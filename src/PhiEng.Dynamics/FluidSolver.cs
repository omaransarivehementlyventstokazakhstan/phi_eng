using System.Collections.Generic;
using System.Numerics;
using PhiEng.Collision;

namespace PhiEng.Dynamics;

/// <summary>
/// Managed SPH fluid stepping (2D). Mirrors the Rust <c>FluidState2</c>
/// solver (<c>crates/dynamics/src/fluid.rs</c>) stage by stage:
/// grid rebuild → density/pressure → adaptive resample → boundary ghosts →
/// pressure forces (+ Monaghan viscosity, cohesion) → integrate (+ ambient
/// turbulence, XSPH, speed cap) → shape collision (two-way impulses) →
/// buoyancy/drag coupling. Behavior parity with Rust is qualitative
/// (tolerance-based, like the native/managed rigid equivalence); exact
/// float-bit equality across languages is not claimed.
/// </summary>
public static class FluidIntegration
{
    private const float Eps = 1e-6f;
    private const float EpsSq = 1e-12f;

    /// <summary>
    /// Advance the fluid one step and couple it to <paramref name="bodies"/>
    /// (two-way). Runs before broadphase so the rigid pipeline observes
    /// post-fluid velocities; a no-op when the fluid holds no particles.
    /// </summary>
    public static void StepFluid2(
        FluidState2 fluid,
        List<RigidBody2> bodies,
        Vector2 gravity,
        AmbientConfig2 ambient,
        float time,
        float dt)
    {
        if (!fluid.Config.Enabled || fluid.Particles.Count == 0 || dt <= 0.0f)
        {
            return;
        }
        uint substeps = CflSubsteps2(fluid, dt);
        float subDt = dt / substeps;
        for (uint s = 0; s < substeps; s++)
        {
            Substep2(fluid, bodies, gravity, ambient, time, subDt);
        }
    }

    /// <summary>2D poly6 kernel: <c>4 / (pi h^8) (h^2 - r^2)^3</c>, zero beyond <c>h</c>.</summary>
    internal static float Poly6(float r2, float h)
    {
        if (r2 >= h * h || r2 < 0.0f)
        {
            return 0.0f;
        }
        float h2 = h * h;
        float t = h2 - r2;
        return 4.0f / (MathF.PI * h2 * h2 * h2 * h2) * t * t * t;
    }

    /// <summary>2D spiky gradient: <c>-30 / (pi h^5) (h - r)^2 * (r_vec / r)</c>.</summary>
    internal static Vector2 SpikyGrad(Vector2 rVec, float r, float h)
    {
        if (r >= h || r < Eps)
        {
            return Vector2.Zero;
        }
        float s = -30.0f / (MathF.PI * h * h * h * h * h) * (h - r) * (h - r);
        return rVec / r * s;
    }

    /// <summary>
    /// Pressure from the configured equation of state (tension-free).
    /// <c>gamma == 1</c> is the legacy linear form; <c>gamma &gt; 1</c> is
    /// the Tait/WCSPH form <c>B*((rho/rho0)^gamma - 1)</c> with <c>B = k*rho0</c>.
    /// </summary>
    internal static float Pressure(float density, float restDensity, float stiffness, float gamma)
    {
        if (gamma == 1.0f)
        {
            return MathF.Max(0.0f, stiffness * (density - restDensity));
        }
        float rho0 = MathF.Max(restDensity, Eps);
        return MathF.Max(0.0f, stiffness * rho0 * (MathF.Pow(density / rho0, gamma) - 1.0f));
    }

    /// <summary>
    /// Monaghan artificial-viscosity pressure term for an approaching pair
    /// (<c>(v_i - v_j) . (x_i - x_j) &lt; 0</c>), else zero.
    /// </summary>
    internal static float ArtificialPressure(Vector2 delta, Vector2 vRel, float rSq, float h, float rhoBar, float c, float alpha)
    {
        float vd = Vector2.Dot(vRel, delta);
        if (vd >= 0.0f)
        {
            return 0.0f;
        }
        float mu = h * vd / (rSq + 0.01f * h * h);
        return -alpha * c * mu / MathF.Max(rhoBar, Eps);
    }

    /// <summary>Bounded cohesion kernel (simplified Akinci): <c>1 - r/h</c> on <c>[0, h)</c>.</summary>
    internal static float Cohesion(float dist, float h)
    {
        if (dist >= h)
        {
            return 0.0f;
        }
        return 1.0f - dist / h;
    }

    /// <summary>Rebuild the neighbor grid from current positions.</summary>
    internal static void RebuildGrid2(FluidState2 fluid)
    {
        foreach (var ids in fluid.Grid.Values)
        {
            ids.Clear();
        }
        fluid.ClearsSinceCompact += 1;
        if (fluid.ClearsSinceCompact >= 256)
        {
            fluid.ClearsSinceCompact = 0;
            var dead = new List<(int, int)>();
            foreach (var kv in fluid.Grid)
            {
                if (kv.Value.Count == 0)
                {
                    dead.Add(kv.Key);
                }
            }
            foreach (var key in dead)
            {
                fluid.GridPool.Add(fluid.Grid[key]);
                fluid.Grid.Remove(key);
            }
        }
        float h = fluid.Config.SmoothingLength;
        var particles = fluid.Particles;
        for (int i = 0; i < particles.Count; i++)
        {
            var p = particles[i].Position;
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y))
            {
                continue;
            }
            var key = ((int)MathF.Floor(p.X / h), (int)MathF.Floor(p.Y / h));
            if (!fluid.Grid.TryGetValue(key, out var ids))
            {
                if (fluid.GridPool.Count > 0)
                {
                    ids = fluid.GridPool[fluid.GridPool.Count - 1];
                    fluid.GridPool.RemoveAt(fluid.GridPool.Count - 1);
                }
                else
                {
                    ids = new List<int>();
                }
                fluid.Grid[key] = ids;
            }
            ids.Add(i);
        }
    }

    /// <summary>
    /// Indices of particles that may lie within the support radius of
    /// particle <c>i</c>, in deterministic cell order. Includes <c>i</c> itself.
    /// </summary>
    internal static void Neighbors2(FluidState2 fluid, int i, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        Vector2 p = fluid.Particles[i].Position;
        int cx = (int)MathF.Floor(p.X / h);
        int cy = (int)MathF.Floor(p.Y / h);
        var particles = fluid.Particles;
        for (int oy = -1; oy <= 1; oy++)
        {
            for (int ox = -1; ox <= 1; ox++)
            {
                if (fluid.Grid.TryGetValue((cx + ox, cy + oy), out var ids))
                {
                    foreach (int j in ids)
                    {
                        if ((particles[j].Position - p).LengthSquared() < h * h)
                        {
                            output.Add(j);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// SPH pressure interpolated at a world-space point:
    /// <c>p(x) = sum m_j (p_j / rho_j) W(|x - x_j|)</c>.
    /// </summary>
    internal static float SamplePressure2(FluidState2 fluid, Vector2 point)
    {
        float h = MathF.Max(fluid.Config.SmoothingLength, Eps);
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        float p = 0.0f;
        var particles = fluid.Particles;
        for (int oy = -1; oy <= 1; oy++)
        {
            for (int ox = -1; ox <= 1; ox++)
            {
                if (fluid.Grid.TryGetValue((cx + ox, cy + oy), out var ids))
                {
                    foreach (int j in ids)
                    {
                        var pj = particles[j];
                        if ((pj.Position - point).LengthSquared() < h * h)
                        {
                            p += pj.Mass * (pj.Pressure / MathF.Max(pj.Density, Eps)) * Poly6((pj.Position - point).LengthSquared(), h);
                        }
                    }
                }
            }
        }
        return MathF.Max(p, 0.0f);
    }

    /// <summary>Surface samples (point, outward normal, area) for buoyancy.</summary>
    internal static void SurfaceSamples2(FluidState2 fluid, RigidBody2 body)
    {
        fluid.SampleScratch.Clear();
        switch (body.Shape)
        {
            case CircleShape c:
            {
                float radius = c.Radius;
                const int n = 12;
                float area = 2.0f * MathF.PI * radius / n;
                for (int k = 0; k < n; k++)
                {
                    float a = 2.0f * MathF.PI * k / n;
                    var local = new Vector2(MathF.Cos(a), MathF.Sin(a));
                    fluid.SampleScratch.Add((
                        body.Transform.TransformPoint(local * radius),
                        body.Transform.RotateVector(local),
                        area));
                }
                break;
            }
            case BoxShape2 b:
            {
                var he = b.HalfExtents;
                float area = 4.0f * (he.X + he.Y) / 16.0f;
                var nx = Vector2.UnitX;
                var ny = Vector2.UnitY;
                var edges = new (Vector2 Normal, Vector2 Origin, Vector2 Tangent)[]
                {
                    (new Vector2(0.0f, -1.0f), new Vector2(-he.X, -he.Y), nx),
                    (new Vector2(0.0f, 1.0f), new Vector2(-he.X, he.Y), nx),
                    (new Vector2(-1.0f, 0.0f), new Vector2(-he.X, -he.Y), ny),
                    (new Vector2(1.0f, 0.0f), new Vector2(he.X, -he.Y), ny),
                };
                foreach (var (normal, origin, tangent) in edges)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        float t = (k + 0.5f) / 4.0f;
                        float extent = tangent == nx ? he.X : he.Y;
                        var local = origin + tangent * (2.0f * extent * t);
                        fluid.SampleScratch.Add((
                            body.Transform.TransformPoint(local),
                            body.Transform.RotateVector(normal),
                            area));
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    /// <summary>
    /// Particle-vs-body boundary test. Returns the outward normal
    /// (body -&gt; particle), penetration depth, and a contact point on
    /// the body surface. Null when separated.
    /// </summary>
    internal static (Vector2 Normal, float Depth, Vector2 Point)? CollideParticleBody2(
        FluidState2 fluid, Vector2 particle, float pr, RigidBody2 body)
    {
        switch (body.Shape)
        {
            case CircleShape c:
            {
                Vector2 delta = particle - body.Transform.Position;
                float dist = delta.Length();
                if (dist >= c.Radius + pr)
                {
                    return null;
                }
                Vector2 normal = dist > Eps ? delta / dist : Vector2.UnitX;
                float depth = c.Radius + pr - dist;
                Vector2 point = body.Transform.Position + normal * c.Radius;
                return (normal, depth, point);
            }
            case BoxShape2 b:
            {
                var he = b.HalfExtents;
                Vector2 rel = particle - body.Transform.Position;
                var rot = body.Transform.Rotation;
                var local = new Vector2(
                    rot.M11 * rel.X + rot.M21 * rel.Y,
                    rot.M12 * rel.X + rot.M22 * rel.Y);
                var clamped = new Vector2(
                    System.Math.Clamp(local.X, -he.X, he.X),
                    System.Math.Clamp(local.Y, -he.Y, he.Y));
                Vector2 diff = local - clamped;
                if (diff.LengthSquared() > EpsSq)
                {
                    float dist = diff.Length();
                    if (dist >= pr)
                    {
                        return null;
                    }
                    Vector2 localN = diff / dist;
                    Vector2 normal = body.Transform.RotateVector(localN);
                    float depth = pr - dist;
                    Vector2 point = body.Transform.TransformPoint(clamped);
                    return (normal, depth, point);
                }
                else
                {
                    float dx = he.X - MathF.Abs(local.X);
                    float dy = he.Y - MathF.Abs(local.Y);
                    Vector2 localN;
                    float depth;
                    if (dx < dy)
                    {
                        localN = new Vector2(Copysign1(local.X), 0.0f);
                        depth = pr + dx;
                    }
                    else
                    {
                        localN = new Vector2(0.0f, Copysign1(local.Y));
                        depth = pr + dy;
                    }
                    Vector2 normal = body.Transform.RotateVector(localN);
                    Vector2 point = body.Transform.TransformPoint(clamped);
                    return (normal, depth, point);
                }
            }
            case PolygonShape poly:
            {
                fluid.PolyScratch.Clear();
                fluid.PolyScratch.Capacity = System.Math.Max(fluid.PolyScratch.Capacity, poly.Vertices.Count);
                foreach (var v in poly.Vertices)
                {
                    fluid.PolyScratch.Add(body.Transform.TransformPoint(v));
                }
                var hit = SAT.CirclePolygon(particle, pr, fluid.PolyScratch);
                if (hit is null)
                {
                    return null;
                }
                Vector2 centroid = SAT.PolygonCentroid(fluid.PolyScratch);
                Vector2 normal = SAT.OrientNormal(hit.Value.Normal, centroid, particle);
                Vector2 point = particle - normal * pr;
                return (normal, hit.Value.Penetration, point);
            }
            default:
                return null;
        }
    }

    /// <summary>Replicates Rust <c>f32::signum().copysign(1.0)</c> (zero maps to +1).</summary>
    internal static float Copysign1(float x)
    {
        if (x < 0.0f || (x == 0.0f && float.IsNegative(x)))
        {
            return -1.0f;
        }
        return 1.0f;
    }

    /// <summary>Number of CFL substeps for this step (<c>1</c> when disabled).</summary>
    internal static uint CflSubsteps2(FluidState2 fluid, float dt)
    {
        var cfg = fluid.Config;
        uint max = cfg.MaxSubsteps < 1 ? 1u : (uint)cfg.MaxSubsteps;
        if (cfg.Cfl <= 0.0f || max <= 1)
        {
            return 1;
        }
        float h = MathF.Max(cfg.SmoothingLength, Eps);
        float rho0 = MathF.Max(cfg.RestDensity, Eps);
        float c0 = cfg.Gamma == 1.0f
            ? MathF.Sqrt(MathF.Max(cfg.Stiffness, 0.0f))
            : MathF.Sqrt(cfg.Gamma * MathF.Max(cfg.Stiffness, 0.0f) * rho0);
        float vmax = 0.0f;
        foreach (var p in fluid.Particles)
        {
            vmax = MathF.Max(vmax, p.Velocity.Length());
        }
        float speed = c0 + vmax;
        if (!float.IsFinite(speed) || speed <= Eps)
        {
            return 1;
        }
        float dtMax = cfg.Cfl * h / speed;
        if (!float.IsFinite(dtMax) || dtMax <= 0.0f)
        {
            return 1;
        }
        long n = (long)MathF.Ceiling(dt / dtMax);
        return (uint)System.Math.Clamp(n, 1L, (long)max);
    }

    /// <summary>Recompute per-particle density (poly6 sum) and pressure (EOS).</summary>
    internal static void ComputeDensityPressure2(FluidState2 fluid, float h, List<int> nbrs)
    {
        var cfg = fluid.Config;
        var particles = fluid.Particles;
        for (int i = 0; i < particles.Count; i++)
        {
            Neighbors2(fluid, i, nbrs);
            Vector2 pi = particles[i].Position;
            float rho = 0.0f;
            foreach (int j in nbrs)
            {
                rho += particles[j].Mass * Poly6((particles[j].Position - pi).LengthSquared(), h);
            }
            var p = particles[i];
            p.Density = System.Math.Clamp(rho, 0.0f, 3.0f * MathF.Max(cfg.RestDensity, Eps));
            particles[i] = p;
        }
        for (int i = 0; i < particles.Count; i++)
        {
            var p = particles[i];
            int ph = p.Phase <= 3 ? p.Phase : 3;
            float rho0 = cfg.PhaseRestDensity[ph];
            p.Pressure = Pressure(p.Density, rho0, cfg.Stiffness, cfg.Gamma);
            particles[i] = p;
        }
    }

    /// <summary>
    /// Fingerprint of everything <see cref="RefreshBoundaries2"/> depends on:
    /// the kernel config plus each static body's transform and shape.
    /// FNV-1a over little-endian field bytes.
    /// </summary>
    internal static ulong BoundaryFingerprint2(FluidConfig2 cfg, List<RigidBody2> bodies)
    {
        const ulong offset = 1469598103934665603ul;
        const ulong prime = 1099511628211ul;
        ulong h = offset;
        void Mix(byte[] bytes)
        {
            foreach (byte b in bytes)
            {
                h ^= b;
                h *= prime;
            }
        }
        Mix(BitConverter.GetBytes(cfg.SmoothingLength));
        Mix(BitConverter.GetBytes(cfg.ParticleMass));
        Mix(BitConverter.GetBytes(cfg.RestDensity));
        foreach (var body in bodies)
        {
            if (!body.IsStatic)
            {
                continue;
            }
            Mix(BitConverter.GetBytes(body.Transform.Position.X));
            Mix(BitConverter.GetBytes(body.Transform.Position.Y));
            var rot = body.Transform.Rotation;
            Mix(BitConverter.GetBytes(rot.M11));
            Mix(BitConverter.GetBytes(rot.M12));
            Mix(BitConverter.GetBytes(rot.M21));
            Mix(BitConverter.GetBytes(rot.M22));
            switch (body.Shape)
            {
                case CircleShape c:
                    Mix(new byte[] { 1 });
                    Mix(BitConverter.GetBytes(c.Radius));
                    break;
                case BoxShape2 b:
                    Mix(new byte[] { 2 });
                    Mix(BitConverter.GetBytes(b.HalfExtents.X));
                    Mix(BitConverter.GetBytes(b.HalfExtents.Y));
                    break;
                case PolygonShape poly:
                    Mix(new byte[] { 3 });
                    Mix(BitConverter.GetBytes(poly.Vertices.Count));
                    foreach (var v in poly.Vertices)
                    {
                        Mix(BitConverter.GetBytes(v.X));
                        Mix(BitConverter.GetBytes(v.Y));
                    }
                    break;
                default:
                    Mix(new byte[] { 0 });
                    break;
            }
        }
        return h;
    }

    /// <summary>Regenerate boundary (ghost) particles from static bodies.</summary>
    internal static void RefreshBoundaries2(FluidState2 fluid, List<RigidBody2> bodies, float spacing)
    {
        fluid.Boundary.Clear();
        float rest = fluid.Config.RestDensity;
        float mass = MathF.Max(fluid.Config.ParticleMass, Eps);
        spacing = MathF.Max(spacing, Eps);
        foreach (var body in bodies)
        {
            if (!body.IsStatic)
            {
                continue;
            }
            switch (body.Shape)
            {
                case CircleShape c:
                {
                    int n = System.Math.Max(1, (int)MathF.Ceiling(2.0f * MathF.PI * c.Radius / spacing));
                    for (int k = 0; k < n; k++)
                    {
                        float a = 2.0f * MathF.PI * k / n;
                        var local = new Vector2(MathF.Cos(a), MathF.Sin(a)) * c.Radius;
                        fluid.Boundary.Add(new FluidParticle2
                        {
                            Position = body.Transform.TransformPoint(local),
                            Velocity = Vector2.Zero,
                            Density = rest,
                            Pressure = 0.0f,
                            Mass = mass,
                            Phase = 0,
                        });
                    }
                    break;
                }
                case BoxShape2 b:
                {
                    var he = b.HalfExtents;
                    var nx = Vector2.UnitX;
                    var ny = Vector2.UnitY;
                    var edges = new (Vector2 Normal, Vector2 Origin, Vector2 Tangent)[]
                    {
                        (new Vector2(0.0f, -1.0f), new Vector2(-he.X, -he.Y), nx),
                        (new Vector2(0.0f, 1.0f), new Vector2(-he.X, he.Y), nx),
                        (new Vector2(-1.0f, 0.0f), new Vector2(-he.X, -he.Y), ny),
                        (new Vector2(1.0f, 0.0f), new Vector2(he.X, -he.Y), ny),
                    };
                    foreach (var (_, origin, tangent) in edges)
                    {
                        float extent = tangent == nx ? he.X : he.Y;
                        int n = System.Math.Max(1, (int)MathF.Ceiling(2.0f * extent / spacing));
                        for (int k = 0; k < n; k++)
                        {
                            float t = (k + 0.5f) / n;
                            var local = origin + tangent * (2.0f * extent * t);
                            fluid.Boundary.Add(new FluidParticle2
                            {
                                Position = body.Transform.TransformPoint(local),
                                Velocity = Vector2.Zero,
                                Density = rest,
                                Pressure = 0.0f,
                                Mass = mass,
                                Phase = 0,
                            });
                        }
                    }
                    break;
                }
                case PolygonShape poly:
                {
                    for (int e = 0; e < poly.Vertices.Count; e++)
                    {
                        Vector2 a = poly.Vertices[e];
                        Vector2 bb = poly.Vertices[(e + 1) % poly.Vertices.Count];
                        Vector2 edge = bb - a;
                        float len = edge.Length();
                        if (len < Eps)
                        {
                            continue;
                        }
                        int n = System.Math.Max(1, (int)MathF.Ceiling(len / spacing));
                        for (int k = 0; k < n; k++)
                        {
                            float t = (k + 0.5f) / n;
                            var local = a + edge * t;
                            fluid.Boundary.Add(new FluidParticle2
                            {
                                Position = body.Transform.TransformPoint(local),
                                Velocity = Vector2.Zero,
                                Density = rest,
                                Pressure = 0.0f,
                                Mass = mass,
                                Phase = 0,
                            });
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    /// <summary>Rebuild the boundary neighbor grid.</summary>
    internal static void RebuildBoundaryGrid2(FluidState2 fluid, float h)
    {
        fluid.BoundaryGrid.Clear();
        var boundary = fluid.Boundary;
        for (int i = 0; i < boundary.Count; i++)
        {
            var p = boundary[i].Position;
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y))
            {
                continue;
            }
            var key = ((int)MathF.Floor(p.X / h), (int)MathF.Floor(p.Y / h));
            if (!fluid.BoundaryGrid.TryGetValue(key, out var ids))
            {
                ids = new List<int>();
                fluid.BoundaryGrid[key] = ids;
            }
            ids.Add(i);
        }
    }

    /// <summary>Boundary indices within the support radius of a world-space point.</summary>
    internal static void BoundaryNear2(FluidState2 fluid, Vector2 point, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        var boundary = fluid.Boundary;
        for (int oy = -1; oy <= 1; oy++)
        {
            for (int ox = -1; ox <= 1; ox++)
            {
                if (fluid.BoundaryGrid.TryGetValue((cx + ox, cy + oy), out var ids))
                {
                    foreach (int j in ids)
                    {
                        if ((boundary[j].Position - point).LengthSquared() < h * h)
                        {
                            output.Add(j);
                        }
                    }
                }
            }
        }
    }

    /// <summary>Fluid-particle indices within the support radius of a world-space point.</summary>
    internal static void FluidNear2(FluidState2 fluid, Vector2 point, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        var particles = fluid.Particles;
        for (int oy = -1; oy <= 1; oy++)
        {
            for (int ox = -1; ox <= 1; ox++)
            {
                if (fluid.Grid.TryGetValue((cx + ox, cy + oy), out var ids))
                {
                    foreach (int j in ids)
                    {
                        if ((particles[j].Position - point).LengthSquared() < h * h)
                        {
                            output.Add(j);
                        }
                    }
                }
            }
        }
    }

    /// <summary>Boundary density/pressure from nearby fluid particles (two-pass WCSPH).</summary>
    internal static void ComputeBoundaryDensityPressure2(FluidState2 fluid, float h, List<int> nbrs)
    {
        var cfg = fluid.Config;
        var boundary = fluid.Boundary;
        var particles = fluid.Particles;
        for (int bi = 0; bi < boundary.Count; bi++)
        {
            Vector2 bp = boundary[bi].Position;
            FluidNear2(fluid, bp, nbrs);
            float rho = 0.0f;
            foreach (int j in nbrs)
            {
                rho += particles[j].Mass * Poly6((particles[j].Position - bp).LengthSquared(), h);
            }
            var g = boundary[bi];
            g.Density = System.Math.Clamp(rho, 0.0f, 3.0f * MathF.Max(cfg.RestDensity, Eps));
            g.Pressure = Pressure(rho, cfg.RestDensity, cfg.Stiffness, cfg.Gamma);
            boundary[bi] = g;
        }
    }

    /// <summary>Adaptive-resolution pass: merge over-dense close pairs, then split under-dense.</summary>
    internal static void Resample2(FluidState2 fluid, float h)
    {
        var cfg = fluid.Config;
        var particles = fluid.Particles;
        float rest = MathF.Max(cfg.RestDensity, Eps);
        float splitLo = cfg.SplitDensityRatio * rest;
        float mergeHi = cfg.MergeDensityRatio * rest;
        float minMass = MathF.Max(cfg.ParticleMass, Eps) * 0.125f;

        int n = particles.Count;
        var absorbed = new bool[n];
        float mergeDistSq = 0.25f * h * (0.25f * h);
        for (int i = 0; i < n; i++)
        {
            if (absorbed[i] || particles[i].Density <= mergeHi)
            {
                continue;
            }
            for (int j = i - 1; j >= 0; j--)
            {
                if (absorbed[j] || particles[j].Density <= mergeHi)
                {
                    continue;
                }
                if ((particles[i].Position - particles[j].Position).LengthSquared() < mergeDistSq)
                {
                    float mi = particles[i].Mass;
                    float mj = particles[j].Mass;
                    float mt = mi + mj;
                    if (mt > Eps)
                    {
                        var pi = particles[i];
                        pi.Velocity = (particles[i].Velocity * mi + particles[j].Velocity * mj) / mt;
                        pi.Mass = mt;
                        particles[i] = pi;
                    }
                    absorbed[j] = true;
                    break;
                }
            }
        }
        bool any = false;
        foreach (bool a in absorbed)
        {
            if (a)
            {
                any = true;
                break;
            }
        }
        if (any)
        {
            var kept = new List<FluidParticle2>(n);
            for (int idx = 0; idx < particles.Count; idx++)
            {
                if (!absorbed[idx])
                {
                    kept.Add(particles[idx]);
                }
            }
            particles.Clear();
            particles.AddRange(kept);
        }

        var output = new List<FluidParticle2>(particles.Count);
        var off = new Vector2(0.25f * h, 0.0f);
        foreach (var p in particles)
        {
            if (p.Density < splitLo && p.Mass > minMass && output.Count + 1 < FluidState2.MaxParticles)
            {
                float half = p.Mass * 0.5f;
                var c0 = p;
                c0.Mass = half;
                c0.Position = p.Position + off;
                output.Add(c0);
                var c1 = p;
                c1.Mass = half;
                c1.Position = p.Position - off;
                output.Add(c1);
            }
            else
            {
                output.Add(p);
            }
        }
        particles.Clear();
        particles.AddRange(output);
    }

    /// <summary>One fluid substep (the full SPH pipeline) over interval <c>dt</c>.</summary>
    internal static void Substep2(
        FluidState2 fluid,
        List<RigidBody2> bodies,
        Vector2 gravity,
        AmbientConfig2 ambient,
        float time,
        float dt)
    {
        var cfg = fluid.Config;
        float h = MathF.Max(cfg.SmoothingLength, Eps);
        var particles = fluid.Particles;

        RebuildGrid2(fluid);

        var nbrs = new List<int>();
        var bnbrs = new List<int>();
        ComputeDensityPressure2(fluid, h, nbrs);

        if (cfg.AdaptiveResolution)
        {
            Resample2(fluid, h);
            RebuildGrid2(fluid);
            ComputeDensityPressure2(fluid, h, nbrs);
        }

        if (cfg.BoundaryParticles)
        {
            ulong fp = BoundaryFingerprint2(cfg, bodies);
            if (fluid.BoundaryRevision != fp || fluid.Boundary.Count == 0)
            {
                RefreshBoundaries2(fluid, bodies, h * 0.5f);
                RebuildBoundaryGrid2(fluid, h);
                fluid.BoundaryRevision = fp;
                fluid.BoundaryRegenCount++;
            }
            ComputeBoundaryDensityPressure2(fluid, h, nbrs);
        }

        float cs = MathF.Sqrt(MathF.Max(cfg.Stiffness, 0.0f));
        fluid.Forces.Clear();
        for (int k = 0; k < particles.Count; k++)
        {
            fluid.Forces.Add(Vector2.Zero);
        }
        var boundary = fluid.Boundary;
        for (int i = 0; i < particles.Count; i++)
        {
            Neighbors2(fluid, i, nbrs);
            Vector2 pi = particles[i].Position;
            Vector2 vi = particles[i].Velocity;
            float rhoI = MathF.Max(particles[i].Density, Eps);
            float pI = particles[i].Pressure;
            Vector2 f = Vector2.Zero;
            foreach (int j in nbrs)
            {
                if (j == i)
                {
                    continue;
                }
                var pj = particles[j];
                float rhoJ = MathF.Max(pj.Density, Eps);
                Vector2 delta = pi - pj.Position;
                float dist = delta.Length();
                Vector2 grad = SpikyGrad(delta, dist, h);
                float term = pI / (rhoI * rhoI) + pj.Pressure / (rhoJ * rhoJ);
                if (cfg.ArtificialViscosity > 0.0f && dist > Eps)
                {
                    float rhoBar = 0.5f * (rhoI + rhoJ);
                    term += ArtificialPressure(delta, vi - pj.Velocity, dist * dist, h, rhoBar, cs, cfg.ArtificialViscosity);
                }
                f -= grad * (pj.Mass * term);
                if (cfg.SurfaceTension > 0.0f && dist > Eps)
                {
                    float coh = Cohesion(dist, h);
                    if (coh > 0.0f)
                    {
                        f -= delta / dist * (cfg.SurfaceTension * particles[i].Mass * pj.Mass * coh);
                    }
                }
            }
            if (cfg.BoundaryParticles)
            {
                BoundaryNear2(fluid, pi, bnbrs);
                foreach (int bk in bnbrs)
                {
                    var bp = boundary[bk];
                    Vector2 delta = pi - bp.Position;
                    float dist = delta.Length();
                    Vector2 grad = SpikyGrad(delta, dist, h);
                    float rhoB = MathF.Max(bp.Density, Eps);
                    f -= grad * (bp.Mass * (pI / (rhoI * rhoI) + bp.Pressure / (rhoB * rhoB)));
                }
            }
            fluid.Forces[i] = f;
        }

        var turb = ambient.Turbulence;
        bool turbOn = turb.Enabled && turb.Amplitude > 0.0f;
        for (int i = 0; i < particles.Count; i++)
        {
            Neighbors2(fluid, i, nbrs);
            float rhoI = MathF.Max(particles[i].Density, Eps);
            Vector2 acc = gravity + fluid.Forces[i] / rhoI;
            if (turbOn)
            {
                Vector2 q = particles[i].Position * turb.Frequency;
                float t = time * turb.TimeScale;
                Vector2 nv = turb.DivergenceFree
                    ? TurbulenceNoise.Curl2(q.X, q.Y, t, turb.Seed, turb.Octaves)
                    : new Vector2(
                        TurbulenceNoise.Fbm2(q.X, q.Y, t, turb.Seed, turb.Octaves),
                        TurbulenceNoise.Fbm2(q.X, q.Y, t, turb.Seed ^ 0x55aa55aa, turb.Octaves));
                if (nv.LengthSquared() > 1.0f)
                {
                    nv /= nv.Length();
                }
                acc += nv * turb.Amplitude;
            }
            Vector2 v = particles[i].Velocity + acc * dt;
            if (cfg.Viscosity > 0.0f)
            {
                Vector2 ppos = particles[i].Position;
                Vector2 vi0 = particles[i].Velocity;
                Vector2 smooth = Vector2.Zero;
                foreach (int j in nbrs)
                {
                    if (j == i)
                    {
                        continue;
                    }
                    var pj = particles[j];
                    float rhoBar = 0.5f * (rhoI + MathF.Max(pj.Density, Eps));
                    smooth += (pj.Velocity - vi0) * (pj.Mass / rhoBar)
                        * Poly6((pj.Position - ppos).LengthSquared(), h);
                }
                v += smooth * cfg.Viscosity;
            }
            float speed = v.Length();
            if (speed > cfg.MaxVelocity)
            {
                v *= cfg.MaxVelocity / speed;
            }
            if (!float.IsFinite(v.X) || !float.IsFinite(v.Y))
            {
                v = Vector2.Zero;
            }
            var pp = particles[i];
            pp.Velocity = v;
            Vector2 pos = pp.Position + v * dt;
            if (!float.IsFinite(pos.X) || !float.IsFinite(pos.Y))
            {
                pos = pp.Position;
            }
            pp.Position = pos;
            particles[i] = pp;
        }

        float pr = MathF.Max(cfg.ParticleRadius, 0.0f);
        for (int i = 0; i < particles.Count; i++)
        {
            foreach (var body in bodies)
            {
                Vector2 ppos = particles[i].Position;
                var hit = CollideParticleBody2(fluid, ppos, pr, body);
                if (hit is null)
                {
                    continue;
                }
                if (hit.Value.Depth <= 0.0f)
                {
                    continue;
                }
                var (normal, depth, point) = hit.Value;
                var pp = particles[i];
                pp.Position = ppos + normal * depth;
                float vn = Vector2.Dot(pp.Velocity, normal);
                if (vn < 0.0f)
                {
                    pp.Velocity -= normal * vn;
                    body.ApplyImpulse(normal * (vn * pp.Mass), point);
                    if (cfg.WallFriction > 0.0f)
                    {
                        float mu = System.Math.Clamp(cfg.WallFriction, 0.0f, 1.0f);
                        Vector2 vt = pp.Velocity;
                        Vector2 wallV = body.Velocity;
                        Vector2 wallVt = wallV - normal * Vector2.Dot(wallV, normal);
                        Vector2 deltaV = (wallVt - vt) * mu;
                        pp.Velocity += deltaV;
                        body.ApplyImpulse(-deltaV * pp.Mass, point);
                    }
                    if (!body.IsStatic)
                    {
                        body.IsSleeping = false;
                    }
                }
                particles[i] = pp;
            }
        }

        RebuildGrid2(fluid);
        foreach (var body in bodies)
        {
            if (body.IsStatic)
            {
                continue;
            }
            SurfaceSamples2(fluid, body);
            if (fluid.SampleScratch.Count == 0)
            {
                continue;
            }
            float drag = MathF.Max(fluid.Config.Drag, 0.0f);
            bool wet = false;
            foreach (var (point, normal, area) in fluid.SampleScratch)
            {
                float p = SamplePressure2(fluid, point);
                if (p <= 0.0f)
                {
                    continue;
                }
                wet = true;
                Vector2 r = point - body.Transform.Position;
                Vector2 vRel = body.Velocity
                    + new Vector2(-body.AngularVelocity * r.Y, body.AngularVelocity * r.X)
                    - ambient.Drag.Wind;
                float spd = vRel.Length();
                Vector2 force = -normal * (p * area);
                if (spd > Eps)
                {
                    force -= vRel * (drag * area * spd);
                }
                body.ApplyForce(force, point, dt);
            }
            if (wet)
            {
                body.IsSleeping = false;
            }
        }
    }
}
