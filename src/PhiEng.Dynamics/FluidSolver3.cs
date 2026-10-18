using System.Collections.Generic;
using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Managed SPH fluid stepping (3D). Mirrors the Rust <c>FluidState3</c>
/// solver (<c>crates/dynamics/src/fluid3.rs</c>) stage by stage; see
/// <see cref="FluidIntegration"/> for the parity contract.
/// </summary>
public static class FluidIntegration3
{
    private const float Eps = 1e-6f;
    private const float EpsSq = 1e-12f;

    /// <summary>
    /// Advance the 3D fluid one step and couple it to <paramref name="bodies"/>
    /// (two-way). A no-op when the fluid holds no particles.
    /// </summary>
    public static void StepFluid3(
        FluidState3 fluid,
        List<RigidBody3> bodies,
        Vector3 gravity,
        AmbientConfig3 ambient,
        float time,
        float dt)
    {
        if (!fluid.Config.Enabled || fluid.Particles.Count == 0 || dt <= 0.0f)
        {
            return;
        }
        uint substeps = CflSubsteps3(fluid, dt);
        float subDt = dt / substeps;
        for (uint s = 0; s < substeps; s++)
        {
            Substep3(fluid, bodies, gravity, ambient, time, subDt);
        }
    }

    /// <summary>3D poly6 kernel: <c>315 / (64 pi h^9) (h^2 - r^2)^3</c>.</summary>
    internal static float Poly6(float r2, float h)
    {
        if (r2 >= h * h || r2 < 0.0f)
        {
            return 0.0f;
        }
        float h2 = h * h;
        float t = h2 - r2;
        return 315.0f / (64.0f * MathF.PI * h2 * h2 * h2 * h2 * h) * t * t * t;
    }

    /// <summary>3D spiky gradient: <c>-45 / (pi h^6) (h - r)^2 * (r_vec / r)</c>.</summary>
    internal static Vector3 SpikyGrad(Vector3 rVec, float r, float h)
    {
        if (r >= h || r < Eps)
        {
            return Vector3.Zero;
        }
        float s = -45.0f / (MathF.PI * h * h * h * h * h * h) * (h - r) * (h - r);
        return rVec / r * s;
    }

    /// <summary>Pressure from the configured equation of state (tension-free).</summary>
    internal static float Pressure(float density, float restDensity, float stiffness, float gamma)
    {
        if (gamma == 1.0f)
        {
            return MathF.Max(0.0f, stiffness * (density - restDensity));
        }
        float rho0 = MathF.Max(restDensity, Eps);
        return MathF.Max(0.0f, stiffness * rho0 * (MathF.Pow(density / rho0, gamma) - 1.0f));
    }

    /// <summary>Monaghan artificial-viscosity pressure term for an approaching pair.</summary>
    internal static float ArtificialPressure(Vector3 delta, Vector3 vRel, float rSq, float h, float rhoBar, float c, float alpha)
    {
        float vd = Vector3.Dot(vRel, delta);
        if (vd >= 0.0f)
        {
            return 0.0f;
        }
        float mu = h * vd / (rSq + 0.01f * h * h);
        return -alpha * c * mu / MathF.Max(rhoBar, Eps);
    }

    /// <summary>Bounded cohesion kernel: <c>1 - r/h</c> on <c>[0, h)</c>.</summary>
    internal static float Cohesion(float dist, float h)
    {
        if (dist >= h)
        {
            return 0.0f;
        }
        return 1.0f - dist / h;
    }

    internal static Vector3 NormalizeOrZero(Vector3 v)
    {
        float len = v.Length();
        return len > 0.0f && float.IsFinite(len) ? v / len : Vector3.Zero;
    }

    /// <summary>Rebuild the neighbor grid from current positions.</summary>
    internal static void RebuildGrid3(FluidState3 fluid)
    {
        foreach (var ids in fluid.Grid.Values)
        {
            ids.Clear();
        }
        fluid.ClearsSinceCompact += 1;
        if (fluid.ClearsSinceCompact >= 256)
        {
            fluid.ClearsSinceCompact = 0;
            var dead = new List<(int, int, int)>();
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
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || !float.IsFinite(p.Z))
            {
                continue;
            }
            var key = ((int)MathF.Floor(p.X / h), (int)MathF.Floor(p.Y / h), (int)MathF.Floor(p.Z / h));
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

    /// <summary>Indices of particles within the support radius of particle <c>i</c>, deterministic order.</summary>
    internal static void Neighbors3(FluidState3 fluid, int i, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        Vector3 p = fluid.Particles[i].Position;
        int cx = (int)MathF.Floor(p.X / h);
        int cy = (int)MathF.Floor(p.Y / h);
        int cz = (int)MathF.Floor(p.Z / h);
        var particles = fluid.Particles;
        for (int oz = -1; oz <= 1; oz++)
        {
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (fluid.Grid.TryGetValue((cx + ox, cy + oy, cz + oz), out var ids))
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
    }

    /// <summary>SPH pressure interpolated at a world-space point.</summary>
    internal static float SamplePressure3(FluidState3 fluid, Vector3 point)
    {
        float h = MathF.Max(fluid.Config.SmoothingLength, Eps);
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        int cz = (int)MathF.Floor(point.Z / h);
        float p = 0.0f;
        var particles = fluid.Particles;
        for (int oz = -1; oz <= 1; oz++)
        {
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (fluid.Grid.TryGetValue((cx + ox, cy + oy, cz + oz), out var ids))
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
        }
        return MathF.Max(p, 0.0f);
    }

    /// <summary>Surface samples (point, outward normal, area) for buoyancy.</summary>
    internal static void SurfaceSamples3(FluidState3 fluid, RigidBody3 body)
    {
        fluid.SampleScratch.Clear();
        switch (body.Shape)
        {
            case SphereShape3 s:
            {
                float radius = s.Radius;
                var dirs = new Vector3[]
                {
                    Vector3.UnitX, -Vector3.UnitX,
                    Vector3.UnitY, -Vector3.UnitY,
                    Vector3.UnitZ, -Vector3.UnitZ,
                    NormalizeOrZero(new Vector3(1.0f, 1.0f, 1.0f)),
                    NormalizeOrZero(new Vector3(1.0f, 1.0f, -1.0f)),
                    NormalizeOrZero(new Vector3(1.0f, -1.0f, 1.0f)),
                    NormalizeOrZero(new Vector3(1.0f, -1.0f, -1.0f)),
                    NormalizeOrZero(new Vector3(-1.0f, 1.0f, 1.0f)),
                    NormalizeOrZero(new Vector3(-1.0f, 1.0f, -1.0f)),
                    NormalizeOrZero(new Vector3(-1.0f, -1.0f, 1.0f)),
                    NormalizeOrZero(new Vector3(-1.0f, -1.0f, -1.0f)),
                };
                float area = 4.0f * MathF.PI * radius * radius / dirs.Length;
                foreach (var d in dirs)
                {
                    fluid.SampleScratch.Add((
                        body.Transform.TransformPoint(d * radius),
                        body.Transform.RotateVector(d),
                        area));
                }
                break;
            }
            case BoxShape3 b:
            {
                var he = b.HalfExtents;
                float area = 8.0f * (he.X * he.Y + he.Y * he.Z + he.Z * he.X) / 24.0f;
                var faces = new (Vector3 Normal, float Fixed, int Axis)[]
                {
                    (Vector3.UnitY, -he.Y, 0),
                    (-Vector3.UnitY, -he.Y, 0),
                    (Vector3.UnitX, -he.X, 1),
                    (-Vector3.UnitX, -he.X, 1),
                    (Vector3.UnitZ, -he.Z, 2),
                    (-Vector3.UnitZ, -he.Z, 2),
                };
                foreach (var (normal, fixedV, axis) in faces)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        float t0, t1;
                        switch (k)
                        {
                            case 0: t0 = -0.75f; t1 = -0.75f; break;
                            case 1: t0 = 0.75f; t1 = -0.75f; break;
                            case 2: t0 = -0.75f; t1 = 0.75f; break;
                            default: t0 = 0.75f; t1 = 0.75f; break;
                        }
                        Vector3 local = axis switch
                        {
                            0 => new Vector3(he.X * t0, fixedV, he.Z * t1),
                            1 => new Vector3(fixedV, he.Y * t0, he.Z * t1),
                            _ => new Vector3(he.X * t0, he.Y * t1, fixedV),
                        };
                        fluid.SampleScratch.Add((
                            body.Transform.TransformPoint(local),
                            body.Transform.RotateVector(normal),
                            area));
                    }
                }
                break;
            }
            case CapsuleShape3 c:
            {
                float r = c.Radius;
                float hl = c.HalfLength;
                float sideArea = 4.0f * MathF.PI * r * hl / 16.0f;
                float capArea = 2.0f * MathF.PI * r * r;
                foreach (float sy in new float[] { -hl * 0.5f, hl * 0.5f })
                {
                    for (int k = 0; k < 8; k++)
                    {
                        float a = 2.0f * MathF.PI * k / 8.0f;
                        var dir = new Vector3(MathF.Cos(a), 0.0f, MathF.Sin(a));
                        fluid.SampleScratch.Add((
                            body.Transform.TransformPoint(dir * r + Vector3.UnitY * sy),
                            body.Transform.RotateVector(dir),
                            sideArea));
                    }
                }
                foreach (float s in new float[] { -1.0f, 1.0f })
                {
                    fluid.SampleScratch.Add((
                        body.Transform.TransformPoint(Vector3.UnitY * s * (hl + r)),
                        body.Transform.RotateVector(Vector3.UnitY * s),
                        capArea));
                }
                break;
            }
            default:
                break;
        }
    }

    /// <summary>
    /// Particle-vs-body boundary test. Hulls use a nearest-vertex-sphere
    /// approximation (same documented gap as Rust).
    /// </summary>
    internal static (Vector3 Normal, float Depth, Vector3 Point)? CollideParticleBody3(
        FluidState3 fluid, Vector3 particle, float pr, RigidBody3 body)
    {
        switch (body.Shape)
        {
            case SphereShape3 s:
            {
                Vector3 delta = particle - body.Transform.Position;
                float dist = delta.Length();
                if (dist >= s.Radius + pr)
                {
                    return null;
                }
                Vector3 normal = dist > Eps ? delta / dist : Vector3.UnitX;
                float depth = s.Radius + pr - dist;
                Vector3 point = body.Transform.Position + normal * s.Radius;
                return (normal, depth, point);
            }
            case BoxShape3 b:
            {
                var he = b.HalfExtents;
                Vector3 rel = particle - body.Transform.Position;
                Vector3 local = Vector3.Transform(rel, Quaternion.Inverse(body.Transform.Rotation));
                var clamped = new Vector3(
                    System.Math.Clamp(local.X, -he.X, he.X),
                    System.Math.Clamp(local.Y, -he.Y, he.Y),
                    System.Math.Clamp(local.Z, -he.Z, he.Z));
                Vector3 diff = local - clamped;
                if (diff.LengthSquared() > EpsSq)
                {
                    float dist = diff.Length();
                    if (dist >= pr)
                    {
                        return null;
                    }
                    Vector3 normal = Vector3.Transform(diff / dist, body.Transform.Rotation);
                    float depth = pr - dist;
                    Vector3 point = body.Transform.TransformPoint(clamped);
                    return (normal, depth, point);
                }
                else
                {
                    float dx = he.X - MathF.Abs(local.X);
                    float dy = he.Y - MathF.Abs(local.Y);
                    float dz = he.Z - MathF.Abs(local.Z);
                    Vector3 axis;
                    float sn;
                    float d;
                    if (dx <= dy && dx <= dz)
                    {
                        axis = Vector3.UnitX; sn = Copysign1(local.X); d = dx;
                    }
                    else if (dy <= dz)
                    {
                        axis = Vector3.UnitY; sn = Copysign1(local.Y); d = dy;
                    }
                    else
                    {
                        axis = Vector3.UnitZ; sn = Copysign1(local.Z); d = dz;
                    }
                    Vector3 normal = Vector3.Transform(axis * sn, body.Transform.Rotation);
                    Vector3 point = body.Transform.TransformPoint(clamped);
                    return (normal, pr + d, point);
                }
            }
            case CapsuleShape3 c:
            {
                float r = c.Radius;
                float hl = c.HalfLength;
                Vector3 rel = particle - body.Transform.Position;
                Vector3 local = Vector3.Transform(rel, Quaternion.Inverse(body.Transform.Rotation));
                float sy = System.Math.Clamp(local.Y, -hl, hl);
                var seg = new Vector3(0.0f, sy, 0.0f);
                Vector3 diff = local - seg;
                if (diff.LengthSquared() > EpsSq)
                {
                    float dist = diff.Length();
                    if (dist >= r + pr)
                    {
                        return null;
                    }
                    Vector3 normal = Vector3.Transform(diff / dist, body.Transform.Rotation);
                    float depth = r + pr - dist;
                    Vector3 point = body.Transform.TransformPoint(seg + NormalizeOrZero(diff) * r);
                    return (normal, depth, point);
                }
                else
                {
                    var radial = new Vector3(local.X, 0.0f, local.Z);
                    Vector3 dir = radial.LengthSquared() > EpsSq ? NormalizeOrZero(radial) : Vector3.UnitX;
                    Vector3 normal = Vector3.Transform(dir, body.Transform.Rotation);
                    Vector3 point = body.Transform.TransformPoint(seg + dir * r);
                    return (normal, pr + r - diff.Length(), point);
                }
            }
            case ConvexHullShape3 hull:
            {
                (Vector3 Normal, float Depth, Vector3 Point)? best = null;
                foreach (var v in hull.Vertices)
                {
                    Vector3 w = body.Transform.TransformPoint(v);
                    Vector3 delta = particle - w;
                    float dist = delta.Length();
                    if (dist < pr)
                    {
                        Vector3 normal = dist > Eps ? delta / dist : Vector3.UnitX;
                        float depth = pr - dist;
                        if (best is null || depth > best.Value.Depth)
                        {
                            best = (normal, depth, w);
                        }
                    }
                }
                return best;
            }
            default:
                return null;
        }
    }

    internal static float Copysign1(float x)
    {
        if (x < 0.0f || (x == 0.0f && float.IsNegative(x)))
        {
            return -1.0f;
        }
        return 1.0f;
    }

    /// <summary>Number of CFL substeps for this step (<c>1</c> when disabled).</summary>
    internal static uint CflSubsteps3(FluidState3 fluid, float dt)
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
    internal static void ComputeDensityPressure3(FluidState3 fluid, float h, List<int> nbrs)
    {
        var cfg = fluid.Config;
        var particles = fluid.Particles;
        for (int i = 0; i < particles.Count; i++)
        {
            Neighbors3(fluid, i, nbrs);
            Vector3 pi = particles[i].Position;
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
    /// Fingerprint of everything <see cref="RefreshBoundaries3"/> depends on:
    /// the kernel config plus each static body's transform and shape.
    /// FNV-1a over little-endian field bytes.
    /// </summary>
    internal static ulong BoundaryFingerprint3(FluidConfig3 cfg, List<RigidBody3> bodies)
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
            Mix(BitConverter.GetBytes(body.Transform.Position.Z));
            var q = body.Transform.Rotation;
            Mix(BitConverter.GetBytes(q.X));
            Mix(BitConverter.GetBytes(q.Y));
            Mix(BitConverter.GetBytes(q.Z));
            Mix(BitConverter.GetBytes(q.W));
            switch (body.Shape)
            {
                case SphereShape3 s:
                    Mix(new byte[] { 1 });
                    Mix(BitConverter.GetBytes(s.Radius));
                    break;
                case BoxShape3 b:
                    Mix(new byte[] { 2 });
                    Mix(BitConverter.GetBytes(b.HalfExtents.X));
                    Mix(BitConverter.GetBytes(b.HalfExtents.Y));
                    Mix(BitConverter.GetBytes(b.HalfExtents.Z));
                    break;
                case CapsuleShape3 c:
                    Mix(new byte[] { 3 });
                    Mix(BitConverter.GetBytes(c.Radius));
                    Mix(BitConverter.GetBytes(c.HalfLength));
                    break;
                case ConvexHullShape3 hull:
                    Mix(new byte[] { 4 });
                    Mix(BitConverter.GetBytes(hull.Vertices.Count));
                    foreach (var v in hull.Vertices)
                    {
                        Mix(BitConverter.GetBytes(v.X));
                        Mix(BitConverter.GetBytes(v.Y));
                        Mix(BitConverter.GetBytes(v.Z));
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
    internal static void RefreshBoundaries3(FluidState3 fluid, List<RigidBody3> bodies, float spacing)
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
                case SphereShape3 s:
                {
                    int n = (int)System.Math.Clamp(
                        MathF.Ceiling(4.0f * MathF.PI * s.Radius * s.Radius / (spacing * spacing)),
                        1.0f, 512.0f);
                    float golden = MathF.PI * (3.0f - MathF.Sqrt(5.0f));
                    for (int k = 0; k < n; k++)
                    {
                        float z = 1.0f - 2.0f * (k + 0.5f) / n;
                        float r = MathF.Sqrt(MathF.Max(0.0f, 1.0f - z * z));
                        float theta = golden * k;
                        var local = new Vector3(MathF.Cos(theta) * r, z, MathF.Sin(theta) * r) * s.Radius;
                        fluid.Boundary.Add(new FluidParticle3
                        {
                            Position = body.Transform.TransformPoint(local),
                            Velocity = Vector3.Zero,
                            Density = rest,
                            Pressure = 0.0f,
                            Mass = mass,
                            Phase = 0,
                        });
                    }
                    break;
                }
                case BoxShape3 b:
                {
                    var he = b.HalfExtents;
                    var faces = new (Vector3 Normal, float Fixed, int Axis)[]
                    {
                        (Vector3.UnitY, -he.Y, 0),
                        (-Vector3.UnitY, -he.Y, 0),
                        (Vector3.UnitX, -he.X, 1),
                        (-Vector3.UnitX, -he.X, 1),
                        (Vector3.UnitZ, -he.Z, 2),
                        (-Vector3.UnitZ, -he.Z, 2),
                    };
                    foreach (var (_, fixedV, axis) in faces)
                    {
                        float area = axis switch
                        {
                            0 => 4.0f * he.Y * he.Z,
                            1 => 4.0f * he.X * he.Z,
                            _ => 4.0f * he.X * he.Y,
                        };
                        int n = (int)System.Math.Clamp(
                            MathF.Ceiling(area / (spacing * spacing)), 1.0f, 512.0f);
                        for (int k = 0; k < n; k++)
                        {
                            float t = (k + 0.5f) / n;
                            Vector3 local = axis switch
                            {
                                0 => new Vector3((2.0f * t - 1.0f) * he.X, fixedV, (2.0f * ((t * 7.0f) % 1.0f) - 1.0f) * he.Z),
                                1 => new Vector3(fixedV, (2.0f * t - 1.0f) * he.Y, (2.0f * ((t * 7.0f) % 1.0f) - 1.0f) * he.Z),
                                _ => new Vector3((2.0f * t - 1.0f) * he.X, (2.0f * ((t * 7.0f) % 1.0f) - 1.0f) * he.Y, fixedV),
                            };
                            fluid.Boundary.Add(new FluidParticle3
                            {
                                Position = body.Transform.TransformPoint(local),
                                Velocity = Vector3.Zero,
                                Density = rest,
                                Pressure = 0.0f,
                                Mass = mass,
                                Phase = 0,
                            });
                        }
                    }
                    break;
                }
                case CapsuleShape3 c:
                {
                    float r = c.Radius;
                    float hl = c.HalfLength;
                    float sideArea = 4.0f * MathF.PI * r * hl;
                    int n = (int)System.Math.Clamp(
                        MathF.Ceiling(sideArea / (spacing * spacing)), 1.0f, 256.0f);
                    foreach (float sy in new float[] { -hl * 0.5f, hl * 0.5f })
                    {
                        for (int k = 0; k < n; k++)
                        {
                            float a = 2.0f * MathF.PI * k / n;
                            var local = new Vector3(MathF.Cos(a) * r, sy, MathF.Sin(a) * r);
                            fluid.Boundary.Add(new FluidParticle3
                            {
                                Position = body.Transform.TransformPoint(local),
                                Velocity = Vector3.Zero,
                                Density = rest,
                                Pressure = 0.0f,
                                Mass = mass,
                                Phase = 0,
                            });
                        }
                    }
                    foreach (float s in new float[] { -1.0f, 1.0f })
                    {
                        fluid.Boundary.Add(new FluidParticle3
                        {
                            Position = body.Transform.TransformPoint(Vector3.UnitY * s * (hl + r)),
                            Velocity = Vector3.Zero,
                            Density = rest,
                            Pressure = 0.0f,
                            Mass = mass,
                            Phase = 0,
                        });
                    }
                    break;
                }
                case ConvexHullShape3 hull:
                {
                    foreach (var v in hull.Vertices)
                    {
                        fluid.Boundary.Add(new FluidParticle3
                        {
                            Position = body.Transform.TransformPoint(v),
                            Velocity = Vector3.Zero,
                            Density = rest,
                            Pressure = 0.0f,
                            Mass = mass,
                            Phase = 0,
                        });
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    /// <summary>Rebuild the boundary neighbor grid.</summary>
    internal static void RebuildBoundaryGrid3(FluidState3 fluid, float h)
    {
        fluid.BoundaryGrid.Clear();
        var boundary = fluid.Boundary;
        for (int i = 0; i < boundary.Count; i++)
        {
            var p = boundary[i].Position;
            if (!float.IsFinite(p.X))
            {
                continue;
            }
            var key = ((int)MathF.Floor(p.X / h), (int)MathF.Floor(p.Y / h), (int)MathF.Floor(p.Z / h));
            if (!fluid.BoundaryGrid.TryGetValue(key, out var ids))
            {
                ids = new List<int>();
                fluid.BoundaryGrid[key] = ids;
            }
            ids.Add(i);
        }
    }

    /// <summary>Boundary indices within the support radius of a world-space point.</summary>
    internal static void BoundaryNear3(FluidState3 fluid, Vector3 point, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        int cz = (int)MathF.Floor(point.Z / h);
        var boundary = fluid.Boundary;
        for (int oz = -1; oz <= 1; oz++)
        {
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (fluid.BoundaryGrid.TryGetValue((cx + ox, cy + oy, cz + oz), out var ids))
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
    }

    /// <summary>Fluid-particle indices within the support radius of a world point.</summary>
    internal static void FluidNear3(FluidState3 fluid, Vector3 point, List<int> output)
    {
        output.Clear();
        float h = fluid.Config.SmoothingLength;
        int cx = (int)MathF.Floor(point.X / h);
        int cy = (int)MathF.Floor(point.Y / h);
        int cz = (int)MathF.Floor(point.Z / h);
        var particles = fluid.Particles;
        for (int oz = -1; oz <= 1; oz++)
        {
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (fluid.Grid.TryGetValue((cx + ox, cy + oy, cz + oz), out var ids))
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
    }

    /// <summary>Boundary density/pressure from nearby fluid particles.</summary>
    internal static void ComputeBoundaryDensityPressure3(FluidState3 fluid, float h, List<int> nbrs)
    {
        var cfg = fluid.Config;
        var boundary = fluid.Boundary;
        var particles = fluid.Particles;
        for (int bi = 0; bi < boundary.Count; bi++)
        {
            Vector3 bp = boundary[bi].Position;
            FluidNear3(fluid, bp, nbrs);
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
    internal static void Resample3(FluidState3 fluid, float h)
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
            var kept = new List<FluidParticle3>(n);
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

        var output = new List<FluidParticle3>(particles.Count);
        var off = new Vector3(0.25f * h, 0.0f, 0.0f);
        foreach (var p in particles)
        {
            if (p.Density < splitLo && p.Mass > minMass && output.Count + 1 < FluidState3.MaxParticles)
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
    internal static void Substep3(
        FluidState3 fluid,
        List<RigidBody3> bodies,
        Vector3 gravity,
        AmbientConfig3 ambient,
        float time,
        float dt)
    {
        var cfg = fluid.Config;
        float h = MathF.Max(cfg.SmoothingLength, Eps);
        var particles = fluid.Particles;

        RebuildGrid3(fluid);

        var nbrs = new List<int>();
        ComputeDensityPressure3(fluid, h, nbrs);

        if (cfg.AdaptiveResolution)
        {
            Resample3(fluid, h);
            RebuildGrid3(fluid);
            ComputeDensityPressure3(fluid, h, nbrs);
        }

        if (cfg.BoundaryParticles)
        {
            ulong fp = BoundaryFingerprint3(cfg, bodies);
            if (fluid.BoundaryRevision != fp || fluid.Boundary.Count == 0)
            {
                RefreshBoundaries3(fluid, bodies, h * 0.5f);
                RebuildBoundaryGrid3(fluid, h);
                fluid.BoundaryRevision = fp;
                fluid.BoundaryRegenCount++;
            }
            ComputeBoundaryDensityPressure3(fluid, h, nbrs);
        }

        float cs = MathF.Sqrt(MathF.Max(cfg.Stiffness, 0.0f));
        fluid.Forces.Clear();
        for (int k = 0; k < particles.Count; k++)
        {
            fluid.Forces.Add(Vector3.Zero);
        }
        var bnbrs = new List<int>();
        var boundary = fluid.Boundary;
        for (int i = 0; i < particles.Count; i++)
        {
            Neighbors3(fluid, i, nbrs);
            Vector3 pi = particles[i].Position;
            Vector3 vi = particles[i].Velocity;
            float rhoI = MathF.Max(particles[i].Density, Eps);
            float pI = particles[i].Pressure;
            Vector3 f = Vector3.Zero;
            foreach (int j in nbrs)
            {
                if (j == i)
                {
                    continue;
                }
                var pj = particles[j];
                float rhoJ = MathF.Max(pj.Density, Eps);
                Vector3 delta = pi - pj.Position;
                float dist = delta.Length();
                Vector3 grad = SpikyGrad(delta, dist, h);
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
                BoundaryNear3(fluid, pi, bnbrs);
                foreach (int bk in bnbrs)
                {
                    var bp = boundary[bk];
                    Vector3 delta = pi - bp.Position;
                    float dist = delta.Length();
                    Vector3 grad = SpikyGrad(delta, dist, h);
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
            Neighbors3(fluid, i, nbrs);
            float rhoI = MathF.Max(particles[i].Density, Eps);
            Vector3 acc = gravity + fluid.Forces[i] / rhoI;
            if (turbOn)
            {
                Vector3 q = particles[i].Position * turb.Frequency;
                float t = time * turb.TimeScale;
                Vector3 nv = turb.DivergenceFree
                    ? TurbulenceNoise.Curl3(q.X, q.Y, q.Z, t, turb.Seed, turb.Octaves)
                    : new Vector3(
                        TurbulenceNoise.Fbm3(q.X, q.Y, q.Z, t, turb.Seed, turb.Octaves),
                        TurbulenceNoise.Fbm3(q.X, q.Y, q.Z, t, turb.Seed ^ 0x55aa55aa, turb.Octaves),
                        TurbulenceNoise.Fbm3(q.X, q.Y, q.Z, t, turb.Seed ^ 0xaa55aa55, turb.Octaves));
                if (nv.LengthSquared() > 1.0f)
                {
                    nv /= nv.Length();
                }
                acc += nv * turb.Amplitude;
            }
            Vector3 v = particles[i].Velocity + acc * dt;
            if (cfg.Viscosity > 0.0f)
            {
                Vector3 ppos = particles[i].Position;
                Vector3 vi0 = particles[i].Velocity;
                Vector3 smooth = Vector3.Zero;
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
            if (!float.IsFinite(v.X) || !float.IsFinite(v.Y) || !float.IsFinite(v.Z))
            {
                v = Vector3.Zero;
            }
            var pp = particles[i];
            pp.Velocity = v;
            Vector3 pos = pp.Position + v * dt;
            if (!float.IsFinite(pos.X) || !float.IsFinite(pos.Y) || !float.IsFinite(pos.Z))
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
                Vector3 ppos = particles[i].Position;
                var hit = CollideParticleBody3(fluid, ppos, pr, body);
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
                float vn = Vector3.Dot(pp.Velocity, normal);
                if (vn < 0.0f)
                {
                    pp.Velocity -= normal * vn;
                    body.ApplyImpulse(normal * (vn * pp.Mass), point);
                    if (cfg.WallFriction > 0.0f)
                    {
                        float mu = System.Math.Clamp(cfg.WallFriction, 0.0f, 1.0f);
                        Vector3 vt = pp.Velocity;
                        Vector3 wallV = body.Velocity;
                        Vector3 wallVt = wallV - normal * Vector3.Dot(wallV, normal);
                        Vector3 deltaV = (wallVt - vt) * mu;
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

        RebuildGrid3(fluid);
        foreach (var body in bodies)
        {
            if (body.IsStatic)
            {
                continue;
            }
            SurfaceSamples3(fluid, body);
            if (fluid.SampleScratch.Count == 0)
            {
                continue;
            }
            float drag = MathF.Max(fluid.Config.Drag, 0.0f);
            bool wet = false;
            foreach (var (point, normal, area) in fluid.SampleScratch)
            {
                float p = SamplePressure3(fluid, point);
                if (p <= 0.0f)
                {
                    continue;
                }
                wet = true;
                Vector3 r = point - body.Transform.Position;
                Vector3 vRel = body.Velocity + Vector3.Cross(body.AngularVelocity, r) - ambient.Drag.Wind;
                float spd = vRel.Length();
                Vector3 force = -normal * (p * area);
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
