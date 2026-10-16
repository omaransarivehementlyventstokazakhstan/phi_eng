using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Gravitational force kernels + velocity kicks for the managed engine.
/// Mirrors <c>phieng_dynamics::gravity</c> op-for-op where feasible;
/// cross-language bit-equality is NOT claimed (transcendentals may
/// differ) — managed/native agreement is asserted within tolerances.
/// </summary>
public static class GravityIntegration
{
    private const float EpsSq = 1e-12f;

    /// <summary>Acceleration of a point toward a central attractor of strength mu.</summary>
    public static Vector2 CentralAccel2(Vector2 pos, Vector2 attractor, float mu, float softening)
    {
        Vector2 d = attractor - pos;
        float d2 = d.LengthSquared() + softening * softening;
        if (d2 < EpsSq) return Vector2.Zero;
        float inv = 1.0f / MathF.Sqrt(d2);
        return d * (mu * inv / d2);
    }

    /// <summary>3D counterpart of <see cref="CentralAccel2"/>.</summary>
    public static Vector3 CentralAccel3(Vector3 pos, Vector3 attractor, float mu, float softening)
    {
        Vector3 d = attractor - pos;
        float d2 = d.LengthSquared() + softening * softening;
        if (d2 < EpsSq) return Vector3.Zero;
        float inv = 1.0f / MathF.Sqrt(d2);
        return d * (mu * inv / d2);
    }

    /// <summary>
    /// Symmetric pairwise accelerations (Newton's third law): unscaled,
    /// caller applies per-body scale. Zero for coincident pairs.
    /// </summary>
    public static (Vector2 Ai, Vector2 Aj) PairAccelerations2(
        Vector2 pi, float mi, Vector2 pj, float mj, float bigG, float softening)
    {
        Vector2 delta = pj - pi;
        float d2 = delta.LengthSquared() + softening * softening;
        if (d2 < EpsSq) return (Vector2.Zero, Vector2.Zero);
        float inv = 1.0f / MathF.Sqrt(d2);
        float s = bigG * inv / d2;
        return (delta * (s * mj), delta * (-(s * mi)));
    }

    /// <summary>3D counterpart of <see cref="PairAccelerations2"/>.</summary>
    public static (Vector3 Ai, Vector3 Aj) PairAccelerations3(
        Vector3 pi, float mi, Vector3 pj, float mj, float bigG, float softening)
    {
        Vector3 delta = pj - pi;
        float d2 = delta.LengthSquared() + softening * softening;
        if (d2 < EpsSq) return (Vector3.Zero, Vector3.Zero);
        float inv = 1.0f / MathF.Sqrt(d2);
        float s = bigG * inv / d2;
        return (delta * (s * mj), delta * (-(s * mi)));
    }

    /// <summary>
    /// Gravity velocity kick over 2D bodies: uniform background field
    /// plus Central/N-body terms per <paramref name="cfg"/>, times
    /// per-body scale, over interval <paramref name="h"/>. When
    /// <paramref name="drag"/> is set (Verlet mode), linear damping
    /// rides the kick as symmetric drag. Index order, deterministic.
    /// Statics never move but still exert by stored mass.
    /// </summary>
    public static void Kick2(List<RigidBody2> bodies, GravityConfig2 cfg, Vector2 uniform, float h, bool drag)
    {
        foreach (var b in bodies)
        {
            if (b.IsStatic) continue;
            Vector2 dv = uniform * (b.GravityScale * h);
            if (drag) dv -= b.Velocity * (b.LinearDamping * h);
            b.Velocity += dv;
        }
        switch (cfg.Mode)
        {
            case GravityMode.Uniform:
                break;
            case GravityMode.Central:
                foreach (var b in bodies)
                {
                    if (b.IsStatic) continue;
                    Vector2 a = CentralAccel2(b.Transform.Position, cfg.Attractor, cfg.AttractorMu, cfg.Softening);
                    Vector2 dv = a * (b.GravityScale * h);
                    if (drag) dv -= b.Velocity * (b.LinearDamping * h);
                    b.Velocity += dv;
                }
                break;
            case GravityMode.NBody:
                if (cfg.BigG != 0.0f)
                {
                    for (int i = 0; i < bodies.Count; i++)
                    {
                        for (int j = i + 1; j < bodies.Count; j++)
                        {
                            var bi = bodies[i];
                            var bj = bodies[j];
                            var (ai, aj) = PairAccelerations2(
                                bi.Transform.Position, bi.Mass,
                                bj.Transform.Position, bj.Mass,
                                cfg.BigG, cfg.Softening);
                            if (!bi.IsStatic)
                            {
                                Vector2 dv = ai * (bi.GravityScale * h);
                                if (drag) dv -= bi.Velocity * (bi.LinearDamping * h);
                                bi.Velocity += dv;
                            }
                            if (!bj.IsStatic)
                            {
                                Vector2 dv = aj * (bj.GravityScale * h);
                                if (drag) dv -= bj.Velocity * (bj.LinearDamping * h);
                                bj.Velocity += dv;
                            }
                        }
                    }
                }
                break;
            case GravityMode.BarnesHut:
                if (cfg.BigG != 0.0f)
                    BarnesHut.ApplyForces2(bodies, cfg.BigG, cfg.Softening, cfg.Theta, h, drag);
                break;
        }
    }

    /// <summary>3D counterpart of <see cref="Kick2"/>.</summary>
    public static void Kick3(List<RigidBody3> bodies, GravityConfig3 cfg, Vector3 uniform, float h, bool drag)
    {
        foreach (var b in bodies)
        {
            if (b.IsStatic) continue;
            Vector3 dv = uniform * (b.GravityScale * h);
            if (drag) dv -= b.Velocity * (b.LinearDamping * h);
            b.Velocity += dv;
        }
        switch (cfg.Mode)
        {
            case GravityMode.Uniform:
                break;
            case GravityMode.Central:
                foreach (var b in bodies)
                {
                    if (b.IsStatic) continue;
                    Vector3 a = CentralAccel3(b.Transform.Position, cfg.Attractor, cfg.AttractorMu, cfg.Softening);
                    Vector3 dv = a * (b.GravityScale * h);
                    if (drag) dv -= b.Velocity * (b.LinearDamping * h);
                    b.Velocity += dv;
                }
                break;
            case GravityMode.NBody:
                if (cfg.BigG != 0.0f)
                {
                    for (int i = 0; i < bodies.Count; i++)
                    {
                        for (int j = i + 1; j < bodies.Count; j++)
                        {
                            var bi = bodies[i];
                            var bj = bodies[j];
                            var (ai, aj) = PairAccelerations3(
                                bi.Transform.Position, bi.Mass,
                                bj.Transform.Position, bj.Mass,
                                cfg.BigG, cfg.Softening);
                            if (!bi.IsStatic)
                            {
                                Vector3 dv = ai * (bi.GravityScale * h);
                                if (drag) dv -= bi.Velocity * (bi.LinearDamping * h);
                                bi.Velocity += dv;
                            }
                            if (!bj.IsStatic)
                            {
                                Vector3 dv = aj * (bj.GravityScale * h);
                                if (drag) dv -= bj.Velocity * (bj.LinearDamping * h);
                                bj.Velocity += dv;
                            }
                        }
                    }
                }
                break;
            case GravityMode.BarnesHut:
                if (cfg.BigG != 0.0f)
                    BarnesHut.ApplyForces3(bodies, cfg.BigG, cfg.Softening, cfg.Theta, h, drag);
                break;
        }
    }
}
