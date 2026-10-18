using System.Numerics;
using Shape2Polygon = PhiEng.Dynamics.PolygonShape;
using Shape2Circle = PhiEng.Dynamics.CircleShape;
using Shape2Box = PhiEng.Dynamics.BoxShape2;

namespace PhiEng.Dynamics;

/// <summary>
/// Ambient force kernels + velocity kicks for the managed engine:
/// spring (SHM), aerodynamic drag, and deterministic turbulence. Mirrors
/// Rust <c>phieng_dynamics::ambient</c>; cross-language bit-equality is
/// NOT claimed (noise uses different float ops) — agreement is asserted
/// within tolerances, and the noise field is statistically equivalent.
/// </summary>
public static class AmbientIntegration
{
    private const float Eps = 1e-6f;

    private static float DragArea2(RigidBody2 b)
    {
        return b.Shape switch
        {
            Shape2Circle c => 2.0f * MathF.Max(c.Radius, 0.0f),
            Shape2Box box => 2.0f * (MathF.Abs(box.HalfExtents.X) + MathF.Abs(box.HalfExtents.Y)),
            Shape2Polygon p => PolygonArea(p),
            _ => 1.0f,
        };
    }

    private static float PolygonArea(Shape2Polygon p)
    {
        float r2 = 0.0f;
        foreach (var v in p.Vertices)
        {
            r2 = MathF.Max(r2, v.LengthSquared());
        }
        return 2.0f * MathF.Sqrt(r2);
    }

    private static float DragArea3(RigidBody3 b)
    {
        return b.Shape switch
        {
            SphereShape3 s => MathF.PI * MathF.Max(s.Radius, 0.0f) * MathF.Max(s.Radius, 0.0f),
            BoxShape3 box => 4.0f * (MathF.Abs(box.HalfExtents.X) * MathF.Abs(box.HalfExtents.Y)
                + MathF.Abs(box.HalfExtents.Y) * MathF.Abs(box.HalfExtents.Z)
                + MathF.Abs(box.HalfExtents.Z) * MathF.Abs(box.HalfExtents.X)) / 3.0f,
            CapsuleShape3 c => 4.0f * MathF.Max(c.Radius, 0.0f) * MathF.Max(c.HalfLength, 0.0f)
                + MathF.PI * MathF.Max(c.Radius, 0.0f) * MathF.Max(c.Radius, 0.0f),
            ConvexHullShape3 h => HullArea(h),
            _ => 1.0f,
        };
    }

    private static float HullArea(ConvexHullShape3 h)
    {
        float r2 = 0.0f;
        foreach (var v in h.Vertices)
        {
            r2 = MathF.Max(r2, v.LengthSquared());
        }
        return MathF.PI * r2;
    }

    private static Vector2 ClampLengthMax(Vector2 v, float max)
    {
        float len = v.Length();
        return len > max && len > Eps ? v * (max / len) : v;
    }

    private static Vector3 ClampLengthMax(Vector3 v, float max)
    {
        float len = v.Length();
        return len > max && len > Eps ? v * (max / len) : v;
    }

    /// <summary>
    /// Apply the 2D ambient kick (spring + drag + turbulence) over
    /// interval <paramref name="h"/>. Skips static bodies and bodies with
    /// <c>AmbientScale == 0</c>. Deterministic, allocation-free, index order.
    /// </summary>
    public static void Kick2(List<RigidBody2> bodies, AmbientConfig2 cfg, float time, float h)
    {
        bool springOn = cfg.Spring.Enabled;
        bool dragOn = cfg.Drag.Enabled;
        bool turbOn = cfg.Turbulence.Enabled;
        if (!springOn && !dragOn && !turbOn) return;

        foreach (var b in bodies)
        {
            if (b.IsStatic || b.IsSleeping || b.AmbientScale == 0.0f) continue;
            float scale = b.AmbientScale;
            Vector2 dv = Vector2.Zero;

            if (springOn)
            {
                Vector2 sol = b.Transform.Position - cfg.Spring.Anchor;
                bool within = cfg.Spring.MaxDisplacement <= 0.0f
                    || sol.LengthSquared() <= cfg.Spring.MaxDisplacement * cfg.Spring.MaxDisplacement;
                if (within)
                {
                    Vector2 f = sol * (-cfg.Spring.Stiffness) - b.Velocity * cfg.Spring.Damping;
                    dv += f * (b.InvMass * scale * h);
                }
            }

            if (dragOn)
            {
                float aw = -cfg.Drag.Angular * b.AngularVelocity * b.InvInertia * scale * h;
                if (float.IsFinite(aw)) b.AngularVelocity += aw;
                Vector2 vRel = b.Velocity - cfg.Drag.Wind;
                float speed = vRel.Length();
                Vector2 f = vRel * (-cfg.Drag.Linear);
                if (speed > Eps)
                {
                    float area = DragArea2(b);
                    f -= vRel * (cfg.Drag.Quadratic * cfg.Drag.AirDensity * area * speed);
                }
                dv += f * (b.InvMass * scale * h);
            }

            if (turbOn)
            {
                Vector2 p = b.Transform.Position * cfg.Turbulence.Frequency;
                float t = time * cfg.Turbulence.TimeScale;
                Vector2 n = cfg.Turbulence.DivergenceFree
                    ? TurbulenceNoise.Curl2(p.X, p.Y, t, cfg.Turbulence.Seed, cfg.Turbulence.Octaves)
                    : new Vector2(
                        TurbulenceNoise.Fbm2(p.X, p.Y, t, cfg.Turbulence.Seed, cfg.Turbulence.Octaves),
                        TurbulenceNoise.Fbm2(p.X, p.Y, t, cfg.Turbulence.Seed ^ 0x55aa55aau, cfg.Turbulence.Octaves));
                n = ClampLengthMax(n, 1.0f);
                dv += n * (cfg.Turbulence.Amplitude * scale * h);
            }

            if (float.IsFinite(dv.X) && float.IsFinite(dv.Y)) b.Velocity += dv;
        }
    }

    /// <summary>3D counterpart of <see cref="Kick2"/>.</summary>
    public static void Kick3(List<RigidBody3> bodies, AmbientConfig3 cfg, float time, float h)
    {
        bool springOn = cfg.Spring.Enabled;
        bool dragOn = cfg.Drag.Enabled;
        bool turbOn = cfg.Turbulence.Enabled;
        if (!springOn && !dragOn && !turbOn) return;

        foreach (var b in bodies)
        {
            if (b.IsStatic || b.IsSleeping || b.AmbientScale == 0.0f) continue;
            float scale = b.AmbientScale;
            Vector3 dv = Vector3.Zero;

            if (springOn)
            {
                Vector3 sol = b.Transform.Position - cfg.Spring.Anchor;
                bool within = cfg.Spring.MaxDisplacement <= 0.0f
                    || sol.LengthSquared() <= cfg.Spring.MaxDisplacement * cfg.Spring.MaxDisplacement;
                if (within)
                {
                    Vector3 f = sol * (-cfg.Spring.Stiffness) - b.Velocity * cfg.Spring.Damping;
                    dv += f * (b.InvMass * scale * h);
                }
            }

            if (dragOn)
            {
                Vector3 spin = b.InvInertia.TransformVector(b.AngularVelocity);
                Vector3 aw = -(cfg.Drag.Angular * scale * h) * spin;
                if (float.IsFinite(aw.X) && float.IsFinite(aw.Y) && float.IsFinite(aw.Z))
                {
                    b.AngularVelocity += aw;
                }
                Vector3 vRel = b.Velocity - cfg.Drag.Wind;
                float speed = vRel.Length();
                Vector3 f = vRel * (-cfg.Drag.Linear);
                if (speed > Eps)
                {
                    float area = DragArea3(b);
                    f -= vRel * (cfg.Drag.Quadratic * cfg.Drag.AirDensity * area * speed);
                }
                dv += f * (b.InvMass * scale * h);
            }

            if (turbOn)
            {
                Vector3 p = b.Transform.Position * cfg.Turbulence.Frequency;
                float t = time * cfg.Turbulence.TimeScale;
                Vector3 n = cfg.Turbulence.DivergenceFree
                    ? TurbulenceNoise.Curl3(p.X, p.Y, p.Z, t, cfg.Turbulence.Seed, cfg.Turbulence.Octaves)
                    : new Vector3(
                        TurbulenceNoise.Fbm3(p.X, p.Y, p.Z, t, cfg.Turbulence.Seed, cfg.Turbulence.Octaves),
                        TurbulenceNoise.Fbm3(p.X, p.Y, p.Z, t, cfg.Turbulence.Seed ^ 0x55aa55aau, cfg.Turbulence.Octaves),
                        TurbulenceNoise.Fbm3(p.X, p.Y, p.Z, t, cfg.Turbulence.Seed ^ 0xaa55aa55u, cfg.Turbulence.Octaves));
                n = ClampLengthMax(n, 1.0f);
                dv += n * (cfg.Turbulence.Amplitude * scale * h);
            }

            if (float.IsFinite(dv.X) && float.IsFinite(dv.Y) && float.IsFinite(dv.Z)) b.Velocity += dv;
        }
    }
}
