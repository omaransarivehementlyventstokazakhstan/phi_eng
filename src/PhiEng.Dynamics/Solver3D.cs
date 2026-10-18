using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// 3D contact constraint: normal + 2 friction tangents, accumulated impulses.
/// A struct so constraints live inline in their list (no per-contact heap
/// object). The solver accesses them by index with write-back.
/// Mirrors Rust <c>ContactConstraint3D</c>.
/// </summary>
public struct ContactConstraint3D
{
    /// <summary>First body id.</summary>
    public int BodyA { get; set; }
    /// <summary>Second body id.</summary>
    public int BodyB { get; set; }
    /// <summary>Contact point in world space.</summary>
    public Vector3 Point { get; set; }
    /// <summary>Contact normal (from A to B, unit length).</summary>
    public Vector3 Normal { get; set; }
    /// <summary>First friction tangent (unit, perpendicular to normal).</summary>
    public Vector3 Tangent1 { get; set; }
    /// <summary>Second friction tangent (unit, perpendicular to both).</summary>
    public Vector3 Tangent2 { get; set; }
    /// <summary>Penetration depth.</summary>
    public float Penetration { get; set; }
    /// <summary>Friction coefficient (mu).</summary>
    public float Friction { get; set; }
    /// <summary>Restitution (bounciness).</summary>
    public float Restitution { get; set; }
    /// <summary>Accumulated normal impulse j_n (persisted via contact cache).</summary>
    public float NormalImpulse { get; set; }
    /// <summary>Accumulated first-tangent impulse (Coulomb-clamped).</summary>
    public float Tangent1Impulse { get; set; }
    /// <summary>Accumulated second-tangent impulse (Coulomb-clamped).</summary>
    public float Tangent2Impulse { get; set; }
    /// <summary>Restitution bias velocity for this step (pre-solve, never cached).</summary>
    public float RestBias { get; set; }
    /// <summary>Effective mass along the normal, snapshotted once per step.</summary>
    public float MassNormal { get; set; }
    /// <summary>Effective mass along the first tangent, snapshotted once per step.</summary>
    public float MassTangent1 { get; set; }
    /// <summary>Effective mass along the second tangent, snapshotted once per step.</summary>
    public float MassTangent2 { get; set; }
    /// <summary>
    /// World-frame inverse inertias, snapshotted once per step so the
    /// per-iteration impulse applications don't rebuild R·I⁻¹·Rᵀ.
    /// </summary>
    public Matrix3x3 InvInertiaWorldA { get; set; }
    /// <summary>World-frame inverse inertia of body B, snapshotted once per step.</summary>
    public Matrix3x3 InvInertiaWorldB { get; set; }

    /// <summary>Builds a contact with tangents derived from the normal and zeroed impulses.</summary>
    public ContactConstraint3D(int a, int b, Vector3 point, Vector3 normal,
        float penetration, float friction, float restitution)
    {
        BodyA = a; BodyB = b; Point = point; Normal = normal;
        Penetration = penetration; Friction = friction; Restitution = restitution;
        Tangent1 = MathF.Abs(normal.X) < 0.9f
            ? Vector3.Normalize(Vector3.Cross(Vector3.UnitX, normal))
            : Vector3.Normalize(Vector3.Cross(Vector3.UnitY, normal));
        Tangent2 = Vector3.Cross(normal, Tangent1);
    }
}

/// <summary>
/// 3D sequential impulse solver with Coulomb friction (2 tangents).
/// Mirrors Rust <c>Solver3D</c>.
/// </summary>
public class Solver3D
{
    /// <summary>
    /// Maximum Baumgarte bias velocity (m/s). Mirrors Rust
    /// <c>Solver3D::MAX_BIAS_VELOCITY</c>.
    /// </summary>
    public const float MaxBiasVelocity = 2.0f;
    /// <summary>Maximum solver iterations per step.</summary>
    public int Iterations { get; set; } = 8;
    /// <summary>Bauman correction factor (penetration slop).</summary>
    public float Slop { get; set; } = 0.005f;
    /// <summary>Baumgarte stabilization factor.</summary>
    public float Baumgarte { get; set; } = 0.2f;
    /// <summary>Early-exit tolerance. Mirrors Rust <c>Solver3D</c>.</summary>
    public float EarlyExitTolerance { get; set; } = 1e-6f;
    /// <summary>
    /// Solve the tangential (Coulomb friction) pass. Default true;
    /// disabling halves solve cost (set via <see cref="SolverQuality.Low"/>).
    /// </summary>
    public bool EnableFriction { get; set; } = true;
    /// <summary>
    /// Apply restitution bounces. Default true; disabling keeps the
    /// Baumgarte positional correction (set via <see cref="SolverQuality.Low"/>).
    /// </summary>
    public bool EnableRestitution { get; set; } = true;

    /// <summary>
    /// Apply a quality tier preset (iteration budget plus friction and
    /// restitution switches). <see cref="SolverQuality.Ultra"/> restores
    /// the default configuration.
    /// </summary>
    public void ApplyQuality(SolverQuality quality)
    {
        Iterations = quality switch
        {
            SolverQuality.Ultra => 8,
            SolverQuality.High => 6,
            SolverQuality.Medium => 4,
            SolverQuality.Low => 2,
            _ => 1,
        };
        EnableFriction = quality <= SolverQuality.Medium;
        EnableRestitution = quality <= SolverQuality.Medium;
    }

    /// <summary>Returns the iterations actually run.</summary>
    public int Solve(List<RigidBody3> bodies, List<ContactConstraint3D> constraints, float dt)
    {
        for (int i = 0; i < constraints.Count; i++)
        {
            var c = constraints[i];
            if (BothFrozen(bodies, c.BodyA, c.BodyB))
            {
                c.NormalImpulse = 0.0f;
                c.Tangent1Impulse = 0.0f;
                c.Tangent2Impulse = 0.0f;
                c.RestBias = 0.0f;
                constraints[i] = c;
                continue;
            }
            ComputeMassTerms(bodies, ref c);
            if (!EnableFriction)
            {
                c.Tangent1Impulse = 0.0f;
                c.Tangent2Impulse = 0.0f;
            }
            ApplyWarmStarting(bodies, c);
            c.RestBias = EnableRestitution ? RestitutionBias(bodies, c) : 0.0f;
            constraints[i] = c;
        }
        int ran = 0;
        for (int i = 0; i < Iterations; i++)
        {
            ran++;
            float maxApplied = 0.0f;
            for (int j = 0; j < constraints.Count; j++)
            {
                var c = constraints[j];
                if (BothFrozen(bodies, c.BodyA, c.BodyB)) continue;
                maxApplied = MathF.Max(maxApplied, SolveNormal(bodies, ref c, dt));
                constraints[j] = c;
            }
            if (EnableFriction)
            {
                for (int j = 0; j < constraints.Count; j++)
                {
                    var c = constraints[j];
                    if (BothFrozen(bodies, c.BodyA, c.BodyB)) continue;
                    maxApplied = MathF.Max(maxApplied, SolveFriction(bodies, ref c));
                    constraints[j] = c;
                }
            }
            if (maxApplied < EarlyExitTolerance)
                break;
        }
        return ran;
    }

    private static bool BothFrozen(List<RigidBody3> bodies, int a, int b)
    {
        var ba = bodies[a];
        var bb = bodies[b];
        return (ba.IsStatic || ba.IsSleeping) && (bb.IsStatic || bb.IsSleeping);
    }

    private static float EffectiveMass(
        RigidBody3 a, RigidBody3 b, Vector3 ra, Vector3 rb, Vector3 dir,
        Matrix3x3 invInertiaWA, Matrix3x3 invInertiaWB)
    {
        var raCross = Vector3.Cross(ra, dir);
        var rbCross = Vector3.Cross(rb, dir);
        // World-frame tensors: lever arms and directions are world-space.
        // Frozen bodies (static or sleeping) count as infinite mass.
        float ima = (a.IsStatic || a.IsSleeping) ? 0.0f : a.InvMass;
        float imb = (b.IsStatic || b.IsSleeping) ? 0.0f : b.InvMass;
        float k = ima + imb
            + Vector3.Dot(invInertiaWA.TransformVector(raCross), raCross)
            + Vector3.Dot(invInertiaWB.TransformVector(rbCross), rbCross);
        return k > 0 ? 1.0f / k : 0.0f;
    }

    private static void ComputeMassTerms(List<RigidBody3> bodies, ref ContactConstraint3D c)
    {
        var a = bodies[c.BodyA];
        var b = bodies[c.BodyB];
        var ra = c.Point - a.Transform.Position;
        var rb = c.Point - b.Transform.Position;
        c.InvInertiaWorldA = (a.IsStatic || a.IsSleeping) ? Matrix3x3.Zero : a.WorldInvInertia();
        c.InvInertiaWorldB = (b.IsStatic || b.IsSleeping) ? Matrix3x3.Zero : b.WorldInvInertia();
        c.MassNormal = EffectiveMass(a, b, ra, rb, c.Normal, c.InvInertiaWorldA, c.InvInertiaWorldB);
        c.MassTangent1 = EffectiveMass(a, b, ra, rb, c.Tangent1, c.InvInertiaWorldA, c.InvInertiaWorldB);
        c.MassTangent2 = EffectiveMass(a, b, ra, rb, c.Tangent2, c.InvInertiaWorldA, c.InvInertiaWorldB);
    }

    private static void ApplyWarmStarting(List<RigidBody3> bodies, ContactConstraint3D c)
    {
        if (c.NormalImpulse == 0 && c.Tangent1Impulse == 0 && c.Tangent2Impulse == 0) return;
        var p = c.Normal * c.NormalImpulse + c.Tangent1 * c.Tangent1Impulse + c.Tangent2 * c.Tangent2Impulse;
        var ra = c.Point - bodies[c.BodyA].Transform.Position;
        var rb = c.Point - bodies[c.BodyB].Transform.Position;
        ApplyWarmImpulse(bodies, c, ra, rb, p);
    }

    private float SolveNormal(List<RigidBody3> bodies, ref ContactConstraint3D c, float dt)
    {
        var a = bodies[c.BodyA];
        var b = bodies[c.BodyB];
        var ra = c.Point - a.Transform.Position;
        var rb = c.Point - b.Transform.Position;
        var va = a.Velocity + Vector3.Cross(a.AngularVelocity, ra);
        var vb = b.Velocity + Vector3.Cross(b.AngularVelocity, rb);
        float vn = Vector3.Dot(vb - va, c.Normal);
        float bias = c.RestBias > 0 ? 0
            : dt > 0 ? MathF.Min(Baumgarte * MathF.Max(0, c.Penetration - Slop) / dt, MaxBiasVelocity) : 0;
        float lambda = c.MassNormal * (c.RestBias - vn + bias);
        float old = c.NormalImpulse;
        c.NormalImpulse = MathF.Max(0, old + lambda);
        float applied = c.NormalImpulse - old;
        if (applied != 0)
            ApplyImpulse(bodies, c, ra, rb, c.Normal * applied);
        return MathF.Abs(applied);
    }

    private static float SolveFriction(List<RigidBody3> bodies, ref ContactConstraint3D c)
    {
        var a = bodies[c.BodyA];
        var b = bodies[c.BodyB];
        var ra = c.Point - a.Transform.Position;
        var rb = c.Point - b.Transform.Position;
        var va = a.Velocity + Vector3.Cross(a.AngularVelocity, ra);
        var vb = b.Velocity + Vector3.Cross(b.AngularVelocity, rb);
        var rel = vb - va;
        float maxF = c.Friction * c.NormalImpulse;

        float vt1 = Vector3.Dot(rel, c.Tangent1);
        float old1 = c.Tangent1Impulse;
        c.Tangent1Impulse = System.Math.Clamp(old1 + c.MassTangent1 * -vt1, -maxF, maxF);
        float ap1 = c.Tangent1Impulse - old1;

        float vt2 = Vector3.Dot(rel, c.Tangent2);
        float old2 = c.Tangent2Impulse;
        c.Tangent2Impulse = System.Math.Clamp(old2 + c.MassTangent2 * -vt2, -maxF, maxF);
        float ap2 = c.Tangent2Impulse - old2;

        if (ap1 != 0 || ap2 != 0)
            ApplyImpulse(bodies, c, ra, rb, c.Tangent1 * ap1 + c.Tangent2 * ap2);
        return MathF.Max(MathF.Abs(ap1), MathF.Abs(ap2));
    }

    private static float RestitutionBias(List<RigidBody3> bodies, ContactConstraint3D c)
    {
        var a = bodies[c.BodyA];
        var b = bodies[c.BodyB];
        var ra = c.Point - a.Transform.Position;
        var rb = c.Point - b.Transform.Position;
        var va = a.Velocity + Vector3.Cross(a.AngularVelocity, ra);
        var vb = b.Velocity + Vector3.Cross(b.AngularVelocity, rb);
        float vn = Vector3.Dot(vb - va, c.Normal);
        return vn < -1.0f ? -c.Restitution * vn : 0.0f;
    }

    private static void ApplyImpulse(
        List<RigidBody3> bodies, int ai, int bi,
        Vector3 ra, Vector3 rb, Vector3 p,
        Matrix3x3 invInertiaWA, Matrix3x3 invInertiaWB)
    {
        // Frozen bodies (static or sleeping) take no impulse, mirroring
        // the infinite-mass mass terms above.
        var a = bodies[ai];
        var b = bodies[bi];
        if (!a.IsStatic && !a.IsSleeping)
        {
            a.Velocity -= p * a.InvMass;
            a.AngularVelocity -= invInertiaWA.TransformVector(Vector3.Cross(ra, p));
        }
        if (!b.IsStatic && !b.IsSleeping)
        {
            b.Velocity += p * b.InvMass;
            b.AngularVelocity += invInertiaWB.TransformVector(Vector3.Cross(rb, p));
        }
    }

    private static void ApplyImpulse(
        List<RigidBody3> bodies, ContactConstraint3D c, Vector3 ra, Vector3 rb, Vector3 p)
        => ApplyImpulse(bodies, c.BodyA, c.BodyB, ra, rb, p, c.InvInertiaWorldA, c.InvInertiaWorldB);

    private static void ApplyWarmImpulse(
        List<RigidBody3> bodies, ContactConstraint3D c, Vector3 ra, Vector3 rb, Vector3 p)
    {
        // Warm starting runs before mass terms exist: use fresh tensors.
        var a = bodies[c.BodyA];
        var b = bodies[c.BodyB];
        ApplyImpulse(bodies, c.BodyA, c.BodyB, ra, rb, p, a.WorldInvInertia(), b.WorldInvInertia());
    }
}
