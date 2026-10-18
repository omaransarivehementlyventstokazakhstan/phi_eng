using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Contact constraint for the solver (2D sequential impulse with Coulomb friction).
/// A struct so constraints live inline in their list (no per-contact heap
/// object). The solver accesses them by index with write-back.
/// Mirrors the Rust <c>ContactConstraint</c> in crates/dynamics/src/solver.rs.
/// </summary>
public struct ContactConstraint
{
    /// <summary>First body id.</summary>
    public int BodyA { get; set; }
    /// <summary>Second body id.</summary>
    public int BodyB { get; set; }
    /// <summary>Contact point in world space.</summary>
    public Vector2 Point { get; set; }
    /// <summary>Contact normal (from A to B, unit length).</summary>
    public Vector2 Normal { get; set; }
    /// <summary>Penetration depth.</summary>
    public float Penetration { get; set; }
    /// <summary>Friction coefficient (mu).</summary>
    public float Friction { get; set; }
    /// <summary>Restitution (bounciness).</summary>
    public float Restitution { get; set; }
    /// <summary>Accumulated normal impulse j_n (persisted via contact cache).</summary>
    public float NormalImpulse { get; set; }
    /// <summary>Accumulated tangent impulse j_t (persisted, Coulomb-clamped).</summary>
    public float TangentImpulse { get; set; }
    /// <summary>
    /// Restitution bias velocity for this step, computed once from the
    /// PRE-solve relative velocity (Box2D-style velocity bias). Must NOT be
    /// recomputed per iteration: after iteration 1 the contact separates,
    /// and a fresh target of 0 would un-apply the bounce within the frame.
    /// Recomputed every step; never cached across frames.
    /// </summary>
    public float RestBias { get; set; }
    /// <summary>
    /// Effective mass along the normal, snapshotted once per step (same
    /// policy as the 3D solver and the Rust side). Recomputed every step;
    /// never cached across frames.
    /// </summary>
    public float MassNormal { get; set; }
    /// <summary>Effective mass along the tangent, snapshotted once per step.</summary>
    public float MassTangent { get; set; }

    /// <summary>Zeroed constraint (impulses start unaccumulated).</summary>
    public ContactConstraint()
    {
        NormalImpulse = 0.0f;
        TangentImpulse = 0.0f;
        RestBias = 0.0f;
        MassNormal = 0.0f;
        MassTangent = 0.0f;
    }

    /// <summary>
    /// Tangent direction: 90-degree CCW rotation of the normal.
    /// Must stay in sync with the Rust <c>ContactConstraint::tangent</c>.
    /// </summary>
    public Vector2 Tangent => new Vector2(-Normal.Y, Normal.X);
}

/// <summary>
/// Solver quality tier: a preset for iteration budget plus friction and
/// restitution switches. Mirrors Rust
/// <c>phieng_dynamics::solver::SolverQuality</c>. Tiers only ever reduce
/// work vs <see cref="SolverQuality.Ultra"/>; Ultra restores defaults.
/// </summary>
public enum SolverQuality
{
    /// <summary>8 iterations, friction, restitution (the default).</summary>
    Ultra,
    /// <summary>6 iterations, friction, restitution.</summary>
    High,
    /// <summary>4 iterations, friction, restitution.</summary>
    Medium,
    /// <summary>2 iterations, no friction, no restitution.</summary>
    Low,
    /// <summary>1 iteration, no friction, no restitution.</summary>
    Minimal,
}

/// <summary>
/// Constraint solver for physics engine.
/// Mirrors the Rust <c>Solver</c> in crates/dynamics/src/solver.rs.
/// </summary>
public class Solver
{
    /// <summary>
    /// Maximum Baumgarte bias velocity (m/s). Mirrors Rust
    /// <c>Solver::MAX_BIAS_VELOCITY</c>: deep overlaps resolve gradually
    /// instead of ejecting past discrete collision.
    /// </summary>
    public const float MaxBiasVelocity = 2.0f;
    /// <summary>Maximum solver iterations per step.</summary>
    public int Iterations { get; set; }
    /// <summary>Bauman correction factor (penetration slop).</summary>
    public float Slop { get; set; }
    /// <summary>Baumgarte stabilization factor.</summary>
    public float Baumgarte { get; set; }
    /// <summary>
    /// Solve disjoint constraint islands in parallel (see <see cref="IslandManager"/>).
    /// Default off: the serial path is the reference behavior. Mirrors Rust.
    /// </summary>
    public bool Parallel { get; set; }
    /// <summary>
    /// Early-exit tolerance: an iteration applying less than this (max
    /// absolute impulse delta) ends the solve. Mirrors Rust.
    /// </summary>
    public float EarlyExitTolerance { get; set; }
    /// <summary>
    /// Solve the tangential (Coulomb friction) pass. Default true;
    /// disabling halves solve cost (set via <see cref="SolverQuality.Low"/>).
    /// </summary>
    public bool EnableFriction { get; set; }
    /// <summary>
    /// Apply restitution bounces. Default true; disabling keeps the
    /// Baumgarte positional correction (set via <see cref="SolverQuality.Low"/>).
    /// </summary>
    public bool EnableRestitution { get; set; }

    /// <summary>Default solver: 8 iterations, slop 0.005, Baumgarte 0.2.</summary>
    public Solver()
    {
        Iterations = 8;
        Slop = 0.005f;
        Baumgarte = 0.2f;
        EarlyExitTolerance = 1e-6f;
        EnableFriction = true;
        EnableRestitution = true;
        Parallel = false;
    }

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

    /// <summary>
    /// Solve all constraints. Each iteration runs a full normal pass over all
    /// contacts before the friction pass, so friction responds to the settled
    /// post-normal velocities rather than half-updated per-contact state.
    /// Returns the iterations actually run (early exit when converged).
    /// </summary>
    public int Solve(List<RigidBody2> bodies, List<ContactConstraint> constraints, float dt)
    {
        // Mass terms + warm start + restitution targets from pre-solve state.
        // With friction off, cached tangent impulses are dropped so no
        // stale sideways push survives. Fully frozen pairs (both ends
        // static or sleeping) carry zero velocity: skip them so sleeping
        // islands cost nothing in the solver.
        for (int i = 0; i < constraints.Count; i++)
        {
            var c = constraints[i];
            if (BothFrozen(bodies, c.BodyA, c.BodyB))
            {
                c.NormalImpulse = 0.0f;
                c.TangentImpulse = 0.0f;
                c.RestBias = 0.0f;
                constraints[i] = c;
                continue;
            }
            ComputeMassTerms(bodies, ref c);
            if (!EnableFriction)
            {
                c.TangentImpulse = 0.0f;
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

    /// <summary>
    /// Snapshot per-step effective masses from step-start state.
    /// </summary>
    private static void ComputeMassTerms(List<RigidBody2> bodies, ref ContactConstraint constraint)
    {
        var bodyA = bodies[constraint.BodyA];
        var bodyB = bodies[constraint.BodyB];
        float ima = (bodyA.IsStatic || bodyA.IsSleeping) ? 0.0f : bodyA.InvMass;
        float imb = (bodyB.IsStatic || bodyB.IsSleeping) ? 0.0f : bodyB.InvMass;
        float iia = (bodyA.IsStatic || bodyA.IsSleeping) ? 0.0f : bodyA.InvInertia;
        float iib = (bodyB.IsStatic || bodyB.IsSleeping) ? 0.0f : bodyB.InvInertia;
        Vector2 rA = constraint.Point - bodyA.Transform.Position;
        Vector2 rB = constraint.Point - bodyB.Transform.Position;
        constraint.MassNormal = EffectiveMass(
            ima, imb,
            iia, iib,
            rA, rB, constraint.Normal);
        constraint.MassTangent = EffectiveMass(
            ima, imb,
            iia, iib,
            rA, rB, constraint.Tangent);
    }

    /// <summary>
    /// True when both constraint ends are frozen (static or sleeping):
    /// zero velocity by construction, so the pair needs no solve work.
    /// </summary>
    private static bool BothFrozen(List<RigidBody2> bodies, int a, int b)
    {
        var ba = bodies[a];
        var bb = bodies[b];
        return (ba.IsStatic || ba.IsSleeping) && (bb.IsStatic || bb.IsSleeping);
    }

    /// <summary>
    /// Parallel island solve: per-island index lists in canonical order,
    /// global early-exit shared through a per-iteration barrier (bit-exact
    /// vs serial — per-island early exit would diverge). Islands are
    /// body-disjoint, and C# reference-type bodies make cross-thread
    /// mutation of distinct objects memory-safe by construction.
    /// </summary>
    public int SolveParallel(
        List<RigidBody2> bodies,
        List<ContactConstraint> constraints,
        List<List<int>> islandConstraints,
        float dt)
    {
        // Mass terms + warm start + restitution targets (serial pre-loop,
        // exactly as Solve — the threaded section below covers iterations).
        for (int i = 0; i < constraints.Count; i++)
        {
            var c = constraints[i];
            ComputeMassTerms(bodies, ref c);
            if (!EnableFriction)
            {
                c.TangentImpulse = 0.0f;
            }
            ApplyWarmStarting(bodies, c);
            c.RestBias = EnableRestitution ? RestitutionBias(bodies, c) : 0.0f;
            constraints[i] = c;
        }
        if (islandConstraints.Count <= 1)
        {
            var all = new List<int>(constraints.Count);
            for (int i = 0; i < constraints.Count; i++) all.Add(i);
            return SolveIndexed(bodies, constraints, all, dt);
        }

        int n = islandConstraints.Count;
        var maxSlots = new float[n];
        using var barrier = new System.Threading.Barrier(n);
        int ranMax = 0;
        object ranLock = new();
        System.Threading.Tasks.Parallel.For(0, n, ii =>
        {
            var list = islandConstraints[ii];
            int ran = 0;
            for (int it = 0; it < Iterations; it++)
            {
                ran++;
                float localMax = 0.0f;
                foreach (int ci in list)
                {
                    var c = constraints[ci];
                    if (BothFrozen(bodies, c.BodyA, c.BodyB)) continue;
                    localMax = MathF.Max(localMax, SolveNormal(bodies, ref c, dt));
                    constraints[ci] = c;
                }
                if (EnableFriction)
                {
                    foreach (int ci in list)
                    {
                        var c = constraints[ci];
                        if (BothFrozen(bodies, c.BodyA, c.BodyB)) continue;
                        localMax = MathF.Max(localMax, SolveFriction(bodies, ref c));
                        constraints[ci] = c;
                    }
                }
                maxSlots[ii] = localMax;
                barrier.SignalAndWait();
                float globalMax = 0.0f;
                for (int k = 0; k < n; k++)
                    globalMax = MathF.Max(globalMax, maxSlots[k]);
                if (globalMax < EarlyExitTolerance)
                    break;
            }
            lock (ranLock)
            {
                if (ran > ranMax) ranMax = ran;
            }
        });
        return ranMax;
    }

    /// <summary>
    /// Iteration core over an explicit constraint index subset (stable
    /// order). <see cref="Solve(List{RigidBody2}, List{ContactConstraint}, float)"/>
    /// is this with 0..count. Used by single-island fast paths.
    /// </summary>
    public int SolveIndexed(
        List<RigidBody2> bodies,
        List<ContactConstraint> constraints,
        List<int> indices,
        float dt)
    {
        int ran = 0;
        for (int i = 0; i < Iterations; i++)
        {
            ran++;
            float maxApplied = 0.0f;
            foreach (int j in indices)
            {
                var c = constraints[j];
                if (BothFrozen(bodies, c.BodyA, c.BodyB)) continue;
                maxApplied = MathF.Max(maxApplied, SolveNormal(bodies, ref c, dt));
                constraints[j] = c;
            }
            if (EnableFriction)
            {
                foreach (int j in indices)
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

    /// <summary>
    /// Apply cached impulses from the previous frame directly (no re-scaling:
    /// the stored scalars already are impulses).
    /// </summary>
    private static void ApplyWarmStarting(List<RigidBody2> bodies, ContactConstraint constraint)
    {
        if (constraint.NormalImpulse == 0.0f && constraint.TangentImpulse == 0.0f)
        {
            return;
        }
        Vector2 tangent = constraint.Tangent;
        Vector2 p = constraint.Normal * constraint.NormalImpulse + tangent * constraint.TangentImpulse;

        Vector2 rA = constraint.Point - bodies[constraint.BodyA].Transform.Position;
        Vector2 rB = constraint.Point - bodies[constraint.BodyB].Transform.Position;

        ApplyImpulse(bodies, constraint.BodyA, constraint.BodyB, rA, rB, p);
    }

    /// <summary>
    /// Solve the normal (non-penetration + restitution + Baumgarte) constraint
    /// with impulse accumulation (j_n clamped &gt;= 0).
    /// </summary>
    private float SolveNormal(List<RigidBody2> bodies, ref ContactConstraint constraint, float dt)
    {
        var bodyA = bodies[constraint.BodyA];
        var bodyB = bodies[constraint.BodyB];

        Vector2 rA = constraint.Point - bodyA.Transform.Position;
        Vector2 rB = constraint.Point - bodyB.Transform.Position;

        // Relative velocity at contact point
        Vector2 vA = bodyA.Velocity + Vec2ToAngular(rA, bodyA.AngularVelocity);
        Vector2 vB = bodyB.Velocity + Vec2ToAngular(rB, bodyB.AngularVelocity);
        float vn = Vector2.Dot(vB - vA, constraint.Normal);

        float massN = constraint.MassNormal;
        if (massN == 0.0f)
        {
            return 0.0f;
        }

        // Restitution bias was fixed pre-step (see RestBias); the
        // Baumgarte term handles position correction. Baumgarte is skipped
        // while restitution is active: the bounce separates the contact
        // anyway, and stacking the positional push on top of the bounce
        // velocity pumps energy into the rebound.
        float bias = constraint.RestBias > 0.0f
            ? 0.0f
            : dt > 0.0f
                ? MathF.Min(Baumgarte * MathF.Max(0.0f, constraint.Penetration - Slop) / dt, MaxBiasVelocity)
                : 0.0f;

        float lambda = massN * (constraint.RestBias - vn + bias);

        // Accumulate and clamp (non-penetration is one-sided)
        float oldImpulse = constraint.NormalImpulse;
        constraint.NormalImpulse = MathF.Max(0.0f, oldImpulse + lambda);
        float applied = constraint.NormalImpulse - oldImpulse;

        if (applied != 0.0f)
        {
            ApplyImpulse(bodies, constraint.BodyA, constraint.BodyB, rA, rB, constraint.Normal * applied);
        }
        return MathF.Abs(applied);
    }

    /// <summary>
    /// Solve the tangential (Coulomb friction) constraint with accumulation.
    /// Clamps |j_t| &lt;= mu * j_n (friction cone).
    /// </summary>
    private static float SolveFriction(List<RigidBody2> bodies, ref ContactConstraint constraint)
    {
        Vector2 tangent = constraint.Tangent;
        var bodyA = bodies[constraint.BodyA];
        var bodyB = bodies[constraint.BodyB];

        Vector2 rA = constraint.Point - bodyA.Transform.Position;
        Vector2 rB = constraint.Point - bodyB.Transform.Position;

        Vector2 vA = bodyA.Velocity + Vec2ToAngular(rA, bodyA.AngularVelocity);
        Vector2 vB = bodyB.Velocity + Vec2ToAngular(rB, bodyB.AngularVelocity);
        float vt = Vector2.Dot(vB - vA, tangent);

        float massT = constraint.MassTangent;
        if (massT == 0.0f)
        {
            return 0.0f;
        }

        float lambda = massT * -vt;

        // Coulomb cone clamp against the current accumulated normal impulse
        float maxFriction = constraint.Friction * constraint.NormalImpulse;
        float oldImpulse = constraint.TangentImpulse;
        constraint.TangentImpulse = MathF.Max(-maxFriction, MathF.Min(maxFriction, oldImpulse + lambda));
        float applied = constraint.TangentImpulse - oldImpulse;

        if (applied != 0.0f)
        {
            ApplyImpulse(bodies, constraint.BodyA, constraint.BodyB, rA, rB, tangent * applied);
        }
        return MathF.Abs(applied);
    }

    /// <summary>
    /// Restitution target from pre-solve relative normal velocity.
    /// Bounce only for significant impacts (Box2D uses 1 m/s threshold).
    /// </summary>
    private static float RestitutionBias(List<RigidBody2> bodies, ContactConstraint constraint)
    {
        var bodyA = bodies[constraint.BodyA];
        var bodyB = bodies[constraint.BodyB];
        Vector2 rA = constraint.Point - bodyA.Transform.Position;
        Vector2 rB = constraint.Point - bodyB.Transform.Position;
        Vector2 vA = bodyA.Velocity + Vec2ToAngular(rA, bodyA.AngularVelocity);
        Vector2 vB = bodyB.Velocity + Vec2ToAngular(rB, bodyB.AngularVelocity);
        float vn = Vector2.Dot(vB - vA, constraint.Normal);
        return vn < -1.0f ? -constraint.Restitution * vn : 0.0f;
    }

    /// <summary>
    /// Effective mass along a direction:
    /// 1 / (inv_m_a + inv_m_b + inv_i_a*(r_a x d)^2 + inv_i_b*(r_b x d)^2).
    /// Returns 0 when both bodies are static along this direction.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float EffectiveMass(
        float invMassA, float invMassB,
        float invInertiaA, float invInertiaB,
        Vector2 rA, Vector2 rB, Vector2 dir)
    {
        float rACross = rA.X * dir.Y - rA.Y * dir.X;
        float rBCross = rB.X * dir.Y - rB.Y * dir.X;
        float k = invMassA + invMassB +
                  invInertiaA * rACross * rACross +
                  invInertiaB * rBCross * rBCross;
        return k > 0.0f ? 1.0f / k : 0.0f;
    }

    /// <summary>Apply a world-space impulse P at the contact (A -= P, B += P).</summary>
    private static void ApplyImpulse(
        List<RigidBody2> bodies, int bodyA, int bodyB,
        Vector2 rA, Vector2 rB, Vector2 p)
    {
        // Frozen bodies (static or sleeping) take no impulse, mirroring
        // the infinite-mass mass terms above.
        var a = bodies[bodyA];
        var b = bodies[bodyB];
        if (!a.IsStatic && !a.IsSleeping)
        {
            a.Velocity -= p * a.InvMass;
            a.AngularVelocity -= (rA.X * p.Y - rA.Y * p.X) * a.InvInertia;
        }
        if (!b.IsStatic && !b.IsSleeping)
        {
            b.Velocity += p * b.InvMass;
            b.AngularVelocity += (rB.X * p.Y - rB.Y * p.X) * b.InvInertia;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static Vector2 Vec2ToAngular(Vector2 r, float angularVelocity)
    {
        return new Vector2(-r.Y, r.X) * angularVelocity;
    }
}
