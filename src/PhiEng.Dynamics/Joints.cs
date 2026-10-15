using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Joint constraint types for 2D (distance, hinge, prismatic, fixed).
/// Mirrors Rust <c>Joint2</c> in crates/dynamics/src/joint.rs — same
/// formulations, constants, validation rules, and free-limit conventions.
/// Joints solve after contacts each iteration and own their accumulated
/// impulses, so warm starting is automatic.
/// </summary>
public abstract class Joint2
{
    /// <summary>Joint id (doubles as the index in World2.Joints).</summary>
    public int Id { get; set; }
    /// <summary>First body id.</summary>
    public int BodyA { get; protected set; }
    /// <summary>Second body id.</summary>
    public int BodyB { get; protected set; }

    /// <summary>Re-applies the stored impulse before iterating (warm start).</summary>
    public abstract void WarmStart(List<RigidBody2> bodies);
    /// <summary>Runs one sequential-impulse iteration.</summary>
    public abstract void Solve(List<RigidBody2> bodies, float dt);

    /// <summary>
    /// Shift survivor endpoints after a body removal (Vec.RemoveAt semantics).
    /// Returns false when the joint touched the removed body (caller drops it).
    /// </summary>
    public bool Remap(int removed)
    {
        if (BodyA == removed || BodyB == removed) return false;
        if (BodyA > removed) BodyA--;
        if (BodyB > removed) BodyB--;
        return true;
    }

    /// <summary>Rejects unknown or identical joint endpoints.</summary>
    protected static void ValidateEndpoints(int bodyA, int bodyB, int bodyCount)
    {
        if (bodyA < 0 || bodyB < 0 || bodyA >= bodyCount || bodyB >= bodyCount)
            throw new ArgumentException($"Unknown joint body id ({bodyA}, {bodyB}).");
        if (bodyA == bodyB)
            throw new ArgumentException("Joint requires two distinct bodies.");
    }

    /// <summary>Rejects non-finite joint anchors.</summary>
    protected static void ValidateAnchor(Vector2 anchor, string name)
    {
        if (!float.IsFinite(anchor.X) || !float.IsFinite(anchor.Y))
            throw new ArgumentException($"Non-finite joint {name}.");
    }

    /// <summary>Rejects non-finite or non-zero compliance (rigid joints only).</summary>
    protected static void ValidateCompliance(float compliance)
    {
        if (!float.IsFinite(compliance))
            throw new ArgumentException("Non-finite joint compliance.");
        if (compliance != 0.0f)
            throw new ArgumentException("Non-zero compliance not supported in v1 (rigid joints only).");
    }

    /// <summary>Effective mass along a direction (same formula as the contact solver).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    protected static float EffectiveMass(
        float invMassA, float invMassB, float invInertiaA, float invInertiaB,
        Vector2 rA, Vector2 rB, Vector2 dir)
    {
        float raCross = rA.X * dir.Y - rA.Y * dir.X;
        float rbCross = rB.X * dir.Y - rB.Y * dir.X;
        float k = invMassA + invMassB
            + invInertiaA * raCross * raCross
            + invInertiaB * rbCross * rbCross;
        return k > 0.0f ? 1.0f / k : 0.0f;
    }

    /// <summary>90-degree CCW perpendicular (angular velocity contribution at r).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    protected static Vector2 Perp(Vector2 r) => new(-r.Y, r.X);

    /// <summary>Applies impulse P to body B and -P to body A, with torques.</summary>
    protected static void ApplyPairImpulse(
        List<RigidBody2> bodies, int bodyA, int bodyB, Vector2 p, Vector2 pa, Vector2 pb)
    {
        var a = bodies[bodyA];
        var b = bodies[bodyB];
        var ra = pa - a.Transform.Position;
        var rb = pb - b.Transform.Position;
        a.Velocity -= p * a.InvMass;
        b.Velocity += p * b.InvMass;
        a.AngularVelocity -= (ra.X * p.Y - ra.Y * p.X) * a.InvInertia;
        b.AngularVelocity += (rb.X * p.Y - rb.Y * p.X) * b.InvInertia;
    }

    /// <summary>Wraps an angle to [-pi, pi].</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    protected static float WrapAngle(float a)
    {
        a %= MathF.Tau;
        if (a > MathF.PI) a -= MathF.Tau;
        else if (a < -MathF.PI) a += MathF.Tau;
        return a;
    }
}

/// <summary>
/// Rigid rod between two anchors: |worldB - worldA| == rest length.
/// Bilateral (pushes and pulls). Mirrors Rust <c>DistanceJoint</c>.
/// </summary>
public sealed class DistanceJoint : Joint2
{
    /// <summary>Positional correction strength (fraction of the
    /// positional error corrected per step, scaled by 1/dt).</summary>
    public const float Baumgarte = 0.2f;

    /// <summary>Anchor on body A, in A's local frame.</summary>
    public Vector2 AnchorA { get; }
    /// <summary>Anchor on body B, in B's local frame.</summary>
    public Vector2 AnchorB { get; }
    /// <summary>Rest length in world units (>= 0).</summary>
    public float RestLength { get; }
    /// <summary>Accumulated impulse (warm start, persists across steps).</summary>
    public float Impulse { get; set; }

    /// <summary>
    /// Creates a validated distance joint. Throws <see cref="ArgumentException"/>
    /// on unknown/stale bodies, shared endpoints, non-finite anchors, negative
    /// rest length, or non-zero compliance (rigid-only v1).
    /// </summary>
    /// <param name="bodyCount">Current body count, for endpoint validation.</param>
    public DistanceJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float restLength, float compliance, int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA, nameof(anchorA));
        ValidateAnchor(anchorB, nameof(anchorB));
        if (!float.IsFinite(restLength))
            throw new ArgumentException("Non-finite joint rest length.");
        if (restLength < 0.0f)
            throw new ArgumentException("Joint rest length must be >= 0.");
        ValidateCompliance(compliance);
        BodyA = bodyA; BodyB = bodyB;
        AnchorA = anchorA; AnchorB = anchorB;
        RestLength = restLength;
    }

    private Vector2 WorldA(List<RigidBody2> bodies) => bodies[BodyA].Transform.TransformPoint(AnchorA);
    private Vector2 WorldB(List<RigidBody2> bodies) => bodies[BodyB].Transform.TransformPoint(AnchorB);

    /// <inheritdoc/>
    public override void WarmStart(List<RigidBody2> bodies)
    {
        if (Impulse == 0.0f) return;
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var delta = pb - pa;
        float dist = delta.Length();
        if (dist <= 1e-6f) return;
        var n = delta / dist;
        ApplyPairImpulse(bodies, BodyA, BodyB, -n * Impulse, pa, pb);
    }

    /// <inheritdoc/>
    public override void Solve(List<RigidBody2> bodies, float dt)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var delta = pb - pa;
        float dist = delta.Length();
        if (dist <= 1e-6f) return;
        var n = delta / dist;
        float c = dist - RestLength;
        if (MathF.Abs(c) < 1e-6f) return;

        var a = bodies[BodyA];
        var b = bodies[BodyB];
        var ra = pa - a.Transform.Position;
        var rb = pb - b.Transform.Position;
        var va = a.Velocity + Perp(ra) * a.AngularVelocity;
        var vb = b.Velocity + Perp(rb) * b.AngularVelocity;
        float vn = Vector2.Dot(vb - va, n);
        float m = EffectiveMass(a.InvMass, b.InvMass, a.InvInertia, b.InvInertia, ra, rb, n);
        if (m == 0.0f) return;

        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = m * (vn + bias);
        Impulse += lambda;
        ApplyPairImpulse(bodies, BodyA, BodyB, -n * lambda, pa, pb);
    }
}

/// <summary>
/// Revolute joint: local anchors must coincide; rotation is free unless
/// angle limits are enforced. <c>LimitMin &gt; LimitMax</c> means a FREE
/// hinge. Mirrors Rust <c>HingeJoint</c>.
/// </summary>
public sealed class HingeJoint : Joint2
{
    /// <summary>Positional correction strength (fraction of the
    /// positional error corrected per step, scaled by 1/dt).</summary>
    public const float Baumgarte = 0.2f;

    /// <summary>Anchor on body A, in A's local frame.</summary>
    public Vector2 AnchorA { get; }
    /// <summary>Anchor on body B, in B's local frame.</summary>
    public Vector2 AnchorB { get; }
    /// <summary>Lower bound (radians or world units); Min &gt; Max means free.</summary>
    public float LimitMin { get; }
    /// <summary>Upper bound (radians or world units); Min &gt; Max means free.</summary>
    public float LimitMax { get; }
    /// <summary>Accumulated point impulse in world frame
    /// (warm start; axes are world-fixed so it is directly reusable).</summary>
    public Vector2 Impulse { get; set; }
    /// <summary>Accumulated angular impulse (warm start).</summary>
    public float AngularImpulse { get; set; }

    /// <summary>
    /// Creates a validated hinge joint. <c>limitMin &gt; limitMax</c> selects a
    /// free hinge; otherwise the relative angle is clamped into range.
    /// Throws <see cref="ArgumentException"/> on bad endpoints, non-finite
    /// anchors/limits, or non-zero compliance.
    /// </summary>
    /// <param name="bodyCount">Current body count, for endpoint validation.</param>
    public HingeJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float limitMin, float limitMax, float compliance, int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA, nameof(anchorA));
        ValidateAnchor(anchorB, nameof(anchorB));
        if (!float.IsFinite(limitMin) || !float.IsFinite(limitMax))
            throw new ArgumentException("Non-finite hinge limits.");
        ValidateCompliance(compliance);
        BodyA = bodyA; BodyB = bodyB;
        AnchorA = anchorA; AnchorB = anchorB;
        LimitMin = limitMin; LimitMax = limitMax;
    }

    private Vector2 WorldA(List<RigidBody2> bodies) => bodies[BodyA].Transform.TransformPoint(AnchorA);
    private Vector2 WorldB(List<RigidBody2> bodies) => bodies[BodyB].Transform.TransformPoint(AnchorB);
    private float RelativeAngle(List<RigidBody2> bodies) =>
        WrapAngle(bodies[BodyB].RotationAngle - bodies[BodyA].RotationAngle);

    /// <inheritdoc/>
    public override void WarmStart(List<RigidBody2> bodies)
    {
        if (Impulse == Vector2.Zero && AngularImpulse == 0.0f) return;
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        if (Impulse != Vector2.Zero)
            ApplyPairImpulse(bodies, BodyA, BodyB, Impulse, pa, pb);
        if (AngularImpulse != 0.0f)
        {
            float lam = AngularImpulse;
            var a = bodies[BodyA];
            var b = bodies[BodyB];
            a.AngularVelocity -= lam * a.InvInertia;
            b.AngularVelocity += lam * b.InvInertia;
        }
    }

    /// <inheritdoc/>
    public override void Solve(List<RigidBody2> bodies, float dt)
    {
        SolvePoint(bodies, dt);
        SolveLimits(bodies, dt);
    }

    private void SolvePoint(List<RigidBody2> bodies, float dt)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var delta = pb - pa;
        if (delta.LengthSquared() < 1e-12f) return;
        foreach (var axis in new[] { Vector2.UnitX, Vector2.UnitY })
        {
            float c = Vector2.Dot(delta, axis);
            if (MathF.Abs(c) < 1e-6f) continue;
            var a = bodies[BodyA];
            var b = bodies[BodyB];
            var ra = pa - a.Transform.Position;
            var rb = pb - b.Transform.Position;
            var va = a.Velocity + Perp(ra) * a.AngularVelocity;
            var vb = b.Velocity + Perp(rb) * b.AngularVelocity;
            float vn = Vector2.Dot(vb - va, axis);
            float m = EffectiveMass(a.InvMass, b.InvMass, a.InvInertia, b.InvInertia, ra, rb, axis);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (vn + bias);
            Impulse += axis * lambda;
            ApplyPairImpulse(bodies, BodyA, BodyB, axis * lambda, pa, pb);
        }
    }

    private void SolveLimits(List<RigidBody2> bodies, float dt)
    {
        if (LimitMin > LimitMax) return; // free hinge
        float theta = RelativeAngle(bodies);
        float c = theta < LimitMin ? theta - LimitMin
            : theta > LimitMax ? theta - LimitMax
            : 0.0f;
        if (c == 0.0f) return;
        var a = bodies[BodyA];
        var b = bodies[BodyB];
        float wRel = b.AngularVelocity - a.AngularVelocity;
        float k = a.InvInertia + b.InvInertia;
        if (k <= 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -1.0f / k * (wRel + bias);
        AngularImpulse += lambda;
        a.AngularVelocity -= lambda * a.InvInertia;
        b.AngularVelocity += lambda * b.InvInertia;
    }
}

/// <summary>
/// Slider joint: free translation along the axis (A's local frame),
/// locked perpendicular, rotation locked to the latched pose.
/// <c>LimitMin &gt; LimitMax</c> means free travel.
/// Mirrors Rust <c>PrismaticJoint</c>.
/// </summary>
public sealed class PrismaticJoint : Joint2
{
    /// <summary>Positional correction strength (fraction of the
    /// positional error corrected per step, scaled by 1/dt).</summary>
    public const float Baumgarte = 0.2f;

    /// <summary>Anchor on body A, in A's local frame.</summary>
    public Vector2 AnchorA { get; }
    /// <summary>Anchor on body B, in B's local frame.</summary>
    public Vector2 AnchorB { get; }
    /// <summary>Slide axis in A's local frame (unit length).</summary>
    public Vector2 AxisA { get; }
    /// <summary>Lower bound (radians or world units); Min &gt; Max means free.</summary>
    public float LimitMin { get; }
    /// <summary>Upper bound (radians or world units); Min &gt; Max means free.</summary>
    public float LimitMax { get; }
    /// <summary>Latched relative angle (null until first solve).</summary>
    public float? RefAngle { get; set; }
    /// <summary>Accumulated perpendicular impulse (warm start).</summary>
    public float PerpImpulse { get; set; }
    /// <summary>Accumulated angular impulse (warm start).</summary>
    public float AngularImpulse { get; set; }
    /// <summary>Accumulated travel-limit impulse (warm start).</summary>
    public float LimitImpulse { get; set; }

    /// <summary>
    /// Creates a validated prismatic joint. The axis is normalized at creation.
    /// <c>limitMin &gt; limitMax</c> means free travel. Throws
    /// <see cref="ArgumentException"/> on bad endpoints, non-finite/zero axis,
    /// non-finite limits, or non-zero compliance.
    /// </summary>
    /// <param name="bodyCount">Current body count, for endpoint validation.</param>
    public PrismaticJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        Vector2 axisA, float limitMin, float limitMax, float compliance, int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA, nameof(anchorA));
        ValidateAnchor(anchorB, nameof(anchorB));
        if (!float.IsFinite(axisA.X) || !float.IsFinite(axisA.Y))
            throw new ArgumentException("Non-finite prismatic axis.");
        if (axisA.Length() <= 1e-6f)
            throw new ArgumentException("Prismatic axis must be non-zero.");
        if (!float.IsFinite(limitMin) || !float.IsFinite(limitMax))
            throw new ArgumentException("Non-finite prismatic limits.");
        ValidateCompliance(compliance);
        BodyA = bodyA; BodyB = bodyB;
        AnchorA = anchorA; AnchorB = anchorB;
        AxisA = axisA / axisA.Length();
        LimitMin = limitMin; LimitMax = limitMax;
    }

    private Vector2 WorldA(List<RigidBody2> bodies) => bodies[BodyA].Transform.TransformPoint(AnchorA);
    private Vector2 WorldB(List<RigidBody2> bodies) => bodies[BodyB].Transform.TransformPoint(AnchorB);
    private Vector2 AxisWorld(List<RigidBody2> bodies) =>
        Vector2.Transform(AxisA, bodies[BodyA].Transform.Rotation);

    /// <inheritdoc/>
    public override void WarmStart(List<RigidBody2> bodies)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        if (PerpImpulse != 0.0f)
        {
            var perpAx = Perp(AxisWorld(bodies));
            ApplyPairImpulse(bodies, BodyA, BodyB, perpAx * PerpImpulse, pa, pb);
        }
        if (LimitImpulse != 0.0f)
        {
            var ax = AxisWorld(bodies);
            ApplyPairImpulse(bodies, BodyA, BodyB, ax * LimitImpulse, pa, pb);
        }
        if (AngularImpulse != 0.0f)
        {
            float lam = AngularImpulse;
            var a = bodies[BodyA];
            var b = bodies[BodyB];
            a.AngularVelocity -= lam * a.InvInertia;
            b.AngularVelocity += lam * b.InvInertia;
        }
    }

    /// <inheritdoc/>
    public override void Solve(List<RigidBody2> bodies, float dt)
    {
        RefAngle ??= WrapAngle(bodies[BodyB].RotationAngle - bodies[BodyA].RotationAngle);
        SolvePerp(bodies, dt);
        SolveAngular(bodies, dt);
        SolveLimits(bodies, dt);
    }

    private void SolvePerp(List<RigidBody2> bodies, float dt)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var perpAx = Perp(AxisWorld(bodies));
        float c = Vector2.Dot(pb - pa, perpAx);
        if (MathF.Abs(c) < 1e-6f) return;
        var a = bodies[BodyA];
        var b = bodies[BodyB];
        var ra = pa - a.Transform.Position;
        var rb = pb - b.Transform.Position;
        var va = a.Velocity + Perp(ra) * a.AngularVelocity;
        var vb = b.Velocity + Perp(rb) * b.AngularVelocity;
        float vn = Vector2.Dot(vb - va, perpAx);
        float m = EffectiveMass(a.InvMass, b.InvMass, a.InvInertia, b.InvInertia, ra, rb, perpAx);
        if (m == 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -m * (vn + bias);
        PerpImpulse += lambda;
        ApplyPairImpulse(bodies, BodyA, BodyB, perpAx * lambda, pa, pb);
    }

    private void SolveAngular(List<RigidBody2> bodies, float dt)
    {
        float theta = WrapAngle(bodies[BodyB].RotationAngle - bodies[BodyA].RotationAngle);
        float c = WrapAngle(theta - RefAngle!.Value);
        if (MathF.Abs(c) < 1e-6f) return;
        var a = bodies[BodyA];
        var b = bodies[BodyB];
        float wRel = b.AngularVelocity - a.AngularVelocity;
        float k = a.InvInertia + b.InvInertia;
        if (k <= 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -1.0f / k * (wRel + bias);
        AngularImpulse += lambda;
        a.AngularVelocity -= lambda * a.InvInertia;
        b.AngularVelocity += lambda * b.InvInertia;
    }

    private void SolveLimits(List<RigidBody2> bodies, float dt)
    {
        if (LimitMin > LimitMax) return;
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var ax = AxisWorld(bodies);
        float s = Vector2.Dot(pb - pa, ax);
        float c = s < LimitMin ? s - LimitMin : s > LimitMax ? s - LimitMax : 0.0f;
        if (c == 0.0f) return;
        var a = bodies[BodyA];
        var b = bodies[BodyB];
        var ra = pa - a.Transform.Position;
        var rb = pb - b.Transform.Position;
        var va = a.Velocity + Perp(ra) * a.AngularVelocity;
        var vb = b.Velocity + Perp(rb) * b.AngularVelocity;
        float vn = Vector2.Dot(vb - va, ax);
        float m = EffectiveMass(a.InvMass, b.InvMass, a.InvInertia, b.InvInertia, ra, rb, ax);
        if (m == 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -m * (vn + bias);
        LimitImpulse += lambda;
        ApplyPairImpulse(bodies, BodyA, BodyB, ax * lambda, pa, pb);
    }
}

/// <summary>
/// Weld joint: anchors must coincide and relative angle must hold its
/// latched value (captured at first solve). Mirrors Rust <c>FixedJoint</c>.
/// </summary>
public sealed class FixedJoint : Joint2
{
    /// <summary>Positional correction strength (fraction of the
    /// positional error corrected per step, scaled by 1/dt).</summary>
    public const float Baumgarte = 0.2f;

    /// <summary>Anchor on body A, in A's local frame.</summary>
    public Vector2 AnchorA { get; }
    /// <summary>Anchor on body B, in B's local frame.</summary>
    public Vector2 AnchorB { get; }
    /// <summary>Latched relative angle (null until first solve).</summary>
    public float? RefAngle { get; set; }
    /// <summary>Accumulated point impulse in world frame
    /// (warm start; axes are world-fixed so it is directly reusable).</summary>
    public Vector2 Impulse { get; set; }
    /// <summary>Accumulated angular impulse (warm start).</summary>
    public float AngularImpulse { get; set; }

    /// <summary>
    /// Creates a validated weld joint. The relative angle is latched at first
    /// solve. Throws <see cref="ArgumentException"/> on bad endpoints,
    /// non-finite anchors, or non-zero compliance.
    /// </summary>
    /// <param name="bodyCount">Current body count, for endpoint validation.</param>
    public FixedJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float compliance, int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA, nameof(anchorA));
        ValidateAnchor(anchorB, nameof(anchorB));
        ValidateCompliance(compliance);
        BodyA = bodyA; BodyB = bodyB;
        AnchorA = anchorA; AnchorB = anchorB;
    }

    private Vector2 WorldA(List<RigidBody2> bodies) => bodies[BodyA].Transform.TransformPoint(AnchorA);
    private Vector2 WorldB(List<RigidBody2> bodies) => bodies[BodyB].Transform.TransformPoint(AnchorB);

    /// <inheritdoc/>
    public override void WarmStart(List<RigidBody2> bodies)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        if (Impulse != Vector2.Zero)
            ApplyPairImpulse(bodies, BodyA, BodyB, Impulse, pa, pb);
        if (AngularImpulse != 0.0f)
        {
            float lam = AngularImpulse;
            var a = bodies[BodyA];
            var b = bodies[BodyB];
            a.AngularVelocity -= lam * a.InvInertia;
            b.AngularVelocity += lam * b.InvInertia;
        }
    }

    /// <inheritdoc/>
    public override void Solve(List<RigidBody2> bodies, float dt)
    {
        RefAngle ??= WrapAngle(bodies[BodyB].RotationAngle - bodies[BodyA].RotationAngle);
        SolvePoint(bodies, dt);
        SolveAngular(bodies, dt);
    }

    private void SolvePoint(List<RigidBody2> bodies, float dt)
    {
        var pa = WorldA(bodies);
        var pb = WorldB(bodies);
        var delta = pb - pa;
        if (delta.LengthSquared() < 1e-12f) return;
        foreach (var axis in new[] { Vector2.UnitX, Vector2.UnitY })
        {
            float c = Vector2.Dot(delta, axis);
            if (MathF.Abs(c) < 1e-6f) continue;
            var a = bodies[BodyA];
            var b = bodies[BodyB];
            var ra = pa - a.Transform.Position;
            var rb = pb - b.Transform.Position;
            var va = a.Velocity + Perp(ra) * a.AngularVelocity;
            var vb = b.Velocity + Perp(rb) * b.AngularVelocity;
            float vn = Vector2.Dot(vb - va, axis);
            float m = EffectiveMass(a.InvMass, b.InvMass, a.InvInertia, b.InvInertia, ra, rb, axis);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (vn + bias);
            Impulse += axis * lambda;
            ApplyPairImpulse(bodies, BodyA, BodyB, axis * lambda, pa, pb);
        }
    }

    private void SolveAngular(List<RigidBody2> bodies, float dt)
    {
        float theta = WrapAngle(bodies[BodyB].RotationAngle - bodies[BodyA].RotationAngle);
        float c = WrapAngle(theta - RefAngle!.Value);
        if (MathF.Abs(c) < 1e-6f) return;
        var a = bodies[BodyA];
        var b = bodies[BodyB];
        float wRel = b.AngularVelocity - a.AngularVelocity;
        float k = a.InvInertia + b.InvInertia;
        if (k <= 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -1.0f / k * (wRel + bias);
        AngularImpulse += lambda;
        a.AngularVelocity -= lambda * a.InvInertia;
        b.AngularVelocity += lambda * b.InvInertia;
    }
}
