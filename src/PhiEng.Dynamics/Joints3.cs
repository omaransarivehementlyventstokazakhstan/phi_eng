using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// A rigid rod between two anchor points: |world_b - world_a| == rest_length.
/// Bilateral (pushes AND pulls), so the accumulated impulse is unclamped.
/// Mirrors Rust <c>DistanceJoint3</c> in crates/dynamics/src/joint3.rs.
/// </summary>
public struct DistanceJoint3
{
    /// <summary>Joint id (doubles as index).</summary>
    public int Id;
    /// <summary>First body id.</summary>
    public int BodyA;
    /// <summary>Second body id.</summary>
    public int BodyB;
    /// <summary>Anchor on A in A's local frame.</summary>
    public Vector3 AnchorA;
    /// <summary>Anchor on B in B's local frame.</summary>
    public Vector3 AnchorB;
    /// <summary>Maintained anchor separation.</summary>
    public float RestLength;
    /// <summary>Accumulated impulse (warm start).</summary>
    public float Impulse;
}

/// <summary>
/// A revolute (hinge) joint: the two local anchors must coincide
/// (point-to-point, 3 linear constraints), the two local hinge axes must
/// align (2 angular constraints), and twist about the hinge axis is free
/// unless twist limits are enforced (1 angular constraint when out of range).
/// Mirrors Rust <c>HingeJoint3</c>.
/// </summary>
public struct HingeJoint3
{
    /// <summary>Joint id (doubles as index).</summary>
    public int Id;
    /// <summary>First body id.</summary>
    public int BodyA;
    /// <summary>Second body id.</summary>
    public int BodyB;
    /// <summary>Anchor on A in A's local frame.</summary>
    public Vector3 AnchorA;
    /// <summary>Anchor on B in B's local frame.</summary>
    public Vector3 AnchorB;
    /// <summary>Hinge axis on A in A's local frame.</summary>
    public Vector3 AxisA;
    /// <summary>Hinge axis on B in B's local frame.</summary>
    public Vector3 AxisB;
    /// <summary>Twist lower bound (radians); min &gt; max selects a free hinge.</summary>
    public float LimitMin;
    /// <summary>Twist upper bound (radians).</summary>
    public float LimitMax;
    /// <summary>Accumulated point impulse in world frame (warm start).</summary>
    public Vector3 Impulse;
    /// <summary>Accumulated first alignment impulse (warm start).</summary>
    public float AlignU1;
    /// <summary>Accumulated second alignment impulse (warm start).</summary>
    public float AlignU2;
    /// <summary>Accumulated twist-limit impulse (warm start).</summary>
    public float TwistImpulse;
}

/// <summary>
/// A prismatic (slider) joint: translation is free along <c>AxisA</c>
/// (A's local frame), locked on the perpendicular plane (2 linear
/// constraints), and relative orientation is locked to the pose at first
/// solve (latched <see cref="RefOrientation"/>, 3 angular constraints).
/// Mirrors Rust <c>PrismaticJoint3</c>.
/// </summary>
public struct PrismaticJoint3
{
    /// <summary>Joint id (doubles as index).</summary>
    public int Id;
    /// <summary>First body id.</summary>
    public int BodyA;
    /// <summary>Second body id.</summary>
    public int BodyB;
    /// <summary>Anchor on A in A's local frame.</summary>
    public Vector3 AnchorA;
    /// <summary>Anchor on B in B's local frame.</summary>
    public Vector3 AnchorB;
    /// <summary>Slide axis in A's local frame.</summary>
    public Vector3 AxisA;
    /// <summary>Travel lower bound; min &gt; max selects free travel.</summary>
    public float LimitMin;
    /// <summary>Travel upper bound.</summary>
    public float LimitMax;
    /// <summary>Latched relative orientation (captured at first solve).</summary>
    public Quaternion? RefOrientation;
    /// <summary>Accumulated first perpendicular impulse (warm start).</summary>
    public float PerpU1;
    /// <summary>Accumulated second perpendicular impulse (warm start).</summary>
    public float PerpU2;
    /// <summary>Accumulated angular impulse in world frame (warm start).</summary>
    public Vector3 AngularImpulse;
    /// <summary>Accumulated travel-limit impulse (warm start).</summary>
    public float LimitImpulse;
}

/// <summary>
/// A weld (fixed) joint: the two anchors must coincide (3 linear
/// constraints) AND the relative orientation must hold its latched value
/// (3 angular constraints, captured at first solve). Mirrors Rust
/// <c>FixedJoint3</c>.
/// </summary>
public struct FixedJoint3
{
    /// <summary>Joint id (doubles as index).</summary>
    public int Id;
    /// <summary>First body id.</summary>
    public int BodyA;
    /// <summary>Second body id.</summary>
    public int BodyB;
    /// <summary>Anchor on A in A's local frame.</summary>
    public Vector3 AnchorA;
    /// <summary>Anchor on B in B's local frame.</summary>
    public Vector3 AnchorB;
    /// <summary>Latched relative orientation (captured at first solve).</summary>
    public Quaternion? RefOrientation;
    /// <summary>Accumulated point impulse in world frame (warm start).</summary>
    public Vector3 Impulse;
    /// <summary>Accumulated angular impulse in world frame (warm start).</summary>
    public Vector3 AngularImpulse;
}

/// <summary>
/// Any 3D joint type. The solver dispatches per variant each iteration.
/// Mirrors Rust <c>Joint3</c>.
/// </summary>
/// <summary>3D joint discriminator.</summary>
public enum Joint3Type
{
    /// <summary>Distance rod.</summary>
    Distance3,
    /// <summary>Revolute hinge.</summary>
    Hinge3,
    /// <summary>Linear slider.</summary>
    Prismatic3,
    /// <summary>Rigid weld.</summary>
    Fixed3,
}

/// <summary>
/// 3D joint container with type dispatch. Mirrors Rust <c>Joint3</c> enum.
/// </summary>
public sealed class Joint3
{
    /// <summary>Active variant.</summary>
    public Joint3Type Type { get; }
    /// <summary>Distance payload (set when <see cref="Type"/> is Distance3).</summary>
    public DistanceJoint3? Distance { get; set; }
    /// <summary>Hinge payload (set when <see cref="Type"/> is Hinge3).</summary>
    public HingeJoint3? Hinge { get; set; }
    /// <summary>Prismatic payload (set when <see cref="Type"/> is Prismatic3).</summary>
    public PrismaticJoint3? Prismatic { get; set; }
    /// <summary>Fixed payload (set when <see cref="Type"/> is Fixed3).</summary>
    public FixedJoint3? Fixed { get; set; }

    /// <summary>Joint container; exactly the matching payload must be set.</summary>
    public Joint3(Joint3Type type, DistanceJoint3? d, HingeJoint3? h,
        PrismaticJoint3? p, FixedJoint3? f)
    {
        Type = type;
        Distance = d;
        Hinge = h;
        Prismatic = p;
        Fixed = f;
    }

    /// <summary>Wraps a distance joint.</summary>
    public static Joint3 MakeDistance(DistanceJoint3 d) =>
        new(Joint3Type.Distance3, d, null, null, null);
    /// <summary>Wraps a hinge joint.</summary>
    public static Joint3 MakeHinge(HingeJoint3 h) =>
        new(Joint3Type.Hinge3, null, h, null, null);
    /// <summary>Wraps a prismatic joint.</summary>
    public static Joint3 MakePrismatic(PrismaticJoint3 p) =>
        new(Joint3Type.Prismatic3, null, null, p, null);
    /// <summary>Wraps a fixed joint.</summary>
    public static Joint3 MakeFixed(FixedJoint3 f) =>
        new(Joint3Type.Fixed3, null, null, null, f);

    /// <summary>Joint id (doubles as index).</summary>
    public int Id => Type switch
    {
        Joint3Type.Distance3 => Distance!.Value.Id,
        Joint3Type.Hinge3 => Hinge!.Value.Id,
        Joint3Type.Prismatic3 => Prismatic!.Value.Id,
        Joint3Type.Fixed3 => Fixed!.Value.Id,
        _ => throw new InvalidOperationException(),
    };

    /// <summary>Assigns the joint id (used to preserve id == index).</summary>
    public void SetId(int id)
    {
        switch (Type)
        {
            case Joint3Type.Distance3:
            {
                var d = Distance!.Value;
                d.Id = id;
                Distance = d;
                break;
            }
            case Joint3Type.Hinge3:
            {
                var h = Hinge!.Value;
                h.Id = id;
                Hinge = h;
                break;
            }
            case Joint3Type.Prismatic3:
            {
                var p = Prismatic!.Value;
                p.Id = id;
                Prismatic = p;
                break;
            }
            case Joint3Type.Fixed3:
            {
                var f = Fixed!.Value;
                f.Id = id;
                Fixed = f;
                break;
            }
        }
    }

    /// <summary>Body ids touched by this joint.</summary>
    public (int, int) Bodies() => Type switch
    {
        Joint3Type.Distance3 => (Distance!.Value.BodyA, Distance.Value.BodyB),
        Joint3Type.Hinge3 => (Hinge!.Value.BodyA, Hinge.Value.BodyB),
        Joint3Type.Prismatic3 => (Prismatic!.Value.BodyA, Prismatic.Value.BodyB),
        Joint3Type.Fixed3 => (Fixed!.Value.BodyA, Fixed.Value.BodyB),
        _ => throw new InvalidOperationException(),
    };

    /// <summary>Shifts survivor endpoints down after a body removal.</summary>
    public void Shift(int removed)
    {
        switch (Type)
        {
            case Joint3Type.Distance3:
            {
                var d = Distance!.Value;
                if (d.BodyA > removed) d.BodyA--;
                if (d.BodyB > removed) d.BodyB--;
                Distance = d;
                break;
            }
            case Joint3Type.Hinge3:
            {
                var h = Hinge!.Value;
                if (h.BodyA > removed) h.BodyA--;
                if (h.BodyB > removed) h.BodyB--;
                Hinge = h;
                break;
            }
            case Joint3Type.Prismatic3:
            {
                var p = Prismatic!.Value;
                if (p.BodyA > removed) p.BodyA--;
                if (p.BodyB > removed) p.BodyB--;
                Prismatic = p;
                break;
            }
            case Joint3Type.Fixed3:
            {
                var f = Fixed!.Value;
                if (f.BodyA > removed) f.BodyA--;
                if (f.BodyB > removed) f.BodyB--;
                Fixed = f;
                break;
            }
        }
    }

    /// <summary>Remaps endpoints through a removal shift; false when the joint touched the removed body.</summary>
    public bool Remap(int removed)
    {
        var (a, b) = Bodies();
        if (a == removed || b == removed) return false;
        Shift(removed);
        return true;
    }

    /// <summary>Applies the stored impulse before iterating (warm start).</summary>
    public void WarmStart(List<RigidBody3> bodies)
    {
        switch (Type)
        {
            case Joint3Type.Distance3:
            {
                var d = Distance!.Value;
                Joint3Math.DistanceWarmStart(ref d, bodies);
                Distance = d;
                break;
            }
            case Joint3Type.Hinge3:
            {
                var h = Hinge!.Value;
                Joint3Math.HingeWarmStart(ref h, bodies);
                Hinge = h;
                break;
            }
            case Joint3Type.Prismatic3:
            {
                var p = Prismatic!.Value;
                Joint3Math.PrismaticWarmStart(ref p, bodies);
                Prismatic = p;
                break;
            }
            case Joint3Type.Fixed3:
            {
                var f = Fixed!.Value;
                Joint3Math.FixedWarmStart(ref f, bodies);
                Fixed = f;
                break;
            }
        }
    }

    /// <summary>Runs one sequential-impulse iteration.</summary>
    public void Solve(List<RigidBody3> bodies, float dt)
    {
        switch (Type)
        {
            case Joint3Type.Distance3:
            {
                var d = Distance!.Value;
                Joint3Math.DistanceSolve(ref d, bodies, dt);
                Distance = d;
                break;
            }
            case Joint3Type.Hinge3:
            {
                var h = Hinge!.Value;
                Joint3Math.HingeSolve(ref h, bodies, dt);
                Hinge = h;
                break;
            }
            case Joint3Type.Prismatic3:
            {
                var p = Prismatic!.Value;
                Joint3Math.PrismaticSolve(ref p, bodies, dt);
                Prismatic = p;
                break;
            }
            case Joint3Type.Fixed3:
            {
                var f = Fixed!.Value;
                Joint3Math.FixedSolve(ref f, bodies, dt);
                Fixed = f;
                break;
            }
        }
    }
}

/// <summary>
/// 3D joint math: validation, construction, and per-type solve.
/// Mirrors Rust <c>joint3.rs</c> op-for-op.
/// </summary>
public static class Joint3Math
{
    /// <summary>Baumgarte stabilization factor for joints.</summary>
    public const float Baumgarte = 0.2f;

    // ---- validation ----

    /// <summary>Rejects unknown or identical joint endpoints.</summary>
    public static void ValidateEndpoints(int bodyA, int bodyB, int bodyCount)
    {
        if (bodyA >= bodyCount || bodyB >= bodyCount)
            throw new ArgumentException("unknown or stale body id");
        if (bodyA == bodyB)
            throw new ArgumentException("joint requires two distinct bodies");
    }

    /// <summary>Rejects non-finite joint anchors.</summary>
    public static void ValidateAnchor(Vector3 anchor)
    {
        if (!float.IsFinite(anchor.X) || !float.IsFinite(anchor.Y) || !float.IsFinite(anchor.Z))
            throw new ArgumentException("non-finite joint anchor");
    }

    /// <summary>Rejects non-finite or non-zero compliance (rigid joints only).</summary>
    public static void ValidateCompliance(float compliance)
    {
        if (!float.IsFinite(compliance))
            throw new ArgumentException("non-finite compliance");
        if (compliance != 0.0f)
            throw new ArgumentException("non-zero compliance not supported in v1 (rigid joints only)");
    }

    /// <summary>Rejects non-finite twist limits (inverted bounds select a free hinge).</summary>
    public static void ValidateTwistLimits(float limitMin, float limitMax)
    {
        if (!float.IsFinite(limitMin) || !float.IsFinite(limitMax))
            throw new ArgumentException("non-finite twist limit");
    }

    /// <summary>Rejects non-finite or zero axes; returns the normalized axis.</summary>
    public static Vector3 ValidateAxis(Vector3 axis)
    {
        if (!float.IsFinite(axis.X) || !float.IsFinite(axis.Y) || !float.IsFinite(axis.Z))
            throw new ArgumentException("non-finite axis");
        float len = axis.Length();
        if (len <= 1e-6f)
            throw new ArgumentException("zero axis");
        return axis / len;
    }

    /// <summary>Rejects non-finite travel limits (inverted bounds select free travel).</summary>
    public static void ValidateTravelLimits(float limitMin, float limitMax)
    {
        if (!float.IsFinite(limitMin) || !float.IsFinite(limitMax))
            throw new ArgumentException("non-finite travel limit");
    }

    // ---- validated factory methods (mirror Rust make_*_joint_3) ----

    /// <summary>Builds a validated distance joint.</summary>
    public static Joint3 MakeDistanceJoint3(
        int id, int bodyA, int bodyB,
        Vector3 anchorA, Vector3 anchorB, float restLength, float compliance,
        int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA);
        ValidateAnchor(anchorB);
        if (!float.IsFinite(restLength))
            throw new ArgumentException("non-finite rest length");
        if (restLength < 0.0f)
            throw new ArgumentException("negative rest length");
        ValidateCompliance(compliance);
        return Joint3.MakeDistance(new DistanceJoint3
        {
            Id = id, BodyA = bodyA, BodyB = bodyB,
            AnchorA = anchorA, AnchorB = anchorB,
            RestLength = restLength, Impulse = 0,
        });
    }

    /// <summary>Builds a validated hinge joint.</summary>
    public static Joint3 MakeHingeJoint3(
        int id, int bodyA, int bodyB,
        Vector3 anchorA, Vector3 anchorB,
        Vector3 axisA, Vector3 axisB,
        float limitMin, float limitMax, float compliance,
        int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA);
        ValidateAnchor(anchorB);
        var axA = ValidateAxis(axisA);
        var axB = ValidateAxis(axisB);
        ValidateTwistLimits(limitMin, limitMax);
        ValidateCompliance(compliance);
        return Joint3.MakeHinge(new HingeJoint3
        {
            Id = id, BodyA = bodyA, BodyB = bodyB,
            AnchorA = anchorA, AnchorB = anchorB,
            AxisA = axA, AxisB = axB,
            LimitMin = limitMin, LimitMax = limitMax,
            Impulse = Vector3.Zero,
            AlignU1 = 0, AlignU2 = 0,
            TwistImpulse = 0,
        });
    }

    /// <summary>Builds a validated prismatic joint.</summary>
    public static Joint3 MakePrismaticJoint3(
        int id, int bodyA, int bodyB,
        Vector3 anchorA, Vector3 anchorB,
        Vector3 axisA,
        float limitMin, float limitMax, float compliance,
        int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA);
        ValidateAnchor(anchorB);
        var ax = ValidateAxis(axisA);
        ValidateTravelLimits(limitMin, limitMax);
        ValidateCompliance(compliance);
        return Joint3.MakePrismatic(new PrismaticJoint3
        {
            Id = id, BodyA = bodyA, BodyB = bodyB,
            AnchorA = anchorA, AnchorB = anchorB,
            AxisA = ax, LimitMin = limitMin, LimitMax = limitMax,
            RefOrientation = null,
            PerpU1 = 0, PerpU2 = 0,
            AngularImpulse = Vector3.Zero,
            LimitImpulse = 0,
        });
    }

    /// <summary>Builds a validated fixed joint.</summary>
    public static Joint3 MakeFixedJoint3(
        int id, int bodyA, int bodyB,
        Vector3 anchorA, Vector3 anchorB,
        float compliance, int bodyCount)
    {
        ValidateEndpoints(bodyA, bodyB, bodyCount);
        ValidateAnchor(anchorA);
        ValidateAnchor(anchorB);
        ValidateCompliance(compliance);
        return Joint3.MakeFixed(new FixedJoint3
        {
            Id = id, BodyA = bodyA, BodyB = bodyB,
            AnchorA = anchorA, AnchorB = anchorB,
            RefOrientation = null,
            Impulse = Vector3.Zero,
            AngularImpulse = Vector3.Zero,
        });
    }

    // ---- quaternion helpers ----

    /// <summary>Relative rotation from frame A to frame B.</summary>
    public static Quaternion RelativeQuat(Quaternion qa, Quaternion qb)
    {
        return Quaternion.Normalize(Quaternion.Multiply(qb, Quaternion.Inverse(qa)));
    }

    /// <summary>Flips to the hemisphere containing the identity (shortest path).</summary>
    public static Quaternion ShortestPath(Quaternion q)
    {
        return q.W < 0.0f ? new Quaternion(-q.X, -q.Y, -q.Z, -q.W) : q;
    }

    /// <summary>Small-angle rotation vector of a relative orientation.</summary>
    public static Vector3 OrientationError(Quaternion qRel)
    {
        var q = ShortestPath(qRel);
        return new Vector3(q.X, q.Y, q.Z) * 2.0f;
    }

    /// <summary>Signed twist angle of a relative orientation about an axis.</summary>
    public static float TwistAbout(Quaternion qRel, Vector3 axis)
    {
        var q = ShortestPath(qRel);
        float tw = 2.0f * MathF.Atan2(q.X * axis.X + q.Y * axis.Y + q.Z * axis.Z, q.W);
        return WrapAngle(tw);
    }

    /// <summary>Wraps an angle to [-pi, pi].</summary>
    public static float WrapAngle(float a)
    {
        const float tau = MathF.PI * 2.0f;
        a %= tau;
        if (a > MathF.PI) a -= tau;
        else if (a < -MathF.PI) a += tau;
        return a;
    }

    /// <summary>Orthonormal basis perpendicular to a unit vector.</summary>
    public static (Vector3 U1, Vector3 U2) StableBasis3(Vector3 n)
    {
#if DEBUG
        // Rust version validates input is normalized
        if (MathF.Abs(n.LengthSquared() - 1.0f) >= 1e-3f)
            throw new ArgumentException("StableBasis3 requires a unit vector");
#endif
        Vector3 u;
        if (Vector3.Cross(n, Vector3.UnitX).LengthSquared() > 1e-12f)
            u = Vector3.Normalize(Vector3.Cross(n, Vector3.UnitX));
        else
            u = Vector3.Normalize(Vector3.Cross(n, Vector3.UnitY));
        return (u, Vector3.Cross(n, u));
    }

    // ---- mass terms ----

    /// <summary>Effective mass along a direction (same formula as the contact solver).</summary>
    public static float EffectiveMass3(
        float invMassA, float invMassB,
        Matrix3x3 invInertiaA, Matrix3x3 invInertiaB,
        Vector3 rA, Vector3 rB, Vector3 dir)
    {
        var rACross = Vector3.Cross(rA, dir);
        var rBCross = Vector3.Cross(rB, dir);
        float k = invMassA + invMassB +
                  Vector3.Dot(invInertiaA.TransformVector(rACross), rACross) +
                  Vector3.Dot(invInertiaB.TransformVector(rBCross), rBCross);
        return k > 0.0f ? 1.0f / k : 0.0f;
    }

    /// <summary>Effective mass about an axis from inverse inertia tensors.</summary>
    public static float AngularMass3(Matrix3x3 invInertiaA, Matrix3x3 invInertiaB, Vector3 axis)
    {
        float k = Vector3.Dot(invInertiaA.TransformVector(axis), axis) +
                  Vector3.Dot(invInertiaB.TransformVector(axis), axis);
        return k > 0.0f ? 1.0f / k : 0.0f;
    }

    // ---- impulse application ----

    /// <summary>Applies impulse P to body B and -P to body A, with torques.</summary>
    public static void ApplyPairImpulse3(
        List<RigidBody3> bodies, int bodyA, int bodyB,
        Vector3 p, Vector3 pa, Vector3 pb)
    {
        var a = bodies[bodyA];
        var b = bodies[bodyB];
        Vector3 ra = pa - a.Transform.Position;
        Vector3 rb = pb - b.Transform.Position;

        a.Velocity -= p * a.InvMass;
        b.Velocity += p * b.InvMass;

        a.AngularVelocity -= a.WorldInvInertia().TransformVector(Vector3.Cross(ra, p));
        b.AngularVelocity += b.WorldInvInertia().TransformVector(Vector3.Cross(rb, p));

        bodies[bodyA] = a;
        bodies[bodyB] = b;
    }

    /// <summary>Applies a pure angular impulse pair.</summary>
    public static void ApplyAngularVec3(
        List<RigidBody3> bodies, int bodyA, int bodyB, Vector3 j)
    {
        var a = bodies[bodyA];
        var b = bodies[bodyB];
        a.AngularVelocity -= a.WorldInvInertia().TransformVector(j);
        b.AngularVelocity += b.WorldInvInertia().TransformVector(j);
        bodies[bodyA] = a;
        bodies[bodyB] = b;
    }

    /// <summary>Applies a scalar angular impulse about an axis.</summary>
    public static void ApplyAngular3(
        List<RigidBody3> bodies, int bodyA, int bodyB, Vector3 axis, float lambda)
    {
        var a = bodies[bodyA];
        var b = bodies[bodyB];
        a.AngularVelocity -= a.WorldInvInertia().TransformVector(axis * lambda);
        b.AngularVelocity += b.WorldInvInertia().TransformVector(axis * lambda);
        bodies[bodyA] = a;
        bodies[bodyB] = b;
    }

    // ---- per-type solve ----

    /// <summary>World-space anchor A for a distance joint.</summary>
    public static Vector3 DistanceWorldA(DistanceJoint3 d, List<RigidBody3> bodies) =>
        bodies[d.BodyA].Transform.TransformPoint(d.AnchorA);
    /// <summary>World-space anchor B for a distance joint.</summary>
    public static Vector3 DistanceWorldB(DistanceJoint3 d, List<RigidBody3> bodies) =>
        bodies[d.BodyB].Transform.TransformPoint(d.AnchorB);

    /// <summary>World-space anchor A for a hinge joint.</summary>
    public static Vector3 HingeWorldA(HingeJoint3 h, List<RigidBody3> bodies) =>
        bodies[h.BodyA].Transform.TransformPoint(h.AnchorA);
    /// <summary>World-space anchor B for a hinge joint.</summary>
    public static Vector3 HingeWorldB(HingeJoint3 h, List<RigidBody3> bodies) =>
        bodies[h.BodyB].Transform.TransformPoint(h.AnchorB);

    /// <summary>World-space anchor A for a prismatic joint.</summary>
    public static Vector3 PrismaticWorldA(PrismaticJoint3 p, List<RigidBody3> bodies) =>
        bodies[p.BodyA].Transform.TransformPoint(p.AnchorA);
    /// <summary>World-space anchor B for a prismatic joint.</summary>
    public static Vector3 PrismaticWorldB(PrismaticJoint3 p, List<RigidBody3> bodies) =>
        bodies[p.BodyB].Transform.TransformPoint(p.AnchorB);

    /// <summary>World-space anchor A for a fixed joint.</summary>
    public static Vector3 FixedWorldA(FixedJoint3 f, List<RigidBody3> bodies) =>
        bodies[f.BodyA].Transform.TransformPoint(f.AnchorA);
    /// <summary>World-space anchor B for a fixed joint.</summary>
    public static Vector3 FixedWorldB(FixedJoint3 f, List<RigidBody3> bodies) =>
        bodies[f.BodyB].Transform.TransformPoint(f.AnchorB);

    /// <summary>Computes the distance joint axis and current separation.</summary>
    public static (Vector3? Axis, float Dist) AxisInfo(DistanceJoint3 d, List<RigidBody3> bodies)
    {
        Vector3 delta = DistanceWorldB(d, bodies) - DistanceWorldA(d, bodies);
        float dist = delta.Length();
        if (dist > 1e-6f)
            return (delta / dist, dist);
        return (null, dist);
    }

    /// <summary>Applies stored impulse before iterating (warm start) for a distance joint.</summary>
    public static void DistanceWarmStart(ref DistanceJoint3 d, List<RigidBody3> bodies)
    {
        if (d.Impulse == 0.0f) return;
        var (n, _) = AxisInfo(d, bodies);
        if (n == null) return;
        ApplyPairImpulse3(bodies, d.BodyA, d.BodyB, -n.Value * d.Impulse,
            DistanceWorldA(d, bodies), DistanceWorldB(d, bodies));
    }

    /// <summary>Runs one sequential-impulse solve step for a distance joint.</summary>
    public static void DistanceSolve(ref DistanceJoint3 d, List<RigidBody3> bodies, float dt)
    {
        var (nOpt, dist) = AxisInfo(d, bodies);
        if (nOpt == null) return;
        Vector3 n = nOpt.Value;
        float c = dist - d.RestLength;
        if (MathF.Abs(c) < 1e-6f) return;

        Vector3 pa = DistanceWorldA(d, bodies);
        Vector3 pb = DistanceWorldB(d, bodies);
        Vector3 ra = pa - bodies[d.BodyA].Transform.Position;
        Vector3 rb = pb - bodies[d.BodyB].Transform.Position;

        Vector3 va = bodies[d.BodyA].Velocity + Vector3.Cross(bodies[d.BodyA].AngularVelocity, ra);
        Vector3 vb = bodies[d.BodyB].Velocity + Vector3.Cross(bodies[d.BodyB].AngularVelocity, rb);
        float vn = Vector3.Dot(vb - va, n);

        float m = EffectiveMass3(
            bodies[d.BodyA].InvMass, bodies[d.BodyB].InvMass,
            bodies[d.BodyA].WorldInvInertia(), bodies[d.BodyB].WorldInvInertia(),
            ra, rb, n);
        if (m == 0.0f) return;

        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = m * (vn + bias);
        d.Impulse += lambda;
        ApplyPairImpulse3(bodies, d.BodyA, d.BodyB, -n * lambda, pa, pb);
    }

    /// <summary>World-space hinge axis on body A.</summary>
    public static Vector3 HingeAxisAWorld(HingeJoint3 h, List<RigidBody3> bodies) =>
        bodies[h.BodyA].Transform.RotateVector(h.AxisA);
    /// <summary>World-space hinge axis on body B.</summary>
    public static Vector3 HingeAxisBWorld(HingeJoint3 h, List<RigidBody3> bodies) =>
        bodies[h.BodyB].Transform.RotateVector(h.AxisB);

    /// <summary>Applies stored impulse before iterating (warm start) for a hinge joint.</summary>
    public static void HingeWarmStart(ref HingeJoint3 h, List<RigidBody3> bodies)
    {
        Vector3 pa = HingeWorldA(h, bodies);
        Vector3 pb = HingeWorldB(h, bodies);
        if (h.Impulse != Vector3.Zero)
            ApplyPairImpulse3(bodies, h.BodyA, h.BodyB, h.Impulse, pa, pb);
        if (h.AlignU1 != 0.0f || h.AlignU2 != 0.0f)
        {
            var (u1, u2) = StableBasis3(HingeAxisAWorld(h, bodies));
            if (h.AlignU1 != 0.0f)
                ApplyAngular3(bodies, h.BodyA, h.BodyB, u1, h.AlignU1);
            if (h.AlignU2 != 0.0f)
                ApplyAngular3(bodies, h.BodyA, h.BodyB, u2, h.AlignU2);
        }
        if (h.TwistImpulse != 0.0f)
            ApplyAngular3(bodies, h.BodyA, h.BodyB, HingeAxisAWorld(h, bodies), h.TwistImpulse);
    }

    /// <summary>Runs one sequential-impulse solve step for a hinge joint.</summary>
    public static void HingeSolve(ref HingeJoint3 h, List<RigidBody3> bodies, float dt)
    {
        HingeSolvePoint(ref h, bodies, dt);
        HingeSolveAlign(ref h, bodies, dt);
        HingeSolveTwist(ref h, bodies, dt);
    }

    static void HingeSolvePoint(ref HingeJoint3 h, List<RigidBody3> bodies, float dt)
    {
        Vector3 pa = HingeWorldA(h, bodies);
        Vector3 pb = HingeWorldB(h, bodies);
        Vector3 delta = pb - pa;
        if (delta.LengthSquared() < 1e-12f) return;
        foreach (Vector3 axis in new[] { Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ })
        {
            float c = Vector3.Dot(delta, axis);
            if (MathF.Abs(c) < 1e-6f) continue;
            Vector3 ra = pa - bodies[h.BodyA].Transform.Position;
            Vector3 rb = pb - bodies[h.BodyB].Transform.Position;
            Vector3 va = bodies[h.BodyA].Velocity + Vector3.Cross(bodies[h.BodyA].AngularVelocity, ra);
            Vector3 vb = bodies[h.BodyB].Velocity + Vector3.Cross(bodies[h.BodyB].AngularVelocity, rb);
            float vn = Vector3.Dot(vb - va, axis);
            float m = EffectiveMass3(
                bodies[h.BodyA].InvMass, bodies[h.BodyB].InvMass,
                bodies[h.BodyA].WorldInvInertia(), bodies[h.BodyB].WorldInvInertia(),
                ra, rb, axis);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (vn + bias);
            h.Impulse += axis * lambda;
            ApplyPairImpulse3(bodies, h.BodyA, h.BodyB, axis * lambda, pa, pb);
        }
    }

    static void HingeSolveAlign(ref HingeJoint3 h, List<RigidBody3> bodies, float dt)
    {
        Vector3 axA = HingeAxisAWorld(h, bodies);
        Vector3 axB = HingeAxisBWorld(h, bodies);
        Vector3 err = Vector3.Cross(axA, axB);
        if (err.LengthSquared() < 1e-12f) return;
        var (u1, u2) = StableBasis3(axA);
        var ia = bodies[h.BodyA].WorldInvInertia();
        var ib = bodies[h.BodyB].WorldInvInertia();
        Vector3 wRel = bodies[h.BodyB].AngularVelocity - bodies[h.BodyA].AngularVelocity;
        foreach ((Vector3 u, bool isU1) in new[] { (u1, true), (u2, false) })
        {
            float c = Vector3.Dot(err, u);
            if (MathF.Abs(c) < 1e-6f) continue;
            float m = AngularMass3(ia, ib, u);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (Vector3.Dot(wRel, u) + bias);
            if (isU1) h.AlignU1 += lambda; else h.AlignU2 += lambda;
            ApplyAngular3(bodies, h.BodyA, h.BodyB, u, lambda);
        }
    }

    static void HingeSolveTwist(ref HingeJoint3 h, List<RigidBody3> bodies, float dt)
    {
        if (h.LimitMin > h.LimitMax) return;
        Vector3 u = HingeAxisAWorld(h, bodies);
        Quaternion qRel = RelativeQuat(
            bodies[h.BodyA].Transform.Rotation,
            bodies[h.BodyB].Transform.Rotation);
        float tw = TwistAbout(qRel, u);
        float c = tw < h.LimitMin ? tw - h.LimitMin
            : tw > h.LimitMax ? tw - h.LimitMax
            : 0.0f;
        if (c == 0.0f) return;
        var ia = bodies[h.BodyA].WorldInvInertia();
        var ib = bodies[h.BodyB].WorldInvInertia();
        Vector3 wRel = bodies[h.BodyB].AngularVelocity - bodies[h.BodyA].AngularVelocity;
        float m = AngularMass3(ia, ib, u);
        if (m == 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -m * (Vector3.Dot(wRel, u) + bias);
        h.TwistImpulse += lambda;
        ApplyAngular3(bodies, h.BodyA, h.BodyB, u, lambda);
    }

    /// <summary>World-space prismatic slide axis.</summary>
    public static Vector3 PrismaticAxisWorld(PrismaticJoint3 p, List<RigidBody3> bodies) =>
        bodies[p.BodyA].Transform.RotateVector(p.AxisA);

    /// <summary>Applies stored impulse before iterating (warm start) for a prismatic joint.</summary>
    public static void PrismaticWarmStart(ref PrismaticJoint3 p, List<RigidBody3> bodies)
    {
        Vector3 pa = PrismaticWorldA(p, bodies);
        Vector3 pb = PrismaticWorldB(p, bodies);
        if (p.PerpU1 != 0.0f || p.PerpU2 != 0.0f)
        {
            var (u1, u2) = StableBasis3(PrismaticAxisWorld(p, bodies));
            if (p.PerpU1 != 0.0f)
                ApplyPairImpulse3(bodies, p.BodyA, p.BodyB, u1 * p.PerpU1, pa, pb);
            if (p.PerpU2 != 0.0f)
                ApplyPairImpulse3(bodies, p.BodyA, p.BodyB, u2 * p.PerpU2, pa, pb);
        }
        if (p.LimitImpulse != 0.0f)
            ApplyPairImpulse3(bodies, p.BodyA, p.BodyB,
                PrismaticAxisWorld(p, bodies) * p.LimitImpulse, pa, pb);
        if (p.AngularImpulse != Vector3.Zero)
            ApplyAngularVec3(bodies, p.BodyA, p.BodyB, p.AngularImpulse);
    }

    /// <summary>Runs one sequential-impulse solve step for a prismatic joint.</summary>
    public static void PrismaticSolve(ref PrismaticJoint3 p, List<RigidBody3> bodies, float dt)
    {
        if (p.RefOrientation == null)
            p.RefOrientation = RelativeQuat(
                bodies[p.BodyA].Transform.Rotation,
                bodies[p.BodyB].Transform.Rotation);
        PrismaticSolvePerp(ref p, bodies, dt);
        PrismaticSolveAngular(ref p, bodies, dt);
        PrismaticSolveTravel(ref p, bodies, dt);
    }

    static void PrismaticSolvePerp(ref PrismaticJoint3 p, List<RigidBody3> bodies, float dt)
    {
        Vector3 pa = PrismaticWorldA(p, bodies);
        Vector3 pb = PrismaticWorldB(p, bodies);
        Vector3 delta = pb - pa;
        if (delta.LengthSquared() < 1e-12f) return;
        var (u1, u2) = StableBasis3(PrismaticAxisWorld(p, bodies));
        var ia = bodies[p.BodyA].WorldInvInertia();
        var ib = bodies[p.BodyB].WorldInvInertia();
        foreach ((Vector3 u, bool isU1) in new[] { (u1, true), (u2, false) })
        {
            float c = Vector3.Dot(delta, u);
            if (MathF.Abs(c) < 1e-6f) continue;
            Vector3 ra = pa - bodies[p.BodyA].Transform.Position;
            Vector3 rb = pb - bodies[p.BodyB].Transform.Position;
            Vector3 va = bodies[p.BodyA].Velocity + Vector3.Cross(bodies[p.BodyA].AngularVelocity, ra);
            Vector3 vb = bodies[p.BodyB].Velocity + Vector3.Cross(bodies[p.BodyB].AngularVelocity, rb);
            float vn = Vector3.Dot(vb - va, u);
            float m = EffectiveMass3(
                bodies[p.BodyA].InvMass, bodies[p.BodyB].InvMass,
                ia, ib, ra, rb, u);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (vn + bias);
            if (isU1) p.PerpU1 += lambda; else p.PerpU2 += lambda;
            ApplyPairImpulse3(bodies, p.BodyA, p.BodyB, u * lambda, pa, pb);
        }
    }

    static void PrismaticSolveAngular(ref PrismaticJoint3 p, List<RigidBody3> bodies, float dt)
    {
        if (p.RefOrientation == null) return;
        Quaternion qRel = RelativeQuat(
            bodies[p.BodyA].Transform.Rotation,
            bodies[p.BodyB].Transform.Rotation);
        Vector3 err = OrientationError(qRel * Quaternion.Inverse(p.RefOrientation.Value));
        if (err.LengthSquared() < 1e-12f) return;
        var ia = bodies[p.BodyA].WorldInvInertia();
        var ib = bodies[p.BodyB].WorldInvInertia();
        Vector3 wRel = bodies[p.BodyB].AngularVelocity - bodies[p.BodyA].AngularVelocity;
        foreach ((Vector3 axis, int slot) in new[] { (Vector3.UnitX, 0), (Vector3.UnitY, 1), (Vector3.UnitZ, 2) })
        {
            float c = Vector3.Dot(err, axis);
            if (MathF.Abs(c) < 1e-6f) continue;
            float m = AngularMass3(ia, ib, axis);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (Vector3.Dot(wRel, axis) + bias);
            if (slot == 0) p.AngularImpulse.X += lambda;
            else if (slot == 1) p.AngularImpulse.Y += lambda;
            else p.AngularImpulse.Z += lambda;
            ApplyAngular3(bodies, p.BodyA, p.BodyB, axis, lambda);
        }
    }

    static void PrismaticSolveTravel(ref PrismaticJoint3 p, List<RigidBody3> bodies, float dt)
    {
        if (p.LimitMin > p.LimitMax) return;
        Vector3 pa = PrismaticWorldA(p, bodies);
        Vector3 pb = PrismaticWorldB(p, bodies);
        Vector3 ax = PrismaticAxisWorld(p, bodies);
        float s = Vector3.Dot(pb - pa, ax);
        float c = s < p.LimitMin ? s - p.LimitMin
            : s > p.LimitMax ? s - p.LimitMax
            : 0.0f;
        if (c == 0.0f) return;
        Vector3 ra = pa - bodies[p.BodyA].Transform.Position;
        Vector3 rb = pb - bodies[p.BodyB].Transform.Position;
        Vector3 va = bodies[p.BodyA].Velocity + Vector3.Cross(bodies[p.BodyA].AngularVelocity, ra);
        Vector3 vb = bodies[p.BodyB].Velocity + Vector3.Cross(bodies[p.BodyB].AngularVelocity, rb);
        float vn = Vector3.Dot(vb - va, ax);
        float m = EffectiveMass3(
            bodies[p.BodyA].InvMass, bodies[p.BodyB].InvMass,
            bodies[p.BodyA].WorldInvInertia(), bodies[p.BodyB].WorldInvInertia(),
            ra, rb, ax);
        if (m == 0.0f) return;
        float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
        float lambda = -m * (vn + bias);
        p.LimitImpulse += lambda;
        ApplyPairImpulse3(bodies, p.BodyA, p.BodyB, ax * lambda, pa, pb);
    }

    /// <summary>Applies stored impulse before iterating (warm start) for a fixed joint.</summary>
    public static void FixedWarmStart(ref FixedJoint3 f, List<RigidBody3> bodies)
    {
        Vector3 pa = FixedWorldA(f, bodies);
        Vector3 pb = FixedWorldB(f, bodies);
        if (f.Impulse != Vector3.Zero)
            ApplyPairImpulse3(bodies, f.BodyA, f.BodyB, f.Impulse, pa, pb);
        if (f.AngularImpulse != Vector3.Zero)
            ApplyAngularVec3(bodies, f.BodyA, f.BodyB, f.AngularImpulse);
    }

    /// <summary>Runs one sequential-impulse solve step for a fixed joint.</summary>
    public static void FixedSolve(ref FixedJoint3 f, List<RigidBody3> bodies, float dt)
    {
        if (f.RefOrientation == null)
            f.RefOrientation = RelativeQuat(
                bodies[f.BodyA].Transform.Rotation,
                bodies[f.BodyB].Transform.Rotation);
        FixedSolvePoint(ref f, bodies, dt);
        FixedSolveAngular(ref f, bodies, dt);
    }

    static void FixedSolvePoint(ref FixedJoint3 f, List<RigidBody3> bodies, float dt)
    {
        Vector3 pa = FixedWorldA(f, bodies);
        Vector3 pb = FixedWorldB(f, bodies);
        Vector3 delta = pb - pa;
        if (delta.LengthSquared() >= 1e-12f)
        {
            foreach (Vector3 axis in new[] { Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ })
            {
                float c = Vector3.Dot(delta, axis);
                if (MathF.Abs(c) < 1e-6f) continue;
                Vector3 ra = pa - bodies[f.BodyA].Transform.Position;
                Vector3 rb = pb - bodies[f.BodyB].Transform.Position;
                Vector3 va = bodies[f.BodyA].Velocity + Vector3.Cross(bodies[f.BodyA].AngularVelocity, ra);
                Vector3 vb = bodies[f.BodyB].Velocity + Vector3.Cross(bodies[f.BodyB].AngularVelocity, rb);
                float vn = Vector3.Dot(vb - va, axis);
                float m = EffectiveMass3(
                    bodies[f.BodyA].InvMass, bodies[f.BodyB].InvMass,
                    bodies[f.BodyA].WorldInvInertia(), bodies[f.BodyB].WorldInvInertia(),
                    ra, rb, axis);
                if (m == 0.0f) continue;
                float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
                float lambda = -m * (vn + bias);
                f.Impulse += axis * lambda;
                ApplyPairImpulse3(bodies, f.BodyA, f.BodyB, axis * lambda, pa, pb);
            }
        }
    }

    static void FixedSolveAngular(ref FixedJoint3 f, List<RigidBody3> bodies, float dt)
    {
        if (f.RefOrientation == null) return;
        Quaternion qRel = RelativeQuat(
            bodies[f.BodyA].Transform.Rotation,
            bodies[f.BodyB].Transform.Rotation);
        Vector3 err = OrientationError(qRel * Quaternion.Inverse(f.RefOrientation.Value));
        if (err.LengthSquared() < 1e-12f) return;
        var ia = bodies[f.BodyA].WorldInvInertia();
        var ib = bodies[f.BodyB].WorldInvInertia();
        Vector3 wRel = bodies[f.BodyB].AngularVelocity - bodies[f.BodyA].AngularVelocity;
        foreach ((Vector3 axis, int slot) in new[] { (Vector3.UnitX, 0), (Vector3.UnitY, 1), (Vector3.UnitZ, 2) })
        {
            float c = Vector3.Dot(err, axis);
            if (MathF.Abs(c) < 1e-6f) continue;
            float m = AngularMass3(ia, ib, axis);
            if (m == 0.0f) continue;
            float bias = dt > 0.0f ? Baumgarte * c / dt : 0.0f;
            float lambda = -m * (Vector3.Dot(wRel, axis) + bias);
            if (slot == 0) f.AngularImpulse.X += lambda;
            else if (slot == 1) f.AngularImpulse.Y += lambda;
            else f.AngularImpulse.Z += lambda;
            ApplyAngular3(bodies, f.BodyA, f.BodyB, axis, lambda);
        }
    }
}
