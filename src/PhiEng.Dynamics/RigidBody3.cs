using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>3D shape discriminator.</summary>
public enum Shape3Type
{
    /// <summary>Sphere collider.</summary>
    Sphere,
    /// <summary>Axis-aligned box collider.</summary>
    Box,
    /// <summary>Capsule collider.</summary>
    Capsule,
    /// <summary>Convex hull collider.</summary>
    ConvexHull,
}

/// <summary>3D shape base class.</summary>
public abstract class Shape3
{
    /// <summary>Shape discriminator.</summary>
    public Shape3Type Type { get; set; }
}

/// <summary>Sphere collider.</summary>
public class SphereShape3 : Shape3
{
    /// <summary>Sphere radius (must be positive).</summary>
    public float Radius { get; set; }
    /// <summary>Sphere collider.</summary>
    public SphereShape3(float radius) { Radius = radius; Type = Shape3Type.Sphere; }
}

/// <summary>Axis-aligned box collider (in local space).</summary>
public class BoxShape3 : Shape3
{
    /// <summary>Box half extents (must be positive).</summary>
    public Vector3 HalfExtents { get; set; }
    /// <summary>Box collider.</summary>
    public BoxShape3(Vector3 halfExtents) { HalfExtents = halfExtents; Type = Shape3Type.Box; }
}

/// <summary>Capsule collider aligned with the local Y axis.</summary>
public class CapsuleShape3 : Shape3
{
    /// <summary>Capsule radius (must be positive).</summary>
    public float Radius { get; set; }
    /// <summary>Capsule half length (must be positive; use sphere for zero length).</summary>
    public float HalfLength { get; set; }
    /// <summary>Capsule collider.</summary>
    public CapsuleShape3(float radius, float halfLength)
    {
        Radius = radius; HalfLength = halfLength; Type = Shape3Type.Capsule;
    }
}

/// <summary>Convex hull collider (vertex cloud).</summary>
public class ConvexHullShape3 : Shape3
{
    /// <summary>Hull vertices.</summary>
    public List<Vector3> Vertices { get; set; } = new();
    /// <summary>Convex hull collider.</summary>
    public ConvexHullShape3(List<Vector3> vertices) { Vertices = vertices; Type = Shape3Type.ConvexHull; }
}

/// <summary>
/// 3D rigid body with 6 DOF. Mirrors Rust <c>RigidBody3</c>.
/// </summary>
public class RigidBody3
{
    /// <summary>Body id (doubles as index into the world's body list).</summary>
    public int Id { get; set; }
    /// <summary>Transform (position + rotation).</summary>
    public Transform3 Transform { get; set; }
    /// <summary>Linear velocity.</summary>
    public Vector3 Velocity { get; set; }
    /// <summary>Angular velocity in world space.</summary>
    public Vector3 AngularVelocity { get; set; }
    /// <summary>Mass.</summary>
    public float Mass { get; set; }
    /// <summary>Inverse mass (0 for static bodies).</summary>
    public float InvMass { get; set; }
    /// <summary>Moment of inertia tensor (body frame).</summary>
    public Matrix3x3 Inertia { get; set; }
    /// <summary>Inverse inertia tensor (zero for static bodies).</summary>
    public Matrix3x3 InvInertia { get; set; }
    /// <summary>Linear damping.</summary>
    public float LinearDamping { get; set; }
    /// <summary>Angular damping.</summary>
    public float AngularDamping { get; set; }
    /// <summary>
    /// Gravitational coupling multiplier (see
    /// <c>RigidBody2.GravityScale</c>).
    /// </summary>
    public float GravityScale { get; set; }
    /// <summary>
    /// Collision group this body belongs to. A pair may collide only
    /// when each body accepts the other's group. A group is one bit
    /// (the Box2D convention). Mirrors Rust
    /// <c>RigidBody2::collision_group</c>.
    /// </summary>
    public uint CollisionGroup { get; set; } = CollisionFilter.DefaultGroup;
    /// <summary>
    /// Collision groups this body accepts, as a bit field of the group
    /// flags. Mirrors Rust
    /// <c>RigidBody2::collision_mask</c>; the default accepts every
    /// layer, so an unconfigured body collides with everything.
    /// </summary>
    public uint CollisionMask { get; set; } = CollisionFilter.DefaultMask;
    /// <summary>
    /// Is a sensor: this body reports contacts but never pushes. A
    /// sensor's contacts are reported as contact events and are not
    /// solved, so the sensor falls through whatever it touches. That
    /// is what a trigger volume is. Mirrors Rust
    /// <c>RigidBody3::is_sensor</c>.
    /// </summary>
    public bool IsSensor { get; set; }
    /// <summary>
    /// Ambient-force coupling multiplier (see <c>RigidBody2.AmbientScale</c>).
    /// </summary>
    public float AmbientScale { get; set; }
    /// <summary>
    /// Current temperature in Kelvin. Mirrors Rust
    /// <c>RigidBody3::temperature</c>; default 293.15 K.
    /// </summary>
    public float Temperature { get; set; }

    /// <summary>
    /// The nuclear state (cold: the nuclear phase). See
    /// <see cref="RigidBody2.Nuclear"/>.
    /// </summary>
    public NuclearState Nuclear { get; } = new();
    /// <summary>
    /// Thermal-expansion factor currently applied to this body's shape
    /// (1.0 = not expanded). See <see cref="RigidBody2.ExpansionScale"/>.
    /// </summary>
    public float ExpansionScale { get; set; }
    /// <summary>Surface and mass properties (shapes inherit it).</summary>
    public PhysicsMaterial Material { get; set; }
    /// <summary>Whether the body is immovable.</summary>
    public bool IsStatic { get; set; }
    /// <summary>Whether the body is sleeping.</summary>
    public bool IsSleeping { get; set; }
    /// <summary>Collision shape (null until set).</summary>
    public Shape3? Shape { get; set; }

    /// <summary>Dynamic body with unit mass properties until shaped.</summary>
    public RigidBody3(int id)
    {
        Id = id;
        Transform = new Transform3();
        Velocity = Vector3.Zero;
        AngularVelocity = Vector3.Zero;
        Mass = 1.0f;
        InvMass = 1.0f;
        Inertia = Matrix3x3.Identity;
        InvInertia = Matrix3x3.Identity;
        LinearDamping = 0.01f;
        AngularDamping = 0.01f;
        GravityScale = 1.0f;
        CollisionGroup = CollisionFilter.DefaultGroup;
        CollisionMask = CollisionFilter.DefaultMask;
        AmbientScale = 1.0f;
        Temperature = ThermalConfig2.DefaultTemperature;
        ExpansionScale = 1.0f;
        Material = PhysicsMaterial.Default;
        IsStatic = false;
        IsSleeping = false;
        Shape = null;
    }

    /// <summary>Immovable body (zero inverse mass/inertia).</summary>
    public static RigidBody3 StaticBody(int id)
    {
        var body = new RigidBody3(id);
        body.IsStatic = true;
        body.InvMass = 0.0f;
        body.InvInertia = Matrix3x3.Zero;
        return body;
    }

    /// <summary>Sets the shape and recalculates mass properties from density.</summary>
    public void SetShape(Shape3 shape, float density)
    {
        ShapeValidation.ValidateShape3(shape);
        density = ShapeValidation.ValidateDensity(IsStatic, density);
        var current = Material ?? PhysicsMaterial.Default;
        // Only the density changes; the thermal properties (specific heat,
        // conductivity, emissivity) are carried over, matching the Rust
        // `set_shape` (which touches `material.density` only).
        Material = new PhysicsMaterial(
            current.Friction,
            current.Restitution,
            density,
            current.SpecificHeat,
            current.ThermalConductivity,
            current.Emissivity);
        Shape = shape;
        var (mass, inertia) = MassProperties(shape, density);
        ApplyMass(mass, inertia);
    }

    /// <summary>Assigns a full material, recomputing mass/inertia from its density.</summary>
    public void SetMaterial(PhysicsMaterial material)
    {
        Material = material.Clone();
        Material.Density = ShapeValidation.ValidateDensity(IsStatic, Material.Density);
        if (Shape != null)
            SetShape(Shape, Material.Density);
    }

    private void ApplyMass(float mass, Matrix3x3 inertia)
    {
        mass = MathF.Max(PhysicsMaterial.MinDensity, mass);
        Mass = mass;
        InvMass = IsStatic ? 0.0f : 1.0f / Mass;
        Inertia = inertia;
        InvInertia = IsStatic ? Matrix3x3.Zero : inertia.Inverse();
    }

    private static (float Mass, Matrix3x3 Inertia) MassProperties(Shape3 shape, float density)
    {
        switch (shape)
        {
            case SphereShape3 s:
            {
                float volume = (4.0f / 3.0f) * MathF.PI * s.Radius * s.Radius * s.Radius;
                float mass = volume * density;
                float i = 0.4f * mass * s.Radius * s.Radius;
                return (mass, Matrix3x3.FromDiagonal(new Vector3(i, i, i)));
            }
            case BoxShape3 b:
            {
                float w = b.HalfExtents.X * 2, h = b.HalfExtents.Y * 2, d = b.HalfExtents.Z * 2;
                float mass = w * h * d * density;
                return (mass, Matrix3x3.FromDiagonal(new Vector3(
                    mass * (h * h + d * d) / 12.0f,
                    mass * (w * w + d * d) / 12.0f,
                    mass * (w * w + h * h) / 12.0f)));
            }
            case CapsuleShape3 c:
            {
                // Y-axis capsule: cylinder + two hemispherical caps.
                // Mirrors InertiaTensor3::capsule in math.rs.
                float length = c.HalfLength * 2.0f;
                float vCyl = MathF.PI * c.Radius * c.Radius * length;
                float vSph = (4.0f / 3.0f) * MathF.PI * c.Radius * c.Radius * c.Radius;
                float mCyl = density * vCyl;
                float mSph = density * vSph;
                float transverse = mCyl * (3.0f * c.Radius * c.Radius + length * length) / 12.0f
                    + mSph * (0.4f * c.Radius * c.Radius + c.HalfLength * c.HalfLength);
                float axial = 0.5f * mCyl * c.Radius * c.Radius + 0.4f * mSph * c.Radius * c.Radius;
                float mass = mCyl + mSph;
                return (mass, Matrix3x3.FromDiagonal(new Vector3(transverse, axial, transverse)));
            }
            default:
                return (1.0f, Matrix3x3.Identity);
        }
    }

    /// <summary>
    /// Inverse inertia in world space: R * I^-1 * R^T. Torques live in
    /// world space and must use this, never the body-frame tensor.
    /// Mirrors the Rust <c>world_inv_inertia</c>.
    /// </summary>
    public Matrix3x3 WorldInvInertia()
    {
        var r = Matrix3x3.FromQuaternion(Transform.Rotation);
        return Matrix3x3.Multiply(Matrix3x3.Multiply(r, InvInertia), r.Transpose());
    }

    /// <summary>Applies an impulse at a point (linear + torque).</summary>
    public void ApplyImpulse(Vector3 impulse, Vector3 point)
    {
        if (IsStatic) return;
        // An externally applied impulse is a wake event.
        IsSleeping = false;
        Velocity += impulse * InvMass;
        var r = point - Transform.Position;
        AngularVelocity += WorldInvInertia().TransformVector(Vector3.Cross(r, impulse));
    }

    /// <summary>
    /// Apply a force at a point for <paramref name="dt"/> seconds
    /// (impulse F * dt). Mirrors the Rust <c>apply_force</c>.
    /// </summary>
    public void ApplyForce(Vector3 force, Vector3 point, float dt)
    {
        if (IsStatic) return;
        // An externally applied force is a wake event.
        IsSleeping = false;
        Velocity += force * InvMass * dt;
        var r = point - Transform.Position;
        AngularVelocity += WorldInvInertia().TransformVector(Vector3.Cross(r, force)) * dt;
    }

    /// <summary>Integrates motion with semi-implicit Euler (applies damping).</summary>
    public void Integrate(float dt)
    {
        if (IsStatic) return;
        Velocity *= 1.0f - MathF.Min(1.0f, LinearDamping * dt);
        AngularVelocity *= 1.0f - MathF.Min(1.0f, AngularDamping * dt);
        var t = Transform;
        t.Position += Velocity * dt;
        t.Rotation = IntegrateRotation(t.Rotation, AngularVelocity, dt);
        Transform = t;
    }

    /// <summary>
    /// Rotation update shared by <see cref="Integrate"/> and
    /// <see cref="IntegrateDrift"/> (single source of truth for the
    /// left-composition convention).
    /// </summary>
    private static Quaternion IntegrateRotation(Quaternion rotation, Vector3 angularVelocity, float dt)
    {
        float angSpeed = angularVelocity.Length();
        // The gate is on the angle, not the speed: a slow spin over a
        // long step still turns. Mirrors the Rust integrate exactly.
        float angle = angSpeed * dt;
        if (angle > 1e-6f)
        {
            // Angular velocity is world-space, so the increment composes
            // on the left (mirrors the Rust integrate).
            var axis = angularVelocity / angSpeed;
            // Built from the deterministic sine and cosine rather than
            // Quaternion.CreateFromAxisAngle, which uses the platform math
            // library and therefore differs between Windows and Linux in
            // the last ULP. See PhiEng.Math.DeterministicTrig.
            var (cosHalf, sinHalf) = DeterministicTrig.ComputeCosSin(angle * 0.5f);
            var dq = new Quaternion(axis.X * sinHalf, axis.Y * sinHalf, axis.Z * sinHalf, cosHalf);
            return DeterministicRotation.Normalize(DeterministicRotation.Multiply(dq, rotation));
        }
        return rotation;
    }

    /// <summary>
    /// Drift step for velocity-Verlet integration (see
    /// <c>Gravity</c>): positions and rotation advance a full step with
    /// the post-kick velocities. Linear damping is NOT applied here —
    /// in Verlet mode it rides the gravity kicks as symmetric drag.
    /// Angular damping matches <see cref="Integrate"/>.
    /// </summary>
    public void IntegrateDrift(float dt)
    {
        if (IsStatic) return;
        AngularVelocity *= 1.0f - MathF.Min(1.0f, AngularDamping * dt);
        var t = Transform;
        t.Position += Velocity * dt;
        t.Rotation = IntegrateRotation(t.Rotation, AngularVelocity, dt);
        Transform = t;
    }

    /// <summary>Whether the body is slow enough to sleep.</summary>
    public bool ShouldSleep(float linearThreshold, float angularThreshold)
    {
        if (IsStatic) return true;
        return Velocity.Length() < linearThreshold && AngularVelocity.Length() < angularThreshold;
    }
}
