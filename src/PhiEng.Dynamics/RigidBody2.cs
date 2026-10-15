using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// 2D Shape enum for collision detection
/// </summary>
public enum Shape2Type
{
    /// <summary>Circle collider.</summary>
    Circle,
    /// <summary>Axis-aligned box collider (in local space).</summary>
    Box,
    /// <summary>Convex polygon collider.</summary>
    Polygon
}

/// <summary>
/// Rigid body in 2D
/// </summary>
public class RigidBody2
{
    /// <summary>Body id (doubles as index into the world's body list).</summary>
    public int Id { get; set; }
    /// <summary>Transform (position + rotation).</summary>
    public Transform2 Transform { get; set; }
    /// <summary>Linear velocity.</summary>
    public Vector2 Velocity { get; set; }
    /// <summary>Angular velocity in radians/sec.</summary>
    public float AngularVelocity { get; set; }
    /// <summary>Mass.</summary>
    public float Mass { get; set; }
    /// <summary>Inverse mass (0 for static bodies).</summary>
    public float InvMass { get; set; }
    /// <summary>Moment of inertia.</summary>
    public float Inertia { get; set; }
    /// <summary>Inverse inertia (0 for static bodies).</summary>
    public float InvInertia { get; set; }
    /// <summary>Linear damping.</summary>
    public float LinearDamping { get; set; }
    /// <summary>Angular damping.</summary>
    public float AngularDamping { get; set; }
    /// <summary>
    /// Surface and mass properties (friction, restitution, density).
    /// Shapes inherit their body's material.
    /// </summary>
    public PhysicsMaterial Material { get; set; }
    /// <summary>Whether the body is immovable.</summary>
    public bool IsStatic { get; set; }
    /// <summary>Whether the body is sleeping.</summary>
    public bool IsSleeping { get; set; }
    /// <summary>Collision shape (null until set).</summary>
    public Shape2? Shape { get; set; }

    /// <summary>Dynamic body with unit mass properties until shaped.</summary>
    public RigidBody2(int id)
    {
        Id = id;
        Transform = new Transform2();
        Velocity = Vector2.Zero;
        AngularVelocity = 0.0f;
        Mass = 1.0f;
        InvMass = 1.0f;
        Inertia = 0.5f;
        InvInertia = 2.0f;
        LinearDamping = 0.01f;
        AngularDamping = 0.01f;
        Material = PhysicsMaterial.Default;
        IsStatic = false;
        IsSleeping = false;
        Shape = null;
    }

    /// <summary>Immovable body (zero inverse mass/inertia).</summary>
    public static RigidBody2 StaticBody(int id)
    {
        var body = new RigidBody2(id);
        body.IsStatic = true;
        body.InvMass = 0.0f;
        body.InvInertia = 0.0f;
        return body;
    }

    /// <summary>
    /// Set shape and recalculate mass properties from the material density.
    /// Also stores <paramref name="density"/> into the material. Rejects
    /// ambiguous input (see <see cref="ShapeValidation"/>) instead of
    /// silently substituting defaults.
    /// </summary>
    public void SetShape(Shape2 shape, float density)
    {
        ShapeValidation.ValidateShape2(shape);
        density = ShapeValidation.ValidateDensity(IsStatic, density);
        // Fresh instance: never mutate a material the user may share.
        var current = Material ?? PhysicsMaterial.Default;
        Material = new PhysicsMaterial(current.Friction, current.Restitution, density);
        Shape = shape;
        switch (shape.Type)
        {
            case Shape2Type.Circle:
                var circle = shape as CircleShape;
                if (circle != null)
                {
                    float area = MathF.PI * circle.Radius * circle.Radius;
                    Mass = area * density;
                    Inertia = 0.5f * Mass * circle.Radius * circle.Radius;
                }
                break;
            case Shape2Type.Box:
                var box = shape as BoxShape2;
                if (box != null)
                {
                    float width = box.HalfExtents.X * 2.0f;
                    float height = box.HalfExtents.Y * 2.0f;
                    float area = width * height;
                    Mass = area * density;
                    Inertia = Mass * (width * width + height * height) / 12.0f;
                }
                break;
            case Shape2Type.Polygon:
                var poly = shape as PolygonShape;
                if (poly != null)
                {
                    var (area, _) = PolygonAreaAndCentroid(poly.Vertices);
                    Mass = MathF.Abs(area) * density;
                    Inertia = PolygonInertia(poly.Vertices, Mass);
                }
                break;
        }
        ApplyMass(Mass, Inertia);
    }

    private static (float Area, Vector2 Centroid) PolygonAreaAndCentroid(List<Vector2> vertices)
    {
        float area = 0.0f;
        Vector2 centroid = Vector2.Zero;
        for (int i = 0; i < vertices.Count; i++)
        {
            int j = (i + 1) % vertices.Count;
            float cross = vertices[i].X * vertices[j].Y - vertices[j].X * vertices[i].Y;
            area += cross;
            centroid += (vertices[i] + vertices[j]) * cross;
        }
        area *= 0.5f;
        centroid /= area * 6.0f;
        return (area, centroid);
    }

    private static float PolygonInertia(List<Vector2> vertices, float mass)
    {
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            int j = (i + 1) % vertices.Count;
            var a = vertices[i];
            var b = vertices[j];
            float cross = a.X * b.Y - b.X * a.Y;
            float dotA = a.X * a.X + a.Y * a.Y;
            float dotB = b.X * b.X + b.Y * b.Y;
            numerator += (dotA + dotB + a.X * b.X + a.Y * b.Y) * cross;
            denominator += cross;
        }
        if (MathF.Abs(denominator) < 1e-6f) return 0.0f;
        return mass * numerator / (6.0f * denominator);
    }

    /// <summary>
    /// Assign a full material (defensive copy), recomputing mass/inertia
    /// from the current shape (if any) with the material's density.
    /// Friction keeps sanitize-to-range behavior; density follows the
    /// rejection rule (see <see cref="ShapeValidation"/>).
    /// </summary>
    public void SetMaterial(PhysicsMaterial material)
    {
        Material = material.Clone();
        // Density rejects (no silent defaulting); friction sanitize lives
        // in the PhysicsMaterial constructor.
        Material.Density = ShapeValidation.ValidateDensity(IsStatic, Material.Density);
        if (Shape != null)
        {
            // SetShape preserves the material's friction/restitution and
            // recomputes mass/inertia with the new density.
            SetShape(Shape, Material.Density);
        }
    }

    /// <summary>Store mass properties, honoring the static flag for inverses.</summary>
    private void ApplyMass(float mass, float inertia)
    {
        mass = MathF.Max(PhysicsMaterial.MinDensity, mass);
        inertia = MathF.Max(PhysicsMaterial.MinDensity, inertia);
        Mass = mass;
        InvMass = IsStatic ? 0.0f : 1.0f / Mass;
        Inertia = inertia;
        InvInertia = IsStatic ? 0.0f : 1.0f / Inertia;
    }

    /// <summary>
    /// Apply a force at a point for <paramref name="dt"/> seconds
    /// (impulse F * dt). Mirrors the Rust <c>apply_force</c>.
    /// </summary>
    public void ApplyForce(Vector2 force, Vector2 point, float dt)
    {
        if (IsStatic) return;

        Velocity += force * InvMass * dt;

        Vector2 r = point - Transform.Position;
        float torque = r.X * force.Y - r.Y * force.X;
        AngularVelocity += torque * InvInertia * dt;
    }

    /// <summary>Applies an impulse at a point (linear + torque).</summary>
    public void ApplyImpulse(Vector2 impulse, Vector2 point)
    {
        if (IsStatic) return;

        Velocity += impulse * InvMass;

        Vector2 r = point - Transform.Position;
        float torque = r.X * impulse.Y - r.Y * impulse.X;
        AngularVelocity += torque * InvInertia;
    }

    /// <summary>Integrates motion with semi-implicit Euler (applies damping).</summary>
    public void Integrate(float dt)
    {
        if (IsStatic) return;

        // Apply damping
        float damping = MathF.Min(1.0f, LinearDamping * dt);
        Velocity *= 1.0f - damping;

        float angDamping = MathF.Min(1.0f, AngularDamping * dt);
        AngularVelocity *= 1.0f - angDamping;

        // Update position
        Vector2 newPosition = Transform.Position + Velocity * dt;

        // Update rotation - extract angle from matrix.
        // Matrix3x2.CreateRotation(a) has M11=cos(a), M21=-sin(a), so the
        // angle is Atan2(-M21, M11). (Plain Atan2(M21, M11) returns -angle
        // and makes bodies ping-pong between +angle and -angle every frame.)
        float currentAngle = MathF.Atan2(-Transform.Rotation.M21, Transform.Rotation.M11);
        float newAngle = currentAngle + AngularVelocity * dt;
        Matrix3x2 newRotation = Matrix3x2.CreateRotation(newAngle);

        // Create new transform with updated values
        Transform = new Transform2(newPosition, newAngle);
    }

    /// <summary>Current rotation angle in radians.</summary>
    public float RotationAngle => MathF.Atan2(-Transform.Rotation.M21, Transform.Rotation.M11);

    /// <summary>Whether the body is slow enough to sleep.</summary>
    public bool ShouldSleep(float linearThreshold, float angularThreshold)
    {
        if (IsStatic) return true;

        float linearSpeed = Velocity.Length();
        float angularSpeed = MathF.Abs(AngularVelocity);

        return linearSpeed < linearThreshold && angularSpeed < angularThreshold;
    }
}

/// <summary>
/// 2D Shape base class
/// </summary>
public abstract class Shape2
{
    /// <summary>Shape discriminator.</summary>
    public Shape2Type Type { get; set; }
}

/// <summary>
/// Circle shape
/// </summary>
public class CircleShape : Shape2
{
    /// <summary>Circle radius (must be positive).</summary>
    public float Radius { get; set; }

    /// <summary>Circle collider.</summary>
    public CircleShape(float radius)
    {
        Radius = radius;
        Type = Shape2Type.Circle;
    }
}

/// <summary>
/// Box shape in 2D
/// </summary>
public class BoxShape2 : Shape2
{
    /// <summary>Box half extents (must be positive).</summary>
    public Vector2 HalfExtents { get; set; }

    /// <summary>Axis-aligned box collider (in local space).</summary>
    public BoxShape2(Vector2 halfExtents)
    {
        HalfExtents = halfExtents;
        Type = Shape2Type.Box;
    }
}

/// <summary>
/// Polygon shape in 2D
/// </summary>
public class PolygonShape : Shape2
{
    /// <summary>Convex polygon vertices.</summary>
    public List<Vector2> Vertices { get; set; } = new();

    /// <summary>Convex polygon collider.</summary>
    public PolygonShape(List<Vector2> vertices)
    {
        Vertices = vertices;
        Type = Shape2Type.Polygon;
    }
}
