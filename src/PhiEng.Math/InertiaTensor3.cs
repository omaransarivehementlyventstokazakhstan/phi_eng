using System.Numerics;

namespace PhiEng.Math;

/// <summary>
/// 3x3 inertia tensor for rigid body dynamics.
/// Mirrors the Rust <c>InertiaTensor3</c> in crates/math/src/math.rs.
/// World tensor: I_world = R * I_local * R^T.
/// </summary>
public struct InertiaTensor3
{
    public Matrix3x3 M { get; set; }

    public InertiaTensor3(Matrix3x3 m)
    {
        M = m;
    }

    public static InertiaTensor3 FromDiagonal(float ixx, float iyy, float izz)
        => new(Matrix3x3.FromDiagonal(new Vector3(ixx, iyy, izz)));

    public static InertiaTensor3 Sphere(float radius, float mass)
    {
        float i = 0.4f * mass * radius * radius;
        return FromDiagonal(i, i, i);
    }

    public static InertiaTensor3 Box(Vector3 halfExtents, float mass)
    {
        float w = halfExtents.X * 2.0f;
        float h = halfExtents.Y * 2.0f;
        float d = halfExtents.Z * 2.0f;
        return FromDiagonal(
            mass * (h * h + d * d) / 12.0f,
            mass * (w * w + d * d) / 12.0f,
            mass * (w * w + h * h) / 12.0f);
    }

    public static InertiaTensor3 Capsule(float radius, float halfLength, float mass)
    {
        float length = halfLength * 2.0f;
        float vCyl = MathF.PI * radius * radius * length;
        float vSph = (4.0f / 3.0f) * MathF.PI * radius * radius * radius;
        float rho = mass / MathF.Max(1e-12f, vCyl + vSph);
        float mCyl = rho * vCyl;
        float mSph = rho * vSph;
        float transverse = mCyl * (3.0f * radius * radius + length * length) / 12.0f
            + mSph * (0.4f * radius * radius + halfLength * halfLength);
        float axial = 0.5f * mCyl * radius * radius + 0.4f * mSph * radius * radius;
        return FromDiagonal(transverse, axial, transverse);
    }

    public InertiaTensor3 TransformBy(Quaternion rotation)
    {
        var r = Matrix3x3.FromQuaternion(rotation);
        return new InertiaTensor3(Matrix3x3.Multiply(Matrix3x3.Multiply(r, M), r.Transpose()));
    }

    public InertiaTensor3 Inverse() => new(M.Inverse());
    public Vector3 Apply(Vector3 v) => M.TransformVector(v);
}
