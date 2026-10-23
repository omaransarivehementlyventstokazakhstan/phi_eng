using System.Numerics;

namespace PhiEng.Math;

/// <summary>
/// Quaternion rotation with an exactly mirrored operation order.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust side's <c>glam::Quat::mul_vec3</c> term for
/// term, so the two engines agree bit-for-bit:
/// <c>v*(w*w-b.b) + b*((v.b)*2) + (b×v)*(w*2)</c>. The platform
/// <c>Vector3.Transform</c> computes the same rotation through a
/// different association and lands a ULP away on general inputs, so
/// code that must match the core bit-for-bit rotates here.
/// </para>
/// </remarks>
public static class DeterministicRotation
{
    /// <summary>Rotate <paramref name="v"/> by the unit quaternion <paramref name="q"/>.</summary>
    public static Vector3 RotateVector(Quaternion q, Vector3 v)
    {
        float w = q.W;
        var b = new Vector3(q.X, q.Y, q.Z);
        float b2 = Vector3.Dot(b, b);
        var t1 = v * (w * w - b2);
        var t2 = b * (Vector3.Dot(v, b) * 2.0f);
        var t3 = Vector3.Cross(b, v) * (w * 2.0f);
        return t1 + t2 + t3;
    }

    /// <summary>
    /// Normalize a quaternion the way the Rust side (glam) does:
    /// component-wise division by the length.
    /// </summary>
    /// <remarks>
    /// The compiled core normalizes SIMD lanes by division, not by
    /// scaling with the reciprocal (that is only what the scalar
    /// fallback source reads as). Division and reciprocal-scaling
    /// round differently, so code that must match the core bit-for-bit
    /// normalizes here. The length-squared sums pairwise, as the
    /// compiled dot does.
    /// </remarks>
    public static Quaternion Normalize(Quaternion q)
    {
        // One operation per statement throughout.
        float xx = q.X * q.X;
        float yy = q.Y * q.Y;
        float zz = q.Z * q.Z;
        float ww = q.W * q.W;
        float xz = xx + zz;
        float yw = yy + ww;
        float ls = xz + yw;
        float len = MathF.Sqrt(ls);
        float nx = q.X / len;
        float ny = q.Y / len;
        float nz = q.Z / len;
        float nw = q.W / len;
        return new Quaternion(nx, ny, nz, nw);
    }

    /// <summary>
    /// Compose two rotations: <paramref name="a"/> after <paramref name="b"/>.
    /// </summary>
    /// <remarks>
    /// The Hamilton product in exactly the association the Rust side
    /// (glam) uses. The platform multiply associates its twelve
    /// additions differently and lands ULPs away on general inputs,
    /// which rotation integration then carries into every attitude.
    /// </remarks>
    public static Quaternion Multiply(Quaternion a, Quaternion b)
    {
        // The compiled core (SIMD shuffle multiply) pairs the terms:
        // (t0+t1) + (t2-t3) per lane — never the scalar left fold.
        // Products first, then pure-add combines, so no statement
        // mixes a multiply into an add (which the JIT could fuse).
        float ax = a.W * b.X;
        float bx = a.X * b.W;
        float cx = a.Y * b.Z;
        float dx = a.Z * b.Y;
        float ay = a.W * b.Y;
        float by = a.X * b.Z;
        float cy = a.Y * b.W;
        float dy = a.Z * b.X;
        float az = a.W * b.Z;
        float bz = a.X * b.Y;
        float cz = a.Y * b.X;
        float dz = a.Z * b.W;
        float aw = a.W * b.W;
        float bw = a.X * b.X;
        float cw = a.Y * b.Y;
        float dw = a.Z * b.Z;
        float qx0 = ax + bx;
        float qx1 = cx - dx;
        float qx = qx0 + qx1;
        float qy0 = ay - by;
        float qy1 = cy + dy;
        float qy = qy0 + qy1;
        float qz0 = az + bz;
        float qz1 = dz - cz;
        float qz = qz0 + qz1;
        float qw0 = aw - bw;
        float qw2 = 0.0f - cw;
        float qw3 = qw2 - dw;
        float qw = qw0 + qw3;
        return new Quaternion(qx, qy, qz, qw);
    }
}
