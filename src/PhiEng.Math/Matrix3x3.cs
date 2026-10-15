using System.Numerics;

namespace PhiEng.Math;

/// <summary>
/// 3x3 Matrix for inertia tensors
/// </summary>
public struct Matrix3x3
{
    public float M11 { get; set; }
    public float M12 { get; set; }
    public float M13 { get; set; }
    public float M21 { get; set; }
    public float M22 { get; set; }
    public float M23 { get; set; }
    public float M31 { get; set; }
    public float M32 { get; set; }
    public float M33 { get; set; }

    public Matrix3x3()
    {
        M11 = M22 = M33 = 1.0f;
        M12 = M13 = M21 = M23 = M31 = M32 = 0.0f;
    }

    public Matrix3x3(float m11, float m12, float m13,
                     float m21, float m22, float m23,
                     float m31, float m32, float m33)
    {
        M11 = m11; M12 = m12; M13 = m13;
        M21 = m21; M22 = m22; M23 = m23;
        M31 = m31; M32 = m32; M33 = m33;
    }

    public static Matrix3x3 Identity => new(1, 0, 0, 0, 1, 0, 0, 0, 1);
    public static Matrix3x3 Zero => new(0, 0, 0, 0, 0, 0, 0, 0, 0);

    public static Matrix3x3 FromDiagonal(Vector3 diagonal)
    {
        return new Matrix3x3(
            diagonal.X, 0, 0,
            0, diagonal.Y, 0,
            0, 0, diagonal.Z
        );
    }

    public Vector3 TransformVector(Vector3 v)
    {
        return new Vector3(
            M11 * v.X + M12 * v.Y + M13 * v.Z,
            M21 * v.X + M22 * v.Y + M23 * v.Z,
            M31 * v.X + M32 * v.Y + M33 * v.Z
        );
    }

    public Matrix3x3 Transpose()
    {
        return new Matrix3x3(
            M11, M21, M31,
            M12, M22, M32,
            M13, M23, M33
        );
    }

    public static Matrix3x3 Multiply(Matrix3x3 a, Matrix3x3 b)
    {
        return new Matrix3x3(
            a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31,
            a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32,
            a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33,
            a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31,
            a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32,
            a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33,
            a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31,
            a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32,
            a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33
        );
    }

    public static Matrix3x3 FromQuaternion(Quaternion q)
    {
        float xx = q.X * q.X, yy = q.Y * q.Y, zz = q.Z * q.Z;
        float xy = q.X * q.Y, xz = q.X * q.Z, yz = q.Y * q.Z;
        float wx = q.W * q.X, wy = q.W * q.Y, wz = q.W * q.Z;
        return new Matrix3x3(
            1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy),
            2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
            2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy)
        );
    }

    public Matrix3x3 Inverse()
    {
        // For diagonal matrix (common case in physics)
        if (M12 == 0 && M13 == 0 && M21 == 0 && M23 == 0 && M31 == 0 && M32 == 0)
        {
            return new Matrix3x3(
                1.0f / M11, 0, 0,
                0, 1.0f / M22, 0,
                0, 0, 1.0f / M33
            );
        }

        // General inverse using cofactors
        float det = M11 * (M22 * M33 - M23 * M32) -
                    M12 * (M21 * M33 - M23 * M31) +
                    M13 * (M21 * M32 - M22 * M31);

        if (MathF.Abs(det) < 1e-10f)
        {
            return Zero; // Singular matrix
        }

        float invDet = 1.0f / det;

        return new Matrix3x3(
            (M22 * M33 - M23 * M32) * invDet, -(M12 * M33 - M13 * M32) * invDet, (M12 * M23 - M13 * M22) * invDet,
            -(M21 * M33 - M23 * M31) * invDet, (M11 * M33 - M13 * M31) * invDet, -(M11 * M23 - M13 * M21) * invDet,
            (M21 * M32 - M22 * M31) * invDet, -(M11 * M32 - M12 * M31) * invDet, (M11 * M22 - M12 * M21) * invDet
        );
    }
}
