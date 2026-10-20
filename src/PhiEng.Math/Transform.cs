using System.Numerics;

namespace PhiEng.Math;

/// <summary>
/// 2D Transform with position and rotation
/// </summary>
public struct Transform2
{
    public Vector2 Position { get; set; }
    public Matrix3x2 Rotation { get; set; }

    public Transform2()
    {
        Position = Vector2.Zero;
        Rotation = Matrix3x2.Identity;
    }

    public Transform2(Vector2 position, float angle)
    {
        Position = position;
        // Built from the deterministic sine and cosine rather than
        // Matrix3x2.CreateRotation, which uses the platform math library
        // and therefore differs between Windows and Linux in the last
        // ULP. See DeterministicTrig for why that matters.
        var (cos, sin) = DeterministicTrig.ComputeCosSin(angle);
        Rotation = new Matrix3x2(cos, sin, -sin, cos, 0.0f, 0.0f);
    }

    public Vector2 TransformPoint(Vector2 point)
    {
        return Vector2.Transform(point, Rotation) + Position;
    }

    public Vector2 RotateVector(Vector2 vector)
    {
        return Vector2.Transform(vector, Rotation);
    }
}

/// <summary>
/// 3D Transform with position and rotation
/// </summary>
public struct Transform3
{
    public Vector3 Position { get; set; }
    public Quaternion Rotation { get; set; }

    public Transform3()
    {
        Position = Vector3.Zero;
        Rotation = Quaternion.Identity;
    }

    public Transform3(Vector3 position, Quaternion rotation)
    {
        Position = position;
        Rotation = rotation;
    }

    public Vector3 TransformPoint(Vector3 point)
    {
        return Vector3.Transform(point, Rotation) + Position;
    }

    public Vector3 RotateVector(Vector3 vector)
    {
        return Vector3.Transform(vector, Rotation);
    }
}
