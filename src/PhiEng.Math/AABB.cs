using System.Numerics;

namespace PhiEng.Math;

/// <summary>
/// Axis-Aligned Bounding Box in 2D
/// </summary>
public struct AABB2
{
    public Vector2 Min { get; set; }
    public Vector2 Max { get; set; }

    public AABB2()
    {
        Min = Vector2.Zero;
        Max = Vector2.Zero;
    }

    public AABB2(Vector2 min, Vector2 max)
    {
        Min = min;
        Max = max;
    }

    public static AABB2 FromCenterHalfExtents(Vector2 center, Vector2 halfExtents)
    {
        return new AABB2(center - halfExtents, center + halfExtents);
    }

    public Vector2 Center => (Min + Max) * 0.5f;
    public Vector2 HalfExtents => (Max - Min) * 0.5f;
    public Vector2 Size => Max - Min;
    public float Volume => Size.X * Size.Y;

    public bool Contains(Vector2 point)
    {
        return point.X >= Min.X && point.X <= Max.X &&
               point.Y >= Min.Y && point.Y <= Max.Y;
    }

    public bool Intersects(AABB2 other)
    {
        return Min.X <= other.Max.X && Max.X >= other.Min.X &&
               Min.Y <= other.Max.Y && Max.Y >= other.Min.Y;
    }

    public AABB2 Merge(AABB2 other)
    {
        return new AABB2(
            Vector2.Min(Min, other.Min),
            Vector2.Max(Max, other.Max)
        );
    }

    public AABB2 Expand(Vector2 amount)
    {
        return new AABB2(Min - amount, Max + amount);
    }
}

/// <summary>
/// Axis-Aligned Bounding Box in 3D
/// </summary>
public struct AABB3
{
    public Vector3 Min { get; set; }
    public Vector3 Max { get; set; }

    public AABB3()
    {
        Min = Vector3.Zero;
        Max = Vector3.Zero;
    }

    public AABB3(Vector3 min, Vector3 max)
    {
        Min = min;
        Max = max;
    }

    public static AABB3 FromCenterHalfExtents(Vector3 center, Vector3 halfExtents)
    {
        return new AABB3(center - halfExtents, center + halfExtents);
    }

    public Vector3 Center => (Min + Max) * 0.5f;
    public Vector3 HalfExtents => (Max - Min) * 0.5f;
    public Vector3 Size => Max - Min;
    public float Volume => Size.X * Size.Y * Size.Z;

    public bool Contains(Vector3 point)
    {
        return point.X >= Min.X && point.X <= Max.X &&
               point.Y >= Min.Y && point.Y <= Max.Y &&
               point.Z >= Min.Z && point.Z <= Max.Z;
    }

    public bool Intersects(AABB3 other)
    {
        return Min.X <= other.Max.X && Max.X >= other.Min.X &&
               Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
               Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
    }

    public AABB3 Merge(AABB3 other)
    {
        return new AABB3(
            Vector3.Min(Min, other.Min),
            Vector3.Max(Max, other.Max)
        );
    }

    public AABB3 Expand(Vector3 amount)
    {
        return new AABB3(Min - amount, Max + amount);
    }
}
