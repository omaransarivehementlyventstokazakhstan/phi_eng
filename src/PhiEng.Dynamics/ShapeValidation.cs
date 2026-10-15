using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Boundary rule (never-guess policy), mirroring Rust <c>ShapeError</c> +
/// <c>validate_shape2/3</c>: ambiguous input is rejected at creation with
/// <see cref="ArgumentException"/> instead of being silently clamped.
/// Friction/restitution keep sanitize-to-range behavior (universally-agreed
/// ranges); density and geometry reject.
/// </summary>
internal static class ShapeValidation
{
    public static void ValidateShape2(Shape2 shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        switch (shape)
        {
            case CircleShape c:
                if (!float.IsFinite(c.Radius))
                    throw new ArgumentException("circle radius must be finite", nameof(shape));
                if (c.Radius <= 0.0f)
                    throw new ArgumentException("circle radius must be positive", nameof(shape));
                break;
            case BoxShape2 b:
                if (!float.IsFinite(b.HalfExtents.X) || !float.IsFinite(b.HalfExtents.Y))
                    throw new ArgumentException("box half-extents must be finite", nameof(shape));
                if (b.HalfExtents.X <= 0.0f || b.HalfExtents.Y <= 0.0f)
                    throw new ArgumentException("box half-extents must be positive", nameof(shape));
                break;
            case PolygonShape p:
                if (p.Vertices == null || p.Vertices.Any(v => !float.IsFinite(v.X) || !float.IsFinite(v.Y)))
                    throw new ArgumentException("polygon vertices must be finite", nameof(shape));
                // Count first: area helpers divide by length.
                if (p.Vertices.Count < 3 || DistinctCount(p.Vertices) < 3)
                    throw new ArgumentException("polygon needs 3+ distinct vertices", nameof(shape));
                if (PolygonArea(p.Vertices) == 0.0f)
                    throw new ArgumentException("polygon needs nonzero area", nameof(shape));
                break;
            default:
                throw new ArgumentException($"unknown 2D shape type {shape.GetType().Name}", nameof(shape));
        }
    }

    public static void ValidateShape3(Shape3 shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        switch (shape)
        {
            case SphereShape3 s:
                if (!float.IsFinite(s.Radius))
                    throw new ArgumentException("sphere radius must be finite", nameof(shape));
                if (s.Radius <= 0.0f)
                    throw new ArgumentException("sphere radius must be positive", nameof(shape));
                break;
            case BoxShape3 b:
                if (!float.IsFinite(b.HalfExtents.X) || !float.IsFinite(b.HalfExtents.Y) || !float.IsFinite(b.HalfExtents.Z))
                    throw new ArgumentException("box half-extents must be finite", nameof(shape));
                if (b.HalfExtents.X <= 0.0f || b.HalfExtents.Y <= 0.0f || b.HalfExtents.Z <= 0.0f)
                    throw new ArgumentException("box half-extents must be positive", nameof(shape));
                break;
            case CapsuleShape3 c:
                if (!float.IsFinite(c.Radius) || !float.IsFinite(c.HalfLength))
                    throw new ArgumentException("capsule params must be finite", nameof(shape));
                if (c.Radius <= 0.0f)
                    throw new ArgumentException("capsule radius must be positive", nameof(shape));
                if (c.HalfLength <= 0.0f)
                    throw new ArgumentException("zero-length capsule: use Sphere instead", nameof(shape));
                break;
            case ConvexHullShape3 h:
                if (h.Vertices == null || h.Vertices.Any(v => !float.IsFinite(v.X) || !float.IsFinite(v.Y) || !float.IsFinite(v.Z)))
                    throw new ArgumentException("hull vertices must be finite", nameof(shape));
                if (DistinctCount(h.Vertices) < 4)
                    throw new ArgumentException("convex hull needs 4+ distinct vertices", nameof(shape));
                break;
            default:
                throw new ArgumentException($"unknown 3D shape type {shape.GetType().Name}", nameof(shape));
        }
    }

    /// Density rule: always finite; positive unless the body is static
    /// (statics ignore mass, so any finite density is accepted for them).
    public static float ValidateDensity(bool isStatic, float density, string paramName = "density")
    {
        if (!float.IsFinite(density))
            throw new ArgumentException("density must be finite", paramName);
        if (!isStatic && density <= 0.0f)
            throw new ArgumentException("dynamic body needs positive density", paramName);
        return density;
    }

    public static void ValidatePosition(Vector2 p, string paramName = "position")
    {
        if (!float.IsFinite(p.X) || !float.IsFinite(p.Y))
            throw new ArgumentException("position must be finite", paramName);
    }

    public static void ValidatePosition(Vector3 p, string paramName = "position")
    {
        if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || !float.IsFinite(p.Z))
            throw new ArgumentException("position must be finite", paramName);
    }

    private static int DistinctCount(List<Vector2> vs)
    {
        int n = 0;
        for (int i = 0; i < vs.Count; i++)
        {
            bool dup = false;
            for (int j = 0; j < i; j++)
                if (vs[j] == vs[i]) { dup = true; break; }
            if (!dup) n++;
        }
        return n;
    }

    private static int DistinctCount(List<Vector3> vs)
    {
        int n = 0;
        for (int i = 0; i < vs.Count; i++)
        {
            bool dup = false;
            for (int j = 0; j < i; j++)
                if (vs[j] == vs[i]) { dup = true; break; }
            if (!dup) n++;
        }
        return n;
    }

    private static float PolygonArea(List<Vector2> vertices)
    {
        float area = 0.0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            int j = (i + 1) % vertices.Count;
            area += vertices[i].X * vertices[j].Y - vertices[j].X * vertices[i].Y;
        }
        return area * 0.5f;
    }
}
