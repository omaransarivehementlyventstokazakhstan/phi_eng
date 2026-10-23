using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>The nearest surface point of a 2D shape. Mirrors Rust <c>ShapeHit2</c>.</summary>
public struct ShapeHit2
{
    /// <summary>Whether the query point is inside the shape.</summary>
    public bool Inside;
    /// <summary>The closest point in the shape (the query point itself when inside).</summary>
    public Vector2 Point;
    /// <summary>The outward escape direction at the surface.</summary>
    public Vector2 Normal;
    /// <summary>The distance to the surface (zero when inside).</summary>
    public float Distance;
}

/// <summary>The nearest surface point of a 3D shape. Mirrors Rust <c>ShapeHit3</c>.</summary>
public struct ShapeHit3
{
    public bool Inside;
    public Vector3 Point;
    public Vector3 Normal;
    public float Distance;
}

/// <summary>The nearest surface point a world closest-point query found. Mirrors Rust <c>WorldClosestHit2</c>.</summary>
public struct WorldClosestHit2
{
    public int Body;
    public bool Inside;
    public Vector2 Point;
    public Vector2 Normal;
    public float Distance;
}

/// <summary>The 3D form of <see cref="WorldClosestHit2"/>.</summary>
public struct WorldClosestHit3
{
    public int Body;
    public bool Inside;
    public Vector3 Point;
    public Vector3 Normal;
    public float Distance;
}

/// <summary>
/// Analytic point queries against the 2D shapes.
/// </summary>
/// <remarks>
/// Mirrors the Rust <c>phieng_dynamics::shape_query</c> module
/// operation for operation, so the two engines agree bit-for-bit.
/// <para>
/// <c>Point</c> is the closest point <em>in</em> the shape (the query
/// point itself when inside); <c>Normal</c> is the outward escape
/// direction (the least-penetration axis inside a box); <c>Distance</c>
/// is zero when inside. Winding is derived from the polygon's own
/// signed area, so both windings behave the same.
/// </para>
/// </remarks>
public static class ShapeQuery2
{
    private static Vector2 UnitOr(Vector2 v, Vector2 fallback)
    {
        float len = v.Length();
        return len > 1.17549435e-38f && float.IsFinite(len) ? v / len : fallback;
    }

    private static Vector2 SegmentClosest(Vector2 a, Vector2 b, Vector2 p)
    {
        var ab = b - a;
        float lenSq = ab.LengthSquared();
        if (lenSq <= 1.17549435e-38f) return a;
        float t = System.Math.Clamp(Vector2.Dot(p - a, ab) / lenSq, 0.0f, 1.0f);
        return a + ab * t;
    }

    private static bool PolygonIsCcw(IReadOnlyList<Vector2> vertices)
    {
        float area2 = 0.0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            var a = vertices[i];
            var b = vertices[(i + 1) % vertices.Count];
            area2 += a.X * b.Y - b.X * a.Y;
        }
        return area2 > 0.0f;
    }

    /// <summary>Whether <paramref name="point"/> is inside the shape at the given pose.</summary>
    public static bool PointInShape(Shape2 shape, Vector2 position, Matrix3x2 rotation, Vector2 point) =>
        ClosestPoint(shape, position, rotation, point) is { } hit && hit.Inside;

    /// <summary>The nearest point in the shape, its outward normal, and the distance.</summary>
    public static ShapeHit2? ClosestPoint(Shape2 shape, Vector2 position, Matrix3x2 rotation, Vector2 point)
    {
        switch (shape)
        {
            case CircleShape circle:
            {
                float r = MathF.Abs(circle.Radius);
                var offset = point - position;
                float len = offset.Length();
                var normal = UnitOr(offset, Vector2.UnitY);
                bool inside = len <= r;
                return new ShapeHit2
                {
                    Inside = inside,
                    Point = inside ? point : position + normal * r,
                    Normal = normal,
                    Distance = inside ? 0.0f : len - r,
                };
            }
            case BoxShape2 box:
            {
                var he = box.HalfExtents;
                var d = point - position;
                // Local coordinates: the rotation's columns are the
                // body's axes, so the transpose multiplies by them.
                var local = new Vector2(
                    d.X * rotation.M11 + d.Y * rotation.M12,
                    d.X * rotation.M21 + d.Y * rotation.M22);
                var clamped = new Vector2(
                    System.Math.Clamp(local.X, -he.X, he.X),
                    System.Math.Clamp(local.Y, -he.Y, he.Y));
                bool inside = clamped == local;
                var delta = local - clamped;
                float distance = delta.Length();
                Vector2 normalLocal;
                if (inside)
                {
                    float dx = he.X - MathF.Abs(local.X);
                    float dy = he.Y - MathF.Abs(local.Y);
                    normalLocal = dx <= dy
                        ? new Vector2(MathF.Sign(local.X), 0.0f)
                        : new Vector2(0.0f, MathF.Sign(local.Y));
                }
                else
                {
                    normalLocal = UnitOr(delta, Vector2.UnitX);
                }
                var world = clamped.X * new Vector2(rotation.M11, rotation.M12)
                    + clamped.Y * new Vector2(rotation.M21, rotation.M22);
                var n = normalLocal.X * new Vector2(rotation.M11, rotation.M12)
                    + normalLocal.Y * new Vector2(rotation.M21, rotation.M22);
                return new ShapeHit2
                {
                    Inside = inside,
                    Point = inside ? point : position + world,
                    Normal = n,
                    Distance = distance,
                };
            }
            case PolygonShape polygon:
            {
                int n = polygon.Vertices.Count;
                if (n < 3) return null;
                var world = new Vector2[n];
                for (int i = 0; i < n; i++)
                {
                    var v = polygon.Vertices[i];
                    world[i] = position
                        + v.X * new Vector2(rotation.M11, rotation.M12)
                        + v.Y * new Vector2(rotation.M21, rotation.M22);
                }
                bool ccw = PolygonIsCcw(polygon.Vertices);
                bool inside = true;
                bool found = false;
                float bestD = 0.0f;
                Vector2 bestP = Vector2.Zero;
                Vector2 bestN = Vector2.Zero;
                for (int i = 0; i < n; i++)
                {
                    var a = world[i];
                    var b = world[(i + 1) % n];
                    var edge = b - a;
                    float edgeLen = edge.Length();
                    if (edgeLen <= 1.17549435e-38f) continue;
                    var outward = ccw
                        ? new Vector2(edge.Y, -edge.X) / edgeLen
                        : new Vector2(-edge.Y, edge.X) / edgeLen;
                    if (Vector2.Dot(point - a, outward) > 0.0f) inside = false;
                    var onEdge = SegmentClosest(a, b, point);
                    float dd = (point - onEdge).Length();
                    if (!found || dd < bestD)
                    {
                        found = true;
                        bestD = dd;
                        bestP = onEdge;
                        bestN = outward;
                    }
                }
                if (!found) return null;
                return new ShapeHit2
                {
                    Inside = inside,
                    Point = inside ? point : bestP,
                    Normal = bestN,
                    Distance = inside ? 0.0f : bestD,
                };
            }
            default:
                return null;
        }
    }

    private static List<Vector2>? Outline(Shape2 shape, Vector2 position, Matrix3x2 rotation)
    {
        switch (shape)
        {
            case CircleShape:
                return null;
            case BoxShape2 box:
            {
                var he = box.HalfExtents;
                var local = new[]
                {
                    new Vector2(-he.X, -he.Y), new Vector2(he.X, -he.Y),
                    new Vector2(he.X, he.Y), new Vector2(-he.X, he.Y),
                };
                var world = new List<Vector2>(4);
                foreach (var v in local)
                {
                    world.Add(position
                        + v.X * new Vector2(rotation.M11, rotation.M12)
                        + v.Y * new Vector2(rotation.M21, rotation.M22));
                }
                return world;
            }
            case PolygonShape polygon:
            {
                var world = new List<Vector2>(polygon.Vertices.Count);
                foreach (var v in polygon.Vertices)
                {
                    world.Add(position
                        + v.X * new Vector2(rotation.M11, rotation.M12)
                        + v.Y * new Vector2(rotation.M21, rotation.M22));
                }
                return world;
            }
            default:
                return null;
        }
    }

    /// <summary>
    /// Whether two shapes overlap (touch or penetrate).
    /// </summary>
    /// <remarks>
    /// SAT per pair, plus an explicit containment check: SAT reports
    /// separation for a shape wholly inside another (no separating axis
    /// through the interior), so a small circle inside a large box must
    /// be caught by the point test.
    /// </remarks>
    public static bool Overlap(
        Shape2 a, Vector2 aPosition, Matrix3x2 aRotation,
        Shape2 b, Vector2 bPosition, Matrix3x2 bRotation)
    {
        bool hit;
        if (a is CircleShape ca && b is CircleShape cb)
        {
            float sum = MathF.Abs(ca.Radius) + MathF.Abs(cb.Radius);
            hit = (bPosition - aPosition).LengthSquared() <= sum * sum;
        }
        else if (a is CircleShape circleA)
        {
            hit = Collision.SAT.CirclePolygon(
                aPosition, MathF.Abs(circleA.Radius),
                Outline(b, bPosition, bRotation) ?? new List<Vector2>()) != null;
        }
        else if (b is CircleShape circleB)
        {
            hit = Collision.SAT.CirclePolygon(
                bPosition, MathF.Abs(circleB.Radius),
                Outline(a, aPosition, aRotation) ?? new List<Vector2>()) != null;
        }
        else
        {
            hit = Collision.SAT.PolygonPolygon(
                Outline(a, aPosition, aRotation) ?? new List<Vector2>(),
                Outline(b, bPosition, bRotation) ?? new List<Vector2>()) != null;
        }
        return hit
            || PointInShape(a, aPosition, aRotation, bPosition)
            || PointInShape(b, bPosition, bRotation, aPosition);
    }
}

/// <summary>Analytic point and overlap queries against the 3D shapes. Mirrors the Rust module exactly.</summary>
public static class ShapeQuery3
{
    private static Vector3 UnitOr(Vector3 v, Vector3 fallback)
    {
        float len = v.Length();
        return len > 1.17549435e-38f && float.IsFinite(len) ? v / len : fallback;
    }

    private static Vector3 SegmentClosest(Vector3 a, Vector3 b, Vector3 p)
    {
        var ab = b - a;
        float lenSq = ab.LengthSquared();
        if (lenSq <= 1.17549435e-38f) return a;
        float t = System.Math.Clamp(Vector3.Dot(p - a, ab) / lenSq, 0.0f, 1.0f);
        return a + ab * t;
    }

    /// <summary>Whether <paramref name="point"/> is inside the shape at the given pose.</summary>
    public static bool PointInShape(Shape3 shape, Vector3 position, Quaternion rotation, Vector3 point) =>
        ClosestPoint(shape, position, rotation, point) is { } hit && hit.Inside;

    /// <summary>The nearest point in the shape, its outward normal, and the distance.</summary>
    /// <remarks>The convex hull and convex mesh report null, matching the 3D ray cast.</remarks>
    public static ShapeHit3? ClosestPoint(Shape3 shape, Vector3 position, Quaternion rotation, Vector3 point)
    {
        switch (shape)
        {
            case SphereShape3 sphere:
            {
                float r = MathF.Abs(sphere.Radius);
                var offset = point - position;
                float len = offset.Length();
                var normal = UnitOr(offset, Vector3.UnitY);
                bool inside = len <= r;
                return new ShapeHit3
                {
                    Inside = inside,
                    Point = inside ? point : position + normal * r,
                    Normal = normal,
                    Distance = inside ? 0.0f : len - r,
                };
            }
            case BoxShape3 box:
            {
                var he = box.HalfExtents;
                // Rust uses the quaternion inverse (conjugate for a unit
                // quaternion) through the deterministic rotation.
                var local = DeterministicRotation.RotateVector(
                    Quaternion.Conjugate(rotation), point - position);
                var clamped = new Vector3(
                    System.Math.Clamp(local.X, -he.X, he.X),
                    System.Math.Clamp(local.Y, -he.Y, he.Y),
                    System.Math.Clamp(local.Z, -he.Z, he.Z));
                bool inside = clamped == local;
                var delta = local - clamped;
                float distance = delta.Length();
                Vector3 normalLocal;
                if (inside)
                {
                    float dx = he.X - MathF.Abs(local.X);
                    float dy = he.Y - MathF.Abs(local.Y);
                    float dz = he.Z - MathF.Abs(local.Z);
                    if (dx <= dy && dx <= dz) normalLocal = new Vector3(MathF.Sign(local.X), 0.0f, 0.0f);
                    else if (dy <= dz) normalLocal = new Vector3(0.0f, MathF.Sign(local.Y), 0.0f);
                    else normalLocal = new Vector3(0.0f, 0.0f, MathF.Sign(local.Z));
                }
                else
                {
                    normalLocal = UnitOr(delta, Vector3.UnitX);
                }
                return new ShapeHit3
                {
                    Inside = inside,
                    Point = inside
                        ? point
                        : position + DeterministicRotation.RotateVector(rotation, clamped),
                    Normal = DeterministicRotation.RotateVector(rotation, normalLocal),
                    Distance = distance,
                };
            }
            case CapsuleShape3 capsule:
            {
                float r = MathF.Abs(capsule.Radius);
                var axis = DeterministicRotation.RotateVector(rotation, Vector3.UnitY)
                    * MathF.Abs(capsule.HalfLength);
                var a = position + axis;
                var b = position - axis;
                var onAxis = SegmentClosest(a, b, point);
                var offset = point - onAxis;
                float len = offset.Length();
                var normal = UnitOr(offset, DeterministicRotation.RotateVector(rotation, Vector3.UnitX));
                bool inside = len <= r;
                return new ShapeHit3
                {
                    Inside = inside,
                    Point = inside ? point : onAxis + normal * r,
                    Normal = normal,
                    Distance = inside ? 0.0f : len - r,
                };
            }
            default:
                return null;
        }
    }

    /// <summary>The support function of a shape at a pose, matching the narrowphase's mapping.</summary>
    private static Func<Vector3, Vector3> SupportFor(Shape3 shape, Vector3 position, Quaternion rotation)
    {
        switch (shape)
        {
            case SphereShape3 sphere:
            {
                float r = MathF.Abs(sphere.Radius);
                return dir => position + Vector3.Normalize(dir) * r;
            }
            case BoxShape3 box:
            {
                var he = box.HalfExtents;
                return dir =>
                {
                    var localDir = DeterministicRotation.RotateVector(Quaternion.Conjugate(rotation), dir);
                    var localSupport = new Vector3(
                        localDir.X > 0.0f ? he.X : -he.X,
                        localDir.Y > 0.0f ? he.Y : -he.Y,
                        localDir.Z > 0.0f ? he.Z : -he.Z);
                    return position + DeterministicRotation.RotateVector(rotation, localSupport);
                };
            }
            case CapsuleShape3 capsule:
            {
                float r = MathF.Abs(capsule.Radius);
                var axis = DeterministicRotation.RotateVector(rotation, Vector3.UnitY)
                    * MathF.Abs(capsule.HalfLength);
                var pa = position + axis;
                var pb = position - axis;
                return dir => Collision.GjkEpa3D.SupportCapsule(pa, pb, r, dir);
            }
            case ConvexHullShape3 hull:
            {
                var verts = new List<Vector3>(hull.Vertices.Count);
                foreach (var v in hull.Vertices)
                {
                    verts.Add(position + DeterministicRotation.RotateVector(rotation, v));
                }
                return dir => Collision.GjkEpa3D.SupportHull(verts, dir);
            }
            default:
                return _ => position;
        }
    }

    /// <summary>
    /// Whether two shapes overlap (touch or penetrate).
    /// </summary>
    /// <remarks>
    /// GJK decides, with the point test on top so a shape wholly inside
    /// another is never reported as separated.
    /// </remarks>
    public static bool Overlap(
        Shape3 a, Vector3 aPosition, Quaternion aRotation,
        Shape3 b, Vector3 bPosition, Quaternion bRotation)
    {
        var (result, _, _) = Collision.GjkEpa3D.Gjk(
            SupportFor(a, aPosition, aRotation),
            SupportFor(b, bPosition, bRotation));
        return result == Collision.GjkResult3D.Colliding
            || PointInShape(a, aPosition, aRotation, bPosition)
            || PointInShape(b, bPosition, bRotation, aPosition);
    }
}
