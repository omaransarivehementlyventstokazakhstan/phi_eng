using System.Numerics;

namespace PhiEng.Collision;

/// <summary>
/// Ray casts against the shapes the engine knows.
///
/// A ray cast answers "what does this ray hit first?" -- the query a
/// game needs to pick a body under the mouse, to fire a hitscan, or to
/// sweep a character controller.
///
/// Every method here is analytic and exact for the shape it takes, and
/// every one of them mirrors <c>phieng_collision::query</c>, so the two
/// agree bit for bit.
/// </summary>
public static class Query
{
    /// <summary>A direction this small is treated as parallel to an axis.</summary>
    private const float ParallelEps = 1e-9f;

    /// <summary>The 2D cross product: the z component of the 3D cross.</summary>
    private static float Cross2(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;

    /// <summary>
    /// Ray against a circle. A ray that starts inside the circle does
    /// not hit it: the hit is for a ray that enters.
    /// </summary>
    public static bool RayCircle(
        Vector2 origin, Vector2 direction, Vector2 center, float radius, out float t, out Vector2 point, out Vector2 normal)
    {
        t = 0.0f;
        point = Vector2.Zero;
        normal = Vector2.Zero;
        var m = origin - center;
        var a = Vector2.Dot(direction, direction);
        if (a <= 0.0f)
        {
            return false;
        }
        var b = 2.0f * Vector2.Dot(m, direction);
        var c = Vector2.Dot(m, m) - radius * radius;
        var disc = b * b - 4.0f * a * c;
        if (disc < 0.0f)
        {
            return false;
        }
        var root = MathF.Sqrt(disc);
        t = (-b - root) / (2.0f * a);
        if (t < 0.0f)
        {
            t = (-b + root) / (2.0f * a);
        }
        if (t < 0.0f)
        {
            return false;
        }
        point = origin + direction * t;
        normal = (point - center) / radius;
        return true;
    }

    /// <summary>
    /// Ray against an oriented box. <paramref name="rotation"/> is the
    /// box's orientation and <paramref name="halfExtents"/> its size
    /// along the box's own axes.
    /// </summary>
    public static bool RayObb(
        Vector2 origin, Vector2 direction, Vector2 center, Matrix3x2 rotation,
        Vector2 halfExtents, out float t, out Vector2 point, out Vector2 normal)
    {
        t = 0.0f;
        point = Vector2.Zero;
        normal = Vector2.Zero;
        // Into the box's own frame. The Matrix3x2 constructor takes
        // (m11, m12, m21, m22, m31, m32), so the transpose swaps the
        // two off-diagonal entries.
        var inv = new Matrix3x2(
            rotation.M11, rotation.M21, rotation.M12,
            rotation.M22, rotation.M31, rotation.M32);
        var o = Vector2.Transform(origin - center, inv);
        var d = Vector2.Transform(direction, inv);

        var tEnter = float.MinValue;
        var tExit = float.MaxValue;
        var enterAxis = 0;
        var enterSign = 1.0f;

        for (int axis = 0; axis < 2; axis++)
        {
            var h = axis == 0 ? halfExtents.X : halfExtents.Y;
            var oa = axis == 0 ? o.X : o.Y;
            var da = axis == 0 ? d.X : d.Y;
            if (MathF.Abs(da) < ParallelEps)
            {
                // Parallel to this slab: a miss unless the ray is inside it.
                if (MathF.Abs(oa) > h)
                {
                    return false;
                }
                continue;
            }
            var invD = 1.0f / da;
            var t1 = (-h - oa) * invD;
            var t2 = (h - oa) * invD;
            float near, far;
            if (t1 <= t2)
            {
                near = t1;
                far = t2;
            }
            else
            {
                near = t2;
                far = t1;
            }
            if (near > tEnter)
            {
                tEnter = near;
                enterAxis = axis;
                enterSign = da > 0.0f ? -1.0f : 1.0f;
            }
            if (far < tExit)
            {
                tExit = far;
            }
            if (tEnter > tExit)
            {
                return false;
            }
        }

        if (tEnter < 0.0f)
        {
            // The ray starts inside the box: it does not hit it.
            return false;
        }
        t = tEnter;
        point = origin + direction * t;
        var localNormal = Vector2.Zero;
        if (enterAxis == 0)
        {
            localNormal.X = enterSign;
        }
        else
        {
            localNormal.Y = enterSign;
        }
        normal = Vector2.Transform(localNormal, rotation);
        return true;
    }

    /// <summary>
    /// Ray against a convex polygon. The vertices may be in either
    /// winding order: the normal is always turned to face the ray, so
    /// the answer does not depend on it.
    /// </summary>
    public static bool RayPolygon(
        Vector2 origin, Vector2 direction, Vector2[] vertices,
        out float t, out Vector2 point, out Vector2 normal)
    {
        t = 0.0f;
        point = Vector2.Zero;
        normal = Vector2.Zero;
        int n = vertices.Length;
        if (n < 3)
        {
            return false;
        }
        if (PointInConvexPolygon(origin, vertices))
        {
            // The ray starts inside: it does not hit the polygon.
            return false;
        }

        var bestT = float.MaxValue;
        var bestNormal = Vector2.Zero;
        for (int i = 0; i < n; i++)
        {
            var a = vertices[i];
            var b = vertices[(i + 1) % n];
            var edge = b - a;
            var denom = Cross2(direction, edge);
            if (MathF.Abs(denom) < ParallelEps)
            {
                continue;
            }
            var offset = a - origin;
            var hit = Cross2(offset, edge) / denom;
            var s = Cross2(offset, direction) / denom;
            if (s < 0.0f || s > 1.0f || hit < 0.0f)
            {
                continue;
            }
            if (hit < bestT)
            {
                bestT = hit;
                // The edge normal, facing the ray.
                var edgeNormal = new Vector2(edge.Y, -edge.X);
                var len = edgeNormal.Length();
                if (len > 0.0f)
                {
                    edgeNormal /= len;
                }
                if (Vector2.Dot(edgeNormal, direction) > 0.0f)
                {
                    edgeNormal = -edgeNormal;
                }
                bestNormal = edgeNormal;
            }
        }
        if (bestT == float.MaxValue)
        {
            return false;
        }
        t = bestT;
        point = origin + direction * t;
        normal = bestNormal;
        return true;
    }

    /// <summary>
    /// Whether a point is inside a convex polygon. The answer does not
    /// depend on the winding order.
    /// </summary>
    public static bool PointInConvexPolygon(Vector2 point, Vector2[] vertices)
    {
        int n = vertices.Length;
        if (n < 3)
        {
            return false;
        }
        bool positive = false;
        bool negative = false;
        for (int i = 0; i < n; i++)
        {
            var a = vertices[i];
            var b = vertices[(i + 1) % n];
            var cross = Cross2(b - a, point - a);
            if (cross > 0.0f)
            {
                positive = true;
            }
            else if (cross < 0.0f)
            {
                negative = true;
            }
            if (positive && negative)
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Ray against a sphere. A ray that starts inside the sphere does not
    /// hit it.
    /// </summary>
    public static bool RaySphere(
        Vector3 origin, Vector3 direction, Vector3 center, float radius,
        out float t, out Vector3 point, out Vector3 normal)
    {
        t = 0.0f;
        point = Vector3.Zero;
        normal = Vector3.Zero;
        var m = origin - center;
        var a = Vector3.Dot(direction, direction);
        if (a <= 0.0f)
        {
            return false;
        }
        var b = 2.0f * Vector3.Dot(m, direction);
        var c = Vector3.Dot(m, m) - radius * radius;
        var disc = b * b - 4.0f * a * c;
        if (disc < 0.0f)
        {
            return false;
        }
        var root = MathF.Sqrt(disc);
        t = (-b - root) / (2.0f * a);
        if (t < 0.0f)
        {
            t = (-b + root) / (2.0f * a);
        }
        if (t < 0.0f)
        {
            return false;
        }
        point = origin + direction * t;
        normal = (point - center) / radius;
        return true;
    }

    /// <summary>
    /// Ray against an oriented box. <paramref name="rotation"/> is the
    /// box's orientation and <paramref name="halfExtents"/> its size
    /// along the box's own axes.
    /// </summary>
    public static bool RayObb(
        Vector3 origin, Vector3 direction, Vector3 center, Quaternion rotation,
        Vector3 halfExtents, out float t, out Vector3 point, out Vector3 normal)
    {
        t = 0.0f;
        point = Vector3.Zero;
        normal = Vector3.Zero;
        // The box's own axes, the same ones the OBB-OBB test uses.
        var ax = Vector3.Transform(Vector3.UnitX, rotation);
        var ay = Vector3.Transform(Vector3.UnitY, rotation);
        var az = Vector3.Transform(Vector3.UnitZ, rotation);
        // The ray in the box's frame.
        var offset = origin - center;
        var o = new Vector3(Vector3.Dot(offset, ax), Vector3.Dot(offset, ay), Vector3.Dot(offset, az));
        var d = new Vector3(Vector3.Dot(direction, ax), Vector3.Dot(direction, ay), Vector3.Dot(direction, az));

        var tEnter = float.MinValue;
        var tExit = float.MaxValue;
        var enterAxis = 0;
        var enterSign = 1.0f;

        for (int axis = 0; axis < 3; axis++)
        {
            var h = axis == 0 ? halfExtents.X : (axis == 1 ? halfExtents.Y : halfExtents.Z);
            var oa = axis == 0 ? o.X : (axis == 1 ? o.Y : o.Z);
            var da = axis == 0 ? d.X : (axis == 1 ? d.Y : d.Z);
            if (MathF.Abs(da) < ParallelEps)
            {
                if (MathF.Abs(oa) > h)
                {
                    return false;
                }
                continue;
            }
            var invD = 1.0f / da;
            var t1 = (-h - oa) * invD;
            var t2 = (h - oa) * invD;
            float near, far;
            if (t1 <= t2)
            {
                near = t1;
                far = t2;
            }
            else
            {
                near = t2;
                far = t1;
            }
            if (near > tEnter)
            {
                tEnter = near;
                enterAxis = axis;
                enterSign = da > 0.0f ? -1.0f : 1.0f;
            }
            if (far < tExit)
            {
                tExit = far;
            }
            if (tEnter > tExit)
            {
                return false;
            }
        }

        if (tEnter < 0.0f)
        {
            return false;
        }
        t = tEnter;
        point = origin + direction * t;
        var localNormal = Vector3.Zero;
        if (enterAxis == 0)
        {
            localNormal.X = enterSign;
        }
        else if (enterAxis == 1)
        {
            localNormal.Y = enterSign;
        }
        else
        {
            localNormal.Z = enterSign;
        }
        // Back into world space, through the same axes.
        normal = ax * localNormal.X + ay * localNormal.Y + az * localNormal.Z;
        return true;
    }

    /// <summary>
    /// Ray against a capsule: the segment <paramref name="a"/> to
    /// <paramref name="b"/> with the given radius.
    /// </summary>
    public static bool RayCapsule(
        Vector3 origin, Vector3 direction, Vector3 a, Vector3 b, float radius,
        out float t, out Vector3 point, out Vector3 normal)
    {
        t = 0.0f;
        point = Vector3.Zero;
        normal = Vector3.Zero;
        var axis = b - a;
        var axisLenSq = axis.LengthSquared();
        if (axisLenSq < ParallelEps)
        {
            // A sphere.
            return RaySphere(origin, direction, a, radius, out t, out point, out normal);
        }

        // The infinite cylinder around the axis, then the two caps.
        var o = origin - a;
        var d = direction;
        var axisDotD = Vector3.Dot(axis, d);
        var axisDotO = Vector3.Dot(axis, o);

        var aQuad = Vector3.Dot(d, d) - axisDotD * axisDotD / axisLenSq;
        var bQuad = Vector3.Dot(d, o) - axisDotD * axisDotO / axisLenSq;
        var cQuad = Vector3.Dot(o, o) - axisDotO * axisDotO / axisLenSq - radius * radius;

        bool hitAny = false;
        float bestT = float.MaxValue;
        Vector3 bestPoint = Vector3.Zero;
        Vector3 bestNormal = Vector3.Zero;

        if (MathF.Abs(aQuad) > ParallelEps)
        {
            var disc = bQuad * bQuad - aQuad * cQuad;
            if (disc >= 0.0f)
            {
                var root = MathF.Sqrt(disc);
                foreach (var candidate in new[] { (-bQuad - root) / aQuad, (-bQuad + root) / aQuad })
                {
                    if (candidate < 0.0f)
                    {
                        continue;
                    }
                    var hitPoint = origin + d * candidate;
                    // The hit must be on the segment, not past its ends.
                    var s = Vector3.Dot(hitPoint - a, axis) / axisLenSq;
                    if (s < 0.0f || s > 1.0f)
                    {
                        continue;
                    }
                    var hitNormal = (hitPoint - (a + axis * s)) / radius;
                    if (candidate < bestT)
                    {
                        bestT = candidate;
                        bestPoint = hitPoint;
                        bestNormal = hitNormal;
                        hitAny = true;
                    }
                }
            }
        }

        // The caps.
        foreach (var cap in new[] { a, b })
        {
            if (RaySphere(origin, direction, cap, radius, out var capT, out var capPoint, out var capNormal)
                && capT < bestT)
            {
                bestT = capT;
                bestPoint = capPoint;
                bestNormal = capNormal;
                hitAny = true;
            }
        }

        if (!hitAny)
        {
            return false;
        }
        t = bestT;
        point = bestPoint;
        normal = bestNormal;
        return true;
    }
}
