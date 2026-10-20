using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Collision;

/// <summary>Which features produced the 3D SAT minimum axis.</summary>
public enum ObbContactKind { FaceA, FaceB, EdgeEdge }

/// <summary>
/// 3D separating-axis test for oriented boxes. Mirrors sat3d.rs.
/// Face axes give exact contact normals (GJK/EPA normals wobble ~1
/// degree on face contact and feed a torque feedback loop).
/// </summary>
public static class Sat3D
{
    private static Vector3 Axis(int k) => k == 0 ? Vector3.UnitX : k == 1 ? Vector3.UnitY : Vector3.UnitZ;

    public static (Vector3 Normal, float Penetration, ObbContactKind Kind)? ObbObb(
        Vector3 centerA, Vector3 halfA, Quaternion rotA,
        Vector3 centerB, Vector3 halfB, Quaternion rotB)
    {
        Vector3 Ax(int k) => Vector3.Transform(Axis(k), rotA);
        Vector3 Bx(int k) => Vector3.Transform(Axis(k), rotB);
        static float Comp(Vector3 v, int k) => k == 0 ? v.X : k == 1 ? v.Y : v.Z;
        var d = centerB - centerA;

        float bestOverlap = float.MaxValue;
        var bestAxis = Vector3.UnitX;
        var bestKind = ObbContactKind.FaceA;

        for (int i = 0; i < 3; i++)
        {
            var axis = Ax(i);
            float ra = Comp(halfA, i);
            float rb = Comp(halfB, 0) * MathF.Abs(Vector3.Dot(axis, Bx(0)))
                     + Comp(halfB, 1) * MathF.Abs(Vector3.Dot(axis, Bx(1)))
                     + Comp(halfB, 2) * MathF.Abs(Vector3.Dot(axis, Bx(2)));
            if (MathF.Abs(Vector3.Dot(d, axis)) > ra + rb) return null;
            float overlap = ra + rb - MathF.Abs(Vector3.Dot(d, axis));
            if (overlap < bestOverlap) { bestOverlap = overlap; bestAxis = axis; bestKind = ObbContactKind.FaceA; }
        }
        for (int i = 0; i < 3; i++)
        {
            var axis = Bx(i);
            float rb = Comp(halfB, i);
            float ra = Comp(halfA, 0) * MathF.Abs(Vector3.Dot(axis, Ax(0)))
                     + Comp(halfA, 1) * MathF.Abs(Vector3.Dot(axis, Ax(1)))
                     + Comp(halfA, 2) * MathF.Abs(Vector3.Dot(axis, Ax(2)));
            if (MathF.Abs(Vector3.Dot(d, axis)) > ra + rb) return null;
            float overlap = ra + rb - MathF.Abs(Vector3.Dot(d, axis));
            if (overlap < bestOverlap) { bestOverlap = overlap; bestAxis = axis; bestKind = ObbContactKind.FaceB; }
        }
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                var axis = Vector3.Cross(Ax(i), Bx(j));
                if (axis.LengthSquared() < 1e-12f) continue;
                axis = Vector3.Normalize(axis);
                float ra = Comp(halfA, 0) * MathF.Abs(Vector3.Dot(axis, Ax(0)))
                         + Comp(halfA, 1) * MathF.Abs(Vector3.Dot(axis, Ax(1)))
                         + Comp(halfA, 2) * MathF.Abs(Vector3.Dot(axis, Ax(2)));
                float rb = Comp(halfB, 0) * MathF.Abs(Vector3.Dot(axis, Bx(0)))
                         + Comp(halfB, 1) * MathF.Abs(Vector3.Dot(axis, Bx(1)))
                         + Comp(halfB, 2) * MathF.Abs(Vector3.Dot(axis, Bx(2)));
                if (MathF.Abs(Vector3.Dot(d, axis)) > ra + rb) return null;
                float overlap = ra + rb - MathF.Abs(Vector3.Dot(d, axis));
                if (overlap < bestOverlap - 1e-6f) { bestOverlap = overlap; bestAxis = axis; bestKind = ObbContactKind.EdgeEdge; }
            }
        }

        var normal = Vector3.Dot(d, bestAxis) < 0 ? -bestAxis : bestAxis;
        return (normal, bestOverlap, bestKind);
    }
}

/// <summary>
/// Separating Axis Theorem for convex polygons in 2D
/// </summary>
public static class SAT
{
    /// <summary>
    /// SAT for polygon vs polygon collision
    /// Returns normal and penetration, or null if separated
    /// </summary>
    public static (Vector2 Normal, float Penetration)? PolygonPolygon(List<Vector2> a, List<Vector2> b)
    {
        if (a.Count < 3 || b.Count < 3)
        {
            return null;
        }

        float bestOverlap = float.MaxValue;
        Vector2 bestAxis = Vector2.Zero;

        // Check axes from polygon A
        for (int i = 0; i < a.Count; i++)
        {
            int j = (i + 1) % a.Count;
            Vector2 edge = a[j] - a[i];
            if (edge.LengthSquared() < 1e-12f)
            {
                continue;
            }
            Vector2 normal = Vector2.Normalize(new Vector2(-edge.Y, edge.X));

            // Project both polygons onto this normal
            var (minA, maxA) = ProjectPolygon(a, normal);
            var (minB, maxB) = ProjectPolygon(b, normal);

            // Check for separation
            if (maxA < minB || maxB < minA)
            {
                return null; // Separated
            }

            // Calculate overlap (positive when overlapping)
            float overlap = MathF.Min(maxA, maxB) - MathF.Max(minA, minB);

            if (overlap < bestOverlap)
            {
                bestOverlap = overlap;
                bestAxis = normal;
            }
        }

        // Check axes from polygon B
        for (int i = 0; i < b.Count; i++)
        {
            int j = (i + 1) % b.Count;
            Vector2 edge = b[j] - b[i];
            if (edge.LengthSquared() < 1e-12f)
            {
                continue;
            }
            Vector2 normal = Vector2.Normalize(new Vector2(-edge.Y, edge.X));

            // Project both polygons onto this normal
            var (minA, maxA) = ProjectPolygon(a, normal);
            var (minB, maxB) = ProjectPolygon(b, normal);

            // Check for separation
            if (maxA < minB || maxB < minA)
            {
                return null; // Separated
            }

            // Calculate overlap (positive when overlapping)
            float overlap = MathF.Min(maxA, maxB) - MathF.Max(minA, minB);

            if (overlap < bestOverlap)
            {
                bestOverlap = overlap;
                bestAxis = normal;
            }
        }

        if (bestAxis == Vector2.Zero)
        {
            return null; // All edges degenerate.
        }
        return (bestAxis, bestOverlap);
    }

    /// <summary>
    /// SAT for circle vs polygon collision. Projects both shapes onto each
    /// polygon edge normal plus the closest-vertex axis (parity with the
    /// Rust implementation).
    /// Normal orientation is undefined — orient A-&gt;B with <see cref="OrientNormal"/>.
    /// </summary>
    public static (Vector2 Normal, float Penetration)? CirclePolygon(Vector2 center, float radius, List<Vector2> polygon)
    {
        if (polygon.Count < 3 || radius <= 0.0f)
        {
            return null;
        }

        float bestOverlap = float.MaxValue;
        Vector2 bestAxis = Vector2.Zero;

        for (int i = 0; i < polygon.Count; i++)
        {
            int j = (i + 1) % polygon.Count;
            Vector2 edge = polygon[j] - polygon[i];
            if (edge.LengthSquared() < 1e-12f)
            {
                continue;
            }
            Vector2 normal = Vector2.Normalize(new Vector2(-edge.Y, edge.X));

            // Project polygon and circle onto this normal.
            var (minPoly, maxPoly) = ProjectPolygon(polygon, normal);
            float centerProj = Vector2.Dot(normal, center);
            float minCircle = centerProj - radius;
            float maxCircle = centerProj + radius;

            // Check for separation
            if (maxPoly < minCircle || maxCircle < minPoly)
            {
                return null; // Separated
            }

            // Calculate overlap (positive when overlapping)
            float overlap = MathF.Min(maxPoly, maxCircle) - MathF.Max(minPoly, minCircle);

            if (overlap < bestOverlap)
            {
                bestOverlap = overlap;
                bestAxis = normal;
            }
        }

        // Closest-vertex axis: a circle sitting past a polygon corner can
        // overlap every face-normal projection while being separated from
        // the polygon (the face axes alone cannot see the corner gap). The
        // axis from the closest vertex to the center separates exactly in
        // that case. This test can only turn a contact into a separation,
        // never the reverse: overlapping shapes overlap on every axis.
        {
            Vector2 closest = polygon[0];
            float bestDistSq = Vector2.Dot(center - closest, center - closest);
            for (int i = 1; i < polygon.Count; i++)
            {
                Vector2 d = center - polygon[i];
                float distSq = Vector2.Dot(d, d);
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    closest = polygon[i];
                }
            }
            Vector2 toCenter = center - closest;
            float lenSq = Vector2.Dot(toCenter, toCenter);
            if (lenSq >= 1e-12f)
            {
                float invLen = 1.0f / MathF.Sqrt(lenSq);
                Vector2 axis = new Vector2(toCenter.X * invLen, toCenter.Y * invLen);
                var (minPoly, maxPoly) = ProjectPolygon(polygon, axis);
                float centerProj = Vector2.Dot(axis, center);
                if (maxPoly < centerProj - radius || centerProj + radius < minPoly)
                {
                    return null; // Separated past a corner.
                }
            }
        }

        // Check if circle center is inside polygon: push out along the CLOSEST
        // edge (least penetration). `dist` is negative for interior points
        // (measured against the inward edge normal), so the closest face is
        // the MAXIMUM dist. Tracking the minimum would eject through the
        // farthest face instead.
        if (PointInPolygon(center, polygon))
        {
            // Find closest edge and push out
            float bestDist = float.MinValue;
            Vector2 closestNormal = Vector2.Zero;

            for (int i = 0; i < polygon.Count; i++)
            {
                int j = (i + 1) % polygon.Count;
                Vector2 edge = polygon[j] - polygon[i];
                if (edge.LengthSquared() < 1e-12f)
                {
                    continue;
                }
                Vector2 normal = Vector2.Normalize(new Vector2(-edge.Y, edge.X));
                float dist = Vector2.Dot(normal, polygon[i] - center);

                if (dist > bestDist)
                {
                    bestDist = dist;
                    closestNormal = normal;
                }
            }

            if (closestNormal == Vector2.Zero)
            {
                return null; // All edges degenerate.
            }
            float penetration = radius - bestDist;
            return (-closestNormal, penetration);
        }

        if (bestAxis == Vector2.Zero)
        {
            return null; // All edges degenerate.
        }
        return (bestAxis, bestOverlap);
    }

    /// <summary>
    /// Project a polygon onto an axis
    /// </summary>
    private static (float Min, float Max) ProjectPolygon(List<Vector2> polygon, Vector2 axis)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (var vertex in polygon)
        {
            float projection = Vector2.Dot(axis, vertex);
            min = MathF.Min(min, projection);
            max = MathF.Max(max, projection);
        }

        return (min, max);
    }

    /// <summary>
    /// Check if a point is inside a polygon
    /// </summary>
    public static bool PointInPolygon(Vector2 point, List<Vector2> polygon)
    {
        bool inside = false;
        int j = polygon.Count - 1;

        for (int i = 0; i < polygon.Count; i++)
        {
            if ((polygon[i].Y > point.Y) != (polygon[j].Y > point.Y))
            {
                float slope = (point.Y - polygon[i].Y) / (polygon[j].Y - polygon[i].Y);
                float x = polygon[i].X + slope * (polygon[j].X - polygon[i].X);

                if (point.X < x)
                {
                    inside = !inside;
                }
            }
            j = i;
        }

        return inside;
    }

    /// <summary>
    /// Orient a SAT normal so it points from centroid A to centroid B.
    /// </summary>
    public static Vector2 OrientNormal(Vector2 normal, Vector2 centroidA, Vector2 centroidB)
    {
        return Vector2.Dot(centroidB - centroidA, normal) < 0.0f ? -normal : normal;
    }

    /// <summary>
    /// Centroid (average) of a vertex set.
    /// </summary>
    public static Vector2 PolygonCentroid(List<Vector2> verts)
    {
        if (verts.Count == 0)
        {
            return Vector2.Zero;
        }
        Vector2 sum = Vector2.Zero;
        foreach (var v in verts)
        {
            sum += v;
        }
        return sum / verts.Count;
    }

    /// <summary>
    /// SAT for AABB vs AABB (axis-aligned fast path). Normal points A-&gt;B.
    /// </summary>
    public static (Vector2 Normal, float Penetration)? AabbAabb(AABB2 a, AABB2 b)
    {
        float overlapX = MathF.Min(a.Max.X, b.Max.X) - MathF.Max(a.Min.X, b.Min.X);
        if (overlapX < 0.0f)
        {
            return null;
        }
        float overlapY = MathF.Min(a.Max.Y, b.Max.Y) - MathF.Max(a.Min.Y, b.Min.Y);
        if (overlapY < 0.0f)
        {
            return null;
        }

        if (overlapX < overlapY)
        {
            float dir = b.Center.X >= a.Center.X ? 1.0f : -1.0f;
            return (new Vector2(dir, 0.0f), overlapX);
        }
        float dirY = b.Center.Y >= a.Center.Y ? 1.0f : -1.0f;
        return (new Vector2(0.0f, dirY), overlapY);
    }

    /// <summary>
    /// Generate contact points for polygon-polygon collision.
    /// Reference/incident face clipping (Sutherland–Hodgman against the side
    /// planes of the reference edge, then filtering to points behind the
    /// reference face). Tries A-as-reference first, falls back to B-as-reference.
    /// Returns up to 2 contact points.
    /// </summary>
    public static List<Vector2> PolygonPolygonContactPoints(List<Vector2> a, List<Vector2> b, Vector2 normal)
    {
        var contacts = new List<Vector2>();
        var tmp = new List<Vector2>();
        PolygonPolygonContactPointsInto(a, b, normal, contacts, tmp);
        return contacts;
    }

    /// <summary>
    /// Allocation-free variant: appends up to 2 contact points to
    /// <paramref name="output"/> (untouched when there is no contact),
    /// reusing <paramref name="tmp"/> scratch.
    /// </summary>
    public static void PolygonPolygonContactPointsInto(
        List<Vector2> a, List<Vector2> b, Vector2 normal,
        List<Vector2> output, List<Vector2> tmp)
    {
        if (a.Count < 3 || b.Count < 3 || normal.LengthSquared() < 1e-12f)
        {
            return;
        }

        // Ensure the normal points from A to B.
        Vector2 n = OrientNormal(normal, PolygonCentroid(a), PolygonCentroid(b));

        int before = output.Count;
        ClipContactsInto(a, b, n, output, tmp);
        if (output.Count == before)
        {
            ClipContactsInto(b, a, -n, output, tmp);
        }
        if (output.Count > before + 2)
        {
            output.RemoveRange(before + 2, output.Count - (before + 2));
        }
    }

    /// <summary>
    /// Clip the incident edge of <paramref name="incident"/> against the
    /// reference edge of <paramref name="reference"/>, appending to
    /// <paramref name="output"/> with <paramref name="tmp"/> as scratch.
    /// </summary>
    private static void ClipContactsInto(
        List<Vector2> reference, List<Vector2> incident, Vector2 normal,
        List<Vector2> output, List<Vector2> tmp)
    {
        var (refIdx, _) = FindBestEdge(reference, normal);
        var (incIdx, _) = FindBestEdge(incident, -normal);

        Vector2 r1 = reference[refIdx];
        Vector2 r2 = reference[(refIdx + 1) % reference.Count];
        Vector2 s1 = incident[incIdx];
        Vector2 s2 = incident[(incIdx + 1) % incident.Count];

        Vector2 edge = r2 - r1;
        if (edge.LengthSquared() < 1e-12f)
        {
            return;
        }

        // Reference face normal, oriented to agree with the collision normal
        // (handles both CW and CCW windings).
        Vector2 faceNormal = Vector2.Normalize(new Vector2(edge.Y, -edge.X));
        if (Vector2.Dot(faceNormal, normal) < 0.0f)
        {
            faceNormal = -faceNormal;
        }

        // Clip incident segment to the side slab of the reference edge,
        // ping-ponging between tmp and output's tail region.
        Vector2 tangent = Vector2.Normalize(r2 - r1);
        int before = output.Count;
        tmp.Clear();
        tmp.Add(s1);
        tmp.Add(s2);
        ClipToPlaneInto(tmp, -tangent, Vector2.Dot(-tangent, r1), output);
        if (output.Count == before)
        {
            return;
        }
        tmp.Clear();
        for (int i = before; i < output.Count; i++) tmp.Add(output[i]);
        output.RemoveRange(before, output.Count - before);
        ClipToPlaneInto(tmp, tangent, Vector2.Dot(tangent, r2), output);
        if (output.Count == before)
        {
            return;
        }

        // Keep points behind (or on) the reference face (in-place filter).
        float refOffset = Vector2.Dot(faceNormal, r1);
        int kept = 0;
        for (int i = before; i < output.Count; i++)
        {
            if (Vector2.Dot(faceNormal, output[i]) - refOffset <= 0.001f)
            {
                output[before + kept] = output[i];
                kept++;
            }
        }
        output.RemoveRange(before + kept, output.Count - (before + kept));
    }

    /// <summary>
    /// Sutherland–Hodgman clip of a point list against half-plane dot(n,p) &lt;= offset,
    /// appending to <paramref name="output"/>.
    /// </summary>
    private static void ClipToPlaneInto(IReadOnlyList<Vector2> points, Vector2 n, float offset, List<Vector2> output)
    {
        if (points.Count == 0)
        {
            return;
        }
        Vector2 prev = points[points.Count - 1];
        bool prevInside = Vector2.Dot(prev, n) <= offset;

        foreach (var cur in points)
        {
            bool curInside = Vector2.Dot(cur, n) <= offset;
            if (curInside)
            {
                if (!prevInside)
                {
                    output.Add(IntersectEdge(prev, cur, n, offset));
                }
                output.Add(cur);
            }
            else if (prevInside)
            {
                output.Add(IntersectEdge(prev, cur, n, offset));
            }
            prev = cur;
            prevInside = curInside;
        }
    }

    /// <summary>
    /// Intersection of segment prev-&gt;cur with plane dot(n,p) = offset.
    /// </summary>
    private static Vector2 IntersectEdge(Vector2 prev, Vector2 cur, Vector2 n, float offset)
    {
        Vector2 d = cur - prev;
        float denom = Vector2.Dot(n, d);
        if (MathF.Abs(denom) < 1e-9f)
        {
            return cur;
        }
        float t = (offset - Vector2.Dot(n, prev)) / denom;
        t = MathF.Max(0.0f, MathF.Min(1.0f, t));
        return prev + d * t;
    }

    /// <summary>
    /// Find the edge whose outward normal best aligns with the given direction.
    /// Polygons are expected wound CCW (outward = (edge.Y, -edge.X)).
    /// </summary>
    private static (int EdgeIndex, Vector2 Vertex) FindBestEdge(List<Vector2> polygon, Vector2 dir)
    {
        float bestDot = float.MinValue;
        int edgeIdx = 0;

        for (int i = 0; i < polygon.Count; i++)
        {
            int j = (i + 1) % polygon.Count;
            Vector2 edge = polygon[j] - polygon[i];
            if (edge.LengthSquared() < 1e-12f)
            {
                continue;
            }
            Vector2 outward = Vector2.Normalize(new Vector2(edge.Y, -edge.X));
            float dot = Vector2.Dot(outward, dir);

            if (dot > bestDot)
            {
                bestDot = dot;
                edgeIdx = i;
            }
        }

        return (edgeIdx, polygon[edgeIdx]);
    }
}
