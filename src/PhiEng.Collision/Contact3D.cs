using System.Numerics;

namespace PhiEng.Collision;

/// <summary>
/// 3D contact utilities: box-face clipping manifolds, closest-point
/// queries, segment-segment closest points.
/// Mirrors crates/collision/src/contact3d.rs.
/// </summary>
public static class Contact3D
{
    public static List<Vector3> ClipPolygonPlane(List<Vector3> polygon, Vector3 normal, float offset)
    {
        var output = new List<Vector3>(polygon.Count + 1);
        ClipPolygonPlaneInto(polygon, normal, offset, output);
        return output;
    }

    /// <summary>
    /// Allocation-free Sutherland-Hodgman clip, appending to
    /// <paramref name="output"/>.
    /// </summary>
    public static void ClipPolygonPlaneInto(IReadOnlyList<Vector3> polygon, Vector3 normal, float offset, List<Vector3> output)
    {
        if (polygon.Count == 0) return;
        for (int i = 0; i < polygon.Count; i++)
        {
            var cur = polygon[i];
            var next = polygon[(i + 1) % polygon.Count];
            float dc = Vector3.Dot(normal, cur) - offset;
            float dn = Vector3.Dot(normal, next) - offset;
            if (dc <= 0) output.Add(cur);
            if ((dc > 0 && dn < 0) || (dc < 0 && dn > 0))
            {
                float denom = dc - dn;
                float t = MathF.Abs(denom) < 1e-9f ? 0 : dc / denom;
                t = System.Math.Clamp(t, 0, 1);
                output.Add(cur + (next - cur) * t);
            }
        }
    }

    /// <summary>
    /// Box vs box manifold via reference/incident face clipping.
    /// <paramref name="normal"/> points from A to B. Returns up to 4
    /// points (may be empty for edge-on faces; caller should fall back).
    /// Mirrors <c>box_box_contact_points</c> in contact3d.rs.
    /// </summary>
    public static List<Vector3> BoxBoxContactPoints(
        Vector3 centerA, Vector3 heA, Quaternion rotA,
        Vector3 centerB, Vector3 heB, Quaternion rotB,
        Vector3 normal)
    {
        var output = new List<Vector3>(4);
        var tmp = new List<Vector3>(8);
        BoxBoxContactPointsInto(centerA, heA, rotA, centerB, heB, rotB, normal, output, tmp);
        return output;
    }

    /// <summary>
    /// Allocation-free variant: appends up to 4 contact points to
    /// <paramref name="output"/> (untouched when faces are edge-on),
    /// reusing <paramref name="tmp"/> scratch.
    /// </summary>
    public static void BoxBoxContactPointsInto(
        Vector3 centerA, Vector3 heA, Quaternion rotA,
        Vector3 centerB, Vector3 heB, Quaternion rotB,
        Vector3 normal,
        List<Vector3> output, List<Vector3> tmp)
    {
        int refK = 1; float refS = 1, best = float.MinValue;
        for (int k = 0; k < 3; k++)
            for (int si = 0; si < 2; si++)
            {
                float s = si == 0 ? 1.0f : -1.0f;
                float score = Vector3.Dot(Vector3.Transform(Axis(k) * s, rotA), normal);
                if (score > best) { best = score; refK = k; refS = s; }
            }
        int incK = 1; float incS = 1; best = float.MinValue;
        for (int k = 0; k < 3; k++)
            for (int si = 0; si < 2; si++)
            {
                float s = si == 0 ? 1.0f : -1.0f;
                float score = Vector3.Dot(Vector3.Transform(Axis(k) * s, rotB), -normal);
                if (score > best) { best = score; incK = k; incS = s; }
            }

        // Stage the incident quad in tmp.
        int ii = (incK + 1) % 3, jj = (incK + 2) % 3;
        tmp.Clear();
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
            {
                float si = a == 0 ? -1.0f : 1.0f;
                float sj = b == 0 ? -1.0f : 1.0f;
                tmp.Add(centerB + Vector3.Transform(
                    Axis(ii) * (Comp(heB, ii) * si) + Axis(jj) * (Comp(heB, jj) * sj) + Axis(incK) * (Comp(heB, incK) * incS), rotB));
            }

        int i = (refK + 1) % 3, j = (refK + 2) % 3;
        var u = Vector3.Transform(Axis(i), rotA);
        var v = Vector3.Transform(Axis(j), rotA);
        float hu = Comp(heA, i), hv = Comp(heA, j);
        var faceCenter = centerA + Vector3.Transform(Axis(refK) * (Comp(heA, refK) * refS), rotA);
        var nR = Vector3.Transform(Axis(refK) * refS, rotA);

        float cu = Vector3.Dot(u, faceCenter);
        float cv = Vector3.Dot(v, faceCenter);
        int before = output.Count;
        // tmp -> output
        ClipPolygonPlaneInto(tmp, u, cu + hu, output);
        if (output.Count == before) return;
        // output -> tmp
        tmp.Clear();
        for (int k = before; k < output.Count; k++) tmp.Add(output[k]);
        output.RemoveRange(before, output.Count - before);
        ClipPolygonPlaneInto(tmp, -u, -cu + hu, output);
        if (output.Count == before) return;
        // output -> tmp
        tmp.Clear();
        for (int k = before; k < output.Count; k++) tmp.Add(output[k]);
        output.RemoveRange(before, output.Count - before);
        ClipPolygonPlaneInto(tmp, v, cv + hv, output);
        if (output.Count == before) return;
        // output -> tmp
        tmp.Clear();
        for (int k = before; k < output.Count; k++) tmp.Add(output[k]);
        output.RemoveRange(before, output.Count - before);
        ClipPolygonPlaneInto(tmp, -v, -cv + hv, output);
        if (output.Count == before) return;

        // Keep points behind (or on) the reference face, capped at 4.
        float refOff = Vector3.Dot(nR, faceCenter);
        int kept = 0;
        for (int k = before; k < output.Count && kept < 4; k++)
        {
            if (Vector3.Dot(nR, output[k]) - refOff <= 0.01f)
            {
                output[before + kept] = output[k];
                kept++;
            }
        }
        output.RemoveRange(before + kept, output.Count - (before + kept));
    }

    private static Vector3 Axis(int k) => k == 0 ? Vector3.UnitX : k == 1 ? Vector3.UnitY : Vector3.UnitZ;

    private static float Comp(Vector3 v, int k) => k == 0 ? v.X : k == 1 ? v.Y : v.Z;

    /// <summary>
    /// Closest point on a box surface to a query point, plus the outward
    /// normal there. Mirrors <c>closest_point_on_box</c> in contact3d.rs.
    /// </summary>
    public static (Vector3 Point, Vector3 Normal) ClosestPointOnBox(
        Vector3 query, Vector3 center, Vector3 he, Quaternion rot)
    {
        var inv = Quaternion.Inverse(rot);
        var local = Vector3.Transform(query - center, inv);
        var clamped = new Vector3(
            System.Math.Clamp(local.X, -he.X, he.X),
            System.Math.Clamp(local.Y, -he.Y, he.Y),
            System.Math.Clamp(local.Z, -he.Z, he.Z));
        if ((local - clamped).LengthSquared() > 1e-12f)
        {
            var world = center + Vector3.Transform(clamped, rot);
            return (world, Vector3.Normalize(query - world));
        }
        float dx = he.X - MathF.Abs(local.X);
        float dy = he.Y - MathF.Abs(local.Y);
        float dz = he.Z - MathF.Abs(local.Z);
        int axisIdx;
        float s;
        if (dx <= dy && dx <= dz) { axisIdx = 0; s = System.Math.Sign(local.X); }
        else if (dy <= dz) { axisIdx = 1; s = System.Math.Sign(local.Y); }
        else { axisIdx = 2; s = System.Math.Sign(local.Z); }
        if (s == 0) s = 1;
        var n = Vector3.Transform(Axis(axisIdx) * s, rot);
        var outLocal = clamped;
        if (axisIdx == 0) outLocal.X = he.X * s;
        else if (axisIdx == 1) outLocal.Y = he.Y * s;
        else outLocal.Z = he.Z * s;
        return (center + Vector3.Transform(outLocal, rot), n);
    }

    public static (Vector3 C1, Vector3 C2) ClosestPointsSegmentSegment(
        Vector3 p1, Vector3 q1, Vector3 p2, Vector3 q2)
    {
        var d1 = q1 - p1;
        var d2 = q2 - p2;
        var r = p1 - p2;
        float a = Vector3.Dot(d1, d1);
        float e = Vector3.Dot(d2, d2);
        float f = Vector3.Dot(d2, r);
        if (a <= 1e-10f && e <= 1e-10f) return (p1, p2);
        float s, t;
        if (a <= 1e-10f) { s = 0; t = System.Math.Clamp(f / e, 0, 1); }
        else
        {
            float c = Vector3.Dot(d1, r);
            if (e <= 1e-10f) { t = 0; s = System.Math.Clamp(-c / a, 0, 1); }
            else
            {
                float b = Vector3.Dot(d1, d2);
                float denom = a * e - b * b;
                s = denom != 0 ? System.Math.Clamp((b * f - c * e) / denom, 0, 1) : 0;
                t = (b * s + f) / e;
                if (t < 0) { t = 0; s = System.Math.Clamp(-c / a, 0, 1); }
                else if (t > 1) { t = 1; s = System.Math.Clamp((b - c) / a, 0, 1); }
            }
        }
        return (p1 + d1 * s, p2 + d2 * t);
    }
}
