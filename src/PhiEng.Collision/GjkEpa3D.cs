using System.Numerics;

namespace PhiEng.Collision;

/// <summary>
/// Result of a 3D GJK query. Mirrors the Rust <c>GjkResult</c>.
/// </summary>
public enum GjkResult3D
{
    Separated,
    Colliding,
}

/// <summary>
/// EPA output: contact normal (A to B), penetration depth, witness points.
/// Mirrors the Rust <c>EpaResult</c>.
/// </summary>
public class EpaResult3D
{
    public Vector3 Normal { get; set; }
    public float Depth { get; set; }
    public Vector3 ContactPointA { get; set; }
    public Vector3 ContactPointB { get; set; }
}

/// <summary>
/// 3D GJK + EPA narrowphase. Mirrors crates/collision/src/gjk3d.rs and epa.rs.
/// </summary>
public static class GjkEpa3D
{
    public const int MaxIterations = 32;
    public const int EpaMaxIterations = 64;
    public const float Epsilon = 1e-6f;
    public const float EpaTolerance = 1e-4f;

    public static Vector3 SupportSphere(Vector3 center, float radius, Vector3 dir)
    {
        if (dir.LengthSquared() < 1e-12f) return center + new Vector3(radius, 0, 0);
        return center + Vector3.Normalize(dir) * radius;
    }

    public static Vector3 SupportObb(Vector3 center, Vector3 halfExtents, Quaternion rotation, Vector3 dir)
    {
        var inv = Quaternion.Inverse(rotation);
        var local = Vector3.Transform(dir, inv);
        var s = new Vector3(
            local.X > 0 ? halfExtents.X : local.X < 0 ? -halfExtents.X : 0,
            local.Y > 0 ? halfExtents.Y : local.Y < 0 ? -halfExtents.Y : 0,
            local.Z > 0 ? halfExtents.Z : local.Z < 0 ? -halfExtents.Z : 0);
        return center + Vector3.Transform(s, rotation);
    }

    public static Vector3 SupportCapsule(Vector3 a, Vector3 b, float radius, Vector3 dir)
    {
        float len = dir.Length();
        var d = len > 1e-10f ? dir / len : Vector3.UnitX;
        var ab = b - a;
        float t = Vector3.Dot(ab, d);
        Vector3 closest = MathF.Abs(t) < 1e-10f ? (a + b) * 0.5f : t > 0 ? b : a;
        return closest + d * radius;
    }

    public static Vector3 SupportHull(IReadOnlyList<Vector3> verts, Vector3 dir)
    {
        float best = float.MinValue;
        var p = Vector3.Zero;
        foreach (var v in verts)
        {
            float x = Vector3.Dot(v, dir);
            if (x > best) { best = x; p = v; }
        }
        return p;
    }

    private sealed class Simplex
    {
        public readonly Vector3[] Points = new Vector3[4];
        public int Count;
        public void Push(Vector3 p) => Points[Count++] = p;
    }

    private static bool UpdateSimplex(Simplex s, ref Vector3 dir)
    {
        // Mirrors Rust update_simplex: Ericson Voronoi-region reductions.
        // The simplex is always reduced to the feature closest to the
        // origin; keeping a stale triangle whose closest feature is an
        // edge makes GJK cycle to iteration exhaustion and falsely
        // report separation (missed contact, tunneling).
        switch (s.Count)
        {
            case 2:
            {
                // a = newest point.
                var a = s.Points[1];
                var b = s.Points[0];
                var ab = b - a;
                var ao = -a;
                if (Vector3.Dot(ab, ao) <= 0)
                {
                    // Origin in vertex A's region: drop B.
                    s.Points[0] = a;
                    s.Count = 1;
                    dir = ao;
                }
                else
                {
                    dir = Vector3.Cross(Vector3.Cross(ab, ao), ab);
                    if (dir.LengthSquared() < Epsilon)
                        dir = MathF.Abs(ab.X) > MathF.Abs(ab.Y)
                            ? new Vector3(-ab.Z, 0, ab.X)
                            : new Vector3(0, ab.Z, -ab.Y);
                }
                return false;
            }
            case 3:
                return SolveTriangle(s, s.Points[2], s.Points[1], s.Points[0], ref dir);
            case 4:
                return SolveTetrahedron(s, s.Points[3], s.Points[2], s.Points[1], s.Points[0], ref dir);
            default: return false;
        }
    }

    private static bool SolveTriangle(Simplex s, Vector3 a, Vector3 b, Vector3 c, ref Vector3 dir)
    {
        var ab = b - a;
        var ac = c - a;
        var ao = -a;

        // Vertex A region.
        float d1 = Vector3.Dot(ab, ao);
        float d2 = Vector3.Dot(ac, ao);
        if (d1 <= 0 && d2 <= 0)
        {
            if (ao.LengthSquared() < Epsilon) return true;
            s.Points[0] = a;
            s.Count = 1;
            dir = ao;
            return false;
        }

        // Vertex B region.
        var bo = -b;
        float d3 = Vector3.Dot(ab, bo);
        float d4 = Vector3.Dot(ac, bo);
        if (d3 >= 0 && d4 <= d3)
        {
            if (bo.LengthSquared() < Epsilon) return true;
            s.Points[0] = b;
            s.Count = 1;
            dir = bo;
            return false;
        }

        // Edge AB region.
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0)
        {
            // d1 - d3 = |ab|^2; guard the degenerate a == b case.
            if (d1 - d3 < 1e-12f)
            {
                s.Points[0] = a;
                s.Count = 1;
                dir = ao;
                return false;
            }
            float v = d1 / (d1 - d3);
            var closest = a + ab * v;
            if (closest.LengthSquared() < Epsilon) return true;
            s.Points[0] = b; s.Points[1] = a;
            s.Count = 2;
            dir = -closest;
            return false;
        }

        // Vertex C region.
        var co = -c;
        float d5 = Vector3.Dot(ab, co);
        float d6 = Vector3.Dot(ac, co);
        if (d6 >= 0 && d5 <= d6)
        {
            if (co.LengthSquared() < Epsilon) return true;
            s.Points[0] = c;
            s.Count = 1;
            dir = co;
            return false;
        }

        // Edge AC region.
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
        {
            // d2 - d6 = |ac|^2; guard the degenerate a == c case.
            if (d2 - d6 < 1e-12f)
            {
                s.Points[0] = a;
                s.Count = 1;
                dir = ao;
                return false;
            }
            float w = d2 / (d2 - d6);
            var closest = a + ac * w;
            if (closest.LengthSquared() < Epsilon) return true;
            s.Points[0] = c; s.Points[1] = a;
            s.Count = 2;
            dir = -closest;
            return false;
        }

        // Edge BC region.
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        {
            // (d4 - d3) + (d5 - d6) = |bc|^2; guard b == c.
            if ((d4 - d3) + (d5 - d6) < 1e-12f)
            {
                s.Points[0] = b;
                s.Count = 1;
                dir = bo;
                return false;
            }
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            var closest = b + (c - b) * w;
            if (closest.LengthSquared() < Epsilon) return true;
            s.Points[0] = c; s.Points[1] = b;
            s.Count = 2;
            dir = -closest;
            return false;
        }

        // Face region: origin projects inside the triangle.
        float denom = va + vb + vc;
        if (MathF.Abs(denom) < 1e-12f)
        {
            // Degenerate (zero-area) triangle: keep the newest vertex.
            s.Points[0] = a;
            s.Count = 1;
            dir = ao;
            return false;
        }
        float fv = vb / denom;
        float fw = vc / denom;
        var faceClosest = a + ab * fv + ac * fw;
        if (faceClosest.LengthSquared() < Epsilon) return true;
        var n = Vector3.Cross(ab, ac);
        // Face normal pointing toward the origin side.
        dir = Vector3.Dot(n, ao) > 0 ? n : -n;
        return false;
    }

    private static bool SolveTetrahedron(Simplex s, Vector3 a, Vector3 b, Vector3 c, Vector3 d, ref Vector3 dir)
    {
        // All four faces are tested, not just the three containing the
        // newest point: skipping the opposite face is only valid when
        // every prior reduction was exact.
        var faces = new[] { (a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a) };
        foreach (var (p0, p1, p2, opp) in faces)
        {
            var n = Vector3.Cross(p1 - p0, p2 - p0);
            if (n.LengthSquared() < 1e-24f) continue; // Degenerate face.
            if (Vector3.Dot(n, opp - p0) > 0) n = -n; // Orient outward.
            if (Vector3.Dot(n, -p0) > 0)
            {
                // Origin outside this face: reduce to it.
                s.Points[0] = p0; s.Points[1] = p1; s.Points[2] = p2;
                s.Count = 3;
                dir = n;
                return false;
            }
        }
        // Origin inside (or on) all faces: containment.
        return true;
    }

    private static Vector3 ClosestPtSegment(Vector3 a, Vector3 b)
    {
        var ab = b - a;
        float denom = Vector3.Dot(ab, ab);
        if (denom < 1e-24f) return a;
        float t = Vector3.Dot(-a, ab) / denom;
        if (t <= 0) return a;
        if (t >= 1) return b;
        return a + ab * t;
    }

    private static Vector3 ClosestPtTriangle(Vector3 a, Vector3 b, Vector3 c)
    {
        var ab = b - a;
        var ac = c - a;
        var ao = -a;
        float d1 = Vector3.Dot(ab, ao);
        float d2 = Vector3.Dot(ac, ao);
        if (d1 <= 0 && d2 <= 0) return a;
        var bo = -b;
        float d3 = Vector3.Dot(ab, bo);
        float d4 = Vector3.Dot(ac, bo);
        if (d3 >= 0 && d4 <= d3) return b;
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0)
        {
            float denom = d1 - d3;
            return MathF.Abs(denom) < 1e-12f ? a : a + ab * (d1 / denom);
        }
        var co = -c;
        float d5 = Vector3.Dot(ab, co);
        float d6 = Vector3.Dot(ac, co);
        if (d6 >= 0 && d5 <= d6) return c;
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
        {
            float denom = d2 - d6;
            return MathF.Abs(denom) < 1e-12f ? a : a + ac * (d2 / denom);
        }
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        {
            float denom = (d4 - d3) + (d5 - d6);
            return MathF.Abs(denom) < 1e-12f ? b : b + (c - b) * ((d4 - d3) / denom);
        }
        float fdenom = va + vb + vc;
        if (MathF.Abs(fdenom) < 1e-12f) return a;
        return a + ab * (vb / fdenom) + ac * (vc / fdenom);
    }

    private static (bool Contained, float Distance) SimplexOriginInfo(Simplex s)
    {
        switch (s.Count)
        {
            case 0: return (false, 0);
            case 1: return (s.Points[0].LengthSquared() < Epsilon, s.Points[0].Length());
            case 2:
            {
                var closest = ClosestPtSegment(s.Points[1], s.Points[0]);
                return (closest.LengthSquared() < Epsilon, closest.Length());
            }
            case 3:
            {
                var closest = ClosestPtTriangle(s.Points[2], s.Points[1], s.Points[0]);
                return (closest.LengthSquared() < Epsilon, closest.Length());
            }
            default:
            {
                var a = s.Points[3]; var b = s.Points[2];
                var c = s.Points[1]; var d = s.Points[0];
                var faces = new[] { (a, b, c), (a, c, d), (a, d, b), (b, d, c) };
                bool insideAll = true;
                float minDist = float.MaxValue;
                foreach (var (p0, p1, p2) in faces)
                {
                    var n = Vector3.Cross(p1 - p0, p2 - p0);
                    if (n.LengthSquared() < 1e-24f) continue;
                    var interior = a + b + c + d - p0 - p1 - p2;
                    if (Vector3.Dot(n, interior - p0) > 0) n = -n;
                    float planeDist = Vector3.Dot(Vector3.Normalize(n), -p0);
                    if (planeDist > 0) insideAll = false;
                    minDist = MathF.Min(minDist, ClosestPtTriangle(p0, p1, p2).Length());
                }
                return (insideAll, minDist == float.MaxValue ? 0 : minDist);
            }
        }
    }

    public static (GjkResult3D Result, List<Vector3> Simplex, float Distance) Gjk(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB)
    {
        var dir = new Vector3(1, 0, 0);
        var s = new Simplex();
        var p0 = supportA(dir) - supportB(-dir);
        s.Push(p0);
        dir = -p0;

        for (int i = 0; i < MaxIterations; i++)
        {
            if (dir.LengthSquared() < Epsilon)
                return (GjkResult3D.Colliding, new List<Vector3>(s.Points[..s.Count]), 0);
            var p = supportA(dir) - supportB(-dir);
            if (Vector3.Dot(p, dir) < -Epsilon)
            {
                // Sound separation proof, but the current simplex is only
                // a subset of the Minkowski difference: report its true
                // closest distance, never simplex point zero.
                var (_, dist) = SimplexOriginInfo(s);
                return (GjkResult3D.Separated, new List<Vector3>(s.Points[..s.Count]), MathF.Max(0, dist));
            }
            s.Push(p);
            if (UpdateSimplex(s, ref dir))
                return (GjkResult3D.Colliding, new List<Vector3>(s.Points[..s.Count]), 0);
        }
        // Iteration exhaustion: never report separation on a guess. A
        // tetrahedron containing the origin is a collision; otherwise
        // fall back to the true closest distance of the final simplex.
        var (contained, fallDist) = SimplexOriginInfo(s);
        return contained
            ? (GjkResult3D.Colliding, new List<Vector3>(s.Points[..s.Count]), 0)
            : (GjkResult3D.Separated, new List<Vector3>(s.Points[..s.Count]), fallDist);
    }

    internal sealed class PolyFace
    {
        public int A, B, C;
        public Vector3 Normal;
        public float Distance;
    }

    private static void AddHorizonEdge(List<(int, int)> horizon, int u, int v)
    {
        for (int idx = horizon.Count - 1; idx >= 0; idx--)
        {
            if (horizon[idx].Item1 == v && horizon[idx].Item2 == u)
            {
                horizon.RemoveAt(idx);
                return;
            }
        }
        horizon.Add((u, v));
    }

    private static Vector3 Bary(Vector3 a, Vector3 b, Vector3 c, Vector3 p)
    {
        var v0 = b - a; var v1 = c - a; var v2 = p - a;
        float d00 = Vector3.Dot(v0, v0), d01 = Vector3.Dot(v0, v1), d11 = Vector3.Dot(v1, v1);
        float d20 = Vector3.Dot(v2, v0), d21 = Vector3.Dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        if (MathF.Abs(denom) < 1e-9f) return new Vector3(1 / 3f, 1 / 3f, 1 / 3f);
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        return new Vector3(1 - v - w, v, w);
    }

    /// <summary>
    /// Reusable EPA working buffers. A world keeps one and passes it to
    /// <see cref="GjkEpa(Func{Vector3, Vector3}, Func{Vector3, Vector3}, EpaScratch)"/>
    /// for every GJK-positive pair, so the steady state performs no
    /// allocations inside EPA.
    /// </summary>
    public sealed class EpaScratch
    {
        internal readonly List<Vector3> Vertices = new(64);
        internal readonly List<PolyFace> Faces = new(64);
        internal readonly List<(int, int)> Horizon = new(16);
        internal readonly List<bool> Visible = new(64);
    }

    public static EpaResult3D? GjkEpa(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB)
        => GjkEpa(supportA, supportB, new EpaScratch());

    public static EpaResult3D? GjkEpa(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB,
        EpaScratch scratch)
    {
        var (result, simplex, _) = Gjk(supportA, supportB);
        if (result != GjkResult3D.Colliding) return null;
        // GJK often terminates on a lower-dimensional simplex (origin on
        // an edge/vertex) or a boundary tetrahedron. Feeding those to EPA
        // poisons it (faces through the origin all read distance ~0 and
        // EPA orbits them to exhaustion), so EPA only ever starts from a
        // STRICTLY containing tetrahedron built by search below.
        if (simplex.Count >= 4 && OriginInTetStrict(simplex[0], simplex[1], simplex[2], simplex[3], StrictTetMargin))
            return Epa(supportA, supportB, simplex, scratch);
        var tet = SeedTetrahedron(supportA, supportB);
        if (tet == null) return null;
        return Epa(supportA, supportB, tet, scratch);
    }

    /// <summary>
    /// Minimum clearance required of an EPA seed tetrahedron. Overlaps
    /// shallower than this are solver-irrelevant (slop = 0.005). Mirrors
    /// Rust <c>STRICT_TET_MARGIN</c>.
    /// </summary>
    public const float StrictTetMargin = 1e-4f;

    /// <summary>
    /// Strict containment: origin bounded away from every face by at
    /// least <paramref name="margin"/>, with every opposite vertex bounded
    /// away from its face plane by <paramref name="margin"/> too. The
    /// altitude requirement is load-bearing: near-coplanar quadruples
    /// (slivers) decide outward orientation by catastrophic cancellation —
    /// float noise, opposite signs in Rust vs C# — so a clearance-only test
    /// calls the same sliver "containing" in one language and "outside" in
    /// the other. Mirrors Rust <c>origin_in_tet_strict</c>.
    /// </summary>
    public static bool OriginInTetStrict(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float margin)
    {
        if (MathF.Abs(Vector3.Dot(Vector3.Cross(b - a, c - a), d - a)) < 1e-12f)
            return false;
        var faces = new[] { (a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a) };
        foreach (var (v0, v1, v2, opp) in faces)
        {
            var n = Vector3.Cross(v1 - v0, v2 - v0);
            if (n.LengthSquared() < 1e-20f) return false;
            n = Vector3.Normalize(n);
            if (MathF.Abs(Vector3.Dot(n, opp - v0)) < margin) return false;
            var outward = Vector3.Dot(n, opp - v0) > 0 ? -n : n;
            if (Vector3.Dot(outward, -v0) > -margin) return false;
        }
        return true;
    }

    private static float TetClearance(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        if (MathF.Abs(Vector3.Dot(Vector3.Cross(b - a, c - a), d - a)) < 1e-12f)
            return float.MinValue;
        var faces = new[] { (a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a) };
        float clearance = float.MaxValue;
        foreach (var (v0, v1, v2, opp) in faces)
        {
            var n = Vector3.Cross(v1 - v0, v2 - v0);
            if (n.LengthSquared() < 1e-20f) return float.MinValue;
            n = Vector3.Normalize(n);
            clearance = MathF.Min(clearance, MathF.Abs(Vector3.Dot(n, opp - v0)));
            var outward = Vector3.Dot(n, opp - v0) > 0 ? -n : n;
            clearance = MathF.Min(clearance, -Vector3.Dot(outward, -v0));
        }
        return clearance;
    }

    /// <summary>
    /// Build EPA's initial tetrahedron by deterministic search: support
    /// points in 6 axis + 8 diagonal directions, then the quadruple with
    /// maximum face clearance. Mirrors Rust <c>seed_tetrahedron</c>.
    /// </summary>
    public static List<Vector3>? SeedTetrahedron(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB)
    {
        Vector3 Support(Vector3 dir) => supportA(dir) - supportB(-dir);
        float s = 0.57735027f;
        var dirs = new[]
        {
            Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ,
            -Vector3.UnitX, -Vector3.UnitY, -Vector3.UnitZ,
            new Vector3(1, 1, 1) * s, new Vector3(1, 1, -1) * s,
            new Vector3(1, -1, 1) * s, new Vector3(1, -1, -1) * s,
            new Vector3(-1, 1, 1) * s, new Vector3(-1, 1, -1) * s,
            new Vector3(-1, -1, 1) * s, new Vector3(-1, -1, -1) * s,
        };
        var pts = new List<Vector3>(14);
        foreach (var dir in dirs)
        {
            var p = Support(dir);
            // Dedupe: repeated support points add only degenerate quadruples.
            bool dup = false;
            foreach (var q in pts)
                if ((q - p).LengthSquared() < 1e-12f) { dup = true; break; }
            if (!dup) pts.Add(p);
        }
        List<Vector3>? best = null;
        float bestClearance = StrictTetMargin;
        for (int i = 0; i < pts.Count; i++)
            for (int j = i + 1; j < pts.Count; j++)
                for (int k = j + 1; k < pts.Count; k++)
                    for (int l = k + 1; l < pts.Count; l++)
                    {
                        float clearance = TetClearance(pts[i], pts[j], pts[k], pts[l]);
                        // Strict `>`: first-best wins ties, keeping the
                        // search deterministic.
                        if (clearance > StrictTetMargin && clearance > bestClearance)
                        {
                            bestClearance = clearance;
                            best = new List<Vector3> { pts[i], pts[j], pts[k], pts[l] };
                        }
                    }
        return best;
    }

    /// <summary>
    /// Origin inside tetrahedron, with tolerance for boundary cases.
    /// Flat (zero-volume) tetrahedra never count. Mirrors Rust <c>origin_in_tet</c>.
    /// </summary>
    public static bool OriginInTet(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        if (MathF.Abs(Vector3.Dot(Vector3.Cross(b - a, c - a), d - a)) < 1e-12f)
            return false;
        var faces = new[] { (a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a) };
        foreach (var (v0, v1, v2, opp) in faces)
        {
            var n = Vector3.Cross(v1 - v0, v2 - v0);
            if (n.LengthSquared() < 1e-20f) return false;
            n = Vector3.Normalize(n);
            var outward = Vector3.Dot(n, opp - v0) > 0 ? -n : n;
            if (Vector3.Dot(outward, -v0) > 1e-6f) return false;
        }
        return true;
    }

    private static EpaResult3D? Epa(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB,
        List<Vector3> simplex)
        => Epa(supportA, supportB, simplex, new EpaScratch());

    private static EpaResult3D? Epa(
        Func<Vector3, Vector3> supportA, Func<Vector3, Vector3> supportB,
        List<Vector3> simplex, EpaScratch scratch)
    {
        var verts = scratch.Vertices;
        verts.Clear();
        verts.AddRange(simplex);
        var faces = scratch.Faces;
        faces.Clear();
        int[,] init = { { 0, 1, 2, 3 }, { 0, 3, 1, 2 }, { 0, 2, 3, 1 }, { 1, 3, 2, 0 } };
        for (int k = 0; k < 4; k++)
        {
            var v0 = verts[init[k, 0]]; var v1 = verts[init[k, 1]];
            var v2 = verts[init[k, 2]]; var opp = verts[init[k, 3]];
            var n = Vector3.Cross(v1 - v0, v2 - v0);
            if (n.Length() < 1e-10f) continue;
            n = Vector3.Normalize(n);
            if (Vector3.Dot(n, opp - v0) > 0) n = -n;
            faces.Add(new PolyFace { A = init[k, 0], B = init[k, 1], C = init[k, 2], Normal = n, Distance = Vector3.Dot(n, v0) });
        }

        for (int iter = 0; iter < EpaMaxIterations; iter++)
        {
            if (faces.Count == 0) break;
            int best = 0;
            float min = float.MaxValue;
            for (int i = 0; i < faces.Count; i++)
                if (faces[i].Distance < min) { min = faces[i].Distance; best = i; }
            var face = faces[best];
            var p = supportA(face.Normal) - supportB(-face.Normal);
            float pd = Vector3.Dot(p, face.Normal);
            if (pd - face.Distance < EpaTolerance)
            {
                var v0 = verts[face.A]; var v1 = verts[face.B]; var v2 = verts[face.C];
                var bary = Bary(v0, v1, v2, p);
                var c = v0 * bary.X + v1 * bary.Y + v2 * bary.Z;
                return new EpaResult3D { Normal = face.Normal, Depth = face.Distance, ContactPointA = c, ContactPointB = c };
            }

            verts.Add(p);
            int ni = verts.Count - 1;
            var visible = scratch.Visible;
            visible.Clear();
            for (int vi = 0; vi < faces.Count; vi++)
                visible.Add(Vector3.Dot(faces[vi].Normal, p - verts[faces[vi].A]) > 0);
            // All faces visible: the apex overhangs the whole polytope,
            // so the horizon is empty and expanding would annihilate it.
            // Return the current minimum face instead: its distance
            // UNDERestimates the true depth, and an underestimated depth
            // is a weaker solver push — always stable.
            if (visible.All(v => v))
            {
                var w0 = verts[face.A]; var w1 = verts[face.B]; var w2 = verts[face.C];
                var wbary = Bary(w0, w1, w2, p);
                var wc = w0 * wbary.X + w1 * wbary.Y + w2 * wbary.Z;
                return new EpaResult3D { Normal = face.Normal, Depth = face.Distance, ContactPointA = wc, ContactPointB = wc };
            }
            var horizon = scratch.Horizon;
            horizon.Clear();
            for (int i = 0; i < faces.Count; i++)
            {
                if (!visible[i]) continue;
                var fa = faces[i].A; var fb = faces[i].B; var fc = faces[i].C;
                // Unrolled edge loop (avoids per-iteration array alloc).
                AddHorizonEdge(horizon, fa, fb);
                AddHorizonEdge(horizon, fb, fc);
                AddHorizonEdge(horizon, fc, fa);
            }
            for (int i = faces.Count - 1; i >= 0; i--)
                if (visible[i]) faces.RemoveAt(i);
            foreach (var (u, v) in horizon)
            {
                var v0 = verts[u]; var v1 = verts[v]; var v2 = p;
                var n = Vector3.Cross(v1 - v0, v2 - v0);
                if (n.Length() < 1e-10f) continue;
                n = Vector3.Normalize(n);
                if (Vector3.Dot(n, v0) < 0) n = -n;
                float d = Vector3.Dot(n, v0);
                if (d > -1e-6f)
                    faces.Add(new PolyFace { A = u, B = v, C = ni, Normal = n, Distance = MathF.Max(0, d) });
            }
        }

        if (faces.Count == 0) return null;
        var bf = faces.MinBy(f => f.Distance)!;
        var sv = verts[bf.A] * 0.34f + verts[bf.B] * 0.33f + verts[bf.C] * 0.33f;
        return new EpaResult3D { Normal = bf.Normal, Depth = bf.Distance, ContactPointA = sv, ContactPointB = sv };
    }
}
