using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Deterministic Barnes-Hut approximation for N-body gravity (managed
/// mirror of <c>phieng_dynamics::barnes_hut</c>): same index-order
/// insertion, same midpoint routing, same fixed-order traversal and
/// self-force exclusion. Cross-language bit-equality is NOT claimed
/// (transcendentals may differ) — managed/native agreement is asserted
/// within tolerances. Same-mode managed runs are deterministic.
/// </summary>
public static class BarnesHut
{
    private const float EpsSq = 1e-12f;

    /// <summary>Maximum subdivision depth (bounds coincident-point splitting).</summary>
    private const uint MaxTreeDepth = 32;

    // ------------------------------------------------------------------
    // 2D quadtree.
    // ------------------------------------------------------------------

    private sealed class QuadNode
    {
        public Vector2 Min;
        public Vector2 Max;
        public readonly List<int> Bodies = new();
        // Child indices in fixed order: 0=(-x,-y), 1=(+x,-y), 2=(-x,+y), 3=(+x,+y). -1 = empty.
        public readonly int[] Children = { -1, -1, -1, -1 };
        public float Mass;
        public Vector2 Com;

        public bool IsLeaf()
        {
            return Children[0] < 0 && Children[1] < 0 && Children[2] < 0 && Children[3] < 0;
        }

        public float Extent()
        {
            return MathF.Max(Max.X - Min.X, Max.Y - Min.Y);
        }
    }

    private static int RouteQuad(QuadNode node, Vector2 p)
    {
        Vector2 mid = (node.Min + node.Max) * 0.5f;
        int ix = p.X < mid.X ? 0 : 1;
        int iy = p.Y < mid.Y ? 0 : 1;
        return iy * 2 + ix;
    }

    private static void SubdivideQuad(List<QuadNode> nodes, int nodeIdx)
    {
        var parent = nodes[nodeIdx];
        Vector2 mid = (parent.Min + parent.Max) * 0.5f;
        var quads = new (Vector2 Lo, Vector2 Hi)[]
        {
            (parent.Min, mid),
            (new Vector2(mid.X, parent.Min.Y), new Vector2(parent.Max.X, mid.Y)),
            (new Vector2(parent.Min.X, mid.Y), new Vector2(mid.X, parent.Max.Y)),
            (mid, parent.Max),
        };
        int Base = nodes.Count;
        foreach (var (lo, hi) in quads)
        {
            nodes.Add(new QuadNode { Min = lo, Max = hi });
        }
        // Pushed in order 0..3, so arena indices are contiguous.
        var node = nodes[nodeIdx];
        node.Children[0] = Base;
        node.Children[1] = Base + 1;
        node.Children[2] = Base + 2;
        node.Children[3] = Base + 3;
    }

    private static void InsertQuad(List<QuadNode> nodes, int nodeIdx, int bodyIdx, List<Vector2> positions, uint depth)
    {
        var node = nodes[nodeIdx];
        if (!node.IsLeaf())
        {
            int child = RouteQuad(node, positions[bodyIdx]);
            InsertQuad(nodes, node.Children[child], bodyIdx, positions, depth + 1);
            return;
        }
        if (node.Bodies.Count == 0 || depth >= MaxTreeDepth || node.Extent() == 0.0f)
        {
            node.Bodies.Add(bodyIdx);
            return;
        }
        SubdivideQuad(nodes, nodeIdx);
        node = nodes[nodeIdx];
        var existing = new List<int>(node.Bodies);
        node.Bodies.Clear();
        foreach (int b in existing)
        {
            int c = RouteQuad(node, positions[b]);
            InsertQuad(nodes, node.Children[c], b, positions, depth + 1);
        }
        int nc = RouteQuad(node, positions[bodyIdx]);
        InsertQuad(nodes, node.Children[nc], bodyIdx, positions, depth + 1);
    }

    private static void AccumulateQuad(List<QuadNode> nodes, int nodeIdx, List<Vector2> positions, List<float> masses)
    {
        var node = nodes[nodeIdx];
        if (node.IsLeaf())
        {
            float mass = 0.0f;
            Vector2 moment = Vector2.Zero;
            foreach (int b in node.Bodies)
            {
                mass += masses[b];
                moment += positions[b] * masses[b];
            }
            node.Mass = mass;
            node.Com = mass > 0.0f ? moment / mass : (node.Min + node.Max) * 0.5f;
            return;
        }
        float m = 0.0f;
        Vector2 mom = Vector2.Zero;
        for (int c = 0; c < 4; c++)
        {
            int child = node.Children[c];
            if (child < 0) continue;
            AccumulateQuad(nodes, child, positions, masses);
            var cn = nodes[child];
            m += cn.Mass;
            mom += cn.Com * cn.Mass;
        }
        node.Mass = m;
        node.Com = m > 0.0f ? mom / m : (node.Min + node.Max) * 0.5f;
    }

    private static void WalkQuad(
        List<QuadNode> nodes, int nodeIdx,
        Vector2 qpos, int qidx,
        List<Vector2> positions, List<float> masses,
        float bigG, float softening, float theta,
        ref Vector2 acc)
    {
        var node = nodes[nodeIdx];
        if (node.IsLeaf())
        {
            foreach (int k in node.Bodies)
            {
                if (k == qidx) continue;
                var (a, _) = GravityIntegration.PairAccelerations2(qpos, 1.0f, positions[k], masses[k], bigG, softening);
                acc += a;
            }
            return;
        }
        int routed = RouteQuad(node, qpos);
        for (int c = 0; c < 4; c++)
        {
            int child = node.Children[c];
            if (child < 0) continue;
            if (c == routed)
            {
                WalkQuad(nodes, child, qpos, qidx, positions, masses, bigG, softening, theta, ref acc);
                continue;
            }
            var cn = nodes[child];
            Vector2 d = cn.Com - qpos;
            float dist2 = d.LengthSquared() + softening * softening;
            if (dist2 < EpsSq) continue;
            if (cn.Extent() / MathF.Sqrt(dist2) < theta)
            {
                var (a, _) = GravityIntegration.PairAccelerations2(qpos, 1.0f, cn.Com, cn.Mass, bigG, softening);
                acc += a;
            }
            else
            {
                WalkQuad(nodes, child, qpos, qidx, positions, masses, bigG, softening, theta, ref acc);
            }
        }
    }

    /// <summary>
    /// Barnes-Hut force contribution over 2D bodies (tree part only —
    /// the caller runs the uniform background pass first, mirroring
    /// the Rust kick structure).
    /// </summary>
    public static void ApplyForces2(List<RigidBody2> bodies, float bigG, float softening, float theta, float h, bool drag)
    {
        int n = bodies.Count;
        if (n == 0) return;
        var positions = new List<Vector2>(n);
        var masses = new List<float>(n);
        foreach (var b in bodies)
        {
            positions.Add(b.Transform.Position);
            masses.Add(b.Mass);
        }
        var nodes = new List<QuadNode>
        {
            new()
            {
                Min = new Vector2(float.PositiveInfinity, float.PositiveInfinity),
                Max = new Vector2(float.NegativeInfinity, float.NegativeInfinity),
            },
        };
        for (int i = 0; i < n; i++)
        {
            var root = nodes[0];
            if (positions[i].X < root.Min.X) root.Min = new Vector2(positions[i].X, root.Min.Y);
            if (positions[i].Y < root.Min.Y) root.Min = new Vector2(root.Min.X, positions[i].Y);
            if (positions[i].X > root.Max.X) root.Max = new Vector2(positions[i].X, root.Max.Y);
            if (positions[i].Y > root.Max.Y) root.Max = new Vector2(root.Max.X, positions[i].Y);
        }
        for (int i = 0; i < n; i++)
        {
            InsertQuad(nodes, 0, i, positions, 0);
        }
        AccumulateQuad(nodes, 0, positions, masses);
        for (int i = 0; i < n; i++)
        {
            var b = bodies[i];
            if (b.IsStatic || b.IsSleeping) continue;
            Vector2 acc = Vector2.Zero;
            WalkQuad(nodes, 0, positions[i], i, positions, masses, bigG, softening, theta, ref acc);
            Vector2 dv = acc * (b.GravityScale * h);
            if (drag) dv -= b.Velocity * (b.LinearDamping * h);
            b.Velocity += dv;
        }
    }

    // ------------------------------------------------------------------
    // 3D octree (same contract as the quadtree above).
    // ------------------------------------------------------------------

    private sealed class OctNode
    {
        public Vector3 Min;
        public Vector3 Max;
        public readonly List<int> Bodies = new();
        // Child indices in fixed order: bit 0 = +x, bit 1 = +y, bit 2 = +z.
        public readonly int[] Children = { -1, -1, -1, -1, -1, -1, -1, -1 };
        public float Mass;
        public Vector3 Com;

        public bool IsLeaf()
        {
            for (int i = 0; i < 8; i++)
                if (Children[i] >= 0) return false;
            return true;
        }

        public float Extent()
        {
            return MathF.Max(Max.X - Min.X, MathF.Max(Max.Y - Min.Y, Max.Z - Min.Z));
        }
    }

    private static int RouteOct(OctNode node, Vector3 p)
    {
        Vector3 mid = (node.Min + node.Max) * 0.5f;
        int ix = p.X < mid.X ? 0 : 1;
        int iy = p.Y < mid.Y ? 0 : 1;
        int iz = p.Z < mid.Z ? 0 : 1;
        return iz * 4 + iy * 2 + ix;
    }

    private static void SubdivideOct(List<OctNode> nodes, int nodeIdx)
    {
        var parent = nodes[nodeIdx];
        Vector3 mid = (parent.Min + parent.Max) * 0.5f;
        int Base = nodes.Count;
        for (int iz = 0; iz < 2; iz++)
        {
            for (int iy = 0; iy < 2; iy++)
            {
                for (int ix = 0; ix < 2; ix++)
                {
                    var lo = new Vector3(
                        ix == 0 ? parent.Min.X : mid.X,
                        iy == 0 ? parent.Min.Y : mid.Y,
                        iz == 0 ? parent.Min.Z : mid.Z);
                    var hi = new Vector3(
                        ix == 0 ? mid.X : parent.Max.X,
                        iy == 0 ? mid.Y : parent.Max.Y,
                        iz == 0 ? mid.Z : parent.Max.Z);
                    nodes.Add(new OctNode { Min = lo, Max = hi });
                }
            }
        }
        // Pushed iz-major in 0..8 order, so arena indices are contiguous.
        var node = nodes[nodeIdx];
        for (int i = 0; i < 8; i++)
        {
            node.Children[i] = Base + i;
        }
    }

    private static void InsertOct(List<OctNode> nodes, int nodeIdx, int bodyIdx, List<Vector3> positions, uint depth)
    {
        var node = nodes[nodeIdx];
        if (!node.IsLeaf())
        {
            int child = RouteOct(node, positions[bodyIdx]);
            InsertOct(nodes, node.Children[child], bodyIdx, positions, depth + 1);
            return;
        }
        if (node.Bodies.Count == 0 || depth >= MaxTreeDepth || node.Extent() == 0.0f)
        {
            node.Bodies.Add(bodyIdx);
            return;
        }
        SubdivideOct(nodes, nodeIdx);
        node = nodes[nodeIdx];
        var existing = new List<int>(node.Bodies);
        node.Bodies.Clear();
        foreach (int b in existing)
        {
            int c = RouteOct(node, positions[b]);
            InsertOct(nodes, node.Children[c], b, positions, depth + 1);
        }
        int nc = RouteOct(node, positions[bodyIdx]);
        InsertOct(nodes, node.Children[nc], bodyIdx, positions, depth + 1);
    }

    private static void AccumulateOct(List<OctNode> nodes, int nodeIdx, List<Vector3> positions, List<float> masses)
    {
        var node = nodes[nodeIdx];
        if (node.IsLeaf())
        {
            float mass = 0.0f;
            Vector3 moment = Vector3.Zero;
            foreach (int b in node.Bodies)
            {
                mass += masses[b];
                moment += positions[b] * masses[b];
            }
            node.Mass = mass;
            node.Com = mass > 0.0f ? moment / mass : (node.Min + node.Max) * 0.5f;
            return;
        }
        float m = 0.0f;
        Vector3 mom = Vector3.Zero;
        for (int c = 0; c < 8; c++)
        {
            int child = node.Children[c];
            if (child < 0) continue;
            AccumulateOct(nodes, child, positions, masses);
            var cn = nodes[child];
            m += cn.Mass;
            mom += cn.Com * cn.Mass;
        }
        node.Mass = m;
        node.Com = m > 0.0f ? mom / m : (node.Min + node.Max) * 0.5f;
    }

    private static void WalkOct(
        List<OctNode> nodes, int nodeIdx,
        Vector3 qpos, int qidx,
        List<Vector3> positions, List<float> masses,
        float bigG, float softening, float theta,
        ref Vector3 acc)
    {
        var node = nodes[nodeIdx];
        if (node.IsLeaf())
        {
            foreach (int k in node.Bodies)
            {
                if (k == qidx) continue;
                var (a, _) = GravityIntegration.PairAccelerations3(qpos, 1.0f, positions[k], masses[k], bigG, softening);
                acc += a;
            }
            return;
        }
        int routed = RouteOct(node, qpos);
        for (int c = 0; c < 8; c++)
        {
            int child = node.Children[c];
            if (child < 0) continue;
            if (c == routed)
            {
                WalkOct(nodes, child, qpos, qidx, positions, masses, bigG, softening, theta, ref acc);
                continue;
            }
            var cn = nodes[child];
            Vector3 d = cn.Com - qpos;
            float dist2 = d.LengthSquared() + softening * softening;
            if (dist2 < EpsSq) continue;
            if (cn.Extent() / MathF.Sqrt(dist2) < theta)
            {
                var (a, _) = GravityIntegration.PairAccelerations3(qpos, 1.0f, cn.Com, cn.Mass, bigG, softening);
                acc += a;
            }
            else
            {
                WalkOct(nodes, child, qpos, qidx, positions, masses, bigG, softening, theta, ref acc);
            }
        }
    }

    /// <summary>3D counterpart of <see cref="ApplyForces2"/>.</summary>
    public static void ApplyForces3(List<RigidBody3> bodies, float bigG, float softening, float theta, float h, bool drag)
    {
        int n = bodies.Count;
        if (n == 0) return;
        var positions = new List<Vector3>(n);
        var masses = new List<float>(n);
        foreach (var b in bodies)
        {
            positions.Add(b.Transform.Position);
            masses.Add(b.Mass);
        }
        var nodes = new List<OctNode>
        {
            new()
            {
                Min = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity),
                Max = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity),
            },
        };
        for (int i = 0; i < n; i++)
        {
            var root = nodes[0];
            if (positions[i].X < root.Min.X) root.Min = new Vector3(positions[i].X, root.Min.Y, root.Min.Z);
            if (positions[i].Y < root.Min.Y) root.Min = new Vector3(root.Min.X, positions[i].Y, root.Min.Z);
            if (positions[i].Z < root.Min.Z) root.Min = new Vector3(root.Min.X, root.Min.Y, positions[i].Z);
            if (positions[i].X > root.Max.X) root.Max = new Vector3(positions[i].X, root.Max.Y, root.Max.Z);
            if (positions[i].Y > root.Max.Y) root.Max = new Vector3(root.Max.X, positions[i].Y, root.Max.Z);
            if (positions[i].Z > root.Max.Z) root.Max = new Vector3(root.Max.X, root.Max.Y, positions[i].Z);
        }
        for (int i = 0; i < n; i++)
        {
            InsertOct(nodes, 0, i, positions, 0);
        }
        AccumulateOct(nodes, 0, positions, masses);
        for (int i = 0; i < n; i++)
        {
            var b = bodies[i];
            if (b.IsStatic || b.IsSleeping) continue;
            Vector3 acc = Vector3.Zero;
            WalkOct(nodes, 0, positions[i], i, positions, masses, bigG, softening, theta, ref acc);
            Vector3 dv = acc * (b.GravityScale * h);
            if (drag) dv -= b.Velocity * (b.LinearDamping * h);
            b.Velocity += dv;
        }
    }
}
