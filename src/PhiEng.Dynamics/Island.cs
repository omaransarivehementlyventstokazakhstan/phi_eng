namespace PhiEng.Dynamics;

/// <summary>
/// Deterministic constraint islands for parallel solving. Mirrors Rust
/// <c>island.rs</c> in crates/dynamics/src/island.rs — same union-find
/// over canonically sorted edges, same canonical island order, same
/// stable within-island constraint order.
/// </summary>
public sealed class Island
{
    /// <summary>Sorted body indices owned by this island.</summary>
    public List<int> Bodies { get; } = new();
    /// <summary>Constraint indices (stable input order).</summary>
    public List<int> Constraints { get; } = new();
}

/// <summary>Deterministic island builder (stateless, rebuilt every step).</summary>
public static class IslandManager
{
    /// <summary>
    /// Build islands from contact pairs and joint endpoint pairs.
    /// Self-pairs and out-of-range ids are dropped. Output is canonical:
    /// sorted by smallest body id, members ascending.
    /// </summary>
    public static List<Island> BuildIslands(
        int bodyCount, IList<(int, int)> contacts, IList<(int, int)> joints)
    {
        var islands = new List<Island>();
        if (bodyCount <= 0) return islands;

        var edges = new List<(int, int)>(contacts.Count + joints.Count);
        foreach (var (a, b) in contacts) AddEdge(edges, a, b, bodyCount);
        foreach (var (a, b) in joints) AddEdge(edges, a, b, bodyCount);
        edges.Sort();
        // Dedup (sorted, so linear).
        int kept = 0;
        for (int i = 0; i < edges.Count; i++)
        {
            if (i == 0 || edges[i] != edges[i - 1])
                edges[kept++] = edges[i];
        }
        if (kept < edges.Count) edges.RemoveRange(kept, edges.Count - kept);

        // Union-find with path compression + union by rank.
        var parent = new int[bodyCount];
        var rank = new byte[bodyCount];
        for (int i = 0; i < bodyCount; i++) parent[i] = i;
        int Find(int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        void Union(int a, int b)
        {
            int ra = Find(a), rb = Find(b);
            if (ra == rb) return;
            if (rank[ra] < rank[rb]) parent[ra] = rb;
            else if (rank[ra] > rank[rb]) parent[rb] = ra;
            else { parent[rb] = ra; rank[ra]++; }
        }
        foreach (var (a, b) in edges) Union(a, b);

        var present = new bool[bodyCount];
        foreach (var (a, b) in edges) { present[a] = true; present[b] = true; }
        var groups = new Dictionary<int, Island>();
        for (int i = 0; i < bodyCount; i++)
        {
            if (!present[i]) continue;
            int r = Find(i);
            if (!groups.TryGetValue(r, out var isl))
            {
                isl = new Island();
                groups[r] = isl;
            }
            isl.Bodies.Add(i);
        }
        islands.AddRange(groups.Values);
        islands.Sort((x, y) => x.Bodies[0].CompareTo(y.Bodies[0]));
        return islands;
    }

    private static void AddEdge(List<(int, int)> edges, int a, int b, int n)
    {
        if (a < 0 || b < 0 || a >= n || b >= n || a == b) return;
        edges.Add(a < b ? (a, b) : (b, a));
    }

    /// <summary>Body-index → island-index map (usize::MAX equivalent: -1).</summary>
    public static int[] IslandIndexMap(List<Island> islands, int bodyCount)
    {
        var map = new int[bodyCount];
        Array.Fill(map, -1);
        for (int ii = 0; ii < islands.Count; ii++)
            foreach (int b in islands[ii].Bodies)
                if (b >= 0 && b < bodyCount) map[b] = ii;
        return map;
    }

    /// <summary>
    /// Assign each contact pair to its island, keeping input order per
    /// island. Both endpoints must share the island (union-find guarantee).
    /// </summary>
    public static List<List<int>> AssignConstraints(
        List<Island> islands, IList<(int, int)> contactPairs, int[] map)
    {
        var assigned = new List<List<int>>(islands.Count);
        for (int i = 0; i < islands.Count; i++) assigned.Add(new List<int>());
        for (int ci = 0; ci < contactPairs.Count; ci++)
        {
            var (a, b) = contactPairs[ci];
            if (a < 0 || b < 0 || a >= map.Length || b >= map.Length) continue;
            int ia = map[a], ib = map[b];
            if (ia < 0 || ia != ib) continue; // corrupt input: skip loudly elsewhere
            assigned[ia].Add(ci);
        }
        return assigned;
    }
}
