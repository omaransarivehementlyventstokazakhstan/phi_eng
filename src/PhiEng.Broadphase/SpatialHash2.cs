using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Broadphase;

/// <summary>
/// Spatial Hash Grid for broad phase collision detection
/// </summary>
public class SpatialHash2
{
    public float CellSize { get; set; }
    private Dictionary<(int, int), List<int>> Grid { get; set; } = new();
    private int _count;
    private int _clearsSinceCompact;

    public SpatialHash2(float cellSize)
    {
        // Guard against zero/negative/non-finite cell sizes: they would
        // produce unbounded cell ranges (hang) on insert.
        CellSize = float.IsFinite(cellSize) && cellSize > 0.0f ? cellSize : 1.0f;
    }

    /// <summary>
    /// Clear all bodies, keeping every allocation. <c>Dictionary.Clear</c>
    /// would drop each cell's list (with its buffer), forcing a
    /// reallocation per occupied cell on the next insert; clearing each
    /// list in place avoids that. Vacated cells are compacted out every
    /// 256 clears. Mirrors the Rust <c>SpatialHash2::clear</c>.
    /// </summary>
    public void Clear()
    {
        foreach (var list in Grid.Values) list.Clear();
        _count = 0;
        if (++_clearsSinceCompact >= 256)
        {
            _clearsSinceCompact = 0;
            var empty = new List<(int, int)>();
            foreach (var kv in Grid)
                if (kv.Value.Count == 0) empty.Add(kv.Key);
            foreach (var key in empty) Grid.Remove(key);
        }
    }

    public void Insert(int bodyId, Vector2 min, Vector2 max)
    {
        // Non-finite bounds (escaped NaN/inf velocity) cannot map to
        // cells and must not reach the int casts below.
        if (!float.IsFinite(min.X) || !float.IsFinite(min.Y)
            || !float.IsFinite(max.X) || !float.IsFinite(max.Y))
        {
            return;
        }
        _count++;

        // Calculate which cells this body overlaps
        int cellMinX = (int)MathF.Floor(min.X / CellSize);
        int cellMinY = (int)MathF.Floor(min.Y / CellSize);
        int cellMaxX = (int)MathF.Floor(max.X / CellSize);
        int cellMaxY = (int)MathF.Floor(max.Y / CellSize);

        for (int cx = cellMinX; cx <= cellMaxX; cx++)
        {
            for (int cy = cellMinY; cy <= cellMaxY; cy++)
            {
                var key = (cx, cy);
                if (!Grid.ContainsKey(key))
                {
                    Grid[key] = new List<int>();
                }
                Grid[key].Add(bodyId);
            }
        }
    }

    public List<(int, int)> GetPairs()
    {
        var pairs = new List<(int, int)>();
        CollectPairs(pairs);
        return pairs;
    }

    /// <summary>
    /// Append pairs to <paramref name="output"/> without allocating.
    /// Contract: <paramref name="output"/> must be empty. Dedupes by
    /// sorting instead of a HashSet: no hash allocations and
    /// deterministic pair order.
    /// </summary>
    public void CollectPairs(List<(int, int)> output)
    {
        // Sort-based dedupe: measured faster than hash-based for typical
        // pair counts, and keeps output order deterministic.
        foreach (var cellIds in Grid.Values)
        {
            for (int i = 0; i < cellIds.Count; i++)
            {
                for (int j = i + 1; j < cellIds.Count; j++)
                {
                    int a = cellIds[i];
                    int b = cellIds[j];
                    output.Add(a < b ? (a, b) : (b, a));
                }
            }
        }
        output.Sort();
        int kept = 0;
        for (int i = 0; i < output.Count; i++)
        {
            if (i == 0 || output[i] != output[i - 1])
            {
                output[kept] = output[i];
                kept++;
            }
        }
        if (kept < output.Count)
            output.RemoveRange(kept, output.Count - kept);
    }

    public List<int> GetBodiesInCell((int, int) key)
    {
        return Grid.ContainsKey(key) ? Grid[key] : new List<int>();
    }

    public int Count => _count;
    public bool IsEmpty => _count == 0;
}
