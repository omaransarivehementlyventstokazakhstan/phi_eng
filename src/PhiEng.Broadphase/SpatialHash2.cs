using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Broadphase;

/// <summary>
/// Broad-phase backend selection. Mirrors the Rust
/// <c>BroadphaseMode</c> in <c>crates/broadphase/src/broadphase_mode.rs</c>;
/// the discriminant values are part of the FFI contract and must stay in
/// sync (see <c>phieng_world_set_broadphase_mode</c>).
/// </summary>
public enum BroadphaseMode
{
    /// <summary>All-pairs test. Fastest for tiny scenes (&lt; ~50 bodies).</summary>
    BruteForce = 0,

    /// <summary>Uniform spatial hash grid. Default; best for 50–5000 bodies.</summary>
    SpatialHash = 1,

    /// <summary>Balanced AABB tree. Best for 5000+ bodies and clustered layouts.</summary>
    DynamicAABBTree = 2,

    /// <summary>Auto-select from live body count (see <c>AutoSelect</c>).</summary>
    Hybrid = 3,
}

/// <summary>Auto-selection thresholds for <see cref="BroadphaseMode.Hybrid"/>.</summary>
public static class BroadphaseModeExtensions
{
    /// <summary>Body count below which <see cref="BroadphaseMode.Hybrid"/> picks all-pairs.</summary>
    public const int BruteForceMaxBodies = 50;

    /// <summary>Body count below which <see cref="BroadphaseMode.Hybrid"/> picks the grid.</summary>
    public const int SpatialHashMaxBodies = 5000;

    /// <summary>
    /// Resolve a <see cref="BroadphaseMode.Hybrid"/> request to a concrete
    /// backend from a live body count. Mirrors
    /// <c>BroadphaseMode::auto_select</c>; never returns
    /// <see cref="BroadphaseMode.Hybrid"/> itself.
    /// </summary>
    public static BroadphaseMode AutoSelect(this BroadphaseMode mode, int bodyCount)
    {
        if (mode != BroadphaseMode.Hybrid)
        {
            return mode;
        }
        if (bodyCount < BruteForceMaxBodies)
        {
            return BroadphaseMode.BruteForce;
        }
        return bodyCount < SpatialHashMaxBodies
            ? BroadphaseMode.SpatialHash
            : BroadphaseMode.DynamicAABBTree;
    }
}

/// <summary>
/// Spatial Hash Grid for broad phase collision detection
/// </summary>
public class SpatialHash2
{
    public float CellSize { get; set; }
    private Dictionary<(int, int), List<int>> Grid { get; set; } = new();
    private int _count;
    private int _clearsSinceCompact;
    // Every inserted body id this frame (insertion order), and bodies whose
    // AABB spans too many cells to grid. Mirrors Rust SpatialHash2.
    private readonly List<int> _all = new();
    private readonly List<int> _oversized = new();

    /// <summary>
    /// Maximum cells a single body may occupy. A body whose AABB exceeds
    /// this is tracked in the oversized list and paired against the whole
    /// scene instead of being written into the grid (a 1000-unit body at a
    /// 1.0 cell size would touch millions of cells and hang the step).
    /// Sized so ordinary scenes never trigger it.
    /// </summary>
    public const int MaxCellsPerBody = 4096;
    /// <summary>Cell-index magnitude beyond which a coordinate is treated
    /// as oversized (guards the int casts from saturating).</summary>
    private const float MaxCellIndex = 2_000_000_000.0f;

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
        _all.Clear();
        _oversized.Clear();
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
        _all.Add(bodyId);

        // Cells spanned, computed in float to avoid overflow on
        // extreme-but-finite coordinates (int casts below saturate).
        float spanX = (max.X - min.X) / CellSize;
        float spanY = (max.Y - min.Y) / CellSize;
        float cellsF = (spanX + 1.0f) * (spanY + 1.0f);
        float originF = System.MathF.Max(
            System.MathF.Max(System.MathF.Abs(min.X), System.MathF.Abs(max.X)),
            System.MathF.Max(System.MathF.Abs(min.Y), System.MathF.Abs(max.Y))) / CellSize;

        // Oversized (too many cells, or coords beyond safe cell indices):
        // skip the grid, pair against the whole scene in CollectPairs.
        if (cellsF > MaxCellsPerBody || originF > MaxCellIndex)
        {
            _oversized.Add(bodyId);
            return;
        }

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
        // Oversized bodies are not in the grid: pair each against every
        // body. Determinism holds (_all/_oversized are insertion-ordered,
        // and the final sort+dedupe canonicalizes).
        foreach (int o in _oversized)
        {
            foreach (int b in _all)
            {
                if (o != b) output.Add(o < b ? (o, b) : (b, o));
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
