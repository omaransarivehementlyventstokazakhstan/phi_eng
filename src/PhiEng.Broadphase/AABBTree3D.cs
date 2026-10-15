using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Broadphase;

/// <summary>
/// 3D AABB tree (flat broadphase over stored AABBs).
/// Mirrors crates/broadphase/src/aabb_tree3d.rs.
/// </summary>
public class AABBTree3D
{
    private readonly List<(int BodyId, AABB3 Box)> _entries = new();

    public void Clear() => _entries.Clear();

    public void Insert(int bodyId, Vector3 min, Vector3 max)
        => _entries.Add((bodyId, new AABB3(min, max)));

    public void Remove(int bodyId)
        => _entries.RemoveAll(e => e.BodyId == bodyId);

    public List<int> Query(AABB3 box)
    {
        var results = new List<int>();
        foreach (var (id, b) in _entries)
            if (b.Intersects(box)) results.Add(id);
        return results;
    }

    public List<(int, int)> GetPairs()
    {
        var pairs = new List<(int, int)>();
        CollectPairs(pairs);
        return pairs;
    }

    /// <summary>
    /// Append pairs to <paramref name="output"/> without allocating.
    /// Contract: <paramref name="output"/> must be empty.
    /// </summary>
    public void CollectPairs(List<(int, int)> output)
    {
        for (int i = 0; i < _entries.Count; i++)
            for (int j = i + 1; j < _entries.Count; j++)
            {
                var (idA, boxA) = _entries[i];
                var (idB, boxB) = _entries[j];
                if (idA != idB && boxA.Intersects(boxB))
                    output.Add((idA, idB));
            }
    }

    public int Count => _entries.Count;
    public bool IsEmpty => _entries.Count == 0;
}
