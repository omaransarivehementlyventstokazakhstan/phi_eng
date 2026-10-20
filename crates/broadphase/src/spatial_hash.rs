use glam::Vec2;
use std::collections::HashMap;

/// Cell key for the spatial hash grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey {
    pub x: i32,
    pub y: i32,
}

impl CellKey {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Spatial Hash Grid for broad phase collision detection
#[derive(Debug, Clone)]
pub struct SpatialHash2 {
    /// Cell size for the hash grid
    pub cell_size: f32,
    /// Map from cell key to body IDs
    grid: HashMap<CellKey, Vec<usize>>,
    /// Number of inserted bodies (the grid owns no AABB copies; the
    /// world rebuilds it from live body positions every step).
    count: usize,
    /// Recycled cell buffers. Vacated cells return their `Vec` here instead
    /// of freeing it; new cells draw from it, so re-appearing cells never
    /// re-allocate. Internal (the public API is unchanged).
    pool: Vec<Vec<usize>>,
    /// Every inserted body id this frame (insertion order). Used to pair
    /// oversized bodies against the whole scene.
    all: Vec<usize>,
    /// Bodies whose AABB spans more cells than [`Self::MAX_CELLS_PER_BODY`].
    /// Such bodies are NOT written into the grid (a 1000-unit body at a
    /// 1.0 cell size would touch millions of cells and hang the step);
    /// instead they are paired against every body in `collect_pairs`.
    oversized: Vec<usize>,
}

impl SpatialHash2 {
    /// Maximum cells a single body may occupy in the grid. A body whose
    /// AABB exceeds this is tracked in the oversized list and paired
    /// against the whole scene instead — correctness is preserved
    /// (it still collides with everything) while keeping insert cost
    /// bounded. Sized so ordinary scenes never trigger it (e.g. a
    /// 64x64-cell, 90x90-unit body at cell size 1.0).
    pub const MAX_CELLS_PER_BODY: usize = 4096;
    /// Cell-index magnitude beyond which a coordinate is treated as
    /// oversized (guards the `as i32` casts from saturating to i32::MIN/MAX
    /// and producing degenerate ranges for extreme-but-finite positions).
    const MAX_CELL_INDEX: f32 = 2_000_000_000.0;

    pub fn new(cell_size: f32) -> Self {
        // Guard against zero/negative/infinite cell sizes: they would produce
        // unbounded cell ranges (hang) on insert.
        let cell_size = if cell_size.is_finite() && cell_size > 0.0 {
            cell_size
        } else {
            1.0
        };
        Self {
            cell_size,
            grid: HashMap::new(),
            count: 0,
            pool: Vec::new(),
            all: Vec::new(),
            oversized: Vec::new(),
        }
    }

    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// `HashMap::clear` would drop each cell's `Vec` (with its buffer),
    /// forcing a reallocation per occupied cell on the very next insert.
    /// Instead each cell's buffer is moved into the pool, where the next
    /// insert picks it up again: nothing is allocated, and the map holds
    /// only the cells that are actually in use.
    ///
    /// Keeping the vacated cells around instead (the old behaviour) grows
    /// the map without bound between compactions, and a large map turns
    /// every lookup into a cache miss: the insert pass costs ~650 ns per
    /// body in a settled 1500-body pile, essentially all of it in the
    /// hash lookup. Vacating the cells keeps the map at the size of the
    /// current frame and the lookup back in cache.
    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// One pass over the map: every cell is emptied in place, and the
    /// cells that are now empty are dropped and their buffers recycled
    /// through the pool.
    ///
    /// Dropping the empty cells is what keeps the map small. The cells
    /// of a scene that sweeps its bodies across many positions pile up
    /// otherwise - a settled 1500-body 2D pile held 3 327 878 cells for
    /// 1501 bodies - and every lookup in a hash table with millions of
    /// entries is a cache miss: the insert pass cost ~650 ns per body.
    /// The cells that are still in use are kept, so the next insert
    /// finds them instead of recreating them: no per-frame churn.
    pub fn clear(&mut self) {
        let pool = &mut self.pool;
        self.grid.retain(|_, ids| {
            if ids.is_empty() {
                pool.push(std::mem::take(ids));
                false
            } else {
                ids.clear();
                true
            }
        });
        self.count = 0;
        self.all.clear();
        self.oversized.clear();
    }

    /// Insert a body into the grid
    pub fn insert(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        // Non-finite bounds (escaped NaN/inf velocity) cannot map to
        // cells — and must not reach the `as i32` casts below, which
        // saturate to i32::MIN/MAX and would hang the cell loops.
        if !min.x.is_finite() || !min.y.is_finite() || !max.x.is_finite() || !max.y.is_finite() {
            return;
        }
        self.count += 1;
        self.all.push(body_id);

        // Cells spanned per axis, computed in float to avoid overflow on
        // extreme-but-finite coordinates (integer casts below saturate).
        let span_x = (max.x - min.x) / self.cell_size;
        let span_y = (max.y - min.y) / self.cell_size;
        let cells_f = (span_x + 1.0) * (span_y + 1.0);
        let origin_f = (min.x.abs().max(max.x.abs()) / self.cell_size)
            .max(min.y.abs().max(max.y.abs()) / self.cell_size);

        // Oversized (too many cells, or coords beyond safe cell indices):
        // skip the grid and pair against the whole scene in `collect_pairs`.
        if cells_f > Self::MAX_CELLS_PER_BODY as f32 || origin_f > Self::MAX_CELL_INDEX {
            self.oversized.push(body_id);
            return;
        }

        // Calculate which cells this body overlaps.
        // Use floor (not truncation) so negative coords map correctly,
        // and do NOT clamp to 0 �?" CellKey supports negative cells.
        let cell_min_x = (min.x / self.cell_size).floor() as i32;
        let cell_min_y = (min.y / self.cell_size).floor() as i32;
        let cell_max_x = (max.x / self.cell_size).floor() as i32;
        let cell_max_y = (max.y / self.cell_size).floor() as i32;

        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                let key = CellKey::new(cx, cy);
                self.grid
                    .entry(key)
                    .or_insert_with(|| self.pool.pop().unwrap_or_default())
                    .push(body_id);
            }
        }
    }

    /// Get potential collision pairs
    pub fn get_pairs(&mut self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        self.collect_pairs(&mut pairs);
        pairs
    }

    /// Append potential collision pairs to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Dedupes by sorting (deterministic pair order); measured faster
    /// than hash-based dedupe for typical pair counts.
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for cell_ids in self.grid.values() {
            for i in 0..cell_ids.len() {
                for j in i + 1..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    // Canonical pair key.
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        // Oversized bodies are not in the grid: pair each against every
        // body (their huge AABBs overlap anything in range). Determinism
        // holds — `all`/`oversized` are insertion-ordered and the final
        // sort+dedup canonicalizes.
        for &o in &self.oversized {
            for &b in &self.all {
                if o != b {
                    out.push(if o < b { (o, b) } else { (b, o) });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// How many cells the grid currently holds.
    ///
    /// A grid that holds many more cells than the scene has bodies is
    /// bloated: the cells of a scene that swept its bodies across many
    /// positions are still around, and every lookup in a large hash
    /// table is a cache miss.
    pub fn cell_count(&self) -> usize {
        self.grid.len()
    }

    /// Get bodies in a cell
    pub fn get_bodies_in_cell(&self, key: &CellKey) -> Vec<usize> {
        self.grid.get(key).cloned().unwrap_or_default()
    }

    /// Get the number of bodies
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for SpatialHash2 {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash_insert_and_get_pairs() {
        let mut hash = SpatialHash2::new(1.0);

        // Insert two overlapping bodies
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let pairs = hash.get_pairs();
        assert!(pairs.contains(&(0, 1)));
    }

    #[test]
    fn test_spatial_hash_no_collision() {
        let mut hash = SpatialHash2::new(1.0);

        // Insert two non-overlapping bodies
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));

        let pairs = hash.get_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash_clear() {
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.clear();

        assert!(hash.is_empty());
    }

    #[test]
    fn test_oversized_body_pairs_with_all_and_is_fast() {
        // A body too large for the grid (radius 1000 -> ~4M cells) must
        // not hang insert, and must still pair with every other body.
        let mut hash = SpatialHash2::new(1.0);
        let t = std::time::Instant::now();
        hash.insert(0, Vec2::new(-1000.0, -1000.0), Vec2::new(1000.0, 1000.0));
        // A far-away small body (no cell overlap with the huge AABB's
        // center region would be irrelevant: the huge body overlaps it).
        hash.insert(1, Vec2::new(500.0, 500.0), Vec2::new(501.0, 501.0));
        assert!(
            t.elapsed().as_millis() < 500,
            "oversized insert must be fast, took {:?}",
            t.elapsed()
        );

        let pairs = hash.get_pairs();
        assert!(
            pairs.contains(&(0, 1)),
            "oversized body must pair with the small body"
        );
    }

    #[test]
    fn test_normal_body_stays_in_grid() {
        // A normal-sized body never takes the oversized path: two
        // overlapping small bodies still pair via cells.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert!(hash.oversized.is_empty());
        assert!(hash.get_pairs().contains(&(0, 1)));
    }

    #[test]
    fn test_spatial_hash_negative_coords() {
        let mut hash = SpatialHash2::new(1.0);

        // Floor at y=-1.5..-0.5 and a ball just above it must share a cell
        // only when actually overlapping — and negative-x bodies must NOT
        // all collapse into cell 0.
        hash.insert(0, Vec2::new(-10.0, -1.5), Vec2::new(10.0, -0.5));
        hash.insert(1, Vec2::new(-0.3, -0.4), Vec2::new(0.3, 0.2));
        let pairs = hash.get_pairs();
        assert!(pairs.contains(&(0, 1)));

        // Far-apart negative bodies must not pair.
        let mut hash2 = SpatialHash2::new(1.0);
        hash2.insert(0, Vec2::new(-10.0, -10.0), Vec2::new(-9.0, -9.0));
        hash2.insert(1, Vec2::new(-1.0, -1.0), Vec2::new(0.0, 0.0));
        assert!(hash2.get_pairs().is_empty());
    }

    #[test]
    fn test_spatial_hash_rejects_non_finite_bounds() {
        let mut hash = SpatialHash2::new(1.0);
        // NaN/inf bounds must be dropped, never hang the cell loops.
        hash.insert(0, Vec2::new(f32::NAN, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(0.0, 0.0), Vec2::new(f32::INFINITY, 1.0));
        assert!(hash.get_pairs().is_empty());
        assert!(hash.is_empty());
        assert_eq!(hash.len(), 0);
    }

    #[test]
    fn test_spatial_hash_clear_reuses_buffers_without_leaking_ids() {
        // A cleared cell's buffer goes back into the pool and is handed
        // out again on the next insert. If the recycled buffer still held
        // the previous frame's ids, the cell would report pairs that do
        // not exist (and ids that are not in the scene at all).
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(0.5, 0.5), Vec2::new(2.5, 2.5));
        assert_eq!(hash.get_pairs().len(), 1);

        hash.clear();

        // A different body in the same cell must not inherit the old
        // ids.
        hash.insert(7, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let pairs = hash.get_pairs();
        assert!(
            pairs.is_empty(),
            "a recycled cell must not carry the previous frame's ids: {pairs:?}"
        );
        assert_eq!(hash.len(), 1);

        // And the recycled buffer still works for a second body.
        hash.insert(8, Vec2::new(0.5, 0.5), Vec2::new(2.5, 2.5));
        assert_eq!(hash.get_pairs(), vec![(7, 8)]);
    }

    #[test]
    fn test_spatial_hash_clear_empties_every_cell() {
        // After a clear every cell must be empty: either the cell was
        // dropped (a bloated table) or its list was cleared in place (a
        // small table). No cell may still hold ids.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(100.0, 100.0), Vec2::new(101.0, 101.0));
        hash.clear();
        for (key, ids) in hash.grid.iter() {
            assert!(
                ids.is_empty(),
                "cell {key:?} still holds ids after a clear: {ids:?}"
            );
        }
        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash_reinsert_after_clear_keeps_pairs() {
        // The full cycle: clear, re-insert, and the pairs must be the
        // same as they were before the clear.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let before = hash.get_pairs();
        hash.clear();
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert_eq!(hash.get_pairs(), before);
    }

    #[test]
    fn test_spatial_hash_len_counts_inserts() {
        let mut hash = SpatialHash2::new(1.0);
        assert_eq!(hash.len(), 0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));
        assert_eq!(hash.len(), 2);
    }
}
