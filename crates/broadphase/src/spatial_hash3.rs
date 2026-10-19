use glam::Vec3;
use std::collections::HashMap;

/// Cell key for the 3D spatial hash grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellKey3 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// 3D spatial hash grid — the 3D counterpart of [`super::SpatialHash2`].
///
/// Same contract as the 2D grid: pooled per-cell buffers (no steady-state
/// allocation), a bounded insert cost (oversized bodies are paired against
/// the whole scene instead of filling millions of cells), and canonical
/// sorted+deduped pair output so every backend hands the narrowphase the
/// same pair sequence.
#[derive(Debug, Clone)]
pub struct SpatialHash3 {
    /// Cell size for the hash grid.
    pub cell_size: f32,
    /// Map from cell key to body IDs.
    grid: HashMap<CellKey3, Vec<usize>>,
    /// Number of inserted bodies.
    count: usize,
    /// Steps since vacated cells were compacted.
    clears_since_compact: u32,
    /// Recycled cell buffers.
    pool: Vec<Vec<usize>>,
    /// Every inserted body id this frame (insertion order).
    all: Vec<usize>,
    /// Bodies too large for the grid; paired against the whole scene.
    oversized: Vec<usize>,
}

impl SpatialHash3 {
    /// Maximum cells a single body may occupy. A body whose AABB exceeds
    /// this is tracked in the oversized list and paired against the whole
    /// scene instead — correctness is preserved (it still collides with
    /// everything) while insert cost stays bounded.
    pub const MAX_CELLS_PER_BODY: usize = 4096;
    /// Cell-index magnitude beyond which a coordinate is treated as
    /// oversized (guards the `as i32` casts from saturating and
    /// producing degenerate ranges for extreme-but-finite positions).
    const MAX_CELL_INDEX: f32 = 2_000_000_000.0;

    pub fn new(cell_size: f32) -> Self {
        // Guard against zero/negative/infinite cell sizes: they would
        // produce unbounded cell ranges (a hang) on insert.
        let cell_size = if cell_size.is_finite() && cell_size > 0.0 {
            cell_size
        } else {
            1.0
        };
        Self {
            cell_size,
            grid: HashMap::new(),
            count: 0,
            clears_since_compact: 0,
            pool: Vec::new(),
            all: Vec::new(),
            oversized: Vec::new(),
        }
    }

    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// Vacated cells are compacted out every 256 clears so roaming worlds
    /// cannot accumulate them unboundedly (amortized cost ~zero).
    pub fn clear(&mut self) {
        for ids in self.grid.values_mut() {
            ids.clear();
        }
        self.count = 0;
        self.all.clear();
        self.oversized.clear();
        self.clears_since_compact += 1;
        if self.clears_since_compact >= 256 {
            self.clears_since_compact = 0;
            let grid = &mut self.grid;
            let pool = &mut self.pool;
            grid.retain(|_, ids| {
                if ids.is_empty() {
                    pool.push(std::mem::take(ids));
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Insert a body into the grid.
    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        // Non-finite bounds (escaped NaN/inf velocity) cannot map to
        // cells — and must not reach the `as i32` casts below, which
        // saturate to i32::MIN/MAX and would hang the cell loops.
        if !min.x.is_finite()
            || !min.y.is_finite()
            || !min.z.is_finite()
            || !max.x.is_finite()
            || !max.y.is_finite()
            || !max.z.is_finite()
        {
            return;
        }
        self.count += 1;
        self.all.push(body_id);

        // Cells spanned per axis, computed in float to avoid overflow on
        // extreme-but-finite coordinates.
        let span_x = (max.x - min.x) / self.cell_size;
        let span_y = (max.y - min.y) / self.cell_size;
        let span_z = (max.z - min.z) / self.cell_size;
        let cells_f = (span_x + 1.0) * (span_y + 1.0) * (span_z + 1.0);
        let origin_f = (min.x.abs().max(max.x.abs()) / self.cell_size)
            .max(min.y.abs().max(max.y.abs()) / self.cell_size)
            .max(min.z.abs().max(max.z.abs()) / self.cell_size);

        // Oversized (too many cells, or coords beyond safe cell indices):
        // skip the grid and pair against the whole scene.
        if cells_f > Self::MAX_CELLS_PER_BODY as f32 || origin_f > Self::MAX_CELL_INDEX {
            self.oversized.push(body_id);
            return;
        }

        // Floor (not truncation) so negative coords map correctly.
        let cell_min_x = (min.x / self.cell_size).floor() as i32;
        let cell_min_y = (min.y / self.cell_size).floor() as i32;
        let cell_min_z = (min.z / self.cell_size).floor() as i32;
        let cell_max_x = (max.x / self.cell_size).floor() as i32;
        let cell_max_y = (max.y / self.cell_size).floor() as i32;
        let cell_max_z = (max.z / self.cell_size).floor() as i32;

        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                for cz in cell_min_z..=cell_max_z {
                    let key = CellKey3::new(cx, cy, cz);
                    self.grid
                        .entry(key)
                        .or_insert_with(|| self.pool.pop().unwrap_or_default())
                        .push(body_id);
                }
            }
        }
    }

    /// Append every overlapping pair to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Output is sorted and deduped, so every backend produces the same
    /// pair sequence.
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for cell_ids in self.grid.values() {
            for i in 0..cell_ids.len() {
                for j in (i + 1)..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        // Oversized bodies are not in the grid: pair each against every
        // body (their huge AABBs overlap anything in range).
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

    /// Get the number of bodies.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for SpatialHash3 {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash3_pairs() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        hash.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        hash.insert(2, Vec3::new(10.0, 10.0, 10.0), Vec3::new(12.0, 12.0, 12.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_spatial_hash3_no_collision() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash3_negative_coords() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(
            0,
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(-9.0, -9.0, -9.0),
        );
        hash.insert(1, Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0));
        hash.insert(2, Vec3::new(-9.5, -9.5, -9.5), Vec3::new(-8.0, -8.0, -8.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 2)]);
    }

    #[test]
    fn test_spatial_hash3_rejects_non_finite() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::ZERO, Vec3::new(f32::INFINITY, 1.0, 1.0));
        assert!(hash.is_empty());
        assert_eq!(hash.len(), 0);
    }

    #[test]
    fn test_spatial_hash3_oversized_pairs_with_all() {
        // A body too large for the grid must not hang insert, and must
        // still pair with every other body.
        let mut hash = SpatialHash3::new(1.0);
        let start = std::time::Instant::now();
        hash.insert(0, Vec3::splat(-1000.0), Vec3::splat(1000.0));
        hash.insert(1, Vec3::splat(500.0), Vec3::splat(501.0));
        assert!(
            start.elapsed().as_millis() < 500,
            "oversized insert must be fast, took {:?}",
            start.elapsed()
        );

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(
            pairs.contains(&(0, 1)),
            "oversized body must pair with the small body"
        );
    }

    #[test]
    fn test_spatial_hash3_clear_reuses_buffers() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        hash.clear();
        assert!(hash.is_empty());

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash3_has_no_false_negatives() {
        use crate::brute_force3::BruteForce3;
        let mut hash = SpatialHash3::new(1.0);
        let mut bf = BruteForce3::new();
        // Deterministic pseudo-cluster: 30 bodies in 4 tight groups.
        let mut bounds: Vec<(Vec3, Vec3)> = Vec::new();
        for i in 0..30usize {
            let g = (i % 4) as f32;
            let k = (i / 4) as f32;
            let min = Vec3::new(g * 40.0 + k * 0.5, k * 0.5, k * 0.25);
            let max = min + Vec3::splat(2.0);
            hash.insert(i, min, max);
            bf.insert(i, min, max);
            bounds.push((min, max));
        }

        let mut grid_pairs = Vec::new();
        hash.collect_pairs(&mut grid_pairs);
        let mut exact_pairs = Vec::new();
        bf.collect_pairs(&mut exact_pairs);

        // The grid reports *cell* overlaps, so it may emit extra pairs
        // (the narrowphase rejects them). What it must never do is miss a
        // genuinely overlapping pair — that would be a false negative.
        for pair in &exact_pairs {
            assert!(
                grid_pairs.contains(pair),
                "grid missed an overlapping pair {pair:?}"
            );
        }
        // Every extra pair the grid adds must be a false positive: the
        // two AABBs do not actually overlap.
        for pair in &grid_pairs {
            if exact_pairs.contains(pair) {
                continue;
            }
            let (a, b) = *pair;
            let (amin, amax) = bounds[a];
            let (bmin, bmax) = bounds[b];
            let overlaps = amax.x >= bmin.x
                && amin.x <= bmax.x
                && amax.y >= bmin.y
                && amin.y <= bmax.y
                && amax.z >= bmin.z
                && amin.z <= bmax.z;
            assert!(!overlaps, "grid emitted {pair:?} as a non-overlapping pair");
        }
    }
}
