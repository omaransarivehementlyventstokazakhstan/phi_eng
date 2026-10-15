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
#[derive(Debug)]
pub struct SpatialHash2 {
    /// Cell size for the hash grid
    pub cell_size: f32,
    /// Map from cell key to body IDs
    grid: HashMap<CellKey, Vec<usize>>,
    /// Number of inserted bodies (the grid owns no AABB copies; the
    /// world rebuilds it from live body positions every step).
    count: usize,
    /// Steps since vacated cells were compacted (see [`SpatialHash2::clear`]).
    clears_since_compact: u32,
}

impl SpatialHash2 {
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
            clears_since_compact: 0,
        }
    }

    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// `HashMap::clear` would drop each cell's `Vec` (with its buffer),
    /// forcing a reallocation per occupied cell on the very next insert.
    /// Instead each cell list is cleared in place. Vacated cells leave
    /// empty entries behind; they are compacted out every 256 clears so
    /// roaming worlds cannot accumulate them unboundedly (amortized cost
    /// ~zero).
    pub fn clear(&mut self) {
        for ids in self.grid.values_mut() {
            ids.clear();
        }
        self.count = 0;
        self.clears_since_compact += 1;
        if self.clears_since_compact >= 256 {
            self.clears_since_compact = 0;
            self.grid.retain(|_, ids| !ids.is_empty());
        }
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

        // Calculate which cells this body overlaps.
        // Use floor (not truncation) so negative coords map correctly,
        // and do NOT clamp to 0 — CellKey supports negative cells.
        let cell_min_x = (min.x / self.cell_size).floor() as i32;
        let cell_min_y = (min.y / self.cell_size).floor() as i32;
        let cell_max_x = (max.x / self.cell_size).floor() as i32;
        let cell_max_y = (max.y / self.cell_size).floor() as i32;

        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                let key = CellKey::new(cx, cy);
                self.grid.entry(key).or_default().push(body_id);
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
        out.sort_unstable();
        out.dedup();
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
    fn test_spatial_hash_len_counts_inserts() {
        let mut hash = SpatialHash2::new(1.0);
        assert_eq!(hash.len(), 0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));
        assert_eq!(hash.len(), 2);
    }
}
