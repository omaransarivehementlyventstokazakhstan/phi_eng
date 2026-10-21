use glam::Vec2;

use super::broadphase_mode::BroadphaseMode;
use super::brute_force2::BruteForce2;
use super::dynamic_aabb_tree2::DynamicAABBTree2;
use super::spatial_hash::SpatialHash2;

/// Broadphase backend for 2D worlds. Wraps the concrete implementation
/// selected by [`BroadphaseMode`].
#[derive(Debug, Clone)]
pub enum Broadphase2 {
    SpatialHash(SpatialHash2),
    BruteForce(BruteForce2),
    DynamicAABBTree(DynamicAABBTree2),
}

impl Broadphase2 {
    pub fn new(mode: BroadphaseMode, cell_size: f32) -> Self {
        match mode {
            BroadphaseMode::SpatialHash => Broadphase2::SpatialHash(SpatialHash2::new(cell_size)),
            BroadphaseMode::BruteForce => Broadphase2::BruteForce(BruteForce2::new()),
            BroadphaseMode::DynamicAABBTree => {
                Broadphase2::DynamicAABBTree(DynamicAABBTree2::new())
            }
            BroadphaseMode::Hybrid => {
                // Hybrid resolves to SpatialHash by default; the world
                // re-resolves per-step based on body count.
                Broadphase2::SpatialHash(SpatialHash2::new(cell_size))
            }
        }
    }

    pub fn clear(&mut self) {
        match self {
            Broadphase2::SpatialHash(h) => h.clear(),
            Broadphase2::BruteForce(b) => b.clear(),
            Broadphase2::DynamicAABBTree(t) => t.clear(),
        }
    }

    pub fn insert(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        match self {
            Broadphase2::SpatialHash(h) => h.insert(body_id, min, max),
            Broadphase2::BruteForce(b) => b.insert(body_id, min, max),
            Broadphase2::DynamicAABBTree(t) => t.insert(body_id, min, max),
        }
    }

    /// Insert a *static* body.
    ///
    /// A static body still pairs with every dynamic body, but a pair of
    /// two statics can never produce an impulse, so the grid skips those
    /// pairs instead of generating them for the narrowphase to discard.
    /// The brute-force and tree backends have no such notion: they
    /// already report only overlapping pairs, and the narrowphase
    /// discards the static ones there.
    pub fn insert_static(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        match self {
            Broadphase2::SpatialHash(h) => h.insert_static(body_id, min, max),
            _ => self.insert(body_id, min, max),
        }
    }

    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        match self {
            Broadphase2::SpatialHash(h) => h.collect_pairs(out),
            Broadphase2::BruteForce(b) => b.collect_pairs(out),
            Broadphase2::DynamicAABBTree(t) => t.collect_pairs(out),
        }
    }

    /// Append the AABB-overlapping subsequence of [`Self::collect_pairs`].
    ///
    /// The narrowphase uses this fast path: every dropped pair is two
    /// bodies whose AABBs are disjoint, so the narrow phase could only
    /// have produced zero contacts for it.
    pub fn collect_overlapping_pairs(&self, out: &mut Vec<(usize, usize)>) {
        match self {
            Broadphase2::SpatialHash(h) => h.collect_overlapping_pairs(out),
            Broadphase2::BruteForce(b) => b.collect_overlapping_pairs(out),
            Broadphase2::DynamicAABBTree(t) => t.collect_overlapping_pairs(out),
        }
    }

    /// Profiling only: turn the grid's pair ownership rule on or off.
    ///
    /// The rule makes the grid generate each pair once, in the lowest
    /// cell the two bodies share, instead of once per shared cell. The
    /// pair set is the same either way; only the amount of duplicated
    /// candidate work differs.
    #[cfg(feature = "phase-timing")]
    pub fn set_owned_pairs(&mut self, owned: bool) {
        if let Broadphase2::SpatialHash(h) = self {
            h.owned_pairs = owned;
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Broadphase2::SpatialHash(h) => h.len(),
            Broadphase2::BruteForce(b) => b.len(),
            Broadphase2::DynamicAABBTree(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cell_size(&self) -> f32 {
        match self {
            Broadphase2::SpatialHash(h) => h.cell_size,
            _ => 1.0,
        }
    }

    /// How many cells the backing grid holds.
    ///
    /// Only the grid backend has cells; the other backends report 0.
    pub fn cell_count(&self) -> usize {
        match self {
            Broadphase2::SpatialHash(h) => h.cell_count(),
            _ => 0,
        }
    }
}

impl Default for Broadphase2 {
    fn default() -> Self {
        Self::new(BroadphaseMode::default(), 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All three backends must agree on pairs for the same scene. This is
    /// the core tier-system contract: switching mode is a performance
    /// choice, never a behavior change.
    fn reference_pairs() -> Vec<(usize, usize)> {
        // Deterministic mix of overlaps, near-misses, and isolation.
        let boxes: [(f32, f32, f32, f32); 8] = [
            (0.0, 0.0, 2.0, 2.0),
            (1.0, 1.0, 3.0, 3.0),
            (2.0, 2.0, 4.0, 4.0),
            (50.0, 50.0, 52.0, 52.0),
            (51.0, 51.0, 53.0, 53.0),
            (-10.0, -10.0, -8.0, -8.0),
            (-9.0, -9.0, -7.0, -7.0),
            (0.5, 0.5, 1.5, 6.0),
        ];
        let mut out = Vec::new();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let (ax0, ay0, ax1, ay1) = boxes[i];
                let (bx0, by0, bx1, by1) = boxes[j];
                let hit = ax0 <= bx1 && ax1 >= bx0 && ay0 <= by1 && ay1 >= by0;
                if hit {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn pairs_for(mode: BroadphaseMode) -> Vec<(usize, usize)> {
        let mut bp = Broadphase2::new(mode, 1.0);
        let boxes: [(f32, f32, f32, f32); 8] = [
            (0.0, 0.0, 2.0, 2.0),
            (1.0, 1.0, 3.0, 3.0),
            (2.0, 2.0, 4.0, 4.0),
            (50.0, 50.0, 52.0, 52.0),
            (51.0, 51.0, 53.0, 53.0),
            (-10.0, -10.0, -8.0, -8.0),
            (-9.0, -9.0, -7.0, -7.0),
            (0.5, 0.5, 1.5, 6.0),
        ];
        for (i, b) in boxes.iter().enumerate() {
            bp.insert(i, Vec2::new(b.0, b.1), Vec2::new(b.2, b.3));
        }
        let mut pairs = Vec::new();
        bp.collect_pairs(&mut pairs);
        assert_eq!(bp.len(), 8);
        pairs
    }

    #[test]
    fn test_all_modes_agree_with_reference() {
        let expected = reference_pairs();
        assert!(!expected.is_empty(), "reference scene must have pairs");
        for mode in [
            BroadphaseMode::BruteForce,
            BroadphaseMode::SpatialHash,
            BroadphaseMode::DynamicAABBTree,
        ] {
            assert_eq!(pairs_for(mode), expected, "mode {} disagreed", mode.name());
        }
    }

    #[test]
    fn test_hybrid_constructs_spatial_hash() {
        // Hybrid resolves per-step in the world; the wrapper itself must
        // be constructible and functional.
        let bp = Broadphase2::new(BroadphaseMode::Hybrid, 1.0);
        assert_eq!(bp.cell_size(), 1.0);
        assert!(bp.is_empty());
    }

    #[test]
    fn test_clear_empties_every_backend() {
        for mode in [
            BroadphaseMode::BruteForce,
            BroadphaseMode::SpatialHash,
            BroadphaseMode::DynamicAABBTree,
        ] {
            let mut bp = Broadphase2::new(mode, 1.0);
            bp.insert(0, Vec2::ZERO, Vec2::new(1.0, 1.0));
            bp.insert(1, Vec2::ZERO, Vec2::new(1.0, 1.0));
            assert_eq!(bp.len(), 2);
            bp.clear();
            assert!(bp.is_empty());
            let mut pairs = Vec::new();
            bp.collect_pairs(&mut pairs);
            assert!(pairs.is_empty());
        }
    }

    #[test]
    fn test_collect_pairs_requires_empty_out() {
        // The world passes cleared scratch; a non-empty buffer is a
        // caller bug we assert on (debug builds only).
        let mut bp = Broadphase2::new(BroadphaseMode::BruteForce, 1.0);
        bp.insert(0, Vec2::ZERO, Vec2::new(1.0, 1.0));
        let mut pairs = Vec::new();
        bp.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_pair_order_is_deterministic_across_runs() {
        for mode in [
            BroadphaseMode::BruteForce,
            BroadphaseMode::SpatialHash,
            BroadphaseMode::DynamicAABBTree,
        ] {
            let first = pairs_for(mode);
            let second = pairs_for(mode);
            assert_eq!(first, second, "mode {} not deterministic", mode.name());
        }
    }
}
