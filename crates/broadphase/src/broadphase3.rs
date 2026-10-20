use glam::Vec3;

use super::broadphase_mode::BroadphaseMode;
use super::brute_force3::BruteForce3;
use super::dynamic_aabb_tree3::DynamicAABBTree3;
use super::spatial_hash3::SpatialHash3;

/// Broadphase backend for 3D worlds. Wraps the concrete implementation
/// selected by [`BroadphaseMode`].
#[derive(Debug, Clone)]
pub enum Broadphase3 {
    BruteForce(BruteForce3),
    SpatialHash(SpatialHash3),
    DynamicAABBTree(DynamicAABBTree3),
}

impl Broadphase3 {
    pub fn new(mode: BroadphaseMode) -> Self {
        match mode {
            BroadphaseMode::BruteForce => Broadphase3::BruteForce(BruteForce3::new()),
            BroadphaseMode::SpatialHash | BroadphaseMode::Hybrid => {
                Broadphase3::SpatialHash(SpatialHash3::new(1.0))
            }
            BroadphaseMode::DynamicAABBTree => {
                Broadphase3::DynamicAABBTree(DynamicAABBTree3::new())
            }
        }
    }

    pub fn clear(&mut self) {
        match self {
            Broadphase3::BruteForce(b) => b.clear(),
            Broadphase3::SpatialHash(h) => h.clear(),
            Broadphase3::DynamicAABBTree(t) => t.clear(),
        }
    }

    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        match self {
            Broadphase3::BruteForce(b) => b.insert(body_id, min, max),
            Broadphase3::SpatialHash(h) => h.insert(body_id, min, max),
            Broadphase3::DynamicAABBTree(t) => t.insert(body_id, min, max),
        }
    }

    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        match self {
            Broadphase3::BruteForce(b) => b.collect_pairs(out),
            Broadphase3::SpatialHash(h) => h.collect_pairs(out),
            Broadphase3::DynamicAABBTree(t) => t.collect_pairs(out),
        }
    }

    /// Append the AABB-overlapping subsequence of [`Self::collect_pairs`].
    ///
    /// The narrowphase uses this fast path: every dropped pair is two
    /// bodies whose AABBs are disjoint, so the narrow phase could only
    /// have produced zero contacts for it.
    pub fn collect_overlapping_pairs(&self, out: &mut Vec<(usize, usize)>) {
        match self {
            Broadphase3::BruteForce(b) => b.collect_overlapping_pairs(out),
            Broadphase3::SpatialHash(h) => h.collect_overlapping_pairs(out),
            Broadphase3::DynamicAABBTree(t) => t.collect_overlapping_pairs(out),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Broadphase3::BruteForce(b) => b.len(),
            Broadphase3::SpatialHash(h) => h.len(),
            Broadphase3::DynamicAABBTree(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many cells the backing grid holds.
    ///
    /// Only the grid backend has cells; the other backends report 0.
    pub fn cell_count(&self) -> usize {
        match self {
            Broadphase3::SpatialHash(h) => h.cell_count(),
            _ => 0,
        }
    }
}

impl Default for Broadphase3 {
    fn default() -> Self {
        Self::new(BroadphaseMode::default())
    }
}
