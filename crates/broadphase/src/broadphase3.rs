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
}

impl Default for Broadphase3 {
    fn default() -> Self {
        Self::new(BroadphaseMode::default())
    }
}
