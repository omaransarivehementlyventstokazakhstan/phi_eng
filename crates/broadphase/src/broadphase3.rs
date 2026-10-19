use glam::Vec3;

use super::broadphase_mode::BroadphaseMode;
use super::brute_force3::BruteForce3;
use super::dynamic_aabb_tree3::DynamicAABBTree3;

/// Broadphase backend for 3D worlds. Wraps the concrete implementation
/// selected by [`BroadphaseMode`].
#[derive(Debug, Clone)]
pub enum Broadphase3 {
    BruteForce(BruteForce3),
    DynamicAABBTree(DynamicAABBTree3),
}

impl Broadphase3 {
    pub fn new(mode: BroadphaseMode) -> Self {
        match mode {
            BroadphaseMode::BruteForce => Broadphase3::BruteForce(BruteForce3::new()),
            BroadphaseMode::DynamicAABBTree => {
                Broadphase3::DynamicAABBTree(DynamicAABBTree3::new())
            }
            // 3D SpatialHash not yet implemented; fall back to BruteForce.
            BroadphaseMode::SpatialHash | BroadphaseMode::Hybrid => {
                Broadphase3::BruteForce(BruteForce3::new())
            }
        }
    }

    pub fn clear(&mut self) {
        match self {
            Broadphase3::BruteForce(b) => b.clear(),
            Broadphase3::DynamicAABBTree(t) => t.clear(),
        }
    }

    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        match self {
            Broadphase3::BruteForce(b) => b.insert(body_id, min, max),
            Broadphase3::DynamicAABBTree(t) => t.insert(body_id, min, max),
        }
    }

    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        match self {
            Broadphase3::BruteForce(b) => b.collect_pairs(out),
            Broadphase3::DynamicAABBTree(t) => t.collect_pairs(out),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Broadphase3::BruteForce(b) => b.len(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> [(Vec3, Vec3); 6] {
        [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0)),
            (Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0)),
            (Vec3::new(2.5, 2.5, 2.5), Vec3::new(4.0, 4.0, 4.0)),
            (Vec3::new(50.0, 50.0, 50.0), Vec3::new(52.0, 52.0, 52.0)),
            (Vec3::new(51.0, 51.0, 51.0), Vec3::new(53.0, 53.0, 53.0)),
            (
                Vec3::new(-30.0, -30.0, -30.0),
                Vec3::new(-28.0, -28.0, -28.0),
            ),
        ]
    }

    #[test]
    fn test_both_modes_agree() {
        // The tier system is a performance choice only: brute force and
        // the tree must produce identical pair sets for the same scene.
        let s = scene();
        let mut results = Vec::new();
        for mode in [BroadphaseMode::BruteForce, BroadphaseMode::DynamicAABBTree] {
            let mut bp = Broadphase3::new(mode);
            for (i, (min, max)) in s.iter().enumerate() {
                bp.insert(i, *min, *max);
            }
            let mut pairs = Vec::new();
            bp.collect_pairs(&mut pairs);
            assert_eq!(bp.len(), 6);
            results.push(pairs);
        }
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn test_spatial_hash_falls_back_to_brute_force() {
        // 3D has no spatial-hash backend yet; the constructor must map it
        // to brute force rather than panicking or losing pairs.
        let bp = Broadphase3::new(BroadphaseMode::SpatialHash);
        assert!(matches!(bp, Broadphase3::BruteForce(_)));
        let bp = Broadphase3::new(BroadphaseMode::Hybrid);
        assert!(matches!(bp, Broadphase3::BruteForce(_)));
    }

    #[test]
    fn test_clear_empties_every_backend() {
        for mode in [BroadphaseMode::BruteForce, BroadphaseMode::DynamicAABBTree] {
            let mut bp = Broadphase3::new(mode);
            bp.insert(0, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
            bp.insert(1, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
            assert_eq!(bp.len(), 2);
            bp.clear();
            assert!(bp.is_empty());
            let mut pairs = Vec::new();
            bp.collect_pairs(&mut pairs);
            assert!(pairs.is_empty());
        }
    }

    #[test]
    fn test_empty_tree_collects_nothing() {
        let bp = Broadphase3::new(BroadphaseMode::DynamicAABBTree);
        let mut pairs = Vec::new();
        bp.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }
}
