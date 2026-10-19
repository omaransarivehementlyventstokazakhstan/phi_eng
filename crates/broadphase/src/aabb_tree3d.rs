use glam::Vec3;
use phieng_math::AABB3;

/// Flat 3D broadphase over stored AABBs. (Named "tree" for API parity;
/// the current implementation is a brute-force list, which is cache-
/// friendly and exact for the body counts this engine targets.)
///
/// Legacy surface: prefer [`Broadphase3`](super::Broadphase3), which
/// exposes this same all-pairs backend as its `BruteForce` mode alongside
/// the real [`DynamicAABBTree3`](super::DynamicAABBTree3). Kept so existing
/// callers do not break.
#[derive(Debug)]
pub struct AABBTree3D {
    body_aabbs: Vec<(usize, AABB3)>,
}

impl AABBTree3D {
    pub fn new() -> Self {
        Self {
            body_aabbs: Vec::new(),
        }
    }

    /// Clear all entries (the world rebuilds the broadphase every step).
    pub fn clear(&mut self) {
        self.body_aabbs.clear();
    }

    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        let aabb = AABB3::new(min, max);
        self.body_aabbs.push((body_id, aabb));
    }

    pub fn query(&self, query_aabb: &AABB3) -> Vec<usize> {
        let mut results = Vec::new();
        for (body_id, aabb) in &self.body_aabbs {
            if aabb.intersects(query_aabb) {
                results.push(*body_id);
            }
        }
        results
    }

    pub fn get_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        self.collect_pairs(&mut pairs);
        pairs
    }

    /// Append potential collision pairs to `out` without allocating.
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Pairs are sorted and deduped so every backend hands the narrowphase
    /// the same pair sequence (the sequential solver accumulates in pair
    /// order, so ordering must not vary between backends).
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for i in 0..self.body_aabbs.len() {
            for j in (i + 1)..self.body_aabbs.len() {
                let (id_a, aabb_a) = &self.body_aabbs[i];
                let (id_b, aabb_b) = &self.body_aabbs[j];
                if id_a == id_b {
                    continue;
                }
                if aabb_a.intersects(aabb_b) {
                    out.push((*id_a, *id_b));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    pub fn len(&self) -> usize {
        self.body_aabbs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.body_aabbs.is_empty()
    }
}

impl Default for AABBTree3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_tree3d_insert_and_query() {
        let mut tree = AABBTree3D::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        tree.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));

        let query = AABB3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let results = tree.query(&query);
        assert!(results.contains(&1));
    }

    #[test]
    fn test_aabb_tree3d_no_collision() {
        let mut tree = AABBTree3D::new();
        tree.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        tree.insert(1, Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        let query = AABB3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let results = tree.query(&query);
        assert!(results.contains(&0));
        assert!(!results.contains(&1));
    }
}
