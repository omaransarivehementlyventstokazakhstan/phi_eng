use glam::Vec3;
use phieng_math::AABB3;

/// O(n^2) all-pairs broadphase for 3D. Best for tiny scenes (< ~50 bodies).
/// Every pair is tested exactly once; pairs are emitted in index order.
#[derive(Debug, Default)]
pub struct BruteForce3 {
    entries: Vec<(usize, AABB3)>,
}

impl BruteForce3 {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        self.entries.push((body_id, AABB3::new(min, max)));
    }

    /// Append every overlapping pair to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Sorted and deduped so every backend hands the narrowphase the same
    /// pair sequence (the sequential solver accumulates in pair order, so
    /// an unsorted backend would perturb trajectories at the 1e-13 level).
    /// Append the ids of the bodies whose AABB overlaps `aabb`.
    ///
    /// The bodies whose AABB overlaps the query box, in ascending id
    /// order and without duplicates.
    pub fn query_aabb(&self, aabb: &AABB3, out: &mut Vec<usize>) {
        for (body_id, body_aabb) in &self.entries {
            if body_aabb.intersects(aabb) {
                out.push(*body_id);
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                let (id_a, aabb_a) = &self.entries[i];
                let (id_b, aabb_b) = &self.entries[j];
                if aabb_a.intersects(aabb_b) {
                    out.push((*id_a, *id_b));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Same as [`Self::collect_pairs`]: brute force tests every pair's
    /// AABB overlap already, so its output is exactly the
    /// AABB-overlapping subsequence by construction. Provided so the
    /// narrowphase fast path is uniform across backends.
    pub fn collect_overlapping_pairs(&self, out: &mut Vec<(usize, usize)>) {
        self.collect_pairs(out);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Clone for BruteForce3 {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brute_force3_pairs() {
        let mut bf = BruteForce3::new();
        bf.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        bf.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        bf.insert(2, Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 11.0, 11.0));

        let mut pairs = Vec::new();
        bf.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_brute_force3_no_collision() {
        let mut bf = BruteForce3::new();
        bf.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        bf.insert(1, Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        let mut pairs = Vec::new();
        bf.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }
}
