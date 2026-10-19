use glam::Vec2;
use phieng_math::AABB2;

/// O(n^2) all-pairs broadphase. Best for tiny scenes (< ~50 bodies) where
/// grid/tree overhead dominates. Every pair is tested exactly once;
/// pairs are emitted in index order (deterministic).
#[derive(Debug, Default)]
pub struct BruteForce2 {
    entries: Vec<(usize, AABB2)>,
}

impl BruteForce2 {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        self.entries.push((body_id, AABB2::new(min, max)));
    }

    /// Append every overlapping pair to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Output is sorted and deduped so every backend hands the narrowphase
    /// the *same* pair sequence — the sequential solver accumulates in pair
    /// order, so an unsorted backend would perturb trajectories at the 1e-13
    /// level (positions still match, but not bit-exactly).
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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Clone for BruteForce2 {
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
    fn test_brute_force2_pairs() {
        let mut bf = BruteForce2::new();
        bf.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        bf.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        bf.insert(2, Vec2::new(10.0, 10.0), Vec2::new(11.0, 11.0));

        let mut pairs = Vec::new();
        bf.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_brute_force2_no_collision() {
        let mut bf = BruteForce2::new();
        bf.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        bf.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));

        let mut pairs = Vec::new();
        bf.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }
}
