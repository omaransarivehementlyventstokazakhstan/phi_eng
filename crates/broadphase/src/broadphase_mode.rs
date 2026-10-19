/// Broad-phase collision detection mode.
///
/// Selected per-world; `Hybrid` auto-selects based on live body count so
/// callers do not need to tune it manually. Every mode is deterministic
/// (pairs are sorted + deduped before leaving the broadphase) and uses
/// pooled scratch where possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BroadphaseMode {
    /// O(n^2) all-pairs test. Fastest for tiny scenes (< ~50 bodies),
    /// no grid/tree overhead.
    BruteForce,
    /// Spatial hash grid. Best for uniform distributions, 50-5000 bodies.
    /// Current default for 2D worlds.
    #[default]
    SpatialHash,
    /// Dynamic AABB tree. Best for large scenes (5000+ bodies) and
    /// clustered distributions where a hash grid degenerates.
    DynamicAABBTree,
    /// Auto-select: BruteForce < 50, SpatialHash 50-5000,
    /// DynamicAABBTree > 5000.
    Hybrid,
}

impl BroadphaseMode {
    /// Auto-select a concrete mode from a live body count. Mirrors the
    /// thresholds documented in OPTIMIZATION_PLAN.md section 2.1.
    pub fn auto_select(body_count: usize) -> BroadphaseMode {
        if body_count < 50 {
            BroadphaseMode::BruteForce
        } else if body_count < 5000 {
            BroadphaseMode::SpatialHash
        } else {
            BroadphaseMode::DynamicAABBTree
        }
    }

    /// Human-readable name for logging / debug output.
    pub fn name(self) -> &'static str {
        match self {
            BroadphaseMode::BruteForce => "BruteForce",
            BroadphaseMode::SpatialHash => "SpatialHash",
            BroadphaseMode::DynamicAABBTree => "DynamicAABBTree",
            BroadphaseMode::Hybrid => "Hybrid",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_spatial_hash() {
        // Preserves pre-tier-system behavior bit-for-bit.
        assert_eq!(BroadphaseMode::default(), BroadphaseMode::SpatialHash);
        assert_eq!(BroadphaseMode::default().name(), "SpatialHash");
    }

    #[test]
    fn test_auto_select_thresholds() {
        assert_eq!(BroadphaseMode::auto_select(0), BroadphaseMode::BruteForce);
        assert_eq!(BroadphaseMode::auto_select(49), BroadphaseMode::BruteForce);
        assert_eq!(BroadphaseMode::auto_select(50), BroadphaseMode::SpatialHash);
        assert_eq!(
            BroadphaseMode::auto_select(4999),
            BroadphaseMode::SpatialHash
        );
        assert_eq!(
            BroadphaseMode::auto_select(5000),
            BroadphaseMode::DynamicAABBTree
        );
    }

    #[test]
    fn test_auto_select_never_returns_hybrid() {
        // Hybrid is a request, never a resolved backend: resolving it
        // twice must be a fixed point or Hybrid would thrash the tree.
        for n in [0usize, 10, 100, 10_000] {
            let resolved = BroadphaseMode::auto_select(n);
            assert_ne!(resolved, BroadphaseMode::Hybrid);
            assert_eq!(BroadphaseMode::auto_select(n), resolved);
        }
    }

    #[test]
    fn test_names_are_distinct() {
        let all = [
            BroadphaseMode::BruteForce,
            BroadphaseMode::SpatialHash,
            BroadphaseMode::DynamicAABBTree,
            BroadphaseMode::Hybrid,
        ];
        let mut names: Vec<&str> = all.iter().map(|m| m.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 4);
    }
}
