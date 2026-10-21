use glam::Vec3;

use super::broadphase_mode::BroadphaseMode;
use super::brute_force3::BruteForce3;
use super::dynamic_aabb_tree3::DynamicAABBTree3;
use super::spatial_hash3::SpatialHash3;

/// Broadphase backend for 3D worlds. Wraps the concrete implementation
/// selected by [`BroadphaseMode`].
///
/// The grid is the largest variant by a wide margin, and the enum is
/// held by value in the world. The lint about that is not actionable
/// here: the enum is never copied per step (the world is stepped
/// through `&mut self`), and boxing the grid would put a pointer chase
/// in the pair loop, which runs thousands of times per step.
#[allow(clippy::large_enum_variant)]
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

    /// Begin a step: reset what the step holds, keep what it does not.
    ///
    /// The grid keeps its cells across steps, so a body that did not
    /// move keeps its place in them; the other backends rebuild from
    /// scratch every step and have nothing to keep.
    pub fn begin_frame(&mut self) {
        match self {
            Broadphase3::BruteForce(b) => b.clear(),
            Broadphase3::SpatialHash(h) => h.begin_frame(),
            Broadphase3::DynamicAABBTree(t) => t.clear(),
        }
    }

    /// Close the frame: drop the bodies the world no longer has.
    ///
    /// The grid keeps its cells across steps, so a body that is not
    /// inserted any more has to be taken out of them. The other
    /// backends rebuild from scratch every step, so they have nothing
    /// to close.
    pub fn end_frame(&mut self) {
        match self {
            Broadphase3::SpatialHash(h) => h.end_frame(),
            Broadphase3::BruteForce(_) | Broadphase3::DynamicAABBTree(_) => {}
        }
    }

    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        match self {
            Broadphase3::BruteForce(b) => b.insert(body_id, min, max),
            Broadphase3::SpatialHash(h) => h.insert(body_id, min, max),
            Broadphase3::DynamicAABBTree(t) => t.insert(body_id, min, max),
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
    pub fn insert_static(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        match self {
            Broadphase3::SpatialHash(h) => h.insert_static(body_id, min, max),
            _ => self.insert(body_id, min, max),
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

    /// Profiling only: turn the grid's persistent cells on or off.
    ///
    /// The persistent grid keeps its cells across steps and only
    /// re-inserts the bodies that moved; the other behaviour wipes
    /// every cell and re-inserts every body each step. The pair set is
    /// the same either way; only the amount of work differs.
    #[cfg(feature = "phase-timing")]
    pub fn set_persistent(&mut self, persistent: bool) {
        if let Broadphase3::SpatialHash(h) = self {
            h.persistent = persistent;
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
        if let Broadphase3::SpatialHash(h) = self {
            h.owned_pairs = owned;
        }
    }

    /// Profiling only: turn the grid's static-cell skip on or off.
    ///
    /// The skip drops a cell that holds nothing but static bodies
    /// without walking its candidate pairs. The pair set is the same
    /// either way; only the amount of candidate work differs.
    #[cfg(feature = "phase-timing")]
    pub fn set_skip_static_cells(&mut self, skip: bool) {
        if let Broadphase3::SpatialHash(h) = self {
            h.skip_static_cells = skip;
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
