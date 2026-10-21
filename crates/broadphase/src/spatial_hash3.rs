use glam::Vec3;
use phieng_math::{fxhash::FxHashMap, AABB3};

/// Cell key for the 3D spatial hash grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellKey3 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// 3D spatial hash grid — the 3D counterpart of [`super::SpatialHash2`].
///
/// Same contract as the 2D grid: pooled per-cell buffers (no steady-state
/// allocation), a bounded insert cost (oversized bodies are paired against
/// the whole scene instead of filling millions of cells), and canonical
/// sorted+deduped pair output so every backend hands the narrowphase the
/// same pair sequence.
#[derive(Debug, Clone)]
pub struct SpatialHash3 {
    /// Cell size for the hash grid.
    pub cell_size: f32,
    /// Map from cell key to body IDs.
    grid: FxHashMap<CellKey3, Vec<usize>>,
    /// Number of inserted bodies.
    count: usize,
    /// Steps since vacated cells were compacted.
    /// Recycled cell buffers.
    pool: Vec<Vec<usize>>,
    /// Every inserted body id this frame (insertion order).
    all: Vec<usize>,
    /// Bodies too large for the grid; paired against the whole scene.
    oversized: Vec<usize>,
    /// Which body ids are static, indexed by body id (see the 2D grid).
    ///
    /// A pair of two static bodies can never produce an impulse (both
    /// have infinite mass), so the narrowphase discards it. The grid
    /// knows which bodies are static and skips those pairs at the
    /// source, instead of generating thousands of them and throwing
    /// them away one step later.
    statics: Vec<bool>,
    /// The lowest cell each body occupies, indexed by body id.
    ///
    /// A body is inserted into every cell its AABB spans, so a pair of
    /// bodies that share several cells is found in each of them — the
    /// same pair, several times over, before the sort and dedupe that
    /// the output contract requires. The lowest cell of each body is
    /// enough to pick one owner for every pair: the owner is
    /// `(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))`, the lowest cell
    /// both bodies share, and the pair is emitted only there. Every
    /// shared cell still contains both bodies, so no pair is lost.
    cell_min: Vec<(i32, i32, i32)>,
    /// The highest cell each body occupies, indexed by body id.
    ///
    /// Together with `cell_min` this is the body's *cell range*: the
    /// exact set of cells it is written into. The range is a pure
    /// function of the AABB and the cell size, so two AABBs with the
    /// same range put the body in exactly the same cells — which is
    /// what lets the grid keep a body's membership across steps.
    cell_max: Vec<(i32, i32, i32)>,
    /// The bodies the grid currently holds, with their cell membership.
    ///
    /// In the persistent mode the cells survive from step to step, so
    /// the grid has to know which bodies it still holds: a body the
    /// world no longer has has to be taken out of the cells, or it would
    /// keep pairing with the rest of the scene forever.
    tracked: Vec<usize>,
    /// Whether each body id is in `tracked`, indexed by body id.
    tracked_flag: Vec<bool>,
    /// The step each body was last inserted in, indexed by body id.
    ///
    /// A body that is not inserted in the current step is not in the
    /// world any more, and the grid drops it.
    seen: Vec<u32>,
    /// The current step, bumped by `begin_frame`.
    epoch: u32,
    /// Keep the cells across steps and only re-insert the bodies that
    /// moved.
    ///
    /// Profiling switch only (`phase-timing` feature): turning it off
    /// restores the behaviour of wiping every cell and re-inserting
    /// every body each step, so the two can be A/B measured inside one
    /// binary. The shipped build always keeps the cells, and has no
    /// branch in the insert path.
    #[cfg(feature = "phase-timing")]
    pub persistent: bool,
    /// Only skip a body whose AABB is *bit-identical* to the one it was
    /// inserted with.
    ///
    /// Profiling switch only (`phase-timing` feature), and off by
    /// default: the shipped rule skips any body whose cell range is
    /// unchanged, which is a pure integer comparison — the membership
    /// depends on the range and nothing else, so both rules produce the
    /// same pairs. The strict rule re-inserts a body whose range
    /// survived a move that did not change its cell, which is more work
    /// for the same result. This switch exists so the two can be
    /// measured against each other.
    #[cfg(feature = "phase-timing")]
    pub persistent_strict: bool,
    /// Generate each pair once, in its owner cell, instead of once per
    /// shared cell.
    ///
    /// Profiling switch only (`phase-timing` feature): turning it off
    /// restores the behaviour of generating the pair in every cell the
    /// two bodies share, so the two can be A/B measured inside one
    /// binary. The shipped build has no such switch and no branch in
    /// the pair loop.
    #[cfg(feature = "phase-timing")]
    pub owned_pairs: bool,
    /// Skip a cell that holds nothing but static bodies.
    ///
    /// Profiling switch only (`phase-timing` feature): turning it off
    /// restores the behaviour of walking every candidate pair in every
    /// cell and dropping the static-static ones one by one, so the two
    /// can be A/B measured inside one binary.
    #[cfg(feature = "phase-timing")]
    pub skip_static_cells: bool,
    /// The AABB of every body, indexed by body id. Used only by
    /// [`Self::collect_overlapping_pairs`] to drop candidate pairs whose
    /// AABBs are disjoint before they are pushed, sorted and deduplicated.
    aabbs: Vec<(Vec3, Vec3)>,
}

impl SpatialHash3 {
    /// Maximum cells a single body may occupy. A body whose AABB exceeds
    /// this is tracked in the oversized list and paired against the whole
    /// scene instead — correctness is preserved (it still collides with
    /// everything) while insert cost stays bounded.
    pub const MAX_CELLS_PER_BODY: usize = 4096;
    /// Cell-index magnitude beyond which a coordinate is treated as
    /// oversized (guards the `as i32` casts from saturating and
    /// producing degenerate ranges for extreme-but-finite positions).
    const MAX_CELL_INDEX: f32 = 2_000_000_000.0;

    pub fn new(cell_size: f32) -> Self {
        // Guard against zero/negative/infinite cell sizes: they would
        // produce unbounded cell ranges (a hang) on insert.
        let cell_size = if cell_size.is_finite() && cell_size > 0.0 {
            cell_size
        } else {
            1.0
        };
        Self {
            cell_size,
            grid: FxHashMap::default(),
            count: 0,
            pool: Vec::new(),
            all: Vec::new(),
            oversized: Vec::new(),
            statics: Vec::new(),
            cell_min: Vec::new(),
            cell_max: Vec::new(),
            tracked: Vec::new(),
            tracked_flag: Vec::new(),
            seen: Vec::new(),
            epoch: 0,
            #[cfg(feature = "phase-timing")]
            persistent: true,
            #[cfg(feature = "phase-timing")]
            persistent_strict: false,
            #[cfg(feature = "phase-timing")]
            owned_pairs: true,
            #[cfg(feature = "phase-timing")]
            skip_static_cells: true,
            aabbs: Vec::new(),
        }
    }

    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// Each cell's buffer is moved into the pool, where the next insert
    /// picks it up again: nothing is allocated, and the map holds only
    /// the cells that are actually in use.
    ///
    /// Keeping the vacated cells around instead grows the map without
    /// bound, and a large map turns every lookup into a cache miss.
    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// One pass over the map: every cell is emptied in place, and the
    /// cells that are now empty are dropped and their buffers recycled
    /// through the pool.
    ///
    /// Dropping the empty cells is what keeps the map small. The cells
    /// of a scene that sweeps its bodies across many positions pile up
    /// otherwise - a settled 1500-body 2D pile held 3 327 878 cells for
    /// 1501 bodies - and every lookup in a hash table with millions of
    /// entries is a cache miss: the insert pass cost ~650 ns per body.
    /// The cells that are still in use are kept, so the next insert
    /// finds them instead of recreating them: no per-frame churn.
    pub fn clear(&mut self) {
        let pool = &mut self.pool;
        self.grid.retain(|_, ids| {
            if ids.is_empty() {
                pool.push(std::mem::take(ids));
                false
            } else {
                ids.clear();
                true
            }
        });
        self.count = 0;
        self.all.clear();
        self.oversized.clear();
        // A hard reset also drops what the grid was holding: nothing
        // survives it.
        self.tracked.clear();
        self.tracked_flag.clear();
        self.seen.clear();
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Begin a step: reset what the step holds, keep what it does not.
    ///
    /// The cells survive the step, so the membership of a body that did
    /// not move survives it too, and only the bodies that moved are
    /// re-inserted. The frame-scoped lists (the bodies inserted this
    /// step, the oversized ones) are reset either way.
    ///
    /// [`Self::clear`] still wipes the grid completely; this is the
    /// step-scoped reset.
    pub fn begin_frame(&mut self) {
        if !self.keeps_cells() {
            self.clear();
            return;
        }
        self.epoch = self.epoch.wrapping_add(1);
        self.count = 0;
        self.all.clear();
        self.oversized.clear();
    }

    /// Whether the grid keeps its cells across steps.
    ///
    /// The shipped build always does; the profiling build can turn it
    /// off to measure the behaviour it replaced. The `cfg` folds the
    /// check away, so the insert path carries no branch.
    #[inline]
    fn keeps_cells(&self) -> bool {
        #[cfg(feature = "phase-timing")]
        {
            self.persistent
        }
        #[cfg(not(feature = "phase-timing"))]
        {
            true
        }
    }

    /// Insert a *static* body into the grid (see the 2D grid).
    ///
    /// A static body still pairs with every dynamic body, but two
    /// statics can never touch, so the grid skips those pairs.
    pub fn insert_static(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        self.insert_impl(body_id, min, max, true);
    }

    /// Whether a body is only skipped when its AABB is bit-identical to
    /// the one it was inserted with.
    ///
    /// The shipped build is not strict: it skips any body whose *cell
    /// range* is unchanged. The cells a body occupies are exactly the
    /// cells of its range, and the range is a tuple of integers, so an
    /// unchanged range means an unchanged membership — the pair set is
    /// the same either way, and there is no float comparison involved.
    ///
    /// The strict variant re-inserts a body whose range survived a
    /// move that did not change its cell. It produces the same pairs
    /// for more work. The profiling build can select it, so the
    /// measurement can be repeated.
    #[inline]
    fn strict_skip(&self) -> bool {
        #[cfg(feature = "phase-timing")]
        {
            self.persistent_strict
        }
        #[cfg(not(feature = "phase-timing"))]
        {
            false
        }
    }

    /// Whether a body id was inserted as static.
    #[inline]
    fn is_static(&self, body_id: usize) -> bool {
        self.statics.get(body_id).copied().unwrap_or(false)
    }

    /// Insert a body into the grid.
    pub fn insert(&mut self, body_id: usize, min: Vec3, max: Vec3) {
        self.insert_impl(body_id, min, max, false);
    }

    /// The shared body of [`Self::insert`] and [`Self::insert_static`].
    fn insert_impl(&mut self, body_id: usize, min: Vec3, max: Vec3, is_static: bool) {
        // Non-finite bounds (escaped NaN/inf velocity) cannot map to
        // cells — and must not reach the `as i32` casts below, which
        // saturate to i32::MIN/MAX and would hang the cell loops.
        if !min.x.is_finite()
            || !min.y.is_finite()
            || !min.z.is_finite()
            || !max.x.is_finite()
            || !max.y.is_finite()
            || !max.z.is_finite()
        {
            return;
        }
        self.count += 1;
        self.all.push(body_id);
        if self.aabbs.len() <= body_id {
            self.aabbs.resize(body_id + 1, (Vec3::ZERO, Vec3::ZERO));
        }
        if self.statics.len() <= body_id {
            self.statics.resize(body_id + 1, false);
        }
        // The static flag is rewritten every step, so a body that
        // changed kind cannot keep a stale one.
        self.statics[body_id] = is_static;
        let old_aabb = self.aabbs[body_id];
        self.aabbs[body_id] = (min, max);

        // Cells spanned per axis, computed in float to avoid overflow on
        // extreme-but-finite coordinates.
        let span_x = (max.x - min.x) / self.cell_size;
        let span_y = (max.y - min.y) / self.cell_size;
        let span_z = (max.z - min.z) / self.cell_size;
        let cells_f = (span_x + 1.0) * (span_y + 1.0) * (span_z + 1.0);
        let origin_f = (min.x.abs().max(max.x.abs()) / self.cell_size)
            .max(min.y.abs().max(max.y.abs()) / self.cell_size)
            .max(min.z.abs().max(max.z.abs()) / self.cell_size);

        // Oversized (too many cells, or coords beyond safe cell indices):
        // skip the grid and pair against the whole scene.
        if cells_f > Self::MAX_CELLS_PER_BODY as f32 || origin_f > Self::MAX_CELL_INDEX {
            // A body that used to fit in the grid has to leave it.
            if self.keeps_cells() {
                self.untrack(body_id);
            }
            self.oversized.push(body_id);
            return;
        }

        // Floor (not truncation) so negative coords map correctly.
        let cell_min_x = (min.x / self.cell_size).floor() as i32;
        let cell_min_y = (min.y / self.cell_size).floor() as i32;
        let cell_min_z = (min.z / self.cell_size).floor() as i32;
        let cell_max_x = (max.x / self.cell_size).floor() as i32;
        let cell_max_y = (max.y / self.cell_size).floor() as i32;
        let cell_max_z = (max.z / self.cell_size).floor() as i32;

        // The persistent mode: a body that did not move keeps its cells.
        if self.keeps_cells() {
            self.persistent_insert(
                body_id, min, max, old_aabb, cell_min_x, cell_min_y, cell_min_z, cell_max_x,
                cell_max_y, cell_max_z,
            );
            return;
        }

        // Remember the lowest cell this body occupies: it is what picks
        // the single owner cell of every pair the body takes part in.
        self.insert_into_cells(
            body_id, cell_min_x, cell_min_y, cell_min_z, cell_max_x, cell_max_y, cell_max_z,
        );
    }

    /// Put a body into the cells of its range and record where it is.
    ///
    /// Both modes go through here, so the grid's bookkeeping — the
    /// cells a body occupies, the bodies it holds, the step it was last
    /// seen in — is the same whichever mode ran. That matters because
    /// the profiling build switches between them from step to step: a
    /// mode that left the bookkeeping behind would hand the next mode a
    /// grid it cannot make sense of.
    #[allow(clippy::too_many_arguments)]
    fn insert_into_cells(
        &mut self,
        body_id: usize,
        cell_min_x: i32,
        cell_min_y: i32,
        cell_min_z: i32,
        cell_max_x: i32,
        cell_max_y: i32,
        cell_max_z: i32,
    ) {
        if self.cell_min.len() <= body_id {
            self.cell_min.resize(body_id + 1, (0, 0, 0));
        }
        if self.cell_max.len() <= body_id {
            self.cell_max.resize(body_id + 1, (0, 0, 0));
        }
        if self.tracked_flag.len() <= body_id {
            self.tracked_flag.resize(body_id + 1, false);
        }
        self.cell_min[body_id] = (cell_min_x, cell_min_y, cell_min_z);
        self.cell_max[body_id] = (cell_max_x, cell_max_y, cell_max_z);

        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                for cz in cell_min_z..=cell_max_z {
                    let key = CellKey3::new(cx, cy, cz);
                    self.grid
                        .entry(key)
                        .or_insert_with(|| self.pool.pop().unwrap_or_default())
                        .push(body_id);
                }
            }
        }

        if self.seen.len() <= body_id {
            self.seen.resize(body_id + 1, 0);
        }
        self.seen[body_id] = self.epoch;
        if !self.tracked_flag[body_id] {
            self.tracked.push(body_id);
            self.tracked_flag[body_id] = true;
        }
    }

    /// The persistent-mode insert: keep the cells a body already
    /// occupies whenever it can, and only move it when it has to.
    ///
    /// The cell range is a pure function of the AABB and the cell size,
    /// and the cells a body occupies are exactly the cells of its
    /// range. So a body whose range is unchanged is in exactly the same
    /// cells as before, and its membership can stay as it is: the grid
    /// only has to refresh the AABB it filters pairs with.
    #[allow(clippy::too_many_arguments)]
    fn persistent_insert(
        &mut self,
        body_id: usize,
        min: Vec3,
        max: Vec3,
        old_aabb: (Vec3, Vec3),
        cell_min_x: i32,
        cell_min_y: i32,
        cell_min_z: i32,
        cell_max_x: i32,
        cell_max_y: i32,
        cell_max_z: i32,
    ) {
        if self.seen.len() <= body_id {
            self.seen.resize(body_id + 1, 0);
        }
        self.seen[body_id] = self.epoch;

        let was_tracked = self.tracked_flag.get(body_id).copied().unwrap_or(false);
        if was_tracked {
            let old_min = self.cell_min[body_id];
            let old_max = self.cell_max[body_id];
            let same_range = old_min == (cell_min_x, cell_min_y, cell_min_z)
                && old_max == (cell_max_x, cell_max_y, cell_max_z);
            let same_aabb = old_aabb == (min, max);
            if same_range && (same_aabb || !self.strict_skip()) {
                // The membership is unchanged: the AABB in `self.aabbs`
                // was already refreshed by the caller.
                return;
            }
            self.remove_from_cells(body_id, old_min, old_max);
        }

        self.insert_into_cells(
            body_id, cell_min_x, cell_min_y, cell_min_z, cell_max_x, cell_max_y, cell_max_z,
        );
    }

    /// Take a body out of the cells it occupies.
    ///
    /// A cell that runs empty is dropped from the map and its buffer
    /// recycled, which is what keeps the map at the size of the scene's
    /// current extent instead of every position the scene ever visited.
    fn remove_from_cells(&mut self, body_id: usize, min: (i32, i32, i32), max: (i32, i32, i32)) {
        for cx in min.0..=max.0 {
            for cy in min.1..=max.1 {
                for cz in min.2..=max.2 {
                    let key = CellKey3::new(cx, cy, cz);
                    let empty = match self.grid.get_mut(&key) {
                        Some(ids) => {
                            if let Some(pos) = ids.iter().position(|&id| id == body_id) {
                                ids.swap_remove(pos);
                            }
                            ids.is_empty()
                        }
                        None => false,
                    };
                    if empty {
                        if let Some(mut ids) = self.grid.remove(&key) {
                            ids.clear();
                            self.pool.push(ids);
                        }
                    }
                }
            }
        }
    }

    /// Take a body out of the grid entirely.
    fn untrack(&mut self, body_id: usize) {
        if !self.tracked_flag.get(body_id).copied().unwrap_or(false) {
            return;
        }
        let min = self.cell_min[body_id];
        let max = self.cell_max[body_id];
        self.remove_from_cells(body_id, min, max);
        self.tracked_flag[body_id] = false;
        if let Some(pos) = self.tracked.iter().position(|&id| id == body_id) {
            self.tracked.swap_remove(pos);
        }
    }

    /// Drop the bodies the world no longer has.
    ///
    /// The cells survive the step, so a body that is not inserted any
    /// more has to be taken out of them by hand — otherwise it would
    /// keep pairing with the rest of the scene forever.
    pub fn end_frame(&mut self) {
        if !self.keeps_cells() {
            return;
        }
        let epoch = self.epoch;
        let mut i = 0;
        while i < self.tracked.len() {
            let body_id = self.tracked[i];
            if self.seen.get(body_id).copied() != Some(epoch) {
                let min = self.cell_min[body_id];
                let max = self.cell_max[body_id];
                self.remove_from_cells(body_id, min, max);
                self.tracked_flag[body_id] = false;
                self.tracked.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Append every overlapping pair to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Output is sorted and deduped, so every backend produces the same
    /// pair sequence.
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for (key, cell_ids) in &self.grid {
            // A cell of nothing but statics cannot produce a pair.
            if !self.cell_can_pair(cell_ids) {
                continue;
            }
            for i in 0..cell_ids.len() {
                for j in (i + 1)..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    // Two statics can never touch.
                    if self.is_static(a) && self.is_static(b) {
                        continue;
                    }
                    // Emit the pair in its owner cell only (see
                    // `owns_pair`).
                    if !self.owns_pair(a, b, key.x, key.y, key.z) {
                        continue;
                    }
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        // Oversized bodies are not in the grid: pair each against every
        // body (their huge AABBs overlap anything in range).
        for &o in &self.oversized {
            for &b in &self.all {
                if o != b {
                    out.push(if o < b { (o, b) } else { (b, o) });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Whether two bodies' AABBs overlap.
    ///
    /// Conservative with respect to the narrow phase: an AABB contains
    /// its shape, so two bodies whose AABBs do not overlap cannot touch.
    fn aabbs_overlap(a: usize, b: usize, aabbs: &[(Vec3, Vec3)]) -> bool {
        match (aabbs.get(a), aabbs.get(b)) {
            (Some((amin, amax)), Some((bmin, bmax))) => {
                amin.x <= bmax.x
                    && amax.x >= bmin.x
                    && amin.y <= bmax.y
                    && amax.y >= bmin.y
                    && amin.z <= bmax.z
                    && amax.z >= bmin.z
            }
            // An id without a recorded AABB cannot be filtered.
            _ => true,
        }
    }

    /// Append the ids of the bodies whose AABB overlaps `aabb`.
    ///
    /// The bodies whose AABB overlaps the query box, in ascending id
    /// order and without duplicates: a body that spans several cells
    /// is in each of them, and it is reported once.
    pub fn query_aabb(&self, aabb: &AABB3, out: &mut Vec<usize>) {
        let cell_min_x = (aabb.min.x / self.cell_size).floor() as i32;
        let cell_min_y = (aabb.min.y / self.cell_size).floor() as i32;
        let cell_min_z = (aabb.min.z / self.cell_size).floor() as i32;
        let cell_max_x = (aabb.max.x / self.cell_size).floor() as i32;
        let cell_max_y = (aabb.max.y / self.cell_size).floor() as i32;
        let cell_max_z = (aabb.max.z / self.cell_size).floor() as i32;
        let before = out.len();
        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                for cz in cell_min_z..=cell_max_z {
                    if let Some(ids) = self.grid.get(&CellKey3::new(cx, cy, cz)) {
                        out.extend_from_slice(ids);
                    }
                }
            }
        }
        if out.len() > before {
            out[before..].sort_unstable();
            let tail = &mut out[before..];
            let mut write = 0;
            for read in 0..tail.len() {
                if read == 0 || tail[read] != tail[write - 1] {
                    tail[write] = tail[read];
                    write += 1;
                }
            }
            out.truncate(before + write);
        }
    }

    /// Whether a cell can produce a pair at all.
    ///
    /// A cell that holds only static bodies cannot: two statics never
    /// touch, so every candidate pair in it would be dropped anyway.
    /// Skipping the cell whole is what keeps a static-heavy scene from
    /// walking its quadratic number of static-static candidates.
    #[inline]
    fn cell_can_pair(&self, cell_ids: &[usize]) -> bool {
        // The profiling build can turn the rule off to measure the
        // behaviour it replaced.
        #[cfg(feature = "phase-timing")]
        if !self.skip_static_cells {
            return true;
        }
        !cell_ids.iter().all(|&id| self.is_static(id))
    }

    /// Whether cell `(cx, cy, cz)` is the single owner of the pair
    /// `(a, b)`.
    ///
    /// The owner is the lowest cell both bodies span, so a pair that
    /// shares several cells is generated in exactly one of them. The
    /// pair set is unchanged -- the same pairs, in the same canonical
    /// order -- but the list that has to be sorted and deduplicated is
    /// as long as the pair set instead of a multiple of it.
    #[inline]
    fn owns_pair(&self, a: usize, b: usize, cx: i32, cy: i32, cz: i32) -> bool {
        // The profiling build can turn the rule off to measure the
        // behaviour it replaced.
        #[cfg(feature = "phase-timing")]
        if !self.owned_pairs {
            return true;
        }
        let (ax, ay, az) = self.cell_min[a];
        let (bx, by, bz) = self.cell_min[b];
        cx == ax.max(bx) && cy == ay.max(by) && cz == az.max(bz)
    }

    /// Append the subset of potential collision pairs whose AABBs overlap.
    ///
    /// This is the narrowphase fast path: every dropped pair is two
    /// bodies whose AABBs are disjoint, so neither body can touch the
    /// other and the narrow phase would have produced zero contacts for
    /// it. The emitted sequence is the AABB-overlapping subsequence of
    /// what [`Self::collect_pairs`] emits (same canonical order), so
    /// switching the narrowphase to this method cannot change the
    /// constraint stream or the trajectory. Contract: `out` must be
    /// empty (the world passes cleared scratch).
    ///
    /// The CCD swept-volume test deliberately keeps using
    /// [`Self::collect_pairs`]: a pair whose current AABBs are disjoint
    /// can still overlap once swept along the velocities, and dropping
    /// it here would blind the substepping decision.
    pub fn collect_overlapping_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for (key, cell_ids) in &self.grid {
            // A cell of nothing but statics cannot produce a pair.
            if !self.cell_can_pair(cell_ids) {
                continue;
            }
            for i in 0..cell_ids.len() {
                for j in (i + 1)..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    // Emit the pair in its owner cell only (see
                    // `owns_pair`).
                    if !self.owns_pair(a, b, key.x, key.y, key.z) {
                        continue;
                    }
                    if !Self::aabbs_overlap(a, b, &self.aabbs) {
                        continue;
                    }
                    // Canonical pair key.
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        // Oversized bodies are not in the grid: pair each against every
        // body whose AABB it overlaps.
        for &o in &self.oversized {
            for &b in &self.all {
                if o != b && Self::aabbs_overlap(o, b, &self.aabbs) {
                    out.push(if o < b { (o, b) } else { (b, o) });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// How many cells the grid currently holds.
    ///
    /// A grid that holds many more cells than the scene has bodies is
    /// bloated, and every lookup in a large hash table is a cache miss.
    pub fn cell_count(&self) -> usize {
        self.grid.len()
    }

    /// Get the number of bodies.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for SpatialHash3 {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash3_pairs() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        hash.insert(1, Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        hash.insert(2, Vec3::new(10.0, 10.0, 10.0), Vec3::new(12.0, 12.0, 12.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_spatial_hash3_no_collision() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash3_negative_coords() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(
            0,
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(-9.0, -9.0, -9.0),
        );
        hash.insert(1, Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0));
        hash.insert(2, Vec3::new(-9.5, -9.5, -9.5), Vec3::new(-8.0, -8.0, -8.0));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 2)]);
    }

    #[test]
    fn test_spatial_hash3_rejects_non_finite() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::ZERO, Vec3::new(f32::INFINITY, 1.0, 1.0));
        assert!(hash.is_empty());
        assert_eq!(hash.len(), 0);
    }

    #[test]
    fn test_spatial_hash3_oversized_pairs_with_all() {
        // A body too large for the grid must not hang insert, and must
        // still pair with every other body.
        let mut hash = SpatialHash3::new(1.0);
        let start = std::time::Instant::now();
        hash.insert(0, Vec3::splat(-1000.0), Vec3::splat(1000.0));
        hash.insert(1, Vec3::splat(500.0), Vec3::splat(501.0));
        assert!(
            start.elapsed().as_millis() < 500,
            "oversized insert must be fast, took {:?}",
            start.elapsed()
        );

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(
            pairs.contains(&(0, 1)),
            "oversized body must pair with the small body"
        );
    }

    #[test]
    fn test_spatial_hash3_clear_reuses_buffers() {
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        hash.insert(1, Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        hash.clear();
        assert!(hash.is_empty());

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash3_clear_reuses_buffers_without_leaking_ids() {
        // A cleared cell's buffer goes back into the pool and is handed
        // out again on the next insert. If the recycled buffer still held
        // the previous frame's ids, the cell would report pairs that do
        // not exist.
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        hash.insert(1, Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.5, 2.5, 2.5));
        let mut first = Vec::new();
        hash.collect_pairs(&mut first);
        assert_eq!(first.len(), 1);

        hash.clear();

        hash.insert(7, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(
            pairs.is_empty(),
            "a recycled cell must not carry the previous frame's ids: {pairs:?}"
        );
        assert_eq!(hash.len(), 1);

        hash.insert(8, Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.5, 2.5, 2.5));
        let mut pairs2 = Vec::new();
        hash.collect_pairs(&mut pairs2);
        assert_eq!(pairs2, vec![(7, 8)]);
    }

    #[test]
    fn test_spatial_hash3_has_no_false_negatives() {
        use crate::brute_force3::BruteForce3;
        let mut hash = SpatialHash3::new(1.0);
        let mut bf = BruteForce3::new();
        // Deterministic pseudo-cluster: 30 bodies in 4 tight groups.
        let mut bounds: Vec<(Vec3, Vec3)> = Vec::new();
        for i in 0..30usize {
            let g = (i % 4) as f32;
            let k = (i / 4) as f32;
            let min = Vec3::new(g * 40.0 + k * 0.5, k * 0.5, k * 0.25);
            let max = min + Vec3::splat(2.0);
            hash.insert(i, min, max);
            bf.insert(i, min, max);
            bounds.push((min, max));
        }

        let mut grid_pairs = Vec::new();
        hash.collect_pairs(&mut grid_pairs);
        let mut exact_pairs = Vec::new();
        bf.collect_pairs(&mut exact_pairs);

        // The grid reports *cell* overlaps, so it may emit extra pairs
        // (the narrowphase rejects them). What it must never do is miss a
        // genuinely overlapping pair — that would be a false negative.
        for pair in &exact_pairs {
            assert!(
                grid_pairs.contains(pair),
                "grid missed an overlapping pair {pair:?}"
            );
        }
        // Every extra pair the grid adds must be a false positive: the
        // two AABBs do not actually overlap.
        for pair in &grid_pairs {
            if exact_pairs.contains(pair) {
                continue;
            }
            let (a, b) = *pair;
            let (amin, amax) = bounds[a];
            let (bmin, bmax) = bounds[b];
            let overlaps = amax.x >= bmin.x
                && amin.x <= bmax.x
                && amax.y >= bmin.y
                && amin.y <= bmax.y
                && amax.z >= bmin.z
                && amin.z <= bmax.z;
            assert!(!overlaps, "grid emitted {pair:?} as a non-overlapping pair");
        }
    }

    #[test]
    fn test_overlapping_pairs_drops_disjoint_cell_sharers() {
        // Two bodies sharing grid cells without overlapping: the
        // unfiltered sequence reports the pair (the CCD swept test
        // relies on that), while the narrowphase fast path must not.
        let mut hash = SpatialHash3::new(1.0);
        hash.insert(0, Vec3::new(-10.0, -1.5, -1.5), Vec3::new(10.0, -0.5, 0.5));
        hash.insert(1, Vec3::new(-0.3, -0.4, -0.4), Vec3::new(0.3, 0.2, 0.2));
        let mut filtered = Vec::new();
        hash.collect_overlapping_pairs(&mut filtered);
        assert!(
            filtered.is_empty(),
            "disjoint AABBs must be filtered: {filtered:?}"
        );
        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.contains(&(0, 1)));
    }

    #[test]
    fn test_overlapping_pairs_agrees_with_brute_force() {
        use crate::brute_force3::BruteForce3;
        let scene: Vec<(Vec3, Vec3)> = vec![
            (Vec3::ZERO, Vec3::splat(2.0)),
            (Vec3::splat(1.0), Vec3::splat(3.0)),
            (Vec3::new(2.5, 0.0, 0.0), Vec3::new(3.5, 1.0, 1.0)),
            (Vec3::splat(50.0), Vec3::splat(52.0)),
            (Vec3::splat(-1000.0), Vec3::splat(1000.0)),
            (Vec3::splat(500.0), Vec3::splat(501.0)),
            (Vec3::splat(2000.0), Vec3::splat(2001.0)),
        ];
        let mut grid = SpatialHash3::new(1.0);
        let mut bf = BruteForce3::new();
        for (i, (min, max)) in scene.iter().enumerate() {
            grid.insert(i, *min, *max);
            bf.insert(i, *min, *max);
        }
        let mut a = Vec::new();
        grid.collect_overlapping_pairs(&mut a);
        let mut b = Vec::new();
        bf.collect_overlapping_pairs(&mut b);
        assert_eq!(a, b, "grid and brute force must filter identically");
        // Sanity: the oversized body pairs with what it overlaps, not
        // with the far body.
        assert!(a.contains(&(4, 5)));
        assert!(!a.iter().any(|&(x, y)| x == 6 || y == 6));
    }
}
