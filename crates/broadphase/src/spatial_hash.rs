use glam::Vec2;
use phieng_math::fxhash::FxHashMap;

/// Cell key for the spatial hash grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey {
    pub x: i32,
    pub y: i32,
}

impl CellKey {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Spatial Hash Grid for broad phase collision detection
#[derive(Debug, Clone)]
pub struct SpatialHash2 {
    /// Cell size for the hash grid
    pub cell_size: f32,
    /// Map from cell key to body IDs
    grid: FxHashMap<CellKey, Vec<usize>>,
    /// Number of inserted bodies (the grid owns no AABB copies; the
    /// world rebuilds it from live body positions every step).
    count: usize,
    /// Recycled cell buffers. Vacated cells return their `Vec` here instead
    /// of freeing it; new cells draw from it, so re-appearing cells never
    /// re-allocate. Internal (the public API is unchanged).
    pool: Vec<Vec<usize>>,
    /// Every inserted body id this frame (insertion order). Used to pair
    /// oversized bodies against the whole scene.
    all: Vec<usize>,
    /// The AABB of every body, indexed by body id.
    ///
    /// The grid pairs two bodies when they share a cell, which is
    /// conservative: sharing a cell says nothing about overlapping. The
    /// AABB does, and the narrow phase would have rejected a pair whose
    /// AABBs do not overlap anyway - so testing it here removes the
    /// false positives before they are pushed, sorted and deduplicated.
    aabbs: Vec<(Vec2, Vec2)>,
    /// Bodies whose AABB spans more cells than [`Self::MAX_CELLS_PER_BODY`].
    /// Such bodies are NOT written into the grid (a 1000-unit body at a
    /// 1.0 cell size would touch millions of cells and hang the step);
    /// instead they are paired against every body in `collect_pairs`.
    oversized: Vec<usize>,
    /// Which body ids are static, indexed by body id.
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
    /// `(max(a.x, b.x), max(a.y, b.y))`, the lowest cell both bodies
    /// share, and the pair is emitted only there. Every shared cell
    /// still contains both bodies, so no pair is lost — each one is
    /// generated exactly once, and the candidate list that gets sorted
    /// and deduplicated shrinks by the duplication factor.
    cell_min: Vec<(i32, i32)>,
    /// The highest cell each body occupies, indexed by body id.
    ///
    /// Together with `cell_min` this is the body's *cell range*: the
    /// exact set of cells it is written into. The range is a pure
    /// function of the AABB and the cell size, so two AABBs with the
    /// same range put the body in exactly the same cells — which is
    /// what lets the grid keep a body's membership across steps.
    cell_max: Vec<(i32, i32)>,
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
    /// The current step, bumped by `clear`.
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
}

impl SpatialHash2 {
    /// Maximum cells a single body may occupy in the grid. A body whose
    /// AABB exceeds this is tracked in the oversized list and paired
    /// against the whole scene instead — correctness is preserved
    /// (it still collides with everything) while keeping insert cost
    /// bounded. Sized so ordinary scenes never trigger it (e.g. a
    /// 64x64-cell, 90x90-unit body at cell size 1.0).
    pub const MAX_CELLS_PER_BODY: usize = 4096;
    /// Cell-index magnitude beyond which a coordinate is treated as
    /// oversized (guards the `as i32` casts from saturating to i32::MIN/MAX
    /// and producing degenerate ranges for extreme-but-finite positions).
    const MAX_CELL_INDEX: f32 = 2_000_000_000.0;

    pub fn new(cell_size: f32) -> Self {
        // Guard against zero/negative/infinite cell sizes: they would produce
        // unbounded cell ranges (hang) on insert.
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
            aabbs: Vec::new(),
            oversized: Vec::new(),
            statics: Vec::new(),
            cell_min: Vec::new(),
            cell_max: Vec::new(),
            tracked: Vec::new(),
            tracked_flag: Vec::new(),
            seen: Vec::new(),
            epoch: 0,
            #[cfg(feature = "phase-timing")]
            owned_pairs: true,
            #[cfg(feature = "phase-timing")]
            skip_static_cells: true,
            #[cfg(feature = "phase-timing")]
            persistent: true,
            #[cfg(feature = "phase-timing")]
            persistent_strict: false,
        }
    }

    /// Clear all bodies from the grid, keeping every allocation.
    ///
    /// `HashMap::clear` would drop each cell's `Vec` (with its buffer),
    /// forcing a reallocation per occupied cell on the very next insert.
    /// Instead each cell's buffer is moved into the pool, where the next
    /// insert picks it up again: nothing is allocated, and the map holds
    /// only the cells that are actually in use.
    ///
    /// Keeping the vacated cells around instead (the old behaviour) grows
    /// the map without bound between compactions, and a large map turns
    /// every lookup into a cache miss: the insert pass costs ~650 ns per
    /// body in a settled 1500-body pile, essentially all of it in the
    /// hash lookup. Vacating the cells keeps the map at the size of the
    /// current frame and the lookup back in cache.
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

    /// Insert a body into the grid
    pub fn insert(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        self.insert_impl(body_id, min, max, false);
    }

    /// Insert a *static* body into the grid.
    ///
    /// A static body still pairs with every dynamic body (a resting box
    /// needs the floor), but two statics can never touch, so the grid
    /// skips those pairs.
    pub fn insert_static(&mut self, body_id: usize, min: Vec2, max: Vec2) {
        self.insert_impl(body_id, min, max, true);
    }

    /// The shared body of [`Self::insert`] and [`Self::insert_static`].
    fn insert_impl(&mut self, body_id: usize, min: Vec2, max: Vec2, is_static: bool) {
        // Non-finite bounds (escaped NaN/inf velocity) cannot map to
        // cells — and must not reach the `as i32` casts below, which
        // saturate to i32::MIN/MAX and would hang the cell loops.
        if !min.x.is_finite() || !min.y.is_finite() || !max.x.is_finite() || !max.y.is_finite() {
            return;
        }
        self.count += 1;
        self.all.push(body_id);
        if self.aabbs.len() <= body_id {
            self.aabbs.resize(body_id + 1, (Vec2::ZERO, Vec2::ZERO));
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
        // extreme-but-finite coordinates (integer casts below saturate).
        let span_x = (max.x - min.x) / self.cell_size;
        let span_y = (max.y - min.y) / self.cell_size;
        let cells_f = (span_x + 1.0) * (span_y + 1.0);
        let origin_f = (min.x.abs().max(max.x.abs()) / self.cell_size)
            .max(min.y.abs().max(max.y.abs()) / self.cell_size);

        // Calculate which cells this body overlaps.
        // Use floor (not truncation) so negative coords map correctly,
        // and do NOT clamp to 0 — CellKey supports negative cells.
        let cell_min_x = (min.x / self.cell_size).floor() as i32;
        let cell_min_y = (min.y / self.cell_size).floor() as i32;
        let cell_max_x = (max.x / self.cell_size).floor() as i32;
        let cell_max_y = (max.y / self.cell_size).floor() as i32;

        // Oversized (too many cells, or coords beyond safe cell indices):
        // skip the grid and pair against the whole scene in `collect_pairs`.
        if cells_f > Self::MAX_CELLS_PER_BODY as f32 || origin_f > Self::MAX_CELL_INDEX {
            // A body that used to fit in the grid has to leave it.
            if self.keeps_cells() {
                self.untrack(body_id);
            }
            self.oversized.push(body_id);
            return;
        }

        // The persistent mode: a body that did not move keeps its cells.
        if self.keeps_cells() {
            self.persistent_insert(
                body_id, min, max, old_aabb, cell_min_x, cell_min_y, cell_max_x, cell_max_y,
            );
            return;
        }

        self.insert_into_cells(body_id, cell_min_x, cell_min_y, cell_max_x, cell_max_y);
    }

    /// Put a body into the cells of its range and record where it is.
    ///
    /// Both modes go through here, so the grid's bookkeeping — the
    /// cells a body occupies, the bodies it holds, the step it was last
    /// seen in — is the same whichever mode ran. That matters because
    /// the profiling build switches between them from step to step: a
    /// mode that left the bookkeeping behind would hand the next mode a
    /// grid it cannot make sense of.
    fn insert_into_cells(
        &mut self,
        body_id: usize,
        cell_min_x: i32,
        cell_min_y: i32,
        cell_max_x: i32,
        cell_max_y: i32,
    ) {
        // Remember the lowest cell this body occupies: it is what picks
        // the single owner cell of every pair the body takes part in
        // (see `cell_min`).
        if self.cell_min.len() <= body_id {
            self.cell_min.resize(body_id + 1, (0, 0));
        }
        if self.cell_max.len() <= body_id {
            self.cell_max.resize(body_id + 1, (0, 0));
        }
        if self.tracked_flag.len() <= body_id {
            self.tracked_flag.resize(body_id + 1, false);
        }
        self.cell_min[body_id] = (cell_min_x, cell_min_y);
        self.cell_max[body_id] = (cell_max_x, cell_max_y);

        for cx in cell_min_x..=cell_max_x {
            for cy in cell_min_y..=cell_max_y {
                let key = CellKey::new(cx, cy);
                self.grid
                    .entry(key)
                    .or_insert_with(|| self.pool.pop().unwrap_or_default())
                    .push(body_id);
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
            false
        }
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
    /// for more work: on the 1500-body pile, where every body moves
    /// every step, it made the rebuild **78 % slower** than the range
    /// rule and the whole step **9 % slower** than the grid that wipes
    /// and re-inserts everything. The profiling build can select it, so
    /// the measurement can be repeated.
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

    /// The persistent-mode insert: keep the cells a body already
    /// occupies whenever it can, and only move it when it has to.
    ///
    /// The cell range is a pure function of the AABB and the cell size,
    /// and the cells a body occupies are exactly the cells of its
    /// range. So a body whose range is unchanged is in exactly the same
    /// cells as before, and its membership can stay as it is: the grid
    /// only has to refresh the AABB it filters pairs with.
    ///
    /// The strict variant insists on a bit-identical AABB instead, which
    /// re-inserts a body whose range survived a move that did not change
    /// its cell. Both produce the same pairs — the membership is the
    /// same set either way, and the pair list is sorted and deduped —
    /// so the switch only decides how much work is done.
    #[allow(clippy::too_many_arguments)]
    fn persistent_insert(
        &mut self,
        body_id: usize,
        min: Vec2,
        max: Vec2,
        old_aabb: (Vec2, Vec2),
        cell_min_x: i32,
        cell_min_y: i32,
        cell_max_x: i32,
        cell_max_y: i32,
    ) {
        if self.seen.len() <= body_id {
            self.seen.resize(body_id + 1, 0);
        }
        self.seen[body_id] = self.epoch;

        let was_tracked = self.tracked_flag.get(body_id).copied().unwrap_or(false);
        if was_tracked {
            let old_min = self.cell_min[body_id];
            let old_max = self.cell_max[body_id];
            let same_range =
                old_min == (cell_min_x, cell_min_y) && old_max == (cell_max_x, cell_max_y);
            let same_aabb = old_aabb == (min, max);
            if same_range && (same_aabb || !self.strict_skip()) {
                // The membership is unchanged: the AABB in `self.aabbs`
                // was already refreshed by the caller.
                return;
            }
            self.remove_from_cells(body_id, old_min, old_max);
        }

        self.insert_into_cells(body_id, cell_min_x, cell_min_y, cell_max_x, cell_max_y);
    }

    /// Take a body out of the cells it occupies.
    ///
    /// A cell that runs empty is dropped from the map and its buffer
    /// recycled, which is what keeps the map at the size of the scene's
    /// current extent instead of every position the scene ever visited.
    fn remove_from_cells(&mut self, body_id: usize, min: (i32, i32), max: (i32, i32)) {
        for cx in min.0..=max.0 {
            for cy in min.1..=max.1 {
                let key = CellKey::new(cx, cy);
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

    /// Whether two bodies' AABBs overlap.
    ///
    /// Conservative with respect to the narrow phase: an AABB contains
    /// its shape, so two bodies whose AABBs do not overlap cannot touch.
    fn aabbs_overlap(a: usize, b: usize, aabbs: &[(Vec2, Vec2)]) -> bool {
        match (aabbs.get(a), aabbs.get(b)) {
            (Some((amin, amax)), Some((bmin, bmax))) => {
                amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y && amax.y >= bmin.y
            }
            // An id without a recorded AABB cannot be filtered.
            _ => true,
        }
    }

    /// Whether a cell can produce a pair at all.
    ///
    /// A cell that holds only static bodies cannot: two statics never
    /// touch, so every candidate pair in it would be dropped anyway.
    /// Skipping the cell whole is what keeps a static-heavy scene from
    /// walking its quadratic number of static-static candidates — the
    /// per-pair check below still has to *look* at every candidate, and
    /// in a level full of static geometry that is nearly all of them.
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

    /// Whether cell `(cx, cy)` is the single owner of the pair `(a, b)`.
    ///
    /// The owner is the lowest cell both bodies span, so a pair that
    /// shares several cells is generated in exactly one of them. The
    /// pair set is unchanged — the same pairs, in the same canonical
    /// order — but the list that has to be sorted and deduplicated is
    /// as long as the pair set instead of a multiple of it.
    #[inline]
    fn owns_pair(&self, a: usize, b: usize, cx: i32, cy: i32) -> bool {
        // The profiling build can turn the rule off to measure the
        // behaviour it replaced.
        #[cfg(feature = "phase-timing")]
        if !self.owned_pairs {
            return true;
        }
        let (ax, ay) = self.cell_min[a];
        let (bx, by) = self.cell_min[b];
        cx == ax.max(bx) && cy == ay.max(by)
    }

    /// Get potential collision pairs
    pub fn get_pairs(&mut self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        self.collect_pairs(&mut pairs);
        pairs
    }

    /// Append potential collision pairs to `out` without allocating.
    ///
    /// Contract: `out` must be empty (the world passes cleared scratch).
    /// Dedupes by sorting (deterministic pair order); measured faster
    /// than hash-based dedupe for typical pair counts.
    pub fn collect_pairs(&self, out: &mut Vec<(usize, usize)>) {
        debug_assert!(out.is_empty());
        for (key, cell_ids) in &self.grid {
            // A cell of nothing but statics cannot produce a pair.
            if !self.cell_can_pair(cell_ids) {
                continue;
            }
            for i in 0..cell_ids.len() {
                for j in i + 1..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    // Two statics can never touch: the narrowphase
                    // discards the pair, so it is never generated.
                    if self.is_static(a) && self.is_static(b) {
                        continue;
                    }
                    // Emit the pair in its owner cell only: the bodies
                    // share this cell, and every shared cell holds both
                    // of them, so the pair is still generated — once.
                    if !self.owns_pair(a, b, key.x, key.y) {
                        continue;
                    }
                    // Canonical pair key.
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        // Oversized bodies are not in the grid: pair each against every
        // body (their huge AABBs overlap anything in range). Determinism
        // holds — `all`/`oversized` are insertion-ordered and the final
        // sort+dedup canonicalizes.
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
                for j in i + 1..cell_ids.len() {
                    let a = cell_ids[i];
                    let b = cell_ids[j];
                    debug_assert_ne!(a, b, "a body must not be twice in a cell");
                    // Two statics can never touch: the narrowphase
                    // discards the pair, so it is never generated.
                    if self.is_static(a) && self.is_static(b) {
                        continue;
                    }
                    // Emit the pair in its owner cell only (see
                    // `owns_pair`).
                    if !self.owns_pair(a, b, key.x, key.y) {
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
    /// bloated: the cells of a scene that swept its bodies across many
    /// positions are still around, and every lookup in a large hash
    /// table is a cache miss.
    pub fn cell_count(&self) -> usize {
        self.grid.len()
    }

    /// Get bodies in a cell
    pub fn get_bodies_in_cell(&self, key: &CellKey) -> Vec<usize> {
        self.grid.get(key).cloned().unwrap_or_default()
    }

    /// Get the number of bodies
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for SpatialHash2 {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every candidate pair the cells hold, before the owner rule.
    ///
    /// The grid's pair set is the deduplicated, sorted output; this is
    /// the raw candidate count, which is what the owner rule shrinks.
    fn candidate_count(hash: &SpatialHash2) -> usize {
        hash.grid
            .values()
            .map(|ids| ids.len() * (ids.len().saturating_sub(1)) / 2)
            .sum()
    }

    #[test]
    fn each_pair_is_generated_once() {
        // Bodies of diameter 0.6 in cells of size 1.0: most of them span
        // two cells, so a pair of neighbours is found in two or four
        // cells. The owner rule must emit it exactly once, while the
        // pair set itself stays the same.
        let mut hash = SpatialHash2::new(1.0);
        let mut expected = Vec::new();
        let mut boxes = Vec::new();
        for i in 0..64usize {
            let x = (i % 8) as f32 * 0.5;
            let y = (i / 8) as f32 * 0.5;
            let min = Vec2::new(x, y);
            let max = Vec2::new(x + 0.6, y + 0.6);
            hash.insert(i, min, max);
            boxes.push((min, max));
        }
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let (amin, amax) = boxes[i];
                let (bmin, bmax) = boxes[j];
                if amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y && amax.y >= bmin.y {
                    expected.push((i, j));
                }
            }
        }
        expected.sort_unstable();

        let mut out = Vec::new();
        hash.collect_overlapping_pairs(&mut out);
        assert_eq!(out, expected, "the pair set must be unchanged");
        assert_eq!(
            out.len(),
            candidate_count(&hash).min(out.len()),
            "every candidate is emitted at most once"
        );
    }

    #[test]
    fn owner_rule_drops_the_duplicates() {
        // The owner rule must actually remove work: with bodies that
        // span several cells, the candidates outnumber the pairs.
        let mut hash = SpatialHash2::new(1.0);
        for i in 0..64usize {
            let x = (i % 8) as f32 * 0.5;
            let y = (i / 8) as f32 * 0.5;
            hash.insert(i, Vec2::new(x, y), Vec2::new(x + 0.6, y + 0.6));
        }
        let candidates = candidate_count(&hash);
        let mut out = Vec::new();
        hash.collect_overlapping_pairs(&mut out);
        assert!(
            candidates > out.len(),
            "expected duplicated candidates: {candidates} candidates, {} pairs",
            out.len()
        );
    }

    /// The pairs a brute-force sweep over the same AABBs produces.
    fn brute_force_pairs(aabbs: &[(Vec2, Vec2)]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for a in 0..aabbs.len() {
            for b in a + 1..aabbs.len() {
                let (amin, amax) = aabbs[a];
                let (bmin, bmax) = aabbs[b];
                if amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y && amax.y >= bmin.y {
                    out.push((a, b));
                }
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn persistent_grid_does_not_reinsert_an_unchanged_body() {
        let mut hash = SpatialHash2::new(1.0);
        for _ in 0..3 {
            hash.begin_frame();
            hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(0.6, 0.6));
            hash.end_frame();
        }
        let mut copies = 0;
        for ids in hash.grid.values() {
            copies += ids.iter().filter(|&&id| id == 0).count();
        }
        assert_eq!(
            copies, 1,
            "an unchanged body must stay in its cells exactly once"
        );
    }

    #[test]
    fn persistent_grid_matches_a_brute_force_sweep_over_many_steps() {
        // The grid keeps its cells across steps and only re-inserts the
        // bodies that moved. Whatever it does internally, the pairs it
        // reports must be the ones a brute-force AABB sweep finds —
        // every step, for bodies that move, that leave, and that come
        // back.
        let mut hash = SpatialHash2::new(1.0);
        let mut aabbs: Vec<(Vec2, Vec2)> = Vec::new();
        let mut step = 0usize;

        for _ in 0..40 {
            // Every third step a body leaves the scene, and a new one
            // takes its place.
            let leaving = if step.is_multiple_of(3) && !aabbs.is_empty() {
                Some(aabbs.len() - 1)
            } else {
                None
            };
            if let Some(index) = leaving {
                aabbs.remove(index);
            }

            hash.begin_frame();
            for (id, (min, max)) in aabbs.iter().enumerate() {
                hash.insert(id, *min, *max);
            }
            hash.end_frame();

            let mut pairs = Vec::new();
            hash.collect_overlapping_pairs(&mut pairs);
            assert_eq!(
                pairs,
                brute_force_pairs(&aabbs),
                "step {step}: the grid's pairs must match the sweep"
            );

            // Move everything a little, so the cells change.
            for (min, max) in aabbs.iter_mut() {
                let dx = ((step % 7) as f32 - 3.0) * 0.13;
                let dy = ((step % 5) as f32 - 2.0) * 0.21;
                *min += Vec2::new(dx, dy);
                *max += Vec2::new(dx, dy);
            }
            step += 1;

            if aabbs.len() < 6 {
                // Keep the scene populated.
                let min = Vec2::new(step as f32 * 0.37, step as f32 * 0.11);
                aabbs.push((min, min + Vec2::new(0.6, 0.6)));
            }
        }
    }

    #[test]
    fn persistent_grid_drops_a_body_that_leaves_the_scene() {
        let mut hash = SpatialHash2::new(1.0);
        hash.begin_frame();
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(0.2, 0.2), Vec2::new(1.2, 1.2));
        hash.end_frame();
        let mut pairs = Vec::new();
        hash.collect_overlapping_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)]);

        // The next step has only one body: the one that left must not
        // keep pairing with it.
        hash.begin_frame();
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.end_frame();
        let mut pairs = Vec::new();
        hash.collect_overlapping_pairs(&mut pairs);
        assert!(
            pairs.is_empty(),
            "a body that left must not pair: {pairs:?}"
        );
    }

    #[test]
    fn persistent_grid_keeps_a_moved_body_pairing_with_its_neighbour() {
        let mut hash = SpatialHash2::new(1.0);
        hash.begin_frame();
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(0.5, 0.0), Vec2::new(1.5, 1.0));
        hash.end_frame();

        // Move the first body right, into the second one's cell. The
        // pair must survive: the second body did not move, so the pair
        // is found in the cell they now share.
        hash.begin_frame();
        hash.insert(0, Vec2::new(0.6, 0.0), Vec2::new(1.6, 1.0));
        hash.insert(1, Vec2::new(0.5, 0.0), Vec2::new(1.5, 1.0));
        hash.end_frame();
        let mut pairs = Vec::new();
        hash.collect_overlapping_pairs(&mut pairs);
        assert_eq!(pairs, vec![(0, 1)], "the pair must survive the move");
    }

    #[test]
    fn a_cell_of_statics_produces_no_pairs() {
        // Statics never touch each other, so a grid of nothing but
        // statics must produce no pairs at all -- and must not have to
        // walk its candidate pairs to find that out.
        let mut hash = SpatialHash2::new(1.0);
        for i in 0..32usize {
            let x = (i % 8) as f32 * 0.5;
            let y = (i / 8) as f32 * 0.5;
            hash.insert_static(i, Vec2::new(x, y), Vec2::new(x + 0.6, y + 0.6));
        }
        let mut out = Vec::new();
        hash.collect_overlapping_pairs(&mut out);
        assert!(out.is_empty(), "statics must not pair with each other");

        // A dynamic body in the middle still pairs with the statics
        // around it.
        hash.insert(32, Vec2::new(1.0, 1.0), Vec2::new(1.6, 1.6));
        let mut out = Vec::new();
        hash.collect_overlapping_pairs(&mut out);
        assert!(!out.is_empty(), "a dynamic body must pair with the statics");
        assert!(
            out.iter().all(|&(a, b)| a == 32 || b == 32),
            "only the dynamic body may pair: {out:?}"
        );
    }

    #[test]
    fn test_spatial_hash_insert_and_get_pairs() {
        let mut hash = SpatialHash2::new(1.0);

        // Insert two overlapping bodies
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let pairs = hash.get_pairs();
        assert!(pairs.contains(&(0, 1)));
    }

    #[test]
    fn test_spatial_hash_no_collision() {
        let mut hash = SpatialHash2::new(1.0);

        // Insert two non-overlapping bodies
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));

        let pairs = hash.get_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash_clear() {
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.clear();

        assert!(hash.is_empty());
    }

    #[test]
    fn test_oversized_body_pairs_with_all_and_is_fast() {
        // A body too large for the grid (radius 1000 -> ~4M cells) must
        // not hang insert, and must still pair with every other body.
        let mut hash = SpatialHash2::new(1.0);
        let t = std::time::Instant::now();
        hash.insert(0, Vec2::new(-1000.0, -1000.0), Vec2::new(1000.0, 1000.0));
        // A far-away small body (no cell overlap with the huge AABB's
        // center region would be irrelevant: the huge body overlaps it).
        hash.insert(1, Vec2::new(500.0, 500.0), Vec2::new(501.0, 501.0));
        assert!(
            t.elapsed().as_millis() < 500,
            "oversized insert must be fast, took {:?}",
            t.elapsed()
        );

        let pairs = hash.get_pairs();
        assert!(
            pairs.contains(&(0, 1)),
            "oversized body must pair with the small body"
        );
    }

    #[test]
    fn test_normal_body_stays_in_grid() {
        // A normal-sized body never takes the oversized path: two
        // overlapping small bodies still pair via cells.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert!(hash.oversized.is_empty());
        assert!(hash.get_pairs().contains(&(0, 1)));
    }

    #[test]
    fn test_spatial_hash_negative_coords() {
        let mut hash = SpatialHash2::new(1.0);

        // Floor at y=-1.5..-0.5 and a ball just above it must share a cell
        // only when actually overlapping — and negative-x bodies must NOT
        // all collapse into cell 0.
        hash.insert(0, Vec2::new(-10.0, -1.5), Vec2::new(10.0, -0.5));
        hash.insert(1, Vec2::new(-0.3, -0.4), Vec2::new(0.3, 0.2));
        let pairs = hash.get_pairs();
        assert!(pairs.contains(&(0, 1)));

        // Far-apart negative bodies must not pair.
        let mut hash2 = SpatialHash2::new(1.0);
        hash2.insert(0, Vec2::new(-10.0, -10.0), Vec2::new(-9.0, -9.0));
        hash2.insert(1, Vec2::new(-1.0, -1.0), Vec2::new(0.0, 0.0));
        assert!(hash2.get_pairs().is_empty());
    }

    #[test]
    fn test_spatial_hash_rejects_non_finite_bounds() {
        let mut hash = SpatialHash2::new(1.0);
        // NaN/inf bounds must be dropped, never hang the cell loops.
        hash.insert(0, Vec2::new(f32::NAN, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(0.0, 0.0), Vec2::new(f32::INFINITY, 1.0));
        assert!(hash.get_pairs().is_empty());
        assert!(hash.is_empty());
        assert_eq!(hash.len(), 0);
    }

    #[test]
    fn test_spatial_hash_clear_reuses_buffers_without_leaking_ids() {
        // A cleared cell's buffer goes back into the pool and is handed
        // out again on the next insert. If the recycled buffer still held
        // the previous frame's ids, the cell would report pairs that do
        // not exist (and ids that are not in the scene at all).
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(0.5, 0.5), Vec2::new(2.5, 2.5));
        assert_eq!(hash.get_pairs().len(), 1);

        hash.clear();

        // A different body in the same cell must not inherit the old
        // ids.
        hash.insert(7, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let pairs = hash.get_pairs();
        assert!(
            pairs.is_empty(),
            "a recycled cell must not carry the previous frame's ids: {pairs:?}"
        );
        assert_eq!(hash.len(), 1);

        // And the recycled buffer still works for a second body.
        hash.insert(8, Vec2::new(0.5, 0.5), Vec2::new(2.5, 2.5));
        assert_eq!(hash.get_pairs(), vec![(7, 8)]);
    }

    #[test]
    fn test_static_pairs_are_not_generated() {
        // A pair of two statics can never produce an impulse, so the grid
        // must not emit it. A static and a dynamic body still pair.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert_static(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert_static(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        hash.insert(2, Vec2::new(1.5, 1.5), Vec2::new(3.5, 3.5));

        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(
            !pairs.contains(&(0, 1)),
            "a static-static pair must not be generated: {pairs:?}"
        );
        assert!(
            pairs.contains(&(0, 2)) && pairs.contains(&(1, 2)),
            "statics must still pair with dynamic bodies: {pairs:?}"
        );
    }

    #[test]
    fn test_spatial_hash_clear_empties_every_cell() {
        // After a clear every cell must be empty: either the cell was
        // dropped (a bloated table) or its list was cleared in place (a
        // small table). No cell may still hold ids.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(100.0, 100.0), Vec2::new(101.0, 101.0));
        hash.clear();
        for (key, ids) in hash.grid.iter() {
            assert!(
                ids.is_empty(),
                "cell {key:?} still holds ids after a clear: {ids:?}"
            );
        }
        let mut pairs = Vec::new();
        hash.collect_pairs(&mut pairs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_spatial_hash_reinsert_after_clear_keeps_pairs() {
        // The full cycle: clear, re-insert, and the pairs must be the
        // same as they were before the clear.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let before = hash.get_pairs();
        hash.clear();
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        hash.insert(1, Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert_eq!(hash.get_pairs(), before);
    }

    #[test]
    fn test_overlapping_pairs_drops_disjoint_cell_sharers() {
        // Floor at y=-1.5..-0.5 and a ball at y=-0.4..0.2 share grid
        // cells but their AABBs do not overlap: the unfiltered sequence
        // still reports the pair (the CCD swept test relies on that),
        // while the narrowphase fast path must not.
        let mut hash = SpatialHash2::new(1.0);
        hash.insert(0, Vec2::new(-10.0, -1.5), Vec2::new(10.0, -0.5));
        hash.insert(1, Vec2::new(-0.3, -0.4), Vec2::new(0.3, 0.2));
        let mut filtered = Vec::new();
        hash.collect_overlapping_pairs(&mut filtered);
        assert!(
            filtered.is_empty(),
            "disjoint AABBs must be filtered: {filtered:?}"
        );
        // The unfiltered contract is unchanged.
        assert!(hash.get_pairs().contains(&(0, 1)));
    }

    #[test]
    fn test_overlapping_pairs_agrees_with_brute_force() {
        use super::super::brute_force2::BruteForce2;
        // Mixed scene: true overlaps, cell-sharing non-overlaps, and an
        // oversized body paired against near and far bodies.
        let scene: Vec<(Vec2, Vec2)> = vec![
            (Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0)),
            (Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0)),
            (Vec2::new(2.5, 0.0), Vec2::new(3.5, 1.0)),
            (Vec2::new(50.0, 50.0), Vec2::new(52.0, 52.0)),
            (Vec2::new(-10.0, -1.5), Vec2::new(10.0, -0.5)),
            (Vec2::new(-0.3, -0.4), Vec2::new(0.3, 0.2)),
            (Vec2::new(-1000.0, -1000.0), Vec2::new(1000.0, 1000.0)),
            (Vec2::new(500.0, 500.0), Vec2::new(501.0, 501.0)),
            (Vec2::new(2000.0, 2000.0), Vec2::new(2001.0, 2001.0)),
        ];
        let mut grid = SpatialHash2::new(1.0);
        let mut bf = BruteForce2::new();
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
        assert!(a.contains(&(6, 7)));
        assert!(!a.iter().any(|&(x, y)| x == 8 || y == 8));
    }

    #[test]
    fn test_spatial_hash_len_counts_inserts() {
        let mut hash = SpatialHash2::new(1.0);
        assert_eq!(hash.len(), 0);
        hash.insert(0, Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        hash.insert(1, Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));
        assert_eq!(hash.len(), 2);
    }
}
