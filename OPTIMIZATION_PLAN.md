# Low-End Device Optimization Plan for phieng

## Executive Summary

Target: Run 60 FPS on devices with 1-2 CPU cores, 512MB-1GB RAM, no SIMD/FPU (e.g., Raspberry Pi Zero, old phones, embedded devices).

Current baseline: ~2ms/step for 1500 bodies (2D) on desktop. Target: <16ms/step for 500 bodies on low-end.

### How to measure a change here

Absolute timings on a loaded machine drift by ±20 %, and two builds of
**identical** source differ by 6–10 % per phase because the code layout
moved — a fixed property of each binary, so it never averages out. A
cross-binary A/B therefore cannot resolve anything below ~15 %, and a
before/after comparison minutes apart is not evidence at all.

The harness that removes the confounds lives in
`crates/engine/examples/bench_solver_ab.rs`: **one binary** (so no
layout difference), **one world** (so no heap-layout difference), and
the two variants **alternating every 20 steps** (so the pile's own
settling drift hits both equally). `PERF_BASELINE.md` § "Measuring on
this machine" has the full table of confounds and their fixes. Every
"measured" number below that is quoted with the method that produced
it.

---

## Priority 1: Zero-Cost / High-Impact Optimizations (Week 1-2)

### 1.1 Object Pooling for Zero Allocations — ✅ DONE (0.4.x)
- **Target**: Zero allocations/step after warmup
- **Areas**: Contact constraints, broadphase pairs, narrowphase scratch buffers, joint arrays
- **Implementation**: Pre-allocated `Vec` pools with `clear()` reuse, `Pool<T>` wrapper
- **Shipped**: the broadphase grid (`SpatialHash2`), the contact cache
  (`World2/3::contact_cache`), and the SPH fluid neighbor grids
  (`FluidState2/3`) now **recycle vacated per-cell / per-key `Vec`
  buffers through internal pools** instead of freeing them. Measured:
  dynamic pile 1500 bodies **~467 → 0.8 allocs/step** (2D), **14 → 0.0**
  (3D); resting stack 0.00. Memory-only; `contact_cache.len()` stays
  exact and every determinism hash is unchanged.
- **Shipped (parallel islands)**: the island partition is rebuilt every
  step (bodies are added, removed and reindexed), so its buffers are
  pooled in `island::IslandScratch` (owned by the world's scratch) and
  the grouping is done with a counting sort over the union-find roots —
  no per-island index `Vec` is ever created. The parallel solver also
  runs on a **fixed worker pool** (capped at the available parallelism)
  instead of one OS thread per island: a settled pile has ~140 islands,
  and spawning ~140 threads per step cost more than the solve it
  parallelised (the parallel path was measurably *slower* than the
  serial one). Measured on the 1500-body pile: **785 → 0.3
  allocs/step**, i.e. the parallel path now matches the serial path's
  allocation floor, and `serial_hash == parallel_hash` still holds.
  Below 256 island constraints the serial path is used (same result,
  no thread overhead).
- **Shipped (Barnes-Hut)**: the tree arena, the per-leaf index buffers
  and the `positions`/`masses` scratch are all pooled in the thread-local
  tree pool, and the "move the bodies down" step now reads the leaf's
  list *in place* (an internal node's list is never read again) instead
  of taking and freeing it. Measured on the 1500-body pile with
  Barnes-Hut gravity: **2473 → 17 allocations per kick**. The tree
  structure and the per-leaf ordering are unchanged, so the gravity
  trajectories (and their determinism hashes) are identical.

### 1.2 Configurable Solver Quality Tiers
```rust
pub enum SolverQuality {
    Ultra,    // 8 iterations, full friction, CCD
    High,     // 6 iterations, full friction
    Medium,   // 4 iterations, simplified friction
    Low,      // 2 iterations, no friction, no restitution
    Minimal,  // 1 iteration, position correction only
}
```
- Map to `Solver.iterations`, `Solver.enable_friction`, `Solver.enable_restitution`, `Solver.enable_ccd`
- Default: `High` (desktop), `Low` (auto-detect low-end)

### 1.3 Collision Detection Tier System
```rust
pub enum CollisionTier {
    Tier1_AABBOnly,      // AABB overlap only (fastest)
    Tier2_SphereAABB,    // Sphere vs AABB (fast)
    Tier3_SphereSphere,  // Sphere vs Sphere (fast)
    Tier4_BoxBox,        // SAT OBB-OBB (medium)
    Tier5_GJK_EPA,       // Convex vs Convex (slow)
}
```
- Early exit at lowest tier that separates
- Configurable max tier per quality setting

### 1.4 Sleep/Wake Optimization
- **Island-based sleeping**: Only wake islands with external forces
- **Sleep counters**: Deferred sleep (wait N frames before sleeping)
- **Velocity threshold scaling**: Scale with mass for stability
- **Static body exclusion**: Never process static bodies in solver

---

## Priority 2: Medium-Impact Optimizations (Week 3-4)

### 2.1 Broadphase Tier System — ✅ DONE (0.5.x)
```rust
pub enum BroadphaseMode {
    BruteForce,     // < 50 bodies
    SpatialHash,    // 50-5000 bodies (current)
    DynamicAABBTree, // 5000+ bodies
    Hybrid,         // Auto-select
}
```
- **Shipped**: `BroadphaseMode` with `BruteForce` (explicit all-pairs),
  `SpatialHash` (existing grid), `DynamicAABBTree` (new balanced tree
  for 2D and 3D, bottom-up rebuild past the 2×log₂ threshold), and
  `Hybrid` (auto-select from live body count each step). Modes are
  behavior-identical (canonical sorted+deduped pairs); default stays the
  grid; mode persists in snapshot schema v8. FFI + managed C# mirror
  included. Pinned by B01–B06 (Rust↔C# parity).

### 2.2 SIMD Math Paths (Rust) — 🟡 PARTIAL (primitives shipped, hot paths not wired)
- Use `packed_simd` or `std::simd` for Vec2/Vec3/Mat2/Mat3
- Feature flag: `simd` (disabled by default for compatibility)
- **Shipped**: `phieng-math` has a `simd` feature with `std::arch`
  batched primitives (f32x4/f32x8) behind it, off by default.
- **Measured, not shipped (narrowphase)**: the 2D narrowphase projection
  (`crates/collision/src/sat.rs`, `project_polygon`) was the best
  candidate - it is a two-term dot per vertex, so a batched form can be
  made *bit-identical* (the SSE horizontal add preserves the
  `ax*vx + ay*vy` order per lane). Measured on the reference
  micro-benchmark it is **slower than the scalar loop it would
  replace**: 0.83x at 4 vertices, 0.97x at 8, 0.59x at 16, 0.47x at 32.
  The scalar `axis.dot(v)` loop is already efficient, and the
  horizontal add plus the per-chunk overhead cost more than they save.
  So no feature flag, no `unsafe` pointer cast, and no code change -
  the finding is recorded here so it is not re-attempted blind.
- **Remaining**: the real opportunity is an **SoA-backed** hot path
  (the engine is AoS today: `RigidBody2` interleaves `velocity` with
  20+ other fields, so 4 consecutive bodies' velocities are ~152
  bytes apart and there is no contiguous `[f32; 8]` to load). The
  experimental SoA prototype is present but disabled
  (`crates/dynamics/src/lib.rs`). The solver itself cannot be batched
  at all: it is strict Gauss-Seidel (each constraint observes the
  previous one's impulse writes), so batching there would change the
  trajectory. **SoA data layout is the remaining work, not SIMD
  intrinsics.**

### 2.3 Fixed-Point Math Option
- `FixedPoint` feature flag
- `i32`/`i64` based with 16.16 or 32.32 format
- For ARM Cortex-M, RISC-V without FPU

### 2.4 Temporal Coherence / Warm-Starting — 🟡 PARTIAL
- Persist contact points across frames (already done)
- Persist joint impulses (already done)
- Add **constraint graph caching** for static scenes — **measured, and the
  obvious version of it does not pay**
  - Moving the existing "static-static pairs cannot produce an impulse"
    check from the narrowphase loop to the broadphase output is
    bit-identical (the surviving pairs keep their relative order) and
    looks like a clear win for static-heavy scenes. Measured on the new
    dense static benchmark (400 overlapping static boxes + 20 dynamic
    bodies, ~1.1 ms/step): **1.106 → 1.106 ms/step — no change.**
  - The reason: the narrowphase *already* skipped those pairs, so the
    only cost left is the broadphase *generating* them, and pruning
    after the fact cannot avoid that. A real win needs the broadphase
    itself not to emit them, which is a wider change (the broadphase is
    shared with the managed C# mirror and the B-tier tests).
  - The benchmark scene is kept: the repo had no static-heavy scenario
    to measure, and the next attempt needs one.
- **Measured, not shipped (rigid-body layout)**: the solver is the
  largest phase of the 2D pile (40 % of the step) and its inner loop
  reads position, velocity, angular velocity, inverse mass/inertia and
  the two flags for every constraint it touches. `RigidBody2` is 176
  bytes and the compiler's field reordering spread those hot fields
  across three cache lines (position @16, velocity @56, angular
  velocity @96, flags @168). Two layouts were measured:
  - `repr(C)` with the hot fields declared first: the hot block packs
    into bytes 0–58 (one line), but the struct grows to 192 bytes with
    16-byte alignment, so the body array no longer packs as tightly.
    Interleaved A/B (two binaries, nine alternating rounds, same
    machine): pile 2D iters=8 **0.920 → 1.162 ms/step (+26 %)**,
    iters=1 +8.8 %, the settled census unchanged (−0.6 %).
  - `repr(C, align(64))`: the body starts on a cache-line boundary, so
    the hot block never straddles two lines. Interleaved A/B: pile 2D
    iters=1 −3.7 %, iters=8 +3.2 %, census −2.6 %, islands serial
    +7.0 %, pile 3D −0.5 % — inside the noise, and the 192-byte body
    costs 9 % more memory.
  So the cache-line win does not pay for the density loss on this
  workload. A re-measurement of the `repr(C)` hot-first variant with
  the better harness (one binary, one heap layout, the two paths
  alternating every 20 steps) confirmed it: **every phase** was 1–7 %
  *slower* with the reordered body, and the whole step 4 % slower.
  The field reorder stays rejected.
- **Shipped (hot-field side array)**: the remaining lever, and the one
  that does not pay for density. Instead of reordering `RigidBody2`,
  the solver gathers the fields it actually reads — position, velocity,
  angular velocity, inverse mass, inverse inertia and the two frozen
  flags — into one cache-line-aligned 32-byte record per body, solves
  on that, and writes the velocities back. The body array keeps its
  176-byte layout; only the solver's *view* of it is packed, so the
  solver's working set for the 1500-body pile drops from 264 KB to
  48 KB. The physics has a single implementation (a `SolveBody` trait
  the fat body and the packed record both implement), the same values
  are copied, and the same arithmetic runs in the same order, so every
  pinned hash is unchanged. Measured with the same harness: the solver
  phase **−6 % to −10 %**, the whole step **−2 % to −5.5 %**.
- **Shipped (grid hashing)**: the contact cache already used FxHash —
  a single multiply-rotate per word — because its keys are internal
  `(body_a, body_b)` pairs. The broad-phase grid had the same kind of
  key (an internal `CellKey` of two `i32`s) but used std's default
  SipHash, and the grid is rebuilt every step: 1500 bodies × up to 4
  cells each is ~6000 hash lookups per step, each a random probe into a
  large table. FxHash now lives in `phieng-math` so both crates share
  it. The grid's *output* is unaffected — `collect_pairs` sorts and
  dedupes, so the emitted sequence is a canonical set and does not
  depend on the hash order — and every pinned hash is unchanged.
  Measured on the rebuild phase, the phase the hasher touches (three
  consecutive runs each): **0.230 / 0.245 / 0.292 ms (SipHash)** →
  **0.170 / 0.185 / 0.188 ms (FxHash)**. Interleaved A/B over 13
  alternating rounds: islands serial −12.6 %, islands parallel
  −18.5 %, pile 3D −9.5 %; the 1500-body pile is solver-bound, so
  its grid is a smaller share.
- **Shipped (static-pair pruning)**: the static-heavy scene generated
  **8 010** candidate pairs to end up with **14** live ones, and the
  generation was 91 % of the step. Every one of those pairs was static
  against static: the narrowphase discards them immediately (a static
  body has infinite mass, so the constraint would solve to zero), so
  the work was thrown away one step later. The grid now learns which
  bodies are static (`insert_static`) and skips a candidate pair whose
  *both* ends are static — a static still pairs with every dynamic
  body, so only the pair of two statics is dropped. The surviving
  pairs are the same set in the same relative order (the emitted
  sequence is sorted and deduped), so the constraint stream, the
  contact cache and the trajectory are unchanged: every pinned hash
  is identical, verified with the hash guard on both sides of the A/B.
  The static scene's pair phase: **0.999 → 0.054 ms/step (18×)**,
  and its raw pair count from 8 010 to 18. Interleaved A/B over 9
  alternating rounds: islands serial −10.2 %, islands parallel
  −11.7 %, pile 3D −8.6 %, pile 2D −2 % to −3 %.
- **Shipped (warm-start lookup)**: the warm start looked the contact
  cache up twice per constraint — `contains_key` to learn whether the
  pair was new (and so might need a sleeper woken) and then `get` to
  find the cached contact to match. The first lookup folds into the
  second. The wake still happens before the solver runs and cannot
  change what the match found: it only clears sleep flags, and the
  cache holds the previous frame's contacts either way. Interleaved A/B:
  pile 2D settled census −12.8 %, islands serial −6.3 %, islands
  parallel −4.2 %, pile 2D iters=8 −1.9 %, pile 3D unchanged.
- **Measured, not shipped (grid cell size)**: the grid's cell (1.0) is
  much larger than the bodies (0.3), so a cell holds many mutually
  disjoint bodies: the pile examines **25 296** candidate pairs to
  produce 1 953. A smaller cell was tried (0.5) and it made things
  *worse* (41 879 candidates): a smaller cell means more cells, and a
  body then spans more of them, so the same pair is found through more
  cells. The candidate count is inherent to a uniform grid at this
  density; the AABB filter is what keeps it cheap, and it is already
  in place.
- **Remaining**: an actual persistent static constraint graph, or
  static-awareness in the broadphase.
- **Shipped (narrowphase input filtering)**: every broadphase backend
  now also offers `collect_overlapping_pairs` — the subset of candidate
  pairs whose AABBs overlap — and both `step_inner` narrowphase loops
  consume it instead of `collect_pairs`. Pairs with disjoint AABBs can
  never produce contacts (an AABB contains its shape), so the
  constraint stream is unchanged. Measured on the corrected trajectory
  (median of 7, hash guard green both ways): pile 2D iters=8
  **1.513 → 0.918 ms/step (−39 %)**, census **1.616 → 1.417 (−12 %)**,
  pile 3D iters=8 **0.238 → 0.139 (−42 %)**. The CCD swept-volume test
  deliberately keeps the unfiltered sequence: a currently-disjoint pair
  can still overlap once swept along the velocities.
- **Fixed (narrowphase correctness)**: `sat_circle_polygon` (Rust) and
  `SAT.CirclePolygon` (C#) tested only polygon face normals, so a
  circle grazing a polygon corner reported a phantom contact (face
  projections overlap while the corner axis separates). Both now also
  test the closest-vertex axis, separation-only: it can only turn a
  contact into a separation, never the reverse, and true contacts keep
  their exact normal and penetration. This moved the 2D pins
  deliberately (`0xc3f59dbc6e87256a` → `0xb12fcb9a19506312`,
  `0xc509c18cd50043b4` → `0x4acb6694f3a9917`); box-only and 3D scenes
  are unchanged. The pair filter above depends on this fix: without it
  the filter would drop pairs the old narrowphase turned into
  (phantom) contacts.

---

## Priority 3: Advanced Optimizations (Week 5+)

### 3.1 GPU Compute (Optional)
- Broadphase on GPU (compute shader)
- Narrowphase SAT on GPU
- Requires `wgpu`/`vulkano` feature

### 3.2 Continuous Collision Detection (CCD) Tiers — ✅ DONE (0.5.x)
- Off by default on low-end
- Swept AABB for fast movers only
- Full GJK-raycast only on High/Ultra quality
- **Shipped**: `CcdTier` (`Off` / `SweptAABB` / `FullGJK`) with
  `ccd_tier_from_quality` mapping (`Ultra`→Full, `High`→Swept, else
  Off). `SweptAABB` checks swept-volume overlap per broadphase pair and
  skips substepping entirely when nothing fast is near anything;
  `FullGJK` keeps the legacy peak-speed substepping. Default is `Off`,
  so existing trajectories are bit-identical. Tier persists in the v6
  `ccd` block and crosses the FFI + managed C# mirror.

### 3.3 Scene Graph / Instancing — ❌ NOT STARTED
- Shared shapes across bodies
- Instanced rendering data prep

---

## Configuration API — ✅ DONE (0.5.x, Rust core)

`PhysicsConfig` unifies `solver_quality`, `collision_tier`,
`broadphase_mode`, sleep thresholds, `ccd`, and `thermal` with
`Default` (= pre-tier-system behavior), `low_end()`, and
`high_fidelity()` presets plus `apply_to_2d`/`apply_to_3d`.
(`max_bodies`/`max_contacts` caps, `fixed_point`, and `simd` remain
future work — see §§2.2–2.3.)

```rust
// Shipped today (crates/dynamics/src/physics_config.rs):
pub struct PhysicsConfig {
    pub solver_quality: SolverQuality,
    pub collision_tier: CollisionTier,
    pub broadphase_mode: BroadphaseMode,
    pub enable_sleep: bool,
    pub linear_sleep_threshold: f32,
    pub angular_sleep_threshold: f32,
    pub ccd: CcdConfig,
    pub thermal: ThermalConfig2D,
}

// Still on the table (see §§2.2–2.3):
//   pub max_bodies: usize,
//   pub max_contacts_per_step: usize,
//   pub fixed_point: bool,
//   pub simd: bool,
```

---

## Metrics to Track

| Metric | Target (Low-End) | Measurement |
|--------|------------------|-------------|
| Step time (500 bodies) | < 16ms | `bench_large` |
| Allocations/step | 0 | Custom allocator |
| Memory footprint | < 50MB | `heaptrack` |
| Solver iterations avg | 2-4 | Stats |
| Sleep % of bodies | > 80% | Stats |
| Cache miss rate | < 5% | `perf` |

---

## Implementation Order

1. **Week 1**: Object pooling + Solver quality tiers
2. **Week 2**: Collision tiers + Sleep optimization
2. **Week 3**: Broadphase tiers + Config API
3. **Week 4**: Fixed-point + SIMD feature flags
4. **Week 5+**: Profiling, tuning, documentation

---

## Backward Compatibility

- All new features **opt-in** via `PhysicsConfig`
- Default behavior unchanged
- Existing API unchanged
- Feature flags for optional dependencies