# Low-End Device Optimization Plan for phieng

## Executive Summary

Target: Run 60 FPS on devices with 1-2 CPU cores, 512MB-1GB RAM, no SIMD/FPU (e.g., Raspberry Pi Zero, old phones, embedded devices).

Current baseline: ~2ms/step for 1500 bodies (2D) on desktop. Target: <16ms/step for 500 bodies on low-end.

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
- **Remaining**: an actual persistent static constraint graph, or
  static-awareness in the broadphase.
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