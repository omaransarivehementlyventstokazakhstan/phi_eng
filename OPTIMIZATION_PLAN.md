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
- **Remaining**: the Barnes-Hut tree arena is pooled, but its per-kick
  `positions`/`masses` scratch vectors and the per-leaf index buffers
  still allocate per kick.

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
- **Remaining**: the solver / broadphase / narrowphase hot paths still
  use the scalar code; the batched primitives are not called from them
  yet, so the feature currently changes no trajectory.

### 2.3 Fixed-Point Math Option
- `FixedPoint` feature flag
- `i32`/`i64` based with 16.16 or 32.32 format
- For ARM Cortex-M, RISC-V without FPU

### 2.4 Temporal Coherence / Warm-Starting
- Persist contact points across frames (already done)
- Persist joint impulses (already done)
- Add **constraint graph caching** for static scenes

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
pub struct PhysicsConfig {
    pub solver_quality: SolverQuality,
    pub collision_tier: CollisionTier,
    pub broadphase_mode: BroadphaseMode,
    pub max_bodies: usize,
    pub max_contacts_per_step: usize,
    pub enable_sleep: bool,
    pub enable_ccd: bool,
    pub fixed_point: bool,
    pub simd: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        // Auto-detect or sensible defaults
    }
}
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