# PhiEng

Deterministic 2D + 3D rigid-body physics with SPH fluids: a Rust core
with a mirroring C# implementation and a C ABI for P/Invoke.
Data-oriented, allocation-conscious, `f32` throughout, tick-based (no
wall-clock inside the simulation).

## Quickstart (5 minutes)

### Rust (crates.io)

```bash
# Add to your Cargo.toml
[dependencies]
phieng = "0.3.1"

# Or use the full workspace for all crates
phieng-math = "0.3.1"
phieng-collision = "0.3.1"
phieng-broadphase = "0.3.1"
phieng-dynamics = "0.3.1"
```

```rust
use phieng::{World2, Shape2};

fn main() {
    let mut world = World2::new();
    world.gravity = phieng::math::Vec2::new(0.0, -9.81);

    // Static floor
    world.add_static_body(
        phieng::math::Vec2::new(0.0, -1.0),
        Shape2::Box { half_extents: phieng::math::Vec2::new(10.0, 0.5) },
    ).unwrap();

    // Drop a box
    let _ = world.add_dynamic_body(
        phieng::math::Vec2::new(0.0, 5.0),
        Shape2::Box { half_extents: phieng::math::Vec2::new(0.5, 0.5) },
        1.0,
    );

    // Step simulation
    for _ in 0..300 {
        world.step(1.0 / 60.0);
    }
}
```

### C# (NuGet)

```bash
dotnet add package PhiEng
dotnet add package PhiEng.Native  # for native FFI bridge
```

```csharp
using PhiEng.Dynamics;
using PhiEng.Math;

var world = new World2();
world.Gravity = new Vector2(0, -9.81f);

world.AddStaticBody(new Vector2(0, -1), new BoxShape2(new Vector2(10, 0.5f)));
int ball = world.AddDynamicBody(new Vector2(0, 5), new CircleShape(0.5f), 1.0f);

for (int i = 0; i < 300; i++) world.Step(1.0f / 60.0f);

var pos = world.Bodies[ball].Transform.Position;
Console.WriteLine($"Ball at ({pos.X:F2}, {pos.Y:F2})");
```

### Native FFI (C/C++)

```bash
# Download from GitHub Releases or build from source
cargo build --target x86_64-pc-windows-gnu --release -p phieng
# → target/x86_64-pc-windows-gnu/release/phieng.dll
```

---

## Layout

| Rust (`crates/`) | C# (`src/`) | Role |
|---|---|---|
| `math` | `PhiEng.Math` | Vectors, quaternions, transforms, AABBs, inertia tensors |
| `collision` | `PhiEng.Collision` | 2D SAT, 3D SAT / GJK / EPA, contact generation |
| `broadphase` | `PhiEng.Broadphase` | Spatial hash (2D), flat AABB tree (3D) |
| `dynamics` | `PhiEng.Dynamics` | Bodies, materials, sequential-impulse solvers, worlds, 2D joints, snapshots |
| `engine` | `PhiEng` (root) + `PhiEng.Native` | Re-exports, C ABI (`phieng_world_*`, `phieng_world3_*`, `phieng_world_*_joint`) |

`tests/PhiEng.Tests` mirrors the Rust integration tests. The two
implementations track each other algorithm-for-algorithm (same iteration
counts, tolerances, and policy decisions).

## Compatibility

| Surface | Version / requirement |
|---|---|
| Rust crates (`phieng-*`) | 0.3.1, MSRV 1.78, edition 2021 |
| NuGet packages (`PhiEng.*`) | 0.3.1, .NET 10 |
| Snapshot schema | v3 (v1/v2 files are rejected, never misparsed) |
| C ABI | `phieng_world_*` / `phieng_world3_*` symbols, `phieng.dll` / `libphieng.so(.dylib)` |
| Determinism | bit-exact within one backend; behavior parity (not bit parity) across Rust↔C# |

## Quickstart

```powershell
# From the repo root:
# Rust: full workspace tests
cargo test --workspace

# Rust: 2D demo (deterministic — identical output every run)
cargo run -p phieng --example demo_2d

# Rust: SPH fluid dam break (prints collapse readings)
cargo run -p phieng --example fluid_dam --release

# Rust: large-scene benchmark (pair/contact census, allocation census, solver sweep)
cargo run -p phieng --example bench_large --release
cargo run -p phieng --example bench_large --release -- no-exit

# C# tests (.NET 10)
dotnet test tests/PhiEng.Tests/PhiEng.Tests.csproj

# Test-ID parity between Rust and C# (S/W/N/D/F/V/J blocks)
python tools/check_test_parity.py

# Rebuild the native DLL consumed by PhiEng.Native (staged at native/win-x64/)
cargo build --target x86_64-pc-windows-gnu --release -p phieng
```

## Build profiles (portable vs native)

The default build — and the shipped `native/win-x64` DLL — targets
portable baseline x86-64 and runs everywhere. For local benchmarking or
hosts that control their hardware, an opt-in native profile compiles
with `-C target-cpu=native` (AVX2/AVX512/FMA where present):

```powershell
# Windows: native cdylib (do NOT stage over native/win-x64/)
powershell -ExecutionPolicy Bypass -File tools/build-native.ps1
# ...or run the native bench directly:
powershell -ExecutionPolicy Bypass -File tools/build-native.ps1 -Bench
```

```bash
# Linux/macOS:
./tools/build-native.sh [--bench]
```

Rules for the native profile:

- **Never redistribute** native binaries as the portable artifact.
- **Trajectory hashes are recorded per profile.** SIMD codegen may
  reorder reductions at the ulp level, so cross-profile bit-parity is
  not promised; within one profile, determinism still holds bit-exact.
- CI's `bench-native` job is informational only (runner noise forbids
  timing gates); `bench-smoke` gates the portable hash + alloc census.

## Narrowphase dispatch (3D)

| Pair | Path | Why |
|---|---|---|
| Box–box | 3D SAT (6 face + 9 edge axes) | Exact face normals; GJK/EPA normals wobble ~1° on face contact and feed a torque jitter loop |
| Sphere–sphere / sphere–box | Analytic closest-point | Exact, no iteration |
| Capsule / hull vs anything | GJK + EPA | General convex support-function path |
| ConvexMesh vs anything | GJK + EPA over mesh vertices | Same path as hulls; single midpoint contact (mesh towers pile instead of stacking — see `absolute_stress_tests.md`) |

GJK uses Ericson Voronoi-region reductions (segment/triangle/tetrahedron),
an honest closest-distance report on separation (never `simplex[0]`), and a
containment-checked iteration-exhaustion fallthrough. EPA only ever starts
from a *strictly* containing tetrahedron (origin and every opposite vertex
bounded off every face plane by `1e-4`); otherwise a deterministic
14-direction support search builds one. If no strict tet exists (grazing
touch or needle overlap with Minkowski inradius < `1e-4`), EPA returns no
contact for that frame rather than a guessed depth.

## Guarantees and policies

- **Per-language determinism, bit-exact.** Same scene + same tick count =
  identical bytes, verified by `test_determinism_bit_exact` (Rust) and the
  C# determinism test. No wall-clock, no iteration-order nondeterminism
  (HashSet iteration order was removed from the broadphase for exactly this
  reason). Rust↔C# bit-parity is *not* claimed (different float libraries);
  behavior parity is.
- **Never guess a contact.** A guessed depth injects energy through
  Baumgarte stabilization and destabilizes stacks. Degenerate simplexes,
  grazing touches, and sliver overlaps skip the frame and self-heal as
  penetration deepens or separates. In user terms: a capsule/hull pair in
  a needle configuration may sink an extra ~3 mm for one frame (at 60 Hz
  under Earth gravity) and is pushed back out over the following frames;
  box–box and sphere pairs never take this path (analytic/SAT dispatch).
  The skip is bounded and convergent, never accumulating.
- **Sleeping is partial (known limitation).** A body flagged asleep is
  *not* skipped by the pipeline and there is **no auto-wake on contact**:
  a sleeping body keeps accumulating gravity and solver impulses each
  step (velocities are zeroed at the end of the step), and only a manual
  `wake_body` call clears the flag. Do not rely on sleep for performance
  yet; it currently only damps residual velocity. Full sleeping (pipeline
  skip + auto-wake + cascade) is a planned separate phase. See
  `LIMITATIONS.md`.
- **Solver (2D and 3D):** max 8 iterations, slop `0.005`, Baumgarte `0.2`,
  warm starting, early exit when max applied impulse `< 1e-6` (resting
  contacts typically converge in 1 iteration; proven trajectory-neutral by
  identical final-state hashes with early exit on/off). The Baumgarte
  bias velocity is capped at 2.0 m/s (`Solver::MAX_BIAS_VELOCITY`,
  mirrored in C#) so deep overlaps resolve gradually instead of
  ejecting; contacts shallower than ~17 cm at 60 Hz are bit-identical
  with or without the cap. Effective masses are snapshotted once per
  step (2D and 3D alike); joints solve after contacts each iteration
  with the same budget.
- **Joints (2D and 3D):** distance rods, revolute hinges (optional angle
  limits), prismatic sliders (optional travel limits, rotation locked to
  the latched pose), and fixed welds — all rigid, bilateral,
  warm-started, solved after contacts. Creation rejects ambiguous input
  (`JointError` in Rust, `ArgumentException` in C#); non-zero compliance
  is rejected, not faked. `min > max` bounds select the free/limitless
  mode explicitly. Removal drops touching joints and remaps survivors.
- **SPH fluids (2D and 3D, Rust stepping):** weakly-compressible
  particles (poly6 density, linear pressure EOS, spiky pressure force,
  XSPH viscosity) stepped as phase 1b of each world step, with two-way
  rigid coupling (boundary impulses, pressure-sampled buoyancy,
  Morison-style drag). New `ConvexMesh` shape carries exact
  Mirtich-style mass properties. Fluid state persists in snapshots
  (schema v3) and crosses FFI; C# carries fluid state but does not step
  it yet.
- **Runtime validation:** every step guards against NaN/Inf state and
  clamps runaway penetration, coordinates, and velocities
  (`ValidationConfig`: `Default`/`Strict`/`Permissive`; per-step
  `ValidationReport`). Report-only for mass-ratio outliers. Zero-cost
  when the world holds no issues to correct; joint-less worlds step
  exactly as before (proven by unchanged determinism hashes).
- **Snapshots:** save/load 2D and 3D worlds as JSON or versioned binary
  (`snapshot.rs`, schema v3: v2 added convex-mesh shapes, v3 added fluid
  state). `Exact` mode restores warm-start caches, joint impulses,
  latched angles, and fluid particles for bit-identical continuation;
  `Fresh` mode restores structure + config with cold solver state.
  Mass is recomputed through the validated constructors on load, never
  trusted from the file; hostile inputs are rejected, never panicked.
- **Operating range:** coordinates within roughly ±100 units (`f32`;
  EPA seed margin `1e-4` absolute, consistent with the absolute slop).
- **Body ids are indices** (`id == index` invariant, maintained by
  `remove_body`'s swap-remove bookkeeping; contact-cache keys are remapped
  through removal so survivors keep warm-start impulses).
- **Creation validates, never sanitizes (except friction/restitution).**
  Zero-size shapes, degenerate polygons/hulls, zero-length capsules ("use
  Sphere instead"), zero-density dynamics, and any non-finite input are
  rejected at creation (`Result<_, ShapeError>` in Rust,
  `ArgumentException` in C#). Friction/restitution keep clamp-to-range
  (universally-agreed ranges); everything else ambiguous is an error.

## Performance tiers (how to read the numbers)

`bench_large` (Rust, counting allocator) reports medians over 5×300-step
windows after settling, plus `avg pairs/step` and `avg contacts/step` —
always read time *with* those counts, and with the solver sweep
(`iters=1` vs `iters=8`), because cost is per-contact per-iteration:

- **Empty world:** `0.00` allocs/step. **Resting stack, converged
  (early exit):** `~0.15` allocs/step, `~5` B/step — the steady state
  performs no per-frame allocation (broadphase pair buffers, contact
  cache entries, and EPA working sets are all reused caller-owned
  scratch). This is the number that matters for low-spec hardware.
- **Dynamic pile (1500 bodies, ~1260 pairs, ~1400 contacts):** allocation
  churn is dominated by contact-cache entry birth/death as fringe pairs
  flicker, and is *trajectory-dependent* (pair order changes alter float
  summation order, which alters the pile's evolution — both deterministic,
  both correct). Compare builds by trajectory hash first; only compare
  churn across identical hashes. Do not tune the allocator against a
  single pile history.
- C# mirrors the same methodology (GC allocation counters, solver sweep).

## Baseline matrix 1a (AoS + islands control, 2026-10-14)

Medians over 5×300-step windows, portable profile
(`x86_64-pc-windows-gnu` release / .NET Release). Timing is machine- and
noise-sensitive (±40%); hashes are the determinism gate.

| Scene | Rust serial | Rust parallel | C# serial | C# parallel |
|---|---|---|---|---|
| pile 2D, iters=1 | 2.778 ms/step | — | 4.187 ms/step | — |
| pile 2D, iters=8 | 5.314 ms/step | — | 4.469 ms/step | — |
| pile 2D pairs/contacts | 1629 / 1785 | — | 1215–1766 / 1362–1959 | — |
| pile 2D allocs | 466.8/step | — | gc0 ≤ 0.027/step | — |
| islands (8×5 stacks) | 0.0473 ms/step | 0.0951 (0.50x) | 0.0693 ms/step | 0.1247 (0.56x) |
| islands identical? | — | **true** | — | **true** |
| pile 3D (400), iters=1 | 0.413 ms/step | n/a (2D-only) | 0.762 ms/step | n/a |
| pile 3D (400), iters=8 | 0.723 ms/step | n/a | 1.230 ms/step | n/a |
| pile 3D pairs/contacts | 592 / 904 | — | 561–579 / 891–913 | — |
| pile 3D allocs | 14.0/step | — | gc0 0.000/step | — |

Hashes: pile 2D `0xe767a92fb630ba4b`, islands `0xfe0d182ff4321dad`
(serial == parallel), pile3d `0x3c0815ebe61347f4`. Hashes moved in
0.3.1 (3D friction cone is now a true cone clamp, default sleep
thresholds 0.01 → 0.05, distance-joint zero-separation fallback, EPA
seed-margin scaling). C# hashes are run-local (worlds keep evolving
across windows); the gate is serial == parallel within a run.

Findings for the SoA verdict: (1) the 41-body islands scene sits
**below the parallelization threshold** — thread/barrier overhead
dominates (0.50x Rust scoped-threads, 0.56x C# threadpool this run;
an earlier C# run showed 2.16x, i.e. noise dominates at this size).
Parallel scaling must be judged on larger per-island workloads.
(2) C# 2D pile is within ~1.5x of Rust despite class indirection —
pointer-chasing is not yet proven dominant. The SoA spike (Phase 1b)
must beat these cells beyond noise, or AoS stays.

## Phase 1b shootout + verdict: AoS stays (HOLD confirmed)

Minimal SoA contact-solver spikes were built in both languages
(gather hot fields → same sequential-impulse math op-for-op →
scatter; contacts only, serial only), proven bit-identical to AoS
first (6 differential tests: hand-built solver case + 300-step
stacks + 120-step 240-body piles + joint scene, all hashes equal),
then timed on the same scenes. Gather cost is included in every SoA
number, which favors AoS.

| Cell | AoS | SoA | Speedup | Identical |
|---|---|---|---|---|
| Rust pile-1500 iters=1 | 2.997 | 3.095 | 0.97x | true |
| Rust pile-1500 iters=8 | 3.514 | 3.519 | 1.00x | true |
| Rust islands | 0.0438 | 0.0473 | 0.93x | true |
| C# pile-1500 iters=1 | 3.273 | 3.189 | 1.03x | True |
| C# pile-1500 iters=8 | 3.427 | 3.549 | 0.97x | True |
| C# islands | 0.0637 | 0.0668 | 0.95x | True |

(ms/step, medians of 5×300; SoA pile hashes `0xde0584218f45a2c7`
/ `0x40535d8960d217f0` equal AoS bit-for-bit **as of 0.3.0** — the
spikes were deleted after measurement and the 0.3.1 physics fixes move
every hash, so these are historical readings, not reproducible.)

**Verdict: HOLD, on measured evidence.** All six cells sit within
noise of 1.00x. At iters=8 (solver-dominated) Rust reads exactly
1.00x — the solver core is layout-neutral: the working set
(~1500 inline bodies, hot-first field order, `split_at_mut`) lives
in cache across iterations, so flat arrays buy nothing and gather
costs ~3–7%. Even C# class indirection shows no penalty for the
same reason (repeated in-cache touches, not streaming).
The spikes were deleted after measurement; AoS stays. Revisit only
with a scene whose working set escapes cache or a profile showing
chase-dominated stalls — not on speculation.

## Test inventory

- Rust: unit tests per crate (SAT, GJK, EPA, broadphase, solver, world) +
  engine integration tests. Includes a 1000-case deterministic
  (xorshift64*) EPA-vs-SAT fuzz over random OBB pairs — boolean agreement
  outside a `1e-3` grazing margin, depth within `0.02 + 5%`, normal in the
  same hemisphere — plus pinned regressions (GJK false-separation pair,
  needle-overlap pair, strict-seed cases).
- C#: same coverage via xUnit, including its own 1000-case fuzz with an
  independent xorshift implementation (stable across .NET versions).
- Robustness suite (IDs S/W/N/D/F/V/J/E/P, mirrored both languages,
  parity enforced by `tools/check_test_parity.py`): sleep semantics,
  warm-start/cache eviction, never-guess self-heal proofs, solver
  numerical limits, shape-edge validation, FFI boundary + dispose stress,
  runtime validation layer (V01–V11), joint constraints (J31–J44) plus the
  native joint bridge (F31–F33), snapshots (E01–E07) plus the native
  save/load bridge (F34–F36), parallel island solver (P31–P34,
  bit-identical to serial; `Solver.Parallel`, off by default).
- Cross-language equivalence (F28): identical fixed scenes through the
  managed engine and the Rust core via FFI, agreeing within tolerance
  (bit-parity not claimed; behavior parity proven). Known gaps live in
  `LIMITATIONS.md`, each pinned by a passing `known_limitation_` test.
- Absolute stress suite (`crates/engine/tests/brutal.rs`, 13 tests):
  collapsing towers, restitution storms, joint gauntlets, mass-ratio
  soaks, mesh tumblers, snapshot-mid-chaos resumes, hostile input.
  Each test asserts survival and prints a `BRUTAL ...` readings line;
  results are logged in `absolute_stress_tests.md`. Names deliberately
  avoid the parity-ID pattern, so no C# mirrors are required.
