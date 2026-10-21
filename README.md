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
phieng = "0.4.0"

# Or use the full workspace for all crates
phieng-math = "0.4.0"
phieng-collision = "0.4.0"
phieng-broadphase = "0.4.0"
phieng-dynamics = "0.4.0"
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
| `broadphase` | `PhiEng.Broadphase` | Tiered backends (`BroadphaseMode`: brute force, spatial hash, dynamic AABB tree, hybrid auto-select) |
| `dynamics` | `PhiEng.Dynamics` | Bodies, materials, sequential-impulse solvers, worlds, 2D joints, snapshots |
| `engine` | `PhiEng` (root) + `PhiEng.Native` | Re-exports, C ABI (`phieng_world_*`, `phieng_world3_*`, `phieng_world_*_joint`) |

`tests/PhiEng.Tests` mirrors the Rust integration tests. The two
implementations track each other algorithm-for-algorithm (same iteration
counts, tolerances, and policy decisions).

## Compatibility

| Surface | Version / requirement |
|---|---|
| Rust crates (`phieng-*`) | 0.4.0, MSRV 1.78, edition 2021 |
| NuGet packages (`PhiEng.*`) | 0.4.0, .NET 10 |
| Snapshot schema | v12 (loaders accept v3–v12; older unknown versions are rejected, never misparsed) |
| C ABI | `phieng_world_*` / `phieng_world3_*` symbols, `phieng.dll` / `libphieng.so(.dylib)` |
| Determinism | bit-exact within one backend; behavior parity (not bit parity) across Rust↔C# |

## Quickstart

```powershell
# From the repo root:
# Rust: full workspace tests (examples excluded from the test build;
# clippy --all-targets still compiles them)
cargo test --workspace --lib --tests

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
  timing gates); `bench-smoke` gates the portable hash + alloc census —
  the benchmark now *asserts* the pinned trajectory hashes and fails the
  job if the physics moved. The cheap canonical scenes are pinned in
  `crates/engine/tests/determinism_hash.rs`, which runs in the normal
  test suite on every platform.

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
- **Platform-independent rigid-body trajectories.** The rigid-body core
  never calls the platform math library: the rotation integration uses
  `phieng-math`'s deterministic `atan2`/`sin`/`cos`, which are built from
  IEEE-exact operations only and therefore return bit-identical results on
  every OS and CPU. This matters because `sinf`/`cosf`/`atan2f` disagree
  between glibc and the Microsoft CRT in the last ULP, and one ULP is
  enough to send a 240-step trajectory in two different directions — the
  canonical 2D scene used to hash differently on Linux and Windows. The
  pinned trajectory hashes (`determinism_hash.rs`, `bench_large.rs`) are
  therefore the same constants on every platform, and the rotation
  integration is bit-identical across Rust and C# as well (R03). The
  fluid and ambient subsystems still use the platform math library for
  their particle placement, so *their* trajectories remain
  platform-dependent; see `LIMITATIONS.md`.
- **Never guess a contact.** A guessed depth injects energy through
  Baumgarte stabilization and destabilizes stacks. Degenerate simplexes,
  grazing touches, and sliver overlaps skip the frame and self-heal as
  penetration deepens or separates. In user terms: a capsule/hull pair in
  a needle configuration may sink an extra ~3 mm for one frame (at 60 Hz
  under Earth gravity) and is pushed back out over the following frames;
  box–box and sphere pairs never take this path (analytic/SAT dispatch).
  The skip is bounded and convergent, never accumulating.
- **Sleeping is full but support-gated.** A body flagged asleep is
  frozen: skipped in gravity, ambient, fluid force application,
  integration, and the contact solver (treated as infinite mass, exactly
  like statics). Wakes happen on new contact (with transitive cascade
  over contact and joint partners), lost contact (support pulled away),
  direct impulse/force application, and joint-partner motion — plus an
  explicit `wake_body` that cascades the same way. A body falls asleep
  only when slow **and** supported by a current contact (ballistic
  apexes and drifters never sleep), so no wake/sleep history exists and
  save/load stays exact. See `LIMITATIONS.md`.
- **Solver (2D and 3D):** quality tiers `Ultra` (8 iterations) down
  to `Minimal` (1 iteration) via `SolverQuality` (`ApplyQuality()` in
  C#); `Low`/`Minimal` skip the friction pass (~half the solve cost),
  with `enable_friction` / `enable_restitution` switches. Slop `0.005`,
  Baumgarte `0.2`, warm starting, early exit when max applied impulse
  `< 1e-6` (resting contacts typically converge in 1 iteration; proven
  trajectory-neutral by identical final-state hashes with early exit
  on/off). The Baumgarte
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
  particles (poly6 density, spiky pressure force, XSPH viscosity)
  stepped as phase 1b of each world step, with two-way rigid coupling
  (boundary impulses, pressure-sampled buoyancy, Morison-style drag).
  The pressure model is selectable: the legacy **linear EOS**
  (`gamma == 1`) or a **Tait/WCSPH** equation of state (`gamma > 1`,
  classic water `7.0`) that supports deep columns with less artificial
  stiffness; optional **Monaghan artificial viscosity** damps
  approaching particles, and optional **Akinci-style surface tension**
  (cohesion) holds droplets together and rounds free surfaces. Optional
  **CFL substepping** (`cfl`, `max_substeps`) keeps stiff Tait configs
  stable, and the fluid feels ambient **weather** — the body drag uses
  the ambient wind and turbulence drives the particles. Optional
  **adaptive resolution** (per-particle mass; split under-dense / merge
  over-dense, mass-conserving) refines resolved regions, and optional
  **boundary particles** sample static bodies into fixed ghost particles
  that exert SPH pressure on the fluid (a pressure-based wall that
   reduces tunneling), and optional **wall no-slip** (`wall_friction`,
   free-slip by default) drives the tangential velocity toward the wall's
   for a curved Poiseuille-like channel profile. Optional **two-phase**
   fluids give each particle a phase index (0..=3) with per-phase rest
   density and mass, so a heavier phase sinks under gravity
   (Rayleigh–Taylor, oil/water buoyancy). All default
   off, so existing fluids are bit-identical. New
   `ConvexMesh` shape carries exact Mirtich-style mass properties. Fluid
   state persists in snapshots (config block extended at schema v5) and
   crosses FFI (config set/get plus phase-aware insert); the managed C#
   solver mirrors the Rust pipeline stage-by-stage (behavior parity,
   tolerance-based) and steps as phase 1b of managed world steps.
- **Gravitational physics (2D and 3D, both languages):** uniform field
  (status quo, bit-identical when untouched), central attractor
  (`mu` at a fixed point, O(n)), full N-body pairwise Newtonian
  gravity with softening (symmetric ±F, momentum conserved to float
  rounding, O(n²)), and a deterministic Barnes–Hut approximation
  (`theta`, O(n log n), crossover ~700 bodies at θ=0.5, exact stays
  default). Per-body `gravity_scale` (1.0 coupled, 0.0 immune,
  negative repels); massless probes need no exemption (tiny density +
  scale). Optional velocity-Verlet kick-drift-kick wraps the unchanged
  contact solver (linear damping rides the kicks as symmetric drag).
  Same-mode runs are bit-identical; exact-vs-BH trajectories differ by
  design, never compared. Orbital scenes must zero damping and disable
  sleep (thresholds to 0.0); the uniform vector still applies as a
  background field in every mode, and the fluid keeps using it only.
  Bulk FFI (`add_bodies`, `get_transforms_bulk`) plus per-frame-safe
  `Try*` getters and `SuppressGCTransition` on trivial calls keep
  managed↔native polling to one transition per frame; gravity config,
  scales, and sleep thresholds all cross FFI with the same
  null-tolerant status-code conventions.
- **Ambient force fields (2D and 3D, both languages):** three
  independently-toggled fields applied as a pre-solve velocity kick at
  the same pipeline slot as gravity (and with the same Verlet half-kick
  contract). A **spring / simple-harmonic-motion** field
  (`F = -k*(x - anchor) - c*v`, with an optional rest-length leash) gives
  the textbook damped oscillator (`ω = sqrt(k/m)`, `ζ = c/(2*sqrt(k*m))`);
  **air resistance** combines linear (Stokes) and quadratic
  (`0.5*rho*Cd*A*|v_rel|*v_rel`) drag against an ambient wind plus an
  angular-drag torque, with a per-shape reference area; and
  **turbulence** is a deterministic seeded value-noise fBm field
  (optionally divergence-free via curl noise), bounded to the configured
  amplitude. Per-body `ambient_scale` (1.0 coupled, 0.0 immune, negative
  is the caller's physics) mirrors `gravity_scale`; `World2/3::time`
  advances by `dt` each step and drives time-varying turbulence.
  Same-seed runs are bit-identical per backend; Rust↔C# cross-language
  bit-equality is *not* claimed (the noise uses different float ops), so
  parity binds to statistical/energy bounds. Ambient config, the clock,
  and per-body scales cross FFI; a default config is a no-op.
- **Thermodynamics & radiation (2D and 3D, Rust core + FFI):**
  per-body temperature (Kelvin, default 293.15 K) evolved by a thermal
  phase running after integration: contact-driven **conduction**,
  Stefan–Boltzmann T⁴ **radiation** (ambient + body-to-body, distance-
  bounded), and Newtonian **convection** to ambient temperature. Material
  carries `specific_heat`, `thermal_conductivity`, `emissivity`;
  `ThermalConfig2/3` toggles each mechanism plus clamps and the radiation
  cutoff. Off by default — bit-identical trajectories. Static bodies act
  as fixed-temperature reservoirs. Three opt-in sub-features: radiation
  **occlusion** (a third body blocks the line of sight), **forced
  convection** (`h + h_forced·|v|`), and **thermal expansion** (the shape
  scales by `1 + alpha·(T − T_ref)`, applied relative to the authored
  shape so the factor never compounds). Crosses FFI
  (`phieng_world[_3]_set/get_thermal_config`,
  `set/get_body_temperature`, `set/get_body_thermal_material`) and
  persists in schema v8 (sub-features in v10). The managed C# mirror
  (`PhiEng.Dynamics`) steps the same thermal subsystem; its snapshot
  layout is byte-identical to the Rust writer's, so a file written by
  either engine loads in the other (F40).
- **Radiation model expansion (2D and 3D, both languages):** three
  opt-in options on `ThermalConfig2/3`, all defaulting to the
  historical behaviour so existing trajectories stay bit-identical:
  a **symmetric view-factor model** (`ViewFactorModel::Symmetric`) whose
  exchange rate no longer depends on which body of a pair the loop
  visits first, a **sky/ground split** of the environment radiation
  (each body's up-facing area fraction splits the exchange between a
  cold sky and a warm ground), and a **partially transparent occluder**
  (`occluder_transmittance`) that attenuates an occluded exchange
  instead of blocking it. The occlusion test itself was fixed: it used
  to exempt blockers sitting near an endpoint, which let radiation leak
  through the gap. Persists in schema v11. Mirrored in both languages
  (U01–U12).
- **Nuclear physics (2D and 3D, both languages):** a game-scale
  radioactive-decay, activation and fission model. A body carries a
  **nuclide** (from a config-supplied table: half-life, decay energy,
  the fraction that leaves as ionizing radiation, a daughter nuclide,
  absorption and fission cross-sections, the neutron yield and whether
  it is fissile) plus a **nuclear state** (the activity, the remaining
  fuel, the free and delayed neutron populations and the accumulated
  dose). The phase runs *before* the thermal phase: the energy released
  by decay and fission becomes heat, which the thermal phase turns into
  temperature and then into thermal radiation. **Fission** consumes
  fuel, releases `fission_energy` and produces `neutron_yield`
  neutrons, a fraction of which are *delayed* — so a chain reaction
  goes prompt-critical rather than immediately divergent. Free neutrons
  are shared with the nearby bodies (a diffusion approximation, not a
  transport solve) and the ionizing share of the energy is deposited
  as **dose** in the bodies that are in range. Off by default, so a
  default world is bit-identical to a pre-nuclear one. The decay uses
  the engine's **deterministic** exponential, so a half-life drifts by
  the same amount on every platform and Rust and C# agree bit-for-bit.
  Crosses FFI (`phieng_world[_3]_set/get_nuclear_config`,
  `phieng_world[_3]_set_isotope`,
  `phieng_world[_3]_set/get_body_nuclear_state`) and persists in schema
  v12. Mirrored in both languages (Y01–Y28).
- **Broad-phase tiers (2D and 3D, both languages):** `BroadphaseMode`
  (`BruteForce` / `SpatialHash` / `DynamicAABBTree` / `Hybrid`), a
  dynamic AABB tree backend for 2D and 3D, and an explicit O(n²) brute
  force backend. Modes are behavior-identical (canonical sorted+deduped
  pair sequence); `Hybrid` auto-selects from live body count each step
  (brute force below 50, grid 50–5000, tree above). Default stays the
  grid. Mode persists in schema v8; FFI
  `phieng_world[_3]_set/get_broadphase_mode` plus managed
  `PhiNativeWorld(3)` wrappers.
- **Collision tiers (2D and 3D, Rust + managed C#):** `CollisionTier`
  (`Tier1_AABBOnly` … `Tier5_GJK_EPA`) ceilings each shape pair at the
  cheapest separating test (sphere-AABB, sphere-sphere, SAT box-box,
  GJK/EPA); pairs above the ceiling are skipped, never guessed. Maps
  from `SolverQuality`; off by default (full fidelity). FFI
  `phieng_world[_3]_set/get_collision_tier` plus managed
  `PhiNativeWorld(3).SetCollisionTier/GetCollisionTier`; persists in
  schema v9. Pinned by H01–H05 (Rust↔C# parity).
- **CCD tiers (2D and 3D, both languages):** `CcdTier`
  (`Off` / `SweptAABB` / `FullGJK`) selects which bodies substep —
  swept-AABB substeps only pairs whose swept volumes overlap (sparse
  scenes pay nothing), full keeps the legacy peak-speed behavior. Maps
  from `SolverQuality` (`Ultra`→Full, `High`→Swept, else Off); persists
  in the v6 `ccd` block's `tier` field.
- **Collision filtering (2D and 3D, both languages):** every body
  carries a collision *layer* (the group it belongs to) and a *mask*
  (the layers it accepts). Two bodies may collide only when each
  accepts the other's layer, so one side is enough to veto a pair. The
  default filter (layer 1, mask all) accepts everything, so a scene
  that never configures one behaves exactly as before — every pinned
  trajectory hash is unchanged. `RigidBody2/3.CollisionGroup` /
  `CollisionMask`; FFI `phieng_world[_3]_set/get_body_collision_filter`
  plus managed `PhiNativeWorld(3).Set/GetBodyCollisionFilter`;
  persists in schema v13 (older files load with the default). Pinned by
  L01–L06 (Rust↔C# parity).
- **Scene queries (2D and 3D, both languages):** `World2/3::overlap_aabb`
  returns the bodies whose AABB overlaps a query box, in ascending id
  order, honouring the collision filter. The query runs against the
  *current* body positions — the broad phase's grid holds the positions
  from before the last solve, so a query must not report against those.
  Every broad-phase backend also answers an AABB query of its own, and
  all three answer alike. FFI `phieng_world[_3]_overlap_aabb` (a null
  output array asks for the count alone) plus managed
  `World2/3.OverlapAabb` and `PhiNativeWorld(3).OverlapAabb`. Pinned by
  X01–X06 (Rust↔C# parity).
- **SIMD math paths (Rust, opt-in):** `phieng-math` `simd` feature flag
  (off by default for MSRV compatibility) enables batched primitives
  (`add4`/`sub4`/`mul4`/`dot4`, `dot2_batch`, `add_vec2_batch`) using
  x86_64/x86 SSE intrinsics with scalar fallbacks everywhere else.
  The element-wise ops are bit-exact vs their scalar fallback; `dot4`
  is not (the SSE horizontal add reassociates the sum), so its test
  compares with a tolerance. Default builds are untouched.
- **PhysicsConfig facade (Rust):** `PhysicsConfig` unifies
  `solver_quality`, `collision_tier`, `broadphase_mode`, sleep
  thresholds, `ccd`, and `thermal` with `Default` (= pre-tier-system
  behavior), `low_end()`, and `high_fidelity()` presets plus
  `apply_to_2d`/`apply_to_3d`.
- **Runtime validation:** every step guards against NaN/Inf state and
  clamps runaway penetration, coordinates, and velocities
  (`ValidationConfig`: `Default`/`Strict`/`Permissive`; per-step
  `ValidationReport`). Report-only for mass-ratio outliers. Zero-cost
  when the world holds no issues to correct; joint-less worlds step
  exactly as before (proven by unchanged determinism hashes).
- **Snapshots:** save/load 2D and 3D worlds as JSON or versioned binary
  (`snapshot.rs`, schema v12: v2 added convex-mesh shapes, v3 added fluid
  state, v4 added the gravity block + per-body scales, v5 added the
  ambient-force block + simulation clock + per-body ambient scales, v6
  added the CCD block, v7 added solver friction/restitution switches, v8
  added the thermal block + per-body temperature + broadphase-mode tag, v9
  added the collision-tier block, v10 added the thermal sub-feature block
  + the per-body expansion factor, v11 added the radiation model block,
  v12 added the nuclear block + the per-body nuclear state —
  all conditional, so default worlds still write byte-identical older
  schemas and loaders accept v3–v12).
  `Exact` mode restores warm-start caches, joint impulses,
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
  (early exit):** `0.00` allocs/step, `0` B/step — the steady state
  performs no per-frame allocation (broadphase pair buffers, contact
  cache entries, and EPA working sets are all reused caller-owned
  scratch). This is the number that matters for low-spec hardware.
- **Dynamic pile (1500 bodies, ~1650 pairs, ~1850 contacts):** the
  broadphase grid, the contact cache, and the SPH fluid neighbor grids
  all **recycle their per-key / per-cell buffers through internal
  pools** — vacated entries return their `Vec` to a pool instead of the
  allocator, and new entries draw from it. The pile now runs at
  **0.0 allocs/step** (was ~467: the churn was cell/entry birth-death,
  now effectively zero). Pooling is
  memory-only and keeps `contact_cache.len()` exact, so trajectories are
  bit-identical. Residual churn is *trajectory-dependent* (pair order
  changes float summation order, which changes the pile's evolution —
  both deterministic, both correct); compare builds by trajectory hash
  first.
- C# mirrors the same methodology (GC allocation counters, solver sweep).

## Baseline matrix 1a (AoS + islands control, 2026-10-14)

Medians over 5×300-step windows, portable profile
(`x86_64-pc-windows-gnu` release / .NET Release). Timing is machine- and
noise-sensitive (±40%); hashes are the determinism gate.

| Scene | Rust serial | Rust parallel | C# serial | C# parallel |
|---|---|---|---|---|
| pile 2D, iters=1 | 2.778 ms/step | — | 4.187 ms/step | — |
| pile 2D, iters=8 | 5.314 ms/step | — | 4.469 ms/step | — |
| pile 2D pairs/contacts | 1646 / 1846 | — | 1215–1766 / 1362–1959 | — |
| pile 2D allocs | 0.8/step | — | gc0 ≤ 0.027/step | — |
| islands (8×5 stacks) | 0.0473 ms/step | 0.0951 (0.50x) | 0.0693 ms/step | 0.1247 (0.56x) |
| islands identical? | — | **true** | — | **true** |
| pile 3D (400), iters=1 | 0.413 ms/step | n/a (2D-only) | 0.762 ms/step | n/a |
| pile 3D (400), iters=8 | 0.723 ms/step | n/a | 1.230 ms/step | n/a |
| pile 3D pairs/contacts | 592 / 904 | — | 561–579 / 891–913 | — |
| pile 3D allocs | 0.0/step | — | gc0 0.000/step | — |

Hashes: pile 2D `0x4acb6694f3a9917`, islands `0xd4a3a56f185c13b2`
(serial == parallel), pile3d `0x66ae94fb463de60`. Hashes moved in
0.4.0 (3D friction cone is now a true cone clamp, default sleep
thresholds 0.01 → 0.05, distance-joint zero-separation fallback, EPA
seed-margin scaling). C# hashes are run-local (worlds keep evolving
across windows); the gate is serial == parallel within a run.
The 0.4.x broadphase/contact-cache buffer pools are memory-only and left
every hash above unchanged.

The 2D hashes moved again when the rotation integration was made
platform-independent (see *Determinism* below): the old pins were the
glibc values, and the same code produced a different trajectory on
Windows. The current pins are the same on every platform.

The 2D hashes moved once more when the circle-vs-polygon SAT gained its
missing corner-axis test: the old code reported phantom contacts for
circles grazing polygon corners (face-normal projections can all
overlap while the shapes are separated). Scenes without
circle-vs-polygon corner grazing (islands, pile 3D) are unchanged.

## Baseline matrix 1b (broadphase overhaul, 2026-10-21)

The numbers below are the current state on an AMD Ryzen 7 7445HS
(Windows x64, release). `PERF_BASELINE.md` holds the full table, the
per-phase split, and the measurement discipline.

| Scene | Rust serial | Rust parallel | Notes |
|---|---|---|---|
| pile 2D (1500), iters=1 | 0.566 ms/step | — | was 2.45 ms/step before the overhaul (**−77 %**) |
| pile 2D (1500), iters=8 | 0.821 ms/step | — | was 6.18 ms/step (**−87 %**) |
| pile 2D, settled census | 1.297 ms/step | — | was 3.92 ms/step (**−67 %**) |
| pile 2D allocs | **0.0/step** | — | was 0.8/step |
| islands (8×5 stacks) | 0.0299 ms/step | 0.0330 | 41 bodies: below the parallelization threshold |
| pile 3D (400), iters=8 | 0.153 ms/step | n/a (2D-only) | was 0.302 ms/step (**−49 %**) |
| pile 3D allocs | 0.0/step | — | |
| statics (420) | 0.030 ms/step | — | the grid keeps its cells now: rebuild 0.0063 → 0.0029 ms |

Those are interleaved A/B numbers (the two builds alternate, nine
rounds) on one machine — the only comparison that survives a loaded
box, since the absolute timings drift by ±20 % between runs.

Every one of those changes leaves the pinned hashes bit-identical: the
grid compaction, the FxHash switch, the AABB pair filter and the
static-pair pruning all change *which* pairs are examined, never the
constraint stream. The hashes that did move (the 2D pair) moved for a
different reason: the circle-vs-polygon corner-axis fix removed phantom
contacts, and that changes the physics on purpose.

Where the time goes now (2D pile, per step): the solver 40 %, the
narrowphase 20 %, pair generation 14 %, the grid rebuild 10 %.

## Baseline matrix 1c (packed solver view, 2026-10-21)

The solver's inner loop reads six fields of each body for both ends of
every constraint, on every iteration. In `RigidBody2` those fields span
three cache lines, so the 1500-body pile cost the solver a 264 KB
working set. The solver now gathers them into one cache-line-aligned
32-byte record per body (48 KB for the same pile) and writes the
velocities back when it is done — the body array keeps its layout, only
the solver's *view* of it is packed.

| Measure | Value | Notes |
|---|---|---|
| solver phase | **−6 % to −10 %** | three independent repeats |
| whole 2D pile step | **−2 % to −5.5 %** | same repeats |
| every pinned hash | **bit-identical** | 2D, 3D, islands serial **and** parallel |

Those numbers come from `cargo run -p phieng --example
bench_solver_ab --release --features phase-timing`, which puts both
variants in one binary, one world and one heap layout, and alternates
them every 20 steps. Cross-binary comparisons on a loaded machine
cannot resolve anything below ~15 %: two builds of *identical* source
differ by 6–10 % per phase because the code layout moved.
`PERF_BASELINE.md` documents all three confounds and their fixes.

## Baseline matrix 1d (pair ownership rule, 2026-10-21)

A body sits in every grid cell its AABB spans, so a pair of neighbours
that share two or four cells was found in each of them — the same pair,
several times over, before the sort and dedupe. The grid now gives
every pair one *owner* cell: the lowest cell both bodies span, which
both of them therefore occupy. No pair is lost; each is generated
exactly once.

| Measure | Value | Notes |
|---|---|---|
| broadphase pair phase | **−18 % to −20 %** | four independent repeats |
| whole 2D pile step | **−3 % to −7 %** | same repeats |
| every pinned hash | **bit-identical** | the pair set is unchanged |

## Baseline matrix 1e (static-cell skip, 2026-10-21)

Two statics never touch, so every candidate pair in a cell that holds
nothing but static bodies was walked and then dropped one by one. In a
level full of static geometry that walk is nearly all of the pair
phase. The grid now drops such a cell whole.

| Measure | Value | Notes |
|---|---|---|
| static scene pair phase | **−88 %** | 5 376 static-static candidates → none walked |
| static scene step | **−46 %** | 1.9x faster; 0.062 → 0.033 ms/step |
| 1500-body pile | **unchanged** | every cell holds a dynamic body |
| every pinned hash | **bit-identical** | the pair set is unchanged |

## Baseline matrix 1f (persistent grid, 2026-10-21)

The grid used to wipe every cell and re-insert every body, every step
— 2 400 hash lookups for the static-heavy scene's 400 boxes alone. It
now keeps its cells: a body whose cell range is unchanged keeps its
place in them, and only the bodies that moved are re-inserted. The
cell range is a tuple of integers, so the comparison involves no float
rounding, and the pair set is unchanged.

| Measure | Value | Notes |
|---|---|---|
| static scene rebuild | **−53 %** | 0.0063 → 0.0029 ms/step |
| static scene step | **−32 %** | 0.042 → 0.030 ms/step |
| 1500-body pile rebuild | **−33 % to −40 %** | its bodies all move, so less to skip |
| 1500-body pile step | **−1 % to −6 %** | |
| every pinned hash | **bit-identical** | 2D, 3D, islands serial **and** parallel |

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
- Robustness suite (IDs S/W/N/D/F/V/J/E/P/G/A/T/B/Q/C/K, mirrored both languages,
  parity enforced by `tools/check_test_parity.py`): sleep semantics,
  warm-start/cache eviction, never-guess self-heal proofs, solver
  numerical limits, shape-edge validation, FFI boundary + dispose stress,
  runtime validation layer (V01–V11), joint constraints (J31–J44) plus the
  native joint bridge (F31–F33), snapshots (E01–E07) plus the native
  save/load bridge (F34–F36), gravitational physics (G01–G10: orbit,
  determinism, Verlet energy order, softening safety, probes, snapshot
  round-trip, BH determinism, plus a compact 3D set), ambient forces
  (A01–A05: spring/SHM period, drag terminal velocity, turbulence
  determinism, default no-op, scale immunity), thermal subsystem
  (T01–T24: conduction, radiation, convection, clamping, reservoirs,
  no-op paths, determinism; 3D conduction + determinism; radiation
  occlusion, forced convection, thermal expansion including the
  no-compounding guarantee), broad-phase
  tiers (B01–B06: backend parity 2D/3D, Hybrid auto-select, default
  backend, mid-run switch transparency, body-id invariant, FFI
  round-trip), solver quality tiers (Q01–Q05), CCD tiers (C01–C04),
  3D joint constraints (K31–K44), parallel island
  solver (P31–P34, bit-identical to serial; `Solver.Parallel`, off by
  default).
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
- Orbital & SHM/SPH benchmark suites (Rust-only, analytic where the
  method allows): `crates/engine/tests/orbital.rs` (7 celestial-mechanics
  scenarios) and the SHM/SPH benchmarks in
  `crates/dynamics/src/{ambient,fluid,fluid3}.rs` (damped-oscillator
  regimes, driven resonance, undamped-energy/critical-settle/stiffness
  scaling, hydrostatic column (linear + Tait EOS), dam-break surge and
  obstacle, droplet, sloshing, channel flow). Gaps (no-slip walls,
  Martin–Moyce √t, penalty-floor hydrostatic overshoot) are pinned in
  `LIMITATIONS.md`; readings in `absolute_stress_tests.md`.
- Thermal, nuclear, fuzz and tier stress suites (Rust-only):
  `crates/engine/tests/thermal_stress.rs` (9 tests: conduction and
  convection against closed forms, radiative antisymmetry, the sky/ground
  split, thermal shock, expansion, clamp boundedness),
  `crates/engine/tests/nuclear_stress.rs` (7 tests: the half-life law,
  the energy split, the fission scaling law, runaway boundedness, the
  capture chain, dose clamping),
  `crates/engine/tests/snapshot_fuzz.rs` (6 tests: 20 000+ mutated
  snapshots that must be refused or must reproduce the same world) and
  `crates/engine/tests/tier_sweep.rs` (3 tests: what the broad-phase
  backends, the collision tiers and the solver-quality tiers cost).
  Each prints a `THERMAL/NUCLEAR/FUZZ/TIER ...` readings line; all
  readings are in `absolute_stress_tests.md`. Names deliberately avoid
  the parity-ID pattern, so no C# mirrors are required.
