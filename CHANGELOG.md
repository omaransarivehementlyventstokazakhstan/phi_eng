# Changelog

All notable changes to PhiEng are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Fixed
- **Circle-vs-polygon narrowphase no longer reports phantom contacts.**
  `sat_circle_polygon` (Rust) and `SAT.CirclePolygon` (C#) tested only
  the polygon's face normals, so a circle sitting past a polygon corner
  — close enough to overlap every face-normal projection yet separated
  along the corner axis — was reported as a contact with a bogus
  penetration (the minimum face overlap). Both implementations now also
  test the axis from the closest polygon vertex to the circle center,
  which separates exactly in that case. The test only ever turns a
  contact into a separation, never the reverse: genuinely overlapping
  shapes overlap on every axis. True corner contacts keep their
  previous normal and penetration (the vertex axis is used for
  separation only, not for the minimum-overlap computation).

  The pinned trajectory hashes moved because the old pins contained
  phantom-contact motion (re-pinned below); scenes without
  circle-vs-polygon corner grazing are unchanged:
  - determinism scene 2D: `0xc3f59dbc6e87256a` → `0xb12fcb9a19506312`
  - bench pile 2D: `0xc509c18cd50043b4` → `0x4acb6694f3a9917`
  - bench islands, determinism scene 3D and bench pile 3D are unchanged.

### Changed
- **The broad phase is substantially faster, with identical results.**
  Three changes, each measured and each leaving every pinned hash
  bit-identical:
  - **The grid is compacted every step.** It used to keep every cell it
    had ever seen and compact only every 256 clears, so a scene that
    swept its bodies across many positions left millions of cells
    behind — a settled 1500-body 2D pile held **3 327 878 cells for
    1501 bodies**, and every lookup in a hash table that large is a
    cache miss. The insert pass cost ~650 ns per body, essentially all
    of it in the hash lookup.
  - **The grid is hashed with FxHash** instead of std's SipHash. The
    contact cache already used FxHash (its keys are internal
    `(body_a, body_b)` pairs); the grid's key is the same kind of
    internal value. The hasher now lives in `phieng-math` so both
    crates share it. The grid's *output* is unaffected — the emitted
    pair sequence is sorted and deduped, so it does not depend on the
    hash order.
  - **Static-static pairs are no longer generated.** Two statics can
    never produce an impulse, and the narrowphase discarded those
    pairs one step later; the static-heavy scene generated **8 010**
    candidate pairs to end up with **14** live ones, and that
    generation was 91 % of the step. The grid now learns which bodies
    are static and skips those pairs at the source. A static still pairs
    with every dynamic body.
  Measured on the corrected trajectory (AMD Ryzen 7 7445HS, release,
  interleaved A/B with the hash guard green on both sides): the static
  scene's pair phase **0.999 → 0.054 ms/step (18x)**; islands serial
  −10 %, islands parallel −12 %, pile 3D −9 %. End to end against the
  pre-overhaul build, the 2D pile's 8-iteration step went
  **5.03 → 0.87 ms/step (−83 %)**, its settled census
  **3.26 → 1.26 ms/step (−61 %)** and the 3D pile
  **0.228 → 0.126 ms/step (−45 %)**, with per-step allocations at
  **0.0**.
- **The warm start looks the contact cache up once per constraint**
  instead of twice. The `contains_key` probe that told the warm start
  whether a pair was new folds into the lookup that fetches the cached
  contact to match. The wake still happens before the solver runs, and
  it cannot change what the match found: it only clears sleep flags,
  and the cache holds the previous frame's contacts either way.
- **The contact solver runs on a packed copy of the bodies' hot fields.**
  The solver's inner loop reads position, velocity, angular velocity,
  inverse mass and inverse inertia for both ends of every constraint, on
  every iteration. In `RigidBody2` those fields sit at offsets 16, 56,
  96, 100 and 104 of a 176-byte body, and the frozen flags at 168/169 —
  so one body spans three cache lines and the 1500-body pile costs the
  solver a 264 KB working set. The solver now gathers exactly those
  fields into one cache-line-aligned 32-byte record per body (48 KB for
  the same pile) and writes the velocities back when it is done.

  The result is **bit-identical**: the copy holds the same values, the
  same arithmetic runs in the same order, and the physics has a single
  implementation shared by both body views. Every pinned hash is
  unchanged — 2D, 3D, islands serial **and** parallel.

  Measured with `crates/engine/examples/bench_solver_ab.rs` (one world,
  one heap layout, the two solver paths alternating every 20 steps, so
  neither the code layout nor the settling drift can favour either
  side): the solver phase **−6 % to −10 %** and the whole step
  **−2 % to −5.5 %** over three independent repeats.
- **The broad phase generates each pair once, not once per shared cell.**
  A body is inserted into every grid cell its AABB spans, so two
  neighbours that share two or four cells were found in each of them —
  the same pair, several times over, before the sort and dedupe the
  output contract requires. The grid now picks one *owner* cell for
  every pair: the lowest cell both bodies span, which both of them
  therefore occupy. Every shared cell still contains both bodies, so no
  pair is lost — each one is generated exactly once, and the list that
  has to be sorted and deduplicated is as long as the pair set instead
  of a multiple of it.

  The pair set is unchanged (the same pairs, in the same canonical
  order), so every pinned hash is bit-identical — 2D, 3D, islands serial
  **and** parallel.

  Measured with the same harness (the grid's rule toggled between
  segments, everything else held fixed): the pair phase **−18 % to
  −20 %** and the whole step **−3 % to −7 %** over four independent
  repeats, with every other phase inside the noise band.
- **A cell of nothing but statics is skipped whole.** Two statics never
  touch, so every candidate pair in such a cell was walked and then
  dropped one by one. In a level full of static geometry that walk is
  nearly all of the pair phase: the static-heavy scene held **5 376**
  static-static candidates to end up with **18** live pairs, and the
  pair phase was **49 %** of that scene's step. The grid now drops such
  a cell without looking at its pairs.

  The pair set is unchanged, so every pinned hash is bit-identical.

  Measured with the same harness: the static-heavy scene's pair phase
  **−88 %** and its whole step **−46 %** (a 1.9x faster step), with
  the 1500-body pile unchanged (every one of its cells holds a dynamic
  body, so there is nothing to skip).

### Added
- **Four new absolute-stress suites, with readings.** The thermal,
  nuclear, snapshot-loader and tier systems had no absolute-stress
  coverage at all; they are now measured, not just asserted. Each suite
  is Rust-only (the names deliberately avoid the parity-ID pattern, so
  no C# mirrors are required) and each test prints one readings line
  that lands in `absolute_stress_tests.md`:
  - `crates/engine/tests/thermal_stress.rs` (9 tests): the conduction
    exchange against an exponential (fitted log-slope −0.5584, τ =
    1.791 s), convective cooling against the analytic exponential
    (492.354 K analytic vs 492.323 K measured), thermal expansion
    against the closed form (1.706850), radiative-exchange
    antisymmetry (bit-identical when the bodies are swapped), the
    sky/ground split (300.0000 K uniform vs 309.5192 K with a 100 K
    sky and a 400 K ground), a 549.9 MJ thermal shock that converges to
    a 0.0663 K gap, the stiffness of the radiative rate cap, and the
    5000 K clamp.
  - `crates/engine/tests/nuclear_stress.rs` (7 tests): the half-life law
    (100 Bq → 1.5625 Bq over six half-lives), the decay-energy split
    (734.970 J heat + 734.970 J ionizing = 1469.940 J released), the
    fission scaling law (linear in the fuel), a runaway that burns its
    fuel out instead of exploding, the capture chain, dose clamping and
    finite behaviour at an activity of 1.5625e28.
  - `crates/engine/tests/snapshot_fuzz.rs` (6 tests): 13 848 single-bit
    flips, 1731 truncations, 2000 random buffers, 2008 malformed JSON
    documents and 127 doctored length fields — all refused, and never
    parsed into a *different* world.
  - `crates/engine/tests/tier_sweep.rs` (3 tests): what the four
    broad-phase backends, the five collision tiers and the five solver
    quality tiers actually cost, and the assertion that all four
    broad-phase backends agree bit-for-bit.

### Fixed
- **Rigid-body trajectories are now platform-independent.** The rotation
  integration used the platform math library, and `sinf`/`cosf`/`atan2f`
  disagree between glibc (Linux) and the Microsoft CRT (Windows) in the
  last ULP — a difference the compiler can also introduce or remove by
  rewriting an inlined call. That single ULP was enough to send the
  canonical 2D scene in two different directions: one body's rotation
  matrix diverged at step 45, and the contact solver amplified it into a
  different final state by step 240. The same code, with the same
  inputs, produced different pinned hashes on Linux and Windows.

  The rigid-body core now uses its own trigonometry
  (`phieng-math`'s `trig` module, mirrored in C# as
  `PhiEng.Math.DeterministicTrig`), built only from operations IEEE-754
  specifies exactly, so it returns bit-identical results on every
  platform — and the same results in Rust and in the managed C# mirror.
  This covers the 2D and 3D rotation integration, the rotation-angle
  getters, and the joint twist/angle extraction.

  The pinned trajectory hashes are therefore the same constants on every
  platform, and the 2D ones moved (re-pinned below) because the
  arithmetic changed:
  - determinism scene 2D: `0xcaebbed0a86ca012` → `0xc3f59dbc6e87256a`
  - bench pile 2D: `0x8619b41d69ad35ee` → `0xc509c18cd50043b4`
  - bench islands: `0xdb0f2a0fff8b7820` → `0xd4a3a56f185c13b2`
  - determinism scene 3D and bench pile 3D are unchanged.

  The fluid and ambient subsystems still use the platform math library
  for their particle placement, so their trajectories remain
  platform-dependent; `LIMITATIONS.md` records the gap.

### Added
- **Radiation model expansion (2D and 3D, both languages).** Three
  opt-in options on `ThermalConfig2/3`, all of which default to the
  historical behaviour so existing trajectories stay bit-identical:
  - a **symmetric view-factor model** (`ViewFactorModel::Symmetric`):
    the view factor becomes `min(r_i * r_j / d^2, 1)` and the exchange
    uses the mean of the two emitting areas, so the exchange rate no
    longer depends on which body of the pair the loop visits first;
  - a **sky/ground split** of the environment radiation: each body's
    up-facing area fraction (a box's top edge, a box's top face in 3D,
    isotropic for everything else) splits the exchange between a cold
    sky and a warm ground. A sky and ground equal to the ambient are
    bit-identical to the historical single-temperature exchange;
  - a **partially transparent occluder** (`occluder_transmittance`):
    an occluded exchange is attenuated by the transmitted fraction
    instead of being blocked completely. Zero (the default) keeps the
    historical "fully opaque" behaviour bit-for-bit.
- The occlusion test was **fixed**: it used to require a blocker to be
  closer to the segment's midpoint than the endpoints are, which
  wrongly exempted blockers sitting next to one of the two exchanging
  bodies and let radiation leak through the gap.
- The new options persist in **snapshot schema v11** (its own block, so
  every earlier payload still decodes) and cross FFI with the thermal
  config.
- **Nuclear physics (2D and 3D, both languages).** A game-scale
  radioactive-decay, activation and fission model. A body carries a
  **nuclide** (from a config-supplied table: half-life, decay energy,
  the fraction that leaves as ionizing radiation, a daughter nuclide,
  absorption and fission cross-sections, the neutron yield and whether
  it is fissile) plus a **nuclear state** (the activity, the remaining
  fuel, the free and delayed neutron populations and the accumulated
  dose). The phase runs *before* the thermal phase: the energy released
  by decay and fission becomes heat, which the thermal phase turns
  into temperature and then into thermal radiation. Fission consumes
  fuel, releases energy and produces neutrons, a fraction of which are
  *delayed* - so a chain reaction goes prompt-critical rather than
  immediately divergent. Free neutrons are shared with the nearby
  bodies and the ionizing share of the energy is deposited as **dose**
  in the bodies that are in range. Off by default, so a default world
  is bit-identical to a pre-nuclear one.
- The decay uses the engine's **deterministic** exponential, not the
  platform libm: `exp` is exactly the function whose last-ULP
  disagreement between glibc and the Microsoft CRT broke the Linux CI
  run, and a half-life that drifts by a different amount on every
  platform would be worse - a chain reaction is a feedback loop, so it
  amplifies even a tiny difference into a different outcome.
- The nuclear subsystem crosses FFI
  (`phieng_world[_3]_set/get_nuclear_config`,
  `phieng_world[_3]_set_isotope`,
  `phieng_world[_3]_set/get_body_nuclear_state`) and persists in
  **snapshot schema v12** (its own block, so every earlier payload
  still decodes).

### Fixed
- **The nuclear subsystem could poison a body's temperature with NaN.**
  The released decay energy is `activity * decay_energy`, which overflows
  to infinity for extreme inputs, and `infinity * 0.0` is NaN - so the
  heat/ionizing split produced NaN and the body kept a NaN temperature
  for good (poisoning the thermal phase, the radiation phase and the
  trajectory hash). The energy product is now saturated instead of
  overflowing, and the split has an explicit zero-ionizing branch.
- **A fission could produce heat and neutrons the body did not have
  fuel for.** The fission count was derived from the fuel but never
  capped by it, so a body kept producing energy long after its fuel ran
  out - free energy, and free neutrons, in the exact regime the feature
  targets. The count is now capped by the fuel actually present.
- **A zero-length step re-applied the previous step's nuclear heat.**
  The nuclear phase computes a heat budget and the world applies it
  afterwards, so a stale budget survived the phase's early return and was
  applied again every step - an unbounded temperature ramp with no decay
  happening. The budget is now cleared before the early returns.
- **The sky and ground temperatures were ignored when they were equal
  to each other.** The split only activated when the two differed, so
  "sky = ground = 200 K" silently fell back to the ambient
  temperature. The split now applies whenever the configured
  temperatures differ from the ambient, and stays bit-identical to the
  historical exchange when they do not.
- **The managed mirror's snapshot layout disagreed with the Rust core's
  for the new nuclear fields.** The C# reader did not read the new
  `activation_rate` and `capture_progress` fields, so a managed-written
  file could not be read back by the managed engine (the shape tag
  desynchronised). The managed layout now matches the Rust core
  field-for-field, and the round-trip is pinned by the tests.
- **The managed mirror compared the expansion factor against
  `float.Epsilon`,** which in C# is the smallest *subnormal*
  (1.4e-45), not the machine epsilon (1.19e-7) that the Rust code
  uses (`f32::EPSILON`). The two engines therefore rescaled a body at
  different times, producing different shapes, masses and trajectories.
  Both now use the machine epsilon.
- **`atan2(-0.0, x > 0)` returned `+0.0`** instead of the IEEE-754
  `-0.0` that the platform libm returns. The deterministic
  trigonometry now preserves the sign of a zero, and the test compares
  the bits (because `0.0 == -0.0` is true).
- **A world with more than 4096 nuclides could be saved but not
  loaded.** The reader rejected the over-long table; the writer did not.
  The writer now refuses it too, so a save can never produce a file the
  same engine cannot load.

### Changed
- The C ABI's thermal config struct **grew**: `ThermalConfig2Ffi` and
  `ThermalConfig3Ffi` carry four new fields (the view-factor model, the
  sky and ground temperatures, and the occluder transmittance) after the
  existing ones. The existing fields keep their offsets, so a C caller
  that only sets the old fields still works, but a caller that allocates
  the struct itself must be recompiled against the new header. The
  managed `ThermalConfig2Ffi` mirror was extended to match.

### Tests
- **Nuclear parity block Y01-Y28** (mirrored Rust↔C#): a disabled
  config is a no-op, the activity halves after a half-life, decay heats
  the body, the temperature respects the thermal clamps, the activity
  is clamped, the energy budget balances, a stable nuclide never decays,
  a body without a nuclide is untouched, a fission consumes fuel and
  releases heat, a supercritical chain reaction grows while a
  subcritical one dies out, delayed neutrons hold the release back, the
  fuel never goes negative, the 3D mirror decays too, determinism plus
  parallel == serial, a static body does not heat up, the nuclide
  table is usable, the decay constant is ln(2) over the half-life, the
  dose is deposited in the nearby bodies and clamped, the nuclear state
  round-trips through a snapshot, and the config and the per-body state
  round-trip through the native FFI.
- **Radiation parity block U01-U12** (mirrored Rust↔C#): the legacy
  view-factor model is the default and matches the historical formula
  bit-for-bit, the symmetric model matches it for equal bodies, is
  order-independent and differs for unequal bodies, the sky/ground split
  changes the exchange and is bit-identical to the historical one when
  the sky and ground equal the ambient, an opaque occluder blocks the
  exchange (including an off-axis one next to an endpoint), a partial
  occluder attenuates it, a fully transparent one does not block,
  determinism plus parallel == serial, and a snapshot round-trip.
- **Deterministic-trigonometry parity block R01–R07** (mirrored
  Rust↔C#): pinned cosine/sine bit patterns (R01), pinned `atan2` bit
  patterns (R02), the engine's own 2D rotation integration pinned to
  exact bits (R03), pinned `exp` bit patterns (R04), pinned `ln` bit
  patterns (R05), agreement with the platform libm within a tolerance
  (R06), and `exp`/`ln` as mutual inverses (R07) — the properties
  that make rigid-body trajectories platform-independent and
  Rust/C#-identical.
- **Broad-phase tier parity block B01–B06** (mirrored Rust↔C#): every
  backend bit-identical trajectories (2D B01, 3D B02), Hybrid matches its
  auto-selected backend (B03), default stays the pre-tier-system grid
  (B04), mid-run backend switch is transparent (B05), body-id invariant
  and FFI round-trip (B06).
- **Solver quality parity block Q01–Q05** (mirrored Rust↔C#): tier
  presets map to iteration budgets and friction/restitution switches
  (Q01), friction-off sliding preserved in 2D/3D (Q02, Q04),
  restitution-off no-bounce (Q03), solver-flag snapshot round-trip at
  schema v7 with default worlds staying on older schemas (Q05).
- **Sleep contract rework (S01, S02)**: the old known-limitation tests
  (no auto-wake, no cascade) are replaced by the new behavior —
  auto-wake on new contact with impact response (S01), transitive wake
  cascade through a resting stack plus settle-back-without-tunneling
  (S02).
- **CCD parity block C01–C04** (mirrored Rust↔C#): fast-ball-vs-thin-wall
  tunneling prevented (2D C01, 3D C02), default-off bit-identical
  trajectories plus schema-v3 JSON without a `ccd` key (C03), and CCD
  snapshot round-trip through JSON and binary (schema v6) with the fast
  ball still caught after save/load (C04).
- **SHM & SPH benchmark suite**: 18 canonical, well-known problems added
  (Rust-only). SHM: damped-oscillator regimes vs the closed-form
  solution, energy-decay rate, `T = 2π√(m/k)` mass scaling, gravity rest
  point, a driven-resonance curve (`ω_r = ω₀√(1-2ζ²)`), undamped-energy
  boundedness (no secular drift), critical-damping fastest-settle, and
  `ω ∝ √k` stiffness scaling. SPH: hydrostatic column (linear + Tait
  EOS), Martin–Moyce dam-break surge, dam break over an obstacle,
  droplet cohesion, free-blob energy, sloshing damping, and channel
  flow. Where the method can't hit the textbook result the test pins the
  gap instead of faking it (no-slip walls, √t front law, penalty-floor
  hydrostatic overshoot) — documented in `LIMITATIONS.md`, readings in
  `absolute_stress_tests.md`.
- **Two-phase SPH test**: `sph_two_phase_heavy_sinks` — a heavy droplet
  (phase 0) submerged in light fluid (phase 1) sinks under gravity.

### Fixed
- **Managed C# snapshot: the CCD tier was 4 bytes wide instead of 1.**
  The managed writer emitted `Ccd.Tier` as a 32-bit int while the Rust
  writer emits a one-byte tag, so every v6+ payload was misaligned
  between the two engines. Latent until now: a managed world only
  wrote a v6+ payload when the CCD config was non-default, and the
  native→managed direction (F36) used a default (v3) payload. Now
  fixed on both sides, and the managed snapshot matches the Rust
  layout byte-for-byte.
- **Managed C# `SetShape`/`PhysicsMaterial.Clone` dropped the thermal
  material properties.** Both rebuilt the material from
  `(friction, restitution, density)` only, silently resetting
  `SpecificHeat`, `ThermalConductivity` and `Emissivity` to their
  defaults. The Rust `set_shape` only touches `material.density`.
  Harmless before the managed thermal mirror existed; now that the
  managed engine steps thermal, it would have reset the thermal
  material on every expansion rescale.
- **Parallel island solver could hang the run (Rust, 2D)**: the shared
  max-applied-impulse slot had a data race. The barrier orders
  "arrived at iteration *g*" against "left iteration *g*", but nothing
  ordered one island's *next* write against another island's *current*
  read — a fast island could overwrite its slot for *g+1* while a slow
  island was still reading it for *g*. The slow island then computed a
  smaller global max, decided to exit early, and left the barrier while
  the others waited for it: a deadlock, not a wrong number. Symptom was a
  *hung* test on a 2-core Windows CI runner (`brutal_parallel_parity_scale_2d`
  sat for 20+ minutes), and the same class of hang was the likely cause
  of the "The operation was canceled" job timeouts with no failing
  step. Fixed by double-buffering the slots per iteration generation
  (a value being read can no longer be overwritten) and by replacing
  `std::sync::Barrier` with a timeout barrier, so any future
  desynchronization panics with a diagnostic instead of hanging the
  run. Regression test: `brutal_parallel_island_stress_2d` (12
  independent islands, repeated rounds, serial cross-check) — it fails
  loudly on the old code and passes on the fixed code.
- **Broadphase hang on oversized bodies (Rust + C#)**: the 2D spatial
  hash inserted a body into *every* cell its AABB overlapped, so a large
  body (e.g. a 1000-unit radius circle at cell size 1.0) touched
  millions of cells and stalled the step. Bodies whose AABB exceeds
  `SpatialHash2::MAX_CELLS_PER_BODY` (~4096 cells) are now tracked in an
  oversized list and paired against the whole scene — correctness is
  preserved (they still collide with everything) while insert cost stays
  bounded. Ordinary scenes never trigger it (bit-identical trajectories).
- **Property tests generated pathological shapes**: the proptest shape
  strategies drew dimensions up to 1000 units, which is what triggered
  the broadphase hang above; shape sizes are now bounded to a physically
  sane 0.1–5.0 units (`energy`/`determinism`/`non-finite` property tests
  went from a multi-hour hang to <0.1 s).

### CI
- **Caching + timeouts**: every Rust job now uses `Swatinem/rust-cache`
  and the .NET jobs use the `setup-dotnet` NuGet cache, so runs no longer
  rebuild all dependencies from scratch. Every job has an explicit
  `timeout-minutes` so a future hang fails fast instead of consuming the
  6-hour default.
- **Split `rust` job into `rust-lint` + `rust-test`**: fmt + clippy +
  doc + a full debug test build and run no longer fit one 30-minute
  budget (green runs were being killed at exactly 30:00 with "The
  operation was canceled" and no failing step). Each half keeps a
  fail-fast timeout that fits it (30 min lint, 45 min tests).
- **Scope debug tests to `--lib --tests`**: skips compiling the 4
  examples in the test job (clippy `--all-targets` still builds them) —
  identical test coverage with fewer Windows debug links, the slowest
  part of the job.
- **Release soak job**: a new `rust-soak` job runs the full workspace
  suite under `--release`, pinning the determinism hashes and allocation
  counts to optimized-code results (the regime where vectorization / FMA
  contraction would alter floats).

### Changed
- **Near-zero-allocation steady state**: the broadphase grid
  (`SpatialHash2`), the contact cache (`World2/3::contact_cache`), and
  the SPH fluid neighbor grids (`FluidState2/3`) now recycle vacated
  per-cell / per-key `Vec` buffers through internal pools instead of
  returning them to the allocator. Measured on the
  1500-body dynamic pile: **~467 → 0.8 allocs/step** (2D), **14 → 0.0**
  (3D), resting stack 0.00. Purely a memory optimization — `contact_cache
  .len()` still counts live pairs exactly, the public API is unchanged,
  and every determinism hash (pile 2D, islands, pile 3D) is
  bit-identical.
- **Persisted boundary ghosts**: SPH ghost particles used to be
  resampled from every static body on every substep; now a fingerprint
  (kernel config + static transforms/shapes) skips both the resample
  and the ghost-grid rebuild while the scene is static. Ghost
  density/pressure still recompute per pass, so results are unchanged —
  boundary scenes are bit-identical and the regen counter
  (`boundary_regens` / `FluidState.BoundaryRegenCount`) proves the skip.
  Managed C# solver mirrors the same scheme.

### Changed
- **Narrowphase SIMD was measured and not shipped.** The 2D projection
  (`project_polygon`) was the best candidate: it is a two-term dot per
  vertex, so a batched SSE form can be made *bit-identical* (the
  horizontal add preserves the `ax*vx + ay*vy` order per lane). It
  measured **slower** than the scalar loop it would replace - 0.83x at 4
  vertices, 0.97x at 8, 0.59x at 16, 0.47x at 32 - because the scalar
  `axis.dot(v)` loop is already efficient and the horizontal add plus
  the per-chunk overhead cost more than they save. No feature flag, no
  `unsafe` pointer cast and no code change; the measurement is recorded
  in `OPTIMIZATION_PLAN.md` §2.2 so it is not re-attempted blind. The
  real opportunity is an SoA-backed hot path (the engine is AoS today).
- **Static-static pair pruning was measured and not shipped.** Moving the
  existing "static-static pairs cannot produce an impulse" check from the
  narrowphase loop to the broadphase output looked like an obvious win
  for static-heavy scenes - it is bit-identical, because the surviving
  pairs keep their relative order. It was measured on a new dense
  static-scene benchmark (400 overlapping static boxes + 20 dynamic
  bodies, ~1.1 ms/step): **1.106 → 1.106 ms/step, i.e. no change at
  all.** The reason is that the narrowphase *already* skipped those
  pairs, so the only remaining cost is the broadphase *generating* them -
  and pruning after the fact cannot avoid that. A real win would need the
  broadphase itself not to emit them, which is a wider API change
  (the broadphase is shared with the managed C# mirror). The benchmark
  scene is kept, because the repo had no static-heavy scenario to
  measure.

### Added
- **Pooled Barnes-Hut tree buffers (Rust)**: the tree arena, the per-leaf
  index buffers and the `positions`/`masses` scratch are pooled in the
  thread-local tree pool, and the "move the bodies down" step reads the
  leaf's list in place (an internal node's list is never read again)
  instead of taking and freeing it. Measured on the 1500-body pile with
  Barnes-Hut gravity: **2473 → 17 allocations per kick**. The tree
  structure and the per-leaf ordering are unchanged, so the gravity
  trajectories (and their determinism hashes) are identical.
- **Pooled parallel-island partition + fixed worker pool (Rust)**: the
  island partition is rebuilt every step (bodies are added, removed and
  reindexed), so its buffers are now pooled in a new
  `island::IslandScratch` owned by the world's scratch. The grouping is
  done with a counting sort over the union-find roots, so no per-island
  index `Vec` is ever created. The parallel solver also runs on a fixed
  worker pool (capped at the available parallelism) instead of one OS
  thread per island — a settled pile has ~140 islands, and spawning
  ~140 threads per step cost more than the solve it was parallelising
  (the parallel path was measurably *slower* than the serial one).
  Measured on the 1500-body pile: **785 → 0.3 allocations per step**,
  i.e. the parallel path now matches the serial path's allocation
  floor. `serial_hash == parallel_hash` still holds: the islands are
  body-disjoint, so the per-island computation, the max-slot layout and
  the per-generation barrier are unchanged. Below 256 island constraints
  the serial path is used (identical result, no thread overhead).
- **Managed C# thermal mirror**: `PhiEng.Dynamics` `World2/3` now step
  the same thermal subsystem as the Rust core — conduction, radiation,
  convection, plus the three opt-in sub-features (radiation occlusion,
  forced convection, thermal expansion). The managed phase runs in the
  same position as the Rust one (after velocity integration, before the
  sleep check). `RigidBody2/3` gained `Temperature` and `ExpansionScale`;
  `ThermalConfig2` gained `IsActive()`/`IsDefault()` and the shared
  physical constants; the new `ThermalIntegration`/`ThermalGeometry`
  types carry the ported algorithm and the shape-geometry helpers.
  The managed snapshot layout is now **byte-identical to the Rust
  writer's** (schema v10, including the v8 thermal block, the v9
  collision-tier block and the broad-phase tag the managed engine does
  not model), so a file written by either engine loads in the other.
  Parity block T26–T37 (mirrored Rust ↔ managed C#) plus F40, which
  loads a managed-written file in the Rust core.
- **Thermal sub-features (Rust + managed C#)**: three opt-in additions to
  the thermal subsystem, all off by default so default trajectories stay
  bit-identical.
  - **Radiation occlusion**: `radiation_occlusion` blocks body-to-body
    radiation when a third body lies between the pair (a coarse
    segment/sphere test against the blocker's bounding radius).
  - **Forced convection**: `forced_convection_coefficient` makes the
    convection coefficient `h + h_forced·|v|`, so a body moving through
    still air sheds heat faster than a stationary one.
  - **Thermal expansion**: `expansion_coefficient` +
    `expansion_reference_temperature` scale a body's shape by
    `1 + alpha·(T − T_ref)` and refresh its mass and inertia. The
    applied factor is tracked per body and applied *relative to the
    authored shape*, so a body held at a constant temperature does not
    grow without bound.
  New FFI: `phieng_world_set_body_velocity`,
  `phieng_world[_3]_get_body_shape_radius`; managed
  `ThermalConfig2/3` properties plus
  `PhiNativeWorld(3).SetBodyVelocity/GetBodyShapeRadius`.
  Persisted through **snapshot schema v10** (conditional: default
  worlds stay on older schemas byte-identical, and v8/v9 payloads still
  load). Parity block T18–T24 (mirrored Rust + C#).
- **Solver quality tiers (Rust + managed C#)**: `SolverQuality`
  (`Ultra` 8 iters down to `Minimal` 1 iter) plus `enable_friction` /
  `enable_restitution` switches on `Solver`/`Solver3D`
  (`ApplyQuality()` in C#). Low/Minimal skip the friction pass
  (~half the solve cost). Persisted through **snapshot schema v7**
  (conditional: only worlds with disabled switches bump; everything
  else stays byte-identical).
- **Full sleeping (Rust + managed C#)**: sleeping bodies are frozen —
  skipped in gravity, ambient, fluid force application, integration,
  and the contact solver (treated as infinite mass, exactly like
  statics), so resting piles cost ~nothing. Wakes: new-contact
  auto-wake, lost-contact auto-wake (support pulled away), direct
  impulse/force application, joint partners, and transitive
  `wake_body` cascades. A body falls asleep only when slow *and*
  supported by a current contact (ballistic apexes and drifters never
  sleep), so no wake history exists and save/load stays exact with no
  schema change. Replaces the old partial-sleep behavior (which moved
  "sleeping" bodies); S01/S02 now pin the new contract.
- **CCD via adaptive substepping (Rust + managed C#)**: a new
  `phieng_dynamics::ccd::CcdConfig` (`enabled`, `max_movement` default
  `0.1`, `max_substeps` default `8`) shared by 2D and 3D worlds. When
  enabled, each world step splits into substeps so no body travels more
  than `max_movement` per substep (linear plus angular sweep); each
  substep runs the full pipeline. Off by default — bit-identical
  trajectories. Persisted through **snapshot schema v6** (conditional:
  default worlds stay v3/v4/v5 byte-identical), crossed via new
  `phieng_world[_3]_set/get_ccd_config` FFI plus managed
  `PhiNativeWorld(3).SetCcdConfig/GetCcdConfig`, and mirrored in
  managed `World2/3.Step` with native-vs-managed behavior parity.
- **Fluid wall no-slip** (`wall_friction` ∈ [0, 1], default `0` =
  free-slip, bit-identical): at a particle/body contact the tangential
  velocity is driven toward the wall's tangential velocity (reaction to
  dynamic bodies), removing tangential momentum and curving a driven
  channel toward the parabolic Poiseuille/Chapman profile. Opt-in;
  extends snapshot schema v5 and the managed `FluidConfig`/`Snapshot`
  mirror. (Also fixes the v5 schema trigger to include
  `boundary_particles`.)
- **Ambient force fields (Rust + managed C#)**: a new
  `phieng_dynamics::ambient` module with three independently-toggled
  fields applied as a pre-solve velocity kick (same pipeline slot and
  Verlet half-kick contract as gravity):
  - **Spring / simple-harmonic-motion** (`SpringConfig2/3`): a linear
    restoring force `F = -k*(x - anchor) - c*v` with an optional
    rest-length leash, giving the textbook damped oscillator
    (`ω = sqrt(k/m)`, `ζ = c/(2*sqrt(k*m))`).
  - **Air resistance** (`DragConfig2/3`): linear (Stokes) plus quadratic
    (`0.5*rho*Cd*A*|v_rel|*v_rel`) drag against an ambient wind, with an
    angular-drag torque. Reference area is derived per shape
    (circle/sphere analytic, box/capsule/polygon hull proxies).
  - **Air turbulence** (`TurbulenceConfig2/3`): deterministic seeded
    value-noise fBm (see `phieng_dynamics::noise`), optionally
    divergence-free via curl noise. Bounded to the configured amplitude.
  - Per-body `ambient_scale` coupling multiplier (0 = immune), mirroring
    `gravity_scale`.
  - Clock: `World2/3::time` advances by `dt` each step and drives
    time-varying turbulence.
- **Ambient FFI surface**: `phieng_world[_3]_set/get_ambient_config`
  (repr(C) `AmbientConfig2Ffi`/`AmbientConfig3Ffi`),
  `phieng_world[_3]_set/get_time`, and
  `phieng_world[_3]_set_body_ambient_scale`.
- **Managed mirror**: `Ambient.cs`, `AmbientIntegration.cs`,
  `TurbulenceNoise.cs` in `PhiEng.Dynamics`; `World2/3.Ambient` + `Time`;
  `RigidBody2/3.AmbientScale`; native wrappers in `PhiNativeWorld(3)`.
- **Snapshot schema v5**: adds the ambient config block, the simulation
  clock, and per-body `ambient_scale` (conditional: default worlds stay
  v3 and gravity-only worlds stay v4, byte-identical to older files).
- **Ambient parity block A01–A05** (mirrored Rust↔C#): spring period,
  drag terminal velocity, turbulence determinism, default no-op, and
  scale immunity.
- **Fluid SPH expansion (Rust; config mirrored to C#)**: new
  `FluidConfig2/3` tunables, all default-off so existing fluids are
  bit-identical:
  - **Tait equation of state** (`gamma`, classic water value `7`):
    `p = max(0, k*rho0*((rho/rho0)^gamma - 1))` replaces the linear EOS
    when `gamma > 1`, supporting deep hydrostatic columns with far less
    artificial stiffness. `gamma == 1` keeps the legacy linear form.
  - **Monaghan artificial viscosity** (`artificial_viscosity`, `[0, 2]`):
    a pairwise viscous pressure damping approaching particles; stable at
    low XSPH settings.
  - **Surface tension / cohesion** (`surface_tension`): a bounded
    Akinci-style attractive force that holds droplets together and
    rounds free surfaces.
  - **CFL substepping** (`cfl`, `max_substeps`): when enabled, the fluid
    pipeline runs `ceil(dt / (cfl*h/(c0+v)))` times (capped) so stiff
    Tait configs stay stable at 60 Hz. Off by default (single pass,
    bit-identical).
  - **Fluid weather coupling**: the SPH body drag now uses the ambient
    wind as the relative velocity (a current sweeps floating bodies),
    and the ambient turbulence field drives fluid particles directly.
  - **Adaptive resolution** (`adaptive_resolution`): particles carry a
    per-particle mass; under-dense particles split (mass halved) and
    over-dense close pairs merge (mass/momentum combined), conserving
    total mass, bounded by the particle cap and a minimum child mass.
    Off by default; best suited to calm/settled scenes.
  - **Boundary particles** (`boundary_particles`): static bodies are
    sampled into fixed ghost particles (circle/box/polygon in 2D;
    sphere/box/capsule/hull in 3D) that exert SPH pressure on the
    fluid via a two-pass density/pressure scheme — a pressure-based wall
    that reduces tunneling vs the penalty boundary alone. Ghosts persist
    across substeps and regenerate only when the static set/transforms or
    kernel config change (FNV fingerprinted); their density/pressure
    still recompute every pass. Never persisted.
  - **Two-phase fluids**: particles carry a per-particle `phase` index
    (0..=3) and each phase gets its own `phase_rest_density` /
    `phase_mass`, so a heavier phase sinks under gravity (Rayleigh–Taylor,
    oil/water buoyancy). Added via
    `FluidState2/3::add_particles_phase`; single-phase fluids keep phase 0
    and are bit-identical. Multi-phase caveats (phase-clamp, EOS-only) are
    documented in `LIMITATIONS.md`.
  - **Fluid FFI surface**: `phieng_world[_3]_set/get_fluid_config`
    (repr(C) `FluidConfig2Ffi`/`FluidConfig3Ffi`, validated on parse —
    bad input rejected without mutating the world) and
    `phieng_world[_3]_fluid_add_phase` for phase-tagged particle insert,
    with managed `PhiNativeWorld(3).SetFluidConfig/GetFluidConfig/
    AddFluidParticles(Phase)/FluidCount/FluidClear/GetFluidParticles`
    and native interop round-trip tests.
  - **Managed C# fluid stepping**: `FluidIntegration.StepFluid2` /
    `FluidIntegration3.StepFluid3` mirror the Rust solver stage-by-stage
    (kernels, neighbor grid, density/pressure incl. two-phase EOS,
    adaptive resampling, boundary ghosts, Monaghan viscosity, cohesion,
    CFL substepping, XSPH, two-way rigid coupling, ambient
    wind/turbulence) and run as phase 1b of managed `World2/3.Step`.
    Cross-language contract is behavior parity (tolerance), not
    bit-equality; `FluidTests` (determinism, containment, two-phase,
    mass conservation, turbulence determinism, 3D) plus
    native-vs-managed equivalence tests pin it.
  - Preserved through snapshot schema v5 (fluid config block extended,
    version-gated); the schema switches to v5 when any fluid-expansion
    field is non-default.

- **Thermodynamics & radiation (Rust + managed C#)**: a new
  `phieng_dynamics::thermal` module that adds per-body temperature
  evolution via three heat-transfer mechanisms, run as a new phase 5c
  of `World2/3::step_inner` after integration and before sleep check.
  Sleeping bodies still exchange heat; thermal transfer never wakes
  bodies.
  - **Conduction**: contact-driven, series thermal resistance via the
    harmonic mean of both bodies' conductivities, contact geometry from
    narrowphase results. Off by default.
  - **Radiation**: Stefan–Boltzmann T⁴ law, body-to-environment plus
    body-to-body (distance-bounded, simplified view factor), effective
    emissivity for two-gray-body exchange. Off by default.
  - **Convection**: Newton's law of cooling to ambient temperature.
    Off by default.
  - Per-body `temperature` (Kelvin, default 293.15 K) and material
    properties `specific_heat` J/(kg·K), `thermal_conductivity` W/(m·K),
    `emissivity` 0..1 (extend `PhysicsMaterial`).
  - `ThermalConfig2/3`: master switch, ambient temperature,
    Stefan–Boltzmann constant, convection coefficient, per-mechanism
    toggles, temperature clamps, radiation cutoff factor, max rate.
- **Thermal FFI surface**: `phieng_world[_3]_set/get_thermal_config`
  (repr(C) `ThermalConfig2Ffi`/`ThermalConfig3Ffi`),
  `phieng_world[_3]_set/get_body_temperature`,
  `phieng_world[_3]_set/get_body_thermal_material`.
- **Collision tiers (Rust + managed C#)**: `CollisionTier`
  (`Tier1_AABBOnly` ... `Tier5_GJK_EPA`) caps each shape pair at the
  cheapest separating test (sphere-AABB, sphere-sphere, SAT box-box,
  GJK/EPA); pairs above the ceiling are *skipped* (no contact
  generated), never guessed. Off by default, so a default world is
  bit-identical; maps from `SolverQuality`. FFI
  `phieng_world[_3]_set/get_collision_tier` (repr(C)
  `CollisionTierConfigFfi`) plus managed
  `PhiNativeWorld(3).SetCollisionTier/GetCollisionTier` and
  `PhiEng.Broadphase.CollisionTier` / `CollisionTierConfig`.
  Persisted through **snapshot schema v9** (conditional: default worlds
  stay on their old schema byte-identical). Parity block H01–H05
  (mirrored Rust→C#): the default is full fidelity, a ceiling
  below a pair's required tier really does stop the contact, a ceiling
  at that tier keeps it, the 3D mirror behaves the same, and capped
  runs stay deterministic.
- **Broad-phase tiers (Rust + managed C#)**: `BroadphaseMode`
  (`BruteForce` / `SpatialHash` / `DynamicAABBTree` / `Hybrid`), a
  `DynamicAABBTree` backend for 2D and 3D, and an explicit O(n²)
  `BruteForce` backend — so the mode is a performance choice, never a
  behavior change (every backend emits the same sorted+deduped pair
  sequence; all modes agree bit-identically on B01/B02). Selection:
  brute force below 50 bodies, spatial hash 50–5000, dynamic tree above
  (see `BroadphaseMode::auto_select`); `Hybrid` re-resolves from the live
  body count at the top of every step. Default stays the pre-tier-system
  grid, so existing callers and files are unaffected. FFI
  `phieng_world[_3]_set/get_broadphase_mode` plus managed
  `PhiNativeWorld(3).SetBroadphaseMode/GetBroadphaseMode` and
  `PhiEng.Broadphase.BroadphaseMode`. Mode persists via schema-v8
  `broadphase_mode` tag (conditional: default worlds stay on older
  schemas byte-identical).
- **3D spatial-hash broadphase**: `SpatialHash3` (the 3D counterpart of
  `SpatialHash2`) with the same contract — pooled per-cell buffers (no
  steady-state allocation), a bounded insert cost (oversized bodies pair
  against the whole scene instead of filling millions of cells), and
  canonical sorted+deduped pairs. 3D `SpatialHash`/`Hybrid` used to fall
  back to all-pairs (O(n²)); they now use the real grid. The
  default 3D backend therefore changed from brute force to the grid —
  a performance change only, with bit-identical trajectories pinned by
  B02 (all three 3D backends agree).
- **CCD tiers (Rust + managed C#)**: `CcdTier`
  (`Off` / `SweptAABB` / `FullGJK`) selects how aggressively tunneling is
  prevented — off pays nothing, swept-AABB substeps only pairs whose
  swept volumes overlap during the step, full substeps every body. Tier
  maps from `SolverQuality` (`Ultra`→Full, `High`→Swept, else Off) and
  persists via the v6 `ccd` block's new `tier` field (conditional: files
  without CCD stay on older schemas byte-identical). FFI
  `CcdConfigFfi.tier` plus managed `CcdConfig.Tier`; C01–C04 and the FFI
  round-trips pin the new field.
- **PhysicsConfig facade (Rust)**: `PhysicsConfig` unifies
  `solver_quality`, `collision_tier`, `broadphase_mode`, sleep
  thresholds, `ccd`, and `thermal` under one struct with
  `Default` (= pre-tier-system behavior exactly), `low_end()`, and
  `high_fidelity()` presets, plus `apply_to_2d`/`apply_to_3d` to push a
  config into a live world idempotently.
- **Managed mirror**: `ThermalConfig.cs` in `PhiEng.Dynamics`;
  `World2/3.Thermal` + `RigidBody2/3.Temperature`;
  native wrappers in `PhiNativeWorld(3)`.
- **Snapshot schema v8**: adds the thermal config block, per-body
  temperature, and material thermal properties (conditional: default
  worlds stay on older schemas byte-identical).
- **Snapshot schema v9**: adds the collision-tier block (conditional:
  default worlds stay on their old schema byte-identical; older files
  load with the tier ceiling at full fidelity).
- **Thermal parity block T01–T17** (mirrored Rust↔C#): conduction
  equilibrium, conductivity scaling, zero-conductivity no-flow,
  temperature clamping (max/min), static-body thermal reservoir,
  disabled-config no-op, conduction disabled, determinism.

## [0.4.0] - 2026-10-16

### Added
- **Gravitational physics core (Rust)**: `GravityMode` (uniform /
  central attractor / full N-body pairwise with softening),
  per-body `gravity_scale`, and a velocity-Verlet integrator option,
  in a new `phieng_dynamics::gravity` module. Snapshot schema is now
  v4 (conditional: default-gravity worlds still write byte-identical
  v3; loaders accept both).
- **Barnes-Hut approximation (Rust)**: `GravityMode::BarnesHut`
  with per-world `theta` replaces the O(n^2) pairwise loop with a
  deterministic quadtree/octree (index-order insertion, fixed-order
  traversal/accumulation, self-force exclusion, depth-capped
  termination). Same-mode runs are bit-identical; exact-vs-BH
  trajectories differ by design. Measured crossover at theta 0.5 is
  ~700 bodies (2.5x faster at n=2000); exact stays the default.
- **Gravity FFI surface**: `set/get_gravity_config` (mode, integrator,
  G, softening, attractor, mu, theta), per-body
  `set/get_body_gravity_scale`, and `set/get_sleep_thresholds` (2D +
  3D, same null-tolerant status-code conventions).
- **Managed C# gravity mirror**: `GravityMode`/`Integrator`/
  `GravityConfig2/3`, per-body `GravityScale`, mode-aware kicks +
  velocity-Verlet drift and a deterministic Barnes-Hut port in
  `PhiEng.Dynamics`, with snapshot schema v4 support (conditional
  versions, range-checked) in `Snapshot.cs`. Managed/native agreement
  is within tolerances (bit-equality never claimed cross-language).
- **Bulk FFI transfers**: `add_bodies` / `get_transforms_bulk`
  (2D + 3D) move whole arrays in one transition (per-slot `-1`/NaN
  failure semantics, no rollback); `Try*` getter overloads and
  `SuppressGCTransition` on trivial calls cut polling overhead.
- **Gravity parity block G01–G10** (mirrored Rust↔C#, enforced):
  orbit stability, N-body determinism, Verlet energy order,
  softening safety, probe exemption, snapshot round-trips, BH
  determinism, plus a compact 3D set.
- **`bench_gravity` example**: exact-vs-BH crossover harness
  (deterministic hashes, portable profile).

## [0.3.1] - 2026-10-15

### Fixed
- **3D friction cone was a box, not a cone** (`Solver3D`): tangent
  impulses were clamped independently per axis (`|jt1| <= mu*jn`,
  `|jt2| <= mu*jn`), allowing `|jt|` up to `sqrt(2) * mu*jn` — 41%
  over the Coulomb cone at 45 degrees. The tangent impulse *vector*
  is now clamped to the disk `|jt| <= mu*jn`. Surfaces that should
  grip (boxes on slopes, corner contacts) no longer slide.
- **Distance joint deadlock at zero separation**: when both anchors
  coincided while `rest_length > 0`, `axis()` returned `None` and the
  frame was skipped — the joint exerted zero force forever. It now
  falls back to the relative-velocity axis (then `Vec2::X` when both
  bodies are at rest), so overlapping bodies push apart to rest
  length.
- **EPA seeding failed for small shapes**: `STRICT_TET_MARGIN` was an
  absolute `1e-4`, so shapes smaller than the margin could never
  seed a tetrahedron — `gjk_epa` returned `None` and the collision
  was missed. The margin now scales with the candidate support
  points' own extent (AABB diagonal), keeping it a relative tolerance
  at any scene scale.
- **GJK iteration exhaustion could report separation for an actual
  overlap**: with a full tetrahedron (`count == 4`) after 32
  iterations, GJK now reports `Colliding` instead of falling through
  to the simplex distance — closes a tunneling path for deep,
  high-iteration overlaps.
- **Near-zero effective mass produced inf/NaN**: the guard was
  `mass_n == 0.0` (exact). Subnormal effective mass (e.g. near-static
  bodies) yielded huge `lambda` values that poisoned velocities with
  inf/NaN. Guard is now `mass_n < EPS^2`.

### Changed
- **Default sleep thresholds 0.01 -> 0.05** (linear and angular, 2D
  and 3D): bodies no longer sleep while still visibly moving
  (0.01 m/s = 0.6 mm/frame at 60 Hz; now ~3 mm/frame).
- **Shared numeric tolerance constants** (`phieng_math`): `EPS`,
  `EPS_SQ`, `EPS_LINE`, `EPS_PLANE`, `EPA_TOL`, `GJK_EPS`,
  `MAX_BIAS_VELOCITY`, `DEFAULT_SLOP`, `DEFAULT_BAUMGARTE`,
  `DEFAULT_ITERATIONS`, `EARLY_EXIT_TOL`,
  `LINEAR_SLEEP_THRESHOLD`, `ANGULAR_SLEEP_THRESHOLD`,
  `CONTACT_MATCH_DIST_SQ`, `RESTITUTION_THRESHOLD`, and `scaled_eps()`
  replace per-file magic numbers across the solver, joints, world,
  fluid, narrowphase, and material code. `MIN_DENSITY` is now
  `phieng_math::EPS`.
- **Release profile**: `lto = "thin"`, `codegen-units = 1`;
  `overflow-checks = true` in dev/test/release.
- **Staged native library rebuilt in release**: the committed
  win-x64 `phieng.dll` was a stale debug build (2.6 MB); it is now
  the release build (0.87 MB) and matches the current sources.
- **Baseline hashes moved** (README baseline matrix): the physics
  fixes above change trajectories, so the recorded pile/islands/
  pile3d hashes were re-recorded. The determinism gate (serial ==
  parallel, snapshot round-trip bit-identical) still passes.

## [0.3.0] - 2026-10-15

### Added
- **SPH fluid (2D + 3D)**: weakly-compressible particle fluid
  (`FluidState2`/`FluidState3`: poly6 density, linear pressure
  equation of state, spiky-gradient pressure force, XSPH viscosity)
  stepped as phase 1b of `World2::step`/`World3::step`, with two-way
  rigid coupling (boundary impulses plus pressure-sampled buoyancy and
  Morison-style drag). Empty fluid is a no-op. C# fluid stepping is
  still pending (state round-trips; it does not simulate).
- **Fluid snapshots (schema v3)**: particle positions/velocities and
  fluid config persist in JSON and binary, round-trip bit-exactly, and
  continue bit-identically in Rust and C# data paths.
- **Fluid FFI bridge**: `physics_world[_3]_fluid_add/count/clear/get`
  with null-safety and atomic validation, mirrored in
  `NativeMethods`.
- **ConvexMesh shape (3D)**: closed, outward-wound convex triangle mesh
  with exact Mirtich-style mass properties (volume, center of mass, and
  inertia tensor), strict creation-time validation
  (`TooFewMeshVertices`, `InvalidMeshIndex`, `DegenerateMesh`,
  `NonConvexMesh`), and automatic center-of-mass recentering on
  `set_shape`. Collision reuses the convex GJK path.

### Changed
- **Snapshots are now schema v3**: fluid state section plus the v2
  `convexmesh` shape DTO/binary tag `4`; loaders still reject unknown
  versions instead of guessing. Managed (C#) mesh DTO support is still
  pending.

### Fixed
- **Capped Baumgarte bias velocity at 2.0 m/s** (Rust `Solver::MAX_BIAS_VELOCITY`,
  mirrored in C# `Solver`/`Solver3D`): deep overlaps now resolve
  gradually over several frames instead of ejecting bodies past discrete
  collision. Contacts shallower than ~17 cm at 60 Hz are bit-identical,
  so settled stacks, joints, and warm-start behavior are unchanged.

## [0.2.0] - 2026-10-14

### Added
- **3D Joints**: `DistanceJoint3`, `HingeJoint3`, `PrismaticJoint3`, `FixedJoint3`
  with full validation, warm-start, limits, and snapshot support
  (K31–K44, F37–F39 parity tests)
- **Multi-RID Native Builds**: Portable baseline binaries for
  `win-x64`, `linux-x64`, `osx-x64`, `osx-arm64` via
  `tools/build-native-all.{ps1,sh}`
- **Cross-Platform Native Loading**: `DllImportResolver` in
  `NativeMethods` resolves `phieng.dll/.so/.dylib` from
  `runtimes/{rid}/native/` automatically
- **NuGet Packaging**: All public assemblies (`PhiEng`,
  `PhiEng.Math`, `PhiEng.Collision`, `PhiEng.Broadphase`,
  `PhiEng.Dynamics`, `PhiEng.Native`) publishable with
  `GenerateDocumentationFile`, versioned via `Directory.Build.props`
- **Hello-World Samples**: Five 5-minute samples in `samples/`
  (`hello-2d`, `hello-3d`, `hello-joint`, `hello-snapshot`, `hello-native`)
  using project references for instant local iteration
- **CI Enhancements**: `host-smoke` (from-scratch host example) and
  `readme-smoke` (hello samples validation) jobs

### Changed
- **3D Snapshots**: Added `Joint3Dto` serialization with exact/fresh modes
- **Native Loading**: Replaced hardcoded DLL path with `DllImportResolver`
  that probes `runtimes/{rid}/native/` next to the assembly
- **Native Distribution**: Native binaries now copied to output via
  `Content` items with `CopyToOutputDirectory="PreserveNewest"`

### Fixed
- **Parity Script**: Updated to recognize `unsafe` C# test methods
- **Host Smoke CI**: Fixed broken `examples/host/PhiEng.Host.csproj`
- **Native DLL Loading**: Fixed `DllNotFoundException` via `Content`
  items with `CopyToOutputDirectory="PreserveNewest"`

## [0.1.0] - 2026-10-13 (initial)

### Added
- Core 2D/3D rigid-body dynamics (sequential impulse solver)
- 2D joints (distance, hinge, prismatic, fixed) with validation
- 2D/3D collision detection (SAT, GJK/EPA, broadphase)
- Snapshots (JSON + binary, exact/fresh modes, CRC32)
- FFI bridge (`phieng_world_add_joint`, snapshots, etc.)
- Property-based tests + differential fuzzing (1000 cases)
- Parallel island solver (bit-identical to serial)
- Benchmarks (Rust `bench_large`, C# `PhiEng.Bench`)

---

## Versioning Policy

- **SemVer** (`MAJOR.MINOR.PATCH`)
- **MAJOR**: Breaking API changes (e.g., removed types, changed signatures)
- **MINOR**: New features, new joint types, new public APIs
- **PATCH**: Bug fixes, performance improvements, CI/docs

Version is single-sourced in `Directory.Build.props` (`<Version>`).
Release flow: tag `vX.Y.Z` → CI builds/publishes NuGet + GitHub Release.

---

## Release Checklist

- [ ] All tests green (Rust + C# + parity)
- [ ] `cargo fmt --check`, `cargo clippy -D warnings`, `cargo doc -D warnings`
- [ ] `dotnet build -c Release`, `dotnet test`, `dotnet pack`
- [ ] `cargo build --release -p phieng` (all RIDs via `build-native-all`)
- [ ] Update `CHANGELOG.md`
- [ ] Tag `vX.Y.Z` and push
- [ ] CI publishes NuGet packages + GitHub Release with native artifacts