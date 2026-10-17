# Changelog

All notable changes to PhiEng are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Added
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
    that reduces tunneling vs the penalty boundary alone. Regenerated
    every substep (statics don't move); never persisted.
  - Preserved through snapshot schema v5 (fluid config block extended,
    version-gated); the schema switches to v5 when any fluid-expansion
    field is non-default.

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