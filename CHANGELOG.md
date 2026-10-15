# Changelog

All notable changes to PhiEng are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

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