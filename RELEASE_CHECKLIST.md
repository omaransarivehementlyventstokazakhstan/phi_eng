# PhiEng release checklist

Run top to bottom for every release. Nothing ships with a red box.

## 1. Hygiene gate (mirrors CI)

- [ ] `cargo fmt --all --check` clean.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo doc --workspace --no-deps` clean with `RUSTDOCFLAGS=-D warnings`.
- [ ] `cargo test --workspace` green (all targets, not just `--lib`).
- [ ] `dotnet test PhiEng.slnx -c Release` green.
- [ ] `python tools/check_test_parity.py` reports PARITY OK.

## 2. Version alignment

- [ ] `Cargo.toml` workspace `version`, `Directory.Build.props` `<Version>`,
      and the top `CHANGELOG.md` entry all agree (CI enforces the first two).
- [ ] `CHANGELOG.md`: `[Unreleased]` section renamed to the version with
      today's date; entries describe user-visible behavior changes.

## 3. Native artifacts

- [ ] Rebuild portable binaries for all RIDs
      (`tools/build-native-all.ps1 -Rids win-x64,linux-x64,osx-x64,osx-arm64`).
- [ ] Restage into `src/PhiEng.Native/runtimes/` **and** the legacy
      `native/` dir. Stale DLLs fail cross-language tests with a schema
      mismatch (seen before: F36 against a v1 DLL) — never skip this.
- [ ] `dotnet test` passes against the restaged binaries (covers F36).

## 4. Unsafe audit (per release)

Library `unsafe` inventory (all documented at the use site; re-verify
any new additions):

- `crates/dynamics/src/solver.rs` — parallel island solve reconstructs
  `&mut` slices from raw addresses over disjoint index sets. Sound
  iff body-disjointness AND constraint-index disjointness hold;
  debug builds re-verify both before spawning threads. The shared
  max-applied-impulse slots are **double-buffered per generation**
  (one buffer per parity): a single shared slot races, because the
  barrier orders "arrived at generation g" against "left generation g"
  but not one thread's next write against another thread's current
  read — that race let islands disagree on the early-exit decision and
  hang the run. The barrier also carries a timeout, so any future
  desync panics loudly instead of hanging CI.
- `crates/dynamics/src/world.rs` — parallel joint solve, same contract
  (disjoint island index sets, debug-verified).
- `crates/engine/src/ffi.rs` — raw-pointer boundary: every entry point
  tolerates null (documented error codes); slices are built from
  caller-validated lengths and capped (`1_000_000`, particle caps).
- `crates/engine/examples/bench_large.rs` — counting global allocator
  (example only, never ships in the library).
- `crates/math/src/simd.rs` — `std::arch` SSE intrinsics (`_mm_loadu_ps`
  / `_mm_storeu_ps` family) behind the opt-in `simd` feature flag.
  Sound iff every pointer is valid for 4 `f32` reads/writes and every
  test in `simd::tests` (tolerance-compared vs the scalar fallback; the
  element-wise ops are bit-exact, `dot4` reassociates the sum) passes;
  non-x86 targets compile the scalar fallback with no `unsafe` at all.

Rules for new `unsafe`: debug assertion or test proving the contract,
plus a line in this section.

## 5. Fuzz + soak

- [ ] `cargo test -p phieng --test differential_fuzz` green (covers
      rigid bodies + fluid particles).
- [ ] Release brutal suite green with readings refreshed in
      `absolute_stress_tests.md` if any threshold moved:
      `cargo test --release -p phieng --test brutal -- --nocapture --test-threads=1`.

## 6. Docs

- [ ] README feature list / quickstart / compat matrix current.
- [ ] `CHANGELOG.md` finalized (see §2).
- [ ] `LIMITATIONS.md` reflects any fixed or newly found gaps.
- [ ] New public API has rustdoc + XML doc comments (CS1591-clean on
      `PhiEng.Dynamics`; see `API_STABILITY.md`).

## 7. Publish

- [ ] Tag `vX.Y.Z`; `release-native.yml` stages per-RID archives.
- [ ] `publish-nuget.yml` dry-run first (`workflow_dispatch` without
      version), then publish on release.
- [ ] After publish: bump workspace + props versions to next `-dev`?
      (Current policy: version files always hold the *next* release
      number; see `API_STABILITY.md`.)
- [ ] Rename the GitHub repository itself if not yet done
      (`github.com/omaransarivehementlyventstokazakhstan/phi_eng`);
      code URLs already point at the new name.
