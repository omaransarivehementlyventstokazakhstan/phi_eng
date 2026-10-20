# Contributing to PhiEng

Thanks for contributing. This document describes the workflow and the
rules that keep the test suite, the cross-language parity, and the
determinism guarantees intact. If a rule below seems arbitrary, it is
almost certainly pinned by a failing test somewhere — ask before
bending it.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Workflow

1. Fork the repo and create a branch from `main`.
2. Make your change (see style and testing rules below).
3. Run the full hygiene gate locally (see below) — CI runs the same
   commands, and every job must pass before merge.
4. Open a pull request against `main` describing behavior changes,
   new test IDs, and any snapshot schema impact.
5. Keep the PR focused: one feature or fix per PR; docs (`CHANGELOG.md`,
   `LIMITATIONS.md`, `README.md`) updated in the same PR as the code.

## Hygiene gate (mirrors CI)

Run these from the repo root; all must be clean:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps
cargo test --workspace --lib --tests
dotnet test PhiEng.slnx -c Release
python tools/check_test_parity.py   # must print PARITY OK
```

Additional gates CI runs: `rust-soak` (`cargo test --workspace --release`),
`version-check` (Cargo workspace version == MSBuild version),
`msrv` (Rust 1.78 library build), `bench-smoke`, `host-smoke`,
`readme-smoke`. See `.github/workflows/ci.yml` and
`RELEASE_CHECKLIST.md` §1 for the full list.

## Code style

- Rust: `cargo fmt` is the arbiter; `clippy -D warnings` must stay
  clean. Every `pub` item needs a `///` rustdoc comment
  (`RUSTDOCFLAGS=-D warnings` fails the build otherwise).
- C#: XML doc comments on all `public` types and members
  (`PhiEng.Dynamics` is CS1591-clean; keep it that way). No emojis in
  code or comments.
- Simulation code is `f32` throughout and tick-based — no wall-clock
  inside `step()`. Same inputs + same steps must give bit-identical
  state within one backend.
- MSRV for library code is Rust **1.78**. Examples and benches may use
  newer sugar; library code may not.

## Testing rules

- **Mirror discipline.** Behavioral tests carry a parity ID
  (`S01`, `W06`, `Q01`, `C01`, `T01`, `B01`, `K31`, …) and must exist in
  both Rust (`crates/`) and C# (`tests/PhiEng.Tests/`), with matching
  names (`fn t01_…` ↔ `T01_…`). `tools/check_test_parity.py` enforces
  this — a green suite with a red parity check means the mirror is
  drifting. Rust-only exception: analytic benchmarks and fuzz suites
  that have no meaningful C# counterpart.
- **Determinism.** Same build + same inputs + same steps = bit-identical
  state, within one backend. Cross-language parity is *behavioral*
  (tolerance), never bit-exact — different float libraries. Never
  "fix" a test by loosening a hash or tolerance without understanding
  why it moved; a moved hash is a behavior change until proven otherwise.
  The **rigid-body core** goes further: it must stay platform-independent
  (Linux, Windows and macOS agree bit-for-bit, and so do Rust and the
  managed C# mirror). That means no platform math library in the rigid
  path — use `phieng_math::trig` / `PhiEng.Math.DeterministicTrig` for
  sine, cosine and `atan2`. The fluid and ambient subsystems are exempt
  (see `LIMITATIONS.md`).
- **Never guess.** Ambiguous inputs are rejected at creation
  (`ShapeError`, `JointError`, `ArgumentException`), never silently
  clamped — except friction/restitution, which keep clamp-to-range.
  The same rule applies to snapshots: unknown versions, tags, or
  trailing bytes are errors, never reinterpretation.
- **New snapshot fields are conditional.** Omit-when-default on write
  (`skip_serializing_if`), default-on-read for old files, version-gated
  in binary. Default worlds must stay byte-identical to older schemas;
  add a round-trip test in both JSON and binary.
- **New `unsafe` needs a contract proof.** A debug assertion or test
  proving the safety condition, plus a line in `RELEASE_CHECKLIST.md`
  §4 (unsafe audit). See the solver's island-disjointness checks for
  the pattern.

## FFI rules (`crates/engine/src/ffi.rs` ↔ `PhiEng.Native`)

- Null pointers yield documented error codes (`0`/`-1`), never UB.
- Validate-then-commit: reject invalid input without mutating the world.
- New `extern "C"` functions need: a `#[repr(C)]` struct if they take
  config, `from_ffi`/`to_ffi` converters with full validation, a 2D +
  `world3` 3D pair, `LibraryImport` declarations plus managed wrappers
  in `PhiNativeWorld(3)`, and round-trip tests on both sides.

## Docs to update with your change

- `CHANGELOG.md` `[Unreleased]` — behavior changes only, user-facing
  wording (see existing entries for the house style).
- `LIMITATIONS.md` — every known gap needs a pinned
  `known_limitation_` test; document the gap instead of faking it.
- `README.md` compat matrix / feature list — if you touched the
  snapshot schema, solver defaults, or the public API surface.
- `OPTIMIZATION_PLAN.md` — mark plan items done with measured numbers,
  not adjectives.

## Release process

Releases follow `RELEASE_CHECKLIST.md` top to bottom (hygiene gate,
version alignment, native artifacts, unsafe audit, fuzz + soak, docs,
publish). Version bumps touch `Cargo.toml` and
`Directory.Build.props` together — CI's `version-check` job enforces
they agree.
