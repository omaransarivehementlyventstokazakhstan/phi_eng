# PhiEng API stability policy

## What counts as public API

- **Rust**: all `pub` items of the `phieng`, `phieng-dynamics`,
  `phieng-collision`, `phieng-broadphase`, and `phieng-math` crates,
  including FFI symbol names and signatures in `phieng::ffi`.
- **C#**: all `public` types and members of the `PhiEng.*` assemblies,
  including P/Invoke signatures in `PhiEng.Native`.
- **Snapshots**: the JSON field set and the binary layout at a given
  `SNAPSHOT_SCHEMA_VERSION` / `Snapshot.SchemaVersion`. Loaders reject
  unknown versions instead of guessing; old files never silently parse
  under a new version.

## Version rules (SemVer)

- **MAJOR**: removed types/functions/symbols, changed signatures,
  changed snapshot binary layout without a version bump path, changed
  deterministic behavior the parity suite pins (bit-exact trajectories).
- **MINOR**: new shapes, joints, fluid features, FFI additions,
  snapshot schema bumps (with reject-old behavior), new test-ID blocks.
- **PATCH**: bug fixes that preserve all of the above, doc updates,
  internal refactors.

## Guarantees

- Determinism: same build + same inputs + same steps = bit-identical
  state, within one backend (Rust) and across backends where the
  equivalence tests (`F28`, `F36`, `P31–P34`) pin it.
- MSRV for Rust library code: **1.78** (verified by the `msrv` CI job
  building `--workspace --lib`). Examples and benches may use newer sugar.
- Snapshot loaders never guess: unknown versions, tags, or trailing
  bytes are errors, never silent reinterpretation.
- FFI never panics across the boundary on documented inputs: null
  pointers yield documented error codes; hostile payloads yield errors.

## Non-goals (explicitly unstable)

- Snapshot binary bytes across versions (only same-version round-trips
  are pinned).
- Timing/performance characteristics (only trajectory hashes and
  allocation counts in bench smoke).
- Internal module layout and private helpers.
