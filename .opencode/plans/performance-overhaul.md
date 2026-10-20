# Performance overhaul plan — phieng (Rust core)

## Non-negotiables
1. **Bit-identical only.** The five pinned hashes must not move:
   - pile 2D `0x4acb6694f3a9917` (`crates/engine/examples/bench_large.rs`)
   - islands `0xd4a3a56f185c13b2` (serial *and* parallel)
   - pile 3D `0x66ae94fb463de60`
   - determinism 2D `0xb12fcb9a19506312`, 3D `0xd057471127ba77ab`
     (`crates/engine/tests/determinism_hash.rs`)
   - the R01–R07 bit-pattern pins (`crates/engine/tests/determinism_trig.rs`)
   No re-pinning, no MAJOR version bump. (The 2D pins moved once,
   deliberately, when the circle-vs-polygon SAT gained its missing
   corner-axis test and stopped reporting phantom contacts.)
2. **Rust core first.** The managed C# mirror is a later pass.
3. **Layout work is in scope.**
4. MSRV 1.78 for library code; no public API removals;
   `tools/check_test_parity.py` must stay green.

## The measured baseline (release, Ryzen 7 7445HS, `bench_large`)

| Scene | iters=1 | iters=8 | pairs/step | contacts/step | allocs/step |
|---|---|---|---|---|---|
| pile 2D, 1500 bodies | 2.154 ms/step | 5.132 ms/step | 1578 | 1744 | 0.8 |
| pile 3D, 400 bodies | 0.235 ms/step | 0.259 ms/step | 400 | 601 | 0.0 |
| islands, 40 bodies | serial 0.0256 | parallel 0.0289 (0.88x, slower) | - | - | 0.0 |
| statics, 420 bodies | 1.124 ms/step | | | 12 contacts | |
| Barnes-Hut, 1500 bodies | | | | | 17.5 allocs/step |

The baseline does **not** give the phase split: the early exit means
"iters=8" is not eight iterations, so the apparent per-constraint cost is
an artefact. Phase 0 exists to answer that before anything is touched.

## Protocol (per change, no exceptions)
1. One change = one commit, isolated, revertible by `git revert`.
2. Record the baseline with the harness before; re-run after.
3. Keep only if it wins >= 3% on the target scene AND every hash is
   bit-identical. Otherwise revert and record the negative result in
   `OPTIMIZATION_PLAN.md` (the repo's existing "measured, not shipped"
   discipline).
4. After every Rust change: rebuild and **restage
   `src/PhiEng.Native/runtimes/win-x64/native/phieng.dll`** (the Windows CI
   leg uses the committed DLL; it does not rebuild it), then `dotnet test`.
5. Full local gate before each commit: `cargo fmt --all --check`,
   `clippy --workspace --all-targets -D warnings`,
   `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps`,
   `cargo test --workspace --lib --tests` (Debug),
   `cargo test --workspace --release`, `cargo +1.78 build --workspace --lib`,
   `dotnet test -c Debug` and `-c Release`,
   `python tools/check_test_parity.py`,
   `cargo run -p phieng --example bench_large --release` (hash gates), the
   five readme samples and the host smoke.

## Phase 0 — Instrumentation (no behaviour change)
- **0.1** `phase-timing` cargo feature (mirrors the `simd` feature
  precedent): per-phase wall times in `World2`/`World3`, off by default,
  zero cost and zero API surface when off.
- **0.2** `crates/engine/examples/bench_phases.rs` printing the phase
  split. Names must not match `fn <letter><2 digits>` (the parity checker
  scans `crates/**/*.rs`, examples included).
- **0.3** `tools/bench_ab.py`: permanent A/B harness (median + spread,
  baseline comparison, hash re-check).
- **0.4** `PERF_BASELINE.md`: the frozen baseline.

**Kill criterion:** if the phase split shows the solver is not dominant,
Phase 3 is dropped and the effort moves to the dominant phase.

## Phase 1 — Zero-risk waste removal (bit-identical by construction)
1.1 Hybrid broadphase rebuilt every step (`world.rs:962`, `world.rs:2022`).
1.2 CCD `SweptAABB` per-step `Vec` + duplicated broadphase rebuild
    (`world.rs:563-564`, `world.rs:1702`).
1.3 Double hash lookup in the contact cache (`world.rs:698`+`705`,
    `world.rs:1824`+`1830`).
1.4 Contact cache three full passes per step (`world.rs:759-794`).
1.5 `wake_body` is O(cache) per call, called per new contact
    (`world.rs:916-949`).
1.6 SPH per-substep `Vec` + `boundary_grid.clear()` dropping cell buffers
    (`fluid.rs:912-913`, `fluid.rs:747`; `fluid3.rs:1004,1034`).
1.7 SPH neighbour list rebuilt 3x per particle per substep
    (`fluid.rs:595,946,1004`).
1.8 Snapshot save copies the payload twice; load copies it again
    (`snapshot.rs:4694`, `4882`, `4239`, `4284`). Output bytes unchanged.
1.9 Thermal expansion clones the shape every step (`thermal.rs:401`,
    `thermal.rs:448`).
1.10 `validate_body_2d` runs per body per step even with `check_finite`
    off (`world.rs:894`).

## Phase 2 — Algorithmic wins (bit-identical, more invasive)
2.1 Per-pair world-space vertex rebuilds (`world.rs:1157-1178` etc.).
2.2 Radiation O(n^2) with per-pair surface properties (`radiation.rs:93-123`).
2.3 Nuclear fission scans all bodies twice per source (`nuclear.rs:429-451`).
2.4 Island builder re-sorts what the broadphase already sorted (`island.rs:121`).
2.5 Parallel path builds the island partition, then may discard it
    (`world.rs:385`, `world.rs:398`).
2.6 SPH particle-vs-body O(particles x bodies) (`fluid.rs:1057`).
2.7 Broadphase cleared and rebuilt every step (`world.rs:967-973`).

## Phase 3 — Memory layout (gated on Phase 0)
- **3.1 (recommended):** reorder `RigidBody2`/`RigidBody3` so the hot
  fields are contiguous, `#[repr(C)]` for a deterministic layout. Pure
  data movement; no public API change.
- **3.2 (escalation):** an interleaved hot-block side array (28 B/body),
  built before the solve and written back after, with exact write-back
  ordering around the joint and integration phases.

## Phase 4 — Parallelism
The parallel path is 0.88x (slower) because `std::thread::scope` spawns
and joins OS threads twice per step (`solver.rs:502`, `world.rs:466`).
- **4.1** Persistent worker pool (spin-wait barrier) instead of per-step
  spawn/join. Bit-identical: islands are disjoint and the max-impulse
  reduction is already global.
- **4.2** Re-tune `PARALLEL_MIN_CONSTRAINTS = 256` (`world.rs:40`) with
  real measurements.
- **4.3** A 3D parallel solver — deferred (highest risk).

## Phase 5 — Documentation
`OPTIMIZATION_PLAN.md` (measured numbers, including negatives),
`LIMITATIONS.md` (allocation baseline), `CHANGELOG.md`, `README.md`
performance matrix, `absolute_stress_tests.md` if a reading moves.

## Out of scope
The managed C# mirror; full SoA; GPU; fixed-point; SIMD (already measured
slower); re-pinning any hash.

## Risks
| Risk | Mitigation |
|---|---|
| A change moves a hash | The hash gates are the safety net (`bench_large` panics, `determinism_hash` asserts, parallel == serial is asserted). Every commit is verified before it lands. |
| A change is a net loss | A/B harness + per-change revert + the "measured, not shipped" log. |
| The native DLL goes stale | Restage + `dotnet test` after every Rust change. |
| A new test trips the parity checker | No `fn <letter><2 digits>` names anywhere in `crates/`. |
| MSRV 1.78 breaks | The `msrv` job builds `--workspace --lib`; no new language features in library code. |
| Scope creep into the C# mirror | A separate plan after this one is green. |

## Success criteria
Phase 0 lands with a phase split and a frozen baseline. Phases 1-2 land
as a series of individually-measured, individually-revertible commits,
each hash-clean. Phase 3 lands only if the data justifies it. The 2D
pile's `iters=8` number drops by a measured margin with all five hashes
unchanged and the full CI gate green on Windows and Linux.
