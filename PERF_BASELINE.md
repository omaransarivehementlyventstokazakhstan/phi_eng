# Performance baseline

The frozen reference for every performance change in this repository.

**Rule: a performance change must leave every pinned trajectory hash
bit-identical.** The hashes are the gate; the timings are the reason to
optimise. `tools/bench_ab.py` re-checks them on every measurement, so a
change that moves the physics fails loudly instead of silently.

## Pinned hashes (never move without a deliberate re-pin)

| Scene | Hash | Source |
| ----- | ---- | ------ |
| pile 2D (1500 bodies) | `0x4acb6694f3a9917` | `crates/engine/examples/bench_large.rs` |
| islands (8 stacks, 40 bodies) | `0xd4a3a56f185c13b2` | same (serial **and** parallel) |
| pile 3D (400 bodies) | `0x66ae94fb463de60` | same |
| determinism 2D | `0xb12fcb9a19506312` | `crates/engine/tests/determinism_hash.rs` |
| determinism 3D | `0xd057471127ba77ab` | same |

## Machine

AMD Ryzen 7 7445HS, 12 threads, 15.2 GB, Windows x64, release build
(`lto = "thin"`, `codegen-units = 1`, `opt-level = 3`).

## Whole-step baseline (`bench_large`, median of 5 windows)

| Measure | Baseline | Notes |
| ------- | -------- | ----- |
| `pile2d_iters1` | **0.620 ms/step** | 1500 bodies, 1 solver iteration |
| `pile2d_iters8` | **0.874 ms/step** | 1500 bodies, 8 solver iterations |
| `pile2d_census` | **1.377 ms/step** | the settled pile, 8 iterations, after the sweep |
| `pile2d_allocs` | **0.0 allocs/step** | counting global allocator |
| `islands_serial` | **0.0265 ms/step** | 40 bodies, 8 stacks |
| `islands_parallel` | **0.0260 ms/step** | serial and parallel tied at this size |
| `pile3d_iters8` | **0.140 ms/step** | 400 bodies |

These numbers describe the corrected trajectory (after the
circle-vs-polygon corner-axis fix below re-pinned the 2D hashes). The
pre-fix numbers are kept in git history; do not compare across the
re-pin except through `bench_ab.py` on a single trajectory.

The machine-readable form is `PERF_BASELINE.json`; regenerate both with
`python tools/bench_ab.py --reps 5 --save PERF_BASELINE.json`.

## Phase split (the `phase-timing` feature)

`cargo run -p phieng --example bench_large --release --features phase-timing`
prints where a step's time goes. The instrumentation is observational
only — the hashes are identical with the feature on and off.

### pile 2D — 1500 bodies, settled, 1634 cached pairs, 1844 cached contacts

| Phase | ms/step | Share |
| ----- | ------- | ----- |
| gravity | 0.0052 | 0.3 % |
| fluid | 0.0001 | 0.0 % |
| broadphase: rebuild | 0.1785 | 11.7 % |
| broadphase: pairs | 0.2162 | 14.2 % |
| narrowphase | 0.2900 | 19.1 % |
| cache (warm start) | 0.1169 | 7.7 % |
| solver | 0.5830 | 38.3 % |
| integrate | 0.1036 | 6.8 % |
| thermal | 0.0002 | 0.0 % |
| sleep | 0.0206 | 1.4 % |
| **TOTAL** | **1.5218** | 99.9 % of the wall clock |

**The solver is now the largest phase (38 %)** on the corrected
trajectory, followed by the narrowphase (19 %) and pair generation
(14 %). The grid-compaction work cut the broad-phase from 72 % of the
step to 26 %; the corrected pile settles into a denser configuration
with more contacts in flight, which is why the solver dominates here.

### The other scenes (`cargo run -p phieng --example bench_phases --release --features phase-timing`)

| Scene | ms/step | Broadphase share | Raw pairs | Cached pairs |
| ----- | ------- | ---------------- | --------- | ------------ |
| pile 3D (400 bodies) | 0.184 | 33 % | 400 | 400 |
| islands (40 bodies) | 0.036 | 22 % | 44 | 43 |
| **statics (420 bodies)** | **0.205** | **41 %** | **18** | **14** |
| fluid (96 SPH particles) | 0.114 | 1 % | — | — |

The static-heavy scene was the pathological case: the broad phase used
to generate **8 010** candidate pairs to end up with **14** live ones,
and that generation was 91 % of the step. The grid no longer generates
static-static pairs (they can never produce an impulse), so the scene
now emits **18** raw pairs and its pair phase is 18x faster.

## What this rules in and out

- **In:** the solver (40 % of the pile step on the corrected
  trajectory) and the narrowphase (19 %).
- **In:** broad-phase pair generation (15 % here) — the AABB pair filter
  and the static-pair pruning already took the static scene's pair phase
  from 91 % to 41 % of its step (see below).
- **In, conditionally:** the rigid-body memory layout. The solver is
  the dominant phase on the corrected trajectory, so the layout work
  from the plan is back on the table — pending its own measurement.
  (A `repr(C)` field reorder was measured and did **not** pay; see
  `OPTIMIZATION_PLAN.md`.)
- **Out:** SIMD (already measured slower than the scalar loop it would
  replace — see `OPTIMIZATION_PLAN.md` §2.2).

## The AABB pair filter (shipped)

The narrowphase fast path (`collect_overlapping_pairs` on every
backend, wired into both `step_inner` narrowphase loops; the CCD swept
test deliberately keeps the unfiltered sequence): pairs whose AABBs are
disjoint can never produce contacts, so they are dropped before the
push/sort/dedupe/narrowphase work. Measured on the corrected
trajectory, filter on vs off (median of 7, same trajectory, hash guard
green both ways):

| Measure | Filter on | Filter off | Delta |
| ------- | --------- | ---------- | ----- |
| `pile2d_iters8` | 0.918 | 1.513 | **−39 %** |
| `pile2d_census` | 1.417 | 1.616 | **−12 %** |
| `pile3d_iters8` | 0.139 | 0.238 | **−42 %** |

No hash moves in either direction: the filter drops only pairs the
(corrected) narrowphase rejects anyway.

## The static-pair pruning (shipped)

The grid learns which bodies are static and never generates a
static-static pair: two statics can never produce an impulse, so the
narrowphase discarded those pairs one step later. The static-heavy
scene's pair phase went from **0.999 to 0.054 ms/step (18x)** and its
raw pair count from 8 010 to 18. Interleaved A/B (9 alternating
rounds, hash guard green on both sides): islands serial −10.2 %,
islands parallel −11.7 %, pile 3D −8.6 %, pile 2D −2 % to −3 %.

## The rule for every change

1. One change, one commit, revertible on its own.
2. Measure before and after with `tools/bench_ab.py` (median of 5, the
   hash guard on). On a loaded machine, prefer the interleaved
   A/B (`tools/` script) — the absolute numbers drift by ±20 % between
   runs, so a before/after comparison minutes apart is not evidence.
3. Keep it only if it wins **at least 3 %** on the target scene **and**
   every hash is bit-identical.
4. Otherwise revert it and record the negative result in
   `OPTIMIZATION_PLAN.md` — a measured "no" is worth as much as a "yes".
