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
| `pile2d_iters1` | **0.555 ms/step** | 1500 bodies, 1 solver iteration |
| `pile2d_iters8` | **0.873 ms/step** | 1500 bodies, 8 solver iterations |
| `pile2d_census` | **1.263 ms/step** | the settled pile, 8 iterations, after the sweep |
| `pile2d_allocs` | **0.0 allocs/step** | counting global allocator |
| `islands_serial` | **0.0219 ms/step** | 40 bodies, 8 stacks |
| `islands_parallel` | **0.0209 ms/step** | serial and parallel tied at this size |
| `pile3d_iters8` | **0.126 ms/step** | 400 bodies |

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
| gravity | 0.0047 | 0.4 % |
| fluid | 0.0001 | 0.0 % |
| broadphase: rebuild | 0.1258 | 10.0 % |
| broadphase: pairs | 0.1706 | 13.5 % |
| narrowphase | 0.2539 | 20.1 % |
| cache (warm start) | 0.0819 | 6.5 % |
| solver | 0.5003 | 39.7 % |
| integrate | 0.0971 | 7.7 % |
| thermal | 0.0002 | 0.0 % |
| sleep | 0.0194 | 1.5 % |
| **TOTAL** | **1.2603** | 99.9 % of the wall clock |

**The solver is now the largest phase (38 %)** on the corrected
trajectory, followed by the narrowphase (19 %) and pair generation
(14 %). The grid-compaction work cut the broad-phase from 72 % of the
step to 26 %; the corrected pile settles into a denser configuration
with more contacts in flight, which is why the solver dominates here.

### The other scenes (`cargo run -p phieng --example bench_phases --release --features phase-timing`)

| Scene | ms/step | Broadphase share | Raw pairs | Cached pairs |
| ----- | ------- | ---------------- | --------- | ------------ |
| pile 3D (400 bodies) | 0.120 | 30 % | 400 | 400 |
| islands (40 bodies) | 0.032 | 20 % | 44 | 43 |
| **statics (420 bodies)** | **0.033** | **21 %** | **18** | **14** |
| fluid (96 SPH particles) | 0.069 | 1 % | — | — |

The static-heavy scene was the pathological case: the broad phase used
to generate **8 010** candidate pairs to end up with **14** live ones,
and that generation was 91 % of the step. The grid no longer generates
static-static pairs (they can never produce an impulse), so the scene
now emits **18** raw pairs. Two more changes took it further: a pair is
now generated once, in the cell that owns it, instead of once per cell
the two bodies share; and a cell that holds nothing but static bodies
is skipped whole, instead of having its quadratic number of
static-static candidates walked and dropped one by one. The scene's
pair phase went **0.999 → 0.003 ms/step**.

## What this rules in and out

- **In:** the solver (40 % of the pile step on the corrected
  trajectory) and the narrowphase (20 %).
- **In:** broad-phase pair generation (14 % here) — the AABB pair filter,
  the static-pair pruning, the owner rule and the static-cell skip
  already took the static scene's pair phase from 91 % to 21 % of its
  step (see below).
- **In, done:** the rigid-body memory layout. The solver is the dominant
  phase on the corrected trajectory, and the packed solver view (see
  below) now keeps the solver's working set in 48 KB instead of 264 KB.
  A `repr(C)` field reorder of `RigidBody2` itself was measured and did
  **not** pay — see `OPTIMIZATION_PLAN.md`.
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

## Measuring on this machine (read this before trusting a number)

The absolute timings on this box drift by ±20 % between runs, and two
builds of **identical** source differ by 6–10 % per phase because the
code layout moved. That offset is a fixed property of each binary, so
it never averages out: a cross-binary A/B cannot resolve anything below
about 15 %. Three confounds, and how to remove each:

| Confound | Symptom | Fix |
| -------- | ------- | --- |
| Code layout | every phase moves together by 5–10 % | put both variants in **one binary** and select at run time |
| Heap layout | every phase moves together by 5–10 % | the extra scratch changes where the body's array lands; keep the **same world and heap** for both variants |
| Work drift | whichever variant is measured *last* looks cheapest | the pile keeps settling while the measurement runs; **alternate the variants** every few steps, not every few hundred |

`crates/engine/examples/bench_solver_ab.rs` is the harness that applies
all three fixes at once: one world, one heap layout, the two solver
paths alternating every 20 steps. It is the reference for any change
that is too small to survive a cross-binary comparison.

```text
cargo run -p phieng --example bench_solver_ab --release --features phase-timing
```

## The packed solver view (shipped)
The solver's inner loop reads position, velocity, angular velocity,
inverse mass and inverse inertia for both ends of every constraint, on
every iteration. In `RigidBody2` those fields sit at offsets 16, 56, 96,
100 and 104 of a 176-byte body and the frozen flags at 168/169, so one
body spans **three cache lines** and the 1500-body pile costs the
solver a 264 KB working set.

The solver now runs on a packed copy of exactly those fields — one
32-byte record per body, cache-line aligned — so the working set drops
to **48 KB**. The copy holds the same values and the same arithmetic
runs in the same order, so the result is bit-identical; only velocity
and angular velocity change during the solve, and both are written back
before the solve returns.

Measured with the harness above (three independent repeats):

| Phase | Packed | Direct | Delta |
| ----- | ------ | ------ | ----- |
| solver | 202–215 ms | 216–240 ms | **−6 % to −10 %** |
| TOTAL (wall clock) | 497–510 ms | 519–535 ms | **−2 % to −5.5 %** |

Every other phase stayed within noise, as it must: the packed view only
touches the solver. All hashes are bit-identical (2D, 3D, islands serial
**and** parallel).

The advantage grows with the scene: at 4500 bodies the same comparison
gives the solver phase **−14.9 %** and the whole step **−7.0 %**, because
the direct view's 176-byte bodies no longer fit in cache while the
packed view's 32-byte records still do.

## The pair ownership rule (shipped)

A body is inserted into every grid cell its AABB spans, so two
neighbours that share two or four cells were found in each of them — the
same pair, several times over, before the sort and dedupe the output
contract requires. The grid now picks one *owner* cell per pair: the
lowest cell both bodies span, which both of them therefore occupy. No
pair is lost (every shared cell still holds both bodies), each pair is
generated exactly once, and the list that has to be sorted and
deduplicated is as long as the pair set instead of a multiple of it.

Measured with the same harness (the rule toggled between segments,
everything else held fixed):

| Phase | Owned | Unowned | Delta |
| ----- | ----- | ------- | ----- |
| broadphase: pairs | 61.7–68.4 ms | 76.7–83.4 ms | **−18 % to −20 %** |
| TOTAL (wall clock) | 476–525 ms | 498–541 ms | **−3 % to −7 %** |

Every other phase stayed inside the noise band, as it must: the rule
only changes how many times a pair is *examined*, not which pairs exist.
The pair set is identical, so all hashes are bit-identical.

## The static-cell skip (shipped)

Two statics never touch, so every candidate pair in a cell that holds
nothing but static bodies was walked and then dropped one by one. In a
level full of static geometry that walk is nearly all of the pair
phase: the static-heavy scene held **5 376** static-static candidates
to end up with **18** live pairs, and the pair phase was **49 %** of
that scene's step. The grid now drops such a cell without looking at
its pairs.

Measured with the same harness (the skip toggled between segments,
everything else held fixed):

| Scene | Measure | Skipping | Walking | Delta |
| ----- | ------- | -------- | ------- | ----- |
| statics (420 bodies) | broadphase: pairs | 1.07 ms | 9.16 ms | **−88 %** |
| statics (420 bodies) | TOTAL (wall clock) | 9.74 ms | 18.10 ms | **−46 %** |
| pile 2D (1500 bodies) | broadphase: pairs | 65.0 ms | 65.2 ms | −0.4 % |
| pile 2D (1500 bodies) | TOTAL (wall clock) | 494 ms | 499 ms | −1.0 % |

The pile is untouched because every one of its cells holds a dynamic
body — there is nothing to skip. The pair set is identical either way,
so all hashes are bit-identical.

## The overhaul, end to end

Interleaved A/B of the pre-overhaul build (commit `7c9b842`) against the
current one, nine alternating rounds on the same machine:

| Measure | Before | After | Delta |
| ------- | ------ | ----- | ----- |
| `pile2d_iters1` | 2.850 ms | 0.555 ms | **−80.5 %** |
| `pile2d_iters8` | 5.027 ms | 0.873 ms | **−82.6 %** |
| `pile2d_census` | 3.262 ms | 1.263 ms | **−61.3 %** |
| `islands_serial` | 0.0243 ms | 0.0219 ms | −9.9 % |
| `islands_parallel` | 0.0253 ms | 0.0209 ms | −17.4 % |
| `pile3d_iters8` | 0.228 ms | 0.126 ms | **−44.7 %** |

The pile-2D hash differs between the two builds **by design**: the
circle-vs-polygon corner-axis fix removed phantom contacts, which
changes the physics on purpose. The islands and 3D hashes are identical
across the whole overhaul.
