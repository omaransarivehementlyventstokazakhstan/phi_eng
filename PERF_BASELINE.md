# Performance baseline

The frozen reference for every performance change in this repository.

**Rule: a performance change must leave every pinned trajectory hash
bit-identical.** The hashes are the gate; the timings are the reason to
optimise. `tools/bench_ab.py` re-checks them on every measurement, so a
change that moves the physics fails loudly instead of silently.

## Pinned hashes (never move without a deliberate re-pin)

| Scene | Hash | Source |
| ----- | ---- | ------ |
| pile 2D (1500 bodies) | `0xc509c18cd50043b4` | `crates/engine/examples/bench_large.rs` |
| islands (8 stacks, 40 bodies) | `0xd4a3a56f185c13b2` | same (serial **and** parallel) |
| pile 3D (400 bodies) | `0x66ae94fb463de60` | same |
| determinism 2D | `0xc3f59dbc6e87256a` | `crates/engine/tests/determinism_hash.rs` |
| determinism 3D | `0xd057471127ba77ab` | same |

## Machine

AMD Ryzen 7 7445HS, 12 threads, 15.2 GB, Windows x64, release build
(`lto = "thin"`, `codegen-units = 1`, `opt-level = 3`).

## Whole-step baseline (`bench_large`, median of 5 windows)

| Measure | Baseline | Notes |
| ------- | -------- | ----- |
| `pile2d_iters1` | **2.307 ms/step** | 1500 bodies, 1 solver iteration |
| `pile2d_iters8` | **5.108 ms/step** | 1500 bodies, 8 solver iterations |
| `pile2d_census` | **3.183 ms/step** | the settled pile, 8 iterations, after the sweep |
| `pile2d_allocs` | **0.8 allocs/step** | counting global allocator |
| `islands_serial` | **0.0236 ms/step** | 40 bodies, 8 stacks |
| `islands_parallel` | **0.0310 ms/step** | **the parallel path is 1.31x SLOWER than serial** |
| `pile3d_iters8` | **0.232 ms/step** | 400 bodies |

The machine-readable form is `PERF_BASELINE.json`; regenerate both with
`python tools/bench_ab.py --reps 5 --save PERF_BASELINE.json`.

## Phase split (the `phase-timing` feature)

`cargo run -p phieng --example bench_large --release --features phase-timing`
prints where a step's time goes. The instrumentation is observational
only — the hashes are identical with the feature on and off.

### pile 2D — 1500 bodies, settled, 1578 cached pairs, 1744 cached contacts

| Phase | ms/step | Share |
| ----- | ------- | ----- |
| gravity | 0.0036 | 0.1 % |
| fluid | 0.0001 | 0.0 % |
| **broadphase: rebuild** | **0.9880** | **32.4 %** |
| **broadphase: pairs** | **1.2175** | **39.9 %** |
| narrowphase | 0.3408 | 11.2 % |
| cache (warm start) | 0.0894 | 2.9 % |
| solver | 0.3130 | 10.3 % |
| integrate | 0.0866 | 2.8 % |
| thermal | 0.0004 | 0.0 % |
| sleep | 0.0123 | 0.4 % |
| **TOTAL** | **3.0516** | 93.5 % of the wall clock |

**The broad-phase is 72 % of the step.** The solver is 10 %.

### The other scenes (`cargo run -p phieng --example bench_phases --release --features phase-timing`)

| Scene | ms/step | Broadphase share | Raw pairs | Cached pairs |
| ----- | ------- | ---------------- | --------- | ------------ |
| pile 3D (400 bodies) | 0.189 | 43 % | 1 968 | 400 |
| islands (40 bodies) | 0.024 | 29 % | 51 | 43 |
| **statics (420 bodies)** | **0.549** | **96 %** | **10 203** | **25** |
| fluid (96 SPH particles) | 0.046 | 1 % | — | — |

The static-heavy scene is the pathological case: the broad phase
generates **10 203** candidate pairs to end up with **25** live ones, and
that generation is 90 % of the step.

## What this rules in and out

- **In:** the broad-phase (rebuild and pair generation). It is the
  overwhelming majority of the time.
- **In:** the narrowphase (11 %) and the solver (10 %).
- **Out, for now:** the rigid-body memory layout. The solver is 10 % of
  the step, not the bottleneck the earlier estimate suggested; the
  layout work only pays off if the solver becomes the dominant phase.
- **Out:** SIMD (already measured slower than the scalar loop it would
  replace — see `OPTIMIZATION_PLAN.md` §2.2).

## The rule for every change

1. One change, one commit, revertible on its own.
2. Measure before and after with `tools/bench_ab.py` (median of 5, the
   hash guard on).
3. Keep it only if it wins **at least 3 %** on the target scene **and**
   every hash is bit-identical.
4. Otherwise revert it and record the negative result in
   `OPTIMIZATION_PLAN.md` — a measured "no" is worth as much as a "yes".
