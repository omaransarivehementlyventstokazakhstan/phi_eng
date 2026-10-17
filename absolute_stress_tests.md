# Absolute Stress Tests — Readings

Brutal, in-scope soak suite: `crates/engine/tests/brutal.rs` (13 tests).
Each test asserts survival (finite state, no escape) and prints one
`BRUTAL ...` readings line. Run command (release build required —
debug is 10–50x slower):

```text
cargo test --release -p phieng --test brutal -- --nocapture --test-threads=1
```

## Readings (release, Windows x64, 2026-10-15)

| Test | Bodies | Steps | Wall ms | ms/step | Peak speed | Reading |
|---|---|---|---|---|---|---|
| deep_overlap_eject_2d | 2 | 240 | 0.2 | 0.0006 | 2.001 | 90% overlap separates gently to final_x=7.960, no NaN (was 10.740 / 42.217 pre-cap) |
| determinism_double_run_2d | 11 | 600 | 2.8 | 0.0047 | 11.215 | run1 == run2 bit-exact, hash `0xe0f4f2748072d545` |
| friction_shove_2d | 13 | 600 | 7.5 | 0.0125 | 73.499 | 8 m/s shove ejects debris; min_y=-290.59, **tunneled=true** (unchanged by cap: impact-driven, not bias-driven) |
| hostile_soak_2d | 2 | 120 | 0.1 | 0.0007 | 0.000 | 6/6 hostile ops rejected cleanly |
| joint_gauntlet_2d | 7 | 1200 | 1.9 | 0.0016 | 16.911 | 6 distance links, max rest-length err 0.0063 (bit-identical pre/post cap) |
| mass_ratio_soak_2d | 3 | 20000 | 39.7 | 0.0020 | 0.296 | 10,000:1 ratio, light body rests at y=0.994 (bit-identical pre/post cap) |
| mega_tower_collapse_2d | 37 | 1800 | 81.7 | 0.0454 | 240.074 | 36-box collapse still tunnels (min_y=-3522.97) via impact speeds, not bias; debris escapes to max_abs_x=282.75 |
| mesh_stack_3d | 7 | 900 | 7.7 | 0.0086 | 9.901 | all 6 levels sank to y≈-0.02 (floor pile, no floor tunneling) |
| mesh_tumbler_3d | 19 | 900 | 20.4 | 0.0226 | 11.701 | min_y=-0.12, 14 pairs / 14 contacts, all contained |
| needle_impact_2d | 2 | 300 | 0.4 | 0.0013 | 60.000 | 60 m/s needle rebounded to final_x=-18.26, **tunneled=false** |
| parallel_parity_scale_2d | 21 | 300 | — | serial 0.0192 / parallel 0.0569 | — | serial == parallel bit-exact, hash `0x7fa0fadeb1ea3f37` (bit-identical pre/post cap: shallow contacts untouched) |
| restitution_storm_2d | 39 | 1200 | 7.0 | 0.0058 | 23.549 | 36 restitution-1.0 balls contained, energy bounded (bit-identical pre/post cap) |
| snapshot_midchaos_3d | 11 | 240 | — | — | — | json 19821 B vs bin 2890 B; json == bin == direct continuation `0x33ef9bb330a33282` |

Result: **13 passed, 0 failed** (total suite wall time ≈ 0.21 s in release).
(Tested on a Ryzen 7 7745Hs with 16 gigs of ram).
## Findings worth knowing

1. **Bias cap fixed the explosion class.** Capping Baumgarte bias at
   2.0 m/s (`Solver::MAX_BIAS_VELOCITY`, mirrored in C#) cut deep-overlap
   ejection from 10.7 to 2.0 m/s with gentle separation (final_x 42 →
   8.0). Shallow contacts (< ~17 cm at 60 Hz) are bit-identical: joint
   gauntlet, mass-ratio soak, storm, needle, and parallel-parity hashes
   are unchanged pre/post cap.
2. **Tower/shove tunneling persists via impact momentum, not bias.**
   The shove's peak (73.5) and min_y (−290.59) are byte-identical
   pre/post cap: that escape is driven by the 8 m/s impact itself
   (restitution path, where Baumgarte is skipped by design), plus a thin
   1 m floor and no CCD. Fixing that residue needs CCD or impact-speed
   management — a bigger, separate change. Both tests now bound the
   envelope (finite, in-universe) and report tunneling honestly.
3. **Mesh-vs-mesh towers do not hold.** The GJK catch-all emits a
   single support-midpoint contact per pair, so six stacked convex-mesh
   boxes sink through each other into a floor pile (all levels end at
   y≈−0.02, none pass the floor). Same weakness class as the old
   box-vs-capsule midpoint bug. Mesh-vs-analytic contacts (floor, walls)
   hold fine — the tumbler stays contained. Future work: mesh contact
   manifold refinement.
4. **Joints are rock solid.** Six-link chain holds rest length to
   6.3 mm after 1200 swinging steps; 10,000:1 mass ratio shows 0.3 m/s
   peak jitter over 20,000 steps.
5. **Determinism + snapshots hold under chaos.** Double runs match
   bit-exactly; JSON and binary mid-chaos resumes match the
   uninterrupted run bit-exactly (binary is ~6.9× smaller than JSON).
6. **Parallel costs more than it saves at this scale.** 21-body scene:
   parallel 0.057 ms/step vs serial 0.019 ms/step — thread overhead
   dominates; results are bit-identical.
7. **Restitution storms stay bounded.** 36 fully bouncy balls peak at
   23.5 m/s with no escape and no energy blowup.

## Orbital mechanics suite (2026-10-16)

Ultra-complex celestial-mechanics benchmarks in `crates/engine/tests/orbital.rs`
(7 scenarios × 2 integrators = 14 readings). Canonical units per test
(documented in-file). Cross-language parity: managed/native agreement is
asserted within tolerances (bit-equality never claimed).

Run command (release required):

```text
cargo test --release -p phieng --test orbital -- --nocapture --test-threads=1
```

| Test | Bodies | Steps | Wall ms | ms/step | Peak speed | Reading |
|---|---|---|---|---|---|---|
| burrau_euler | 3 | 3000 | 0.6 | 0.0002 | 4.978 | e_ratio=-1.139 min_sep=0.027 max_r=27.9 |
| burrau_verlet | 3 | 3000 | 0.7 | 0.0002 | 4.723 | e_ratio=-1.057 min_sep=0.060 max_r=4.6 |
| figure8_euler | 3 | 1898 | 0.8 | 0.0004 | 1.285 | return=0.048 e_drift=2.4e-4 min_sep=0.671 |
| figure8_verlet | 3 | 1898 | 0.6 | 0.0003 | 1.272 | return=0.0004 e_drift=6.3e-6 min_sep=0.690 |
| geo_ring_euler | 12 | 9537 | 13.2 | 0.0014 | 1.582 | ang_err=0.01° r_err=0.0000 |
| geo_ring_verlet | 12 | 9537 | 12.7 | 0.0013 | 1.581 | ang_err=0.01° r_err=0.0000 |
| hohmann_euler | 1 | 1938 | 0.5 | 0.0002 | 3.652 | r_apo=20.001 hold_err=0.0010 |
| hohmann_verlet | 1 | 1938 | 0.5 | 0.0002 | 3.651 | r_apo=20.000 hold_err=0.0001 |
| leo_constellation_euler | 66 | 2000 | 7.2 | 0.0036 | 10.023 | r_err=0.0023 |
| leo_constellation_verlet | 66 | 2000 | 6.9 | 0.0034 | 10.000 | r_err=0.0000 |
| molniya_euler | 1 | 3372 | 0.4 | 0.0001 | 5.323 | closure=0.077 r_apo=34.01 plane_err=0.00° |
| molniya_verlet | 1 | 3372 | 0.4 | 0.0001 | 5.323 | closure=0.027 r_apo=34.00 plane_err=0.00° |
| plummer_exact | 1000 | 200 | 647.5 | 3.2377 | 0.361 | e_drift=3.9e-4 Q=1.016 |
| plummer_bh | 1000 | 200 | 2956.4 | 14.7822 | 0.361 | e_drift=7.1e-6 |

Result: **7 passed, 0 failed** (total suite wall time ≈ 4.3 s in release).
(Tested on a Ryzen 7 7745Hs with 16 gigs of ram.)

### Orbital findings

1. **Verlet dominates Euler on every long-orbit benchmark.** Figure-8
   energy drift: 6.3e-6 (Verlet) vs 2.4e-4 (Euler) — 38× better.
   Hohmann hold error: 0.0001 vs 0.0010 — 10× better. GEO ring phase
   error: 0.01° both (dominated by output quantization, not
   integrator). LEO constellation radius error: 0.0000 vs 0.0023.
2. **Figure-8 choreography is reproducible.** After 5 periods the
   configuration returns to within 0.0004 (Verlet) of the initial
   state — the periodic solution is stable under this integrator.
3. **Hohmann transfer is accurate to 0.1%.** Arrival radius 20.000
   (Verlet) vs analytic 20.00; circularization hold error 0.0001.
4. **Molniya orbit closes.** One full period returns to within 0.027
   (Verlet) of start; apogee 34.00 vs analytic 34.0; orbital plane
   preserved to 0.00°.
5. **Plummer cluster is a proper virial equilibrium (Q = 1.016).** The
   sampled initial conditions satisfy the virial theorem
   (2K/|U| ≈ 1) — the correct Plummer velocity distribution gives the
   equilibrium kinetic energy — and over 200 steps the energy drift is
   tiny (exact 3.9e-4, BH 7.1e-6), so the integrator tracks the
   equilibrium well. BH remains ~4.6× slower than exact at N=1000
   (tree-build overhead dominates at this scale).
