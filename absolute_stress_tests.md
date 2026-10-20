# Absolute Stress Tests — Readings

Brutal, in-scope soak suite: `crates/engine/tests/brutal.rs` (13 tests).
Each test asserts survival (finite state, no escape) and prints one
`BRUTAL ...` readings line. Run command (release build required —
debug is 10–50x slower):

```text
cargo test --release -p phieng --test brutal -- --nocapture --test-threads=1
```

This document also holds the four later suites, each in its own
section, each with its own run command and readings:

| Suite | File | Tests | Readings prefix |
| ----- | ---- | ----- | ---------------- |
| Thermal & radiation | `crates/engine/tests/thermal_stress.rs` | 9 | `THERMAL ...` |
| Nuclear | `crates/engine/tests/nuclear_stress.rs` | 7 | `NUCLEAR ...` |
| Snapshot fuzzing | `crates/engine/tests/snapshot_fuzz.rs` | 6 | `FUZZ ...` |
| Tier performance | `crates/engine/tests/tier_sweep.rs` | 3 | `TIER ...` |

(The wall times of the four later sections were measured on a Ryzen 7
7445HS / 15.2 GB; the earlier sections were measured on a 7745Hs / 16 GB.
The *physics* readings are exact and machine-independent; the wall times
are not comparable across the two machines.)

## Readings (release, Windows x64, 2026-10-15)

| Test                      | Bodies | Steps | Wall ms | ms/step                         | Peak speed | Reading                                                                                                          |
| ------------------------- | ------ | ----- | ------- | ------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| deep_overlap_eject_2d     | 2      | 240   | 0.2     | 0.0006                          | 2.001      | 90% overlap separates gently to final_x=7.960, no NaN (was 10.740 / 42.217 pre-cap)                              |
| determinism_double_run_2d | 11     | 600   | 2.8     | 0.0047                          | 11.215     | run1 == run2 bit-exact, hash `0xe0f4f2748072d545`                                                                |
| friction_shove_2d         | 13     | 600   | 7.5     | 0.0125                          | 73.499     | 8 m/s shove ejects debris; min_y=-290.59, **tunneled=true** (unchanged by cap: impact-driven, not bias-driven)   |
| hostile_soak_2d           | 2      | 120   | 0.1     | 0.0007                          | 0.000      | 6/6 hostile ops rejected cleanly                                                                                 |
| joint_gauntlet_2d         | 7      | 1200  | 1.9     | 0.0016                          | 16.911     | 6 distance links, max rest-length err 0.0063 (bit-identical pre/post cap)                                        |
| mass_ratio_soak_2d        | 3      | 20000 | 39.7    | 0.0020                          | 0.296      | 10,000:1 ratio, light body rests at y=0.994 (bit-identical pre/post cap)                                         |
| mega_tower_collapse_2d    | 37     | 1800  | 81.7    | 0.0454                          | 240.074    | 36-box collapse still tunnels (min_y=-3522.97) via impact speeds, not bias; debris escapes to max_abs_x=282.75   |
| mesh_stack_3d             | 7      | 900   | 7.7     | 0.0086                          | 9.901      | all 6 levels sank to y≈-0.02 (floor pile, no floor tunneling)                                                    |
| mesh_tumbler_3d           | 19     | 900   | 20.4    | 0.0226                          | 11.701     | min_y=-0.12, 14 pairs / 14 contacts, all contained                                                               |
| needle_impact_2d          | 2      | 300   | 0.4     | 0.0013                          | 60.000     | 60 m/s needle rebounded to final_x=-18.26, **tunneled=false**                                                    |
| parallel_parity_scale_2d  | 21     | 300   | —       | serial 0.0192 / parallel 0.0569 | —          | serial == parallel bit-exact, hash `0x7fa0fadeb1ea3f37` (bit-identical pre/post cap: shallow contacts untouched) |
| restitution_storm_2d      | 39     | 1200  | 7.0     | 0.0058                          | 23.549     | 36 restitution-1.0 balls contained, energy bounded (bit-identical pre/post cap)                                  |
| snapshot_midchaos_3d      | 11     | 240   | —       | —                               | —          | json 19821 B vs bin 2890 B; json == bin == direct continuation `0x33ef9bb330a33282`                              |

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
   1 m floor and CCD off by default. Enabling CCD
   (`CcdConfig.enabled`, C01–C04) arrests this class of escape by
   substepping fast bodies; the brutal readings above keep CCD off to
   pin the discrete-stepping baseline. Both tests bound the
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

| Test                     | Bodies | Steps | Wall ms | ms/step | Peak speed | Reading                                    |
| ------------------------ | ------ | ----- | ------- | ------- | ---------- | ------------------------------------------ |
| burrau_euler             | 3      | 3000  | 0.6     | 0.0002  | 4.978      | e_ratio=-1.139 min_sep=0.027 max_r=27.9    |
| burrau_verlet            | 3      | 3000  | 0.7     | 0.0002  | 4.723      | e_ratio=-1.057 min_sep=0.060 max_r=4.6     |
| figure8_euler            | 3      | 1898  | 0.8     | 0.0004  | 1.285      | return=0.048 e_drift=2.4e-4 min_sep=0.671  |
| figure8_verlet           | 3      | 1898  | 0.6     | 0.0003  | 1.272      | return=0.0004 e_drift=6.3e-6 min_sep=0.690 |
| geo_ring_euler           | 12     | 9537  | 13.2    | 0.0014  | 1.582      | ang_err=0.01° r_err=0.0000                 |
| geo_ring_verlet          | 12     | 9537  | 12.7    | 0.0013  | 1.581      | ang_err=0.01° r_err=0.0000                 |
| hohmann_euler            | 1      | 1938  | 0.5     | 0.0002  | 3.652      | r_apo=20.001 hold_err=0.0010               |
| hohmann_verlet           | 1      | 1938  | 0.5     | 0.0002  | 3.651      | r_apo=20.000 hold_err=0.0001               |
| leo_constellation_euler  | 66     | 2000  | 7.2     | 0.0036  | 10.023     | r_err=0.0023                               |
| leo_constellation_verlet | 66     | 2000  | 6.9     | 0.0034  | 10.000     | r_err=0.0000                               |
| molniya_euler            | 1      | 3372  | 0.4     | 0.0001  | 5.323      | closure=0.077 r_apo=34.01 plane_err=0.00°  |
| molniya_verlet           | 1      | 3372  | 0.4     | 0.0001  | 5.323      | closure=0.027 r_apo=34.00 plane_err=0.00°  |
| plummer_exact            | 1000   | 200   | 647.5   | 3.2377  | 0.361      | e_drift=3.9e-4 Q=1.016                     |
| plummer_bh               | 1000   | 200   | 2956.4  | 14.7822 | 0.361      | e_drift=7.1e-6                             |

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

## Memory / allocation census (2026-10-17)

`bench_large` (release, counting allocator), dynamic pile 1500 bodies
(2D) / 400 (3D), medians after settling:

| Scene | allocs/step | bytes/step |
|---|---|---|
| pile 2D (before pools) | 466.8 | ~15 000 |
| **pile 2D (after pools)** | **0.8** | **54** |
| pile 3D (before) | 14.0 | — |
| **pile 3D (after)** | **0.0** | **0** |
| resting stack | 0.00 | 0 |
| empty world | 0.00 | 0 |

The broadphase grid (`SpatialHash2`) and contact cache
(`World2/3::contact_cache`) recycle vacated per-cell / per-key `Vec`
buffers through internal pools; vacated entries are still dropped every
frame (so `contact_cache.len()` stays exact) and compaction every 256
steps bounds the cell map for roaming worlds. All determinism hashes
(pile 2D `0xc509c18cd50043b4`, islands `0xd4a3a56f185c13b2`, pile3d
`0x66ae94fb463de60`) are **unchanged** — the optimization is
memory-only. (Those are the current pins; the rotation integration was
later made platform-independent, which re-pinned the 2D hashes.)

## SHM & SPH benchmark suite (2026-10-17)

Canonical, well-known problems, checked against closed-form solutions
where the method can hit them and against qualitative trends where it
cannot (gaps pinned in `LIMITATIONS.md`). 20 tests; all green
(**debug** wall clock, Windows x64, single-threaded: **~0.1 s SHM +
~32.9 s SPH ≈ 33.0 s total** — dominated by the 3D hydrostatic column).

### Simple harmonic motion (damped linear oscillator)
The spring field is `m x'' + c x' + k x = 0`, so the analytic solution
is exact:

| Test                               | Check                                                                      | Wall (debug) | Result |
| ---------------------------------- | -------------------------------------------------------------------------- | ------------ | ------ |
| `shm_damped_regimes_analytic`      | under/critical/over vs closed form (<0.03)                                 | 0.01 s       | pass   |
| `shm_energy_decay_rate`            | envelope slope ≈ `-2 ζ ω₀` (log-fit)                                       | <0.01 s      | pass   |
| `shm_period_mass_scaling`          | `T = 2π√(m/k)` across masses                                               | 0.01 s       | pass   |
| `shm_vertical_gravity_equilibrium` | rest point shifts `mg/k`, period fixed                                     | 0.01 s       | pass   |
| `shm_driven_resonance_peak`        | amplitude peaks at `ω₀√(1-2ζ²)`; matches `F₀/(m√((ω₀²-ω_d²)²+(2ζω₀ω_d)²))` | 0.08 s       | pass   |
| `shm_undamped_energy_bounded`      | symplectic Euler: no secular energy drift (band < 5%)                      | <0.01 s      | pass   |
| `shm_critical_settles_fastest`     | ζ=1 reaches a rest band faster than under/over                             | 0.01 s       | pass   |
| `shm_stiffness_scaling`            | `ω ∝ √k` (`period(k=160) = period(k=40)/2`)                                | <0.01 s      | pass   |
| `shm3_damped_regimes_analytic`     | 3D mirror (bit-identical to 2D)                                            | <0.01 s      | pass   |

### SPH fluid

| Test                           | Canonical problem               | Assertion                                           | Wall (debug) | Result |
| ------------------------------ | ------------------------------- | --------------------------------------------------- | ------------ | ------ |
| `sph_hydrostatic_pressure`     | still column, `p ≈ ρgh`         | order-of-magnitude + monotone depth                 | 2.55 s       | pass   |
| `sph_hydrostatic_tait`         | column, Tait EOS (γ=7)          | tighter `p ≈ ρgh` (0.5–2.0×)                        | 2.83 s       | pass   |
| `sph_dam_break_surge_front`    | Martin & Moyce                  | advances + contained (√t law is a documented gap)   | 0.14 s       | pass   |
| `sph_dam_break_obstacle`       | surge over a bump               | flow rides over, no obstacle penetration            | 0.50 s       | pass   |
| `sph_droplet_cohesion`         | surface tension holds a droplet | bounded radius                                      | 0.41 s       | pass   |
| `sph_free_blob_energy_bounded` | no gravity/visc/surface         | energy conserved (bounded)                          | 0.28 s       | pass   |
| `sph_sloshing_settles`         | kicked tank                     | slosh damps monotonically (peak speed decays)       | 0.44 s       | pass   |
| `sph_poiseuille_profile`       | channel + wall friction         | center fastest (curved, no-slip)                    | 1.04 s       | pass   |
| `sph_wall_friction_slows_flow` | channel, free vs no-slip        | no-slip removes tangential momentum                 | 2.37 s       | pass   |
| `sph_two_phase_heavy_sinks`   | heavy droplet in light fluid    | heavy phase sinks under gravity                    | 0.98 s       | pass   |
| `sph3_hydrostatic_pressure`    | 3D column                       | order-of-magnitude `ρgh`                            | 21.37 s      | pass   |

Findings worth knowing:
- **The 3D hydrostatic column dominates the suite runtime** (~21 of
  ~32 s debug: 512 particles × 900 steps × 3D neighbor search in an
  unoptimized build); every other test is sub-second. Release is ~10×
  faster.
- **SHM is analytically exact** (to the integrator's O(dt) accuracy);
  the 2D and 3D springs produce bit-identical trajectories.
- **Hydrostatic support works** but the penalty floor compresses the
  bottom decile, so `p_bottom` overshoots `ρgh` (~2.3× in 2D) — a
  boundary-model artifact, not a solver bug.
- **No-slip walls are opt-in** (`wall_friction ∈ [0,1]`, default 0 =
  free-slip): with it off the channel is flat, with it on the profile
  curves toward the center and tangential momentum is removed. It is a
  penalty velocity match, not a full ghost-viscosity no-slip, so the
  profile is qualitatively Chapman.
- **Martin & Moyce √t front law is not reproduced** by penalty-boundary
  weakly-compressible SPH; the front advances but the quantitative law
  is a gap.
- **Penalty boundaries tunnel** at high speed (no CCD): a violent
  dam-break ejected a fast particle through a thin wall until the test
  used thick walls.

## CCD parity block (2026-10-18)

Conservative advancement via adaptive substepping
(`CcdConfig.enabled`, `max_movement` 0.05, `max_substeps` 64), pinned
by `crates/engine/tests/ccd.rs` (C01–C04, mirrored in
`tests/PhiEng.Tests/CcdTests.cs`):

| Test | Scene | Result |
| ---- | ----- | ------ |
| `c01_thin_wall_2d` | ball r=0.25 at 60 m/s vs 0.1-thick wall, 120 steps | tunnels to y &lt; −1 off; rests y ∈ (0.2, 0.6) on |
| `c02_thin_wall_3d` | sphere r=0.25 at 120 m/s vs 0.1-thick slab, 120 steps | tunnels to y &lt; −1 off; rests y ∈ (0.2, 0.6) on |
| `c03_default_off_bit_identical` | settling stack, CCD enabled but capped at 1 substep, 300 steps | bitwise identical to CCD-off; default JSON has no `ccd` key |
| `c04_ccd_snapshot_roundtrip` | CCD wall scene through JSON + binary (schema v6) | config preserved; fast ball still caught after load |

CCD cost is proportional to fast-body speed (substeps × the normal
pipeline); calm scenes take exactly one substep, so resting piles pay
nothing. Off (default) is bit-identical to legacy stepping.

## Thermal & radiation stress suite (2026-10-20)

The thermal phase (conduction, convection, radiation, expansion) and the
schema-v11 radiation-model expansion had **no** absolute-stress coverage.
`crates/engine/tests/thermal_stress.rs` (9 tests) checks the model against
a closed-form solution wherever one exists, and against boundedness,
antisymmetry and conservation everywhere else. Each test prints one
`THERMAL ...` readings line.

Run command (release required):

```text
cargo test --release -p phieng --test thermal_stress -- --nocapture --test-threads=1
```

| Test | Bodies | Steps | Wall ms | ms/step | Reading |
| ---- | ------ | ----- | ------- | ------- | ------- |
| `thermal_conduction_exchange_is_exponential` | 3 | 600 | 0.9 | 0.0014 | gap 99.07 → 0.3760 m; log-slope −0.5584 → τ = 1.791 s; pair heat 545 760 000.0 → 545 758 272.9 |
| `thermal_convective_cooling_matches_analytic` | 1 | 600 | 0.1 | 0.0002 | τ = 22.500 s; analytic 492.354 K vs measured 492.323 K (err 0.0311 K) |
| `thermal_expansion_matches_formula` | 1 | 200 | 0.1 | 0.0003 | radius 1.706850 = closed form 1.706850; per-step compounding would have reached 2.748e46 |
| `thermal_extreme_values_stay_finite` | 2 | 600 | 42.1 | 0.0702 | hot and cold both clamp at 4999.99 K; expansion 0.0100; no NaN, no runaway |
| `thermal_radiation_is_stiff_but_bounded` | 1 | 600 | 0.1 | 0.0002 | final 4000.08 K; lost 999.92 K; rate cap 1000 K/step; peak 5000.0 K |
| `thermal_radiative_exchange_is_antisymmetric` | 2 | 1 | 0.0 | 0.0000 | cold gain 0.022064 / hot loss 2.462463 — bit-identical when the bodies are swapped |
| `thermal_shock_converges_without_explosion` | 3 | 1200 | 1.1 | 0.0009 | heat 549900000.0 → 549898173.7; 5000 K → 302.41 K, 300 K → 302.35 K (gap 0.0663 K); peak speed 0.0000 |
| `thermal_sky_ground_split_changes_equilibrium` | 1 | 600 | 0.0 | 0.0000 | uniform ambient 300.0000 K; sky 100 / ground 400 → 309.5192 K; sky 100 / ground 100 → 283.8012 K |
| `thermal_symmetric_view_factor_matches_legacy` | 2 | 1 | 0.0 | 0.0000 | equal bodies: legacy 0.022064 = symmetric 0.022064; unequal: legacy 0.022064 vs symmetric 0.033112 |

Result: **9 passed, 0 failed** (total suite wall time ≈ 0.04 s in release).
(Tested on a Ryzen 7 7445HS with 15.2 gigs of ram.)

### Thermal findings

1. **Conduction is a clean exponential and it conserves the pair's heat.**
   The temperature difference decays with a fitted log-slope of −0.5584
   (τ = 1.791 s) while the pair's total heat moves by 1 727.1 out of
   545.76 M — the only leak is the cold, heavy reservoir's own exchange
   with the floor. The exponential is asserted as a straight line in log
   space, not as a curve fit.
2. **Convection matches the analytic exponential to 0.0311 K** (492.354 K
   analytic vs 492.323 K measured after 600 steps) — a 6.3e-5 relative
   error, at the limit of the f32 state.
3. **Thermal expansion is linear in T, not compounding.** The radius lands
   on the closed form 1.706850; compounding the same factor every step
   would have reached 2.748e46 — infinity — within 200 steps.
4. **Extreme temperatures are bounded by the clamp, not by luck.** With
   both bodies driven into the 5000 K clamp the world stays finite for
   600 steps, both bodies settle at 4999.99 K, and the expansion stays at
   0.0100. This is the slowest test in the suite (42.1 ms) because the
   clamp is re-evaluated on every step.
5. **Radiative cooling is stiff but still bounded.** A 5000 K body wants
   to radiate ~10^8 W and would empty itself in a single step; the
   1000 K/step rate cap is what keeps it integrable. The cap does *not*
   freeze the cooling — the body still ends 999.92 K cooler.
6. **The pair radiative exchange is exactly antisymmetric.** Swapping
   which of the two bodies is hot leaves both the cold body's gain
   (0.022064) and the hot body's loss (2.462463) bit-identical. That only
   holds for the symmetric view-factor model; the legacy one is
   order-dependent, which is the entire reason the symmetric model
   exists.
7. **A thermal shock converges instead of exploding.** A cold box holding
   549900000.0 units of heat against a 5000 K body meets at 302.41 K /
   302.35 K — a residual gap of 0.0663 K after 1200 steps — with a peak
   speed of exactly 0.0000: the exchange is thermal, not mechanical, and
   the bodies never move.
8. **The sky/ground split moves the equilibrium by ±3 %.** A uniform 300 K
   ambient holds a body at exactly 300.0000 K; splitting the sky and the
   ground (100 K / 400 K) pushes it to 309.5192 K, and a cold sky
   (100 K / 100 K) pulls it down to 283.8012 K. A one-sided ambient is
   not a split one, and the difference is not a rounding artifact.
9. **The symmetric view-factor model is a strict refinement of the legacy
   one.** For equal bodies the two agree to six decimals (0.022064), so
   the model change is a no-op for symmetric scenes. For unequal bodies
   the legacy model returns the *same* number (0.022064 — it is
   size-blind) while the symmetric model returns 0.033112, a 50 % larger
   exchange. Scenes that mix body sizes will therefore see different
   thermal results after the upgrade; that is the intended fix and it is
   called out in `LIMITATIONS.md`.

## Nuclear stress suite (2026-10-20)

The nuclear subsystem (decay, capture/activation, fission, chain
reaction, dose) shipped with schema v12 and had **no** absolute-stress
coverage. `crates/engine/tests/nuclear_stress.rs` (7 tests) pins the
half-life law, the energy budget, the fission scaling law and the
boundedness of a runaway. Each test prints one `NUCLEAR ...` readings
line.

Run command (release required):

```text
cargo test --release -p phieng --test nuclear_stress -- --nocapture --test-threads=1
```

| Test | Bodies | Steps | Wall ms | ms/step | Reading |
| ---- | ------ | ----- | ------- | ------- | ------- |
| `nuclear_decay_matches_half_life` | 1 | 60 | 0.0 | 0.0002 | activity 1.5625 = expected 1.5625 (six half-lives from 100 Bq); energy 1469.94 J, heat 734.97 J; 293.15 → 294.1898 K (expected 294.1898) |
| `nuclear_decay_energy_splits_into_heat_and_radiation` | 1 | 60 | 0.0 | 0.0000 | heat 734.970 + radiation 734.970 = 1469.940 (50 % ionizing fraction) |
| `nuclear_fission_release_scales_with_fuel` | 2 | 300 | 0.0 | 0.0000 | fuel 0.00 → 0.0, 0.25 → 503.9, 0.50 → 1016.1, 0.75 → 1511.3, 1.00 → 2032.1 (J) |
| `nuclear_supercritical_runaway_stays_bounded` | 2 | 600 | 0.1 | 0.0002 | fuel_left 0.000000 (fully consumed); peak_activity 0.000; peak_temp 356.68 K; total_heat 22454.5 J |
| `nuclear_capture_chain_advances` | 1 | 60 | 0.0 | 0.0000 | chain_len 61; chain head [1, 2, 3, 2, 3, 2, 3, 2]; final isotope 3 |
| `nuclear_dose_accumulates_and_clamps` | 2 | 120 | 0.1 | 0.0005 | dose 1.0000 clamped at 1.0 Sv; source range 2.0 m |
| `nuclear_extreme_activity_stays_finite` | 1 | 60 | 0.0 | 0.0003 | activity 15625000235116659685573263360.0000 (finite, no NaN/inf); temperature clamps at 5000.0000 K |

Result: **7 passed, 0 failed** (total suite wall time ≈ 0.01 s in release).
(Tested on a Ryzen 7 7445HS with 15.2 gigs of ram.)

### Nuclear findings

1. **The half-life law is exact.** With a 10 s half-life and a time scale
   of 60 nuclear seconds per simulation second, 60 steps are six
   half-lives, and the activity lands on 100 / 64 = 1.5625 Bq to within
   1e-3 relative. The decay is a true exponential in the *nuclear* clock,
   not in the simulation clock.
2. **The energy budget closes from the other side.** The discrete release
   is a left Riemann sum (`A₀ · E · dt · (1 − r^N)/(1 − r)`), so the
   energy released over the window is 1469.94 J; only the non-ionizing
   half (734.97 J) reaches the body, and the body's temperature rises to
   exactly the analytic 294.1898 K from 293.15 K. Heat in, temperature
   out — the two sides agree to 1e-2 relative.
3. **The energy split is exact.** A 50 % ionizing fraction gives
   734.970 J of heat and 734.970 J of ionizing radiation, summing to the
   1469.940 J released. No energy appears or vanishes at the split.
4. **Fission release is linear in the fuel.** Sweeping the fuel fraction
   from 0.00 to 1.00 gives 0.0 → 503.9 → 1016.1 → 1511.3 → 2032.1 J —
   linear, with the 0.25 step landing on 503.9 of a 2032.1 total
   (24.8 %).
5. **A runaway burns out instead of exploding.** The supercritical scene
   consumes its fuel completely (fuel_left 0.000000), and the whole
   reaction stays at 356.68 K — 83.5 °C — for 22 454.5 J of total heat,
   with a peak activity of 0.000. The mean-field model has **no true
   criticality threshold**: the fission rate is proportional to the fuel,
   so any fuel at all burns, and the neutron spread cannot extinguish a
   reaction the way a stochastic one can. The suite pins the
   *boundedness*, not a criticality threshold the model does not have.
6. **The capture chain cycles.** 60 steps produce a 61-link chain whose
   head runs 1, 2, 3, 2, 3, 2, 3, 2 — capture products decay back into
   the chain, so a body with a capture-prone isotope keeps cycling rather
   than settling on the first product.
7. **Dose accumulates and clamps.** Two bodies 2.0 m apart accumulate a
   dose that clamps at exactly 1.0 (Sv), the configured maximum, instead
   of growing without bound.
8. **Extreme activity stays finite.** An activity of
   15625000235116659685573263360 (1.5625e28 — ten orders of magnitude past
   anything physical) neither overflows to
   infinity nor produces a NaN; the temperature simply clamps at the
   5000 K limit. The state stays loadable and saveable.

## Snapshot fuzzing suite (2026-10-20)

The snapshot loader is the one place where a corrupt file meets an
engine that will happily simulate whatever it is handed.
`crates/engine/tests/snapshot_fuzz.rs` (6 tests) mutates a valid
snapshot in every way that matters and asserts that the loader either
refuses the file or reproduces *exactly* the same world — never a
different one. Each test prints one `FUZZ ...` readings line.

Run command (release required):

```text
cargo test --release -p phieng --test snapshot_fuzz -- --nocapture --test-threads=1
```

| Test | Cases | Wall ms | ms/case | Reading |
| ---- | ----- | ------- | ------- | ------- |
| `fuzz_truncated_binary_never_parses` | 1731 | 0.2 | 0.0001 | every prefix of the file (all 1731 lengths) rejected; a prefix may only load as the same world |
| `fuzz_bit_flips_never_misparse` | 13848 | 2.3 | 0.0002 | every single-bit flip of the file (13848 = 1731 B × 8) rejected; same_world=0, different_world=0 |
| `fuzz_garbage_bytes_never_parse` | 2000 | 2.6 | 0.0013 | random buffers of random length rejected; accepted=0 |
| `fuzz_malformed_json_never_parses` | 2008 | 3.4 | 0.0017 | JSON mutations (truncations, byte flips, injected keys) rejected; accepted=0 |
| `fuzz_doctored_length_field_rejected` | 127 | 0.0 | 0.0002 | 127 doctored payload-length values (±1…±63 around the true length) rejected; accepted=0 |
| `fuzz_unknown_schema_version_rejected` | 301 | 0.1 | 0.0002 | 301 schema versions probed; 300 rejected; **accepted_outside_range=0**, accepted_inside_range=1 |

Result: **6 passed, 0 failed** (total suite wall time ≈ 0.01 s in release).
(Tested on a Ryzen 7 7445HS with 15.2 gigs of ram.)

### Fuzz findings

1. **Not one single-bit flip is accepted.** The suite flips each of the
   13 848 bits of a 1731-byte snapshot in turn — 13 848 separate
   corruptions of a single file — and the loader rejects every one of
   them. Zero files parsed into a different world (`different_world=0`),
   which is the property that actually matters: a corrupt file must not
   become a *plausible but wrong* simulation.
2. **Truncation is caught at every length.** All 1731 prefixes of the
   file are rejected, including the ones that stop just short of a
   complete body record. The loader validates the envelope length before
   it looks at the payload, so there is no "read past the end" class.
3. **Random bytes never parse.** 2000 random buffers of random length are
   all rejected, with no panic and no partial load.
4. **Malformed JSON never parses.** 2008 JSON mutations — truncations at
   many offsets, byte flips, injected keys — are all rejected. The JSON
   and binary paths fail independently, which matters because they are
   independent parsers.
5. **A doctored length field is always rejected.** Every one of the 127
   plausible wrong lengths (the true payload length ±1…±63) fails the
   envelope check.
6. **The one real hole: the header is not checksummed.** The envelope's
   CRC32 covers the **payload only** — not the magic, not the version,
   not the kind, not the length. A single-bit flip is always caught
   (because it lands in the payload, or because a version flip lands
   outside the readable range and is refused); but a *crafted* version
   field inside the readable range is accepted and the payload is then
   parsed **under the wrong layout**. The suite measures the exposure
   exactly: of 301 probed versions, **all 300 outside the readable range
   3..=12 are rejected**, and exactly **one inside the range (version 3)
   loads a v12 payload as if it were a v3 payload** — a silently
   different world. The test asserts the hard guarantee (out-of-range
   versions are never guessed) and reports the soft one, so a change in
   the exposure is visible in the readings. This is documented in
   `LIMITATIONS.md`; fixing it properly means either checksumming the
   header (a format break) or adding a layout self-check to the reader.

## Tier performance sweep (2026-10-20)

The tier blocks (B01–B06 broad-phase backends, Q01–Q05 solver quality,
H01–H05 collision tiers) pin what each tier *computes*. None of them
prices it, and a tier nobody has measured is a tier nobody can budget
for. `crates/engine/tests/tier_sweep.rs` (3 tests) measures the tiers on
the same scene, and asserts that the broad-phase backends agree
bit-for-bit. Each test prints one `TIER ...` readings line.

Run command (release required):

```text
cargo test --release -p phieng --test tier_sweep -- --nocapture --test-threads=1
```

### Broad-phase backends (300 bodies, 60 steps)

| Backend | ms/step | Note |
| ------- | ------- | ---- |
| BruteForce | 0.1862 | O(n²) pair scan |
| SpatialHash | 0.1893 | grid backend |
| DynamicAABBTree | 0.1805 | tree backend |
| Hybrid | 0.2147 | tree + hash, auto-selected per frame |

All four produce the **same** state hash `0x11aecc01950b82fc`.

### Collision tiers (200 bodies, 120 steps)

| Tier | ms/step | Contacts | Note |
| ---- | ------- | -------- | ---- |
| Tier1_AABBOnly | 0.0352 | 0 | broad-phase only — no manifold is produced |
| Tier2_SphereAABB | 0.0452 | 67 | |
| Tier3_SphereSphere | 0.0341 | 64 | |
| Tier4_BoxBox | 0.1333 | 282 | SAT box-box |
| Tier5_GJK_EPA | 0.1344 | 282 | general convex hulls |

### Solver quality tiers (200 bodies, 120 steps)

| Quality | ms/step | Mean-height drift vs Ultra |
| ------- | ------- | --------------------------- |
| Ultra | 0.1478 | +0.00000 |
| High | 0.1212 | +0.12700 |
| Medium | 0.1110 | +0.03155 |
| Low | 0.1014 | −0.51218 |
| Minimal | 0.0973 | −0.47369 |

Result: **3 passed, 0 failed** (total suite wall time ≈ 0.17 s in release).
(Tested on a Ryzen 7 7445HS with 15.2 gigs of ram. The per-tier milliseconds
above are from the reference run and move by a few percent between runs; the
contact counts, the drifts and the hash are exact.)

### Tier findings

1. **The broad-phase is not the bottleneck at this scale.** Brute force
   (0.1862 ms/step) is no slower than the spatial hash (0.1893 ms/step)
   and the dynamic AABB tree (0.1805 ms/step) on a 300-body pile. The
   narrow-phase and the solver dominate the frame, so a scene of this
   size gains nothing from a smarter broad-phase — the O(n²) scan is
   buried under the cost of the contacts it finds.
2. **All four backends agree bit-for-bit.** The hash
   `0x11aecc01950b82fc` is identical across BruteForce, SpatialHash,
   DynamicAABBTree and Hybrid. The backend is an implementation detail,
   not a modelling choice.
3. **Hybrid is the slowest backend** (0.2147 ms/step, +19 % over the
   tree) because it pays for the tree *and* the grid on the frames where
   it uses both. It is the safest default for mixed scenes, not the
   fastest one.
4. **The collision tiers are priced in contacts, not in milliseconds.**
   Going from Tier3 (0.0341 ms/step, 64 contacts) to Tier4 (0.1333 ms/step,
   282 contacts) costs 3.9× the time for 4.4× the contacts — roughly
   linear, with no cliff and no free lunch.
5. **GJK/EPA costs +0.8 % over the box-box tier** (0.1344 vs 0.1333
   ms/step) for the same 282 contacts. For a scene made of boxes, Tier5
   is pure overhead; for a scene made of hulls it is the only tier that
   produces correct contacts at all.
6. **Tier1 produces no contacts at all.** It is a broad-phase-only
   ceiling: the AABB test runs, but no narrowphase manifold is produced,
   so nothing collides (0 contacts, 0.0352 ms/step) and the pile falls
   straight through. That is the documented meaning of the ceiling — a
   pair above it is skipped, not approximated.
7. **The solver quality tiers buy ~34 % time, and the accuracy is not
   monotonic.** Ultra → Minimal is 0.1478 → 0.0973 ms/step (−34 %), but
   the *Low* tier drifts further from the Ultra reference (−0.51218 mean
   body height over 120 steps) than the *Minimal* tier does (−0.47369).
   Low spends its budget on iterations that converge to a slightly worse
   answer than Minimal's; if accuracy matters, do not assume the ladder
   is ordered.
