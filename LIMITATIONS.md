# Known limitations

Every entry here is pinned by a passing test whose name starts with
`known_limitation_` (correctness tests fail red = bug; limitation tests
pass = documented gap). Do not "fix" these quietly — each has a planned
phase or an explicit wont-fix rationale. Test-ID parity between Rust and
C# is enforced by `tools/check_test_parity.py`
(ID blocks: S01–S05 sleep, W06–W10 warm start, N11–N15 never-guess,
S16–S21 solver limits, D22–D27 dispatch/shape edges, F28–F33 FFI-only,
V01–V11 validation, J31–J44 joints, E01–E07 snapshots, F34–F36 native
snapshots, P31–P34 parallel islands, A01–A05 ambient forces, C01–C04 CCD,
Q01–Q05 solver quality, T01–T24 thermal, B01–B06 broadphase tiers,
K31–K44 3D joints,
H01–H05 collision tiers, R01–R07 deterministic trigonometry,
U01–U12 radiation expansion, Y01–Y28 nuclear physics).

## The nuclear model is game-scale, not reactor-scale (Y01–Y28)

The nuclear subsystem is a **rate-based** model, not a transport solve:

- **Neutrons travel by a nearest-neighbour spread**, not by straight-line
  transport. A body's free neutrons are shared equally among the
  bodies within `neutron_range` (the share is uniform among the
  neighbours). Real neutron transport would need per-neutron ray casting
  against the broadphase — the same machinery the CCD tier uses — and a
  scattering model. The diffusion approximation reproduces the
  *qualitative* behaviour (a chain reaction grows above the critical
  mass, dies out below it) and not the quantitative one.
- **Fission is a rate, not a stochastic event.** The fission rate is
  `flux * cross_section * fuel * rate`, so it is smooth in time. Real
  fission is Poisson-distributed. The consequence is that the engine
  cannot express "this one lump of fuel happened to fission now", and
  a chain reaction never shows the statistical jitter of a real one.
  Where randomness is wanted, the caller can drive it from the
  deterministic counter-based hash the ambient turbulence uses.
- **The nuclide table is flat.** A nuclide has one daughter, so a decay
  chain is a chain, not a tree: branching ratios (the fraction of
  decays that produce each daughter) are not modelled, and the
  activation path is a single `daughter` field.
- **Capture is mean-field, not per-neutron.** A body with a daughter
  accumulates capture progress at a rate proportional to the flux and
  the absorption cross-section, and becomes the daughter when that
  progress reaches one. A real capture is a per-neutron event with a
  branching ratio, so the model gets the *average* right and not the
  individual event. It is deterministic by construction (no RNG), which
  is the trade the engine makes.
- **There is no delayed-neutron *precursor* chain.** The delayed
  fraction is modelled as a single exponential leak with one delay
  time, not as the six precursor groups a real reactor has. The
  qualitative effect (prompt-critical versus delayed-critical) is
  captured; the timescales are not.
- **Dose is a scalar accumulator**, not a spatial field: it records
  how much ionizing energy a body received, with an inverse-square
  weight, and stops there. There is no organ dose, no dose rate, and
  no biological effect beyond whatever the game applies to the number.
- **The per-body state stores the nuclide *index***, so reordering the
  table changes what a saved body means. The table and the state
  travel in the same snapshot block, so a file round-trips correctly,
  but two snapshots with different tables are not comparable.

## Cross-platform determinism: the rigid core only (R01–R07)

The **rigid-body** core is platform-independent: the same inputs and
the same number of steps give the same bits on Linux, Windows and macOS,
and the same bits in Rust and in the managed C# mirror. The rotation
integration uses the engine's own trigonometry
(`phieng-math`'s `trig` module / `PhiEng.Math.DeterministicTrig`), which
is built only from operations IEEE-754 specifies exactly. Without it,
`sinf`/`cosf`/`atan2f` disagree between glibc and the Microsoft CRT in
the last ULP, and the compiler can introduce or remove that difference
by rewriting an inlined call — one ULP, and a 240-step trajectory
diverges completely.

The **fluid and ambient** subsystems are the exception: their particle
placement still calls the platform math library (`MathF.Cos`/`MathF.Sin`
in `FluidSolver`/`FluidSolver3`, and the turbulence field in
`ambient.rs`). Their trajectories are therefore reproducible *within* one
platform but not across platforms, and Rust↔C# agreement there stays
statistical rather than bit-exact. Moving them onto the deterministic
helpers is mechanical and is the natural next step; it was left out
because those subsystems have no pinned trajectory hash today, so
re-pinning them is a separate, deliberate act.

## Sleep is full but support-gated (S01, S02)

Sleeping bodies are frozen: skipped in gravity, ambient, fluid force
application, integration, and the contact solver (treated as infinite
mass, exactly like statics). What wakes them:

- **New contact** auto-wakes (an uncached pair involving a sleeper),
  with a transitive cascade over contact and joint partners.
- **Lost contact** auto-wakes (a vanished pair re-evaluates support),
  also cascaded — a pulled-out-from-under stack falls instead of
  floating.
- **Direct impulse/force** (`apply_impulse`/`apply_force`, including
  fluid coupling) wakes its target; explicit `wake_body` still works
  and cascades the same way.

A body falls asleep only when slow **and** supported by a current
contact — ballistic apexes and drifting bodies never sleep, so no
wake/sleep history exists and save/load stays exact (schema unchanged).

Two consequences to know:

- An unsupported slow body (apex, drifter) keeps full pipeline cost
  instead of sleeping; this is rare and intentional.
- Joints with an awake partner keep their bodies awake (a jointed
  sleeper wakes the moment its partner moves); fully frozen joints
  are skipped.

## Convex-hull mass properties are a stub (D27)

`ConvexHull` bodies use `mass = 1, I = diag(1,1,1)` regardless of density
or vertex layout. Two identical hulls with different densities behave
identically. Prefer capsules/boxes — or the new `ConvexMesh` shape, which
carries exact closed-mesh mass properties (Mirtich-style) — for dynamic
bodies whose inertia matters.

## CCD via adaptive substepping (C01–C04)

Collision is discrete (`step(dt)`, semi-implicit Euler) by default, so
fast or thin bodies tunnel; behavior in tunneling regimes is bounded
(no NaN, finite speeds) but not correct. Opt-in CCD
(`CcdConfig.enabled`, both languages, snapshot schema v6, FFI
`phieng_world[_3]_set/get_ccd_config`) splits a step into substeps so
no body travels more than `max_movement` per substep (default 0.1,
capped by `max_substeps`, default 8) — a thin wall thinner than that
budget still tunnels, by design. Off (default) is bit-identical to
legacy stepping. The `tier` field (`CcdTier`: Off / SweptAABB / FullGJK)
selects *which bodies* substep: `SweptAABB` only substeps when some
broadphase pair's swept volumes overlap (sparse scenes pay nothing),
`FullGJK` keeps the legacy peak-speed behavior. Note "FullGJK" names
the substepping thoroughness, not a true time-of-impact solve — there
is still no TOI root-finding or swept-shape narrowphase in this engine.
Two related notes: EPA normals are verified in
the shallow/moderate-penetration regime the solver maintains — deeply
interpenetrated pairs (past any range the solver would let persist) can
report flipped axes, which is another reason penetration must be
arrested early rather than resolved late.

## One scene query so far (AABB overlap), no raycasts or triggers — broadphase tiers, rigid-only joints, snapshots are versioned v13

Broad-phase backends (`BroadphaseMode`: brute force, spatial hash,
dynamic AABB tree, hybrid auto-select — B01–B06) are behavior-identical
by construction: every backend emits the same sorted+deduped pair
sequence, so switching mode mid-run is transparent. Known caveats:

- **Grid backends report cell overlaps, not AABB overlaps.** A spatial
  hash's `collect_pairs` emits every pair that shares a cell, so it can
  report pairs whose AABBs do not actually touch (the narrowphase
  rejects them, so the simulation is unaffected — B01/B02 pin
  bit-identical trajectories across backends). What a grid must never
  do is *miss* an overlapping pair; that is a false negative and would
  drop a real contact. The step pipeline itself consumes the narrower
  `collect_overlapping_pairs` instead (same sequence minus the
  AABB-disjoint pairs, which can never produce contacts); the CCD swept
  test keeps the unfiltered sequence on purpose.
- **The dynamic tree has no incremental update.** The world rebuilds the
  whole tree every step (`clear` + `insert`), and `remove` rebuilds
  bottom-up from the surviving leaves. Remote bodies that never move
  still pay full insert cost each step — the win over brute force comes
  from pair pruning, not from amortization.
- **Monotone insertion degenerates the tree.** The SAH descent alone
  turns a sorted-insert stream into a linked list; a height-triggered
  bottom-up rebuild (past 2×log₂) restores balance. The threshold is
  conservative, so adversarial orderings pay one O(n log n) rebuild
  before settling.

The scene query surface is one query deep so far: an AABB overlap
(\overlap_aabb\). A ray cast, a shape overlap, a point test and a
closest-point query are not built yet, so a game that needs to pick a
body under the mouse, sweep a character controller or fire a hitscan
still has to do that work itself.

Collision filtering (L01–L06) is applied in the narrowphase and in
the CCD swept test, not in the broad phase: the grid still emits the
candidate pair and the narrow phase is the first place that knows
whether the two bodies may collide. The pair census therefore does not
shrink when a body is filtered — only the contacts do.

Joints exist in both dimensions (distance, hinge with angle limits,
prismatic with travel limits, fixed weld — rigid only, see
`crates/dynamics/src/joint.rs` / `joint3.rs`, the J31–J44 (2D) and
K31–K44 (3D) parity tests, plus the generic
`phieng_world_add_joint` / `phieng_world3_add_joint` FFI bridges and
their managed mirrors). Not yet built: non-zero compliance (springs —
rejected with `JointError::UnsupportedCompliance` rather than faked),
motors, and breakable limits. The ray cast covers the analytic shapes
(a circle, an oriented box and a convex polygon in 2D; a sphere, an
oriented box and a capsule in 3D) but not the convex ones: a ray that
would hit a convex hull or a convex mesh is reported as a miss. The
other shape queries (shape overlap, point test, closest point) are not
built yet. The ray cast's distance is the parameter along the direction
as it was given: a direction that is not a unit vector scales the
range, and so does `max_t`. A caller that wants a distance in world
units should normalize the direction first.
Vehicles exist in both dimensions (raycast suspension with bump
stops and friction-circle tires in 3D, the same tire model top-down
in 2D — see `crates/dynamics/src/vehicle.rs` / `vehicle3.rs`, the
I01–I08 parity tests, and the `phieng_world[_3]_vehicle_step` FFI
bridges with their managed mirrors). They are stateless by design:
no wheel-spin memory, no slip-ratio state, no holding brake at
standstill, no load transfer and no anti-roll — a driven wheel that
unloads simply pulls less. Two envelope edges to respect: a
hardpoint driven below the ground surface reads as airborne (rays
that start inside a shape report a miss, the house rule), so size
`max_suspension_force` to stop the chassis before the springs run
out; and a belly graze hands the motion to the contact solver, whose
sub-penetration touches can resolve a last bit apart across mirrors
(the I04/I08 pins stay in the contact-free envelope for exactly this
reason).
Save/load exists (JSON + versioned binary, exact + fresh modes, 2D + 3D,
see `crates/dynamics/src/snapshot.rs` and the E01–E07 parity tests).
The parallel island solver (`Solver.Parallel`, P31–P34) is 2D-only;
3D worlds always take the serial path. The parallel path shares one
early-exit decision across islands (a per-island exit would diverge from
serial), so the islands rendezvous on a barrier every iteration; that
barrier carries a timeout, and a desynchronization panics with a
diagnostic rather than hanging the run (a shared single-slot buffer used
to race exactly that way — see `CHANGELOG.md`).

## Solver mass-ratio limit (S16, S17 — quantified)

The sequential-impulse solver has no mass-ratio clamp. Measured:
1000:1 is stable over 10,000 steps (light-body jitter peak < 1.0,
rests 5 mm into slop); 1e6:1 stays bounded over 1200 steps (settles
within centimeters, no NaN, no escape). Beyond that, expect
jitter/drift — unmeasured territory, not a supported regime.

## Fluid stepping (Rust core + managed C# mirror)

SPH particles (`FluidState2`/`FluidState3`) step inside
`World2::step`/`World3::step` and couple two-way with rigid bodies
(SAT/analytic boundary impulses plus pressure-sampled buoyancy and
drag), in both languages. Cross-language equivalence is behavior
parity (tolerance-based, like the rigid equivalence tests), not
bit-equality: the managed solver mirrors the Rust pipeline
stage-by-stage (density/pressure, adaptive resampling, boundary
ghosts, pressure forces, XSPH, collision, buoyancy/drag) but uses the
managed noise library, so turbulence-driven trajectories differ
across languages while calm scenes agree tightly. The pressure model
is linear by default (`gamma == 1`); setting
`gamma > 1` selects the Tait/WCSPH EOS, and `artificial_viscosity` /
`surface_tension` add Monaghan viscosity / simplified Akinci cohesion.
CFL substepping (`cfl`, `max_substeps`) keeps stiff Tait configs stable
by subdividing the step when `dt >= cfl*h/(c0+v)`; it is off by default
and costs `substeps`× the fluid work. Two remaining caveats: the
cohesion kernel is a bounded linear spline, not the full Akinci spline
(stable and tunable, not a surface-energy match), and boundaries are
still penalty-based (positional push-out + reaction impulses), not
boundary/ghost particles — a particle can still outrun pressure buildup
at extreme speeds (the `drag` coefficient caps entry speed instead).
Adaptive resolution (`adaptive_resolution`) is opt-in and conserves
mass (verified), but its split criterion keys on low density, which also
describes free surfaces — so it is stable for calm/settled scenes and
should be avoided in violent dam-break-style transients where surface
splitting can over-refine and destabilize; merging is a simple
closest-pair combine, not a full resampling scheme. Boundary particles
  (`boundary_particles`) add pressure-based walls (two-pass WCSPH) that
  reduce tunneling. Ghosts persist across substeps and regenerate only
  when the static set/transforms or kernel config change (fingerprinted);
  their density/pressure still recompute every pass — but the ghosts
  themselves are not persisted and the penalty boundary still runs
  underneath, so extreme-speed
tunneling is mitigated rather than eliminated. Fluid state
persists through snapshots (config block extended at schema v5, both
languages) and crosses the FFI bridge (config set/get plus
phase-aware particle insert). 2D polygons and 3D hulls/meshes are
impulse-coupled only (no pressure sampling — no winding contract for
outward normals, nearest-vertex response for hulls/meshes).

## FFI shape subset (F29)

The C ABI (and therefore `PhiNativeWorld(3)`) exposes Circle/Box
(2D) and Sphere/Box/Capsule (3D) only. Polygon, ConvexHull, and ConvexMesh
bodies exist in the native engines but cannot be created through the
bridge; unknown shape tags are rejected cleanly (`-1` / `0`, no UB).
ConvexMesh snapshot DTOs are new in schema v2; managed mesh support is
still pending, so mesh snapshots round-trip in Rust only for now.

## Sliver-skip policy (N11–N15)

Capsule/hull pairs whose Minkowski inradius around the origin is below
`1e-4` (grazing touches, needle overlaps) yield no contact for that
frame and self-heal as penetration deepens or separates. The skip is
bounded and convergent, never accumulating; box–box and sphere pairs
never take this path. A skipped frame under Earth gravity at 60 Hz costs
roughly ~3 mm / ~0.16 m/s on the affected pair, recovered over the
following frames.

## Ambient forces are a game-y approximation (A03, A05)

The ambient fields (spring/SHM, air resistance, turbulence) are physical
in form but tuned for gameplay, not CFD. Specific documented gaps:

- **Cross-language noise is not bit-equal.** The seeded turbulence field
  is bit-identical *within* one backend (A03 pins per-backend
  determinism), but Rust and C# use different float operation orders, so
  cross-language trajectories diverge — the same rule as gravity.
  Parity binds to statistical/energy/amplitude bounds, never exact
  positions.
- **Drag reference area is a per-shape proxy** (2D: circle `2r`, box
  `2*(hx+hy)`, polygon bounding diameter; 3D: sphere `πr²`, box mean
  face area, capsule side+caps, hull bounding sphere) — an
  orientation-independent characteristic area, not a projected-area
  solve. Retune the quadratic coefficient or use per-body
  `ambient_scale` for exact behavior.
- **Statics and sleeping bodies are skipped.** Statics never receive
  ambient force (as with gravity), and `ambient_scale == 0` makes a body
  immune (A05). Sleeping bodies are not woken by ambient fields (see the
  sleep limitation above); disable sleep (thresholds to `0.0`) for
  scenes where turbulence must keep bodies in motion.
- **Turbulence is an acceleration; spring and drag are forces.**
  Turbulence is mass-independent (like gravity) so heterogeneous debris
  moves coherently; spring and drag divide by mass, giving the proper
  oscillator `ω = sqrt(k/m)` and terminal velocity `g*m/c`.

## Residual per-step allocations

The broadphase grid and contact cache now recycle their buffers through
internal pools (dynamic pile: ~0.8 allocs/step 2D, 0.0 3D; resting 0.0).
The Barnes-Hut tree arena is also pooled (thread-local arena reused
across kicks instead of rebuilt), so BH gravity scenes no longer pay
per-kick tree allocation. One source still allocates and is a
documented gap, not a regression:

- **Parallel island solve** allocates its per-step island/constraint
  scratch (off by default; serial path is allocation-free).
- The broadphase grid and contact cache still **compact vacated entries
  every 256 steps** (recycling buffers to their pool) to bound memory for
  roaming worlds; churn is bounded by the compaction cadence.

## Fluid SPH benchmark gaps (pinned by the SHM/SPH suite)

Canonical fluid problems are tested (see `absolute_stress_tests.md`).
Where the method cannot hit the textbook result, the test pins the
limitation rather than faking it:

- **No-slip walls are opt-in.** `wall_friction` ∈ [0, 1] (default `0` =
  free-slip, bit-identical) drives the tangential particle velocity
  toward the wall's at each contact. With it off a force-driven channel
  is **flat**; with it on the profile curves toward the center
  (`sph_poiseuille_profile`), and it removes tangential momentum
  (`sph_wall_friction_slows_flow`). It is a penalty velocity match, not
  a full ghost-viscosity / Akinci no-slip, so the profile is
  qualitatively — not quantitatively — Chapman.
- **The Martin & Moyce √t surge-front law is not reproduced.**
  `sph_dam_break_surge_front` asserts the front advances and stays
  contained, not the quantitative √t position law.
- **Hydrostatic pressure overshoots at the floor.** The penalty floor
  compresses the bottom particle layer, so the measured `p_bottom` can
  exceed `ρ g h` (measured ~2.3× in the 2D column).
  `sph_hydrostatic_pressure` asserts monotone depth and
  order-of-magnitude support.
- **Penalty boundaries tunnel at high speed** (no CCD, as elsewhere): a
  violent dam break can eject a fast particle through a thin wall. The
  benchmark uses thick walls.

## Two-phase fluid limitations

- **Phase index is clamped to 0..=3.** `add_particles_phase` masks
  out-of-range phase values to 3; at most 4 phases are supported.
- **Per-phase equation of state only.** Each phase sets its own rest
  density and particle mass (so a heavier phase packs more mass per
  volume and sinks). There is no per-phase surface tension, viscosity, or
  immiscibility model — phases interact through the same shared pressure
  kernel, so they mix like miscible fluids with different densities rather
  than separating like oil/water.
- **No interfacial tension.** The cohesion kernel is single-phase; a
  heavy/light interface is a density gradient, not a surface-energy
  boundary. For true immiscible multiphase flow, surface tension between
  phases is still missing.


## Collision tiers are a fidelity-for-cost trade (H01–H05)

The narrowphase tier ceiling (`CollisionTier`: AABB-only, sphere-AABB,
sphere-sphere, SAT box-box, GJK/EPA) is opt-in and off by default. Two
documented consequences:

- **A capped ceiling silently stops contacts.** A pair whose required
  tier exceeds the ceiling is *skipped*, not approximated, so the bodies
  pass through each other. That is the intended trade (cheaper
  narrowphase for scenes that never need exact contact), but it is a
  fidelity change, not a performance-only change: use it only for
  scenes whose shape mix fits under the ceiling.
- **The ceiling is per-world, not per-pair.** There is no per-body or
  per-shape override, so a scene mixing hulls (Tier5) with boxes (Tier4)
  cannot keep exact hull contacts under a Tier4 ceiling.

## Thermodynamics & radiation (T01–T24)

The thermal subsystem adds per-body temperature evolution via conduction,
radiation, and convection, plus three opt-in sub-features: radiation
occlusion, forced convection, and thermal expansion. Documented gaps:

- **No phase change.** Bodies carry fixed material properties; there is
  no melting/solidification or latent heat. A body's specific heat,
  conductivity, and emissivity never change with temperature.
- **Radiation occlusion is a coarse test.** `radiation_occlusion`
  (opt-in, per-world) blocks body-to-body radiation when a third body
  lies between the pair. The test is a segment/sphere overlap against
  the blocker's bounding radius, not an exact shadow volume: a body
  that is *near* the line of sight but does not actually touch it can
  still block radiation, and the blocker is treated as fully opaque
  regardless of its size or material. Off by default (an O(n) scan per
  pair).
- **Forced convection is a linear speed model.** The convection
  coefficient becomes `h + h_forced · |v|`, where `|v|` is the body's
  speed relative to the (still) ambient. There is no boundary-layer
  model, no flow direction (a body moving *towards* a wall sheds heat
  exactly like one moving away), and no natural-convection model
  (Rayleigh–Bénard, etc.). Opt-in per-world.
- **Thermal expansion rescales the shape, not the mesh topology.**
  `expansion_coefficient` (opt-in, per-world) scales a body's shape by
  `1 + alpha · (T − T_ref)` and refreshes its mass and inertia. The
  factor is applied relative to the *authored* shape (the applied
  factor is tracked per body and persisted in snapshots), so a body
  held at a constant temperature does not grow without bound. Off by
  default, so default trajectories stay bit-identical. Gaps: expansion
  is uniform (no anisotropic or phase-specific swelling), the shape is
  rescaled in place (vertex data is not regenerated, so a polygon's
  original vertex count and winding are preserved but its tessellation
  is not refined), and a body that changes shape mid-step has its
  broad-phase bounds refreshed on the next step.
- **The symmetric view-factor model changes results for mixed-size
  scenes.** `ViewFactorModel::Symmetric` (the default) computes a
  symmetric pair exchange; the legacy model is order-dependent and
  *size-blind* — it returns the same exchange for a pair of equal bodies
  and for a pair of unequal ones. For equal bodies the two models agree
  to six decimals, so symmetric scenes are unaffected; for a pair with a
  1:2 radius ratio the symmetric model returns ~50 % more exchange
  (0.033112 vs 0.022064 in the stress suite). That difference is the
  point of the change — the legacy number was wrong for unequal bodies —
  but it means a scene that mixes body sizes sees different thermal
  results after the upgrade. Pinned by
  `thermal_symmetric_view_factor_matches_legacy`.
- **Static bodies are thermal reservoirs.** A static body's temperature
  never changes (infinite heat capacity); it can supply or absorb
  unlimited heat without temperature drift. This is intentional for
  performance and simplicity.
- **2D conduction uses simplified geometry.** Contacts in 2D are points;
  the effective contact width is derived from the smaller body's
  characteristic length, not the actual contact area. Conductance is
  scaled by `CONDUCTION_SCALE = 500` for game-timescale dynamics rather
  than strict SI units.
- **Cross-language parity is tolerance-based.** Rust and C# thermal
  trajectories agree on statistical/energy/amplitude bounds but are not
  bit-equal (same rule as ambient turbulence and gravity). T01–T24 pin
  per-backend determinism, not cross-language bit-equality.
- **Managed C# thermal mirror covers fewer shapes.** `PhiEng.Dynamics`
  `World2/3` now steps the same thermal subsystem as the Rust core
  (conduction, radiation, convection, plus the three opt-in
  sub-features), and its snapshot layout is byte-identical to the Rust
  writer's, so a file written by either engine loads in the other. Gaps:
  the managed side has no `ConvexMesh` shape (the Rust 3D hull is a
  placeholder in *both* engines — mass 1, identity inertia — so the two
  agree), and the managed 2D/3D solvers are independent
  implementations, so thermal trajectories agree on the qualitative
  outcome (which way the heat flowed) rather than bit-for-bit.

## The snapshot envelope does not checksum its header

The binary snapshot envelope is `magic | version | kind | length |
payload | crc32`, and the CRC32 covers the **payload only** — not the
magic, not the version, not the kind, not the length. Consequences:

- **A single-bit corruption is always caught.** A bit flip either lands
  in the payload (the CRC fails) or in the header (the magic/kind
  checks fail, or the version lands outside the readable range and is
  refused). All 13 848 single-bit flips of a 1731-byte snapshot are
  rejected, and so is every one of its 1731 prefixes.
- **A crafted version field inside the readable range is accepted.**
  Version 3 can be written over a version-12 header: the CRC still
  matches (the payload did not change), the version passes the range
  check, and the payload is then parsed **under the v3 layout** — a
  silently different world. The loader never guesses a version outside
  the readable range 3..=12 (all 300 probed out-of-range versions are
  refused), but inside the range the layout is taken on trust.
- **Fixing it means changing the format.** Either the CRC covers the
  header (every existing file stops loading — a format break) or the
  reader self-checks the layout it just parsed (e.g. that the body
  count and the trailing bytes are consistent) and refuses a payload
  that does not fit the version it was given. The second option is
  backward compatible and is the planned fix.

Pinned by `fuzz_unknown_schema_version_rejected` (the out-of-range
rejection is asserted; the in-range exposure is measured and reported in
the readings) and `fuzz_bit_flips_never_misparse`.
