# Known limitations

Every entry here is pinned by a passing test whose name starts with
`known_limitation_` (correctness tests fail red = bug; limitation tests
pass = documented gap). Do not "fix" these quietly — each has a planned
phase or an explicit wont-fix rationale. Test-ID parity between Rust and
C# is enforced by `tools/check_test_parity.py`
(ID blocks: S01–S05 sleep, W06–W10 warm start, N11–N15 never-guess,
S16–S21 solver limits, D22–D27 dispatch/shape edges, F28–F33 FFI-only,
V01–V11 validation, J31–J44 joints, E01–E07 snapshots, F34–F36 native
snapshots, P31–P34 parallel islands, A01–A05 ambient forces).

## Sleep is partial (S01, S02)

Bodies flagged asleep are **not** skipped anywhere in the pipeline, and
there is **no auto-wake on contact and no wake cascade**. Concretely:

- A sleeping body hit by a moving body does not wake; it keeps
  accumulating gravity and solver impulses each step with velocities
  zeroed at step end (effectively static-like until manually woken).
- Waking the bottom of a sleeping stack does not wake the bodies above.

User impact: sleep currently only damps residual velocity; it saves no
CPU. Call `wake_body` explicitly after teleporting, spawning onto, or
otherwise disturbing a sleeping body. Full sleeping (pipeline skip +
auto-wake on new contact + cascade) is a planned separate phase with its
own fuzz/test cycle — deliberately *not* done inside an edge-case
validation phase, so validation results stay interpretable.

## Convex-hull mass properties are a stub (D27)

`ConvexHull` bodies use `mass = 1, I = diag(1,1,1)` regardless of density
or vertex layout. Two identical hulls with different densities behave
identically. Prefer capsules/boxes — or the new `ConvexMesh` shape, which
carries exact closed-mesh mass properties (Mirtich-style) — for dynamic
bodies whose inertia matters.

## No CCD (N12)

Collision is discrete (`step(dt)`, semi-implicit Euler). Fast or thin
bodies tunnel; behavior in tunneling regimes is bounded (no NaN, finite
speeds) but not correct. Two related notes: EPA normals are verified in
the shallow/moderate-penetration regime the solver maintains — deeply
interpenetrated pairs (past any range the solver would let persist) can
report flipped axes, which is another reason penetration must be
arrested early rather than resolved late. Conservative advancement is
planned after 3D joints.

## No raycasts, triggers — joints are 2D-only, snapshots are versioned v5

2D joints exist (distance, hinge with angle limits, prismatic with travel
limits, fixed weld — rigid only, see `crates/dynamics/src/joint.rs` and
the J31–J44 parity tests, plus the generic `phieng_world_add_joint`
FFI bridge). Not yet built: 3D joints of any kind, non-zero compliance
(springs — rejected with `JointError::UnsupportedCompliance` rather than
faked), motors, and breakable limits. No ray/shape queries beyond internal
broadphase iteration, no sensor flags or contact callbacks.
Save/load exists (JSON + versioned binary, exact + fresh modes, 2D + 3D,
see `crates/dynamics/src/snapshot.rs` and the E01–E07 parity tests).
The parallel island solver (`Solver.Parallel`, P31–P34) is 2D-only;
3D worlds always take the serial path. Conservative advancement is
planned after 3D joints.

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
Two sources still allocate and are documented gaps, not regressions:

- **Barnes-Hut** rebuilds its arena and per-leaf index `Vec`s every kick
  (the tree is not pooled) — a gravity-scene cost only; the exact N-body
  path stays allocation-free.
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



