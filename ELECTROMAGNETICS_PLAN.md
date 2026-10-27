# Electromagnetics — plan and status

This is the roadmap and the record for the `phieng-electromagnetics`
module. The status column is the state of the tree, not an aspiration.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where does the code live? | A separate crate `phieng-electromagnetics`, mirrored by `src/PhiEng.Electromagnetics` | The field solvers are body-agnostic; the crate can be tested without a world. `dynamics` owns the per-body state and the kick, exactly as it does for quantum. |
| Where does per-body state live? | On the body (`ElectroState2/3`) | Survives removals and reindexing; the hot part of the body struct is untouched. |
| Pairwise or grid? | **Two exclusive modes** | Applying both counts the same field twice. The uniform external fields are added in both modes. |
| Units | Game-scale defaults (`k = 1`, `c = 1`) with SI constants available | `f32` cannot multiply two elementary charges without losing precision. |
| Determinism | Only IEEE-exact operations, no platform libm | The EM path joins the rigid core: bit-identical across platforms and Rust/C#. |
| Default | Off | Every existing pinned trajectory hash is unchanged. |
| 2D field model | 2D Poisson grid = line-charge Green's function; pairwise = the 3D law sampled on a plane | The two are not interchangeable; documented as a limitation. |
| Snapshot | Schema v17, conditional; default worlds still emit v16 | Same rule every earlier subsystem followed. |

## What landed

| Area | Scope | Status |
|---|---|---|
| Crate foundations | `ElectroMode`, `FieldBoundary`, configs, per-body state, constants, validation | **Done** |
| Electrostatics | Softened Coulomb pair forces, field and potential queries, superposition | **Done** — EM01–EM03, EM10 |
| Coupling | Per-body `charge_scale`, static exerts without moving, sleeping inert, default-off bit identity | **Done** — EM04–EM06 |
| Conductors | Isolated (fixed charge) and grounded (drain) | **Done** — EM08 |
| Magnetostatics | Uniform `E`/`B`, `q(E + v x B)`, 3D dipole torque | **Done** |
| Field mode | Red-black SOR Poisson solve, bilinear/trilinear sampling, field pushes | **Done** — EM07 |
| Snapshot | Schema v17, config + per-body + solved potential, JSON and binary, both languages | **Done** — EM12 |
| FFI | Config/body/charge/queries, size-asserted structs, 2D+3D, managed wrappers | **Done** — EM11 |
| C# mirror | `PhiEng.Electromagnetics`, `ElectroIntegration`, body/world fields, snapshot | **Done** |
| Docs | CHANGELOG, LIMITATIONS, README, parity tool | **Done** |

## What a later phase should add

- **FDTD Maxwell waves** (2D TMz/TEz and 3D Yee), with CFL substepping,
  PEC/periodic/Mur boundaries, charge-conserving current deposition and
  radiation. This is the largest remaining piece; it belongs in its own
  crate module and its own test block.
- **Per-cell materials** (`eps_r`, `mu_r`, `sigma` arrays) for
  waveguides, dielectric interfaces and lossy media.
- **Contact charge sharing** between touching conductors (a capacitance
  graph), and induced surface charge / polarisability for dielectrics.
- **Pairwise magnetostatics** (Biot–Savart between moving charges) and
  magnetic materials.
- **A charge Barnes–Hut tree** for very large `Pairwise` scenes (the
  current path is `O(n^2)`, like exact gravity).
- **Lumped circuits** (RC/RL/RLC) coupled to bodies.

## The traps this design avoids

- **Double counting** — the mode enum is exclusive; there is no hybrid.
- **Sign conventions** — the 2D `Bz` cross product and the Coulomb sign
  are pinned by analytic tests.
- **Default drift** — the EM phase is a branch on `enabled`; the pinned
  hashes did not move.
- **Snapshot growth** — the grid and snapshot have one shared cap
  (262,144 cells), so a save can never produce a file the engine cannot
  load.
- **C# mirror drift** — the field/body/config layouts have Rust size
  asserts and C# `StructLayout` mirrors; the snapshot has JSON and binary
  round-trips in both languages.
