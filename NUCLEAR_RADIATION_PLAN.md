# Nuclear and radiation expansion — plan and status

This is the roadmap for the two subsystems added after the thermal
module: the **radiation model expansion** and the **nuclear physics**
subsystem. It records the decisions that were made and why, so the next
change does not have to re-derive them.

The status column is the state of the tree, not an aspiration.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Rate-based or transport-based neutrons? | **Rate-based** (a nearest-neighbour spread) | Transport needs per-neutron ray casting against the broadphase and a scattering model: a second physics subsystem, not a feature of this one. The rate model reproduces the qualitative chain-reaction behaviour, which is what a game needs. `LIMITATIONS.md` records the gap. |
| Where does the nuclear state live? | **On the body** (`NuclearState`) | A world side table would have to be kept in sync with the world's add/remove paths — and `remove_body` re-assigns the ids, so the table would need the same shifting. On the body it follows the body for free. The fields sit at the cold tail of the struct. |
| Is the nuclide table fixed or configurable? | **Configurable, with a built-in default table** | Users define their own nuclides; the engine ships a placeholder at index 0 ("no nuclide") so the default table is one entry long. |
| Stochastic fission? | **No — rate-based** | A deterministic engine is worth more than Poisson jitter, and the caller can add jitter from the deterministic counter-based hash the ambient turbulence already uses. |
| Where does the nuclear energy go? | **Into the thermal phase** | The nuclear phase runs *before* the thermal phase and fills a per-body heat budget; the thermal phase turns it into temperature, and the radiation then carries it away. One place writes the temperature. |
| Do the new physics options change the default trajectories? | **No** | Every new option defaults to the historical behaviour, so the pinned trajectory hashes do not move. This is the repo's standing rule and it is what let every phase here land without re-pinning. |
| Cross-platform? | **Yes, bit-for-bit** | The decay uses `exp`, which is exactly the libm function that broke CI in the first place. The deterministic exponential from Phase 0 is used instead. |

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Deterministic `exp`/`ln` (`phieng-math` + `PhiEng.Math`) | **Done** — R04–R07 |
| 1 | Extract the radiation model into its own module (a pure move) | **Done** — the pinned hashes did not move, which is the proof it was pure |
| 2 | Radiation model expansion: symmetric view factors, sky/ground split, transparent occluders, the occlusion fix | **Done** — U01–U12, schema v11 |
| 3 | Nuclear foundations: the nuclide table, the decay, the energy budget, the thermal coupling | **Done** — Y01–Y24 |
| 4 | Fission and the rate-based chain reaction, the delayed neutrons | **Done** — Y09–Y14 |
| 5 | The dose: depositing the ionizing energy into the bodies in range | **Done** — Y21–Y22 |
| 6 | The snapshot (schema v12), the FFI, the managed mirror | **Done** — Y23–Y24, F40 |

## What a later phase could add

- **Real neutron transport.** Straight-line neutrons with broadphase
  ray casting, scattering, and a dose field rather than a scalar. The
  ray-casting machinery already exists (the CCD tier uses it), so the
  expensive part is the scattering model and the cost.
- **Branching ratios.** A nuclide with a table of daughters and the
  fraction of decays that produce each, instead of a single `daughter`.
- **Delayed-neutron precursor groups.** Six precursor groups with
  their own half-lives instead of one exponential leak.
- **Per-body optical properties** (absorptivity, reflectivity,
  transmissivity). Deferred from Phase 2 so the radiation expansion
  would not have to grow the hot `PhysicsMaterial` struct; it belongs
  with the nuclear material properties, if they are added at all.
- **The ionizing band in the radiation module.** The nuclear phase
  currently deposits the dose directly. Routing it through the radiation
  module as a second band would let shielding, partial transparency and
  the view-factor machinery apply to ionizing radiation too.

## The trap that shaped all of it

The first CI failure of this work was **not** a physics bug: the
canonical 2D scene hashed differently on Linux and Windows because
`atan2` and `sin_cos` disagree between glibc and the Microsoft CRT in
the last ULP. One ULP, and a 240-step trajectory diverged completely.

Every transcendental in a simulation path is a portability hazard, so
the engine now has `phieng-math::trig` and `phieng-math::exponential`
for sine, cosine, `atan2`, `exp` and `ln`, and the managed mirror has
the same two modules. The rule is in `CONTRIBUTING.md`: *the rigid path
must not call the platform math library*.

The fluid and ambient subsystems still do (their particle placement and
turbulence field), which is recorded as a known gap.
