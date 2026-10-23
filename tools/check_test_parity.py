#!/usr/bin/env python3
"""Enforce Rust<->C# test parity for the robustness suite (and beyond).

Convention: every mirrored robustness test embeds a shared numeric ID in
its name:
    Rust:  fn s01_sleeping_hit_wakes(...)      (snake_case, letter lowercase)
    C#:    public void S01_SleepingHitWakes()  (PascalCase, letter uppercase)

ID blocks (numeric ranges are disjoint, so the reused S is unambiguous):
    S01-S05  sleep + solver interaction
    W06-W10  warm start + cache eviction
    N11-N15  never-guess / self-heal policy
    S16-S21  solver numerical limits
    D22-D27  2D/3D dispatch + shape edges
    F28-F40  cross-language / FFI boundary (F28-F30 C#-only by nature,
             exempt; F31-F33 native joint bridge, F34-F36 native snapshots,
             mirrored in Rust FFI tests; F40 managed-file/native-core
             cross-boundary check, C#-only by nature (it needs the native
             DLL on both sides), exempt)
    F41-F46  the native write surface (mirrored both languages: a body
             can be placed, spun, pushed and woken, the bodies can be
             enumerated, and bad input is refused)
    Z01-Z12  contact events and sensors (mirrored both languages: a pair
             that starts or stops touching produces an event, a resting
             pair produces none, a sensor reports its contacts and never
             pushes, the events are off by default, FFI round-trip, the
             3D sensor reports too, a pair reports one began event
             however many points it touches, the events are the last
             step's only, and removing a body ends its contacts)
    V01-V11  runtime validation layer (mirrored both languages)
    J31-J44  joint constraints (mirrored both languages)
    E01-E07  snapshots save/load (mirrored both languages)
    G01-G10  gravitational physics (mirrored both languages: G01-G07
             2D contract — orbit, determinism, Verlet energy order,
             softening safety, probes, snapshot round-trip, BH
             determinism; G08-G10 compact 3D set — orbit, determinism,
             snapshot)
    A01-A05  ambient forces (mirrored both languages: A01 spring/SHM
             period, A02 linear-drag terminal velocity, A03 turbulence
             determinism, A04 default no-op, A05 ambient-scale immunity)
    T01-T17  thermal subsystem (mirrored both languages: conduction,
             radiation, convection, clamping, reservoirs, no-op paths,
             determinism; 3D conduction + determinism)
    B01-B06  broad-phase tiers (mirrored both languages: backend parity
             2D/3D, Hybrid auto-select, default backend, mid-run switch
             transparency, body-id invariant, FFI round-trip)
    H01-H05  collision tiers (mirrored both languages: default is full
             fidelity, ceiling below the required tier stops contacts,
             ceiling at the required tier keeps them, 3D mirror,
             determinism under a capped ceiling)
    L01-L06  collision filtering (mirrored both languages: the default
             filter accepts every layer, a layer that does not accept the
             other side rejects the pair, one side is enough to veto it,
             3D mirror, snapshot round-trip, FFI round-trip)
    X01-X06  scene queries (mirrored both languages: the overlap query
             returns the bodies in the box in id order, the collision
             filter applies to queries, the query sees the current
             positions, 3D mirror, FFI round-trip)
    X07-X12  the ray cast (mirrored both languages: the ray finds the
             body in front of it, a ray that misses hits nothing, the
             collision filter applies, the nearest body wins, 3D mirror,
             FFI round-trip)
    Q01-Q05  solver quality tiers (mirrored both languages: iteration
             budgets, friction/restitution switches, snapshot round-trip)
    C01-C04  CCD tiers (mirrored both languages: thin-wall capture 2D/3D,
             default-off bit-identical, snapshot round-trip)
    K31-K44  3D joint constraints (mirrored both languages: distance,
              hinge, prismatic, fixed)
    R01-R07  deterministic scalar functions (mirrored both languages:
              pinned cos/sin bit patterns, pinned atan2 bit patterns,
              the engine's own 2D rotation integration pinned to exact
              bits, pinned exp bit patterns, pinned ln bit patterns,
              agreement with the platform libm within a tolerance, and
              exp/ln as mutual inverses - the properties that make
              rigid-body trajectories platform-independent and
              Rust/C#-identical)
    M01-M08  the quantum math layer (mirrored both languages: the
              complex arithmetic's pinned bit patterns, the complex
              transcendentals' pinned bit patterns, the edge cases
              where a naive formula overflows, the random number
              generator's pinned stream, reproducible seeding, the
              float mapping, the normal deviate, and a stream that
              resumes exactly from a saved state)
    M17-M26  the stationary Schrodinger equation (mirrored both
              languages: the infinite well's spectrum, the harmonic
              oscillator's spectrum, a ground state that is normalised
              and symmetric, the harmonic ground state's peak, the
              Sturm count, the tridiagonal solve, the lowest
              eigenvector, the double well, the soft Coulomb ground
              state, and the grid)
    M09-M16  the quantum vocabulary (mirrored both languages: a state
              normalises and its Born probabilities follow, a Hermitian
              expectation is real, the Born probability is a projector
              expectation, the overlap and the fidelity, the dense
              matrix operations, the eigensolver's spectrum, the
              eigenvectors that rebuild the matrix, and the power
              iteration's dominant eigenvalue)
    M27-M34  the time evolution (mirrored both languages: the
              transform round-trips, the Crank-Nicolson method conserves
              the norm and the energy and converges at the second order,
              the split operator conserves the norm and refuses a
              mismatched boundary, the probabilities sum to one, the
              sampling follows the Born rule, the field is inert until
              switched on, and the FFI round-trips the configuration and
              the amplitudes)
    M39-M48  the gates and circuits (mirrored both languages: the
              single-qubit gate identities, the entangling gates' truth
              tables, the Bloch sphere, the measurement rule, the
              Deutsch-Jozsa and Bernstein-Vazirani oracles in one query,
              the Grover search, the Fourier transform round trip, the
              teleportation fidelity and the superdense coding round
              trip)
    M49-M56  the game-facing quantum layer (mirrored both languages:
              the Born-rule sampling pinned shot for shot, the seeded
              stream's reproducibility, the Bohmian velocity of a
              stationary and a flowing state, the driven body surfacing
              deterministically, the Madelung dispersion, the classical
              limit, the visualisation feed, and the native bridge round
              trip)
    M57-M64  spin and particles (mirrored both languages: the Pauli
              algebra, the Stern-Gerlach measurement, the Rabi flop, the
              Zeeman splitting and Larmor precession, the bosonic ladder,
              the fermionic anticommutation that is the Pauli principle,
              the Fermi gas, and the spin rotations)
    M65-M72  entanglement and quantum information (mirrored both
              languages: the Bell states, half a pair maximally mixed,
              the von Neumann entropy, the CHSH game pinned past the
              classical bound, the noisy channels, the Lindblad step and
              its trace, the mixed-state ladder, and the three-qubit
              partial traces)
    M73-M80  the 3D grid and the fields (mirrored both languages: the
              grid bookkeeping, the cubic box and harmonic trap spectra
              with their triple degeneracies bit-identical, the
              normalised tensor-product ground state, the matrix-free
              versus dense cross-check, the zero-flux ring, the
              quarter-flux lift, and the flux periodicity)
    M35-M38  the quantum field in the snapshot (mirrored both languages:
              a world with a wavefunction round-trips through the binary
              and the JSON with its configuration and its amplitudes
              intact, a world without one keeps the schema it had before
              the field existed, a file written before v15 loads with the
              field off, and the bytes the Rust writer produces are the
              bytes the managed mirror produces)
    U01-U12  radiation model expansion (mirrored both languages: the
              legacy view-factor model is the default and matches the
              historical formula, the symmetric model matches it for
              equal bodies and is order-independent, it differs for
              unequal bodies, the sky/ground split changes the exchange
              and is bit-identical to the historical one when the sky
              and ground equal the ambient, an opaque occluder blocks
              the exchange (including an off-axis one next to an
              endpoint), a partial occluder attenuates it, a fully
              transparent one does not block, determinism plus parallel
              == serial, and a snapshot round-trip)
    Y01-Y28  nuclear physics (mirrored both languages: a disabled
              config is a no-op, the activity halves after a half-life,
              decay heats the body, the temperature respects the thermal
              clamps, the activity is clamped, the energy budget
              balances, a stable nuclide never decays, a body without a
              nuclide is untouched, a fission consumes fuel and releases
              heat, a supercritical chain reaction grows while a
              subcritical one dies out, delayed neutrons hold the
              release back, the fuel never goes negative, the 3D mirror
              decays too, determinism plus parallel == serial, a static
              body does not heat up, the nuclide table is usable, the
              decay constant is ln(2) over the half-life, the dose is
              deposited in the nearby bodies and clamped, the nuclear
              state round-trips through a snapshot, and the config and
              the per-body state round-trip through the native FFI)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# F28-F30, F36, F40 are C#-only by nature: F28-F30 predate the mirror blocks,
# F36 executes the managed loader against a Rust-written file (the
# Rust side pins the writer half via e07/f34 round-trips), and F40
# executes the Rust loader against a managed-written file (it needs the
# native DLL on both sides).
EXEMPT_CS_ONLY = {"F28", "F29", "F30", "F36", "F40"}

RUST_FN = re.compile(r"\bfn\s+([A-Za-z])(\d{2})\w*")
CS_METHOD = re.compile(r"public\s+(?:unsafe\s+)?(?:void|async\s+Task)\s+([A-Za-z])(\d{2})\w*\s*\(")


def collect(pattern: re.Pattern, files) -> set[str]:
    ids: set[str] = set()
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"warn: cannot read {f}: {e}")
            continue
        for m in pattern.finditer(text):
            ids.add(f"{m.group(1).upper()}{m.group(2)}")
    return ids


def main() -> int:
    rust_files = list((ROOT / "crates").rglob("*.rs"))
    cs_files = list((ROOT / "tests").rglob("*.cs"))
    if not rust_files:
        print("error: no Rust files found under crates/")
        return 1
    if not cs_files:
        print("error: no C# files found under tests/")
        return 1

    rust_ids = collect(RUST_FN, rust_files)
    cs_ids = collect(CS_METHOD, cs_files)

    rust_only = sorted(rust_ids - cs_ids - EXEMPT_CS_ONLY)
    # Exemptions are C#-only by nature; a Rust-side F-test is itself drift.
    cs_only = sorted((cs_ids - rust_ids) - EXEMPT_CS_ONLY)
    exempt_present = sorted(cs_ids & EXEMPT_CS_ONLY)

    print(f"Rust IDs ({len(rust_ids)}): {sorted(rust_ids)}")
    print(f"C# IDs ({len(cs_ids)}): {sorted(cs_ids)}")
    print(f"Exempt C#-only present: {exempt_present}")

    failed = False
    if rust_only:
        print(f"DRIFT: IDs only in Rust (missing C# mirror): {rust_only}")
        failed = True
    if cs_only:
        print(f"DRIFT: IDs only in C# (missing Rust mirror): {cs_only}")
        failed = True
    missing_exempt = sorted(EXEMPT_CS_ONLY - cs_ids)
    if missing_exempt:
        print(f"NOTE: exempt C#-only IDs not found (tests not yet written?): {missing_exempt}")
    print("PARITY " + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
