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
    F28-F36  cross-language / FFI boundary (F28-F30 C#-only by nature,
             exempt; F31-F33 native joint bridge, F34-F36 native snapshots,
             mirrored in Rust FFI tests)
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
    Q01-Q05  solver quality tiers (mirrored both languages: iteration
             budgets, friction/restitution switches, snapshot round-trip)
    C01-C04  CCD tiers (mirrored both languages: thin-wall capture 2D/3D,
             default-off bit-identical, snapshot round-trip)
    K31-K44  3D joint constraints (mirrored both languages: distance,
             hinge, prismatic, fixed)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# F28-F30, F36 are C#-only by nature: F28-F30 predate the mirror blocks,
# and F36 executes the managed loader against a Rust-written file (the
# Rust side pins the writer half via e07/f34 round-trips).
EXEMPT_CS_ONLY = {"F28", "F29", "F30", "F36"}

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
