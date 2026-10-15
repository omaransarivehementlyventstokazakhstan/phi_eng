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

This script extracts IDs from Rust test fns (crates/**/*.rs) and C# test
methods (tests/**/*.cs) and fails (exit 1) on any drift outside the
exemption list. Run it manually or wire it into CI; a green suite with a
red parity check means the mirror is drifting.
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
