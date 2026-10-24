#!/usr/bin/env python3
"""A/B benchmark harness with a hash guard.

Runs a benchmark target N times, reports the median and the spread, and
re-checks the pinned trajectory hashes. The hash check is the point: a
performance change that moves a pinned hash is a behaviour change, not an
optimisation, and this script fails loudly instead of letting it slip.

Usage:
    python tools/bench_ab.py --target pile2d
    python tools/bench_ab.py --target pile2d --baseline PERF_BASELINE.md
    python tools/bench_ab.py --all

The targets are the scenes `bench_large` prints. The hash guard reads the
hashes the run prints and compares them with the pinned constants in
`crates/engine/examples/bench_large.rs`.
"""
import argparse
import io
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The pinned trajectory hashes. Source of truth: bench_large.rs. Keep this
# table in sync with it — a mismatch here is a failed gate, never a
# silently updated constant.
PINNED = {
    "pile2d": ("0x4acb6694f3a9917", "pile 2D"),
    "islands": ("0xd4a3a56f185c13b2", "islands"),
    "pile3d": ("0x4aad96014e832f5b", "pile 3D"),
}

# The measurement lines `bench_large` prints, and the label used for them.
MEASURES = [
    ("pile2d", r"iters=1: median ([0-9.]+) ms/step", "pile2d_iters1"),
    ("pile2d", r"iters=8: median ([0-9.]+) ms/step", "pile2d_iters8"),
    ("pile2d", r"per step: ([0-9.]+) ms", "pile2d_census"),
    ("pile2d", r"allocs/step: ([0-9.]+)", "pile2d_allocs"),
    ("islands", r"\[islands\] serial=([0-9.]+) ms/step", "islands_serial"),
    ("islands", r"parallel=([0-9.]+) ms/step", "islands_parallel"),
    ("pile3d", r"\[pile3d\] iters=8: median ([0-9.]+) ms/step", "pile3d_iters8"),
]


def run_bench(reps: int, extra_args=None):
    """Run `bench_large` `reps` times; return the parsed outputs."""
    cmd = ["cargo", "run", "-p", "phieng", "--example", "bench_large", "--release"]
    if extra_args:
        cmd += extra_args
    outputs = []
    for _ in range(reps):
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            sys.exit("bench_large failed (exit %d) — the hash guard tripped"
                     % proc.returncode)
        outputs.append((proc.stdout, elapsed))
    return outputs


def parse(outputs):
    """Median + spread per measure, plus the hashes the run reported."""
    values = {}
    for target, pattern, label in MEASURES:
        samples = []
        for out, _ in outputs:
            m = re.search(pattern, out)
            if m:
                samples.append(float(m.group(1)))
        if samples:
            values[label] = {
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
                "n": len(samples),
            }
    hashes = {}
    for out, _ in outputs:
        m = re.search(r"final hash: (0x[0-9a-f]+)", out)
        if m:
            hashes["pile2d"] = m.group(1)
        m = re.search(r"\[islands\].*serial=(0x[0-9a-f]+)", out)
        if m:
            hashes["islands"] = m.group(1)
        m = re.search(r"final hash3d: (0x[0-9a-f]+)", out)
        if m:
            hashes["pile3d"] = m.group(1)
    return values, hashes


def check_hashes(hashes):
    """Fail loudly if a pinned hash moved."""
    bad = []
    for key, (expected, label) in PINNED.items():
        got = hashes.get(key)
        if got is None:
            bad.append("%s: the run did not report a hash" % label)
        elif got != expected:
            bad.append("%s: got %s, expected %s" % (label, got, expected))
    if bad:
        print("HASH GUARD TRIPPED:")
        for b in bad:
            print("  " + b)
        print("\nThe physics moved. A performance change must be "
              "bit-identical; if the move was intended, re-pin the "
              "constants in bench_large.rs, README.md and "
              "absolute_stress_tests.md and say so in the CHANGELOG.")
        sys.exit(1)
    print("hash guard: OK (" + ", ".join(
        "%s %s" % (PINNED[k][1], hashes.get(k, "?")) for k in sorted(PINNED)) + ")")


def load_baseline(path):
    """Read a baseline JSON blob written next to PERF_BASELINE.md."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None
    return json.loads(io.open(p, encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="the scene to report (informational; all are measured)")
    ap.add_argument("--reps", type=int, default=5, help="runs of bench_large (default 5)")
    ap.add_argument("--baseline", help="a baseline JSON file to compare against")
    ap.add_argument("--save", help="write the measurements to this JSON file")
    ap.add_argument("--all", action="store_true", help="report every measure")
    args = ap.parse_args()

    print("running bench_large %d times ..." % args.reps)
    outputs = run_bench(args.reps)
    values, hashes = parse(outputs)

    check_hashes(hashes)

    print("")
    print("%-22s %10s %10s %10s" % ("measure", "median", "min", "max"))
    for label, v in values.items():
        print("%-22s %10.4f %10.4f %10.4f" % (label, v["median"], v["min"], v["max"]))

    baseline = load_baseline(args.baseline) if args.baseline else None
    if baseline:
        print("")
        print("%-22s %10s %10s %9s" % ("measure", "baseline", "now", "delta"))
        for label, v in values.items():
            base_entry = baseline.get(label)
            if base_entry is None:
                continue
            base = base_entry["median"] if isinstance(base_entry, dict) else base_entry
            delta = 100.0 * (v["median"] - base) / base if base else 0.0
            flag = ""
            if delta < -3.0:
                flag = "  faster"
            elif delta > 3.0:
                flag = "  SLOWER"
            print("%-22s %10.4f %10.4f %8.1f%%%s"
                  % (label, base, v["median"], delta, flag))

    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = ROOT / p
        io.open(p, "w", encoding="utf-8", newline="\n").write(
            json.dumps(values, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
