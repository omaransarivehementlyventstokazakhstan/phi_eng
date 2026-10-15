#!/usr/bin/env bash
# Opt-in native-CPU build: compiles the Rust core + bench with
# `-C target-cpu=native` (AVX2/AVX512, FMA where the host has it).
#
# The DEFAULT build (and the shipped native/win-x64 DLL) stays portable
# baseline x86-64. This profile is for local benchmarking and hosts that
# control their deployment hardware. Binaries built here REQUIRE the host
# CPU class — never redistribute them as the portable artifact.
#
# Bit-parity note: SIMD codegen may reorder reductions at the ulp level,
# so trajectory hashes are recorded SEPARATELY per profile. Within one
# profile, determinism still holds bit-exact (same binary twice).
#
# Usage (from repo root):
#   ./tools/build-native.sh [--bench]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUSTFLAGS="-C target-cpu=native"

if [[ "${1:-}" == "--bench" ]]; then
    cargo run --manifest-path "$ROOT/Cargo.toml" -p phieng --example bench_large --release
else
    cargo build --manifest-path "$ROOT/Cargo.toml" --release -p phieng
    echo "Native cdylib built for the local host (NOT the shipped portable artifact)."
    echo "Do NOT copy this over native/win-x64/ (that slot is the portable build)."
fi
