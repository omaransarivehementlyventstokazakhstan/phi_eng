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
#   powershell -ExecutionPolicy Bypass -File tools/build-native.ps1
#   powershell -ExecutionPolicy Bypass -File tools/build-native.ps1 -Bench

param([switch]$Bench)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:RUSTFLAGS = "-C target-cpu=native"

if ($Bench) {
    cargo run --manifest-path "$root/Cargo.toml" -p phieng --example bench_large --release
} else {
    # Native cdylib for the local host (NOT the shipped portable DLL).
    cargo build --manifest-path "$root/Cargo.toml" --target x86_64-pc-windows-gnu --release -p phieng
    Write-Host "Native cdylib at target/x86_64-pc-windows-gnu/release/phieng.dll"
    Write-Host "Do NOT copy this over native/win-x64/ (that slot is the portable build)."
}
