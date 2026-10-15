#!/usr/bin/env bash
# Builds portable baseline native binaries for all supported RIDs.
# Outputs to runtimes/{rid}/native/ for NuGet packaging.
#
# Supported RIDs:
#   - win-x64      (Windows x64, GNU toolchain)
#   - linux-x64    (Linux x64, glibc)
#   - osx-x64      (macOS x64)
#   - osx-arm64    (macOS ARM64)
#
# The portable baseline uses NO `-C target-cpu=native` — it's the
# baseline that runs on any x86-64 / ARM64 CPU of the target OS.
#
# Bit-parity: trajectory hashes are recorded per RID and stored in
# README.md. Within a RID, determinism is bit-exact.
#
# Usage (from repo root):
#   ./tools/build-native-all.sh [--clean] [--rids win-x64,linux-x64]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIDS=("win-x64" "linux-x64" "osx-x64" "osx-arm64")
CLEAN=false

# Parse args
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
        --rids=*) IFS=',' read -ra RIDS <<< "${arg#--rids=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "=== Building portable native binaries for RIDs: ${RIDS[*]} ==="

# Ensure Rust targets are installed
declare -A TARGET_MAP=(
    ["win-x64"]="x86_64-pc-windows-gnu"
    ["linux-x64"]="x86_64-unknown-linux-gnu"
    ["osx-x64"]="x86_64-apple-darwin"
    ["osx-arm64"]="aarch64-apple-darwin"
)
declare -A OUTPUT_MAP=(
    ["win-x64"]="phieng.dll"
    ["linux-x64"]="libphieng.so"
    ["osx-x64"]="libphieng.dylib"
    ["osx-arm64"]="libphieng.dylib"
)

for rid in "${RIDS[@]}"; do
    target="${TARGET_MAP[$rid]}"
    echo "Ensuring Rust target: $target"
    rustup target add "$target" 2>/dev/null || true
done

if [[ "$CLEAN" == true ]]; then
    echo "Cleaning previous builds..."
    rm -rf "$ROOT/target"
fi

declare -A BUILT
for rid in "${RIDS[@]}"; do
    target="${TARGET_MAP[$rid]}"
    output="${OUTPUT_MAP[$rid]}"
    echo "Building for $rid ($target)..."
    cargo build --manifest-path "$ROOT/Cargo.toml" --target "$target" --release -p phieng

    src="$ROOT/target/$target/release/$output"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: Build output not found: $src" >&2
        exit 1
    fi

    # Stage to runtimes/{rid}/native/ for NuGet packaging
    dest_dir="$ROOT/src/PhiEng.Native/runtimes/$rid/native"
    mkdir -p "$dest_dir"
    dest="$dest_dir/$output"
    cp -f "$src" "$dest"
    echo "  Staged to $dest"
    BUILT[$rid]="$dest"
done

# Legacy copy for win-x64 (backward compat with existing tests)
legacy_dir="$ROOT/native/win-x64"
mkdir -p "$legacy_dir"
cp -f "${BUILT[win-x64]}" "$legacy_dir/phieng.dll"
echo "Legacy copy: $legacy_dir/phieng.dll"

echo "=== All RIDs built and staged ==="
for rid in "${RIDS[@]}"; do
    echo "  $rid: ${BUILT[$rid]}"
done