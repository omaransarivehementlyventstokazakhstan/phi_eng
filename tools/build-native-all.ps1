<##
.SYNOPSIS
    Builds portable baseline native binaries for all supported RIDs.
    Outputs to runtimes/{rid}/native/ for NuGet packaging.

    Supported RIDs:
      - win-x64      (Windows x64, MSVC or GNU toolchain)
      - linux-x64    (Linux x64, glibc)
      - osx-x64      (macOS x64)
      - osx-arm64    (macOS ARM64)

    The portable baseline uses NO `-C target-cpu=native` — it's the
    baseline that runs on any x86-64 / ARM64 CPU of the target OS.

    Bit-parity: trajectory hashes are recorded per RID and stored in
    README.md. Within a RID, determinism is bit-exact.

    Usage (from repo root):
      powershell -ExecutionPolicy Bypass -File tools/build-native-all.ps1
      powershell -ExecutionPolicy Bypass -File tools/build-native-all.ps1 -Rids win-x64,linux-x64
#>

param(
    [string[]]$Rids = @("win-x64", "linux-x64", "osx-x64", "osx-arm64"),
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$projectRoot = $root

Write-Host "=== Building portable native binaries for RIDs: $($Rids -join ', ') ==="

# Ensure Rust targets are installed
foreach ($rid in $Rids) {
    $target = switch ($rid) {
        "win-x64"      { "x86_64-pc-windows-gnu" }
        "linux-x64"    { "x86_64-unknown-linux-gnu" }
        "osx-x64"      { "x86_64-apple-darwin" }
        "osx-arm64"    { "aarch64-apple-darwin" }
        default        { throw "Unknown RID: $rid" }
    }
    Write-Host "Ensuring Rust target: $target"
    rustup target add $target 2>$null | Out-Null
}

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning previous builds..."
    Remove-Item -Recurse -Force "$projectRoot/target" -ErrorAction SilentlyContinue
}

# Build for each RID
$built = @{}
foreach ($rid in $Rids) {
    $target = switch ($rid) {
        "win-x64"      { "x86_64-pc-windows-gnu" }
        "linux-x64"    { "x86_64-unknown-linux-gnu" }
        "osx-x64"      { "x86_64-apple-darwin" }
        "osx-arm64"    { "aarch64-apple-darwin" }
    }
    $outputName = switch ($rid) {
        "win-x64"      { "phieng.dll" }
        "linux-x64"    { "libphieng.so" }
        "osx-x64"      { "libphieng.dylib" }
        "osx-arm64"    { "libphieng.dylib" }
    }

    Write-Host "Building for $rid ($target)..."
    cargo build --manifest-path "$projectRoot/Cargo.toml" --target $target --release -p phieng

    $src = "$projectRoot/target/$target/release/$outputName"
    if (-not (Test-Path $src)) {
        throw "Build output not found: $src"
    }

    # Stage to runtimes/{rid}/native/ for NuGet packaging
    $destDir = "$projectRoot/src/PhiEng.Native/runtimes/$rid/native"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $dest = "$destDir/$outputName"
    Copy-Item -Force $src $dest
    Write-Host "  Staged to $dest"
    $built[$rid] = $dest
}

# Also copy the win-x64 portable build to the legacy native/win-x64/ location
# for backward compatibility with existing test setup
$legacyDir = "$projectRoot/native/win-x64"
New-Item -ItemType Directory -Force -Path $legacyDir | Out-Null
$win64path = $built["win-x64"]
Copy-Item -Force $win64path "$legacyDir/phieng.dll"
Write-Host "Legacy copy: $legacyDir/phieng.dll"

Write-Host "=== All RIDs built and staged ==="
foreach ($rid in $Rids) {
    $path = $built[$rid]
    Write-Host ("  {0}: {1}" -f $rid, $path)
}