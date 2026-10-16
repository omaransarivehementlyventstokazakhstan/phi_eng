# PhiEng — Complete Setup from Scratch

This document walks through every step required to clone, build, test, and run PhiEng on a fresh machine. No prior state is assumed.

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|-----------------|-------|
| Git | 2.40+ | |
| Rust | 1.78+ (stable) | MSRV = 1.78; install via [rustup.rs](https://rustup.rs) |
| .NET SDK | 10.0.x | [dotnet.microsoft.com](https://dotnet.microsoft.com/download/dotnet/10.0) |
| Python | 3.10+ | Only for `tools/check_test_parity.py` |
| CMake | 3.25+ | Required by some `windows-sys` bindings on Windows |
| Visual Studio 2022 or Build Tools | 17.8+ | Windows: "Desktop development with C++" workload |

### Verify prerequisites

```bash
git --version
rustc --version          # ≥ 1.78
cargo --version
dotnet --version         # 10.0.x
python --version         # 3.10+
cmake --version          # 3.25+
```

---

## 1. Clone the repository

```bash
git clone https://github.com/omaransarivehementlyventstokazakhstan/phi_eng.git
cd phieng
```

The repo already contains the Windows native DLL staged at:
```
src/PhiEng.Native/runtimes/win-x64/native/phieng.dll
```
Linux/macOS native libraries are **built by CI**; they are not committed.

---

## 2. Rust workspace — build & test

```bash
# Debug build (all crates)
cargo build --workspace

# Release build (optimized, used by benches and native DLL)
cargo build --workspace --release

# All tests (unit + integration)
cargo test --workspace

# Specific test suites
cargo test --lib                          # 172 unit/integration tests
cargo test --test brutal                  # 13 stress tests
cargo test --test parallel_tests          # 4 parallel island tests
cargo test --test snapshot_tests          # 7 snapshot round-trip tests
cargo test --test differential_fuzz       # 1000-case EPA-vs-SAT fuzz
cargo test --test property_tests          # Proptest stateful tests
cargo test --test robustness              # Robustness suite (S/W/N/D/F/V/J)
cargo test --test integration             # Cross-crate integration
```

### Run examples (Rust)

```bash
# 2D deterministic demo (identical output every run)
cargo run -p phieng --example demo_2d

# SPH fluid dam-break (release build recommended)
cargo run -p phieng --example fluid_dam --release

# Large-scene benchmark (prints allocation census + trajectory hash)
cargo run -p phieng --example bench_large --release

# Same benchmark, keeps process alive for profiling
cargo run -p phieng --example bench_large --release -- no-exit

# Native-CPU profile (AVX2/FMA on capable hardware)
# Windows PowerShell:
powershell -ExecutionPolicy Bypass -File tools/build-native.ps1 -Bench
# Linux/macOS:
./tools/build-native.sh --bench
```

---

## 3. .NET solution — build & test

```bash
# Restore dependencies
dotnet restore PhiEng.slnx

# Debug build
dotnet build PhiEng.slnx

# Release build (used by NuGet pack)
dotnet build PhiEng.slnx -c Release

# Run all C# tests (147 tests)
dotnet test tests/PhiEng.Tests/PhiEng.Tests.csproj

# Or full solution
dotnet test PhiEng.slnx -c Release
```

### Run samples (C#)

Each sample is a 5-minute hello-world:

```bash
dotnet run --project samples/hello-2d/hello-2d.csproj
dotnet run --project samples/hello-3d/hello-3d.csproj
dotnet run --project samples/hello-joint/hello-joint.csproj
dotnet run --project samples/hello-snapshot/hello-snapshot.csproj
dotnet run --project samples/hello-native/hello-native.csproj   # uses FFI bridge
```

---

## 4. Cross-language parity check

```bash
python tools/check_test_parity.py
# → PARITY OK
```

This verifies that Rust and C# test IDs align (S/W/N/D/F/V/J/E/P blocks).

---

## 5. Native library (Rust cdylib) — rebuild from source

The staged `win-x64` DLL is a portable baseline x86-64 binary. To rebuild it or target other platforms:

### Windows (PowerShell)

```powershell
# Portable baseline (what CI ships)
powershell -ExecutionPolicy Bypass -File tools/build-native-all.ps1 -Rids win-x64,linux-x64,osx-x64,osx-arm64

# Native-CPU profile (opt-in, NOT redistributable as portable)
powershell -ExecutionPolicy Bypass -File tools/build-native.ps1
# Or with bench:
powershell -ExecutionPolicy Bypass -File tools/build-native.ps1 -Bench
```

Outputs land in `target/<target-triple>/release/phieng.dll` (or `.so`/`.dylib`).

### Linux / macOS (bash)

```bash
# Portable baseline for all RIDs
./tools/build-native-all.sh win-x64 linux-x64 osx-x64 osx-arm64

# Native-CPU profile
./tools/build-native.sh --bench
```

### Staging for C# consumption

After building, copy the artifact into the NuGet package layout:

```bash
# Windows example
mkdir -p src/PhiEng.Native/runtimes/win-x64/native
cp target/x86_64-pc-windows-gnu/release/phieng.dll src/PhiEng.Native/runtimes/win-x64/native/

# Linux example
mkdir -p src/PhiEng.Native/runtimes/linux-x64/native
cp target/x86_64-unknown-linux-gnu/release/libphieng.so src/PhiEng.Native/runtimes/linux-x64/native/
```

The `PhiEng.Native` csproj automatically packages everything under `runtimes/{rid}/native/` into the NuGet package. The `DllImportResolver` in `NativeMethods.cs` loads the correct file for the current RID at runtime.

---

## 6. Packing NuGet packages locally

```bash
dotnet pack PhiEng.slnx -c Release --no-build -o ./local-nuget
```

Outputs (version 0.3.1):
```
local-nuget/
├── PhiEng.0.3.1.nupkg
├── PhiEng.0.3.1.snupkg
├── PhiEng.Math.0.3.1.nupkg
├── PhiEng.Math.0.3.1.snupkg
├── PhiEng.Collision.0.3.1.nupkg
├── PhiEng.Collision.0.3.1.snupkg
├── PhiEng.Broadphase.0.3.1.nupkg
├── PhiEng.Broadphase.0.3.1.snupkg
├── PhiEng.Dynamics.0.3.1.nupkg
├── PhiEng.Dynamics.0.3.1.snupkg
├── PhiEng.Native.0.3.1.nupkg
├── PhiEng.Native.0.3.1.snupkg
└── PhiEng.Bench.0.3.1.nupkg
```

Install locally for testing:

```bash
dotnet add package PhiEng --source ./local-nuget
dotnet add package PhiEng.Native --source ./local-nuget
```

---

## 7. CI / GitHub Actions — what runs on push

| Job | OS | Purpose |
|-----|----|---------|
| `rust` | ubuntu-latest, windows-latest | fmt, clippy, doc, tests |
| `csharp` | ubuntu-latest, windows-latest | dotnet test + parity (ubuntu builds `libphieng.so`) |
| `version-check` | ubuntu-latest | Cargo version = MSBuild version |
| `msrv` | ubuntu-latest | Build with Rust 1.78 |
| `bench-smoke` | ubuntu-latest | Trajectory hash + alloc census (portable) |
| `bench-native` | ubuntu-latest | Native-CPU bench (informational, continue-on-error) |
| `host-smoke` | windows-latest | Host example headless run |
| `readme-smoke` | ubuntu-latest | All 5 hello samples compile & run (ubuntu builds `.so`) |

All jobs must pass before merge.

---

## 8. Publishing a release (maintainer only)

### Prerequisites
- `NUGET_API_KEY` secret configured in GitHub repository settings
- Write access to `omaransarivehementlyventstokazakhstan/phi_eng`

### Steps

1. **Finalize changelog** — rename `[Unreleased]` section in `CHANGELOG.md` to `[0.3.1] - <date>`
2. **Tag and push**:
   ```bash
   git tag v0.3.1
   git push origin v0.3.1
   ```
3. **Watch CI** — two release workflows trigger:
   - `release-native.yml` → builds 4 RIDs, creates `.zip`/`.tar.gz` assets, attaches to GitHub Release
   - `publish-nuget.yml` → packs and pushes all 7 packages to nuget.org
4. **Bump version files** to next `-dev` (see `API_STABILITY.md`):
   - `Cargo.toml` workspace `version = "0.4.0-dev"`
   - `Directory.Build.props` `<Version>0.4.0-dev</Version>`
5. **Commit and push** the version bump.

---

## 9. Common issues & fixes

| Symptom | Fix |
|---------|-----|
| `DllNotFoundException: phieng` on Linux | Run `cargo build --release -p phieng` and stage `libphieng.so` to `src/PhiEng.Native/runtimes/linux-x64/native/` |
| `CS1591` warnings on build | Run `dotnet build -p:TreatWarningsAsErrors=false` — XML docs are complete on `PhiEng.Dynamics` public surface |
| `error[E0463]: crate \`log\` not found` | `cargo build -p phieng-dynamics --features log` — `log` is optional |
| `RUSTDOCFLAGS=-D warnings` fails | Usually a missing `///` doc on a public item; `cargo doc --workspace --no-deps` locally to see |
| `dotnet test` hangs on Ubuntu | Ensure `libphieng.so` is staged; FFI tests skip gracefully but hello-native needs it |
| MSRV build fails | Check `ci.yml` uses `dtolnay/rust-toolchain@1.78` — update if MSRV bumps |

---

## 10. Directory layout cheat sheet

```
phieng/
├── .github/workflows/     # CI + release automation
├── crates/                # Rust workspace members
│   ├── math/              # phieng-math
│   ├── collision/         # phieng-collision
│   ├── broadphase/        # phieng-broadphase
│   ├── dynamics/          # phieng-dynamics
│   └── engine/            # phieng (cdylib + rlib, FFI entry)
├── src/                   # C# projects
│   ├── PhiEng.Math/
│   ├── PhiEng.Collision/
│   ├── PhiEng.Broadphase/
│   ├── PhiEng.Dynamics/
│   ├── PhiEng.Native/     # FFI bridge + staged runtimes/
│   ├── PhiEng/            # Facade package
│   └── PhiEng.Bench/      # Allocation/performance benchmarks
├── tests/PhiEng.Tests/    # xUnit tests (147 tests, mirrors Rust)
├── samples/               # 5 hello-world samples
├── examples/host/         # Headless host example (CI smoke)
├── tools/                 # build-native scripts, parity checker
├── artifacts/             # Local NuGet output (gitignored)
├── Cargo.toml             # Workspace root
├── Directory.Build.props  # Shared MSBuild props (version, license, repo)
├── PhiEng.slnx            # Solution file
├── CHANGELOG.md
├── RELEASE_CHECKLIST.md
├── API_STABILITY.md
├── LIMITATIONS.md
├── SECURITY.md
├── UNLICENSE
└── README.md
```

---

## 11. License

This project is released under **The Unlicense** (public domain dedication). See `UNLICENSE` for full text.

---

## 12. Support / contributing

- Bug reports: GitHub Issues
- Security: GitHub Security Advisories (see `SECURITY.md`)
- Questions: Discussions tab

Before contributing, read `API_STABILITY.md` (versioning/compatibility policy) and `RELEASE_CHECKLIST.md` (release gates).