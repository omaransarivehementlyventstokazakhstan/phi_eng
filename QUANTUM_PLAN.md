# Quantum mechanics in PhiEng — the plan

This is the working plan for adding quantum mechanics to the engine: a
ladder of stages, each one a shippable, revertible commit, each one
landing the Rust core, the managed C# mirror, the C ABI and the
mirrored parity tests together.

The ladder is ordered by *dependency and value*, not by difficulty: the
gates and circuits (Stage E) are conceptually easier than the
time-dependent Schrödinger equation (Stage D) but need the complex layer
and nothing else, so they can be built alongside.

## The decisions behind the ladder

| Question | Answer | Why |
| --- | --- | --- |
| How strict is the Rust↔C# parity? | **Bit-identical throughout.** | The rigid core already claims it; the quantum stack is pure arithmetic and can hold the same line. The cost is that every kernel is written twice, operation for operation. |
| Which strand leads? | **Wave physics first, gates alongside.** | The Schrödinger equation is the physical heart; the gates are self-contained and quick once the complex layer exists. |
| Where does the code live? | **A new `crates/quantum` crate.** | Mirrors the broadphase/collision split, keeps `world.rs` from growing further, and gives the mirror a parallel `src/PhiEng.Quantum` home. |
| How does it interleave with the classical roadmap? | **Quantum first, then resume.** | The quantum work touches the math layer, the crate graph, the FFI and the snapshot schema; doing it in one stretch avoids paying that integration cost twice. |

## The ground rules

These are the gates the CI already runs. A stage that breaks one is not
done.

- **Bit-identical Rust↔C#, and platform-independent.** Every
  transcendental goes through `phieng-math`'s deterministic helpers
  (`trig`, `exponential`, and from Stage A `complex` and `rng`). The
  platform libm is never called.
- **f32 state, f64 scratch, rounded once.** The same precision model
  the deterministic trigonometry uses.
- **Pinned bit patterns.** Each kernel gets a module-local
  `*_pinned_bit_patterns` unit test *and* a mirrored parity ID. A
  change in either fails loudly.
- **Mirrored parity IDs.** The quantum stack uses the letter **M**
  (`a b c d e f g h j k l n p q r s t u v w x y z` are taken). The
  parity checker fails the build on drift.
- **Inert by default.** The quantum subsystem is off by default, so the
  pinned rigid-body hashes never move and no re-pin is needed.
- **The subsystem attaches to the world.** No standalone handle: the
  exports are `phieng_world_quantum_*` and `phieng_world3_quantum_*`.
- **The snapshot appends.** New state lands in a new version-gated
  block at the end of the payload, behind a *named* constant
  (`SNAPSHOT_SCHEMA_QUANTUM`), never `VERSION - 1`.
- **Zero steady-state allocation.** The grids use pooled scratch
  buffers, like the solver does.

### The five determinism traps

1. **FMA contraction.** Rust and RyuJIT do not contract `a*b + c` into
   a fused multiply-add by default, but a hand-written `mul_add` on one
   side only would silently diverge. Never use `f32::mul_add` in a
   mirrored kernel; write the multiply and the add separately.
2. **`Math.Round` is banker's rounding in C#.** Use
   `MidpointRounding.AwayFromZero`, the repo's existing pattern.
3. **Complex division.** The naive `(ac+bd)/(c²+d²)` overflows. Use
   Smith's algorithm, in the same branch order on both sides.
4. **The modulus.** `sqrt(re² + im²)` overflows above `|z| ~ 2^38` and
   underflows below `|z| ~ 2^-38`. Scale by the larger component first.
5. **The Fourier twiddles.** `exp(-2πik/N)` must be built from the
   deterministic trigonometry, with the angle computed in an identical
   expression. A libm `cosf` there would poison every stage after it.

## The ladder

| Stage | What lands | IDs |
| --- | --- | --- |
| **A** | The complex layer and the seeded random generator (`phieng-math`). | M01–M08 |
| **B** | States, operators, dense matrices, the Hermitian Jacobi solver. | M09–M16 |
| **C** | The stationary 1D Schrödinger equation: the grid, the potentials, the Hamiltonian, the Sturm bisection, the imaginary-time ground state. | M17–M26 |
| **D** | The time evolution: Crank–Nicolson, the split operator, the deterministic FFT, the first FFI exports, snapshot v15. | M27–M34 |
| **E** | The gates and circuits: the qubits, the Bloch sphere, the gates, the measurement, the algorithms. | M35–M44 |
| **F** | The game-facing layer: Born-rule bodies, the Madelung quantum-fluid, the visualisation feed. | M45–M52 |
| **G** | Spin and particles: the Pauli algebra, the Stern–Gerlach, the second quantisation. | M53–M60 |
| **H** | Entanglement and quantum information: the Bell states, the CHSH game, the density matrices, the Lindblad master equation. | M61–M68 |
| **I** | The 3D grid and the fields: the vector potential, the Aharonov–Bohm effect, the 3D FFI and snapshot. | M69–M76 |
| **J** | The advanced: a tight-binding band structure, a Hartree–Fock-lite, a DMRG-lite, the scattering, the parameter-shift rule. | M77–M84 |

### What each stage must prove

The tests assert the *physics*, not just the code:

- **A** — the arithmetic and the transcendentals are bit-identical on
  both sides; the generator's stream is reproducible and resumable.
- **B** — ⟨ψ|ψ⟩ = 1; ⟨ψ|A|ψ⟩ real for a Hermitian A; the Born
  probability equals ⟨ψ|Π|ψ⟩; the fidelity of a state with itself is 1.
- **C** — the infinite well's E_n = n²π²ℏ²/(2mL²) to 1e-4 relative; the
  harmonic oscillator's E_n = (n+½)ℏω; the eigenstates normalised and
  orthogonal; the double well's ground state below the barrier.
- **D** — the norm conserved to 1e-5 over 1 000 steps; the energy
  conserved with the observed convergence order 2; the free Gaussian
  packet's width matching the analytic one to 1 %; Crank–Nicolson and
  the split operator agreeing to 1e-3.
- **E** — H² = X² = I; S² = Z; T⁴ = I; the CNOT's truth table; the
  Grover success probability; the QFT's inverse; the teleportation's
  fidelity 1.
- **F** — the sampled positions follow |ψ|² (a χ² fit over 10 000
  samples); the same seed gives the same samples; the quantum fluid's
  dispersion matches the analytic one to 5 %; the classical limit
  (ℏ → 0) recovers the ordinary fluid.
- **G** — the Stern–Gerlach split exactly ±½; the anticommutator 1 to
  1e-6; the Fermi gas's ground-state energy.
- **H** — the Bell state's partial trace = I/2; the entropy of a
  product state 0 and of a Bell state 1 bit; the trace preserved to
  1e-5 under the Lindblad evolution; CHSH ≈ 2√2 quantum, ≤ 2 classical.
- **I** — the Aharonov–Bohm phase = qΦ/ℏ to 1 %; the 3D harmonic
  oscillator's ground state to 1e-3.
- **J** — the SSH model's edge states at the topological boundary; the
  parameter-shift gradient to 1e-4; the transmission coefficient.

## The commit template

Each stage ships as one to three revertible commits, each green:

```
feat(quantum): the complex layer and the seeded RNG        -> M01-M08
feat(quantum): states, operators and the Hermitian solver   -> M09-M16
feat(quantum): the 1D stationary Schrodinger equation        -> M17-M26
...
```

The per-commit gate, unchanged from the rest of the repo:

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features phieng/phase-timing -- -D warnings
cargo doc --workspace --no-deps
cargo test --workspace --lib --tests          (Debug and Release)
cargo +1.78 build --workspace --lib
dotnet test PhiEng.slnx -c Debug  &&  -c Release
python tools/check_test_parity.py              -> PARITY OK
cargo run -p phieng --example bench_large --release   -> the pinned hashes unmoved
dotnet run --project examples/host/... -- --smoke --steps 600  + the five samples
wsl linux_gate.sh
```

Plus, per stage: the CHANGELOG, the README, LIMITATIONS,
`API_STABILITY.md` (new exports are a MINOR bump), this file, the
parity checker's ID list, and the native DLL restaged.

## The progress

- [x] **Stage A** — the complex layer and the seeded random generator.
      `crates/math/src/complex.rs`, `crates/math/src/rng.rs`, mirrored by
      `src/PhiEng.Math/Complex32.cs` and `src/PhiEng.Math/DeterministicRng.cs`.
      Pinned by M01–M08.
- [ ] **Stage B** — states, operators, matrices, the Hermitian solver.
- [ ] **Stage C** — the stationary 1D Schrödinger equation.
- [ ] **Stage D** — the time evolution, the FFI, snapshot v15.
- [ ] **Stage E** — the gates and circuits.
- [ ] **Stage F** — the game-facing layer.
- [ ] **Stage G** — spin and particles.
- [ ] **Stage H** — entanglement and information.
- [ ] **Stage I** — the 3D grid and the fields.
- [ ] **Stage J** — the advanced.

## The crate layout

```
crates/quantum/src/
  lib.rs
  state.rs        QuantumState: the norms, the Born probabilities, the expectation values
  operator.rs     the QuantumOperator trait: matrix-free, so the PDE Hamiltonian is never materialised
  matrix.rs       ComplexMatrix: the dense operators, the dagger, the tensor product
  linalg.rs       the Hermitian Jacobi solver, the power iteration, the tridiagonal solves
  fft.rs          the deterministic radix-2 FFT
  hamiltonian.rs  the 1D/2D/3D discrete Hamiltonians and the potentials
  evolution.rs    the Crank-Nicolson and the split operator
  spin.rs         the Pauli algebra and the spinors
  fields.rs       the density matrices, the channels, the Lindblad master equation
  gates.rs        the single- and multi-qubit gates, the Bloch sphere
  circuit.rs      the circuits, the measurement, the algorithms
  sampling.rs     the Born-rule sampling

src/PhiEng.Quantum/   the managed mirror, file for file
```

The complex type and the generator live in `phieng-math`, not in the
quantum crate: they are general-purpose primitives, they belong beside
the deterministic trigonometry, and every later crate may want them.
