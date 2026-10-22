//! The quantum gates: the matrices, their action on a register, the
//! Bloch sphere and the measurement.
//!
//! # The qubit order
//!
//! Qubit 0 is the least significant bit of the basis index
//! (little-endian, the Qiskit convention): the amplitude of the basis
//! state `|q_{n-1} ... q_1 q_0>` lives at `amplitudes[index]` with
//! `index = q_0 + 2 q_1 + ...`. Every function here and in the managed
//! mirror uses this order, so the two agree on which amplitude is
//! which.
//!
//! # How a gate is applied
//!
//! A gate is never embedded in the full `2^n` by `2^n` matrix: the
//! amplitudes that differ only in the gate's qubits are visited in
//! ascending index order and the small matrix is applied to each pair
//! (or quadruple, or octet) in place. The pairs are disjoint, so the
//! in-place update reads both amplitudes before it writes either, and
//! the order is fixed — the two engines agree bit-for-bit.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::matrix::ComplexMatrix;
use crate::state::QuantumState;

/// `1 / sqrt(2)`, the Hadamard normalisation, as exact bits.
///
/// The correctly rounded `f32` value, written as bits so the two
/// engines cannot disagree on how it was computed. A function rather
/// than a constant because `f32::from_bits` is not a `const fn` on the
/// MSRV toolchain.
#[inline]
pub fn inv_sqrt_2() -> f32 {
    f32::from_bits(0x3F3504F3)
}

/// The Hadamard gate, `H = (1/sqrt(2)) [[1, 1], [1, -1]]`.
pub fn h_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    let s = inv_sqrt_2();
    m.set(0, 0, Complex32::new(s, 0.0));
    m.set(0, 1, Complex32::new(s, 0.0));
    m.set(1, 0, Complex32::new(s, 0.0));
    m.set(1, 1, Complex32::new(-s, 0.0));
    m
}

/// The Pauli X gate (the bit flip).
pub fn x_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 1, Complex32::one());
    m.set(1, 0, Complex32::one());
    m
}

/// The Pauli Y gate.
pub fn y_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 1, Complex32::new(0.0, -1.0));
    m.set(1, 0, Complex32::new(0.0, 1.0));
    m
}

/// The Pauli Z gate (the phase flip).
pub fn z_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::one());
    m.set(1, 1, Complex32::new(-1.0, 0.0));
    m
}

/// The S gate (the square root of Z).
pub fn s_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::one());
    m.set(1, 1, Complex32::i());
    m
}

/// The T gate (the fourth root of Z).
pub fn t_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::one());
    m.set(1, 1, Complex32::exp_i(std::f32::consts::FRAC_PI_4));
    m
}

/// The X rotation by `theta`: `RX = cos(t/2) I - i sin(t/2) X`.
pub fn rx_gate(theta: f32) -> ComplexMatrix {
    let (c, s) = phieng_math::trig::compute_cos_sin(theta * 0.5);
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::new(c, 0.0));
    m.set(0, 1, Complex32::new(0.0, -s));
    m.set(1, 0, Complex32::new(0.0, -s));
    m.set(1, 1, Complex32::new(c, 0.0));
    m
}

/// The Y rotation by `theta`: `RY = cos(t/2) I - i sin(t/2) Y`.
pub fn ry_gate(theta: f32) -> ComplexMatrix {
    let (c, s) = phieng_math::trig::compute_cos_sin(theta * 0.5);
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::new(c, 0.0));
    m.set(0, 1, Complex32::new(-s, 0.0));
    m.set(1, 0, Complex32::new(s, 0.0));
    m.set(1, 1, Complex32::new(c, 0.0));
    m
}

/// The Z rotation by `theta`: `diag(e^(-it/2), e^(it/2))`.
pub fn rz_gate(theta: f32) -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::exp_i(-theta * 0.5));
    m.set(1, 1, Complex32::exp_i(theta * 0.5));
    m
}

/// The general single-qubit gate `U3(theta, phi, lambda)`.
pub fn u3_gate(theta: f32, phi: f32, lambda: f32) -> ComplexMatrix {
    let (ct, st) = phieng_math::trig::compute_cos_sin(theta * 0.5);
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::new(ct, 0.0));
    m.set(0, 1, Complex32::exp_i(lambda).scale(-st));
    m.set(1, 0, Complex32::exp_i(phi).scale(st));
    m.set(1, 1, Complex32::exp_i(phi + lambda).scale(ct));
    m
}

/// The controlled-NOT: flips the target where the control is set.
pub fn cnot_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(4, 4);
    m.set(0, 0, Complex32::one());
    m.set(1, 1, Complex32::one());
    m.set(2, 3, Complex32::one());
    m.set(3, 2, Complex32::one());
    m
}

/// The controlled-Z: a phase flip where both qubits are set.
pub fn cz_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::identity(4);
    m.set(3, 3, Complex32::new(-1.0, 0.0));
    m
}

/// The SWAP gate: exchanges the two qubits.
pub fn swap_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(4, 4);
    m.set(0, 0, Complex32::one());
    m.set(1, 2, Complex32::one());
    m.set(2, 1, Complex32::one());
    m.set(3, 3, Complex32::one());
    m
}

/// The Toffoli gate (controlled-controlled-NOT): flips the target where
/// both controls are set.
pub fn toffoli_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::identity(8);
    m.set(6, 6, Complex32::zero());
    m.set(7, 7, Complex32::zero());
    m.set(6, 7, Complex32::one());
    m.set(7, 6, Complex32::one());
    m
}

/// The Fredkin gate (controlled-SWAP): exchanges the targets where the
/// control is set.
pub fn fredkin_gate() -> ComplexMatrix {
    let mut m = ComplexMatrix::identity(8);
    m.set(5, 5, Complex32::zero());
    m.set(6, 6, Complex32::zero());
    m.set(5, 6, Complex32::one());
    m.set(6, 5, Complex32::one());
    m
}

/// Applies a 2 by 2 gate to one qubit of a register.
///
/// The pairs of amplitudes that differ only in `qubit` are visited in
/// ascending index order. Returns false when the qubit is out of range
/// or the gate is not 2 by 2.
pub fn apply_single(state: &mut QuantumState, gate: &ComplexMatrix, qubit: usize) -> bool {
    let dim = state.dim();
    if gate.rows() != 2 || gate.cols() != 2 {
        return false;
    }
    // The register must hold a whole number of qubits: its length is a
    // power of two, and the qubit must be one of them.
    if dim == 0 || dim & (dim - 1) != 0 {
        return false;
    }
    if qubit >= usize::BITS as usize || (1usize << qubit) >= dim {
        return false;
    }
    let mask = 1usize << qubit;
    let g00 = gate.get(0, 0);
    let g01 = gate.get(0, 1);
    let g10 = gate.get(1, 0);
    let g11 = gate.get(1, 1);
    let amps = state.amplitudes_mut();
    for i in 0..dim {
        if i & mask != 0 {
            continue;
        }
        let j = i | mask;
        if j >= dim {
            continue;
        }
        let a = amps[i];
        let b = amps[j];
        amps[i] = g00 * a + g01 * b;
        amps[j] = g10 * a + g11 * b;
    }
    true
}

/// Applies a controlled phase: the amplitudes where both `control` and
/// `target` are set are multiplied by `e^(i theta)`.
///
/// Returns false when either qubit is out of range.
pub fn apply_controlled_phase(
    state: &mut QuantumState,
    control: usize,
    target: usize,
    theta: f32,
) -> bool {
    let dim = state.dim();
    if control >= usize::BITS as usize
        || target >= usize::BITS as usize
        || (1usize << control) >= dim
        || (1usize << target) >= dim
    {
        return false;
    }
    let phase = Complex32::exp_i(theta);
    let mask = (1usize << control) | (1usize << target);
    let amps = state.amplitudes_mut();
    for (i, a) in amps.iter_mut().enumerate() {
        if i & mask == mask {
            *a = *a * phase;
        }
    }
    true
}

/// Applies a CNOT with the given control and target.
///
/// Returns false when either qubit is out of range.
pub fn apply_cnot(state: &mut QuantumState, control: usize, target: usize) -> bool {
    let dim = state.dim();
    if control >= usize::BITS as usize
        || target >= usize::BITS as usize
        || (1usize << control) >= dim
        || (1usize << target) >= dim
        || control == target
    {
        return false;
    }
    let cmask = 1usize << control;
    let tmask = 1usize << target;
    let amps = state.amplitudes_mut();
    for i in 0..dim {
        if i & cmask == 0 || i & tmask != 0 {
            continue;
        }
        let j = i | tmask;
        amps.swap(i, j);
    }
    true
}

/// Applies a Toffoli with two controls and a target.
///
/// Returns false when any qubit is out of range or two coincide.
pub fn apply_toffoli(
    state: &mut QuantumState,
    control_a: usize,
    control_b: usize,
    target: usize,
) -> bool {
    let dim = state.dim();
    if control_a == control_b || control_a == target || control_b == target {
        return false;
    }
    for q in [control_a, control_b, target] {
        if q >= usize::BITS as usize || (1usize << q) >= dim {
            return false;
        }
    }
    let amask = 1usize << control_a;
    let bmask = 1usize << control_b;
    let tmask = 1usize << target;
    let amps = state.amplitudes_mut();
    for i in 0..dim {
        if i & amask == 0 || i & bmask == 0 || i & tmask != 0 {
            continue;
        }
        let j = i | tmask;
        amps.swap(i, j);
    }
    true
}

/// Applies a SWAP of two qubits.
///
/// Returns false when either qubit is out of range or the two coincide.
pub fn apply_swap(state: &mut QuantumState, a: usize, b: usize) -> bool {
    let dim = state.dim();
    if a == b {
        return false;
    }
    for q in [a, b] {
        if q >= usize::BITS as usize || (1usize << q) >= dim {
            return false;
        }
    }
    let amask = 1usize << a;
    let bmask = 1usize << b;
    let amps = state.amplitudes_mut();
    // Exactly one of the two bits set: the pair straddles the swap.
    for i in 0..dim {
        let has_a = i & amask != 0;
        let has_b = i & bmask != 0;
        if has_a == has_b || !has_a {
            continue;
        }
        let j = (i & !amask) | bmask;
        amps.swap(i, j);
    }
    true
}

/// The Bloch-sphere coordinates `(x, y, z)` of a single-qubit state.
///
/// Returns `None` when the state is not one qubit. The coordinates of a
/// pure state satisfy `x² + y² + z² = 1`; a mixed state would lie
/// inside the sphere, which a state vector cannot express.
pub fn bloch(state: &QuantumState) -> Option<(f32, f32, f32)> {
    if state.dim() != 2 {
        return None;
    }
    let a0 = state.amplitudes()[0];
    let a1 = state.amplitudes()[1];
    let cross = a0.conj() * a1;
    Some((
        2.0 * cross.re,
        2.0 * cross.im,
        a0.norm_sqr() - a1.norm_sqr(),
    ))
}

/// Measures one qubit of a register.
///
/// The outcome is drawn from the Born rule with the given stream, and
/// the state collapses onto the measured branch, renormalised. Returns
/// `None` when the qubit is out of range.
pub fn measure(state: &mut QuantumState, qubit: usize, rng: &mut DeterministicRng) -> Option<bool> {
    let dim = state.dim();
    if qubit >= usize::BITS as usize || (1usize << qubit) >= dim {
        return None;
    }
    let mask = 1usize << qubit;
    let mut p0 = 0.0f32;
    for (i, a) in state.amplitudes().iter().enumerate() {
        if i & mask == 0 {
            p0 += a.norm_sqr();
        }
    }
    let outcome = rng.next_f32() >= p0;
    let p = if outcome { 1.0 - p0 } else { p0 };
    if !p.is_finite() || p <= 0.0 {
        return None;
    }
    // Collapse onto the measured branch and renormalise.
    let inv = 1.0 / p.sqrt();
    for (i, a) in state.amplitudes_mut().iter_mut().enumerate() {
        let set = i & mask != 0;
        if set == outcome {
            *a = a.scale(inv);
        } else {
            *a = Complex32::zero();
        }
    }
    Some(outcome)
}
