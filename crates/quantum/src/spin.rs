//! Spin-1/2: the Pauli algebra, the Stern-Gerlach measurement, and
//! precession in a magnetic field.
//!
//! A spin-1/2 particle lives in a two-dimensional Hilbert space, so a
//! spinor *is* a single-qubit state and everything in
//! [`crate::gates`] applies. This module gives the spin vocabulary its
//! own home: the Pauli matrices as observables, the Stern-Gerlach
//! measurement of `S_z`, the Zeeman Hamiltonian of a uniform field, and
//! the Rabi flop it drives.
//!
//! Natural units throughout (`hbar = 1`): energies and frequencies
//! share the same numbers. The spatial part of the Pauli equation —
//! the spinor wavefunction with minimal coupling — lives with the
//! fields in Stage I; what is here is the spin degree of freedom
//! alone.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::gates;
use crate::matrix::ComplexMatrix;
use crate::state::QuantumState;

/// The Pauli X matrix: `S_x` up to the factor `hbar / 2`.
pub fn pauli_x() -> ComplexMatrix {
    gates::x_gate()
}

/// The Pauli Y matrix: `S_y` up to the factor `hbar / 2`.
pub fn pauli_y() -> ComplexMatrix {
    gates::y_gate()
}

/// The Pauli Z matrix: `S_z` up to the factor `hbar / 2`.
pub fn pauli_z() -> ComplexMatrix {
    gates::z_gate()
}

/// The spin-up state `|up>`: the +1 eigenstate of `S_z`.
pub fn spin_up() -> QuantumState {
    QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()])
}

/// The spin-down state `|down>`: the -1 eigenstate of `S_z`.
pub fn spin_down() -> QuantumState {
    QuantumState::from_amplitudes(vec![Complex32::zero(), Complex32::one()])
}

/// Measures `S_z` of a spinor: true for up, false for down.
///
/// The outcome is drawn from the Born rule with the given stream, and
/// the spinor collapses onto the measured branch. Returns `None` when
/// the state is not one spin.
pub fn measure_sz(state: &mut QuantumState, rng: &mut DeterministicRng) -> Option<bool> {
    if state.dim() != 2 {
        return None;
    }
    // The register reports true for |1>; spin up is |0>.
    gates::measure(state, 0, rng).map(|one| !one)
}

/// The probability of measuring spin up.
pub fn up_probability(state: &QuantumState) -> f32 {
    if state.dim() != 2 {
        return 0.0;
    }
    state.amplitudes()[0].norm_sqr()
}

/// The Zeeman Hamiltonian of a uniform field: `H = -mu (B . sigma) / 2`.
///
/// With the field along z the energies are `∓mu B / 2`: the splitting
/// is the Larmor frequency itself in natural units.
pub fn zeeman_hamiltonian(field: [f32; 3], moment: f32) -> ComplexMatrix {
    let mut h = ComplexMatrix::zeros(2, 2);
    let terms = [
        (gates::x_gate(), field[0]),
        (gates::y_gate(), field[1]),
        (gates::z_gate(), field[2]),
    ];
    for (pauli, component) in terms {
        let scaled = pauli.scale(-moment * component * 0.5);
        for r in 0..2 {
            for c in 0..2 {
                h.set(r, c, h.get(r, c) + scaled.get(r, c));
            }
        }
    }
    h
}

/// Evolves a spinor under a transverse field: the Rabi flop.
///
/// The Hamiltonian is `(omega / 2) sigma_x`, so the exact propagator is
/// the X rotation by `omega * t`: no stepper, no error, just the gate.
/// Starting from spin up, the down probability is `sin²(omega t / 2)` —
/// a pi pulse (`t = pi / omega`) flips the spin all the way over.
pub fn rabi_evolve(state: &mut QuantumState, omega: f32, t: f32) -> bool {
    if state.dim() != 2 {
        return false;
    }
    gates::apply_single(state, &gates::rx_gate(omega * t), 0)
}

/// The spin vector `(<S_x>, <S_y>, <S_z>)` of a spinor, in units of
/// `hbar = 1` (so `S_i = sigma_i / 2`).
///
/// Returns `None` when the state is not one spin.
pub fn spin_vector(state: &QuantumState) -> Option<(f32, f32, f32)> {
    let (x, y, z) = gates::bloch(state)?;
    Some((x * 0.5, y * 0.5, z * 0.5))
}
