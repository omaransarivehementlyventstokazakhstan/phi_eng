//! Small many-body problems: the Hubbard dimer and the entanglement
//! spectrum.
//!
//! Two sites with repulsion already show what mean-field theory keeps
//! and what it throws away: the restricted Hartree-Fock iteration
//! below is variational (it can never beat the exact answer) and the
//! gap between the two is the correlation energy. The entanglement
//! spectrum is the heart of the density-matrix renormalisation group:
//! it tells which states of half a system matter, and the weight it
//! drops is the truncation error. The managed mirror iterates the
//! same maps in the same order.

use phieng_math::complex::Complex32;

use crate::density::DensityMatrix;
use crate::linalg::eigen_hermitian;
use crate::matrix::ComplexMatrix;
use crate::state::QuantumState;

/// The singlet sector of the Hubbard dimer: two sites, two electrons.
///
/// The basis holds both electrons on site 0, both on site 1, and the
/// singlet split across the sites. Either electron can hop, so each
/// double occupancy couples to the split singlet with `√2 t`. Returns
/// the spectrum in ascending order.
pub fn hubbard_dimer_exact(hopping: f32, u: f32) -> Vec<f32> {
    let t = Complex32::new(-hopping * std::f32::consts::SQRT_2, 0.0);
    let uu = Complex32::new(u, 0.0);
    let mut m = ComplexMatrix::zeros(3, 3);
    m.set(0, 0, uu);
    m.set(1, 1, uu);
    m.set(0, 2, t);
    m.set(2, 0, t);
    m.set(1, 2, t);
    m.set(2, 1, t);
    eigen_hermitian(&m, 32, 1e-8).eigenvalues
}

/// The restricted Hartree-Fock energy of the Hubbard dimer, by
/// self-consistent iteration, with the converged density beside it.
///
/// The Fock matrix in the site basis is `F_ii = U n_i / 2` — each
/// electron feels the opposite spin's density — with `-t` between the
/// sites. Its lowest eigenvector, doubly occupied, is the new density,
/// mixed half-and-half with the old one for stability. The iteration
/// count is fixed, so the result is deterministic to the bit; the
/// energy it returns is variational, never below the exact one.
pub fn hubbard_dimer_rhf(hopping: f32, u: f32, iterations: usize) -> (f32, ComplexMatrix) {
    // The bonding orbital, doubly occupied: uniform unit density.
    let mut p = ComplexMatrix::zeros(2, 2);
    for r in 0..2 {
        for c in 0..2 {
            p.set(r, c, Complex32::one());
        }
    }
    let hop = Complex32::new(-hopping, 0.0);
    for _ in 0..iterations {
        let mut f = ComplexMatrix::zeros(2, 2);
        f.set(0, 0, Complex32::new(u * p.get(0, 0).re / 2.0, 0.0));
        f.set(1, 1, Complex32::new(u * p.get(1, 1).re / 2.0, 0.0));
        f.set(0, 1, hop);
        f.set(1, 0, hop);
        let solved = eigen_hermitian(&f, 32, 1e-8);
        // The new density is the lowest orbital, doubly occupied,
        // mixed half-and-half with the old one.
        let c0 = solved.eigenvectors.get(0, 0);
        let c1 = solved.eigenvectors.get(1, 0);
        let two = Complex32::new(2.0, 0.0);
        let half = Complex32::new(0.5, 0.0);
        let q00 = two * c0 * c0.conj();
        let q01 = two * c0 * c1.conj();
        let q11 = two * c1 * c1.conj();
        p.set(0, 0, half * (p.get(0, 0) + q00));
        p.set(0, 1, half * (p.get(0, 1) + q01));
        p.set(1, 0, half * (p.get(1, 0) + q01.conj()));
        p.set(1, 1, half * (p.get(1, 1) + q11));
    }
    // The energy: the one-body trace plus half the interaction trace,
    // because each pair would otherwise be counted twice.
    let e_one = (p.get(0, 1) + p.get(1, 0)).re * -hopping;
    let e_int = u * (p.get(0, 0).re * p.get(0, 0).re + p.get(1, 1).re * p.get(1, 1).re) / 4.0;
    (e_one + e_int, p)
}

/// A transverse-field Ising chain: ferromagnetic `j` along z, field
/// `h` along x, open ends. `n` qubits, so the matrix is 2^n by 2^n —
/// for the small chains exact diagonalisation can still hold.
pub fn tfi_chain(n: usize, j: f32, h: f32) -> ComplexMatrix {
    use crate::spin::{pauli_x, pauli_z};
    let id = ComplexMatrix::identity(2);
    let mut full = ComplexMatrix::zeros(1 << n.max(1), 1 << n.max(1));
    let n = n.max(1);
    // The bonds: -J Z_i Z_{i+1}.
    for i in 0..n.saturating_sub(1) {
        let mut term = ComplexMatrix::from_diagonal(&[Complex32::one()]);
        for site in 0..n {
            let factor = if site == i || site == i + 1 {
                pauli_z()
            } else {
                id.clone()
            };
            term = term.kron(&factor);
        }
        let scaled = term.scale(-j);
        for r in 0..full.rows() {
            for c in 0..full.cols() {
                full.set(r, c, full.get(r, c) + scaled.get(r, c));
            }
        }
    }
    // The field: -h X_i.
    for i in 0..n {
        let mut term = ComplexMatrix::from_diagonal(&[Complex32::one()]);
        for site in 0..n {
            let factor = if site == i { pauli_x() } else { id.clone() };
            term = term.kron(&factor);
        }
        let scaled = term.scale(-h);
        for r in 0..full.rows() {
            for c in 0..full.cols() {
                full.set(r, c, full.get(r, c) + scaled.get(r, c));
            }
        }
    }
    full
}

/// The entanglement spectrum across a cut: the eigenvalues of the
/// reduced density matrix, largest first.
///
/// This is the truncation step of the density-matrix renormalisation
/// group: the spectrum tells which states of the kept half matter,
/// and the weight it drops is the error of keeping only a few. Here
/// the state is exact and small; DMRG earns its keep by truncating
/// this spectrum for states no exact method can hold.
pub fn entanglement_spectrum(
    state: &QuantumState,
    keep: &[usize],
    total_qubits: usize,
) -> Vec<f32> {
    let rho = DensityMatrix::from_pure(state);
    let mut spectrum = rho.partial_trace(keep, total_qubits).eigenvalues();
    spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    spectrum
}
