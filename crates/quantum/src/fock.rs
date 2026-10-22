//! Second quantization: Fock states, ladder operators, and the Fermi
//! gas.
//!
//! Where a wavefunction says *where* a particle is, the Fock space says
//! *how many* excitations each mode holds. The ladder operators move
//! between those occupancy states: `a†` creates a quantum, `a`
//! destroys one, and the number operator `N = a† a` counts them.
//!
//! Bosons stack without limit (up to the truncation: the matrices here
//! keep the lowest `cutoff` levels); fermions refuse double occupancy,
//! which is the Pauli exclusion principle as a one-line theorem.

use phieng_math::complex::Complex32;

use crate::matrix::ComplexMatrix;

/// The bosonic annihilation operator in a `cutoff`-dimensional Fock
/// space: `a|n> = sqrt(n) |n - 1>`.
pub fn boson_annihilation(cutoff: usize) -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(cutoff, cutoff);
    for n in 1..cutoff {
        m.set(n - 1, n, Complex32::new((n as f32).sqrt(), 0.0));
    }
    m
}

/// The bosonic creation operator: the dagger of
/// [`boson_annihilation`].
pub fn boson_creation(cutoff: usize) -> ComplexMatrix {
    boson_annihilation(cutoff).dagger()
}

/// The number operator `N = a† a`: diagonal with `0, 1, 2, ...`.
pub fn number_operator(cutoff: usize) -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(cutoff, cutoff);
    for n in 0..cutoff {
        m.set(n, n, Complex32::new(n as f32, 0.0));
    }
    m
}

/// The fermionic annihilation operator for one mode, in the `(|0>, |1>)`
/// basis: `a|1> = |0>`, and `a|0> = 0`.
pub fn fermion_annihilation() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 1, Complex32::one());
    m
}

/// The fermionic creation operator: the dagger of
/// [`fermion_annihilation`].
pub fn fermion_creation() -> ComplexMatrix {
    fermion_annihilation().dagger()
}

/// The anticommutator `{a, b} = ab + ba`.
pub fn anticommutator(a: &ComplexMatrix, b: &ComplexMatrix) -> ComplexMatrix {
    a.mat_mul(b).add(&b.mat_mul(a))
}

/// The commutator `[a, b] = ab - ba`.
pub fn commutator(a: &ComplexMatrix, b: &ComplexMatrix) -> ComplexMatrix {
    a.mat_mul(b).sub(&b.mat_mul(a))
}

/// The ground-state energy of spinless fermions filling single-particle
/// levels: at zero temperature the particles occupy the lowest levels
/// first, so the energy is the sum of the `n_particles` smallest
/// entries of `levels`.
///
/// Returns 0.0 when asked for more particles than levels: the extra
/// fermions have nowhere to go, and inventing levels for them would be
/// guessing.
pub fn fermi_gas_energy(levels: &[f32], n_particles: usize) -> f32 {
    if n_particles == 0 || n_particles > levels.len() {
        return 0.0;
    }
    // The levels arrive sorted in every call site that matters; the
    // selection below takes the smallest entries however they arrive.
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.iter().take(n_particles).sum()
}
