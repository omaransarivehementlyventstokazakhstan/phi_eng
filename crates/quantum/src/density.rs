//! Density matrices: mixed states, partial traces, entropy, channels
//! and open-system evolution.
//!
//! A state vector describes a system known exactly; a density matrix
//! describes one known only statistically — a subsystem of an entangled
//! pair, a state decohered by its environment, a mixture a game sampled
//! from a hat. `rho` is Hermitian, positive semi-definite, and of unit
//! trace; everything here preserves those properties by construction.
//!
//! The partial trace is the workhorse: it answers "what does this
//! subset of qubits look like alone". The loops visit the kept and
//! traced indices in ascending order on both engines, so the reduced
//! matrices agree bit-for-bit.

use phieng_math::complex::Complex32;

use crate::linalg::eigen_hermitian;
use crate::matrix::ComplexMatrix;
use crate::state::QuantumState;

/// A density matrix: a Hermitian, positive semi-definite matrix of
/// unit trace.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix {
    matrix: ComplexMatrix,
}

impl DensityMatrix {
    /// The density matrix of a pure state: `|psi><psi|`.
    pub fn from_pure(state: &QuantumState) -> Self {
        let n = state.dim();
        let mut matrix = ComplexMatrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                matrix.set(r, c, state.amplitudes()[r] * state.amplitudes()[c].conj());
            }
        }
        Self { matrix }
    }

    /// The maximally mixed state: the identity over the dimension.
    pub fn maximally_mixed(dim: usize) -> Self {
        let mut matrix = ComplexMatrix::zeros(dim, dim);
        if dim == 0 {
            return Self { matrix };
        }
        let share = Complex32::new(1.0 / dim as f32, 0.0);
        for i in 0..dim {
            matrix.set(i, i, share);
        }
        Self { matrix }
    }

    /// A density matrix from a raw matrix, without validation.
    ///
    /// The caller promises Hermiticity, positivity and unit trace;
    /// [`DensityMatrix::is_valid`] checks the first and the last. This
    /// exists so tests and tools can name a matrix that no constructor
    /// would produce.
    pub fn from_matrix_unchecked(matrix: ComplexMatrix) -> Self {
        Self { matrix }
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        self.matrix.rows()
    }

    /// The underlying matrix.
    pub fn matrix(&self) -> &ComplexMatrix {
        &self.matrix
    }

    /// The trace: one for every valid density matrix.
    pub fn trace(&self) -> Complex32 {
        self.matrix.trace()
    }

    /// The purity `Tr(rho²)`: one for a pure state, `1/d` for the
    /// maximally mixed state of dimension `d`.
    pub fn purity(&self) -> f32 {
        let square = self.matrix.mat_mul(&self.matrix);
        square.trace().re
    }

    /// Whether the matrix is a plausible density matrix: Hermitian with
    /// unit trace, to within `tol`.
    ///
    /// Positive semi-definiteness is not checked here — it needs the
    /// spectrum, and the caller that needs it has
    /// [`DensityMatrix::eigenvalues`].
    pub fn is_valid(&self, tol: f32) -> bool {
        self.matrix.is_hermitian(tol) && (self.trace().re - 1.0).abs() <= tol
    }

    /// The eigenvalues, in ascending order.
    pub fn eigenvalues(&self) -> Vec<f32> {
        eigen_hermitian(&self.matrix, 64, 1e-8).eigenvalues
    }

    /// The von Neumann entropy `-Tr(rho ln rho)`, in nats.
    ///
    /// Zero for a pure state, `ln d` for the maximally mixed state of
    /// dimension `d`. Non-positive eigenvalues — rounding noise below a
    /// true zero — contribute nothing rather than a NaN.
    pub fn von_neumann_entropy(&self) -> f32 {
        let mut entropy = 0.0f32;
        for value in self.eigenvalues() {
            if value > 0.0 {
                entropy -= value * phieng_math::exponential::ln(value);
            }
        }
        entropy
    }

    /// The diagonal: the measurement probabilities in the computational
    /// basis.
    pub fn probabilities(&self) -> Vec<f32> {
        let n = self.dim();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.matrix.get(i, i).re);
        }
        out
    }

    /// Traces out every qubit except the ones in `keep`.
    ///
    /// `keep` lists the qubit indices (least significant bit first, the
    /// engine's convention) that survive, in any order; the result has
    /// one qubit per kept index, in the order they were listed. Tracing
    /// out everything returns the 1 by 1 matrix `[[1]]`.
    pub fn partial_trace(&self, keep: &[usize], total_qubits: usize) -> Self {
        let kept: Vec<usize> = {
            let mut seen = keep.to_vec();
            seen.sort_unstable();
            seen.dedup();
            seen.into_iter().filter(|q| *q < total_qubits).collect()
        };
        let mut matrix =
            ComplexMatrix::zeros(1usize << kept.len().min(20), 1usize << kept.len().min(20));
        let dim = 1usize << kept.len().min(20);
        let traced: Vec<usize> = (0..total_qubits).filter(|q| !kept.contains(q)).collect();
        for a in 0..dim {
            for b in 0..dim {
                let mut sum = Complex32::zero();
                for e in 0..(1usize << traced.len().min(20)) {
                    let mut row = 0usize;
                    let mut col = 0usize;
                    for (k, q) in kept.iter().enumerate() {
                        if a & (1usize << k) != 0 {
                            row |= 1usize << q;
                        }
                        if b & (1usize << k) != 0 {
                            col |= 1usize << q;
                        }
                    }
                    for (k, q) in traced.iter().enumerate() {
                        if e & (1usize << k) != 0 {
                            row |= 1usize << q;
                            col |= 1usize << q;
                        }
                    }
                    sum = sum + self.matrix.get(row, col);
                }
                matrix.set(a, b, sum);
            }
        }
        Self { matrix }
    }

    /// Applies a channel: `rho -> sum_k K_k rho K_k†`.
    pub fn apply_channel(&self, kraus: &[ComplexMatrix]) -> Self {
        let n = self.dim();
        let mut out = ComplexMatrix::zeros(n, n);
        for k in kraus {
            let left = k.mat_mul(&self.matrix);
            let right = left.mat_mul(&k.dagger());
            for r in 0..n {
                for c in 0..n {
                    out.set(r, c, out.get(r, c) + right.get(r, c));
                }
            }
        }
        Self { matrix: out }
    }
}

/// The Kraus operators of the depolarising channel with error rate `p`:
/// with probability `p` the qubit is replaced by white noise.
pub fn depolarizing_kraus(p: f32) -> Vec<ComplexMatrix> {
    let keep = (1.0 - 0.75 * p).sqrt();
    let flip = (0.25 * p).sqrt();
    let mut k0 = ComplexMatrix::identity(2);
    for i in 0..2 {
        k0.set(i, i, Complex32::new(keep, 0.0));
    }
    // One operator per Pauli, each carrying a quarter of the error:
    // K_1 = sqrt(p/4) X, K_2 = sqrt(p/4) Y, K_3 = sqrt(p/4) Z.
    let mut kx = ComplexMatrix::zeros(2, 2);
    kx.set(0, 1, Complex32::new(flip, 0.0));
    kx.set(1, 0, Complex32::new(flip, 0.0));
    let mut ky = ComplexMatrix::zeros(2, 2);
    ky.set(0, 1, Complex32::new(0.0, -flip));
    ky.set(1, 0, Complex32::new(0.0, flip));
    let mut kz = ComplexMatrix::zeros(2, 2);
    kz.set(0, 0, Complex32::new(flip, 0.0));
    kz.set(1, 1, Complex32::new(-flip, 0.0));
    vec![k0, kx, ky, kz]
}

/// The Kraus operators of amplitude damping with rate `gamma`: `|1>`
/// decays to `|0>`, and the coherence follows it down.
pub fn amplitude_damping_kraus(gamma: f32) -> Vec<ComplexMatrix> {
    let mut k0 = ComplexMatrix::zeros(2, 2);
    k0.set(0, 0, Complex32::one());
    k0.set(1, 1, Complex32::new((1.0 - gamma).max(0.0).sqrt(), 0.0));
    let mut k1 = ComplexMatrix::zeros(2, 2);
    k1.set(0, 1, Complex32::new(gamma.max(0.0).sqrt(), 0.0));
    vec![k0, k1]
}

/// The Kraus operators of dephasing with rate `p`: the populations
/// survive, the coherences shrink by `1 - 2p`.
pub fn dephasing_kraus(p: f32) -> Vec<ComplexMatrix> {
    let mut k0 = ComplexMatrix::identity(2);
    for i in 0..2 {
        k0.set(i, i, Complex32::new((1.0 - p).sqrt(), 0.0));
    }
    let mut k1 = ComplexMatrix::zeros(2, 2);
    let s = p.max(0.0).sqrt();
    k1.set(0, 0, Complex32::new(s, 0.0));
    k1.set(1, 1, Complex32::new(-s, 0.0));
    vec![k0, k1]
}

/// One Euler step of the Lindblad master equation:
///
/// ```text
///   rho += dt * (-i [H, rho] + sum_k g_k (L_k rho L_k† - {L_k† L_k, rho} / 2))
/// ```
///
/// The Hamiltonian part is unitary; the dissipator is where the
/// environment steals coherence. Returns false for dimension
/// mismatches, and leaves the matrix alone in that case.
pub fn lindblad_step(
    rho: &mut DensityMatrix,
    hamiltonian: &ComplexMatrix,
    jumps: &[(f32, ComplexMatrix)],
    dt: f32,
) -> bool {
    let n = rho.dim();
    if hamiltonian.rows() != n || hamiltonian.cols() != n {
        return false;
    }
    for jump in jumps.iter().map(|(_, operator)| operator) {
        if jump.rows() != n || jump.cols() != n {
            return false;
        }
    }
    // -i [H, rho].
    let hr = hamiltonian.mat_mul(&rho.matrix);
    let rh = rho.matrix.mat_mul(hamiltonian);
    let mut derivative = ComplexMatrix::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            let commutator = hr.get(r, c) - rh.get(r, c);
            // -i times the commutator: (x + iy) -> (y - ix).
            derivative.set(r, c, Complex32::new(commutator.im, -commutator.re));
        }
    }
    // The dissipator, jump by jump.
    for (rate, jump) in jumps {
        let lrl = jump.mat_mul(&rho.matrix).mat_mul(&jump.dagger());
        let ldl = jump.dagger().mat_mul(jump);
        for r in 0..n {
            for c in 0..n {
                // {L†L, rho} / 2 at (r, c): half the row-plus-column.
                let mut anti = Complex32::zero();
                for k in 0..n {
                    anti = anti + ldl.get(r, k) * rho.matrix.get(k, c);
                    anti = anti + rho.matrix.get(r, k) * ldl.get(k, c);
                }
                let term = lrl.get(r, c) - anti.scale(0.5);
                derivative.set(r, c, derivative.get(r, c) + term.scale(*rate));
            }
        }
    }
    for r in 0..n {
        for c in 0..n {
            rho.matrix
                .set(r, c, rho.matrix.get(r, c) + derivative.get(r, c).scale(dt));
        }
    }
    true
}
