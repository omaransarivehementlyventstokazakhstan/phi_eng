//! The linear algebra a quantum simulation needs.

use phieng_math::complex::Complex32;
use phieng_math::trig::{atan2, compute_cos_sin};

use crate::matrix::ComplexMatrix;
use crate::operator::QuantumOperator;
use crate::state::QuantumState;

/// The result of a Hermitian eigendecomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianEigen {
    /// The eigenvalues, in ascending order.
    pub eigenvalues: Vec<f32>,
    /// The eigenvectors, one per column, in the same order as the
    /// eigenvalues.
    pub eigenvectors: ComplexMatrix,
}

impl HermitianEigen {
    /// The eigenvector of the `index`-th eigenvalue, normalised.
    pub fn eigenvector(&self, index: usize) -> QuantumState {
        let n = self.eigenvectors.cols();
        let mut amps = Vec::with_capacity(n);
        for r in 0..self.eigenvectors.rows() {
            amps.push(self.eigenvectors.get(r, index));
        }
        let mut s = QuantumState::from_amplitudes(amps);
        s.normalize();
        s
    }
}

/// The Hermitian eigendecomposition of a matrix, by the cyclic Jacobi
/// method.
///
/// The Jacobi method rotates the matrix until it is diagonal, and the
/// rotations are accumulated into the eigenvectors. It is chosen here
/// for two reasons: it is unconditionally stable (no matrix grows), and
/// it is *exactly* reproducible — a fixed sweep order, a fixed pivot
/// rule, no randomness and no iteration count that depends on the
/// data.
///
/// The matrix must be Hermitian; a non-Hermitian matrix is not checked
/// here (the check is O(n²) and the caller usually knows), and the
/// result is whatever the algorithm produces.
///
/// The convergence test is the squared off-diagonal weight: the sweep
/// stops when it falls below `tol` squared, or after `max_sweeps`
/// sweeps. A typical Hermitian matrix converges in five to eight sweeps.
///
/// # Examples
///
/// ```
/// use phieng_quantum::{ComplexMatrix, eigen_hermitian};
///
/// // A matrix with the two eigenvalues 1 and 2.
/// let m = ComplexMatrix::from_diagonal(&[
///     phieng_math::complex::Complex32::new(1.0, 0.0),
///     phieng_math::complex::Complex32::new(2.0, 0.0),
/// ]);
/// let e = eigen_hermitian(&m, 32, 1e-6);
/// assert!((e.eigenvalues[0] - 1.0).abs() < 1e-6);
/// assert!((e.eigenvalues[1] - 2.0).abs() < 1e-6);
/// ```
pub fn eigen_hermitian(a: &ComplexMatrix, max_sweeps: usize, tol: f32) -> HermitianEigen {
    let n = a.rows();
    let mut m = a.clone();
    let mut v = ComplexMatrix::identity(n);
    if n == 0 {
        return HermitianEigen {
            eigenvalues: Vec::new(),
            eigenvectors: v,
        };
    }

    for _ in 0..max_sweeps {
        // The squared weight of the off-diagonal: the measure of how
        // far the matrix is from diagonal.
        let mut off = 0.0f32;
        for p in 0..n {
            for q in p + 1..n {
                off += m.get(p, q).norm_sqr();
            }
        }
        if off <= tol * tol {
            break;
        }

        for p in 0..n {
            for q in p + 1..n {
                let apq = m.get(p, q);
                let r = apq.abs();
                if !r.is_finite() || r <= 0.0 {
                    continue;
                }
                // The phase of the entry and the rotation that zeroes
                // it: tan(2 theta) = 2|a_pq| / (a_qq - a_pp).
                //
                // The rotation is G = [[c, s*conj(phi)], [-s*phi, c]] on
                // the (p, q) plane, so that A' = G^H A G has a zero
                // (p, q) entry. Writing it the other way round (with the
                // signs of s swapped) would need the opposite angle.
                let phase = apq.scale(1.0 / r);
                let app = m.get(p, p).re;
                let aqq = m.get(q, q).re;
                let theta = 0.5 * atan2(2.0 * r, aqq - app);
                let (c, s) = compute_cos_sin(theta);
                let gpp = Complex32::new(c, 0.0);
                let gpq = Complex32::new(s, 0.0) * phase.conj();
                let gqp = Complex32::new(-s, 0.0) * phase;
                let gqq = Complex32::new(c, 0.0);

                // A <- A G: the columns p and q are mixed.
                for k in 0..n {
                    let akp = m.get(k, p);
                    let akq = m.get(k, q);
                    m.set(k, p, gpp * akp + gqp * akq);
                    m.set(k, q, gpq * akp + gqq * akq);
                }
                // A <- G^H A: the rows p and q are mixed. The row of
                // G^H is the conjugate of the *column* of G, so the
                // row p takes gqp's conjugate and the row q takes
                // gpq's.
                for j in 0..n {
                    let apj = m.get(p, j);
                    let aqj = m.get(q, j);
                    m.set(p, j, gpp.conj() * apj + gqp.conj() * aqj);
                    m.set(q, j, gpq.conj() * apj + gqq.conj() * aqj);
                }
                // V <- V G: the same column mixing, on the eigenvectors.
                for k in 0..v.rows() {
                    let vkp = v.get(k, p);
                    let vkq = v.get(k, q);
                    v.set(k, p, gpp * vkp + gqp * vkq);
                    v.set(k, q, gpq * vkp + gqq * vkq);
                }
            }
        }
    }

    // The eigenvalues are the diagonal now; sort them ascending and
    // carry the eigenvectors along.
    let mut order: Vec<usize> = (0..n).collect();
    let diag: Vec<f32> = (0..n).map(|i| m.get(i, i).re).collect();
    order.sort_by(|&a, &b| {
        diag[a]
            .partial_cmp(&diag[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = ComplexMatrix::zeros(n, n);
    for (new_index, &old_index) in order.iter().enumerate() {
        eigenvalues.push(diag[old_index]);
        for r in 0..n {
            eigenvectors.set(r, new_index, v.get(r, old_index));
        }
    }

    HermitianEigen {
        eigenvalues,
        eigenvectors,
    }
}

/// The eigenpair of the largest magnitude, by the power iteration.
///
/// The iteration starts from a fixed vector — a superposition with every
/// amplitude equal — and repeatedly applies the operator and
/// normalises. It converges to the eigenvector of the largest-magnitude
/// eigenvalue, which for the Hamiltonian is the *highest* energy state.
///
/// The convergence test is the change in the eigenvalue between two
/// iterations. The iteration count is capped, so a degenerate spectrum
/// (where the test never passes) returns the best estimate rather than
/// spinning.
pub fn dominant_eigenpair(
    op: &dyn QuantumOperator,
    dim: usize,
    iterations: usize,
    tol: f32,
) -> (f32, QuantumState) {
    let mut state = QuantumState::zeros(dim);
    for a in state.amplitudes_mut().iter_mut() {
        *a = Complex32::new(1.0, 0.0);
    }
    state.normalize();

    let mut last = 0.0f32;
    for _ in 0..iterations {
        op.apply(state.amplitudes_mut());
        state.normalize();
        // The Rayleigh quotient: the expectation of the operator in the
        // current state.
        let value = state.expectation(op).re;
        if (value - last).abs() <= tol {
            return (value, state);
        }
        last = value;
    }
    (last, state)
}
