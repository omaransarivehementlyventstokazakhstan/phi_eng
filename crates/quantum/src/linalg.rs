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
    let mut applied = vec![Complex32::zero(); dim];
    for _ in 0..iterations {
        op.apply_to(state.amplitudes(), &mut applied);
        state.amplitudes_mut().copy_from_slice(&applied[..dim]);
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

/// The number of eigenvalues of a symmetric tridiagonal matrix below `x`.
///
/// This is the Sturm count: the `LDL^T` factorisation of `T - xI` has
/// one negative pivot for each eigenvalue below `x`, and the pivots can
/// be counted without factorising the matrix.
///
/// It is the workhorse of the tridiagonal eigensolver: a bisection on
/// the count finds every eigenvalue of a banded Hamiltonian without ever
/// building a dense matrix.
pub fn sturm_count(diagonal: &[f32], off_diagonal: &[f32], x: f32) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }
    // The pivots of the factorisation span many orders of magnitude:
    // the Hamiltonian of a thousand-point grid is nearly singular, and
    // its pivots run from a billion down to a billionth. In `f32` that
    // overflows and the count turns to noise in the middle of the
    // spectrum, so the count is carried in `f64`.
    //
    // The `f64` is scratch only — the query point is an `f32` and the
    // answer is a count — and the arithmetic is exact and the same in
    // both languages, so the two engines still agree bit for bit.
    let x = x as f64;
    let mut count = 0usize;
    let mut pivot = diagonal[0] as f64 - x;
    if pivot == 0.0 {
        // A zero pivot would divide by zero below; the convention is to
        // treat it as a tiny negative one, which keeps the count
        // monotone. Without this the very first pivot of a spectrum
        // centred on the query point poisons every count.
        pivot = -1e-30;
    }
    if pivot < 0.0 {
        count += 1;
    }
    // Index loop: the recurrence walks the diagonal and the
    // off-diagonal together, one step apart.
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
        if pivot == 0.0 {
            pivot = -1e-30;
        }
        let e = off_diagonal.get(i - 1).copied().unwrap_or(0.0) as f64;
        pivot = diagonal[i] as f64 - x - e * e / pivot;
        if pivot < 0.0 {
            count += 1;
        }
    }
    count
}

/// The eigenvalues of a symmetric tridiagonal matrix, in ascending order.
///
/// The matrix is given by its diagonal `d` and its off-diagonal `e`,
/// where `e[i]` is the entry `H[i][i+1]`. The eigenvalues are found by
/// bisecting the Sturm count: the count jumps by one at each
/// eigenvalue, so a bisection on the count pins every one of them.
///
/// The bisection runs a fixed number of steps, so the answer is
/// reproducible to the last bit: the same matrix gives the same
/// eigenvalues on every platform and in both languages.
///
/// A matrix whose off-diagonal is shorter than `d.len() - 1` is read as
/// if the missing entries were zero.
pub fn tridiagonal_eigenvalues(diagonal: &[f32], off_diagonal: &[f32]) -> Vec<f32> {
    tridiagonal_lowest_eigenvalues(diagonal, off_diagonal, diagonal.len())
}

/// The lowest `count` eigenvalues of a symmetric tridiagonal matrix, in
/// ascending order.
///
/// The same bisection as [`tridiagonal_eigenvalues`], but it stops after
/// the lowest `count` — the ground state needs the first two, and the
/// whole spectrum of a thousand-point Hamiltonian costs a hundred times
/// more.
pub fn tridiagonal_lowest_eigenvalues(
    diagonal: &[f32],
    off_diagonal: &[f32],
    count: usize,
) -> Vec<f32> {
    let n = diagonal.len();
    if n == 0 || count == 0 {
        return Vec::new();
    }
    // Gershgorin's circle bounds the spectrum: every eigenvalue lies
    // within one of the rows' discs.
    let mut lower = f32::INFINITY;
    let mut upper = f32::NEG_INFINITY;
    for (i, d) in diagonal.iter().enumerate() {
        let mut radius = 0.0f32;
        if i > 0 {
            radius += off_diagonal.get(i - 1).copied().unwrap_or(0.0).abs();
        }
        if i + 1 < n {
            radius += off_diagonal.get(i).copied().unwrap_or(0.0).abs();
        }
        lower = lower.min(*d - radius);
        upper = upper.max(*d + radius);
    }
    // The bounds are exclusive: the smallest eigenvalue is *above* the
    // lower bound and the largest is *below* the upper one.
    lower -= 1.0;
    upper += 1.0;

    let mut values = Vec::with_capacity(count.min(n));
    for k in 0..count.min(n) {
        let mut lo = lower;
        let mut hi = upper;
        for _ in 0..100 {
            // Half of each, so the sum cannot overflow.
            let mid = 0.5 * lo + 0.5 * hi;
            if sturm_count(diagonal, off_diagonal, mid) <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        values.push(0.5 * lo + 0.5 * hi);
    }
    values
}

/// The ground state of a Hermitian operator, by the imaginary-time
/// relaxation.
///
/// The state is evolved by `psi <- psi - dtau (H psi - shift psi)` and
/// renormalised. Each energy component is multiplied by
/// `1 - dtau (E_n - shift)`, so the component whose energy is lowest
/// grows the fastest — after enough steps the state *is* the ground
/// state.
///
/// The shift must be an energy **at or above the top of the
/// spectrum**. Then every factor is at least one and none of them
/// changes sign, so the evolution is stable and only the lowest state
/// wins. A shift *inside* the spectrum makes the factors change sign
/// and the relaxation oscillates instead of relaxing — the energy
/// bounces between two values and never settles.
///
/// The convergence test is the change in the energy between two steps.
///
/// Returns the energy and the state.
pub fn ground_state(
    op: &dyn QuantumOperator,
    dim: usize,
    dtau: f32,
    iterations: usize,
    tol: f32,
    shift: f32,
) -> (f32, QuantumState) {
    let mut state = QuantumState::zeros(dim);
    for (i, a) in state.amplitudes_mut().iter_mut().enumerate() {
        // A symmetric bump in the middle: it has an overlap with the
        // even states, and a state with no overlap with the ground
        // state would never find it.
        let u = 2.0 * i as f32 / (dim.max(2) - 1) as f32 - 1.0;
        *a = Complex32::new(1.0 - u.abs(), 0.0);
    }
    state.normalize();

    let mut applied = vec![Complex32::zero(); dim];
    let mut next = vec![Complex32::zero(); dim];
    let mut energy = 0.0f32;
    let mut last_energy = f32::INFINITY;
    for _ in 0..iterations {
        op.apply_to(state.amplitudes(), &mut applied);
        // The Rayleigh quotient: the expectation of the operator.
        let mut e = Complex32::zero();
        for (a, b) in state.amplitudes().iter().zip(applied.iter()) {
            e = e + a.conj() * *b;
        }
        energy = e.re;
        for i in 0..dim {
            next[i] = state.amplitudes()[i]
                - (applied[i] - state.amplitudes()[i].scale(shift)).scale(dtau);
        }
        state.amplitudes_mut().copy_from_slice(&next[..dim]);
        if !state.normalize() {
            break;
        }
        if (energy - last_energy).abs() <= tol {
            break;
        }
        last_energy = energy;
    }
    (energy, state)
}

/// Solves the tridiagonal system `A x = b` by the Thomas algorithm.
///
/// `A` is given by its diagonal `d` and its off-diagonal `e`, where
/// `e[i]` is the entry `A[i][i+1]`. The system is factored in place
/// conceptually: the elimination is a forward sweep and the solution a
/// back substitution.
///
/// The algorithm has no pivoting, so it needs a diagonally dominant
/// matrix — which the Hamiltonian's `H - shift I` is, when the shift
/// is near the bottom of the spectrum.
///
/// Returns false when the system is singular (a zero pivot) or the
/// buffers are too short; `out` is then undefined.
pub fn tridiagonal_solve(
    diagonal: &[f32],
    off_diagonal: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> bool {
    let n = diagonal.len();
    if n == 0 {
        return false;
    }
    let mut multipliers = vec![0.0f32; n];
    tridiagonal_solve_with(diagonal, off_diagonal, rhs, out, &mut multipliers)
}

/// The Thomas algorithm with the caller's scratch for the elimination
/// multipliers.
///
/// The multipliers are the `c'_i` of the classical algorithm: the
/// forward sweep divides the off-diagonal by the running pivot, and the
/// back substitution multiplies by the same numbers. They are kept
/// because they cannot be recovered in the back sweep.
pub fn tridiagonal_solve_with(
    diagonal: &[f32],
    off_diagonal: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    multipliers: &mut [f32],
) -> bool {
    let n = diagonal.len();
    if n == 0 || rhs.len() < n || out.len() < n || multipliers.len() < n {
        return false;
    }
    // The forward sweep, in place in `out`.
    let mut pivot = diagonal[0];
    if pivot == 0.0 || !pivot.is_finite() {
        return false;
    }
    out[0] = rhs[0] / pivot;
    if !out[0].is_finite() {
        return false;
    }
    for i in 1..n {
        let e = off_diagonal.get(i - 1).copied().unwrap_or(0.0);
        // The elimination multiplier and the modified pivot.
        let multiplier = e / pivot;
        multipliers[i] = multiplier;
        pivot = diagonal[i] - e * multiplier;
        if pivot == 0.0 || !pivot.is_finite() {
            return false;
        }
        out[i] = (rhs[i] - e * out[i - 1]) / pivot;
        if !out[i].is_finite() {
            return false;
        }
    }
    // The back substitution.
    for i in (0..n - 1).rev() {
        out[i] -= multipliers[i + 1] * out[i + 1];
    }
    true
}

/// The eigenvector of the lowest eigenvalue of a symmetric tridiagonal
/// matrix, by inverse iteration.
///
/// The method solves `(H - shift I) x = v` repeatedly, with `shift` just
/// below the lowest eigenvalue, and normalises each time. The
/// eigenvector closest to the shift grows the fastest under the
/// iteration, so the state converges to the ground state — and it
/// converges *quadratically*, which is what distinguishes it from the
/// relaxation and the power iteration.
///
/// Those two converge at a rate set by the ratio of the spectral width
/// to the gap, and the width of a discretised Hamiltonian is enormous:
/// the ground state of a thousand-point well would need hundreds of
/// thousands of steps. The inverse iteration needs a handful.
///
/// The shift must be *below* the lowest eigenvalue: exactly on it the
/// matrix is singular and the solve fails.
pub fn lowest_eigenvector(
    diagonal: &[f32],
    off_diagonal: &[f32],
    eigenvalue: f32,
    iterations: usize,
) -> Option<QuantumState> {
    let n = diagonal.len();
    if n == 0 {
        return None;
    }
    // Just below the eigenvalue, so the system is not singular.
    let scale = diagonal
        .iter()
        .fold(0.0f32, |acc, d| acc.max(d.abs()))
        .max(1.0);
    let shift = eigenvalue - 1e-6 * scale;

    let mut shifted_diagonal = diagonal.to_vec();
    for d in shifted_diagonal.iter_mut() {
        *d -= shift;
    }

    // The start: a ramp, so the state has an overlap with the ground
    // state and not only with the odd ones.
    let mut state = QuantumState::zeros(n);
    for (i, a) in state.amplitudes_mut().iter_mut().enumerate() {
        let x = i as f32 / (n.max(2) - 1) as f32;
        *a = Complex32::new(x, 0.0);
    }
    state.normalize();

    let mut rhs = vec![0.0f32; n];
    let mut solution = vec![0.0f32; n];
    for _ in 0..iterations {
        // The right-hand side is the current state: the Hamiltonian is
        // real, so the solve is a real one.
        for (r, amp) in rhs.iter_mut().zip(state.amplitudes().iter()) {
            *r = amp.re;
        }
        if !tridiagonal_solve(&shifted_diagonal, off_diagonal, &rhs, &mut solution) {
            return None;
        }
        for (amp, value) in state.amplitudes_mut().iter_mut().zip(solution.iter()) {
            *amp = Complex32::new(*value, 0.0);
        }
        if !state.normalize() {
            return None;
        }
    }
    Some(state)
}

/// Solves the tridiagonal system `A x = b` by the Thomas algorithm,
/// with a complex diagonal.
///
/// This is the form the time evolution needs: the Crank-Nicolson
/// matrix has a complex diagonal and a real off-diagonal.
///
/// Returns false when the system is singular (a zero pivot) or the
/// buffers are too short; `out` is then undefined.
pub fn tridiagonal_solve_complex(
    diagonal: &[Complex32],
    off_diagonal: &[Complex32],
    rhs: &[Complex32],
    out: &mut [Complex32],
) -> bool {
    let n = diagonal.len();
    if n == 0 {
        return false;
    }
    let mut multipliers = vec![Complex32::zero(); n];
    tridiagonal_solve_complex_with(diagonal, off_diagonal, rhs, out, &mut multipliers)
}

/// The complex Thomas algorithm with the caller's scratch for the
/// elimination multipliers.
pub fn tridiagonal_solve_complex_with(
    diagonal: &[Complex32],
    off_diagonal: &[Complex32],
    rhs: &[Complex32],
    out: &mut [Complex32],
    multipliers: &mut [Complex32],
) -> bool {
    let n = diagonal.len();
    if n == 0 || rhs.len() < n || out.len() < n || multipliers.len() < n {
        return false;
    }
    // The forward sweep: the elimination multipliers are kept in the
    // scratch.
    let mut pivot = diagonal[0];
    if pivot.is_zero() || !pivot.is_finite() {
        return false;
    }
    out[0] = rhs[0] / pivot;
    if !out[0].is_finite() {
        return false;
    }
    for i in 1..n {
        let e = off_diagonal
            .get(i - 1)
            .copied()
            .unwrap_or(Complex32::zero());
        // The elimination multiplier and the modified pivot.
        let multiplier = e / pivot;
        multipliers[i] = multiplier;
        pivot = diagonal[i] - e * multiplier;
        if pivot.is_zero() || !pivot.is_finite() {
            return false;
        }
        out[i] = (rhs[i] - e * out[i - 1]) / pivot;
        if !out[i].is_finite() {
            return false;
        }
    }
    // The back substitution.
    for i in (0..n - 1).rev() {
        out[i] = out[i] - multipliers[i + 1] * out[i + 1];
    }
    true
}
