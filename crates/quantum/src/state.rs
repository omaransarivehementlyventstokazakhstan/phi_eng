//! The quantum state.

use phieng_math::complex::Complex32;

use crate::operator::QuantumOperator;

/// A state in a finite-dimensional Hilbert space: a vector of complex
/// amplitudes.
///
/// The state is *not* required to be normalised — a state that has just
/// come out of a solver may carry a scale — but everything that reads a
/// probability expects a normalised one. [`QuantumState::normalize`]
/// fixes that, and [`QuantumState::is_normalized`] says whether it is
/// already.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumState {
    dim: usize,
    amps: Vec<Complex32>,
}

impl QuantumState {
    /// The zero state of a space of `dim` dimensions.
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            amps: vec![Complex32::zero(); dim],
        }
    }

    /// The state with the given amplitudes.
    pub fn from_amplitudes(amps: Vec<Complex32>) -> Self {
        let dim = amps.len();
        Self { dim, amps }
    }

    /// The `index`-th basis state: one in that place, zero elsewhere.
    pub fn basis(dim: usize, index: usize) -> Self {
        let mut s = Self::zeros(dim);
        if index < dim {
            s.amps[index] = Complex32::one();
        }
        s
    }

    /// The dimension of the space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The amplitudes.
    pub fn amplitudes(&self) -> &[Complex32] {
        &self.amps
    }

    /// The amplitudes, mutable.
    pub fn amplitudes_mut(&mut self) -> &mut [Complex32] {
        &mut self.amps
    }

    /// The squared norm, `⟨ψ|ψ⟩`.
    pub fn norm_sqr(&self) -> f32 {
        let mut sum = 0.0f32;
        for a in self.amps.iter() {
            sum += a.norm_sqr();
        }
        sum
    }

    /// The norm, `sqrt(⟨ψ|ψ⟩)`.
    pub fn norm(&self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Scale the state to unit norm.
    ///
    /// Returns false and changes nothing when the state has no norm to
    /// scale by — a zero state cannot be normalised.
    pub fn normalize(&mut self) -> bool {
        let n = self.norm();
        if !n.is_finite() || n <= 0.0 {
            return false;
        }
        let inv = 1.0 / n;
        for a in self.amps.iter_mut() {
            *a = a.scale(inv);
        }
        true
    }

    /// Whether the state is normalised, to within `tol`.
    pub fn is_normalized(&self, tol: f32) -> bool {
        (self.norm_sqr() - 1.0).abs() <= tol
    }

    /// The Born probabilities: `|amplitude|²` for each basis state.
    ///
    /// These are the probabilities of measuring each outcome, and they
    /// sum to one for a normalised state.
    pub fn born_probabilities(&self) -> Vec<f32> {
        self.amps.iter().map(|a| a.norm_sqr()).collect()
    }

    /// The expectation value `⟨ψ|A|ψ⟩`.
    pub fn expectation(&self, op: &dyn QuantumOperator) -> Complex32 {
        let n = op.dim().min(self.dim);
        let mut applied = vec![Complex32::zero(); n];
        op.apply_to(&self.amps[..n], &mut applied);
        let mut sum = Complex32::zero();
        for (a, b) in self.amps[..n].iter().zip(applied.iter()) {
            sum = sum + a.conj() * *b;
        }
        sum
    }

    /// The overlap `⟨ψ|φ⟩`.
    pub fn overlap(&self, other: &QuantumState) -> Complex32 {
        let n = self.dim.min(other.dim);
        let mut sum = Complex32::zero();
        for (a, b) in self.amps[..n].iter().zip(other.amps[..n].iter()) {
            sum = sum + a.conj() * *b;
        }
        sum
    }

    /// The fidelity `|⟨ψ|φ⟩|²`, the probability of measuring one state
    /// as the other.
    pub fn fidelity(&self, other: &QuantumState) -> f32 {
        self.overlap(other).norm_sqr()
    }

    /// The phase that aligns this state with `other`.
    ///
    /// Two states that differ only by a global phase are the same
    /// physical state; this returns the factor that turns one into the
    /// other, so two runs can be compared without the phase spoiling
    /// the comparison.
    pub fn phase_to(&self, other: &QuantumState) -> Complex32 {
        let o = other.overlap(self);
        if o.is_zero() {
            return Complex32::one();
        }
        o.inv()
    }
}
