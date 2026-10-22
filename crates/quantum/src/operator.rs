//! The operator trait.

use phieng_math::complex::Complex32;

use crate::matrix::ComplexMatrix;

/// Something that acts on a quantum state.
///
/// The trait is *matrix-free* on purpose. The Hamiltonian of a
/// wavefunction on a grid is a banded operator, and materialising it as
/// a dense matrix would cost a million entries to compute what a
/// three-point stencil computes in three operations. So the trait asks
/// for one thing: apply me to a vector.
///
/// The application writes into a separate buffer rather than in place.
/// A stencil operator reads its neighbours while it writes, so it
/// cannot work in place without a copy — and a copy that allocates on
/// every call is a per-frame allocation, which this engine does not
/// have. The two-buffer form lets the caller own the scratch.
pub trait QuantumOperator {
    /// The dimension of the space this operator acts on.
    fn dim(&self) -> usize;

    /// Apply the operator to a state, into a separate buffer.
    ///
    /// `out` must be at least [`QuantumOperator::dim`] long; a shorter
    /// buffer is left alone.
    fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]);
}

/// A dense operator: a matrix, applied as a matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseOperator {
    matrix: ComplexMatrix,
}

impl DenseOperator {
    /// The operator of `matrix`.
    pub fn new(matrix: ComplexMatrix) -> Self {
        Self { matrix }
    }

    /// The operator's matrix.
    pub fn matrix(&self) -> &ComplexMatrix {
        &self.matrix
    }
}

impl QuantumOperator for DenseOperator {
    fn dim(&self) -> usize {
        self.matrix.rows()
    }

    fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]) {
        let n = self.matrix.rows().min(psi.len()).min(out.len());
        if n == 0 {
            return;
        }
        self.matrix.mat_vec(&psi[..n], &mut out[..n]);
    }
}

/// A diagonal operator: a vector of eigenvalues.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalOperator {
    values: Vec<Complex32>,
}

impl DiagonalOperator {
    /// The operator with `values` on its diagonal.
    pub fn new(values: Vec<Complex32>) -> Self {
        Self { values }
    }

    /// The operator's diagonal.
    pub fn values(&self) -> &[Complex32] {
        &self.values
    }
}

impl QuantumOperator for DiagonalOperator {
    fn dim(&self) -> usize {
        self.values.len()
    }

    fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]) {
        let n = self.values.len().min(psi.len()).min(out.len());
        for (value, (src, dst)) in self
            .values
            .iter()
            .zip(psi.iter().zip(out.iter_mut()))
            .take(n)
        {
            *dst = *value * *src;
        }
    }
}

/// The identity operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityOperator {
    dim: usize,
}

impl IdentityOperator {
    /// The identity on a space of `dim` dimensions.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl QuantumOperator for IdentityOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]) {
        let n = self.dim.min(psi.len()).min(out.len());
        out[..n].copy_from_slice(&psi[..n]);
    }
}
