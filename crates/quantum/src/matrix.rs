//! The dense complex matrix.

use phieng_math::complex::Complex32;

/// A dense complex matrix, stored row by row.
///
/// The dense representation is for the *small* systems: a qubit pair, a
/// gate, a handful of basis states. The operators of a wavefunction on
/// a grid are never dense — they are applied as a stencil through
/// [`crate::operator::QuantumOperator`], because a dense matrix of a
/// thousand points would be a million entries nobody wants.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex32>,
}

impl ComplexMatrix {
    /// A matrix of `rows` by `cols` zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex32::zero(); rows * cols],
        }
    }

    /// The `n` by `n` identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, Complex32::one());
        }
        m
    }

    /// The matrix with `values` on its diagonal.
    pub fn from_diagonal(values: &[Complex32]) -> Self {
        let n = values.len();
        let mut m = Self::zeros(n, n);
        for (i, v) in values.iter().enumerate() {
            m.set(i, i, *v);
        }
        m
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The entry at `(row, col)`.
    ///
    /// Out-of-range indices return zero rather than panicking: a matrix
    /// is data, and a bad index should not take a simulation down.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex32 {
        if row >= self.rows || col >= self.cols {
            return Complex32::zero();
        }
        self.data[row * self.cols + col]
    }

    /// The entry at `(row, col)`.
    ///
    /// Out-of-range indices are ignored.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Complex32) {
        if row >= self.rows || col >= self.cols {
            return;
        }
        self.data[row * self.cols + col] = value;
    }

    /// The matrix's entries, row by row.
    pub fn data(&self) -> &[Complex32] {
        &self.data
    }

    /// The matrix's entries, mutable, row by row.
    pub fn data_mut(&mut self) -> &mut [Complex32] {
        &mut self.data
    }

    /// `self` applied to a vector: `out = self * psi`.
    ///
    /// `out` is written in full; a shorter `out` is a no-op.
    pub fn mat_vec(&self, psi: &[Complex32], out: &mut [Complex32]) {
        if out.len() < self.rows {
            return;
        }
        for (r, out_r) in out.iter_mut().enumerate().take(self.rows) {
            let base = r * self.cols;
            let mut sum = Complex32::zero();
            for (c, psi_c) in psi.iter().enumerate().take(self.cols) {
                sum = sum + self.data[base + c] * *psi_c;
            }
            *out_r = sum;
        }
    }

    /// The matrix product `self * other`.
    ///
    /// The dimensions must agree; a mismatch returns an empty matrix
    /// rather than panicking.
    pub fn mat_mul(&self, other: &ComplexMatrix) -> ComplexMatrix {
        if self.cols != other.rows {
            return ComplexMatrix::zeros(0, 0);
        }
        if other.cols == 0 {
            return ComplexMatrix::zeros(0, 0);
        }
        let mut out = ComplexMatrix::zeros(self.rows, other.cols);
        for (r, out_row) in out.data.chunks_mut(other.cols).enumerate() {
            for (c, out_c) in out_row.iter_mut().enumerate() {
                let mut sum = Complex32::zero();
                for k in 0..self.cols {
                    sum = sum + self.get(r, k) * other.get(k, c);
                }
                *out_c = sum;
            }
        }
        out
    }

    /// The conjugate transpose, `self^dagger`.
    pub fn dagger(&self) -> ComplexMatrix {
        let mut out = ComplexMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c).conj());
            }
        }
        out
    }

    /// The trace.
    pub fn trace(&self) -> Complex32 {
        let n = self.rows.min(self.cols);
        let mut sum = Complex32::zero();
        for i in 0..n {
            sum = sum + self.get(i, i);
        }
        sum
    }

    /// The sum of every entry's squared modulus.
    ///
    /// This is the squared Frobenius norm.
    pub fn frobenius_sqr(&self) -> f32 {
        let mut sum = 0.0f32;
        for v in self.data.iter() {
            sum += v.norm_sqr();
        }
        sum
    }

    /// Whether the matrix is Hermitian, to within `tol`.
    ///
    /// A Hermitian matrix is its own conjugate transpose: the diagonal
    /// is real and the off-diagonal entries come in conjugate pairs.
    pub fn is_hermitian(&self, tol: f32) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                let a = self.get(r, c);
                let b = self.get(c, r).conj();
                if (a.re - b.re).abs() > tol || (a.im - b.im).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// The tensor product, `self (x) other`.
    ///
    /// The dimension of the result is the product of the dimensions.
    pub fn kron(&self, other: &ComplexMatrix) -> ComplexMatrix {
        let mut out = ComplexMatrix::zeros(self.rows * other.rows, self.cols * other.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let a = self.get(r, c);
                for r2 in 0..other.rows {
                    for c2 in 0..other.cols {
                        out.set(
                            r * other.rows + r2,
                            c * other.cols + c2,
                            a * other.get(r2, c2),
                        );
                    }
                }
            }
        }
        out
    }

    /// The sum.
    pub fn add(&self, other: &ComplexMatrix) -> ComplexMatrix {
        if self.rows != other.rows || self.cols != other.cols {
            return ComplexMatrix::zeros(0, 0);
        }
        let mut out = ComplexMatrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] + other.data[i];
        }
        out
    }

    /// The difference.
    pub fn sub(&self, other: &ComplexMatrix) -> ComplexMatrix {
        if self.rows != other.rows || self.cols != other.cols {
            return ComplexMatrix::zeros(0, 0);
        }
        let mut out = ComplexMatrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] - other.data[i];
        }
        out
    }

    /// The product with a real number.
    pub fn scale(&self, s: f32) -> ComplexMatrix {
        let mut out = ComplexMatrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].scale(s);
        }
        out
    }
}
