//! Tight-binding lattices: bands, gaps and edge modes.
//!
//! A particle that hops between neighbouring sites with amplitude `t`
//! has the band `E(k) = -2t cos(k)`. Alternating the hopping (the SSH
//! chain) opens a gap — and in the topological phase the gap shelters
//! two zero-energy modes bound to the ends. The managed mirror builds
//! the same matrices entry for entry.

use phieng_math::complex::Complex32;

use crate::matrix::ComplexMatrix;

/// A chain of `n` sites with uniform hopping `t`.
///
/// Open ends by default; `periodic` closes the ring. The diagonal is
/// zero: the on-site energy is the zero of energy here.
pub fn tight_binding_chain(n: usize, hopping: f32, periodic: bool) -> ComplexMatrix {
    let n = n.max(2);
    let mut m = ComplexMatrix::zeros(n, n);
    let t = Complex32::new(-hopping, 0.0);
    for i in 0..n - 1 {
        m.set(i, i + 1, t);
        m.set(i + 1, i, t);
    }
    if periodic && n >= 3 {
        m.set(0, n - 1, t);
        m.set(n - 1, 0, t);
    }
    m
}

/// An SSH chain of `cells` unit cells (two sites each): hopping `v`
/// inside the cell, `w` between cells, open ends.
///
/// For `v < w` the chain is topological and shelters two near-zero
/// edge modes, split only by their overlap across the chain; for
/// `v > w` it is an ordinary gapped insulator with no states near
/// zero.
pub fn ssh_chain(cells: usize, v: f32, w: f32) -> ComplexMatrix {
    let cells = cells.max(1);
    let n = 2 * cells;
    let mut m = ComplexMatrix::zeros(n, n);
    let a = Complex32::new(-v, 0.0);
    let b = Complex32::new(-w, 0.0);
    for j in 0..cells {
        m.set(2 * j, 2 * j + 1, a);
        m.set(2 * j + 1, 2 * j, a);
    }
    for j in 0..cells - 1 {
        m.set(2 * j + 1, 2 * j + 2, b);
        m.set(2 * j + 2, 2 * j + 1, b);
    }
    m
}
