//! The separable three-dimensional Hamiltonian.
//!
//! A particle in a box, a harmonic trap, or any potential that splits
//! as `V(x, y, z) = Vx(x) + Vy(y) + Vz(z)` separates: its eigenstates
//! are tensor products of one-dimensional eigenstates, and its
//! energies are sums. This module never builds the full matrix for
//! that case — it solves three small one-dimensional problems and
//! combines them — and it applies the operator as three one-dimensional
//! stencils for the general one.
//!
//! Everything here assumes the Dirichlet condition on all three axes:
//! the ends are held at zero, so the unknowns are the interior points
//! and every spectrum below comes from the tridiagonal solvers.

use phieng_math::complex::Complex32;

use crate::grid::Grid1D;
use crate::grid3::Grid3D;
use crate::hamiltonian::{Boundary1D, Hamiltonian1D};
use crate::linalg::{lowest_eigenvector, tridiagonal_lowest_eigenvalues};
use crate::potential::Potential1D;
use crate::state::QuantumState;

/// A separable Hamiltonian on a [`Grid3D`]: three one-dimensional
/// Hamiltonians plus an optional non-separable remainder sampled on
/// the full grid.
#[derive(Clone, Debug)]
pub struct Hamiltonian3D {
    grid: Grid3D,
    hx: Hamiltonian1D,
    hy: Hamiltonian1D,
    hz: Hamiltonian1D,
    nonseparable: Vec<f32>,
}

impl Hamiltonian3D {
    /// A separable Hamiltonian from three one-dimensional problems.
    ///
    /// Returns `None` when the pieces do not fit: the grids must match
    /// the axes, the spacings must agree exactly, and all three
    /// conditions must be Dirichlet.
    pub fn separable(
        grid: Grid3D,
        hx: Hamiltonian1D,
        hy: Hamiltonian1D,
        hz: Hamiltonian1D,
    ) -> Option<Self> {
        let (nx, ny, nz) = grid.shape();
        let (dx, dy, dz) = grid.spacings();
        let fits = |h: &Hamiltonian1D, n: usize, dx: f32| {
            h.grid().n() == n && h.grid().dx() == dx && h.boundary() == Boundary1D::Dirichlet
        };
        if !fits(&hx, nx, dx) || !fits(&hy, ny, dy) || !fits(&hz, nz, dz) {
            return None;
        }
        let n = grid.n();
        Some(Self {
            grid,
            hx,
            hy,
            hz,
            nonseparable: vec![0.0; n],
        })
    }

    /// Adds a non-separable remainder, sampled on the full grid in
    /// x-fastest order.
    ///
    /// Returns false and changes nothing when the slice is short.
    pub fn with_nonseparable(&mut self, values: &[f32]) -> bool {
        if values.len() < self.grid.n() {
            return false;
        }
        self.nonseparable[..self.grid.n()].copy_from_slice(&values[..self.grid.n()]);
        true
    }

    /// The grid.
    pub fn grid(&self) -> &Grid3D {
        &self.grid
    }

    /// The number of unknowns: the interior points.
    pub fn unknowns(&self) -> usize {
        self.grid.dirichlet_unknowns()
    }

    /// The lowest `count` energies: sums of one-dimensional energies,
    /// lowest first.
    ///
    /// Each axis contributes its lowest few levels (`count + 2`, at
    /// least six); every combination is an energy of the separable
    /// problem, and the lowest `count` of them are returned. The
    /// non-separable remainder is not included: it shifts the true
    /// spectrum, and the dense cross-check below is how that shift is
    /// measured.
    pub fn separable_energies(&self, count: usize) -> Vec<f32> {
        // How many one-dimensional levels feed the product sums: the
        // `count` lowest sums use axis indices below `count`, because the
        // `count` candidates (i,0,0) cap the count-th sum from above while
        // every sum with an index past `count - 1` lies strictly above
        // that cap (one-dimensional bound states do not degenerate). So
        // `count + 2` levels per axis provably contain them all; the floor
        // of 6 keeps small requests efficient and the ceiling is the
        // number of unknowns on the axis.
        let take = (count + 2).max(6).min(self.hx.unknowns().max(1));
        let ex = tridiagonal_lowest_eigenvalues(&self.hx.diagonal(), &self.hx.off_diagonal(), take);
        let ey = tridiagonal_lowest_eigenvalues(&self.hy.diagonal(), &self.hy.off_diagonal(), take);
        let ez = tridiagonal_lowest_eigenvalues(&self.hz.diagonal(), &self.hz.off_diagonal(), take);
        let mut sums = Vec::with_capacity(ex.len() * ey.len() * ez.len());
        for x in ex.iter() {
            for y in ey.iter() {
                for z in ez.iter() {
                    // Add the three axis energies smallest-first: f32
                    // addition is commutative but not associative, so the
                    // permutations of one degenerate level (say (1,0,0),
                    // (0,1,0), (0,0,1)) would otherwise land on different
                    // last-bit roundings. Sorting the three summands makes
                    // every permutation of the same multiset bit-identical,
                    // which is what the physical permutation symmetry
                    // demands. The C# mirror sorts the same way.
                    let mut parts = [*x, *y, *z];
                    parts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    sums.push((parts[0] + parts[1]) + parts[2]);
                }
            }
        }
        sums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sums.truncate(count);
        sums
    }

    /// The ground state of the separable problem: the tensor product of
    /// the three one-dimensional ground states.
    ///
    /// The unknown `u` at interior point `(ix, iy, iz)` carries
    /// `psix[ix-1] * psiy[iy-1] * psiz[iz-1]`.
    pub fn separable_ground_state(&self) -> (f32, QuantumState) {
        let ground = |h: &Hamiltonian1D| {
            let e0 = h.lowest_eigenvalues(1).first().copied().unwrap_or(0.0);
            let state = lowest_eigenvector(&h.diagonal(), &h.off_diagonal(), e0, 50)
                .unwrap_or_else(|| {
                    let mut s = QuantumState::zeros(h.unknowns());
                    for a in s.amplitudes_mut().iter_mut() {
                        *a = Complex32::one();
                    }
                    s.normalize();
                    s
                });
            (e0, state)
        };
        let (ex, sx) = ground(&self.hx);
        let (ey, sy) = ground(&self.hy);
        let (ez, sz) = ground(&self.hz);
        let n = self.unknowns();
        let mut amps = vec![Complex32::zero(); n];
        for (u, amp) in amps.iter_mut().enumerate() {
            let Some((ix, iy, iz)) = self.grid.unknown_to_grid(u) else {
                continue;
            };
            *amp = sx.amplitudes()[ix - 1] * sy.amplitudes()[iy - 1] * sz.amplitudes()[iz - 1];
        }
        let mut state = QuantumState::from_amplitudes(amps);
        state.normalize();
        (ex + ey + ez, state)
    }

    /// Applies the operator to a state, into a separate buffer.
    ///
    /// Three one-dimensional stencils plus the sampled potentials, all
    /// in place conceptually: the ends read as zero. Returns false for
    /// a short buffer.
    pub fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]) -> bool {
        let n = self.unknowns();
        if psi.len() < n || out.len() < n {
            return false;
        }
        let (nx, ny, nz) = self.grid.shape();
        let (mx, my, mz) = (nx - 2, ny - 2, nz - 2);
        // The kinetic coefficients already carry their spacings.
        let cx = self.hx.kinetic_coefficient();
        let cy = self.hy.kinetic_coefficient();
        let cz = self.hz.kinetic_coefficient();
        let vx = self.hx.potential();
        let vy = self.hy.potential();
        let vz = self.hz.potential();
        for u in 0..n {
            let ix = u % mx;
            let iy = (u / mx) % my;
            let iz = u / (mx * my);
            // The x stencil: neighbours left and right, zero past the ends.
            let xm = if ix == 0 {
                Complex32::zero()
            } else {
                psi[u - 1]
            };
            let xp = if ix + 1 >= mx {
                Complex32::zero()
            } else {
                psi[u + 1]
            };
            // The y stencil: a row stride away.
            let ym = if iy == 0 {
                Complex32::zero()
            } else {
                psi[u - mx]
            };
            let yp = if iy + 1 >= my {
                Complex32::zero()
            } else {
                psi[u + mx]
            };
            // The z stencil: a plane stride away.
            let mxy = mx * my;
            let zm = if iz == 0 {
                Complex32::zero()
            } else {
                psi[u - mxy]
            };
            let zp = if iz + 1 >= mz {
                Complex32::zero()
            } else {
                psi[u + mxy]
            };
            let centre = psi[u];
            let lap = (xm - centre.scale(2.0) + xp).scale(cx)
                + (ym - centre.scale(2.0) + yp).scale(cy)
                + (zm - centre.scale(2.0) + zp).scale(cz);
            // The grid point of this unknown is one past it on every axis.
            let gi = (ix + 1) + nx * ((iy + 1) + ny * (iz + 1));
            let potential = vx[ix + 1]
                + vy[iy + 1]
                + vz[iz + 1]
                + self.nonseparable[gi.min(self.nonseparable.len() - 1)];
            out[u] = centre.scale(potential) - lap;
        }
        true
    }

    /// The dense matrix of the operator, for small grids.
    ///
    /// Column `j` is the operator applied to the `j`-th basis state:
    /// exact, and exactly as expensive as it sounds. Beyond a few
    /// points per axis the matrix-free [`Hamiltonian3D::apply_to`] is
    /// the only sane form.
    pub fn dense_matrix(&self) -> crate::matrix::ComplexMatrix {
        let n = self.unknowns();
        let mut m = crate::matrix::ComplexMatrix::zeros(n, n);
        if n == 0 {
            return m;
        }
        let mut basis = vec![Complex32::zero(); n];
        let mut column = vec![Complex32::zero(); n];
        // Index loop: each pass reuses the same two buffers, so there
        // is no second slice to zip with.
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            for v in basis.iter_mut() {
                *v = Complex32::zero();
            }
            basis[j] = Complex32::one();
            if !self.apply_to(&basis, &mut column) {
                continue;
            }
            for (i, value) in column.iter().enumerate() {
                m.set(i, j, *value);
            }
        }
        m
    }

    /// A 1D grid along the x axis, for tests and tools.
    pub fn axis_grid_x(n: usize, min: f32, max: f32) -> Grid1D {
        Grid1D::new(n, min, max)
    }
}

/// The one-dimensional harmonic trap, for tests and tools.
pub fn harmonic_1d(n: usize, min: f32, max: f32, omega: f32) -> Hamiltonian1D {
    Hamiltonian1D::new(
        Grid1D::new(n, min, max),
        Potential1D::Harmonic { omega },
        1.0,
        1.0,
    )
}
