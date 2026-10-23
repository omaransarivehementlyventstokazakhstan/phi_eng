//! A charged particle on a ring threaded by magnetic flux: the
//! Aharonov-Bohm effect.
//!
//! The flux never touches the particle — the magnetic field vanishes on
//! the ring itself — and still the spectrum moves with it. The vector
//! potential enters the hopping amplitudes as Peierls phases: hopping
//! one way around the ring picks up `e^(-iθ)` per link and the other
//! way `e^(+iθ)`, with `θ = 2π Φ / (N Φ_0)` for `N` links and flux
//! `Φ` in units of the flux quantum `Φ_0`.
//!
//! The energies are `E_m = 2c (1 - cos(2π (m - f) / N))`: the whole
//! spectrum slides with the flux and returns to itself every quantum.
//! The managed mirror builds the same matrix entry for entry.

use phieng_math::complex::Complex32;

use crate::linalg::eigen_hermitian;
use crate::matrix::ComplexMatrix;

/// A particle on a ring of `n` points threaded by magnetic flux.
#[derive(Clone, Debug, PartialEq)]
pub struct RingHamiltonian {
    n: usize,
    hopping: f32,
    flux: f32,
}

impl RingHamiltonian {
    /// A ring of `n` points of radius `radius`, threaded by `flux`
    /// flux quanta.
    ///
    /// The hopping is the finite-difference kinetic coefficient
    /// `hbar² / (2 m dx²)` with `dx = 2 pi R / n`.
    pub fn new(n: usize, radius: f32, mass: f32, hbar: f32, flux: f32) -> Self {
        let n = n.max(3);
        let dx = 2.0 * std::f32::consts::PI * radius / n as f32;
        let hopping = hbar * hbar / (2.0 * mass.max(f32::MIN_POSITIVE) * dx * dx);
        Self { n, hopping, flux }
    }

    /// The number of points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The hopping amplitude.
    pub fn hopping(&self) -> f32 {
        self.hopping
    }

    /// The flux in flux quanta.
    pub fn flux(&self) -> f32 {
        self.flux
    }

    /// The Peierls phase per link: `e^(-i 2 pi f / N)`.
    pub fn link_phase(&self) -> Complex32 {
        let theta = 2.0 * std::f32::consts::PI * self.flux / self.n as f32;
        Complex32::exp_i(-theta)
    }

    /// The Hamiltonian matrix: `2c` on the diagonal and the phased
    /// hopping on the ring, closed around the end.
    pub fn matrix(&self) -> ComplexMatrix {
        let n = self.n;
        let mut m = ComplexMatrix::zeros(n, n);
        let phase = self.link_phase();
        for j in 0..n {
            m.set(j, j, Complex32::new(2.0 * self.hopping, 0.0));
            let next = (j + 1) % n;
            // Hopping forward picks up the phase; hopping back picks
            // up its conjugate, so the matrix stays Hermitian.
            m.set(j, next, phase.scale(-self.hopping));
            m.set(next, j, phase.conj().scale(-self.hopping));
        }
        m
    }

    /// The spectrum, in ascending order.
    pub fn eigenvalues(&self) -> Vec<f32> {
        eigen_hermitian(&self.matrix(), 64, 1e-8).eigenvalues
    }
}
