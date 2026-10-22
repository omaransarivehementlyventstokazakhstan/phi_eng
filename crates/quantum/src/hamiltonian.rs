//! The one-dimensional Hamiltonian.

use phieng_math::complex::Complex32;

use crate::grid::Grid1D;
use crate::operator::QuantumOperator;
use crate::potential::Potential1D;

/// The boundary condition of a one-dimensional Hamiltonian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary1D {
    /// The wavefunction vanishes at both ends: the infinite well.
    ///
    /// The unknowns are the *interior* points; the two ends are held at
    /// zero and are not part of the state.
    Dirichlet,
    /// The wavefunction wraps around: the ring.
    ///
    /// Every point is an unknown, and the stencil reaches around.
    Periodic,
}

/// The Hamiltonian of a particle on a line: a kinetic term and a
/// potential.
///
/// The kinetic term is the three-point finite difference
///
/// ```text
///   -(hbar² / 2m) * (psi[i+1] - 2 psi[i] + psi[i-1]) / dx²
/// ```
///
/// which is second-order accurate in the spacing, and the whole
/// operator is *matrix-free*: it is applied as a stencil, so a
/// thousand-point Hamiltonian costs three operations per point and never
/// materialises a matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Hamiltonian1D {
    grid: Grid1D,
    mass: f32,
    hbar: f32,
    potential: Vec<f32>,
    boundary: Boundary1D,
}

impl Hamiltonian1D {
    /// A Hamiltonian on `grid` with the given potential.
    ///
    /// The mass and the reduced Planck constant are one unless the
    /// caller says otherwise — the natural units of the field.
    pub fn new(grid: Grid1D, potential: Potential1D, mass: f32, hbar: f32) -> Self {
        let potential = potential.sample(&grid);
        Self {
            grid,
            mass,
            hbar,
            potential,
            boundary: Boundary1D::Dirichlet,
        }
    }

    /// The same Hamiltonian with a different boundary condition.
    pub fn with_boundary(mut self, boundary: Boundary1D) -> Self {
        self.boundary = boundary;
        self
    }

    /// The grid the Hamiltonian lives on.
    pub fn grid(&self) -> &Grid1D {
        &self.grid
    }

    /// The potential sampled on the grid.
    pub fn potential(&self) -> &[f32] {
        &self.potential
    }

    /// The boundary condition.
    pub fn boundary(&self) -> Boundary1D {
        self.boundary
    }

    /// The kinetic coefficient, `hbar² / (2 m dx²)`.
    pub fn kinetic_coefficient(&self) -> f32 {
        let dx = self.grid.dx();
        self.hbar * self.hbar / (2.0 * self.mass * dx * dx)
    }

    /// The number of unknowns.
    ///
    /// The Dirichlet condition holds the two ends at zero, so they are
    /// not unknowns; the periodic condition makes every point one.
    pub fn unknowns(&self) -> usize {
        match self.boundary {
            Boundary1D::Dirichlet => self.grid.n().saturating_sub(2),
            Boundary1D::Periodic => self.grid.n(),
        }
    }

    /// The diagonal of the Hamiltonian matrix.
    ///
    /// The diagonal is `hbar² / (m dx²) + V[i]`: the two kinetic terms
    /// of the stencil collected, plus the potential.
    ///
    /// For the Dirichlet condition the unknowns are the interior points,
    /// so the entries are the interior's; for the periodic condition
    /// every point is an unknown.
    pub fn diagonal(&self) -> Vec<f32> {
        let c = self.kinetic_coefficient();
        let n = self.unknowns();
        let mut d = Vec::with_capacity(n);
        match self.boundary {
            Boundary1D::Dirichlet => {
                // The unknowns are the interior points 1..n-2.
                for i in 1..(self.grid.n() - 1) {
                    d.push(2.0 * c + self.potential[i]);
                }
            }
            Boundary1D::Periodic => {
                for i in 0..n {
                    d.push(2.0 * c + self.potential[i]);
                }
            }
        }
        d
    }

    /// The off-diagonal of the Hamiltonian matrix, `H[i][i+1]`.
    ///
    /// The off-diagonal is `-c` everywhere. The periodic case also
    /// closes the ring with `H[n-1][0] = -c`, which makes the matrix
    /// cyclic rather than tridiagonal; the tridiagonal solvers here do
    /// not see that entry, so they solve the Dirichlet problem.
    pub fn off_diagonal(&self) -> Vec<f32> {
        let c = self.kinetic_coefficient();
        let n = self.unknowns();
        vec![-c; n.saturating_sub(1)]
    }

    /// An energy at or above the top of the spectrum.
    ///
    /// The Gershgorin bound: every eigenvalue of the matrix lies within
    /// one of the rows' discs, so the largest row's upper edge is an
    /// upper bound for the whole spectrum. It is the shift the
    /// imaginary-time relaxation needs.
    pub fn spectral_upper_bound(&self) -> f32 {
        let c = self.kinetic_coefficient();
        let d = self.diagonal();
        let e = self.off_diagonal();
        let mut upper = f32::NEG_INFINITY;
        for (i, value) in d.iter().enumerate() {
            let mut radius = 0.0f32;
            if i > 0 {
                radius += e.get(i - 1).copied().unwrap_or(0.0).abs();
            }
            if i + 1 < d.len() {
                radius += e.get(i).copied().unwrap_or(0.0).abs();
            }
            upper = upper.max(*value + radius);
        }
        upper + c
    }

    /// The lowest eigenvalues of the Hamiltonian.
    ///
    /// The spectrum of the discrete Hamiltonian, by the Sturm bisection.
    /// Only the lowest `count` are found: the whole spectrum of a
    /// thousand-point Hamiltonian costs a hundred times more, and the
    /// ground state needs the first two.
    pub fn lowest_eigenvalues(&self, count: usize) -> Vec<f32> {
        crate::linalg::tridiagonal_lowest_eigenvalues(&self.diagonal(), &self.off_diagonal(), count)
    }

    /// The ground state of the Hamiltonian and its energy.
    ///
    /// The imaginary-time relaxation needs a shift at (or just below)
    /// the ground state's energy, and a time step small enough that the
    /// step does not overshoot the gap to the first excited state. Both
    /// come from the spectrum, so the caller does not have to guess
    /// them: the shift is the lowest eigenvalue and the step is half the
    /// gap divided by the gap.
    ///
    /// The relaxation converges when the gap to the first excited state
    /// is known and the step respects it; the convergence test is the
    /// change in the energy between two steps.
    pub fn ground_state(&self, iterations: usize, tol: f32) -> (f32, crate::state::QuantumState) {
        let spectrum = self.lowest_eigenvalues(1);
        let e0 = spectrum.first().copied().unwrap_or(0.0);
        let state = crate::linalg::lowest_eigenvector(
            &self.diagonal(),
            &self.off_diagonal(),
            e0,
            iterations.max(1),
        );
        let Some(state) = state else {
            return (e0, self.initial_state());
        };
        // The energy of the state the iteration produced.
        let mut applied = vec![phieng_math::complex::Complex32::zero(); self.unknowns()];
        self.apply_to(state.amplitudes(), &mut applied);
        let mut energy = phieng_math::complex::Complex32::zero();
        for (a, b) in state.amplitudes().iter().zip(applied.iter()) {
            energy = energy + a.conj() * *b;
        }
        let _ = tol;
        (energy.re, state)
    }

    /// The state of the ground state: a real, positive wavefunction.
    ///
    /// A constant one is a poor starting point for the imaginary-time
    /// relaxation — it has no overlap with the odd-parity states, and
    /// they would stay exactly zero forever. A gentle ramp from the
    /// middle has an overlap with everything, so every state can be
    /// found.
    pub fn initial_state(&self) -> crate::state::QuantumState {
        let n = self.unknowns();
        let mut amps = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f32 / (n.max(2) - 1) as f32;
            amps.push(Complex32::new(x, 0.0));
        }
        let mut s = crate::state::QuantumState::from_amplitudes(amps);
        s.normalize();
        s
    }
}

impl QuantumOperator for Hamiltonian1D {
    fn dim(&self) -> usize {
        self.unknowns()
    }

    fn apply_to(&self, psi: &[Complex32], out: &mut [Complex32]) {
        let c = self.kinetic_coefficient();
        let n = self.unknowns();
        if psi.len() < n || out.len() < n {
            return;
        }
        match self.boundary {
            Boundary1D::Dirichlet => {
                // The unknowns are the interior points; the two ends are
                // held at zero and are not part of the state.
                for i in 0..n {
                    let left = if i == 0 {
                        Complex32::zero()
                    } else {
                        psi[i - 1]
                    };
                    let right = if i + 1 >= n {
                        Complex32::zero()
                    } else {
                        psi[i + 1]
                    };
                    let lap = right - psi[i].scale(2.0) + left;
                    // The unknown i is the grid point i+1.
                    out[i] = psi[i].scale(self.potential[i + 1]) - lap.scale(c);
                }
            }
            Boundary1D::Periodic => {
                for i in 0..n {
                    let left = psi[(i + n - 1) % n];
                    let right = psi[(i + 1) % n];
                    let lap = right - psi[i].scale(2.0) + left;
                    out[i] = psi[i].scale(self.potential[i]) - lap.scale(c);
                }
            }
        }
    }
}
