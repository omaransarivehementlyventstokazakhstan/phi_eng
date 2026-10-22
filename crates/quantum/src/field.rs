//! The quantum field: a wavefunction attached to a world.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::evolution::{crank_nicolson, split_operator, EvolutionMethod, EvolutionScratch};
use crate::grid::Grid1D;
use crate::hamiltonian::{Boundary1D, Hamiltonian1D};
use crate::linalg::lowest_eigenvector;
use crate::madelung::DENSITY_FLOOR;
use crate::operator::QuantumOperator;
use crate::potential::Potential1D;
use crate::state::QuantumState;

/// How a wavefunction is configured.
///
/// The defaults are the ones a world starts with: the field is off,
/// and the grid is a small harmonic trap.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumConfig {
    /// Whether the field evolves at all.
    ///
    /// Off by default: a world that never asks for a wavefunction must
    /// behave exactly as it did before the field existed, down to the
    /// last bit of its trajectory.
    pub enabled: bool,
    /// The number of grid points.
    pub grid_size: usize,
    /// The left end of the grid.
    pub domain_min: f32,
    /// The right end of the grid.
    pub domain_max: f32,
    /// The mass of the particle.
    pub mass: f32,
    /// The reduced Planck constant.
    pub hbar: f32,
    /// The frequency of the harmonic trap the particle sits in.
    pub omega: f32,
    /// How the wavefunction is advanced.
    pub method: EvolutionMethod,
    /// The boundary condition.
    ///
    /// The Crank-Nicolson method needs the Dirichlet condition and the
    /// split operator needs the periodic one; a mismatched pair is
    /// refused rather than silently wrong.
    pub boundary: Boundary1D,
    /// The seed of the measurement stream.
    pub seed: u64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_size: 256,
            domain_min: -10.0,
            domain_max: 10.0,
            mass: 1.0,
            hbar: 1.0,
            omega: 1.0,
            method: EvolutionMethod::CrankNicolson,
            boundary: Boundary1D::Dirichlet,
            seed: 1,
        }
    }
}

/// A wavefunction on a line, attached to a world.
///
/// The field owns the grid, the Hamiltonian, the state and the scratch
/// the evolution needs, so a step allocates nothing.
#[derive(Clone, Debug)]
pub struct QuantumField {
    config: QuantumConfig,
    grid: Grid1D,
    hamiltonian: Hamiltonian1D,
    state: QuantumState,
    scratch: EvolutionScratch,
    rng: DeterministicRng,
    /// The cumulative probability of the measurement stream.
    cumulative: Vec<f32>,
    time: f32,
}

impl Default for QuantumField {
    fn default() -> Self {
        Self::new(QuantumConfig::default())
    }
}

impl QuantumField {
    /// A field with the given configuration.
    ///
    /// The wavefunction starts in the ground state of the trap, so a
    /// field that is stepped without being excited does not move — a
    /// stationary state is the easiest thing to check.
    pub fn new(config: QuantumConfig) -> Self {
        let grid = Grid1D::new(config.grid_size, config.domain_min, config.domain_max);
        let hamiltonian = Hamiltonian1D::new(
            grid.clone(),
            Potential1D::Harmonic {
                omega: config.omega,
            },
            config.mass,
            config.hbar,
        )
        .with_boundary(config.boundary);
        let state = Self::ground_state_of(&hamiltonian);
        let scratch = EvolutionScratch::new(state.dim());
        let rng = DeterministicRng::new(config.seed);
        let cumulative = vec![0.0f32; state.dim()];
        Self {
            config,
            grid,
            hamiltonian,
            state,
            scratch,
            rng,
            cumulative,
            time: 0.0,
        }
    }

    /// The ground state of a Hamiltonian, or the ramp if the solver
    /// cannot find one.
    fn ground_state_of(h: &Hamiltonian1D) -> QuantumState {
        let eigenvalue = h.lowest_eigenvalues(1).first().copied().unwrap_or(0.0);
        lowest_eigenvector(&h.diagonal(), &h.off_diagonal(), eigenvalue, 50)
            .unwrap_or_else(|| h.initial_state())
    }

    /// The field's configuration.
    pub fn config(&self) -> &QuantumConfig {
        &self.config
    }

    /// Whether the field is on.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// The grid the wavefunction lives on.
    pub fn grid(&self) -> &Grid1D {
        &self.grid
    }

    /// The wavefunction.
    pub fn state(&self) -> &QuantumState {
        &self.state
    }

    /// The wavefunction, mutable: a caller may put a state of its own
    /// in the field.
    pub fn state_mut(&mut self) -> &mut QuantumState {
        &mut self.state
    }

    /// The time the wavefunction has been evolved for.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Sets the field's clock.
    ///
    /// The clock is structural state — it tells how long the
    /// wavefunction has been evolved — so the snapshot loader restores
    /// it exactly rather than resetting it.
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// The measurement stream's seed.
    pub fn seed(&self) -> u64 {
        self.config.seed
    }

    /// Replaces the configuration and rebuilds the field.
    ///
    /// The wavefunction starts again in the ground state: a new grid is
    /// a new wavefunction.
    pub fn configure(&mut self, config: QuantumConfig) {
        *self = Self::new(config);
    }

    /// Draws outcomes from the Born rule.
    ///
    /// Each draw is an outcome with the state's probability, so the
    /// distribution of the draws is the probability distribution. The
    /// stream is seeded, so the same seed gives the same draws — a
    /// measurement can be replayed.
    pub fn sample(&mut self, out: &mut [f32]) -> usize {
        let n = self.state.dim().min(out.len());
        for o in out.iter_mut().take(n) {
            match self.draw_index() {
                Some(index) => *o = index as f32,
                None => return 0,
            }
        }
        n
    }

    /// Draws one outcome index from the Born rule.
    ///
    /// Returns `None` when the state is empty or carries no probability.
    /// The cumulative distribution lives in the field's scratch, so a
    /// draw allocates nothing.
    fn draw_index(&mut self) -> Option<usize> {
        let n = self.state.dim();
        if n == 0 || self.cumulative.len() < n {
            return None;
        }
        let mut total = 0.0f32;
        for (c, a) in self
            .cumulative
            .iter_mut()
            .zip(self.state.amplitudes().iter())
            .take(n)
        {
            total += a.norm_sqr();
            *c = total;
        }
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let u = self.rng.next_f32() * total;
        let mut index = 0usize;
        for (i, c) in self.cumulative.iter().enumerate().take(n) {
            if u < *c {
                index = i;
                break;
            }
            index = i;
        }
        Some(index)
    }

    /// Draws a position on the grid from the Born rule.
    ///
    /// The outcome index becomes the grid point it names: for the
    /// Dirichlet condition unknown `i` is grid point `i + 1` (the ends
    /// are held at zero and never drawn), for the periodic condition it
    /// is grid point `i`. Returns `None` when there is nothing to draw.
    pub fn sample_position(&mut self) -> Option<f32> {
        let index = self.draw_index()?;
        Some(self.grid_position(index))
    }

    /// The grid position an unknown index names.
    fn grid_position(&self, index: usize) -> f32 {
        match self.config.boundary {
            Boundary1D::Dirichlet => self.grid.x_at(index + 1),
            Boundary1D::Periodic => self.grid.x_at(index),
        }
    }

    /// The unknown index nearest to a position, clamped to the state.
    fn nearest_index(&self, x: f32) -> Option<usize> {
        let n = self.state.dim();
        if n == 0 || !x.is_finite() {
            return None;
        }
        let dx = self.grid.dx();
        if !dx.is_finite() || dx <= 0.0 {
            return None;
        }
        let q = (x - self.grid.x_min()) / dx;
        // A quotient this large is beyond any grid anyone will build;
        // clamping keeps the float-to-integer cast below in range, where
        // the two engines disagree on out-of-range casts.
        let p = (q.clamp(-9.0e15, 9.0e15) + 0.5).floor() as i64;
        let point = match self.config.boundary {
            // The unknowns are the interior points 1..=n.
            Boundary1D::Dirichlet => p.clamp(1, n as i64) as usize - 1,
            Boundary1D::Periodic => p.rem_euclid(n as i64) as usize,
        };
        Some(point.min(n - 1))
    }

    /// The Bohmian velocity at a position: `v = J / rho` with the
    /// probability current `J = (hbar / m) Im(psi* dpsi/dx)`.
    ///
    /// A stationary state has no current anywhere, so it reports zero;
    /// a state with a phase gradient reports the flow. Central
    /// differences on the three-point stencil around the nearest grid
    /// point; where the density is at the floor the velocity is zero
    /// rather than infinite.
    pub fn velocity_at(&self, x: f32) -> f32 {
        let Some(i) = self.nearest_index(x) else {
            return 0.0;
        };
        let n = self.state.dim();
        let amps = self.state.amplitudes();
        let at = |k: usize| -> Complex32 {
            match self.config.boundary {
                Boundary1D::Dirichlet => {
                    if k >= n {
                        Complex32::zero()
                    } else {
                        amps[k]
                    }
                }
                Boundary1D::Periodic => amps[k % n],
            }
        };
        let left = if i == 0 {
            match self.config.boundary {
                Boundary1D::Dirichlet => Complex32::zero(),
                Boundary1D::Periodic => at(n - 1),
            }
        } else {
            at(i - 1)
        };
        let centre = at(i);
        let right = if i + 1 >= n {
            match self.config.boundary {
                Boundary1D::Dirichlet => Complex32::zero(),
                Boundary1D::Periodic => at(0),
            }
        } else {
            at(i + 1)
        };
        let rho = centre.norm_sqr();
        if rho <= DENSITY_FLOOR {
            return 0.0;
        }
        let dx = self.grid.dx();
        let deriv = (right - left).scale(1.0 / (2.0 * dx));
        let current =
            self.config.hbar / self.config.mass.max(f32::MIN_POSITIVE) * (centre.conj() * deriv).im;
        current / rho
    }

    /// The force a classical particle feels at a position:
    /// `F = -d(V + Q)/dx`, the classical potential plus the quantum
    /// potential.
    ///
    /// The quantum potential is evaluated point by point from the
    /// wavefunction's density, so no buffer is needed; where the
    /// density is at the floor the quantum term is zero. This is the
    /// Bohmian force: a game that integrates it moves bodies the way
    /// the wavefunction pushes.
    pub fn force_at(&self, x: f32) -> f32 {
        let Some(i) = self.nearest_index(x) else {
            return 0.0;
        };
        let n = self.state.dim();
        let dx = self.grid.dx();
        // The density and the potential on the five-point footprint the
        // central difference of Q needs, with the ends held at zero for
        // the Dirichlet condition and wrapped for the periodic one.
        let lo = i.saturating_sub(2);
        let mut rho = [0.0f32; 5];
        let mut pot = [0.0f32; 5];
        for k in 0..5 {
            let (r, v) = self.density_and_potential_at(lo + k, n);
            rho[k] = r;
            pot[k] = v;
        }
        // The quantum potential at the three central points.
        let mut q = [0.0f32; 3];
        for (k, slot) in q.iter_mut().enumerate() {
            *slot = Self::point_quantum_potential(
                &rho[k..k + 3],
                dx,
                self.config.hbar,
                self.config.mass,
            );
        }
        // The classical potential at the three central points.
        let mut v = [0.0f32; 3];
        for (k, slot) in v.iter_mut().enumerate() {
            *slot = self.point_potential(lo + 1 + k, n);
        }
        -((v[2] + q[2]) - (v[0] + q[0])) / (2.0 * dx)
    }

    /// The density and the classical potential at unknown `i`.
    ///
    /// Out-of-range indices are vacuum with the potential of the
    /// nearest end for the Dirichlet condition, and wrap for the
    /// periodic one.
    fn density_and_potential_at(&self, i: usize, n: usize) -> (f32, f32) {
        match self.config.boundary {
            Boundary1D::Dirichlet => {
                if i >= n {
                    (0.0, self.point_potential(n.saturating_sub(1), n))
                } else {
                    (
                        self.state.amplitudes()[i].norm_sqr(),
                        self.point_potential(i, n),
                    )
                }
            }
            Boundary1D::Periodic => {
                let k = i % n;
                (
                    self.state.amplitudes()[k].norm_sqr(),
                    self.point_potential(k, n),
                )
            }
        }
    }

    /// The classical potential at unknown `i`, clamped to the state.
    fn point_potential(&self, i: usize, n: usize) -> f32 {
        let states = self.hamiltonian.potential().len();
        if n == 0 || states == 0 {
            return 0.0;
        }
        let k = i.min(n - 1).min(states - 1);
        match self.config.boundary {
            Boundary1D::Dirichlet => self
                .hamiltonian
                .potential()
                .get(k + 1)
                .copied()
                .unwrap_or(0.0),
            Boundary1D::Periodic => self.hamiltonian.potential().get(k).copied().unwrap_or(0.0),
        }
    }

    /// The quantum potential of three consecutive densities.
    fn point_quantum_potential(s: &[f32], dx: f32, hbar: f32, mass: f32) -> f32 {
        if s.len() < 3 {
            return 0.0;
        }
        let centre = s[1].max(0.0).sqrt();
        if centre <= DENSITY_FLOOR {
            return 0.0;
        }
        let left = s[0].max(0.0).sqrt();
        let right = s[2].max(0.0).sqrt();
        let lap = (left - centre * 2.0 + right) / (dx * dx);
        -(hbar * hbar) / (2.0 * mass.max(f32::MIN_POSITIVE)) * lap / centre
    }

    /// The probability of each outcome: the squared modulus of each
    /// amplitude.
    ///
    /// The probabilities of a normalised state sum to one.
    pub fn probabilities(&self, out: &mut [f32]) -> usize {
        let n = self.state.dim().min(out.len());
        for (o, a) in out.iter_mut().zip(self.state.amplitudes().iter()).take(n) {
            *o = a.norm_sqr();
        }
        n
    }

    /// The density on the grid: the squared modulus of each amplitude.
    ///
    /// The same numbers as [`QuantumField::probabilities`], under the
    /// name a renderer asks for.
    pub fn density(&self, out: &mut [f32]) -> usize {
        self.probabilities(out)
    }

    /// The phase on the grid: the argument of each amplitude, in
    /// `(-pi, pi]`, from the deterministic arctangent.
    ///
    /// A stationary state with a fixed global phase reads flat; the
    /// interference fringes of a superposition read as ramps.
    pub fn phase(&self, out: &mut [f32]) -> usize {
        let n = self.state.dim().min(out.len());
        for (o, a) in out.iter_mut().zip(self.state.amplitudes().iter()).take(n) {
            *o = phieng_math::trig::atan2(a.im, a.re);
        }
        n
    }

    /// Advances the wavefunction by one step.
    ///
    /// Returns false when the configuration and the method disagree —
    /// the Crank-Nicolson method needs the Dirichlet condition and the
    /// split operator the periodic one — in which case the wavefunction
    /// is left alone.
    pub fn step(&mut self, dt: f32) -> bool {
        if !self.config.enabled {
            return true;
        }
        let advanced = match self.config.method {
            EvolutionMethod::CrankNicolson => {
                crank_nicolson(&self.hamiltonian, &mut self.state, dt, &mut self.scratch)
            }
            EvolutionMethod::SplitOperator => {
                split_operator(&self.hamiltonian, &mut self.state, dt, &mut self.scratch)
            }
        };
        if advanced {
            self.time += dt;
        }
        advanced
    }

    /// The energy of the wavefunction: the expectation of the
    /// Hamiltonian.
    pub fn energy(&self) -> f32 {
        let n = self.state.dim();
        let mut applied = vec![Complex32::zero(); n];
        self.hamiltonian
            .apply_to(self.state.amplitudes(), &mut applied);
        let mut sum = Complex32::zero();
        for (a, b) in self.state.amplitudes().iter().zip(applied.iter()) {
            sum = sum + a.conj() * *b;
        }
        sum.re
    }
}
