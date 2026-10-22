//! The quantum field: a wavefunction attached to a world.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::evolution::{crank_nicolson, split_operator, EvolutionMethod, EvolutionScratch};
use crate::grid::Grid1D;
use crate::hamiltonian::{Boundary1D, Hamiltonian1D};
use crate::linalg::lowest_eigenvector;
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

    /// Draws outcomes from the Born rule.
    ///
    /// Each draw is an outcome with the state's probability, so the
    /// distribution of the draws is the probability distribution. The
    /// stream is seeded, so the same seed gives the same draws — a
    /// measurement can be replayed.
    pub fn sample(&mut self, out: &mut [f32]) -> usize {
        let n = self.state.dim().min(out.len());
        // The cumulative probability of the outcomes.
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
        // The draws: the inverse-CDF method. Each draw is the first
        // outcome whose cumulative probability passes it.
        for o in out.iter_mut().take(n) {
            let u = self.rng.next_f32() * total;
            let mut index = 0usize;
            for (i, c) in self.cumulative.iter().enumerate().take(n) {
                if u < *c {
                    index = i;
                    break;
                }
                index = i;
            }
            *o = index as f32;
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
