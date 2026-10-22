//! The time evolution of a wavefunction.
//!
//! # The two methods
//!
//! **Crank-Nicolson** advances the time-dependent Schrodinger
//! equation
//!
//! ```text
//!   (I + i dt H / 2 hbar) psi^(n+1) = (I - i dt H / 2 hbar) psi^n
//! ```
//!
//! which is second-order accurate in the step, unconditionally stable
//! (it never blows up, whatever the step), and conserves the norm
//! exactly — the method is unitary to the rounding. The left-hand side
//! is a tridiagonal system, and the right-hand side is a stencil, so a
//! step costs one tridiagonal solve and two passes over the state.
//!
//! It needs the Dirichlet condition: the tridiagonal solver has no
//! wrap-around.
//!
//! **The split operator** advances
//!
//! ```text
//!   psi^(n+1) = e^(-i T dt / hbar) e^(-i V dt / hbar) psi^n
//! ```
//!
//! in two half-steps, each of which is diagonal in its own basis: the
//! potential one is diagonal in position, the kinetic one in momentum.
//! The momentum step is a Fourier transform, so the method costs one
//! transform per step and is second-order accurate as well. It needs
//! the periodic condition — a transform wraps around.
//!
//! # The scratch
//!
//! Both methods need buffers: the tridiagonal matrix, the right-hand
//! side, the elimination multipliers and the transform's workspace.
//! They are owned by the caller ([`EvolutionScratch`]) and reused, so a
//! step allocates nothing.

use phieng_math::complex::Complex32;
use phieng_math::trig::compute_cos_sin;

use crate::fft;
use crate::hamiltonian::{Boundary1D, Hamiltonian1D};
use crate::operator::QuantumOperator;
use crate::state::QuantumState;

/// How a wavefunction is advanced in time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolutionMethod {
    /// Crank-Nicolson: one tridiagonal solve per step. Needs the
    /// Dirichlet condition.
    CrankNicolson,
    /// The split operator: one Fourier transform per step. Needs the
    /// periodic condition.
    SplitOperator,
}

/// The buffers an evolution needs, so that a step allocates nothing.
#[derive(Clone, Debug, Default)]
pub struct EvolutionScratch {
    /// The diagonal of the Crank-Nicolson matrix.
    diagonal: Vec<Complex32>,
    /// The off-diagonal of the Crank-Nicolson matrix.
    off_diagonal: Vec<Complex32>,
    /// The right-hand side of the Crank-Nicolson system.
    rhs: Vec<Complex32>,
    /// The elimination multipliers of the tridiagonal solve.
    multipliers: Vec<Complex32>,
    /// The transform's workspace.
    workspace: Vec<Complex32>,
    /// The phase of the potential step.
    potential_phase: Vec<Complex32>,
    /// The phase of the kinetic step.
    kinetic_phase: Vec<Complex32>,
}

impl EvolutionScratch {
    /// Scratch for a state of `dim` amplitudes.
    pub fn new(dim: usize) -> Self {
        Self {
            diagonal: vec![Complex32::zero(); dim],
            off_diagonal: vec![Complex32::zero(); dim],
            rhs: vec![Complex32::zero(); dim],
            multipliers: vec![Complex32::zero(); dim],
            workspace: vec![Complex32::zero(); dim],
            potential_phase: vec![Complex32::zero(); dim],
            kinetic_phase: vec![Complex32::zero(); dim],
        }
    }
}

/// Advances a state by one step of the Crank-Nicolson method.
///
/// The state must match the Hamiltonian's dimension. The step is
/// refused (and the state left alone) when the Hamiltonian is periodic
/// — the tridiagonal solver has no wrap-around — or when the
/// dimensions do not match.
pub fn crank_nicolson(
    h: &Hamiltonian1D,
    state: &mut QuantumState,
    dt: f32,
    scratch: &mut EvolutionScratch,
) -> bool {
    if h.boundary() != Boundary1D::Dirichlet {
        return false;
    }
    let n = h.dim();
    if state.dim() != n {
        return false;
    }
    if scratch.diagonal.len() < n {
        *scratch = EvolutionScratch::new(n);
    }
    let hbar = h.hbar();
    if !hbar.is_finite() || hbar <= 0.0 {
        return false;
    }
    // The step's coefficient: i dt / (2 hbar).
    let step = Complex32::new(0.0, dt / (2.0 * hbar));

    // The left-hand side: I + i dt H / (2 hbar).
    let diagonal = h.diagonal();
    let off_diagonal = h.off_diagonal();
    for i in 0..n {
        scratch.diagonal[i] = Complex32::one() + step * diagonal[i];
        scratch.off_diagonal[i] = if i + 1 < n {
            step * off_diagonal[i]
        } else {
            Complex32::zero()
        };
    }

    // The right-hand side: (I - i dt H / (2 hbar)) psi.
    let mut applied = vec![Complex32::zero(); n];
    h.apply_to(state.amplitudes(), &mut applied);
    for (r, (psi, applied_i)) in scratch
        .rhs
        .iter_mut()
        .zip(state.amplitudes().iter().zip(applied.iter()))
        .take(n)
    {
        *r = *psi - step * *applied_i;
    }

    // The solve. The matrix is tridiagonal, and the Thomas algorithm
    // needs no pivoting because the Crank-Nicolson matrix is
    // diagonally dominant.
    let solved = crate::linalg::tridiagonal_solve_complex_with(
        &scratch.diagonal[..n],
        &scratch.off_diagonal[..n],
        &scratch.rhs[..n],
        &mut scratch.workspace[..n],
        &mut scratch.multipliers[..n],
    );
    if !solved {
        return false;
    }
    state
        .amplitudes_mut()
        .copy_from_slice(&scratch.workspace[..n]);
    true
}

/// Advances a state by one step of the split-operator method.
///
/// The state must match the Hamiltonian's dimension, and the grid's
/// length must be a power of two — the transform refuses any other
/// length. The Hamiltonian must be periodic: a transform wraps around.
pub fn split_operator(
    h: &Hamiltonian1D,
    state: &mut QuantumState,
    dt: f32,
    scratch: &mut EvolutionScratch,
) -> bool {
    if h.boundary() != Boundary1D::Periodic {
        return false;
    }
    let n = h.dim();
    if state.dim() != n || !fft::is_power_of_two(n) {
        return false;
    }
    if scratch.workspace.len() < n {
        *scratch = EvolutionScratch::new(n);
    }
    let hbar = h.hbar();
    if !hbar.is_finite() || hbar <= 0.0 {
        return false;
    }
    // The potential step: a phase per point.
    let potential = h.potential();
    for (phase, v) in scratch
        .potential_phase
        .iter_mut()
        .zip(potential.iter())
        .take(n)
    {
        let angle = -dt * *v / hbar;
        *phase = Complex32::exp_i(angle);
    }
    // The kinetic step: a phase per mode. The energy of the mode is the
    // one of the *discrete* kinetic operator, so the two halves of the
    // split are the same operator the Hamiltonian is built from: the
    // discrete energy of the mode j is 4 c sin^2(pi j / N), with c the
    // kinetic coefficient.
    let c = h.kinetic_coefficient();
    for (j, phase) in scratch.kinetic_phase.iter_mut().enumerate().take(n) {
        // The mode's index, folded so that the modes run from -n/2 to
        // n/2 in the order the transform produces them. The sine is
        // the same for j and j-N, so the folding is only for the
        // sign of the mode.
        let folded = if j <= n / 2 {
            j as i64
        } else {
            j as i64 - n as i64
        };
        // The sine comes from the deterministic trigonometry: the
        // platform libm's `sin` is not portable, and this crate promises
        // never to call it.
        let angle = (std::f64::consts::PI * folded as f64 / n as f64) as f32;
        let s = compute_cos_sin(angle).1 as f64;
        let energy = 4.0 * c as f64 * s * s;
        let phase_angle = (-dt as f64 * energy / hbar as f64) as f32;
        *phase = Complex32::exp_i(phase_angle);
    }

    // The potential half-step.
    for i in 0..n {
        state.amplitudes_mut()[i] = state.amplitudes()[i] * scratch.potential_phase[i];
    }
    // The kinetic half-step: into the momentum basis and back.
    if !fft::transform(state.amplitudes(), false, &mut scratch.workspace[..n]) {
        return false;
    }
    for j in 0..n {
        scratch.workspace[j] = scratch.workspace[j] * scratch.kinetic_phase[j];
    }
    if !fft::transform(
        &scratch.workspace[..n],
        true,
        &mut state.amplitudes_mut()[..n],
    ) {
        return false;
    }
    true
}
