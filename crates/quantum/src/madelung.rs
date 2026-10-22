//! The Madelung fluid: the hydrodynamic face of a wavefunction.
//!
//! Writing the wavefunction as an amplitude times a phase,
//! `psi = sqrt(rho) e^(i S / hbar)`, turns the Schrödinger equation
//! into fluid equations for the density `rho` and the velocity
//! `v = (hbar / m) grad S`:
//!
//! ```text
//!   ∂rho/∂t + div(rho v) = 0
//!   ∂v/∂t + (v·grad) v = -grad(V + Q)
//! ```
//!
//! with the **quantum potential** `Q = -(hbar² / 2m) (lap sqrt(rho)) /
//! sqrt(rho)`. The first line is conservation of probability; the
//! second is Euler's equation with one extra force. Turn `hbar` to
//! zero and the extra force vanishes: the fluid goes classical.
//!
//! The fluid here lives on a periodic one-dimensional grid and steps
//! explicitly, so it is a laboratory rather than a production solver:
//! it pins the dispersion relation a game effect would lean on, and the
//! same `quantum_potential` below is what the world's
//! [`crate::field::QuantumField::force_at`] evaluates for rigid
//! bodies. The arithmetic is plain `f32` with IEEE-exact square roots,
//! in a fixed order, so the managed mirror agrees bit-for-bit.

/// The floor under a density before it is square-rooted.
///
/// A density below this is treated as vacuum: its quantum potential is
/// zero rather than infinite. The value is far below any physical
/// density a simulation carries, so it only ever triggers at a true
/// node or outside the wavefunction's support.
pub const DENSITY_FLOOR: f32 = 1e-12;

/// The quantum potential `Q = -(hbar² / 2m) (lap s) / s` with
/// `s = sqrt(rho)`, on a periodic grid.
///
/// Central differences throughout; a point whose amplitude is at the
/// floor gets `Q = 0` instead of an infinity. Returns false when the
/// buffers disagree in length or the parameters are not usable, and
/// leaves `out` alone in that case.
pub fn quantum_potential(rho: &[f32], dx: f32, hbar: f32, mass: f32, out: &mut [f32]) -> bool {
    let n = rho.len();
    if out.len() < n || n == 0 {
        return false;
    }
    if !dx.is_finite() || dx <= 0.0 || !hbar.is_finite() || hbar < 0.0 {
        return false;
    }
    if !mass.is_finite() || mass <= 0.0 {
        return false;
    }
    let coeff = -(hbar * hbar) / (2.0 * mass);
    for i in 0..n {
        let s_here = rho[i].max(0.0).sqrt();
        if s_here <= DENSITY_FLOOR {
            out[i] = 0.0;
            continue;
        }
        let s_left = rho[(i + n - 1) % n].max(0.0).sqrt();
        let s_right = rho[(i + 1) % n].max(0.0).sqrt();
        let lap = (s_left - s_here * 2.0 + s_right) / (dx * dx);
        out[i] = coeff * lap / s_here;
    }
    true
}

/// A one-dimensional Madelung fluid on a periodic grid.
#[derive(Clone, Debug)]
pub struct QuantumFluid1D {
    n: usize,
    dx: f32,
    hbar: f32,
    mass: f32,
    rho: Vec<f32>,
    vel: Vec<f32>,
}

/// Scratch for a fluid step, so that stepping allocates nothing.
#[derive(Clone, Debug)]
pub struct MadelungScratch {
    /// The next density.
    pub rho_next: Vec<f32>,
    /// The next velocity.
    pub vel_next: Vec<f32>,
    /// The quantum potential of the current density.
    pub potential: Vec<f32>,
}

impl MadelungScratch {
    /// Scratch for a fluid of `n` points.
    pub fn new(n: usize) -> Self {
        Self {
            rho_next: vec![0.0; n],
            vel_next: vec![0.0; n],
            potential: vec![0.0; n],
        }
    }
}

impl QuantumFluid1D {
    /// A fluid of `n` points with spacing `dx`, uniform density `rho0`
    /// and zero velocity.
    pub fn new(n: usize, dx: f32, hbar: f32, mass: f32, rho0: f32) -> Self {
        Self {
            n,
            dx,
            hbar,
            mass,
            rho: vec![rho0.max(0.0); n],
            vel: vec![0.0; n],
        }
    }

    /// The number of points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The spacing.
    pub fn dx(&self) -> f32 {
        self.dx
    }

    /// The density.
    pub fn rho(&self) -> &[f32] {
        &self.rho
    }

    /// The velocity.
    pub fn vel(&self) -> &[f32] {
        &self.vel
    }

    /// Replaces the density.
    ///
    /// Returns false when the slice is short; a negative entry is
    /// clamped to zero rather than admitted.
    pub fn set_density(&mut self, values: &[f32]) -> bool {
        if values.len() < self.n {
            return false;
        }
        for (r, v) in self.rho.iter_mut().zip(values.iter()).take(self.n) {
            *r = v.max(0.0);
        }
        true
    }

    /// Replaces the velocity.
    ///
    /// Returns false when the slice is short.
    pub fn set_velocity(&mut self, values: &[f32]) -> bool {
        if values.len() < self.n {
            return false;
        }
        self.vel[..self.n].copy_from_slice(&values[..self.n]);
        true
    }

    /// The total mass: the density integrated over the grid.
    pub fn mass_total(&self) -> f32 {
        let mut sum = 0.0f32;
        for r in self.rho.iter() {
            sum += *r;
        }
        sum * self.dx
    }

    /// Advances the fluid by one explicit Euler step.
    ///
    /// Continuity moves the density with the flux `rho v`; the momentum
    /// equation advects the velocity and pushes it down the gradient of
    /// the quantum potential. Returns false for a non-positive step, a
    /// mismatched scratch, or unusable parameters, and leaves the fluid
    /// alone in that case.
    pub fn step(&mut self, dt: f32, scratch: &mut MadelungScratch) -> bool {
        let n = self.n;
        if n == 0
            || !dt.is_finite()
            || dt <= 0.0
            || scratch.rho_next.len() < n
            || scratch.vel_next.len() < n
            || scratch.potential.len() < n
        {
            return false;
        }
        if !quantum_potential(
            &self.rho,
            self.dx,
            self.hbar,
            self.mass,
            &mut scratch.potential[..n],
        ) {
            return false;
        }
        let dx = self.dx;
        for i in 0..n {
            let left = (i + n - 1) % n;
            let right = (i + 1) % n;
            let flux_left = self.rho[left] * self.vel[left];
            let flux_right = self.rho[right] * self.vel[right];
            let next_rho = self.rho[i] - dt * (flux_right - flux_left) / (2.0 * dx);
            scratch.rho_next[i] = next_rho.max(0.0);
            let advect = self.vel[i] * (self.vel[right] - self.vel[left]) / (2.0 * dx);
            let push = (scratch.potential[right] - scratch.potential[left]) / (2.0 * dx);
            scratch.vel_next[i] = self.vel[i] - dt * (advect + push);
        }
        self.rho.copy_from_slice(&scratch.rho_next[..n]);
        self.vel.copy_from_slice(&scratch.vel_next[..n]);
        true
    }
}
