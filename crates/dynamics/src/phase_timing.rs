//! Per-phase wall-clock instrumentation for the step pipeline.
//!
//! This module only exists when the `phase-timing` feature is enabled.
//! With the feature off the step pipeline contains no clock reads and no
//! branches, and the crate's public API is exactly what it was before —
//! a performance change must not change what a dependent crate can see.
//!
//! The instrumentation is *observational only*: it reads a clock and
//! accumulates elapsed time. It never touches the simulation state, so
//! the trajectory hashes are identical with the feature on and off. The
//! only cost is the clock reads themselves (two per phase per step,
//! roughly 200 ns on a modern desktop — three orders of magnitude below
//! a typical step).
//!
//! ```
//! # #[cfg(feature = "phase-timing")]
//! # fn demo() {
//! use phieng_dynamics::PhaseTimings;
//!
//! let timings = PhaseTimings::default();
//! assert_eq!(timings.steps, 0);
//! # }
//! ```

/// Accumulated wall-clock time per step phase, in milliseconds.
///
/// The phases mirror the numbered sections of `World2::step_inner` and
/// `World3::step_inner`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhaseTimings {
    /// Gravity kick plus the ambient-force kick.
    pub gravity: f64,
    /// The SPH fluid phase (zero when the fluid holds no particles).
    pub fluid: f64,
    /// Broad-phase backend rebuild (clear + re-insert every body).
    pub broadphase_rebuild: f64,
    /// Broad-phase pair generation (the cell scan, the sort and the
    /// deduplication).
    pub broadphase_pairs: f64,
    /// Narrow-phase contact generation.
    pub narrowphase: f64,
    /// Contact-cache warm start and constraint validation.
    pub cache: f64,
    /// The contact solver, the joint solver and the cache write-back.
    pub solver: f64,
    /// Motion integration (Euler or the Verlet drift + second half-kick).
    pub integrate: f64,
    /// The nuclear and thermal phases.
    pub thermal: f64,
    /// Body validation and the sleep sweep.
    pub sleep: f64,
    /// How many steps were accounted for.
    pub steps: u64,
}

impl PhaseTimings {
    /// The sum of every phase, in milliseconds.
    ///
    /// This is the accounted time; it is less than the wall time of the
    /// step because the substepping decision, the clock reads and the
    /// bookkeeping around the phases are not attributed to a phase.
    pub fn total_ms(&self) -> f64 {
        self.gravity
            + self.fluid
            + self.broadphase_rebuild
            + self.broadphase_pairs
            + self.narrowphase
            + self.cache
            + self.solver
            + self.integrate
            + self.thermal
            + self.sleep
    }

    /// The accounted milliseconds per step.
    pub fn ms_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_ms() / self.steps as f64
        }
    }

    /// Add one step's worth of accumulators into this total.
    pub(crate) fn merge(&mut self, other: PhaseTimings) {
        self.gravity += other.gravity;
        self.fluid += other.fluid;
        self.broadphase_rebuild += other.broadphase_rebuild;
        self.broadphase_pairs += other.broadphase_pairs;
        self.narrowphase += other.narrowphase;
        self.cache += other.cache;
        self.solver += other.solver;
        self.integrate += other.integrate;
        self.thermal += other.thermal;
        self.sleep += other.sleep;
        self.steps += other.steps;
    }

    /// Forget everything measured so far.
    pub fn reset(&mut self) {
        *self = PhaseTimings::default();
    }
}
