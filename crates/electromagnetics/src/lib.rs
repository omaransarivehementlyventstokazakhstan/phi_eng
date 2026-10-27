//! Electromagnetics: electrostatics, magnetostatics and the field solvers.
//!
//! # What this crate is
//!
//! The crate is deliberately **body-agnostic**: every routine takes plain
//! positions, charges and field values. The rigid-body crate owns the
//! per-body state and the world-level orchestration; this crate owns the
//! arithmetic and the grids. That is the same split `phieng-quantum` uses,
//! and it keeps the field solvers independently testable.
//!
//! * [`electrostatics`] - Coulomb pair forces, the electric field and the
//!   potential at a point, with softening at close range.
//! * [`magnetostatics`] - the Lorentz acceleration in uniform fields and
//!   the torque on a magnetic dipole.
//! * [`poisson`] - the electrostatic grid: a red-black SOR solve of
//!   `nabla^2 phi = -rho / eps`, sampled back at arbitrary points.
//! * [`state`] - the shared configuration and per-body state types.
//!
//! # The guarantee
//!
//! Only operations IEEE-754 specifies exactly are used: add, subtract,
//! multiply, divide, `sqrt`, `min`, `max` and `abs`. There is no platform
//! math library on the path, so a trajectory is bit-identical on every
//! platform and the managed C# mirror can match it to the last bit.
//!
//! The two force modes are **mutually exclusive**: pairwise Coulomb and
//! the field grid are never applied together, because that would count
//! the same field twice.

pub mod constants;
pub mod electrostatics;
pub mod magnetostatics;
pub mod poisson;
pub mod state;

pub use constants::*;
pub use electrostatics::*;
pub use magnetostatics::*;
pub use poisson::*;
pub use state::*;
