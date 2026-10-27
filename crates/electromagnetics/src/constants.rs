//! Constants and unit helpers.
//!
//! The engine is **unit-agnostic**: the defaults are game-scale
//! (`k = 1`, `c = 1` cell per second), exactly like gravity's `big_g = 1`.
//! The SI values are provided for callers that want physical units, but
//! `f32` cannot represent the product of two elementary charges without
//! losing most of its precision, and an SI light speed needs device-scale
//! cells to stay stable at a 60 Hz step. See `LIMITATIONS.md`.

/// Coulomb's constant in engine units. Same role as gravity's `big_g`.
pub const COULOMB_K: f32 = 1.0;

/// The speed of light in engine units: one grid cell per simulated
/// second. Only the field solvers use it.
pub const LIGHT_SPEED: f32 = 1.0;

/// Coulomb's constant in SI units (`8.9875517923e9 N m^2 / C^2`).
pub const SI_COULOMB_K: f32 = 8.987_552e9;

/// The speed of light in SI units, in metres per second.
pub const SI_LIGHT_SPEED: f32 = 2.997_924_5e8;

/// The largest charge magnitude the engine accepts.
///
/// The largest `f32` is about `3.4e38`; two charges at this bound give a
/// product of `1e30`, which stays finite even after a unit Coulomb
/// constant, so a pair force can never overflow to infinity and then
/// poison a trajectory with NaN.
pub const MAX_CHARGE: f32 = 1.0e15;

/// The largest acceleration the electric/magnetic kick may apply.
pub const MAX_ACCELERATION: f32 = 1.0e30;

/// The smallest relative permittivity/permeability the config accepts.
pub const MIN_MATERIAL: f32 = 1.0e-6;

/// The largest grid cell count a field may allocate (in memory).
///
/// Deliberately equal to the snapshot cap: a grid the engine can build is
/// always a grid it can save, so a save can never produce a file the same
/// engine refuses to load. 262,144 cells is 512x512 in 2D or 64^3 in 3D.
pub const MAX_FIELD_CELLS: usize = 262_144;

/// The largest grid cell count a snapshot may carry.
pub const MAX_SNAPSHOT_CELLS: usize = 262_144;

/// The vacuum permittivity implied by a Coulomb constant: `1 / (4 pi k)`.
///
/// `k` must be finite and positive; the configuration validation enforces
/// that before this is called.
#[inline]
pub fn eps0(coulomb_k: f32) -> f32 {
    1.0 / (4.0 * core::f32::consts::PI * coulomb_k)
}

/// The vacuum permeability implied by `k` and the propagation speed:
/// `mu0 = 1 / (eps0 * c^2)`.
#[inline]
pub fn mu0(coulomb_k: f32, light_speed: f32) -> f32 {
    1.0 / (eps0(coulomb_k) * light_speed * light_speed)
}
