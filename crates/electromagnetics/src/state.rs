//! Shared configuration and per-body state types.

use glam::{Vec2, Vec3};

use crate::constants::{COULOMB_K, MAX_ACCELERATION, MAX_CHARGE, MAX_FIELD_CELLS, MIN_MATERIAL};

/// How the electromagnetic force is computed each step.
///
/// The discriminants are part of the snapshot file format: never reorder
/// them. `Pairwise` and `Field` are mutually exclusive; applying both
/// would count the same field twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ElectroMode {
    /// No electromagnetic force. A world with this mode behaves exactly
    /// as it did before the module existed.
    Off = 0,
    /// Direct Coulomb sum over every body pair, plus the uniform external
    /// fields. Exact for point charges, `O(n^2)`.
    Pairwise = 1,
    /// A grid field (see [`crate::poisson`]). The uniform external fields
    /// are still added; the pairwise Coulomb sum is not.
    Field = 2,
}

/// The boundary condition of a field grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FieldBoundary {
    /// The potential is held at zero on the boundary.
    Dirichlet = 0,
    /// The normal derivative is zero (a mirror wall).
    Neumann = 1,
    /// The grid wraps around.
    Periodic = 2,
}

/// The grid configuration of a field-mode solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldConfig2 {
    /// The number of cells along X.
    pub nx: u32,
    /// The number of cells along Y.
    pub ny: u32,
    /// The cell size in scene units.
    pub dx: f32,
    /// The world-space position of cell `(0, 0)`.
    pub origin: Vec2,
    /// Fixed SOR iterations per solve.
    pub iterations: u32,
    /// SOR over-relaxation in `(0, 2)`.
    pub omega: f32,
    /// The boundary condition.
    pub boundary: FieldBoundary,
    /// Relative permittivity of the medium.
    pub eps_r: f32,
    /// Whether the solved field is stored in a snapshot.
    pub snapshot_field: bool,
}

impl Default for FieldConfig2 {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 128,
            dx: 0.1,
            origin: Vec2::ZERO,
            iterations: 40,
            omega: 1.8,
            boundary: FieldBoundary::Dirichlet,
            eps_r: 1.0,
            snapshot_field: true,
        }
    }
}

/// The 3D counterpart of [`FieldConfig2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldConfig3 {
    /// The number of cells along X.
    pub nx: u32,
    /// The number of cells along Y.
    pub ny: u32,
    /// The number of cells along Z.
    pub nz: u32,
    /// The cell size in scene units.
    pub dx: f32,
    /// The world-space position of cell `(0, 0, 0)`.
    pub origin: Vec3,
    /// Fixed SOR iterations per solve.
    pub iterations: u32,
    /// SOR over-relaxation in `(0, 2)`.
    pub omega: f32,
    /// The boundary condition.
    pub boundary: FieldBoundary,
    /// Relative permittivity of the medium.
    pub eps_r: f32,
    /// Whether the solved field is stored in a snapshot.
    pub snapshot_field: bool,
}

impl Default for FieldConfig3 {
    fn default() -> Self {
        Self {
            nx: 32,
            ny: 32,
            nz: 32,
            dx: 0.25,
            origin: Vec3::ZERO,
            iterations: 40,
            omega: 1.8,
            boundary: FieldBoundary::Dirichlet,
            eps_r: 1.0,
            snapshot_field: true,
        }
    }
}

impl FieldConfig2 {
    /// The number of cells.
    pub fn cells(&self) -> usize {
        self.nx as usize * self.ny as usize
    }
}

impl FieldConfig3 {
    /// The number of cells.
    pub fn cells(&self) -> usize {
        self.nx as usize * self.ny as usize * self.nz as usize
    }
}

/// The electromagnetic configuration of a 2D world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectroConfig2 {
    /// Master switch. Off by default, so a world that never asks for
    /// electromagnetics keeps its trajectory to the last bit.
    pub enabled: bool,
    /// The force model.
    pub mode: ElectroMode,
    /// Coulomb's constant, in scene units.
    pub coulomb_k: f32,
    /// Softening length; finite and `>= 0`.
    pub softening: f32,
    /// The uniform external electric field.
    pub uniform_e: Vec2,
    /// The uniform external magnetic field (out of plane, `Bz`).
    pub uniform_b: f32,
    /// Relative permittivity of the medium (scales the Coulomb force).
    pub permittivity: f32,
    /// Relative permeability of the medium (reserved for magnetostatics).
    pub permeability: f32,
    /// The largest acceleration a body may receive.
    pub max_acceleration: f32,
    /// The field grid (ignored unless `mode == Field`).
    pub field: FieldConfig2,
}

impl Default for ElectroConfig2 {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ElectroMode::Off,
            coulomb_k: COULOMB_K,
            softening: 0.0,
            uniform_e: Vec2::ZERO,
            uniform_b: 0.0,
            permittivity: 1.0,
            permeability: 1.0,
            max_acceleration: MAX_ACCELERATION,
            field: FieldConfig2::default(),
        }
    }
}

impl ElectroConfig2 {
    /// True for a freshly constructed config. The snapshot writer emits
    /// the pre-electromagnetics schema for worlds satisfying this.
    pub fn is_default(&self) -> bool {
        !self.enabled
            && self.mode == ElectroMode::Off
            && self.coulomb_k == COULOMB_K
            && self.softening == 0.0
            && self.uniform_e == Vec2::ZERO
            && self.uniform_b == 0.0
            && self.permittivity == 1.0
            && self.permeability == 1.0
            && self.max_acceleration == MAX_ACCELERATION
            && self.field == FieldConfig2::default()
    }

    /// Validate every field. Rejecting is the house rule: never guess.
    pub fn validate(&self) -> Result<(), ElectroError> {
        if !self.enabled {
            // A disabled config is inert; only the master switch matters.
            return Ok(());
        }
        if !self.coulomb_k.is_finite() {
            return Err(ElectroError::NonFinite);
        }
        if !self.softening.is_finite() || self.softening < 0.0 {
            return Err(ElectroError::NegativeSoftening);
        }
        if !self.uniform_e.is_finite() || !self.uniform_b.is_finite() {
            return Err(ElectroError::NonFinite);
        }
        if !self.permittivity.is_finite() || self.permittivity < MIN_MATERIAL {
            return Err(ElectroError::InvalidPermittivity);
        }
        if !self.permeability.is_finite() || self.permeability < MIN_MATERIAL {
            return Err(ElectroError::InvalidPermeability);
        }
        if !self.max_acceleration.is_finite() || self.max_acceleration <= 0.0 {
            return Err(ElectroError::InvalidAcceleration);
        }
        if self.mode == ElectroMode::Field {
            validate_field2(&self.field)?;
        }
        Ok(())
    }
}

/// The 3D counterpart of [`ElectroConfig2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectroConfig3 {
    /// Master switch.
    pub enabled: bool,
    /// The force model.
    pub mode: ElectroMode,
    /// Coulomb's constant, in scene units.
    pub coulomb_k: f32,
    /// Softening length; finite and `>= 0`.
    pub softening: f32,
    /// The uniform external electric field.
    pub uniform_e: Vec3,
    /// The uniform external magnetic field.
    pub uniform_b: Vec3,
    /// Relative permittivity of the medium.
    pub permittivity: f32,
    /// Relative permeability of the medium.
    pub permeability: f32,
    /// The largest acceleration a body may receive.
    pub max_acceleration: f32,
    /// The field grid (ignored unless `mode == Field`).
    pub field: FieldConfig3,
}

impl Default for ElectroConfig3 {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ElectroMode::Off,
            coulomb_k: COULOMB_K,
            softening: 0.0,
            uniform_e: Vec3::ZERO,
            uniform_b: Vec3::ZERO,
            permittivity: 1.0,
            permeability: 1.0,
            max_acceleration: MAX_ACCELERATION,
            field: FieldConfig3::default(),
        }
    }
}

impl ElectroConfig3 {
    /// See [`ElectroConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        !self.enabled
            && self.mode == ElectroMode::Off
            && self.coulomb_k == COULOMB_K
            && self.softening == 0.0
            && self.uniform_e == Vec3::ZERO
            && self.uniform_b == Vec3::ZERO
            && self.permittivity == 1.0
            && self.permeability == 1.0
            && self.max_acceleration == MAX_ACCELERATION
            && self.field == FieldConfig3::default()
    }

    /// See [`ElectroConfig2::validate`].
    pub fn validate(&self) -> Result<(), ElectroError> {
        if !self.enabled {
            return Ok(());
        }
        if !self.coulomb_k.is_finite() {
            return Err(ElectroError::NonFinite);
        }
        if !self.softening.is_finite() || self.softening < 0.0 {
            return Err(ElectroError::NegativeSoftening);
        }
        if !self.uniform_e.is_finite() || !self.uniform_b.is_finite() {
            return Err(ElectroError::NonFinite);
        }
        if !self.permittivity.is_finite() || self.permittivity < MIN_MATERIAL {
            return Err(ElectroError::InvalidPermittivity);
        }
        if !self.permeability.is_finite() || self.permeability < MIN_MATERIAL {
            return Err(ElectroError::InvalidPermeability);
        }
        if !self.max_acceleration.is_finite() || self.max_acceleration <= 0.0 {
            return Err(ElectroError::InvalidAcceleration);
        }
        if self.mode == ElectroMode::Field {
            validate_field3(&self.field)?;
        }
        Ok(())
    }
}

fn validate_field2(f: &FieldConfig2) -> Result<(), ElectroError> {
    if f.nx == 0 || f.ny == 0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.dx.is_finite() || f.dx <= 0.0 || !f.origin.is_finite() {
        return Err(ElectroError::InvalidGrid);
    }
    if f.cells() > MAX_FIELD_CELLS {
        return Err(ElectroError::GridTooLarge);
    }
    if f.iterations == 0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.omega.is_finite() || f.omega <= 0.0 || f.omega >= 2.0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.eps_r.is_finite() || f.eps_r < MIN_MATERIAL {
        return Err(ElectroError::InvalidPermittivity);
    }
    Ok(())
}

fn validate_field3(f: &FieldConfig3) -> Result<(), ElectroError> {
    if f.nx == 0 || f.ny == 0 || f.nz == 0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.dx.is_finite() || f.dx <= 0.0 || !f.origin.is_finite() {
        return Err(ElectroError::InvalidGrid);
    }
    if f.cells() > MAX_FIELD_CELLS {
        return Err(ElectroError::GridTooLarge);
    }
    if f.iterations == 0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.omega.is_finite() || f.omega <= 0.0 || f.omega >= 2.0 {
        return Err(ElectroError::InvalidGrid);
    }
    if !f.eps_r.is_finite() || f.eps_r < MIN_MATERIAL {
        return Err(ElectroError::InvalidPermittivity);
    }
    Ok(())
}

/// Why an electromagnetic configuration or operation was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectroError {
    /// A value was not finite.
    NonFinite,
    /// The softening length was negative.
    NegativeSoftening,
    /// The relative permittivity was zero or negative.
    InvalidPermittivity,
    /// The relative permeability was zero or negative.
    InvalidPermeability,
    /// The acceleration cap was zero or negative.
    InvalidAcceleration,
    /// The grid dimensions, spacing or solver tuning were invalid.
    InvalidGrid,
    /// The grid would allocate more cells than the engine allows.
    GridTooLarge,
}

/// The electromagnetic state of a 2D body.
///
/// It lives on the body (like the thermal and nuclear per-body state) so
/// that it follows the body through removals and so that the hot part of
/// the body struct is untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectroState2 {
    /// The net electric charge.
    pub charge: f32,
    /// How strongly the body feels electric/magnetic forces. `0` makes a
    /// body immune (it still exerts); negative is the caller's physics.
    pub charge_scale: f32,
    /// Reserved for induced-current models.
    pub conductivity: f32,
    /// Relative permittivity of the body.
    pub permittivity: f32,
    /// Relative permeability of the body.
    pub permeability: f32,
    /// The out-of-plane magnetic moment.
    pub magnetic_moment: f32,
    /// The capacitance used by a grounded conductor.
    pub capacitance: f32,
    /// The resistance to ground; zero means the conductor never drains.
    pub resistance: f32,
    /// Whether the body is a conductor.
    pub is_conductor: bool,
    /// Whether a conductor is grounded (its charge drains).
    pub is_grounded: bool,
}

impl Default for ElectroState2 {
    fn default() -> Self {
        Self {
            charge: 0.0,
            charge_scale: 1.0,
            conductivity: 0.0,
            permittivity: 1.0,
            permeability: 1.0,
            magnetic_moment: 0.0,
            capacitance: 1.0,
            resistance: 0.0,
            is_conductor: false,
            is_grounded: false,
        }
    }
}

impl ElectroState2 {
    /// True for a freshly constructed state.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Clamp to the engine's safe ranges, dropping non-finite values.
    ///
    /// The per-step validation calls this: a direct field write can put
    /// anything in a public body, and a non-finite charge would poison
    /// every other body in the pairwise sum.
    pub fn sanitized(&self) -> Self {
        let mut s = *self;
        if !s.charge.is_finite() {
            s.charge = 0.0;
        }
        s.charge = s.charge.clamp(-MAX_CHARGE, MAX_CHARGE);
        if !s.charge_scale.is_finite() {
            s.charge_scale = 1.0;
        }
        if !s.conductivity.is_finite() || s.conductivity < 0.0 {
            s.conductivity = 0.0;
        }
        if !s.permittivity.is_finite() || s.permittivity < MIN_MATERIAL {
            s.permittivity = 1.0;
        }
        if !s.permeability.is_finite() || s.permeability < MIN_MATERIAL {
            s.permeability = 1.0;
        }
        if !s.magnetic_moment.is_finite() {
            s.magnetic_moment = 0.0;
        }
        if !s.capacitance.is_finite() || s.capacitance <= 0.0 {
            s.capacitance = 1.0;
        }
        if !s.resistance.is_finite() || s.resistance < 0.0 {
            s.resistance = 0.0;
        }
        s
    }
}

/// The electromagnetic state of a 3D body. See [`ElectroState2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectroState3 {
    /// The net electric charge.
    pub charge: f32,
    /// The coupling multiplier.
    pub charge_scale: f32,
    /// Reserved for induced-current models.
    pub conductivity: f32,
    /// Relative permittivity of the body.
    pub permittivity: f32,
    /// Relative permeability of the body.
    pub permeability: f32,
    /// The magnetic dipole moment.
    pub magnetic_moment: Vec3,
    /// The capacitance used by a grounded conductor.
    pub capacitance: f32,
    /// The resistance to ground; zero means the conductor never drains.
    pub resistance: f32,
    /// Whether the body is a conductor.
    pub is_conductor: bool,
    /// Whether a conductor is grounded.
    pub is_grounded: bool,
}

impl Default for ElectroState3 {
    fn default() -> Self {
        Self {
            charge: 0.0,
            charge_scale: 1.0,
            conductivity: 0.0,
            permittivity: 1.0,
            permeability: 1.0,
            magnetic_moment: Vec3::ZERO,
            capacitance: 1.0,
            resistance: 0.0,
            is_conductor: false,
            is_grounded: false,
        }
    }
}

impl ElectroState3 {
    /// True for a freshly constructed state.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// See [`ElectroState2::sanitized`].
    pub fn sanitized(&self) -> Self {
        let mut s = *self;
        if !s.charge.is_finite() {
            s.charge = 0.0;
        }
        s.charge = s.charge.clamp(-MAX_CHARGE, MAX_CHARGE);
        if !s.charge_scale.is_finite() {
            s.charge_scale = 1.0;
        }
        if !s.conductivity.is_finite() || s.conductivity < 0.0 {
            s.conductivity = 0.0;
        }
        if !s.permittivity.is_finite() || s.permittivity < MIN_MATERIAL {
            s.permittivity = 1.0;
        }
        if !s.permeability.is_finite() || s.permeability < MIN_MATERIAL {
            s.permeability = 1.0;
        }
        if !s.magnetic_moment.is_finite() {
            s.magnetic_moment = Vec3::ZERO;
        }
        if !s.capacitance.is_finite() || s.capacitance <= 0.0 {
            s.capacitance = 1.0;
        }
        if !s.resistance.is_finite() || s.resistance < 0.0 {
            s.resistance = 0.0;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_default() {
        assert!(ElectroConfig2::default().is_default());
        assert!(ElectroConfig3::default().is_default());
        assert_eq!(ElectroMode::Field as u32, 2);
        assert_eq!(FieldBoundary::Periodic as u32, 2);
    }

    #[test]
    fn validate_rejects_hostile_values() {
        let mut c = ElectroConfig2 {
            enabled: true,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
        c.softening = -1.0;
        assert_eq!(c.validate(), Err(ElectroError::NegativeSoftening));
        c.softening = 0.0;
        c.permittivity = 0.0;
        assert_eq!(c.validate(), Err(ElectroError::InvalidPermittivity));
        c.permittivity = 1.0;
        c.uniform_e = Vec2::new(f32::NAN, 0.0);
        assert_eq!(c.validate(), Err(ElectroError::NonFinite));
    }

    #[test]
    fn field_validation_rejects_bad_grids() {
        let mut c = ElectroConfig2 {
            enabled: true,
            mode: ElectroMode::Field,
            ..Default::default()
        };
        c.field.nx = 0;
        assert_eq!(c.validate(), Err(ElectroError::InvalidGrid));
        c.field.nx = 128;
        c.field.omega = 2.0;
        assert_eq!(c.validate(), Err(ElectroError::InvalidGrid));
        c.field.omega = 1.8;
        c.field.nx = 4000;
        c.field.ny = 4000;
        assert_eq!(c.validate(), Err(ElectroError::GridTooLarge));
    }

    #[test]
    fn sanitize_clamps_and_drops_nonfinite() {
        let s = ElectroState2 {
            charge: f32::INFINITY,
            charge_scale: f32::NAN,
            permittivity: 0.0,
            ..Default::default()
        }
        .sanitized();
        assert_eq!(s.charge, 0.0);
        assert_eq!(s.charge_scale, 1.0);
        assert_eq!(s.permittivity, 1.0);
    }
}
