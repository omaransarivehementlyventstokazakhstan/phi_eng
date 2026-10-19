//! Surface/material properties for rigid bodies.
//!
//! A [`PhysicsMaterial`] bundles friction, restitution and density. Bodies own
//! exactly one material (shapes inherit their body's material); contacts
//! combine the two bodies' materials with [`PhysicsMaterial::combine_friction`] /
//! [`PhysicsMaterial::combine_restitution`]. Mirrored by `PhiEng.Dynamics.PhysicsMaterial` in C#.

/// Minimum sane density: guards against zero/negative mass (which would
/// produce infinite inverse mass and poison velocities with inf/NaN).
/// Single source of truth with the rest of the engine (`phieng_math::EPS`).
pub const MIN_DENSITY: f32 = phieng_math::EPS;

/// Surface and mass properties of a rigid body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsMaterial {
    /// Sliding resistance, typically 0.0 (frictionless) to 1.0+ (grippy).
    pub friction: f32,
    /// Bounciness, 0.0 (dead) to 1.0 (perfectly elastic).
    pub restitution: f32,
    /// Mass per unit area (2D) / volume (3D); derives mass and inertia.
    pub density: f32,
    /// Specific heat capacity J/(kg·K). Default: 900.0.
    pub specific_heat: f32,
    /// Thermal conductivity W/(m·K). Default: 1.0.
    pub thermal_conductivity: f32,
    /// Surface emissivity 0..1. Default: 0.9.
    pub emissivity: f32,
}

impl PhysicsMaterial {
    pub const DEFAULT_FRICTION: f32 = 0.4;
    pub const DEFAULT_RESTITUTION: f32 = 0.1;
    pub const DEFAULT_DENSITY: f32 = 1.0;
    pub const DEFAULT_SPECIFIC_HEAT: f32 = 900.0;
    pub const DEFAULT_THERMAL_CONDUCTIVITY: f32 = 1.0;
    pub const DEFAULT_EMISSIVITY: f32 = 0.9;

    /// Build a material, sanitizing inputs (negative friction/density and
    /// non-finite values fall back to defaults; restitution clamps to [0, 1]).
    pub fn new(friction: f32, restitution: f32, density: f32) -> Self {
        Self {
            friction: if friction.is_finite() && friction >= 0.0 {
                friction
            } else {
                Self::DEFAULT_FRICTION
            },
            restitution: if restitution.is_finite() {
                restitution.clamp(0.0, 1.0)
            } else {
                Self::DEFAULT_RESTITUTION
            },
            density: if density.is_finite() && density > 0.0 {
                density
            } else {
                Self::DEFAULT_DENSITY
            },
            specific_heat: Self::DEFAULT_SPECIFIC_HEAT,
            thermal_conductivity: Self::DEFAULT_THERMAL_CONDUCTIVITY,
            emissivity: Self::DEFAULT_EMISSIVITY,
        }
    }

    /// Combine two friction coefficients with the geometric mean, so a
    /// frictionless surface (`mu = 0`) always slides regardless of the other
    /// body, and equal materials keep their value (`sqrt(mu*mu) = mu`).
    pub fn combine_friction(a: f32, b: f32) -> f32 {
        (a.max(0.0) * b.max(0.0)).sqrt()
    }

    /// Combine restitution with the maximum, so a bouncy object always
    /// bounces even against a dead surface.
    pub fn combine_restitution(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: Self::DEFAULT_FRICTION,
            restitution: Self::DEFAULT_RESTITUTION,
            density: Self::DEFAULT_DENSITY,
            specific_heat: Self::DEFAULT_SPECIFIC_HEAT,
            thermal_conductivity: Self::DEFAULT_THERMAL_CONDUCTIVITY,
            emissivity: Self::DEFAULT_EMISSIVITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_defaults() {
        let m = PhysicsMaterial::default();
        assert_eq!(m.friction, 0.4);
        assert_eq!(m.restitution, 0.1);
        assert_eq!(m.density, 1.0);
    }

    #[test]
    fn test_material_new_sanitizes() {
        let m = PhysicsMaterial::new(f32::NAN, 5.0, -2.0);
        assert_eq!(m.friction, PhysicsMaterial::DEFAULT_FRICTION);
        assert_eq!(m.restitution, 1.0);
        assert_eq!(m.density, PhysicsMaterial::DEFAULT_DENSITY);
        assert_eq!(m.specific_heat, PhysicsMaterial::DEFAULT_SPECIFIC_HEAT);
        assert_eq!(
            m.thermal_conductivity,
            PhysicsMaterial::DEFAULT_THERMAL_CONDUCTIVITY
        );
        assert_eq!(m.emissivity, PhysicsMaterial::DEFAULT_EMISSIVITY);

        let m = PhysicsMaterial::new(0.8, 0.6, 2.0);
        assert_eq!(m.friction, 0.8);
        assert_eq!(m.restitution, 0.6);
        assert_eq!(m.density, 2.0);
    }

    #[test]
    fn test_combine_friction_geometric_mean() {
        assert!((PhysicsMaterial::combine_friction(0.9, 0.9) - 0.9).abs() < 1e-6);
        assert!((PhysicsMaterial::combine_friction(1.0, 0.25) - 0.5).abs() < 1e-6);
        // Frictionless side always wins: the pair must slide.
        assert_eq!(PhysicsMaterial::combine_friction(0.0, 0.9), 0.0);
    }

    #[test]
    fn test_combine_restitution_max() {
        assert_eq!(PhysicsMaterial::combine_restitution(0.9, 0.0), 0.9);
        assert_eq!(PhysicsMaterial::combine_restitution(0.2, 0.5), 0.5);
    }
}
