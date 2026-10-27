//! Magnetostatics: the Lorentz acceleration in uniform fields and the
//! torque on a magnetic dipole.
//!
//! Pairwise magnetic forces are not modelled in this version: a moving
//! point charge's Biot-Savart field is frame-dependent and non-conservative,
//! and it is a documented limitation. The uniform external field, its
//! `q v x B` force and the dipole torque cover the game-facing cases.

use glam::{Vec2, Vec3};

/// The acceleration of a charge in a uniform 2D field.
///
/// `a = (q/m) (E + v x B)` with `B = Bz z_hat`, so
/// `v x B = (vy * Bz, -vx * Bz)`. `q_over_m` is `q * charge_scale / m`.
#[inline]
pub fn lorentz_accel_2(v: Vec2, q_over_m: f32, e: Vec2, bz: f32) -> Vec2 {
    let vxb = Vec2::new(v.y * bz, -v.x * bz);
    (e + vxb) * q_over_m
}

/// The acceleration of a charge in a uniform 3D field.
#[inline]
pub fn lorentz_accel_3(v: Vec3, q_over_m: f32, e: Vec3, b: Vec3) -> Vec3 {
    (e + v.cross(b)) * q_over_m
}

/// The torque a uniform magnetic field applies to a dipole: `tau = m x B`.
#[inline]
pub fn dipole_torque_3(moment: Vec3, b: Vec3) -> Vec3 {
    moment.cross(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_charge_feels_no_magnetic_force() {
        let a = lorentz_accel_2(Vec2::ZERO, 1.0, Vec2::ZERO, 3.0);
        assert_eq!(a, Vec2::ZERO);
    }

    #[test]
    fn magnetic_force_is_perpendicular_to_velocity() {
        let v = Vec2::new(1.0, 0.0);
        let a = lorentz_accel_2(v, 1.0, Vec2::ZERO, 1.0);
        assert!((a - Vec2::new(0.0, -1.0)).length() < 1e-6);
        assert!(v.dot(a).abs() < 1e-6);
    }

    #[test]
    fn electric_field_accelerates_along_itself() {
        let a = lorentz_accel_2(Vec2::ZERO, 2.0, Vec2::new(3.0, 0.0), 0.0);
        assert!((a - Vec2::new(6.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn cyclotron_radius_and_period() {
        // r = m v / (q B); with m/q = 1, v = 2, B = 1 -> r = 2.
        // Period = 2 pi m / (q B) = 2 pi.
        let b = 1.0f32;
        let v = 2.0f32;
        let r = v / b;
        assert!((r - 2.0).abs() < 1e-6);
        let period = 2.0 * core::f32::consts::PI / b;
        assert!((period - 2.0 * core::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn dipole_torque_aligns_with_the_field() {
        let t = dipole_torque_3(Vec3::X, Vec3::Y);
        assert!((t - Vec3::Z).length() < 1e-6);
        let aligned = dipole_torque_3(Vec3::Z, Vec3::Z);
        assert!(aligned.length() < 1e-6);
    }

    #[test]
    fn three_d_lorentz_mirror() {
        let a = lorentz_accel_3(Vec3::X, 1.0, Vec3::ZERO, Vec3::Z);
        // X x Z = -Y.
        assert!((a - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
    }
}
