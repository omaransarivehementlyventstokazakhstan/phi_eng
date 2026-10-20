//! Radiative heat transfer (2D and 3D).
//!
//! Position in the step: the thermal phase calls into this module after
//! integration, so it sees the final positions of the frame.
//!
//! The model is the standard gray-body exchange:
//!
//! * body to environment: `Q = eps * sigma * A * (T^4 - T_amb^4)`
//!   (Stefan-Boltzmann),
//! * body to body: `Q = eps_eff * sigma * A * F * (T_i^4 - T_j^4)` with
//!   the two-gray-body effective emissivity
//!   `eps_eff = 1 / (1/eps_i + 1/eps_j - 1)` and the view factor
//!   `F = min((r_j / d)^2, 1)`.
//!
//! That pair term is algebraically identical to the standard radiation
//! resistance network
//! `(1-eps_i)/(A_i F) + 1/(A_i F) + (1-eps_j)/(A_j F)`, i.e. it
//! already assumes Kirchhoff surfaces (`absorptivity = emissivity`).
//!
//! The 2D and 3D variants deliberately differ: 2D treats a body's
//! "surface area" as its perimeter and uses the *characteristic length*
//! in the pair term, while 3D uses the true surface area. The asymmetry
//! is reproduced verbatim so the managed C# mirror matches the Rust
//! core bit-for-bit; see `src/PhiEng.Dynamics/ThermalIntegration.cs`.

use super::body::{RigidBody2, RigidBody3};
use super::thermal::{ThermalConfig2D, ThermalConfig3D};

pub(crate) fn compute_radiation_2d(
    bodies: &[RigidBody2],
    config: &ThermalConfig2D,
    heat_flux: &mut [f32],
    surface_area: &[f32],
) {
    let sigma = config.stefan_boltzmann;
    let t_amb = config.ambient_temperature;
    let t_amb_sq = t_amb * t_amb;
    let t_amb_fourth = t_amb_sq * t_amb_sq;
    let cutoff_factor = config.radiation_radius_factor;

    for (i, body_i) in bodies.iter().enumerate() {
        let Some(_) = &body_i.shape else {
            continue;
        };

        let eps_i = body_i.material.emissivity.clamp(0.0, 1.0);
        if eps_i <= 0.0 {
            continue;
        }

        let a_surf = surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }

        let t_i = body_i.temperature.clamp(0.0, config.max_temperature);
        let t_i_sq = t_i * t_i;
        let t_i_fourth = t_i_sq * t_i_sq;

        // q_env is the radiative *loss* (positive when hotter than ambient),
        // but heat_flux accumulates heat *gained*, so subtract it.
        let q_env = eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth);
        heat_flux[i] -= q_env;

        let r_i = body_i
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        let l_i = body_i
            .shape
            .as_ref()
            .map(|s| s.characteristic_length())
            .unwrap_or(0.0);

        for (j, body_j) in bodies.iter().enumerate().skip(i + 1) {
            if body_j.shape.is_none() {
                continue;
            }

            let eps_j = body_j.material.emissivity.clamp(0.0, 1.0);
            if eps_j <= 0.0 {
                continue;
            }

            let r_j = body_j
                .shape
                .as_ref()
                .map(|s| s.bounding_radius())
                .unwrap_or(0.0);
            let dist = (body_i.transform.position - body_j.transform.position).length();
            let cutoff = cutoff_factor * (r_i + r_j);
            if dist > cutoff || dist <= 0.0 {
                continue;
            }

            let eps_eff = 1.0 / (1.0 / eps_i + 1.0 / eps_j - 1.0);
            if eps_eff <= 0.0 {
                continue;
            }

            let t_j = body_j.temperature.clamp(0.0, config.max_temperature);
            let t_j_sq = t_j * t_j;
            let t_j_fourth = t_j_sq * t_j_sq;

            let view_factor = (r_j / dist).powi(2).min(1.0);

            // Occlusion: a third body between the pair blocks the line of
            // sight, so the pair exchanges no radiative heat. Opt-in: it
            // costs an O(n) scan per pair.
            if config.radiation_occlusion && is_occluded_2d(bodies, i, j) {
                continue;
            }

            let q_rad = eps_eff * sigma * l_i * view_factor * (t_i_fourth - t_j_fourth);

            heat_flux[i] -= q_rad;
            heat_flux[j] += q_rad;
        }
    }
}

/// True when a third body lies between bodies `a` and `b` (2D), blocking
/// the line of sight. The blocker must be *between* them along the
/// segment and strictly closer to the midpoint (so the endpoints
/// themselves never block each other).
fn is_occluded_2d(bodies: &[RigidBody2], a: usize, b: usize) -> bool {
    let pa = bodies[a].transform.position;
    let pb = bodies[b].transform.position;
    let ab = pb - pa;
    let len_sq = ab.length_squared();
    if len_sq <= 0.0 {
        return false;
    }
    let mid = (pa + pb) * 0.5;
    let half = len_sq.sqrt() * 0.5;

    for (idx, body) in bodies.iter().enumerate() {
        if idx == a || idx == b || body.shape.is_none() {
            continue;
        }
        let p = body.transform.position;
        // Parameter along the segment; must be strictly inside.
        let t = (p - pa).dot(ab) / len_sq;
        if !(0.0 < t && t < 1.0) {
            continue;
        }
        // Distance from the segment.
        let closest = pa + ab * t;
        let d = (p - closest).length();
        let radius = body
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        // The blocker must reach the line (within its radius) and sit
        // closer to the midpoint than the endpoints do.
        if d <= radius && (p - mid).length() < half {
            return true;
        }
    }
    false
}

/// True when a third body lies between bodies `a` and `b` (3D), blocking
/// the line of sight. 3D counterpart of [`is_occluded_2d`].
fn is_occluded_3d(bodies: &[RigidBody3], a: usize, b: usize) -> bool {
    let pa = bodies[a].transform.position;
    let pb = bodies[b].transform.position;
    let ab = pb - pa;
    let len_sq = ab.length_squared();
    if len_sq <= 0.0 {
        return false;
    }
    let mid = (pa + pb) * 0.5;
    let half = len_sq.sqrt() * 0.5;

    for (idx, body) in bodies.iter().enumerate() {
        if idx == a || idx == b || body.shape.is_none() {
            continue;
        }
        let p = body.transform.position;
        let t = (p - pa).dot(ab) / len_sq;
        if !(0.0 < t && t < 1.0) {
            continue;
        }
        let closest = pa + ab * t;
        let d = (p - closest).length();
        let radius = body
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        if d <= radius && (p - mid).length() < half {
            return true;
        }
    }
    false
}

pub(crate) fn compute_radiation_3d(
    bodies: &[RigidBody3],
    config: &ThermalConfig3D,
    heat_flux: &mut [f32],
    surface_area: &[f32],
) {
    let sigma = config.stefan_boltzmann;
    let t_amb = config.ambient_temperature;
    let t_amb_sq = t_amb * t_amb;
    let t_amb_fourth = t_amb_sq * t_amb_sq;
    let cutoff_factor = config.radiation_radius_factor;

    for (i, body_i) in bodies.iter().enumerate() {
        if body_i.shape.is_none() {
            continue;
        }

        let eps_i = body_i.material.emissivity.clamp(0.0, 1.0);
        if eps_i <= 0.0 {
            continue;
        }

        let a_surf = surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }

        let t_i = body_i.temperature.clamp(0.0, config.max_temperature);
        let t_i_sq = t_i * t_i;
        let t_i_fourth = t_i_sq * t_i_sq;

        // q_env is the radiative *loss* (positive when hotter than ambient),
        // but heat_flux accumulates heat *gained*, so subtract it.
        let q_env = eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth);
        heat_flux[i] -= q_env;

        let r_i = body_i
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);

        for (j, body_j) in bodies.iter().enumerate().skip(i + 1) {
            if body_j.shape.is_none() {
                continue;
            }

            let eps_j = body_j.material.emissivity.clamp(0.0, 1.0);
            if eps_j <= 0.0 {
                continue;
            }

            let r_j = body_j
                .shape
                .as_ref()
                .map(|s| s.bounding_radius())
                .unwrap_or(0.0);
            let dist = (body_i.transform.position - body_j.transform.position).length();
            let cutoff = cutoff_factor * (r_i + r_j);
            if dist > cutoff || dist <= 0.0 {
                continue;
            }

            let eps_eff = 1.0 / (1.0 / eps_i + 1.0 / eps_j - 1.0);
            if eps_eff <= 0.0 {
                continue;
            }

            let t_j = body_j.temperature.clamp(0.0, config.max_temperature);
            let t_j_sq = t_j * t_j;
            let t_j_fourth = t_j_sq * t_j_sq;

            let view_factor = (r_j / dist).powi(2).min(1.0);

            // Occlusion (3D): see `is_occluded_2d`.
            if config.radiation_occlusion && is_occluded_3d(bodies, i, j) {
                continue;
            }

            let q_rad = eps_eff * sigma * a_surf * view_factor * (t_i_fourth - t_j_fourth);

            heat_flux[i] -= q_rad;
            heat_flux[j] += q_rad;
        }
    }
}
