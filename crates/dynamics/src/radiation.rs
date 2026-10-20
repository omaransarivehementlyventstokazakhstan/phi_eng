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
use super::thermal::{ThermalConfig2D, ThermalConfig3D, ViewFactorModel};

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

    // Sky / ground environment split. A non-positive sky or ground
    // temperature means "use the ambient". When the two end up equal
    // (the default) the historical single-temperature exchange is used,
    // so the result stays bit-identical.
    let (t_sky, t_ground) = environment_temperatures(config);
    let split_environment = t_sky != t_ground;
    let t_sky_fourth = fourth_power(t_sky);
    let t_ground_fourth = fourth_power(t_ground);

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
        let q_env = if split_environment {
            // Split the body's surface between the sky and the ground.
            let f_up = body_i
                .shape
                .as_ref()
                .map(|s| s.up_facing_fraction())
                .unwrap_or(0.5);
            eps_i
                * sigma
                * a_surf
                * (f_up * (t_i_fourth - t_sky_fourth)
                    + (1.0 - f_up) * (t_i_fourth - t_ground_fourth))
        } else {
            eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth)
        };
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
            let l_j = body_j
                .shape
                .as_ref()
                .map(|s| s.characteristic_length())
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
            let mut occluded = false;
            if config.radiation_occlusion && is_occluded_2d(bodies, i, j) {
                occluded = true;
            }

            let q_rad = match config.view_factor_model {
                ViewFactorModel::Legacy => {
                    if occluded {
                        if config.occluder_transmittance <= 0.0 {
                            // Fully opaque: the historical behaviour.
                            continue;
                        }
                        // A partially transparent occluder attenuates the
                        // exchange instead of blocking it completely.
                        eps_eff
                            * sigma
                            * l_i
                            * view_factor
                            * (t_i_fourth - t_j_fourth)
                            * config.occluder_transmittance
                    } else {
                        eps_eff * sigma * l_i * view_factor * (t_i_fourth - t_j_fourth)
                    }
                }
                ViewFactorModel::Symmetric => {
                    // Order-independent view factor: the geometric mean
                    // of the two directed factors, so the exchange rate
                    // does not depend on which body the pair loop
                    // happened to visit first.
                    let f = (r_i * r_j / (dist * dist)).min(1.0);
                    // The mean of the two emitting areas; for two equal
                    // bodies this is exactly the shared area.
                    let a_eff = 0.5 * (l_i + l_j) * f;
                    let q = eps_eff * sigma * a_eff * (t_i_fourth - t_j_fourth);
                    if occluded {
                        // A partially transparent occluder attenuates the
                        // exchange instead of blocking it completely.
                        q * config.occluder_transmittance
                    } else {
                        q
                    }
                }
            };

            heat_flux[i] -= q_rad;
            heat_flux[j] += q_rad;
        }
    }
}

/// True when a third body lies between bodies `a` and `b` (2D), blocking
/// the line of sight.
///
/// The blocker must project strictly inside the segment (so the two
/// endpoints never block each other) and reach the line of sight.
///
/// The historical implementation additionally required the blocker to be
/// closer to the segment's midpoint than the endpoints are. That test
/// was wrong: a blocker sitting just next to one of the two bodies
/// blocks the line of sight just as effectively as one in the middle,
/// so the exemption let radiation leak through gaps next to the
/// endpoints.
fn is_occluded_2d(bodies: &[RigidBody2], a: usize, b: usize) -> bool {
    let pa = bodies[a].transform.position;
    let pb = bodies[b].transform.position;
    let ab = pb - pa;
    let len_sq = ab.length_squared();
    if len_sq <= 0.0 {
        return false;
    }
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
        // The blocker must reach the line of sight.
        if d <= radius {
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
        if d <= radius {
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

    // Sky / ground environment split (see the 2D counterpart).
    let (t_sky, t_ground) = environment_temperatures(config);
    let split_environment = t_sky != t_ground;
    let t_sky_fourth = fourth_power(t_sky);
    let t_ground_fourth = fourth_power(t_ground);

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
        let q_env = if split_environment {
            // Split the body's surface between the sky and the ground.
            let f_up = body_i
                .shape
                .as_ref()
                .map(|s| s.up_facing_fraction())
                .unwrap_or(0.5);
            eps_i
                * sigma
                * a_surf
                * (f_up * (t_i_fourth - t_sky_fourth)
                    + (1.0 - f_up) * (t_i_fourth - t_ground_fourth))
        } else {
            eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth)
        };
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
            let mut occluded = false;
            if config.radiation_occlusion && is_occluded_3d(bodies, i, j) {
                occluded = true;
            }

            let q_rad = match config.view_factor_model {
                ViewFactorModel::Legacy => {
                    if occluded {
                        if config.occluder_transmittance <= 0.0 {
                            // Fully opaque: the historical behaviour.
                            continue;
                        }
                        eps_eff
                            * sigma
                            * a_surf
                            * view_factor
                            * (t_i_fourth - t_j_fourth)
                            * config.occluder_transmittance
                    } else {
                        eps_eff * sigma * a_surf * view_factor * (t_i_fourth - t_j_fourth)
                    }
                }
                ViewFactorModel::Symmetric => {
                    // Order-independent view factor (see the 2D
                    // counterpart).
                    let f = (r_i * r_j / (dist * dist)).min(1.0);
                    // The mean of the two emitting areas.
                    let a_eff = 0.5 * (a_surf + surface_area[j]) * f;
                    let q = eps_eff * sigma * a_eff * (t_i_fourth - t_j_fourth);
                    if occluded {
                        q * config.occluder_transmittance
                    } else {
                        q
                    }
                }
            };

            heat_flux[i] -= q_rad;
            heat_flux[j] += q_rad;
        }
    }
}

/// The sky and ground temperatures used by the environment radiation.
///
/// A non-positive sky or ground temperature means "use the ambient
/// temperature". When the two resolve to the same value (the default)
/// the environment exchange is the historical single-temperature one.
fn environment_temperatures(config: &ThermalConfig2D) -> (f32, f32) {
    let ambient = config.ambient_temperature;
    let sky = if config.sky_temperature > 0.0 {
        config.sky_temperature
    } else {
        ambient
    };
    let ground = if config.ground_temperature > 0.0 {
        config.ground_temperature
    } else {
        ambient
    };
    (sky, ground)
}

/// `t^4`, computed the same way the model has always computed it.
fn fourth_power(t: f32) -> f32 {
    let sq = t * t;
    sq * sq
}
