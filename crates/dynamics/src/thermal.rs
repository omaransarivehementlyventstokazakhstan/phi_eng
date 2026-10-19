//! Thermal simulation: per-body temperature evolution via conduction,
//! radiation, and convection.
//!
//! Position in [`crate::World2::step`]: a *thermal phase* runs immediately
//! after velocity integration and before sleep check. It reads contact
//! data from the just-completed narrowphase pass and updates body
//! temperatures. It does NOT modify velocity, so it neither wakes sleeping
//! bodies nor is affected by sleep.
//!
//! Heat-transfer mechanisms:
//! - **Conduction**: contact-driven, based on thermal conductivities and
//!   contact geometry.
//! - **Radiation**: long-range, Stefan–Boltzmann law, with distance-based
//!   cutoff.
//! - **Convection**: ambient coupling, Newton's law of cooling.
//!
//! Determinism: pure per-body computation in index order, no RNG, no
//! allocation (buffers are pooled and recycled).

use super::body::{RigidBody2, RigidBody3};
use super::solver::ContactConstraint;
use super::solver3d::ContactConstraint3D;
#[cfg(test)]
use glam::Vec2;
use phieng_math::{DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MIN_TEMPERATURE, STEFAN_BOLTZMANN};

/// Global thermal simulation settings (2D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalConfig2D {
    pub enabled: bool,
    pub ambient_temperature: f32,
    pub stefan_boltzmann: f32,
    pub convection_coefficient: f32,
    pub radiation_enabled: bool,
    pub conduction_enabled: bool,
    pub convection_enabled: bool,
    pub max_temperature: f32,
    pub min_temperature: f32,
    pub radiation_radius_factor: f32,
    pub max_temperature_rate: f32,
}

impl Default for ThermalConfig2D {
    fn default() -> Self {
        Self {
            enabled: false,
            ambient_temperature: DEFAULT_TEMPERATURE,
            stefan_boltzmann: STEFAN_BOLTZMANN,
            convection_coefficient: 10.0,
            radiation_enabled: true,
            conduction_enabled: true,
            convection_enabled: true,
            max_temperature: MAX_TEMPERATURE,
            min_temperature: MIN_TEMPERATURE,
            radiation_radius_factor: 5.0,
            max_temperature_rate: 100.0,
        }
    }
}

impl ThermalConfig2D {
    pub fn is_active(&self) -> bool {
        self.enabled
            && (self.radiation_enabled || self.conduction_enabled || self.convection_enabled)
            && self.ambient_temperature.is_finite()
            && self.ambient_temperature > 0.0
    }
}

/// Global thermal simulation settings (3D).
pub type ThermalConfig3D = ThermalConfig2D;

/// Per-world thermal scratch state (2D).
#[derive(Debug)]
pub struct ThermalState2 {
    pub config: ThermalConfig2D,
    heat_flux: Vec<f32>,
    surface_area: Vec<f32>,
}

impl ThermalState2 {
    pub fn new() -> Self {
        Self {
            config: ThermalConfig2D::default(),
            heat_flux: Vec::new(),
            surface_area: Vec::new(),
        }
    }
}

impl Default for ThermalState2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-world thermal scratch state (3D).
pub type ThermalState3 = ThermalState2;

/// Minimum conduction distance to avoid division by zero.
const MIN_CONDUCTION_DIST: f32 = 1e-4;
/// Conduction scaling factor to make thermal dynamics visible on game
/// timescales (seconds) rather than minutes for realistic materials.
const CONDUCTION_SCALE: f32 = 500.0;

/// Zero the per-body flux accumulator and refresh the cached surface area.
///
/// Written as a zipped iterator pass (not an index loop) so the borrow
/// stays on one body at a time without `bodies[i]` indexing.
fn reset_thermal_scratch<S: SurfaceArea>(bodies: &[S], flux: &mut [f32], areas: &mut [f32]) {
    for ((f, a), body) in flux.iter_mut().zip(areas.iter_mut()).zip(bodies.iter()) {
        *f = 0.0;
        *a = body.surface_area_m2();
    }
}

/// Per-shape surface area, used only by the scratch refresh above.
trait SurfaceArea {
    fn surface_area_m2(&self) -> f32;
}

impl SurfaceArea for RigidBody2 {
    fn surface_area_m2(&self) -> f32 {
        self.shape.as_ref().map(|s| s.surface_area()).unwrap_or(0.0)
    }
}

impl SurfaceArea for RigidBody3 {
    fn surface_area_m2(&self) -> f32 {
        self.shape.as_ref().map(|s| s.surface_area()).unwrap_or(0.0)
    }
}

/// Run the thermal phase for a 2D world.
///
/// `contacts` are the narrowphase results from this step. Sleeping bodies
/// still participate in heat transfer.
pub fn step_thermal_2d(
    bodies: &mut [RigidBody2],
    contacts: &[ContactConstraint],
    config: &ThermalConfig2D,
    state: &mut ThermalState2,
    dt: f32,
) {
    if !config.is_active() || dt <= 0.0 || bodies.is_empty() {
        return;
    }

    let n = bodies.len();
    if state.heat_flux.len() != n {
        state.heat_flux.resize(n, 0.0);
    }
    if state.surface_area.len() != n {
        state.surface_area.resize(n, 0.0);
    }

    reset_thermal_scratch(bodies, &mut state.heat_flux, &mut state.surface_area);

    if config.conduction_enabled {
        compute_conduction_2d(bodies, contacts, state);
    }

    if config.radiation_enabled {
        compute_radiation_2d(bodies, config, state);
    }

    if config.convection_enabled {
        compute_convection_2d(bodies, config, state);
    }

    update_temperatures_2d(bodies, config, state, dt);
}

fn compute_conduction_2d(
    bodies: &[RigidBody2],
    contacts: &[ContactConstraint],
    state: &mut ThermalState2,
) {
    for contact in contacts {
        let a = contact.body_a;
        let b = contact.body_b;
        if a >= bodies.len() || b >= bodies.len() {
            continue;
        }

        let body_a = &bodies[a];
        let body_b = &bodies[b];

        let k_a = body_a.material.thermal_conductivity;
        let k_b = body_b.material.thermal_conductivity;
        if k_a <= 0.0 || k_b <= 0.0 {
            continue;
        }

        let k_eff = (k_a * k_b) / (k_a + k_b);

        let char_a = body_a
            .shape
            .as_ref()
            .map(|s| s.characteristic_length())
            .unwrap_or(MIN_CONDUCTION_DIST);
        let char_b = body_b
            .shape
            .as_ref()
            .map(|s| s.characteristic_length())
            .unwrap_or(MIN_CONDUCTION_DIST);
        let w_contact = char_a.min(char_b).max(MIN_CONDUCTION_DIST);

        let g_cond = CONDUCTION_SCALE * k_eff * w_contact;

        let t_a = body_a.temperature;
        let t_b = body_b.temperature;
        let q_dot = g_cond * (t_a - t_b);

        state.heat_flux[a] -= q_dot;
        state.heat_flux[b] += q_dot;
    }
}

fn compute_radiation_2d(
    bodies: &[RigidBody2],
    config: &ThermalConfig2D,
    state: &mut ThermalState2,
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

        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }

        let t_i = body_i.temperature.clamp(0.0, config.max_temperature);
        let t_i_sq = t_i * t_i;
        let t_i_fourth = t_i_sq * t_i_sq;

        // q_env is the radiative *loss* (positive when hotter than ambient),
        // but heat_flux accumulates heat *gained*, so subtract it.
        let q_env = eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth);
        state.heat_flux[i] -= q_env;

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

            let q_rad = eps_eff * sigma * l_i * view_factor * (t_i_fourth - t_j_fourth);

            state.heat_flux[i] -= q_rad;
            state.heat_flux[j] += q_rad;
        }
    }
}

fn compute_convection_2d(
    bodies: &[RigidBody2],
    config: &ThermalConfig2D,
    state: &mut ThermalState2,
) {
    let h = config.convection_coefficient;
    let t_amb = config.ambient_temperature;

    for (i, body) in bodies.iter().enumerate() {
        if body.shape.is_none() {
            continue;
        }
        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }
        // q_conv is the convective *loss* (Newton's law, positive when
        // hotter than ambient); heat_flux accumulates heat gained.
        let q_conv = h * a_surf * (body.temperature - t_amb);
        state.heat_flux[i] -= q_conv;
    }
}

fn update_temperatures_2d(
    bodies: &mut [RigidBody2],
    config: &ThermalConfig2D,
    state: &mut ThermalState2,
    dt: f32,
) {
    let max_delta = config.max_temperature_rate * dt;
    let min_t = config.min_temperature;
    let max_t = config.max_temperature;

    for (body, flux) in bodies.iter_mut().zip(state.heat_flux.iter()) {
        // Statics are infinite-capacity reservoirs: they exchange heat but
        // never change temperature.
        if body.is_static {
            continue;
        }

        let c_i = body.mass * body.material.specific_heat;
        if c_i <= 0.0 {
            continue;
        }

        let delta_t = ((*flux / c_i) * dt).clamp(-max_delta, max_delta);
        body.temperature = (body.temperature + delta_t).clamp(min_t, max_t);
    }
}

/// Run the thermal phase for a 3D world.
pub fn step_thermal_3d(
    bodies: &mut [RigidBody3],
    contacts: &[ContactConstraint3D],
    config: &ThermalConfig3D,
    state: &mut ThermalState3,
    dt: f32,
) {
    if !config.is_active() || dt <= 0.0 || bodies.is_empty() {
        return;
    }

    let n = bodies.len();
    if state.heat_flux.len() != n {
        state.heat_flux.resize(n, 0.0);
    }
    if state.surface_area.len() != n {
        state.surface_area.resize(n, 0.0);
    }

    reset_thermal_scratch(bodies, &mut state.heat_flux, &mut state.surface_area);

    if config.conduction_enabled {
        compute_conduction_3d(bodies, contacts, state);
    }

    if config.radiation_enabled {
        compute_radiation_3d(bodies, config, state);
    }

    if config.convection_enabled {
        compute_convection_3d(bodies, config, state);
    }

    update_temperatures_3d(bodies, config, state, dt);
}

fn compute_conduction_3d(
    bodies: &[RigidBody3],
    contacts: &[ContactConstraint3D],
    state: &mut ThermalState3,
) {
    for contact in contacts {
        let a = contact.body_a;
        let b = contact.body_b;
        if a >= bodies.len() || b >= bodies.len() {
            continue;
        }

        let body_a = &bodies[a];
        let body_b = &bodies[b];

        let k_a = body_a.material.thermal_conductivity;
        let k_b = body_b.material.thermal_conductivity;
        if k_a <= 0.0 || k_b <= 0.0 {
            continue;
        }

        let k_eff = (k_a * k_b) / (k_a + k_b);

        let r_a = body_a
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(MIN_CONDUCTION_DIST);
        let r_b = body_b
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(MIN_CONDUCTION_DIST);
        let r_min = r_a.min(r_b).max(MIN_CONDUCTION_DIST);

        let d_pen = contact.penetration.max(0.0);
        let a_contact = std::f32::consts::PI * r_min * d_pen.max(0.01 * r_min);
        let d_char = r_min * 0.5;

        let g_cond = CONDUCTION_SCALE * k_eff * a_contact / d_char;

        let t_a = body_a.temperature;
        let t_b = body_b.temperature;
        let q_dot = g_cond * (t_a - t_b);

        state.heat_flux[a] -= q_dot;
        state.heat_flux[b] += q_dot;
    }
}

fn compute_radiation_3d(
    bodies: &[RigidBody3],
    config: &ThermalConfig3D,
    state: &mut ThermalState3,
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

        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }

        let t_i = body_i.temperature.clamp(0.0, config.max_temperature);
        let t_i_sq = t_i * t_i;
        let t_i_fourth = t_i_sq * t_i_sq;

        // q_env is the radiative *loss* (positive when hotter than ambient),
        // but heat_flux accumulates heat *gained*, so subtract it.
        let q_env = eps_i * sigma * a_surf * (t_i_fourth - t_amb_fourth);
        state.heat_flux[i] -= q_env;

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

            let q_rad = eps_eff * sigma * a_surf * view_factor * (t_i_fourth - t_j_fourth);

            state.heat_flux[i] -= q_rad;
            state.heat_flux[j] += q_rad;
        }
    }
}

fn compute_convection_3d(
    bodies: &[RigidBody3],
    config: &ThermalConfig3D,
    state: &mut ThermalState3,
) {
    let h = config.convection_coefficient;
    let t_amb = config.ambient_temperature;

    for (i, body) in bodies.iter().enumerate() {
        if body.shape.is_none() {
            continue;
        }
        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }
        // q_conv is the convective *loss* (Newton's law, positive when
        // hotter than ambient); heat_flux accumulates heat gained.
        let q_conv = h * a_surf * (body.temperature - t_amb);
        state.heat_flux[i] -= q_conv;
    }
}

fn update_temperatures_3d(
    bodies: &mut [RigidBody3],
    config: &ThermalConfig3D,
    state: &mut ThermalState3,
    dt: f32,
) {
    let max_delta = config.max_temperature_rate * dt;
    let min_t = config.min_temperature;
    let max_t = config.max_temperature;

    for (body, flux) in bodies.iter_mut().zip(state.heat_flux.iter()) {
        // Statics are infinite-capacity reservoirs: they exchange heat but
        // never change temperature.
        if body.is_static {
            continue;
        }

        let c_i = body.mass * body.material.specific_heat;
        if c_i <= 0.0 {
            continue;
        }

        let delta_t = ((*flux / c_i) * dt).clamp(-max_delta, max_delta);
        body.temperature = (body.temperature + delta_t).clamp(min_t, max_t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{Shape2, Shape3};

    fn make_body_2d(id: usize, temperature: f32, conductivity: f32) -> RigidBody2 {
        let mut body = RigidBody2::new(id);
        body.temperature = temperature;
        body.material.thermal_conductivity = conductivity;
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0).unwrap();
        body
    }

    fn make_body_3d(id: usize, temperature: f32, conductivity: f32) -> RigidBody3 {
        let mut body = RigidBody3::new(id);
        body.temperature = temperature;
        body.material.thermal_conductivity = conductivity;
        body.set_shape(Shape3::Sphere { radius: 1.0 }, 1.0).unwrap();
        body
    }

    fn make_contact_2d(a: usize, b: usize) -> ContactConstraint {
        ContactConstraint {
            body_a: a,
            body_b: b,
            point: Vec2::ZERO,
            normal: Vec2::X,
            penetration: 0.01,
            friction: 0.5,
            restitution: 0.0,
            normal_impulse: 0.0,
            tangent_impulse: 0.0,
            rest_bias: 0.0,
            mass_normal: 0.0,
            mass_tangent: 0.0,
        }
    }

    fn make_contact_3d(a: usize, b: usize) -> ContactConstraint3D {
        ContactConstraint3D {
            body_a: a,
            body_b: b,
            point: glam::Vec3::ZERO,
            normal: glam::Vec3::X,
            tangent1: glam::Vec3::Y,
            tangent2: glam::Vec3::Z,
            penetration: 0.01,
            friction: 0.5,
            restitution: 0.0,
            normal_impulse: 0.0,
            tangent1_impulse: 0.0,
            tangent2_impulse: 0.0,
            rest_bias: 0.0,
            mass_normal: 0.0,
            mass_tangent1: 0.0,
            mass_tangent2: 0.0,
            inv_inertia_world_a: glam::Mat3::ZERO,
            inv_inertia_world_b: glam::Mat3::ZERO,
        }
    }

    /// Run one body under `config` for `steps` frames at 1/60 s.
    fn run_2d(temp: f32, config: ThermalConfig2D, steps: usize) -> f32 {
        let mut bodies = vec![make_body_2d(0, temp, 1.0)];
        let mut state = ThermalState2::new();
        for _ in 0..steps {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
        }
        bodies[0].temperature
    }

    #[test]
    fn t01_conduction_reaches_equilibrium() {
        let mut bodies = vec![make_body_2d(0, 400.0, 10.0), make_body_2d(1, 200.0, 10.0)];
        let contacts = vec![make_contact_2d(0, 1)];
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: false,
            ambient_temperature: 200.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..500 {
            step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        let diff = (bodies[0].temperature - bodies[1].temperature).abs();
        assert!(diff < 5.0, "Temperatures should converge, diff={}", diff);
    }

    #[test]
    fn t02_conduction_scales_with_conductivity() {
        let run = |k: f32| {
            let mut bodies = vec![make_body_2d(0, 400.0, k), make_body_2d(1, 200.0, k)];
            let contacts = vec![make_contact_2d(0, 1)];
            let config = ThermalConfig2D {
                enabled: true,
                conduction_enabled: true,
                radiation_enabled: false,
                convection_enabled: false,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut state = ThermalState2::new();

            for _ in 0..60 {
                step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
            }
            (bodies[0].temperature - bodies[1].temperature).abs()
        };

        let diff_low = run(1.0);
        let diff_high = run(100.0);
        assert!(
            diff_high < diff_low,
            "Higher conductivity should equilibrate faster: low={}, high={}",
            diff_low,
            diff_high
        );
    }

    #[test]
    fn t03_zero_conductivity_no_heat_flow() {
        let mut bodies = vec![make_body_2d(0, 400.0, 0.0), make_body_2d(1, 200.0, 0.0)];
        let contacts = vec![make_contact_2d(0, 1)];
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: false,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..100 {
            step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        assert_eq!(bodies[0].temperature, 400.0);
        assert_eq!(bodies[1].temperature, 200.0);
    }

    #[test]
    fn t09_temperature_clamped_at_max() {
        let mut bodies = vec![make_body_2d(0, 4999.0, 1.0)];
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 5000.0,
            max_temperature: 5000.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..100 {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
        }

        assert!(bodies[0].temperature <= 5000.0);
    }

    #[test]
    fn t10_temperature_clamped_at_min() {
        let mut bodies = vec![make_body_2d(0, 1.0, 1.0)];
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 0.0,
            min_temperature: 0.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..100 {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
        }

        assert!(bodies[0].temperature >= 0.0);
    }

    #[test]
    fn t11_static_body_is_reservoir() {
        let mut bodies = vec![make_body_2d(0, 400.0, 10.0), {
            let mut b = make_body_2d(1, 200.0, 10.0);
            b.is_static = true;
            b
        }];
        let contacts = vec![make_contact_2d(0, 1)];
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: false,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..500 {
            step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        assert_eq!(bodies[1].temperature, 200.0);
        assert!(bodies[0].temperature > 200.0);
        assert!(bodies[0].temperature < 400.0);
    }

    #[test]
    fn t12_disabled_config_is_noop() {
        let mut bodies = vec![make_body_2d(0, 400.0, 10.0), make_body_2d(1, 200.0, 10.0)];
        let contacts = vec![make_contact_2d(0, 1)];
        let config = ThermalConfig2D::default();
        let mut state = ThermalState2::new();

        for _ in 0..100 {
            step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        assert_eq!(bodies[0].temperature, 400.0);
        assert_eq!(bodies[1].temperature, 200.0);
    }

    #[test]
    fn t13_conduction_disabled() {
        let mut bodies = vec![make_body_2d(0, 400.0, 10.0), make_body_2d(1, 200.0, 10.0)];
        let contacts = vec![make_contact_2d(0, 1)];
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: false,
            convection_enabled: false,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..100 {
            step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        assert_eq!(bodies[0].temperature, 400.0);
        assert_eq!(bodies[1].temperature, 200.0);
    }

    #[test]
    fn t16_determinism() {
        let run = || {
            let mut bodies = vec![make_body_2d(0, 400.0, 10.0), make_body_2d(1, 200.0, 10.0)];
            let contacts = vec![make_contact_2d(0, 1)];
            let config = ThermalConfig2D {
                enabled: true,
                conduction_enabled: true,
                radiation_enabled: true,
                convection_enabled: true,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut state = ThermalState2::new();

            for _ in 0..100 {
                step_thermal_2d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
            }
            bodies[0].temperature
        };

        assert_eq!(run(), run());
    }

    // ---- Radiation (T04, T05) ----

    /// Hotter than ambient: net radiative loss must cool the body.
    #[test]
    fn t04_radiation_cools_hot_body() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let final_t = run_2d(500.0, config, 100);
        assert!(
            final_t < 500.0,
            "hot body must radiate heat away, got {}",
            final_t
        );
    }

    /// Colder than ambient: net radiative gain must warm the body.
    #[test]
    fn t05_radiation_warms_cold_body() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let final_t = run_2d(200.0, config, 100);
        assert!(
            final_t > 200.0,
            "cold body must absorb radiation, got {}",
            final_t
        );
    }

    // ---- Convection (T06) ----

    /// Newton's law of cooling: the body approaches ambient monotonically
    /// and never overshoots past it.
    #[test]
    fn t06_convection_approaches_ambient_monotonically() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: true,
            ambient_temperature: 300.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut bodies = vec![make_body_2d(0, 500.0, 1.0)];
        let mut state = ThermalState2::new();
        let mut prev = bodies[0].temperature;
        for _ in 0..600 {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            let t = bodies[0].temperature;
            assert!(t <= prev + 1e-4, "convection must not heat a hot body");
            assert!(t >= 300.0 - 1e-4, "must not cross below ambient");
            prev = t;
        }
        // After 10 s with h=10, A=2pi, C=pi*900: tau = C/(hA) ~ 45 s,
        // so 10 s is ~20% of the way; require clear but partial decay.
        assert!(prev < 500.0 && prev > 300.0, "final temp {}", prev);
    }

    // ---- Body-to-body radiation cutoff (T07, T08) ----

    fn run_pair_2d(dx: f32, config: ThermalConfig2D, steps: usize) -> (f32, f32) {
        let mut bodies = vec![make_body_2d(0, 400.0, 1.0), make_body_2d(1, 200.0, 1.0)];
        bodies[0].transform.position = Vec2::ZERO;
        bodies[1].transform.position = Vec2::new(dx, 0.0);
        let mut state = ThermalState2::new();
        for _ in 0..steps {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
        }
        (bodies[0].temperature, bodies[1].temperature)
    }

    /// T07: nearby bodies exchange radiative heat.
    #[test]
    fn t07_nearby_bodies_exchange_radiation() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 293.15,
            radiation_radius_factor: 5.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        // Overlapping (dx=1.0) is well inside the cutoff.
        let (hot, cold) = run_pair_2d(1.0, config, 200);
        assert!(
            cold > 200.0,
            "nearby cold body must gain heat, got {}",
            cold
        );
        assert!(hot < 400.0, "nearby hot body must lose heat, got {}", hot);
    }

    /// T08: beyond the cutoff, body-to-body exchange stops.
    #[test]
    fn t08_radiation_cutoff_stops_exchange() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: true,
            conduction_enabled: false,
            convection_enabled: false,
            ambient_temperature: 293.15,
            // Each body has bounding radius 1.0, so the cutoff is
            // 5.0*(1+1) = 10.0 units. Place them at 1000 units apart.
            radiation_radius_factor: 5.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let (hot, cold) = run_pair_2d(1000.0, config, 200);
        // Both still exchange with the 293.15 K ambient independently,
        // so each moves toward ambient but NOT toward each other past the
        // cutoff. The cold body warming above ambient would require
        // body-to-body flux; assert it only got the (weaker) ambient term.
        assert!(cold > 200.0, "cold body still sees ambient, got {}", cold);
        assert!(
            hot > 293.15,
            "hot body must still be above ambient, got {}",
            hot
        );
        // With no body-to-body term the cold body cannot exceed ambient.
        assert!(
            cold <= 293.15 + 1e-3,
            "cold body must not gain from a distant body, got {}",
            cold
        );
    }

    // ---- 3D conduction (uses make_body_3d) ----

    /// 3D conduction moves heat hot→cold through a contact.
    #[test]
    fn t14_3d_conduction_moves_heat() {
        let mut bodies = vec![make_body_3d(0, 400.0, 10.0), make_body_3d(1, 200.0, 10.0)];
        let contacts = vec![make_contact_3d(0, 1)];
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: false,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut state = ThermalState2::new();

        for _ in 0..500 {
            step_thermal_3d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }

        assert!(
            bodies[0].temperature < 400.0,
            "hot 3D body must cool, got {}",
            bodies[0].temperature
        );
        assert!(
            bodies[1].temperature >= 200.0,
            "cold 3D body is a reservoir or warms, got {}",
            bodies[1].temperature
        );
    }

    /// 3D determinism mirrors the 2D contract.
    #[test]
    fn t15_3d_determinism() {
        let run = || {
            let mut bodies = vec![make_body_3d(0, 400.0, 10.0), make_body_3d(1, 200.0, 10.0)];
            let contacts = vec![make_contact_3d(0, 1)];
            let config = ThermalConfig2D {
                enabled: true,
                radiation_enabled: true,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut state = ThermalState2::new();
            for _ in 0..100 {
                step_thermal_3d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
            }
            bodies[0].temperature
        };
        assert_eq!(run(), run());
    }

    /// T14/T15 disabled toggles (mirrors the 2D T12/T13 contract in 3D).
    #[test]
    fn t17_3d_disabled_config_is_noop() {
        let mut bodies = vec![make_body_3d(0, 400.0, 10.0), make_body_3d(1, 200.0, 10.0)];
        let contacts = vec![make_contact_3d(0, 1)];
        let config = ThermalConfig2D::default(); // disabled
        let mut state = ThermalState2::new();
        for _ in 0..100 {
            step_thermal_3d(&mut bodies, &contacts, &config, &mut state, 1.0 / 60.0);
        }
        assert_eq!(bodies[0].temperature, 400.0);
        assert_eq!(bodies[1].temperature, 200.0);
    }
}
