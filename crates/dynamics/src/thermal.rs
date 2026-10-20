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
//! Opt-in sub-features (all default to off, so default trajectories are
//! bit-identical):
//! - **Radiation occlusion**: a third body between two bodies blocks their
//!   radiative exchange (coarse segment/sphere test).
//! - **Forced convection**: the convection coefficient grows with the
//!   body's speed (`h + h_forced·|v|`).
//! - **Thermal expansion**: the shape is scaled by `1 + alpha·(T − T_ref)`.
//!   The applied factor is tracked per body ([`RigidBody2::expansion_scale`])
//!   and applied *relative to the authored shape*, so the factor never
//!   compounds across steps.
//!
//! Determinism: pure per-body computation in index order, no RNG, no
//! allocation (buffers are pooled and recycled).

use super::body::{RigidBody2, RigidBody3};
use super::solver::ContactConstraint;
use super::solver3d::ContactConstraint3D;
#[cfg(test)]
use glam::Vec2;
use phieng_math::{DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MIN_TEMPERATURE, STEFAN_BOLTZMANN};

/// Which exchange model the body-to-body radiation uses.
///
/// [`ViewFactorModel::Legacy`] is the historical model and the default:
/// the view factor is `min((r_j / d)^2, 1)` (the small-source
/// approximation, using the *second* body of the pair) and the exchange
/// rate is `eps_eff * sigma * A * F * (T_i^4 - T_j^4)`.
///
/// [`ViewFactorModel::Symmetric`] replaces that with a model that does
/// not depend on the order of the pair: the view factor is
/// `min(r_i * r_j / d^2, 1)` and the exchange rate is
/// `sigma * A_eff * (T_i^4 - T_j^4) / (1/eps_i + 1/eps_j - 1)` with
/// `A_eff = A_i * F`. This is the standard parallel-plate gray-body
/// result generalised to unequal areas. For two equal-sized bodies it
/// reduces *exactly* to the legacy formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewFactorModel {
    /// Historical model: order-dependent view factor (the default).
    #[default]
    Legacy,
    /// Order-independent view factor and exchange rate.
    Symmetric,
}

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
    /// Block body-to-body radiation when another body lies between the
    /// pair (line-of-sight test). Off by default: it is O(n) per pair and
    /// changes exchanged heat, so it is opt-in.
    pub radiation_occlusion: bool,
    /// Extra convection coefficient per unit of relative speed
    /// (W/(m²·K) per m/s). Models forced convection over a moving
    /// surface. Off by default (0.0 = free convection only).
    pub forced_convection_coefficient: f32,
    /// Linear thermal-expansion coefficient (1/K). Scales a body's shape by
    /// `1 + alpha * (T - reference_temperature)`. Off by default
    /// (0.0 = no expansion) so default trajectories are bit-identical.
    pub expansion_coefficient: f32,
    /// Reference temperature (K) the expansion is measured against.
    pub expansion_reference_temperature: f32,
    /// Exchange model used by the body-to-body radiation.
    ///
    /// `Legacy` (the default) keeps the historical, order-dependent
    /// view factor; `Symmetric` uses the order-independent model.
    /// Switching models changes the exchanged heat, so it is opt-in.
    pub view_factor_model: ViewFactorModel,
    /// Sky temperature (K) used by the environment radiation.
    ///
    /// `<= 0` means "use `ambient_temperature`". When the sky and the
    /// ground differ, a body's environment exchange is split between
    /// them by its up-facing area fraction; when they are equal (the
    /// default) the historical single-temperature exchange is used and
    /// the result is bit-identical to the pre-split model.
    pub sky_temperature: f32,
    /// Ground temperature (K) used by the environment radiation.
    /// `<= 0` means "use `ambient_temperature`".
    pub ground_temperature: f32,
    /// Fraction of the radiation that passes through an occluding body
    /// (0.0 = fully opaque, the historical behaviour; 1.0 = the occluder
    /// does not block at all).
    pub occluder_transmittance: f32,
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
            radiation_occlusion: false,
            forced_convection_coefficient: 0.0,
            expansion_coefficient: 0.0,
            expansion_reference_temperature: DEFAULT_TEMPERATURE,
            view_factor_model: ViewFactorModel::Legacy,
            sky_temperature: 0.0,
            ground_temperature: 0.0,
            occluder_transmittance: 0.0,
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
        super::radiation::compute_radiation_2d(
            bodies,
            config,
            &mut state.heat_flux,
            &state.surface_area,
        );
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

fn compute_convection_2d(
    bodies: &[RigidBody2],
    config: &ThermalConfig2D,
    state: &mut ThermalState2,
) {
    let h = config.convection_coefficient;
    let t_amb = config.ambient_temperature;
    let h_forced = config.forced_convection_coefficient;

    for (i, body) in bodies.iter().enumerate() {
        if body.shape.is_none() {
            continue;
        }
        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }
        // Forced convection: the coefficient grows with the surface speed
        // (a body moving through still air sheds heat faster).
        let h_eff = if h_forced > 0.0 {
            h + h_forced * body.velocity.length()
        } else {
            h
        };
        // q_conv is the convective *loss* (Newton's law, positive when
        // hotter than ambient); heat_flux accumulates heat gained.
        let q_conv = h_eff * a_surf * (body.temperature - t_amb);
        state.heat_flux[i] -= q_conv;
    }
}

/// Target linear scale factor for a body at `temperature`, given the
/// expansion coefficient and reference temperature.
///
/// Returns 1.0 (no expansion) when the coefficient is 0, and clamps to a
/// small positive floor so a body never collapses to a degenerate shape.
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

pub fn expansion_factor(config: &ThermalConfig2D, temperature: f32) -> f32 {
    let alpha = config.expansion_coefficient;
    if !alpha.is_finite() || alpha == 0.0 {
        return 1.0;
    }
    let dt = temperature - config.expansion_reference_temperature;
    if !dt.is_finite() {
        return 1.0;
    }
    let factor = 1.0 + alpha * dt;
    // A non-positive or non-finite factor would produce a degenerate
    // shape (zero radius, inverted hull); clamp instead.
    if !factor.is_finite() || factor <= 0.0 {
        return 1.0;
    }
    factor.clamp(0.01, 100.0)
}

/// Rescale a body's shape from its currently-applied expansion factor to
/// the target factor.
///
/// The shape is multiplied by `target / applied` (not by `target`), so
/// the result is always `authored_shape * target` regardless of how the
/// body reached its current temperature. Applying `target` directly
/// would compound the factor on every step (a body held at a constant
/// temperature would grow without bound).
fn rescale_for_expansion_2d(body: &mut RigidBody2, target: f32) {
    let applied = body.expansion_scale;
    if !applied.is_finite() || applied <= 0.0 {
        // Defensive: an invalid tracker would make the ratio NaN.
        body.expansion_scale = 1.0;
        return;
    }
    if (target - applied).abs() <= f32::EPSILON {
        return;
    }
    let Some(shape) = body.shape.clone() else {
        return;
    };
    let ratio = target / applied;
    if !ratio.is_finite() || ratio <= 0.0 {
        return;
    }
    let scaled = shape.scaled(ratio);
    // `set_shape` recomputes mass/inertia from the (unchanged) density,
    // so the body stays consistent with its new size.
    let density = body.material.density;
    if body.set_shape(scaled, density).is_ok() {
        body.material.density = density;
        body.expansion_scale = target;
    }
}

/// Apply thermal expansion to 2D bodies: rescale each shape to match its
/// expansion factor and refresh the mass properties.
///
/// Off by default (`expansion_coefficient == 0`), in which case this is a
/// no-op and every trajectory stays bit-identical.
pub fn apply_expansion_2d(bodies: &mut [RigidBody2], config: &ThermalConfig2D) {
    if !config.expansion_coefficient.is_finite() || config.expansion_coefficient == 0.0 {
        return;
    }
    for body in bodies.iter_mut() {
        let target = expansion_factor(config, body.temperature);
        rescale_for_expansion_2d(body, target);
    }
}

/// Apply thermal expansion to 3D bodies. See [`apply_expansion_2d`].
pub fn apply_expansion_3d(bodies: &mut [RigidBody3], config: &ThermalConfig3D) {
    if !config.expansion_coefficient.is_finite() || config.expansion_coefficient == 0.0 {
        return;
    }
    for body in bodies.iter_mut() {
        let applied = body.expansion_scale;
        if !applied.is_finite() || applied <= 0.0 {
            body.expansion_scale = 1.0;
            continue;
        }
        let target = expansion_factor(config, body.temperature);
        if (target - applied).abs() <= f32::EPSILON {
            continue;
        }
        let Some(shape) = body.shape.clone() else {
            continue;
        };
        let ratio = target / applied;
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let scaled = shape.scaled(ratio);
        let density = body.material.density;
        if body.set_shape(scaled, density).is_ok() {
            body.material.density = density;
            body.expansion_scale = target;
        }
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
        super::radiation::compute_radiation_3d(
            bodies,
            config,
            &mut state.heat_flux,
            &state.surface_area,
        );
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

fn compute_convection_3d(
    bodies: &[RigidBody3],
    config: &ThermalConfig3D,
    state: &mut ThermalState3,
) {
    let h = config.convection_coefficient;
    let t_amb = config.ambient_temperature;
    let h_forced = config.forced_convection_coefficient;

    for (i, body) in bodies.iter().enumerate() {
        if body.shape.is_none() {
            continue;
        }
        let a_surf = state.surface_area[i];
        if a_surf <= 0.0 {
            continue;
        }
        // Forced convection (3D): see `compute_convection_2d`.
        let h_eff = if h_forced > 0.0 {
            h + h_forced * body.velocity.length()
        } else {
            h
        };
        // q_conv is the convective *loss* (Newton's law, positive when
        // hotter than ambient); heat_flux accumulates heat gained.
        let q_conv = h_eff * a_surf * (body.temperature - t_amb);
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

    // ---- Radiation occlusion (T18) ----

    /// T18: a body between two others blocks body-to-body radiation when
    /// occlusion is enabled.
    ///
    /// The A/B uses the *same* three bodies in both runs and only flips
    /// the occlusion flag, so the blocker's own heat exchange (which is
    /// unaffected: it is not between any pair that changes) cancels out
    /// and the difference isolates the blocked 0-1 exchange.
    #[test]
    fn t18_radiation_occlusion_blocks_intermediate_body() {
        let make = |x: f32, temp: f32| {
            let mut b = make_body_2d(0, temp, 10.0);
            b.transform.position = Vec2::new(x, 0.0);
            b
        };
        // Hot body, cold body, and a blocker sitting between them.
        let build = |occlusion: bool| {
            let mut bodies = vec![make(0.0, 600.0), make(2.0, 300.0), make(1.0, 400.0)];
            let config = ThermalConfig2D {
                enabled: true,
                radiation_enabled: true,
                conduction_enabled: false,
                convection_enabled: false,
                radiation_occlusion: occlusion,
                ambient_temperature: 293.15,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut state = ThermalState2::new();
            for _ in 0..10 {
                step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            }
            bodies[0].temperature
        };

        let blocked = build(true);
        let unblocked = build(false);
        assert!(
            blocked > unblocked,
            "occlusion must reduce body-to-body exchange: blocked {} vs unblocked {}",
            blocked,
            unblocked
        );
    }

    // ---- Forced convection (T19) ----

    /// T19: a moving body loses more heat than a stationary one when
    /// forced convection is enabled.
    #[test]
    fn t19_forced_convection_cools_moving_body_faster() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: true,
            forced_convection_coefficient: 50.0,
            ambient_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };

        let run = |speed: f32| {
            let mut bodies = vec![make_body_2d(0, 500.0, 10.0)];
            bodies[0].velocity = Vec2::new(speed, 0.0);
            let mut state = ThermalState2::new();
            for _ in 0..60 {
                step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            }
            bodies[0].temperature
        };

        let still = run(0.0);
        let moving = run(10.0);
        assert!(
            moving < still,
            "moving body must cool faster: moving {} vs still {}",
            moving,
            still
        );
    }

    // ---- Thermal expansion (T20, T21) ----

    /// T20: a hot body expands when an expansion coefficient is set.
    #[test]
    fn t20_expansion_scales_shape_with_temperature() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: false,
            expansion_coefficient: 1.0e-3,
            expansion_reference_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };

        let mut bodies = vec![make_body_2d(0, 500.0, 10.0)];
        let radius_before = bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        let mut state = ThermalState2::new();
        for _ in 0..10 {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            apply_expansion_2d(&mut bodies, &config);
        }
        let radius_after = bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);

        assert!(
            radius_after > radius_before,
            "hot body must expand: {} -> {}",
            radius_before,
            radius_after
        );
    }

    /// T22: expansion is applied by the world step, not only by the
    /// thermal phase in isolation.
    ///
    /// A body that is hot and never exchanges heat must still reach (and
    /// hold) the target scale `1 + alpha * (T - T_ref)`.
    #[test]
    fn t22_expansion_applies_in_world_step() {
        use crate::body::{RigidBody2, Shape2};
        use crate::world::World2;

        let mut w = World2::new();
        w.thermal.config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: false,
            expansion_coefficient: 1.0e-3,
            expansion_reference_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        body.temperature = 500.0;
        w.add_body(body).expect("adds body");

        let before = w.bodies[0].shape.as_ref().unwrap().bounding_radius();
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        let after = w.bodies[0].shape.as_ref().unwrap().bounding_radius();

        let expected = 1.0 + 1.0e-3 * (500.0 - 293.15);
        assert!(
            after > before,
            "hot body must expand: {} -> {}",
            before,
            after
        );
        assert!(
            (after - expected).abs() < 1e-4,
            "expected scale {}, got {}",
            expected,
            after
        );
    }

    /// T24: 3D expansion mirrors the 2D contract.
    ///
    /// A hot sphere must reach the target scale, and a hot convex mesh
    /// must expand without the mesh recentering drifting.
    #[test]
    fn t24_expansion_3d() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: false,
            expansion_coefficient: 1.0e-3,
            expansion_reference_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };

        // Sphere.
        let mut body = crate::body::RigidBody3::new(0);
        body.set_shape(crate::body::Shape3::Sphere { radius: 1.0 }, 1.0)
            .expect("valid shape");
        body.temperature = 500.0;
        let mut bodies = vec![body];
        let mut state = ThermalState2::new();
        for _ in 0..10 {
            step_thermal_3d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            apply_expansion_3d(&mut bodies, &config);
        }
        let radius = bodies[0].shape.as_ref().unwrap().bounding_radius();
        let expected = 1.0 + 1.0e-3 * (500.0 - 293.15);
        assert!(
            (radius - expected).abs() < 1e-4,
            "expected scale {}, got {}",
            expected,
            radius
        );

        // Convex mesh: the vertices must scale, and the recentering in
        // `set_shape` must not drift the mesh away from the body origin.
        let mut mesh = crate::body::RigidBody3::new(0);
        mesh.set_shape(
            crate::body::Shape3::ConvexHull {
                vertices: vec![
                    glam::Vec3::new(-1.0, -1.0, -1.0),
                    glam::Vec3::new(1.0, -1.0, -1.0),
                    glam::Vec3::new(1.0, 1.0, -1.0),
                    glam::Vec3::new(-1.0, 1.0, -1.0),
                    glam::Vec3::new(0.0, 0.0, 1.0),
                ],
            },
            1.0,
        )
        .expect("valid shape");
        mesh.temperature = 500.0;
        let radius_before = mesh.shape.as_ref().unwrap().bounding_radius();
        let mut meshes = vec![mesh];
        let mut state2 = ThermalState2::new();
        for _ in 0..10 {
            step_thermal_3d(&mut meshes, &[], &config, &mut state2, 1.0 / 60.0);
            apply_expansion_3d(&mut meshes, &config);
        }
        let radius = meshes[0].shape.as_ref().unwrap().bounding_radius();
        // The hull's bounding radius is not 1.0, so compare the *ratio*.
        assert!(
            (radius / radius_before - expected).abs() < 1e-4,
            "convex hull expected scale {}, got {}",
            expected,
            radius / radius_before
        );
    }

    // ----------------------------------------------------------------
    // Managed-mirror parity block (T26-T37).
    //
    // These mirror the managed C# tests in
    // `tests/PhiEng.Tests/ThermalManagedTests.cs`, which drive the same
    // scenarios through the pure-managed `World2`. The two engines have
    // separate solver implementations, so the assertions are on the
    // qualitative outcome (which way the heat flowed, whether the shape
    // grew), not on bit-equal numbers.
    // ----------------------------------------------------------------

    /// Builds a 2D world with the given thermal config and one dynamic
    /// circle body at `temperature`.
    fn managed_world_2d(config: ThermalConfig2D, temperature: f32) -> crate::world::World2 {
        use crate::body::{RigidBody2, Shape2};
        let mut w = crate::world::World2::new();
        w.thermal.config = config;
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        body.temperature = temperature;
        w.add_body(body).expect("adds body");
        w
    }

    /// T26: conduction between two contacting bodies reaches equilibrium.
    #[test]
    fn t26_managed_conduction_reaches_equilibrium() {
        use crate::body::{RigidBody2, Shape2};
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: false,
            ambient_temperature: 200.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = crate::world::World2::new();
        w.thermal.config = config;
        w.add_static_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 1.0 })
            .expect("adds static body");
        let mut ball = RigidBody2::new(1);
        ball.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        ball.temperature = 400.0;
        w.add_body(ball).expect("adds body");
        for body in w.bodies.iter_mut() {
            body.material.specific_heat = 900.0;
            body.material.thermal_conductivity = 10.0;
        }

        for _ in 0..500 {
            w.step(1.0 / 60.0);
        }

        let hot = w.bodies[1].temperature;
        let cold = w.bodies[0].temperature;
        assert!(hot < 400.0 && hot > 200.0, "hot body must cool: {}", hot);
        assert!(cold >= 200.0, "cold body must warm or hold: {}", cold);
    }

    /// T27: a hot body radiates heat away.
    #[test]
    fn t27_managed_radiation_cools_hot_body() {
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: true,
            convection_enabled: false,
            ambient_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 500.0);
        for _ in 0..100 {
            w.step(1.0 / 60.0);
        }
        assert!(
            w.bodies[0].temperature < 500.0,
            "hot body must radiate heat away: {}",
            w.bodies[0].temperature
        );
    }

    /// T28: a cold body absorbs radiation from the environment.
    #[test]
    fn t28_managed_radiation_warms_cold_body() {
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: true,
            convection_enabled: false,
            ambient_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 200.0);
        for _ in 0..100 {
            w.step(1.0 / 60.0);
        }
        assert!(
            w.bodies[0].temperature > 200.0,
            "cold body must absorb radiation: {}",
            w.bodies[0].temperature
        );
    }

    /// T29: the temperature is clamped to the configured maximum.
    #[test]
    fn t29_managed_temperature_clamped_at_max() {
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: false,
            convection_enabled: true,
            ambient_temperature: 1000.0,
            max_temperature: 400.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 100.0);
        for _ in 0..200 {
            w.step(1.0 / 60.0);
        }
        assert!(
            w.bodies[0].temperature <= 400.0 + 1e-3,
            "temperature must respect the cap: {}",
            w.bodies[0].temperature
        );
    }

    /// T30: a static body is an infinite-capacity thermal reservoir.
    #[test]
    fn t30_managed_static_body_is_reservoir() {
        use crate::body::{RigidBody2, Shape2};
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: true,
            radiation_enabled: false,
            convection_enabled: true,
            ambient_temperature: 200.0,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = crate::world::World2::new();
        w.thermal.config = config;
        w.add_static_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 1.0 })
            .expect("adds static body");
        // The reservoir sits at the ambient temperature.
        w.bodies[0].temperature = 200.0;
        let mut ball = RigidBody2::new(1);
        ball.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        ball.temperature = 400.0;
        w.add_body(ball).expect("adds body");

        for _ in 0..300 {
            w.step(1.0 / 60.0);
        }

        assert_eq!(w.bodies[0].temperature, 200.0);
        assert!(
            w.bodies[1].temperature < 400.0,
            "the dynamic body must cool: {}",
            w.bodies[1].temperature
        );
    }

    /// T31: a disabled thermal config is a no-op.
    #[test]
    fn t31_managed_disabled_config_is_noop() {
        let config = ThermalConfig2D {
            enabled: false,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 400.0);
        for _ in 0..100 {
            w.step(1.0 / 60.0);
        }
        assert_eq!(w.bodies[0].temperature, 400.0);
    }

    /// T32: the managed thermal phase is deterministic.
    #[test]
    fn t32_managed_thermal_determinism() {
        let run = || {
            let config = ThermalConfig2D {
                enabled: true,
                ambient_temperature: 250.0,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut w = managed_world_2d(config, 400.0);
            for _ in 0..100 {
                w.step(1.0 / 60.0);
            }
            w.bodies[0].temperature
        };
        assert_eq!(run(), run());
    }

    /// T33: radiation occlusion blocks the exchange between two bodies
    /// when a third body lies between them.
    #[test]
    fn t33_managed_radiation_occlusion_blocks_exchange() {
        use crate::body::{RigidBody2, Shape2};
        let build = |occlusion: bool| {
            let config = ThermalConfig2D {
                enabled: true,
                conduction_enabled: false,
                radiation_enabled: true,
                convection_enabled: false,
                radiation_occlusion: occlusion,
                ambient_temperature: 293.15,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut w = crate::world::World2::new();
            w.thermal.config = config;
            for (i, (x, temp)) in [(0.0f32, 600.0f32), (2.0, 300.0), (1.0, 400.0)]
                .into_iter()
                .enumerate()
            {
                let mut body = RigidBody2::new(i);
                body.set_shape(Shape2::Circle { radius: 0.4 }, 1.0)
                    .expect("valid shape");
                body.transform.position = Vec2::new(x, 0.0);
                body.temperature = temp;
                w.add_body(body).expect("adds body");
            }
            for _ in 0..10 {
                w.step(1.0 / 60.0);
            }
            w.bodies[0].temperature
        };
        let blocked = build(true);
        let unblocked = build(false);
        assert!(
            blocked > unblocked,
            "occlusion must reduce body-to-body exchange: blocked {} vs unblocked {}",
            blocked,
            unblocked
        );
    }

    /// T34: forced convection cools a moving body faster.
    #[test]
    fn t34_managed_forced_convection_cools_moving_body_faster() {
        use crate::body::{RigidBody2, Shape2};
        let run = |speed: f32| {
            let config = ThermalConfig2D {
                enabled: true,
                conduction_enabled: false,
                radiation_enabled: false,
                convection_enabled: true,
                forced_convection_coefficient: 50.0,
                ambient_temperature: 293.15,
                max_temperature_rate: 10000.0,
                ..Default::default()
            };
            let mut w = crate::world::World2::new();
            w.thermal.config = config;
            let mut body = RigidBody2::new(0);
            body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
                .expect("valid shape");
            body.temperature = 500.0;
            body.velocity = Vec2::new(speed, 0.0);
            w.add_body(body).expect("adds body");
            for _ in 0..60 {
                w.step(1.0 / 60.0);
            }
            w.bodies[0].temperature
        };
        let still = run(0.0);
        let moving = run(10.0);
        assert!(
            moving < still,
            "moving body must cool faster: moving {} vs still {}",
            moving,
            still
        );
    }

    /// T35: thermal expansion scales a hot body's shape to the target
    /// factor, without compounding.
    #[test]
    fn t35_managed_expansion_scales_shape() {
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: false,
            convection_enabled: false,
            expansion_coefficient: 1.0e-3,
            expansion_reference_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 500.0);
        let radius_before = w.bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        let radius_after = w.bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        assert!(
            radius_after > radius_before,
            "hot body must expand: {} -> {}",
            radius_before,
            radius_after
        );
        let expected = 1.0 + 1.0e-3 * (500.0 - 293.15);
        assert!(
            (radius_after - expected).abs() < 1e-4,
            "expected scale {}, got {}",
            expected,
            radius_after
        );
    }

    /// T36: expansion is opt-in — with the default (zero) coefficient the
    /// shape must not change.
    #[test]
    fn t36_managed_expansion_disabled_is_noop() {
        let config = ThermalConfig2D {
            enabled: true,
            conduction_enabled: false,
            radiation_enabled: false,
            convection_enabled: false,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };
        let mut w = managed_world_2d(config, 500.0);
        let radius_before = w.bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        let radius_after = w.bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        assert_eq!(radius_before, radius_after);
    }

    /// T37: the thermal state round-trips through a snapshot (per-body
    /// temperature, material thermal properties, the config and the
    /// applied expansion factor).
    #[test]
    fn t37_managed_thermal_snapshot_roundtrip() {
        use crate::body::{RigidBody2, Shape2};
        let mut w = crate::world::World2::new();
        w.thermal.config = ThermalConfig2D {
            enabled: true,
            ambient_temperature: 320.0,
            convection_coefficient: 12.5,
            radiation_occlusion: true,
            forced_convection_coefficient: 7.25,
            expansion_coefficient: 1.5e-4,
            expansion_reference_temperature: 280.0,
            max_temperature_rate: 42.0,
            ..Default::default()
        };
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        body.temperature = 456.0;
        body.expansion_scale = 1.25;
        body.material.specific_heat = 1234.0;
        body.material.thermal_conductivity = 5.5;
        body.material.emissivity = 0.42;
        w.add_body(body).expect("adds body");

        let bytes = crate::snapshot::world2_to_binary(&w, crate::snapshot::LoadMode::Exact)
            .expect("serializes");
        let back = crate::snapshot::world2_from_binary(&bytes, crate::snapshot::LoadMode::Exact)
            .expect("deserializes");

        assert!(back.thermal.config.enabled);
        assert_eq!(back.thermal.config.ambient_temperature, 320.0);
        assert_eq!(back.thermal.config.convection_coefficient, 12.5);
        assert!(back.thermal.config.radiation_occlusion);
        assert_eq!(back.thermal.config.forced_convection_coefficient, 7.25);
        assert_eq!(back.thermal.config.expansion_coefficient, 1.5e-4);
        assert_eq!(back.thermal.config.expansion_reference_temperature, 280.0);
        assert_eq!(back.thermal.config.max_temperature_rate, 42.0);

        assert_eq!(back.bodies[0].temperature, 456.0);
        assert_eq!(back.bodies[0].expansion_scale, 1.25);
        assert_eq!(back.bodies[0].material.specific_heat, 1234.0);
        assert_eq!(back.bodies[0].material.thermal_conductivity, 5.5);
        assert_eq!(back.bodies[0].material.emissivity, 0.42);
    }

    /// T23: expansion must not compound.
    ///
    /// A body held at a constant temperature has a *fixed* target scale.
    /// Applying the factor directly to the (already-expanded) shape would
    /// multiply it every step, so a body held hot would grow without
    /// bound. The applied factor is tracked per body, so the shape stays
    /// at exactly `authored * target`.
    #[test]
    fn t23_expansion_does_not_compound() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: false,
            expansion_coefficient: 1.0e-3,
            expansion_reference_temperature: 293.15,
            max_temperature_rate: 10000.0,
            ..Default::default()
        };

        let run = |steps: usize| {
            let mut bodies = vec![make_body_2d(0, 500.0, 10.0)];
            let mut state = ThermalState2::new();
            for _ in 0..steps {
                step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
                apply_expansion_2d(&mut bodies, &config);
            }
            bodies[0]
                .shape
                .as_ref()
                .map(|s| s.bounding_radius())
                .unwrap_or(0.0)
        };

        let after_1 = run(1);
        let after_100 = run(100);
        assert_eq!(
            after_1, after_100,
            "a body at a constant temperature must not keep growing"
        );
        // The expected target is 1 + alpha * (T - T_ref).
        let expected = 1.0 + 1.0e-3 * (500.0 - 293.15);
        assert!(
            (after_100 - expected).abs() < 1e-4,
            "expected scale {}, got {}",
            expected,
            after_100
        );
    }

    /// T21: expansion is opt-in. With the default (zero) coefficient the
    /// shape must not change at all, so default trajectories stay
    /// bit-identical.
    #[test]
    fn t21_expansion_disabled_is_noop() {
        let config = ThermalConfig2D {
            enabled: true,
            radiation_enabled: false,
            conduction_enabled: false,
            convection_enabled: false,
            ..Default::default()
        };

        let mut bodies = vec![make_body_2d(0, 500.0, 10.0)];
        let radius_before = bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);
        let mut state = ThermalState2::new();
        for _ in 0..10 {
            step_thermal_2d(&mut bodies, &[], &config, &mut state, 1.0 / 60.0);
            apply_expansion_2d(&mut bodies, &config);
        }
        let radius_after = bodies[0]
            .shape
            .as_ref()
            .map(|s| s.bounding_radius())
            .unwrap_or(0.0);

        assert_eq!(radius_before, radius_after);
    }
}
