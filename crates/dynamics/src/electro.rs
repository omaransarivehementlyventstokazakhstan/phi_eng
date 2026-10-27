//! The world-facing electromagnetic phase: per-body state lives on the
//! bodies, this module owns the world configuration, the scratch and the
//! field grid, and applies the velocity kicks.
//!
//! See [`crate::world`] for the phase order. The pairwise and field
//! modes are mutually exclusive: the field mode replaces the pairwise
//! sum entirely, and the uniform external fields are added in both.
//!
//! Exertion always uses the body's stored `charge`; `charge_scale`
//! multiplies only the force the body *feels*, exactly like
//! `gravity_scale` does for mass.

use glam::{Vec2, Vec3};
use phieng_electromagnetics::{
    dipole_torque_3, lorentz_accel_2, lorentz_accel_3, pair_force_2, pair_force_3, ElectroConfig2,
    ElectroConfig3, ElectroError, ElectroMode, PoissonGrid2, PoissonGrid3,
};

use super::body::{RigidBody2, RigidBody3};

/// Reusable per-step scratch for the 2D phase.
#[derive(Debug, Clone, Default)]
pub struct ElectroScratch2 {
    positions: Vec<Vec2>,
    charges: Vec<f32>,
    scales: Vec<f32>,
    inv_masses: Vec<f32>,
}

/// Reusable per-step scratch for the 3D phase.
#[derive(Debug, Clone, Default)]
pub struct ElectroScratch3 {
    positions: Vec<Vec3>,
    charges: Vec<f32>,
    scales: Vec<f32>,
    inv_masses: Vec<f32>,
}

/// The electromagnetic state of a 2D world.
#[derive(Debug)]
pub struct ElectroWorldState2 {
    /// The configuration. Assigning it directly is allowed; the grid is
    /// rebuilt lazily when the field configuration changes.
    pub config: ElectroConfig2,
    /// The field grid, when the mode is [`ElectroMode::Field`].
    pub poisson: Option<PoissonGrid2>,
    /// The simulation time the phase has seen.
    pub time: f32,
    /// Whether the last field solve was refused because the grid was too
    /// large to snapshot. Never set by the arithmetic itself.
    pub last_field_stalled: bool,
    scratch: ElectroScratch2,
}

impl Default for ElectroWorldState2 {
    fn default() -> Self {
        Self {
            config: ElectroConfig2::default(),
            poisson: None,
            time: 0.0,
            last_field_stalled: false,
            scratch: ElectroScratch2::default(),
        }
    }
}

impl ElectroWorldState2 {
    /// A fresh state with the default (off) configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when the phase should run.
    pub fn is_active(&self) -> bool {
        self.config.enabled
    }

    /// Validate and install a configuration.
    ///
    /// The grid is not built here; the first step builds it from the
    /// bodies' current positions. Validation is validate-then-commit: a
    /// bad configuration leaves the old one untouched.
    pub fn configure(&mut self, config: ElectroConfig2) -> Result<(), ElectroError> {
        config.validate()?;
        self.config = config;
        self.poisson = None;
        self.time = 0.0;
        self.last_field_stalled = false;
        self.scratch = ElectroScratch2::default();
        Ok(())
    }

    /// The field grid, if one is built.
    pub fn field(&self) -> Option<&PoissonGrid2> {
        self.poisson.as_ref()
    }

    /// Ensure the grid matches the current field configuration, building
    /// and solving it from `bodies` when needed.
    fn ensure_field(&mut self, bodies: &[RigidBody2]) -> Result<(), ElectroError> {
        let want = self.config.field;
        let matches = self.poisson.as_ref().is_some_and(|g| *g.config() == want);
        if matches {
            return Ok(());
        }
        let mut grid = PoissonGrid2::new(want, self.config.coulomb_k)?;
        for body in bodies {
            grid.deposit_charge(body.transform.position, body.electro.charge);
        }
        grid.solve();
        self.poisson = Some(grid);
        Ok(())
    }
}

/// The electromagnetic state of a 3D world.
#[derive(Debug)]
pub struct ElectroWorldState3 {
    /// The configuration.
    pub config: ElectroConfig3,
    /// The field grid, when the mode is [`ElectroMode::Field`].
    pub poisson: Option<PoissonGrid3>,
    /// The simulation time the phase has seen.
    pub time: f32,
    /// Whether the last field solve was refused.
    pub last_field_stalled: bool,
    scratch: ElectroScratch3,
}

impl Default for ElectroWorldState3 {
    fn default() -> Self {
        Self {
            config: ElectroConfig3::default(),
            poisson: None,
            time: 0.0,
            last_field_stalled: false,
            scratch: ElectroScratch3::default(),
        }
    }
}

impl ElectroWorldState3 {
    /// A fresh state with the default (off) configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when the phase should run.
    pub fn is_active(&self) -> bool {
        self.config.enabled
    }

    /// Validate and install a configuration.
    pub fn configure(&mut self, config: ElectroConfig3) -> Result<(), ElectroError> {
        config.validate()?;
        self.config = config;
        self.poisson = None;
        self.time = 0.0;
        self.last_field_stalled = false;
        self.scratch = ElectroScratch3::default();
        Ok(())
    }

    /// The field grid, if one is built.
    pub fn field(&self) -> Option<&PoissonGrid3> {
        self.poisson.as_ref()
    }

    fn ensure_field(&mut self, bodies: &[RigidBody3]) -> Result<(), ElectroError> {
        let want = self.config.field;
        let matches = self.poisson.as_ref().is_some_and(|g| *g.config() == want);
        if matches {
            return Ok(());
        }
        let mut grid = PoissonGrid3::new(want, self.config.coulomb_k)?;
        for body in bodies {
            grid.deposit_charge(body.transform.position, body.electro.charge);
        }
        grid.solve();
        self.poisson = Some(grid);
        Ok(())
    }
}

#[inline]
fn clamp_accel_2(a: Vec2, max: f32) -> Vec2 {
    let l2 = a.length_squared();
    if !l2.is_finite() {
        return Vec2::ZERO;
    }
    if l2 > max * max {
        a * (max / l2.sqrt())
    } else {
        a
    }
}

#[inline]
fn clamp_accel_3(a: Vec3, max: f32) -> Vec3 {
    let l2 = a.length_squared();
    if !l2.is_finite() {
        return Vec3::ZERO;
    }
    if l2 > max * max {
        a * (max / l2.sqrt())
    } else {
        a
    }
}

/// Apply the electromagnetic velocity kick over `h` to every 2D body.
///
/// Idle bodies (static or sleeping) never receive a force; static bodies
/// still contribute their charge to the field. The uniform external
/// fields are added in both modes.
pub fn electro_kick_2(bodies: &mut [RigidBody2], state: &mut ElectroWorldState2, h: f32) {
    let cfg = state.config;
    if !cfg.enabled || h == 0.0 || bodies.is_empty() {
        return;
    }
    if cfg.mode == ElectroMode::Field {
        // Building the grid from the current positions gives the first
        // step a field; later steps reuse the solve from `electro_step`.
        let _ = state.ensure_field(bodies);
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    let softening = cfg.softening;
    let max = cfg.max_acceleration;

    if cfg.mode == ElectroMode::Pairwise {
        let n = bodies.len();
        let scratch = &mut state.scratch;
        scratch.positions.clear();
        scratch.charges.clear();
        scratch.scales.clear();
        scratch.inv_masses.clear();
        scratch.positions.reserve(n);
        scratch.charges.reserve(n);
        scratch.scales.reserve(n);
        scratch.inv_masses.reserve(n);
        for body in bodies.iter() {
            scratch.positions.push(body.transform.position);
            scratch.charges.push(body.electro.charge);
            scratch.scales.push(body.electro.charge_scale);
        }
        for (i, body) in bodies.iter_mut().enumerate() {
            if body.is_static || body.is_sleeping {
                continue;
            }
            let q_eff = scratch.charges[i] * scratch.scales[i];
            let inv_m = body.inv_mass;
            let mut acc = Vec2::ZERO;
            if q_eff != 0.0 {
                let p = scratch.positions[i];
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let qj = scratch.charges[j];
                    if qj == 0.0 {
                        continue;
                    }
                    acc += pair_force_2(p, q_eff, scratch.positions[j], qj, k_eff, softening);
                }
                // Uniform electric field.
                acc += cfg.uniform_e * q_eff;
                // Uniform magnetic field (Bz out of plane).
                let bz = cfg.uniform_b;
                if bz != 0.0 {
                    acc += lorentz_accel_2(body.velocity, q_eff, Vec2::ZERO, bz);
                }
            }
            let a = clamp_accel_2(acc * inv_m, max);
            body.velocity += a * h;
        }
        return;
    }

    if cfg.mode == ElectroMode::Field {
        let Some(grid) = state.poisson.as_ref() else {
            return;
        };
        for body in bodies.iter_mut() {
            if body.is_static || body.is_sleeping {
                continue;
            }
            let q_eff = body.electro.charge * body.electro.charge_scale;
            if q_eff == 0.0 {
                continue;
            }
            let e = grid.field_at(body.transform.position) + cfg.uniform_e;
            let a = lorentz_accel_2(body.velocity, q_eff, e, cfg.uniform_b);
            body.velocity += clamp_accel_2(a * body.inv_mass, max) * h;
        }
    }
}

/// Advance the per-step electromagnetic bookkeeping: drain grounded
/// conductors and, in field mode, rebuild and solve the grid from the
/// bodies' current positions.
pub fn electro_step_2(bodies: &mut [RigidBody2], state: &mut ElectroWorldState2, dt: f32) {
    if !state.config.enabled {
        return;
    }
    drain_grounded_2(bodies, dt);
    if state.config.mode == ElectroMode::Field {
        match state.ensure_field(bodies) {
            Ok(()) => {
                if let Some(grid) = state.poisson.as_mut() {
                    grid.clear_sources();
                    for body in bodies.iter() {
                        grid.deposit_charge(body.transform.position, body.electro.charge);
                    }
                    grid.solve();
                }
            }
            Err(_) => {
                state.last_field_stalled = true;
                state.poisson = None;
            }
        }
    }
    state.time += dt;
}

fn drain_grounded_2(bodies: &mut [RigidBody2], dt: f32) {
    for body in bodies.iter_mut() {
        let e = &mut body.electro;
        if !e.is_conductor || !e.is_grounded || e.charge == 0.0 {
            continue;
        }
        let tau = e.capacitance * e.resistance;
        if tau.is_finite() && tau > 0.0 && dt.is_finite() && dt > 0.0 {
            e.charge *= phieng_math::exponential::exp(-dt / tau);
        } else {
            e.charge = 0.0;
        }
    }
}

/// Apply the electromagnetic velocity kick over `h` to every 3D body.
pub fn electro_kick_3(bodies: &mut [RigidBody3], state: &mut ElectroWorldState3, h: f32) {
    let cfg = state.config;
    if !cfg.enabled || h == 0.0 || bodies.is_empty() {
        return;
    }
    if cfg.mode == ElectroMode::Field {
        let _ = state.ensure_field(bodies);
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    let softening = cfg.softening;
    let max = cfg.max_acceleration;

    if cfg.mode == ElectroMode::Pairwise {
        let n = bodies.len();
        let scratch = &mut state.scratch;
        scratch.positions.clear();
        scratch.charges.clear();
        scratch.scales.clear();
        scratch.inv_masses.clear();
        scratch.positions.reserve(n);
        scratch.charges.reserve(n);
        scratch.scales.reserve(n);
        scratch.inv_masses.reserve(n);
        for body in bodies.iter() {
            scratch.positions.push(body.transform.position);
            scratch.charges.push(body.electro.charge);
            scratch.scales.push(body.electro.charge_scale);
            scratch.inv_masses.push(body.inv_mass);
        }
        for (i, body) in bodies.iter_mut().enumerate() {
            if body.is_static || body.is_sleeping {
                continue;
            }
            let q_eff = scratch.charges[i] * scratch.scales[i];
            let inv_m = scratch.inv_masses[i];
            let mut acc = Vec3::ZERO;
            if q_eff != 0.0 {
                let p = scratch.positions[i];
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let qj = scratch.charges[j];
                    if qj == 0.0 {
                        continue;
                    }
                    acc += pair_force_3(p, q_eff, scratch.positions[j], qj, k_eff, softening);
                }
                acc += cfg.uniform_e * q_eff;
                if cfg.uniform_b != Vec3::ZERO {
                    acc += lorentz_accel_3(body.velocity, q_eff, Vec3::ZERO, cfg.uniform_b);
                }
            }
            let a = clamp_accel_3(acc * inv_m, max);
            body.velocity += a * h;
            apply_dipole_torque_3(body, cfg.uniform_b, h);
        }
        return;
    }

    if cfg.mode == ElectroMode::Field {
        let Some(grid) = state.poisson.as_ref() else {
            return;
        };
        for body in bodies.iter_mut() {
            if body.is_static || body.is_sleeping {
                continue;
            }
            let q_eff = body.electro.charge * body.electro.charge_scale;
            if q_eff != 0.0 {
                let e = grid.field_at(body.transform.position) + cfg.uniform_e;
                let a = lorentz_accel_3(body.velocity, q_eff, e, cfg.uniform_b);
                body.velocity += clamp_accel_3(a * body.inv_mass, max) * h;
            }
            apply_dipole_torque_3(body, cfg.uniform_b, h);
        }
    }
}

/// See [`electro_step_2`].
pub fn electro_step_3(bodies: &mut [RigidBody3], state: &mut ElectroWorldState3, dt: f32) {
    if !state.config.enabled {
        return;
    }
    drain_grounded_3(bodies, dt);
    if state.config.mode == ElectroMode::Field {
        match state.ensure_field(bodies) {
            Ok(()) => {
                if let Some(grid) = state.poisson.as_mut() {
                    grid.clear_sources();
                    for body in bodies.iter() {
                        grid.deposit_charge(body.transform.position, body.electro.charge);
                    }
                    grid.solve();
                }
            }
            Err(_) => {
                state.last_field_stalled = true;
                state.poisson = None;
            }
        }
    }
    state.time += dt;
}

fn drain_grounded_3(bodies: &mut [RigidBody3], dt: f32) {
    for body in bodies.iter_mut() {
        let e = &mut body.electro;
        if !e.is_conductor || !e.is_grounded || e.charge == 0.0 {
            continue;
        }
        let tau = e.capacitance * e.resistance;
        if tau.is_finite() && tau > 0.0 && dt.is_finite() && dt > 0.0 {
            e.charge *= phieng_math::exponential::exp(-dt / tau);
        } else {
            e.charge = 0.0;
        }
    }
}

fn apply_dipole_torque_3(body: &mut RigidBody3, b: Vec3, h: f32) {
    if b == Vec3::ZERO || body.electro.magnetic_moment == Vec3::ZERO {
        return;
    }
    let torque = dipole_torque_3(body.electro.magnetic_moment, b);
    body.angular_velocity += body.world_inv_inertia() * torque * h;
}

/// The electric field at a world point, including the uniform external
/// field. In field mode the grid is sampled when it exists; otherwise the
/// stored charges are summed directly (so a query never returns a stale
/// grid).
pub fn electric_field_at_2(bodies: &[RigidBody2], state: &ElectroWorldState2, p: Vec2) -> Vec2 {
    let cfg = state.config;
    let mut e = cfg.uniform_e;
    if cfg.mode == ElectroMode::Field {
        if let Some(grid) = state.field() {
            return grid.field_at(p) + cfg.uniform_e;
        }
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    for body in bodies {
        e += pair_force_2(
            p,
            1.0,
            body.transform.position,
            body.electro.charge,
            k_eff,
            cfg.softening,
        );
    }
    e
}

/// The electric potential at a world point. The uniform field uses the
/// gauge `V(0) = 0`, so it contributes `-E . p`.
pub fn electric_potential_at_2(bodies: &[RigidBody2], state: &ElectroWorldState2, p: Vec2) -> f32 {
    let cfg = state.config;
    if cfg.mode == ElectroMode::Field {
        if let Some(grid) = state.field() {
            return grid.potential_at(p) - cfg.uniform_e.dot(p);
        }
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    let mut v = -cfg.uniform_e.dot(p);
    for body in bodies {
        let d = p - body.transform.position;
        let r2 = d.length_squared() + cfg.softening * cfg.softening;
        if r2 < phieng_math::EPS_SQ || !r2.is_finite() {
            continue;
        }
        v += k_eff * body.electro.charge / r2.sqrt();
    }
    v
}

/// 3D counterpart of [`electric_field_at_2`].
pub fn electric_field_at_3(bodies: &[RigidBody3], state: &ElectroWorldState3, p: Vec3) -> Vec3 {
    let cfg = state.config;
    let mut e = cfg.uniform_e;
    if cfg.mode == ElectroMode::Field {
        if let Some(grid) = state.field() {
            return grid.field_at(p) + cfg.uniform_e;
        }
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    for body in bodies {
        e += pair_force_3(
            p,
            1.0,
            body.transform.position,
            body.electro.charge,
            k_eff,
            cfg.softening,
        );
    }
    e
}

/// 3D counterpart of [`electric_potential_at_2`].
pub fn electric_potential_at_3(bodies: &[RigidBody3], state: &ElectroWorldState3, p: Vec3) -> f32 {
    let cfg = state.config;
    if cfg.mode == ElectroMode::Field {
        if let Some(grid) = state.field() {
            return grid.potential_at(p) - cfg.uniform_e.dot(p);
        }
    }
    let k_eff = cfg.coulomb_k / cfg.permittivity;
    let mut v = -cfg.uniform_e.dot(p);
    for body in bodies {
        let d = p - body.transform.position;
        let r2 = d.length_squared() + cfg.softening * cfg.softening;
        if r2 < phieng_math::EPS_SQ || !r2.is_finite() {
            continue;
        }
        v += k_eff * body.electro.charge / r2.sqrt();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{Shape2, Shape3};

    fn world2_with(charges: &[(f32, f32)], config: ElectroConfig2) -> (crate::World2, Vec<usize>) {
        let mut w = crate::World2::new();
        let mut ids = Vec::new();
        for &(x, q) in charges {
            let id = w
                .add_dynamic_body(glam::Vec2::new(x, 0.0), Shape2::Circle { radius: 0.1 }, 1.0)
                .unwrap();
            w.bodies[id].electro.charge = q;
            ids.push(id);
        }
        w.electro.configure(config).unwrap();
        (w, ids)
    }

    fn pairwise_on() -> ElectroConfig2 {
        ElectroConfig2 {
            enabled: true,
            mode: ElectroMode::Pairwise,
            ..Default::default()
        }
    }

    #[test]
    fn opposite_charges_accelerate_toward_each_other() {
        let (mut w, ids) = world2_with(&[(-1.0, 1.0), (1.0, -1.0)], pairwise_on());
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        let a = w.bodies[ids[0]].transform.position.x;
        let b = w.bodies[ids[1]].transform.position.x;
        assert!(a > -1.0, "left body did not move right: {a}");
        assert!(b < 1.0, "right body did not move left: {b}");
    }

    #[test]
    fn default_electro_is_a_noop() {
        let mut w = crate::World2::new();
        let id = w
            .add_dynamic_body(
                glam::Vec2::new(0.0, 5.0),
                Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .unwrap();
        w.bodies[id].electro.charge = 1.0;
        let mut reference = crate::World2::new();
        let rid = reference
            .add_dynamic_body(
                glam::Vec2::new(0.0, 5.0),
                Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .unwrap();
        reference.bodies[rid].electro.charge = 1.0;
        for _ in 0..30 {
            w.step(1.0 / 60.0);
            reference.step(1.0 / 60.0);
        }
        assert_eq!(
            w.bodies[id].transform.position,
            reference.bodies[rid].transform.position
        );
    }

    #[test]
    fn field_mode_solves_and_pushes() {
        let mut cfg = pairwise_on();
        cfg.mode = ElectroMode::Field;
        cfg.field.nx = 32;
        cfg.field.ny = 32;
        cfg.field.dx = 0.2;
        cfg.field.origin = glam::Vec2::splat(-3.2);
        cfg.field.iterations = 100;
        let (mut w, ids) = world2_with(&[(-1.0, 1.0), (1.0, 1.0)], cfg);
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        let left = w.bodies[ids[0]].transform.position.x;
        let right = w.bodies[ids[1]].transform.position.x;
        assert!(left < -1.0, "left body did not move away: {left}");
        assert!(right > 1.0, "right body did not move away: {right}");
        assert!(w.electro.field().is_some());
    }

    #[test]
    fn grounded_conductor_drains() {
        let mut cfg = pairwise_on();
        cfg.mode = ElectroMode::Off;
        let mut w = crate::World2::new();
        let id = w
            .add_dynamic_body(
                glam::Vec2::new(0.0, 0.0),
                Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .unwrap();
        w.bodies[id].electro = phieng_electromagnetics::ElectroState2 {
            charge: 5.0,
            is_conductor: true,
            is_grounded: true,
            resistance: 0.0,
            ..Default::default()
        };
        w.electro.configure(cfg).unwrap();
        w.step(1.0 / 60.0);
        assert_eq!(w.bodies[id].electro.charge, 0.0);
    }

    #[test]
    fn three_d_opposite_charges_attract() {
        let mut w = crate::World3::new();
        let mut ids = Vec::new();
        for (x, q) in [(-1.0f32, 1.0f32), (1.0, -1.0)] {
            let id = w
                .add_dynamic_body(
                    glam::Vec3::new(x, 0.0, 0.0),
                    Shape3::Sphere { radius: 0.1 },
                    1.0,
                )
                .unwrap();
            w.bodies[id].electro.charge = q;
            ids.push(id);
        }
        w.electro
            .configure(ElectroConfig3 {
                enabled: true,
                mode: ElectroMode::Pairwise,
                ..Default::default()
            })
            .unwrap();
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        assert!(w.bodies[ids[0]].transform.position.x > -1.0);
        assert!(w.bodies[ids[1]].transform.position.x < 1.0);
    }
}
