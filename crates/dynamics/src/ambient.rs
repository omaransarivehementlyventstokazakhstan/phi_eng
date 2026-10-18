//! Ambient force fields applied to rigid bodies each step: a linear
//! spring / simple-harmonic oscillator (SHM), aerodynamic drag (air
//! resistance), and deterministic air turbulence.
//!
//! Position in [`crate::World2::step`]: an *ambient kick* runs immediately
//! after the gravity kick, before the fluid phase and the solver, so it is
//! a pure velocity kick that both integrators (semi-implicit Euler and
//! velocity-Verlet) consume unchanged. In Verlet mode the world calls the
//! kick twice with `h = dt / 2` — exactly the gravity contract — so the
//! force is evaluated at the correct sub-step positions/velocities.
//!
//! Force conventions:
//! - **Spring (SHM):** force `F = -k*(x - anchor) - c*v`, applied as
//!   `dv = F * inv_mass * ambient_scale * h`. Mass-dependent, so the
//!   natural frequency is `ω = sqrt(k/m)` and the damping ratio is
//!   `ζ = c / (2*sqrt(k*m))` — the textbook damped-oscillator system.
//! - **Drag (air resistance):** force
//!   `F = -(linear + quadratic*air_density*area*|v_rel|) * v_rel`,
//!   `v_rel = v - wind`, plus an angular drag torque. Mass-dependent.
//! - **Turbulence:** an acceleration `dv = amplitude * n * ambient_scale * h`
//!   where `n` is a bounded `[-1, 1]` deterministic noise vector (see
//!   [`crate::noise`]). Mass-independent like gravity, so a cloud of
//!   differently-sized debris reacts coherently.
//!
//! Determinism: pure per-body computation in index order, no RNG, no
//! allocation, no hash maps. Same-mode runs are bit-identical. Cross-
//! language equality with the C# mirror is *not* claimed (see the noise
//! module note).

use super::body::{RigidBody2, RigidBody3, Shape2, Shape3};
use crate::noise::{curl_2, curl_3, fbm_2, fbm_3};
use glam::{Vec2, Vec3};
use phieng_math::EPS;

/// Spring / simple-harmonic-motion field (2D). Disabled by default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpringConfig2 {
    /// Master switch. When false the whole spring block is skipped.
    pub enabled: bool,
    /// World-space rest point the spring pulls toward.
    pub anchor: Vec2,
    /// Hooke stiffness `k >= 0` (`F = -k * (x - anchor)`). Finite.
    pub stiffness: f32,
    /// Viscous damping `c >= 0` (`F -= c * v`; `0` = undamped SHM). Finite.
    pub damping: f32,
    /// Rest-length leash: offsets longer than this receive no spring force
    /// (`0` = unlimited). Finite and `>= 0`. Guards runaway shake-outs.
    pub max_displacement: f32,
}

impl SpringConfig2 {
    /// True for a freshly constructed (no-op) spring field.
    pub fn is_default(&self) -> bool {
        self.enabled == SpringConfig2::default().enabled
            && self.anchor == Vec2::ZERO
            && self.stiffness == 0.0
            && self.damping == 0.0
            && self.max_displacement == 0.0
    }
}

/// Aerodynamic drag field (2D). Disabled by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragConfig2 {
    /// Master switch.
    pub enabled: bool,
    /// Linear (Stokes / low-Reynolds) coefficient `>= 0`. Finite.
    pub linear: f32,
    /// Quadratic (pressure / high-Reynolds) coefficient `>= 0`. Finite.
    /// The force is `-quadratic * air_density * area * |v_rel| * v_rel`.
    pub quadratic: f32,
    /// Air density multiplier `>= 0` (folded into the quadratic term so
    /// the coefficient reads as physical). Finite.
    pub air_density: f32,
    /// Angular drag torque coefficient `>= 0` (damps spin). Finite.
    pub angular: f32,
    /// Ambient air velocity used for the relative velocity `v - wind`.
    pub wind: Vec2,
}

impl Default for DragConfig2 {
    fn default() -> Self {
        Self {
            enabled: false,
            linear: 0.0,
            quadratic: 0.0,
            air_density: 1.0,
            angular: 0.0,
            wind: Vec2::ZERO,
        }
    }
}

impl DragConfig2 {
    /// True for a freshly constructed (no-op) drag field.
    pub fn is_default(&self) -> bool {
        self.enabled == DragConfig2::default().enabled
            && self.linear == 0.0
            && self.quadratic == 0.0
            && self.air_density == 1.0
            && self.angular == 0.0
            && self.wind == Vec2::ZERO
    }
}

/// Deterministic turbulence field (2D). Disabled by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceConfig2 {
    /// Master switch.
    pub enabled: bool,
    /// Peak acceleration magnitude `>= 0` (output is bounded to this).
    pub amplitude: f32,
    /// Spatial frequency: world positions are multiplied by this before
    /// sampling noise. Larger = finer eddies. Finite and `> 0` to be
    /// meaningful (`0` gives a spatially constant field).
    pub frequency: f32,
    /// Temporal rate: the world clock is multiplied by this. `0` freezes
    /// the field. Finite.
    pub time_scale: f32,
    /// Noise seed (distinct seeds give independent fields).
    pub seed: u32,
    /// fBm octave count, clamped to `1..=8`.
    pub octaves: u8,
    /// Curl-noise mode: divergence-free (incompressible) flow.
    pub divergence_free: bool,
}

impl Default for TurbulenceConfig2 {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            frequency: 1.0,
            time_scale: 1.0,
            seed: 0,
            octaves: 3,
            divergence_free: false,
        }
    }
}

impl TurbulenceConfig2 {
    /// True for a freshly constructed (no-op) turbulence field.
    pub fn is_default(&self) -> bool {
        let d = TurbulenceConfig2::default();
        self.enabled == d.enabled
            && self.amplitude == d.amplitude
            && self.frequency == d.frequency
            && self.time_scale == d.time_scale
            && self.seed == d.seed
            && self.octaves == d.octaves
            && self.divergence_free == d.divergence_free
    }
}

/// Aggregate ambient-force configuration for a 2D world.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AmbientConfig2 {
    /// Spring / SHM field.
    pub spring: SpringConfig2,
    /// Aerodynamic drag field.
    pub drag: DragConfig2,
    /// Turbulence field.
    pub turbulence: TurbulenceConfig2,
}

impl AmbientConfig2 {
    /// True when every sub-field is a no-op (snapshot writes may then omit
    /// the block so pre-ambient files stay byte-identical).
    pub fn is_default(&self) -> bool {
        self.spring.is_default() && self.drag.is_default() && self.turbulence.is_default()
    }

    /// True when at least one sub-field is enabled (fast hot-loop gate).
    pub fn is_active(&self) -> bool {
        self.spring.enabled || self.drag.enabled || self.turbulence.enabled
    }
}

/// 3D spring / SHM field (see [`SpringConfig2`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpringConfig3 {
    /// Master switch.
    pub enabled: bool,
    /// World-space rest point.
    pub anchor: Vec3,
    /// Hooke stiffness `k >= 0`.
    pub stiffness: f32,
    /// Viscous damping `c >= 0`.
    pub damping: f32,
    /// Rest-length leash (`0` = unlimited).
    pub max_displacement: f32,
}

impl SpringConfig3 {
    /// See [`SpringConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        self.enabled == SpringConfig3::default().enabled
            && self.anchor == Vec3::ZERO
            && self.stiffness == 0.0
            && self.damping == 0.0
            && self.max_displacement == 0.0
    }
}

/// 3D aerodynamic drag field (see [`DragConfig2`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragConfig3 {
    /// Master switch.
    pub enabled: bool,
    /// Linear drag coefficient `>= 0`.
    pub linear: f32,
    /// Quadratic drag coefficient `>= 0`.
    pub quadratic: f32,
    /// Air density multiplier `>= 0`.
    pub air_density: f32,
    /// Angular drag torque coefficient `>= 0`.
    pub angular: f32,
    /// Ambient air velocity.
    pub wind: Vec3,
}

impl Default for DragConfig3 {
    fn default() -> Self {
        Self {
            enabled: false,
            linear: 0.0,
            quadratic: 0.0,
            air_density: 1.0,
            angular: 0.0,
            wind: Vec3::ZERO,
        }
    }
}

impl DragConfig3 {
    /// See [`DragConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        let d = DragConfig3::default();
        self.enabled == d.enabled
            && self.linear == 0.0
            && self.quadratic == 0.0
            && self.air_density == 1.0
            && self.angular == 0.0
            && self.wind == Vec3::ZERO
    }
}

/// 3D deterministic turbulence field (see [`TurbulenceConfig2`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceConfig3 {
    /// Master switch.
    pub enabled: bool,
    /// Peak acceleration magnitude `>= 0`.
    pub amplitude: f32,
    /// Spatial frequency.
    pub frequency: f32,
    /// Temporal rate.
    pub time_scale: f32,
    /// Noise seed.
    pub seed: u32,
    /// fBm octaves, clamped to `1..=8`.
    pub octaves: u8,
    /// Curl-noise (divergence-free) mode.
    pub divergence_free: bool,
}

impl Default for TurbulenceConfig3 {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            frequency: 1.0,
            time_scale: 1.0,
            seed: 0,
            octaves: 3,
            divergence_free: false,
        }
    }
}

impl TurbulenceConfig3 {
    /// See [`TurbulenceConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        let d = TurbulenceConfig3::default();
        self.enabled == d.enabled
            && self.amplitude == d.amplitude
            && self.frequency == d.frequency
            && self.time_scale == d.time_scale
            && self.seed == d.seed
            && self.octaves == d.octaves
            && self.divergence_free == d.divergence_free
    }
}

/// Aggregate ambient-force configuration for a 3D world.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AmbientConfig3 {
    /// Spring / SHM field.
    pub spring: SpringConfig3,
    /// Aerodynamic drag field.
    pub drag: DragConfig3,
    /// Turbulence field.
    pub turbulence: TurbulenceConfig3,
}

impl AmbientConfig3 {
    /// See [`AmbientConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        self.spring.is_default() && self.drag.is_default() && self.turbulence.is_default()
    }

    /// See [`AmbientConfig2::is_active`].
    pub fn is_active(&self) -> bool {
        self.spring.enabled || self.drag.enabled || self.turbulence.enabled
    }
}

/// Orientation-independent characteristic cross-section (a length in 2D)
/// used as the drag reference area. Shapes without a closed form fall
/// back to their bounding radius; shapeless bodies use `1.0`. Documented
/// approximation — quadratic drag is a tunable gameplay force, not a CFD
/// solve.
fn drag_area_2(body: &RigidBody2) -> f32 {
    match &body.shape {
        Some(Shape2::Circle { radius }) => 2.0 * radius.max(0.0),
        Some(Shape2::Box { half_extents }) => 2.0 * (half_extents.x.abs() + half_extents.y.abs()),
        Some(Shape2::Polygon { vertices }) => {
            let mut r2 = 0.0f32;
            for v in vertices {
                r2 = r2.max(v.length_squared());
            }
            2.0 * r2.sqrt()
        }
        None => 1.0,
    }
}

/// 3D projected reference area. See [`drag_area_2`].
fn drag_area_3(body: &RigidBody3) -> f32 {
    use std::f32::consts::PI;
    match &body.shape {
        Some(Shape3::Sphere { radius }) => PI * radius.max(0.0) * radius.max(0.0),
        Some(Shape3::Box { half_extents }) => {
            let (x, y, z) = (
                half_extents.x.abs(),
                half_extents.y.abs(),
                half_extents.z.abs(),
            );
            // Mean of the three face areas (orientation-independent proxy).
            4.0 * (x * y + y * z + z * x) / 3.0
        }
        Some(Shape3::Capsule {
            radius,
            half_length,
        }) => {
            let r = radius.max(0.0);
            let hl = half_length.max(0.0);
            // Cylindrical side + two hemispherical caps.
            4.0 * r * hl + PI * r * r
        }
        Some(Shape3::ConvexHull { vertices }) | Some(Shape3::ConvexMesh { vertices, .. }) => {
            let mut r2 = 0.0f32;
            for v in vertices {
                r2 = r2.max(v.length_squared());
            }
            PI * r2
        }
        None => 1.0,
    }
}

/// Spring acceleration (force per unit mass) for a body at `pos` with
/// velocity `v`, or `None` when the leash suppresses the pull.
#[inline]
fn spring_force_2(cfg: &SpringConfig2, pos: Vec2, v: Vec2) -> Option<Vec2> {
    let sol = pos - cfg.anchor;
    if cfg.max_displacement > 0.0
        && sol.length_squared() > cfg.max_displacement * cfg.max_displacement
    {
        return None;
    }
    Some(sol * (-cfg.stiffness) - v * cfg.damping)
}

/// 3D counterpart of [`spring_force_2`].
#[inline]
fn spring_force_3(cfg: &SpringConfig3, pos: Vec3, v: Vec3) -> Option<Vec3> {
    let sol = pos - cfg.anchor;
    if cfg.max_displacement > 0.0
        && sol.length_squared() > cfg.max_displacement * cfg.max_displacement
    {
        return None;
    }
    Some(sol * (-cfg.stiffness) - v * cfg.damping)
}

/// Apply the 2D ambient kick (spring + drag + turbulence) over interval `h`.
/// Skips static bodies (as gravity does) and bodies with
/// `ambient_scale == 0`. Deterministic, allocation-free, index order.
pub fn ambient_kick_2(bodies: &mut [RigidBody2], cfg: &AmbientConfig2, time: f32, h: f32) {
    let spring_on = cfg.spring.enabled;
    let drag_on = cfg.drag.enabled;
    let turb_on = cfg.turbulence.enabled;
    if !spring_on && !drag_on && !turb_on {
        return;
    }
    for b in bodies.iter_mut() {
        if b.is_static || b.is_sleeping || b.ambient_scale == 0.0 {
            continue;
        }
        let scale = b.ambient_scale;
        let mut dv = Vec2::ZERO;

        if spring_on {
            if let Some(f) = spring_force_2(&cfg.spring, b.transform.position, b.velocity) {
                dv += f * (b.inv_mass * scale * h);
            }
        }

        if drag_on {
            // Angular drag is independent of the linear kick; apply it
            // even if the relative velocity is zero.
            let aw = -cfg.drag.angular * b.angular_velocity * b.inv_inertia * scale * h;
            if aw.is_finite() {
                b.angular_velocity += aw;
            }
            let v_rel = b.velocity - cfg.drag.wind;
            let speed = v_rel.length();
            let mut f = v_rel * (-cfg.drag.linear);
            if speed > EPS {
                let area = drag_area_2(b);
                f -= v_rel * (cfg.drag.quadratic * cfg.drag.air_density * area * speed);
            }
            dv += f * (b.inv_mass * scale * h);
        }

        if turb_on {
            let p = b.transform.position * cfg.turbulence.frequency;
            let t = time * cfg.turbulence.time_scale;
            let n = if cfg.turbulence.divergence_free {
                curl_2(p.x, p.y, t, cfg.turbulence.seed, cfg.turbulence.octaves)
            } else {
                Vec2::new(
                    fbm_2(p.x, p.y, t, cfg.turbulence.seed, cfg.turbulence.octaves),
                    fbm_2(
                        p.x,
                        p.y,
                        t,
                        cfg.turbulence.seed ^ 0x55aa_55aa,
                        cfg.turbulence.octaves,
                    ),
                )
            };
            // Bound the contribution so |dv_turb| <= amplitude*scale*h.
            let n = n.clamp_length_max(1.0);
            dv += n * (cfg.turbulence.amplitude * scale * h);
        }

        if dv.x.is_finite() && dv.y.is_finite() {
            b.velocity += dv;
        }
    }
}

/// Apply the 3D ambient kick. See [`ambient_kick_2`].
pub fn ambient_kick_3(bodies: &mut [RigidBody3], cfg: &AmbientConfig3, time: f32, h: f32) {
    let spring_on = cfg.spring.enabled;
    let drag_on = cfg.drag.enabled;
    let turb_on = cfg.turbulence.enabled;
    if !spring_on && !drag_on && !turb_on {
        return;
    }
    for b in bodies.iter_mut() {
        if b.is_static || b.is_sleeping || b.ambient_scale == 0.0 {
            continue;
        }
        let scale = b.ambient_scale;
        let mut dv = Vec3::ZERO;

        if spring_on {
            if let Some(f) = spring_force_3(&cfg.spring, b.transform.position, b.velocity) {
                dv += f * (b.inv_mass * scale * h);
            }
        }

        if drag_on {
            let aw = -(cfg.drag.angular * scale * h) * (b.inv_inertia * b.angular_velocity);
            if aw.x.is_finite() && aw.y.is_finite() && aw.z.is_finite() {
                b.angular_velocity += aw;
            }
            let v_rel = b.velocity - cfg.drag.wind;
            let speed = v_rel.length();
            let mut f = v_rel * (-cfg.drag.linear);
            if speed > EPS {
                let area = drag_area_3(b);
                f -= v_rel * (cfg.drag.quadratic * cfg.drag.air_density * area * speed);
            }
            dv += f * (b.inv_mass * scale * h);
        }

        if turb_on {
            let p = b.transform.position * cfg.turbulence.frequency;
            let t = time * cfg.turbulence.time_scale;
            let n = if cfg.turbulence.divergence_free {
                curl_3(
                    p.x,
                    p.y,
                    p.z,
                    t,
                    cfg.turbulence.seed,
                    cfg.turbulence.octaves,
                )
            } else {
                Vec3::new(
                    fbm_3(
                        p.x,
                        p.y,
                        p.z,
                        t,
                        cfg.turbulence.seed,
                        cfg.turbulence.octaves,
                    ),
                    fbm_3(
                        p.x,
                        p.y,
                        p.z,
                        t,
                        cfg.turbulence.seed ^ 0x55aa_55aa,
                        cfg.turbulence.octaves,
                    ),
                    fbm_3(
                        p.x,
                        p.y,
                        p.z,
                        t,
                        cfg.turbulence.seed ^ 0xaa55_aa55,
                        cfg.turbulence.octaves,
                    ),
                )
            };
            let n = n.clamp_length_max(1.0);
            dv += n * (cfg.turbulence.amplitude * scale * h);
        }

        if dv.x.is_finite() && dv.y.is_finite() && dv.z.is_finite() {
            b.velocity += dv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{RigidBody2, RigidBody3, Shape2, Shape3};

    fn dyn2(pos: Vec2, radius: f32, density: f32) -> RigidBody2 {
        let mut b = RigidBody2::new(0);
        b.set_shape(Shape2::Circle { radius }, density).unwrap();
        b.transform.position = pos;
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
        b
    }

    #[test]
    fn test_default_is_noop() {
        let mut b = dyn2(Vec2::new(1.0, 2.0), 0.5, 1.0);
        b.velocity = Vec2::new(3.0, -1.0);
        let before = b.velocity;
        let cfg = AmbientConfig2::default();
        assert!(cfg.is_default() && !cfg.is_active());
        ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 12.0, 1.0 / 60.0);
        assert_eq!(b.velocity, before);
    }

    #[test]
    fn test_spring_shm_period() {
        // Undamped SHM through the real kick: the body returns to its
        // start after one period T = 2*pi*sqrt(m/k).
        let k = 40.0f32;
        let mut b = dyn2(Vec2::new(1.0, 0.0), 0.5, 1.0);
        let omega = (k / b.mass).sqrt();
        let period = 2.0 * std::f32::consts::PI / omega;
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: k,
                damping: 0.0,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        let dt = 1.0 / 8000.0;
        let steps = (period / dt).round() as usize;
        for _ in 0..steps {
            ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
            b.transform.position += b.velocity * dt;
        }
        assert!(
            (b.transform.position.x - 1.0).abs() < 0.02,
            "SHM did not close a period: x={}",
            b.transform.position.x
        );
    }

    #[test]
    fn test_spring_leash() {
        let mut b = dyn2(Vec2::new(10.0, 0.0), 0.5, 1.0);
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: 1000.0,
                damping: 0.0,
                max_displacement: 1.0,
            },
            ..Default::default()
        };
        ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, 1.0 / 60.0);
        assert_eq!(b.velocity, Vec2::ZERO, "leash did not suppress the pull");
    }

    #[test]
    fn test_spring_damped_decays() {
        let k = 100.0f32;
        let c = 2.0 * (k * 1.0f32).sqrt(); // critical damping for m=1
        let mut b = dyn2(Vec2::new(2.0, 0.0), 0.5, 1.0);
        b.velocity = Vec2::ZERO;
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: k,
                damping: c,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        let dt = 1.0 / 240.0;
        let mut peak = b.transform.position.length();
        for _ in 0..600 {
            ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
            b.transform.position += b.velocity * dt;
            peak = peak.max(b.transform.position.length());
        }
        assert!(
            b.transform.position.length() < 0.05,
            "critically damped spring did not settle: x={}",
            b.transform.position.length()
        );
        let _ = peak;
    }

    #[test]
    fn test_drag_terminal_velocity_linear() {
        // Linear drag + gravity reaches v_term = g*m / linear.
        // Here we only test the drag half by hand-integrating v under
        // constant gravity against the ambient kick.
        let linear = 3.0f32;
        let g = 9.81f32;
        let mut b = dyn2(Vec2::ZERO, 0.5, 1.0);
        let cfg = AmbientConfig2 {
            drag: DragConfig2 {
                enabled: true,
                linear,
                quadratic: 0.0,
                air_density: 1.0,
                angular: 0.0,
                wind: Vec2::ZERO,
            },
            ..Default::default()
        };
        let dt = 1.0 / 240.0;
        for _ in 0..4000 {
            b.velocity += Vec2::new(0.0, -g) * dt; // gravity kick
            ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
        }
        // Steady state: drag force balances gravity => v = -g*m/linear.
        let expected = -g * b.mass / linear;
        assert!(
            (b.velocity.y - expected).abs() < 0.5,
            "linear terminal velocity off: {} want {}",
            b.velocity.y,
            expected
        );
    }

    #[test]
    fn test_drag_zero_at_rest_with_wind() {
        // A body co-moving with the wind feels no drag: velocity is
        // unchanged.
        let mut b = dyn2(Vec2::new(5.0, 5.0), 0.5, 1.0);
        b.velocity = Vec2::new(2.0, 0.0);
        let cfg = AmbientConfig2 {
            drag: DragConfig2 {
                enabled: true,
                linear: 5.0,
                quadratic: 2.0,
                air_density: 1.0,
                angular: 0.0,
                wind: Vec2::new(2.0, 0.0),
            },
            ..Default::default()
        };
        ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, 1.0 / 60.0);
        assert_eq!(
            b.velocity,
            Vec2::new(2.0, 0.0),
            "wind-matched body got drag"
        );
    }

    #[test]
    fn test_turbulence_determinism() {
        let cfg = AmbientConfig2 {
            turbulence: TurbulenceConfig2 {
                enabled: true,
                amplitude: 5.0,
                frequency: 0.5,
                time_scale: 1.0,
                seed: 42,
                octaves: 4,
                divergence_free: false,
            },
            ..Default::default()
        };
        let run = || {
            let mut b = dyn2(Vec2::new(1.0, 2.0), 0.5, 1.0);
            for i in 0..30 {
                ambient_kick_2(
                    std::slice::from_mut(&mut b),
                    &cfg,
                    i as f32 * 0.01,
                    1.0 / 60.0,
                );
                b.transform.position += b.velocity * (1.0 / 60.0);
            }
            b.velocity
        };
        let a = run();
        let c = run();
        assert_eq!(a.x.to_bits(), c.x.to_bits());
        assert_eq!(a.y.to_bits(), c.y.to_bits());
    }

    #[test]
    fn test_turbulence_bounded() {
        let amp = 7.0f32;
        let cfg = AmbientConfig2 {
            turbulence: TurbulenceConfig2 {
                enabled: true,
                amplitude: amp,
                frequency: 0.7,
                time_scale: 0.5,
                seed: 7,
                octaves: 5,
                divergence_free: true,
            },
            ..Default::default()
        };
        let dt = 1.0 / 60.0;
        for i in 0..500 {
            let mut b = dyn2(Vec2::new(i as f32 * 0.13, i as f32 * -0.07), 0.5, 1.0);
            ambient_kick_2(std::slice::from_mut(&mut b), &cfg, i as f32 * 0.02, dt);
            let dv = b.velocity.length();
            assert!(
                dv <= amp * dt + 1e-3,
                "turbulence exceeded amplitude bound: {dv}"
            );
        }
    }

    #[test]
    fn test_ambient_scale_immune() {
        let mut b = dyn2(Vec2::new(3.0, 0.0), 0.5, 1.0);
        b.ambient_scale = 0.0;
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: 500.0,
                damping: 0.5,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, 1.0 / 60.0);
        assert_eq!(b.velocity, Vec2::ZERO);
    }

    #[test]
    fn test_static_skipped() {
        let mut b = RigidBody2::static_body(0);
        b.set_shape(Shape2::Circle { radius: 0.5 }, 1.0).unwrap();
        b.transform.position = Vec2::new(2.0, 0.0);
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: 500.0,
                damping: 0.0,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, 1.0 / 60.0);
        assert_eq!(b.velocity, Vec2::ZERO);
    }

    #[test]
    fn test_ambient3_determinism_and_default() {
        let mut b = RigidBody3::new(0);
        b.set_shape(Shape3::Sphere { radius: 0.5 }, 1.0).unwrap();
        b.transform.position = Vec3::new(1.0, 2.0, 3.0);
        b.linear_damping = 0.0;
        let d = AmbientConfig3::default();
        assert!(d.is_default() && !d.is_active());
        let before = b.velocity;
        ambient_kick_3(std::slice::from_mut(&mut b), &d, 0.0, 1.0 / 60.0);
        assert_eq!(b.velocity, before);

        let cfg = AmbientConfig3 {
            turbulence: TurbulenceConfig3 {
                enabled: true,
                amplitude: 3.0,
                frequency: 0.4,
                time_scale: 1.0,
                seed: 9,
                octaves: 3,
                divergence_free: true,
            },
            ..Default::default()
        };
        let run = || {
            let mut b = RigidBody3::new(0);
            b.set_shape(Shape3::Sphere { radius: 0.5 }, 1.0).unwrap();
            b.transform.position = Vec3::new(1.0, 2.0, 3.0);
            for i in 0..25 {
                ambient_kick_3(
                    std::slice::from_mut(&mut b),
                    &cfg,
                    i as f32 * 0.02,
                    1.0 / 60.0,
                );
                b.transform.position += b.velocity * (1.0 / 60.0);
            }
            b.velocity
        };
        let a = run();
        let c = run();
        assert_eq!(a.x.to_bits(), c.x.to_bits());
        assert_eq!(a.y.to_bits(), c.y.to_bits());
        assert_eq!(a.z.to_bits(), c.z.to_bits());
    }

    // ---- Canonical damped-oscillator (SHM) benchmark suite ----
    // The engine's spring is `m x'' + c x' + k x = 0`; the closed-form
    // solution is known, so each regime is checked analytically.

    /// Integrate one body under the spring kick (semi-implicit Euler),
    /// returning `(t, x, v)` samples. Anchor at origin.
    fn spring_series(
        k: f32,
        c: f32,
        x0: f32,
        v0: f32,
        dt: f32,
        steps: usize,
    ) -> Vec<(f32, f32, f32)> {
        let mut b = dyn2(Vec2::new(x0, 0.0), 0.5, 1.0);
        b.velocity = Vec2::new(v0, 0.0);
        let cfg = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: k,
                damping: c,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        let mut out = Vec::with_capacity(steps);
        for n in 0..steps {
            ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
            b.transform.position += b.velocity * dt;
            out.push(((n + 1) as f32 * dt, b.transform.position.x, b.velocity.x));
        }
        out
    }

    #[test]
    fn shm_damped_regimes_analytic() {
        let k = 40.0f32;
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let w0 = (k / m).sqrt();
        let dt = 1.0 / 8000.0;
        let steps = 16000; // 2 s
        let (x0, v0) = (1.0f32, 0.0f32);
        let crossings = |series: &[(f32, f32, f32)]| {
            let mut n = 0;
            let mut prev = x0;
            for &(_, x, _) in series {
                if (x > 0.0) != (prev > 0.0) {
                    n += 1;
                }
                prev = x;
            }
            n
        };

        // Underdamped (zeta = 0.1): oscillatory decay.
        let zeta = 0.1f32;
        let c = 2.0 * zeta * (k * m).sqrt();
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let s = spring_series(k, c, x0, v0, dt, steps);
        for &(t, x, _) in &s {
            let analytic = (-zeta * w0 * t).exp()
                * (x0 * (wd * t).cos() + ((v0 + zeta * w0 * x0) / wd) * (wd * t).sin());
            assert!(
                (x - analytic).abs() < 0.03,
                "underdamped x={x} analytic={analytic} t={t}"
            );
        }
        assert!(
            crossings(&s) >= 3,
            "underdamped should oscillate ({})",
            crossings(&s)
        );

        // Critically damped (zeta = 1): x = (x0 + (v0 + w0 x0) t) e^{-w0 t}.
        let c = 2.0 * (k * m).sqrt();
        let s = spring_series(k, c, x0, v0, dt, steps);
        for &(t, x, _) in &s {
            let analytic = (x0 + (v0 + w0 * x0) * t) * (-w0 * t).exp();
            assert!(
                (x - analytic).abs() < 0.03,
                "critical x={x} analytic={analytic} t={t}"
            );
        }
        assert_eq!(crossings(&s), 0, "critically damped must not oscillate");

        // Overdamped (zeta = 3): two real exponentials, no oscillation.
        let zeta = 3.0f32;
        let c = 2.0 * zeta * (k * m).sqrt();
        let disc = (zeta * zeta - 1.0).sqrt();
        let r1 = -w0 * (zeta - disc);
        let r2 = -w0 * (zeta + disc);
        let a = (v0 - r2 * x0) / (r1 - r2);
        let bcoef = (r1 * x0 - v0) / (r1 - r2);
        let s = spring_series(k, c, x0, v0, dt, steps);
        for &(t, x, _) in &s {
            let analytic = a * (r1 * t).exp() + bcoef * (r2 * t).exp();
            assert!(
                (x - analytic).abs() < 0.03,
                "overdamped x={x} analytic={analytic} t={t}"
            );
        }
        assert_eq!(crossings(&s), 0, "overdamped must not oscillate");
    }

    #[test]
    fn shm_energy_decay_rate() {
        // Underdamped energy envelope: E(t) ~ E0 e^{-2 zeta w0 t}.
        let k = 40.0f32;
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let w0 = (k / m).sqrt();
        let zeta = 0.05f32;
        let c = 2.0 * zeta * (k * m).sqrt();
        let dt = 1.0 / 4000.0;
        let s = spring_series(k, c, 1.0, 0.0, dt, 4000 * 6);
        let mut peaks: Vec<(f32, f32)> = Vec::new();
        for w in s.windows(2) {
            if w[0].2 > 0.0 && w[1].2 <= 0.0 {
                peaks.push((w[1].0, w[1].1));
            }
        }
        assert!(peaks.len() >= 5, "too few peaks: {}", peaks.len());
        let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &(t, x) in &peaks {
            let e = (0.5 * k * x * x).max(1e-9) as f64;
            let (xt, yt) = (t as f64, e.ln());
            sx += xt;
            sy += yt;
            sxx += xt * xt;
            sxy += xt * yt;
            n += 1.0;
        }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let expected = -2.0 * zeta as f64 * w0 as f64;
        assert!(
            (slope - expected).abs() < 0.2 * expected.abs(),
            "energy decay slope {slope} want {expected}"
        );
    }

    #[test]
    fn shm_period_mass_scaling() {
        // T = 2 pi sqrt(m/k): periods scale as sqrt(mass).
        let k = 50.0f32;
        let dt = 1.0 / 8000.0;
        for &density in &[0.5f32, 1.0, 2.0] {
            let mut b = dyn2(Vec2::new(1.0, 0.0), 0.5, density);
            let t_expected = 2.0 * std::f32::consts::PI * (b.mass / k).sqrt();
            let steps = (3.0 * t_expected / dt) as usize;
            let mut cross_times: Vec<f32> = Vec::new();
            let mut prev = b.transform.position.x;
            let mut t = 0.0f32;
            for _ in 0..steps {
                ambient_kick_2(std::slice::from_mut(&mut b), &cfg_spring(k), 0.0, dt);
                b.transform.position += b.velocity * dt;
                t += dt;
                let x = b.transform.position.x;
                if prev > 0.0 && x <= 0.0 {
                    cross_times.push(t);
                }
                prev = x;
            }
            assert!(
                cross_times.len() >= 2,
                "density {density}: too few crossings"
            );
            let period = cross_times[1] - cross_times[0];
            assert!(
                (period - t_expected).abs() < 0.05 * t_expected,
                "density {density}: period {period} want {t_expected}"
            );
        }
    }

    fn cfg_spring(k: f32) -> AmbientConfig2 {
        AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::ZERO,
                stiffness: k,
                damping: 0.0,
                max_displacement: 0.0,
            },
            ..Default::default()
        }
    }

    #[test]
    fn shm_vertical_gravity_equilibrium() {
        // Gravity shifts the rest point by m g / k; period is unchanged.
        let g = 9.81f32;
        let k = 40.0f32;
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, -g);
        let id = world
            .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[id].linear_damping = 0.0;
        world.linear_sleep_threshold = 0.0;
        world.ambient.spring = SpringConfig2 {
            enabled: true,
            anchor: Vec2::ZERO,
            stiffness: k,
            damping: 0.0,
            max_displacement: 0.0,
        };
        let m = world.bodies[id].mass;
        let y_eq = -m * g / k;
        let period = 2.0 * std::f32::consts::PI * (m / k).sqrt();
        let dt = 1.0 / 2000.0;
        let settle = (period / dt) as usize;
        let (mut sum, mut cnt) = (0.0f32, 0usize);
        for n in 0..(3 * settle) {
            world.step(dt);
            if n >= settle {
                sum += world.bodies[id].transform.position.y;
                cnt += 1;
            }
        }
        let mean = sum / cnt as f32;
        assert!(
            (mean - y_eq).abs() < 0.02,
            "gravity rest point {mean} want {y_eq}"
        );
    }

    #[test]
    fn shm_driven_resonance_peak() {
        // Driven damped oscillator: steady-state amplitude peaks near
        // omega_r = w0 sqrt(1 - 2 zeta^2).
        let k = 40.0f32;
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let w0 = (k / m).sqrt();
        let zeta = 0.08f32;
        let c = 2.0 * zeta * (k * m).sqrt();
        let f0 = 1.0f32;
        let dt = 1.0 / 1000.0;

        let amplitude = |wd: f32| -> f32 {
            let mut b = dyn2(Vec2::ZERO, 0.5, 1.0);
            let cfg = AmbientConfig2 {
                spring: SpringConfig2 {
                    enabled: true,
                    anchor: Vec2::ZERO,
                    stiffness: k,
                    damping: c,
                    max_displacement: 0.0,
                },
                ..Default::default()
            };
            let period = 2.0 * std::f32::consts::PI / wd;
            let drive_steps = (15.0 * period / dt) as usize;
            let measure_steps = (2.0 * period / dt) as usize;
            let mut t = 0.0f32;
            let mut peak = 0.0f32;
            for n in 0..(drive_steps + measure_steps) {
                let f = f0 * (wd * t).cos();
                b.velocity += Vec2::new(f, 0.0) * (b.inv_mass * dt);
                ambient_kick_2(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
                b.transform.position += b.velocity * dt;
                t += dt;
                if n >= drive_steps {
                    peak = peak.max(b.transform.position.x.abs());
                }
            }
            peak
        };

        let w_r = w0 * (1.0 - 2.0 * zeta * zeta).max(0.0).sqrt();
        let mut best_w = 0.0f32;
        let mut best_a = 0.0f32;
        for i in 0..=40 {
            let wd = (0.4 + 1.2 * i as f32 / 40.0) * w0;
            let a = amplitude(wd);
            if a > best_a {
                best_a = a;
                best_w = wd;
            }
        }
        assert!(
            (best_w - w_r).abs() < 0.15 * w0,
            "resonance peak at {best_w} want ~{w_r}"
        );
        let denom =
            ((w0 * w0 - best_w * best_w).powi(2) + (2.0 * zeta * w0 * best_w).powi(2)).sqrt();
        let analytic = f0 / (m * denom);
        assert!(
            (best_a - analytic).abs() < 0.3 * analytic,
            "peak amplitude {best_a} vs analytic {analytic}"
        );
    }

    #[test]
    fn shm3_damped_regimes_analytic() {
        // 3D mirror: the spring is isotropic, so the x-motion matches 2D.
        let k = 40.0f32;
        let mut probe = RigidBody3::new(0);
        probe
            .set_shape(Shape3::Sphere { radius: 0.5723572 }, 1.0)
            .unwrap();
        let m = probe.mass;
        let w0 = (k / m).sqrt();
        let zeta = 0.1f32;
        let c = 2.0 * zeta * (k * m).sqrt();
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let cfg = AmbientConfig3 {
            spring: SpringConfig3 {
                enabled: true,
                anchor: Vec3::ZERO,
                stiffness: k,
                damping: c,
                max_displacement: 0.0,
            },
            ..Default::default()
        };
        let mut b = RigidBody3::new(0);
        b.set_shape(Shape3::Sphere { radius: 0.5723572 }, 1.0)
            .unwrap();
        b.transform.position = Vec3::new(1.0, 0.0, 0.0);
        b.linear_damping = 0.0;
        let dt = 1.0 / 8000.0;
        for n in 0..8000 {
            ambient_kick_3(std::slice::from_mut(&mut b), &cfg, 0.0, dt);
            b.transform.position += b.velocity * dt;
            let t = (n + 1) as f32 * dt;
            // x(t) = e^{-zeta w0 t}(x0 cos(wd t) + ((v0 + zeta w0 x0)/wd) sin(wd t));
            // with x0 = 1, v0 = 0 the sin term is (zeta w0 / wd) sin(wd t).
            let analytic =
                (-zeta * w0 * t).exp() * ((wd * t).cos() + ((zeta * w0) / wd) * (wd * t).sin());
            assert!(
                (b.transform.position.x - analytic).abs() < 0.05,
                "3D underdamped x={} analytic={analytic}",
                b.transform.position.x
            );
        }
    }

    #[test]
    fn shm_undamped_energy_bounded() {
        // Symplectic Euler introduces no secular energy drift: the
        // undamped oscillator's energy stays in a narrow band around E0.
        let k = 40.0f32;
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let dt = 1.0 / 4000.0;
        let s = spring_series(k, 0.0, 1.0, 0.0, dt, 4000 * 6);
        let energy = |x: f32, v: f32| 0.5 * k * x * x + 0.5 * m * v * v;
        let e0 = energy(1.0, 0.0);
        let (mut emin, mut emax) = (e0, e0);
        for &(_, x, v) in &s {
            let e = energy(x, v);
            emin = emin.min(e);
            emax = emax.max(e);
        }
        assert!(
            (emax - emin) < 0.05 * e0,
            "energy band too wide: {emin}..{emax}"
        );
        assert!(
            (emax - e0).abs() < 0.05 * e0,
            "energy drifted up: {emax} vs {e0}"
        );
        assert!(
            (emin - e0).abs() < 0.05 * e0,
            "energy drifted down: {emin} vs {e0}"
        );
    }

    #[test]
    fn shm_critical_settles_fastest() {
        // Critical damping reaches a rest band in the least time.
        let k = 40.0f32;
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let dt = 1.0 / 2000.0;
        let settle_time = |zeta: f32| -> f32 {
            let c = 2.0 * zeta * (k * m).sqrt();
            let s = spring_series(k, c, 1.0, 0.0, dt, 2000 * 10);
            let band = 0.01;
            let mut last = 0.0f32;
            for &(t, x, _) in &s {
                if x.abs() >= band {
                    last = t;
                }
            }
            last
        };
        let under = settle_time(0.3);
        let crit = settle_time(1.0);
        let over = settle_time(3.0);
        assert!(
            crit < under,
            "critical ({crit}) should beat underdamped ({under})"
        );
        assert!(
            crit < over,
            "critical ({crit}) should beat overdamped ({over})"
        );
    }

    #[test]
    fn shm_stiffness_scaling() {
        // omega = sqrt(k/m): the period scales as 1/sqrt(k).
        let m = dyn2(Vec2::ZERO, 0.5, 1.0).mass;
        let dt = 1.0 / 8000.0;
        let period = |k: f32| -> f32 {
            let mut b = dyn2(Vec2::new(1.0, 0.0), 0.5, 1.0);
            let t_expected = 2.0 * std::f32::consts::PI * (m / k).sqrt();
            let steps = (3.0 * t_expected / dt) as usize;
            let mut cross_times: Vec<f32> = Vec::new();
            let mut prev = b.transform.position.x;
            let mut t = 0.0f32;
            for _ in 0..steps {
                ambient_kick_2(std::slice::from_mut(&mut b), &cfg_spring(k), 0.0, dt);
                b.transform.position += b.velocity * dt;
                t += dt;
                let x = b.transform.position.x;
                if prev > 0.0 && x <= 0.0 {
                    cross_times.push(t);
                }
                prev = x;
            }
            cross_times[1] - cross_times[0]
        };
        let t40 = period(40.0);
        let t160 = period(160.0);
        assert!(
            (t160 - t40 / 2.0).abs() < 0.05 * t40,
            "period does not scale as 1/sqrt(k): k=40 T={t40}, k=160 T={t160}"
        );
    }
}
