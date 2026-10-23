//! Arcade cars in 2D: a top-down driver with tire grip and a handbrake.
//!
//! There is no suspension here — the top-down world has no up to hang
//! a ray from — only four (or two) tire patches: engine force along
//! the steered heading, lateral grip that kills the slide, and a
//! handbrake that steals the rear grip so the tail steps out. Like
//! its 3D sibling the model is *stateless*: game re-passes config and
//! input after a load and continues bit for bit.
//!
//! The chassis frame is forward `+X`, left `+Y`: `forward =
//! rotation * (1,0)`. Wheel offsets live in this body frame.

use glam::Vec2;

use super::world::World2;
use phieng_math::trig::compute_cos_sin;

/// The most wheels one vehicle may carry. Fixed so the telemetry is a
/// plain array on both sides of the FFI.
pub const MAX_WHEELS_2D: usize = 8;

/// One wheel's mounting: where it sits and what it does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMount2 {
    /// The wheel in the chassis body frame (x forward, y left).
    pub offset: Vec2,
    /// Whether the input steering yaws this wheel.
    pub steer: bool,
    /// Whether the throttle/brake/handbrake drive this wheel.
    pub drive: bool,
}

/// The vehicle's tuning. Forces marked total are split across wheels.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleConfig2 {
    /// The wheels, in order; only the first `wheel_count` are read.
    pub wheels: [WheelMount2; MAX_WHEELS_2D],
    /// How many wheels are mounted (clamped to `MAX_WHEELS_2D`).
    pub wheel_count: usize,
    /// The total engine force (N), split across the driven wheels.
    pub engine_force: f32,
    /// The total brake force (N), split across every wheel.
    pub brake_force: f32,
    /// The lateral tire stiffness (N·s/m): the grip that kills slide.
    pub grip_lateral: f32,
    /// The friction-circle coefficient: `|F_tire| <= mu * N`.
    pub friction_mu: f32,
    /// The quadratic air drag (N·s²/m²) at the centre of mass.
    pub drag_k: f32,
    /// The rolling resistance (per unit normal load).
    pub rolling_resist: f32,
    /// The maximum steer angle (radians) at full steering input.
    pub max_steer: f32,
    /// The wheel radius (m): the visual spin's divisor only.
    pub wheel_radius: f32,
}

impl Default for VehicleConfig2 {
    fn default() -> Self {
        let mount = WheelMount2 {
            offset: Vec2::ZERO,
            steer: false,
            drive: false,
        };
        Self {
            wheels: [mount; MAX_WHEELS_2D],
            wheel_count: 0,
            engine_force: 6000.0,
            brake_force: 9000.0,
            grip_lateral: 400.0,
            friction_mu: 1.1,
            drag_k: 0.7,
            rolling_resist: 0.02,
            max_steer: 0.5,
            wheel_radius: 0.33,
        }
    }
}

/// The driver's hands, every step. Throttle spans `-1..1` (reverse
/// included); the rest span `0..1`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VehicleInput2 {
    pub throttle: f32,
    pub brake: f32,
    pub steering: f32,
    pub handbrake: f32,
}

/// What one wheel did this step.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct WheelTelemetry2 {
    /// Always true top-down: there is no suspension to leave ground.
    pub in_contact: bool,
    /// Always zero: no suspension, no travel.
    pub compression01: f32,
    /// The static normal load (N): a quarter of the weight per wheel
    /// only when four wheels share it evenly — in general the nominal
    /// share, with no load transfer.
    pub normal_load: f32,
    /// The longitudinal tire force (N, along the wheel forward).
    pub force_long: f32,
    /// The lateral tire force (N, along the wheel left).
    pub force_lat: f32,
    /// The rolling spin (rad/s, visual only): `v_long / R`.
    pub spin: f32,
}

/// A smooth sign with a dead zone: `v / 0.5` clamped to `-1..1`, so a
/// resting wheel feels no force jitter.
fn smooth_sign(v: f32) -> f32 {
    (v / 0.5).clamp(-1.0, 1.0)
}

/// Rotate `(x, y)` by `angle` with the deterministic trigonometry —
/// never the platform `sin_cos`.
fn rotate2(x: f32, y: f32, angle: f32) -> (f32, f32) {
    let (c, s) = compute_cos_sin(angle);
    (x * c - y * s, x * s + y * c)
}

/// Run one vehicle step: push the tires, fill `telemetry`, and return
/// the wheels processed.
///
/// Call it before [`World2::step`] with the same `dt`. Returns 0 and
/// touches nothing for an unknown chassis, no wheels, or a
/// non-positive `dt`.
pub fn step_vehicle2(
    world: &mut World2,
    chassis: usize,
    config: &VehicleConfig2,
    input: &VehicleInput2,
    dt: f32,
    telemetry: &mut [WheelTelemetry2],
) -> usize {
    if !dt.is_finite() || dt <= 0.0 {
        return 0;
    }
    let count = config.wheel_count.min(MAX_WHEELS_2D).min(telemetry.len());
    if count == 0 {
        return 0;
    }
    let Some(chassis_body) = world.bodies.get(chassis).filter(|b| b.id == chassis) else {
        return 0;
    };
    let throttle = input.throttle.clamp(-1.0, 1.0);
    let brake = input.brake.clamp(0.0, 1.0);
    let steering = input.steering.clamp(-1.0, 1.0);
    let handbrake = input.handbrake.clamp(0.0, 1.0);

    let radius = config.wheel_radius.max(1e-3);
    // The nominal load per wheel is Earth-normalized, not read from
    // the world's gravity: the top-down tire has no suspension to
    // weigh anything with, so the load is a pure tire normalization.
    // A world gravity (if any) still drags the chassis as a body.
    let nominal = (chassis_body.mass * 9.81 / count.max(1) as f32).max(1e-6);
    let driven = config.wheels[..count].iter().filter(|w| w.drive).count();
    let engine_each = if driven == 0 {
        0.0
    } else {
        config.engine_force / driven as f32
    };
    let brake_each = config.brake_force / count as f32;

    // The chassis state, read once: every wheel's rates come from the
    // same instant, so the processing order cannot yaw the car.
    let snapshot = world.bodies[chassis].clone();
    let fwd0 = snapshot.transform.rotation * Vec2::X;
    // The forces, deferred: nothing mutates the chassis until every
    // wheel has been measured.
    let mut tire_force = [Vec2::ZERO; MAX_WHEELS_2D];
    let mut tire_point = [Vec2::ZERO; MAX_WHEELS_2D];

    for (i, mount) in config.wheels[..count].iter().enumerate() {
        let hard = snapshot.transform.position + snapshot.transform.rotation * mount.offset;
        // Positive steering turns right (clockwise on screen): the
        // nose leaves +X toward −Y, hence the negation.
        let steer_angle = if mount.steer {
            -steering * config.max_steer
        } else {
            0.0
        };
        let (fx, fy) = rotate2(fwd0.x, fwd0.y, steer_angle);
        let wheel_fwd = Vec2::new(fx, fy);
        // Left of forward: the lateral direction.
        let wheel_lat = Vec2::new(-wheel_fwd.y, wheel_fwd.x);
        // The point velocity carries the yaw into the tire.
        let r = hard - snapshot.transform.position;
        let point_vel = snapshot.velocity + snapshot.angular_velocity * Vec2::new(-r.y, r.x);
        let v_long = point_vel.dot(wheel_fwd);
        let v_lat = point_vel.dot(wheel_lat);
        let mut force_long = 0.0f32;
        if mount.drive {
            force_long += throttle * engine_each;
            force_long -= handbrake * config.brake_force * 0.75 * smooth_sign(v_long);
        }
        force_long -= brake * brake_each * smooth_sign(v_long);
        force_long -= config.rolling_resist * nominal * smooth_sign(v_long);
        let mut grip = config.grip_lateral;
        if !mount.steer {
            // The handbrake steals the rear grip so the tail steps out.
            grip *= 1.0 - 0.9 * handbrake;
        }
        let mut force_lat = -grip * v_lat;
        let cap = config.friction_mu * nominal;
        let total = (force_long * force_long + force_lat * force_lat).sqrt();
        if total > cap {
            let s = cap / total;
            force_long *= s;
            force_lat *= s;
        }
        tire_force[i] = wheel_fwd * force_long + wheel_lat * force_lat;
        tire_point[i] = hard;
        telemetry[i] = WheelTelemetry2 {
            in_contact: true,
            compression01: 0.0,
            normal_load: nominal,
            force_long,
            force_lat,
            spin: v_long / radius,
        };
    }

    for i in 0..count {
        world.bodies[chassis].apply_force(tire_force[i], tire_point[i], dt);
    }

    let body = &world.bodies[chassis];
    let v = body.velocity;
    let speed = v.length();
    if speed > 0.0 {
        let drag = v * (-config.drag_k * speed);
        let centre = body.transform.position;
        world.bodies[chassis].apply_force(drag, centre, dt);
    }
    count
}
