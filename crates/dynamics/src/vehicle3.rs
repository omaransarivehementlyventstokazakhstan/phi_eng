//! Raycast suspension vehicles in 3D: an arcade/sim-cade car.
//!
//! The chassis is an ordinary dynamic rigid body. Each wheel hangs
//! from a hardpoint: a suspension ray measures the ground, a spring
//! pushes the chassis up, and a steady-state tire pushes it forward
//! and sideways. The model is *stateless*: every rate comes from the
//! current rigid-body state, so saving and loading needs no vehicle
//! block — the game re-passes the config and the input and continues
//! bit for bit. Wheel spin for visuals derives from rolling
//! (`v_long / R`); there is no slip-ratio state, no persistent wheel
//! memory, and no holding brake at standstill (all three are
//! follow-up work that needs a snapshot block).
//!
//! One envelope edge to respect: a hardpoint driven below the ground
//! surface reads as airborne, because rays that start inside a shape
//! report a miss (the house rule). The bump-stop cap is what keeps a
//! hard landing out of that regime — size `max_suspension_force` to
//! stop the chassis before the springs run out, and do not drop cars
//! from orbit.
//!
//! The chassis frame is right-handed with forward `+Z` and up `+Y`:
//! `forward = rotation * (0,0,1)`, `up = rotation * (0,1,0)`,
//! `right = rotation * (1,0,0)`. Wheel offsets live in this body
//! frame. Steering yaws the front wheels' forward around up.

use glam::Vec3;

use super::body::{QueryFilter, Shape3};
use super::world::{World3, WorldRayHit3};
use phieng_collision::query::{ray_capsule3, ray_obb3, ray_sphere3, rotate_vec3, Ray3};
use phieng_math::trig::compute_cos_sin;

/// The most wheels one vehicle may carry. Fixed so the telemetry is a
/// plain array on both sides of the FFI.
pub const MAX_WHEELS: usize = 8;

/// One wheel's mounting: where it hangs and what it does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMount3 {
    /// The hardpoint in the chassis body frame (x right, y up, z forward).
    pub offset: Vec3,
    /// Whether the input steering yaws this wheel.
    pub steer: bool,
    /// Whether the throttle/brake/handbrake drive this wheel.
    pub drive: bool,
}

/// The vehicle's tuning. Everything is total-per-vehicle where noted.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleConfig3 {
    /// The wheels, in order; only the first `wheel_count` are read.
    pub wheels: [WheelMount3; MAX_WHEELS],
    /// How many wheels are mounted (clamped to `MAX_WHEELS`).
    pub wheel_count: usize,
    /// The suspension rest length (axle below the hardpoint, metres).
    pub rest_length: f32,
    /// The suspension spring rate (N/m).
    pub spring_k: f32,
    /// The suspension damping rate (N·s/m).
    pub damping_c: f32,
    /// The suspension force cap (N): the bottoming stop.
    pub max_suspension_force: f32,
    /// The wheel radius (m): sets the axle height and the visual spin.
    pub wheel_radius: f32,
    /// The total engine force (N), split across the driven wheels.
    pub engine_force: f32,
    /// The total brake force (N), split across every wheel.
    pub brake_force: f32,
    /// The lateral tire stiffness (N·s/m per unit load).
    pub grip_lateral: f32,
    /// The friction-circle coefficient: `|F_tire| <= mu * N`.
    pub friction_mu: f32,
    /// The quadratic air drag (N·s²/m²) at the centre of mass.
    pub drag_k: f32,
    /// The rolling resistance (per unit normal load).
    pub rolling_resist: f32,
    /// The maximum steer angle (radians) at full steering input.
    pub max_steer: f32,
}

impl Default for VehicleConfig3 {
    fn default() -> Self {
        let mount = WheelMount3 {
            offset: Vec3::ZERO,
            steer: false,
            drive: false,
        };
        Self {
            wheels: [mount; MAX_WHEELS],
            wheel_count: 0,
            rest_length: 0.35,
            spring_k: 22000.0,
            damping_c: 2600.0,
            max_suspension_force: 30000.0,
            wheel_radius: 0.33,
            engine_force: 9000.0,
            brake_force: 14000.0,
            grip_lateral: 600.0,
            friction_mu: 1.1,
            drag_k: 0.9,
            rolling_resist: 0.02,
            max_steer: 0.55,
        }
    }
}

/// The driver's hands, every step. Each channel is clamped to its
/// range: throttle `-1..1` (reverse included), the rest `0..1`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VehicleInput3 {
    pub throttle: f32,
    pub brake: f32,
    pub steering: f32,
    pub handbrake: f32,
}

/// What one wheel did this step: the renderer's feed and the tuner's
/// window.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct WheelTelemetry3 {
    /// Whether the suspension ray touched ground.
    pub in_contact: bool,
    /// The suspension compression, `0` at full droop, `1` bottomed.
    pub compression01: f32,
    /// The suspension force (N): the tire's normal load.
    pub normal_load: f32,
    /// The longitudinal tire force (N, along the wheel forward).
    pub force_long: f32,
    /// The lateral tire force (N, along the wheel lateral).
    pub force_lat: f32,
    /// The rolling spin (rad/s, visual only): `v_long / R`.
    pub spin: f32,
}

/// A smooth sign with a dead zone: `v / 0.5` clamped to `-1..1`, so a
/// resting wheel feels no force jitter.
fn smooth_sign(v: f32) -> f32 {
    (v / 0.5).clamp(-1.0, 1.0)
}

/// Rotate `v` around the unit axis `up` by `angle`, by Rodrigues'
/// formula with the deterministic trigonometry — never the platform
/// `sin_cos`, so both engines agree bit for bit.
fn rotate_around_up(v: Vec3, up: Vec3, angle: f32) -> Vec3 {
    let (c, s) = compute_cos_sin(angle);
    // v cosθ + (up × v) sinθ + up (up·v) (1 − cosθ).
    v * c + up.cross(v) * s + up * (up.dot(v) * (1.0 - c))
}

/// The rigid-body velocity at a world point: `v + ω × r`.
fn velocity_at_point(linear: Vec3, angular: Vec3, centre: Vec3, point: Vec3) -> Vec3 {
    linear + angular.cross(point - centre)
}

/// A suspension ray that refuses to see the chassis: the world's
/// raycast loop, traversal order and tie-break, minus one body.
/// Convex shapes report a miss, exactly as the world raycast does.
fn suspension_ray(
    world: &World3,
    chassis: usize,
    origin: Vec3,
    direction: Vec3,
    max_t: f32,
) -> Option<WorldRayHit3> {
    let filter = QueryFilter::default();
    let ray = Ray3 { origin, direction };
    let mut best: Option<WorldRayHit3> = None;
    for body in &world.bodies {
        if body.id == chassis {
            continue;
        }
        if !filter.accepts(body.collision_group, body.collision_mask) {
            continue;
        }
        if body.is_sensor {
            continue;
        }
        let Some(shape) = body.shape.as_ref() else {
            continue;
        };
        let hit = match shape {
            Shape3::Sphere { radius } => {
                ray_sphere3(ray.origin, ray.direction, body.transform.position, *radius)
            }
            Shape3::Box { half_extents } => ray_obb3(
                ray.origin,
                ray.direction,
                body.transform.position,
                body.transform.rotation,
                *half_extents,
            ),
            Shape3::Capsule {
                radius,
                half_length,
            } => {
                let axis = rotate_vec3(body.transform.rotation, Vec3::Y) * *half_length;
                ray_capsule3(
                    ray.origin,
                    ray.direction,
                    body.transform.position + axis,
                    body.transform.position - axis,
                    *radius,
                )
            }
            Shape3::ConvexHull { .. } | Shape3::ConvexMesh { .. } => None,
        };
        let Some(hit) = hit else { continue };
        if hit.t > max_t {
            continue;
        }
        let nearer = match best {
            Some(current) => hit.t < current.t,
            None => true,
        };
        if nearer {
            best = Some(WorldRayHit3 {
                body: body.id as u32,
                t: hit.t,
                point: hit.point,
                normal: hit.normal,
            });
        }
    }
    best
}

/// Run one vehicle step: raycast the suspension, push the springs and
/// the tires, fill `telemetry`, and return the wheels processed.
///
/// Call it before [`World3::step`] with the same `dt`: the forces
/// integrate straight into the chassis velocities, exactly like every
/// other external force. Returns 0 and touches nothing for an unknown
/// chassis, no wheels, or a non-positive `dt`.
pub fn step_vehicle3(
    world: &mut World3,
    chassis: usize,
    config: &VehicleConfig3,
    input: &VehicleInput3,
    dt: f32,
    telemetry: &mut [WheelTelemetry3],
) -> usize {
    if !dt.is_finite() || dt <= 0.0 {
        return 0;
    }
    let count = config.wheel_count.min(MAX_WHEELS).min(telemetry.len());
    if count == 0 {
        return 0;
    }
    let Some(chassis_body) = world.bodies.get(chassis).filter(|b| b.id == chassis) else {
        return 0;
    };
    if chassis_body.is_static {
        // A parked chassis still reports its suspension, but forces are
        // refused exactly as any external force is.
    }
    let throttle = input.throttle.clamp(-1.0, 1.0);
    let brake = input.brake.clamp(0.0, 1.0);
    let steering = input.steering.clamp(-1.0, 1.0);
    let handbrake = input.handbrake.clamp(0.0, 1.0);

    let rest = config.rest_length.max(1e-3);
    let radius = config.wheel_radius.max(1e-3);
    // The nominal load per wheel: the weight the springs share at rest.
    // Without gravity nothing weighs anything, and the tires go quiet.
    let gravity = world.gravity.length();
    let chassis_mass = chassis_body.mass;
    let nominal = chassis_mass * gravity / count.max(1) as f32;
    let nominal = nominal.max(1e-6);
    let driven = config.wheels[..count].iter().filter(|w| w.drive).count();
    let engine_each = if driven == 0 {
        0.0
    } else {
        config.engine_force / driven as f32
    };
    let brake_each = config.brake_force / count as f32;

    // The chassis state, read once: every wheel's rates come from the
    // same instant, so the processing order cannot lean the car.
    let snapshot = world.bodies[chassis].clone();
    let fwd0 = snapshot.transform.rotation * Vec3::Z;
    let up0 = snapshot.transform.rotation * Vec3::Y;
    // The forces, deferred: one suspension push per wheel at its
    // hardpoint, one tire push at its contact. Nothing mutates the
    // chassis until every wheel has been measured.
    let mut susp_force = [Vec3::ZERO; MAX_WHEELS];
    let mut susp_point = [Vec3::ZERO; MAX_WHEELS];
    let mut tire_force = [Vec3::ZERO; MAX_WHEELS];
    let mut tire_point = [Vec3::ZERO; MAX_WHEELS];
    let mut has_susp = [false; MAX_WHEELS];
    let mut has_tire = [false; MAX_WHEELS];

    for (i, mount) in config.wheels[..count].iter().enumerate() {
        let mut tele = WheelTelemetry3::default();
        let hard = snapshot.transform.position + snapshot.transform.rotation * mount.offset;
        let dir = -up0;
        let hit = suspension_ray(world, chassis, hard, dir, rest + radius);
        let Some(hit) = hit else {
            // Airborne: no spring, no tire, no telemetry but the miss.
            telemetry[i] = tele;
            continue;
        };
        let axle_height = hit.t - radius;
        let compression = (rest - axle_height).clamp(0.0, rest);
        tele.in_contact = true;
        tele.compression01 = compression / rest;
        // The spring and the damper oppose the closing speed of the
        // hardpoint onto the ground's own motion.
        let chassis_vel = velocity_at_point(
            snapshot.velocity,
            snapshot.angular_velocity,
            snapshot.transform.position,
            hard,
        );
        let ground_vel = world.bodies.get(hit.body as usize).map_or(Vec3::ZERO, |g| {
            velocity_at_point(
                g.velocity,
                g.angular_velocity,
                g.transform.position,
                hit.point,
            )
        });
        let closing = (chassis_vel - ground_vel).dot(dir);
        let spring = (config.spring_k * compression + config.damping_c * closing)
            .clamp(0.0, config.max_suspension_force);
        tele.normal_load = spring;
        // The chassis feels the spring at the hardpoint, along its up.
        susp_force[i] = up0 * spring;
        susp_point[i] = hard;
        has_susp[i] = true;

        // The tire: steered forward, load-scaled, friction-capped.
        let steer_angle = if mount.steer {
            steering * config.max_steer
        } else {
            0.0
        };
        let wheel_fwd = rotate_around_up(fwd0, up0, steer_angle);
        let wheel_lat = up0.cross(wheel_fwd);
        let contact_vel = velocity_at_point(
            snapshot.velocity,
            snapshot.angular_velocity,
            snapshot.transform.position,
            hit.point,
        );
        let v_long = contact_vel.dot(wheel_fwd);
        let v_lat = contact_vel.dot(wheel_lat);
        let load = (spring / nominal).clamp(0.0, 1.0);
        let mut force_long = 0.0f32;
        if mount.drive {
            force_long += throttle * engine_each * load;
            // The handbrake leans on the driven wheels extra hard.
            force_long -= handbrake * config.brake_force * 0.75 * load * smooth_sign(v_long);
        }
        force_long -= brake * brake_each * load * smooth_sign(v_long);
        force_long -= config.rolling_resist * spring * smooth_sign(v_long);
        let mut grip = config.grip_lateral;
        if mount.drive {
            // A locked driven wheel slides: the handbrake steals grip.
            grip *= 1.0 - 0.9 * handbrake;
        }
        let mut force_lat = -grip * v_lat * (spring / nominal);
        // The friction circle caps the pair, never each alone.
        let cap = config.friction_mu * spring;
        let total = (force_long * force_long + force_lat * force_lat).sqrt();
        if total > cap {
            let s = cap / total;
            force_long *= s;
            force_lat *= s;
        }
        tele.force_long = force_long;
        tele.force_lat = force_lat;
        tele.spin = v_long / radius;
        tire_force[i] = wheel_fwd * force_long + wheel_lat * force_lat;
        tire_point[i] = hit.point;
        has_tire[i] = true;
        telemetry[i] = tele;
    }

    for i in 0..count {
        if has_susp[i] {
            world.bodies[chassis].apply_force(susp_force[i], susp_point[i], dt);
        }
        if has_tire[i] {
            world.bodies[chassis].apply_force(tire_force[i], tire_point[i], dt);
        }
    }

    // The air: quadratic drag at the centre, no torque.
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
