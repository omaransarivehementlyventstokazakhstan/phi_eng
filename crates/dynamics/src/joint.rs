//! Joint constraints (2D): distance, hinge, prismatic, fixed.
//!
//! Joints are bilateral position constraints solved with the same
//! sequential-impulse machinery as contacts, but they persist across
//! frames: each joint owns its accumulated impulse, so warm starting is
//! automatic (no contact-cache matching needed).
//!
//! Integration contract (see [`crate::World2::step`]):
//! - Joints solve AFTER contacts each iteration, for
//!   [`crate::Solver::iterations`] iterations, so joints see settled velocities.
//! - World anchors are recomputed from local anchors every iteration
//!   (bodies move under iteration).
//! - Removal: joints hold body ids; [`crate::World2::remove_body`] drops joints
//!   touching the removed body and remaps survivors through the index
//!   shift (same bookkeeping as the contact cache).
//!
//! Boundary rule (never-guess policy, see [`crate::ShapeError`]): ambiguous
//! joint descriptions are rejected at creation — non-finite anchors,
//! negative rest length, unknown/stale body ids, or a joint from a body
//! to itself. `compliance > 0` is rejected with
//! [`JointError::UnsupportedCompliance`]: v1 joints are rigid, and
//! silently treating a spring as a rod would diverge the user's mental
//! model from the simulated state. Compliance lands in a follow-up.

use super::body::RigidBody2;
use glam::Vec2;
use phieng_math::{EPS, EPS_SQ};

/// Rejection reasons for joint creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointError {
    /// Unknown or stale body id.
    UnknownBody,
    /// Both ends attached to the same body.
    SameBody,
    /// NaN or infinite anchor, rest length, limit, or axis.
    NonFiniteInput,
    /// Negative rest length, inverted limits, or zero axis.
    InvalidParams,
    /// Non-zero compliance: v1 joints are rigid (see module docs).
    UnsupportedCompliance,
}

impl std::fmt::Display for JointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JointError::UnknownBody => write!(f, "unknown or stale body id"),
            JointError::SameBody => write!(f, "joint requires two distinct bodies"),
            JointError::NonFiniteInput => {
                write!(f, "non-finite (NaN/inf) joint anchor, length, or axis")
            }
            JointError::InvalidParams => write!(
                f,
                "invalid joint params (negative length, inverted limits, zero axis)"
            ),
            JointError::UnsupportedCompliance => write!(
                f,
                "non-zero compliance not supported in v1 (rigid joints only)"
            ),
        }
    }
}

impl std::error::Error for JointError {}

/// Positional correction strength for joints (fraction of the positional
/// error corrected per step, scaled by 1/dt like contact Baumgarte).
/// Matches the contact solver's feel; tuned jointly, not per joint.
pub const JOINT_BAUMGARTE: f32 = phieng_math::DEFAULT_BAUMGARTE;

/// A rigid rod between two anchor points: |world_b - world_a| == rest_length.
/// Bilateral (pushes AND pulls), so the accumulated impulse is unclamped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceJoint {
    /// Joint id (doubles as index into [`crate::World2::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec2,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec2,
    /// Rest length (world units, >= 0).
    pub rest_length: f32,
    /// Accumulated impulse (warm start, persists across steps).
    pub impulse: f32,
}

/// Any joint type. The solver dispatches per variant each iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Joint2 {
    Distance(DistanceJoint),
    Hinge(HingeJoint),
    Prismatic(PrismaticJoint),
    Fixed(FixedJoint),
}

/// Sleep gate shared by joint warm-start and solve. Returns true when
/// the joint must be skipped (both ends frozen: static or sleeping).
/// A sleeping end with an awake partner wakes so joints track moving bodies.
fn joint_sleep_gate(bodies: &mut [RigidBody2], a: usize, b: usize) -> bool {
    let a_frozen = bodies[a].is_static || bodies[a].is_sleeping;
    let b_frozen = bodies[b].is_static || bodies[b].is_sleeping;
    if a_frozen && b_frozen {
        return true;
    }
    if bodies[a].is_sleeping {
        bodies[a].is_sleeping = false;
    }
    if bodies[b].is_sleeping {
        bodies[b].is_sleeping = false;
    }
    false
}

impl Joint2 {
    /// Joint id.
    pub fn id(&self) -> usize {
        match self {
            Joint2::Distance(j) => j.id,
            Joint2::Hinge(j) => j.id,
            Joint2::Prismatic(j) => j.id,
            Joint2::Fixed(j) => j.id,
        }
    }

    /// Whether this joint carries motor configuration (a set speed or
    /// budget, even on a disabled motor): the snapshot schema gate.
    pub fn has_motor(&self) -> bool {
        match self {
            Joint2::Hinge(h) => h.motor_speed != 0.0 || h.motor_max_torque != 0.0,
            Joint2::Prismatic(p) => p.motor_speed != 0.0 || p.motor_max_force != 0.0,
            Joint2::Distance(_) | Joint2::Fixed(_) => false,
        }
    }

    /// Assign the joint id (used by `World2` to preserve id == index).
    pub fn set_id(&mut self, id: usize) {
        match self {
            Joint2::Distance(j) => j.id = id,
            Joint2::Hinge(j) => j.id = id,
            Joint2::Prismatic(j) => j.id = id,
            Joint2::Fixed(j) => j.id = id,
        }
    }

    /// Shift survivor endpoints down after a body removal (`Vec::remove`
    /// semantics: ids above the gap drop by 1). Caller drops joints
    /// touching the removed body first.
    pub fn shift(&mut self, removed: usize) {
        match self {
            Joint2::Distance(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint2::Hinge(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint2::Prismatic(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint2::Fixed(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
        }
    }

    /// Body ids touched by this joint.
    pub fn bodies(&self) -> (usize, usize) {
        match self {
            Joint2::Distance(j) => (j.body_a, j.body_b),
            Joint2::Hinge(j) => (j.body_a, j.body_b),
            Joint2::Prismatic(j) => (j.body_a, j.body_b),
            Joint2::Fixed(j) => (j.body_a, j.body_b),
        }
    }

    /// Apply the stored impulse along the CURRENT constraint direction
    /// before iterating (warm start). Direction is recomputed from live
    /// state, so a stale impulse can never push the wrong way — the
    /// scalar is reused, never the vector. (Hinge point axes are
    /// world-fixed X/Y, so its stored vector is directly reusable.)
    /// A joint with both ends frozen (static or sleeping) is skipped;
    /// a sleeping end with an awake partner wakes first.
    pub fn warm_start(&self, bodies: &mut [RigidBody2]) {
        let (a, b) = self.bodies();
        if joint_sleep_gate(bodies, a, b) {
            return;
        }
        match self {
            Joint2::Distance(j) => j.warm_start(bodies),
            Joint2::Hinge(j) => j.warm_start(bodies),
            Joint2::Prismatic(j) => j.warm_start(bodies),
            Joint2::Fixed(j) => j.warm_start(bodies),
        }
    }

    /// One sequential-impulse iteration. Same sleep gate as
    /// [`Joint2::warm_start`].
    pub fn solve(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let (a, b) = self.bodies();
        if joint_sleep_gate(bodies, a, b) {
            return;
        }
        match self {
            Joint2::Distance(j) => j.solve(bodies, dt),
            Joint2::Hinge(j) => j.solve(bodies, dt),
            Joint2::Prismatic(j) => j.solve(bodies, dt),
            Joint2::Fixed(j) => j.solve(bodies, dt),
        }
    }

    /// Remap body ids through a removal shift (`Vec::remove` semantics:
    /// survivor ids above the gap drop by 1). Returns `false` when the
    /// joint touched the removed body (caller drops it).
    pub fn remap(&mut self, removed: usize) -> bool {
        match self {
            Joint2::Distance(j) => {
                if j.body_a == removed || j.body_b == removed {
                    return false;
                }
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
                true
            }
            Joint2::Hinge(j) => {
                if j.body_a == removed || j.body_b == removed {
                    return false;
                }
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
                true
            }
            Joint2::Prismatic(j) => {
                if j.body_a == removed || j.body_b == removed {
                    return false;
                }
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
                true
            }
            Joint2::Fixed(j) => {
                if j.body_a == removed || j.body_b == removed {
                    return false;
                }
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
                true
            }
        }
    }
}

impl DistanceJoint {
    /// World-space anchor on A.
    fn world_a(&self, bodies: &[RigidBody2]) -> Vec2 {
        let b = &bodies[self.body_a];
        b.transform.transform_point(self.anchor_a)
    }

    /// World-space anchor on B.
    fn world_b(&self, bodies: &[RigidBody2]) -> Vec2 {
        let b = &bodies[self.body_b];
        b.transform.transform_point(self.anchor_b)
    }

    /// Constraint axis (unit, A -> B) and separation. Returns `None`
    /// when the anchors coincide: with no direction the impulse would
    /// be a guess, so the frame is skipped (bounded, self-heals as the
    /// bodies separate — same policy as the never-guess contact path).
    fn axis(&self, bodies: &[RigidBody2]) -> Option<(Vec2, f32)> {
        let delta = self.world_b(bodies) - self.world_a(bodies);
        let dist = delta.length();
        if dist > EPS {
            Some((delta / dist, dist))
        } else if self.rest_length <= EPS {
            None
        } else {
            let va = bodies[self.body_a].velocity;
            let vb = bodies[self.body_b].velocity;
            let rel_vel = vb - va;
            if rel_vel.length_squared() > EPS_SQ {
                Some((rel_vel.normalize(), 0.0))
            } else {
                Some((Vec2::X, 0.0))
            }
        }
    }

    fn warm_start(&self, bodies: &mut [RigidBody2]) {
        if self.impulse == 0.0 {
            return;
        }
        let Some((n, _)) = self.axis(bodies) else {
            return;
        };
        // P = -n * impulse on B (see solve() for the sign convention).
        apply_pair_impulse(
            bodies,
            self.body_a,
            self.body_b,
            -n * self.impulse,
            self.world_a(bodies),
            self.world_b(bodies),
        );
    }

    /// One sequential-impulse iteration for C = dist - rest_length.
    ///
    /// Sign convention (matches contact A -> B normal handling):
    /// stretched (C > 0) with λ > 0 pulls B back along -n and A
    /// forward along +n, i.e. P_on_B = -n * λ.
    /// λ = m * (vn + bias), bias = β * C / dt, bilateral (no clamp).
    fn solve(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let Some((n, dist)) = self.axis(bodies) else {
            return;
        };
        let c = dist - self.rest_length;
        // Resting at rest length: nothing to do (avoids jitter from
        // bias on a satisfied constraint).
        if c.abs() < 1e-6 {
            // Still damp separating/approaching velocity? No — a rigid
            // rod only constrains position; velocity along a satisfied
            // axis is free. Return early.
            return;
        }

        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let (ra, rb) = (
            pa - bodies[self.body_a].transform.position,
            pb - bodies[self.body_b].transform.position,
        );

        let va = bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
        let vb = bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
        let vn = (vb - va).dot(n);

        let m = effective_mass(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].inv_inertia,
            bodies[self.body_b].inv_inertia,
            ra,
            rb,
            n,
        );
        if m == 0.0 {
            return;
        }

        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = m * (vn + bias);
        self.impulse += lambda;
        apply_pair_impulse(bodies, self.body_a, self.body_b, -n * lambda, pa, pb);
    }
}

/// A revolute (hinge) joint: the two local anchors must coincide
/// (point-to-point, 2 linear constraints), rotation is free unless
/// angle limits are enforced (1 angular constraint when out of range).
///
/// Limits convention: `limit_min <= limit_max` enforces the range
/// (radians, on the wrapped relative angle); `limit_min > limit_max`
/// means a FREE hinge. Both bounds must be finite either way — there
/// is no NaN/null "no limit" encoding to misread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HingeJoint {
    /// Joint id (doubles as index into [`crate::World2::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec2,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec2,
    /// Relative-angle lower bound (radians). See type docs for the
    /// free-hinge convention.
    pub limit_min: f32,
    /// Relative-angle upper bound (radians).
    pub limit_max: f32,
    /// Accumulated point impulse in world frame (warm start; axes are
    /// world-fixed X/Y so the vector is directly reusable).
    pub impulse: Vec2,
    /// Accumulated limit impulse (warm start).
    pub angular_impulse: f32,
    /// Motor target relative angular velocity (rad/s, B minus A).
    /// Only acts when `motor_max_torque > 0`.
    pub motor_speed: f32,
    /// Motor torque budget (N·m). Zero disables the motor.
    pub motor_max_torque: f32,
    /// Accumulated motor impulse (warm start, clamped to the budget).
    pub motor_impulse: f32,
}

impl HingeJoint {
    /// World-space anchor on A.
    fn world_a(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    /// World-space anchor on B.
    fn world_b(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    /// Relative angle θb - θa wrapped to [-π, π].
    fn relative_angle(&self, bodies: &[RigidBody2]) -> f32 {
        wrap_angle(
            rotation_angle_of(&bodies[self.body_b]) - rotation_angle_of(&bodies[self.body_a]),
        )
    }

    fn warm_start(&self, bodies: &mut [RigidBody2]) {
        if self.impulse == Vec2::ZERO && self.angular_impulse == 0.0 {
            return;
        }
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.impulse != Vec2::ZERO {
            // Stored axes are world X/Y: apply directly.
            apply_pair_impulse(bodies, self.body_a, self.body_b, self.impulse, pa, pb);
        }
        if self.angular_impulse != 0.0 {
            let lam = self.angular_impulse;
            let (a, b) = if self.body_a < self.body_b {
                let (head, tail) = bodies.split_at_mut(self.body_b);
                (&mut head[self.body_a], &mut tail[0])
            } else {
                let (head, tail) = bodies.split_at_mut(self.body_a);
                (&mut tail[0], &mut head[self.body_b])
            };
            a.angular_velocity -= lam * a.inv_inertia;
            b.angular_velocity += lam * b.inv_inertia;
        }
        if self.motor_impulse != 0.0 {
            let lam = self.motor_impulse;
            let (a, b) = if self.body_a < self.body_b {
                let (head, tail) = bodies.split_at_mut(self.body_b);
                (&mut head[self.body_a], &mut tail[0])
            } else {
                let (head, tail) = bodies.split_at_mut(self.body_a);
                (&mut tail[0], &mut head[self.body_b])
            };
            a.angular_velocity -= lam * a.inv_inertia;
            b.angular_velocity += lam * b.inv_inertia;
        }
    }

    /// One iteration: point coincidence (2 axes) + motor if budgeted
    /// + angle limits if set.
    ///
    /// The motor runs before the limits so a stalled motor leans on
    /// the stops instead of through them.
    fn solve(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        self.solve_point(bodies, dt);
        self.solve_motor(bodies, dt);
        self.solve_limits(bodies, dt);
    }

    /// Point-to-point along world X and Y. C = (pb - pa) · axis.
    /// λ = -m * (vn + β * C / dt), bilateral.
    fn solve_point(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let delta = pb - pa;
        if delta.length_squared() < 1e-12 {
            return; // coincident: satisfied, avoids jitter bias.
        }
        for axis in [Vec2::X, Vec2::Y] {
            let c = delta.dot(axis);
            if c.abs() < 1e-6 {
                continue;
            }
            let ra = pa - bodies[self.body_a].transform.position;
            let rb = pb - bodies[self.body_b].transform.position;
            let va = bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
            let vb = bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
            let vn = (vb - va).dot(axis);
            let m = effective_mass(
                bodies[self.body_a].inv_mass,
                bodies[self.body_b].inv_mass,
                bodies[self.body_a].inv_inertia,
                bodies[self.body_b].inv_inertia,
                ra,
                rb,
                axis,
            );
            if m == 0.0 {
                continue;
            }
            let bias = if dt > 0.0 {
                JOINT_BAUMGARTE * c / dt
            } else {
                0.0
            };
            let lambda = -m * (vn + bias);
            // Accumulate per-axis into the stored world-frame vector.
            self.impulse += axis * lambda;
            apply_pair_impulse(bodies, self.body_a, self.body_b, axis * lambda, pa, pb);
        }
    }

    /// Velocity motor on the hinge axis. C = w_rel - motor_speed;
    /// λ = -m * C, with the ACCUMULATED impulse clamped to the torque
    /// budget (max_torque * dt). Hits the target speed exactly when the
    /// budget covers it, slips below it when stalled. Off when the
    /// budget is zero.
    fn solve_motor(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        if self.motor_max_torque <= 0.0 {
            return;
        }
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        let k = bodies[self.body_a].inv_inertia + bodies[self.body_b].inv_inertia;
        if k <= 0.0 {
            return;
        }
        let m_ang = 1.0 / k;
        let lambda = -m_ang * (w_rel - self.motor_speed);
        let max_impulse = self.motor_max_torque * dt.max(0.0);
        let old = self.motor_impulse;
        self.motor_impulse = (old + lambda).clamp(-max_impulse, max_impulse);
        let applied = self.motor_impulse - old;
        let (a, b) = if self.body_a < self.body_b {
            let (head, tail) = bodies.split_at_mut(self.body_b);
            (&mut head[self.body_a], &mut tail[0])
        } else {
            let (head, tail) = bodies.split_at_mut(self.body_a);
            (&mut tail[0], &mut head[self.body_b])
        };
        a.angular_velocity -= applied * a.inv_inertia;
        b.angular_velocity += applied * b.inv_inertia;
    }

    /// Angle limits. C = θ - clamp(θ) (0 inside range).
    /// λ = -m_ang * (w_rel + β * C / dt), bilateral within the active
    /// side. Skipped entirely for free hinges (min > max).
    fn solve_limits(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        if self.limit_min <= self.limit_max {
            let theta = self.relative_angle(bodies);
            let c = if theta < self.limit_min {
                theta - self.limit_min
            } else if theta > self.limit_max {
                theta - self.limit_max
            } else {
                return; // inside range: free.
            };
            let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
            let k = bodies[self.body_a].inv_inertia + bodies[self.body_b].inv_inertia;
            if k <= 0.0 {
                return;
            }
            let m_ang = 1.0 / k;
            let bias = if dt > 0.0 {
                JOINT_BAUMGARTE * c / dt
            } else {
                0.0
            };
            let lambda = -m_ang * (w_rel + bias);
            self.angular_impulse += lambda;
            // Paired angular apply (split borrow, same pattern as contacts).
            let (a, b) = if self.body_a < self.body_b {
                let (head, tail) = bodies.split_at_mut(self.body_b);
                (&mut head[self.body_a], &mut tail[0])
            } else {
                let (head, tail) = bodies.split_at_mut(self.body_a);
                (&mut tail[0], &mut head[self.body_b])
            };
            a.angular_velocity -= lambda * a.inv_inertia;
            b.angular_velocity += lambda * b.inv_inertia;
        }
    }
}

impl HingeJoint {
    /// Set the velocity motor: target relative angular velocity plus
    /// torque budget. A zero budget disables the motor. The
    /// accumulated impulse survives retuning (the per-step clamp pulls
    /// it inside a shrunk budget on its own).
    pub fn set_motor(&mut self, speed: f32, max_torque: f32) -> Result<(), JointError> {
        if !speed.is_finite() || !max_torque.is_finite() {
            return Err(JointError::NonFiniteInput);
        }
        if max_torque < 0.0 {
            return Err(JointError::InvalidParams);
        }
        self.motor_speed = speed;
        self.motor_max_torque = max_torque;
        Ok(())
    }
}

/// A prismatic (slider) joint: translation is free along `axis_a`
/// (A's local frame), locked perpendicular to it, and relative rotation
/// is locked to the pose at first solve (latched `ref_angle`).
///
/// Travel limits: `limit_min <= limit_max` clamps the along-axis anchor
/// separation into range; `limit_min > limit_max` means FREE travel.
/// Both bounds must be finite either way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrismaticJoint {
    /// Joint id (doubles as index into [`crate::World2::joints`]).
    pub id: usize,
    /// Rail body id.
    pub body_a: usize,
    /// Slider body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec2,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec2,
    /// Slide axis in A's LOCAL frame (unit; normalized at creation).
    pub axis_a: Vec2,
    /// Travel lower bound along the axis (world units).
    pub limit_min: f32,
    /// Travel upper bound along the axis (world units).
    pub limit_max: f32,
    /// Latched relative angle (None until first solve).
    pub ref_angle: Option<f32>,
    /// Accumulated perpendicular impulse (warm start, scalar along the
    /// CURRENT perpendicular — direction recomputed each step).
    pub perp_impulse: f32,
    /// Accumulated angular impulse (warm start).
    pub angular_impulse: f32,
    /// Accumulated limit impulse (warm start, scalar along CURRENT axis).
    pub limit_impulse: f32,
    /// Motor target slide speed along the axis (world units/s).
    /// Only acts when `motor_max_force > 0`.
    pub motor_speed: f32,
    /// Motor force budget (N). Zero disables the motor.
    pub motor_max_force: f32,
    /// Accumulated motor impulse (warm start, clamped to the budget,
    /// scalar along the CURRENT axis).
    pub motor_impulse: f32,
}

impl PrismaticJoint {
    /// Set the velocity motor: target slide speed plus force budget.
    /// A zero budget disables the motor. The accumulated impulse
    /// survives retuning (the per-step clamp pulls it inside a shrunk
    /// budget on its own).
    pub fn set_motor(&mut self, speed: f32, max_force: f32) -> Result<(), JointError> {
        if !speed.is_finite() || !max_force.is_finite() {
            return Err(JointError::NonFiniteInput);
        }
        if max_force < 0.0 {
            return Err(JointError::InvalidParams);
        }
        self.motor_speed = speed;
        self.motor_max_force = max_force;
        Ok(())
    }
}

/// A weld (fixed) joint: the two anchors must coincide AND the relative
/// angle must hold its latched value (captured at first solve). Glues
/// two bodies into one rigid assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedJoint {
    /// Joint id (doubles as index into [`crate::World2::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec2,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec2,
    /// Latched relative angle (None until first solve).
    pub ref_angle: Option<f32>,
    /// Accumulated point impulse in world frame (warm start; axes are
    /// world-fixed X/Y so the vector is directly reusable).
    pub impulse: Vec2,
    /// Accumulated angular impulse (warm start).
    pub angular_impulse: f32,
}

impl PrismaticJoint {
    fn world_a(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    fn world_b(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    /// Slide axis in world frame.
    fn axis_world(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_a].transform.rotate_vector(self.axis_a)
    }

    fn warm_start(&self, bodies: &mut [RigidBody2]) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.perp_impulse != 0.0 {
            let perp_ax = perp(self.axis_world(bodies));
            apply_pair_impulse(
                bodies,
                self.body_a,
                self.body_b,
                perp_ax * self.perp_impulse,
                pa,
                pb,
            );
        }
        if self.limit_impulse != 0.0 {
            let ax = self.axis_world(bodies);
            apply_pair_impulse(
                bodies,
                self.body_a,
                self.body_b,
                ax * self.limit_impulse,
                pa,
                pb,
            );
        }
        if self.motor_impulse != 0.0 {
            let ax = self.axis_world(bodies);
            apply_pair_impulse(
                bodies,
                self.body_a,
                self.body_b,
                ax * self.motor_impulse,
                pa,
                pb,
            );
        }
        if self.angular_impulse != 0.0 {
            let lam = self.angular_impulse;
            let (a, b) = if self.body_a < self.body_b {
                let (head, tail) = bodies.split_at_mut(self.body_b);
                (&mut head[self.body_a], &mut tail[0])
            } else {
                let (head, tail) = bodies.split_at_mut(self.body_a);
                (&mut tail[0], &mut head[self.body_b])
            };
            a.angular_velocity -= lam * a.inv_inertia;
            b.angular_velocity += lam * b.inv_inertia;
        }
    }

    fn solve(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        // Latch the reference angle on first solve (deterministic:
        // pure function of the trajectory up to this step).
        if self.ref_angle.is_none() {
            let rel = wrap_angle(
                rotation_angle_of(&bodies[self.body_b]) - rotation_angle_of(&bodies[self.body_a]),
            );
            self.ref_angle = Some(rel);
        }
        self.solve_perp(bodies, dt);
        self.solve_angular(bodies, dt);
        self.solve_motor(bodies, dt);
        self.solve_limits(bodies, dt);
    }

    /// Perpendicular lock: C = (pb - pa) · perp(axis). Bilateral.
    fn solve_perp(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let perp_ax = perp(self.axis_world(bodies));
        let c = (pb - pa).dot(perp_ax);
        if c.abs() < 1e-6 {
            return;
        }
        let ra = pa - bodies[self.body_a].transform.position;
        let rb = pb - bodies[self.body_b].transform.position;
        let va = bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
        let vb = bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
        let vn = (vb - va).dot(perp_ax);
        let m = effective_mass(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].inv_inertia,
            bodies[self.body_b].inv_inertia,
            ra,
            rb,
            perp_ax,
        );
        if m == 0.0 {
            return;
        }
        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = -m * (vn + bias);
        self.perp_impulse += lambda;
        apply_pair_impulse(bodies, self.body_a, self.body_b, perp_ax * lambda, pa, pb);
    }

    /// Rotation lock to the latched reference. C = wrap(θ - ref).
    fn solve_angular(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let Some(ref_angle) = self.ref_angle else {
            return;
        };
        let theta = wrap_angle(
            rotation_angle_of(&bodies[self.body_b]) - rotation_angle_of(&bodies[self.body_a]),
        );
        let c = wrap_angle(theta - ref_angle);
        if c.abs() < 1e-6 {
            return;
        }
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        let k = bodies[self.body_a].inv_inertia + bodies[self.body_b].inv_inertia;
        if k <= 0.0 {
            return;
        }
        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = -1.0 / k * (w_rel + bias);
        self.angular_impulse += lambda;
        let (a, b) = if self.body_a < self.body_b {
            let (head, tail) = bodies.split_at_mut(self.body_b);
            (&mut head[self.body_a], &mut tail[0])
        } else {
            let (head, tail) = bodies.split_at_mut(self.body_a);
            (&mut tail[0], &mut head[self.body_b])
        };
        a.angular_velocity -= lambda * a.inv_inertia;
        b.angular_velocity += lambda * b.inv_inertia;
    }

    /// Velocity motor along the slide axis. C = v_rel - motor_speed;
    /// λ = -m * C, with the ACCUMULATED impulse clamped to the force
    /// budget (max_force * dt). Hits the target slide speed exactly
    /// when the budget covers it. Off when the budget is zero.
    fn solve_motor(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        if self.motor_max_force <= 0.0 {
            return;
        }
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let ax = self.axis_world(bodies);
        let ra = pa - bodies[self.body_a].transform.position;
        let rb = pb - bodies[self.body_b].transform.position;
        let va = bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
        let vb = bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
        let vn = (vb - va).dot(ax);
        let m = effective_mass(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].inv_inertia,
            bodies[self.body_b].inv_inertia,
            ra,
            rb,
            ax,
        );
        if m == 0.0 {
            return;
        }
        let lambda = -m * (vn - self.motor_speed);
        let max_impulse = self.motor_max_force * dt.max(0.0);
        let old = self.motor_impulse;
        self.motor_impulse = (old + lambda).clamp(-max_impulse, max_impulse);
        let applied = self.motor_impulse - old;
        apply_pair_impulse(bodies, self.body_a, self.body_b, ax * applied, pa, pb);
    }

    /// Travel limits along the axis. s = (pb - pa) · axis;
    /// C = s - clamp(s). Bilateral on the active side; free travel
    /// (min > max) skips entirely.
    fn solve_limits(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        if self.limit_min > self.limit_max {
            return;
        }
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let ax = self.axis_world(bodies);
        let s = (pb - pa).dot(ax);
        let c = if s < self.limit_min {
            s - self.limit_min
        } else if s > self.limit_max {
            s - self.limit_max
        } else {
            return;
        };
        let ra = pa - bodies[self.body_a].transform.position;
        let rb = pb - bodies[self.body_b].transform.position;
        let va = bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
        let vb = bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
        let vn = (vb - va).dot(ax);
        let m = effective_mass(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].inv_inertia,
            bodies[self.body_b].inv_inertia,
            ra,
            rb,
            ax,
        );
        if m == 0.0 {
            return;
        }
        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = -m * (vn + bias);
        self.limit_impulse += lambda;
        apply_pair_impulse(bodies, self.body_a, self.body_b, ax * lambda, pa, pb);
    }
}

impl FixedJoint {
    fn world_a(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    fn world_b(&self, bodies: &[RigidBody2]) -> Vec2 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    fn warm_start(&self, bodies: &mut [RigidBody2]) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.impulse != Vec2::ZERO {
            apply_pair_impulse(bodies, self.body_a, self.body_b, self.impulse, pa, pb);
        }
        if self.angular_impulse != 0.0 {
            let lam = self.angular_impulse;
            let (a, b) = if self.body_a < self.body_b {
                let (head, tail) = bodies.split_at_mut(self.body_b);
                (&mut head[self.body_a], &mut tail[0])
            } else {
                let (head, tail) = bodies.split_at_mut(self.body_a);
                (&mut tail[0], &mut head[self.body_b])
            };
            a.angular_velocity -= lam * a.inv_inertia;
            b.angular_velocity += lam * b.inv_inertia;
        }
    }

    fn solve(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        if self.ref_angle.is_none() {
            let rel = wrap_angle(
                rotation_angle_of(&bodies[self.body_b]) - rotation_angle_of(&bodies[self.body_a]),
            );
            self.ref_angle = Some(rel);
        }
        self.solve_point(bodies, dt);
        self.solve_angular(bodies, dt);
    }

    /// Point coincidence along world X and Y (same as hinge point).
    fn solve_point(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let delta = pb - pa;
        if delta.length_squared() < 1e-12 {
            // Still need the angular lock below; point itself is satisfied.
        } else {
            for axis in [Vec2::X, Vec2::Y] {
                let c = delta.dot(axis);
                if c.abs() < 1e-6 {
                    continue;
                }
                let ra = pa - bodies[self.body_a].transform.position;
                let rb = pb - bodies[self.body_b].transform.position;
                let va =
                    bodies[self.body_a].velocity + perp(ra) * bodies[self.body_a].angular_velocity;
                let vb =
                    bodies[self.body_b].velocity + perp(rb) * bodies[self.body_b].angular_velocity;
                let vn = (vb - va).dot(axis);
                let m = effective_mass(
                    bodies[self.body_a].inv_mass,
                    bodies[self.body_b].inv_mass,
                    bodies[self.body_a].inv_inertia,
                    bodies[self.body_b].inv_inertia,
                    ra,
                    rb,
                    axis,
                );
                if m == 0.0 {
                    continue;
                }
                let bias = if dt > 0.0 {
                    JOINT_BAUMGARTE * c / dt
                } else {
                    0.0
                };
                let lambda = -m * (vn + bias);
                self.impulse += axis * lambda;
                apply_pair_impulse(bodies, self.body_a, self.body_b, axis * lambda, pa, pb);
            }
        }
    }

    /// Angle lock to the latched reference (always enforced).
    fn solve_angular(&mut self, bodies: &mut [RigidBody2], dt: f32) {
        let Some(ref_angle) = self.ref_angle else {
            return;
        };
        let theta = wrap_angle(
            rotation_angle_of(&bodies[self.body_b]) - rotation_angle_of(&bodies[self.body_a]),
        );
        let c = wrap_angle(theta - ref_angle);
        if c.abs() < 1e-6 {
            return;
        }
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        let k = bodies[self.body_a].inv_inertia + bodies[self.body_b].inv_inertia;
        if k <= 0.0 {
            return;
        }
        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = -1.0 / k * (w_rel + bias);
        self.angular_impulse += lambda;
        let (a, b) = if self.body_a < self.body_b {
            let (head, tail) = bodies.split_at_mut(self.body_b);
            (&mut head[self.body_a], &mut tail[0])
        } else {
            let (head, tail) = bodies.split_at_mut(self.body_a);
            (&mut tail[0], &mut head[self.body_b])
        };
        a.angular_velocity -= lambda * a.inv_inertia;
        b.angular_velocity += lambda * b.inv_inertia;
    }
}

/// Body rotation angle from its rotation matrix (same extraction as
/// [`RigidBody2::rotation_angle`], operating on SoA-friendly pieces).
#[inline]
fn rotation_angle_of(body: &RigidBody2) -> f32 {
    let m00 = body.transform.rotation.x_axis.x;
    let m10 = body.transform.rotation.y_axis.x;
    // Deterministic atan2: see `phieng_math::trig`.
    phieng_math::trig::atan2(-m10, m00)
}

/// Wrap an angle to [-π, π].
#[inline]
fn wrap_angle(mut a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    a %= tau;
    if a > std::f32::consts::PI {
        a -= tau;
    } else if a < -std::f32::consts::PI {
        a += tau;
    }
    a
}

/// Validate joint endpoints shared by all joint types.
#[inline]
fn validate_endpoints(body_a: usize, body_b: usize, body_count: usize) -> Result<(), JointError> {
    if body_a >= body_count || body_b >= body_count {
        return Err(JointError::UnknownBody);
    }
    if body_a == body_b {
        return Err(JointError::SameBody);
    }
    Ok(())
}

/// Validate a local anchor.
#[inline]
fn validate_anchor(anchor: Vec2) -> Result<(), JointError> {
    if !anchor.x.is_finite() || !anchor.y.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Validate the compliance field (v1: must be exactly 0).
#[inline]
fn validate_compliance(compliance: f32) -> Result<(), JointError> {
    if !compliance.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    if compliance != 0.0 {
        return Err(JointError::UnsupportedCompliance);
    }
    Ok(())
}

/// Build a validated distance joint (id assigned by the caller/World).
#[allow(clippy::too_many_arguments)]
pub fn make_distance_joint(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec2,
    anchor_b: Vec2,
    rest_length: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint2, JointError> {
    validate_endpoints(body_a, body_b, body_count)?;
    validate_anchor(anchor_a)?;
    validate_anchor(anchor_b)?;
    if !rest_length.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    if rest_length < 0.0 {
        return Err(JointError::InvalidParams);
    }
    validate_compliance(compliance)?;
    Ok(Joint2::Distance(DistanceJoint {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        rest_length,
        impulse: 0.0,
    }))
}

/// Validate hinge angle limits. Both bounds must be finite; `min > max`
/// is the explicit free-hinge convention (see [`HingeJoint`]).
#[inline]
fn validate_hinge_limits(limit_min: f32, limit_max: f32) -> Result<(), JointError> {
    if !limit_min.is_finite() || !limit_max.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Build a validated hinge (revolute) joint (id assigned by caller/World).
/// `limit_min > limit_max` selects a FREE hinge; otherwise the relative
/// angle is clamped into `[limit_min, limit_max]` radians.
#[allow(clippy::too_many_arguments)]
pub fn make_hinge_joint(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec2,
    anchor_b: Vec2,
    limit_min: f32,
    limit_max: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint2, JointError> {
    validate_endpoints(body_a, body_b, body_count)?;
    validate_anchor(anchor_a)?;
    validate_anchor(anchor_b)?;
    validate_hinge_limits(limit_min, limit_max)?;
    validate_compliance(compliance)?;
    Ok(Joint2::Hinge(HingeJoint {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        limit_min,
        limit_max,
        impulse: Vec2::ZERO,
        angular_impulse: 0.0,
        motor_speed: 0.0,
        motor_max_torque: 0.0,
        motor_impulse: 0.0,
    }))
}

/// Validate a slide/hinge axis: finite and non-zero. Returns the
/// normalized axis (canonicalization, not guessing: direction is
/// preserved, only the magnitude is standardized).
#[inline]
fn validate_axis(axis: Vec2) -> Result<Vec2, JointError> {
    if !axis.x.is_finite() || !axis.y.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    let len = axis.length();
    if len <= 1e-6 {
        return Err(JointError::InvalidParams);
    }
    Ok(axis / len)
}

/// Validate travel limits. Both bounds must be finite; `min > max` is
/// the explicit free-travel convention.
#[inline]
fn validate_travel_limits(limit_min: f32, limit_max: f32) -> Result<(), JointError> {
    if !limit_min.is_finite() || !limit_max.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Build a validated prismatic (slider) joint (id assigned by
/// caller/World). The slide axis is given in A's local frame and
/// normalized at creation.
#[allow(clippy::too_many_arguments)]
pub fn make_prismatic_joint(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec2,
    anchor_b: Vec2,
    axis_a: Vec2,
    limit_min: f32,
    limit_max: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint2, JointError> {
    validate_endpoints(body_a, body_b, body_count)?;
    validate_anchor(anchor_a)?;
    validate_anchor(anchor_b)?;
    let axis = validate_axis(axis_a)?;
    validate_travel_limits(limit_min, limit_max)?;
    validate_compliance(compliance)?;
    Ok(Joint2::Prismatic(PrismaticJoint {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        axis_a: axis,
        limit_min,
        limit_max,
        ref_angle: None,
        perp_impulse: 0.0,
        angular_impulse: 0.0,
        limit_impulse: 0.0,
        motor_speed: 0.0,
        motor_max_force: 0.0,
        motor_impulse: 0.0,
    }))
}

/// Build a validated weld (fixed) joint (id assigned by caller/World).
/// The relative angle is latched at first solve.
pub fn make_fixed_joint(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec2,
    anchor_b: Vec2,
    compliance: f32,
    body_count: usize,
) -> Result<Joint2, JointError> {
    validate_endpoints(body_a, body_b, body_count)?;
    validate_anchor(anchor_a)?;
    validate_anchor(anchor_b)?;
    validate_compliance(compliance)?;
    Ok(Joint2::Fixed(FixedJoint {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        ref_angle: None,
        impulse: Vec2::ZERO,
        angular_impulse: 0.0,
    }))
}

/// Effective mass along a direction (same formula as contact solver).
#[inline]
fn effective_mass(
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_inertia_a: f32,
    inv_inertia_b: f32,
    r_a: Vec2,
    r_b: Vec2,
    dir: Vec2,
) -> f32 {
    let ra_cross = r_a.x * dir.y - r_a.y * dir.x;
    let rb_cross = r_b.x * dir.y - r_b.y * dir.x;
    let k = inv_mass_a
        + inv_mass_b
        + inv_inertia_a * ra_cross * ra_cross
        + inv_inertia_b * rb_cross * rb_cross;
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// 90-degree CCW perpendicular (angular velocity contribution at r).
#[inline]
fn perp(r: Vec2) -> Vec2 {
    Vec2::new(-r.y, r.x)
}

/// Apply impulse P to B and -P to A through one paired borrow
/// (same single-check pattern as the contact solver).
#[inline]
fn apply_pair_impulse(
    bodies: &mut [RigidBody2],
    body_a: usize,
    body_b: usize,
    p: Vec2,
    pa: Vec2,
    pb: Vec2,
) {
    debug_assert!(body_a != body_b, "joint must connect two distinct bodies");
    let (a, b) = if body_a < body_b {
        let (head, tail) = bodies.split_at_mut(body_b);
        (&mut head[body_a], &mut tail[0])
    } else {
        let (head, tail) = bodies.split_at_mut(body_a);
        (&mut tail[0], &mut head[body_b])
    };
    let ra = pa - a.transform.position;
    let rb = pb - b.transform.position;

    a.velocity -= p * a.inv_mass;
    b.velocity += p * b.inv_mass;

    a.angular_velocity -= (ra.x * p.y - ra.y * p.x) * a.inv_inertia;
    b.angular_velocity += (rb.x * p.y - rb.y * p.x) * b.inv_inertia;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape2;

    fn world_with_ball_on_rod() -> (crate::World2, usize) {
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world
            .add_static_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
            )
            .expect("valid shape");
        let ball = world
            .add_dynamic_body(Vec2::new(0.0, -2.0), Shape2::Circle { radius: 0.3 }, 1.0)
            .expect("valid shape");
        (world, ball)
    }

    #[test]
    fn j31_distance_rejects_bad_input() {
        let mut world = crate::World2::new();
        world
            .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        world
            .add_static_body(Vec2::X, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        let n = world.bodies.len();
        let a = Vec2::ZERO;
        let b = Vec2::X;
        // Unknown body.
        assert_eq!(
            make_distance_joint(0, 0, 99, a, b, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        // Same body (endpoint check runs before anchor checks).
        assert_eq!(
            make_distance_joint(0, 0, 0, a, b, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        // NaN anchor (distinct valid bodies so the anchor check runs).
        assert_eq!(
            make_distance_joint(0, 0, 1, Vec2::new(f32::NAN, 0.0), b, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // Negative rest length.
        assert_eq!(
            make_distance_joint(0, 0, 1, a, b, -1.0, 0.0, n),
            Err(JointError::InvalidParams)
        );
        // Non-finite rest length.
        assert_eq!(
            make_distance_joint(0, 0, 1, a, b, f32::INFINITY, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // Non-zero compliance rejected (rigid-only v1).
        assert_eq!(
            make_distance_joint(0, 0, 1, a, b, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
    }

    #[test]
    fn j32_distance_holds_rest_length() {
        let (mut world, ball) = world_with_ball_on_rod();
        // Rod from anchor origin to ball center, rest length 2.
        let joint = make_distance_joint(
            0,
            0,
            ball,
            Vec2::ZERO,
            Vec2::ZERO,
            2.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        // Ball hangs straight down at rest length (rod holds, no stretch).
        let p = world.bodies[ball].transform.position;
        let dist = (p - Vec2::ZERO).length();
        assert!(
            (dist - 2.0).abs() < 0.05,
            "rod must hold rest length, dist={dist}, pos={p}"
        );
        assert!(
            p.x.abs() < 0.1,
            "pendulum at rest hangs straight down, x={}",
            p.x
        );
    }

    #[test]
    fn j33_distance_pulls_back() {
        let (mut world, ball) = world_with_ball_on_rod();
        let joint = make_distance_joint(
            0,
            0,
            ball,
            Vec2::ZERO,
            Vec2::ZERO,
            2.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Displace sideways: pendulum must swing back toward center.
        world.bodies[ball].transform.position = Vec2::new(1.0, -1.732);
        world.bodies[ball].velocity = Vec2::ZERO;
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }
        let p = world.bodies[ball].transform.position;
        assert!(
            p.x.abs() < 1.0,
            "displaced pendulum must swing back, x={}",
            p.x
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.velocity.x.is_finite());
        }
    }

    #[test]
    fn j34_joint_removed_with_body() {
        let (mut world, ball) = world_with_ball_on_rod();
        let joint = make_distance_joint(
            0,
            0,
            ball,
            Vec2::ZERO,
            Vec2::ZERO,
            2.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        assert_eq!(world.joints.len(), 1);
        world.remove_body(ball);
        assert!(
            world.joints.is_empty(),
            "joint touching removed body must drop"
        );
        // Stepping a jointless world must not crash.
        for _ in 0..10 {
            world.step(dt_for_test());
        }
    }

    fn dt_for_test() -> f32 {
        1.0 / 60.0
    }

    #[test]
    fn j39_prismatic_rejects_bad_input() {
        let mut world = crate::World2::new();
        world
            .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        world
            .add_static_body(Vec2::X, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        let n = world.bodies.len();
        let a = Vec2::ZERO;
        let b = Vec2::X;
        assert_eq!(
            make_prismatic_joint(0, 0, 99, a, b, Vec2::X, -1.0, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_prismatic_joint(0, 1, 1, a, b, Vec2::X, -1.0, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_prismatic_joint(
                0,
                0,
                1,
                Vec2::new(f32::NAN, 0.0),
                b,
                Vec2::X,
                -1.0,
                1.0,
                0.0,
                n
            ),
            Err(JointError::NonFiniteInput)
        );
        // Zero axis rejected.
        assert_eq!(
            make_prismatic_joint(0, 0, 1, a, b, Vec2::ZERO, -1.0, 1.0, 0.0, n),
            Err(JointError::InvalidParams)
        );
        // NaN axis rejected.
        assert_eq!(
            make_prismatic_joint(0, 0, 1, a, b, Vec2::new(1.0, f32::NAN), -1.0, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // NaN limits rejected.
        assert_eq!(
            make_prismatic_joint(0, 0, 1, a, b, Vec2::X, f32::NAN, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_prismatic_joint(0, 0, 1, a, b, Vec2::X, -1.0, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
        // Inverted bounds are the valid free-travel convention.
        assert!(make_prismatic_joint(0, 0, 1, a, b, Vec2::X, 1.0, -1.0, 0.0, n).is_ok());
    }

    #[test]
    fn j40_prismatic_slides_holds_rail() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        // Static rail along X at the origin.
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        let slider = world
            .add_dynamic_body(
                Vec2::new(0.0, 1.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Slider axis X in the rail's frame; anchors coincide at the
        // slider center; free travel (inverted bounds).
        let joint = make_prismatic_joint(
            0,
            0,
            slider,
            Vec2::new(0.0, 2.0),
            Vec2::ZERO,
            Vec2::X,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Shove it sideways: it must glide along X while the rail holds Y.
        world.bodies[slider].velocity = Vec2::new(5.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }
        let p = world.bodies[slider].transform.position;
        assert!(p.x > 1.0, "slider must glide along the rail, x={}", p.x);
        assert!(
            (p.y - 1.0).abs() < 0.15,
            "rail must hold the slider's height against gravity, y={}",
            p.y
        );
        assert!(p.x.is_finite() && p.y.is_finite());
    }

    #[test]
    fn j41_prismatic_limits_enforced() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::ZERO;
        world
            .add_static_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
            )
            .expect("valid shape");
        let slider = world
            .add_dynamic_body(
                Vec2::new(3.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Travel clamped to [-1, 1] along X: the rail anchor sits at the
        // static center while the slider anchor rides the slider center,
        // so s starts at 3 (beyond the max) and must be pulled back.
        let joint = make_prismatic_joint(
            0,
            0,
            slider,
            Vec2::ZERO,
            Vec2::ZERO,
            Vec2::X,
            -1.0,
            1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        let p = world.bodies[slider].transform.position;
        assert!(
            p.x < 1.2,
            "travel limit must pull the slider back, x={}",
            p.x
        );
        assert!(
            p.y.abs() < 0.15,
            "rail must hold lateral position, y={}",
            p.y
        );
    }

    #[test]
    fn j42_fixed_rejects_bad_input() {
        let mut world = crate::World2::new();
        world
            .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        world
            .add_static_body(Vec2::X, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        let n = world.bodies.len();
        assert_eq!(
            make_fixed_joint(0, 0, 99, Vec2::ZERO, Vec2::X, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_fixed_joint(0, 1, 1, Vec2::ZERO, Vec2::X, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_fixed_joint(0, 0, 1, Vec2::new(0.0, f32::NAN), Vec2::X, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_fixed_joint(0, 0, 1, Vec2::ZERO, Vec2::X, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
    }

    #[test]
    fn j43_fixed_holds_formation() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        // Two boxes glued 1 unit apart, dropped: the formation must
        // survive the fall without contacts to lean on.
        let a = world
            .add_dynamic_body(
                Vec2::new(-0.5, 5.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        let b = world
            .add_dynamic_body(
                Vec2::new(0.5, 5.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Weld B's center to a point 1 unit right of A's center.
        let joint = make_fixed_joint(
            0,
            a,
            b,
            Vec2::new(1.0, 0.0),
            Vec2::ZERO,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        let dt = 1.0 / 60.0;
        for _ in 0..180 {
            world.step(dt);
        }
        let pa = world.bodies[a].transform.position;
        let pb = world.bodies[b].transform.position;
        let gap = (pb - pa).length();
        assert!(
            (gap - 1.0).abs() < 0.1,
            "weld must hold the 1-unit formation, gap={gap}, a={pa}, b={pb}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.angular_velocity.is_finite());
        }
    }

    #[test]
    fn j44_fixed_resists_rotation() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::ZERO;
        let a = world
            .add_dynamic_body(
                Vec2::new(-0.5, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        let b = world
            .add_dynamic_body(
                Vec2::new(0.5, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        world.bodies[a].linear_damping = 0.0;
        world.bodies[a].angular_damping = 0.0;
        world.bodies[b].linear_damping = 0.0;
        world.bodies[b].angular_damping = 0.0;
        let joint = make_fixed_joint(
            0,
            a,
            b,
            Vec2::new(1.0, 0.0),
            Vec2::ZERO,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Spin A hard: the weld must drag B along (relative angle ~ 0).
        world.bodies[a].angular_velocity = 5.0;
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }
        let rel = wrap_angle(world.bodies[b].rotation_angle() - world.bodies[a].rotation_angle());
        assert!(
            rel.abs() < 0.3,
            "weld must keep the bodies co-rotating, rel={rel}"
        );
    }

    fn world_with_hinge() -> (crate::World2, usize) {
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world
            .add_static_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
            )
            .expect("valid shape");
        let bob = world
            .add_dynamic_body(
                Vec2::new(2.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        (world, bob)
    }

    #[test]
    fn j35_hinge_rejects_bad_input() {
        let mut world = crate::World2::new();
        world
            .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        world
            .add_static_body(Vec2::X, Shape2::Circle { radius: 0.5 })
            .expect("valid shape");
        let n = world.bodies.len();
        let a = Vec2::ZERO;
        let b = Vec2::X;
        assert_eq!(
            make_hinge_joint(0, 0, 99, a, b, 0.0, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_hinge_joint(0, 1, 1, a, b, 0.0, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_hinge_joint(0, 0, 1, Vec2::new(f32::NAN, 0.0), b, 0.0, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // NaN limits rejected (free hinge uses inverted finite bounds, not NaN).
        assert_eq!(
            make_hinge_joint(0, 0, 1, a, b, f32::NAN, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_hinge_joint(0, 0, 1, a, b, 0.0, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
        // Inverted bounds are the valid free-hinge convention.
        assert!(make_hinge_joint(0, 0, 1, a, b, 1.0, -1.0, 0.0, n).is_ok());
    }

    #[test]
    fn j36_hinge_pulls_to_anchor() {
        let (mut world, bob) = world_with_hinge();
        // Free hinge (inverted bounds): pin bob's center to the origin.
        let joint = make_hinge_joint(
            0,
            0,
            bob,
            Vec2::ZERO,
            Vec2::ZERO,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        let p = world.bodies[bob].transform.position;
        assert!(
            p.length() < 0.1,
            "hinge must pull the bob to the anchor, pos={p}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.velocity.x.is_finite());
        }
    }

    #[test]
    fn j37_hinge_free_rotation() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::ZERO;
        // Static anchor at (0,-2); its local anchor (0,2) puts the world
        // hinge point at the origin — the bob's center — while the bodies
        // stay 2 units apart (no contact friction to confuse the spin
        // check). Point impulses act through the bob's center of mass,
        // so spin must persist undamped.
        world
            .add_static_body(
                Vec2::new(0.0, -2.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
            )
            .expect("valid shape");
        let bob = world
            .add_dynamic_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        world.bodies[bob].linear_damping = 0.0;
        world.bodies[bob].angular_damping = 0.0;
        let joint = make_hinge_joint(
            0,
            0,
            bob,
            Vec2::new(0.0, 2.0),
            Vec2::ZERO,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        world.bodies[bob].angular_velocity = 2.0;
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step(dt);
        }
        let angle = world.bodies[bob].rotation_angle();
        assert!(
            (angle - 2.0).abs() < 0.2,
            "free hinge must not brake spin, angle={angle}"
        );
        assert!(
            world.bodies[bob].transform.position.length() < 0.05,
            "center-pinned bob must not drift, pos={}",
            world.bodies[bob].transform.position
        );
    }

    #[test]
    fn j38_hinge_limits_enforced() {
        let (mut world, bob) = world_with_hinge();
        // Tight limits around zero: the bob may hang but not windmill.
        let joint = make_hinge_joint(
            0,
            0,
            bob,
            Vec2::ZERO,
            Vec2::ZERO,
            -0.2,
            0.2,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Wind it up and spin it hard: limits must catch it.
        world.bodies[bob].angular_velocity = 10.0;
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        let rel = wrap_angle(world.bodies[bob].rotation_angle() - world.bodies[0].rotation_angle());
        assert!(
            (-0.35..=0.35).contains(&rel),
            "hinge limits must catch the windup, rel={rel}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.angular_velocity.is_finite());
        }
    }

    fn bits(v: f32) -> u32 {
        v.to_bits()
    }

    /// A free hinge pendulum (pivot at the origin, arm below), motorized.
    fn motor_pendulum(speed: f32, torque: f32, gravity_y: f32) -> (crate::World2, usize) {
        let mut world = crate::World2::new();
        world.gravity = Vec2::new(0.0, gravity_y);
        world.linear_sleep_threshold = 0.0;
        world.angular_sleep_threshold = 0.0;
        let pivot = world
            .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.2 })
            .expect("valid shape");
        let arm = world
            .add_dynamic_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.1, 1.0),
                },
                1.0,
            )
            .expect("valid shape");
        let mut joint = make_hinge_joint(
            0,
            pivot,
            arm,
            Vec2::ZERO,
            Vec2::new(0.0, 1.0),
            1.0,
            -1.0,
            0.0,
            2,
        )
        .expect("valid joint");
        if let Joint2::Hinge(h) = &mut joint {
            h.set_motor(speed, torque).expect("valid motor");
        }
        world.add_joint(joint).expect("valid joint");
        (world, arm)
    }

    /// J45: a budgeted hinge motor spins its arm up to the target speed.
    #[test]
    fn j45_hinge_motor_reaches_target_speed() {
        let (mut world, arm) = motor_pendulum(3.0, 1000.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step(dt);
        }
        let w = world.bodies[arm].angular_velocity;
        assert_eq!(bits(w), 0x403f_f7cf);
        assert!((w - 3.0).abs() < 1e-2, "motor must reach 3 rad/s, got {w}");
    }

    /// J46: a starved motor slips below target with its impulse
    /// saturated at the budget, and bad tuning is rejected.
    #[test]
    fn j46_weak_motor_slips_with_saturated_impulse() {
        let (mut world, arm) = motor_pendulum(3.0, 0.5, -9.81);
        // Bad tuning is rejected, not clamped into silence.
        if let Joint2::Hinge(h) = &mut world.joints[0] {
            assert_eq!(h.set_motor(f32::NAN, 1.0), Err(JointError::NonFiniteInput));
            assert_eq!(h.set_motor(1.0, -1.0), Err(JointError::InvalidParams));
        } else {
            panic!("expected a hinge");
        }
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step(dt);
        }
        let w = world.bodies[arm].angular_velocity;
        let imp = match &world.joints[0] {
            Joint2::Hinge(h) => h.motor_impulse,
            _ => panic!("expected a hinge"),
        };
        assert_eq!(bits(imp), 0x3c08_8889);
        // No bit pin on w: the stalled arm is a decaying oscillation,
        // and pinning an oscillator's phase would pin solver jitter,
        // not motor physics. The slip is the bound below; the saturated
        // impulse above is the bit-exact proof the budget binds.
        assert!(
            w.abs() < 0.5,
            "starved motor must slip far below 3 rad/s, got {w}"
        );
        assert!(w > 0.0 && w < 3.0, "starved motor must slip, got {w}");
        assert!(
            (imp.abs() - 0.5 * dt).abs() < 1e-9,
            "impulse must sit on the budget {imp}"
        );
    }

    /// J47: a prismatic motor extends its slider at the target speed.
    #[test]
    fn j47_prismatic_motor_extends_at_target_speed() {
        let mut world = crate::World2::new();
        world.gravity = Vec2::ZERO;
        world.linear_sleep_threshold = 0.0;
        world.angular_sleep_threshold = 0.0;
        let rail = world
            .add_static_body(
                Vec2::ZERO,
                Shape2::Box {
                    half_extents: Vec2::new(5.0, 0.2),
                },
            )
            .expect("valid shape");
        // A sensor rail: joints ignore solidity, so the slide measures
        // the motor alone instead of wrestling contact.
        world.bodies[rail].is_sensor = true;
        let slider = world
            .add_dynamic_body(
                Vec2::ZERO,
                Shape2::Box {
                    half_extents: Vec2::new(0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        let mut joint = make_prismatic_joint(
            0,
            rail,
            slider,
            Vec2::ZERO,
            Vec2::ZERO,
            Vec2::X,
            -10.0,
            10.0,
            0.0,
            2,
        )
        .expect("valid joint");
        if let Joint2::Prismatic(p) = &mut joint {
            p.set_motor(2.0, 10000.0).expect("valid motor");
        }
        world.add_joint(joint).expect("valid joint");
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }
        let p = world.bodies[slider].transform.position;
        assert_eq!(bits(p.x), 0x407f_f522);
        assert!(
            (p.x - 4.0).abs() < 0.05,
            "slider must extend at 2 m/s, got {p}"
        );
        assert!(p.y.abs() < 0.05, "slider must stay on the rail, got {p}");
    }

    /// J48: motors round-trip through save/load (schema v16), and an
    /// unmotorized world stays on its old schema with motors off.
    #[test]
    fn j48_motors_persist_through_save_load() {
        use crate::snapshot::{
            world2_from_binary, world2_to_binary, LoadMode, SNAPSHOT_SCHEMA_VERSION,
        };
        fn json_schema(text: &str) -> u32 {
            let marker = "\"schema\": ";
            let at = text.find(marker).expect("the JSON carries a schema");
            let rest = &text[at + marker.len()..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().expect("the schema is a number")
        }
        // Motorized: schema v16, config and impulse survive Exact.
        let (mut world, arm) = motor_pendulum(3.0, 1000.0, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step(dt);
        }
        let text = crate::snapshot::world2_to_json(&world, LoadMode::Exact).expect("serializes");
        assert_eq!(
            json_schema(&text),
            SNAPSHOT_SCHEMA_VERSION,
            "a motorized world writes v16"
        );
        assert!(text.contains("motor_speed"), "motors must serialize");
        let bytes = world2_to_binary(&world, LoadMode::Exact).expect("serializes");
        let mut back = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
        match &back.joints[0] {
            Joint2::Hinge(h) => {
                assert_eq!(bits(h.motor_speed), bits(3.0));
                assert_eq!(bits(h.motor_max_torque), bits(1000.0));
            }
            _ => panic!("expected a hinge"),
        }
        for _ in 0..60 {
            world.step(dt);
            back.step(dt);
        }
        let (a, b) = (&world.bodies[arm], &back.bodies[arm]);
        assert_eq!(
            (
                bits(a.velocity.x),
                bits(a.velocity.y),
                bits(a.angular_velocity)
            ),
            (
                bits(b.velocity.x),
                bits(b.velocity.y),
                bits(b.angular_velocity)
            ),
            "restored motors must continue bit-identically"
        );
        // Unmotorized: old schema, motors off on load.
        let (mut plain, _) = motor_pendulum(0.0, 0.0, 0.0);
        for _ in 0..60 {
            plain.step(dt);
        }
        let text = crate::snapshot::world2_to_json(&plain, LoadMode::Exact).expect("serializes");
        assert!(
            json_schema(&text) < SNAPSHOT_SCHEMA_VERSION,
            "no motors means no v16"
        );
        assert!(
            !text.contains("motor_speed"),
            "no motors means no motor keys"
        );
        let bytes = world2_to_binary(&plain, LoadMode::Exact).expect("serializes");
        let back = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
        match &back.joints[0] {
            Joint2::Hinge(h) => {
                assert_eq!(h.motor_speed, 0.0);
                assert_eq!(h.motor_max_torque, 0.0);
                assert_eq!(h.motor_impulse, 0.0);
            }
            _ => panic!("expected a hinge"),
        }
    }
}
