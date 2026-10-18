//! Joint constraints (3D): distance, hinge, prismatic, fixed.
//!
//! Same contract as the 2D joints (see [`crate::joint`]), lifted to 3D:
//! bilateral position constraints solved with sequential impulses after
//! contacts each iteration; each joint owns its accumulated impulse, so
//! warm starting is automatic.
//!
//! 3D-specific design (see [`crate::World3::step`], serial only):
//! - Point coincidence is solved about the world X/Y/Z axes (3 linear
//!   constraints); the stored point impulse is a world-frame vector,
//!   directly reusable at warm start like the 2D hinge/fixed impulse.
//! - Orientation error uses the quaternion error vector `2*xyz(qb*qa^-1)`
//!   (shortest path), solved about world axes or a hinge/slide axis.
//! - Twist about a hinge axis is `2*atan2(dot(xyz, axis), w)`, wrapped to
//!   `[-pi, pi]` â€” the analog of the 2D wrapped relative angle, so hinge
//!   twist limits need no latched reference. Prismatic/fixed orientation
//!   locks DO latch `ref_orientation` at first solve (mirror of 2D
//!   `ref_angle`).
//! - World inertia tensors are rebuilt from live orientations on every
//!   impulse application (orientations never change mid-solve â€” only
//!   velocities do â€” so this is pure recompute overhead, never stale).
//! - Never-guess policy: coincident distance anchors, anti-parallel hinge
//!   axes (cross product vanishes), and zero travel all skip the frame
//!   boundedly instead of inventing a direction.
//!
//! Boundary rule (same as 2D): ambiguous descriptions are rejected at
//! creation with [`JointError`]; `compliance > 0` is rejected with
//! [`JointError::UnsupportedCompliance`] (rigid-only v1).

use super::body::RigidBody3;
use super::joint::{JointError, JOINT_BAUMGARTE};
use glam::{Mat3, Quat, Vec3};
use phieng_math::{EPS, EPS_SQ};

/// A rigid rod between two anchor points: |world_b - world_a| == rest_length.
/// Bilateral (pushes AND pulls), so the accumulated impulse is unclamped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceJoint3 {
    /// Joint id (doubles as index into [`crate::World3::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec3,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec3,
    /// Rest length (world units, >= 0).
    pub rest_length: f32,
    /// Accumulated impulse (warm start, persists across steps).
    pub impulse: f32,
}

/// A revolute (hinge) joint: the two local anchors must coincide
/// (point-to-point, 3 linear constraints), the two local hinge axes must
/// align (2 angular constraints), and twist about the hinge axis is free
/// unless twist limits are enforced (1 angular constraint when out of range).
///
/// Limits convention: `limit_min <= limit_max` enforces the twist range
/// (radians); `limit_min > limit_max` means a FREE hinge. Both bounds
/// must be finite either way â€” no NaN "no limit" encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HingeJoint3 {
    /// Joint id (doubles as index into [`crate::World3::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec3,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec3,
    /// Hinge axis in A's LOCAL frame (unit; normalized at creation).
    pub axis_a: Vec3,
    /// Hinge axis in B's LOCAL frame (unit; normalized at creation).
    pub axis_b: Vec3,
    /// Twist lower bound (radians). See type docs for free-hinge convention.
    pub limit_min: f32,
    /// Twist upper bound (radians).
    pub limit_max: f32,
    /// Accumulated point impulse in world frame (warm start; axes are
    /// world-fixed X/Y/Z so the vector is directly reusable).
    pub impulse: Vec3,
    /// Accumulated axis-alignment impulses along the CURRENT stable basis
    /// (x along u1, y along u2 â€” basis recomputed each step, same pattern
    /// as the 2D prismatic perp/limit scalars).
    pub align_impulse: Vec2Pair,
    /// Accumulated twist impulse about the CURRENT hinge axis (warm start,
    /// scalar along the recomputed axis).
    pub twist_impulse: f32,
}

/// Two packed scalars (avoids pulling a 2D vector type into 3D joint state).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2Pair {
    pub x: f32,
    pub y: f32,
}

/// A prismatic (slider) joint: translation is free along `axis_a`
/// (A's local frame), locked on the perpendicular plane (2 linear
/// constraints), and relative orientation is locked to the pose at first
/// solve (latched `ref_orientation`, 3 angular constraints).
///
/// Travel limits: `limit_min <= limit_max` clamps the along-axis anchor
/// separation into range; `limit_min > limit_max` means FREE travel.
/// Both bounds must be finite either way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrismaticJoint3 {
    /// Joint id (doubles as index into [`crate::World3::joints`]).
    pub id: usize,
    /// Rail body id.
    pub body_a: usize,
    /// Slider body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec3,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec3,
    /// Slide axis in A's LOCAL frame (unit; normalized at creation).
    pub axis_a: Vec3,
    /// Travel lower bound along the axis (world units).
    pub limit_min: f32,
    /// Travel upper bound along the axis (world units).
    pub limit_max: f32,
    /// Latched relative orientation (None until first solve).
    pub ref_orientation: Option<Quat>,
    /// Accumulated perpendicular impulses along the CURRENT stable basis
    /// (x along u1, y along u2 â€” recomputed each step).
    pub perp_impulse: Vec2Pair,
    /// Accumulated orientation-lock impulse in world frame (warm start;
    /// axes are world-fixed X/Y/Z so the vector is directly reusable).
    pub angular_impulse: Vec3,
    /// Accumulated travel-limit impulse (warm start, scalar along the
    /// CURRENT slide axis).
    pub limit_impulse: f32,
}

/// A weld (fixed) joint: the two anchors must coincide (3 linear
/// constraints) AND the relative orientation must hold its latched value
/// (3 angular constraints, captured at first solve). Glues two bodies
/// into one rigid assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedJoint3 {
    /// Joint id (doubles as index into [`crate::World3::joints`]).
    pub id: usize,
    /// First body id.
    pub body_a: usize,
    /// Second body id.
    pub body_b: usize,
    /// Anchor on A in A's local frame.
    pub anchor_a: Vec3,
    /// Anchor on B in B's local frame.
    pub anchor_b: Vec3,
    /// Latched relative orientation (None until first solve).
    pub ref_orientation: Option<Quat>,
    /// Accumulated point impulse in world frame (warm start).
    pub impulse: Vec3,
    /// Accumulated orientation-lock impulse in world frame (warm start).
    pub angular_impulse: Vec3,
}

/// Any 3D joint type. The solver dispatches per variant each iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Joint3 {
    Distance(DistanceJoint3),
    Hinge(HingeJoint3),
    Prismatic(PrismaticJoint3),
    Fixed(FixedJoint3),
}

/// Sleep gate shared by joint warm-start and solve (see `joint.rs`):
/// skip fully frozen joints, wake a sleeping end with an awake partner.
fn joint_sleep_gate(bodies: &mut [RigidBody3], a: usize, b: usize) -> bool {
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

impl Joint3 {
    /// Joint id.
    pub fn id(&self) -> usize {
        match self {
            Joint3::Distance(j) => j.id,
            Joint3::Hinge(j) => j.id,
            Joint3::Prismatic(j) => j.id,
            Joint3::Fixed(j) => j.id,
        }
    }

    /// Assign the joint id (used by `World3` to preserve id == index).
    pub fn set_id(&mut self, id: usize) {
        match self {
            Joint3::Distance(j) => j.id = id,
            Joint3::Hinge(j) => j.id = id,
            Joint3::Prismatic(j) => j.id = id,
            Joint3::Fixed(j) => j.id = id,
        }
    }

    /// Shift survivor endpoints down after a body removal (`Vec::remove`
    /// semantics: ids above the gap drop by 1). Caller drops joints
    /// touching the removed body first.
    pub fn shift(&mut self, removed: usize) {
        match self {
            Joint3::Distance(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint3::Hinge(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint3::Prismatic(j) => {
                if j.body_a > removed {
                    j.body_a -= 1;
                }
                if j.body_b > removed {
                    j.body_b -= 1;
                }
            }
            Joint3::Fixed(j) => {
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
            Joint3::Distance(j) => (j.body_a, j.body_b),
            Joint3::Hinge(j) => (j.body_a, j.body_b),
            Joint3::Prismatic(j) => (j.body_a, j.body_b),
            Joint3::Fixed(j) => (j.body_a, j.body_b),
        }
    }

    /// Apply the stored impulses along CURRENT directions before iterating
    /// (warm start). Directions are recomputed from live state, so a stale
    /// scalar can never push the wrong way; world-frame vectors (point /
    /// orientation-lock) are directly reusable. A joint with both ends
    /// frozen (static or sleeping) is skipped; a sleeping end with an
    /// awake partner wakes first.
    pub fn warm_start(&self, bodies: &mut [RigidBody3]) {
        let (a, b) = self.bodies();
        if joint_sleep_gate(bodies, a, b) {
            return;
        }
        match self {
            Joint3::Distance(j) => j.warm_start(bodies),
            Joint3::Hinge(j) => j.warm_start(bodies),
            Joint3::Prismatic(j) => j.warm_start(bodies),
            Joint3::Fixed(j) => j.warm_start(bodies),
        }
    }

    /// One sequential-impulse iteration. Same sleep gate as
    /// [`Joint3::warm_start`].
    pub fn solve(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let (a, b) = self.bodies();
        if joint_sleep_gate(bodies, a, b) {
            return;
        }
        match self {
            Joint3::Distance(j) => j.solve(bodies, dt),
            Joint3::Hinge(j) => j.solve(bodies, dt),
            Joint3::Prismatic(j) => j.solve(bodies, dt),
            Joint3::Fixed(j) => j.solve(bodies, dt),
        }
    }

    /// Remap body ids through a removal shift (`Vec::remove` semantics).
    /// Returns `false` when the joint touched the removed body (caller
    /// drops it).
    pub fn remap(&mut self, removed: usize) -> bool {
        match self {
            Joint3::Distance(j) => {
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
            Joint3::Hinge(j) => {
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
            Joint3::Prismatic(j) => {
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
            Joint3::Fixed(j) => {
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

impl DistanceJoint3 {
    /// World-space anchor on A.
    fn world_a(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    /// World-space anchor on B.
    fn world_b(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    /// Constraint axis (unit, A -> B) and separation. Returns `None`
    /// when the anchors coincide: with no direction the impulse would
    /// be a guess, so the frame is skipped (bounded, self-heals as the
    /// bodies separate â€” same policy as 2D).
    fn axis(&self, bodies: &[RigidBody3]) -> Option<(Vec3, f32)> {
        let delta = self.world_b(bodies) - self.world_a(bodies);
        let dist = delta.length();
        if dist > EPS {
            Some((delta / dist, dist))
        } else {
            // Coincident anchors: satisfied at zero rest length, and a
            // skipped frame otherwise (never guess a direction).
            None
        }
    }

    fn warm_start(&self, bodies: &mut [RigidBody3]) {
        if self.impulse == 0.0 {
            return;
        }
        let Some((n, _)) = self.axis(bodies) else {
            return;
        };
        // P = -n * impulse on B (see solve() for the sign convention).
        apply_pair_impulse_3(
            bodies,
            self.body_a,
            self.body_b,
            -n * self.impulse,
            self.world_a(bodies),
            self.world_b(bodies),
        );
    }

    /// One sequential-impulse iteration for C = dist - rest_length.
    /// Same sign convention as 2D: P_on_B = -n * Î», bilateral.
    /// Î» = m * (vn + bias), bias = Î² * C / dt.
    fn solve(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let Some((n, dist)) = self.axis(bodies) else {
            return;
        };
        let c = dist - self.rest_length;
        // Resting at rest length: nothing to do (avoids jitter from
        // bias on a satisfied constraint).
        if c.abs() < EPS {
            return;
        }

        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let (ra, rb) = (
            pa - bodies[self.body_a].transform.position,
            pb - bodies[self.body_b].transform.position,
        );

        let va = bodies[self.body_a].velocity + bodies[self.body_a].angular_velocity.cross(ra);
        let vb = bodies[self.body_b].velocity + bodies[self.body_b].angular_velocity.cross(rb);
        let vn = (vb - va).dot(n);

        let m = effective_mass_3(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].world_inv_inertia(),
            bodies[self.body_b].world_inv_inertia(),
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
        apply_pair_impulse_3(bodies, self.body_a, self.body_b, -n * lambda, pa, pb);
    }
}

impl HingeJoint3 {
    /// World-space anchor on A.
    fn world_a(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    /// World-space anchor on B.
    fn world_b(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    /// Hinge axis of A in world frame.
    fn axis_a_world(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.rotate_vector(self.axis_a)
    }

    /// Hinge axis of B in world frame.
    fn axis_b_world(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_b].transform.rotate_vector(self.axis_b)
    }

    fn warm_start(&self, bodies: &mut [RigidBody3]) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.impulse != Vec3::ZERO {
            // Stored axes are world X/Y/Z: apply directly.
            apply_pair_impulse_3(bodies, self.body_a, self.body_b, self.impulse, pa, pb);
        }
        if self.align_impulse.x != 0.0 || self.align_impulse.y != 0.0 {
            let (u1, u2) = stable_basis_3(self.axis_a_world(bodies));
            let lam1 = self.align_impulse.x;
            let lam2 = self.align_impulse.y;
            if lam1 != 0.0 {
                apply_angular_3(bodies, self.body_a, self.body_b, u1, lam1);
            }
            if lam2 != 0.0 {
                apply_angular_3(bodies, self.body_a, self.body_b, u2, lam2);
            }
        }
        if self.twist_impulse != 0.0 {
            apply_angular_3(
                bodies,
                self.body_a,
                self.body_b,
                self.axis_a_world(bodies),
                self.twist_impulse,
            );
        }
    }

    /// One iteration: point coincidence (3 axes) + axis alignment (2) +
    /// twist limits if set.
    fn solve(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        self.solve_point(bodies, dt);
        self.solve_align(bodies, dt);
        self.solve_twist(bodies, dt);
    }

    /// Point-to-point along world X, Y and Z. C = (pb - pa) Â· axis.
    /// Î» = -m * (vn + Î² * C / dt), bilateral.
    fn solve_point(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let delta = pb - pa;
        if delta.length_squared() < EPS_SQ {
            return; // coincident: satisfied, avoids jitter bias.
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let c = delta.dot(axis);
            if c.abs() < EPS {
                continue;
            }
            let ra = pa - bodies[self.body_a].transform.position;
            let rb = pb - bodies[self.body_b].transform.position;
            let va = bodies[self.body_a].velocity + bodies[self.body_a].angular_velocity.cross(ra);
            let vb = bodies[self.body_b].velocity + bodies[self.body_b].angular_velocity.cross(rb);
            let vn = (vb - va).dot(axis);
            let m = effective_mass_3(
                bodies[self.body_a].inv_mass,
                bodies[self.body_b].inv_mass,
                bodies[self.body_a].world_inv_inertia(),
                bodies[self.body_b].world_inv_inertia(),
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
            apply_pair_impulse_3(bodies, self.body_a, self.body_b, axis * lambda, pa, pb);
        }
    }

    /// Hinge-axis alignment: drives cross(axA, axB) to zero about the two
    /// stable basis directions perpendicular to A's hinge axis. Bilateral.
    /// Skipped when the axes are (anti-)parallel: the cross product
    /// vanishes, so there is no correction direction to solve along
    /// (bounded skip; twist still acts).
    fn solve_align(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let ax_a = self.axis_a_world(bodies);
        let ax_b = self.axis_b_world(bodies);
        let err = ax_a.cross(ax_b);
        if err.length_squared() < EPS_SQ {
            return;
        }
        let (u1, u2) = stable_basis_3(ax_a);
        let ia = bodies[self.body_a].world_inv_inertia();
        let ib = bodies[self.body_b].world_inv_inertia();
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        for (u, slot) in [
            (u1, &mut self.align_impulse.x),
            (u2, &mut self.align_impulse.y),
        ] {
            let c = err.dot(u);
            if c.abs() < EPS {
                continue;
            }
            let m = angular_mass_3(ia, ib, u);
            if m == 0.0 {
                continue;
            }
            let bias = if dt > 0.0 {
                JOINT_BAUMGARTE * c / dt
            } else {
                0.0
            };
            let lambda = -m * (w_rel.dot(u) + bias);
            *slot += lambda;
            apply_angular_3(bodies, self.body_a, self.body_b, u, lambda);
        }
    }

    /// Twist limits about the hinge axis. C = tw - clamp(tw) (0 inside
    /// range). Î» = -m_ang * (w_rel + Î² * C / dt), bilateral within the
    /// active side. Skipped entirely for free hinges (min > max).
    fn solve_twist(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        if self.limit_min > self.limit_max {
            return;
        }
        let u = self.axis_a_world(bodies);
        let q_rel = relative_quat(
            bodies[self.body_a].transform.rotation,
            bodies[self.body_b].transform.rotation,
        );
        let tw = twist_about(q_rel, u);
        let c = if tw < self.limit_min {
            tw - self.limit_min
        } else if tw > self.limit_max {
            tw - self.limit_max
        } else {
            return; // inside range: free.
        };
        let ia = bodies[self.body_a].world_inv_inertia();
        let ib = bodies[self.body_b].world_inv_inertia();
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        let m = angular_mass_3(ia, ib, u);
        if m == 0.0 {
            return;
        }
        let bias = if dt > 0.0 {
            JOINT_BAUMGARTE * c / dt
        } else {
            0.0
        };
        let lambda = -m * (w_rel.dot(u) + bias);
        self.twist_impulse += lambda;
        apply_angular_3(bodies, self.body_a, self.body_b, u, lambda);
    }
}

impl PrismaticJoint3 {
    fn world_a(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    fn world_b(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    /// Slide axis in world frame.
    fn axis_world(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.rotate_vector(self.axis_a)
    }

    fn warm_start(&self, bodies: &mut [RigidBody3]) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.perp_impulse.x != 0.0 || self.perp_impulse.y != 0.0 {
            let ax = self.axis_world(bodies);
            let (u1, u2) = stable_basis_3(ax);
            if self.perp_impulse.x != 0.0 {
                apply_pair_impulse_3(
                    bodies,
                    self.body_a,
                    self.body_b,
                    u1 * self.perp_impulse.x,
                    pa,
                    pb,
                );
            }
            if self.perp_impulse.y != 0.0 {
                apply_pair_impulse_3(
                    bodies,
                    self.body_a,
                    self.body_b,
                    u2 * self.perp_impulse.y,
                    pa,
                    pb,
                );
            }
        }
        if self.limit_impulse != 0.0 {
            let ax = self.axis_world(bodies);
            apply_pair_impulse_3(
                bodies,
                self.body_a,
                self.body_b,
                ax * self.limit_impulse,
                pa,
                pb,
            );
        }
        if self.angular_impulse != Vec3::ZERO {
            // Stored axes are world X/Y/Z: apply directly.
            apply_angular_vec_3(bodies, self.body_a, self.body_b, self.angular_impulse);
        }
    }

    fn solve(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        // Latch the reference orientation on first solve (deterministic:
        // pure function of the trajectory up to this step).
        if self.ref_orientation.is_none() {
            self.ref_orientation = Some(relative_quat(
                bodies[self.body_a].transform.rotation,
                bodies[self.body_b].transform.rotation,
            ));
        }
        self.solve_perp(bodies, dt);
        self.solve_angular(bodies, dt);
        self.solve_travel(bodies, dt);
    }

    /// Perpendicular lock on the plane normal to the slide axis:
    /// C = (pb - pa) Â· u_i for the stable basis. Bilateral.
    fn solve_perp(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let delta = pb - pa;
        if delta.length_squared() < EPS_SQ {
            return;
        }
        let (u1, u2) = stable_basis_3(self.axis_world(bodies));
        let ia = bodies[self.body_a].world_inv_inertia();
        let ib = bodies[self.body_b].world_inv_inertia();
        for (u, slot) in [
            (u1, &mut self.perp_impulse.x),
            (u2, &mut self.perp_impulse.y),
        ] {
            let c = delta.dot(u);
            if c.abs() < EPS {
                continue;
            }
            let ra = pa - bodies[self.body_a].transform.position;
            let rb = pb - bodies[self.body_b].transform.position;
            let va = bodies[self.body_a].velocity + bodies[self.body_a].angular_velocity.cross(ra);
            let vb = bodies[self.body_b].velocity + bodies[self.body_b].angular_velocity.cross(rb);
            let vn = (vb - va).dot(u);
            let m = effective_mass_3(
                bodies[self.body_a].inv_mass,
                bodies[self.body_b].inv_mass,
                ia,
                ib,
                ra,
                rb,
                u,
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
            *slot += lambda;
            apply_pair_impulse_3(bodies, self.body_a, self.body_b, u * lambda, pa, pb);
        }
    }

    /// Orientation lock to the latched reference, about world X/Y/Z.
    /// C = err Â· axis with err the quaternion error vector of
    /// (q_rel * ref^-1). Always enforced once latched.
    fn solve_angular(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let Some(ref_q) = self.ref_orientation else {
            return;
        };
        let q_rel = relative_quat(
            bodies[self.body_a].transform.rotation,
            bodies[self.body_b].transform.rotation,
        );
        let err = orientation_error(q_rel * ref_q.inverse());
        if err.length_squared() < EPS_SQ {
            return;
        }
        let ia = bodies[self.body_a].world_inv_inertia();
        let ib = bodies[self.body_b].world_inv_inertia();
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        for (axis, slot) in [
            (Vec3::X, &mut self.angular_impulse.x),
            (Vec3::Y, &mut self.angular_impulse.y),
            (Vec3::Z, &mut self.angular_impulse.z),
        ] {
            let c = err.dot(axis);
            if c.abs() < EPS {
                continue;
            }
            let m = angular_mass_3(ia, ib, axis);
            if m == 0.0 {
                continue;
            }
            let bias = if dt > 0.0 {
                JOINT_BAUMGARTE * c / dt
            } else {
                0.0
            };
            let lambda = -m * (w_rel.dot(axis) + bias);
            *slot += lambda;
            apply_angular_3(bodies, self.body_a, self.body_b, axis, lambda);
        }
    }

    /// Travel limits along the slide axis. s = (pb - pa) Â· axis;
    /// C = s - clamp(s). Bilateral on the active side; free travel
    /// (min > max) skips entirely.
    fn solve_travel(&mut self, bodies: &mut [RigidBody3], dt: f32) {
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
        let va = bodies[self.body_a].velocity + bodies[self.body_a].angular_velocity.cross(ra);
        let vb = bodies[self.body_b].velocity + bodies[self.body_b].angular_velocity.cross(rb);
        let vn = (vb - va).dot(ax);
        let m = effective_mass_3(
            bodies[self.body_a].inv_mass,
            bodies[self.body_b].inv_mass,
            bodies[self.body_a].world_inv_inertia(),
            bodies[self.body_b].world_inv_inertia(),
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
        apply_pair_impulse_3(bodies, self.body_a, self.body_b, ax * lambda, pa, pb);
    }
}

impl FixedJoint3 {
    fn world_a(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_a].transform.transform_point(self.anchor_a)
    }

    fn world_b(&self, bodies: &[RigidBody3]) -> Vec3 {
        bodies[self.body_b].transform.transform_point(self.anchor_b)
    }

    fn warm_start(&self, bodies: &mut [RigidBody3]) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        if self.impulse != Vec3::ZERO {
            apply_pair_impulse_3(bodies, self.body_a, self.body_b, self.impulse, pa, pb);
        }
        if self.angular_impulse != Vec3::ZERO {
            apply_angular_vec_3(bodies, self.body_a, self.body_b, self.angular_impulse);
        }
    }

    fn solve(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        if self.ref_orientation.is_none() {
            self.ref_orientation = Some(relative_quat(
                bodies[self.body_a].transform.rotation,
                bodies[self.body_b].transform.rotation,
            ));
        }
        self.solve_point(bodies, dt);
        self.solve_angular(bodies, dt);
    }

    /// Point coincidence along world X, Y and Z (same as hinge point).
    fn solve_point(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let pa = self.world_a(bodies);
        let pb = self.world_b(bodies);
        let delta = pb - pa;
        if delta.length_squared() < EPS_SQ {
            // Still need the orientation lock below; point is satisfied.
        } else {
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                let c = delta.dot(axis);
                if c.abs() < EPS {
                    continue;
                }
                let ra = pa - bodies[self.body_a].transform.position;
                let rb = pb - bodies[self.body_b].transform.position;
                let va =
                    bodies[self.body_a].velocity + bodies[self.body_a].angular_velocity.cross(ra);
                let vb =
                    bodies[self.body_b].velocity + bodies[self.body_b].angular_velocity.cross(rb);
                let vn = (vb - va).dot(axis);
                let m = effective_mass_3(
                    bodies[self.body_a].inv_mass,
                    bodies[self.body_b].inv_mass,
                    bodies[self.body_a].world_inv_inertia(),
                    bodies[self.body_b].world_inv_inertia(),
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
                apply_pair_impulse_3(bodies, self.body_a, self.body_b, axis * lambda, pa, pb);
            }
        }
    }

    /// Orientation lock to the latched reference (always enforced).
    fn solve_angular(&mut self, bodies: &mut [RigidBody3], dt: f32) {
        let Some(ref_q) = self.ref_orientation else {
            return;
        };
        let q_rel = relative_quat(
            bodies[self.body_a].transform.rotation,
            bodies[self.body_b].transform.rotation,
        );
        let err = orientation_error(q_rel * ref_q.inverse());
        if err.length_squared() < EPS_SQ {
            return;
        }
        let ia = bodies[self.body_a].world_inv_inertia();
        let ib = bodies[self.body_b].world_inv_inertia();
        let w_rel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
        for (axis, slot) in [
            (Vec3::X, &mut self.angular_impulse.x),
            (Vec3::Y, &mut self.angular_impulse.y),
            (Vec3::Z, &mut self.angular_impulse.z),
        ] {
            let c = err.dot(axis);
            if c.abs() < EPS {
                continue;
            }
            let m = angular_mass_3(ia, ib, axis);
            if m == 0.0 {
                continue;
            }
            let bias = if dt > 0.0 {
                JOINT_BAUMGARTE * c / dt
            } else {
                0.0
            };
            let lambda = -m * (w_rel.dot(axis) + bias);
            *slot += lambda;
            apply_angular_3(bodies, self.body_a, self.body_b, axis, lambda);
        }
    }
}

/// Relative orientation of B in A's frame, normalized.
#[inline]
fn relative_quat(qa: Quat, qb: Quat) -> Quat {
    (qb * qa.inverse()).normalize()
}

/// Shortest-path form (w >= 0); the error vector below is sign-sensitive.
#[inline]
fn shortest_path(mut q: Quat) -> Quat {
    if q.w < 0.0 {
        q = -q;
    }
    q
}

/// Orientation error vector: `2*xyz` of the shortest-path relative
/// rotation. Zero iff aligned; magnitude ~= angle for small angles.
#[inline]
fn orientation_error(q_rel: Quat) -> Vec3 {
    2.0 * shortest_path(q_rel).xyz()
}

/// Signed twist of `q_rel` about a unit `axis`, wrapped to [-Ï€, Ï€].
#[inline]
fn twist_about(q_rel: Quat, axis: Vec3) -> f32 {
    let q = shortest_path(q_rel);
    let tw = 2.0 * q.xyz().dot(axis).atan2(q.w);
    wrap_angle(tw)
}

/// Wrap an angle to [-Ï€, Ï€] (mirror of the 2D helper).
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

/// Deterministic orthonormal basis perpendicular to a unit vector.
/// For unit input one of the fallback crosses always succeeds
/// (min over unit vectors of the best cross length is ~0.7).
#[inline]
fn stable_basis_3(n: Vec3) -> (Vec3, Vec3) {
    debug_assert!(
        (n.length_squared() - 1.0).abs() < 1e-3,
        "stable basis needs a unit vector"
    );
    let u = if n.cross(Vec3::X).length_squared() > 1e-12 {
        n.cross(Vec3::X).normalize()
    } else {
        n.cross(Vec3::Y).normalize()
    };
    (u, n.cross(u))
}

/// Validate joint endpoints shared by all 3D joint types.
#[inline]
fn validate_endpoints_3(body_a: usize, body_b: usize, body_count: usize) -> Result<(), JointError> {
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
fn validate_anchor_3(anchor: Vec3) -> Result<(), JointError> {
    if !anchor.x.is_finite() || !anchor.y.is_finite() || !anchor.z.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Validate the compliance field (v1: must be exactly 0).
#[inline]
fn validate_compliance_3(compliance: f32) -> Result<(), JointError> {
    if !compliance.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    if compliance != 0.0 {
        return Err(JointError::UnsupportedCompliance);
    }
    Ok(())
}

/// Validate hinge twist limits. Both bounds must be finite; `min > max`
/// is the explicit free-hinge convention.
#[inline]
fn validate_twist_limits(limit_min: f32, limit_max: f32) -> Result<(), JointError> {
    if !limit_min.is_finite() || !limit_max.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Validate a hinge/slide axis: finite and non-zero. Returns the
/// normalized axis (canonicalization, not guessing: direction is
/// preserved, only the magnitude is standardized).
#[inline]
fn validate_axis_3(axis: Vec3) -> Result<Vec3, JointError> {
    if !axis.x.is_finite() || !axis.y.is_finite() || !axis.z.is_finite() {
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
fn validate_travel_limits_3(limit_min: f32, limit_max: f32) -> Result<(), JointError> {
    if !limit_min.is_finite() || !limit_max.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    Ok(())
}

/// Build a validated 3D distance joint (id assigned by the caller/World).
#[allow(clippy::too_many_arguments)]
pub fn make_distance_joint_3(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec3,
    anchor_b: Vec3,
    rest_length: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint3, JointError> {
    validate_endpoints_3(body_a, body_b, body_count)?;
    validate_anchor_3(anchor_a)?;
    validate_anchor_3(anchor_b)?;
    if !rest_length.is_finite() {
        return Err(JointError::NonFiniteInput);
    }
    if rest_length < 0.0 {
        return Err(JointError::InvalidParams);
    }
    validate_compliance_3(compliance)?;
    Ok(Joint3::Distance(DistanceJoint3 {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        rest_length,
        impulse: 0.0,
    }))
}

/// Build a validated 3D hinge (revolute) joint (id assigned by
/// caller/World). Both hinge axes are given in their body's local frame
/// and normalized at creation. `limit_min > limit_max` selects a FREE
/// hinge; otherwise twist about the hinge axis is clamped into
/// `[limit_min, limit_max]` radians.
#[allow(clippy::too_many_arguments)]
pub fn make_hinge_joint_3(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec3,
    anchor_b: Vec3,
    axis_a: Vec3,
    axis_b: Vec3,
    limit_min: f32,
    limit_max: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint3, JointError> {
    validate_endpoints_3(body_a, body_b, body_count)?;
    validate_anchor_3(anchor_a)?;
    validate_anchor_3(anchor_b)?;
    let ax_a = validate_axis_3(axis_a)?;
    let ax_b = validate_axis_3(axis_b)?;
    validate_twist_limits(limit_min, limit_max)?;
    validate_compliance_3(compliance)?;
    Ok(Joint3::Hinge(HingeJoint3 {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        axis_a: ax_a,
        axis_b: ax_b,
        limit_min,
        limit_max,
        impulse: Vec3::ZERO,
        align_impulse: Vec2Pair::default(),
        twist_impulse: 0.0,
    }))
}

/// Build a validated 3D prismatic (slider) joint (id assigned by
/// caller/World). The slide axis is given in A's local frame and
/// normalized at creation.
#[allow(clippy::too_many_arguments)]
pub fn make_prismatic_joint_3(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec3,
    anchor_b: Vec3,
    axis_a: Vec3,
    limit_min: f32,
    limit_max: f32,
    compliance: f32,
    body_count: usize,
) -> Result<Joint3, JointError> {
    validate_endpoints_3(body_a, body_b, body_count)?;
    validate_anchor_3(anchor_a)?;
    validate_anchor_3(anchor_b)?;
    let axis = validate_axis_3(axis_a)?;
    validate_travel_limits_3(limit_min, limit_max)?;
    validate_compliance_3(compliance)?;
    Ok(Joint3::Prismatic(PrismaticJoint3 {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        axis_a: axis,
        limit_min,
        limit_max,
        ref_orientation: None,
        perp_impulse: Vec2Pair::default(),
        angular_impulse: Vec3::ZERO,
        limit_impulse: 0.0,
    }))
}

/// Build a validated 3D weld (fixed) joint (id assigned by caller/World).
/// The relative orientation is latched at first solve.
pub fn make_fixed_joint_3(
    id: usize,
    body_a: usize,
    body_b: usize,
    anchor_a: Vec3,
    anchor_b: Vec3,
    compliance: f32,
    body_count: usize,
) -> Result<Joint3, JointError> {
    validate_endpoints_3(body_a, body_b, body_count)?;
    validate_anchor_3(anchor_a)?;
    validate_anchor_3(anchor_b)?;
    validate_compliance_3(compliance)?;
    Ok(Joint3::Fixed(FixedJoint3 {
        id,
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        ref_orientation: None,
        impulse: Vec3::ZERO,
        angular_impulse: Vec3::ZERO,
    }))
}

/// Effective mass along a direction (same formula as the 3D contact
/// solver: world-space tensors, cross-term quadratic form).
#[inline]
fn effective_mass_3(
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_inertia_a: Mat3,
    inv_inertia_b: Mat3,
    r_a: Vec3,
    r_b: Vec3,
    dir: Vec3,
) -> f32 {
    let r_a_cross = r_a.cross(dir);
    let r_b_cross = r_b.cross(dir);
    let k = inv_mass_a
        + inv_mass_b
        + (inv_inertia_a * r_a_cross).dot(r_a_cross)
        + (inv_inertia_b * r_b_cross).dot(r_b_cross);
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Effective angular mass about a unit axis: torque about `axis` changes
/// the relative angular velocity along `axis` by `k` per unit impulse.
#[inline]
fn angular_mass_3(inv_inertia_a: Mat3, inv_inertia_b: Mat3, axis: Vec3) -> f32 {
    let k = axis.dot(inv_inertia_a * axis) + axis.dot(inv_inertia_b * axis);
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Apply impulse P to B and -P to A through one paired borrow
/// (same single-check pattern as the contact solvers). World tensors
/// are rebuilt from live orientations (orientations never change
/// mid-solve, so they are never stale).
#[inline]
fn apply_pair_impulse_3(
    bodies: &mut [RigidBody3],
    body_a: usize,
    body_b: usize,
    p: Vec3,
    pa: Vec3,
    pb: Vec3,
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

    a.angular_velocity -= a.world_inv_inertia() * ra.cross(p);
    b.angular_velocity += b.world_inv_inertia() * rb.cross(p);
}

/// Apply a pure angular impulse vector J (world frame): A -= J, B += J.
/// Used for world-frame orientation-lock warm starts.
#[inline]
fn apply_angular_vec_3(bodies: &mut [RigidBody3], body_a: usize, body_b: usize, j: Vec3) {
    debug_assert!(body_a != body_b, "joint must connect two distinct bodies");
    let ia = bodies[body_a].world_inv_inertia();
    let ib = bodies[body_b].world_inv_inertia();
    let (a, b) = if body_a < body_b {
        let (head, tail) = bodies.split_at_mut(body_b);
        (&mut head[body_a], &mut tail[0])
    } else {
        let (head, tail) = bodies.split_at_mut(body_a);
        (&mut tail[0], &mut head[body_b])
    };
    a.angular_velocity -= ia * j;
    b.angular_velocity += ib * j;
}

/// Apply a scalar angular impulse about a unit axis: A -= Î»Â·axis,
/// B += Î»Â·axis (torque-space, through the world tensors).
#[inline]
fn apply_angular_3(
    bodies: &mut [RigidBody3],
    body_a: usize,
    body_b: usize,
    axis: Vec3,
    lambda: f32,
) {
    debug_assert!(body_a != body_b, "joint must connect two distinct bodies");
    let ia = bodies[body_a].world_inv_inertia();
    let ib = bodies[body_b].world_inv_inertia();
    let (a, b) = if body_a < body_b {
        let (head, tail) = bodies.split_at_mut(body_b);
        (&mut head[body_a], &mut tail[0])
    } else {
        let (head, tail) = bodies.split_at_mut(body_a);
        (&mut tail[0], &mut head[body_b])
    };
    a.angular_velocity -= ia * (axis * lambda);
    b.angular_velocity += ib * (axis * lambda);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape3;
    use crate::world::World3;
    use glam::Vec3;

    const DT: f32 = 1.0 / 60.0;

    fn two_statics() -> (World3, usize) {
        let mut world = World3::new();
        world
            .add_static_body(Vec3::ZERO, Shape3::Sphere { radius: 0.5 })
            .expect("valid shape");
        world
            .add_static_body(Vec3::X, Shape3::Sphere { radius: 0.5 })
            .expect("valid shape");
        let n = world.bodies.len();
        (world, n)
    }

    fn world_with_ball_on_rod() -> (World3, usize) {
        let mut world = World3::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        world
            .add_static_body(
                Vec3::ZERO,
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
            )
            .expect("valid shape");
        let ball = world
            .add_dynamic_body(
                Vec3::new(0.0, -2.0, 0.0),
                Shape3::Sphere { radius: 0.3 },
                1.0,
            )
            .expect("valid shape");
        (world, ball)
    }

    fn world_with_hinge() -> (World3, usize) {
        let mut world = World3::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        world
            .add_static_body(
                Vec3::ZERO,
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
            )
            .expect("valid shape");
        let bob = world
            .add_dynamic_body(
                Vec3::new(2.0, 0.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        (world, bob)
    }

    /// Signed twist of B relative to A about a world axis.
    fn twist_of(world: &World3, a: usize, b: usize, axis: Vec3) -> f32 {
        let qa = world.bodies[a].transform.rotation;
        let qb = world.bodies[b].transform.rotation;
        twist_about(relative_quat(qa, qb), axis)
    }

    /// Relative orientation error magnitude (0 iff co-oriented).
    fn orientation_err(world: &World3, a: usize, b: usize) -> f32 {
        let qa = world.bodies[a].transform.rotation;
        let qb = world.bodies[b].transform.rotation;
        orientation_error(relative_quat(qa, qb)).length()
    }

    #[test]
    fn k31_distance3_rejects_bad_input() {
        let (_world, n) = two_statics();
        let a = Vec3::ZERO;
        let b = Vec3::X;
        // Unknown body.
        assert_eq!(
            make_distance_joint_3(0, 0, 99, a, b, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        // Same body (endpoint check runs before anchor checks).
        assert_eq!(
            make_distance_joint_3(0, 0, 0, a, b, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        // NaN anchor (distinct valid bodies so the anchor check runs).
        assert_eq!(
            make_distance_joint_3(0, 0, 1, Vec3::new(f32::NAN, 0.0, 0.0), b, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // Negative rest length.
        assert_eq!(
            make_distance_joint_3(0, 0, 1, a, b, -1.0, 0.0, n),
            Err(JointError::InvalidParams)
        );
        // Non-finite rest length.
        assert_eq!(
            make_distance_joint_3(0, 0, 1, a, b, f32::INFINITY, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        // Non-zero compliance rejected (rigid-only v1).
        assert_eq!(
            make_distance_joint_3(0, 0, 1, a, b, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
    }

    #[test]
    fn k32_distance3_holds_rest_length() {
        let (mut world, ball) = world_with_ball_on_rod();
        // Rod from anchor origin to ball center, rest length 2.
        let joint = make_distance_joint_3(
            0,
            0,
            ball,
            Vec3::ZERO,
            Vec3::ZERO,
            2.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        for _ in 0..300 {
            world.step(DT);
        }
        // Ball hangs straight down at rest length (rod holds, no stretch).
        let p = world.bodies[ball].transform.position;
        let dist = p.length();
        assert!(
            (dist - 2.0).abs() < 0.05,
            "rod must hold rest length, dist={dist}, pos={p}"
        );
        assert!(
            p.x.abs() < 0.1 && p.z.abs() < 0.1,
            "pendulum at rest hangs straight down, pos={p}"
        );
    }

    #[test]
    fn k33_distance3_pulls_back() {
        let (mut world, ball) = world_with_ball_on_rod();
        let joint = make_distance_joint_3(
            0,
            0,
            ball,
            Vec3::ZERO,
            Vec3::ZERO,
            2.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Displace sideways: pendulum must swing back toward center.
        world.bodies[ball].transform.position = Vec3::new(1.0, -1.5, 0.5);
        world.bodies[ball].velocity = Vec3::ZERO;
        for _ in 0..120 {
            world.step(DT);
        }
        let p = world.bodies[ball].transform.position;
        assert!(
            p.x.abs() < 1.0,
            "displaced pendulum must swing back, pos={p}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.velocity.x.is_finite());
        }
    }

    #[test]
    fn k34_joint3_removed_with_body() {
        let (mut world, ball) = world_with_ball_on_rod();
        let joint = make_distance_joint_3(
            0,
            0,
            ball,
            Vec3::ZERO,
            Vec3::ZERO,
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
            world.step(DT);
        }
    }

    #[test]
    fn k35_hinge3_rejects_bad_input() {
        let (_world, n) = two_statics();
        let a = Vec3::ZERO;
        let b = Vec3::X;
        assert_eq!(
            make_hinge_joint_3(0, 0, 99, a, b, Vec3::Z, Vec3::Z, 0.0, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_hinge_joint_3(0, 1, 1, a, b, Vec3::Z, Vec3::Z, 0.0, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_hinge_joint_3(
                0,
                0,
                1,
                Vec3::new(f32::NAN, 0.0, 0.0),
                b,
                Vec3::Z,
                Vec3::Z,
                0.0,
                1.0,
                0.0,
                n
            ),
            Err(JointError::NonFiniteInput)
        );
        // Zero hinge axis rejected.
        assert_eq!(
            make_hinge_joint_3(0, 0, 1, a, b, Vec3::ZERO, Vec3::Z, 0.0, 1.0, 0.0, n),
            Err(JointError::InvalidParams)
        );
        // NaN limits rejected (free hinge uses inverted finite bounds).
        assert_eq!(
            make_hinge_joint_3(0, 0, 1, a, b, Vec3::Z, Vec3::Z, f32::NAN, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_hinge_joint_3(0, 0, 1, a, b, Vec3::Z, Vec3::Z, 0.0, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
        // Inverted bounds are the valid free-hinge convention.
        assert!(make_hinge_joint_3(0, 0, 1, a, b, Vec3::Z, Vec3::Z, 1.0, -1.0, 0.0, n).is_ok());
    }

    #[test]
    fn k36_hinge3_pulls_to_anchor() {
        let (mut world, bob) = world_with_hinge();
        // Free hinge (inverted bounds): pin bob's center to the origin.
        let joint = make_hinge_joint_3(
            0,
            0,
            bob,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        for _ in 0..300 {
            world.step(DT);
        }
        let p = world.bodies[bob].transform.position;
        assert!(
            p.length() < 0.15,
            "hinge must pull the bob to the anchor, pos={p}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.velocity.x.is_finite());
        }
    }

    #[test]
    fn k37_hinge3_free_spin() {
        let mut world = World3::new();
        world.gravity = Vec3::ZERO;
        // Static anchor at (0,-2,0); its local anchor (0,2,0) puts the
        // world hinge point at the origin â€” the bob's center â€” while the
        // bodies stay 2 units apart (no contact friction to confuse the
        // spin check). Twist about the hinge axis must persist undamped.
        world
            .add_static_body(
                Vec3::new(0.0, -2.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
            )
            .expect("valid shape");
        let bob = world
            .add_dynamic_body(
                Vec3::ZERO,
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        world.bodies[bob].linear_damping = 0.0;
        world.bodies[bob].angular_damping = 0.0;
        let joint = make_hinge_joint_3(
            0,
            0,
            bob,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        world.bodies[bob].angular_velocity = Vec3::new(0.0, 0.0, 2.0);
        for _ in 0..60 {
            world.step(DT);
        }
        let tw = twist_of(&world, 0, bob, Vec3::Z);
        assert!(
            (tw - 2.0).abs() < 0.3,
            "free hinge must not brake twist, twist={tw}"
        );
        assert!(
            world.bodies[bob].transform.position.length() < 0.05,
            "center-pinned bob must not drift, pos={}",
            world.bodies[bob].transform.position
        );
    }

    #[test]
    fn k38_hinge3_twist_limits_enforced() {
        let (mut world, bob) = world_with_hinge();
        // Tight twist limits around zero: the bob may hang but not windmill.
        let joint = make_hinge_joint_3(
            0,
            0,
            bob,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            -0.2,
            0.2,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Wind it up and spin it hard about the hinge axis: limits catch it.
        world.bodies[bob].angular_velocity = Vec3::new(0.0, 0.0, 10.0);
        for _ in 0..300 {
            world.step(DT);
        }
        let tw = twist_of(&world, 0, bob, Vec3::Z);
        assert!(
            (-0.4..=0.4).contains(&tw),
            "hinge twist limits must catch the windup, twist={tw}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.angular_velocity.x.is_finite());
        }
    }

    #[test]
    fn k39_prismatic3_rejects_bad_input() {
        let (_world, n) = two_statics();
        let a = Vec3::ZERO;
        let b = Vec3::X;
        assert_eq!(
            make_prismatic_joint_3(0, 0, 99, a, b, Vec3::X, -1.0, 1.0, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_prismatic_joint_3(0, 1, 1, a, b, Vec3::X, -1.0, 1.0, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_prismatic_joint_3(
                0,
                0,
                1,
                Vec3::new(f32::NAN, 0.0, 0.0),
                b,
                Vec3::X,
                -1.0,
                1.0,
                0.0,
                n
            ),
            Err(JointError::NonFiniteInput)
        );
        // Zero axis rejected.
        assert_eq!(
            make_prismatic_joint_3(0, 0, 1, a, b, Vec3::ZERO, -1.0, 1.0, 0.0, n),
            Err(JointError::InvalidParams)
        );
        // NaN axis rejected.
        assert_eq!(
            make_prismatic_joint_3(
                0,
                0,
                1,
                a,
                b,
                Vec3::new(1.0, f32::NAN, 0.0),
                -1.0,
                1.0,
                0.0,
                n
            ),
            Err(JointError::NonFiniteInput)
        );
        // NaN travel limits rejected.
        assert_eq!(
            make_prismatic_joint_3(0, 0, 1, a, b, Vec3::X, f32::NAN, 1.0, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_prismatic_joint_3(0, 0, 1, a, b, Vec3::X, -1.0, 1.0, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
        // Inverted bounds are the valid free-travel convention.
        assert!(make_prismatic_joint_3(0, 0, 1, a, b, Vec3::X, 1.0, -1.0, 0.0, n).is_ok());
    }

    #[test]
    fn k40_prismatic3_slides_holds_rail() {
        let mut world = World3::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        // Static rail along X at y = 1.
        world
            .add_static_body(
                Vec3::new(0.0, 1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
            )
            .expect("valid shape");
        let slider = world
            .add_dynamic_body(
                Vec3::new(0.0, 1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Slider axis X in the rail's frame; coincident anchors; free travel.
        let joint = make_prismatic_joint_3(
            0,
            0,
            slider,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::X,
            1.0,
            -1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Shove it along the rail: it must glide in X while Y/Z hold.
        world.bodies[slider].velocity = Vec3::new(5.0, 0.0, 0.0);
        for _ in 0..120 {
            world.step(DT);
        }
        let p = world.bodies[slider].transform.position;
        assert!(p.x > 1.0, "slider must glide along the rail, pos={p}");
        assert!(
            (p.y - 1.0).abs() < 0.15,
            "rail must hold the slider's height against gravity, pos={p}"
        );
        assert!(p.z.abs() < 0.15, "rail must hold lateral Z, pos={p}");
        assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    }

    #[test]
    fn k41_prismatic3_limits_enforced() {
        let mut world = World3::new();
        world.gravity = Vec3::ZERO;
        world
            .add_static_body(
                Vec3::ZERO,
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
            )
            .expect("valid shape");
        let slider = world
            .add_dynamic_body(
                Vec3::new(3.0, 0.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Travel clamped to [-1, 1] along X: s starts at 3 (beyond the
        // max) and must be pulled back.
        let joint = make_prismatic_joint_3(
            0,
            0,
            slider,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::X,
            -1.0,
            1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        for _ in 0..300 {
            world.step(DT);
        }
        let p = world.bodies[slider].transform.position;
        assert!(p.x < 1.2, "travel limit must pull the slider back, pos={p}");
        assert!(
            p.y.abs() < 0.15 && p.z.abs() < 0.15,
            "rail must hold the perpendicular plane, pos={p}"
        );
    }

    #[test]
    fn k42_fixed3_rejects_bad_input() {
        let (_world, n) = two_statics();
        assert_eq!(
            make_fixed_joint_3(0, 0, 99, Vec3::ZERO, Vec3::X, 0.0, n),
            Err(JointError::UnknownBody)
        );
        assert_eq!(
            make_fixed_joint_3(0, 1, 1, Vec3::ZERO, Vec3::X, 0.0, n),
            Err(JointError::SameBody)
        );
        assert_eq!(
            make_fixed_joint_3(0, 0, 1, Vec3::new(0.0, f32::NAN, 0.0), Vec3::X, 0.0, n),
            Err(JointError::NonFiniteInput)
        );
        assert_eq!(
            make_fixed_joint_3(0, 0, 1, Vec3::ZERO, Vec3::X, 0.5, n),
            Err(JointError::UnsupportedCompliance)
        );
    }

    #[test]
    fn k43_fixed3_holds_formation() {
        let mut world = World3::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        // Two boxes glued 1 unit apart, dropped: the formation must
        // survive the fall without contacts to lean on.
        let a = world
            .add_dynamic_body(
                Vec3::new(-0.5, 5.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        let b = world
            .add_dynamic_body(
                Vec3::new(0.5, 5.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        // Weld B's center to a point 1 unit right of A's center.
        let joint = make_fixed_joint_3(
            0,
            a,
            b,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        for _ in 0..180 {
            world.step(DT);
        }
        let pa = world.bodies[a].transform.position;
        let pb = world.bodies[b].transform.position;
        let gap = (pb - pa).length();
        assert!(
            (gap - 1.0).abs() < 0.15,
            "weld must hold the 1-unit formation, gap={gap}, a={pa}, b={pb}"
        );
        for body in &world.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.angular_velocity.x.is_finite());
        }
    }

    #[test]
    fn k44_fixed3_resists_rotation() {
        let mut world = World3::new();
        world.gravity = Vec3::ZERO;
        let a = world
            .add_dynamic_body(
                Vec3::new(-0.5, 0.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        let b = world
            .add_dynamic_body(
                Vec3::new(0.5, 0.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.3, 0.3, 0.3),
                },
                1.0,
            )
            .expect("valid shape");
        world.bodies[a].linear_damping = 0.0;
        world.bodies[a].angular_damping = 0.0;
        world.bodies[b].linear_damping = 0.0;
        world.bodies[b].angular_damping = 0.0;
        let joint = make_fixed_joint_3(
            0,
            a,
            b,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        // Spin A hard: the weld must drag B along (relative error ~ 0).
        world.bodies[a].angular_velocity = Vec3::new(0.0, 0.0, 5.0);
        for _ in 0..120 {
            world.step(DT);
        }
        let err = orientation_err(&world, a, b);
        assert!(
            err < 0.4,
            "weld must keep the bodies co-rotating, err={err}"
        );
    }
}
