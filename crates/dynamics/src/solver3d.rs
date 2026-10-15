use crate::body::RigidBody3;
use glam::{Mat3, Vec3};

#[derive(Debug, Clone)]
pub struct ContactConstraint3D {
    pub body_a: usize,
    pub body_b: usize,
    pub point: Vec3,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub penetration: f32,
    pub friction: f32,
    pub restitution: f32,
    pub normal_impulse: f32,
    pub tangent1_impulse: f32,
    pub tangent2_impulse: f32,
    pub rest_bias: f32,
    pub mass_normal: f32,
    pub mass_tangent1: f32,
    pub mass_tangent2: f32,
    /// World-frame inverse inertias, snapshotted once per step during the
    /// solve setup so the per-iteration impulse
    /// applications don't rebuild `R * I^-1 * R^T` every time.
    pub inv_inertia_world_a: Mat3,
    pub inv_inertia_world_b: Mat3,
}

impl ContactConstraint3D {
    pub fn new(
        body_a: usize,
        body_b: usize,
        point: Vec3,
        normal: Vec3,
        penetration: f32,
        friction: f32,
        restitution: f32,
    ) -> Self {
        let tangent1 = if normal.x.abs() < 0.9 {
            Vec3::X.cross(normal).normalize()
        } else {
            Vec3::Y.cross(normal).normalize()
        };
        let tangent2 = normal.cross(tangent1);

        Self {
            body_a,
            body_b,
            point,
            normal,
            tangent1,
            tangent2,
            penetration,
            friction,
            restitution,
            normal_impulse: 0.0,
            tangent1_impulse: 0.0,
            tangent2_impulse: 0.0,
            rest_bias: 0.0,
            mass_normal: 0.0,
            mass_tangent1: 0.0,
            mass_tangent2: 0.0,
            inv_inertia_world_a: Mat3::IDENTITY,
            inv_inertia_world_b: Mat3::IDENTITY,
        }
    }
}

#[derive(Debug)]
pub struct Solver3D {
    /// Maximum solver iterations per step.
    pub iterations: usize,
    pub slop: f32,
    pub baumgarte: f32,
    /// Early-exit tolerance: an iteration applying less than this
    /// (max absolute impulse delta) ends the solve. See [`crate::Solver`].
    pub early_exit_tolerance: f32,
}

impl Solver3D {
    /// Maximum Baumgarte bias velocity (m/s). Same policy as
    /// [`crate::Solver::MAX_BIAS_VELOCITY`]: deep overlaps resolve
    /// gradually instead of ejecting past discrete collision.
    pub const MAX_BIAS_VELOCITY: f32 = 2.0;

    pub fn new() -> Self {
        Self {
            iterations: 8,
            slop: 0.005,
            baumgarte: 0.2,
            early_exit_tolerance: 1e-6,
        }
    }

    /// Solve all constraints. Returns the iterations actually run
    /// (early exit when converged). See [`crate::Solver::solve`].
    pub fn solve(
        &self,
        bodies: &mut [RigidBody3],
        constraints: &mut [ContactConstraint3D],
        dt: f32,
    ) -> usize {
        for constraint in constraints.iter_mut() {
            self.compute_mass_terms(bodies, constraint);
            self.apply_warm_starting(bodies, constraint);
            constraint.rest_bias = self.restitution_bias(bodies, constraint);
        }

        let mut ran = 0;
        for _ in 0..self.iterations {
            ran += 1;
            let mut max_applied: f32 = 0.0;
            for constraint in constraints.iter_mut() {
                max_applied = max_applied.max(self.solve_normal(bodies, constraint, dt));
            }
            for constraint in constraints.iter_mut() {
                max_applied = max_applied.max(self.solve_friction(bodies, constraint));
            }
            if max_applied < self.early_exit_tolerance {
                break;
            }
        }
        ran
    }

    fn compute_mass_terms(&self, bodies: &[RigidBody3], constraint: &mut ContactConstraint3D) {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        // Snapshot world tensors once; the per-iteration impulse
        // applications below reuse them instead of rebuilding.
        constraint.inv_inertia_world_a = body_a.world_inv_inertia();
        constraint.inv_inertia_world_b = body_b.world_inv_inertia();

        constraint.mass_normal = Self::effective_mass(
            body_a.inv_mass,
            body_b.inv_mass,
            constraint.inv_inertia_world_a,
            constraint.inv_inertia_world_b,
            r_a,
            r_b,
            constraint.normal,
        );
        constraint.mass_tangent1 = Self::effective_mass(
            body_a.inv_mass,
            body_b.inv_mass,
            constraint.inv_inertia_world_a,
            constraint.inv_inertia_world_b,
            r_a,
            r_b,
            constraint.tangent1,
        );
        constraint.mass_tangent2 = Self::effective_mass(
            body_a.inv_mass,
            body_b.inv_mass,
            constraint.inv_inertia_world_a,
            constraint.inv_inertia_world_b,
            r_a,
            r_b,
            constraint.tangent2,
        );
    }

    fn effective_mass(
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

        let term_a = inv_inertia_a * r_a_cross;
        let term_b = inv_inertia_b * r_b_cross;

        let k = inv_mass_a + inv_mass_b + term_a.dot(r_a_cross) + term_b.dot(r_b_cross);

        if k > 0.0 {
            1.0 / k
        } else {
            0.0
        }
    }

    fn apply_warm_starting(&self, bodies: &mut [RigidBody3], constraint: &mut ContactConstraint3D) {
        if constraint.normal_impulse == 0.0
            && constraint.tangent1_impulse == 0.0
            && constraint.tangent2_impulse == 0.0
        {
            return;
        }

        let p = constraint.normal * constraint.normal_impulse
            + constraint.tangent1 * constraint.tangent1_impulse
            + constraint.tangent2 * constraint.tangent2_impulse;

        let r_a = constraint.point - bodies[constraint.body_a].transform.position;
        let r_b = constraint.point - bodies[constraint.body_b].transform.position;

        Self::apply_impulse(bodies, constraint.body_a, constraint.body_b, r_a, r_b, p);
    }

    fn solve_normal(
        &self,
        bodies: &mut [RigidBody3],
        constraint: &mut ContactConstraint3D,
        dt: f32,
    ) -> f32 {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        let v_a = body_a.velocity + body_a.angular_velocity.cross(r_a);
        let v_b = body_b.velocity + body_b.angular_velocity.cross(r_b);
        let relative_velocity = v_b - v_a;

        let vn = relative_velocity.dot(constraint.normal);

        let bias = if constraint.rest_bias > 0.0 {
            0.0
        } else if dt > 0.0 {
            (self.baumgarte * (constraint.penetration - self.slop).max(0.0) / dt)
                .min(Self::MAX_BIAS_VELOCITY)
        } else {
            0.0
        };

        let lambda = constraint.mass_normal * (constraint.rest_bias - vn + bias);

        let old_impulse = constraint.normal_impulse;
        constraint.normal_impulse = (old_impulse + lambda).max(0.0);
        let applied = constraint.normal_impulse - old_impulse;

        if applied != 0.0 {
            let p = constraint.normal * applied;
            Self::apply_impulse_with(
                bodies,
                constraint.body_a,
                constraint.body_b,
                r_a,
                r_b,
                p,
                constraint.inv_inertia_world_a,
                constraint.inv_inertia_world_b,
            );
        }
        applied.abs()
    }

    fn solve_friction(
        &self,
        bodies: &mut [RigidBody3],
        constraint: &mut ContactConstraint3D,
    ) -> f32 {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        let v_a = body_a.velocity + body_a.angular_velocity.cross(r_a);
        let v_b = body_b.velocity + body_b.angular_velocity.cross(r_b);
        let relative_velocity = v_b - v_a;

        let vt1 = relative_velocity.dot(constraint.tangent1);
        let lambda1 = constraint.mass_tangent1 * -vt1;

        let vt2 = relative_velocity.dot(constraint.tangent2);
        let lambda2 = constraint.mass_tangent2 * -vt2;

        let max_friction = constraint.friction * constraint.normal_impulse;

        let old_impulse1 = constraint.tangent1_impulse;
        constraint.tangent1_impulse = (old_impulse1 + lambda1).clamp(-max_friction, max_friction);
        let applied1 = constraint.tangent1_impulse - old_impulse1;

        let old_impulse2 = constraint.tangent2_impulse;
        constraint.tangent2_impulse = (old_impulse2 + lambda2).clamp(-max_friction, max_friction);
        let applied2 = constraint.tangent2_impulse - old_impulse2;

        if applied1 != 0.0 || applied2 != 0.0 {
            let p = constraint.tangent1 * applied1 + constraint.tangent2 * applied2;
            Self::apply_impulse_with(
                bodies,
                constraint.body_a,
                constraint.body_b,
                r_a,
                r_b,
                p,
                constraint.inv_inertia_world_a,
                constraint.inv_inertia_world_b,
            );
        }
        applied1.abs().max(applied2.abs())
    }

    fn restitution_bias(&self, bodies: &[RigidBody3], constraint: &ContactConstraint3D) -> f32 {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        let v_a = body_a.velocity + body_a.angular_velocity.cross(r_a);
        let v_b = body_b.velocity + body_b.angular_velocity.cross(r_b);
        let vn = (v_b - v_a).dot(constraint.normal);

        if vn < -1.0 {
            -constraint.restitution * vn
        } else {
            0.0
        }
    }

    /// Apply an impulse, using the caller-provided world tensors.
    /// Callers pass the snapshot from the constraint when available;
    /// the warm-start path (pre-mass-terms) falls back to fresh tensors.
    ///
    /// Borrows both bodies through one `split_at_mut` (see 2D
    /// `apply_impulse`): ~2 bounds checks per impulse instead of ~6.
    /// `body_a != body_b` always holds for broadphase pairs; out-of-range
    /// indices panic exactly as before.
    ///
    /// The eight parameters are the full impulse context (bodies, contact
    /// geometry, impulse, precomputed tensors); bundling them would hide
    /// the data flow at the three call sites without removing any work.
    #[allow(clippy::too_many_arguments)]
    fn apply_impulse_with(
        bodies: &mut [RigidBody3],
        body_a: usize,
        body_b: usize,
        r_a: Vec3,
        r_b: Vec3,
        p: Vec3,
        inv_inertia_a: Mat3,
        inv_inertia_b: Mat3,
    ) {
        debug_assert!(body_a != body_b, "contact pair must be two distinct bodies");
        let (a, b) = if body_a < body_b {
            let (head, tail) = bodies.split_at_mut(body_b);
            (&mut head[body_a], &mut tail[0])
        } else {
            let (head, tail) = bodies.split_at_mut(body_a);
            (&mut tail[0], &mut head[body_b])
        };

        a.velocity -= p * a.inv_mass;
        b.velocity += p * b.inv_mass;

        let torque_a = r_a.cross(p);
        let torque_b = r_b.cross(p);

        a.angular_velocity -= inv_inertia_a * torque_a;
        b.angular_velocity += inv_inertia_b * torque_b;
    }

    fn apply_impulse(
        bodies: &mut [RigidBody3],
        body_a: usize,
        body_b: usize,
        r_a: Vec3,
        r_b: Vec3,
        p: Vec3,
    ) {
        // World-frame tensors: torques are world-space.
        let inv_inertia_a = bodies[body_a].world_inv_inertia();
        let inv_inertia_b = bodies[body_b].world_inv_inertia();
        Self::apply_impulse_with(
            bodies,
            body_a,
            body_b,
            r_a,
            r_b,
            p,
            inv_inertia_a,
            inv_inertia_b,
        );
    }
}

impl Default for Solver3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_box(id: usize, pos: Vec3, vel: Vec3) -> RigidBody3 {
        let mut b = RigidBody3::new(id);
        b.transform.position = pos;
        b.velocity = vel;
        b.inv_inertia = glam::Mat3::ZERO;
        b
    }

    #[test]
    fn test_solver3d_creation() {
        let solver = Solver3D::new();
        assert_eq!(solver.iterations, 8);
    }

    #[test]
    fn test_friction_kills_tangential_velocity() {
        let mut bodies = vec![
            RigidBody3::static_body(0),
            dynamic_box(1, Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.4, -1.0, 0.0)),
        ];
        let mut constraints = vec![ContactConstraint3D::new(
            0,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::Y,
            0.01,
            0.5,
            0.0,
        )];

        Solver3D::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        assert!(
            bodies[1].velocity.x.abs() < 0.05,
            "friction should stop sliding, vx={}",
            bodies[1].velocity.x
        );
    }

    #[test]
    fn test_effective_mass_uses_world_inertia() {
        use glam::Quat;
        use std::f32::consts::FRAC_PI_2;
        // Non-cube box (he 1,2,3; mass 48 at density 1) rotated 90 deg
        // about Z: local Y lands on world X, so the world-frame inverse
        // inertia along X is 1/Iyy_local = 1/160. A body-frame tensor
        // would wrongly use 1/Ixx_local = 1/208.
        let mut anchor = RigidBody3::static_body(0);
        anchor.transform.position = Vec3::ZERO;
        let mut b = RigidBody3::new(1);
        b.set_shape(
            crate::body::Shape3::Box {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
            },
            1.0,
        )
        .expect("valid shape");
        b.transform.rotation = Quat::from_rotation_z(FRAC_PI_2);
        b.transform.position = Vec3::ZERO;
        b.velocity = Vec3::new(0.0, 0.0, -1.0);
        let mut bodies = vec![anchor, b];
        // Contact above the center, normal +Z: r x n = +X world.
        let mut constraints = vec![ContactConstraint3D::new(
            0,
            1,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Z,
            0.0,
            0.5,
            0.0,
        )];
        // Single iteration: exactly one lambda application, so the
        // response equals the effective-mass prediction with no
        // multi-iteration convergence or angular feedback mixed in.
        // m_n = 1/(1/48 + 1/160) = 36.923; dv_z = 36.923/48 = 0.76923.
        // (A body-frame tensor would give 1/(1/48+1/208) -> dv 0.8125.)
        let solver = Solver3D {
            iterations: 1,
            ..Solver3D::new()
        };
        solver.solve(&mut bodies, &mut constraints, 1.0 / 60.0);
        let vz = bodies[1].velocity.z;
        assert!(
            (vz - (-1.0 + 0.76923)).abs() < 0.005,
            "world-frame inertia response wrong: vz={vz}"
        );
        // Angular kick lands on world X: dw = (r x P)/Iyy_local.
        let wx = bodies[1].angular_velocity.x;
        assert!(
            (wx - 36.923 / 160.0).abs() < 0.005,
            "angular response wrong: wx={wx}"
        );
    }

    #[test]
    fn test_rotation_integrates_in_world_frame() {
        use glam::Quat;
        use std::f32::consts::FRAC_PI_2;
        // Body pre-rotated 90 deg about Y, spinning about world Z.
        // Marker (local +Z) starts at world +X; world-frame integration
        // must swing it toward +Y. Body-frame integration would spin it
        // about world +X instead (marker stuck).
        let mut body = RigidBody3::new(0);
        body.transform.rotation = Quat::from_rotation_y(FRAC_PI_2);
        body.angular_velocity = Vec3::Z;
        body.integrate(0.01);
        let marker = body.transform.transform_point(Vec3::Z);
        assert!(
            marker.y > 0.005,
            "marker must swing toward +Y in world frame, got {marker}"
        );
        assert!(
            (marker.x - 1.0).abs() < 0.01,
            "marker x should stay ~1, got {marker}"
        );
    }

    #[test]
    fn test_early_exit_on_resting_contact() {
        let mut bodies = vec![
            RigidBody3::static_body(0),
            dynamic_box(1, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO),
        ];
        let mut constraints = vec![ContactConstraint3D::new(
            0,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::Y,
            0.001,
            0.5,
            0.0,
        )];

        let ran = Solver3D::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);
        assert_eq!(ran, 1, "resting contact should converge immediately");
    }

    #[test]
    fn test_normal_impulse_positive() {
        let mut bodies = vec![
            RigidBody3::static_body(0),
            dynamic_box(1, Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, -1.0, 0.0)),
        ];
        let mut constraints = vec![ContactConstraint3D::new(
            0,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::Y,
            0.01,
            0.5,
            0.0,
        )];

        Solver3D::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        assert!(constraints[0].normal_impulse >= 0.0);
    }
}
