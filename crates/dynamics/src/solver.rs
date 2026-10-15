use super::body::RigidBody2;
use glam::Vec2;

/// Contact constraint for the solver (2D sequential impulse with Coulomb friction).
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    /// Body A index
    pub body_a: usize,
    /// Body B index
    pub body_b: usize,
    /// Contact point in world space
    pub point: Vec2,
    /// Contact normal (from A to B, unit length)
    pub normal: Vec2,
    /// Penetration depth
    pub penetration: f32,
    /// Friction coefficient (mu)
    pub friction: f32,
    /// Restitution
    pub restitution: f32,
    /// Accumulated normal impulse j_n (persisted across frames via contact cache)
    pub normal_impulse: f32,
    /// Accumulated tangent impulse j_t (persisted, Coulomb-clamped |j_t| <= mu * j_n)
    pub tangent_impulse: f32,
    /// Restitution bias velocity for this step, computed once from the
    /// PRE-solve relative velocity (Box2D-style `velocityBias`). Must NOT be
    /// recomputed per iteration: after iteration 1 the contact separates,
    /// and a fresh target of 0 would un-apply the bounce within the frame.
    /// Recomputed every step; never cached across frames.
    pub rest_bias: f32,
    /// Effective mass along the normal, snapshotted once per step in
    /// [`Solver::solve`] (same policy as the 3D solver's mass terms).
    /// Recomputing it every iteration costs a full cross-term evaluation
    /// per contact per pass with no convergence benefit at our iteration
    /// counts; the snapshot is a pure function of step-start state, so
    /// determinism is preserved.
    pub mass_normal: f32,
    /// Effective mass along the tangent, snapshotted once per step.
    pub mass_tangent: f32,
}

impl ContactConstraint {
    /// Tangent direction: 90-degree CCW rotation of the normal.
    /// Must stay in sync with the C# `ContactConstraint.Tangent` property.
    pub fn tangent(&self) -> Vec2 {
        Vec2::new(-self.normal.y, self.normal.x)
    }
}

/// Constraint solver for physics engine
#[derive(Debug)]
pub struct Solver {
    /// Maximum solver iterations per step.
    pub iterations: usize,
    /// Bauman correction factor (0.0 to 0.2)
    pub slop: f32,
    /// Baumgarte factor (0.0 to 0.3)
    pub baumgarte: f32,
    /// Early-exit tolerance: an iteration that applies less than this
    /// (max absolute impulse delta over all contacts) ends the solve.
    /// Settled scenes converge in 1-3 iterations instead of running all
    /// [`Solver::iterations`]. Deterministic: pure function of state.
    pub early_exit_tolerance: f32,
    /// Solve disjoint constraint islands in parallel with scoped threads
    /// (see [`crate::island`]). Default off: the serial path is the
    /// reference behavior. When on and the step holds 0-1 islands, the
    /// solver takes the exact serial path (zero behavior change).
    pub parallel: bool,
}

impl Solver {
    /// Maximum Baumgarte bias velocity (m/s). Deep overlaps resolve
    /// gradually over several frames instead of ejecting: without the cap,
    /// a clamped 1 m penetration targets ~12 m/s separation, and the
    /// ejected bodies outrun discrete collision (no CCD) and tunnel.
    /// Contacts shallower than ~17 cm at 60 Hz are unaffected, so settled
    /// stacks behave bit-identically with or without the cap.
    pub const MAX_BIAS_VELOCITY: f32 = 2.0;

    pub fn new() -> Self {
        Self {
            iterations: 8,
            slop: 0.005,
            baumgarte: 0.2,
            early_exit_tolerance: 1e-6,
            parallel: false,
        }
    }

    /// Solve all constraints. `dt` is the timestep (used for Baumgarte scaling).
    ///
    /// Each iteration runs a full normal pass over all contacts before the
    /// friction pass, so friction responds to the settled post-normal
    /// velocities rather than half-updated per-contact state.
    ///
    /// Returns the iterations actually run (early exit when converged).
    pub fn solve(
        &self,
        bodies: &mut [RigidBody2],
        constraints: &mut [ContactConstraint],
        dt: f32,
    ) -> usize {
        // Mass terms + warm start + restitution targets from pre-solve state.
        for constraint in constraints.iter_mut() {
            Self::compute_mass_terms(bodies, constraint);
            Self::apply_warm_starting(bodies, constraint);
            constraint.rest_bias = restitution_bias(bodies, constraint);
        }

        let mut ran = 0;
        for _ in 0..self.iterations {
            ran += 1;
            let mut max_applied: f32 = 0.0;
            for constraint in constraints.iter_mut() {
                max_applied = max_applied.max(self.solve_normal(bodies, constraint, dt));
            }
            for constraint in constraints.iter_mut() {
                max_applied = max_applied.max(Self::solve_friction(bodies, constraint));
            }
            if max_applied < self.early_exit_tolerance {
                break;
            }
        }
        ran
    }

    /// Iteration core over an explicit constraint index subset (stable
    /// order). [`Solver::solve`] is this with `0..constraints.len()`.
    /// Used by the parallel island path, where each thread owns a
    /// disjoint index subset (see [`crate::island`]).
    pub fn solve_indexed(
        &self,
        bodies: &mut [RigidBody2],
        constraints: &mut [ContactConstraint],
        indices: &[usize],
        dt: f32,
    ) -> usize {
        let mut ran = 0;
        for _ in 0..self.iterations {
            ran += 1;
            let mut max_applied: f32 = 0.0;
            for &ci in indices {
                // Index provenance: built from live constraint positions by
                // the island partitioner; bounds-checked here so a corrupt
                // index list fails loud, never silently.
                let constraint = &mut constraints[ci];
                max_applied = max_applied.max(self.solve_normal(bodies, constraint, dt));
            }
            for &ci in indices {
                let constraint = &mut constraints[ci];
                max_applied = max_applied.max(Self::solve_friction(bodies, constraint));
            }
            if max_applied < self.early_exit_tolerance {
                break;
            }
        }
        ran
    }

    /// Parallel island solve over a pre-partitioned constraint index map.
    ///
    /// `island_bodies[i]` / `island_constraints[i]` hold the body and
    /// constraint indices of island `i` (canonical island order, stable
    /// within-island order), and islands are body-disjoint (see
    /// [`crate::island`]). `max_slots` is caller-owned scratch with one
    /// `f32` per island, reused across steps (no per-step allocation).
    ///
    /// Convergence parity with the serial path is exact: threads
    /// rendezvous on a barrier after every iteration and share one global
    /// max-applied impulse, so the early-exit decision is identical on
    /// every thread. (Per-island early exit would diverge — a settled
    /// island stopping early while a live island keeps iterating changes
    /// float history vs serial.)
    ///
    /// # Safety
    /// Callers must guarantee body-disjointness AND constraint-index
    /// disjointness across islands (the island builder plus
    /// [`crate::island::assign_constraints`] do; debug builds re-verify
    /// both). Threads reconstruct `&mut` slices from raw addresses over
    /// disjoint index sets, which is sound if and only if no two threads
    /// address the same body or the same constraint. Violating that is
    /// instant undefined behavior — hence the debug verifier, not just
    /// documentation.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_parallel(
        &self,
        bodies: &mut [RigidBody2],
        constraints: &mut [ContactConstraint],
        island_bodies: &[Vec<usize>],
        island_constraints: &[Vec<usize>],
        max_slots: &mut [f32],
        dt: f32,
    ) -> usize {
        debug_assert_eq!(
            island_bodies.len(),
            island_constraints.len(),
            "island body/constraint lists must align"
        );
        debug_assert_eq!(
            max_slots.len(),
            island_constraints.len(),
            "max scratch must cover every island"
        );
        #[cfg(debug_assertions)]
        {
            // Verify BOTH disjointness invariants before spawning threads.
            let mut seen_body = vec![false; bodies.len()];
            for list in island_bodies {
                for &b in list {
                    assert!(b < bodies.len(), "island body out of range");
                    assert!(!seen_body[b], "body {b} in two islands (parallel UB)");
                    seen_body[b] = true;
                }
            }
            let mut seen = vec![false; constraints.len()];
            for list in island_constraints {
                for &ci in list {
                    assert!(ci < constraints.len(), "island constraint out of range");
                    assert!(!seen[ci], "constraint {ci} in two islands (parallel UB)");
                    seen[ci] = true;
                }
            }
        }
        // Exact serial pre-loop (see `Solver::solve`): mass terms, warm
        // start, restitution targets. `World2::step` only SEEDED the
        // accumulators from the cache; applying them to bodies happens
        // here, once — computed serially because every thread would
        // otherwise recompute identical values. The threaded section
        // below covers iterations only.
        for constraint in constraints.iter_mut() {
            Self::compute_mass_terms(bodies, constraint);
            Self::apply_warm_starting(bodies, constraint);
            constraint.rest_bias = restitution_bias(bodies, constraint);
        }
        if island_constraints.len() <= 1 {
            // Degenerate case: behave exactly like the serial path.
            let all: Vec<usize> = (0..constraints.len()).collect();
            return self.solve_indexed(bodies, constraints, &all, dt);
        }

        // Raw addresses travel as `usize` (raw pointers are `!Send`
        // here, and edition-2021 closures capture struct fields, so a
        // carrier struct does not help); slices are rebuilt per thread.
        // Soundness rests entirely on island body-disjointness
        // (debug-verified above): each address is dereferenced only at
        // indices exclusive to its island.
        let bodies_addr = bodies.as_mut_ptr() as usize;
        let bodies_len = bodies.len();
        let constraints_addr = constraints.as_mut_ptr() as usize;
        let constraints_len = constraints.len();
        let max_addr = max_slots.as_mut_ptr() as usize;
        let iters = self.iterations;
        let tol = self.early_exit_tolerance;
        let this = self;
        let barrier = std::sync::Barrier::new(island_constraints.len());

        for slot in max_slots.iter_mut() {
            *slot = 0.0;
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(island_constraints.len());
            for (ii, list) in island_constraints.iter().enumerate() {
                // Shared by reference: `&Barrier` is `Send` (`Barrier: Sync`).
                let barrier = &barrier;
                handles.push(scope.spawn(move || {
                    // SAFETY: per-island exclusive indices (verified above
                    // in debug; partitioner-guaranteed in release) plus a
                    // barrier happens-before between iterations, so the
                    // shared max slots are data-race-free: each thread
                    // writes only its own slot, and reads all slots only
                    // after the rendezvous.
                    let bodies = unsafe {
                        std::slice::from_raw_parts_mut(bodies_addr as *mut RigidBody2, bodies_len)
                    };
                    let constraints = unsafe {
                        std::slice::from_raw_parts_mut(
                            constraints_addr as *mut ContactConstraint,
                            constraints_len,
                        )
                    };
                    let mut ran = 0;
                    for _ in 0..iters {
                        ran += 1;
                        let mut local_max: f32 = 0.0;
                        for &ci in list.iter() {
                            let c = &mut constraints[ci];
                            local_max = local_max.max(this.solve_normal(bodies, c, dt));
                        }
                        for &ci in list.iter() {
                            let c = &mut constraints[ci];
                            local_max = local_max.max(Self::solve_friction(bodies, c));
                        }
                        unsafe {
                            *(max_addr as *mut f32).add(ii) = local_max;
                        }
                        barrier.wait();
                        let mut global_max: f32 = 0.0;
                        for k in 0..island_constraints.len() {
                            global_max =
                                global_max.max(unsafe { *(max_addr as *const f32).add(k) });
                        }
                        if global_max < tol {
                            break;
                        }
                    }
                    ran
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .max()
                .unwrap_or(0)
        })
    }

    /// Snapshot per-step effective masses from step-start state.
    /// Called once per constraint per step (see `mass_normal` docs).
    fn compute_mass_terms(bodies: &[RigidBody2], constraint: &mut ContactConstraint) {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];
        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;
        constraint.mass_normal = effective_mass(
            body_a.inv_mass,
            body_b.inv_mass,
            body_a.inv_inertia,
            body_b.inv_inertia,
            r_a,
            r_b,
            constraint.normal,
        );
        constraint.mass_tangent = effective_mass(
            body_a.inv_mass,
            body_b.inv_mass,
            body_a.inv_inertia,
            body_b.inv_inertia,
            r_a,
            r_b,
            constraint.tangent(),
        );
    }

    /// Apply cached impulses from the previous frame directly (no re-scaling:
    /// the stored scalars already are impulses).
    fn apply_warm_starting(bodies: &mut [RigidBody2], constraint: &ContactConstraint) {
        if constraint.normal_impulse == 0.0 && constraint.tangent_impulse == 0.0 {
            return;
        }
        let tangent = constraint.tangent();
        let p =
            constraint.normal * constraint.normal_impulse + tangent * constraint.tangent_impulse;

        let r_a = constraint.point - bodies[constraint.body_a].transform.position;
        let r_b = constraint.point - bodies[constraint.body_b].transform.position;

        apply_impulse(bodies, constraint.body_a, constraint.body_b, r_a, r_b, p);
    }

    /// Solve the normal (non-penetration + restitution + Baumgarte) constraint
    /// with impulse accumulation (j_n clamped >= 0). Returns the absolute
    /// applied impulse (drives the convergence early-exit).
    fn solve_normal(
        &self,
        bodies: &mut [RigidBody2],
        constraint: &mut ContactConstraint,
        dt: f32,
    ) -> f32 {
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        // Relative velocity at contact point
        let v_a = body_a.velocity + vec2_to_angular(r_a, body_a.angular_velocity);
        let v_b = body_b.velocity + vec2_to_angular(r_b, body_b.angular_velocity);
        let relative_velocity = v_b - v_a;

        let vn = relative_velocity.dot(constraint.normal);

        let mass_n = constraint.mass_normal;
        if mass_n == 0.0 {
            return 0.0;
        }

        // Restitution bias was fixed pre-step (see `rest_bias`); the
        // Baumgarte term handles position correction. Baumgarte is skipped
        // while restitution is active: the bounce separates the contact
        // anyway, and stacking the positional push on top of the bounce
        // velocity pumps energy into the rebound.
        let bias = if constraint.rest_bias > 0.0 {
            0.0
        } else if dt > 0.0 {
            (self.baumgarte * (constraint.penetration - self.slop).max(0.0) / dt)
                .min(Self::MAX_BIAS_VELOCITY)
        } else {
            0.0
        };

        let lambda = mass_n * (constraint.rest_bias - vn + bias);

        // Accumulate and clamp (non-penetration is one-sided)
        let old_impulse = constraint.normal_impulse;
        constraint.normal_impulse = (old_impulse + lambda).max(0.0);
        let applied = constraint.normal_impulse - old_impulse;

        if applied != 0.0 {
            apply_impulse(
                bodies,
                constraint.body_a,
                constraint.body_b,
                r_a,
                r_b,
                constraint.normal * applied,
            );
        }
        applied.abs()
    }

    /// Solve the tangential (Coulomb friction) constraint with accumulation.
    /// Clamps |j_t| <= mu * j_n (friction cone). Returns the absolute
    /// applied impulse.
    fn solve_friction(bodies: &mut [RigidBody2], constraint: &mut ContactConstraint) -> f32 {
        let tangent = constraint.tangent();
        let body_a = &bodies[constraint.body_a];
        let body_b = &bodies[constraint.body_b];

        let r_a = constraint.point - body_a.transform.position;
        let r_b = constraint.point - body_b.transform.position;

        let v_a = body_a.velocity + vec2_to_angular(r_a, body_a.angular_velocity);
        let v_b = body_b.velocity + vec2_to_angular(r_b, body_b.angular_velocity);
        let vt = (v_b - v_a).dot(tangent);

        let mass_t = constraint.mass_tangent;
        if mass_t == 0.0 {
            return 0.0;
        }

        let lambda = mass_t * -vt;

        // Coulomb cone clamp against the current accumulated normal impulse
        let max_friction = constraint.friction * constraint.normal_impulse;
        let old_impulse = constraint.tangent_impulse;
        constraint.tangent_impulse = (old_impulse + lambda).clamp(-max_friction, max_friction);
        let applied = constraint.tangent_impulse - old_impulse;

        if applied != 0.0 {
            apply_impulse(
                bodies,
                constraint.body_a,
                constraint.body_b,
                r_a,
                r_b,
                tangent * applied,
            );
        }
        applied.abs()
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

/// Restitution target from pre-solve relative normal velocity.
/// Bounce only for significant impacts (Box2D uses 1 m/s threshold).
#[inline]
fn restitution_bias(bodies: &[RigidBody2], constraint: &ContactConstraint) -> f32 {
    let body_a = &bodies[constraint.body_a];
    let body_b = &bodies[constraint.body_b];
    let r_a = constraint.point - body_a.transform.position;
    let r_b = constraint.point - body_b.transform.position;
    let v_a = body_a.velocity + vec2_to_angular(r_a, body_a.angular_velocity);
    let v_b = body_b.velocity + vec2_to_angular(r_b, body_b.angular_velocity);
    let vn = (v_b - v_a).dot(constraint.normal);
    if vn < -1.0 {
        -constraint.restitution * vn
    } else {
        0.0
    }
}

/// Effective mass along a direction: 1 / (inv_m_a + inv_m_b + inv_i_a*(r_a x d)^2 + inv_i_b*(r_b x d)^2).
/// Returns 0 when both bodies are static along this direction.
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
    let r_a_cross = r_a.x * dir.y - r_a.y * dir.x;
    let r_b_cross = r_b.x * dir.y - r_b.y * dir.x;
    let k = inv_mass_a
        + inv_mass_b
        + inv_inertia_a * r_a_cross * r_a_cross
        + inv_inertia_b * r_b_cross * r_b_cross;
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Apply a world-space impulse P at the contact (A -= P, B += P).
///
/// Borrows both bodies through one `split_at_mut` instead of six separate
/// indexed accesses, so the hot loop pays ~2 bounds checks per impulse
/// instead of ~6. `body_a != body_b` always holds — broadphase pairs are
/// distinct bodies and FFI ids are validated before use — and out-of-range
/// indices panic exactly as before (pipeline corruption must stay loud).
fn apply_impulse(
    bodies: &mut [RigidBody2],
    body_a: usize,
    body_b: usize,
    r_a: Vec2,
    r_b: Vec2,
    p: Vec2,
) {
    debug_assert!(body_a != body_b, "contact pair must be two distinct bodies");
    // Split once at the higher index: both halves are in range-checked
    // on a single call, then each body is addressed with one check.
    let (a, b) = if body_a < body_b {
        let (head, tail) = bodies.split_at_mut(body_b);
        (&mut head[body_a], &mut tail[0])
    } else {
        let (head, tail) = bodies.split_at_mut(body_a);
        (&mut tail[0], &mut head[body_b])
    };

    a.velocity -= p * a.inv_mass;
    b.velocity += p * b.inv_mass;

    a.angular_velocity -= (r_a.x * p.y - r_a.y * p.x) * a.inv_inertia;
    b.angular_velocity += (r_b.x * p.y - r_b.y * p.x) * b.inv_inertia;
}

/// Convert angular velocity to linear velocity contribution at a point
#[inline]
fn vec2_to_angular(r: Vec2, angular_velocity: f32) -> Vec2 {
    Vec2::new(-r.y, r.x) * angular_velocity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_box(id: usize, pos: Vec2, vel: Vec2) -> RigidBody2 {
        let mut b = RigidBody2::new(id);
        b.transform.position = pos;
        b.velocity = vel;
        // Isolate linear response: lock rotation.
        b.inv_inertia = 0.0;
        b
    }

    fn contact(a: usize, b: usize, point: Vec2, normal: Vec2) -> ContactConstraint {
        ContactConstraint {
            body_a: a,
            body_b: b,
            point,
            normal,
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

    #[test]
    fn test_solver_creation() {
        let solver = Solver::new();
        assert_eq!(solver.iterations, 8);
    }

    #[test]
    fn test_friction_kills_tangential_velocity() {
        // Static floor (A) + slow-sliding dynamic box (B). One frame of
        // friction (mu=0.5) can erase 0.5*jn of tangential velocity.
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::new(0.4, -1.0)),
        ];
        let mut constraints = vec![contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)];

        Solver::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        // Tangential motion must be fully stopped (cone large enough).
        assert!(
            bodies[1].velocity.x.abs() < 0.05,
            "friction should stop sliding, vx={}",
            bodies[1].velocity.x
        );
        // Normal impulse must be non-negative; friction cone respected.
        let c = &constraints[0];
        assert!(c.normal_impulse >= 0.0);
        assert!(
            c.tangent_impulse.abs() <= c.friction * c.normal_impulse + 1e-4,
            "|jt|={} > mu*jn={}",
            c.tangent_impulse.abs(),
            c.friction * c.normal_impulse
        );
    }

    #[test]
    fn test_zero_friction_preserves_sliding() {
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::new(3.0, -1.0)),
        ];
        let mut constraints = vec![ContactConstraint {
            friction: 0.0,
            ..contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)
        }];

        Solver::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        // No friction: tangential velocity untouched, no tangent impulse.
        assert!(
            (bodies[1].velocity.x - 3.0).abs() < 1e-4,
            "vx should be preserved, got {}",
            bodies[1].velocity.x
        );
        assert_eq!(constraints[0].tangent_impulse, 0.0);
    }

    #[test]
    fn test_friction_cone_clamp() {
        // Fast slider, low mu: tangent impulse must saturate at mu * jn.
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::new(20.0, -1.0)),
        ];
        let mut constraints = vec![ContactConstraint {
            friction: 0.1,
            ..contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)
        }];

        Solver::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        let c = &constraints[0];
        let max_f = c.friction * c.normal_impulse;
        assert!(
            (c.tangent_impulse.abs() - max_f).abs() < 1e-3,
            "cone should saturate: |jt|={} max={max_f}",
            c.tangent_impulse.abs()
        );
        // Sliding remains (cone too small to stop it).
        assert!(bodies[1].velocity.x > 10.0);
    }

    #[test]
    fn test_early_exit_on_resting_contact() {
        // Resting contact (zero velocity, penetration within slop):
        // nothing to apply, so the solver must exit after 1 iteration
        // instead of running all 8.
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::ZERO),
        ];
        let mut constraints = vec![ContactConstraint {
            penetration: 0.001,
            ..contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)
        }];

        let ran = Solver::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);
        assert_eq!(ran, 1, "resting contact should converge immediately");
        assert_eq!(constraints[0].normal_impulse, 0.0);
    }

    #[test]
    fn test_no_early_exit_while_resolving() {
        // Deep penetration keeps applying Baumgarte corrections, so the
        // solver must NOT exit after the first iteration.
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::ZERO),
        ];
        let mut constraints = vec![ContactConstraint {
            penetration: 0.5,
            ..contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)
        }];

        let ran = Solver::new().solve(&mut bodies, &mut constraints, 1.0 / 60.0);
        assert!(ran > 1, "resolving contact must iterate, ran={ran}");
        assert!(constraints[0].normal_impulse > 0.0);
    }

    #[test]
    fn test_warm_start_applies_cached_impulse() {
        // Two static-anchored bodies: pre-loaded impulses must move velocities
        // before any iteration runs (iterations=0 proves warm start alone acts).
        let mut bodies = vec![
            RigidBody2::static_body(0),
            dynamic_box(1, Vec2::new(0.0, 0.5), Vec2::ZERO),
        ];
        let mut constraints = vec![ContactConstraint {
            normal_impulse: 2.0,
            tangent_impulse: 1.0,
            ..contact(0, 1, Vec2::new(0.0, 0.0), Vec2::Y)
        }];

        let solver = Solver {
            iterations: 0,
            ..Solver::new()
        };
        solver.solve(&mut bodies, &mut constraints, 1.0 / 60.0);

        // P = n*2 + t*1 with t = (-n.y, n.x) = (-1, 0): dv = ( -1*1*inv_m, 2*inv_m ).
        // Box mass is 1 (RigidBody2::new default) so inv_mass = 1.
        assert!(
            (bodies[1].velocity.x - -1.0).abs() < 1e-5,
            "vx={}",
            bodies[1].velocity.x
        );
        assert!(
            (bodies[1].velocity.y - 2.0).abs() < 1e-5,
            "vy={}",
            bodies[1].velocity.y
        );
    }
}
