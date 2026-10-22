use super::body::{
    validate_density, validate_shape2, validate_shape3, ContactEvent2, ContactEvent3, QueryFilter,
    RigidBody2, RigidBody3, Shape2, Shape3, ShapeError,
};
use super::ccd::{swept_aabb_overlap, swept_aabb_overlap_3, CcdTier};
use super::joint::{Joint2, JointError};
use super::joint3::Joint3;
use super::material::PhysicsMaterial;
use super::solver::{ContactConstraint, Solver};
use super::solver3d::{ContactConstraint3D, Solver3D};
use super::validation::{
    validate_body_2d, validate_body_3d, validate_constraints_2d, validate_constraints_3d,
    ValidationConfig, ValidationReport,
};
use glam::{Vec2, Vec3};
use phieng_broadphase::{Broadphase2, Broadphase3, BroadphaseMode};
use phieng_collision::query::{
    ray_capsule3, ray_circle2, ray_obb2, ray_obb3, ray_polygon2, ray_sphere3,
};
use phieng_collision::{Ray2, Ray3};
use phieng_math::fxhash::FxHashMap;
use phieng_math::{AABB2, AABB3};
use phieng_math::{ANGULAR_SLEEP_THRESHOLD, CONTACT_MATCH_DIST_SQ, EPS, LINEAR_SLEEP_THRESHOLD};

/// Time a block and add the elapsed milliseconds to `$acc`.
///
/// With the `phase-timing` feature off the macro expands to the block
/// alone: no clock read, no branch, no accumulator. With the feature on
/// the accumulator expression is only ever expanded in the timing build,
/// so the default build does not even name it.
#[cfg(feature = "phase-timing")]
macro_rules! timed {
    ($acc:expr, $body:block) => {{
        let __phase_start = ::std::time::Instant::now();
        let __phase_out = $body;
        $acc += __phase_start.elapsed().as_secs_f64() * 1000.0;
        __phase_out
    }};
}

#[cfg(not(feature = "phase-timing"))]
macro_rules! timed {
    ($acc:expr, $body:block) => {{
        $body
    }};
}

/// Cached contact impulses from the previous frame, used for warm starting.
/// Matched to new contacts by proximity of the contact point; the stored
/// normal guards against reusing impulses after the contact flips.
#[derive(Debug, Clone, Copy)]
pub struct CachedContact {
    /// Contact point in world space (last frame)
    pub point: Vec2,
    /// Contact normal (last frame)
    pub normal: Vec2,
    /// Accumulated normal impulse j_n
    pub normal_impulse: f32,
    /// Accumulated tangent impulse j_t
    pub tangent_impulse: f32,
}

/// Maximum squared distance for matching a new contact to a cached one
/// ((0.2 units)^2) — shared with all other crates via `phieng_math`.
/// Minimum number of island constraints before the parallel solver path
/// is worth its worker threads. Below this the serial path is faster (and
/// allocation-free) and produces the identical result.
const PARALLEL_MIN_CONSTRAINTS: usize = 256;

/// Pre-allocated scratch buffers for zero-allocation narrowphase.
///
/// All buffers are pre-allocated with sufficient capacity and reused
/// every step by setting length to 0 instead of clearing (which would
/// drop elements). This eliminates allocations in the steady state.
#[derive(Debug)]
struct ScratchBuffers2D {
    constraints: Vec<ContactConstraint>,
    island_max: Vec<f32>,
    /// Pooled island-partition scratch (see `crate::island::IslandScratch`).
    /// The partition is rebuilt every step; pooling its buffers keeps
    /// the parallel solver path allocation-free in the steady state.
    islands: super::island::IslandScratch,
    /// Pooled contact endpoint pairs (derived from `constraints`).
    contact_pairs: Vec<(usize, usize)>,
    /// Pooled joint endpoint pairs.
    joint_pairs: Vec<(usize, usize)>,
    pairs: Vec<(usize, usize)>,
    verts_a: Vec<Vec2>,
    verts_b: Vec<Vec2>,
    points: Vec<Vec2>,
    clip_tmp: Vec<Vec2>,
    /// One AABB per body, computed in a separate pass before the
    /// broad-phase insert so the insert loop reads a compact array
    /// instead of chasing each body's shape. Pooled like the rest.
    aabbs: Vec<(Vec2, Vec2)>,
    stale: Vec<(usize, usize)>,
    /// The contacts a sensor pair reported, kept apart from the solved
    /// ones: a sensor reports contacts but never pushes.
    sensor_contacts: Vec<ContactConstraint>,
    /// The sensor pairs that were touching last step, so the events can
    /// say which of them ended.
    last_sensor_pairs: Vec<(usize, usize)>,
    /// The sensor pairs touching this step.
    sensor_pairs: Vec<(usize, usize)>,
    /// The contact events of this step.
    events: Vec<ContactEvent2>,
    /// The last known contact of every pair, kept so an *ended* event
    /// can still say where the pair was.
    last_contacts: Vec<(usize, usize, Vec2, Vec2)>,
}

impl ScratchBuffers2D {
    /// Create new scratch buffers with capacity for a large scene.
    fn new() -> Self {
        Self {
            constraints: Vec::with_capacity(8192),
            island_max: Vec::with_capacity(256),
            islands: super::island::IslandScratch::new(),
            contact_pairs: Vec::with_capacity(8192),
            joint_pairs: Vec::with_capacity(256),
            pairs: Vec::with_capacity(16384),
            sensor_contacts: Vec::with_capacity(1024),
            last_sensor_pairs: Vec::with_capacity(1024),
            sensor_pairs: Vec::with_capacity(1024),
            events: Vec::with_capacity(1024),
            last_contacts: Vec::with_capacity(8192),
            verts_a: Vec::with_capacity(1024),
            verts_b: Vec::with_capacity(1024),
            points: Vec::with_capacity(512),
            clip_tmp: Vec::with_capacity(256),
            aabbs: Vec::with_capacity(2048),
            stale: Vec::new(),
        }
    }

    /// Reset all buffers to empty, keeping capacity.
    fn reset(&mut self) {
        self.constraints.clear();
        self.island_max.clear();
        self.contact_pairs.clear();
        self.joint_pairs.clear();
        self.pairs.clear();
        self.verts_a.clear();
        self.verts_b.clear();
        self.points.clear();
        self.clip_tmp.clear();
        self.aabbs.clear();
        self.stale.clear();
        self.sensor_contacts.clear();
        self.events.clear();
    }
}

impl Default for ScratchBuffers2D {
    fn default() -> Self {
        Self::new()
    }
}

/// Physics world in 2D
#[derive(Debug)]
pub struct World2 {
    /// Rigid bodies
    pub bodies: Vec<RigidBody2>,
    /// Broad phase mode (auto-selects BruteForce / SpatialHash / DynamicAABBTree).
    pub broadphase_mode: BroadphaseMode,
    /// Broad phase backend. Resolved from `broadphase_mode` at construction;
    /// `Hybrid` re-resolves each step based on live body count.
    pub broadphase: Broadphase2,
    /// Constraint solver
    pub solver: Solver,
    /// Gravity
    pub gravity: Vec2,
    /// Gravitational force model + integration scheme (see
    /// [`crate::gravity`]). Default is the legacy uniform field with
    /// semi-implicit Euler — bit-identical to pre-gravity behavior.
    pub gravity_config: super::gravity::GravityConfig2,
    /// Ambient force fields — spring (SHM), aerodynamic drag, and
    /// turbulence (see [`crate::ambient`]). Default is a no-op.
    pub ambient: super::ambient::AmbientConfig2,
    /// Collision-detection tier ceiling (see [`crate::collision_tier`]).
    /// Off by default → full fidelity (Tier5).
    pub collision_tier: super::collision_tier::CollisionTierConfig,
    /// Continuous collision detection via adaptive substepping (see
    /// [`crate::ccd`]). Default is a no-op (single legacy substep).
    pub ccd: super::ccd::CcdConfig,
    /// Monotonic simulation clock in seconds, advanced by `dt` each step.
    /// Drives time-varying ambient fields (turbulence).
    pub time: f32,
    /// Next body ID
    pub next_body_id: usize,
    /// Previous frame's contact impulses, keyed by (body_a, body_b).
    /// Fx-hashed: keys are internal body ids (never attacker-controlled),
    /// so SipHash's DOS resistance is pure overhead here.
    pub contact_cache: FxHashMap<(usize, usize), Vec<CachedContact>>,
    /// Collect the contact events (a pair began or stopped touching).
    ///
    /// Off by default: the events cost a pass over the contact cache,
    /// and a scene that never reads them should not pay for it. When on,
    /// the events of the last step are in [`Self::contact_events`].
    pub collect_contact_events: bool,
    /// Bodies touched by a contact this step (parallel to `bodies`).
    /// A slow body may only fall asleep while touched — support-gated
    /// sleep keeps ballistic apexes (no contacts) awake without any
    /// wake/sleep history. Transient scratch; never snapshotted.
    contact_touch: Vec<bool>,
    /// Recycled per-key `CachedContact` buffers. Vacated contact-cache
    /// entries return their `Vec` here instead of freeing it; new keys
    /// draw from it, so the contact cache reaches zero steady-state
    /// allocation even as pairs flicker. Internal (the public cache type
    /// is unchanged and `contact_cache.len()` still counts live pairs).
    contact_cache_pool: Vec<Vec<CachedContact>>,
    /// Fixed timestep
    pub dt: f32,
    /// Velocity threshold for sleeping
    pub linear_sleep_threshold: f32,
    /// Angular velocity threshold for sleeping
    pub angular_sleep_threshold: f32,
    /// Runtime validation configuration
    pub validation: ValidationConfig,
    /// Last validation report
    pub validation_report: ValidationReport,
    /// Joint constraints (see [`crate::joint`]). Ids double as indices
    /// (same invariant as bodies); removal shifts the tail.
    pub joints: Vec<Joint2>,
    /// Next joint ID
    pub next_joint_id: usize,
    /// SPH fluid state (see [`crate::fluid`]). Empty by default, in
    /// which case the fluid phase is a no-op.
    pub fluid: super::fluid::FluidState2,
    /// Thermal simulation state (see [`crate::thermal`]). Disabled by default.
    pub thermal: super::thermal::ThermalState2,
    /// Nuclear simulation state (see [`crate::nuclear`]). Disabled by
    /// default.
    pub nuclear: super::nuclear::NuclearState2,
    /// Per-phase wall-clock accumulators. Only present with the
    /// `phase-timing` feature; the default build has no such field, so
    /// the struct's layout and public API are unchanged.
    #[cfg(feature = "phase-timing")]
    phase_timings: super::phase_timing::PhaseTimings,
    /// Scratch buffers reused every step so the per-frame narrowphase
    /// performs no allocations in the steady state.
    scratch: ScratchBuffers2D,
}

/// The contact data of a sensor pair, if it has one (3D).
fn sensor_contact_3d(contacts: &[ContactConstraint3D], key: (usize, usize)) -> (Vec3, Vec3) {
    contacts
        .iter()
        .find(|c| (c.body_a, c.body_b) == key)
        .map(|c| (c.point, c.normal))
        .unwrap_or((Vec3::ZERO, Vec3::ZERO))
}

/// The two bodies of a pair, in a canonical order.
///
/// The narrow phase does not promise an order, so the events put the
/// smaller id first: an event says *which two* bodies began or
/// stopped touching, and that must not depend on the order the
/// narrow phase happened to produce.
fn event_pair(a: usize, b: usize) -> (u32, u32) {
    if a <= b {
        (a as u32, b as u32)
    } else {
        (b as u32, a as u32)
    }
}

/// The contact data of a sensor pair, if it has one.
fn sensor_contact_2d(contacts: &[ContactConstraint], key: (usize, usize)) -> (Vec2, Vec2) {
    contacts
        .iter()
        .find(|c| (c.body_a, c.body_b) == key)
        .map(|c| (c.point, c.normal))
        .unwrap_or((Vec2::ZERO, Vec2::ZERO))
}

/// What a world ray cast hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRayHit2 {
    /// The body that was hit.
    pub body: u32,
    /// The distance along the ray.
    pub t: f32,
    /// The hit point, in world space.
    pub point: Vec2,
    /// The surface normal at the hit, in world space, facing the ray.
    pub normal: Vec2,
}

/// What a world ray cast hit in 3D. See [`WorldRayHit2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRayHit3 {
    /// The body that was hit.
    pub body: u32,
    /// The distance along the ray.
    pub t: f32,
    /// The hit point, in world space.
    pub point: Vec3,
    /// The surface normal at the hit, in world space, facing the ray.
    pub normal: Vec3,
}

impl World2 {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            broadphase_mode: BroadphaseMode::default(),
            broadphase: Broadphase2::new(BroadphaseMode::default(), 1.0),
            solver: Solver::new(),
            gravity: Vec2::new(0.0, -9.81),
            gravity_config: super::gravity::GravityConfig2::default(),
            ambient: super::ambient::AmbientConfig2::default(),
            collision_tier: super::collision_tier::CollisionTierConfig::default(),
            ccd: super::ccd::CcdConfig::default(),
            time: 0.0,
            next_body_id: 0,
            contact_cache: FxHashMap::default(),
            contact_cache_pool: Vec::new(),
            contact_touch: Vec::new(),
            collect_contact_events: false,
            dt: 1.0 / 60.0,
            linear_sleep_threshold: LINEAR_SLEEP_THRESHOLD,
            angular_sleep_threshold: ANGULAR_SLEEP_THRESHOLD,
            validation: ValidationConfig::default(),
            validation_report: ValidationReport::default(),
            joints: Vec::new(),
            next_joint_id: 0,
            fluid: super::fluid::FluidState2::new(),
            thermal: super::thermal::ThermalState2::new(),
            nuclear: super::nuclear::NuclearState2::new(),
            #[cfg(feature = "phase-timing")]
            phase_timings: super::phase_timing::PhaseTimings::default(),
            scratch: ScratchBuffers2D::new(),
        }
    }

    /// Add a rigid body to the world. The body must already carry a valid
    /// shape (see [`ShapeError`]); position must be finite.
    pub fn add_body(&mut self, mut body: RigidBody2) -> Result<usize, ShapeError> {
        if !body.transform.position.x.is_finite() || !body.transform.position.y.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        if let Some(shape) = &body.shape {
            validate_shape2(shape)?;
            validate_density(body.is_static, body.material.density)?;
        }
        let id = self.next_body_id;
        body.id = id;
        self.next_body_id += 1;
        self.bodies.push(body);
        Ok(id)
    }

    /// Add a static body (immovable)
    pub fn add_static_body(&mut self, position: Vec2, shape: Shape2) -> Result<usize, ShapeError> {
        if !position.x.is_finite() || !position.y.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        let mut body = RigidBody2::static_body(self.next_body_id);
        body.transform.position = position;
        body.set_shape(shape, 1.0)?;
        self.add_body(body)
    }

    /// Add a dynamic body
    pub fn add_dynamic_body(
        &mut self,
        position: Vec2,
        shape: Shape2,
        density: f32,
    ) -> Result<usize, ShapeError> {
        if !position.x.is_finite() || !position.y.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        let mut body = RigidBody2::new(self.next_body_id);
        body.transform.position = position;
        body.set_shape(shape, density)?;
        self.add_body(body)
    }

    /// Add a joint to the world. The joint must already carry validated
    /// endpoints (see [`JointError`]); ids are assigned here to preserve
    /// the id == index invariant (same contract as [`World2::add_body`]).
    pub fn add_joint(&mut self, mut joint: Joint2) -> Result<usize, JointError> {
        let (a, b) = joint.bodies();
        if a >= self.bodies.len() || b >= self.bodies.len() {
            return Err(JointError::UnknownBody);
        }
        if a == b {
            return Err(JointError::SameBody);
        }
        let id = self.next_joint_id;
        joint.set_id(id);
        self.next_joint_id += 1;
        self.joints.push(joint);
        Ok(id)
    }

    /// Remove a joint by id. Ids double as indices; removal shifts the
    /// tail and reassigns ids to preserve the invariant.
    pub fn remove_joint(&mut self, joint_id: usize) {
        if joint_id >= self.joints.len() {
            return;
        }
        self.joints.remove(joint_id);
        for (index, joint) in self.joints.iter_mut().enumerate() {
            joint.set_id(index);
        }
        self.next_joint_id = self.joints.len();
    }

    /// Remove a body from the world.
    ///
    /// Body ids double as indices into [`World2::bodies`] (see
    /// [`World2::add_body`]), so removal shifts the tail and reassigns
    /// ids to preserve the invariant. Contact-cache keys are remapped
    /// through the same shift (entries touching the removed body are
    /// dropped — those contacts are gone — while survivors keep their
    /// warm-start impulses). Pair keys are canonical `(min, max)`, so the
    /// remap re-canonicalizes after substitution. Joints touching the
    /// removed body are dropped; survivors are remapped through the
    /// shift (see [`Joint2::remap`]).
    pub fn remove_body(&mut self, body_id: usize) {
        if body_id >= self.bodies.len() {
            return;
        }
        let last = self.bodies.len() - 1;
        // Drop joints touching the removed body first (they hold stale ids).
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != body_id && b != body_id
        });
        self.bodies.remove(body_id);
        for (index, body) in self.bodies.iter_mut().enumerate() {
            body.id = index;
        }
        self.next_body_id = self.bodies.len();
        // Shift semantics (Vec::remove): survivor joint endpoints above
        // the gap drop by 1.
        for joint in self.joints.iter_mut() {
            joint.shift(body_id);
        }
        if body_id == last {
            // Tail removed: no index shift; drop only removed pair keys.
            self.contact_cache
                .retain(|&(a, b), _| a != body_id && b != body_id);
        } else {
            // Swap-remove moved `last` into `body_id`: substitute, drop
            // keys touching the removed body, re-canonicalize. The
            // substitution is injective on survivors (the only colliding
            // preimage is the dropped key itself), so no merge can occur;
            // `extend` is belt-and-braces against future key-scheme edits.
            for ((a, b), entries) in std::mem::take(&mut self.contact_cache) {
                if a == body_id || b == body_id {
                    continue;
                }
                let na = if a == last { body_id } else { a };
                let nb = if b == last { body_id } else { b };
                let key = if na < nb { (na, nb) } else { (nb, na) };
                self.contact_cache.entry(key).or_default().extend(entries);
            }
        }
    }

    /// Parallel island solve path (see [`crate::island`]).
    ///
    /// Contact warm-start runs serially in [`World2::step`] before this
    /// call; joint warm-start runs serially here after contact solving
    /// (exact same order as the serial path). Contacts and joints then
    /// solve per island on scoped threads, and control returns to
    /// [`World2::step`] for the shared serial cache write-back.
    /// Single-island (or empty) steps delegate to the exact serial
    /// implementation.
    fn solve_parallel_path(&mut self, dt: f32) {
        // Nothing to solve: skip island building entirely (the serial
        // path runs one empty iteration here — observably identical).
        if self.scratch.constraints.is_empty() && self.joints.is_empty() {
            return;
        }

        // NOTE: contact warm-start already ran in `World2::step` (shared
        // code above this call) — it must run exactly once. Applying
        // cached impulses twice would double-count them and diverge from
        // serial. Only joint warm-start repeats below, matching the
        // serial path's position after `Solver::solve`.

        // The island partition is rebuilt from scratch every step (bodies
        // are added, removed and reindexed), so its buffers are pooled:
        // the steady-state allocation count of this path stays at zero
        // instead of O(bodies). The partition is also built even when it
        // turns out to hold a single island, so pooling matters most
        // exactly there (the work is then handed back to the serial
        // solver, but the build has already happened).
        let mut island_scratch = std::mem::take(&mut self.scratch.islands);
        let mut contact_pairs = std::mem::take(&mut self.scratch.contact_pairs);
        let mut joint_pairs = std::mem::take(&mut self.scratch.joint_pairs);
        // `reserve` before the loop: the buffer is pooled and only grows,
        // so the steady state costs no allocation at all. (`extend` over a
        // mapped iterator grows it one element at a time.)
        contact_pairs.clear();
        contact_pairs.reserve(self.scratch.constraints.len());
        for c in &self.scratch.constraints {
            contact_pairs.push((c.body_a, c.body_b));
        }
        joint_pairs.clear();
        joint_pairs.reserve(self.joints.len());
        for j in &self.joints {
            joint_pairs.push(j.bodies());
        }
        island_scratch.build(self.bodies.len(), &contact_pairs, &joint_pairs);
        let island_count = island_scratch.island_count;

        // The parallel path only pays off when there is enough work to
        // amortise the worker threads: a handful of islands with a
        // handful of constraints each is faster (and allocation-free) on
        // the serial path. The serial path is the exact same computation
        // (the islands are body-disjoint), so the result is unchanged.
        let island_work: usize = island_scratch
            .island_constraints
            .iter()
            .map(|l| l.len())
            .sum();
        let use_parallel = island_count > 1 && island_work >= PARALLEL_MIN_CONSTRAINTS;

        if !use_parallel {
            // Degenerate case: exact serial behavior, no thread overhead.
            self.scratch.islands = island_scratch;
            self.scratch.contact_pairs = contact_pairs;
            self.scratch.joint_pairs = joint_pairs;
            self.solver
                .solve(&mut self.bodies, &mut self.scratch.constraints, dt);
            if !self.joints.is_empty() {
                for joint in self.joints.iter() {
                    joint.warm_start(&mut self.bodies);
                }
                let iters = self.solver.iterations;
                for _ in 0..iters {
                    for joint in self.joints.iter_mut() {
                        joint.solve(&mut self.bodies, dt);
                    }
                }
            }
            return;
        }

        #[cfg(debug_assertions)]
        super::island::debug_verify_disjoint_scratch(&island_scratch, self.bodies.len());

        // Per-island constraint and joint index lists (stable: input
        // order within an island).
        island_scratch.assign_constraints(&contact_pairs);
        island_scratch.assign_joints(&joint_pairs);

        // Joint warm-start stays serial (joints vec order, as serial path).
        for joint in self.joints.iter() {
            joint.warm_start(&mut self.bodies);
        }

        // Phase 1: contacts per island, in parallel. Max scratch is
        // resized (only grows) and reused across steps — no per-step
        // allocation in the steady state. The solver sizes it to two
        // slots per island (double-buffered generations).
        self.scratch.island_max.resize(island_count.max(1), 0.0);
        let mut max_slots = std::mem::take(&mut self.scratch.island_max);
        self.solver.solve_parallel(
            &mut self.bodies,
            &mut self.scratch.constraints,
            &island_scratch.island_bodies,
            &island_scratch.island_constraints,
            &mut max_slots,
            dt,
        );
        self.scratch.island_max = max_slots;

        // Phase 2: joints per island, in parallel (same iteration budget).
        // Addresses travel as `usize` (see `Solver::solve_parallel` for
        // the soundness argument: disjoint island index sets,
        // debug-verified).
        let iters = self.solver.iterations;
        let bodies_addr = self.bodies.as_mut_ptr() as usize;
        let bodies_len = self.bodies.len();
        let joints_addr = self.joints.as_mut_ptr() as usize;
        let joints_len = self.joints.len();
        // Same fixed worker pool as the contact phase: islands are dealt
        // round-robin, so each island's joints are still solved by exactly
        // one worker (and the islands are body-disjoint, so the result is
        // unchanged).
        let island_joints = std::mem::take(&mut island_scratch.island_joints);
        let island_count = island_joints.len();
        let worker_count = super::solver::Solver::worker_count_for(island_count);
        std::thread::scope(|scope| {
            for worker in 0..worker_count {
                let lists = &island_joints;
                scope.spawn(move || {
                    // SAFETY: the island lists hold joint indices
                    // exclusive to their island; joint endpoints lie in
                    // this island's disjoint body set (verified above in
                    // debug). Bodies and joints are addressed at disjoint
                    // indices across threads.
                    let bodies = unsafe {
                        std::slice::from_raw_parts_mut(bodies_addr as *mut RigidBody2, bodies_len)
                    };
                    let joints = unsafe {
                        std::slice::from_raw_parts_mut(joints_addr as *mut Joint2, joints_len)
                    };
                    for _ in 0..iters {
                        for (ii, list) in lists.iter().enumerate() {
                            if ii % worker_count != worker {
                                continue;
                            }
                            for &ji in list.iter() {
                                joints[ji].solve(bodies, dt);
                            }
                        }
                    }
                });
            }
        });

        // Hand the pooled buffers back for the next step.
        island_scratch.island_joints = island_joints;
        self.scratch.islands = island_scratch;
        self.scratch.contact_pairs = contact_pairs;
        self.scratch.joint_pairs = joint_pairs;
    }

    /// Step the physics simulation
    /// Advance the simulation by `dt` seconds. With CCD enabled (see
    /// [`crate::ccd`]), the step splits into substeps so no body travels
    /// more than `max_movement` per substep; each substep runs the full
    /// pipeline. Disabled (default) is a single legacy substep,
    /// bit-identical to pre-CCD behavior.
    pub fn step(&mut self, dt: f32) {
        // Reset validation report once for the whole step; substeps merge.
        self.validation_report = ValidationReport::default();
        let n = if self.ccd.is_active() {
            match self.ccd.tier {
                CcdTier::Off => 1,
                CcdTier::SweptAABB => {
                    if self.ccd_needs_substepping_2d(dt, self.ccd.tier) {
                        self.ccd.substeps(dt, self.ccd_peak_speed())
                    } else {
                        1
                    }
                }
                CcdTier::FullGJK => self.ccd.substeps(dt, self.ccd_peak_speed()),
            }
        } else {
            1
        };
        if n <= 1 {
            self.step_inner(dt);
        } else {
            let sub = dt / n as f32;
            for _ in 0..n {
                self.step_inner(sub);
            }
        }
        // 8. Advance the simulation clock (drives time-varying ambient fields).
        self.time += dt;
    }

    /// Peak effective body speed (linear plus angular sweep at the body
    /// extents) for CCD substepping. Static and sleeping bodies carry
    /// zero velocity, so they contribute nothing.
    fn ccd_peak_speed(&self) -> f32 {
        let mut peak = 0.0f32;
        for body in &self.bodies {
            if body.is_static {
                continue;
            }
            let (min, max) = self.get_body_aabb(body);
            let extent = (max - min).length() * 0.5;
            let eff = body.velocity.length() + body.angular_velocity.abs() * extent;
            peak = peak.max(eff);
        }
        peak
    }

    /// Swept-AABB pair overlap: returns true if any broadphase pair's
    /// swept volumes overlap during `dt`, meaning those bodies could
    /// tunnel and should receive substepping. Bodies whose swept AABBs
    /// never overlap pay no per-substep cost.
    fn ccd_needs_substepping_2d(&mut self, dt: f32, tier: CcdTier) -> bool {
        if tier != CcdTier::SweptAABB {
            return false;
        }
        self.update_broadphase();
        let mut pairs = Vec::new();
        self.broadphase.collect_pairs(&mut pairs);
        for &(a, b) in &pairs {
            if a >= self.bodies.len() || b >= self.bodies.len() {
                continue;
            }
            let body_a = &self.bodies[a];
            let body_b = &self.bodies[b];
            if body_a.is_static || body_b.is_static {
                continue;
            }
            // Collision layers: a pair that may not collide must not
            // drag the whole world into substepping either.
            if !super::body::collision_filter_allows(
                body_a.collision_group,
                body_a.collision_mask,
                body_b.collision_group,
                body_b.collision_mask,
            ) {
                continue;
            }
            let (amin, amax) = self.get_body_aabb(body_a);
            let (bmin, bmax) = self.get_body_aabb(body_b);
            let a_extent = (amax - amin).length() * 0.5;
            let b_extent = (bmax - bmin).length() * 0.5;
            if swept_aabb_overlap(
                body_a.transform.position,
                body_a.velocity,
                body_a.angular_velocity,
                a_extent,
                body_b.transform.position,
                body_b.velocity,
                body_b.angular_velocity,
                b_extent,
                dt,
            ) {
                return true;
            }
        }
        false
    }

    fn step_inner(&mut self, dt: f32) {
        #[cfg(feature = "phase-timing")]
        let mut timings = super::phase_timing::PhaseTimings::default();

        // 0. Resolve a Hybrid broadphase request before any broadphase
        // work, so the backend in use for THIS step is decided by this
        // step's body count (not last step's).
        self.resolve_broadphase_mode();

        // 1. Gravity kick (see `crate::gravity`): uniform background field
        // plus Central/N-body terms per `gravity_config`, one full-step
        // kick (Euler) or the first half-kick (Verlet, with linear
        // damping riding as symmetric drag).
        let verlet = self.gravity_config.integrator == super::gravity::Integrator::VelocityVerlet;
        let h = if verlet { dt * 0.5 } else { dt };
        timed!(timings.gravity, {
            super::gravity::gravity_kick_2(
                &mut self.bodies,
                &self.gravity_config,
                self.gravity,
                h,
                verlet,
            );

            // 1a. Ambient forces (spring/SHM, air drag, turbulence). Same slot
            // and same Verlet split as the gravity kick above: a pure velocity
            // kick pre-solve. A no-op when no sub-field is enabled.
            if self.ambient.is_active() {
                super::ambient::ambient_kick_2(&mut self.bodies, &self.ambient, self.time, h);
            }
        });

        // 1b. Fluid phase - SPH particles and two-way body coupling. Runs
        // before broadphase so the rigid pipeline observes post-fluid
        // velocities; a no-op when the fluid holds no particles.
        timed!(timings.fluid, {
            self.fluid
                .step_fluid(&mut self.bodies, self.gravity, &self.ambient, self.time, dt);
        });

        // 2. Broad phase - find potential collision pairs (into scratch).
        // Reset scratch buffers (keeps capacity, including parallel slots).
        timed!(timings.broadphase_aabb, {
            self.scratch.reset();
            self.compute_body_aabbs();
        });
        timed!(timings.broadphase_rebuild, {
            self.broadphase.begin_frame();
            self.insert_body_aabbs();
            // The grid keeps its cells across steps, so the bodies the
            // world no longer has have to be taken out of them.
            self.broadphase.end_frame();
        });
        timed!(timings.broadphase_pairs, {
            // Narrowphase fast path: only AABB-overlapping pairs can
            // produce contacts. The CCD swept test below keeps the
            // unfiltered sequence on purpose.
            self.broadphase
                .collect_overlapping_pairs(&mut self.scratch.pairs);
        });

        // 3. Narrow phase - test actual collisions (into scratch).
        timed!(timings.narrowphase, {
            // The tier gate is a runtime value, so it is resolved once
            // here and the pair loop below calls the narrow phase once.
            let (max_tier, tier_enabled) = if self.collision_tier.is_active() {
                (self.collision_tier.max_tier, true)
            } else {
                (super::collision_tier::CollisionTier::Tier5_GJK_EPA, false)
            };
            for &(a, b) in &self.scratch.pairs {
                // Static-static pairs can never produce an impulse (zero
                // effective mass in every direction), so skip the narrowphase
                // for them entirely. Bit-identical: their constraints would
                // solve to zero and warm-start zeros are no-ops.
                if self.bodies[a].is_static && self.bodies[b].is_static {
                    continue;
                }
                // Collision layers: each body must accept the other's
                // layer. The default filter accepts everything, so an
                // unconfigured scene is unaffected.
                if !super::body::collision_filter_allows(
                    self.bodies[a].collision_group,
                    self.bodies[a].collision_mask,
                    self.bodies[b].collision_group,
                    self.bodies[b].collision_mask,
                ) {
                    continue;
                }
                // A sensor reports contacts but never pushes: its
                // contacts go to the sensor list, not to the solver.
                if self.bodies[a].is_sensor || self.bodies[b].is_sensor {
                    Self::test_collisions_into(
                        &self.bodies,
                        a,
                        b,
                        &mut self.scratch.sensor_contacts,
                        &mut self.scratch.verts_a,
                        &mut self.scratch.verts_b,
                        &mut self.scratch.points,
                        &mut self.scratch.clip_tmp,
                        max_tier,
                        tier_enabled,
                    );
                } else {
                    Self::test_collisions_into(
                        &self.bodies,
                        a,
                        b,
                        &mut self.scratch.constraints,
                        &mut self.scratch.verts_a,
                        &mut self.scratch.verts_b,
                        &mut self.scratch.points,
                        &mut self.scratch.clip_tmp,
                        max_tier,
                        tier_enabled,
                    );
                }
            }
        });

        // 3a. A sensor's contacts are events, not contacts: the pair
        // set is compared with last step's, and a pair that is new (or
        // gone) produces an event. The sensor pairs are in the order the
        // broad phase produced them, which is the sorted pair order, so
        // the two lists merge. The diff runs when the sensor has
        // contacts *or* had them last step — a sensor that just left
        // something has no contacts now, and that is exactly when its
        // "ended" event is due.
        if self.collect_contact_events
            && (!self.scratch.sensor_contacts.is_empty()
                || !self.scratch.last_sensor_pairs.is_empty())
        {
            self.sensor_events_2d();
        }

        // 3b. Warm-start: seed each constraint with cached impulses from the
        // previous frame (matched by contact-point proximity + normal agreement).
        // A pair with no cache entry is a NEW contact: any sleeping party
        // wakes (with cascade) so it can respond to the newcomer. Settled
        // pairs stay cached and keep sleeping. Every constrained body is
        // marked touched (sleep-support tracking for step 6+7).
        timed!(timings.cache, {
            if self.contact_touch.len() != self.bodies.len() {
                self.contact_touch.resize(self.bodies.len(), false);
            }
            for t in self.contact_touch.iter_mut() {
                *t = false;
            }
            // Index loop (not iter_mut): `wake_body` needs `&mut self`
            // while constraints borrow `self.scratch`.
            #[allow(clippy::needless_range_loop)]
            for i in 0..self.scratch.constraints.len() {
                let (a, b) = {
                    let c = &self.scratch.constraints[i];
                    (c.body_a, c.body_b)
                };
                self.contact_touch[a] = true;
                self.contact_touch[b] = true;
                // One hash lookup, not two: the same lookup tells us
                // whether the pair is new (a new pair may need its
                // sleepers woken) and which cached contact to match.
                let mut matched: Option<(f32, f32)> = None;
                let mut is_new = true;
                if let Some(entries) = self.contact_cache.get(&(a, b)) {
                    is_new = false;
                    let constraint = &self.scratch.constraints[i];
                    let mut best: Option<&CachedContact> = None;
                    let mut best_dist = CONTACT_MATCH_DIST_SQ;
                    for entry in entries {
                        // Skip entries whose contact flipped (impulses would push the wrong way).
                        if entry.normal.dot(constraint.normal) <= 0.0 {
                            continue;
                        }
                        let d = (entry.point - constraint.point).length_squared();
                        if d < best_dist {
                            best_dist = d;
                            best = Some(entry);
                        }
                    }
                    if let Some(entry) = best {
                        matched = Some((entry.normal_impulse, entry.tangent_impulse));
                    }
                }
                // A pair that was not touching last step began touching
                // this step: that is the event a trigger wants.
                if is_new && self.collect_contact_events {
                    let c = &self.scratch.constraints[i];
                    let (ea, eb) = event_pair(a, b);
                    self.scratch.events.push(ContactEvent2 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Began,
                        point: c.point,
                        normal: c.normal,
                    });
                }
                // A new pair with a sleeping party wakes it (with a
                // cascade) so it can respond to the newcomer. The wake
                // cannot change what the match above found: it only
                // clears sleep flags, and the cache holds the previous
                // frame's contacts either way.
                if is_new && (self.bodies[a].is_sleeping || self.bodies[b].is_sleeping) {
                    self.wake_body(a);
                    self.wake_body(b);
                }
                if let Some((normal_impulse, tangent_impulse)) = matched {
                    let constraint = &mut self.scratch.constraints[i];
                    constraint.normal_impulse = normal_impulse;
                    constraint.tangent_impulse = tangent_impulse;
                }
            }

            // Validate constraints before solving
            let constraint_report = validate_constraints_2d(
                &self.bodies,
                &mut self.scratch.constraints,
                &self.validation,
            );
            self.validation_report.merge(constraint_report);
        });

        // 4. Solve contacts, then joints (after contacts, same iteration
        // budget). The parallel solver path is used only when requested;
        // it falls back to the exact serial implementation for 0-1 islands.
        timed!(timings.solver, {
            if self.solver.parallel {
                self.solve_parallel_path(dt);
            } else {
                self.solver
                    .solve(&mut self.bodies, &mut self.scratch.constraints, dt);
                if !self.joints.is_empty() {
                    for joint in self.joints.iter() {
                        joint.warm_start(&mut self.bodies);
                    }
                    let iters = self.solver.iterations;
                    for _ in 0..iters {
                        for joint in self.joints.iter_mut() {
                            joint.solve(&mut self.bodies, dt);
                        }
                    }
                }
            }

            // 4b. Write back accumulated impulses for next frame's warm start.
            // Disjoint field borrows: the cache map and its buffer pool.
            let cache = &mut self.contact_cache;
            let pool = &mut self.contact_cache_pool;
            // Remember each pair's last known contact before the lists
            // are cleared, so an *ended* event can still say where the
            // pair was.
            if self.collect_contact_events {
                self.scratch.last_contacts.clear();
                for (key, list) in cache.iter() {
                    if let Some(first) = list.first() {
                        self.scratch
                            .last_contacts
                            .push((key.0, key.1, first.point, first.normal));
                    }
                }
            }
            for list in cache.values_mut() {
                list.clear();
            }
            for constraint in &self.scratch.constraints {
                let key = (constraint.body_a, constraint.body_b);
                cache
                    .entry(key)
                    .or_insert_with(|| pool.pop().unwrap_or_default())
                    .push(CachedContact {
                        point: constraint.point,
                        normal: constraint.normal,
                        normal_impulse: constraint.normal_impulse,
                        tangent_impulse: constraint.tangent_impulse,
                    });
            }
            // Drop vacated entries every frame (so `contact_cache.len()` keeps
            // its exact meaning — number of live pairs) but RECYCLE their
            // buffers into the pool instead of freeing them. New keys draw
            // from the pool, so the cache reaches zero steady-state allocation
            // even as pairs flicker. Memory-only; trajectories bit-identical.
            // A vanished pair is lost contact: wake any sleeping party (loss
            // of support must re-evaluate sleep). Idempotent flags, so the
            // drain order cannot affect determinism.
            let cache = &mut self.contact_cache;
            let pool = &mut self.contact_cache_pool;
            let stale = &mut self.scratch.stale;
            stale.clear();
            cache.retain(|key, list| {
                if list.is_empty() {
                    stale.push(*key);
                    pool.push(std::mem::take(list));
                    false
                } else {
                    true
                }
            });
            // A pair that was touching last step and is not touching
            // now stopped touching: that is the other half of the
            // event. The contact data is the pair's last known one.
            if self.collect_contact_events {
                for &(a, b) in self.scratch.stale.iter() {
                    let (point, normal) = self
                        .scratch
                        .last_contacts
                        .iter()
                        .find(|&&(ka, kb, _, _)| ka == a && kb == b)
                        .map(|&(_, _, p, n)| (p, n))
                        .unwrap_or((Vec2::ZERO, Vec2::ZERO));
                    let (ea, eb) = event_pair(a, b);
                    self.scratch.events.push(ContactEvent2 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Ended,
                        point,
                        normal,
                    });
                }
            }
            // Pop from the end (LIFO): the woken set is order-independent.
            while let Some((a, b)) = self.scratch.stale.pop() {
                if a < self.bodies.len()
                    && b < self.bodies.len()
                    && (self.bodies[a].is_sleeping || self.bodies[b].is_sleeping)
                {
                    self.wake_body(a);
                    self.wake_body(b);
                }
            }
        });

        timed!(timings.integrate, {
            if verlet {
                // 5v. Verlet drift (positions only — no linear damping) plus
                // the second half-kick with forces recomputed at the new
                // positions. The contact solver above ran unchanged.
                // Sleeping bodies are skipped: frozen until woken.
                for body in self.bodies.iter_mut() {
                    if body.is_sleeping {
                        continue;
                    }
                    body.integrate_drift(dt);
                }
                super::gravity::gravity_kick_2(
                    &mut self.bodies,
                    &self.gravity_config,
                    self.gravity,
                    dt * 0.5,
                    true,
                );
                if self.ambient.is_active() {
                    super::ambient::ambient_kick_2(
                        &mut self.bodies,
                        &self.ambient,
                        self.time,
                        dt * 0.5,
                    );
                }
            } else {
                // 5. Integrate motion (sleeping bodies are frozen until woken).
                for body in self.bodies.iter_mut() {
                    if body.is_sleeping {
                        continue;
                    }
                    body.integrate(dt);
                }
            }
        });

        // 5c. Nuclear decay and fission: fills a per-body heat budget
        // (and an ionizing-radiation budget) from the nuclides the
        // bodies carry. Runs before the thermal phase, so the thermal
        // radiation sees the heat in the same step. It modifies
        // temperature only (no velocity change), so it neither wakes nor
        // is gated by sleep.
        timed!(timings.thermal, {
            if self.nuclear.config.is_active() {
                let nuclear_config = &self.nuclear.config;
                let nuclear_scratch = &mut self.nuclear.scratch;
                super::nuclear::step_nuclear_2d(
                    &mut self.bodies,
                    nuclear_config,
                    nuclear_scratch,
                    dt,
                );
                // 5c-2. The nuclear heat raises the body temperatures. The
                // clamps come from the thermal configuration, so the bounds
                // are the same ones the thermal phase enforces; the rate
                // clamp is deliberately not applied, because a nuclear
                // source has a physical rate.
                let thermal_config = self.thermal.config;
                super::nuclear::apply_nuclear_heat_2d(
                    &mut self.bodies,
                    &self.nuclear.scratch,
                    &thermal_config,
                );
            }

            // 5d. Thermal transfer: conduction, radiation, convection. Runs
            // after integrate so it sees final positions; modifies temperature
            // only (no velocity change), so it neither wakes nor is gated by
            // sleep. Reads contact data from the narrowphase pass above.
            let thermal_config = self.thermal.config;
            super::thermal::step_thermal_2d(
                &mut self.bodies,
                &self.scratch.constraints,
                &thermal_config,
                &mut self.thermal,
                dt,
            );
            // 5d. Thermal expansion: rescale shapes by their expansion factor
            // and refresh mass/inertia. No-op unless the coefficient is
            // non-zero (the default), so trajectories stay bit-identical.
            super::thermal::apply_expansion_2d(&mut self.bodies, &thermal_config);
        });

        // 6+7. Validate bodies + update sleep in a single fused sweep.
        // Sleep needs slowness AND support (a current contact, tracked
        // in `contact_touch` during narrowphase): ballistic apexes and
        // drifting bodies never sleep, so no wake/sleep history is
        // needed and save/load stays exact. Sleeping bodies have zeroed
        // velocities every step: the solver may write reaction impulses
        // to a frozen body during the step, and stale velocity would
        // poison the next step's relative velocity (support going soft)
        // since integrate skips sleepers.
        let lin = self.linear_sleep_threshold;
        let ang = self.angular_sleep_threshold;
        timed!(timings.sleep, {
            for (i, body) in self.bodies.iter_mut().enumerate() {
                let report = validate_body_2d(body, &self.validation);
                self.validation_report.merge(report);
                if body.is_static {
                    continue;
                }
                if body.is_sleeping {
                    body.velocity = Vec2::ZERO;
                    body.angular_velocity = 0.0;
                    continue;
                }
                if body.should_sleep(lin, ang) && self.contact_touch[i] {
                    body.is_sleeping = true;
                    body.velocity = Vec2::ZERO;
                    body.angular_velocity = 0.0;
                }
            }
        });

        #[cfg(feature = "phase-timing")]
        {
            timings.steps = 1;
            self.phase_timings.merge(timings);
        }
    }

    /// The per-phase wall-clock accumulators.
    ///
    /// Only available with the `phase-timing` feature; the default build
    /// has no timers and no accumulator, and its public API is unchanged.
    #[cfg(feature = "phase-timing")]
    pub fn phase_timings(&self) -> &super::phase_timing::PhaseTimings {
        &self.phase_timings
    }

    /// Forget everything measured so far.
    #[cfg(feature = "phase-timing")]
    pub fn reset_phase_timings(&mut self) {
        self.phase_timings.reset();
    }

    /// How many candidate pairs the broad phase produced on the last step.
    ///
    /// This is the *raw* count, before the narrow phase filters it: the
    /// difference between this and the contact-cache size is the work the
    /// narrow phase throws away.
    #[cfg(feature = "phase-timing")]
    pub fn broadphase_pair_count(&self) -> usize {
        self.scratch.pairs.len()
    }

    /// How many cells the broad-phase grid currently holds.
    #[cfg(feature = "phase-timing")]
    pub fn broadphase_cell_count(&self) -> usize {
        self.broadphase.cell_count()
    }

    /// Profiling only: turn the grid's persistent cells on or off.
    ///
    /// The persistent grid keeps its cells across steps and only
    /// re-inserts the bodies that moved; the other behaviour wipes
    /// every cell and re-inserts every body each step. The pair set is
    /// the same either way, so the trajectory does not change; only the
    /// amount of work does.
    #[cfg(feature = "phase-timing")]
    pub fn set_broadphase_persistent(&mut self, persistent: bool) {
        self.broadphase.set_persistent(persistent);
    }

    /// Profiling only: whether the persistent grid only skips a body
    /// whose AABB is bit-identical.
    ///
    /// The alternative skips any body whose cell range is unchanged,
    /// which is a pure integer comparison: the cells a body occupies are
    /// exactly the cells of its range, so an unchanged range means an
    /// unchanged membership. Both produce the same pairs.
    #[cfg(feature = "phase-timing")]
    pub fn set_broadphase_persistent_strict(&mut self, strict: bool) {
        self.broadphase.set_persistent_strict(strict);
    }

    /// Profiling only: turn the grid's pair ownership rule on or off.
    ///
    /// The rule makes the grid generate each pair once, in the lowest
    /// cell the two bodies share, instead of once per shared cell. The
    /// pair set is the same either way, so the trajectory does not
    /// change; only the amount of duplicated candidate work does.
    #[cfg(feature = "phase-timing")]
    pub fn set_broadphase_owned_pairs(&mut self, owned: bool) {
        self.broadphase.set_owned_pairs(owned);
    }

    /// Profiling only: turn the grid's static-cell skip on or off.
    ///
    /// The skip drops a cell that holds nothing but static bodies
    /// without walking its candidate pairs. The pair set is the same
    /// either way, so the trajectory does not change; only the amount
    /// of candidate work does.
    #[cfg(feature = "phase-timing")]
    pub fn set_broadphase_skip_static_cells(&mut self, skip: bool) {
        self.broadphase.set_skip_static_cells(skip);
    }

    /// Cast a ray into the world and return the first thing it hits.
    ///
    /// The ray is tested against every body's current shape; the body
    /// that is hit first (the smallest `t`) wins. A body the filter does
    /// not accept is not considered. A ray that starts inside a shape
    /// does not hit it.
    ///
    /// The scan is over the *current* body positions, not the broad
    /// phase's grid, which holds the positions from before the last
    /// solve.
    pub fn raycast(&self, ray: &Ray2, filter: &QueryFilter) -> Option<WorldRayHit2> {
        let mut best: Option<WorldRayHit2> = None;
        for body in &self.bodies {
            if !filter.accepts(body.collision_group, body.collision_mask) {
                continue;
            }
            let Some(shape) = body.shape.as_ref() else {
                continue;
            };
            let hit = match shape {
                Shape2::Circle { radius } => {
                    ray_circle2(ray.origin, ray.direction, body.transform.position, *radius)
                }
                Shape2::Box { half_extents } => ray_obb2(
                    ray.origin,
                    ray.direction,
                    body.transform.position,
                    body.transform.rotation,
                    *half_extents,
                ),
                Shape2::Polygon { vertices } => {
                    // The polygon in world space.
                    let world: Vec<Vec2> = vertices
                        .iter()
                        .map(|v| body.transform.transform_point(*v))
                        .collect();
                    ray_polygon2(ray.origin, ray.direction, &world)
                }
            };
            let Some(hit) = hit else { continue };
            let nearer = match best {
                Some(current) => hit.t < current.t,
                None => true,
            };
            if nearer {
                best = Some(WorldRayHit2 {
                    body: body.id as u32,
                    t: hit.t,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        best
    }

    /// The contact events of the last step.
    ///
    /// One event per pair that began or stopped touching. The events are
    /// only collected when [`Self::collect_contact_events`] is on, and
    /// they are cleared at the start of the next step.
    pub fn contact_events(&self) -> &[ContactEvent2] {
        &self.scratch.events
    }

    /// Work out which sensor pairs began and which ended.
    ///
    /// The sensor pairs of this step are in the order the broad phase
    /// produced them -- the sorted pair order -- and so are last step's,
    /// so the two lists merge instead of being searched.
    fn sensor_events_2d(&mut self) {
        let scratch = &mut self.scratch;

        // The pairs touching a sensor now, in order and without repeats.
        scratch.sensor_pairs.clear();
        for c in &scratch.sensor_contacts {
            let key = (c.body_a, c.body_b);
            if scratch.sensor_pairs.last() != Some(&key) {
                scratch.sensor_pairs.push(key);
            }
        }

        // Merge the two sorted lists and report what changed.
        let mut i = 0;
        let mut j = 0;
        let mut ended: Vec<(usize, usize)> = Vec::new();
        while i < scratch.sensor_pairs.len() || j < scratch.last_sensor_pairs.len() {
            let now = scratch.sensor_pairs.get(i).copied();
            let before = scratch.last_sensor_pairs.get(j).copied();
            match (now, before) {
                (Some(c), Some(p)) if c == p => {
                    i += 1;
                    j += 1;
                }
                (Some(c), Some(p)) => {
                    if c < p {
                        let (point, normal) = sensor_contact_2d(&scratch.sensor_contacts, c);
                        let (ea, eb) = event_pair(c.0, c.1);
                        scratch.events.push(ContactEvent2 {
                            body_a: ea,
                            body_b: eb,
                            kind: super::body::ContactEventKind::Began,
                            point,
                            normal,
                        });
                        i += 1;
                    } else {
                        ended.push(p);
                        j += 1;
                    }
                }
                (Some(c), None) => {
                    let (point, normal) = sensor_contact_2d(&scratch.sensor_contacts, c);
                    let (ea, eb) = event_pair(c.0, c.1);
                    scratch.events.push(ContactEvent2 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Began,
                        point,
                        normal,
                    });
                    i += 1;
                }
                (None, Some(p)) => {
                    ended.push(p);
                    j += 1;
                }
                (None, None) => break,
            }
        }

        // The pairs that ended keep the last contact they had.
        for key in ended {
            let (point, normal) = sensor_contact_2d(&scratch.sensor_contacts, key);
            let (ea, eb) = event_pair(key.0, key.1);
            scratch.events.push(ContactEvent2 {
                body_a: ea,
                body_b: eb,
                kind: super::body::ContactEventKind::Ended,
                point,
                normal,
            });
        }

        // This step's sensor pairs are next step's.
        std::mem::swap(&mut scratch.sensor_pairs, &mut scratch.last_sensor_pairs);
    }

    /// Wake a body and cascade through its contact partners and joint
    /// partners, transitively. The woken set is a fixpoint over
    /// idempotent flags, so map iteration order cannot affect the
    /// result (deterministic). Out-of-range ids are ignored.
    pub fn wake_body(&mut self, body_id: usize) {
        if body_id >= self.bodies.len() {
            return;
        }
        self.bodies[body_id].is_sleeping = false;
        let mut dirty = true;
        while dirty {
            dirty = false;
            for (a, b) in self.contact_cache.keys() {
                let (a, b) = (*a, *b);
                if a < self.bodies.len() && b < self.bodies.len() {
                    if !self.bodies[a].is_sleeping && self.bodies[b].is_sleeping {
                        self.bodies[b].is_sleeping = false;
                        dirty = true;
                    } else if !self.bodies[b].is_sleeping && self.bodies[a].is_sleeping {
                        self.bodies[a].is_sleeping = false;
                        dirty = true;
                    }
                }
            }
            for joint in &self.joints {
                let (a, b) = joint.bodies();
                if a < self.bodies.len() && b < self.bodies.len() {
                    if !self.bodies[a].is_sleeping && self.bodies[b].is_sleeping {
                        self.bodies[b].is_sleeping = false;
                        dirty = true;
                    } else if !self.bodies[b].is_sleeping && self.bodies[a].is_sleeping {
                        self.bodies[a].is_sleeping = false;
                        dirty = true;
                    }
                }
            }
        }
    }

    /// Resolve [`BroadphaseMode::Hybrid`] to a concrete backend from the
    /// live body count.
    ///
    /// Called at the top of every step so the resolved backend is in place
    /// *before* any broadphase work happens. Doing it inside
    /// [`Self::update_broadphase`] would leave the first step running on
    /// whatever backend the world was constructed with, making Hybrid
    /// subtly order-dependent (it resolved a step late).
    fn resolve_broadphase_mode(&mut self) {
        if self.broadphase_mode == BroadphaseMode::Hybrid {
            let resolved = BroadphaseMode::auto_select(self.bodies.len());
            self.broadphase = Broadphase2::new(resolved, self.broadphase.cell_size());
        }
    }

    /// Compute every body's AABB into the pooled scratch buffer.
    fn compute_body_aabbs(&mut self) {
        // The buffer is moved out first so the AABB computation can
        // borrow the world immutably while it fills it.
        let mut aabbs = std::mem::take(&mut self.scratch.aabbs);
        aabbs.clear();
        for body in &self.bodies {
            let (min, max) = self.get_body_aabb(body);
            aabbs.push((min, max));
        }
        self.scratch.aabbs = aabbs;
    }

    /// Insert every body into the broad phase from the AABB buffer.
    fn insert_body_aabbs(&mut self) {
        let aabbs = &self.scratch.aabbs;
        for (body, (min, max)) in self.bodies.iter().zip(aabbs.iter()) {
            if body.is_static {
                // A pair of two statics can never produce an impulse, so
                // the broad phase does not generate it.
                self.broadphase.insert_static(body.id, *min, *max);
            } else {
                self.broadphase.insert(body.id, *min, *max);
            }
        }
    }

    /// Update the broadphase with current body positions (2D)
    fn update_broadphase(&mut self) {
        self.broadphase.begin_frame();
        self.compute_body_aabbs();
        self.insert_body_aabbs();
        // The grid keeps its cells across steps, so the bodies the world
        // no longer has have to be taken out of them.
        self.broadphase.end_frame();
    }

    /// The bodies whose AABB overlaps `aabb`.
    ///
    /// The ids come back in ascending order, and a body is reported
    /// only when the filter accepts it (see [`QueryFilter`]).
    ///
    /// The scan is over the *current* body positions, not over the
    /// broad phase: the grid is rebuilt at the start of the step, so
    /// between steps it holds the positions from before the solve,
    /// and a query must not report against those.
    pub fn overlap_aabb(&self, aabb: &AABB2, filter: &QueryFilter) -> Vec<u32> {
        let mut out = Vec::new();
        for body in &self.bodies {
            if !filter.accepts(body.collision_group, body.collision_mask) {
                continue;
            }
            let (min, max) = self.get_body_aabb(body);
            if min.x <= aabb.max.x
                && max.x >= aabb.min.x
                && min.y <= aabb.max.y
                && max.y >= aabb.min.y
            {
                out.push(body.id as u32);
            }
        }
        out
    }

    /// Get AABB for a 2D body
    fn get_body_aabb(&self, body: &RigidBody2) -> (Vec2, Vec2) {
        match &body.shape {
            Some(Shape2::Circle { radius }) => {
                let r = *radius;
                (
                    body.transform.position - Vec2::new(r, r),
                    body.transform.position + Vec2::new(r, r),
                )
            }
            Some(Shape2::Box { half_extents }) => {
                let he = *half_extents;
                let corners = [
                    Vec2::new(-he.x, -he.y),
                    Vec2::new(he.x, -he.y),
                    Vec2::new(he.x, he.y),
                    Vec2::new(-he.x, he.y),
                ];
                let mut min = body.transform.transform_point(corners[0]);
                let mut max = min;
                for &corner in &corners[1..] {
                    let transformed = body.transform.transform_point(corner);
                    min = min.min(transformed);
                    max = max.max(transformed);
                }
                (min, max)
            }
            Some(Shape2::Polygon { vertices }) => {
                if vertices.is_empty() {
                    return (body.transform.position, body.transform.position);
                }
                let mut min = body.transform.transform_point(vertices[0]);
                let mut max = min;
                for &vertex in &vertices[1..] {
                    let transformed = body.transform.transform_point(vertex);
                    min = min.min(transformed);
                    max = max.max(transformed);
                }
                (min, max)
            }
            None => (body.transform.position, body.transform.position),
        }
    }

    /// Test collision between two 2D bodies, appending constraints to `out`.
    #[allow(clippy::too_many_arguments)]
    fn test_collisions_into(
        bodies: &[RigidBody2],
        a: usize,
        b: usize,
        out: &mut Vec<ContactConstraint>,
        verts_a: &mut Vec<Vec2>,
        verts_b: &mut Vec<Vec2>,
        points: &mut Vec<Vec2>,
        clip_tmp: &mut Vec<Vec2>,
        max_tier: super::collision_tier::CollisionTier,
        tier_enabled: bool,
    ) {
        use phieng_collision::sat::{
            polygon_polygon_contact_points_into, sat_circle_polygon, sat_polygon_polygon,
        };

        let body_a = &bodies[a];
        let body_b = &bodies[b];

        // Tier gate: if the required tier for this shape pair exceeds the
        // configured ceiling, skip the pair (treated as non-colliding).
        if tier_enabled {
            let required = super::collision_tier::required_tier_2d(
                body_a.shape.as_ref().unwrap(),
                body_b.shape.as_ref().unwrap(),
            );
            if required > max_tier {
                return;
            }
        }

        let friction =
            PhysicsMaterial::combine_friction(body_a.material.friction, body_b.material.friction);
        let restitution = PhysicsMaterial::combine_restitution(
            body_a.material.restitution,
            body_b.material.restitution,
        );

        match (&body_a.shape, &body_b.shape) {
            (Some(Shape2::Circle { radius: ra }), Some(Shape2::Circle { radius: rb })) => {
                let delta = body_b.transform.position - body_a.transform.position;
                let dist = delta.length();
                let depth = ra + rb - dist;
                if depth < 0.0 {
                    return;
                }
                let normal = if dist > EPS { delta / dist } else { Vec2::X };
                let point = body_a.transform.position + normal * *ra;
                out.push(ContactConstraint {
                    body_a: a,
                    body_b: b,
                    point,
                    normal,
                    penetration: depth,
                    friction,
                    restitution,
                    normal_impulse: 0.0,
                    tangent_impulse: 0.0,
                    rest_bias: 0.0,
                    mass_normal: 0.0,
                    mass_tangent: 0.0,
                });
            }
            (Some(Shape2::Circle { radius: r }), Some(Shape2::Box { half_extents: he })) => {
                verts_b.clear();
                let he = *he;
                let box_verts = [
                    Vec2::new(-he.x, -he.y),
                    Vec2::new(he.x, -he.y),
                    Vec2::new(he.x, he.y),
                    Vec2::new(-he.x, he.y),
                ];
                for &v in &box_verts {
                    verts_b.push(body_b.transform.transform_point(v));
                }
                let res = sat_circle_polygon(body_a.transform.position, *r, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        body_a.transform.position,
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    let point = body_a.transform.position + normal * *r;
                    out.push(ContactConstraint {
                        body_a: a,
                        body_b: b,
                        point,
                        normal,
                        penetration: depth,
                        friction,
                        restitution,
                        normal_impulse: 0.0,
                        tangent_impulse: 0.0,
                        rest_bias: 0.0,
                        mass_normal: 0.0,
                        mass_tangent: 0.0,
                    });
                }
            }
            (Some(Shape2::Box { half_extents: he }), Some(Shape2::Circle { radius: r })) => {
                verts_a.clear();
                let he = *he;
                let box_verts = [
                    Vec2::new(-he.x, -he.y),
                    Vec2::new(he.x, -he.y),
                    Vec2::new(he.x, he.y),
                    Vec2::new(-he.x, he.y),
                ];
                for &v in &box_verts {
                    verts_a.push(body_a.transform.transform_point(v));
                }
                let res = sat_circle_polygon(body_b.transform.position, *r, verts_a);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        body_b.transform.position,
                        phieng_collision::sat::polygon_centroid(verts_a),
                    );
                    let point = body_b.transform.position + normal * *r;
                    out.push(ContactConstraint {
                        body_a: a,
                        body_b: b,
                        point,
                        normal: -normal,
                        penetration: depth,
                        friction,
                        restitution,
                        normal_impulse: 0.0,
                        tangent_impulse: 0.0,
                        rest_bias: 0.0,
                        mass_normal: 0.0,
                        mass_tangent: 0.0,
                    });
                }
            }
            (Some(Shape2::Box { half_extents: hea }), Some(Shape2::Box { half_extents: heb })) => {
                verts_a.clear();
                verts_b.clear();
                let hea = *hea;
                let heb = *heb;
                let box_a_verts = [
                    Vec2::new(-hea.x, -hea.y),
                    Vec2::new(hea.x, -hea.y),
                    Vec2::new(hea.x, hea.y),
                    Vec2::new(-hea.x, hea.y),
                ];
                let box_b_verts = [
                    Vec2::new(-heb.x, -heb.y),
                    Vec2::new(heb.x, -heb.y),
                    Vec2::new(heb.x, heb.y),
                    Vec2::new(-heb.x, heb.y),
                ];
                for &v in &box_a_verts {
                    verts_a.push(body_a.transform.transform_point(v));
                }
                for &v in &box_b_verts {
                    verts_b.push(body_b.transform.transform_point(v));
                }
                let res = sat_polygon_polygon(verts_a, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        phieng_collision::sat::polygon_centroid(verts_a),
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    points.clear();
                    polygon_polygon_contact_points_into(verts_a, verts_b, normal, points, clip_tmp);
                    if points.is_empty() {
                        polygon_polygon_contact_points_into(
                            verts_b, verts_a, -normal, points, clip_tmp,
                        );
                    }
                    for &point in points.iter().take(2) {
                        out.push(ContactConstraint {
                            body_a: a,
                            body_b: b,
                            point,
                            normal,
                            penetration: depth,
                            friction,
                            restitution,
                            normal_impulse: 0.0,
                            tangent_impulse: 0.0,
                            rest_bias: 0.0,
                            mass_normal: 0.0,
                            mass_tangent: 0.0,
                        });
                    }
                }
            }
            (Some(Shape2::Polygon { vertices: va }), Some(Shape2::Polygon { vertices: vb })) => {
                verts_a.clear();
                verts_b.clear();
                for &v in va {
                    verts_a.push(body_a.transform.transform_point(v));
                }
                for &v in vb {
                    verts_b.push(body_b.transform.transform_point(v));
                }
                let res = sat_polygon_polygon(verts_a, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        phieng_collision::sat::polygon_centroid(verts_a),
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    points.clear();
                    polygon_polygon_contact_points_into(verts_a, verts_b, normal, points, clip_tmp);
                    if points.is_empty() {
                        polygon_polygon_contact_points_into(
                            verts_b, verts_a, -normal, points, clip_tmp,
                        );
                    }
                    for &point in points.iter().take(2) {
                        out.push(ContactConstraint {
                            body_a: a,
                            body_b: b,
                            point,
                            normal,
                            penetration: depth,
                            friction,
                            restitution,
                            normal_impulse: 0.0,
                            tangent_impulse: 0.0,
                            rest_bias: 0.0,
                            mass_normal: 0.0,
                            mass_tangent: 0.0,
                        });
                    }
                }
            }
            (Some(Shape2::Circle { radius: r }), Some(Shape2::Polygon { vertices: v })) => {
                verts_b.clear();
                for &vert in v {
                    verts_b.push(body_b.transform.transform_point(vert));
                }
                let res = sat_circle_polygon(body_a.transform.position, *r, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        body_a.transform.position,
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    let point = body_a.transform.position + normal * *r;
                    out.push(ContactConstraint {
                        body_a: a,
                        body_b: b,
                        point,
                        normal,
                        penetration: depth,
                        friction,
                        restitution,
                        normal_impulse: 0.0,
                        tangent_impulse: 0.0,
                        rest_bias: 0.0,
                        mass_normal: 0.0,
                        mass_tangent: 0.0,
                    });
                }
            }
            (Some(Shape2::Polygon { vertices: v }), Some(Shape2::Circle { radius: r })) => {
                verts_a.clear();
                for &vert in v {
                    verts_a.push(body_a.transform.transform_point(vert));
                }
                let res = sat_circle_polygon(body_b.transform.position, *r, verts_a);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        body_b.transform.position,
                        phieng_collision::sat::polygon_centroid(verts_a),
                    );
                    let point = body_b.transform.position + normal * *r;
                    out.push(ContactConstraint {
                        body_a: a,
                        body_b: b,
                        point,
                        normal: -normal,
                        penetration: depth,
                        friction,
                        restitution,
                        normal_impulse: 0.0,
                        tangent_impulse: 0.0,
                        rest_bias: 0.0,
                        mass_normal: 0.0,
                        mass_tangent: 0.0,
                    });
                }
            }
            (Some(Shape2::Box { half_extents: he }), Some(Shape2::Polygon { vertices: v })) => {
                verts_a.clear();
                verts_b.clear();
                let he = *he;
                let box_verts = [
                    Vec2::new(-he.x, -he.y),
                    Vec2::new(he.x, -he.y),
                    Vec2::new(he.x, he.y),
                    Vec2::new(-he.x, he.y),
                ];
                for &vtx in &box_verts {
                    verts_a.push(body_a.transform.transform_point(vtx));
                }
                for &vert in v {
                    verts_b.push(body_b.transform.transform_point(vert));
                }
                let res = sat_polygon_polygon(verts_a, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        phieng_collision::sat::polygon_centroid(verts_a),
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    points.clear();
                    polygon_polygon_contact_points_into(verts_a, verts_b, normal, points, clip_tmp);
                    if points.is_empty() {
                        polygon_polygon_contact_points_into(
                            verts_b, verts_a, -normal, points, clip_tmp,
                        );
                    }
                    for &point in points.iter().take(2) {
                        out.push(ContactConstraint {
                            body_a: a,
                            body_b: b,
                            point,
                            normal,
                            penetration: depth,
                            friction,
                            restitution,
                            normal_impulse: 0.0,
                            tangent_impulse: 0.0,
                            rest_bias: 0.0,
                            mass_normal: 0.0,
                            mass_tangent: 0.0,
                        });
                    }
                }
            }
            (Some(Shape2::Polygon { vertices: v }), Some(Shape2::Box { half_extents: he })) => {
                verts_a.clear();
                verts_b.clear();
                let he = *he;
                let box_verts = [
                    Vec2::new(-he.x, -he.y),
                    Vec2::new(he.x, -he.y),
                    Vec2::new(he.x, he.y),
                    Vec2::new(-he.x, he.y),
                ];
                for &vert in v {
                    verts_a.push(body_a.transform.transform_point(vert));
                }
                for &vtx in &box_verts {
                    verts_b.push(body_b.transform.transform_point(vtx));
                }
                let res = sat_polygon_polygon(verts_a, verts_b);
                if let Some((normal, depth)) = res {
                    let normal = phieng_collision::sat::orient_normal(
                        normal,
                        phieng_collision::sat::polygon_centroid(verts_a),
                        phieng_collision::sat::polygon_centroid(verts_b),
                    );
                    points.clear();
                    polygon_polygon_contact_points_into(verts_a, verts_b, normal, points, clip_tmp);
                    if points.is_empty() {
                        polygon_polygon_contact_points_into(
                            verts_b, verts_a, -normal, points, clip_tmp,
                        );
                    }
                    for &point in points.iter().take(2) {
                        out.push(ContactConstraint {
                            body_a: a,
                            body_b: b,
                            point,
                            normal,
                            penetration: depth,
                            friction,
                            restitution,
                            normal_impulse: 0.0,
                            tangent_impulse: 0.0,
                            rest_bias: 0.0,
                            mass_normal: 0.0,
                            mass_tangent: 0.0,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

/// Physics world in 3D
#[derive(Debug)]
pub struct World3 {
    /// Rigid bodies
    pub bodies: Vec<RigidBody3>,
    /// Broad phase mode (auto-selects BruteForce / DynamicAABBTree).
    pub broadphase_mode: BroadphaseMode,
    /// Broad phase backend. Resolved from `broadphase_mode` at construction.
    pub broadphase: Broadphase3,
    /// 3D constraint solver
    pub solver: Solver3D,
    /// Gravity
    pub gravity: Vec3,
    /// Gravitational force model + integration scheme (see
    /// [`crate::gravity`]). Default is the legacy uniform field with
    /// semi-implicit Euler — bit-identical to pre-gravity behavior.
    pub gravity_config: super::gravity::GravityConfig3,
    /// Ambient force fields — spring (SHM), aerodynamic drag, and
    /// turbulence (see [`crate::ambient`]). Default is a no-op.
    pub ambient: super::ambient::AmbientConfig3,
    /// Collision-detection tier ceiling (see [`crate::collision_tier`]).
    /// Off by default → full fidelity (Tier5).
    pub collision_tier: super::collision_tier::CollisionTierConfig,
    /// Continuous collision detection via adaptive substepping (see
    /// [`crate::ccd`]). Default is a no-op (single legacy substep).
    pub ccd: super::ccd::CcdConfig,
    /// Monotonic simulation clock in seconds (see [`World2::time`]).
    pub time: f32,
    /// Next body ID
    pub next_body_id: usize,
    /// Fixed timestep
    pub dt: f32,
    /// Velocity threshold for sleeping
    pub linear_sleep_threshold: f32,
    /// Angular velocity threshold for sleeping
    pub angular_sleep_threshold: f32,
    /// Joints (see [`crate::joint3`]; serial only, solved after contacts).
    pub joints: Vec<Joint3>,
    /// Next joint id (doubles as insertion index; see [`World2::add_joint`]).
    pub next_joint_id: usize,
    /// Contact cache for warm starting (Fx-hashed, see [`World2::contact_cache`]).
    pub contact_cache: FxHashMap<(usize, usize), Vec<CachedContact3D>>,
    /// Collect the contact events (a pair began or stopped touching).
    ///
    /// Off by default: the events cost a pass over the contact cache,
    /// and a scene that never reads them should not pay for it. When
    /// on, the events of the last step are in
    /// [`Self::contact_events`].
    pub collect_contact_events: bool,
    /// See [`World2::contact_cache_pool`].
    contact_cache_pool: Vec<Vec<CachedContact3D>>,
    /// Bodies touched by a contact this step (parallel to `bodies`).
    /// See [`World2::contact_touch`]; never snapshotted.
    contact_touch: Vec<bool>,
    /// SPH fluid state (see [`crate::fluid3`]). Empty by default, in
    /// which case the fluid phase is a no-op.
    pub fluid: super::fluid3::FluidState3,
    /// Thermal simulation state (see [`crate::thermal`]). Disabled by default.
    pub thermal: super::thermal::ThermalState3,
    /// Nuclear simulation state (see [`crate::nuclear`]). Disabled by
    /// default.
    pub nuclear: super::nuclear::NuclearState2,
    /// Per-phase wall-clock accumulators. Only present with the
    /// `phase-timing` feature; the default build has no such field, so
    /// the struct's layout and public API are unchanged.
    #[cfg(feature = "phase-timing")]
    phase_timings: super::phase_timing::PhaseTimings,
    /// Runtime validation configuration
    pub validation: ValidationConfig,
    /// Last validation report
    pub validation_report: ValidationReport,
    /// Scratch buffers reused every step so the per-frame narrowphase
    /// performs no allocations in the steady state.
    scratch_constraints: Vec<ContactConstraint3D>,
    scratch_pairs: Vec<(usize, usize)>,
    scratch_points: Vec<Vec3>,
    scratch_clip_tmp: Vec<Vec3>,
    /// One AABB per body, computed in a separate pass before the
    /// broad-phase insert (see `World2::compute_body_aabbs`).
    scratch_aabbs: Vec<(Vec3, Vec3)>,
    epa_buffers: phieng_collision::epa::EpaBuffers,
    /// Vanished contact pairs per step (for loss-of-contact wake).
    /// Reused buffer; never snapshotted.
    scratch_stale: Vec<(usize, usize)>,
    /// The contacts a sensor pair reported, kept apart from the
    /// solved ones: a sensor reports contacts but never pushes.
    scratch_sensor_contacts: Vec<ContactConstraint3D>,
    /// The sensor pairs that were touching last step, so the events
    /// can say which of them ended.
    scratch_last_sensor_pairs: Vec<(usize, usize)>,
    /// The sensor pairs touching this step.
    scratch_sensor_pairs: Vec<(usize, usize)>,
    /// The contact events of this step.
    scratch_events: Vec<ContactEvent3>,
    /// The last known contact of every pair, kept so an *ended*
    /// event can still say where the pair was.
    scratch_last_contacts: Vec<(usize, usize, Vec3, Vec3)>,
}

/// Cached contact impulses for 3D warm starting
#[derive(Debug, Clone, Copy)]
pub struct CachedContact3D {
    pub point: Vec3,
    pub normal: Vec3,
    pub normal_impulse: f32,
    pub tangent1_impulse: f32,
    pub tangent2_impulse: f32,
}

impl World3 {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            broadphase_mode: BroadphaseMode::default(),
            broadphase: Broadphase3::new(BroadphaseMode::default()),
            solver: Solver3D::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            gravity_config: super::gravity::GravityConfig3::default(),
            ambient: super::ambient::AmbientConfig3::default(),
            collision_tier: super::collision_tier::CollisionTierConfig::default(),
            ccd: super::ccd::CcdConfig::default(),
            time: 0.0,
            next_body_id: 0,
            dt: 1.0 / 60.0,
            linear_sleep_threshold: LINEAR_SLEEP_THRESHOLD,
            angular_sleep_threshold: ANGULAR_SLEEP_THRESHOLD,
            contact_cache: FxHashMap::default(),
            collect_contact_events: false,
            contact_cache_pool: Vec::new(),
            contact_touch: Vec::new(),
            joints: Vec::new(),
            next_joint_id: 0,
            fluid: super::fluid3::FluidState3::new(),
            thermal: super::thermal::ThermalState3::new(),
            nuclear: super::nuclear::NuclearState2::new(),
            #[cfg(feature = "phase-timing")]
            phase_timings: super::phase_timing::PhaseTimings::default(),
            validation: ValidationConfig::default(),
            validation_report: ValidationReport::default(),
            scratch_constraints: Vec::new(),
            scratch_pairs: Vec::new(),
            scratch_points: Vec::new(),
            scratch_clip_tmp: Vec::new(),
            scratch_aabbs: Vec::with_capacity(1024),
            epa_buffers: phieng_collision::epa::EpaBuffers::new(),
            scratch_stale: Vec::new(),
            scratch_sensor_contacts: Vec::new(),
            scratch_last_sensor_pairs: Vec::new(),
            scratch_sensor_pairs: Vec::new(),
            scratch_events: Vec::new(),
            scratch_last_contacts: Vec::new(),
        }
    }

    /// Add a rigid body to the world. The body must already carry a valid
    /// shape (see [`ShapeError`]); position must be finite.
    pub fn add_body(&mut self, mut body: RigidBody3) -> Result<usize, ShapeError> {
        let p = body.transform.position;
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        if let Some(shape) = &body.shape {
            validate_shape3(shape)?;
            validate_density(body.is_static, body.material.density)?;
        }
        let id = self.next_body_id;
        body.id = id;
        self.next_body_id += 1;
        self.bodies.push(body);
        Ok(id)
    }

    /// Add a static body (immovable)
    pub fn add_static_body(&mut self, position: Vec3, shape: Shape3) -> Result<usize, ShapeError> {
        if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        let mut body = RigidBody3::static_body(self.next_body_id);
        body.transform.position = position;
        body.set_shape(shape, 1.0)?;
        self.add_body(body)
    }

    /// Add a dynamic body
    pub fn add_dynamic_body(
        &mut self,
        position: Vec3,
        shape: Shape3,
        density: f32,
    ) -> Result<usize, ShapeError> {
        if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
            return Err(ShapeError::NonFiniteInput);
        }
        let mut body = RigidBody3::new(self.next_body_id);
        body.transform.position = position;
        body.set_shape(shape, density)?;
        self.add_body(body)
    }

    /// Add a joint to the world. The joint must already carry validated
    /// endpoints (see [`JointError`]); ids are assigned here to preserve
    /// the id == index invariant (same contract as [`World2::add_joint`]).
    pub fn add_joint(&mut self, mut joint: Joint3) -> Result<usize, JointError> {
        let (a, b) = joint.bodies();
        if a >= self.bodies.len() || b >= self.bodies.len() {
            return Err(JointError::UnknownBody);
        }
        if a == b {
            return Err(JointError::SameBody);
        }
        let id = self.next_joint_id;
        joint.set_id(id);
        self.next_joint_id += 1;
        self.joints.push(joint);
        Ok(id)
    }

    /// Remove a joint by id. Ids double as indices; removal shifts the
    /// tail and reassigns ids to preserve the invariant.
    pub fn remove_joint(&mut self, joint_id: usize) {
        if joint_id >= self.joints.len() {
            return;
        }
        self.joints.remove(joint_id);
        for (index, joint) in self.joints.iter_mut().enumerate() {
            joint.set_id(index);
        }
        self.next_joint_id = self.joints.len();
    }

    /// Remove a body from the world (see [`World2::remove_body`] for the
    /// id/index invariant; contact-cache keys are remapped through the
    /// shift, entries touching the removed body are dropped; joints
    /// touching the removed body are dropped and survivors shift).
    pub fn remove_body(&mut self, body_id: usize) {
        if body_id >= self.bodies.len() {
            return;
        }
        let last = self.bodies.len() - 1;
        // Drop joints touching the removed body first (they hold stale ids).
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != body_id && b != body_id
        });
        self.bodies.remove(body_id);
        for joint in self.joints.iter_mut() {
            joint.shift(body_id);
        }
        for (index, body) in self.bodies.iter_mut().enumerate() {
            body.id = index;
        }
        self.next_body_id = self.bodies.len();
        if body_id == last {
            self.contact_cache
                .retain(|&(a, b), _| a != body_id && b != body_id);
        } else {
            for ((a, b), entries) in std::mem::take(&mut self.contact_cache) {
                if a == body_id || b == body_id {
                    continue;
                }
                let na = if a == last { body_id } else { a };
                let nb = if b == last { body_id } else { b };
                let key = if na < nb { (na, nb) } else { (nb, na) };
                self.contact_cache.entry(key).or_default().extend(entries);
            }
        }
    }

    /// Step the physics simulation
    /// Advance the simulation by `dt` seconds. With CCD enabled (see
    /// [`crate::ccd`]), the step splits into substeps so no body travels
    /// more than `max_movement` per substep; each substep runs the full
    /// pipeline. Disabled (default) is a single legacy substep,
    /// bit-identical to pre-CCD behavior.
    pub fn step(&mut self, dt: f32) {
        // Reset validation report once for the whole step; substeps merge.
        self.validation_report = ValidationReport::default();
        let n = if self.ccd.is_active() {
            match self.ccd.tier {
                CcdTier::Off => 1,
                CcdTier::SweptAABB => {
                    if self.ccd_needs_substepping_3d(dt, self.ccd.tier) {
                        self.ccd.substeps(dt, self.ccd_peak_speed())
                    } else {
                        1
                    }
                }
                CcdTier::FullGJK => self.ccd.substeps(dt, self.ccd_peak_speed()),
            }
        } else {
            1
        };
        if n <= 1 {
            self.step_inner(dt);
        } else {
            let sub = dt / n as f32;
            for _ in 0..n {
                self.step_inner(sub);
            }
        }
        // 8. Advance the simulation clock (drives time-varying ambient fields).
        self.time += dt;
    }

    /// Peak effective body speed (linear plus angular sweep at the body
    /// extents) for CCD substepping. Static bodies carry zero velocity,
    /// so they contribute nothing.
    fn ccd_peak_speed(&self) -> f32 {
        let mut peak = 0.0f32;
        for body in &self.bodies {
            if body.is_static {
                continue;
            }
            let (min, max) = self.get_body_aabb(body);
            let extent = (max - min).length() * 0.5;
            let eff = body.velocity.length() + body.angular_velocity.length() * extent;
            peak = peak.max(eff);
        }
        peak
    }

    /// Swept-AABB pair overlap for 3D: returns true if any broadphase
    /// pair's swept volumes overlap during `dt`.
    fn ccd_needs_substepping_3d(&mut self, dt: f32, tier: CcdTier) -> bool {
        if tier != CcdTier::SweptAABB {
            return false;
        }
        self.update_broadphase();
        let mut pairs = Vec::new();
        self.broadphase.collect_pairs(&mut pairs);
        for &(a, b) in &pairs {
            if a >= self.bodies.len() || b >= self.bodies.len() {
                continue;
            }
            let body_a = &self.bodies[a];
            let body_b = &self.bodies[b];
            if body_a.is_static || body_b.is_static {
                continue;
            }
            // Collision layers: a pair that may not collide must not
            // drag the whole world into substepping either.
            if !super::body::collision_filter_allows(
                body_a.collision_group,
                body_a.collision_mask,
                body_b.collision_group,
                body_b.collision_mask,
            ) {
                continue;
            }
            let (amin, amax) = self.get_body_aabb(body_a);
            let (bmin, bmax) = self.get_body_aabb(body_b);
            let a_extent = (amax - amin).length() * 0.5;
            let b_extent = (bmax - bmin).length() * 0.5;
            if swept_aabb_overlap_3(
                body_a.transform.position,
                body_a.velocity,
                body_a.angular_velocity.length(),
                a_extent,
                body_b.transform.position,
                body_b.velocity,
                body_b.angular_velocity.length(),
                b_extent,
                dt,
            ) {
                return true;
            }
        }
        false
    }

    fn step_inner(&mut self, dt: f32) {
        #[cfg(feature = "phase-timing")]
        let mut timings = super::phase_timing::PhaseTimings::default();

        // 0. Resolve a Hybrid broadphase request (see World2::step_inner).
        self.resolve_broadphase_mode();

        // 1. Gravity kick (see GravityIntegration): uniform background
        // field plus Central/N-body terms per GravityConfig, one
        // full-step kick (Euler) or the first half-kick (Verlet, with
        // linear damping riding as symmetric drag).
        let verlet = self.gravity_config.integrator == super::gravity::Integrator::VelocityVerlet;
        let h = if verlet { dt * 0.5 } else { dt };
        timed!(timings.gravity, {
            super::gravity::gravity_kick_3(
                &mut self.bodies,
                &self.gravity_config,
                self.gravity,
                h,
                verlet,
            );

            // 1a. Ambient forces (spring/SHM, air drag, turbulence). Same slot
            // and same Verlet split as the gravity kick above.
            if self.ambient.is_active() {
                super::ambient::ambient_kick_3(&mut self.bodies, &self.ambient, self.time, h);
            }
        });

        // 1b. Fluid phase - SPH particles and two-way body coupling. Runs
        // before broadphase so the rigid pipeline observes post-fluid
        // velocities; a no-op when the fluid holds no particles.
        timed!(timings.fluid, {
            self.fluid
                .step_fluid(&mut self.bodies, self.gravity, &self.ambient, self.time, dt);
        });

        // 2. Broad phase - find potential collision pairs (into scratch).
        let mut pairs = std::mem::take(&mut self.scratch_pairs);
        timed!(timings.broadphase_aabb, {
            self.compute_body_aabbs();
        });
        timed!(timings.broadphase_rebuild, {
            self.broadphase.begin_frame();
            self.insert_body_aabbs();
            // The grid keeps its cells across steps, so the bodies the
            // world no longer has have to be taken out of them.
            self.broadphase.end_frame();
        });
        timed!(timings.broadphase_pairs, {
            pairs.clear();
            // Narrowphase fast path: only AABB-overlapping pairs can
            // produce contacts. The CCD swept test keeps the unfiltered
            // sequence on purpose.
            self.broadphase.collect_overlapping_pairs(&mut pairs);
        });

        // 3. Narrow phase - test actual collisions (into scratch).
        let mut constraints = std::mem::take(&mut self.scratch_constraints);
        constraints.clear();
        let mut points = std::mem::take(&mut self.scratch_points);
        let mut clip_tmp = std::mem::take(&mut self.scratch_clip_tmp);
        let mut epa_buffers = std::mem::take(&mut self.epa_buffers);
        let mut sensor_contacts = std::mem::take(&mut self.scratch_sensor_contacts);
        sensor_contacts.clear();
        timed!(timings.narrowphase, {
            for &(a, b) in &pairs {
                // Static-static pairs can never produce an impulse (zero
                // effective mass in every direction); skip the narrowphase.
                // Bit-identical: their constraints would solve to zero.
                if self.bodies[a].is_static && self.bodies[b].is_static {
                    continue;
                }
                // Collision layers: each body must accept the other's
                // layer. The default filter accepts everything, so an
                // unconfigured scene is unaffected.
                if !super::body::collision_filter_allows(
                    self.bodies[a].collision_group,
                    self.bodies[a].collision_mask,
                    self.bodies[b].collision_group,
                    self.bodies[b].collision_mask,
                ) {
                    continue;
                }
                // The tier gate is a runtime value, so it is resolved
                // once here and the pair loop below calls the narrow
                // phase once.
                let (max_tier, tier_enabled) = if self.collision_tier.is_active() {
                    (self.collision_tier.max_tier, true)
                } else {
                    (super::collision_tier::CollisionTier::Tier5_GJK_EPA, false)
                };
                // A sensor reports contacts but never pushes: its
                // contacts go to the sensor list, not to the solver.
                if self.bodies[a].is_sensor || self.bodies[b].is_sensor {
                    Self::test_collisions_3d_into(
                        &self.bodies,
                        a,
                        b,
                        &mut sensor_contacts,
                        &mut points,
                        &mut clip_tmp,
                        &mut epa_buffers,
                        max_tier,
                        tier_enabled,
                    );
                } else {
                    Self::test_collisions_3d_into(
                        &self.bodies,
                        a,
                        b,
                        &mut constraints,
                        &mut points,
                        &mut clip_tmp,
                        &mut epa_buffers,
                        max_tier,
                        tier_enabled,
                    );
                }
            }
        });

        // 3a. A sensor's contacts are events, not contacts: the pair
        // set is compared with last step's, and a pair that is new (or
        // gone) produces an event. The diff runs when the sensor has
        // contacts *or* had them last step — a sensor that just left
        // something has no contacts now, and that is exactly when its
        // "ended" event is due.
        if self.collect_contact_events
            && (!self.scratch_sensor_contacts.is_empty()
                || !self.scratch_last_sensor_pairs.is_empty())
        {
            self.sensor_events_3d(&sensor_contacts);
        }

        // 3b. Warm-start from cached impulses. A pair with no cache
        // entry is a NEW contact: any sleeping party wakes (with
        // cascade) so it can respond to the newcomer. Every constrained
        // body is marked touched (sleep-support tracking for step 6+7).
        timed!(timings.cache, {
            if self.contact_touch.len() != self.bodies.len() {
                self.contact_touch.resize(self.bodies.len(), false);
            }
            for t in self.contact_touch.iter_mut() {
                *t = false;
            }
            for constraint in constraints.iter_mut() {
                let (a, b) = (constraint.body_a, constraint.body_b);
                self.contact_touch[a] = true;
                self.contact_touch[b] = true;
                // One hash lookup, not two: the same lookup tells us
                // whether the pair is new (a new pair may need its
                // sleepers woken) and which cached contact to match.
                let mut is_new = true;
                if let Some(entries) = self.contact_cache.get(&(a, b)) {
                    is_new = false;
                    let mut best: Option<&CachedContact3D> = None;
                    let mut best_dist = 0.04f32;
                    for entry in entries {
                        if entry.normal.dot(constraint.normal) <= 0.0 {
                            continue;
                        }
                        let d = (entry.point - constraint.point).length_squared();
                        if d < best_dist {
                            best_dist = d;
                            best = Some(entry);
                        }
                    }
                    if let Some(entry) = best {
                        constraint.normal_impulse = entry.normal_impulse;
                        constraint.tangent1_impulse = entry.tangent1_impulse;
                        constraint.tangent2_impulse = entry.tangent2_impulse;
                    }
                }
                // A pair that was not touching last step began touching
                // this step: that is the event a trigger wants.
                if is_new && self.collect_contact_events {
                    let (ea, eb) = event_pair(a, b);
                    self.scratch_events.push(ContactEvent3 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Began,
                        point: constraint.point,
                        normal: constraint.normal,
                    });
                }
                // A new pair with a sleeping party wakes it (with a
                // cascade) so it can respond to the newcomer. The wake
                // cannot change what the match above found: it only
                // clears sleep flags, and the cache holds the previous
                // frame's contacts either way.
                if is_new && (self.bodies[a].is_sleeping || self.bodies[b].is_sleeping) {
                    self.wake_body(a);
                    self.wake_body(b);
                }
            }

            // Validate constraints before solving
            let constraint_report =
                validate_constraints_3d(&self.bodies, &mut constraints, &self.validation);
            self.validation_report.merge(constraint_report);
        });

        // 4. Solve constraints
        timed!(timings.solver, {
            self.solver.solve(&mut self.bodies, &mut constraints, dt);

            // 4b. Solve joints (after contacts, same iteration budget).
            // Joints own their impulses, so warm starting is direct.
            // Skipped entirely when the world holds no joints (zero cost).
            // Serial only (see `crate::joint3`).
            if !self.joints.is_empty() {
                for joint in self.joints.iter() {
                    joint.warm_start(&mut self.bodies);
                }
                let iters = self.solver.iterations;
                for _ in 0..iters {
                    for joint in self.joints.iter_mut() {
                        joint.solve(&mut self.bodies, dt);
                    }
                }
            }

            // 4b. Write back accumulated impulses for next frame's warm
            // start. Entry vectors are cleared in place and reused; vacated
            // entries are dropped (keeping `contact_cache.len()` exact) but
            // their buffers are recycled via the pool. Zero steady-state
            // allocation; trajectories bit-identical.
            let cache = &mut self.contact_cache;
            let pool = &mut self.contact_cache_pool;
            // Remember each pair's last known contact before the lists
            // are cleared, so an *ended* event can still say where the
            // pair was.
            if self.collect_contact_events {
                self.scratch_last_contacts.clear();
                for (key, list) in cache.iter() {
                    if let Some(first) = list.first() {
                        self.scratch_last_contacts
                            .push((key.0, key.1, first.point, first.normal));
                    }
                }
            }
            for list in cache.values_mut() {
                list.clear();
            }
            for constraint in &constraints {
                let key = (constraint.body_a, constraint.body_b);
                cache
                    .entry(key)
                    .or_insert_with(|| pool.pop().unwrap_or_default())
                    .push(CachedContact3D {
                        point: constraint.point,
                        normal: constraint.normal,
                        normal_impulse: constraint.normal_impulse,
                        tangent1_impulse: constraint.tangent1_impulse,
                        tangent2_impulse: constraint.tangent2_impulse,
                    });
            }
            // A vanished pair is lost contact: wake any sleeping party
            // (loss of support must re-evaluate sleep; see World2).
            // Idempotent flags: drain order cannot affect determinism.
            self.scratch_stale.clear();
            let stale = &mut self.scratch_stale;
            cache.retain(|key, list| {
                if list.is_empty() {
                    stale.push(*key);
                    pool.push(std::mem::take(list));
                    false
                } else {
                    true
                }
            });
            // A pair that was touching last step and is not touching
            // now stopped touching: that is the other half of the
            // event. The contact data is the pair's last known one.
            if self.collect_contact_events {
                for &(a, b) in self.scratch_stale.iter() {
                    let (point, normal) = self
                        .scratch_last_contacts
                        .iter()
                        .find(|&&(ka, kb, _, _)| ka == a && kb == b)
                        .map(|&(_, _, p, n)| (p, n))
                        .unwrap_or((Vec3::ZERO, Vec3::ZERO));
                    let (ea, eb) = event_pair(a, b);
                    self.scratch_events.push(ContactEvent3 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Ended,
                        point,
                        normal,
                    });
                }
            }
            // Pop from the end (LIFO): the woken set is order-independent.
            while let Some((a, b)) = self.scratch_stale.pop() {
                if a < self.bodies.len()
                    && b < self.bodies.len()
                    && (self.bodies[a].is_sleeping || self.bodies[b].is_sleeping)
                {
                    self.wake_body(a);
                    self.wake_body(b);
                }
            }
        });

        timed!(timings.integrate, {
            if verlet {
                // 5v. Verlet drift (positions only — no linear damping) plus
                // the second half-kick with forces recomputed at the new
                // positions. The contact solver above ran unchanged.
                for body in self.bodies.iter_mut() {
                    if body.is_sleeping {
                        continue;
                    }
                    body.integrate_drift(dt);
                }
                super::gravity::gravity_kick_3(
                    &mut self.bodies,
                    &self.gravity_config,
                    self.gravity,
                    dt * 0.5,
                    true,
                );
                if self.ambient.is_active() {
                    super::ambient::ambient_kick_3(
                        &mut self.bodies,
                        &self.ambient,
                        self.time,
                        dt * 0.5,
                    );
                }
            } else {
                // 5. Integrate motion (sleeping bodies are frozen until woken).
                for body in self.bodies.iter_mut() {
                    if body.is_sleeping {
                        continue;
                    }
                    body.integrate(dt);
                }
            }
        });

        // 5c. Nuclear decay and fission (see World2::step_inner phase 5c).
        timed!(timings.thermal, {
            if self.nuclear.config.is_active() {
                let nuclear_config = &self.nuclear.config;
                let nuclear_scratch = &mut self.nuclear.scratch;
                super::nuclear::step_nuclear_3d(
                    &mut self.bodies,
                    nuclear_config,
                    nuclear_scratch,
                    dt,
                );
                let thermal_config = self.thermal.config;
                super::nuclear::apply_nuclear_heat_3d(
                    &mut self.bodies,
                    &self.nuclear.scratch,
                    &thermal_config,
                );
            }

            // 5d. Thermal transfer (see World2::step_inner phase 5d).
            let thermal_config = self.thermal.config;
            super::thermal::step_thermal_3d(
                &mut self.bodies,
                &constraints,
                &thermal_config,
                &mut self.thermal,
                dt,
            );
            // 5d. Thermal expansion (see World2::step_inner).
            super::thermal::apply_expansion_3d(&mut self.bodies, &thermal_config);
        });

        // 6+7. Validate bodies + update sleep in a single fused sweep
        // (bit-identical to validate-all-then-sleep-all; see World2).
        // Sleep needs slowness AND support (a current contact): ballistic
        // apexes and drifting bodies never sleep, so no wake/sleep
        // history is needed and save/load stays exact. Sleeping
        // velocities are re-zeroed every step (see World2).
        let lin = self.linear_sleep_threshold;
        let ang = self.angular_sleep_threshold;
        timed!(timings.sleep, {
            for (i, body) in self.bodies.iter_mut().enumerate() {
                let report = validate_body_3d(body, &self.validation);
                self.validation_report.merge(report);
                if body.is_static {
                    continue;
                }
                if body.is_sleeping {
                    body.velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                    continue;
                }
                if body.should_sleep(lin, ang) && self.contact_touch[i] {
                    body.is_sleeping = true;
                    body.velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                }
            }
        });

        #[cfg(feature = "phase-timing")]
        {
            timings.steps = 1;
            self.phase_timings.merge(timings);
        }

        // Return scratch buffers for the next step.
        self.scratch_pairs = pairs;
        self.scratch_constraints = constraints;
        self.scratch_points = points;
        self.scratch_clip_tmp = clip_tmp;
        self.epa_buffers = epa_buffers;
    }

    /// The per-phase wall-clock accumulators.
    ///
    /// Only available with the `phase-timing` feature; the default build
    /// has no timers and no accumulator, and its public API is unchanged.
    #[cfg(feature = "phase-timing")]
    pub fn phase_timings(&self) -> &super::phase_timing::PhaseTimings {
        &self.phase_timings
    }

    /// Forget everything measured so far.
    #[cfg(feature = "phase-timing")]
    pub fn reset_phase_timings(&mut self) {
        self.phase_timings.reset();
    }

    /// How many candidate pairs the broad phase produced on the last step.
    ///
    /// This is the *raw* count, before the narrow phase filters it: the
    /// difference between this and the contact-cache size is the work the
    /// narrow phase throws away.
    #[cfg(feature = "phase-timing")]
    pub fn broadphase_pair_count(&self) -> usize {
        self.scratch_pairs.len()
    }

    /// How many cells the broad-phase grid currently holds.
    #[cfg(feature = "phase-timing")]
    pub fn broadphase_cell_count(&self) -> usize {
        self.broadphase.cell_count()
    }

    /// Resolve [`BroadphaseMode::Hybrid`] to a concrete 3D backend from the
    /// live body count (see [`World2::resolve_broadphase_mode`]).
    fn resolve_broadphase_mode(&mut self) {
        if self.broadphase_mode == BroadphaseMode::Hybrid {
            let resolved = BroadphaseMode::auto_select(self.bodies.len());
            self.broadphase = Broadphase3::new(resolved);
        }
    }

    /// Compute every body's AABB into the pooled scratch buffer.
    fn compute_body_aabbs(&mut self) {
        // The buffer is moved out first so the AABB computation can
        // borrow the world immutably while it fills it.
        let mut aabbs = std::mem::take(&mut self.scratch_aabbs);
        aabbs.clear();
        for body in &self.bodies {
            let (min, max) = self.get_body_aabb(body);
            aabbs.push((min, max));
        }
        self.scratch_aabbs = aabbs;
    }

    /// Insert every body into the broad phase from the AABB buffer.
    fn insert_body_aabbs(&mut self) {
        let aabbs = &self.scratch_aabbs;
        for (body, (min, max)) in self.bodies.iter().zip(aabbs.iter()) {
            if body.is_static {
                // A pair of two statics can never produce an impulse, so
                // the broad phase does not generate it.
                self.broadphase.insert_static(body.id, *min, *max);
            } else {
                self.broadphase.insert(body.id, *min, *max);
            }
        }
    }

    /// Update the broadphase with current body positions (3D)
    fn update_broadphase(&mut self) {
        self.broadphase.begin_frame();
        self.compute_body_aabbs();
        self.insert_body_aabbs();
        // The grid keeps its cells across steps, so the bodies the world
        // no longer has have to be taken out of them.
        self.broadphase.end_frame();
    }

    /// The bodies whose AABB overlaps `aabb`.
    ///
    /// The ids come back in ascending order, and a body is reported
    /// only when the filter accepts it (see [`QueryFilter`]).
    ///
    /// The scan is over the *current* body positions, not over the
    /// broad phase: the grid is rebuilt at the start of the step, so
    /// between steps it holds the positions from before the solve,
    /// and a query must not report against those.
    pub fn overlap_aabb(&self, aabb: &AABB3, filter: &QueryFilter) -> Vec<u32> {
        let mut out = Vec::new();
        for body in &self.bodies {
            if !filter.accepts(body.collision_group, body.collision_mask) {
                continue;
            }
            let (min, max) = self.get_body_aabb(body);
            if min.x <= aabb.max.x
                && max.x >= aabb.min.x
                && min.y <= aabb.max.y
                && max.y >= aabb.min.y
                && min.z <= aabb.max.z
                && max.z >= aabb.min.z
            {
                out.push(body.id as u32);
            }
        }
        out
    }

    /// Get AABB for a body (exact for rotated boxes)
    fn get_body_aabb(&self, body: &RigidBody3) -> (Vec3, Vec3) {
        match &body.shape {
            Some(Shape3::Sphere { radius }) => {
                let min = body.transform.position - Vec3::new(*radius, *radius, *radius);
                let max = body.transform.position + Vec3::new(*radius, *radius, *radius);
                (min, max)
            }
            Some(Shape3::Box { half_extents }) => {
                let he = *half_extents;
                let corners = [
                    Vec3::new(-he.x, -he.y, -he.z),
                    Vec3::new(he.x, -he.y, -he.z),
                    Vec3::new(he.x, he.y, -he.z),
                    Vec3::new(-he.x, he.y, -he.z),
                    Vec3::new(-he.x, -he.y, he.z),
                    Vec3::new(he.x, -he.y, he.z),
                    Vec3::new(he.x, he.y, he.z),
                    Vec3::new(-he.x, he.y, he.z),
                ];
                let mut min = body.transform.transform_point(corners[0]);
                let mut max = min;
                for &corner in &corners[1..] {
                    let transformed = body.transform.transform_point(corner);
                    min = min.min(transformed);
                    max = max.max(transformed);
                }
                (min, max)
            }
            Some(Shape3::Capsule {
                radius,
                half_length,
            }) => {
                // Segment endpoints in world space (capsule axis is
                // local Y); pad by the radius. The old axis-aligned
                // version missed collisions for rotated capsules.
                let axis = body.transform.rotation * Vec3::Y * *half_length;
                let pa = body.transform.position + axis;
                let pb = body.transform.position - axis;
                let r = Vec3::new(*radius, *radius, *radius);
                (pa.min(pb) - r, pa.max(pb) + r)
            }
            Some(Shape3::ConvexHull { vertices }) => {
                if vertices.is_empty() {
                    return (body.transform.position, body.transform.position);
                }
                let mut min = body.transform.transform_point(vertices[0]);
                let mut max = min;
                for &vertex in &vertices[1..] {
                    let transformed = body.transform.transform_point(vertex);
                    min = min.min(transformed);
                    max = max.max(transformed);
                }
                (min, max)
            }
            Some(Shape3::ConvexMesh { vertices, .. }) => {
                // Convex support set: brute-force vertex bounds, exact.
                if vertices.is_empty() {
                    return (body.transform.position, body.transform.position);
                }
                let mut min = body.transform.transform_point(vertices[0]);
                let mut max = min;
                for &vertex in &vertices[1..] {
                    let transformed = body.transform.transform_point(vertex);
                    min = min.min(transformed);
                    max = max.max(transformed);
                }
                (min, max)
            }
            None => (body.transform.position, body.transform.position),
        }
    }

    /// Test collision between two 3D bodies, appending constraints to
    /// `out` (up to 4 contact points per pair).
    ///
    /// Dispatch by shape pair so the common cases use exact analytic
    /// narrowphases: 3D SAT for box-box (exact face normals — GJK/EPA
    /// normals wobble ~1 degree on face contact and that tilt feeds a
    /// torque feedback loop), closest-point for sphere pairs. GJK/EPA
    /// covers capsules and convex hulls. `points`/`clip_tmp` are
    /// caller-owned scratch; nothing allocates here except in the rare
    /// capsule/hull support-closure path.
    #[allow(clippy::too_many_arguments)]
    fn test_collisions_3d_into(
        bodies: &[RigidBody3],
        a: usize,
        b: usize,
        out: &mut Vec<ContactConstraint3D>,
        points: &mut Vec<Vec3>,
        clip_tmp: &mut Vec<Vec3>,
        epa_buffers: &mut phieng_collision::epa::EpaBuffers,
        max_tier: super::collision_tier::CollisionTier,
        tier_enabled: bool,
    ) {
        use phieng_collision::contact3d::{box_box_contact_points_into, closest_point_on_box};
        use phieng_collision::sat3d::{sat_obb_obb, ObbContactKind};

        let body_a = &bodies[a];
        let body_b = &bodies[b];

        if tier_enabled {
            let Some(sa) = body_a.shape.as_ref() else {
                return;
            };
            let Some(sb) = body_b.shape.as_ref() else {
                return;
            };
            let required = super::collision_tier::required_tier_3d(sa, sb);
            if required > max_tier {
                return;
            }
        }

        let friction =
            PhysicsMaterial::combine_friction(body_a.material.friction, body_b.material.friction);
        let restitution = PhysicsMaterial::combine_restitution(
            body_a.material.restitution,
            body_b.material.restitution,
        );

        match (&body_a.shape, &body_b.shape) {
            (Some(Shape3::Box { half_extents: hea }), Some(Shape3::Box { half_extents: heb })) => {
                let Some((normal, depth, kind)) = sat_obb_obb(
                    body_a.transform.position,
                    *hea,
                    body_a.transform.rotation,
                    body_b.transform.position,
                    *heb,
                    body_b.transform.rotation,
                ) else {
                    return;
                };
                let depth = depth.max(0.0);
                points.clear();
                match kind {
                    ObbContactKind::FaceA => {
                        box_box_contact_points_into(
                            body_a.transform.position,
                            *hea,
                            body_a.transform.rotation,
                            body_b.transform.position,
                            *heb,
                            body_b.transform.rotation,
                            normal,
                            points,
                            clip_tmp,
                        );
                        if points.is_empty() {
                            box_box_contact_points_into(
                                body_b.transform.position,
                                *heb,
                                body_b.transform.rotation,
                                body_a.transform.position,
                                *hea,
                                body_a.transform.rotation,
                                -normal,
                                points,
                                clip_tmp,
                            );
                        }
                    }
                    ObbContactKind::FaceB => {
                        box_box_contact_points_into(
                            body_b.transform.position,
                            *heb,
                            body_b.transform.rotation,
                            body_a.transform.position,
                            *hea,
                            body_a.transform.rotation,
                            -normal,
                            points,
                            clip_tmp,
                        );
                    }
                    ObbContactKind::EdgeEdge => {
                        // Transient edge contact: single midpoint of the
                        // extreme supports.
                        let sa = Self::support_function_for(body_a);
                        let sb = Self::support_function_for(body_b);
                        points.push((sa(normal) + sb(-normal)) * 0.5);
                    }
                };
                if points.is_empty() {
                    return;
                }
                out.reserve(points.len().min(4));
                for &point in points.iter().take(4) {
                    out.push(ContactConstraint3D::new(
                        a,
                        b,
                        point,
                        normal,
                        depth,
                        friction,
                        restitution,
                    ));
                }
            }
            (Some(Shape3::Sphere { radius: ra }), Some(Shape3::Sphere { radius: rb })) => {
                let delta = body_b.transform.position - body_a.transform.position;
                let dist = delta.length();
                let depth = ra + rb - dist;
                if depth < 0.0 {
                    return;
                }
                let normal = if dist > EPS { delta / dist } else { Vec3::X };
                out.push(ContactConstraint3D::new(
                    a,
                    b,
                    body_a.transform.position + normal * *ra,
                    normal,
                    depth,
                    friction,
                    restitution,
                ));
            }
            (Some(Shape3::Sphere { radius: r }), Some(Shape3::Box { half_extents: he })) => {
                let c = body_a.transform.position;
                let (q, _) = closest_point_on_box(
                    c,
                    body_b.transform.position,
                    *he,
                    body_b.transform.rotation,
                );
                let diff = c - q;
                let dist = diff.length();
                if dist > *r {
                    return;
                }
                // diff points from the box surface toward the sphere
                // center, i.e. B -> A; flip to the A -> B convention.
                // Falls back to the EPA normal when the center is
                // inside the box.
                let normal = if dist > EPS {
                    -(diff / dist)
                } else {
                    let sa = Self::support_function_for(body_a);
                    let sb = Self::support_function_for(body_b);
                    match phieng_collision::epa::gjk_epa_with_buffers(&sa, &sb, epa_buffers) {
                        Some(hit) => hit.normal,
                        None => return,
                    }
                };
                out.push(ContactConstraint3D::new(
                    a,
                    b,
                    q,
                    normal,
                    (*r - dist).max(0.0),
                    friction,
                    restitution,
                ));
            }
            (Some(Shape3::Box { half_extents: he }), Some(Shape3::Sphere { radius: r })) => {
                let c = body_b.transform.position;
                let (q, _) = closest_point_on_box(
                    c,
                    body_a.transform.position,
                    *he,
                    body_a.transform.rotation,
                );
                let diff = c - q;
                let dist = diff.length();
                if dist > *r {
                    return;
                }
                // diff points from the box surface toward the sphere
                // center, i.e. A -> B. No flip needed.
                let normal = if dist > EPS {
                    diff / dist
                } else {
                    let sa = Self::support_function_for(body_a);
                    let sb = Self::support_function_for(body_b);
                    match phieng_collision::epa::gjk_epa_with_buffers(&sa, &sb, epa_buffers) {
                        Some(hit) => hit.normal,
                        None => return,
                    }
                };
                out.push(ContactConstraint3D::new(
                    a,
                    b,
                    q,
                    normal,
                    (*r - dist).max(0.0),
                    friction,
                    restitution,
                ));
            }
            // Capsules / hulls: GJK/EPA with support-midpoint contact.
            // Witness refinement: a box support is a far corner (it
            // maximizes all axes), so a raw support midpoint can sit
            // meters from the contact — the lever arm then collapses the
            // effective mass and the solver pushes ~nothing (capsules
            // tunneled through floors). Any box side is replaced by the
            // closest box-surface point to the other side's witness, which
            // by construction lies in the contact region.
            _ => {
                let sa = Self::support_function_for(body_a);
                let sb = Self::support_function_for(body_b);
                let Some(hit) = phieng_collision::epa::gjk_epa_with_buffers(&sa, &sb, epa_buffers)
                else {
                    return;
                };
                let pa = sa(hit.normal);
                let pb = sb(-hit.normal);
                let wa = match &body_a.shape {
                    Some(Shape3::Box { half_extents }) => {
                        closest_point_on_box(
                            pb,
                            body_a.transform.position,
                            *half_extents,
                            body_a.transform.rotation,
                        )
                        .0
                    }
                    _ => pa,
                };
                let wb = match &body_b.shape {
                    Some(Shape3::Box { half_extents }) => {
                        closest_point_on_box(
                            pa,
                            body_b.transform.position,
                            *half_extents,
                            body_b.transform.rotation,
                        )
                        .0
                    }
                    _ => pb,
                };
                out.push(ContactConstraint3D::new(
                    a,
                    b,
                    (wa + wb) * 0.5,
                    hit.normal,
                    hit.depth.max(0.0),
                    friction,
                    restitution,
                ));
            }
        }
    }

    /// Build the support mapping for a body. Heap-allocates the closure
    /// (and a transformed copy for hulls), so the analytic box/sphere
    /// paths above avoid calling it entirely.
    fn support_function_for(body: &RigidBody3) -> Box<dyn Fn(Vec3) -> Vec3 + '_> {
        let transform = body.transform;
        let fallback_pos = transform.position;
        match &body.shape {
            Some(Shape3::Sphere { radius }) => {
                let center = transform.position;
                let radius = *radius;
                Box::new(move |dir: Vec3| center + dir.normalize() * radius)
            }
            Some(Shape3::Box { half_extents }) => {
                let center = transform.position;
                let he = *half_extents;
                let rot = transform.rotation;
                Box::new(move |dir: Vec3| {
                    let local_dir = rot.inverse() * dir;
                    let local_support = Vec3::new(
                        if local_dir.x > 0.0 { he.x } else { -he.x },
                        if local_dir.y > 0.0 { he.y } else { -he.y },
                        if local_dir.z > 0.0 { he.z } else { -he.z },
                    );
                    center + rot * local_support
                })
            }
            Some(Shape3::Capsule {
                radius,
                half_length,
            }) => {
                // World-space segment endpoints; the support itself
                // delegates to the tested gjk3d::support_capsule (endpoint
                // by sign + normalize). Do NOT inline a variant: a prior
                // inline version clamped the direction's Y component to
                // [-1, 1] instead of the segment parameter to [-hl, hl],
                // returning non-maximal supports that made GJK falsely
                // report separation — capsules tunneled through floors.
                let center = transform.position;
                let rot = transform.rotation;
                let r = *radius;
                let pa = center + rot * (Vec3::Y * *half_length);
                let pb = center - rot * (Vec3::Y * *half_length);
                Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_capsule(pa, pb, r, dir))
            }
            Some(Shape3::ConvexHull { vertices }) => {
                let center = transform.position;
                let rot = transform.rotation;
                let verts: Vec<Vec3> = vertices.iter().map(|v| rot * *v + center).collect();
                Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_convex_hull(&verts, dir))
            }
            Some(Shape3::ConvexMesh { vertices, .. }) => {
                // Convex support set: max dot over vertices, same as hulls.
                let center = transform.position;
                let rot = transform.rotation;
                let verts: Vec<Vec3> = vertices.iter().map(|v| rot * *v + center).collect();
                Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_convex_hull(&verts, dir))
            }
            None => Box::new(move |_dir: Vec3| fallback_pos),
        }
    }

    /// Cast a ray into the world and return the first thing it hits.
    ///
    /// See [`World2::raycast`]. The convex shapes (a convex hull and a
    /// convex mesh) are not tested yet: a ray that would hit one is
    /// reported as a miss.
    pub fn raycast(&self, ray: &Ray3, filter: &QueryFilter) -> Option<WorldRayHit3> {
        let mut best: Option<WorldRayHit3> = None;
        for body in &self.bodies {
            if !filter.accepts(body.collision_group, body.collision_mask) {
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
                    // The capsule axis is the body's local Y.
                    let axis = body.transform.rotation * Vec3::Y * *half_length;
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

    /// The contact events of the last step.
    ///
    /// One event per pair that began or stopped touching. The events are
    /// only collected when [`Self::collect_contact_events`] is on, and
    /// they are cleared at the start of the next step.
    pub fn contact_events(&self) -> &[ContactEvent3] {
        &self.scratch_events
    }

    /// Work out which sensor pairs began and which ended (3D).
    fn sensor_events_3d(&mut self, contacts: &[ContactConstraint3D]) {
        self.scratch_sensor_pairs.clear();
        for c in contacts {
            let key = (c.body_a, c.body_b);
            if self.scratch_sensor_pairs.last() != Some(&key) {
                self.scratch_sensor_pairs.push(key);
            }
        }

        let mut i = 0;
        let mut j = 0;
        let mut ended: Vec<(usize, usize)> = Vec::new();
        while i < self.scratch_sensor_pairs.len() || j < self.scratch_last_sensor_pairs.len() {
            let now = self.scratch_sensor_pairs.get(i).copied();
            let before = self.scratch_last_sensor_pairs.get(j).copied();
            match (now, before) {
                (Some(c), Some(p)) if c == p => {
                    i += 1;
                    j += 1;
                }
                (Some(c), Some(p)) => {
                    if c < p {
                        let (point, normal) = sensor_contact_3d(contacts, c);
                        let (ea, eb) = event_pair(c.0, c.1);
                        self.scratch_events.push(ContactEvent3 {
                            body_a: ea,
                            body_b: eb,
                            kind: super::body::ContactEventKind::Began,
                            point,
                            normal,
                        });
                        i += 1;
                    } else {
                        ended.push(p);
                        j += 1;
                    }
                }
                (Some(c), None) => {
                    let (point, normal) = sensor_contact_3d(contacts, c);
                    let (ea, eb) = event_pair(c.0, c.1);
                    self.scratch_events.push(ContactEvent3 {
                        body_a: ea,
                        body_b: eb,
                        kind: super::body::ContactEventKind::Began,
                        point,
                        normal,
                    });
                    i += 1;
                }
                (None, Some(p)) => {
                    ended.push(p);
                    j += 1;
                }
                (None, None) => break,
            }
        }

        for key in ended {
            let (point, normal) = sensor_contact_3d(&self.scratch_sensor_contacts, key);
            let (ea, eb) = event_pair(key.0, key.1);
            self.scratch_events.push(ContactEvent3 {
                body_a: ea,
                body_b: eb,
                kind: super::body::ContactEventKind::Ended,
                point,
                normal,
            });
        }

        std::mem::swap(
            &mut self.scratch_sensor_pairs,
            &mut self.scratch_last_sensor_pairs,
        );
    }

    /// Wake a body and cascade through its contact partners and joint
    /// partners, transitively. The woken set is a fixpoint over
    /// idempotent flags, so map iteration order cannot affect the
    /// result (deterministic). Out-of-range ids are ignored.
    pub fn wake_body(&mut self, body_id: usize) {
        if body_id >= self.bodies.len() {
            return;
        }
        self.bodies[body_id].is_sleeping = false;
        let mut dirty = true;
        while dirty {
            dirty = false;
            for (a, b) in self.contact_cache.keys() {
                let (a, b) = (*a, *b);
                if a < self.bodies.len() && b < self.bodies.len() {
                    if !self.bodies[a].is_sleeping && self.bodies[b].is_sleeping {
                        self.bodies[b].is_sleeping = false;
                        dirty = true;
                    } else if !self.bodies[b].is_sleeping && self.bodies[a].is_sleeping {
                        self.bodies[a].is_sleeping = false;
                        dirty = true;
                    }
                }
            }
            for joint in &self.joints {
                let (a, b) = joint.bodies();
                if a < self.bodies.len() && b < self.bodies.len() {
                    if !self.bodies[a].is_sleeping && self.bodies[b].is_sleeping {
                        self.bodies[b].is_sleeping = false;
                        dirty = true;
                    } else if !self.bodies[b].is_sleeping && self.bodies[a].is_sleeping {
                        self.bodies[a].is_sleeping = false;
                        dirty = true;
                    }
                }
            }
        }
    }
}

impl Default for World2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for World3 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world2_creation() {
        let mut world = World2::new();
        let id = world
            .add_static_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(1.0, 1.0),
                },
            )
            .expect("valid shape");
        assert_eq!(id, 0);
    }

    #[test]
    fn test_world2_add_body() {
        let mut world = World2::new();
        let id = world
            .add_dynamic_body(Vec2::new(0.0, 10.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        assert_eq!(id, 0);
        assert_eq!(world.bodies.len(), 1);
    }

    #[test]
    fn test_world2_time_advances() {
        let mut world = World2::new();
        assert_eq!(world.time, 0.0);
        world.step(1.0 / 60.0);
        assert!((world.time - 1.0 / 60.0).abs() < 1e-9);
        world.step(1.0 / 60.0);
        assert!((world.time - 2.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_world2_spring_oscillates() {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;
        let id = world
            .add_dynamic_body(Vec2::new(1.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[id].linear_damping = 0.0;
        world.linear_sleep_threshold = 0.0;
        world.ambient.spring = super::super::ambient::SpringConfig2 {
            enabled: true,
            anchor: Vec2::ZERO,
            stiffness: 40.0,
            damping: 0.0,
            max_displacement: 0.0,
        };
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        for _ in 0..600 {
            world.step(1.0 / 240.0);
            let x = world.bodies[id].transform.position.x;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        assert!(
            max_x > 0.5 && min_x < -0.5,
            "spring did not oscillate: [{min_x},{max_x}]"
        );
    }

    #[test]
    fn test_world2_ambient_default_noop() {
        // Default ambient config is fully inert: a body's trajectory is
        // exactly what gravity alone produces (kick is gated off).
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        let id = world
            .add_dynamic_body(Vec2::new(0.0, 10.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[id].linear_damping = 0.0;
        assert!(!world.ambient.is_active());
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        // Free-fall: v = g*t exactly (no damping, no contacts).
        let t = 2.0f32;
        assert!(
            (world.bodies[id].velocity.y - (-9.81 * t)).abs() < 1e-2,
            "default ambient perturbed free fall: {}",
            world.bodies[id].velocity.y
        );
    }

    #[test]
    fn test_world3_creation() {
        let mut world = World3::new();
        let id = world
            .add_static_body(
                Vec3::new(0.0, 0.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(1.0, 1.0, 1.0),
                },
            )
            .expect("valid shape");
        assert_eq!(id, 0);
    }

    #[test]
    fn test_world3_box_drop() {
        let mut world = World3::new();
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            )
            .expect("valid shape");
        let box_id = world
            .add_dynamic_body(
                Vec3::new(0.0, 3.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");

        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step(dt);
        }

        let b = &world.bodies[box_id];
        assert!(
            b.transform.position.y > -0.5,
            "box tunneled through floor: y={}",
            b.transform.position.y
        );
        assert!(
            b.velocity.length() < 1.0,
            "box should have settled: v={}",
            b.velocity
        );
    }

    /// Outward-wound box mesh centered at the origin.
    fn box_mesh(half_extents: Vec3) -> (Vec<Vec3>, Vec<u32>) {
        let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
        let v = vec![
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
        ];
        let idx = vec![
            1, 2, 6, 1, 6, 5, // +x
            0, 4, 7, 0, 7, 3, // -x
            3, 7, 6, 3, 6, 2, // +y
            0, 1, 5, 0, 5, 4, // -y
            4, 5, 6, 4, 6, 7, // +z
            0, 3, 2, 0, 2, 1, // -z
        ];
        (v, idx)
    }

    #[test]
    fn test_world3_convex_mesh_drop() {
        // A box convex mesh exercises the mesh AABB/support path through
        // the GJK catch-all: it must rest on the floor, not tunnel.
        let mut world = World3::new();
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            )
            .expect("valid shape");
        let (verts, idx) = box_mesh(Vec3::new(0.5, 0.5, 0.5));
        let mesh_id = world
            .add_dynamic_body(
                Vec3::new(0.0, 3.0, 0.0),
                Shape3::ConvexMesh {
                    vertices: verts,
                    indices: idx,
                },
                1.0,
            )
            .expect("valid shape");

        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step(dt);
        }

        let b = &world.bodies[mesh_id];
        assert!(
            b.transform.position.y > -0.5,
            "mesh tunneled through floor: y={}",
            b.transform.position.y
        );
        assert!(
            b.velocity.length() < 1.0,
            "mesh should have settled: v={}",
            b.velocity
        );
    }

    #[test]
    fn test_world3_sphere_drop() {
        let mut world = World3::new();
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            )
            .expect("valid shape");
        let ball_id = world
            .add_dynamic_body(
                Vec3::new(0.0, 5.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");

        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }

        let b = &world.bodies[ball_id];
        assert!(
            b.transform.position.y < 5.0,
            "ball should have fallen: y={}",
            b.transform.position.y
        );
    }

    /// Regression: a capsule dropped on a box floor must rest on it, not
    /// tunnel. Two independent bugs conspired here and both were caught by
    /// the F28 cross-language equivalence test: (1) the world's capsule
    /// support closure clamped the direction's Y component instead of the
    /// segment parameter (non-maximal supports -> GJK false separation);
    /// (2) the box-vs-capsule contact used the raw support midpoint, which
    /// for a box is a far corner — the lever arm collapsed the effective
    /// mass and the solver pushed ~nothing. Box sides now use the closest
    /// box-surface point to the other witness.
    #[test]
    fn test_world3_capsule_drop() {
        let mut world = World3::new();
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            )
            .expect("valid shape");
        let cap_id = world
            .add_dynamic_body(
                Vec3::new(0.0, 5.0, 0.0),
                Shape3::Capsule {
                    radius: 0.3,
                    half_length: 0.5,
                },
                1.0,
            )
            .expect("valid shape");

        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step(dt);
        }

        let b = &world.bodies[cap_id];
        // Floor top -0.5 + capsule extent 0.8 -> rest center ~0.3.
        assert!(
            b.transform.position.y > -0.2,
            "capsule tunneled through floor: y={}",
            b.transform.position.y
        );
        assert!(
            b.transform.position.y < 1.5,
            "capsule never settled: y={}",
            b.transform.position.y
        );
        assert!(
            b.velocity.length() < 1.0,
            "capsule should have settled: v={}",
            b.velocity
        );
    }

    #[test]
    fn test_world3_stack() {
        let mut world = World3::new();
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            )
            .expect("valid shape");
        let mut ids = Vec::new();
        for i in 0..3 {
            ids.push(
                world
                    .add_dynamic_body(
                        Vec3::new(0.0, 0.05 + i as f32 * 1.02, 0.0),
                        Shape3::Box {
                            half_extents: Vec3::new(0.5, 0.5, 0.5),
                        },
                        1.0,
                    )
                    .expect("valid shape"),
            );
        }

        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step(dt);
        }

        for (level, id) in ids.iter().enumerate() {
            let b = &world.bodies[*id];
            assert!(
                b.transform.position.y > level as f32 - 0.75,
                "stack level {level} tunneled: y={}",
                b.transform.position.y
            );
            assert!(
                b.velocity.length() < 1.0,
                "stack level {level} jittering: v={}",
                b.velocity
            );
        }
    }

    #[test]
    fn test_world2_remove_body_keeps_id_index_invariant() {
        let mut world = World2::new();
        let a = world
            .add_dynamic_body(Vec2::new(-2.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        let b = world
            .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        let c = world
            .add_dynamic_body(Vec2::new(2.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        assert_eq!((a, b, c), (0, 1, 2));
        world.remove_body(1);
        assert_eq!(world.bodies.len(), 2);
        // Tail shifts and ids are reassigned so id == index still holds.
        assert_eq!(world.bodies[0].id, 0);
        assert_eq!(world.bodies[1].id, 1);
        assert_eq!(world.bodies[1].transform.position.x, 2.0);
        // Out-of-range removal is a no-op.
        world.remove_body(99);
        assert_eq!(world.bodies.len(), 2);
        // New bodies append with consistent ids.
        let d = world
            .add_dynamic_body(Vec2::new(0.0, 8.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        assert_eq!(d, 2);
        // World still simulates after removal.
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        assert!(world.bodies[0].transform.position.y < 5.0);
    }

    #[test]
    fn test_world3_remove_body_keeps_id_index_invariant() {
        let mut world = World3::new();
        world
            .add_dynamic_body(
                Vec3::new(-2.0, 5.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        world
            .add_dynamic_body(
                Vec3::new(2.0, 5.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        world.remove_body(0);
        assert_eq!(world.bodies.len(), 1);
        assert_eq!(world.bodies[0].id, 0);
        assert_eq!(world.bodies[0].transform.position.x, 2.0);
        let d = world
            .add_dynamic_body(
                Vec3::new(0.0, 8.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        assert_eq!(d, 1);
        world.step(1.0 / 60.0);
        assert!(world.bodies[1].transform.position.y < 8.0);
    }
}
