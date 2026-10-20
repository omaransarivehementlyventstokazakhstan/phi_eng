//! Tier performance sweep: the broad-phase backends, the solver quality
//! tiers and the collision tiers, measured on the same scene.
//!
//! The tier *blocks* (B01-B06, Q01-Q05, H01-H05) pin the tiers'
//! **correctness** - the backends agree bit-for-bit, the quality tiers
//! change the trajectory in the documented way, the collision tiers cap
//! the contacts. None of them says what the tiers *cost*, and a tier
//! nobody has measured is a tier nobody can price.
//!
//! Each test prints a single `TIER ...` readings line. Run with:
//! `cargo test --release -p phieng --test tier_sweep -- --nocapture --test-threads=1`
//!
//! NOTE: test names here intentionally avoid the parity-ID convention
//! (`tools/check_test_parity.py`), so this file needs no C# mirror.

use glam::Vec2;
use phieng_broadphase::BroadphaseMode;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::collision_tier::CollisionTier;
use phieng_dynamics::solver::SolverQuality;
use phieng_dynamics::World2;
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

/// Print one readings line.
fn tier_report(name: &str, cases: usize, wall_ms: f64, extra: &str) {
    println!(
        "TIER name={name} cases={cases} wall_ms={wall_ms:.1} ms_per_case={:.4} {extra}",
        wall_ms / (cases.max(1)) as f64
    );
}

/// A hash of the world's body state.
fn state_hash(world: &World2) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A pile of mixed boxes and circles, big enough that the broad-phase
/// backends actually differ.
fn pile(bodies: usize) -> World2 {
    let mut world = World2::new();
    world.gravity = Vec2::new(0.0, -9.81);
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(30.0, 0.5),
            },
        )
        .expect("valid shape");
    for i in 0..bodies {
        let x = -12.0 + (i as f32 % 24.0) * 1.0;
        let y = 0.5 + (i / 24) as f32 * 1.05;
        world
            .add_dynamic_body(
                Vec2::new(x, y),
                if i % 3 == 0 {
                    Shape2::Circle { radius: 0.4 }
                } else {
                    Shape2::Box {
                        half_extents: Vec2::new(0.4, 0.4),
                    }
                },
                1.0,
            )
            .expect("valid shape");
    }
    world
}

/// The broad-phase backends must all produce the *same* trajectory, and
/// the spread between them is the cost of the abstraction.
#[test]
fn tier_broadphase_backend_scaling() {
    const BODIES: usize = 300;
    const STEPS: usize = 60;

    let modes = [
        ("BruteForce", BroadphaseMode::BruteForce),
        ("SpatialHash", BroadphaseMode::SpatialHash),
        ("DynamicAABBTree", BroadphaseMode::DynamicAABBTree),
        ("Hybrid", BroadphaseMode::Hybrid),
    ];

    let mut hash = None;
    let mut rows = Vec::new();
    let mut total_wall = 0.0f64;
    for (name, mode) in modes {
        let mut world = pile(BODIES);
        world.broadphase_mode = mode;

        // Warm the world up so the measurement is not dominated by the
        // first frames' broad-phase rebuilds.
        for _ in 0..10 {
            world.step(DT);
        }

        let start = Instant::now();
        for _ in 0..STEPS {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_wall += wall_ms;

        let h = state_hash(&world);
        match hash {
            None => hash = Some(h),
            Some(first) => assert_eq!(
                first, h,
                "the broad-phase backends must agree bit-for-bit ({name})"
            ),
        }
        rows.push(format!("{name}={:.4}", wall_ms / STEPS as f64));
    }

    tier_report(
        "broadphase_backend_scaling",
        4,
        total_wall,
        &format!(
            "bodies={BODIES} steps={STEPS} ms_per_step[{}] hash={:#018x}",
            rows.join(" "),
            hash.unwrap_or(0)
        ),
    );
}

/// The solver quality tiers trade accuracy for time; the sweep measures
/// what each tier costs and how far the trajectory drifts.
#[test]
fn tier_solver_quality_scaling() {
    const BODIES: usize = 200;
    const STEPS: usize = 120;

    let qualities = [
        ("Ultra", SolverQuality::Ultra),
        ("High", SolverQuality::High),
        ("Medium", SolverQuality::Medium),
        ("Low", SolverQuality::Low),
        ("Minimal", SolverQuality::Minimal),
    ];

    let mut reference = None;
    let mut rows = Vec::new();
    let mut total_wall = 0.0f64;
    for (name, quality) in qualities {
        let mut world = pile(BODIES);
        world.solver.apply_quality(quality);

        let start = Instant::now();
        for _ in 0..STEPS {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_wall += wall_ms;

        // How far the cheapest bodies ended from the reference tier.
        let drift: f32 = world
            .bodies
            .iter()
            .skip(1)
            .map(|b| b.transform.position.y)
            .sum::<f32>()
            / (BODIES as f32);
        let h = state_hash(&world);
        if reference.is_none() {
            reference = Some((drift, h));
        }
        let (ref_drift, _) = reference.unwrap();
        rows.push(format!(
            "{name}={:.4}(drift={:+.5})",
            wall_ms / STEPS as f64,
            drift - ref_drift
        ));
    }

    tier_report(
        "solver_quality_scaling",
        5,
        total_wall,
        &format!(
            "bodies={BODIES} steps={STEPS} ms_per_step[{}]",
            rows.join(" ")
        ),
    );
}

/// The collision tiers cap the narrowphase; the sweep measures what
/// each tier costs and how many contacts it produces.
#[test]
fn tier_collision_tier_scaling() {
    const BODIES: usize = 200;
    const STEPS: usize = 120;

    let tiers = [
        ("Tier1_AABBOnly", CollisionTier::Tier1_AABBOnly),
        ("Tier2_SphereAABB", CollisionTier::Tier2_SphereAABB),
        ("Tier3_SphereSphere", CollisionTier::Tier3_SphereSphere),
        ("Tier4_BoxBox", CollisionTier::Tier4_BoxBox),
        ("Tier5_GJK_EPA", CollisionTier::Tier5_GJK_EPA),
    ];

    let mut rows = Vec::new();
    let mut total_wall = 0.0f64;
    for (name, tier) in tiers {
        let mut world = pile(BODIES);
        world.collision_tier.enabled = true;
        world.collision_tier.max_tier = tier;

        let start = Instant::now();
        for _ in 0..STEPS {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_wall += wall_ms;

        let contacts = world.contact_cache.len();
        rows.push(format!(
            "{name}={:.4}(contacts={contacts})",
            wall_ms / STEPS as f64
        ));
    }

    tier_report(
        "collision_tier_scaling",
        5,
        total_wall,
        &format!(
            "bodies={BODIES} steps={STEPS} ms_per_step[{}]",
            rows.join(" ")
        ),
    );
}
