//! Solver quality-tier tests (Q01–Q05): iteration presets plus
//! friction/restitution switches. These embed shared parity IDs mirrored
//! by `tests/PhiEng.Tests/QualityTests.cs` (enforced by
//! `tools/check_test_parity.py`).

use glam::{Vec2, Vec3};
use phieng_dynamics::solver::SolverQuality;
use phieng_dynamics::{world2_from_json, world2_to_json, LoadMode};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

const DT: f32 = 1.0 / 60.0;

fn calm_2d(w: &mut World2) {
    w.gravity = Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn calm_3d(w: &mut World3) {
    w.gravity = Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

/// Q01: quality tiers map to the documented iteration budgets and
/// feature switches, in both dimensions.
#[test]
fn q01_tier_presets() {
    use phieng_dynamics::Solver;
    use phieng_dynamics::Solver3D;
    let mut s = Solver::new();
    assert_eq!(s.iterations, 8);
    assert!(s.enable_friction && s.enable_restitution);
    s.apply_quality(SolverQuality::Ultra);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (8, true, true)
    );
    s.apply_quality(SolverQuality::High);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (6, true, true)
    );
    s.apply_quality(SolverQuality::Medium);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (4, true, true)
    );
    s.apply_quality(SolverQuality::Low);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (2, false, false)
    );
    s.apply_quality(SolverQuality::Minimal);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (1, false, false)
    );
    s.apply_quality(SolverQuality::Ultra);
    assert_eq!(
        (s.iterations, s.enable_friction, s.enable_restitution),
        (8, true, true)
    );

    let mut s3 = Solver3D::new();
    s3.apply_quality(SolverQuality::Low);
    assert_eq!(
        (s3.iterations, s3.enable_friction, s3.enable_restitution),
        (2, false, false)
    );
    s3.apply_quality(SolverQuality::High);
    assert_eq!(
        (s3.iterations, s3.enable_friction, s3.enable_restitution),
        (6, true, true)
    );
}

/// Q02: with friction off, a sliding box keeps its speed; with friction
/// on it grinds to a halt.
#[test]
fn q02_friction_off_slides() {
    let build = |friction: bool| {
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, -0.5),
            Shape2::Box {
                half_extents: Vec2::new(50.0, 0.5),
            },
        )
        .expect("valid shape");
        let id = w
            .add_dynamic_body(
                Vec2::new(0.0, 0.5),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        calm_2d(&mut w);
        w.bodies[id].velocity = Vec2::new(5.0, 0.0);
        if !friction {
            w.solver.apply_quality(SolverQuality::Low);
            // Low also cuts iterations and restitution; restore those so
            // the test isolates friction only.
            w.solver.iterations = 8;
            w.solver.enable_restitution = true;
        }
        (w, id)
    };

    let (mut on, on_id) = build(true);
    for _ in 0..120 {
        on.step(DT);
    }
    assert!(
        on.bodies[on_id].velocity.x.abs() < 0.5,
        "friction should stop the box, vx={}",
        on.bodies[on_id].velocity.x
    );

    let (mut off, off_id) = build(false);
    for _ in 0..120 {
        off.step(DT);
    }
    assert!(
        off.bodies[off_id].velocity.x > 4.5,
        "without friction the box should keep sliding, vx={}",
        off.bodies[off_id].velocity.x
    );
}

/// Q03: with restitution off, a dropped ball does not bounce.
#[test]
fn q03_restitution_off_no_bounce() {
    let build = |restitution: bool| {
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, -0.5),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
        let id = w
            .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        calm_2d(&mut w);
        w.bodies[0].material.restitution = 1.0;
        w.bodies[id].material.restitution = 1.0;
        if !restitution {
            w.solver.enable_restitution = false;
        }
        (w, id)
    };

    let (mut on, on_id) = build(true);
    let mut impacted = false;
    let mut peak = 0.0f32;
    for _ in 0..120 {
        on.step(DT);
        let y = on.bodies[on_id].transform.position.y;
        if y < 1.0 {
            impacted = true;
        }
        if impacted {
            peak = peak.max(y);
        }
    }
    assert!(impacted, "ball never reached the floor");
    assert!(peak > 1.5, "elastic ball should bounce high, peak={peak}");

    let (mut off, off_id) = build(false);
    let mut impacted = false;
    let mut peak_off = 0.0f32;
    for _ in 0..120 {
        off.step(DT);
        let y = off.bodies[off_id].transform.position.y;
        if y < 1.0 {
            impacted = true;
        }
        if impacted {
            peak_off = peak_off.max(y);
        }
    }
    assert!(impacted, "ball never reached the floor");
    // An inelastic ball never climbs back up after first contact.
    assert!(
        peak_off < 1.0,
        "inelastic ball must never leave rest height, peak={peak_off}"
    );
    let y = off.bodies[off_id].transform.position.y;
    assert!(
        (0.3..0.7).contains(&y),
        "inelastic ball should rest on the floor, y={y}"
    );
}

/// Q04: the 3D solver honors the friction switch — same slide test.
#[test]
fn q04_friction_off_slides_3d() {
    let build = |friction: bool| {
        let mut w = World3::new();
        w.add_static_body(
            Vec3::new(0.0, -0.5, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(50.0, 0.5, 50.0),
            },
        )
        .expect("valid shape");
        let id = w
            .add_dynamic_body(
                Vec3::new(0.0, 0.5, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        calm_3d(&mut w);
        w.bodies[id].velocity = Vec3::new(5.0, 0.0, 0.0);
        if !friction {
            w.solver.apply_quality(SolverQuality::Low);
            w.solver.iterations = 8;
            w.solver.enable_restitution = true;
        }
        (w, id)
    };

    let (mut on, on_id) = build(true);
    for _ in 0..120 {
        on.step(DT);
    }
    assert!(
        on.bodies[on_id].velocity.x.abs() < 0.5,
        "3D friction should stop the box, vx={}",
        on.bodies[on_id].velocity.x
    );

    let (mut off, off_id) = build(false);
    for _ in 0..120 {
        off.step(DT);
    }
    assert!(
        off.bodies[off_id].velocity.x > 4.5,
        "3D without friction the box should keep sliding, vx={}",
        off.bodies[off_id].velocity.x
    );
}

/// Q05: solver flags round-trip through JSON (schema v7) and a default
/// world stays on its old schema without friction keys.
#[test]
fn q05_solver_flags_snapshot_roundtrip() {
    let mut w = World2::new();
    w.add_static_body(
        Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    w.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    let json = world2_to_json(&w, LoadMode::Exact).expect("json");
    assert!(
        json.contains("\"schema\": 3"),
        "default worlds stay schema v3"
    );
    // Material friction/restitution are floats; only the solver block
    // would carry literal `false` flags.
    assert!(
        !json.contains("\"friction\": false") && !json.contains("\"restitution\": false"),
        "default worlds must not carry solver flag keys"
    );

    w.solver.apply_quality(SolverQuality::Low);
    let json = world2_to_json(&w, LoadMode::Exact).expect("json");
    assert!(
        json.contains("\"schema\": 7"),
        "flagged worlds use schema v7"
    );
    assert!(
        json.contains("\"friction\": false") && json.contains("\"restitution\": false"),
        "solver flags must persist"
    );
    let back = world2_from_json(&json, LoadMode::Exact).expect("load");
    assert!(!back.solver.enable_friction);
    assert!(!back.solver.enable_restitution);
    assert_eq!(back.solver.iterations, 2);
}
