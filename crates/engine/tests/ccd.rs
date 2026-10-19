//! CCD tests (C01–C04): conservative advancement via adaptive
//! substepping. These embed shared parity IDs mirrored by
//! `tests/PhiEng.Tests/CcdTests.cs` (enforced by
//! `tools/check_test_parity.py`), so the managed and Rust engines are
//! tested against the same contract.

use glam::{Vec2, Vec3};
use phieng_dynamics::ccd::CcdConfig;
use phieng_dynamics::{
    world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, LoadMode,
};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

const DT: f32 = 1.0 / 60.0;

fn clean_2d(w: &mut World2) {
    w.gravity = Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn clean_3d(w: &mut World3) {
    w.gravity = Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn ccd_on() -> CcdConfig {
    CcdConfig {
        enabled: true,
        tier: phieng_dynamics::ccd::CcdTier::FullGJK,
        max_movement: 0.05,
        max_substeps: 64,
    }
}

/// C01: a fast ball tunnels through a thin wall with CCD off, but is
/// caught and comes to rest on it with CCD on.
#[test]
fn c01_thin_wall_2d() {
    let build = || {
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(5.0, 0.05),
            },
        )
        .expect("valid shape");
        let id = w
            .add_dynamic_body(Vec2::new(0.0, 3.0), Shape2::Circle { radius: 0.25 }, 1.0)
            .expect("valid shape");
        clean_2d(&mut w);
        // Perfectly inelastic pair: the caught ball sticks instead of
        // bouncing (restitution combines from both materials).
        w.bodies[0].material.restitution = 0.0;
        w.bodies[id].material.restitution = 0.0;
        w.bodies[id].velocity = Vec2::new(0.0, -60.0);
        (w, id)
    };

    let (mut off, off_id) = build();
    for _ in 0..120 {
        off.step(DT);
    }
    assert!(
        off.bodies[off_id].transform.position.y < -1.0,
        "without CCD the fast ball should tunnel through the thin wall, y={}",
        off.bodies[off_id].transform.position.y
    );

    let (mut on, on_id) = build();
    on.ccd = ccd_on();
    for _ in 0..120 {
        on.step(DT);
    }
    let y = on.bodies[on_id].transform.position.y;
    assert!(
        y > 0.2 && y < 0.6,
        "with CCD the ball should rest on the wall (top 0.05 + r 0.25), y={y}"
    );
    assert!(
        on.bodies[on_id].velocity.length() < 1.0,
        "ball should have settled, v={}",
        on.bodies[on_id].velocity.length()
    );
}

/// C02: the 3D mirror — a fast sphere tunnels through a thin slab with
/// CCD off, but is caught with CCD on.
#[test]
fn c02_thin_wall_3d() {
    let build = || {
        let mut w = World3::new();
        w.add_static_body(
            Vec3::new(0.0, 0.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(5.0, 0.05, 5.0),
            },
        )
        .expect("valid shape");
        let id = w
            .add_dynamic_body(
                Vec3::new(0.0, 3.0, 0.0),
                Shape3::Sphere { radius: 0.25 },
                1.0,
            )
            .expect("valid shape");
        clean_3d(&mut w);
        // Perfectly inelastic pair: the caught sphere sticks instead of
        // bouncing (restitution combines from both materials).
        w.bodies[0].material.restitution = 0.0;
        w.bodies[id].material.restitution = 0.0;
        // 2.0 units/step clears the whole Minkowski sum (0.1 wall + 0.5
        // diameter), so the slab is skipped outright without CCD.
        w.bodies[id].velocity = Vec3::new(0.0, -120.0, 0.0);
        (w, id)
    };

    let (mut off, off_id) = build();
    for _ in 0..120 {
        off.step(DT);
    }
    assert!(
        off.bodies[off_id].transform.position.y < -1.0,
        "without CCD the fast sphere should tunnel, y={}",
        off.bodies[off_id].transform.position.y
    );

    let (mut on, on_id) = build();
    on.ccd = ccd_on();
    for _ in 0..120 {
        on.step(DT);
    }
    let y = on.bodies[on_id].transform.position.y;
    assert!(
        y > 0.2 && y < 0.6,
        "with CCD the sphere should rest on the slab, y={y}"
    );
}

/// C03: CCD enabled but below the speed threshold takes exactly one
/// substep — trajectories are bit-identical to CCD-off, the clock still
/// advances once per step, and default worlds stay schema v3 (no "ccd"
/// key in JSON).
#[test]
fn c03_default_off_bit_identical() {
    let build = || {
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
        clean_2d(&mut w);
        w
    };

    let mut off = build();
    let mut on = build();
    // Enabled but with a movement budget no scene speed can exceed:
    // every step takes exactly one substep.
    on.ccd = CcdConfig {
        enabled: true,
        tier: phieng_dynamics::ccd::CcdTier::FullGJK,
        max_movement: 10.0,
        max_substeps: 8,
    };
    for _ in 0..300 {
        off.step(DT);
        on.step(DT);
    }
    assert_eq!(off.bodies.len(), on.bodies.len());
    for (a, b) in off.bodies.iter().zip(on.bodies.iter()) {
        assert_eq!(
            a.transform.position.x.to_bits(),
            b.transform.position.x.to_bits()
        );
        assert_eq!(
            a.transform.position.y.to_bits(),
            b.transform.position.y.to_bits()
        );
        assert_eq!(a.velocity.x.to_bits(), b.velocity.x.to_bits());
        assert_eq!(a.velocity.y.to_bits(), b.velocity.y.to_bits());
    }
    assert!(
        (off.time - 5.0).abs() < 1e-4,
        "clock must advance once per step"
    );

    let json = world2_to_json(&off, LoadMode::Exact).expect("json");
    assert!(
        !json.contains("\"ccd\""),
        "default worlds must stay schema v3 without a ccd key"
    );
}

/// C04: a CCD world round-trips its config through JSON and binary
/// (schema v6) and keeps stopping the fast ball after a save/load cycle.
#[test]
fn c04_ccd_snapshot_roundtrip() {
    let mut w = World2::new();
    w.add_static_body(
        Vec2::new(0.0, 0.0),
        Shape2::Box {
            half_extents: Vec2::new(5.0, 0.05),
        },
    )
    .expect("valid shape");
    let id = w
        .add_dynamic_body(Vec2::new(0.0, 3.0), Shape2::Circle { radius: 0.25 }, 1.0)
        .expect("valid shape");
    clean_2d(&mut w);
    w.ccd = ccd_on();

    let json = world2_to_json(&w, LoadMode::Exact).expect("json");
    assert!(json.contains("\"ccd\""), "ccd block must persist, v6");
    assert!(
        json.contains("\"schema\": 6"),
        "ccd world must be schema v6"
    );
    let mut rj = world2_from_json(&json, LoadMode::Exact).expect("json load");
    assert!(rj.ccd.is_active());
    assert_eq!(rj.ccd.max_movement, 0.05);
    assert_eq!(rj.ccd.max_substeps, 64);

    let bin = world2_to_binary(&w, LoadMode::Exact).expect("binary");
    let mut rb = world2_from_binary(&bin, LoadMode::Exact).expect("binary load");
    assert!(rb.ccd.is_active());
    assert_eq!(rb.ccd, rj.ccd);

    w.bodies[id].velocity = Vec2::new(0.0, -60.0);
    rj.bodies[id].velocity = Vec2::new(0.0, -60.0);
    rb.bodies[id].velocity = Vec2::new(0.0, -60.0);
    for _ in 0..120 {
        w.step(DT);
        rj.step(DT);
        rb.step(DT);
    }
    for (name, y) in [
        ("live", w.bodies[id].transform.position.y),
        ("json", rj.bodies[id].transform.position.y),
        ("bin", rb.bodies[id].transform.position.y),
    ] {
        assert!(
            y > 0.2 && y < 0.6,
            "{name} world tunneled after save/load, y={y}"
        );
    }
}
