//! Broad-phase tier parity tests (B01–B06, mirrored Rust↔C#).
//!
//! The tier system is a *performance* choice: switching backend must never
//! change what the world simulates. These tests pin that contract by
//! running identical scenes through every mode and requiring bit-identical
//! trajectories, plus the auto-selection thresholds.

use glam::{Vec2, Vec3};
use phieng_broadphase::BroadphaseMode;
use phieng_dynamics::{Shape2, Shape3, World2, World3};

/// Deterministic body layout shared by every mode. Fixed positions (no
/// RNG) so the reference run is reproducible.
fn scene_world2() -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;

    // Ground.
    w.add_static_body(
        Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: Vec2::new(20.0, 0.5),
        },
    )
    .expect("valid shape");

    // A pyramid-ish pile: exercises stacking, contact chains, and sleep-free
    // settling so any backend difference would show up as drift.
    for row in 0..5i32 {
        let count = 7 - 2 * row;
        for c in 0..count {
            let y = row as f32 * 1.05;
            let x = (c as f32 - (count.saturating_sub(1) / 2) as f32) * 1.05;
            w.add_dynamic_body(
                Vec2::new(x, y),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        }
    }
    w
}

fn scene_world3() -> World3 {
    let mut w = World3::new();
    w.gravity = Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;

    w.add_static_body(
        Vec3::new(0.0, -1.0, 0.0),
        Shape3::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");

    for i in 0..8 {
        w.add_dynamic_body(
            Vec3::new(i as f32 * 1.1 - 4.0, 1.0 + i as f32 * 1.1, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    }
    w
}

const MODES: [BroadphaseMode; 3] = [
    BroadphaseMode::BruteForce,
    BroadphaseMode::SpatialHash,
    BroadphaseMode::DynamicAABBTree,
];

/// Trajectory fingerprint: every body's position after N steps.
fn fingerprint2(w: &World2) -> Vec<(Vec2, f32)> {
    w.bodies
        .iter()
        .map(|b| (b.transform.position, b.angular_velocity))
        .collect()
}

fn fingerprint3(w: &World3) -> Vec<(Vec3, Vec3)> {
    w.bodies
        .iter()
        .map(|b| (b.transform.position, b.angular_velocity))
        .collect()
}

/// B01: every 2D backend produces a bit-identical trajectory.
///
/// Pair *sets* are order-normalized (sorted + deduped) before they leave
/// the broadphase, so the narrowphase sees the same constraints regardless
/// of backend — which is what makes this exact equality, not tolerance.
#[test]
fn b01_all_2d_modes_bit_identical() {
    let mut reference: Option<Vec<(Vec2, f32)>> = None;
    let mut reference_mode = BroadphaseMode::SpatialHash;

    for mode in MODES {
        let mut w = scene_world2();
        w.broadphase_mode = mode;
        for _ in 0..240 {
            w.step(1.0 / 60.0);
        }
        let fp = fingerprint2(&w);
        match &reference {
            None => {
                reference = Some(fp);
                reference_mode = mode;
            }
            Some(ref r) => {
                assert_eq!(
                    *r,
                    fp,
                    "2D mode {} diverged from reference mode {}",
                    mode.name(),
                    reference_mode.name()
                );
            }
        }
    }
}

/// B02: every 3D backend produces a bit-identical trajectory.
#[test]
fn b02_all_3d_modes_bit_identical() {
    // All three 3D backends now exist (brute force, spatial hash, tree).
    let mut reference: Option<Vec<(Vec3, Vec3)>> = None;
    for mode in [
        BroadphaseMode::BruteForce,
        BroadphaseMode::SpatialHash,
        BroadphaseMode::DynamicAABBTree,
    ] {
        let mut w = scene_world3();
        w.broadphase_mode = mode;
        for _ in 0..240 {
            w.step(1.0 / 60.0);
        }
        let fp = fingerprint3(&w);
        assert_eq!(w.bodies.len(), 9);
        assert!(fp.iter().all(|(p, _)| p.x.is_finite() && p.y.is_finite()));
        match &reference {
            None => reference = Some(fp),
            Some(r) => assert_eq!(
                *r,
                fp,
                "3D backend {} diverged from the reference",
                mode.name()
            ),
        }
    }
}

/// Build a scene with exactly `count` bodies, all positioned identically
/// regardless of how the count was reached. Both sides of a comparison must
/// come from this one builder, otherwise the *scenes* differ and the test
/// is measuring the wrong thing.
fn scene_world2_sized(count: usize) -> World2 {
    let mut w = scene_world2();
    // Keep the ground plus as many pile bodies as fit.
    while w.bodies.len() > count {
        let last = w.bodies.len() - 1;
        w.remove_body(last);
    }
    // Park any surplus far away and out of contact with the pile so it
    // cannot perturb the settling bodies.
    while w.bodies.len() < count {
        let n = w.bodies.len();
        w.add_dynamic_body(
            Vec2::new(200.0 + n as f32 * 2.0, 200.0),
            Shape2::Circle { radius: 0.25 },
            1.0,
        )
        .expect("valid shape");
    }
    w
}

/// B03: Hybrid resolves to the same backend the thresholds name, and the
/// simulation matches the explicitly-selected backend.
#[test]
fn b03_hybrid_matches_resolved_backend() {
    for count in [3usize, 60, 200] {
        let expected = BroadphaseMode::auto_select(count);

        let mut hybrid = scene_world2_sized(count);
        hybrid.broadphase_mode = BroadphaseMode::Hybrid;
        for _ in 0..60 {
            hybrid.step(1.0 / 60.0);
        }

        let mut explicit = scene_world2_sized(count);
        explicit.broadphase_mode = expected;
        explicit.broadphase = phieng_broadphase::Broadphase2::new(expected, 1.0);
        for _ in 0..60 {
            explicit.step(1.0 / 60.0);
        }

        assert_eq!(
            fingerprint2(&hybrid),
            fingerprint2(&explicit),
            "Hybrid must match its explicitly-selected backend at count={}",
            count
        );
    }
}

/// B04: default world keeps the pre-tier-system backend, so upgrading is
/// a no-op for existing callers.
#[test]
fn b04_default_is_spatial_hash() {
    let w2 = World2::new();
    assert_eq!(w2.broadphase_mode, BroadphaseMode::SpatialHash);
    assert!(matches!(
        w2.broadphase,
        phieng_broadphase::Broadphase2::SpatialHash(_)
    ));

    let w3 = World3::new();
    assert_eq!(w3.broadphase_mode, BroadphaseMode::SpatialHash);
    // 3D has a real grid backend now (it used to fall back to all-pairs).
    assert!(matches!(
        w3.broadphase,
        phieng_broadphase::Broadphase3::SpatialHash(_)
    ));
}

/// B05: switching mode mid-simulation preserves the trajectory.
///
/// This is the contract that makes the tier safe to change at runtime
/// (e.g. an auto-scaler reacting to body count): a switch must not inject
/// energy or drop contacts.
#[test]
fn b05_mode_switch_mid_run_is_transparent() {
    let mut a = scene_world2();
    for _ in 0..120 {
        a.step(1.0 / 60.0);
    }

    let mut b = scene_world2();
    // Same first half...
    for _ in 0..120 {
        b.step(1.0 / 60.0);
    }
    // ...then switch backend, and run the same second half.
    b.broadphase_mode = BroadphaseMode::DynamicAABBTree;
    b.broadphase = phieng_broadphase::Broadphase2::new(BroadphaseMode::DynamicAABBTree, 1.0);
    for _ in 0..120 {
        b.step(1.0 / 60.0);
    }
    for _ in 0..120 {
        a.step(1.0 / 60.0);
    }

    assert_eq!(
        fingerprint2(&a),
        fingerprint2(&b),
        "switching broadphase mid-run must not perturb the trajectory"
    );
}

/// B06: all backends keep body order and the id == index invariant.
///
/// Broadphase backends emit body ids, not indices; if any backend ever
/// reorders or drops an id the contact cache would corrupt (warming the
/// wrong pair). Pin the invariant across modes.
#[test]
fn b06_body_id_invariant_across_modes() {
    for mode in MODES {
        let mut w = scene_world2();
        w.broadphase_mode = mode;
        for _ in 0..60 {
            w.step(1.0 / 60.0);
            for (i, b) in w.bodies.iter().enumerate() {
                assert_eq!(b.id, i, "id != index under mode {}", mode.name());
            }
        }
    }
}
