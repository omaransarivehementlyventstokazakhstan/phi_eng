//! The 3D kinematic character controller (O07–O12, mirrored Rust↔C#).
//!
//! The 3D contract mirrors the 2D one: exact free moves, wall stops
//! without tunnelling, sliding that preserves the free axes, ground from
//! a probe with a slope limit, filter-selected solidity — plus what is
//! specific to 3D: a capsule body is solid through its analytic query,
//! and an unsupported convex hull is walked through (the documented
//! closest-point gap).

use glam::Vec3;
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::character3::{move_character3, CharacterConfig3};
use phieng_dynamics::{Shape3, World3};

fn floor_world() -> World3 {
    let mut w = World3::new();
    w.gravity = Vec3::ZERO;
    w.add_static_body(
        Vec3::new(0.0, -0.5, 0.0),
        Shape3::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");
    w
}

fn wall_world() -> World3 {
    let mut w = World3::new();
    w.gravity = Vec3::ZERO;
    w.add_static_body(
        Vec3::new(2.5, 1.0, 0.0),
        Shape3::Box {
            half_extents: Vec3::new(0.5, 3.0, 3.0),
        },
    )
    .expect("valid shape");
    w
}

fn ramp_world(angle: f32) -> World3 {
    let mut w = World3::new();
    w.gravity = Vec3::ZERO;
    w.add_static_body(
        Vec3::new(2.0, 0.0, 0.0),
        Shape3::Box {
            half_extents: Vec3::new(2.0, 0.3, 2.0),
        },
    )
    .expect("valid shape");
    w.bodies[0].transform.rotation = glam::Quat::from_rotation_z(angle);
    w
}

/// O07: a free move is exact with no contacts; a resting character
/// reports ground.
#[test]
fn o07_a_free_move_is_exact() {
    let w = floor_world();
    let cfg = CharacterConfig3::default();
    let m = move_character3(&w, Vec3::ZERO, &cfg, Vec3::X, &QueryFilter::everything());
    assert_eq!(m.translation, Vec3::X);
    assert_eq!(m.contacts, 0);
    assert!(m.grounded);
    assert_eq!(m.ground_normal, Vec3::Y);

    let air = move_character3(
        &w,
        Vec3::new(0.0, 5.0, 0.0),
        &cfg,
        -Vec3::Y,
        &QueryFilter::everything(),
    );
    assert_eq!(air.translation, -Vec3::Y);
    assert!(!air.grounded);
}

/// O08: a head-on move stops one radius short of the wall face.
#[test]
fn o08_a_head_on_move_stops_at_the_wall() {
    let w = wall_world();
    let cfg = CharacterConfig3::default();
    let m = move_character3(
        &w,
        Vec3::ZERO,
        &cfg,
        Vec3::new(5.0, 0.0, 0.0),
        &QueryFilter::everything(),
    );
    assert!(
        (m.translation.x - 1.6).abs() < 2e-3,
        "stopped at {}",
        m.translation.x
    );
    assert_eq!(m.translation.y, 0.0);
    assert_eq!(m.translation.z, 0.0);
    assert!(m.contacts > 0);
}

/// O09: a diagonal move loses only the blocked component; the other
/// two axes survive exactly.
#[test]
fn o09_a_diagonal_move_slides() {
    let w = wall_world();
    let cfg = CharacterConfig3::default();
    let m = move_character3(
        &w,
        Vec3::ZERO,
        &cfg,
        Vec3::new(5.0, 1.0, 0.5),
        &QueryFilter::everything(),
    );
    assert!(
        (m.translation.x - 1.6).abs() < 2e-3,
        "x blocked at {}",
        m.translation.x
    );
    assert!(
        (m.translation.y - 1.0).abs() < 2e-3,
        "y got {}",
        m.translation.y
    );
    assert!(
        (m.translation.z - 0.5).abs() < 2e-3,
        "z got {}",
        m.translation.z
    );
}

/// O10: ground needs a walkable surface — a floor grounds, a wall does
/// not, and a 60-degree ramp is too steep.
#[test]
fn o10_ground_needs_a_walkable_surface() {
    let cfg = CharacterConfig3::default();
    let floor = move_character3(
        &floor_world(),
        Vec3::ZERO,
        &cfg,
        Vec3::ZERO,
        &QueryFilter::everything(),
    );
    assert!(floor.grounded);
    assert_eq!(floor.ground_normal, Vec3::Y);

    let wall = wall_world();
    let beside = move_character3(
        &wall,
        Vec3::new(1.6, 0.0, 0.0),
        &cfg,
        Vec3::ZERO,
        &QueryFilter::everything(),
    );
    assert!(!beside.grounded, "a wall is not ground");

    let steep = move_character3(
        &ramp_world(std::f32::consts::FRAC_PI_3),
        Vec3::new(2.0, 0.5, 0.0),
        &cfg,
        Vec3::ZERO,
        &QueryFilter::everything(),
    );
    assert!(!steep.grounded, "a 60-degree slope is too steep for ground");
}

/// O11: the filter selects solid bodies, and a capsule body is solid
/// through its analytic query.
#[test]
fn o11_the_filter_and_capsule_solidity() {
    let cfg = CharacterConfig3::default();
    let mut w = wall_world();
    w.bodies[0].collision_group = 2;
    let through = move_character3(
        &w,
        Vec3::ZERO,
        &cfg,
        Vec3::new(5.0, 0.0, 0.0),
        &QueryFilter { group: 1, mask: 1 },
    );
    assert!((through.translation - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    assert_eq!(through.contacts, 0);

    let blocked = move_character3(
        &w,
        Vec3::ZERO,
        &cfg,
        Vec3::new(5.0, 0.0, 0.0),
        &QueryFilter { group: 1, mask: 2 },
    );
    assert!((blocked.translation.x - 1.6).abs() < 2e-3);

    // A capsule wall: shaft face at x = 1.7, radius 0.3 => 1.4.
    let mut cw = World3::new();
    cw.gravity = Vec3::ZERO;
    cw.add_static_body(
        Vec3::new(2.0, 0.0, 0.0),
        Shape3::Capsule {
            radius: 0.3,
            half_length: 1.0,
        },
    )
    .expect("valid shape");
    let capsule = move_character3(
        &cw,
        Vec3::ZERO,
        &cfg,
        Vec3::new(4.0, 0.0, 0.0),
        &QueryFilter::everything(),
    );
    assert!(
        (capsule.translation.x - 1.3).abs() < 2e-3,
        "capsule stop at {}",
        capsule.translation.x
    );
}

/// O12: a walkable ramp is climbed and grounds; an unsupported convex
/// hull is walked through (the documented query gap).
#[test]
fn o12_ramps_climb_and_hulls_are_not_solid() {
    let cfg = CharacterConfig3::default();
    let m = move_character3(
        &ramp_world(std::f32::consts::FRAC_PI_4),
        Vec3::ZERO,
        &cfg,
        Vec3::new(3.0, 0.0, 0.0),
        &QueryFilter::everything(),
    );
    assert!(
        m.translation.y > 0.9,
        "the slide must climb, got {}",
        m.translation.y
    );
    assert!(m.grounded, "a 45-degree ramp is walkable");

    let mut hull_world = World3::new();
    hull_world.gravity = Vec3::ZERO;
    hull_world
        .add_static_body(
            Vec3::new(2.0, 0.0, 0.0),
            Shape3::ConvexHull {
                vertices: vec![
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(3.0, -1.0, -1.0),
                    Vec3::new(2.0, 1.0, -1.0),
                    Vec3::new(2.0, -1.0, 1.0),
                ],
            },
        )
        .expect("valid shape");
    let through = move_character3(
        &hull_world,
        Vec3::ZERO,
        &cfg,
        Vec3::new(4.0, 0.0, 0.0),
        &QueryFilter::everything(),
    );
    assert!(
        (through.translation - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-5,
        "a hull is walked through (documented gap)"
    );
}
