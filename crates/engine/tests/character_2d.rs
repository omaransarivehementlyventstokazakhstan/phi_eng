//! The 2D kinematic character controller (O01–O06, mirrored Rust↔C#).
//!
//! The contract under test: advance freely when nothing is in the way,
//! stop at the first surface, slide the remainder along it, never
//! tunnel (substeps), stay free while resting on a surface (contacts
//! only block motion into them), report ground from a probe, respect
//! the query filter, and climb walkable slopes while refusing steep
//! ones as ground.

use glam::Vec2;
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::character::{move_character2, CharacterConfig2};
use phieng_dynamics::{Shape2, World2};

fn floor_world() -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(
        Vec2::new(0.0, -0.5),
        Shape2::Box {
            half_extents: Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    w
}

fn wall_world() -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(
        Vec2::new(2.5, 0.0),
        Shape2::Box {
            half_extents: Vec2::new(0.5, 3.0),
        },
    )
    .expect("valid shape");
    w
}

fn ramp_world(angle: f32) -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(
        Vec2::new(2.0, 0.0),
        Shape2::Box {
            half_extents: Vec2::new(2.0, 0.3),
        },
    )
    .expect("valid shape");
    w.bodies[0].transform.rotation = glam::Mat2::from_angle(angle);
    w
}

/// O01: a free move is applied exactly, with no contacts; a resting
/// character still reports ground.
#[test]
fn o01_a_free_move_is_exact() {
    let w = floor_world();
    let cfg = CharacterConfig2::default();
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(1.0, 0.0),
        &QueryFilter::everything(),
    );
    assert_eq!(m.translation, Vec2::new(1.0, 0.0));
    assert_eq!(m.contacts, 0);
    assert!(m.grounded, "a character on the floor is grounded");
    assert_eq!(m.ground_normal, Vec2::Y);

    // In mid-air the same move is free and not grounded.
    let m = move_character2(
        &w,
        Vec2::new(0.0, 5.0),
        &cfg,
        Vec2::new(0.0, -1.0),
        &QueryFilter::everything(),
    );
    assert_eq!(m.translation, Vec2::new(0.0, -1.0));
    assert!(!m.grounded);
}

/// O02: a head-on move stops at the wall, one radius short of its face,
/// and never tunnels — even at speeds several radii per call.
#[test]
fn o02_a_head_on_move_stops_at_the_wall() {
    let w = wall_world();
    let cfg = CharacterConfig2::default();
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(5.0, 0.0),
        &QueryFilter::everything(),
    );
    // Face at x = 2, radius 0.4 => 1.6, less the tiny separation skin.
    assert!(
        (m.translation.x - 1.6).abs() < 2e-3,
        "stopped at {}",
        m.translation.x
    );
    assert_eq!(m.translation.y, 0.0);
    assert!(m.contacts > 0, "the wall must register as contacts");
}

/// O03: a diagonal move into a wall loses only the blocked component.
#[test]
fn o03_a_diagonal_move_slides() {
    let w = wall_world();
    let cfg = CharacterConfig2::default();
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(5.0, 1.0),
        &QueryFilter::everything(),
    );
    assert!(
        (m.translation.x - 1.6).abs() < 2e-3,
        "x blocked at {}",
        m.translation.x
    );
    assert!(
        (m.translation.y - 1.0).abs() < 2e-3,
        "y must survive the slide, got {}",
        m.translation.y
    );
}

/// O04: ground comes from the probe and from the slope rule, not from
/// merely touching something.
#[test]
fn o04_ground_needs_a_walkable_surface() {
    let cfg = CharacterConfig2::default();
    // Floor: grounded, normal up.
    let w = floor_world();
    let m = move_character2(&w, Vec2::ZERO, &cfg, Vec2::ZERO, &QueryFilter::everything());
    assert!(m.grounded);
    assert_eq!(m.ground_normal, Vec2::Y);

    // Beside a wall (not above anything): not grounded.
    let w = wall_world();
    let m = move_character2(
        &w,
        Vec2::new(1.6, 0.0),
        &cfg,
        Vec2::ZERO,
        &QueryFilter::everything(),
    );
    assert!(!m.grounded, "a wall is not ground");

    // A 60-degree ramp exceeds the default 50-degree walkable limit.
    let w = ramp_world(std::f32::consts::FRAC_PI_3);
    let m = move_character2(
        &w,
        Vec2::new(2.0, 0.5),
        &cfg,
        Vec2::ZERO,
        &QueryFilter::everything(),
    );
    assert!(!m.grounded, "a 60-degree slope is too steep for ground");
}

/// O05: the query filter selects what is solid for the character.
#[test]
fn o05_the_filter_selects_solid_bodies() {
    let mut w = wall_world();
    let cfg = CharacterConfig2::default();
    // A wall on its own layer is walked through by a layer-1 filter...
    w.bodies[0].collision_group = 2;
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(5.0, 0.0),
        &QueryFilter { group: 1, mask: 1 },
    );
    assert!(
        (m.translation - Vec2::new(5.0, 0.0)).length() < 1e-5,
        "a filtered wall must be walked through, got {:?}",
        m.translation
    );
    assert_eq!(m.contacts, 0);
    // ...and is solid for one that names layer 2.
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(5.0, 0.0),
        &QueryFilter { group: 1, mask: 2 },
    );
    assert!((m.translation.x - 1.6).abs() < 2e-3);
}

/// O06: a walkable ramp is climbed (the slide carries the character up
/// the normal) and reports ground.
#[test]
fn o06_a_walkable_ramp_is_climbed() {
    let w = ramp_world(std::f32::consts::FRAC_PI_4);
    let cfg = CharacterConfig2::default();
    let m = move_character2(
        &w,
        Vec2::ZERO,
        &cfg,
        Vec2::new(3.0, 0.0),
        &QueryFilter::everything(),
    );
    assert!(
        m.translation.y > 0.9,
        "the slide must climb, got {}",
        m.translation.y
    );
    assert!(
        m.translation.x > 1.9,
        "x must still advance, got {}",
        m.translation.x
    );
    assert!(m.grounded, "a 45-degree ramp is walkable");
    assert!(m.ground_normal.y > 0.65, "ramp normal should point up");
}
