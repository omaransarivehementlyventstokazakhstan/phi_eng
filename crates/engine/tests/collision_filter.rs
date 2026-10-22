//! Collision-filter tests (L01–L06, mirrored Rust↔C#).
//!
//! Every body carries a collision *layer* (the group it belongs to) and
//! a *mask* (the layers it accepts). Two bodies may collide only when
//! each accepts the other's layer — the rule every layered physics
//! engine uses. These tests pin:
//!   1. the default filter accepts everything, so a scene that never
//!      configures a filter behaves exactly as it did before layers
//!      existed (and the trajectory hash is unchanged),
//!   2. a layer that does not accept the other side really does stop
//!      the pair from producing a contact,
//!   3. the filter survives a snapshot round trip, and a file written
//!      before the feature existed loads with the default,
//!   4. the native bridge round-trips the filter.

use phieng_dynamics::body::{collision_filter_allows, DEFAULT_COLLISION_GROUP};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

const DT: f32 = 1.0 / 60.0;

fn hash2(w: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &w.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// A ball dropped on a static floor. The ball rests on the floor when
/// the pair collides and falls through when the filter rejects it.
fn drop_2d() -> (World2, usize, usize) {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.add_static_body(
        glam::Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    let ball = w
        .add_dynamic_body(
            glam::Vec2::new(0.0, 1.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    (w, ball, 0)
}

/// The same scene in 3D.
fn drop_3d() -> (World3, usize, usize) {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.add_static_body(
        glam::Vec3::new(0.0, -1.0, 0.0),
        Shape3::Box {
            half_extents: glam::Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");
    let ball = w
        .add_dynamic_body(
            glam::Vec3::new(0.0, 1.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    (w, ball, 0)
}

/// The default filter accepts every layer, so a scene that never
/// configures one behaves exactly as before.
#[test]
fn l01_default_filter_collides_with_everything() {
    let (mut w, _ball, _) = drop_2d();
    assert_eq!(w.bodies[0].collision_group, DEFAULT_COLLISION_GROUP);
    // The default mask accepts every layer.
    assert!(collision_filter_allows(
        w.bodies[0].collision_group,
        w.bodies[0].collision_mask,
        w.bodies[1].collision_group,
        w.bodies[1].collision_mask,
    ));

    // The ball comes to rest on the floor: it stops falling. The
    // floor's top surface is at y = -0.5 and the ball's radius is 0.5,
    // so a resting ball sits at y = 0.
    for _ in 0..240 {
        w.step(DT);
    }
    let ball_y = w.bodies[1].transform.position.y;
    assert!(
        ball_y > -0.2,
        "the ball must rest on the floor, not fall through it (y={ball_y})"
    );
}

/// A body whose mask does not accept the other body's layer does not
/// collide with it: the pair produces no contact at all.
#[test]
fn l02_a_layer_that_does_not_accept_rejects_the_pair() {
    let (mut w, ball, _) = drop_2d();
    // The floor stays on the default layer; the ball moves to a layer
    // the floor does not accept, and the floor does not accept the
    // ball's layer either.
    w.bodies[ball].collision_group = 2;
    w.bodies[ball].collision_mask = 1; // accepts only layer 1
    w.bodies[0].collision_mask = 1; // accepts only layer 1 (the ball is on 2)

    for _ in 0..240 {
        w.step(DT);
    }
    let ball_y = w.bodies[1].transform.position.y;
    assert!(
        ball_y < -1.0,
        "the filtered ball must fall through the floor (y={ball_y})"
    );
}

/// One side is enough to veto the pair: a body that accepts only its
/// own layer does not collide with a body on another layer, even
/// though the other side would accept it.
#[test]
fn l03_one_sided_veto_stops_the_pair() {
    let (mut w, ball, _) = drop_2d();
    // The floor still accepts everything; the ball accepts only itself.
    // A group is one bit, so the ball's group is the flag 0x2 and a mask
    // that accepts it holds that same bit.
    w.bodies[ball].collision_group = 2;
    w.bodies[ball].collision_mask = 2;

    for _ in 0..240 {
        w.step(DT);
    }
    let ball_y = w.bodies[1].transform.position.y;
    assert!(
        ball_y < -1.0,
        "the ball's own veto must stop the pair (y={ball_y})"
    );
}

/// The same rule in 3D.
#[test]
fn l04_filter_applies_in_3d() {
    let (mut w, ball, _) = drop_3d();
    w.bodies[ball].collision_group = 2;
    w.bodies[ball].collision_mask = 1;
    w.bodies[0].collision_mask = 1;

    for _ in 0..240 {
        w.step(DT);
    }
    let ball_y = w.bodies[1].transform.position.y;
    assert!(
        ball_y < -1.0,
        "the filtered ball must fall through the floor (y={ball_y})"
    );
}

/// The filter survives a snapshot round trip, and a file written
/// before the feature existed loads with the default.
#[test]
fn l05_filter_survives_a_snapshot_round_trip() {
    let (mut w, ball, _) = drop_2d();
    w.bodies[ball].collision_group = 7;
    w.bodies[ball].collision_mask = 0b1010;

    let json = phieng_dynamics::world2_to_json(&w, phieng_dynamics::LoadMode::Exact)
        .expect("serializable");
    let restored = phieng_dynamics::world2_from_json(&json, phieng_dynamics::LoadMode::Exact)
        .expect("loadable");
    assert_eq!(restored.bodies[ball].collision_group, 7);
    assert_eq!(restored.bodies[ball].collision_mask, 0b1010);
    assert_eq!(hash2(&restored), hash2(&w));

    // A world whose bodies all carry the default filter writes the
    // oldest schema, so a file without the fields still loads.
    let plain = drop_2d().0;
    let plain_json = phieng_dynamics::world2_to_json(&plain, phieng_dynamics::LoadMode::Exact)
        .expect("serializable");
    assert!(
        !plain_json.contains("collision_group"),
        "a default world must not carry the filter"
    );
    let plain_restored =
        phieng_dynamics::world2_from_json(&plain_json, phieng_dynamics::LoadMode::Exact)
            .expect("loadable");
    assert_eq!(
        plain_restored.bodies[1].collision_group,
        DEFAULT_COLLISION_GROUP
    );
}

/// The native bridge round-trips the filter.
#[test]
fn l06_ffi_round_trips_the_filter() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_get_body_collision_filter,
        phieng_world_new, phieng_world_set_body_collision_filter, phieng_world_step,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        // shape_tag 0 is a circle (radius in `a`).
        let id = phieng_world_add_body(world, 0.0, 1.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0) as u32;
        assert!(id != u32::MAX, "the body must be created");
        assert_eq!(
            phieng_world_set_body_collision_filter(world, id, 5, 0b1100),
            1
        );
        let mut group: u32 = 0;
        let mut mask: u32 = 0;
        assert_eq!(
            phieng_world_get_body_collision_filter(world, id, &mut group, &mut mask),
            1
        );
        assert_eq!(group, 5);
        assert_eq!(mask, 0b1100);
        // A step must not disturb the filter.
        phieng_world_step(world, DT);
        assert_eq!(
            phieng_world_get_body_collision_filter(world, id, &mut group, &mut mask),
            1
        );
        assert_eq!(group, 5);
        assert_eq!(mask, 0b1100);
        phieng_world_free(world);
    }
}
