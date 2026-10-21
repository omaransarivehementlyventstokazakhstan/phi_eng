//! Scene-query tests (X01–X06, mirrored Rust↔C#).
//!
//! A scene query asks the world which bodies are near a place. These
//! tests pin:
//!   1. the query finds the bodies whose box overlaps the query box,
//!      in ascending id order,
//!   2. the collision filter applies to queries as it does to bodies,
//!   3. the query sees the *current* body positions, not the broad
//!      phase's grid (which holds the positions from before the last
//!      solve),
//!   4. the 3D mirror, the native bridge, and the agreement between
//!      the broad phase's own AABB query and the world's.

use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::{Shape2, Shape3, World2, World3};
use phieng_math::{AABB2, AABB3};

/// A floor and three balls at known places.
fn scene_2d() -> World2 {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    w.add_static_body(
        glam::Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    for x in [-5.0f32, 0.0, 5.0] {
        w.add_dynamic_body(glam::Vec2::new(x, 1.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
    }
    w
}

/// X01: the query returns the bodies whose box overlaps, in id order.
#[test]
fn x01_overlap_query_returns_the_bodies_in_the_box() {
    let w = scene_2d();
    // A box around the middle ball only.
    let hits = w.overlap_aabb(
        &AABB2::new(glam::Vec2::new(-1.0, 0.0), glam::Vec2::new(1.0, 2.0)),
        &QueryFilter::everything(),
    );
    assert_eq!(hits, vec![2], "only the middle ball is in the box");

    // A box around all of them.
    let hits = w.overlap_aabb(
        &AABB2::new(glam::Vec2::new(-10.0, 0.0), glam::Vec2::new(10.0, 2.0)),
        &QueryFilter::everything(),
    );
    assert_eq!(hits, vec![1, 2, 3], "the ids come back in ascending order");

    // A box around nothing.
    let hits = w.overlap_aabb(
        &AABB2::new(glam::Vec2::new(100.0, 100.0), glam::Vec2::new(101.0, 101.0)),
        &QueryFilter::everything(),
    );
    assert!(hits.is_empty());
}

/// X02: the collision filter applies to queries.
#[test]
fn x02_query_respects_the_collision_filter() {
    let mut w = scene_2d();
    // The middle ball moves to a group of its own that the query does
    // not accept. A group is one bit (the Box2D convention), so the
    // ball's group is the flag 0x2 and a mask that accepts only that
    // group has the same bit set.
    w.bodies[2].collision_group = 2;
    w.bodies[2].collision_mask = 2;

    let box_around_middle = AABB2::new(glam::Vec2::new(-1.0, 0.0), glam::Vec2::new(1.0, 2.0));

    // A query on the ball's own group finds it. A group is one bit
    // (the Box2D convention), so the ball's group 2 is the flag 0x2 and
    // a mask that accepts it has that bit set.
    let hits = w.overlap_aabb(&box_around_middle, &QueryFilter { group: 2, mask: 2 });
    assert_eq!(hits, vec![2]);

    // A query on the default group does not: the ball accepts only
    // its own group, and the query is not on it. The rule is mutual,
    // so the ball's own veto applies to a query as much as to a body.
    let hits = w.overlap_aabb(&box_around_middle, &QueryFilter::default());
    assert!(hits.is_empty(), "the ball's veto must apply to queries too");

    // A ball that accepts every group is found by a default query.
    w.bodies[2].collision_mask = u32::MAX;
    let hits = w.overlap_aabb(&box_around_middle, &QueryFilter::default());
    assert_eq!(hits, vec![2]);
}

/// X03: the query sees the current positions, not the last step's grid.
#[test]
fn x03_query_sees_the_current_positions() {
    let mut w = scene_2d();
    // Teleport the last ball far away by hand: the broad phase still
    // holds the position from the start of the last step, so a query
    // that used the grid would report the old place.
    w.bodies[3].transform.position = glam::Vec2::new(50.0, 50.0);
    w.bodies[3].velocity = glam::Vec2::ZERO;

    let old_place = w.overlap_aabb(
        &AABB2::new(glam::Vec2::new(4.0, 0.0), glam::Vec2::new(6.0, 2.0)),
        &QueryFilter::everything(),
    );
    assert!(
        old_place.is_empty(),
        "the ball has moved away from its old place"
    );

    let new_place = w.overlap_aabb(
        &AABB2::new(glam::Vec2::new(49.0, 49.0), glam::Vec2::new(51.0, 51.0)),
        &QueryFilter::everything(),
    );
    assert_eq!(
        new_place,
        vec![3],
        "the query must see where the ball is now"
    );
}

/// X04: the same query in 3D.
#[test]
fn x04_overlap_query_in_3d() {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    w.add_static_body(
        glam::Vec3::new(0.0, -1.0, 0.0),
        Shape3::Box {
            half_extents: glam::Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");
    for x in [-5.0f32, 0.0, 5.0] {
        w.add_dynamic_body(
            glam::Vec3::new(x, 1.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    }
    let hits = w.overlap_aabb(
        &AABB3::new(
            glam::Vec3::new(-1.0, 0.0, -1.0),
            glam::Vec3::new(1.0, 2.0, 1.0),
        ),
        &QueryFilter::everything(),
    );
    assert_eq!(hits, vec![2]);
}

/// X05: the native bridge answers the same query.
#[test]
fn x05_ffi_overlap_query() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_new, phieng_world_overlap_aabb,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        // shape_tag 0 is a circle (radius in `a`).
        let a = phieng_world_add_body(world, -5.0, 1.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        let b = phieng_world_add_body(world, 0.0, 1.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        let c = phieng_world_add_body(world, 5.0, 1.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(a >= 0 && b >= 0 && c >= 0);

        // A box around the middle ball.
        let mut ids = [0u32; 8];
        let count = phieng_world_overlap_aabb(
            world,
            -1.0,
            0.0,
            1.0,
            2.0,
            1,
            u32::MAX,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(count, 1, "only the middle ball is in the box");
        assert_eq!(ids[0], b as u32);

        // A box around all of them.
        let count = phieng_world_overlap_aabb(
            world,
            -10.0,
            0.0,
            10.0,
            2.0,
            1,
            u32::MAX,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(count, 3);
        assert_eq!(ids[0], a as u32);
        assert_eq!(ids[1], b as u32);
        assert_eq!(ids[2], c as u32);

        phieng_world_free(world);
    }
}

/// X06: every broad-phase backend answers the AABB query the same way.
#[test]
fn x06_every_broadphase_backend_agrees() {
    use phieng_broadphase::{Broadphase2, BroadphaseMode};

    let boxes: Vec<(glam::Vec2, glam::Vec2)> = vec![
        (glam::Vec2::new(-5.0, -1.0), glam::Vec2::new(-4.0, 0.0)),
        (glam::Vec2::new(-0.5, -1.0), glam::Vec2::new(0.5, 0.0)),
        (glam::Vec2::new(4.0, -1.0), glam::Vec2::new(5.0, 0.0)),
        (glam::Vec2::new(0.0, 3.0), glam::Vec2::new(1.0, 4.0)),
    ];
    let query = AABB2::new(glam::Vec2::new(-1.0, -2.0), glam::Vec2::new(1.0, 5.0));

    let mut answers = Vec::new();
    for mode in [
        BroadphaseMode::SpatialHash,
        BroadphaseMode::BruteForce,
        BroadphaseMode::DynamicAABBTree,
    ] {
        let mut bp = Broadphase2::new(mode, 1.0);
        for (id, (min, max)) in boxes.iter().enumerate() {
            bp.insert(id, *min, *max);
        }
        let mut hits = Vec::new();
        bp.query_aabb(&query, &mut hits);
        answers.push(hits);
    }
    assert_eq!(answers[0], vec![1, 3], "the grid finds the two boxes");
    for hits in &answers[1..] {
        assert_eq!(*hits, answers[0], "every backend must answer alike");
    }
}
