//! Ray-cast tests (X07–X16, mirrored Rust↔C#).
//!
//! A ray cast answers "what does this ray hit first?". These tests pin
//! that the ray finds the body in front of it, that it misses what is
//! not there, that the collision filter applies, that a sensor is not
//! solid, that the ray has a range, and that the shape math agrees
//! between the Rust core and the managed mirror.

use phieng_collision::{Ray2, Ray3};
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::{Shape2, Shape3, World2, World3};

/// A floor with a ball above it and a box beside it.
fn scene_2d() -> (World2, usize, usize) {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    let floor = w
        .add_static_body(
            glam::Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let ball = w
        .add_dynamic_body(
            glam::Vec2::new(5.0, 2.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    (w, floor, ball)
}

/// X07: the ray finds the body in front of it.
#[test]
fn x07_raycast_finds_the_body_in_front() {
    let (w, floor, _ball) = scene_2d();
    let hit = w
        .raycast(
            &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
            &QueryFilter::everything(),
            f32::MAX,
        )
        .expect("the ray must hit the floor");
    assert_eq!(hit.body, floor as u32);
    assert!((hit.t - 10.5).abs() < 1e-4, "t={}", hit.t);
    assert!((hit.point.y + 0.5).abs() < 1e-4, "point={:?}", hit.point);
    assert!((hit.normal - glam::Vec2::Y).length() < 1e-4);
}

/// X08: a ray that misses everything hits nothing.
#[test]
fn x08_raycast_misses() {
    let (w, _floor, _ball) = scene_2d();
    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(100.0, 100.0), glam::Vec2::new(0.0, -1.0)),
        &QueryFilter::everything(),
        f32::MAX,
    );
    assert!(hit.is_none(), "the ray must miss");
}

/// X09: the collision filter applies to a ray cast.
#[test]
fn x09_raycast_applies_the_collision_filter() {
    let (mut w, floor, _ball) = scene_2d();
    w.bodies[floor].collision_group = 2;
    w.bodies[floor].collision_mask = 2;
    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
        &QueryFilter::everything(),
        f32::MAX,
    );
    assert!(hit.is_none(), "the filtered floor must not be hit");

    // A query on the floor's own group finds it.
    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
        &QueryFilter { group: 2, mask: 2 },
        f32::MAX,
    );
    assert!(hit.is_some(), "a query on the floor's group must find it");
}

/// X10: the nearest of two bodies wins.
#[test]
fn x10_raycast_reports_the_nearest_body() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, 0.0);
    let near = w
        .add_static_body(
            glam::Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let far = w
        .add_static_body(
            glam::Vec2::new(0.0, -5.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let hit = w
        .raycast(
            &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
            &QueryFilter::everything(),
            f32::MAX,
        )
        .expect("the ray must hit something");
    assert_eq!(hit.body, near as u32, "the nearer body wins");
    assert_ne!(hit.body, far as u32);
}

/// X11: the 3D mirror.
#[test]
fn x11_raycast_3d() {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    let sphere = w
        .add_static_body(
            glam::Vec3::new(0.0, 0.0, 0.0),
            Shape3::Sphere { radius: 1.0 },
        )
        .expect("valid shape");
    let hit = w
        .raycast(
            &Ray3::new(
                glam::Vec3::new(0.0, 0.0, 10.0),
                glam::Vec3::new(0.0, 0.0, -1.0),
            ),
            &QueryFilter::everything(),
            f32::MAX,
        )
        .expect("the ray must hit something");
    assert_eq!(hit.body, sphere as u32, "the nearer body wins");
}

/// X12: the FFI round trip.
#[test]
fn x12_ffi_raycast() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_new, phieng_world_raycast,
    };
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 0.0, 0, 0.5, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(id >= 0, "the body must be added");
        let mut out = [0.0f32; 6];
        let ok = phieng_world_raycast(
            world,
            0.0,
            10.0,
            0.0,
            -1.0,
            100.0,
            1,
            u32::MAX,
            out.as_mut_ptr(),
        );
        assert_eq!(ok, 1, "the ray must hit the body");
        assert_eq!(out[0] as u32, id as u32, "the body id comes back");
        assert!((out[1] - 9.5).abs() < 1e-4, "t={}", out[1]);
        phieng_world_free(world);
    }
}

/// X13: a ray that starts inside a shape does not hit it.
///
/// The hit is for a ray that *enters*: a ray that starts inside would
/// otherwise report the way out, with the normal pointing away from
/// the ray.
#[test]
fn x13_raycast_from_inside_a_shape_misses() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, 0.0);
    w.add_static_body(glam::Vec2::new(0.0, 0.0), Shape2::Circle { radius: 1.0 })
        .expect("valid shape");
    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1.0, 0.0)),
        &QueryFilter::everything(),
        f32::MAX,
    );
    assert!(hit.is_none(), "a ray that starts inside must not hit it");

    // The same in 3D.
    let mut w3 = World3::new();
    w3.gravity = glam::Vec3::new(0.0, 0.0, 0.0);
    w3.add_static_body(
        glam::Vec3::new(0.0, 0.0, 0.0),
        Shape3::Sphere { radius: 1.0 },
    )
    .expect("valid shape");
    let hit = w3.raycast(
        &Ray3::new(
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
        ),
        &QueryFilter::everything(),
        f32::MAX,
    );
    assert!(hit.is_none(), "a ray that starts inside must not hit it");
}

/// X14: a sensor is not solid: the ray passes through it.
#[test]
fn x14_raycast_passes_through_a_sensor() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, 0.0);
    let sensor = w
        .add_static_body(
            glam::Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(1.0, 1.0),
            },
        )
        .expect("valid shape");
    w.bodies[sensor].is_sensor = true;
    let floor = w
        .add_static_body(
            glam::Vec2::new(0.0, -5.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let hit = w
        .raycast(
            &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
            &QueryFilter::everything(),
            f32::MAX,
        )
        .expect("the ray must hit the floor");
    assert_eq!(hit.body, floor as u32, "the sensor must not stop the ray");
}

/// X15: the ray has a range.
#[test]
fn x15_raycast_respects_its_range() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, 0.0);
    w.add_static_body(
        glam::Vec2::new(0.0, 0.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(1.0, 1.0),
        },
    )
    .expect("valid shape");
    let ray = Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0));
    let hit = w.raycast(&ray, &QueryFilter::everything(), 5.0);
    assert!(hit.is_none(), "the body is out of the ray's range");
    let hit = w.raycast(&ray, &QueryFilter::everything(), 50.0);
    assert!(hit.is_some(), "the body is within the ray's range");
}

/// X16: two bodies on the same group collide whatever their masks say.
///
/// This is Box2D's rule: the group overrides the masks.
#[test]
fn x16_the_same_group_overrides_the_masks() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    let a = w
        .add_dynamic_body(
            glam::Vec2::new(0.0, 0.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    let b = w
        .add_dynamic_body(
            glam::Vec2::new(0.9, 0.0),
            Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    // Both on group 2, and neither mask accepts group 2.
    w.bodies[a].collision_group = 2;
    w.bodies[a].collision_mask = 1;
    w.bodies[b].collision_group = 2;
    w.bodies[b].collision_mask = 1;
    let start_distance = (w.bodies[b].transform.position - w.bodies[a].transform.position).length();
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    let end_distance = (w.bodies[b].transform.position - w.bodies[a].transform.position).length();
    assert!(
        end_distance > start_distance + 0.1,
        "the two bodies on the same group must push each other apart (start={}, end={})",
        start_distance,
        end_distance
    );
}
