//! Ray-cast tests (X07–X12, mirrored Rust↔C#).
//!
//! A ray cast answers "what does this ray hit first?". These tests pin
//! that the ray finds the body in front of it, that it misses what is
//! not there, that the collision filter applies, and that the shape
//! math agrees between the Rust core and the managed mirror.

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
    );
    assert!(hit.is_none(), "the ray must miss");
}

/// X09: the collision filter applies to a ray cast.
#[test]
fn x09_raycast_respects_the_collision_filter() {
    let (mut w, floor, _ball) = scene_2d();
    // The floor moves to a group the query does not accept.
    w.bodies[floor].collision_group = 2;
    w.bodies[floor].collision_mask = 2;

    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
        &QueryFilter::everything(),
    );
    assert!(hit.is_none(), "the filtered floor must not be hit");

    // A query on the floor's own group finds it.
    let hit = w.raycast(
        &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
        &QueryFilter { group: 2, mask: 2 },
    );
    assert!(hit.is_some(), "a query on the floor's group must find it");
}

/// X10: the nearest of two bodies wins.
#[test]
fn x10_raycast_returns_the_nearest_body() {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    let far = w
        .add_static_body(
            glam::Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let near = w
        .add_static_body(
            glam::Vec2::new(0.0, 4.0),
            Shape2::Box {
                half_extents: glam::Vec2::new(1.0, 0.5),
            },
        )
        .expect("valid shape");
    let hit = w
        .raycast(
            &Ray2::new(glam::Vec2::new(0.0, 10.0), glam::Vec2::new(0.0, -1.0)),
            &QueryFilter::everything(),
        )
        .expect("the ray must hit something");
    assert_eq!(hit.body, near as u32, "the nearer body wins");
    assert_ne!(hit.body, far as u32);
}

/// X11: the same in 3D, against a sphere and a box.
#[test]
fn x11_raycast_in_3d() {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    let box_id = w
        .add_static_body(
            glam::Vec3::new(0.0, 0.0, -10.0),
            Shape3::Box {
                half_extents: glam::Vec3::new(10.0, 0.5, 10.0),
            },
        )
        .expect("valid shape");
    let sphere = w
        .add_dynamic_body(
            glam::Vec3::new(0.0, 0.0, 5.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");

    let hit = w
        .raycast(
            &Ray3::new(
                glam::Vec3::new(0.0, 0.0, 10.0),
                glam::Vec3::new(0.0, 0.0, -1.0),
            ),
            &QueryFilter::everything(),
        )
        .expect("the ray must hit something");
    assert_eq!(hit.body, sphere as u32, "the nearer body wins");
    assert_ne!(hit.body, box_id as u32);
    assert!((hit.t - 4.5).abs() < 1e-4, "t={}", hit.t);
    assert!((hit.point.z - 5.5).abs() < 1e-4, "point={:?}", hit.point);
}

/// X12: the native bridge answers the same ray cast.
#[test]
fn x12_ffi_raycast() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_new, phieng_world_raycast,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        // A static box (shape_tag 1) below the ray.
        let id = phieng_world_add_body(world, 0.0, -1.0, 1, 5.0, 0.5, 1.0, 1, 0.5, 0.0);
        assert!(id >= 0);

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
        assert_eq!(ok, 1, "the ray must hit the box");
        assert_eq!(out[0] as u32, id as u32);
        assert!((out[1] - 10.5).abs() < 1e-3, "t={}", out[1]);

        // A ray that misses.
        let ok = phieng_world_raycast(
            world,
            100.0,
            100.0,
            0.0,
            -1.0,
            100.0,
            1,
            u32::MAX,
            out.as_mut_ptr(),
        );
        assert_eq!(ok, 0, "the ray must miss");

        phieng_world_free(world);
    }
}
