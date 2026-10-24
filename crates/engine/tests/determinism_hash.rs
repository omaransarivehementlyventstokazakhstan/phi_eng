//! Absolute determinism anchor.
//!
//! Every other determinism test in this repo is *relative*: it checks
//! that two runs of the same build agree (run1 == run2), that the
//! serial and parallel solvers agree, that the broadphase backends
//! agree, or that a snapshot round-trips. Those are strong, but a bug
//! that shifted every trajectory by the same amount would pass all of
//! them.
//!
//! This module pins the *absolute* trajectory hash of a canonical
//! scene, so any change to the physics — even one that keeps every
//! relative invariant intact — fails loudly.
//!
//! The scene is deliberately small and cheap: this test runs in the
//! normal (debug) test suite on every platform, so it must stay fast
//! while still exercising the interesting paths (boxes and circles, a
//! static floor, stacking, contact warm-starting, sleeping, and the
//! contact cache).
//!
//! The large benchmark scenes are pinned separately, in the
//! `bench_large` example, which is where the release-only performance
//! numbers come from.

use glam::{Vec2, Vec3};
use phieng_dynamics::body::{Shape2, Shape3};
use phieng_dynamics::{World2, World3};

/// Fixed timestep used by every scene here.
const DT: f32 = 1.0 / 60.0;

/// Number of steps the reference scenes are advanced by.
const STEPS: usize = 240;

/// Body count of the 2D reference scene.
const BODIES_2D: usize = 90;

/// Body count of the 3D reference scene.
const BODIES_3D: usize = 60;

/// Pinned trajectory hash of the 2D reference scene.
///
/// Re-pinning is a deliberate act: if this test fails, the physics
/// changed. Either the change is intended (then update this constant
/// and say so in the CHANGELOG) or it is a regression (then fix the
/// physics). Never silence it by loosening the comparison.
const HASH_2D: u64 = 0xb12fcb9a19506312;

const HASH_3D: u64 = 0x2a03c4fa2060939f;

/// The canonical 2D scene: a static floor and a grid of mixed boxes
/// and circles dropped onto it.
fn reference_scene_2d() -> World2 {
    let mut world = World2::new();
    world.gravity = Vec2::new(0.0, -9.81);
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    for i in 0..BODIES_2D {
        let x = -6.0 + (i as f32 % 15.0) * 0.85;
        let y = 2.0 + (i as f32 / 15.0) * 0.8;
        if i % 3 == 0 {
            world
                .add_dynamic_body(
                    Vec2::new(x, y),
                    Shape2::Box {
                        half_extents: Vec2::new(0.3, 0.3),
                    },
                    1.0,
                )
                .expect("valid shape");
        } else {
            world
                .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.3 }, 1.0)
                .expect("valid shape");
        }
    }
    world
}

/// The canonical 3D scene.
fn reference_scene_3d() -> World3 {
    let mut world = World3::new();
    world.gravity = Vec3::new(0.0, -9.81, 0.0);
    world
        .add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
        )
        .expect("valid shape");
    for i in 0..BODIES_3D {
        let x = -4.0 + (i % 10) as f32 * 0.9;
        let z = -2.0 + ((i / 10) % 6) as f32 * 0.9;
        let y = 2.0 + (i / 60) as f32 * 0.8;
        let shape = if i % 3 == 0 {
            Shape3::Box {
                half_extents: Vec3::new(0.3, 0.3, 0.3),
            }
        } else {
            Shape3::Sphere { radius: 0.3 }
        };
        world
            .add_dynamic_body(Vec3::new(x, y, z), shape, 1.0)
            .expect("valid shape");
    }
    world
}

/// FNV-1a over the 2D body state, matching the benchmark's hash so the
/// two can be compared directly.
fn hash_2d(world: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
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

/// FNV-1a over the 3D body state.
fn hash_3d(world: &World3) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.transform.position.z,
            body.velocity.x,
            body.velocity.y,
            body.velocity.z,
            body.angular_velocity.x,
            body.angular_velocity.y,
            body.angular_velocity.z,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// The 2D trajectory is pinned to an absolute value.
#[test]
fn determinism_2d_trajectory_is_pinned() {
    let mut world = reference_scene_2d();
    for _ in 0..STEPS {
        world.step(DT);
    }
    assert_eq!(
        hash_2d(&world),
        HASH_2D,
        "the 2D trajectory changed: the physics moved. If that was intended, \
         re-pin HASH_2D and note it in the CHANGELOG; otherwise this is a \
         regression."
    );
}

/// The 3D trajectory is pinned to an absolute value.
#[test]
fn determinism_3d_trajectory_is_pinned() {
    let mut world = reference_scene_3d();
    for _ in 0..STEPS {
        world.step(DT);
    }
    assert_eq!(
        hash_3d(&world),
        HASH_3D,
        "the 3D trajectory changed: the physics moved. If that was intended, \
         re-pin HASH_3D and note it in the CHANGELOG; otherwise this is a \
         regression."
    );
}

/// The parallel solver must land on the *same* pinned trajectory, not
/// merely on a self-consistent one. Without this, the parallel path
/// could drift away from the serial one and still pass every relative
/// test.
#[test]
fn determinism_parallel_lands_on_the_pinned_trajectory() {
    let mut world = reference_scene_2d();
    world.solver.parallel = true;
    for _ in 0..STEPS {
        world.step(DT);
    }
    assert_eq!(
        hash_2d(&world),
        HASH_2D,
        "the parallel solver no longer lands on the pinned serial trajectory"
    );
}
