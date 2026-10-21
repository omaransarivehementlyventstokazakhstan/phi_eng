//! Native write-surface tests (F41–F46, mirrored Rust↔C#).
//!
//! The bridge could read a body's state but not change it: no way to
//! place a body, spin it, push it or wake it. These tests pin that the
//! write surface is there and that what it writes is what the world
//! sees.

use phieng::ffi::{
    phieng_world_add_body, phieng_world_apply_force, phieng_world_apply_impulse, phieng_world_free,
    phieng_world_get_body_angular_velocity, phieng_world_get_body_id_at,
    phieng_world_get_body_transform, phieng_world_new, phieng_world_set_body_angular_velocity,
    phieng_world_set_body_transform, phieng_world_step, phieng_world_wake_body,
};

const DT: f32 = 1.0 / 60.0;

/// F41: a body can be placed, and the world sees where it went.
#[test]
fn f41_set_body_transform_moves_the_body() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(id >= 0);

        assert_eq!(
            phieng_world_set_body_transform(world, id as u32, 3.0, -2.0, 0.5),
            1
        );

        let mut pos = [0.0f32; 2];
        let mut rot = [0.0f32; 4];
        assert_eq!(
            phieng_world_get_body_transform(world, id as u32, pos.as_mut_ptr(), rot.as_mut_ptr()),
            1
        );
        assert_eq!(pos[0], 3.0);
        assert_eq!(pos[1], -2.0);

        phieng_world_free(world);
    }
}

/// F42: a body can be spun, and the spin survives a step.
#[test]
fn f42_set_body_angular_velocity() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(id >= 0);

        assert_eq!(
            phieng_world_set_body_angular_velocity(world, id as u32, 2.5),
            1
        );
        let mut av = 0.0f32;
        assert_eq!(
            phieng_world_get_body_angular_velocity(world, id as u32, &mut av),
            1
        );
        assert_eq!(av, 2.5);

        phieng_world_step(world, DT);
        assert_eq!(
            phieng_world_get_body_angular_velocity(world, id as u32, &mut av),
            1
        );
        // The spin survives the step: the body's angular damping is
        // 0.01 per second, so it eases off a little and no more.
        assert!(
            (av - 2.5).abs() < 0.01,
            "the spin must survive the step (av={av})"
        );

        phieng_world_free(world);
    }
}

/// F43: an impulse moves a body, and a force does too.
#[test]
fn f43_apply_impulse_and_force() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(id >= 0);

        assert_eq!(phieng_world_apply_impulse(world, id as u32, 2.0, 0.0), 1);
        phieng_world_step(world, DT);
        let mut pos = [0.0f32; 2];
        let mut rot = [0.0f32; 4];
        assert_eq!(
            phieng_world_get_body_transform(world, id as u32, pos.as_mut_ptr(), rot.as_mut_ptr()),
            1
        );
        assert!(
            pos[0] > 0.0,
            "the impulse must have moved the body (x={})",
            pos[0]
        );

        let before = pos[0];
        assert_eq!(
            phieng_world_apply_force(world, id as u32, 10.0, 0.0, 1.0),
            1
        );
        phieng_world_step(world, DT);
        assert_eq!(
            phieng_world_get_body_transform(world, id as u32, pos.as_mut_ptr(), rot.as_mut_ptr()),
            1
        );
        assert!(
            pos[0] > before,
            "the force must have moved the body further (x={})",
            pos[0]
        );

        phieng_world_free(world);
    }
}

/// F44: a body can be woken, and an unknown id is refused.
#[test]
fn f44_wake_body() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let id = phieng_world_add_body(world, 0.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(id >= 0);
        assert_eq!(phieng_world_wake_body(world, id as u32), 1);
        assert_eq!(
            phieng_world_wake_body(world, 9999),
            0,
            "an unknown body must be refused"
        );
        phieng_world_free(world);
    }
}

/// F45: the bodies can be enumerated.
#[test]
fn f45_body_enumeration() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        let a = phieng_world_add_body(world, 0.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        let b = phieng_world_add_body(world, 2.0, 5.0, 0, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
        assert!(a >= 0 && b >= 0);
        assert_eq!(phieng_world_get_body_id_at(world, 0), a);
        assert_eq!(phieng_world_get_body_id_at(world, 1), b);
        assert_eq!(phieng_world_get_body_id_at(world, 2), -1);
        phieng_world_free(world);
    }
}

/// F46: the write surface refuses a bad body id and a bad value.
#[test]
fn f46_write_surface_rejects_bad_input() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        assert_eq!(
            phieng_world_set_body_transform(world, 9999, 0.0, 0.0, 0.0),
            0,
            "an unknown body must be refused"
        );
        assert_eq!(
            phieng_world_set_body_transform(world, 0, f32::NAN, 0.0, 0.0),
            0,
            "a non-finite position must be refused"
        );
        assert_eq!(
            phieng_world_apply_impulse(world, 0, f32::INFINITY, 0.0),
            0,
            "a non-finite impulse must be refused"
        );
        phieng_world_free(world);
    }
}
