//! Native query-surface tests (F47–F50, mirrored Rust↔C#).
//!
//! The scene queries existed in Rust and in the managed API but not
//! across the C ABI, which is the gap these tests close: a point test
//! and a closest-point query are reachable from a native host, agree
//! with the Rust results, and reject bad input cleanly.

use phieng::ffi::{
    phieng_world3_add_body, phieng_world3_closest_point, phieng_world3_free, phieng_world3_new,
    phieng_world3_point_test, phieng_world_add_body, phieng_world_closest_point, phieng_world_free,
    phieng_world_new, phieng_world_point_test, ClosestHit2Ffi, ClosestHit3Ffi,
};

const ALL_GROUP: u32 = 1;
const ALL_MASK: u32 = u32::MAX;

/// F47: a 2D point test reports every containing body, in id order, and
/// a null output asks for the count alone.
#[test]
fn f47_point_test_reports_containing_bodies() {
    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());
        // Two overlapping circles at the origin, one far away.
        let a = phieng_world_add_body(world, 0.0, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        let b = phieng_world_add_body(world, 0.5, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        let far = phieng_world_add_body(world, 10.0, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(a >= 0 && b >= 0 && far >= 0);

        // Count alone.
        assert_eq!(
            phieng_world_point_test(
                world,
                0.0,
                0.0,
                ALL_GROUP,
                ALL_MASK,
                std::ptr::null_mut(),
                0
            ),
            2
        );

        let mut ids = [u32::MAX; 4];
        let n = phieng_world_point_test(
            world,
            0.0,
            0.0,
            ALL_GROUP,
            ALL_MASK,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(n, 2);
        assert_eq!(&ids[..2], &[a as u32, b as u32]);

        // A cap smaller than the hit count truncates, and says so.
        let mut one = [u32::MAX; 1];
        assert_eq!(
            phieng_world_point_test(world, 0.0, 0.0, ALL_GROUP, ALL_MASK, one.as_mut_ptr(), 1),
            1
        );
        assert_eq!(one[0], a as u32);

        // A filter that accepts only another layer sees nothing.
        assert_eq!(
            phieng_world_point_test(world, 0.0, 0.0, 1, 2, std::ptr::null_mut(), 0),
            0
        );

        phieng_world_free(world);
    }
}

/// F48: a 3D point test over a sphere and a box.
#[test]
fn f48_point_test_3d() {
    unsafe {
        let world = phieng_world3_new();
        assert!(!world.is_null());
        // Sphere radius 1 at the origin; box far away.
        let s = phieng_world3_add_body(world, 0.0, 0.0, 0.0, 0, 1.0, 0.0, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(s >= 0);
        let f = phieng_world3_add_body(world, 10.0, 0.0, 0.0, 1, 0.5, 0.5, 0.5, 1.0, 1, 0.5, 0.0);
        assert!(f >= 0);

        let mut ids = [u32::MAX; 4];
        let n = phieng_world3_point_test(
            world,
            0.2,
            0.0,
            0.0,
            ALL_GROUP,
            ALL_MASK,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(n, 1);
        assert_eq!(ids[0], s as u32);

        // Non-finite input is a clean zero.
        assert_eq!(
            phieng_world3_point_test(
                world,
                f32::NAN,
                0.0,
                0.0,
                ALL_GROUP,
                ALL_MASK,
                std::ptr::null_mut(),
                0
            ),
            0
        );

        phieng_world3_free(world);
    }
}

/// F49: a 2D closest-point query fills the struct and agrees with the
/// Rust result.
#[test]
fn f49_closest_point_reports_the_surface() {
    unsafe {
        let world = phieng_world_new();
        let id = phieng_world_add_body(world, 0.0, 0.0, 0, 2.0, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(id >= 0);

        let mut hit = ClosestHit2Ffi {
            body: u32::MAX,
            inside: 0,
            _pad: [0; 3],
            distance: -1.0,
            point_x: 0.0,
            point_y: 0.0,
            normal_x: 0.0,
            normal_y: 0.0,
        };
        // (5, 0) is 3 from a radius-2 circle's surface.
        assert_eq!(
            phieng_world_closest_point(world, 5.0, 0.0, ALL_GROUP, ALL_MASK, 10.0, &mut hit),
            1
        );
        assert_eq!(hit.body, id as u32);
        assert_eq!(hit.inside, 0);
        assert_eq!(hit.distance, 3.0);
        assert_eq!(hit.point_x, 2.0);
        assert_eq!(hit.point_y, 0.0);
        assert_eq!(hit.normal_x, 1.0);
        assert_eq!(hit.normal_y, 0.0);

        // Out of range: no hit, and the struct is untouched.
        let before = hit.distance;
        assert_eq!(
            phieng_world_closest_point(world, 5.0, 0.0, ALL_GROUP, ALL_MASK, 1.0, &mut hit),
            0
        );
        assert_eq!(hit.distance, before);

        // Null output is a clean zero.
        assert_eq!(
            phieng_world_closest_point(
                world,
                0.0,
                0.0,
                ALL_GROUP,
                ALL_MASK,
                1.0,
                std::ptr::null_mut()
            ),
            0
        );

        phieng_world_free(world);
    }
}

/// F50: a 3D closest-point query, including the inside convention and
/// bad-input rejection.
#[test]
fn f50_closest_point_3d_and_rejections() {
    unsafe {
        let world = phieng_world3_new();
        let id = phieng_world3_add_body(world, 0.0, 0.0, 0.0, 0, 1.0, 0.0, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(id >= 0);

        let mut hit = ClosestHit3Ffi {
            body: u32::MAX,
            inside: 0,
            _pad: [0; 3],
            distance: -1.0,
            point_x: 0.0,
            point_y: 0.0,
            point_z: 0.0,
            normal_x: 0.0,
            normal_y: 0.0,
            normal_z: 0.0,
        };
        // Two metres above a unit sphere: distance 2 to the surface.
        assert_eq!(
            phieng_world3_closest_point(world, 0.0, 3.0, 0.0, ALL_GROUP, ALL_MASK, 10.0, &mut hit),
            1
        );
        assert_eq!(hit.body, id as u32);
        assert_eq!(hit.distance, 2.0);
        assert_eq!(hit.point_y, 1.0);
        assert_eq!(hit.normal_y, 1.0);

        // Inside: distance zero, and the normal is the escape direction.
        assert_eq!(
            phieng_world3_closest_point(world, 0.0, 0.5, 0.0, ALL_GROUP, ALL_MASK, 10.0, &mut hit),
            1
        );
        assert_eq!(hit.inside, 1);
        assert_eq!(hit.distance, 0.0);

        // Null world, and a non-finite query point, are clean zeroes.
        assert_eq!(
            phieng_world3_closest_point(
                std::ptr::null(),
                0.0,
                0.0,
                0.0,
                ALL_GROUP,
                ALL_MASK,
                1.0,
                &mut hit
            ),
            0
        );
        assert_eq!(
            phieng_world3_closest_point(
                world,
                0.0,
                f32::INFINITY,
                0.0,
                ALL_GROUP,
                ALL_MASK,
                1.0,
                &mut hit
            ),
            0
        );

        phieng_world3_free(world);
    }
}
