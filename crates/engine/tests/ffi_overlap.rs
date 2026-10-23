//! Native shape-overlap tests (F51–F53, mirrored Rust↔C#).
//!
//! The overlap query is the last one to cross the bridge, and it needs a
//! shape where the other queries needed scalars: the shape arrives as a
//! small JSON blob, the same convention the joint bridge uses. These
//! tests pin that the blob parses, that the answer matches the managed
//! query, and that malformed input is a clean zero.

use phieng::ffi::{
    phieng_world3_add_body, phieng_world3_free, phieng_world3_new, phieng_world3_overlap_shape,
    phieng_world_add_body, phieng_world_free, phieng_world_new, phieng_world_overlap_shape,
};

const GROUP: u32 = 1;
const MASK: u32 = u32::MAX;

fn circle_json(r: f32) -> String {
    format!(r#"{{"type":"circle","radius":{r}}}"#)
}

/// F51: a 2D circle query finds the bodies it touches, matching the
/// managed query.
#[test]
fn f51_overlap_shape_reports_touching_bodies() {
    unsafe {
        let world = phieng_world_new();
        let a = phieng_world_add_body(world, 0.0, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        let b = phieng_world_add_body(world, 1.5, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        let far = phieng_world_add_body(world, 10.0, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(a >= 0 && b >= 0 && far >= 0);

        let json = circle_json(0.6);
        let bytes = json.as_bytes();

        // Count alone: both near bodies (0.6 + 1.0 > 1.5 gap).
        assert_eq!(
            phieng_world_overlap_shape(
                world,
                0.0,
                0.0,
                0.0,
                GROUP,
                MASK,
                bytes.as_ptr(),
                bytes.len() as u32,
                std::ptr::null_mut(),
                0,
            ),
            2
        );

        let mut ids = [u32::MAX; 4];
        let n = phieng_world_overlap_shape(
            world,
            0.0,
            0.0,
            0.0,
            GROUP,
            MASK,
            bytes.as_ptr(),
            bytes.len() as u32,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(n, 2);
        assert_eq!(&ids[..2], &[a as u32, b as u32]);

        // The managed query agrees.
        let world_ref = &*world;
        let managed = world_ref.overlap_shape(
            &phieng_dynamics::Shape2::Circle { radius: 0.6 },
            glam::Vec2::ZERO,
            glam::Mat2::IDENTITY,
            &phieng_dynamics::body::QueryFilter {
                group: GROUP,
                mask: MASK,
            },
        );
        assert_eq!(&ids[..2], managed.as_slice());

        phieng_world_free(world);
    }
}

/// F52: a 3D box query with a rotation, against a sphere and a far box.
#[test]
fn f52_overlap_shape_3d_with_rotation() {
    unsafe {
        let world = phieng_world3_new();
        let sphere =
            phieng_world3_add_body(world, 0.0, 0.0, 0.0, 0, 1.0, 0.0, 0.0, 1.0, 1, 0.5, 0.0);
        let far = phieng_world3_add_body(world, 10.0, 0.0, 0.0, 1, 0.5, 0.5, 0.5, 1.0, 1, 0.5, 0.0);
        assert!(sphere >= 0 && far >= 0);

        let json = r#"{"type":"box","halfExtents":[0.2,0.2,2.0]}"#;
        let bytes = json.as_bytes();
        // A slim box from (0,0,-2) into the sphere, rotated 90 deg about Y
        // so it spans x instead: still overlaps the unit sphere at the
        // origin.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let (qx, qy, qz, qw) = (0.0, half, 0.0, half);
        let mut ids = [u32::MAX; 4];
        let n = phieng_world3_overlap_shape(
            world,
            0.0,
            0.0,
            0.0,
            qx,
            qy,
            qz,
            qw,
            GROUP,
            MASK,
            bytes.as_ptr(),
            bytes.len() as u32,
            ids.as_mut_ptr(),
            ids.len() as u32,
        );
        assert_eq!(n, 1);
        assert_eq!(ids[0], sphere as u32);

        // Out of reach: nothing.
        let n = phieng_world3_overlap_shape(
            world,
            50.0,
            0.0,
            0.0,
            qx,
            qy,
            qz,
            qw,
            GROUP,
            MASK,
            bytes.as_ptr(),
            bytes.len() as u32,
            std::ptr::null_mut(),
            0,
        );
        assert_eq!(n, 0);

        phieng_world3_free(world);
    }
}

/// F53: malformed or unsupported shape blobs are clean zeroes.
#[test]
fn f53_overlap_shape_rejects_bad_input() {
    unsafe {
        let world = phieng_world_new();
        phieng_world_add_body(world, 0.0, 0.0, 0, 1.0, 0.0, 1.0, 1, 0.5, 0.0);

        let mut ids = [u32::MAX; 2];
        let mut run = |json: &str| {
            let bytes = json.as_bytes();
            phieng_world_overlap_shape(
                world,
                0.0,
                0.0,
                0.0,
                GROUP,
                MASK,
                bytes.as_ptr(),
                bytes.len() as u32,
                ids.as_mut_ptr(),
                ids.len() as u32,
            )
        };
        assert_eq!(run("{nope"), 0, "malformed JSON");
        assert_eq!(run(r#"{"type":"triangle"}"#), 0, "unknown type");
        assert_eq!(run(r#"{"type":"circle"}"#), 0, "missing radius");
        assert_eq!(
            run(r#"{"type":"circle","radius":-1}"#),
            0,
            "negative radius"
        );
        assert_eq!(
            run(r#"{"type":"polygon","vertices":[[0,0],[1,0]]}"#),
            0,
            "a polygon needs three points"
        );
        assert_eq!(
            run(r#"{"type":"polygon","vertices":[[0,0],[1,0],[0,null]]}"#),
            0,
            "non-numeric vertex"
        );

        // Null world and null JSON are clean zeroes too.
        let json = circle_json(0.5);
        let bytes = json.as_bytes();
        assert_eq!(
            phieng_world_overlap_shape(
                std::ptr::null(),
                0.0,
                0.0,
                0.0,
                GROUP,
                MASK,
                bytes.as_ptr(),
                bytes.len() as u32,
                ids.as_mut_ptr(),
                ids.len() as u32,
            ),
            0
        );
        assert_eq!(
            phieng_world_overlap_shape(
                world,
                0.0,
                0.0,
                0.0,
                GROUP,
                MASK,
                std::ptr::null(),
                0,
                ids.as_mut_ptr(),
                ids.len() as u32,
            ),
            0
        );

        phieng_world_free(world);
    }
}
