//! Shape-overlap queries (o07–o10, mirrored Rust↔C#).
//!
//! Overlap means "touch or penetrate"; touching counts. SAT/GJK decide
//! per pair, with an explicit containment check so a shape wholly
//! inside another is never reported as separated.

use glam::{Mat2, Quat, Vec2, Vec3};
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::shape_query::{shapes_overlap2, shapes_overlap3};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

fn circle2(r: f32) -> Shape2 {
    Shape2::Circle { radius: r }
}

fn box2(x: f32, y: f32) -> Shape2 {
    Shape2::Box {
        half_extents: Vec2::new(x, y),
    }
}

fn sphere3(r: f32) -> Shape3 {
    Shape3::Sphere { radius: r }
}

fn box3(x: f32, y: f32, z: f32) -> Shape3 {
    Shape3::Box {
        half_extents: Vec3::new(x, y, z),
    }
}

/// o07: the 2D overlap matrix — separated, touching, penetrating, and
/// a small shape wholly inside a large one.
#[test]
fn o07_the_2d_overlap_matrix() {
    let id = Mat2::IDENTITY;
    let cases: [(Shape2, Vec2, Shape2, Vec2, Mat2, bool); 6] = [
        (
            circle2(1.0),
            Vec2::ZERO,
            circle2(1.0),
            Vec2::new(2.5, 0.0),
            id,
            false,
        ),
        (
            circle2(1.0),
            Vec2::ZERO,
            circle2(1.0),
            Vec2::new(2.0, 0.0),
            id,
            true,
        ),
        (
            circle2(1.0),
            Vec2::ZERO,
            circle2(1.0),
            Vec2::new(1.0, 0.0),
            id,
            true,
        ),
        // Deep containment: SAT alone would call this separated.
        (
            circle2(0.2),
            Vec2::ZERO,
            box2(5.0, 5.0),
            Vec2::ZERO,
            id,
            true,
        ),
        (
            box2(1.0, 0.5),
            Vec2::ZERO,
            box2(1.0, 0.5),
            Vec2::new(3.0, 0.0),
            Mat2::from_angle(std::f32::consts::FRAC_PI_4),
            false,
        ),
        (
            box2(1.0, 0.5),
            Vec2::ZERO,
            box2(1.0, 0.5),
            Vec2::new(1.4, 0.0),
            Mat2::from_angle(std::f32::consts::FRAC_PI_4),
            true,
        ),
    ];
    for (i, (a, ap, b, bp, ar, want)) in cases.iter().enumerate() {
        assert_eq!(
            shapes_overlap2(a, *ap, *ar, b, *bp, id),
            *want,
            "2D overlap case {i}"
        );
    }

    // A triangle against a small circle, inside and outside.
    let tri = Shape2::Polygon {
        vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
    };
    assert!(!shapes_overlap2(
        &tri,
        Vec2::ZERO,
        id,
        &circle2(0.2),
        Vec2::new(2.0, 2.0),
        id
    ));
    assert!(shapes_overlap2(
        &tri,
        Vec2::ZERO,
        id,
        &circle2(0.05),
        Vec2::new(0.2, 0.2),
        id
    ));
}

/// o08: the 3D overlap matrix — spheres, a capsule, and a convex hull.
#[test]
fn o08_the_3d_overlap_matrix() {
    let q = Quat::IDENTITY;
    let capsule = Shape3::Capsule {
        radius: 0.5,
        half_length: 1.0,
    };
    let hull = Shape3::ConvexHull {
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
    };
    let cases: [(Shape3, Vec3, Shape3, Vec3, bool); 6] = [
        (
            sphere3(1.0),
            Vec3::ZERO,
            sphere3(1.0),
            Vec3::new(2.5, 0.0, 0.0),
            false,
        ),
        (
            sphere3(1.0),
            Vec3::ZERO,
            sphere3(1.0),
            Vec3::new(2.0, 0.0, 0.0),
            true,
        ),
        // Deep containment through GJK plus the point fallback.
        (
            sphere3(0.1),
            Vec3::ZERO,
            box3(5.0, 5.0, 5.0),
            Vec3::ZERO,
            true,
        ),
        (
            capsule.clone(),
            Vec3::ZERO,
            box3(0.2, 0.2, 0.2),
            Vec3::new(0.6, 0.0, 0.0),
            true,
        ),
        (
            capsule,
            Vec3::ZERO,
            box3(0.2, 0.2, 0.2),
            Vec3::new(2.0, 0.0, 0.0),
            false,
        ),
        (
            hull,
            Vec3::ZERO,
            sphere3(0.1),
            Vec3::new(0.2, 0.2, 0.2),
            true,
        ),
    ];
    for (i, (a, ap, b, bp, want)) in cases.iter().enumerate() {
        assert_eq!(
            shapes_overlap3(a, *ap, q, b, *bp, q),
            *want,
            "3D overlap case {i}"
        );
    }
}

/// o09: a world overlap query returns ids in ascending order, includes
/// sensors, and honors the filter.
#[test]
fn o09_the_world_overlap_is_ordered_and_filtered() {
    let id = Mat2::IDENTITY;
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(Vec2::ZERO, circle2(1.0))
        .expect("valid shape");
    w.add_static_body(Vec2::new(1.5, 0.0), circle2(1.0))
        .expect("valid shape");
    w.bodies[1].is_sensor = true;
    w.add_static_body(Vec2::new(10.0, 0.0), circle2(1.0))
        .expect("valid shape");

    assert_eq!(
        w.overlap_shape(&circle2(0.5), Vec2::ZERO, id, &QueryFilter::everything()),
        vec![0, 1],
        "ascending ids, and the sensor is included"
    );

    w.bodies[1].collision_group = 2;
    assert_eq!(
        w.overlap_shape(
            &circle2(0.5),
            Vec2::ZERO,
            id,
            &QueryFilter { group: 1, mask: 2 }
        ),
        vec![1]
    );
    assert_eq!(
        w.overlap_shape(
            &circle2(0.5),
            Vec2::ZERO,
            id,
            &QueryFilter { group: 1, mask: 1 }
        ),
        vec![0]
    );

    let q = Quat::IDENTITY;
    let mut w3 = World3::new();
    w3.gravity = Vec3::ZERO;
    w3.add_static_body(Vec3::ZERO, sphere3(1.0))
        .expect("valid shape");
    w3.add_static_body(Vec3::new(10.0, 0.0, 0.0), box3(1.0, 1.0, 1.0))
        .expect("valid shape");
    assert_eq!(
        w3.overlap_shape(&sphere3(0.5), Vec3::ZERO, q, &QueryFilter::everything()),
        vec![0]
    );
}

/// o10: overlap is symmetric — swapping the two shapes never changes
/// the answer (a property SAT/GJK could get wrong when one shape is
/// wholly inside the other).
#[test]
fn o10_overlap_is_symmetric() {
    let id = Mat2::IDENTITY;
    let rot = Mat2::from_angle(0.7);
    let pairs: [(Shape2, Vec2, Shape2, Vec2); 5] = [
        (circle2(0.5), Vec2::ZERO, box2(3.0, 3.0), Vec2::ZERO),
        (circle2(1.0), Vec2::ZERO, circle2(1.0), Vec2::new(1.9, 0.0)),
        (
            Shape2::Polygon {
                vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
            },
            Vec2::ZERO,
            box2(0.5, 0.5),
            Vec2::new(0.2, 0.2),
        ),
        (
            box2(1.0, 0.2),
            Vec2::ZERO,
            box2(1.0, 0.2),
            Vec2::new(2.0, 0.0),
        ),
        (
            box2(1.0, 0.2),
            Vec2::ZERO,
            box2(1.0, 0.2),
            Vec2::new(1.9, 0.0),
        ),
    ];
    for (i, (a, ap, b, bp)) in pairs.iter().enumerate() {
        let ab = shapes_overlap2(a, *ap, rot, b, *bp, id);
        let ba = shapes_overlap2(b, *bp, id, a, *ap, rot);
        assert_eq!(ab, ba, "2D symmetry case {i}");
    }

    let q = Quat::from_rotation_y(0.6);
    let hull = Shape3::ConvexHull {
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
    };
    let pairs3: [(Shape3, Vec3, Shape3, Vec3); 4] = [
        (sphere3(0.1), Vec3::ZERO, box3(3.0, 3.0, 3.0), Vec3::ZERO),
        (
            hull.clone(),
            Vec3::ZERO,
            sphere3(0.2),
            Vec3::new(0.1, 0.1, 0.1),
        ),
        (
            sphere3(1.0),
            Vec3::ZERO,
            sphere3(1.0),
            Vec3::new(1.9, 0.0, 0.0),
        ),
        (
            Shape3::Capsule {
                radius: 0.4,
                half_length: 0.8,
            },
            Vec3::ZERO,
            box3(0.5, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
        ),
    ];
    for (i, (a, ap, b, bp)) in pairs3.iter().enumerate() {
        let ab = shapes_overlap3(a, *ap, q, b, *bp, Quat::IDENTITY);
        let ba = shapes_overlap3(b, *bp, Quat::IDENTITY, a, *ap, q);
        assert_eq!(ab, ba, "3D symmetry case {i}");
    }
}
