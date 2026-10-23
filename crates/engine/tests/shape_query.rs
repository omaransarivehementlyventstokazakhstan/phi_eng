//! Point and closest-point shape queries (o01–o06, mirrored Rust↔C#).
//!
//! The semantics under test: `point` is the closest point **in** the
//! shape (the query point itself when inside), `normal` is the outward
//! escape direction (the least-penetration axis inside a box), and
//! `distance` is zero when inside. World queries visit bodies in
//! ascending id order and include sensors.

use glam::{Mat2, Quat, Vec2, Vec3};
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::shape_query::{closest_point2, closest_point3, point_in_shape3};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// o01: containment is exact for a circle, a rotated box and a polygon,
/// and does not depend on the polygon's winding.
#[test]
fn o01_point_containment_is_exact() {
    let id = Mat2::IDENTITY;
    let circle = Shape2::Circle { radius: 2.0 };
    assert!(!phieng_dynamics::shape_query::point_in_shape2(
        &circle,
        Vec2::ZERO,
        id,
        Vec2::new(2.01, 0.0)
    ));
    assert!(phieng_dynamics::shape_query::point_in_shape2(
        &circle,
        Vec2::ZERO,
        id,
        Vec2::new(1.99, 0.0)
    ));
    // On the surface counts as inside (a closed shape).
    assert!(phieng_dynamics::shape_query::point_in_shape2(
        &circle,
        Vec2::ZERO,
        id,
        Vec2::new(2.0, 0.0)
    ));

    // A box rotated 90 degrees: the far x axis is now the short one.
    let bx = Shape2::Box {
        half_extents: Vec2::new(1.0, 0.5),
    };
    let rot = Mat2::from_angle(std::f32::consts::FRAC_PI_2);
    assert!(!phieng_dynamics::shape_query::point_in_shape2(
        &bx,
        Vec2::ZERO,
        rot,
        Vec2::new(0.6, 0.0)
    ));
    assert!(phieng_dynamics::shape_query::point_in_shape2(
        &bx,
        Vec2::ZERO,
        rot,
        Vec2::new(0.0, 0.9)
    ));

    // The same triangle, counter-clockwise and clockwise: same answer.
    let ccw = Shape2::Polygon {
        vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
    };
    let cw = Shape2::Polygon {
        vertices: vec![Vec2::ZERO, Vec2::Y, Vec2::X],
    };
    let inside = Vec2::new(0.2, 0.2);
    let outside = Vec2::new(0.8, 0.8);
    for s in [&ccw, &cw] {
        assert!(phieng_dynamics::shape_query::point_in_shape2(
            s,
            Vec2::ZERO,
            id,
            inside
        ));
        assert!(!phieng_dynamics::shape_query::point_in_shape2(
            s,
            Vec2::ZERO,
            id,
            outside
        ));
    }
}

/// o02: the 2D closest point returns the surface point, the outward
/// normal, and the distance — with the inside convention.
#[test]
fn o02_the_2d_closest_point_is_analytic() {
    let id = Mat2::IDENTITY;
    let circle = Shape2::Circle { radius: 2.0 };
    let out = closest_point2(&circle, Vec2::ZERO, id, Vec2::new(5.0, 0.0)).expect("analytic");
    assert!(!out.inside);
    assert_eq!(bits(out.distance), 0x4040_0000);
    assert_eq!(bits(out.point.x), bits(2.0));
    assert_eq!(bits(out.normal.y), 0x0000_0000);

    // Inside: the query point itself is the closest point in the shape;
    // the normal is the radial escape direction.
    let inside = closest_point2(&circle, Vec2::ZERO, id, Vec2::new(1.0, 0.0)).expect("analytic");
    assert!(inside.inside);
    assert_eq!(inside.distance, 0.0);
    assert_eq!(inside.point, Vec2::new(1.0, 0.0));
    assert_eq!(inside.normal, Vec2::X);

    // A rotated box: (2,0) is 1.5 outside the world-space short axis.
    let bx = Shape2::Box {
        half_extents: Vec2::new(1.0, 0.5),
    };
    let rot = Mat2::from_angle(std::f32::consts::FRAC_PI_2);
    let box_out = closest_point2(&bx, Vec2::ZERO, rot, Vec2::new(2.0, 0.0)).expect("analytic");
    assert!(!box_out.inside);
    assert!((box_out.distance - 1.5).abs() < 1e-6);
    assert!((box_out.point.x - 0.5).abs() < 1e-6);

    // Inside the box the normal is the least-penetration axis.
    let box_in = closest_point2(&bx, Vec2::ZERO, rot, Vec2::new(0.2, 0.1)).expect("analytic");
    assert!(box_in.inside);
    assert_eq!(box_in.point, Vec2::new(0.2, 0.1));
    // local = (0.1, -0.2); the y half-extent (0.5) is the nearer face.
    assert!((box_in.normal - Vec2::X).length() < 1e-6);

    // A triangle: (2,0) is one unit from the hypotenuse-free edge.
    let tri = Shape2::Polygon {
        vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
    };
    let tri_out = closest_point2(&tri, Vec2::ZERO, id, Vec2::new(2.0, 0.0)).expect("analytic");
    assert!(!tri_out.inside);
    assert_eq!(bits(tri_out.distance), 0x3f80_0000);
    assert!((tri_out.point - Vec2::X).length() < 1e-6);
}

/// o03: 3D containment and closest points for the sphere, the box and
/// the capsule (axis along local Y).
#[test]
fn o03_the_3d_queries_are_analytic() {
    let q = Quat::IDENTITY;
    let sphere = Shape3::Sphere { radius: 2.0 };
    let out = closest_point3(&sphere, Vec3::ZERO, q, Vec3::new(5.0, 0.0, 0.0)).expect("analytic");
    assert!(!out.inside);
    assert_eq!(bits(out.distance), 0x4040_0000);
    assert!(point_in_shape3(
        &sphere,
        Vec3::ZERO,
        q,
        Vec3::new(1.0, 0.0, 0.0)
    ));

    // Capsule: r = 0.5, half_length = 1, so the top cap ends at y = 1.5.
    let capsule = Shape3::Capsule {
        radius: 0.5,
        half_length: 1.0,
    };
    let top = closest_point3(&capsule, Vec3::ZERO, q, Vec3::new(0.0, 3.0, 0.0)).expect("analytic");
    assert!(!top.inside);
    assert!((top.distance - 1.5).abs() < 1e-6);
    assert_eq!(bits(top.point.y), bits(1.5));
    assert!((top.normal - Vec3::Y).length() < 1e-6);
    assert!(point_in_shape3(
        &capsule,
        Vec3::ZERO,
        q,
        Vec3::new(0.49, 0.0, 0.0)
    ));
    assert!(!point_in_shape3(
        &capsule,
        Vec3::ZERO,
        q,
        Vec3::new(0.51, 0.0, 0.0)
    ));

    // A box: inside gives the least-penetration axis.
    let bx = Shape3::Box {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
    };
    let inside = closest_point3(&bx, Vec3::ZERO, q, Vec3::new(0.9, 0.0, 0.0)).expect("analytic");
    assert!(inside.inside);
    assert_eq!(inside.point, Vec3::new(0.9, 0.0, 0.0));
    assert!((inside.normal - Vec3::X).length() < 1e-6);
    let side = closest_point3(&bx, Vec3::ZERO, q, Vec3::new(4.0, 0.0, 0.0)).expect("analytic");
    assert!((side.distance - 3.0).abs() < 1e-6);
}

/// o04: the unsupported convex shapes report a miss, matching the ray
/// cast's documented gap.
#[test]
fn o04_convex_shapes_report_a_miss() {
    let q = Quat::IDENTITY;
    let hull = Shape3::ConvexHull {
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
    };
    assert!(closest_point3(&hull, Vec3::ZERO, q, Vec3::X).is_none());
    assert!(!point_in_shape3(&hull, Vec3::ZERO, q, Vec3::ZERO));
    let mesh = Shape3::ConvexMesh {
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
        indices: vec![0, 1, 2, 0, 2, 3],
    };
    assert!(closest_point3(&mesh, Vec3::ZERO, q, Vec3::X).is_none());
}

/// o05: a world point test returns ids in ascending order, includes
/// sensors (a point test is a pick/trigger query), and honors filters.
#[test]
fn o05_the_world_point_test_is_ordered_and_filtered() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(Vec2::ZERO, Shape2::Circle { radius: 1.0 })
        .expect("valid shape");
    w.add_static_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 1.0 })
        .expect("valid shape");
    w.bodies[1].is_sensor = true;
    w.add_static_body(Vec2::new(10.0, 0.0), Shape2::Circle { radius: 1.0 })
        .expect("valid shape");

    assert_eq!(
        w.point_test(Vec2::ZERO, &QueryFilter::everything()),
        vec![0, 1],
        "ascending ids, and the sensor is included"
    );

    // Put body 1 on its own layer. `everything` accepts every layer, so
    // it still sees both; a filter whose mask names one layer sees one.
    w.bodies[1].collision_group = 2;
    assert_eq!(
        w.point_test(Vec2::ZERO, &QueryFilter { group: 1, mask: 2 }),
        vec![1]
    );
    assert_eq!(
        w.point_test(Vec2::ZERO, &QueryFilter { group: 1, mask: 1 }),
        vec![0]
    );
    assert_eq!(
        w.point_test(Vec2::ZERO, &QueryFilter::everything()),
        vec![0, 1]
    );

    // 3D: the same rule.
    let mut w3 = World3::new();
    w3.gravity = Vec3::ZERO;
    w3.add_static_body(Vec3::ZERO, Shape3::Sphere { radius: 1.0 })
        .expect("valid shape");
    w3.add_static_body(
        Vec3::new(4.0, 0.0, 0.0),
        Shape3::Box {
            half_extents: Vec3::splat(0.5),
        },
    )
    .expect("valid shape");
    assert_eq!(
        w3.point_test(Vec3::new(0.2, 0.0, 0.0), &QueryFilter::everything()),
        vec![0]
    );
}

/// o06: the world closest point picks the nearest surface, respects the
/// range, and prefers an inside hit (distance zero) over any outside
/// surface.
#[test]
fn o06_the_world_closest_point_picks_the_nearest() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.add_static_body(Vec2::ZERO, Shape2::Circle { radius: 1.0 })
        .expect("valid shape");
    w.add_static_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 1.0 })
        .expect("valid shape");

    // (1.8, 0): 0.8 from body 0's surface, 0.3 from body 1's.
    let hit = w
        .closest_point(Vec2::new(1.8, 0.0), &QueryFilter::everything(), 5.0)
        .expect("a hit");
    assert_eq!(hit.body, 1);
    assert!(!hit.inside);
    assert_eq!(bits(hit.distance), 0x3e99_9998);

    // The range excludes that surface; the farther body is out too, so
    // nothing is reported.
    assert!(w
        .closest_point(Vec2::new(1.8, 0.0), &QueryFilter::everything(), 0.2)
        .is_none());

    // Inside wins over outside: (1.4, 0) is inside body 1 (distance
    // zero) and 0.4 outside body 0, so the inside hit is reported.
    let inside = w
        .closest_point(Vec2::new(1.4, 0.0), &QueryFilter::everything(), 5.0)
        .expect("a hit");
    assert!(inside.inside);
    assert_eq!(inside.body, 1);
    assert_eq!(inside.distance, 0.0);

    // A point inside both is a tie at distance zero; the lower id wins
    // (bodies are visited in ascending order).
    let tie = w
        .closest_point(Vec2::new(0.9, 0.0), &QueryFilter::everything(), 5.0)
        .expect("a hit");
    assert_eq!(tie.body, 0);

    // 3D: two metres above a unit sphere.
    let mut w3 = World3::new();
    w3.gravity = Vec3::ZERO;
    w3.add_static_body(Vec3::ZERO, Shape3::Sphere { radius: 1.0 })
        .expect("valid shape");
    let hit3 = w3
        .closest_point(Vec3::new(0.0, 3.0, 0.0), &QueryFilter::everything(), 10.0)
        .expect("a hit");
    assert_eq!(hit3.body, 0);
    assert_eq!(bits(hit3.distance), 0x4000_0000);
    assert_eq!(bits(hit3.point.y), bits(1.0));
}
/// o13: the AABB prefilter is invisible - every query returns exactly
/// what a brute-force loop over all bodies returns, including ties.
#[test]
fn o13_the_prefilter_matches_brute_force() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    for iy in 0..12 {
        for ix in 0..12 {
            w.add_static_body(
                Vec2::new(ix as f32, iy as f32),
                Shape2::Circle { radius: 0.45 },
            )
            .expect("valid shape");
        }
    }
    let all = QueryFilter::everything();
    let probes: Vec<Vec2> = (0..40)
        .map(|k| {
            let f = k as f32 * 0.61;
            Vec2::new(f % 11.0, (f * 1.3) % 11.0)
        })
        .collect();

    for p in &probes {
        // Point test against a brute-force loop.
        let mut brute = Vec::new();
        for body in &w.bodies {
            let Some(shape) = body.shape.as_ref() else {
                continue;
            };
            if phieng_dynamics::shape_query::point_in_shape2(
                shape,
                body.transform.position,
                body.transform.rotation,
                *p,
            ) {
                brute.push(body.id as u32);
            }
        }
        assert_eq!(w.point_test(*p, &all), brute, "point test at {p:?}");

        // Closest point against a brute-force loop, ties included.
        let mut best: Option<(u32, f32)> = None;
        for body in &w.bodies {
            let Some(shape) = body.shape.as_ref() else {
                continue;
            };
            let Some(hit) = phieng_dynamics::shape_query::closest_point2(
                shape,
                body.transform.position,
                body.transform.rotation,
                *p,
            ) else {
                continue;
            };
            if !hit.inside && hit.distance > 1000.0 {
                continue;
            }
            if best.map(|(_, d)| hit.distance < d).unwrap_or(true) {
                best = Some((body.id as u32, hit.distance));
            }
        }
        let fast = w
            .closest_point(*p, &all, 1000.0)
            .map(|h| (h.body, h.distance));
        assert_eq!(fast, best, "closest point at {p:?}");
    }

    // Overlap against a brute-force loop.
    let probe = Shape2::Circle { radius: 0.6 };
    for p in probes.iter().take(10) {
        let mut brute = Vec::new();
        for body in &w.bodies {
            let Some(shape) = body.shape.as_ref() else {
                continue;
            };
            if phieng_dynamics::shape_query::shapes_overlap2(
                &probe,
                *p,
                Mat2::IDENTITY,
                shape,
                body.transform.position,
                body.transform.rotation,
            ) {
                brute.push(body.id as u32);
            }
        }
        assert_eq!(
            w.overlap_shape(&probe, *p, Mat2::IDENTITY, &all),
            brute,
            "overlap at {p:?}"
        );
    }
}
