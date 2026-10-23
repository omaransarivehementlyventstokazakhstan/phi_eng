//! Point queries against shapes: containment and the nearest surface
//! point.
//!
//! These are the simplest "pick" primitives: a point test answers
//! "which bodies are under this cursor / inside this trigger", and the
//! closest-point query answers "where is the nearest surface, and how
//! far". Both are analytic per shape, so they are exact and cheap; the
//! convex hull and convex mesh report a miss, exactly as the ray cast
//! does (a documented gap, see `LIMITATIONS.md`).
//!
//! # Semantics
//!
//! `point` is the closest point **in the shape**: outside, the nearest
//! surface point; inside, the query point itself (distance zero). The
//! `normal` is the outward escape direction: outside, the surface
//! normal; inside, the direction of least penetration (for a circle the
//! radial direction, for a box the nearest face's axis, for a polygon
//! the nearest edge's normal). That pair is what depenetration and
//! ground probes need.
//!
//! Winding is not trusted: a polygon's orientation is derived from its
//! own signed area, so a clockwise-authored polygon behaves the same as
//! a counter-clockwise one.

use glam::{Mat2, Vec2, Vec3};

use super::body::{Shape2, Shape3};

/// The nearest surface point of a shape to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeHit2 {
    /// Whether the point is inside the shape.
    pub inside: bool,
    /// The point on the surface nearest the query point. For a point
    /// inside, this is still the surface point (the exit direction
    /// that the normal names).
    pub point: Vec2,
    /// The outward unit normal at that surface point.
    pub normal: Vec2,
    /// The distance from the query point to the surface (zero when
    /// inside).
    pub distance: f32,
}

/// The nearest surface point of a 3D shape. See [`ShapeHit2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeHit3 {
    pub inside: bool,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// The unit vector along `v`, or `fallback` for a zero vector.
fn unit_or(v: Vec2, fallback: Vec2) -> Vec2 {
    let len = v.length();
    if len > f32::MIN_POSITIVE && len.is_finite() {
        v / len
    } else {
        fallback
    }
}

/// The unit vector along `v`, or `fallback` for a zero vector.
fn unit_or3(v: Vec3, fallback: Vec3) -> Vec3 {
    let len = v.length();
    if len > f32::MIN_POSITIVE && len.is_finite() {
        v / len
    } else {
        fallback
    }
}

/// The closest point on the segment `a..b` to `p`.
fn segment_closest(a: Vec2, b: Vec2, p: Vec2) -> Vec2 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= f32::MIN_POSITIVE {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// The closest point on the segment `a..b` to `p` (3D).
fn segment_closest3(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= f32::MIN_POSITIVE {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Whether a convex polygon's vertices wind counter-clockwise.
fn polygon_is_ccw(vertices: &[Vec2]) -> bool {
    let mut area2 = 0.0f32;
    for i in 0..vertices.len() {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        area2 += a.x * b.y - b.x * a.y;
    }
    area2 > 0.0
}

/// Whether `point` is inside the shape at the given pose.
pub fn point_in_shape2(shape: &Shape2, position: Vec2, rotation: Mat2, point: Vec2) -> bool {
    match closest_point2(shape, position, rotation, point) {
        Some(hit) => hit.inside,
        None => false,
    }
}

/// Whether `point` is inside the 3D shape at the given pose.
pub fn point_in_shape3(shape: &Shape3, position: Vec3, rotation: glam::Quat, point: Vec3) -> bool {
    match closest_point3(shape, position, rotation, point) {
        Some(hit) => hit.inside,
        None => false,
    }
}

/// The nearest surface point of a 2D shape, or `None` for a shape the
/// query does not support (there is none in 2D: all shapes are analytic).
pub fn closest_point2(
    shape: &Shape2,
    position: Vec2,
    rotation: Mat2,
    point: Vec2,
) -> Option<ShapeHit2> {
    match shape {
        Shape2::Circle { radius } => {
            let r = radius.abs();
            let offset = point - position;
            let len = offset.length();
            let normal = unit_or(offset, Vec2::Y);
            let inside = len <= r;
            Some(ShapeHit2 {
                inside,
                // Inside: the query point is the closest point in the
                // shape; the normal is the radial escape direction.
                point: if inside { point } else { position + normal * r },
                normal,
                distance: if inside { 0.0 } else { len - r },
            })
        }
        Shape2::Box { half_extents } => {
            let local = rotation.transpose() * (point - position);
            let clamped = Vec2::new(
                local.x.clamp(-half_extents.x, half_extents.x),
                local.y.clamp(-half_extents.y, half_extents.y),
            );
            let inside = clamped == local;
            let delta = local - clamped;
            let distance = delta.length();
            let normal_local = if inside {
                // Least-penetration axis: the nearest face, signed.
                let dx = half_extents.x - local.x.abs();
                let dy = half_extents.y - local.y.abs();
                if dx <= dy {
                    Vec2::new(local.x.signum(), 0.0)
                } else {
                    Vec2::new(0.0, local.y.signum())
                }
            } else {
                unit_or(delta, Vec2::X)
            };
            Some(ShapeHit2 {
                inside,
                point: if inside {
                    point
                } else {
                    position + rotation * clamped
                },
                normal: rotation * normal_local,
                distance,
            })
        }
        Shape2::Polygon { vertices } => {
            if vertices.len() < 3 {
                return None;
            }
            let world: Vec<Vec2> = vertices.iter().map(|v| position + rotation * *v).collect();
            let ccw = polygon_is_ccw(vertices);
            // Inside: on the inner side of every edge. With a CCW
            // polygon the inner side is the left of each directed edge.
            let mut inside = true;
            let mut best: Option<(f32, Vec2, Vec2)> = None;
            for i in 0..world.len() {
                let a = world[i];
                let b = world[(i + 1) % world.len()];
                let edge = b - a;
                let edge_len = edge.length();
                if edge_len <= f32::MIN_POSITIVE {
                    continue;
                }
                let outward = if ccw {
                    Vec2::new(edge.y, -edge.x) / edge_len
                } else {
                    Vec2::new(-edge.y, edge.x) / edge_len
                };
                // Signed distance to the edge's outward half-space.
                let signed = (point - a).dot(outward);
                if signed > 0.0 {
                    inside = false;
                }
                let on_edge = segment_closest(a, b, point);
                let d = (point - on_edge).length();
                if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                    best = Some((d, on_edge, outward));
                }
            }
            best.map(|(d, p, n)| ShapeHit2 {
                inside,
                // Inside: the query point itself; the normal still names
                // the nearest edge, the cheapest way out.
                point: if inside { point } else { p },
                normal: n,
                distance: if inside { 0.0 } else { d },
            })
        }
    }
}

/// The nearest surface point of a 3D shape.
///
/// The convex hull and convex mesh report `None`, matching the ray
/// cast's documented miss for the unsupported convex shapes.
pub fn closest_point3(
    shape: &Shape3,
    position: Vec3,
    rotation: glam::Quat,
    point: Vec3,
) -> Option<ShapeHit3> {
    match shape {
        Shape3::Sphere { radius } => {
            let r = radius.abs();
            let offset = point - position;
            let len = offset.length();
            let normal = unit_or3(offset, Vec3::Y);
            let inside = len <= r;
            Some(ShapeHit3 {
                inside,
                point: if inside { point } else { position + normal * r },
                normal,
                distance: if inside { 0.0 } else { len - r },
            })
        }
        Shape3::Box { half_extents } => {
            let local = rotation.inverse() * (point - position);
            let clamped = Vec3::new(
                local.x.clamp(-half_extents.x, half_extents.x),
                local.y.clamp(-half_extents.y, half_extents.y),
                local.z.clamp(-half_extents.z, half_extents.z),
            );
            let inside = clamped == local;
            let delta = local - clamped;
            let distance = delta.length();
            let normal_local = if inside {
                // Least-penetration axis: the nearest face, signed.
                let dx = half_extents.x - local.x.abs();
                let dy = half_extents.y - local.y.abs();
                let dz = half_extents.z - local.z.abs();
                if dx <= dy && dx <= dz {
                    Vec3::new(local.x.signum(), 0.0, 0.0)
                } else if dy <= dz {
                    Vec3::new(0.0, local.y.signum(), 0.0)
                } else {
                    Vec3::new(0.0, 0.0, local.z.signum())
                }
            } else {
                unit_or3(delta, Vec3::X)
            };
            Some(ShapeHit3 {
                inside,
                point: if inside {
                    point
                } else {
                    position + rotation * clamped
                },
                normal: rotation * normal_local,
                distance,
            })
        }
        Shape3::Capsule {
            radius,
            half_length,
        } => {
            let r = radius.abs();
            // The capsule's axis is the body's local Y (the same
            // convention the ray cast and the narrowphase use).
            let axis = rotation * Vec3::Y * half_length.abs();
            let a = position + axis;
            let b = position - axis;
            let on_axis = segment_closest3(a, b, point);
            let offset = point - on_axis;
            let len = offset.length();
            let normal = unit_or3(offset, rotation * Vec3::X);
            let inside = len <= r;
            Some(ShapeHit3 {
                inside,
                point: if inside { point } else { on_axis + normal * r },
                normal,
                distance: if inside { 0.0 } else { len - r },
            })
        }
        Shape3::ConvexHull { .. } | Shape3::ConvexMesh { .. } => None,
    }
}

/// The polygon outline of a 2D shape in world space (`None` for a
/// circle, which the SAT routines take as a centre and a radius).
fn outline2(shape: &Shape2, position: Vec2, rotation: Mat2) -> Option<Vec<Vec2>> {
    match shape {
        Shape2::Circle { .. } => None,
        Shape2::Box { half_extents } => Some(
            [
                Vec2::new(-half_extents.x, -half_extents.y),
                Vec2::new(half_extents.x, -half_extents.y),
                Vec2::new(half_extents.x, half_extents.y),
                Vec2::new(-half_extents.x, half_extents.y),
            ]
            .iter()
            .map(|v| position + rotation * *v)
            .collect(),
        ),
        Shape2::Polygon { vertices } => {
            Some(vertices.iter().map(|v| position + rotation * *v).collect())
        }
    }
}

/// Whether two shapes overlap (touch or penetrate) in 2D.
///
/// Analytic per pair through the shared SAT routines. Because SAT
/// reports separation for a shape wholly inside another (no separating
/// axis through the interior), containment is checked explicitly, so a
/// small circle inside a large box still overlaps.
pub fn shapes_overlap2(
    a: &Shape2,
    a_position: Vec2,
    a_rotation: Mat2,
    b: &Shape2,
    b_position: Vec2,
    b_rotation: Mat2,
) -> bool {
    use phieng_collision::sat::{sat_circle_polygon, sat_polygon_polygon};

    let hit = match (a, b) {
        (Shape2::Circle { radius: ra }, Shape2::Circle { radius: rb }) => {
            let sum = ra.abs() + rb.abs();
            (b_position - a_position).length_squared() <= sum * sum
        }
        (Shape2::Circle { radius }, other) => {
            let poly = outline2(other, b_position, b_rotation).unwrap_or_default();
            sat_circle_polygon(a_position, radius.abs(), &poly).is_some()
        }
        (other, Shape2::Circle { radius }) => {
            let poly = outline2(other, a_position, a_rotation).unwrap_or_default();
            sat_circle_polygon(b_position, radius.abs(), &poly).is_some()
        }
        _ => {
            let pa = outline2(a, a_position, a_rotation).unwrap_or_default();
            let pb = outline2(b, b_position, b_rotation).unwrap_or_default();
            sat_polygon_polygon(&pa, &pb).is_some()
        }
    };
    hit || point_in_shape2(a, a_position, a_rotation, b_position)
        || point_in_shape2(b, b_position, b_rotation, a_position)
}

/// The support function of a 3D shape at a pose, matching the
/// narrowphase's mapping exactly (see `World3::support_function_for`).
fn support_for3<'a>(
    shape: &'a Shape3,
    position: Vec3,
    rotation: glam::Quat,
) -> Box<dyn Fn(Vec3) -> Vec3 + 'a> {
    match shape {
        Shape3::Sphere { radius } => {
            let r = radius.abs();
            Box::new(move |dir: Vec3| position + dir.normalize() * r)
        }
        Shape3::Box { half_extents } => {
            let he = *half_extents;
            Box::new(move |dir: Vec3| {
                let local_dir = rotation.inverse() * dir;
                let local_support = Vec3::new(
                    if local_dir.x > 0.0 { he.x } else { -he.x },
                    if local_dir.y > 0.0 { he.y } else { -he.y },
                    if local_dir.z > 0.0 { he.z } else { -he.z },
                );
                position + rotation * local_support
            })
        }
        Shape3::Capsule {
            radius,
            half_length,
        } => {
            let r = radius.abs();
            let pa = position + rotation * (Vec3::Y * half_length.abs());
            let pb = position - rotation * (Vec3::Y * half_length.abs());
            Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_capsule(pa, pb, r, dir))
        }
        Shape3::ConvexHull { vertices } => {
            let verts: Vec<Vec3> = vertices.iter().map(|v| rotation * *v + position).collect();
            Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_convex_hull(&verts, dir))
        }
        Shape3::ConvexMesh { vertices, .. } => {
            let verts: Vec<Vec3> = vertices.iter().map(|v| rotation * *v + position).collect();
            Box::new(move |dir: Vec3| phieng_collision::gjk3d::support_convex_hull(&verts, dir))
        }
    }
}

/// Whether two shapes overlap (touch or penetrate) in 3D.
///
/// GJK decides, with an explicit containment check on top: a sphere
/// wholly inside a box has no support plane that separates it, and the
/// point fallback makes that case unambiguous.
pub fn shapes_overlap3(
    a: &Shape3,
    a_position: Vec3,
    a_rotation: glam::Quat,
    b: &Shape3,
    b_position: Vec3,
    b_rotation: glam::Quat,
) -> bool {
    use phieng_collision::gjk3d::{gjk, GjkResult};
    let sa = support_for3(a, a_position, a_rotation);
    let sb = support_for3(b, b_position, b_rotation);
    let (result, _simplex) = gjk(&*sa, &*sb);
    matches!(result, GjkResult::Colliding)
        || point_in_shape3(a, a_position, a_rotation, b_position)
        || point_in_shape3(b, b_position, b_rotation, a_position)
}
