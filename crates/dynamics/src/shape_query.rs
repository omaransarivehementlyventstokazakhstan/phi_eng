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
