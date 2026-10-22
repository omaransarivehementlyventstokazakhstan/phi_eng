//! Ray casts against the shapes the engine knows.
//!
//! A ray cast answers "what does this ray hit first?" — the query a
//! game needs to pick a body under the mouse, to fire a hitscan, or to
//! sweep a character controller.
//!
//! Every function here is analytic and exact for the shape it takes, and
//! every one of them is mirrored by `PhiEng.Collision.Query` in the
//! managed build, so the two agree bit for bit.

use glam::{Mat2, Vec2, Vec3};

/// A ray: an origin and a direction.
///
/// The direction need not be normalized: the hit distance is in units of
/// the direction's length, so a normalized direction gives a distance in
/// world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray2 {
    /// Where the ray starts.
    pub origin: Vec2,
    /// The way the ray points.
    pub direction: Vec2,
}

impl Ray2 {
    /// A ray from `origin` along `direction`.
    pub fn new(origin: Vec2, direction: Vec2) -> Self {
        Self { origin, direction }
    }

    /// The point at distance `t` along the ray.
    pub fn at(&self, t: f32) -> Vec2 {
        self.origin + self.direction * t
    }
}

/// A ray in 3D. See [`Ray2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray3 {
    /// Where the ray starts.
    pub origin: Vec3,
    /// The way the ray points.
    pub direction: Vec3,
}

impl Ray3 {
    /// A ray from `origin` along `direction`.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// The point at distance `t` along the ray.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// What a ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit2 {
    /// The distance along the ray.
    pub t: f32,
    /// The hit point, in world space.
    pub point: Vec2,
    /// The surface normal at the hit, in world space, facing the ray.
    pub normal: Vec2,
}

/// A ray hit in 3D. See [`RayHit2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit3 {
    /// The distance along the ray.
    pub t: f32,
    /// The hit point, in world space.
    pub point: Vec3,
    /// The surface normal at the hit, in world space, facing the ray.
    pub normal: Vec3,
}

/// A direction this small is treated as parallel to an axis.
const PARALLEL_EPS: f32 = 1e-9;

/// The 2D cross product: the z component of the 3D cross.
#[inline]
fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Ray against a circle (2D).
///
/// A ray that starts inside the circle does not hit it: the hit is for
/// a ray that enters.
pub fn ray_circle2(origin: Vec2, direction: Vec2, center: Vec2, radius: f32) -> Option<RayHit2> {
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let m = origin - center;
    let a = direction.dot(direction);
    if !a.is_finite() || a <= 0.0 {
        return None;
    }
    let b = 2.0 * m.dot(direction);
    let c = m.dot(m) - radius * radius;
    if c < 0.0 {
        // The ray starts inside the circle: it does not hit it.
        return None;
    }
    let disc = b * b - 4.0 * a * c;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let mut t = (-b - root) / (2.0 * a);
    if t < 0.0 {
        t = (-b + root) / (2.0 * a);
    }
    if t < 0.0 {
        return None;
    }
    let point = origin + direction * t;
    let normal = (point - center) / radius;
    Some(RayHit2 { t, point, normal })
}

/// Ray against an oriented box (2D).
///
/// `rotation` is the box's orientation and `half_extents` its size
/// along the box's own axes.
pub fn ray_obb2(
    origin: Vec2,
    direction: Vec2,
    center: Vec2,
    rotation: Mat2,
    half_extents: Vec2,
) -> Option<RayHit2> {
    // Into the box's own frame.
    let inv = rotation.transpose();
    let o = inv * (origin - center);
    let d = inv * direction;

    let mut t_enter = f32::MIN;
    let mut t_exit = f32::MAX;
    let mut enter_axis = 0usize;
    let mut enter_sign = 1.0f32;

    for axis in 0..2 {
        let h = half_extents[axis];
        if d[axis].abs() < PARALLEL_EPS {
            // Parallel to this slab: a miss unless the ray is inside it.
            if o[axis].abs() > h {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d[axis];
        let t1 = (-h - o[axis]) * inv_d;
        let t2 = (h - o[axis]) * inv_d;
        let (near, far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        if near > t_enter {
            t_enter = near;
            enter_axis = axis;
            enter_sign = if d[axis] > 0.0 { -1.0 } else { 1.0 };
        }
        if far < t_exit {
            t_exit = far;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    if t_enter < 0.0 {
        // The ray starts inside the box: it does not hit it.
        return None;
    }
    let t = t_enter;
    let point = origin + direction * t;
    let mut local_normal = Vec2::ZERO;
    local_normal[enter_axis] = enter_sign;
    let normal = rotation * local_normal;
    Some(RayHit2 { t, point, normal })
}

/// Twice the signed area of a polygon.
///
/// Zero for a polygon whose vertices are collinear, or all the same
/// point: such a polygon has no inside and is not a shape.
fn polygon_area2(vertices: &[Vec2]) -> f32 {
    let n = vertices.len();
    let mut area = 0.0f32;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        area += cross2(a, b);
    }
    area
}

/// Ray against a convex polygon (2D).
///
/// The vertices may be in either winding order: the normal is always
/// turned to face the ray, so the answer does not depend on it.
pub fn ray_polygon2(origin: Vec2, direction: Vec2, vertices: &[Vec2]) -> Option<RayHit2> {
    let n = vertices.len();
    if n < 3 {
        return None;
    }
    if polygon_area2(vertices) == 0.0 {
        // The polygon has no area: it is a line, not a shape.
        return None;
    }
    if point_in_convex_polygon2(origin, vertices) {
        // The ray starts inside: it does not hit the polygon.
        return None;
    }

    let mut best_t = f32::MAX;
    let mut best_normal = Vec2::ZERO;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let edge = b - a;
        let denom = cross2(direction, edge);
        if denom.abs() < PARALLEL_EPS {
            continue;
        }
        let offset = a - origin;
        let t = cross2(offset, edge) / denom;
        let s = cross2(offset, direction) / denom;
        if s < 0.0 || s > 1.0 || t < 0.0 {
            continue;
        }
        if t < best_t {
            best_t = t;
            // The edge normal, facing the ray.
            let mut normal = Vec2::new(edge.y, -edge.x);
            let len = normal.length();
            if len > 0.0 {
                normal /= len;
            }
            if normal.dot(direction) > 0.0 {
                normal = -normal;
            }
            best_normal = normal;
        }
    }
    if best_t == f32::MAX {
        return None;
    }
    let point = origin + direction * best_t;
    Some(RayHit2 {
        t: best_t,
        point,
        normal: best_normal,
    })
}

/// Whether a point is inside a convex polygon (2D).
///
/// The answer does not depend on the winding order.
pub fn point_in_convex_polygon2(point: Vec2, vertices: &[Vec2]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let cross = cross2(b - a, point - a);
        if cross > 0.0 {
            positive = true;
        } else if cross < 0.0 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }
    true
}

/// Ray against a sphere (3D).
///
/// A ray that starts inside the sphere does not hit it.
pub fn ray_sphere3(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<RayHit3> {
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let m = origin - center;
    let a = direction.dot(direction);
    if !a.is_finite() || a <= 0.0 {
        return None;
    }
    let b = 2.0 * m.dot(direction);
    let c = m.dot(m) - radius * radius;
    if c < 0.0 {
        // The ray starts inside the sphere: it does not hit it.
        return None;
    }
    let disc = b * b - 4.0 * a * c;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let mut t = (-b - root) / (2.0 * a);
    if t < 0.0 {
        t = (-b + root) / (2.0 * a);
    }
    if t < 0.0 {
        return None;
    }
    let point = origin + direction * t;
    let normal = (point - center) / radius;
    Some(RayHit3 { t, point, normal })
}

/// Rotate a vector by a quaternion, spelled out.
///
/// The managed mirror does exactly this arithmetic, so the two produce
/// the same bits: a library rotation would round differently in the two
/// languages and the answers would differ in the last bit.
#[inline]
pub fn rotate_vec3(q: glam::Quat, v: Vec3) -> Vec3 {
    let u = Vec3::new(q.x, q.y, q.z);
    let t = u.cross(v) + q.w * v;
    v + 2.0 * u.cross(t)
}

/// Ray against an oriented box (3D).
///
/// `rotation` is the box's orientation and `half_extents` its size
/// along the box's own axes.
pub fn ray_obb3(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: glam::Quat,
    half_extents: Vec3,
) -> Option<RayHit3> {
    // The box's own axes, the same ones the OBB-OBB test uses. The
    // rotation is spelled out, so the managed mirror can do exactly the
    // same arithmetic: the two agree to the last bit.
    let ax = rotate_vec3(rotation, Vec3::X);
    let ay = rotate_vec3(rotation, Vec3::Y);
    let az = rotate_vec3(rotation, Vec3::Z);
    // The ray in the box's frame.
    let offset = origin - center;
    let o = Vec3::new(offset.dot(ax), offset.dot(ay), offset.dot(az));
    let d = Vec3::new(direction.dot(ax), direction.dot(ay), direction.dot(az));

    let mut t_enter = f32::MIN;
    let mut t_exit = f32::MAX;
    let mut enter_axis = 0usize;
    let mut enter_sign = 1.0f32;

    for axis in 0..3 {
        let h = half_extents[axis];
        if d[axis].abs() < PARALLEL_EPS {
            if o[axis].abs() > h {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d[axis];
        let t1 = (-h - o[axis]) * inv_d;
        let t2 = (h - o[axis]) * inv_d;
        let (near, far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        if near > t_enter {
            t_enter = near;
            enter_axis = axis;
            enter_sign = if d[axis] > 0.0 { -1.0 } else { 1.0 };
        }
        if far < t_exit {
            t_exit = far;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    if t_enter < 0.0 {
        return None;
    }
    let t = t_enter;
    let point = origin + direction * t;
    let mut local_normal = Vec3::ZERO;
    local_normal[enter_axis] = enter_sign;
    // Back into world space, through the same axes.
    let normal = ax * local_normal.x + ay * local_normal.y + az * local_normal.z;
    Some(RayHit3 { t, point, normal })
}

/// Ray against a capsule (3D).
///
/// The capsule is the segment `a` to `b` with the given radius.
pub fn ray_capsule3(
    origin: Vec3,
    direction: Vec3,
    a: Vec3,
    b: Vec3,
    radius: f32,
) -> Option<RayHit3> {
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let axis = b - a;
    let axis_len_sq = axis.length_squared();
    if axis_len_sq < PARALLEL_EPS {
        // A sphere.
        return ray_sphere3(origin, direction, a, radius);
    }
    // The ray starts inside the capsule: it does not hit it.
    {
        let o = origin - a;
        let s = (o.dot(axis) / axis_len_sq).clamp(0.0, 1.0);
        let closest = a + axis * s;
        if (origin - closest).length_squared() < radius * radius {
            return None;
        }
    }

    // The infinite cylinder around the axis, then the two caps.
    let o = origin - a;
    let d = direction;
    let axis_dot_d = axis.dot(d);
    let axis_dot_o = axis.dot(o);

    let a_quad = d.dot(d) - axis_dot_d * axis_dot_d / axis_len_sq;
    let b_quad = d.dot(o) - axis_dot_d * axis_dot_o / axis_len_sq;
    let c_quad = o.dot(o) - axis_dot_o * axis_dot_o / axis_len_sq - radius * radius;

    let mut best: Option<RayHit3> = None;
    if a_quad.abs() > PARALLEL_EPS {
        let disc = b_quad * b_quad - a_quad * c_quad;
        if disc >= 0.0 {
            let root = disc.sqrt();
            for t in [(-b_quad - root) / a_quad, (-b_quad + root) / a_quad] {
                if t < 0.0 {
                    continue;
                }
                let point = origin + d * t;
                // The hit must be on the segment, not past its ends.
                let s = (point - a).dot(axis) / axis_len_sq;
                if s < 0.0 || s > 1.0 {
                    continue;
                }
                let normal = (point - (a + axis * s)) / radius;
                consider(&mut best, RayHit3 { t, point, normal });
            }
        }
    }

    // The caps.
    for cap in [a, b] {
        if let Some(hit) = ray_sphere3(origin, direction, cap, radius) {
            consider(&mut best, hit);
        }
    }
    best
}

/// Keep the nearer of two hits.
fn consider(best: &mut Option<RayHit3>, hit: RayHit3) {
    match best {
        Some(current) if current.t <= hit.t => {}
        _ => *best = Some(hit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray straight down at a unit circle centred on the origin.
    #[test]
    fn ray_hits_a_circle() {
        let hit = ray_circle2(Vec2::new(0.0, 5.0), Vec2::new(0.0, -1.0), Vec2::ZERO, 1.0)
            .expect("the ray must hit the circle");
        assert!((hit.t - 4.0).abs() < 1e-5, "t={}", hit.t);
        assert!((hit.point.y - 1.0).abs() < 1e-5);
        assert!(
            (hit.normal - Vec2::Y).length() < 1e-5,
            "the normal faces the ray"
        );
    }

    /// A ray that misses the circle entirely.
    #[test]
    fn ray_misses_a_circle() {
        assert!(ray_circle2(Vec2::new(5.0, 5.0), Vec2::new(0.0, -1.0), Vec2::ZERO, 1.0).is_none());
    }

    /// A ray at a box, from above and from beside.
    #[test]
    fn ray_hits_an_oriented_box() {
        let half = Vec2::new(1.0, 0.5);
        let hit = ray_obb2(
            Vec2::new(0.0, 5.0),
            Vec2::new(0.0, -1.0),
            Vec2::ZERO,
            Mat2::IDENTITY,
            half,
        )
        .expect("the ray must hit the box");
        assert!((hit.t - 4.5).abs() < 1e-5, "t={}", hit.t);
        assert!((hit.normal - Vec2::Y).length() < 1e-5);

        // A ray that passes beside the box.
        assert!(ray_obb2(
            Vec2::new(5.0, 5.0),
            Vec2::new(0.0, -1.0),
            Vec2::ZERO,
            Mat2::IDENTITY,
            half
        )
        .is_none());
    }

    /// A ray at a rotated box: the hit follows the rotation.
    #[test]
    fn ray_hits_a_rotated_box() {
        let rotation = Mat2::from_angle(std::f32::consts::FRAC_PI_4);
        let hit = ray_obb2(
            Vec2::new(0.0, 5.0),
            Vec2::new(0.0, -1.0),
            Vec2::ZERO,
            rotation,
            Vec2::new(1.0, 0.5),
        )
        .expect("the ray must hit the box");
        // The box is tilted, so its corner stands higher than the
        // axis-aligned box's face: the ray reaches it sooner.
        assert!(hit.t < 4.5, "t={}", hit.t);
    }

    /// A ray at a convex square.
    #[test]
    fn ray_hits_a_convex_polygon() {
        let square = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let hit = ray_polygon2(Vec2::new(0.0, 5.0), Vec2::new(0.0, -1.0), &square)
            .expect("the ray must hit the square");
        assert!((hit.t - 4.0).abs() < 1e-5, "t={}", hit.t);
        assert!((hit.normal - Vec2::Y).length() < 1e-5);

        // The winding must not change the answer.
        let reversed = [
            Vec2::new(-1.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(-1.0, -1.0),
        ];
        let other = ray_polygon2(Vec2::new(0.0, 5.0), Vec2::new(0.0, -1.0), &reversed)
            .expect("the ray must hit the square either way");
        assert!((other.t - hit.t).abs() < 1e-5);
        assert!((other.normal - hit.normal).length() < 1e-5);
    }

    /// A ray that misses the square.
    #[test]
    fn ray_misses_a_convex_polygon() {
        let square = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        assert!(ray_polygon2(Vec2::new(5.0, 5.0), Vec2::new(0.0, -1.0), &square).is_none());
    }

    /// A ray at a sphere.
    #[test]
    fn ray_hits_a_sphere() {
        let hit = ray_sphere3(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ZERO,
            1.0,
        )
        .expect("the ray must hit the sphere");
        assert!((hit.t - 4.0).abs() < 1e-5, "t={}", hit.t);
        assert!((hit.normal - Vec3::Z).length() < 1e-5);
    }

    /// A ray at a capsule: the round side and the round cap.
    #[test]
    fn ray_hits_a_capsule() {
        let a = Vec3::new(0.0, 0.0, -1.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        // At the side of the cylinder.
        let hit = ray_capsule3(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            a,
            b,
            0.5,
        )
        .expect("the ray must hit the capsule");
        assert!((hit.t - 4.5).abs() < 1e-4, "t={}", hit.t);

        // At the round cap.
        let hit = ray_capsule3(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            a,
            b,
            0.5,
        )
        .expect("the ray must hit the cap");
        assert!((hit.t - 3.5).abs() < 1e-4, "t={}", hit.t);
    }
}
