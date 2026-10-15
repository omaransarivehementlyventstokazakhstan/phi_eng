use glam::{Quat, Vec3};

/// Maximum GJK iterations
const GJK_MAX_ITERATIONS: usize = 32;
/// Convergence epsilon
const GJK_EPSILON: f32 = 1e-6;

/// Result of a GJK query
#[derive(Debug, Clone)]
pub enum GjkResult {
    Separated { distance: f32 },
    Colliding,
}

/// A simplex of up to 4 points in 3D
#[derive(Debug, Clone)]
pub struct Simplex {
    pub points: [Vec3; 4],
    pub count: usize,
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplex {
    pub fn new() -> Self {
        Self {
            points: [Vec3::ZERO; 4],
            count: 0,
        }
    }

    pub fn push(&mut self, point: Vec3) {
        self.points[self.count] = point;
        self.count += 1;
    }

    pub fn get(&self, index: usize) -> Vec3 {
        self.points[index]
    }
}

/// Support function for a sphere
pub fn support_sphere(center: Vec3, radius: f32, direction: Vec3) -> Vec3 {
    center + direction.normalize() * radius
}

/// Support function for an oriented bounding box
pub fn support_obb(center: Vec3, half_extents: Vec3, rotation: Quat, direction: Vec3) -> Vec3 {
    let local_dir = rotation.inverse() * direction;
    let local_support = Vec3::new(
        if local_dir.x > 0.0 {
            half_extents.x
        } else if local_dir.x < 0.0 {
            -half_extents.x
        } else {
            0.0
        },
        if local_dir.y > 0.0 {
            half_extents.y
        } else if local_dir.y < 0.0 {
            -half_extents.y
        } else {
            0.0
        },
        if local_dir.z > 0.0 {
            half_extents.z
        } else if local_dir.z < 0.0 {
            -half_extents.z
        } else {
            0.0
        },
    );
    center + rotation * local_support
}

/// Support function for a capsule
pub fn support_capsule(point_a: Vec3, point_b: Vec3, radius: f32, direction: Vec3) -> Vec3 {
    let ab = point_b - point_a;
    let dir_len = direction.length();
    let dir = if dir_len > 1e-10 {
        direction / dir_len
    } else {
        Vec3::X
    };
    let t = ab.dot(dir);
    let closest = if t.abs() < 1e-10 {
        (point_a + point_b) * 0.5
    } else if t > 0.0 {
        point_b
    } else {
        point_a
    };
    closest + dir * radius
}

/// Support function for a convex hull
pub fn support_convex_hull(vertices: &[Vec3], direction: Vec3) -> Vec3 {
    let mut max_dot = f32::MIN;
    let mut max_point = Vec3::ZERO;

    for vertex in vertices {
        let dot = direction.dot(*vertex);
        if dot > max_dot {
            max_dot = dot;
            max_point = *vertex;
        }
    }

    max_point
}

/// Minkowski difference support: support(A, dir) - support(B, -dir)
pub fn support_minkowski(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
    direction: Vec3,
) -> Vec3 {
    support_a(direction) - support_b(-direction)
}

/// Update simplex and search direction for GJK.
/// Returns true when the simplex contains the origin (collision).
///
/// Reductions follow Ericson's Voronoi-region tests (Real-Time Collision
/// Detection 5.1.5): the simplex is always reduced to the feature closest
/// to the origin. Reducing matters — keeping a stale triangle whose
/// closest feature is an edge makes the search direction point away from
/// the origin, GJK cycles until iteration exhaustion, and the fallthrough
/// reports a false separation (missed contact, tunneling).
fn update_simplex(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    match simplex.count {
        2 => {
            // a = newest point.
            let a = simplex.get(1);
            let b = simplex.get(0);
            let ab = b - a;
            let ao = -a;
            if ab.dot(ao) <= 0.0 {
                // Origin is in vertex A's region: drop B.
                simplex.points = [a, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
                simplex.count = 1;
                *direction = ao;
            } else {
                *direction = ab.cross(ao).cross(ab);
                if direction.length_squared() < GJK_EPSILON {
                    *direction = if ab.x.abs() > ab.y.abs() {
                        Vec3::new(-ab.z, 0.0, ab.x)
                    } else {
                        Vec3::new(0.0, ab.z, -ab.y)
                    };
                }
            }
            false
        }
        3 => {
            let a = simplex.get(2);
            let b = simplex.get(1);
            let c = simplex.get(0);
            solve_triangle(simplex, a, b, c, direction)
        }
        4 => {
            let a = simplex.get(3);
            let b = simplex.get(2);
            let c = simplex.get(1);
            let d = simplex.get(0);
            solve_tetrahedron(simplex, a, b, c, d, direction)
        }
        _ => false,
    }
}

/// Triangle solver: Voronoi-region reduction to the feature closest to
/// the origin (Ericson 5.1.5, p = origin). Symmetric in its inputs.
fn solve_triangle(simplex: &mut Simplex, a: Vec3, b: Vec3, c: Vec3, direction: &mut Vec3) -> bool {
    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    // Vertex A region.
    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        if ao.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [a, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 1;
        *direction = ao;
        return false;
    }

    // Vertex B region.
    let bo = -b;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 {
        if bo.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [b, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 1;
        *direction = bo;
        return false;
    }

    // Edge AB region.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        // d1 - d3 = |ab|^2; guard the degenerate a == b case.
        if d1 - d3 < 1e-12 {
            simplex.points = [a, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
            simplex.count = 1;
            *direction = ao;
            return false;
        }
        let v = d1 / (d1 - d3);
        let closest = a + ab * v;
        if closest.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [b, a, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 2;
        *direction = -closest;
        return false;
    }

    // Vertex C region.
    let co = -c;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 {
        if co.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [c, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 1;
        *direction = co;
        return false;
    }

    // Edge AC region.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        // d2 - d6 = |ac|^2; guard the degenerate a == c case.
        if d2 - d6 < 1e-12 {
            simplex.points = [a, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
            simplex.count = 1;
            *direction = ao;
            return false;
        }
        let w = d2 / (d2 - d6);
        let closest = a + ac * w;
        if closest.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [c, a, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 2;
        *direction = -closest;
        return false;
    }

    // Edge BC region.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // (d4 - d3) + (d5 - d6) = |bc|^2; guard b == c.
        if (d4 - d3) + (d5 - d6) < 1e-12 {
            simplex.points = [b, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
            simplex.count = 1;
            *direction = bo;
            return false;
        }
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + (c - b) * w;
        if closest.length_squared() < GJK_EPSILON {
            return true;
        }
        simplex.points = [c, b, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 2;
        *direction = -closest;
        return false;
    }

    // Face region: origin projects inside the triangle.
    let denom = va + vb + vc;
    if denom.abs() < 1e-12 {
        // Degenerate (zero-area) triangle: keep the newest vertex.
        simplex.points = [a, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        simplex.count = 1;
        *direction = ao;
        return false;
    }
    let v = vb / denom;
    let w = vc / denom;
    let closest = a + ab * v + ac * w;
    if closest.length_squared() < GJK_EPSILON {
        return true;
    }
    let n = ab.cross(ac);
    // Face normal pointing toward the origin side.
    *direction = if n.dot(ao) > 0.0 { n } else { -n };
    false
}

/// Tetrahedron solver: reduce to the face the origin is outside of.
/// All four faces are tested (not just the three containing the newest
/// point): skipping the opposite face is only valid when every prior
/// reduction was exact, which a degenerate cycle can violate.
fn solve_tetrahedron(
    simplex: &mut Simplex,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    direction: &mut Vec3,
) -> bool {
    // (p0, p1, p2, opposite vertex) for each face.
    let faces = [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)];
    for (p0, p1, p2, opp) in faces {
        let mut n = (p1 - p0).cross(p2 - p0);
        if n.length_squared() < 1e-24 {
            continue; // Degenerate face: cannot be "outside" of it.
        }
        if n.dot(opp - p0) > 0.0 {
            n = -n; // Orient outward, away from the opposite vertex.
        }
        if n.dot(-p0) > 0.0 {
            // Origin is outside this face: reduce to it.
            simplex.points = [p0, p1, p2, Vec3::ZERO];
            simplex.count = 3;
            *direction = n;
            return false;
        }
    }
    // Origin is inside (or on) all faces: containment.
    true
}

/// Run GJK algorithm to determine if two convex shapes intersect
/// Returns Separated with distance, or Colliding
pub fn gjk(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
) -> (GjkResult, Simplex) {
    let mut direction = Vec3::new(1.0, 0.0, 0.0);
    let mut simplex = Simplex::new();

    let support = support_minkowski(support_a, support_b, direction);
    simplex.push(support);
    direction = -support;

    for _ in 0..GJK_MAX_ITERATIONS {
        if direction.length_squared() < GJK_EPSILON {
            return (GjkResult::Colliding, simplex);
        }

        let new_point = support_minkowski(support_a, support_b, direction);

        if new_point.dot(direction) < -GJK_EPSILON {
            // Sound separation proof (the whole Minkowski difference
            // lies behind the support plane), but the current simplex
            // is only a subset of it: report its true closest distance
            // rather than simplex.get(0).
            let (_, dist) = simplex_origin_info(&simplex);
            return (
                GjkResult::Separated {
                    distance: dist.max(0.0),
                },
                simplex,
            );
        }

        simplex.push(new_point);

        if update_simplex(&mut simplex, &mut direction) {
            return (GjkResult::Colliding, simplex);
        }
    }

    // Iteration exhaustion: never report separation on a guess. A
    // tetrahedron containing the origin is a collision; otherwise fall
    // back to the true closest distance of the final simplex (NOT
    // simplex.get(0), which is just the first support point and can be
    // arbitrarily far from the actual closest feature).
    let (contained, dist) = simplex_origin_info(&simplex);
    if contained {
        (GjkResult::Colliding, simplex)
    } else {
        (GjkResult::Separated { distance: dist }, simplex)
    }
}

/// Closest point on segment AB to the origin.
fn closest_pt_segment(a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let denom = ab.dot(ab);
    if denom < 1e-24 {
        return a;
    }
    let t = (-a).dot(ab) / denom;
    if t <= 0.0 {
        a
    } else if t >= 1.0 {
        b
    } else {
        a + ab * t
    }
}

/// Closest point on triangle ABC to the origin (Ericson 5.1.5).
fn closest_pt_triangle(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bo = -b;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        return if denom.abs() < 1e-12 {
            a
        } else {
            a + ab * (d1 / denom)
        };
    }

    let co = -c;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        return if denom.abs() < 1e-12 {
            a
        } else {
            a + ac * (d2 / denom)
        };
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        return if denom.abs() < 1e-12 {
            b
        } else {
            b + (c - b) * ((d4 - d3) / denom)
        };
    }

    let denom = va + vb + vc;
    if denom.abs() < 1e-12 {
        return a;
    }
    a + ab * (vb / denom) + ac * (vc / denom)
}

/// Containment + closest distance of a simplex to the origin, for the
/// GJK iteration-exhaustion fallthrough.
fn simplex_origin_info(simplex: &Simplex) -> (bool, f32) {
    match simplex.count {
        0 => (false, 0.0),
        1 => (
            simplex.get(0).length_squared() < GJK_EPSILON,
            simplex.get(0).length(),
        ),
        2 => {
            let closest = closest_pt_segment(simplex.get(1), simplex.get(0));
            (closest.length_squared() < GJK_EPSILON, closest.length())
        }
        3 => {
            let closest = closest_pt_triangle(simplex.get(2), simplex.get(1), simplex.get(0));
            (closest.length_squared() < GJK_EPSILON, closest.length())
        }
        _ => {
            let a = simplex.get(3);
            let b = simplex.get(2);
            let c = simplex.get(1);
            let d = simplex.get(0);
            let faces = [(a, b, c), (a, c, d), (a, d, b), (b, d, c)];
            let mut inside_all = true;
            let mut min_dist = f32::MAX;
            for (p0, p1, p2) in faces {
                let mut n = (p1 - p0).cross(p2 - p0);
                if n.length_squared() < 1e-24 {
                    continue;
                }
                // Outward = away from the simplex interior: use the
                // remaining vertex as the interior reference.
                let interior = a + b + c + d - p0 - p1 - p2;
                if n.dot(interior - p0) > 0.0 {
                    n = -n;
                }
                let plane_dist = n.normalize().dot(-p0);
                if plane_dist > 0.0 {
                    inside_all = false;
                }
                let face_closest = closest_pt_triangle(p0, p1, p2);
                min_dist = min_dist.min(face_closest.length());
            }
            (
                inside_all,
                if min_dist == f32::MAX { 0.0 } else { min_dist },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gjk_separated_spheres() {
        let center_a = Vec3::ZERO;
        let radius_a = 1.0;
        let center_b = Vec3::new(3.0, 0.0, 0.0);
        let radius_b = 1.0;

        let support_a = |dir: Vec3| support_sphere(center_a, radius_a, dir);
        let support_b = |dir: Vec3| support_sphere(center_b, radius_b, dir);

        let (result, _) = gjk(&support_a, &support_b);
        match result {
            GjkResult::Separated { distance } => {
                assert!((distance - 1.0).abs() < 1e-4, "distance={}", distance);
            }
            GjkResult::Colliding => panic!("expected separated"),
        }
    }

    #[test]
    fn test_gjk_colliding_spheres() {
        let center_a = Vec3::ZERO;
        let radius_a = 1.0;
        let center_b = Vec3::new(1.0, 0.0, 0.0);
        let radius_b = 1.0;

        let support_a = |dir: Vec3| support_sphere(center_a, radius_a, dir);
        let support_b = |dir: Vec3| support_sphere(center_b, radius_b, dir);

        let (result, _) = gjk(&support_a, &support_b);
        assert!(matches!(result, GjkResult::Colliding), "expected colliding");
    }

    #[test]
    fn test_support_obb() {
        let center = Vec3::ZERO;
        let half_extents = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::IDENTITY;

        let s = support_obb(center, half_extents, rotation, Vec3::X);
        assert_eq!(s, Vec3::new(1.0, 0.0, 0.0));

        let s = support_obb(center, half_extents, rotation, Vec3::Y);
        assert_eq!(s, Vec3::new(0.0, 2.0, 0.0));

        let s = support_obb(center, half_extents, rotation, Vec3::Z);
        assert_eq!(s, Vec3::new(0.0, 0.0, 3.0));
    }

    #[test]
    fn test_support_capsule() {
        let a = Vec3::new(0.0, -1.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let radius = 0.5;

        let s = support_capsule(a, b, radius, Vec3::X);
        assert_eq!(s, Vec3::new(0.5, 0.0, 0.0));

        let s = support_capsule(a, b, radius, Vec3::Y);
        assert_eq!(s, Vec3::new(0.0, 1.5, 0.0));
    }
}
