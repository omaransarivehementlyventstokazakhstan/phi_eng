use glam::Vec2;
use phieng_math::AABB2;

/// Separating Axis Theorem for convex polygons in 2D.
/// Returns the separation axis (unit length, orientation undefined — caller
/// must orient it A->B via `orient_normal`) and penetration depth,
/// or None if separated.
pub fn sat_polygon_polygon(a: &[Vec2], b: &[Vec2]) -> Option<(Vec2, f32)> {
    if a.len() < 3 || b.len() < 3 {
        return None;
    }
    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec2::ZERO;

    // Check axes from polygon A
    for i in 0..a.len() {
        let j = (i + 1) % a.len();
        let edge = a[j] - a[i];
        if edge.length_squared() < 1e-12 {
            continue;
        }
        let normal = Vec2::new(-edge.y, edge.x).normalize_or_zero();

        let (min_a, max_a) = project_polygon(a, normal);
        let (min_b, max_b) = project_polygon(b, normal);

        if max_a < min_b || max_b < min_a {
            return None; // Separated
        }

        let overlap = f32::min(max_a, max_b) - f32::max(min_a, min_b);
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = normal;
        }
    }

    // Check axes from polygon B
    for i in 0..b.len() {
        let j = (i + 1) % b.len();
        let edge = b[j] - b[i];
        if edge.length_squared() < 1e-12 {
            continue;
        }
        let normal = Vec2::new(-edge.y, edge.x).normalize_or_zero();

        let (min_a, max_a) = project_polygon(a, normal);
        let (min_b, max_b) = project_polygon(b, normal);

        if max_a < min_b || max_b < min_a {
            return None; // Separated
        }

        let overlap = f32::min(max_a, max_b) - f32::max(min_a, min_b);
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = normal;
        }
    }

    if best_axis == Vec2::ZERO {
        return None;
    }
    Some((best_axis, best_overlap))
}

/// Orient a SAT normal so it points from centroid A to centroid B.
pub fn orient_normal(normal: Vec2, centroid_a: Vec2, centroid_b: Vec2) -> Vec2 {
    if (centroid_b - centroid_a).dot(normal) < 0.0 {
        -normal
    } else {
        normal
    }
}

/// Centroid (average) of a vertex set.
pub fn polygon_centroid(verts: &[Vec2]) -> Vec2 {
    if verts.is_empty() {
        return Vec2::ZERO;
    }
    let sum = verts.iter().fold(Vec2::ZERO, |acc, v| acc + *v);
    sum / verts.len() as f32
}

/// Project a polygon onto an axis.
fn project_polygon(polygon: &[Vec2], axis: Vec2) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    for vertex in polygon {
        let projection = axis.dot(*vertex);
        min = min.min(projection);
        max = max.max(projection);
    }

    (min, max)
}

/// SAT for circle vs polygon.
/// Projects both shapes onto each polygon edge normal (matching the C#
/// implementation), so the axis/penetration agree across languages.
/// Normal orientation is undefined — orient A->B with `orient_normal`
/// using the circle center and polygon centroid.
pub fn sat_circle_polygon(center: Vec2, radius: f32, polygon: &[Vec2]) -> Option<(Vec2, f32)> {
    if polygon.len() < 3 || radius <= 0.0 {
        return None;
    }
    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec2::ZERO;

    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let edge = polygon[j] - polygon[i];
        if edge.length_squared() < 1e-12 {
            continue;
        }
        let normal = Vec2::new(-edge.y, edge.x).normalize_or_zero();

        // Project polygon and circle onto this normal.
        let (min_poly, max_poly) = project_polygon(polygon, normal);
        let center_proj = normal.dot(center);
        let min_circle = center_proj - radius;
        let max_circle = center_proj + radius;

        if max_poly < min_circle || max_circle < min_poly {
            return None; // Separated
        }

        let overlap = f32::min(max_poly, max_circle) - f32::max(min_poly, min_circle);
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = normal;
        }
    }

    // Circle center inside polygon: push out along the CLOSEST edge
    // (least penetration). `dist` below is negative for interior points
    // (it measures against the inward edge normal), so the closest face is
    // the MAXIMUM dist. Tracking the minimum would eject through the
    // farthest face instead.
    if point_in_polygon(center, polygon) {
        let mut best_dist = f32::MIN;
        let mut closest_normal = Vec2::ZERO;

        for i in 0..polygon.len() {
            let j = (i + 1) % polygon.len();
            let edge = polygon[j] - polygon[i];
            if edge.length_squared() < 1e-12 {
                continue;
            }
            let normal = Vec2::new(-edge.y, edge.x).normalize_or_zero();
            let dist = normal.dot(polygon[i] - center);

            if dist > best_dist {
                best_dist = dist;
                closest_normal = normal;
            }
        }

        if closest_normal == Vec2::ZERO {
            return None; // All edges degenerate.
        }
        let penetration = radius - best_dist;
        return Some((-closest_normal, penetration));
    }

    if best_axis == Vec2::ZERO {
        return None;
    }
    Some((best_axis, best_overlap))
}

/// Check if a point is inside a polygon
fn point_in_polygon(point: Vec2, polygon: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;

    for i in 0..polygon.len() {
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y))
            && (point.x
                < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y)
                    / (polygon[j].y - polygon[i].y)
                    + polygon[i].x)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// SAT for AABB vs AABB (axis-aligned fast path).
pub fn sat_aabb_aabb(a: &AABB2, b: &AABB2) -> Option<(Vec2, f32)> {
    let overlap_x = f32::min(a.max.x, b.max.x) - f32::max(a.min.x, b.min.x);
    if overlap_x < 0.0 {
        return None;
    }
    let overlap_y = f32::min(a.max.y, b.max.y) - f32::max(a.min.y, b.min.y);
    if overlap_y < 0.0 {
        return None;
    }

    if overlap_x < overlap_y {
        let dir = if b.center().x >= a.center().x {
            1.0
        } else {
            -1.0
        };
        Some((Vec2::new(dir, 0.0), overlap_x))
    } else {
        let dir = if b.center().y >= a.center().y {
            1.0
        } else {
            -1.0
        };
        Some((Vec2::new(0.0, dir), overlap_y))
    }
}

/// Generate contact points for polygon-polygon collision.
///
/// Uses reference/incident face clipping (Sutherland–Hodgman against the
/// side planes of the reference edge, then filtering to points behind the
/// reference face). Tries A-as-reference first, falls back to B-as-reference
/// so rotated boxes and generic convex polygons both yield contacts.
/// Returns up to 2 contact points; may be empty if shapes only touch at a
/// single vertex outside both slabs (caller should fall back to a midpoint).
pub fn polygon_polygon_contact_points(a: &[Vec2], b: &[Vec2], normal: Vec2) -> Vec<Vec2> {
    let mut out = Vec::new();
    let mut tmp = Vec::new();
    polygon_polygon_contact_points_into(a, b, normal, &mut out, &mut tmp);
    out
}

/// Allocation-free variant: appends up to 2 contact points to `out`
/// (left untouched when there is no contact), reusing `tmp` scratch.
/// Ideal for per-pair narrowphase loops that must not allocate.
pub fn polygon_polygon_contact_points_into(
    a: &[Vec2],
    b: &[Vec2],
    normal: Vec2,
    out: &mut Vec<Vec2>,
    tmp: &mut Vec<Vec2>,
) {
    if a.len() < 3 || b.len() < 3 || normal.length_squared() < 1e-12 {
        return;
    }
    // Ensure the normal points from A to B.
    let n = orient_normal(normal, polygon_centroid(a), polygon_centroid(b));

    let before = out.len();
    clip_contacts_into(a, b, n, out, tmp);
    if out.len() == before {
        clip_contacts_into(b, a, -n, out, tmp);
    }
    // Cap at 2 points, dropping extras from this call only.
    if out.len() > before + 2 {
        out.truncate(before + 2);
    }
}

/// Allocation-free Sutherland–Hodgman clip of an incident edge against
/// a reference edge: appends clipped points to `out`, using `tmp` as
/// scratch. `out` is left untouched when the clip yields nothing.
fn clip_contacts_into(
    reference: &[Vec2],
    incident: &[Vec2],
    normal: Vec2,
    out: &mut Vec<Vec2>,
    tmp: &mut Vec<Vec2>,
) {
    let (ref_i, _) = find_best_edge(reference, normal);
    let (inc_i, _) = find_best_edge(incident, -normal);

    let r1 = reference[ref_i];
    let r2 = reference[(ref_i + 1) % reference.len()];
    let s1 = incident[inc_i];
    let s2 = incident[(inc_i + 1) % incident.len()];

    // Reference face normal, oriented to agree with the collision normal
    // (handles both CW and CCW windings).
    let edge = r2 - r1;
    if edge.length_squared() < 1e-12 {
        return;
    }
    let mut face_normal = Vec2::new(edge.y, -edge.x).normalize_or_zero(); // outward for CCW
    if face_normal.dot(normal) < 0.0 {
        face_normal = -face_normal;
    }

    // Clip incident segment to the side slab of the reference edge,
    // ping-ponging between `out`'s tail region and `tmp`.
    let tangent = (r2 - r1).normalize_or_zero();
    let before = out.len();
    // Stage the segment in tmp, then clip tmp -> out, then out -> tmp.
    tmp.clear();
    tmp.push(s1);
    tmp.push(s2);
    out.truncate(before);
    clip_to_plane_into(tmp, -tangent, -tangent.dot(r1), out);
    if out.len() == before {
        return;
    }
    tmp.clear();
    // Move current points to tmp, clip back into out.
    tmp.extend_from_slice(&out[before..]);
    out.truncate(before);
    clip_to_plane_into(tmp, tangent, tangent.dot(r2), out);
    if out.len() == before {
        return;
    }

    // Keep points behind (or on) the reference face.
    let ref_offset = face_normal.dot(r1);
    let mut kept = 0;
    for i in before..out.len() {
        if face_normal.dot(out[i]) - ref_offset <= 0.001 {
            out[before + kept] = out[i];
            kept += 1;
        }
    }
    out.truncate(before + kept);
}

/// Allocation-free Sutherland–Hodgman clip of a point list against a
/// single half-plane `dot(n, p) <= offset`: appends clipped points to
/// `out`.
fn clip_to_plane_into(points: &[Vec2], n: Vec2, offset: f32, out: &mut Vec<Vec2>) {
    if points.is_empty() {
        return;
    }
    let mut prev = points[points.len() - 1];
    let mut prev_inside = prev.dot(n) <= offset;

    for &cur in points {
        let cur_inside = cur.dot(n) <= offset;
        if cur_inside {
            if !prev_inside {
                out.push(intersect_edge(prev, cur, n, offset));
            }
            out.push(cur);
        } else if prev_inside {
            out.push(intersect_edge(prev, cur, n, offset));
        }
        prev = cur;
        prev_inside = cur_inside;
    }
}

/// Intersection of segment prev->cur with plane dot(n,p) = offset.
fn intersect_edge(prev: Vec2, cur: Vec2, n: Vec2, offset: f32) -> Vec2 {
    let d = cur - prev;
    let denom = n.dot(d);
    if denom.abs() < 1e-9 {
        return cur;
    }
    let t = (offset - n.dot(prev)) / denom;
    prev + d * t.clamp(0.0, 1.0)
}

/// Find the edge whose outward normal best aligns with `dir`.
/// Polygons are expected wound CCW (outward = (edge.y, -edge.x)).
/// Returns (edge_start_index, outward_normal).
fn find_best_edge(polygon: &[Vec2], dir: Vec2) -> (usize, Vec2) {
    let mut best_dot = f32::MIN;
    let mut best_idx = 0;
    let mut best_normal = Vec2::ZERO;

    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let edge = polygon[j] - polygon[i];
        if edge.length_squared() < 1e-12 {
            continue;
        }
        let outward = Vec2::new(edge.y, -edge.x).normalize_or_zero();
        let dot = outward.dot(dir);
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
            best_normal = outward;
        }
    }

    (best_idx, best_normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square(size: f32) -> Vec<Vec2> {
        let half = size / 2.0;
        vec![
            Vec2::new(-half, -half),
            Vec2::new(half, -half),
            Vec2::new(half, half),
            Vec2::new(-half, half),
        ]
    }

    fn make_square_at(size: f32, offset: Vec2) -> Vec<Vec2> {
        make_square(size).iter().map(|v| *v + offset).collect()
    }

    /// Rotate polygon around origin then translate.
    fn transform_poly(poly: &[Vec2], angle: f32, offset: Vec2) -> Vec<Vec2> {
        let rot = glam::Mat2::from_angle(angle);
        poly.iter().map(|v| rot * *v + offset).collect()
    }

    #[test]
    fn test_sat_no_collision() {
        let a = make_square(1.0);
        let b: Vec<Vec2> = make_square(1.0)
            .iter()
            .map(|v| *v + Vec2::new(5.0, 0.0))
            .collect();

        let result = sat_polygon_polygon(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_sat_collision() {
        let a = make_square(1.0);
        let b: Vec<Vec2> = make_square(1.0)
            .iter()
            .map(|v| *v + Vec2::new(0.5, 0.0))
            .collect();

        let result = sat_polygon_polygon(&a, &b);
        assert!(result.is_some());
    }

    #[test]
    fn test_sat_overlapping_boxes_have_contacts() {
        let a = make_square(1.0);
        let b = make_square_at(1.0, Vec2::new(0.5, 0.0));
        let (n, _) = sat_polygon_polygon(&a, &b).expect("must collide");
        let pts = polygon_polygon_contact_points(&a, &b, n);
        assert!(!pts.is_empty(), "axis-aligned overlap must yield contacts");
        assert!(pts.len() <= 2);
    }

    #[test]
    fn test_sat_rotated_box_has_contacts() {
        let a = make_square(2.0);
        let b = transform_poly(&make_square(2.0), 0.5, Vec2::new(1.0, 0.3));
        let (n, _) = sat_polygon_polygon(&a, &b).expect("must collide");
        let pts = polygon_polygon_contact_points(&a, &b, n);
        assert!(!pts.is_empty(), "rotated overlap must yield contacts");
    }

    #[test]
    fn test_sat_triangle_vs_pentagon() {
        let tri = vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
        ];
        let mut pent = Vec::new();
        for i in 0..5 {
            let ang = 2.0 * std::f32::consts::PI * i as f32 / 5.0;
            pent.push(Vec2::new(ang.cos(), ang.sin()) + Vec2::new(0.5, 0.0));
        }
        let (n, _) = sat_polygon_polygon(&tri, &pent).expect("must collide");
        let pts = polygon_polygon_contact_points(&tri, &pent, n);
        assert!(!pts.is_empty(), "tri vs pentagon must yield contacts");
    }

    #[test]
    fn test_sat_normal_orientation() {
        let a = make_square(1.0);
        let b = make_square_at(1.0, Vec2::new(0.5, 0.0));
        let (n, _) = sat_polygon_polygon(&a, &b).expect("must collide");
        let oriented = orient_normal(n, polygon_centroid(&a), polygon_centroid(&b));
        assert!(oriented.x > 0.0, "normal must point A->B (+x)");
    }

    #[test]
    fn test_circle_polygon_matches_projection() {
        // Circle overlapping the right face of a box.
        let b = make_square(2.0);
        let res = sat_circle_polygon(Vec2::new(1.5, 0.0), 1.0, &b);
        assert!(res.is_some());
        let (_, pen) = res.unwrap();
        assert!(
            (pen - 0.5).abs() < 1e-4,
            "penetration should be 0.5, got {pen}"
        );
    }

    #[test]
    fn test_circle_polygon_separated() {
        let b = make_square(2.0);
        assert!(sat_circle_polygon(Vec2::new(5.0, 0.0), 0.5, &b).is_none());
    }

    #[test]
    fn test_circle_inside_polygon_pushes_out_nearest_face() {
        // Circle center inside the box, near the top face: must eject UP
        // through the nearest face (pen = r + gap), not down through the box.
        let b = make_square(2.0); // half = 1
        let (n, pen) = sat_circle_polygon(Vec2::new(0.0, 0.9), 0.5, &b).expect("must collide");
        assert!(n.y > 0.9, "normal must point up out the near face, got {n}");
        assert!(n.x.abs() < 1e-4);
        assert!(
            (pen - 0.6).abs() < 1e-4,
            "penetration should be 0.6, got {pen}"
        );
    }

    #[test]
    fn test_aabb_aabb() {
        let a = AABB2::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let b = AABB2::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let (n, pen) = sat_aabb_aabb(&a, &b).expect("must overlap");
        assert!((pen - 1.0).abs() < 1e-6);
        assert!(n.length_squared() > 0.5);
        let c = AABB2::new(Vec2::new(5.0, 5.0), Vec2::new(6.0, 6.0));
        assert!(sat_aabb_aabb(&a, &c).is_none());
    }

    #[test]
    fn test_point_in_polygon() {
        let square = make_square(2.0);
        assert!(point_in_polygon(Vec2::new(0.0, 0.0), &square));
        assert!(!point_in_polygon(Vec2::new(2.0, 0.0), &square));
    }
}
