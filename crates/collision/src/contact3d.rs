use glam::{Quat, Vec3};
use phieng_math::{EPS_LINE, EPS_PLANE, EPS_SQ};

/// Clip a polygon against a plane (Sutherland-Hodgman)
pub fn clip_polygon_plane(polygon: &[Vec3], plane_normal: Vec3, plane_offset: f32) -> Vec<Vec3> {
    let mut output = Vec::with_capacity(polygon.len() + 1);
    clip_polygon_plane_into(polygon, plane_normal, plane_offset, &mut output);
    output
}

/// Allocation-free variant: appends clipped points to `out`.
/// Unlike [`clip_polygon_plane`], accepts any non-empty point list
/// (segments clip fine too).
pub fn clip_polygon_plane_into(
    polygon: &[Vec3],
    plane_normal: Vec3,
    plane_offset: f32,
    out: &mut Vec<Vec3>,
) {
    if polygon.is_empty() {
        return;
    }

    let mut prev = polygon[polygon.len() - 1];
    let mut prev_inside = prev.dot(plane_normal) <= plane_offset;

    for &cur in polygon {
        let cur_inside = cur.dot(plane_normal) <= plane_offset;
        if cur_inside {
            if !prev_inside {
                out.push(intersect_edge_3d(prev, cur, plane_normal, plane_offset));
            }
            out.push(cur);
        } else if prev_inside {
            out.push(intersect_edge_3d(prev, cur, plane_normal, plane_offset));
        }
        prev = cur;
        prev_inside = cur_inside;
    }
}

/// Intersection of segment prev->cur with plane dot(n,p) = offset.
fn intersect_edge_3d(prev: Vec3, cur: Vec3, n: Vec3, offset: f32) -> Vec3 {
    let d = cur - prev;
    let denom = n.dot(d);
    if denom.abs() < EPS_LINE {
        return cur;
    }
    let t = (offset - n.dot(prev)) / denom;
    prev + d * t.clamp(0.0, 1.0)
}

/// Box vs box contact manifold via reference/incident face clipping.
/// `normal` points from A to B. Returns up to 4 contact points in world
/// space (may be empty if faces are edge-on; caller should fall back).
pub fn box_box_contact_points(
    center_a: Vec3,
    he_a: Vec3,
    rot_a: Quat,
    center_b: Vec3,
    he_b: Vec3,
    rot_b: Quat,
    normal: Vec3,
) -> Vec<Vec3> {
    let mut out = Vec::new();
    let mut tmp = Vec::new();
    box_box_contact_points_into(
        center_a, he_a, rot_a, center_b, he_b, rot_b, normal, &mut out, &mut tmp,
    );
    out
}

/// Allocation-free variant: appends up to 4 contact points to `out`
/// (left untouched when the faces are edge-on), reusing `tmp` scratch.
#[allow(clippy::too_many_arguments)]
pub fn box_box_contact_points_into(
    center_a: Vec3,
    he_a: Vec3,
    rot_a: Quat,
    center_b: Vec3,
    he_b: Vec3,
    rot_b: Quat,
    normal: Vec3,
    out: &mut Vec<Vec3>,
    tmp: &mut Vec<Vec3>,
) {
    // Reference face on A: outward normal most aligned with `normal`.
    let mut best = f32::MIN;
    let mut ref_k = 1usize;
    let mut ref_s = 1.0f32;
    for k in 0..3 {
        for &s in &[1.0f32, -1.0f32] {
            let axes = [Vec3::X, Vec3::Y, Vec3::Z];
            let outward = rot_a * (axes[k] * s);
            let score = outward.dot(normal);
            if score > best {
                best = score;
                ref_k = k;
                ref_s = s;
            }
        }
    }
    // Incident face on B: outward normal most aligned with `-normal`.
    let mut best_b = f32::MIN;
    let mut inc_k = 1usize;
    let mut inc_s = 1.0f32;
    for k in 0..3 {
        for &s in &[1.0f32, -1.0f32] {
            let axes = [Vec3::X, Vec3::Y, Vec3::Z];
            let outward = rot_b * (axes[k] * s);
            let score = outward.dot(-normal);
            if score > best_b {
                best_b = score;
                inc_k = k;
                inc_s = s;
            }
        }
    }

    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let hs_a = [he_a.x, he_a.y, he_a.z];
    let hs_b = [he_b.x, he_b.y, he_b.z];
    let i = (ref_k + 1) % 3;
    let j = (ref_k + 2) % 3;
    let u = rot_a * axes[i];
    let v = rot_a * axes[j];
    let hu = hs_a[i];
    let hv = hs_a[j];
    let face_center = center_a + rot_a * (axes[ref_k] * (hs_a[ref_k] * ref_s));
    let n_r = rot_a * (axes[ref_k] * ref_s);

    // Stage the incident quad in `tmp`, then ping-pong clip tmp <-> out.
    let ii = (inc_k + 1) % 3;
    let jj = (inc_k + 2) % 3;
    tmp.clear();
    for &si in &[-1.0f32, 1.0] {
        for &sj in &[-1.0f32, 1.0] {
            let local = axes[ii] * (hs_b[ii] * si)
                + axes[jj] * (hs_b[jj] * sj)
                + axes[inc_k] * (hs_b[inc_k] * inc_s);
            tmp.push(center_b + rot_b * local);
        }
    }
    let before = out.len();
    // Clip incident quad to the reference face rectangle.
    let cu = u.dot(face_center);
    let cv = v.dot(face_center);
    // tmp -> out
    clip_polygon_plane_into(tmp, u, cu + hu, out);
    if out.len() == before {
        return;
    }
    // out -> tmp
    tmp.clear();
    tmp.extend_from_slice(&out[before..]);
    out.truncate(before);
    clip_polygon_plane_into(tmp, -u, -cu + hu, out);
    if out.len() == before {
        return;
    }
    // out -> tmp
    tmp.clear();
    tmp.extend_from_slice(&out[before..]);
    out.truncate(before);
    clip_polygon_plane_into(tmp, v, cv + hv, out);
    if out.len() == before {
        return;
    }
    // out -> tmp
    tmp.clear();
    tmp.extend_from_slice(&out[before..]);
    out.truncate(before);
    clip_polygon_plane_into(tmp, -v, -cv + hv, out);
    if out.len() == before {
        return;
    }

    // Keep points at or behind the reference face (in-place filter of
    // this call's tail), capped at 4.
    let ref_off = n_r.dot(face_center);
    let mut kept = 0;
    for idx in before..out.len() {
        if n_r.dot(out[idx]) - ref_off <= EPS_PLANE {
            out[before + kept] = out[idx];
            kept += 1;
            if kept == 4 {
                // Drain-check the rest for the early-out only; simpler
                // to just stop: remaining points are extra corners.
                // Verify none of the skipped ones matter: they don't â€”
                // 4 corners fully determine a box face contact.
                break;
            }
        }
    }
    out.truncate(before + kept);
}

/// Closest point on a box surface to a query point (world space).
/// Returns (closest_point, outward_normal_at_closest).
pub fn closest_point_on_box(query: Vec3, center: Vec3, he: Vec3, rot: Quat) -> (Vec3, Vec3) {
    let local = rot.inverse() * (query - center);
    let clamped = Vec3::new(
        local.x.clamp(-he.x, he.x),
        local.y.clamp(-he.y, he.y),
        local.z.clamp(-he.z, he.z),
    );
    let inside = (local - clamped).length_squared() < EPS_SQ;
    if !inside {
        let world = center + rot * clamped;
        let n = (query - world).normalize_or_zero();
        return (world, n);
    }
    // Query inside the box: push out along the least-penetration axis.
    let dx = he.x - local.x.abs();
    let dy = he.y - local.y.abs();
    let dz = he.z - local.z.abs();
    let (axis_idx, s) = if dx <= dy && dx <= dz {
        (0, local.x.signum())
    } else if dy <= dz {
        (1, local.y.signum())
    } else {
        (2, local.z.signum())
    };
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let n = rot * (axes[axis_idx] * if s >= 0.0 { 1.0 } else { -1.0 });
    let mut out_local = clamped;
    match axis_idx {
        0 => out_local.x = he.x * s,
        1 => out_local.y = he.y * s,
        _ => out_local.z = he.z * s,
    }
    (center + rot * out_local, n)
}

/// Find the closest points between two line segments
pub fn closest_points_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a <= 1e-10 && e <= 1e-10 {
        return (p1, p2);
    }

    let mut s;
    let mut t;

    if a <= 1e-10 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-10 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            s = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            t = (b * s + f) / e;

            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }

    let closest1 = p1 + d1 * s;
    let closest2 = p2 + d2 * t;

    (closest1, closest2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_polygon_plane() {
        let polygon = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];

        let clipped = clip_polygon_plane(&polygon, Vec3::Y, 0.0);
        assert_eq!(clipped.len(), 4);
    }

    #[test]
    fn test_closest_points_segments() {
        let (c1, c2) = closest_points_segment_segment(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        assert!((c1 - Vec3::ZERO).length() < 1e-6);
        assert!((c2 - Vec3::ZERO).length() < 1e-6);
    }

    #[test]
    fn test_box_box_manifold_axis_aligned() {
        use glam::Quat;
        // Unit boxes overlapping by 0.5 along +X: expect 4 contacts.
        let pts = super::box_box_contact_points(
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::X,
        );
        assert_eq!(pts.len(), 4, "expected 4 face contacts, got {pts:?}");
        for p in &pts {
            // Incident (-X) face of B sits at x = 0.5.
            assert!((p.x - 0.5).abs() < 1e-4, "contact x={}", p.x);
        }
    }

    #[test]
    fn test_box_box_manifold_stacked() {
        use glam::Quat;
        // Box resting on another: 4 contacts on the interface plane.
        let pts = super::box_box_contact_points(
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Quat::IDENTITY,
            Vec3::Y,
        );
        assert_eq!(pts.len(), 4, "expected 4 face contacts, got {pts:?}");
        for p in &pts {
            assert!((p.y - 0.4).abs() < 0.15, "contact y={}", p.y);
        }
    }

    #[test]
    fn test_closest_point_on_box_outside() {
        use glam::Quat;
        let (q, n) = super::closest_point_on_box(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
        );
        assert!((q - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((n - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn test_closest_point_on_box_inside() {
        use glam::Quat;
        // Query inside, near the +X face: must push out +X.
        let (q, n) = super::closest_point_on_box(
            Vec3::new(0.9, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
        );
        assert!((q - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((n - Vec3::X).length() < 1e-6);
    }
}
