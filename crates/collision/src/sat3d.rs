use glam::{Quat, Vec3};
use phieng_math::EPS_SQ;

/// Which feature pair produced the minimum-penetration axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObbContactKind {
    /// Minimum axis is a face normal of A (reference face on A).
    FaceA,
    /// Minimum axis is a face normal of B (reference face on B).
    FaceB,
    /// Minimum axis is an edge-edge cross axis.
    EdgeEdge,
}

/// Separating-axis test for two oriented boxes (6 face axes + 9
/// edge-edge cross axes, Gottschalk-style).
///
/// Face axes yield *exact* contact normals, which matters for resting
/// contact: GJK/EPA normals wobble by ~1 degree frame to frame on
/// face-on-face contact, and that tilt feeds a torque feedback loop
/// (growing stack jitter, eventual tunneling). Returns the normal
/// pointing from A to B, the penetration depth, and which features to
/// build the manifold from â€” or None when separated.
pub fn sat_obb_obb(
    center_a: Vec3,
    he_a: Vec3,
    rot_a: Quat,
    center_b: Vec3,
    he_b: Vec3,
    rot_b: Quat,
) -> Option<(Vec3, f32, ObbContactKind)> {
    let ax = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let bx = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    let hea = [he_a.x, he_a.y, he_a.z];
    let heb = [he_b.x, he_b.y, he_b.z];
    let d = center_b - center_a;

    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec3::X;
    let mut best_kind = ObbContactKind::FaceA;

    // Face axes of A.
    for i in 0..3 {
        let axis = ax[i];
        let ra = hea[i];
        let rb = heb[0] * axis.dot(bx[0]).abs()
            + heb[1] * axis.dot(bx[1]).abs()
            + heb[2] * axis.dot(bx[2]).abs();
        if d.dot(axis).abs() > ra + rb {
            return None;
        }
        let overlap = ra + rb - d.dot(axis).abs();
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = axis;
            best_kind = ObbContactKind::FaceA;
        }
    }

    // Face axes of B.
    for i in 0..3 {
        let axis = bx[i];
        let rb = heb[i];
        let ra = hea[0] * axis.dot(ax[0]).abs()
            + hea[1] * axis.dot(ax[1]).abs()
            + hea[2] * axis.dot(ax[2]).abs();
        if d.dot(axis).abs() > ra + rb {
            return None;
        }
        let overlap = ra + rb - d.dot(axis).abs();
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = axis;
            best_kind = ObbContactKind::FaceB;
        }
    }

    // Edge-edge cross axes (skipped when edges are parallel).
    // An edge axis only wins by a clear margin: near axis-alignment a
    // cross axis duplicates a face axis to within an ulp, and flip-flop
    // between face and edge manifolds frame to frame injects jitter.
    for i in 0..3 {
        for j in 0..3 {
            let axis = ax[i].cross(bx[j]);
            if axis.length_squared() < EPS_SQ {
                continue;
            }
            let axis = axis.normalize();
            let ra = hea[0] * axis.dot(ax[0]).abs()
                + hea[1] * axis.dot(ax[1]).abs()
                + hea[2] * axis.dot(ax[2]).abs();
            let rb = heb[0] * axis.dot(bx[0]).abs()
                + heb[1] * axis.dot(bx[1]).abs()
                + heb[2] * axis.dot(bx[2]).abs();
            if d.dot(axis).abs() > ra + rb {
                return None;
            }
            let overlap = ra + rb - d.dot(axis).abs();
            if overlap < best_overlap - 1e-6 {
                best_overlap = overlap;
                best_axis = axis;
                best_kind = ObbContactKind::EdgeEdge;
            }
        }
    }

    let normal = if d.dot(best_axis) < 0.0 {
        -best_axis
    } else {
        best_axis
    };
    Some((normal, best_overlap, best_kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    #[test]
    fn test_sat_face_contact_exact_normal() {
        let (n, d, k) = sat_obb_obb(
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
        )
        .expect("must collide");
        assert!((n - Vec3::X).length() < 1e-6, "normal={n}");
        assert!((d - 0.5).abs() < 1e-6, "depth={d}");
        assert!(k == ObbContactKind::FaceA || k == ObbContactKind::FaceB);
    }

    #[test]
    fn test_sat_separated() {
        assert!(sat_obb_obb(
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
        )
        .is_none());
    }

    #[test]
    fn test_sat_stacked_normal_stable() {
        // Slightly rotated top box: min axis must still be +Y face.
        let rot = Quat::from_rotation_y(0.05);
        let (n, d, k) = sat_obb_obb(
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            rot,
        )
        .expect("must collide");
        assert!(n.y > 0.99, "normal={n}");
        assert!((d - 0.1).abs() < 1e-4, "depth={d}");
        assert_eq!(k, ObbContactKind::FaceA);
    }

    #[test]
    fn test_sat_rotated_45_deg() {
        let rot = Quat::from_rotation_z(FRAC_PI_4);
        let res = sat_obb_obb(
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(2.2, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            rot,
        )
        .expect("must collide");
        assert!(res.0.x > 0.9, "normal={}", res.0);
        assert!(res.1 > 0.0);
    }
}
