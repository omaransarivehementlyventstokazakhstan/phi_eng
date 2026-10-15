use crate::gjk3d::{support_minkowski, GjkResult, Simplex};
use glam::Vec3;

const EPA_MAX_ITERATIONS: usize = 64;
const EPA_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone)]
pub struct EpaResult {
    pub normal: Vec3,
    pub depth: f32,
    pub contact_point_a: Vec3,
    pub contact_point_b: Vec3,
}

#[derive(Debug, Clone)]
struct Face {
    pub indices: [usize; 3],
    pub normal: Vec3,
    pub distance: f32,
}

impl Face {
    fn new(a: usize, b: usize, c: usize, normal: Vec3, distance: f32) -> Self {
        Self {
            indices: [a, b, c],
            normal,
            distance,
        }
    }
}

fn barycentric_coords(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> Vec3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-9 {
        return Vec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    Vec3::new(u, v, w)
}

pub fn gjk_epa(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
) -> Option<EpaResult> {
    gjk_epa_with_buffers(support_a, support_b, &mut EpaBuffers::new())
}

/// Allocation-free variant writing EPA working state into
/// caller-owned `buffers`.
pub fn gjk_epa_with_buffers(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
    buffers: &mut EpaBuffers,
) -> Option<EpaResult> {
    let (result, simplex) = crate::gjk3d::gjk(support_a, support_b);
    match result {
        GjkResult::Colliding => {
            // GJK often terminates on a lower-dimensional simplex
            // (origin on an edge/vertex) or a tetrahedron with the
            // origin on its boundary. Feeding those to EPA poisons it:
            // faces through the origin all read distance ~0 and EPA
            // orbits them until iteration exhaustion, returning a
            // garbage normal/depth. So EPA only ever starts from a
            // STRICTLY containing tetrahedron built by search below.
            // (The GJK simplex is just the overlap trigger.)
            if simplex.count >= 4
                && origin_in_tet_strict(
                    simplex.get(0),
                    simplex.get(1),
                    simplex.get(2),
                    simplex.get(3),
                    STRICT_TET_MARGIN,
                )
            {
                epa_with_buffers(support_a, support_b, &simplex, buffers)
            } else if let Some(tet) = seed_tetrahedron(support_a, support_b) {
                epa_with_buffers(support_a, support_b, &tet, buffers)
            } else {
                // Grazing touch or pathological sliver (inradius below
                // STRICT_TET_MARGIN): report no contact this frame rather
                // than a guessed depth (a guessed depth injects energy via
                // Baumgarte and destabilizes stacks). Penetration below
                // the margin is solver-irrelevant anyway (slop = 0.005);
                // deeper overlaps seed successfully next frame.
                None
            }
        }
        GjkResult::Separated { .. } => None,
    }
}

/// Minimum clearance (origin distance from every face) required of an
/// EPA seed tetrahedron. Faces nearer than this poison EPA expansion
/// (dual ~0-distance faces orbit to iteration exhaustion). Overlaps
/// shallower than this are solver-irrelevant (slop = 0.005).
const STRICT_TET_MARGIN: f32 = 1e-4;

/// Strict containment: origin bounded away from every face by at least
/// `margin`, with non-zero volume AND every opposite vertex bounded away
/// from its face plane by `margin`. The altitude requirement is the load-
/// bearing one: near-coplanar quadruples (slivers) decide the outward
/// orientation by catastrophic cancellation — float noise, opposite signs
/// in Rust vs C# — so a clearance-only test calls the same sliver
/// "containing" in one language and "outside" in the other. Requiring
/// altitude kills slivers before the sign can matter.
fn origin_in_tet_strict(a: Vec3, b: Vec3, c: Vec3, d: Vec3, margin: f32) -> bool {
    if ((b - a).cross(c - a)).dot(d - a).abs() < 1e-12 {
        return false;
    }
    let faces = [(a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a)];
    for (v0, v1, v2, opp) in faces {
        let n = (v1 - v0).cross(v2 - v0);
        if n.length_squared() < 1e-20 {
            return false;
        }
        let n = n.normalize();
        let altitude = n.dot(opp - v0).abs();
        if altitude < margin {
            return false;
        }
        let outward = if n.dot(opp - v0) > 0.0 { -n } else { n };
        if outward.dot(-v0) > -margin {
            return false;
        }
    }
    true
}

/// Minimum face clearance of tetrahedron (abcd): the largest margin by
/// which it strictly contains the origin, or `f32::MIN` when it does
/// not contain the origin at all. Takes the minimum of origin clearance
/// AND opposite-vertex altitude (see [`origin_in_tet_strict`): a sliver
/// can sit 0.78 from the origin on three faces while its fourth face
/// plane grazes a vertex — scoring altitude down-ranks it below any
/// well-conditioned tetrahedron, deterministically, in both languages.
fn tet_clearance(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    if ((b - a).cross(c - a)).dot(d - a).abs() < 1e-12 {
        return f32::MIN;
    }
    let faces = [(a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a)];
    let mut clearance = f32::MAX;
    for (v0, v1, v2, opp) in faces {
        let n = (v1 - v0).cross(v2 - v0);
        if n.length_squared() < 1e-20 {
            return f32::MIN;
        }
        let n = n.normalize();
        clearance = clearance.min(n.dot(opp - v0).abs());
        let outward = if n.dot(opp - v0) > 0.0 { -n } else { n };
        clearance = clearance.min(-outward.dot(-v0));
    }
    clearance
}

/// Build EPA's initial tetrahedron by deterministic search: support
/// points in 6 axis + 8 diagonal directions, then the quadruple with
/// maximum face clearance (first-best wins, so it is reproducible).
/// Returns None when no quadruple strictly contains the origin by
/// [`STRICT_TET_MARGIN`] — a grazing touch, not a missed overlap.
fn seed_tetrahedron(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
) -> Option<Simplex> {
    let support = |dir: Vec3| support_minkowski(support_a, support_b, dir);
    let inv_sqrt3 = 0.57735027;
    let dirs = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        -Vec3::X,
        -Vec3::Y,
        -Vec3::Z,
        Vec3::new(1.0, 1.0, 1.0) * inv_sqrt3,
        Vec3::new(1.0, 1.0, -1.0) * inv_sqrt3,
        Vec3::new(1.0, -1.0, 1.0) * inv_sqrt3,
        Vec3::new(1.0, -1.0, -1.0) * inv_sqrt3,
        Vec3::new(-1.0, 1.0, 1.0) * inv_sqrt3,
        Vec3::new(-1.0, 1.0, -1.0) * inv_sqrt3,
        Vec3::new(-1.0, -1.0, 1.0) * inv_sqrt3,
        Vec3::new(-1.0, -1.0, -1.0) * inv_sqrt3,
    ];
    let mut pts = [Vec3::ZERO; 14];
    let mut n_pts = 0;
    for dir in dirs {
        let p = support(dir);
        // Dedupe: repeated support points add nothing but degenerate
        // quadruples.
        if (0..n_pts).all(|i| (pts[i] - p).length_squared() > 1e-12) {
            pts[n_pts] = p;
            n_pts += 1;
        }
    }
    let mut best: Option<(Simplex, f32)> = None;
    for i in 0..n_pts {
        for j in (i + 1)..n_pts {
            for k in (j + 1)..n_pts {
                for l in (k + 1)..n_pts {
                    let clearance = tet_clearance(pts[i], pts[j], pts[k], pts[l]);
                    let better = match &best {
                        Some((_, c)) => clearance > *c,
                        None => clearance > STRICT_TET_MARGIN,
                    };
                    // Strict `>` (not `>=`): first-best wins ties,
                    // keeping the search deterministic.
                    if clearance > STRICT_TET_MARGIN && better {
                        let mut tet = Simplex::new();
                        tet.push(pts[i]);
                        tet.push(pts[j]);
                        tet.push(pts[k]);
                        tet.push(pts[l]);
                        best = Some((tet, clearance));
                    }
                }
            }
        }
    }
    best.map(|(tet, _)| tet)
}

/// Reusable EPA working buffers. A world keeps one and passes it to
/// [`epa_with_buffers`] for every GJK-positive pair, so the steady
/// state performs no allocations inside EPA.
#[derive(Debug, Default)]
pub struct EpaBuffers {
    vertices: Vec<Vec3>,
    faces: Vec<Face>,
    horizon: Vec<(usize, usize)>,
    visible: Vec<bool>,
}

impl EpaBuffers {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(64),
            faces: Vec::with_capacity(64),
            horizon: Vec::with_capacity(16),
            visible: Vec::with_capacity(64),
        }
    }
}

pub fn epa(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
    simplex: &Simplex,
) -> Option<EpaResult> {
    epa_with_buffers(support_a, support_b, simplex, &mut EpaBuffers::new())
}

/// Allocation-free EPA variant writing into caller-owned `buffers`.
pub fn epa_with_buffers(
    support_a: &dyn Fn(Vec3) -> Vec3,
    support_b: &dyn Fn(Vec3) -> Vec3,
    simplex: &Simplex,
    buffers: &mut EpaBuffers,
) -> Option<EpaResult> {
    if simplex.count != 4 {
        return None;
    }

    let vertices = &mut buffers.vertices;
    vertices.clear();
    for i in 0..4 {
        vertices.push(simplex.get(i));
    }

    let faces = &mut buffers.faces;
    faces.clear();

    let initial_faces = [(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)];

    for &(i0, i1, i2, opp_idx) in &initial_faces {
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        let opp = vertices[opp_idx];

        let n = (v1 - v0).cross(v2 - v0);
        let n_len = n.length();
        if n_len < 1e-10 {
            continue;
        }
        let mut normal = n / n_len;
        if normal.dot(opp - vertices[i0]) > 0.0 {
            normal = -normal;
        }
        let distance = normal.dot(vertices[i0]);
        faces.push(Face::new(i0, i1, i2, normal, distance));
    }

    for _iter in 0..EPA_MAX_ITERATIONS {
        if faces.is_empty() {
            break;
        }

        let mut min_dist = f32::MAX;
        let mut min_idx = 0;
        for (i, face) in faces.iter().enumerate() {
            if face.distance < min_dist {
                min_dist = face.distance;
                min_idx = i;
            }
        }

        let face = &faces[min_idx];
        let support = support_minkowski(support_a, support_b, face.normal);
        let support_dist = support.dot(face.normal);

        if support_dist - face.distance < EPA_TOLERANCE {
            let v0 = vertices[face.indices[0]];
            let v1 = vertices[face.indices[1]];
            let v2 = vertices[face.indices[2]];
            let bary = barycentric_coords(v0, v1, v2, support);

            let contact = v0 * bary.x + v1 * bary.y + v2 * bary.z;

            return Some(EpaResult {
                normal: face.normal,
                depth: face.distance,
                contact_point_a: contact,
                contact_point_b: contact,
            });
        }

        vertices.push(support);
        let new_vertex_idx = vertices.len() - 1;

        let visible = &mut buffers.visible;
        visible.clear();
        visible.extend(
            faces
                .iter()
                .map(|f| f.normal.dot(support - vertices[f.indices[0]]) > 0.0),
        );

        // All faces visible: the apex overhangs the whole polytope, so
        // the horizon is empty and expanding would annihilate it. Return
        // the current minimum face instead: its distance UNDERestimates
        // the true depth (polytope faces cut inside the shape), and an
        // underestimated depth is a weaker solver push — always stable,
        // unlike a guessed overestimate.
        if visible.iter().all(|&v| v) {
            let v0 = vertices[face.indices[0]];
            let v1 = vertices[face.indices[1]];
            let v2 = vertices[face.indices[2]];
            let bary = barycentric_coords(v0, v1, v2, support);
            let contact = v0 * bary.x + v1 * bary.y + v2 * bary.z;
            return Some(EpaResult {
                normal: face.normal,
                depth: face.distance,
                contact_point_a: contact,
                contact_point_b: contact,
            });
        }

        let horizon_edges = &mut buffers.horizon;
        horizon_edges.clear();
        for (i, f) in faces.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            let edges = [
                (f.indices[0], f.indices[1]),
                (f.indices[1], f.indices[2]),
                (f.indices[2], f.indices[0]),
            ];
            for (u, v) in edges {
                let mut found = false;
                for idx in (0..horizon_edges.len()).rev() {
                    if horizon_edges[idx].0 == v && horizon_edges[idx].1 == u {
                        horizon_edges.swap_remove(idx);
                        found = true;
                        break;
                    }
                }
                if !found {
                    horizon_edges.push((u, v));
                }
            }
        }

        // Drop visible faces in place (order is irrelevant to EPA).
        let mut i = 0;
        while i < faces.len() {
            if visible[i] {
                faces.swap_remove(i);
                visible.swap_remove(i);
            } else {
                i += 1;
            }
        }

        for &(u, v) in horizon_edges.iter() {
            let v0 = vertices[u];
            let v1 = vertices[v];
            let v2 = vertices[new_vertex_idx];
            let n = (v1 - v0).cross(v2 - v0);
            let n_len = n.length();
            if n_len < 1e-10 {
                continue;
            }
            let mut normal = n / n_len;
            // Origin is inside the polytope: outward normal points away
            // from it, i.e. normal.dot(v0) > 0.
            if normal.dot(v0) < 0.0 {
                normal = -normal;
            }
            let distance = normal.dot(v0);
            if distance > -1e-6 {
                faces.push(Face::new(u, v, new_vertex_idx, normal, distance.max(0.0)));
            }
        }
    }

    if faces.is_empty() {
        return None;
    }

    let mut min_dist = f32::MAX;
    let mut best_face: Option<Face> = None;
    for f in faces.iter() {
        if f.distance < min_dist {
            min_dist = f.distance;
            best_face = Some(f.clone());
        }
    }

    if let Some(face) = best_face {
        let support = support_minkowski(support_a, support_b, face.normal);
        let v0 = vertices[face.indices[0]];
        let v1 = vertices[face.indices[1]];
        let v2 = vertices[face.indices[2]];
        let bary = barycentric_coords(v0, v1, v2, support);
        let contact = v0 * bary.x + v1 * bary.y + v2 * bary.z;

        return Some(EpaResult {
            normal: face.normal,
            depth: face.distance,
            contact_point_a: contact,
            contact_point_b: contact,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk3d::{support_obb, support_sphere};
    use glam::Quat;

    #[test]
    fn test_epa_sphere_sphere() {
        let center_a = Vec3::ZERO;
        let radius_a = 1.0;
        let center_b = Vec3::new(1.5, 0.0, 0.0);
        let radius_b = 1.0;

        let support_a = |dir: Vec3| support_sphere(center_a, radius_a, dir);
        let support_b = |dir: Vec3| support_sphere(center_b, radius_b, dir);

        let result = gjk_epa(&support_a, &support_b).expect("should collide");
        assert!(result.normal.x.abs() > 0.99, "normal.x={}", result.normal.x);
        assert!((result.depth - 0.5).abs() < 1e-3, "depth={}", result.depth);
    }

    #[test]
    fn test_epa_obb_obb() {
        let center_a = Vec3::ZERO;
        let half_a = Vec3::new(1.0, 1.0, 1.0);
        let rot_a = Quat::IDENTITY;

        let center_b = Vec3::new(1.5, 0.0, 0.0);
        let half_b = Vec3::new(1.0, 1.0, 1.0);
        let rot_b = Quat::IDENTITY;

        let support_a = |dir: Vec3| support_obb(center_a, half_a, rot_a, dir);
        let support_b = |dir: Vec3| support_obb(center_b, half_b, rot_b, dir);

        let result = gjk_epa(&support_a, &support_b).expect("should collide");
        assert!(result.normal.x.abs() > 0.99, "normal.x={}", result.normal.x);
        assert!((result.depth - 0.5).abs() < 1e-3, "depth={}", result.depth);
    }

    #[test]
    fn test_seed_tetrahedron_strict() {
        // Overlapping spheres: the seed tet must strictly contain the
        // origin by STRICT_TET_MARGIN (boundary containment is what
        // poisoned EPA expansion before).
        let support_a = |dir: Vec3| support_sphere(Vec3::ZERO, 1.0, dir);
        let support_b = |dir: Vec3| support_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0, dir);
        let tet = super::seed_tetrahedron(&support_a, &support_b).expect("must seed");
        assert_eq!(tet.count, 4);
        assert!(super::origin_in_tet_strict(
            tet.get(0),
            tet.get(1),
            tet.get(2),
            tet.get(3),
            super::STRICT_TET_MARGIN,
        ));
        // Full pipeline through the seed path still collides.
        let hit = gjk_epa(&support_a, &support_b).expect("should collide");
        assert!(hit.depth >= 0.0, "depth={}", hit.depth);
    }

    #[test]
    fn test_seed_tetrahedron_separated_is_none() {
        let support_a = |dir: Vec3| support_sphere(Vec3::ZERO, 1.0, dir);
        let support_b = |dir: Vec3| support_sphere(Vec3::new(5.0, 0.0, 0.0), 1.0, dir);
        assert!(super::seed_tetrahedron(&support_a, &support_b).is_none());
    }

    /// Regression: needle overlap (SAT depth 0.003, Minkowski inradius
    /// below STRICT_TET_MARGIN — no margin-healthy tetrahedron exists even
    /// among 200 support directions). EPA must either skip (None, the
    /// documented sliver policy: never guess a depth) or agree with SAT —
    /// never return a disagreeing depth. GJK's boolean must be Colliding
    /// (overlap detection is unconditional).
    #[test]
    fn test_epa_needle_pair_skips_or_agrees() {
        use crate::gjk3d::{support_obb, GjkResult};
        let he_a = Vec3::new(1.4051881, 0.80691856, 1.9476976);
        let he_b = Vec3::new(0.96561944, 0.74839616, 0.6716831);
        let rot_a = Quat::from_xyzw(0.10756003, -0.15346055, 0.48880064, -0.85202974);
        let rot_b = Quat::from_xyzw(-0.7761683, 0.58606684, -0.0012072899, 0.2325661);
        let center_a = Vec3::ZERO;
        let center_b = Vec3::new(0.029467806, 2.1654482, -2.1280866);
        let support_a = |dir: Vec3| support_obb(center_a, he_a, rot_a, dir);
        let support_b = |dir: Vec3| support_obb(center_b, he_b, rot_b, dir);
        assert!(matches!(
            crate::gjk3d::gjk(&support_a, &support_b).0,
            GjkResult::Colliding
        ));
        let (sat_normal, sat_depth, _) =
            crate::sat3d::sat_obb_obb(center_a, he_a, rot_a, center_b, he_b, rot_b)
                .expect("SAT must collide");
        assert!((sat_depth - 0.0029).abs() < 0.001, "sat_depth={sat_depth}");
        if let Some(hit) = gjk_epa(&support_a, &support_b) {
            assert!(
                (hit.depth - sat_depth).abs() <= 0.02 + 0.05 * sat_depth,
                "depth mismatch: epa={} sat={}",
                hit.depth,
                sat_depth
            );
            assert!(hit.normal.dot(sat_normal) > 0.7);
        }
    }

    /// N14 probe: deep near-concentric spheres route GJK through the
    /// degenerate-simplex path and can trigger the all-visible guard.
    /// Assert the conservative min-face fallback is stable frame to
    /// frame: same hemisphere, depth in a tight band — never oscillating.
    #[test]
    fn n14_all_visible_guard_stable() {
        use crate::gjk3d::support_sphere;
        let support_a = |dir: Vec3| support_sphere(Vec3::ZERO, 1.0, dir);
        let support_b = |dir: Vec3| support_sphere(Vec3::new(0.1, 0.0, 0.0), 1.0, dir);
        let first = gjk_epa(&support_a, &support_b).expect("n14: must collide");
        for frame in 0..60 {
            let hit = gjk_epa(&support_a, &support_b).expect("n14: must collide");
            assert!(
                hit.normal.dot(first.normal) > 0.9,
                "n14 frame {frame}: normal flipped: {} vs {}",
                hit.normal,
                first.normal
            );
            assert!(
                (hit.depth - first.depth).abs() <= 0.05 * first.depth + 1e-4,
                "n14 frame {frame}: depth oscillated: {} vs {}",
                hit.depth,
                first.depth
            );
        }
    }

    #[test]
    fn test_epa_separated() {
        let center_a = Vec3::ZERO;
        let radius_a = 1.0;
        let center_b = Vec3::new(5.0, 0.0, 0.0);
        let radius_b = 1.0;

        let support_a = |dir: Vec3| support_sphere(center_a, radius_a, dir);
        let support_b = |dir: Vec3| support_sphere(center_b, radius_b, dir);

        assert!(gjk_epa(&support_a, &support_b).is_none());
    }

    /// Regression: fuzz seed 0x9E37… produced two rotated boxes whose
    /// corners interpenetrate (SAT depth 0.44) but GJK cycled to
    /// iteration exhaustion and reported `Separated` — a missed contact.
    /// Root cause was the triangle solver never reducing to edge/vertex
    /// regions plus a fallthrough that reported `simplex.get(0)` as the
    /// distance.
    #[test]
    fn test_gjk_no_false_separation_rotated_boxes() {
        use crate::gjk3d::{support_obb, GjkResult};
        use crate::sat3d::sat_obb_obb;
        let he_a = Vec3::new(0.58751166, 1.9327657, 1.3370173);
        let he_b = Vec3::new(0.44751966, 1.9226546, 0.4482013);
        let rot_a = Quat::from_xyzw(-0.61496, -0.22940487, -0.7301727, -0.1898564);
        let rot_b = Quat::from_xyzw(-0.3793444, -0.05538883, -0.46066985, -0.8005081);
        let center_a = Vec3::ZERO;
        let center_b = Vec3::new(0.8932357, -0.9307329, -1.6851844);

        let support_a = |dir: Vec3| support_obb(center_a, he_a, rot_a, dir);
        let support_b = |dir: Vec3| support_obb(center_b, he_b, rot_b, dir);

        assert!(
            matches!(
                crate::gjk3d::gjk(&support_a, &support_b).0,
                GjkResult::Colliding
            ),
            "GJK must collide on interpenetrating boxes"
        );
        let (sat_normal, sat_depth, _) =
            sat_obb_obb(center_a, he_a, rot_a, center_b, he_b, rot_b).expect("SAT must collide");
        let hit = gjk_epa(&support_a, &support_b).expect("EPA must collide");
        assert!((hit.depth - sat_depth).abs() <= 0.02 + 0.05 * sat_depth);
        assert!(hit.normal.dot(sat_normal) > 0.7);
    }

    /// Fuzz EPA against the SAT box-box reference: 1000 deterministic
    /// (xorshift64*) random OBB pairs. SAT is exact for boxes, so any
    /// EPA disagreement is an EPA bug. Boolean overlap must agree
    /// outside a 1e-3 grazing margin (GJK distance); on overlap, depth
    /// must agree within 0.02 + 5% and the normal must lie in the same
    /// hemisphere as SAT's (a strict axis match is wrong to require:
    /// near-tie minimum axes are all valid, and either side may pick a
    /// different one).
    #[test]
    fn test_epa_fuzz_obb_vs_sat_reference() {
        use crate::gjk3d::{support_obb, GjkResult};
        use crate::sat3d::sat_obb_obb;
        use std::f32::consts::PI;

        struct Rng(u64);
        impl Rng {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn next_f32(&mut self) -> f32 {
                ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
            }
            fn range(&mut self, lo: f32, hi: f32) -> f32 {
                lo + (hi - lo) * self.next_f32()
            }
            fn unit_vector(&mut self) -> Vec3 {
                let z = self.range(-1.0, 1.0);
                let t = self.range(0.0, 2.0 * PI);
                let r = (1.0 - z * z).sqrt().max(0.0);
                Vec3::new(r * t.cos(), r * t.sin(), z)
            }
            fn rotation(&mut self) -> Quat {
                Quat::from_axis_angle(self.unit_vector(), self.range(0.0, 2.0 * PI))
            }
            fn half_extents(&mut self) -> Vec3 {
                Vec3::new(
                    self.range(0.3, 2.0),
                    self.range(0.3, 2.0),
                    self.range(0.3, 2.0),
                )
            }
        }

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut compared = 0u32;
        let mut separated = 0u32;
        let mut skipped_grazing = 0u32;
        let mut sliver_skips = 0u32;
        for i in 0..1000 {
            let he_a = rng.half_extents();
            let he_b = rng.half_extents();
            let rot_a = rng.rotation();
            let rot_b = rng.rotation();
            let center_a = Vec3::ZERO;
            // Alternate near/far pairs so the fuzz covers overlap and
            // separation well outside the grazing margin.
            let offset = if i % 2 == 0 {
                rng.range(0.0, 2.5)
            } else {
                rng.range(2.5, 5.0)
            };
            let center_b = rng.unit_vector() * offset;

            let support_a = |dir: Vec3| support_obb(center_a, he_a, rot_a, dir);
            let support_b = |dir: Vec3| support_obb(center_b, he_b, rot_b, dir);

            match crate::gjk3d::gjk(&support_a, &support_b) {
                (GjkResult::Separated { distance }, _) if distance > 1e-3 => {
                    assert!(
                        sat_obb_obb(center_a, he_a, rot_a, center_b, he_b, rot_b).is_none(),
                        "SAT reports overlap at distance {distance}"
                    );
                    assert!(
                        gjk_epa(&support_a, &support_b).is_none(),
                        "EPA reports overlap at distance {distance}"
                    );
                    separated += 1;
                }
                (GjkResult::Separated { .. }, _) => {
                    skipped_grazing += 1;
                }
                (GjkResult::Colliding, _) => {
                    let (sat_normal, sat_depth, _) =
                        sat_obb_obb(center_a, he_a, rot_a, center_b, he_b, rot_b)
                            .expect("SAT must collide when GJK does");
                    // Grazing overlaps shallower than the seed margin
                    // legitimately yield None (touching, nothing for the
                    // solver to do); anything deeper must agree with SAT.
                    if sat_depth < 2.0 * super::STRICT_TET_MARGIN {
                        skipped_grazing += 1;
                        continue;
                    }
                    let Some(hit) = gjk_epa(&support_a, &support_b) else {
                        // Needle overlaps (Minkowski inradius below the
                        // seed margin — no margin-healthy tetrahedron
                        // exists) legitimately skip: a skipped frame
                        // self-heals, a guessed depth injects energy.
                        // Box-needle penetration is inherently small, so
                        // cap the skip depth: None on a FAT overlap is a
                        // seed bug, not a sliver.
                        assert!(
                            sat_depth < 0.02,
                            "case {i}: EPA skip on fat overlap (depth {sat_depth})"
                        );
                        sliver_skips += 1;
                        continue;
                    };
                    assert!(
                        hit.normal.dot(sat_normal) > 0.7,
                        "case {i}: normal flip: epa={} sat={}",
                        hit.normal,
                        sat_normal
                    );
                    let tol = 0.02 + 0.05 * sat_depth;
                    assert!(
                        (hit.depth - sat_depth).abs() <= tol,
                        "case {i}: depth mismatch: epa={} sat={}",
                        hit.depth,
                        sat_depth
                    );
                    compared += 1;
                }
            }
        }
        assert!(
            compared > 200 && separated > 200 && sliver_skips <= 20,
            "weak fuzz coverage: compared={compared} separated={separated} grazing={skipped_grazing} slivers={sliver_skips}"
        );
    }
}
