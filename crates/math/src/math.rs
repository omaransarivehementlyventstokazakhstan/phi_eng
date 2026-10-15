use glam::{Mat3, Quat, Vec2, Vec3};

pub const EPS: f32 = 1e-6;
pub const EPS_SQ: f32 = 1e-12;
pub const EPS_LINE: f32 = 1e-9;
pub const EPS_PLANE: f32 = 1e-3;
pub const EPA_TOL: f32 = 1e-4;
pub const GJK_EPS: f32 = 1e-6;
pub const MAX_BIAS_VELOCITY: f32 = 2.0;
pub const DEFAULT_SLOP: f32 = 0.005;
pub const DEFAULT_BAUMGARTE: f32 = 0.2;
pub const DEFAULT_ITERATIONS: u32 = 8;
pub const EARLY_EXIT_TOL: f32 = 1e-6;
pub const LINEAR_SLEEP_THRESHOLD: f32 = 0.05;
pub const ANGULAR_SLEEP_THRESHOLD: f32 = 0.05;
pub const CONTACT_MATCH_DIST_SQ: f32 = 0.04;
pub const RESTITUTION_THRESHOLD: f32 = 1.0;

#[inline]
pub const fn scaled_eps(len: f32) -> f32 {
    EPS * len.max(1.0)
}

/// 2D Transform: position + rotation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2 {
    pub position: Vec2,
    pub rotation: glam::Mat2,
}

impl Transform2 {
    pub fn new(position: Vec2, rotation: glam::Mat2) -> Self {
        Self { position, rotation }
    }

    pub fn from_position_angle(position: Vec2, angle: f32) -> Self {
        Self {
            position,
            rotation: glam::Mat2::from_angle(angle),
        }
    }

    pub fn identity() -> Self {
        Self {
            position: Vec2::ZERO,
            rotation: glam::Mat2::IDENTITY,
        }
    }

    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        self.rotation * point + self.position
    }

    pub fn rotate_vector(&self, vector: Vec2) -> Vec2 {
        self.rotation * vector
    }
}

/// 3D Transform: position + rotation (quaternion)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform3 {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Transform3 {
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    pub fn from_euler(position: Vec3, euler: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::from_euler(glam::EulerRot::XYZ, euler.x, euler.y, euler.z),
        }
    }

    pub fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }

    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * point + self.position
    }

    pub fn rotate_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * vector
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.inverse();
        Self {
            position: inv_rot * (-self.position),
            rotation: inv_rot,
        }
    }
}

/// 3x3 Inertia Tensor for rigid body dynamics
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct InertiaTensor3 {
    pub m: Mat3,
}

impl InertiaTensor3 {
    pub fn new(m: Mat3) -> Self {
        Self { m }
    }

    pub fn from_diagonal(ixx: f32, iyy: f32, izz: f32) -> Self {
        Self {
            m: Mat3::from_diagonal(Vec3::new(ixx, iyy, izz)),
        }
    }

    pub fn sphere(radius: f32, mass: f32) -> Self {
        let i = 0.4 * mass * radius * radius;
        Self::from_diagonal(i, i, i)
    }

    pub fn box_inertia(half_extents: Vec3, mass: f32) -> Self {
        let w = half_extents.x * 2.0;
        let h = half_extents.y * 2.0;
        let d = half_extents.z * 2.0;
        let ixx = mass * (h * h + d * d) / 12.0;
        let iyy = mass * (w * w + d * d) / 12.0;
        let izz = mass * (w * w + h * h) / 12.0;
        Self::from_diagonal(ixx, iyy, izz)
    }

    /// Capsule aligned with the Y axis: cylinder plus two hemispherical
    /// caps. Decomposes the total mass by volume, then combines the
    /// cylinder rod inertia with the caps (solid-sphere inertia plus
    /// parallel-axis terms for their ±half_length offsets).
    pub fn capsule(radius: f32, half_length: f32, mass: f32) -> Self {
        use std::f32::consts::PI;
        let length = half_length * 2.0;
        let v_cyl = PI * radius * radius * length;
        let v_sph = (4.0 / 3.0) * PI * radius * radius * radius;
        let density = mass / (v_cyl + v_sph).max(1e-12);
        let m_cyl = density * v_cyl;
        let m_sph = density * v_sph;
        let transverse = m_cyl * (3.0 * radius * radius + length * length) / 12.0
            + m_sph * (0.4 * radius * radius + half_length * half_length);
        let axial = 0.5 * m_cyl * radius * radius + 0.4 * m_sph * radius * radius;
        Self::from_diagonal(transverse, axial, transverse)
    }

    /// Exact mass properties of a closed, outward-wound triangle mesh.
    ///
    /// Mirtich-style signed tetrahedral decomposition about the mesh
    /// origin: each triangle `(a, b, c)` forms the tetrahedron
    /// `(origin, a, b, c)` with signed volume `dot(a, b x c) / 6`, whose
    /// first and second moments accumulate exactly (in `f64`, then cast).
    /// Returns `(mass, center_of_mass, inertia_about_center_of_mass)`.
    ///
    /// Precondition: `indices.len() % 3 == 0`, every index in bounds, and
    /// the triangles form a closed mesh with positive (outward) volume —
    /// see `validate_shape3` in `phieng_dynamics`. Degenerate
    /// input yields zero mass/inertia rather than NaN; callers must
    /// validate first instead of guessing.
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn convex_mesh(vertices: &[Vec3], indices: &[u32], density: f32) -> (f32, Vec3, Self) {
        use glam::{DMat3, DVec3};

        fn outer(u: DVec3, v: DVec3) -> DMat3 {
            DMat3::from_cols(u * v.x, u * v.y, u * v.z)
        }

        fn dv(v: Vec3) -> DVec3 {
            DVec3::new(v.x as f64, v.y as f64, v.z as f64)
        }

        let mut volume = 0.0f64;
        let mut first = DVec3::ZERO;
        let mut second = DMat3::ZERO;
        for tri in indices.chunks_exact(3) {
            let a = dv(vertices[tri[0] as usize]);
            let b = dv(vertices[tri[1] as usize]);
            let c = dv(vertices[tri[2] as usize]);
            // Signed volume of tetrahedron (origin, a, b, c).
            let w = a.dot(b.cross(c)) / 6.0;
            volume += w;
            // First moment: tet centroid is (a + b + c) / 4.
            let s = a + b + c;
            first += (w / 4.0) * s;
            // Second moment of the tet: V/20 * (s s^T + (a a^T + b b^T + c c^T)).
            let q = outer(a, a) + outer(b, b) + outer(c, c);
            second += (w / 20.0) * (outer(s, s) + q);
        }

        let density64 = density as f64;
        let mass = density64 * volume;
        if !volume.is_finite() || volume <= 0.0 || !mass.is_finite() || mass <= 0.0 {
            return (0.0, Vec3::ZERO, Self::from_diagonal(0.0, 0.0, 0.0));
        }
        let com = first / volume;
        let trace = second.x_axis.x + second.y_axis.y + second.z_axis.z;
        let inertia_origin = (DMat3::from_diagonal(DVec3::splat(trace)) - second) * density64;
        // Parallel-axis shift from the origin to the center of mass.
        let shift = (com.length_squared() * DMat3::IDENTITY - outer(com, com)) * mass;
        let inertia_com = inertia_origin - shift;
        let f32v = |v: DVec3| Vec3::new(v.x as f32, v.y as f32, v.z as f32);
        (
            mass as f32,
            f32v(com),
            Self {
                m: Mat3::from_cols(
                    f32v(inertia_com.x_axis),
                    f32v(inertia_com.y_axis),
                    f32v(inertia_com.z_axis),
                ),
            },
        )
    }

    pub fn transform_by(&self, rotation: Quat) -> Self {
        let r = Mat3::from_quat(rotation);
        let r_t = r.transpose();
        Self {
            m: r * self.m * r_t,
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            m: self.m.inverse(),
        }
    }

    pub fn apply(&self, v: Vec3) -> Vec3 {
        self.m * v
    }
}

/// 2D Rotation helpers
pub mod rot2 {
    use super::*;

    pub fn rotation_matrix(angle: f32) -> glam::Mat2 {
        glam::Mat2::from_angle(angle)
    }

    pub fn apply(matrix: glam::Mat2, vector: Vec2) -> Vec2 {
        matrix * vector
    }
}

/// 3D Rotation helpers
pub mod rot3 {
    use super::*;

    pub fn quat_from_euler(x: f32, y: f32, z: f32) -> Quat {
        Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
    }

    pub fn apply(quat: Quat, vector: Vec3) -> Vec3 {
        quat * vector
    }

    pub fn quat_to_mat3(q: Quat) -> Mat3 {
        Mat3::from_quat(q)
    }

    pub fn mat3_to_quaternion(m: Mat3) -> Quat {
        Quat::from_mat3(&m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform2_identity() {
        let t = Transform2::identity();
        let point = Vec2::new(1.0, 2.0);
        assert_eq!(t.transform_point(point), point);
    }

    #[test]
    fn test_transform3_identity() {
        let t = Transform3::identity();
        let point = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(point), point);
    }

    #[test]
    fn test_transform2_rotation() {
        use std::f32::consts::PI;
        let t = Transform2::from_position_angle(Vec2::ZERO, PI / 2.0);
        let point = Vec2::new(1.0, 0.0);
        let result = t.transform_point(point);
        assert!((result - Vec2::new(0.0, 1.0)).length() < 1e-6);
    }

    #[test]
    fn test_transform3_rotation() {
        let t =
            Transform3::from_euler(Vec3::ZERO, Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2));
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = t.transform_point(point);
        assert!((result - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_inertia_tensor_sphere() {
        let tensor = InertiaTensor3::sphere(1.0, 2.0);
        let expected_i = 0.4 * 2.0 * 1.0;
        assert!((tensor.m.x_axis.x - expected_i).abs() < 1e-6);
        assert!((tensor.m.y_axis.y - expected_i).abs() < 1e-6);
        assert!((tensor.m.z_axis.z - expected_i).abs() < 1e-6);
    }

    #[test]
    fn test_inertia_tensor_box() {
        let tensor = InertiaTensor3::box_inertia(Vec3::new(1.0, 2.0, 3.0), 12.0);
        let w = 2.0;
        let h = 4.0;
        let d = 6.0;
        let expected_ixx = 12.0 * (h * h + d * d) / 12.0;
        let expected_iyy = 12.0 * (w * w + d * d) / 12.0;
        let expected_izz = 12.0 * (w * w + h * h) / 12.0;
        assert!((tensor.m.x_axis.x - expected_ixx).abs() < 1e-6);
        assert!((tensor.m.y_axis.y - expected_iyy).abs() < 1e-6);
        assert!((tensor.m.z_axis.z - expected_izz).abs() < 1e-6);
    }

    #[test]
    fn test_inertia_tensor_transform() {
        let local = InertiaTensor3::from_diagonal(1.0, 2.0, 3.0);
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.0, std::f32::consts::FRAC_PI_2);
        let world = local.transform_by(rot);
        assert!(
            (world.m.x_axis.x - 2.0).abs() < 1e-5,
            "ixx after rotZ90: {}",
            world.m.x_axis.x
        );
        assert!(
            (world.m.y_axis.y - 1.0).abs() < 1e-5,
            "iyy after rotZ90: {}",
            world.m.y_axis.y
        );
        assert!(
            (world.m.z_axis.z - 3.0).abs() < 1e-5,
            "izz after rotZ90: {}",
            world.m.z_axis.z
        );
    }

    #[test]
    fn test_inertia_tensor_capsule() {
        // Degenerate capsule (zero half-length) must equal a solid sphere.
        let cap = InertiaTensor3::capsule(1.0, 0.0, 2.0);
        let sph = InertiaTensor3::sphere(1.0, 2.0);
        assert!((cap.m.x_axis.x - sph.m.x_axis.x).abs() < 1e-5);
        assert!((cap.m.y_axis.y - sph.m.y_axis.y).abs() < 1e-5);
        // Long capsule: transverse inertia dominates axial.
        let long = InertiaTensor3::capsule(0.5, 2.0, 10.0);
        assert!(long.m.x_axis.x > long.m.y_axis.y);
        assert!((long.m.x_axis.x - long.m.z_axis.z).abs() < 1e-5);
    }

    #[test]
    fn test_inertia_tensor_inverse() {
        let tensor = InertiaTensor3::from_diagonal(2.0, 4.0, 6.0);
        let inv = tensor.inverse();
        assert!((inv.m.x_axis.x - 0.5).abs() < 1e-6);
        assert!((inv.m.y_axis.y - 0.25).abs() < 1e-6);
        assert!((inv.m.z_axis.z - 1.0 / 6.0).abs() < 1e-6);
    }

    /// Outward-wound unit box mesh (half extents 1, 2, 3).
    fn box_mesh() -> (Vec<Vec3>, Vec<u32>) {
        let v = vec![
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(1.0, -2.0, -3.0),
            Vec3::new(1.0, 2.0, -3.0),
            Vec3::new(-1.0, 2.0, -3.0),
            Vec3::new(-1.0, -2.0, 3.0),
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 2.0, 3.0),
        ];
        // Each quad wound CCW seen from outside (verified by face-normal sign).
        let idx = vec![
            1, 2, 6, 1, 6, 5, // +x
            0, 4, 7, 0, 7, 3, // -x
            3, 7, 6, 3, 6, 2, // +y
            0, 1, 5, 0, 5, 4, // -y
            4, 5, 6, 4, 6, 7, // +z
            0, 3, 2, 0, 2, 1, // -z
        ];
        (v, idx)
    }

    #[test]
    fn test_convex_mesh_box_matches_analytic() {
        let (verts, idx) = box_mesh();
        let (mass, com, tensor) = InertiaTensor3::convex_mesh(&verts, &idx, 2.0);
        let expected = InertiaTensor3::box_inertia(Vec3::new(1.0, 2.0, 3.0), 96.0);
        assert!((mass - 96.0).abs() < 1e-3, "mass={mass}");
        assert!(com.length() < 1e-5, "com={com}");
        for (got, want) in [
            (tensor.m.x_axis.x, expected.m.x_axis.x),
            (tensor.m.y_axis.y, expected.m.y_axis.y),
            (tensor.m.z_axis.z, expected.m.z_axis.z),
        ] {
            let tol = 1e-4 * want.abs().max(1.0);
            assert!((got - want).abs() < tol, "got={got} want={want}");
        }
        assert!(tensor.m.x_axis.y.abs() < 1e-3);
        assert!(tensor.m.x_axis.z.abs() < 1e-3);
        assert!(tensor.m.y_axis.z.abs() < 1e-3);
    }

    #[test]
    fn test_convex_mesh_tetrahedron_exact() {
        // Trirectangular unit tetrahedron, outward wound.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let idx = vec![1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1];
        let (mass, com, tensor) = InertiaTensor3::convex_mesh(&verts, &idx, 1.0);
        assert!((mass - 1.0 / 6.0).abs() < 1e-6, "mass={mass}");
        assert!(
            (com - Vec3::new(0.25, 0.25, 0.25)).length() < 1e-6,
            "com={com}"
        );
        // Diagonal about the COM is 1/80; off-diagonals are +1/480.
        assert!((tensor.m.x_axis.x - 1.0 / 80.0).abs() < 1e-6);
        assert!((tensor.m.y_axis.y - 1.0 / 80.0).abs() < 1e-6);
        assert!((tensor.m.z_axis.z - 1.0 / 80.0).abs() < 1e-6);
        assert!((tensor.m.x_axis.y - 1.0 / 480.0).abs() < 1e-6);
        assert!((tensor.m.y_axis.x - 1.0 / 480.0).abs() < 1e-6);
    }

    #[test]
    fn test_convex_mesh_translation_and_density() {
        let (mut verts, idx) = box_mesh();
        let offset = Vec3::new(5.0, -3.0, 2.0);
        for v in verts.iter_mut() {
            *v += offset;
        }
        let (mass, com, tensor) = InertiaTensor3::convex_mesh(&verts, &idx, 3.0);
        let expected = InertiaTensor3::box_inertia(Vec3::new(1.0, 2.0, 3.0), 48.0 * 3.0);
        assert!((mass - 144.0).abs() < 1e-3, "mass={mass}");
        assert!((com - offset).length() < 1e-4, "com={com}");
        let tol = 1e-4 * expected.m.x_axis.x.abs();
        assert!((tensor.m.x_axis.x - expected.m.x_axis.x).abs() < tol);
        assert!((tensor.m.y_axis.y - expected.m.y_axis.y).abs() < tol);
        assert!((tensor.m.z_axis.z - expected.m.z_axis.z).abs() < tol);
    }

    #[test]
    fn test_convex_mesh_degenerate_yields_zero() {
        // Flat quad (zero volume): no guessing, just zero mass.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let idx = vec![0, 1, 2, 0, 2, 3];
        let (mass, _, tensor) = InertiaTensor3::convex_mesh(&verts, &idx, 1.0);
        assert_eq!(mass, 0.0);
        assert_eq!(tensor.m, Mat3::ZERO);
    }
}
