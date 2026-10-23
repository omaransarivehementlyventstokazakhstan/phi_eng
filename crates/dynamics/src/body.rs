use super::material::{PhysicsMaterial, MIN_DENSITY};
use glam::{Mat2, Mat3, Vec2, Vec3};
use phieng_math::{InertiaTensor3, Transform2, Transform3};

use super::nuclear::NuclearState;
use std::collections::HashSet;

/// The collision layer a body belongs to unless it is configured
/// otherwise.
///
/// Layer 1 is the default so that a body nobody configured shares a
/// layer with every other unconfigured body, and the default mask
/// accepts every layer: an unconfigured scene collides exactly as it
/// did before layers existed.
pub const DEFAULT_COLLISION_GROUP: u32 = 1;

/// The collision mask a body carries unless it is configured otherwise:
/// every layer is accepted.
pub const DEFAULT_COLLISION_MASK: u32 = u32::MAX;

/// Whether two bodies with the given filters may collide.
///
/// The rule is Box2D's: two bodies on the same group always collide —
/// the group overrides the masks — and otherwise each body must accept
/// the other's group. A body that accepts everything therefore
/// collides with everything, and a body that accepts one group collides
/// only with the bodies on it, unless the two share a group — one side
/// is enough to veto the pair.
///
/// A group is one bit, so the test is a bitwise AND of the two group
/// flags against the other's mask. (Box2D's own groups are categories
/// rather than bits; the rule is the same either way.)
#[inline]
pub const fn collision_filter_allows(group_a: u32, mask_a: u32, group_b: u32, mask_b: u32) -> bool {
    // The same group overrides the masks.
    if group_a == group_b && group_a != 0 {
        return true;
    }
    (group_a & mask_b) != 0 && (group_b & mask_a) != 0
}

/// The layers a scene query considers.
///
/// A query behaves like a body on `group`: a body is reported when it
/// accepts the query's group and the query's mask accepts the body's
/// layer -- the same rule two bodies use, so a query and a body
/// agree about whether they would collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryFilter {
    /// The layer the query pretends to be on.
    pub group: u32,
    /// The layers the query accepts.
    pub mask: u32,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            group: DEFAULT_COLLISION_GROUP,
            mask: DEFAULT_COLLISION_MASK,
        }
    }
}

impl QueryFilter {
    /// A filter that accepts every body, whatever its layer.
    pub const fn everything() -> Self {
        Self {
            group: DEFAULT_COLLISION_GROUP,
            mask: DEFAULT_COLLISION_MASK,
        }
    }

    /// Whether this filter accepts a body on `group` with the mask
    /// `mask`.
    #[inline]
    pub const fn accepts(&self, group: u32, mask: u32) -> bool {
        (group & self.mask) != 0 && (self.group & mask) != 0
    }
}

/// What happened to a contact between two bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventKind {
    /// The bodies started touching this step.
    Began,
    /// The bodies stopped touching this step.
    Ended,
}

/// A contact event: two bodies started or stopped touching.
///
/// The world collects one event per pair per step, in the order the
/// pairs were found, and the caller drains them after the step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent2 {
    /// The first body of the pair.
    pub body_a: u32,
    /// The second body of the pair.
    pub body_b: u32,
    /// Whether the contact began or ended.
    pub kind: ContactEventKind,
    /// The contact point, in world space, at the moment the event fired.
    ///
    /// For an ended contact this is the last point the pair was seen
    /// at; for a contact between two sensors it is the point of the
    /// sensor contact.
    pub point: Vec2,
    /// The contact normal, in world space, at the moment the event fired.
    pub normal: Vec2,
}

/// A contact event in 3D. See [`ContactEvent2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent3 {
    /// The first body of the pair.
    pub body_a: u32,
    /// The second body of the pair.
    pub body_b: u32,
    /// Whether the contact began or ended.
    pub kind: ContactEventKind,
    /// The contact point, in world space, at the moment the event fired.
    pub point: Vec3,
    /// The contact normal, in world space, at the moment the event fired.
    pub normal: Vec3,
}

/// Rigid body type in 2D
///
/// Field order is hot-first for cache locality: the per-step hot loop
/// (gravity → solver → integrate → sleep) touches velocity, angular
/// velocity, transform, and inverse mass/inertia on every body, so those
/// sit in the first cache line. Cold setup data (mass, inertia, id) and
/// heap-backed shape live at the tail. Field order is NOT part of the
/// API contract — always construct with field names, never positionally.
#[derive(Debug, Clone)]
pub struct RigidBody2 {
    /// Linear velocity (hot: gravity, solver, integrate, sleep)
    pub velocity: Vec2,
    /// Angular velocity in radians/sec (hot: solver, integrate, sleep)
    pub angular_velocity: f32,
    /// Transform (position + rotation) (hot: broadphase, narrowphase, solver, integrate)
    pub transform: Transform2,
    /// Inverse mass (0 for static) (hot: solver inner loop)
    pub inv_mass: f32,
    /// Inverse inertia (0 for static) (hot: solver inner loop)
    pub inv_inertia: f32,
    /// Is static (immovable) (hot: gravity/integrate early-out)
    pub is_static: bool,
    /// Is sleeping (hot: sleep update)
    pub is_sleeping: bool,
    /// Linear damping (warm: integrate)
    pub linear_damping: f32,
    /// Angular damping (warm: integrate)
    pub angular_damping: f32,
    /// Gravitational coupling multiplier (warm: gravity kick). 1.0 =
    /// full coupling, 0.0 = immune (still collides and still exerts by
    /// mass), negative = repulsion. Only finiteness is enforced
    /// (see `validate_body_2d`); must be finite.
    pub gravity_scale: f32,
    /// Ambient-force coupling multiplier (warm: ambient kick). 1.0 = full
    /// coupling to spring/drag/turbulence, 0.0 = immune, negative is the
    /// caller's physics. Only finiteness is enforced; must be finite.
    pub ambient_scale: f32,
    /// Current temperature in Kelvin (warm: thermal phase). Default 293.15 K.
    pub temperature: f32,
    /// Thermal-expansion factor currently applied to this body's shape
    /// (warm: thermal phase). 1.0 means "not expanded"; the thermal
    /// phase rescales the shape by `target / expansion_scale`, so the
    /// factor is applied relative to the authored shape instead of
    /// compounding every step.
    pub expansion_scale: f32,
    /// Nuclear state (cold: nuclear phase). The default nuclide is
    /// "none", so a body takes no part in the nuclear phase until it is
    /// given one.
    pub nuclear: NuclearState,
    /// Collision group this body belongs to (warm: narrowphase pair
    /// filter). A pair collides only when each body accepts the other's
    /// group.
    ///
    /// A group is **one bit**, the convention Box2D and Bullet use: a
    /// body belongs to exactly one group, and the mask says which of
    /// the groups it collides with. The default is
    /// [`DEFAULT_COLLISION_GROUP`] (the first bit), and the default
    /// mask accepts every group, so a body that is never configured
    /// collides with everything — the behaviour every scene had before
    /// groups existed.
    pub collision_group: u32,
    /// Collision groups this body accepts (warm: narrowphase pair
    /// filter), as a bit field of the group flags. See
    /// `RigidBody2::collision_mask`.
    pub collision_mask: u32,
    /// Is a sensor: this body reports contacts but never pushes. See
    /// `RigidBody2::is_sensor`.
    pub is_sensor: bool,
    /// Surface and mass properties (friction, restitution, density).
    /// Shapes inherit their body's material. (warm: narrowphase)
    pub material: PhysicsMaterial,
    /// Shape (heap allocation for polygons) (warm: broadphase, narrowphase)
    pub shape: Option<Shape2>,
    /// Mass (cold: setup and validation ratios only; hot loop uses inv_mass)
    pub mass: f32,
    /// Moment of inertia (cold: setup only; hot loop uses inv_inertia)
    pub inertia: f32,
    /// Body ID (cold: indexing invariant only)
    pub id: usize,
}

/// 2D Shape enum for collision detection
#[derive(Debug, Clone)]
pub enum Shape2 {
    Circle { radius: f32 },
    Box { half_extents: Vec2 },
    Polygon { vertices: Vec<Vec2> },
}

impl Shape2 {
    /// Uniformly scaled copy of this shape about its origin.
    ///
    /// Used by thermal expansion: a body hotter than the reference
    /// temperature grows by `1 + alpha * dT`. A non-positive or non-finite
    /// factor returns the shape unchanged (never a degenerate shape).
    pub fn scaled(&self, factor: f32) -> Shape2 {
        if !factor.is_finite() || factor <= 0.0 {
            return self.clone();
        }
        match self {
            Shape2::Circle { radius } => Shape2::Circle {
                radius: radius * factor,
            },
            Shape2::Box { half_extents } => Shape2::Box {
                half_extents: *half_extents * factor,
            },
            Shape2::Polygon { vertices } => Shape2::Polygon {
                vertices: vertices.iter().map(|v| *v * factor).collect(),
            },
        }
    }

    /// Surface area (2D: perimeter) in meters.
    pub fn surface_area(&self) -> f32 {
        match self {
            Shape2::Circle { radius } => 2.0 * std::f32::consts::PI * radius,
            Shape2::Box { half_extents } => 4.0 * (half_extents.x + half_extents.y),
            Shape2::Polygon { vertices } => {
                let mut perimeter = 0.0;
                for i in 0..vertices.len() {
                    let j = (i + 1) % vertices.len();
                    perimeter += (vertices[j] - vertices[i]).length();
                }
                perimeter
            }
        }
    }

    /// Characteristic length scale: sqrt(area) for 2D.
    pub fn characteristic_length(&self) -> f32 {
        match self {
            Shape2::Circle { radius } => *radius,
            Shape2::Box { half_extents } => (4.0 * half_extents.x * half_extents.y).sqrt(),
            Shape2::Polygon { vertices } => {
                let (area, _) = polygon_area_and_centroid(vertices);
                area.abs().sqrt()
            }
        }
    }

    /// Fraction of the surface that faces "up" (towards +y), used to
    /// split the environment radiation between the sky and the ground.
    ///
    /// The body's *rotation* is deliberately ignored: the fraction is a
    /// property of the authored shape, not of its orientation. A circle
    /// and a polygon are treated as isotropic (0.5); a box uses the
    /// fraction of its perimeter that is the top edge.
    pub fn up_facing_fraction(&self) -> f32 {
        match self {
            Shape2::Circle { .. } => 0.5,
            Shape2::Box { half_extents } => {
                let perimeter = self.surface_area();
                if perimeter <= 0.0 {
                    return 0.5;
                }
                (2.0 * half_extents.x / perimeter).clamp(0.0, 1.0)
            }
            Shape2::Polygon { .. } => 0.5,
        }
    }

    /// Bounding radius: max distance from centroid to any vertex.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape2::Circle { radius } => *radius,
            Shape2::Box { half_extents } => half_extents.length(),
            Shape2::Polygon { vertices } => {
                let (_, centroid) = polygon_area_and_centroid(vertices);
                vertices
                    .iter()
                    .map(|v| (*v - centroid).length())
                    .fold(0.0, f32::max)
            }
        }
    }
}

/// Rejection reasons for ambiguous body/shape creation.
///
/// Boundary rule (never-guess policy): a shape or body that cannot be
/// described without ambiguity is rejected at creation instead of being
/// silently clamped — clamping hides user errors and diverges the user's
/// mental model from the simulated state. Solver-side floors (e.g.
/// [`MIN_DENSITY`] in `apply_mass`) remain as numerical safety nets, but
/// they are not user-facing policy. Friction/restitution keep their
/// long-standing sanitize-to-range behavior (universally-agreed ranges
/// with obvious clamping); density and geometry reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    /// NaN or infinite value in shape params, density, or position.
    NonFiniteInput,
    /// Circle/sphere radius or box half-extent component <= 0.
    ZeroSize,
    /// Capsule half-length <= 0 (a zero-length capsule is a sphere:
    /// construct `Sphere` instead of relying on silent equivalence).
    ZeroLengthCapsule,
    /// Polygon with fewer than 3 distinct vertices or exactly zero area.
    /// Only exact degeneracy rejects; near-degenerate thin polygons still
    /// simulate (2D SAT is robust to them).
    DegeneratePolygon,
    /// Convex hull with fewer than 4 distinct vertices.
    TooFewHullVertices,
    /// Convex mesh with fewer than 4 distinct vertices or fewer than 4
    /// triangles (a tetrahedron is the smallest closed triangle mesh).
    TooFewMeshVertices,
    /// Convex mesh triangle index out of bounds.
    InvalidMeshIndex,
    /// Convex mesh that is not a closed, consistently-wound manifold with
    /// positive enclosed volume (degenerate triangles, boundary edges,
    /// inconsistent winding, flat or inward-wound soup).
    DegenerateMesh,
    /// Closed convex mesh with a vertex strictly outside some face plane.
    NonConvexMesh,
    /// Non-static body with non-finite or non-positive density. (Static
    /// bodies ignore mass, so any finite density is accepted for them.)
    ZeroDensity,
    /// A dynamic body was given a `ConvexHull`. A hull carries vertices
    /// only, so its volume and inertia cannot be computed exactly
    /// without rebuilding the faces; a dynamic body needs those numbers
    /// to respond to torque. Use `ConvexMesh` (closed triangles) for a
    /// dynamic body, or make the hull static.
    DynamicConvexHull,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::NonFiniteInput => {
                write!(f, "non-finite (NaN/inf) shape, density, or position input")
            }
            ShapeError::ZeroSize => write!(f, "radius or half-extent must be positive"),
            ShapeError::ZeroLengthCapsule => write!(f, "zero-length capsule: use Sphere instead"),
            ShapeError::DegeneratePolygon => {
                write!(f, "polygon needs 3+ distinct vertices and nonzero area")
            }
            ShapeError::TooFewHullVertices => write!(f, "convex hull needs 4+ distinct vertices"),
            ShapeError::TooFewMeshVertices => {
                write!(f, "convex mesh needs 4+ distinct vertices and 4+ triangles")
            }
            ShapeError::InvalidMeshIndex => write!(f, "mesh triangle index out of bounds"),
            ShapeError::DegenerateMesh => {
                write!(
                    f,
                    "mesh must be closed, consistently wound, and enclose positive volume"
                )
            }
            ShapeError::NonConvexMesh => {
                write!(f, "convex mesh has a vertex outside a face plane")
            }
            ShapeError::ZeroDensity => write!(f, "dynamic body needs finite positive density"),
            ShapeError::DynamicConvexHull => write!(
                f,
                "dynamic body cannot use ConvexHull: use ConvexMesh (exact mass/inertia) or make it static"
            ),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Exact-equality distinct count (1-ulp differences still count as
/// distinct — only bitwise duplicates collapse).
fn distinct_count_2d(vertices: &[Vec2]) -> usize {
    let mut distinct = 0;
    for (i, v) in vertices.iter().enumerate() {
        if vertices[..i].iter().all(|u| *u != *v) {
            distinct += 1;
        }
    }
    distinct
}

fn distinct_count_3d(vertices: &[Vec3]) -> usize {
    let mut distinct = 0;
    for (i, v) in vertices.iter().enumerate() {
        if vertices[..i].iter().all(|u| *u != *v) {
            distinct += 1;
        }
    }
    distinct
}

/// Validate a 2D shape without computing anything.
pub fn validate_shape2(shape: &Shape2) -> Result<(), ShapeError> {
    match shape {
        Shape2::Circle { radius } => {
            if !radius.is_finite() {
                Err(ShapeError::NonFiniteInput)
            } else if *radius <= 0.0 {
                Err(ShapeError::ZeroSize)
            } else {
                Ok(())
            }
        }
        Shape2::Box { half_extents } => {
            if !half_extents.x.is_finite() || !half_extents.y.is_finite() {
                Err(ShapeError::NonFiniteInput)
            } else if half_extents.x <= 0.0 || half_extents.y <= 0.0 {
                Err(ShapeError::ZeroSize)
            } else {
                Ok(())
            }
        }
        Shape2::Polygon { vertices } => {
            if vertices
                .iter()
                .any(|v| !v.x.is_finite() || !v.y.is_finite())
            {
                return Err(ShapeError::NonFiniteInput);
            }
            // Length first: area/centroid helpers divide by length.
            if vertices.len() < 3 || distinct_count_2d(vertices) < 3 {
                return Err(ShapeError::DegeneratePolygon);
            }
            let (area, _) = polygon_area_and_centroid(vertices);
            if area == 0.0 {
                return Err(ShapeError::DegeneratePolygon);
            }
            Ok(())
        }
    }
}

/// Validate a 3D shape without computing anything.
// MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn validate_shape3(shape: &Shape3) -> Result<(), ShapeError> {
    match shape {
        Shape3::Sphere { radius } => {
            if !radius.is_finite() {
                Err(ShapeError::NonFiniteInput)
            } else if *radius <= 0.0 {
                Err(ShapeError::ZeroSize)
            } else {
                Ok(())
            }
        }
        Shape3::Box { half_extents } => {
            if !half_extents.x.is_finite()
                || !half_extents.y.is_finite()
                || !half_extents.z.is_finite()
            {
                Err(ShapeError::NonFiniteInput)
            } else if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
                Err(ShapeError::ZeroSize)
            } else {
                Ok(())
            }
        }
        Shape3::Capsule {
            radius,
            half_length,
        } => {
            if !radius.is_finite() || !half_length.is_finite() {
                Err(ShapeError::NonFiniteInput)
            } else if *radius <= 0.0 {
                Err(ShapeError::ZeroSize)
            } else if *half_length <= 0.0 {
                Err(ShapeError::ZeroLengthCapsule)
            } else {
                Ok(())
            }
        }
        Shape3::ConvexHull { vertices } => {
            if vertices
                .iter()
                .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
            {
                return Err(ShapeError::NonFiniteInput);
            }
            // Mass is a stub (no volume integration), so coplanarity is
            // left to the graceful skip policy; only count is enforced.
            if distinct_count_3d(vertices) < 4 {
                return Err(ShapeError::TooFewHullVertices);
            }
            Ok(())
        }
        Shape3::ConvexMesh { vertices, indices } => {
            if vertices
                .iter()
                .any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite())
            {
                return Err(ShapeError::NonFiniteInput);
            }
            if indices.len() % 3 != 0 {
                return Err(ShapeError::DegenerateMesh);
            }
            // A tetrahedron (4 vertices, 4 triangles) is the smallest
            // closed triangle mesh.
            if distinct_count_3d(vertices) < 4 || indices.len() < 12 {
                return Err(ShapeError::TooFewMeshVertices);
            }
            for tri in indices.chunks_exact(3) {
                let (a, b, c) = (tri[0], tri[1], tri[2]);
                if a as usize >= vertices.len()
                    || b as usize >= vertices.len()
                    || c as usize >= vertices.len()
                {
                    return Err(ShapeError::InvalidMeshIndex);
                }
                if a == b || b == c || a == c {
                    return Err(ShapeError::DegenerateMesh);
                }
                let va = vertices[a as usize];
                let vb = vertices[b as usize];
                let vc = vertices[c as usize];
                if (vb - va).cross(vc - va) == Vec3::ZERO {
                    return Err(ShapeError::DegenerateMesh);
                }
            }
            if !convex_mesh_is_closed(vertices.len(), indices) {
                return Err(ShapeError::DegenerateMesh);
            }
            // Inward winding encloses negative volume; flat soup encloses
            // none. Both are rejected instead of guessed at.
            if convex_mesh_signed_volume(vertices, indices) <= 0.0 {
                return Err(ShapeError::DegenerateMesh);
            }
            let tol = 1e-4 * convex_mesh_extent(vertices).max(1e-6);
            if !convex_mesh_is_convex(vertices, indices, tol) {
                return Err(ShapeError::NonConvexMesh);
            }
            Ok(())
        }
    }
}

/// Density rule: always finite; positive unless the body is static
/// (statics ignore mass, so any finite density is accepted for them).
pub fn validate_density(is_static: bool, density: f32) -> Result<f32, ShapeError> {
    if !density.is_finite() {
        return Err(ShapeError::NonFiniteInput);
    }
    if !is_static && density <= 0.0 {
        return Err(ShapeError::ZeroDensity);
    }
    Ok(density)
}

impl RigidBody2 {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            transform: Transform2::identity(),
            velocity: Vec2::ZERO,
            angular_velocity: 0.0,
            mass: 1.0,
            inv_mass: 1.0,
            inertia: 0.5,
            inv_inertia: 2.0,
            linear_damping: 0.01,
            angular_damping: 0.01,
            gravity_scale: 1.0,
            ambient_scale: 1.0,
            temperature: phieng_math::DEFAULT_TEMPERATURE,
            expansion_scale: 1.0,
            nuclear: NuclearState::default(),
            collision_group: DEFAULT_COLLISION_GROUP,
            collision_mask: DEFAULT_COLLISION_MASK,
            is_sensor: false,
            material: PhysicsMaterial::default(),
            is_static: false,
            is_sleeping: false,
            shape: None,
        }
    }

    pub fn static_body(id: usize) -> Self {
        let mut body = Self::new(id);
        body.is_static = true;
        body.inv_mass = 0.0;
        body.inv_inertia = 0.0;
        body
    }

    /// Set shape and recalculate mass properties from the material density.
    /// Also stores `density` into the material. Rejects ambiguous input
    /// (see [`ShapeError`]) instead of silently substituting defaults.
    pub fn set_shape(&mut self, shape: Shape2, density: f32) -> Result<(), ShapeError> {
        validate_shape2(&shape)?;
        let density = validate_density(self.is_static, density)?;
        self.material.density = density;
        let (mass, inertia) = mass_properties_2d(&shape, density);

        self.shape = Some(shape);
        self.apply_mass(mass, inertia);
        Ok(())
    }

    /// Assign a full material, recomputing mass/inertia from the current
    /// shape (if any) with the material's density. Friction/restitution
    /// keep sanitize-to-range behavior; density follows the rejection
    /// rule (see [`ShapeError`]).
    pub fn set_material(&mut self, material: PhysicsMaterial) -> Result<(), ShapeError> {
        let mut sanitized = material;
        if !sanitized.friction.is_finite() || sanitized.friction < 0.0 {
            sanitized.friction = PhysicsMaterial::DEFAULT_FRICTION;
        }
        sanitized.restitution = if sanitized.restitution.is_finite() {
            sanitized.restitution.clamp(0.0, 1.0)
        } else {
            PhysicsMaterial::DEFAULT_RESTITUTION
        };
        validate_density(self.is_static, sanitized.density)?;
        self.material = sanitized;
        if let Some(shape) = self.shape.clone() {
            let (mass, inertia) = mass_properties_2d(&shape, self.material.density);
            self.apply_mass(mass, inertia);
        }
        Ok(())
    }

    /// Store mass properties, honoring the static flag for inverses.
    fn apply_mass(&mut self, mass: f32, inertia: f32) {
        let mass = mass.max(MIN_DENSITY);
        let inertia = inertia.max(MIN_DENSITY);
        self.mass = mass;
        self.inv_mass = if self.is_static { 0.0 } else { 1.0 / mass };
        self.inertia = inertia;
        self.inv_inertia = if self.is_static { 0.0 } else { 1.0 / inertia };
    }

    /// Apply a force at a point for `dt` seconds (impulse `F * dt`).
    pub fn apply_force(&mut self, force: Vec2, point: Vec2, dt: f32) {
        if self.is_static {
            return;
        }
        // An externally applied force is a wake event (see S01): frozen
        // bodies ignore the force itself only when static.
        self.is_sleeping = false;

        // Linear force
        self.velocity += force * self.inv_mass * dt;

        // Torque
        let r = point - self.transform.position;
        let torque = r.x * force.y - r.y * force.x;
        self.angular_velocity += torque * self.inv_inertia * dt;
    }

    /// Apply impulse at a point
    pub fn apply_impulse(&mut self, impulse: Vec2, point: Vec2) {
        if self.is_static {
            return;
        }
        // An externally applied impulse is a wake event (see S01).
        self.is_sleeping = false;

        self.velocity += impulse * self.inv_mass;

        let r = point - self.transform.position;
        let torque = r.x * impulse.y - r.y * impulse.x;
        self.angular_velocity += torque * self.inv_inertia;
    }

    /// Rotation update shared by [`Self::integrate`] and
    /// [`Self::integrate_drift`]: advances the 2D angle by
    /// `angular_velocity * dt`. Single source of truth for the
    /// `atan2(-y_axis.x, x_axis.x)` convention (the negation matters:
    /// plain `atan2(y_axis.x, x_axis.x)` returns `-angle` and makes
    /// bodies ping-pong every frame).
    ///
    /// The trigonometry goes through [`phieng_math::trig`], not the
    /// platform libm: `atan2` and `sin`/`cos` disagree between glibc
    /// and the Microsoft CRT in the last ULP, and that single ULP is
    /// enough to send a 240-step trajectory in two different
    /// directions. See the module docs of `phieng_math::trig`.
    fn integrate_rotation(&mut self, dt: f32) {
        // Mat2::from_angle(a) has x_axis=(cos a, sin a), y_axis=(-sin a, cos a),
        // so the angle is atan2(-y_axis.x, x_axis.x).
        let m00 = self.transform.rotation.x_axis.x;
        let m10 = self.transform.rotation.y_axis.x;
        let current_angle = phieng_math::trig::atan2(-m10, m00);
        let new_angle = current_angle + self.angular_velocity * dt;
        // Same layout as `Mat2::from_angle`, built from the
        // deterministic sine and cosine.
        let (cos, sin) = phieng_math::trig::compute_cos_sin(new_angle);
        self.transform.rotation = Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
    }

    /// Integrate motion (semi-implicit Euler)
    pub fn integrate(&mut self, dt: f32) {
        if self.is_static {
            return;
        }

        // Apply damping
        let damping = self.linear_damping * dt;
        self.velocity *= 1.0 - damping.min(1.0);

        let ang_damping = self.angular_damping * dt;
        self.angular_velocity *= 1.0 - ang_damping.min(1.0);

        // Update position
        self.transform.position += self.velocity * dt;

        self.integrate_rotation(dt);
    }

    /// Drift step for velocity-Verlet integration (see
    /// [`crate::gravity`]): positions and rotation advance a full step
    /// with the post-kick velocities. Linear damping is NOT applied
    /// here — in Verlet mode it rides the gravity kicks as symmetric
    /// drag. Angular damping matches [`Self::integrate`] (multiplicative,
    /// applied before the rotation update).
    pub fn integrate_drift(&mut self, dt: f32) {
        if self.is_static {
            return;
        }

        let ang_damping = self.angular_damping * dt;
        self.angular_velocity *= 1.0 - ang_damping.min(1.0);

        // Update position
        self.transform.position += self.velocity * dt;

        self.integrate_rotation(dt);
    }

    /// Get the transform's rotation angle
    pub fn rotation_angle(&self) -> f32 {
        let m00 = self.transform.rotation.x_axis.x;
        let m10 = self.transform.rotation.y_axis.x;
        phieng_math::trig::atan2(-m10, m00)
    }

    /// Compute kinetic energy: 0.5 * m * v^2 + 0.5 * I * ω^2
    pub fn kinetic_energy(&self) -> f32 {
        if self.is_static {
            return 0.0;
        }
        0.5 * self.mass * self.velocity.length_squared()
            + 0.5 * self.inertia * self.angular_velocity * self.angular_velocity
    }

    /// Check if body should sleep
    pub fn should_sleep(&self, linear_threshold: f32, angular_threshold: f32) -> bool {
        if self.is_static {
            return true;
        }

        let linear_speed = self.velocity.length();
        let angular_speed = self.angular_velocity.abs();

        linear_speed < linear_threshold && angular_speed < angular_threshold
    }
}

/// Mass and inertia for a 2D shape at the given density.
fn mass_properties_2d(shape: &Shape2, density: f32) -> (f32, f32) {
    match shape {
        Shape2::Circle { radius } => {
            let area = std::f32::consts::PI * radius * radius;
            let mass = area * density;
            let inertia = 0.5 * mass * radius * radius;
            (mass, inertia)
        }
        Shape2::Box { half_extents } => {
            let width = half_extents.x * 2.0;
            let height = half_extents.y * 2.0;
            let area = width * height;
            let mass = area * density;
            let inertia = mass * (width * width + height * height) / 12.0;
            (mass, inertia)
        }
        Shape2::Polygon { vertices } => {
            // Signed area depends on winding (negative for clockwise),
            // but mass needs the absolute area.
            let (area, _centroid) = polygon_area_and_centroid(vertices);
            let mass = area.abs() * density;
            let inertia = polygon_inertia(vertices, mass);
            (mass, inertia)
        }
    }
}

/// Mass and inertia tensor for a 3D shape at the given density.
fn mass_properties_3d(shape: &Shape3, density: f32) -> (f32, Mat3) {
    match shape {
        Shape3::Sphere { radius } => {
            let volume = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
            let mass = volume * density;
            let i = 0.4 * mass * radius * radius;
            (mass, Mat3::from_diagonal(glam::Vec3::new(i, i, i)))
        }
        Shape3::Box { half_extents } => {
            let w = half_extents.x * 2.0;
            let h = half_extents.y * 2.0;
            let d = half_extents.z * 2.0;
            let volume = w * h * d;
            let mass = volume * density;
            let ixx = mass * (h * h + d * d) / 12.0;
            let iyy = mass * (w * w + d * d) / 12.0;
            let izz = mass * (w * w + h * h) / 12.0;
            (mass, Mat3::from_diagonal(glam::Vec3::new(ixx, iyy, izz)))
        }
        Shape3::Capsule {
            radius,
            half_length,
        } => {
            use std::f32::consts::PI;
            let length = half_length * 2.0;
            let volume =
                PI * radius * radius * length + (4.0 / 3.0) * PI * radius * radius * radius;
            let mass = volume * density;
            let tensor = phieng_math::InertiaTensor3::capsule(*radius, *half_length, mass);
            (mass, tensor.m)
        }
        Shape3::ConvexHull { vertices: _ } => {
            // Approximate - real implementation would calculate from hull
            let mass = 1.0;
            (mass, Mat3::from_diagonal(glam::Vec3::new(1.0, 1.0, 1.0)))
        }
        Shape3::ConvexMesh { vertices, indices } => {
            // Exact closed-mesh mass properties (Mirtich-style). The mesh
            // is center-of-mass-centered by `set_shape` before this runs,
            // so the returned tensor is already about the body origin.
            let (mass, _, inertia) = InertiaTensor3::convex_mesh(vertices, indices, density);
            (mass, inertia.m)
        }
    }
}

/// Calculate area and centroid of a polygon
fn polygon_area_and_centroid(vertices: &[Vec2]) -> (f32, Vec2) {
    let mut area = 0.0;
    let mut centroid = Vec2::ZERO;

    for i in 0..vertices.len() {
        let j = (i + 1) % vertices.len();
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        area += cross;
        centroid += (vertices[i] + vertices[j]) * cross;
    }

    area *= 0.5;
    centroid /= area * 6.0;

    (area, centroid)
}

/// Calculate moment of inertia of a polygon
fn polygon_inertia(vertices: &[Vec2], mass: f32) -> f32 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..vertices.len() {
        let j = (i + 1) % vertices.len();
        let a = vertices[i];
        let b = vertices[j];

        let cross = a.x * b.y - b.x * a.y;
        let dot_a = a.x * a.x + a.y * a.y;
        let dot_b = b.x * b.x + b.y * b.y;

        numerator += (dot_a + dot_b + a.x * b.x + a.y * b.y) * cross;
        denominator += cross;
    }

    if denominator.abs() < 1e-6 {
        return 0.0;
    }

    mass * numerator / (6.0 * denominator)
}

/// Rigid body type in 3D
#[derive(Debug, Clone)]
pub struct RigidBody3 {
    /// Body ID
    pub id: usize,
    /// Transform (position + rotation)
    pub transform: Transform3,
    /// Linear velocity
    pub velocity: Vec3,
    /// Angular velocity
    pub angular_velocity: Vec3,
    /// Mass
    pub mass: f32,
    /// Inverse mass (0 for static)
    pub inv_mass: f32,
    /// Moment of inertia tensor
    pub inertia: Mat3,
    /// Inverse inertia tensor
    pub inv_inertia: Mat3,
    /// Linear damping
    pub linear_damping: f32,
    /// Angular damping
    pub angular_damping: f32,
    /// Gravitational coupling multiplier (see `RigidBody2::gravity_scale`).
    pub gravity_scale: f32,
    /// Ambient-force coupling multiplier (see `RigidBody2::ambient_scale`).
    pub ambient_scale: f32,
    /// Current temperature in Kelvin (see `RigidBody2::temperature`).
    pub temperature: f32,
    /// Thermal-expansion factor currently applied to this body's shape
    /// (see `RigidBody2::expansion_scale`).
    pub expansion_scale: f32,
    /// Nuclear state (cold: nuclear phase). See `RigidBody2::nuclear`.
    pub nuclear: NuclearState,
    /// Collision layer this body belongs to (warm: narrowphase pair
    /// filter). See `RigidBody2::collision_group`.
    pub collision_group: u32,
    /// Collision groups this body accepts (warm: narrowphase pair
    /// filter), as a bit field of the group flags. See
    /// `RigidBody2::collision_mask`.
    pub collision_mask: u32,
    /// Is a sensor: this body reports contacts but never pushes. See
    /// `RigidBody2::is_sensor`.
    pub is_sensor: bool,
    /// Surface and mass properties (friction, restitution, density).
    /// Shapes inherit their body's material.
    pub material: PhysicsMaterial,
    /// Is static (immovable)
    pub is_static: bool,
    /// Is sleeping
    pub is_sleeping: bool,
    /// Shape (boxed for heap allocation)
    pub shape: Option<Shape3>,
}

/// 3D Shape enum for collision detection
#[derive(Debug, Clone)]
pub enum Shape3 {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Capsule {
        radius: f32,
        half_length: f32,
    },
    ConvexHull {
        vertices: Vec<Vec3>,
    },
    /// Closed, outward-wound convex triangle mesh. `indices` holds flat
    /// triplets into `vertices`. Mass properties are exact
    /// (Mirtich-style); collision uses the convex vertex set, so only
    /// convex meshes are accepted (see [`validate_shape3`]).
    ConvexMesh {
        vertices: Vec<Vec3>,
        indices: Vec<u32>,
    },
}

impl Shape3 {
    /// Uniformly scaled copy of this shape about its origin.
    ///
    /// See [`Shape2::scaled`] for the contract.
    pub fn scaled(&self, factor: f32) -> Shape3 {
        if !factor.is_finite() || factor <= 0.0 {
            return self.clone();
        }
        match self {
            Shape3::Sphere { radius } => Shape3::Sphere {
                radius: radius * factor,
            },
            Shape3::Box { half_extents } => Shape3::Box {
                half_extents: *half_extents * factor,
            },
            Shape3::Capsule {
                radius,
                half_length,
            } => Shape3::Capsule {
                radius: radius * factor,
                half_length: half_length * factor,
            },
            Shape3::ConvexHull { vertices } => Shape3::ConvexHull {
                vertices: vertices.iter().map(|v| *v * factor).collect(),
            },
            Shape3::ConvexMesh { vertices, indices } => Shape3::ConvexMesh {
                vertices: vertices.iter().map(|v| *v * factor).collect(),
                indices: indices.clone(),
            },
        }
    }

    /// Surface area in m².
    pub fn surface_area(&self) -> f32 {
        match self {
            Shape3::Sphere { radius } => 4.0 * std::f32::consts::PI * radius * radius,
            Shape3::Box { half_extents } => {
                let w = 2.0 * half_extents.x;
                let h = 2.0 * half_extents.y;
                let d = 2.0 * half_extents.z;
                2.0 * (w * h + w * d + h * d)
            }
            Shape3::Capsule {
                radius,
                half_length,
            } => {
                4.0 * std::f32::consts::PI * radius * radius
                    + 2.0 * std::f32::consts::PI * radius * 2.0 * half_length
            }
            Shape3::ConvexHull { vertices } => {
                let mut area = 0.0;
                for i in 0..vertices.len() {
                    for j in (i + 1)..vertices.len() {
                        for k in (j + 1)..vertices.len() {
                            let a = vertices[i];
                            let b = vertices[j];
                            let c = vertices[k];
                            area += 0.5 * (b - a).cross(c - a).length();
                        }
                    }
                }
                area
            }
            Shape3::ConvexMesh { vertices, indices } => {
                let mut area = 0.0;
                // MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
                #[allow(clippy::chunks_exact_to_as_chunks)]
                for tri in indices.chunks_exact(3) {
                    let a = vertices[tri[0] as usize];
                    let b = vertices[tri[1] as usize];
                    let c = vertices[tri[2] as usize];
                    area += 0.5 * (b - a).cross(c - a).length();
                }
                area
            }
        }
    }

    /// Fraction of the surface that faces "up" (towards +y), used to
    /// split the environment radiation between the sky and the ground.
    ///
    /// The body's *rotation* is deliberately ignored: the fraction is a
    /// property of the authored shape, not of its orientation. A sphere
    /// and a capsule are isotropic (0.5); a box uses the fraction of
    /// its surface that is the top face; hulls and meshes are treated as
    /// isotropic.
    pub fn up_facing_fraction(&self) -> f32 {
        match self {
            Shape3::Sphere { .. } => 0.5,
            Shape3::Box { half_extents } => {
                let total = self.surface_area();
                if total <= 0.0 {
                    return 0.5;
                }
                let top = 4.0 * half_extents.x * half_extents.z;
                (top / total).clamp(0.0, 1.0)
            }
            Shape3::Capsule { .. } => 0.5,
            Shape3::ConvexHull { .. } => 0.5,
            Shape3::ConvexMesh { .. } => 0.5,
        }
    }

    /// Characteristic length: cube_root(volume).
    pub fn characteristic_length(&self) -> f32 {
        match self {
            Shape3::Sphere { radius } => *radius,
            Shape3::Box { half_extents } => {
                let vol = 8.0 * half_extents.x * half_extents.y * half_extents.z;
                vol.cbrt()
            }
            Shape3::Capsule {
                radius,
                half_length,
            } => {
                let vol = std::f32::consts::PI * radius * radius * 2.0 * half_length
                    + (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
                vol.cbrt()
            }
            Shape3::ConvexHull { vertices } => {
                let mut vol = 0.0f64;
                for i in 0..vertices.len() {
                    for j in (i + 1)..vertices.len() {
                        for k in (j + 1)..vertices.len() {
                            let a = vertices[i];
                            let b = vertices[j];
                            let c = vertices[k];
                            vol += (a.dot(b.cross(c)) / 6.0) as f64;
                        }
                    }
                }
                (vol.abs() as f32).cbrt()
            }
            Shape3::ConvexMesh { vertices, indices } => {
                let vol = convex_mesh_signed_volume(vertices, indices).abs() as f32;
                vol.cbrt()
            }
        }
    }

    /// Bounding radius: max distance from centroid to any vertex.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape3::Sphere { radius } => *radius,
            Shape3::Box { half_extents } => half_extents.length(),
            Shape3::Capsule {
                radius,
                half_length,
            } => *half_length + *radius,
            Shape3::ConvexHull { vertices } => {
                let centroid =
                    vertices.iter().fold(Vec3::ZERO, |acc, v| acc + *v) / vertices.len() as f32;
                vertices
                    .iter()
                    .map(|v| (*v - centroid).length())
                    .fold(0.0, f32::max)
            }
            Shape3::ConvexMesh { vertices, .. } => {
                let centroid =
                    vertices.iter().fold(Vec3::ZERO, |acc, v| acc + *v) / vertices.len() as f32;
                vertices
                    .iter()
                    .map(|v| (*v - centroid).length())
                    .fold(0.0, f32::max)
            }
        }
    }
}

/// Signed enclosed volume of a triangle mesh (positive for outward
/// winding), via tetrahedral decomposition about the mesh origin.
// MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn convex_mesh_signed_volume(vertices: &[Vec3], indices: &[u32]) -> f64 {
    let mut volume = 0.0f64;
    for tri in indices.chunks_exact(3) {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];
        let ax = a.x as f64;
        let ay = a.y as f64;
        let az = a.z as f64;
        let bx = b.x as f64;
        let by = b.y as f64;
        let bz = b.z as f64;
        let cx = c.x as f64;
        let cy = c.y as f64;
        let cz = c.z as f64;
        // dot(a, cross(b, c)) / 6.
        volume +=
            (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6.0;
    }
    volume
}

/// Closed-manifold check: every directed edge appears exactly once, and
/// its reverse exactly once (no boundary edges, no duplicate faces).
// MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn convex_mesh_is_closed(vertex_count: usize, indices: &[u32]) -> bool {
    let _ = vertex_count;
    let mut edges = HashSet::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        for &(u, v) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            if u == v || !edges.insert((u, v)) {
                return false;
            }
        }
    }
    edges.iter().all(|&(u, v)| edges.contains(&(v, u)))
}

/// Convexity check: every vertex must lie behind (or on) every face
/// plane, within a scale-aware tolerance. Callers guarantee non-degenerate
/// triangles and consistent winding.
// MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn convex_mesh_is_convex(vertices: &[Vec3], indices: &[u32], tol: f32) -> bool {
    for tri in indices.chunks_exact(3) {
        let a = vertices[tri[0] as usize];
        let n = (vertices[tri[1] as usize] - a).cross(vertices[tri[2] as usize] - a);
        let normal = n.normalize_or_zero();
        for v in vertices {
            if normal.dot(*v - a) > tol {
                return false;
            }
        }
    }
    true
}

/// Local-space bounding-box diagonal, for scale-aware mesh tolerances.
fn convex_mesh_extent(vertices: &[Vec3]) -> f32 {
    let mut min = vertices[0];
    let mut max = vertices[0];
    for v in &vertices[1..] {
        min = min.min(*v);
        max = max.max(*v);
    }
    (max - min).length()
}

impl RigidBody3 {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            transform: Transform3::identity(),
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            inv_mass: 1.0,
            inertia: Mat3::IDENTITY,
            inv_inertia: Mat3::IDENTITY,
            linear_damping: 0.01,
            angular_damping: 0.01,
            gravity_scale: 1.0,
            ambient_scale: 1.0,
            temperature: phieng_math::DEFAULT_TEMPERATURE,
            expansion_scale: 1.0,
            nuclear: NuclearState::default(),
            collision_group: DEFAULT_COLLISION_GROUP,
            collision_mask: DEFAULT_COLLISION_MASK,
            is_sensor: false,
            material: PhysicsMaterial::default(),
            is_static: false,
            is_sleeping: false,
            shape: None,
        }
    }

    pub fn static_body(id: usize) -> Self {
        let mut body = Self::new(id);
        body.is_static = true;
        body.inv_mass = 0.0;
        body.inv_inertia = Mat3::ZERO;
        body
    }

    /// Set shape and recalculate mass properties from the material density.
    /// Also stores `density` into the material. Rejects ambiguous input
    /// (see [`ShapeError`]) instead of silently substituting defaults.
    pub fn set_shape(&mut self, mut shape: Shape3, density: f32) -> Result<(), ShapeError> {
        validate_shape3(&shape)?;
        // A hull stores vertices but no faces, so its volume and inertia
        // cannot be computed exactly; a dynamic body would run on a stub.
        // Reject it rather than simulate a 1 kg ghost (see ShapeError).
        if !self.is_static && matches!(shape, Shape3::ConvexHull { .. }) {
            return Err(ShapeError::DynamicConvexHull);
        }
        // Recenter convex meshes so the body-frame origin coincides with
        // the center of mass (required for correct rotational dynamics).
        // Topology checks above are translation-invariant, so this is a
        // deterministic normalization, not a guess.
        if let Shape3::ConvexMesh { vertices, indices } = &mut shape {
            let (_, com, _) = InertiaTensor3::convex_mesh(vertices, indices, 1.0);
            if com != Vec3::ZERO {
                for v in vertices.iter_mut() {
                    *v -= com;
                }
            }
        }
        let density = validate_density(self.is_static, density)?;
        self.material.density = density;
        let (mass, inertia) = mass_properties_3d(&shape, density);

        self.shape = Some(shape);
        self.mass = mass.max(MIN_DENSITY);
        self.inv_mass = if self.is_static { 0.0 } else { 1.0 / self.mass };
        self.inertia = inertia;
        self.inv_inertia = if self.is_static {
            Mat3::ZERO
        } else {
            inertia.inverse()
        };
        Ok(())
    }

    /// Assign a full material, recomputing mass/inertia from the current
    /// shape (if any) with the material's density. Friction/restitution
    /// keep sanitize-to-range behavior; density follows the rejection
    /// rule (see [`ShapeError`]).
    pub fn set_material(&mut self, material: PhysicsMaterial) -> Result<(), ShapeError> {
        let mut sanitized = material;
        if !sanitized.friction.is_finite() || sanitized.friction < 0.0 {
            sanitized.friction = PhysicsMaterial::DEFAULT_FRICTION;
        }
        sanitized.restitution = if sanitized.restitution.is_finite() {
            sanitized.restitution.clamp(0.0, 1.0)
        } else {
            PhysicsMaterial::DEFAULT_RESTITUTION
        };
        validate_density(self.is_static, sanitized.density)?;
        self.material = sanitized;
        if let Some(shape) = self.shape.clone() {
            // Shape and density were both just validated.
            self.set_shape(shape, self.material.density)?;
        }
        Ok(())
    }

    /// Inverse inertia in world space: `R * I_local^-1 * R^T`.
    /// Torques from [`Self::apply_impulse`] and the solver live in world
    /// space, so they must use this — never the body-frame tensor.
    pub fn world_inv_inertia(&self) -> Mat3 {
        let r = Mat3::from_quat(self.transform.rotation);
        r * self.inv_inertia * r.transpose()
    }

    /// Apply a force at a point for `dt` seconds (impulse `F * dt`).
    pub fn apply_force(&mut self, force: Vec3, point: Vec3, dt: f32) {
        if self.is_static {
            return;
        }
        // An externally applied force is a wake event (see S01).
        self.is_sleeping = false;

        self.velocity += force * self.inv_mass * dt;

        let r = point - self.transform.position;
        let torque = r.cross(force);
        self.angular_velocity += self.world_inv_inertia() * torque * dt;
    }

    /// Apply impulse at a point
    pub fn apply_impulse(&mut self, impulse: Vec3, point: Vec3) {
        if self.is_static {
            return;
        }
        // An externally applied impulse is a wake event (see S01).
        self.is_sleeping = false;

        self.velocity += impulse * self.inv_mass;

        let r = point - self.transform.position;
        let torque = r.cross(impulse);
        self.angular_velocity += self.world_inv_inertia() * torque;
    }

    /// Rotation update shared by [`Self::integrate`] and
    /// [`Self::integrate_drift`]: quaternion increment from world-space
    /// angular velocity, composed on the LEFT (`q' = dq_world * q`;
    /// right-multiplying would integrate omega as if body-frame).
    ///
    /// The half-angle sine and cosine come from
    /// [`phieng_math::trig`] rather than the platform libm, so the
    /// 3D trajectory is bit-identical across platforms for the same
    /// reason the 2D one is.
    fn integrate_rotation(&mut self, dt: f32) {
        let w = self.angular_velocity;
        let angle = w.length() * dt;
        if angle > 1e-6 {
            let axis = w / w.length();
            let half_angle = angle * 0.5;
            let (cos_half, sin_half) = phieng_math::trig::compute_cos_sin(half_angle);
            let delta_quat = glam::Quat::from_xyzw(
                axis.x * sin_half,
                axis.y * sin_half,
                axis.z * sin_half,
                cos_half,
            );
            self.transform.rotation = (delta_quat * self.transform.rotation).normalize();
        }
    }

    /// Integrate motion (semi-implicit Euler)
    pub fn integrate(&mut self, dt: f32) {
        if self.is_static {
            return;
        }

        // Apply damping
        let damping = self.linear_damping * dt;
        self.velocity *= 1.0 - damping.min(1.0);

        let ang_damping = self.angular_damping * dt;
        self.angular_velocity *= 1.0 - ang_damping.min(1.0);

        // Update position
        self.transform.position += self.velocity * dt;

        self.integrate_rotation(dt);
    }

    /// Drift step for velocity-Verlet integration (see
    /// [`crate::gravity`]): positions and rotation advance a full step
    /// with the post-kick velocities. Linear damping is NOT applied
    /// here — in Verlet mode it rides the gravity kicks as symmetric
    /// drag. Angular damping matches [`Self::integrate`].
    pub fn integrate_drift(&mut self, dt: f32) {
        if self.is_static {
            return;
        }

        let ang_damping = self.angular_damping * dt;
        self.angular_velocity *= 1.0 - ang_damping.min(1.0);

        // Update position
        self.transform.position += self.velocity * dt;

        self.integrate_rotation(dt);
    }

    /// Check if body should sleep
    pub fn should_sleep(&self, linear_threshold: f32, angular_threshold: f32) -> bool {
        if self.is_static {
            return true;
        }

        let linear_speed = self.velocity.length();
        let angular_speed = self.angular_velocity.length();

        linear_speed < linear_threshold && angular_speed < angular_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigid_body_2d_creation() {
        let body = RigidBody2::new(0);
        assert_eq!(body.id, 0);
        assert!(!body.is_static);
    }

    #[test]
    fn test_apply_force_scales_with_dt() {
        // Force F applied for dt must change velocity by F/m*dt
        // (impulse semantics), not by F/m.
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        let mass = body.mass;
        body.apply_force(Vec2::new(10.0, 0.0), Vec2::ZERO, 0.5);
        assert!(
            (body.velocity.x - 10.0 / mass * 0.5).abs() < 1e-5,
            "vx={}",
            body.velocity.x
        );
        // Static bodies ignore forces.
        let mut wall = RigidBody2::static_body(1);
        wall.apply_force(Vec2::new(10.0, 0.0), Vec2::ZERO, 1.0);
        assert_eq!(wall.velocity, Vec2::ZERO);
    }

    #[test]
    fn test_apply_force_3d_scales_with_dt() {
        let mut body = RigidBody3::new(0);
        body.set_shape(Shape3::Sphere { radius: 1.0 }, 1.0)
            .expect("valid shape");
        let mass = body.mass;
        body.apply_force(Vec3::new(0.0, 6.0, 0.0), Vec3::ZERO, 0.25);
        assert!(
            (body.velocity.y - 6.0 / mass * 0.25).abs() < 1e-5,
            "vy={}",
            body.velocity.y
        );
    }

    #[test]
    fn test_rigid_body_2d_set_shape() {
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");

        assert!(body.shape.is_some());
        assert!(body.mass > 0.0);
        assert!(body.inertia > 0.0);
        // Density flows into the material.
        assert_eq!(body.material.density, 1.0);
    }

    #[test]
    fn test_polygon_mass_matches_box_and_winding() {
        use glam::Vec2;
        // Unit square as CCW and CW windings must yield identical mass.
        let ccw = vec![
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(-0.5, 0.5),
        ];
        let mut cw = ccw.clone();
        cw.reverse();
        let mut a = RigidBody2::new(0);
        a.set_shape(Shape2::Polygon { vertices: ccw }, 2.0)
            .expect("valid shape");
        let mut b = RigidBody2::new(1);
        b.set_shape(Shape2::Polygon { vertices: cw }, 2.0)
            .expect("valid shape");
        // Unit square at density 2: mass 2, inertia 2*(1+1)/12.
        for body in [&a, &b] {
            assert!((body.mass - 2.0).abs() < 1e-4, "mass={}", body.mass);
            assert!(
                (body.inertia - 2.0 * 2.0 / 12.0).abs() < 1e-4,
                "inertia={}",
                body.inertia
            );
        }
        assert!((a.mass - b.mass).abs() < 1e-6);
    }

    #[test]
    fn test_rigid_body_2d_set_material() {
        let mut body = RigidBody2::new(0);
        body.set_shape(Shape2::Circle { radius: 1.0 }, 1.0)
            .expect("valid shape");
        let base_mass = body.mass;

        // Density 2 doubles the mass of the same circle (area pi).
        body.set_material(PhysicsMaterial::new(0.8, 0.6, 2.0))
            .expect("valid shape");
        assert_eq!(body.material.friction, 0.8);
        assert_eq!(body.material.restitution, 0.6);
        assert_eq!(body.material.density, 2.0);
        assert!(
            (body.mass - base_mass * 2.0).abs() < 1e-4,
            "mass={}",
            body.mass
        );
        assert!((body.inv_mass - 1.0 / body.mass).abs() < 1e-6);

        // Invalid density is rejected, not sanitized (boundary rule):
        // a dynamic body with zero density has no valid mass.
        let err = body.set_material(PhysicsMaterial {
            friction: 0.5,
            restitution: 0.5,
            density: 0.0,
            ..Default::default()
        });
        assert_eq!(err, Err(ShapeError::ZeroDensity));
        // Friction/restitution keep sanitize-to-range behavior.
        body.set_material(PhysicsMaterial {
            friction: f32::NAN,
            restitution: 5.0,
            density: 1.0,
            ..Default::default()
        })
        .expect("valid shape");
        assert_eq!(body.material.friction, PhysicsMaterial::DEFAULT_FRICTION);
        assert_eq!(body.material.restitution, 1.0);
    }

    #[test]
    fn test_rigid_body_3d_creation() {
        let body = RigidBody3::new(0);
        assert_eq!(body.id, 0);
        assert!(!body.is_static);
    }

    #[test]
    fn test_rigid_body_3d_set_shape() {
        let mut body = RigidBody3::new(0);
        body.set_shape(Shape3::Sphere { radius: 1.0 }, 1.0)
            .expect("valid shape");

        assert!(body.shape.is_some());
        assert!(body.mass > 0.0);
    }

    #[test]
    fn test_static_body() {
        let body = RigidBody2::static_body(0);
        assert!(body.is_static);
        assert_eq!(body.inv_mass, 0.0);
        assert_eq!(body.inv_inertia, 0.0);
    }

    /// Outward-wound box mesh centered at the origin.
    fn box_mesh(half_extents: Vec3) -> (Vec<Vec3>, Vec<u32>) {
        let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
        let v = vec![
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
        ];
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
    fn test_convex_mesh_box_mass_matches_analytic_box() {
        let he = Vec3::new(0.5, 0.5, 0.5);
        let (verts, idx) = box_mesh(he);
        let mut mesh_body = RigidBody3::new(0);
        mesh_body
            .set_shape(
                Shape3::ConvexMesh {
                    vertices: verts,
                    indices: idx,
                },
                2.0,
            )
            .expect("valid mesh");
        let mut box_body = RigidBody3::new(1);
        box_body
            .set_shape(Shape3::Box { half_extents: he }, 2.0)
            .expect("valid shape");
        assert!((mesh_body.mass - box_body.mass).abs() < 1e-4);
        for (m_row, b_row) in [
            (mesh_body.inertia.x_axis, box_body.inertia.x_axis),
            (mesh_body.inertia.y_axis, box_body.inertia.y_axis),
            (mesh_body.inertia.z_axis, box_body.inertia.z_axis),
        ] {
            assert!(
                (m_row - b_row).length() < 1e-3,
                "mesh inertia {m_row} vs box {b_row}"
            );
        }
    }

    #[test]
    fn test_convex_mesh_tetra_exact_and_recentered() {
        // Trirectangular unit tetrahedron authored corner-at-origin.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let idx = vec![1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1];
        let mut body = RigidBody3::new(0);
        body.set_shape(
            Shape3::ConvexMesh {
                vertices: verts,
                indices: idx,
            },
            1.0,
        )
        .expect("valid mesh");
        assert!((body.mass - 1.0 / 6.0).abs() < 1e-6, "mass={}", body.mass);
        assert!((body.inertia.x_axis.x - 1.0 / 80.0).abs() < 1e-6);
        assert!((body.inertia.y_axis.y - 1.0 / 80.0).abs() < 1e-6);
        assert!((body.inertia.z_axis.z - 1.0 / 80.0).abs() < 1e-6);
        assert!((body.inertia.x_axis.y - 1.0 / 480.0).abs() < 1e-6);
        // Stored vertices are recentered so the COM sits at the origin.
        if let Some(Shape3::ConvexMesh { vertices, .. }) = &body.shape {
            let centroid =
                vertices.iter().fold(Vec3::ZERO, |acc, v| acc + *v) / vertices.len() as f32;
            assert!(centroid.length() < 1e-5, "centroid={centroid}");
        } else {
            panic!("expected ConvexMesh shape");
        }
    }

    #[test]
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn test_convex_mesh_rejects_bad_input() {
        let (verts, idx) = box_mesh(Vec3::new(0.5, 0.5, 0.5));
        let mut body = RigidBody3::new(0);
        // Too few triangles.
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: vec![0, 1, 2]
                },
                1.0
            ),
            Err(ShapeError::TooFewMeshVertices)
        );
        // Index count not a multiple of 3.
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: vec![0, 1]
                },
                1.0
            ),
            Err(ShapeError::DegenerateMesh)
        );
        // Out-of-bounds index.
        let mut bad_idx = idx.clone();
        bad_idx[0] = 99;
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: bad_idx
                },
                1.0
            ),
            Err(ShapeError::InvalidMeshIndex)
        );
        // Degenerate triangle (repeated vertex).
        let mut flat_idx = idx.clone();
        flat_idx[0] = flat_idx[1];
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: flat_idx
                },
                1.0
            ),
            Err(ShapeError::DegenerateMesh)
        );
        // Open mesh (one face removed) has boundary edges.
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: idx[6..].to_vec()
                },
                1.0
            ),
            Err(ShapeError::DegenerateMesh)
        );
        // Inward winding encloses negative volume.
        let mut inward = idx.clone();
        for tri in inward.chunks_exact_mut(3) {
            tri.swap(1, 2);
        }
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: verts.clone(),
                    indices: inward
                },
                1.0
            ),
            Err(ShapeError::DegenerateMesh)
        );
        // Dented corner: closed and wound consistently, but concave.
        let mut dented = verts.clone();
        dented[6] = Vec3::ZERO;
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: dented,
                    indices: idx.clone()
                },
                1.0
            ),
            Err(ShapeError::NonConvexMesh)
        );
        // Non-finite vertex.
        let mut nan_verts = verts.clone();
        nan_verts[0].x = f32::NAN;
        assert_eq!(
            body.set_shape(
                Shape3::ConvexMesh {
                    vertices: nan_verts,
                    indices: idx
                },
                1.0
            ),
            Err(ShapeError::NonFiniteInput)
        );
    }

    #[test]
    fn test_rotation_integrates_without_ping_pong() {
        // Regression: angle extraction must return +angle, or bodies
        // oscillate between +a and -a every frame.
        let mut body = RigidBody2::new(0);
        body.angular_velocity = 1.0; // rad/s
        let dt = 1.0 / 60.0;
        let mut prev = 0.0;
        for _ in 0..10 {
            body.integrate(dt);
            let a = body.rotation_angle();
            assert!(
                a > prev,
                "angle must advance monotonically, prev={prev} now={a}"
            );
            prev = a;
        }
        assert!((prev - 10.0 * dt).abs() < 1e-3, "angle={prev}");
    }
}
