//! Versioned snapshots: save/load for 2D and 3D worlds (JSON + binary).
//!
//! Two formats, one semantic content, one schema version:
//! - **JSON**: human-readable, tolerant of unknown *minor* fields, strict
//!   on schema major. Good for debugging, editing, and version control.
//! - **Binary**: little-endian, sectioned, length-prefixed, CRC32-trailer.
//!   Strict: any magic/version/kind/length/CRC mismatch or malformed
//!   payload is [`SnapshotError`], never a panic or a half-loaded world.
//!
//! Two load modes (see [`LoadMode`]):
//! - **Exact**: restores everything, including warm-start contact impulses,
//!   joint accumulators, and latched reference angles. Continuing to step
//!   after an exact load is bit-identical to never having saved.
//! - **Fresh**: restores structure and configuration only. Transient solver
//!   state (contact caches, joint impulses, latched angles) starts cold,
//!   so the first post-load frames re-converge instead of resuming.
//!
//! Boundary rule (never-guess policy): mass and inertia are RECOMPUTED
//! through the existing `set_shape`/`set_material` paths on load, never
//! trusted from the file. Transforms store full matrix/quaternion
//! components as f32 bits (never re-derived from angles). Broadphase,
//! scratch buffers, and EPA working sets are rebuilt, never stored.
//! Anything ambiguous or out of range is [`SnapshotError::Corrupt`].
//!
//! Schema history: v3 is the pre-gravity format (uniform field only).
//! v4 adds the gravity config block (mode, integrator, G, softening,
//! attractor, mu) and per-body `gravity_scale`. Writers emit v3 for
//! default-gravity worlds (byte-identical to older files) and v4
//! otherwise; loaders accept both, filling v3 gravity state with
//! defaults. JSON additive fields carry `#[serde(default)]` (same
//! precedent as fluid/joints); binary stays strict per version.

use glam::{Mat2, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::body::{RigidBody2, RigidBody3, Shape2, Shape3, ShapeError};
use super::joint::{
    make_distance_joint, make_fixed_joint, make_hinge_joint, make_prismatic_joint, Joint2,
    JointError,
};
use super::joint3::{
    make_distance_joint_3, make_fixed_joint_3, make_hinge_joint_3, make_prismatic_joint_3, Joint3,
    Vec2Pair,
};
use super::material::PhysicsMaterial;
use super::validation::ValidationConfig;
use super::{CachedContact, CachedContact3D, World2, World3};

/// Snapshot schema version. Bump on ANY semantic change; loaders reject
/// unknown versions instead of guessing.
/// v9 adds the collision-tier block (narrowphase tier ceiling).
/// v8 adds the thermal block (conduction/radiation/convection config,
/// per-body temperature, and material thermal properties).
/// v7 adds the solver friction/restitution switches.
/// v6 adds the CCD config block (adaptive-substepping tunables).
/// v5 adds the ambient-force config block (spring/drag/turbulence), the
/// simulation clock, and per-body `ambient_scale`.
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 9;
/// Schema that introduced the thermal block (v8). Gates below must use
/// this (not `VERSION`), so v8 payloads keep decoding after future
/// schemas land.
const SNAPSHOT_SCHEMA_THERMAL: u32 = 8;
/// Schema that introduced the collision-tier block (v9). Gates below must
/// use this (not `VERSION`), so v8 payloads keep decoding after future
/// schemas land.
const SNAPSHOT_SCHEMA_COLLISION_TIER: u32 = 9;
/// Schema that introduced the solver friction/restitution switches (v7).
/// Explicit rather than `VERSION - 1`: the schema ladder is not
/// contiguous-by-one (v8 is thermal, v9 is the tier block), so a
/// positional expression silently retargets the gate when the version
/// bumps.
const SNAPSHOT_SCHEMA_SOLVER_FLAGS: u32 = 7;
/// Schema that introduced the ambient block (v5). Gates below must use
/// this (not `VERSION`), so v5 payloads keep decoding after v6/v7 land.
const SNAPSHOT_SCHEMA_AMBIENT: u32 = 5;
/// Schema that introduced the CCD block (v6). Gates below must use this
/// (not `VERSION`), so v6 payloads keep decoding after v7 lands.
const SNAPSHOT_SCHEMA_CCD: u32 = 6;
/// Schema that introduced the gravity block (v4). Worlds without ambient
/// features but with gravity features serialize at this version.
const SNAPSHOT_SCHEMA_GRAVITY: u32 = 4;
/// Oldest schema this build can read. v3 predates gravity fields
/// (uniform field, unit scales, Euler integrator); they load as
/// defaults. Anything outside `[MIN_READABLE, VERSION]` is rejected.
const SNAPSHOT_SCHEMA_MIN_READABLE: u32 = 3;

/// Binary envelope magic (`PHE1` = physics engine snapshot family; the
/// numeric schema version travels as a separate `u32`).
const BINARY_MAGIC: [u8; 4] = *b"PHE1";
/// World kind tags inside the binary envelope.
const KIND_WORLD2: u8 = 2;
const KIND_WORLD3: u8 = 3;
/// Format tags for the FFI `format` parameter (mirrored in C#).
pub const FORMAT_JSON: u32 = 0;
pub const FORMAT_BINARY: u32 = 1;

/// Load mode: exact resume vs cold (fresh) start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Restore everything, including warm-start caches, joint impulses,
    /// and latched reference angles. Bit-identical continuation.
    Exact,
    /// Restore structure + configuration; start solver state cold
    /// (empty caches, zeroed joint impulses, unlatched angles).
    Fresh,
}

/// Rejection reasons for loading. IO vs parse vs version vs corrupt are
/// distinguished so callers (and FFI error codes) can react correctly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// Filesystem / stream failure.
    Io(String),
    /// Malformed JSON or envelope (magic, length, CRC).
    Parse(String),
    /// Unknown schema version (never guess across versions).
    Version(u32),
    /// Structurally valid envelope, semantically invalid world
    /// (bad ids, failed re-validation, out-of-range values).
    Corrupt(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "snapshot IO failure: {e}"),
            SnapshotError::Parse(e) => write!(f, "snapshot parse failure: {e}"),
            SnapshotError::Version(v) => write!(f, "unsupported snapshot schema version: {v}"),
            SnapshotError::Corrupt(e) => write!(f, "corrupt snapshot: {e}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<ShapeError> for SnapshotError {
    fn from(e: ShapeError) -> Self {
        SnapshotError::Corrupt(e.to_string())
    }
}

impl From<JointError> for SnapshotError {
    fn from(e: JointError) -> Self {
        SnapshotError::Corrupt(e.to_string())
    }
}

fn corrupt<T>(msg: impl Into<String>) -> Result<T, SnapshotError> {
    Err(SnapshotError::Corrupt(msg.into()))
}

fn finite(v: f32, what: &str) -> Result<f32, SnapshotError> {
    if v.is_finite() {
        Ok(v)
    } else {
        corrupt(format!("non-finite {what}"))
    }
}

// ============================================================================
// JSON DTOs (serde derives; field names are the stable schema)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Vec2Dto {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Vec3Dto {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MaterialDto {
    friction: f32,
    restitution: f32,
    density: f32,
    /// Specific heat capacity J/(kg·K) (schema v8; default 900.0, elided).
    #[serde(
        default = "default_specific_heat",
        skip_serializing_if = "is_default_specific_heat"
    )]
    specific_heat: f32,
    /// Thermal conductivity W/(m·K) (schema v8; default 1.0, elided).
    #[serde(
        default = "default_thermal_conductivity",
        skip_serializing_if = "is_default_thermal_conductivity"
    )]
    thermal_conductivity: f32,
    /// Surface emissivity 0..1 (schema v8; default 0.9, elided).
    #[serde(
        default = "default_emissivity",
        skip_serializing_if = "is_default_emissivity"
    )]
    emissivity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Shape2Dto {
    Circle { radius: f32 },
    Box { half_extents: Vec2Dto },
    Polygon { vertices: Vec<Vec2Dto> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Shape3Dto {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3Dto,
    },
    Capsule {
        radius: f32,
        half_length: f32,
    },
    ConvexHull {
        vertices: Vec<Vec3Dto>,
    },
    ConvexMesh {
        vertices: Vec<Vec3Dto>,
        indices: Vec<u32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Body2Dto {
    id: u32,
    is_static: bool,
    is_sleeping: bool,
    linear_damping: f32,
    angular_damping: f32,
    material: MaterialDto,
    shape: Option<Shape2Dto>,
    position: Vec2Dto,
    rotation: [f32; 4],
    velocity: Vec2Dto,
    angular_velocity: f32,
    /// Gravitational coupling multiplier (schema v4; defaults to 1.0
    /// for older files and is omitted when 1.0 so v3 output stays
    /// byte-identical).
    #[serde(default = "default_gravity_scale", skip_serializing_if = "is_one")]
    gravity_scale: f32,
    /// Ambient-force coupling multiplier (schema v5; defaults to 1.0 and
    /// is omitted when 1.0 so pre-ambient output stays byte-identical).
    #[serde(default = "default_gravity_scale", skip_serializing_if = "is_one")]
    ambient_scale: f32,
    /// Temperature in Kelvin (schema v8; default 293.15, elided).
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "is_default_temperature"
    )]
    temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Body3Dto {
    id: u32,
    is_static: bool,
    is_sleeping: bool,
    linear_damping: f32,
    angular_damping: f32,
    material: MaterialDto,
    shape: Option<Shape3Dto>,
    position: Vec3Dto,
    rotation: [f32; 4],
    velocity: Vec3Dto,
    angular_velocity: Vec3Dto,
    /// See `Body2Dto::gravity_scale`.
    #[serde(default = "default_gravity_scale", skip_serializing_if = "is_one")]
    gravity_scale: f32,
    /// See `Body2Dto::ambient_scale`.
    #[serde(default = "default_gravity_scale", skip_serializing_if = "is_one")]
    ambient_scale: f32,
    /// See `Body2Dto::temperature`.
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "is_default_temperature"
    )]
    temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Joint2Dto {
    Distance {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec2Dto,
        anchor_b: Vec2Dto,
        rest_length: f32,
        impulse: f32,
    },
    Hinge {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec2Dto,
        anchor_b: Vec2Dto,
        limit_min: f32,
        limit_max: f32,
        impulse: Vec2Dto,
        angular_impulse: f32,
    },
    Prismatic {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec2Dto,
        anchor_b: Vec2Dto,
        axis_a: Vec2Dto,
        limit_min: f32,
        limit_max: f32,
        ref_angle: Option<f32>,
        perp_impulse: f32,
        angular_impulse: f32,
        limit_impulse: f32,
    },
    Fixed {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec2Dto,
        anchor_b: Vec2Dto,
        ref_angle: Option<f32>,
        impulse: Vec2Dto,
        angular_impulse: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Contact2Dto {
    body_a: u32,
    body_b: u32,
    entries: Vec<ContactEntry2Dto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContactEntry2Dto {
    point: Vec2Dto,
    normal: Vec2Dto,
    normal_impulse: f32,
    tangent_impulse: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Contact3Dto {
    body_a: u32,
    body_b: u32,
    entries: Vec<ContactEntry3Dto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContactEntry3Dto {
    point: Vec3Dto,
    normal: Vec3Dto,
    normal_impulse: f32,
    tangent1_impulse: f32,
    tangent2_impulse: f32,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolverDto {
    iterations: u32,
    slop: f32,
    baumgarte: f32,
    early_exit_tolerance: f32,
    /// Friction switch (schema v7; default true, elided when true so
    /// pre-v7 output stays byte-identical).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    friction: bool,
    /// Restitution switch (schema v7; default true, elided when true).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    restitution: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationDto {
    check_finite: bool,
    max_penetration: f32,
    max_mass_ratio: f32,
    max_coordinate: f32,
    max_velocity: f32,
    max_angular_velocity: f32,
    log_warnings: bool,
    auto_correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluidParticle2Dto {
    position: Vec2Dto,
    velocity: Vec2Dto,
    /// Per-particle mass (schema v5; `0` means "use config mass").
    #[serde(default)]
    mass: f32,
    /// Phase index (schema v5; default 0).
    #[serde(default)]
    phase: u8,
}

/// Per-phase fluid definition (schema v5). Omitted when single-phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhasesDto {
    count: u8,
    rest_density: [f32; 4],
    mass: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluidConfig2Dto {
    particle_mass: f32,
    rest_density: f32,
    smoothing_length: f32,
    stiffness: f32,
    /// Tait EOS exponent (schema v5; default 1.0 = linear, elided).
    #[serde(default = "default_gamma", skip_serializing_if = "is_one")]
    gamma: f32,
    /// Monaghan artificial viscosity alpha (schema v5; default 0.0, elided).
    #[serde(default, skip_serializing_if = "is_zero")]
    artificial_viscosity: f32,
    /// Surface-tension strength kappa (schema v5; default 0.0, elided).
    #[serde(default, skip_serializing_if = "is_zero")]
    surface_tension: f32,
    /// CFL substepping factor (schema v5; default 0.0, elided).
    #[serde(default, skip_serializing_if = "is_zero")]
    cfl: f32,
    /// Maximum CFL substeps (schema v5; default 1, elided).
    #[serde(default = "default_max_substeps", skip_serializing_if = "is_one_u32")]
    max_substeps: u32,
    /// Adaptive-resolution flag (schema v5; default false, elided).
    #[serde(default, skip_serializing_if = "is_false")]
    adaptive_resolution: bool,
    /// Split density ratio (schema v5; default 0.5, elided).
    #[serde(default = "default_split_ratio", skip_serializing_if = "is_half")]
    split_density_ratio: f32,
    /// Merge density ratio (schema v5; default 1.6, elided).
    #[serde(
        default = "default_merge_ratio",
        skip_serializing_if = "is_one_point_six"
    )]
    merge_density_ratio: f32,
    /// Boundary-particle flag (schema v5; default false, elided).
    #[serde(default, skip_serializing_if = "is_false")]
    boundary_particles: bool,
    viscosity: f32,
    drag: f32,
    /// Wall friction coefficient (schema v5; default 0.0, elided).
    #[serde(default, skip_serializing_if = "is_zero")]
    wall_friction: f32,
    max_velocity: f32,
    particle_radius: f32,
    enabled: bool,
    /// Per-phase definitions (schema v5; omitted when single-phase).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    phases: Option<PhasesDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Fluid2Dto {
    config: FluidConfig2Dto,
    particles: Vec<FluidParticle2Dto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluidParticle3Dto {
    position: Vec3Dto,
    velocity: Vec3Dto,
    /// Per-particle mass (schema v5; `0` means "use config mass").
    #[serde(default)]
    mass: f32,
    /// Phase index (schema v5; default 0).
    #[serde(default)]
    phase: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluidConfig3Dto {
    particle_mass: f32,
    rest_density: f32,
    smoothing_length: f32,
    stiffness: f32,
    #[serde(default = "default_gamma", skip_serializing_if = "is_one")]
    gamma: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    artificial_viscosity: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    surface_tension: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    cfl: f32,
    #[serde(default = "default_max_substeps", skip_serializing_if = "is_one_u32")]
    max_substeps: u32,
    #[serde(default, skip_serializing_if = "is_false")]
    adaptive_resolution: bool,
    #[serde(default = "default_split_ratio", skip_serializing_if = "is_half")]
    split_density_ratio: f32,
    #[serde(
        default = "default_merge_ratio",
        skip_serializing_if = "is_one_point_six"
    )]
    merge_density_ratio: f32,
    #[serde(default, skip_serializing_if = "is_false")]
    boundary_particles: bool,
    viscosity: f32,
    drag: f32,
    /// Wall friction coefficient (schema v5; default 0.0, elided).
    #[serde(default, skip_serializing_if = "is_zero")]
    wall_friction: f32,
    max_velocity: f32,
    particle_radius: f32,
    enabled: bool,
    /// Per-phase definitions (schema v5; omitted when single-phase).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    phases: Option<PhasesDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Fluid3Dto {
    config: FluidConfig3Dto,
    particles: Vec<FluidParticle3Dto>,
}

impl Default for FluidConfig2Dto {
    fn default() -> Self {
        let c = super::fluid::FluidConfig2::default();
        Self {
            particle_mass: c.particle_mass,
            rest_density: c.rest_density,
            smoothing_length: c.smoothing_length,
            stiffness: c.stiffness,
            gamma: c.gamma,
            artificial_viscosity: c.artificial_viscosity,
            surface_tension: c.surface_tension,
            cfl: c.cfl,
            max_substeps: c.max_substeps,
            adaptive_resolution: c.adaptive_resolution,
            split_density_ratio: c.split_density_ratio,
            merge_density_ratio: c.merge_density_ratio,
            boundary_particles: c.boundary_particles,
            viscosity: c.viscosity,
            drag: c.drag,
            wall_friction: c.wall_friction,
            max_velocity: c.max_velocity,
            particle_radius: c.particle_radius,
            enabled: c.enabled,
            phases: None,
        }
    }
}

impl Default for FluidConfig3Dto {
    fn default() -> Self {
        let c = super::fluid3::FluidConfig3::default();
        Self {
            particle_mass: c.particle_mass,
            rest_density: c.rest_density,
            smoothing_length: c.smoothing_length,
            stiffness: c.stiffness,
            gamma: c.gamma,
            artificial_viscosity: c.artificial_viscosity,
            surface_tension: c.surface_tension,
            cfl: c.cfl,
            max_substeps: c.max_substeps,
            adaptive_resolution: c.adaptive_resolution,
            split_density_ratio: c.split_density_ratio,
            merge_density_ratio: c.merge_density_ratio,
            boundary_particles: c.boundary_particles,
            viscosity: c.viscosity,
            drag: c.drag,
            wall_friction: c.wall_friction,
            max_velocity: c.max_velocity,
            particle_radius: c.particle_radius,
            enabled: c.enabled,
            phases: None,
        }
    }
}

/// Default `gravity_scale` for pre-v4 files (full coupling).
fn default_gravity_scale() -> f32 {
    1.0
}

/// `skip_serializing_if` companion: unit scales are omitted so
/// default-gravity worlds serialize exactly like schema v3.
fn is_one(v: &f32) -> bool {
    *v == 1.0
}

/// `skip_serializing_if` companion: a zero clock is omitted so worlds
/// that never used ambient features serialize exactly like schema v3/v4.
fn is_zero(v: &f32) -> bool {
    *v == 0.0
}

/// Default Tait exponent (linear EOS) for pre-v5 fluid configs.
fn default_gamma() -> f32 {
    1.0
}

/// Default CFL max-substeps (off) for pre-v5 fluid configs.
fn default_max_substeps() -> u32 {
    1
}

/// `skip_serializing_if` companion for `max_substeps` (default 1, elided).
fn is_one_u32(v: &u32) -> bool {
    *v == 1
}

/// `skip_serializing_if` companion for flags defaulting to false.
fn is_false(v: &bool) -> bool {
    !*v
}

/// `skip_serializing_if` companion for flags defaulting to true.
fn is_true(v: &bool) -> bool {
    *v
}

/// Default adaptive-resolution split ratio.
fn default_split_ratio() -> f32 {
    0.5
}

/// Default adaptive-resolution merge ratio.
fn default_merge_ratio() -> f32 {
    1.6
}

/// `skip_serializing_if` companion for `split_density_ratio` (0.5).
fn is_half(v: &f32) -> bool {
    *v == 0.5
}

/// `skip_serializing_if` companion for `merge_density_ratio` (1.6).
fn is_one_point_six(v: &f32) -> bool {
    *v == 1.6
}

/// Gravity configuration DTO (schema v4). Field names mirror
/// [`crate::gravity::GravityConfig2`]; `mode`/`integrator` are the
/// enum discriminants (0 = Uniform/Euler).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GravityConfig2Dto {
    mode: u32,
    integrator: u32,
    big_g: f32,
    softening: f32,
    attractor: Vec2Dto,
    attractor_mu: f32,
    /// Barnes-Hut opening criterion. Required in v4 files like every
    /// other field (only the whole-config default covers v3 files).
    theta: f32,
}

impl Default for GravityConfig2Dto {
    fn default() -> Self {
        Self {
            mode: 0,
            integrator: 0,
            big_g: 1.0,
            softening: 0.0,
            attractor: Vec2Dto { x: 0.0, y: 0.0 },
            attractor_mu: 0.0,
            theta: 0.5,
        }
    }
}

impl GravityConfig2Dto {
    fn is_default(&self) -> bool {
        self.mode == 0
            && self.integrator == 0
            && self.big_g == 1.0
            && self.softening == 0.0
            && self.attractor.x == 0.0
            && self.attractor.y == 0.0
            && self.attractor_mu == 0.0
            && self.theta == 0.5
    }
}

/// 3D counterpart of [`GravityConfig2Dto`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GravityConfig3Dto {
    mode: u32,
    integrator: u32,
    big_g: f32,
    softening: f32,
    attractor: Vec3Dto,
    attractor_mu: f32,
    /// See `GravityConfig2Dto::theta`.
    theta: f32,
}

impl Default for GravityConfig3Dto {
    fn default() -> Self {
        Self {
            mode: 0,
            integrator: 0,
            big_g: 1.0,
            softening: 0.0,
            attractor: Vec3Dto {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            attractor_mu: 0.0,
            theta: 0.5,
        }
    }
}

impl GravityConfig3Dto {
    fn is_default(&self) -> bool {
        self.mode == 0
            && self.integrator == 0
            && self.big_g == 1.0
            && self.softening == 0.0
            && self.attractor.x == 0.0
            && self.attractor.y == 0.0
            && self.attractor.z == 0.0
            && self.attractor_mu == 0.0
            && self.theta == 0.5
    }
}

// ---- CCD config DTO (schema v6) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CcdConfigDto {
    enabled: bool,
    tier: u32,
    max_movement: f32,
    max_substeps: u32,
}

impl Default for CcdConfigDto {
    fn default() -> Self {
        let c = super::ccd::CcdConfig::default();
        Self {
            enabled: c.enabled,
            tier: c.tier as u32,
            max_movement: c.max_movement,
            max_substeps: c.max_substeps,
        }
    }
}

impl CcdConfigDto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.tier == d.tier
            && self.max_movement == d.max_movement
            && self.max_substeps == d.max_substeps
    }
}

fn ccd_config_dto(c: &super::ccd::CcdConfig) -> CcdConfigDto {
    CcdConfigDto {
        enabled: c.enabled,
        tier: c.tier as u32,
        max_movement: c.max_movement,
        max_substeps: c.max_substeps,
    }
}

// ---- Collision-tier config DTO (schema v9) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollisionTierConfigDto {
    /// Master switch. Omitted when false so default worlds stay
    /// byte-identical to older schemas.
    #[serde(default, skip_serializing_if = "is_false")]
    enabled: bool,
    /// Ceiling tag (0 = Tier1_AABBOnly .. 4 = Tier5_GJK_EPA). Omitted at
    /// the default (Tier5) for the same reason.
    #[serde(
        default = "default_collision_tier_tag",
        skip_serializing_if = "is_default_collision_tier_tag"
    )]
    max_tier: u32,
}

fn default_collision_tier_tag() -> u32 {
    super::collision_tier::CollisionTier::Tier5_GJK_EPA as u32
}

fn is_default_collision_tier_tag(v: &u32) -> bool {
    *v == default_collision_tier_tag()
}

impl Default for CollisionTierConfigDto {
    /// Must be the *engine* default (Tier5, full fidelity), not a
    /// zero-valued struct: an absent field in an older file means
    /// "no tier ceiling", and Tier1 (AABB-only) would silently gut the
    /// narrowphase.
    fn default() -> Self {
        Self {
            enabled: false,
            max_tier: default_collision_tier_tag(),
        }
    }
}

impl CollisionTierConfigDto {
    fn is_default(&self) -> bool {
        !self.enabled && is_default_collision_tier_tag(&self.max_tier)
    }
}

fn collision_tier_dto(c: &super::collision_tier::CollisionTierConfig) -> CollisionTierConfigDto {
    CollisionTierConfigDto {
        enabled: c.enabled,
        max_tier: c.max_tier as u32,
    }
}

fn collision_tier_from_dto(
    d: &CollisionTierConfigDto,
) -> Result<super::collision_tier::CollisionTierConfig, SnapshotError> {
    use super::collision_tier::CollisionTier as T;
    let max_tier = match d.max_tier {
        0 => T::Tier1_AABBOnly,
        1 => T::Tier2_SphereAABB,
        2 => T::Tier3_SphereSphere,
        3 => T::Tier4_BoxBox,
        4 => T::Tier5_GJK_EPA,
        other => return corrupt(format!("unknown collision tier tag {other}")),
    };
    Ok(super::collision_tier::CollisionTierConfig {
        enabled: d.enabled,
        max_tier,
    })
}

use super::ccd::CcdTier;

fn ccd_config_from_dto(c: &CcdConfigDto) -> super::ccd::CcdConfig {
    super::ccd::CcdConfig {
        enabled: c.enabled,
        tier: match c.tier {
            0 => CcdTier::Off,
            1 => CcdTier::SweptAABB,
            2 => CcdTier::FullGJK,
            _ => CcdTier::Off,
        },
        max_movement: c.max_movement,
        max_substeps: c.max_substeps,
    }
}

// ---- Thermal config DTO (schema v8) ----

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ThermalConfigDto {
    /// Master switch for thermal stepping.
    #[serde(default, skip_serializing_if = "is_false")]
    enabled: bool,
    /// Ambient/environment temperature in Kelvin.
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "is_default_temperature"
    )]
    ambient_temperature: f32,
    /// Stefan–Boltzmann constant σ in W/(m²·K⁴).
    #[serde(
        default = "default_stefan_boltzmann",
        skip_serializing_if = "is_default_stefan_boltzmann"
    )]
    stefan_boltzmann: f32,
    /// Convection coefficient h in W/(m²·K).
    #[serde(
        default = "default_convection_coefficient",
        skip_serializing_if = "is_default_convection_coefficient"
    )]
    convection_coefficient: f32,
    /// Enable body-to-body + body-to-environment radiation.
    #[serde(default, skip_serializing_if = "is_true")]
    radiation_enabled: bool,
    /// Enable contact-driven conduction.
    #[serde(default, skip_serializing_if = "is_true")]
    conduction_enabled: bool,
    /// Enable ambient convection.
    #[serde(default, skip_serializing_if = "is_true")]
    convection_enabled: bool,
    /// Safety cap on temperature in Kelvin.
    #[serde(
        default = "default_max_temperature",
        skip_serializing_if = "is_default_max_temperature"
    )]
    max_temperature: f32,
    /// Safety floor in Kelvin (absolute zero).
    #[serde(
        default = "default_min_temperature",
        skip_serializing_if = "is_default_min_temperature"
    )]
    min_temperature: f32,
    /// Radiation cutoff as multiplier of combined bounding radii.
    #[serde(
        default = "default_radiation_radius_factor",
        skip_serializing_if = "is_default_radiation_radius_factor"
    )]
    radiation_radius_factor: f32,
    /// Maximum temperature change per step in Kelvin.
    #[serde(
        default = "default_max_temperature_rate",
        skip_serializing_if = "is_default_max_temperature_rate"
    )]
    max_temperature_rate: f32,
}

impl ThermalConfigDto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.ambient_temperature == d.ambient_temperature
            && self.stefan_boltzmann == d.stefan_boltzmann
            && self.convection_coefficient == d.convection_coefficient
            && self.radiation_enabled == d.radiation_enabled
            && self.conduction_enabled == d.conduction_enabled
            && self.convection_enabled == d.convection_enabled
            && self.max_temperature == d.max_temperature
            && self.min_temperature == d.min_temperature
            && self.radiation_radius_factor == d.radiation_radius_factor
            && self.max_temperature_rate == d.max_temperature_rate
    }
}

fn thermal_config_dto(c: &super::thermal::ThermalConfig2D) -> ThermalConfigDto {
    ThermalConfigDto {
        enabled: c.enabled,
        ambient_temperature: c.ambient_temperature,
        stefan_boltzmann: c.stefan_boltzmann,
        convection_coefficient: c.convection_coefficient,
        radiation_enabled: c.radiation_enabled,
        conduction_enabled: c.conduction_enabled,
        convection_enabled: c.convection_enabled,
        max_temperature: c.max_temperature,
        min_temperature: c.min_temperature,
        radiation_radius_factor: c.radiation_radius_factor,
        max_temperature_rate: c.max_temperature_rate,
    }
}

fn thermal_config_from_dto(c: &ThermalConfigDto) -> super::thermal::ThermalConfig2D {
    super::thermal::ThermalConfig2D {
        enabled: c.enabled,
        ambient_temperature: c.ambient_temperature,
        stefan_boltzmann: c.stefan_boltzmann,
        convection_coefficient: c.convection_coefficient,
        radiation_enabled: c.radiation_enabled,
        conduction_enabled: c.conduction_enabled,
        convection_enabled: c.convection_enabled,
        max_temperature: c.max_temperature,
        min_temperature: c.min_temperature,
        radiation_radius_factor: c.radiation_radius_factor,
        max_temperature_rate: c.max_temperature_rate,
    }
}

/// Broad-phase backend tag used by schema-everything worlds (the default
/// grid). Tag values are part of the schema: 0 brute force, 1 spatial
/// hash, 2 dynamic tree, 3 hybrid — matching
/// [`phieng_broadphase::BroadphaseMode`] discriminants *by value*.
fn default_broadphase_mode_tag() -> u32 {
    1
}

/// `skip_serializing_if` companion: the grid tag is elided so default
/// worlds stay byte-identical to pre-tier-system files.
fn is_spatial_hash_tag(v: &u32) -> bool {
    *v == 1
}

fn broadphase_mode_discriminant(m: &phieng_broadphase::BroadphaseMode) -> u32 {
    use phieng_broadphase::BroadphaseMode as M;
    match m {
        M::BruteForce => 0,
        M::SpatialHash => 1,
        M::DynamicAABBTree => 2,
        M::Hybrid => 3,
    }
}

/// Validate a stored backend tag. Unknown tags are `Corrupt`, never a
/// silent default (the serde `Default` only fills fields *absent* from
/// older files, which are the default grid by construction).
fn broadphase_mode_from_tag(tag: u32) -> Result<phieng_broadphase::BroadphaseMode, SnapshotError> {
    use phieng_broadphase::BroadphaseMode as M;
    match tag {
        0 => Ok(M::BruteForce),
        1 => Ok(M::SpatialHash),
        2 => Ok(M::DynamicAABBTree),
        3 => Ok(M::Hybrid),
        _ => corrupt(format!("unknown broadphase mode tag {}", tag)),
    }
}

/// Default specific heat capacity J/(kg·K) for pre-v8 files.
fn default_specific_heat() -> f32 {
    super::material::PhysicsMaterial::DEFAULT_SPECIFIC_HEAT
}
/// Default thermal conductivity W/(m·K) for pre-v8 files.
fn default_thermal_conductivity() -> f32 {
    super::material::PhysicsMaterial::DEFAULT_THERMAL_CONDUCTIVITY
}
/// Default surface emissivity 0..1 for pre-v8 files.
fn default_emissivity() -> f32 {
    super::material::PhysicsMaterial::DEFAULT_EMISSIVITY
}

/// Default temperature in Kelvin for pre-v8 files.
fn default_temperature() -> f32 {
    phieng_math::DEFAULT_TEMPERATURE
}
/// Default Stefan–Boltzmann constant for pre-v8 files.
fn default_stefan_boltzmann() -> f32 {
    phieng_math::STEFAN_BOLTZMANN
}
/// Default convection coefficient for pre-v8 files.
fn default_convection_coefficient() -> f32 {
    10.0
}
/// Default max temperature for pre-v8 files.
fn default_max_temperature() -> f32 {
    phieng_math::MAX_TEMPERATURE
}
/// Default min temperature for pre-v8 files.
fn default_min_temperature() -> f32 {
    phieng_math::MIN_TEMPERATURE
}
/// Default radiation radius factor for pre-v8 files.
fn default_radiation_radius_factor() -> f32 {
    5.0
}
/// Default max temperature rate for pre-v8 files.
fn default_max_temperature_rate() -> f32 {
    100.0
}

/// `skip_serializing_if` companion for specific heat (900.0).
fn is_default_specific_heat(v: &f32) -> bool {
    *v == super::material::PhysicsMaterial::DEFAULT_SPECIFIC_HEAT
}
/// `skip_serializing_if` companion for thermal conductivity (1.0).
fn is_default_thermal_conductivity(v: &f32) -> bool {
    *v == super::material::PhysicsMaterial::DEFAULT_THERMAL_CONDUCTIVITY
}
/// `skip_serializing_if` companion for emissivity (0.9).
fn is_default_emissivity(v: &f32) -> bool {
    *v == super::material::PhysicsMaterial::DEFAULT_EMISSIVITY
}
/// `skip_serializing_if` companion for temperature (293.15 K).
fn is_default_temperature(v: &f32) -> bool {
    *v == phieng_math::DEFAULT_TEMPERATURE
}
/// `skip_serializing_if` companion for Stefan–Boltzmann constant.
fn is_default_stefan_boltzmann(v: &f32) -> bool {
    *v == phieng_math::STEFAN_BOLTZMANN
}
/// `skip_serializing_if` companion for convection coefficient (10.0).
fn is_default_convection_coefficient(v: &f32) -> bool {
    *v == 10.0
}
/// `skip_serializing_if` companion for max temperature (5000.0).
fn is_default_max_temperature(v: &f32) -> bool {
    *v == phieng_math::MAX_TEMPERATURE
}
/// `skip_serializing_if` companion for min temperature (0.0).
fn is_default_min_temperature(v: &f32) -> bool {
    *v == phieng_math::MIN_TEMPERATURE
}
/// `skip_serializing_if` companion for radiation radius factor (5.0).
fn is_default_radiation_radius_factor(v: &f32) -> bool {
    *v == 5.0
}
/// `skip_serializing_if` companion for max temperature rate (100.0).
fn is_default_max_temperature_rate(v: &f32) -> bool {
    *v == 100.0
}

// ---- Ambient force DTOs (schema v5) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Spring2Dto {
    enabled: bool,
    anchor: Vec2Dto,
    stiffness: f32,
    damping: f32,
    max_displacement: f32,
}

impl Default for Spring2Dto {
    fn default() -> Self {
        let c = super::ambient::SpringConfig2::default();
        Self {
            enabled: c.enabled,
            anchor: v2(c.anchor),
            stiffness: c.stiffness,
            damping: c.damping,
            max_displacement: c.max_displacement,
        }
    }
}

impl Spring2Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.anchor.x == d.anchor.x
            && self.anchor.y == d.anchor.y
            && self.stiffness == d.stiffness
            && self.damping == d.damping
            && self.max_displacement == d.max_displacement
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Drag2Dto {
    enabled: bool,
    linear: f32,
    quadratic: f32,
    air_density: f32,
    angular: f32,
    wind: Vec2Dto,
}

impl Default for Drag2Dto {
    fn default() -> Self {
        let c = super::ambient::DragConfig2::default();
        Self {
            enabled: c.enabled,
            linear: c.linear,
            quadratic: c.quadratic,
            air_density: c.air_density,
            angular: c.angular,
            wind: v2(c.wind),
        }
    }
}

impl Drag2Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.linear == d.linear
            && self.quadratic == d.quadratic
            && self.air_density == d.air_density
            && self.angular == d.angular
            && self.wind.x == d.wind.x
            && self.wind.y == d.wind.y
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Turbulence2Dto {
    enabled: bool,
    amplitude: f32,
    frequency: f32,
    time_scale: f32,
    seed: u32,
    octaves: u8,
    divergence_free: bool,
}

impl Default for Turbulence2Dto {
    fn default() -> Self {
        let c = super::ambient::TurbulenceConfig2::default();
        Self {
            enabled: c.enabled,
            amplitude: c.amplitude,
            frequency: c.frequency,
            time_scale: c.time_scale,
            seed: c.seed,
            octaves: c.octaves,
            divergence_free: c.divergence_free,
        }
    }
}

impl Turbulence2Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.amplitude == d.amplitude
            && self.frequency == d.frequency
            && self.time_scale == d.time_scale
            && self.seed == d.seed
            && self.octaves == d.octaves
            && self.divergence_free == d.divergence_free
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AmbientConfig2Dto {
    spring: Spring2Dto,
    drag: Drag2Dto,
    turbulence: Turbulence2Dto,
}

impl AmbientConfig2Dto {
    fn is_default(&self) -> bool {
        self.spring.is_default() && self.drag.is_default() && self.turbulence.is_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Spring3Dto {
    enabled: bool,
    anchor: Vec3Dto,
    stiffness: f32,
    damping: f32,
    max_displacement: f32,
}

impl Default for Spring3Dto {
    fn default() -> Self {
        let c = super::ambient::SpringConfig3::default();
        Self {
            enabled: c.enabled,
            anchor: v3(c.anchor),
            stiffness: c.stiffness,
            damping: c.damping,
            max_displacement: c.max_displacement,
        }
    }
}

impl Spring3Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.anchor.x == d.anchor.x
            && self.anchor.y == d.anchor.y
            && self.anchor.z == d.anchor.z
            && self.stiffness == d.stiffness
            && self.damping == d.damping
            && self.max_displacement == d.max_displacement
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Drag3Dto {
    enabled: bool,
    linear: f32,
    quadratic: f32,
    air_density: f32,
    angular: f32,
    wind: Vec3Dto,
}

impl Default for Drag3Dto {
    fn default() -> Self {
        let c = super::ambient::DragConfig3::default();
        Self {
            enabled: c.enabled,
            linear: c.linear,
            quadratic: c.quadratic,
            air_density: c.air_density,
            angular: c.angular,
            wind: v3(c.wind),
        }
    }
}

impl Drag3Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.linear == d.linear
            && self.quadratic == d.quadratic
            && self.air_density == d.air_density
            && self.angular == d.angular
            && self.wind.x == d.wind.x
            && self.wind.y == d.wind.y
            && self.wind.z == d.wind.z
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Turbulence3Dto {
    enabled: bool,
    amplitude: f32,
    frequency: f32,
    time_scale: f32,
    seed: u32,
    octaves: u8,
    divergence_free: bool,
}

impl Default for Turbulence3Dto {
    fn default() -> Self {
        let c = super::ambient::TurbulenceConfig3::default();
        Self {
            enabled: c.enabled,
            amplitude: c.amplitude,
            frequency: c.frequency,
            time_scale: c.time_scale,
            seed: c.seed,
            octaves: c.octaves,
            divergence_free: c.divergence_free,
        }
    }
}

impl Turbulence3Dto {
    fn is_default(&self) -> bool {
        let d = Self::default();
        self.enabled == d.enabled
            && self.amplitude == d.amplitude
            && self.frequency == d.frequency
            && self.time_scale == d.time_scale
            && self.seed == d.seed
            && self.octaves == d.octaves
            && self.divergence_free == d.divergence_free
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AmbientConfig3Dto {
    spring: Spring3Dto,
    drag: Drag3Dto,
    turbulence: Turbulence3Dto,
}

impl AmbientConfig3Dto {
    fn is_default(&self) -> bool {
        self.spring.is_default() && self.drag.is_default() && self.turbulence.is_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct World2Dto {
    schema: u32,
    gravity: Vec2Dto,
    /// Schema v4; omitted (and defaulted) for v3 files so pre-gravity
    /// snapshots load with a default config.
    #[serde(default, skip_serializing_if = "GravityConfig2Dto::is_default")]
    gravity_config: GravityConfig2Dto,
    /// Schema v5; omitted (and defaulted) for pre-ambient files.
    #[serde(default, skip_serializing_if = "AmbientConfig2Dto::is_default")]
    ambient: AmbientConfig2Dto,
    /// Schema v5; simulation clock. Omitted when zero so pre-ambient
    /// output stays byte-identical.
    #[serde(default, skip_serializing_if = "is_zero")]
    time: f32,
    /// Schema v6; CCD config. Omitted (and defaulted) for pre-CCD files
    /// so v3/v4/v5 output stays byte-identical.
    #[serde(default, skip_serializing_if = "CcdConfigDto::is_default")]
    ccd: CcdConfigDto,
    /// Schema v8; thermal config. Omitted (and defaulted) for
    /// pre-thermal worlds so default output stays byte-identical.
    #[serde(default, skip_serializing_if = "ThermalConfigDto::is_default")]
    thermal: ThermalConfigDto,
    /// Schema v8; broad-phase backend tag (0 brute force, 1 spatial hash,
    /// 2 dynamic tree, 3 hybrid). Omitted when the default grid is in use
    /// so pre-tier-system output stays byte-identical.
    #[serde(
        default = "default_broadphase_mode_tag",
        skip_serializing_if = "is_spatial_hash_tag"
    )]
    broadphase_mode: u32,
    /// Schema v9; narrowphase collision-tier ceiling. Omitted (and
    /// defaulted) for pre-v9 files so default output stays
    /// byte-identical.
    #[serde(default, skip_serializing_if = "CollisionTierConfigDto::is_default")]
    collision_tier: CollisionTierConfigDto,
    dt: f32,
    linear_sleep_threshold: f32,
    angular_sleep_threshold: f32,
    solver: SolverDto,
    validation: ValidationDto,
    next_body_id: u32,
    next_joint_id: u32,
    cell_size: f32,
    bodies: Vec<Body2Dto>,
    joints: Vec<Joint2Dto>,
    contacts: Vec<Contact2Dto>,
    #[serde(default)]
    fluid: Fluid2Dto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Joint3Dto {
    Distance3 {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3Dto,
        anchor_b: Vec3Dto,
        rest_length: f32,
        impulse: f32,
    },
    Hinge3 {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3Dto,
        anchor_b: Vec3Dto,
        axis_a: Vec3Dto,
        axis_b: Vec3Dto,
        limit_min: f32,
        limit_max: f32,
        impulse: Vec3Dto,
        align_u1: f32,
        align_u2: f32,
        twist_impulse: f32,
    },
    Prismatic3 {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3Dto,
        anchor_b: Vec3Dto,
        axis_a: Vec3Dto,
        limit_min: f32,
        limit_max: f32,
        ref_orientation: Option<[f32; 4]>,
        perp_u1: f32,
        perp_u2: f32,
        angular_impulse: Vec3Dto,
        limit_impulse: f32,
    },
    Fixed3 {
        id: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3Dto,
        anchor_b: Vec3Dto,
        ref_orientation: Option<[f32; 4]>,
        impulse: Vec3Dto,
        angular_impulse: Vec3Dto,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct World3Dto {
    schema: u32,
    gravity: Vec3Dto,
    /// Schema v4; omitted (and defaulted) for v3 files.
    #[serde(default, skip_serializing_if = "GravityConfig3Dto::is_default")]
    gravity_config: GravityConfig3Dto,
    /// Schema v5; omitted (and defaulted) for pre-ambient files.
    #[serde(default, skip_serializing_if = "AmbientConfig3Dto::is_default")]
    ambient: AmbientConfig3Dto,
    /// Schema v5; simulation clock (omitted when zero).
    #[serde(default, skip_serializing_if = "is_zero")]
    time: f32,
    /// Schema v6; CCD config. Omitted (and defaulted) for pre-CCD files.
    #[serde(default, skip_serializing_if = "CcdConfigDto::is_default")]
    ccd: CcdConfigDto,
    /// Schema v8; thermal config. Omitted (and defaulted) for
    /// pre-thermal worlds so default output stays byte-identical.
    #[serde(default, skip_serializing_if = "ThermalConfigDto::is_default")]
    thermal: ThermalConfigDto,
    /// Schema v8; broad-phase backend tag (see `World2Dto::broadphase_mode`).
    #[serde(
        default = "default_broadphase_mode_tag",
        skip_serializing_if = "is_spatial_hash_tag"
    )]
    broadphase_mode: u32,
    /// Schema v9; narrowphase collision-tier ceiling. Omitted (and
    /// defaulted) for pre-v9 files so default output stays
    /// byte-identical.
    #[serde(default, skip_serializing_if = "CollisionTierConfigDto::is_default")]
    collision_tier: CollisionTierConfigDto,
    dt: f32,
    linear_sleep_threshold: f32,
    angular_sleep_threshold: f32,
    solver: SolverDto,
    validation: ValidationDto,
    next_body_id: u32,
    // Added with 3D joints: defaulted so pre-joint JSON snapshots load
    // with empty joints; binary stays strict (old payloads fail with a
    // clean truncation error, never misparsed).
    #[serde(default)]
    next_joint_id: u32,
    bodies: Vec<Body3Dto>,
    #[serde(default)]
    joints: Vec<Joint3Dto>,
    contacts: Vec<Contact3Dto>,
    #[serde(default)]
    fluid: Fluid3Dto,
}

// ============================================================================
// DTO conversion (world -> DTO)
// ============================================================================

fn v2(v: Vec2) -> Vec2Dto {
    Vec2Dto { x: v.x, y: v.y }
}

fn v3(v: Vec3) -> Vec3Dto {
    Vec3Dto {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn mat_dto(m: PhysicsMaterial) -> MaterialDto {
    MaterialDto {
        friction: m.friction,
        restitution: m.restitution,
        density: m.density,
        specific_heat: m.specific_heat,
        thermal_conductivity: m.thermal_conductivity,
        emissivity: m.emissivity,
    }
}

fn phases_config_2(c: &super::fluid::FluidConfig2) -> Option<PhasesDto> {
    if c.phase_count <= 1 {
        None
    } else {
        Some(PhasesDto {
            count: c.phase_count,
            rest_density: c.phase_rest_density,
            mass: c.phase_mass,
        })
    }
}

fn phases_config_3(c: &super::fluid3::FluidConfig3) -> Option<PhasesDto> {
    if c.phase_count <= 1 {
        None
    } else {
        Some(PhasesDto {
            count: c.phase_count,
            rest_density: c.phase_rest_density,
            mass: c.phase_mass,
        })
    }
}

fn fluid2_dto(f: &super::fluid::FluidState2) -> Fluid2Dto {
    let c = &f.config;
    Fluid2Dto {
        config: FluidConfig2Dto {
            particle_mass: c.particle_mass,
            rest_density: c.rest_density,
            smoothing_length: c.smoothing_length,
            stiffness: c.stiffness,
            gamma: c.gamma,
            artificial_viscosity: c.artificial_viscosity,
            surface_tension: c.surface_tension,
            cfl: c.cfl,
            max_substeps: c.max_substeps,
            adaptive_resolution: c.adaptive_resolution,
            split_density_ratio: c.split_density_ratio,
            merge_density_ratio: c.merge_density_ratio,
            boundary_particles: c.boundary_particles,
            viscosity: c.viscosity,
            drag: c.drag,
            wall_friction: c.wall_friction,
            max_velocity: c.max_velocity,
            particle_radius: c.particle_radius,
            enabled: c.enabled,
            phases: phases_config_2(c),
        },
        particles: f
            .particles
            .iter()
            .map(|p| FluidParticle2Dto {
                position: v2(p.position),
                velocity: v2(p.velocity),
                mass: p.mass,
                phase: p.phase,
            })
            .collect(),
    }
}

fn fluid3_dto(f: &super::fluid3::FluidState3) -> Fluid3Dto {
    let c = &f.config;
    Fluid3Dto {
        config: FluidConfig3Dto {
            particle_mass: c.particle_mass,
            rest_density: c.rest_density,
            smoothing_length: c.smoothing_length,
            stiffness: c.stiffness,
            gamma: c.gamma,
            artificial_viscosity: c.artificial_viscosity,
            surface_tension: c.surface_tension,
            cfl: c.cfl,
            max_substeps: c.max_substeps,
            adaptive_resolution: c.adaptive_resolution,
            split_density_ratio: c.split_density_ratio,
            merge_density_ratio: c.merge_density_ratio,
            boundary_particles: c.boundary_particles,
            viscosity: c.viscosity,
            drag: c.drag,
            wall_friction: c.wall_friction,
            max_velocity: c.max_velocity,
            particle_radius: c.particle_radius,
            enabled: c.enabled,
            phases: phases_config_3(c),
        },
        particles: f
            .particles
            .iter()
            .map(|p| FluidParticle3Dto {
                position: v3(p.position),
                velocity: v3(p.velocity),
                mass: p.mass,
                phase: p.phase,
            })
            .collect(),
    }
}

fn shape2_dto(s: &Shape2) -> Shape2Dto {
    use super::body::Shape2 as S;
    match s {
        S::Circle { radius } => Shape2Dto::Circle { radius: *radius },
        S::Box { half_extents } => Shape2Dto::Box {
            half_extents: v2(*half_extents),
        },
        S::Polygon { vertices } => Shape2Dto::Polygon {
            vertices: vertices.iter().map(|v| v2(*v)).collect(),
        },
    }
}

fn shape3_dto(s: &Shape3) -> Shape3Dto {
    use super::body::Shape3 as S;
    match s {
        S::Sphere { radius } => Shape3Dto::Sphere { radius: *radius },
        S::Box { half_extents } => Shape3Dto::Box {
            half_extents: v3(*half_extents),
        },
        S::Capsule {
            radius,
            half_length,
        } => Shape3Dto::Capsule {
            radius: *radius,
            half_length: *half_length,
        },
        S::ConvexHull { vertices } => Shape3Dto::ConvexHull {
            vertices: vertices.iter().map(|v| v3(*v)).collect(),
        },
        S::ConvexMesh { vertices, indices } => Shape3Dto::ConvexMesh {
            vertices: vertices.iter().map(|v| v3(*v)).collect(),
            indices: indices.clone(),
        },
    }
}

fn solver_dto_2(w: &World2) -> SolverDto {
    SolverDto {
        iterations: w.solver.iterations as u32,
        slop: w.solver.slop,
        baumgarte: w.solver.baumgarte,
        early_exit_tolerance: w.solver.early_exit_tolerance,
        friction: w.solver.enable_friction,
        restitution: w.solver.enable_restitution,
    }
}

fn solver_dto_3(w: &World3) -> SolverDto {
    SolverDto {
        iterations: w.solver.iterations as u32,
        slop: w.solver.slop,
        baumgarte: w.solver.baumgarte,
        early_exit_tolerance: w.solver.early_exit_tolerance,
        friction: w.solver.enable_friction,
        restitution: w.solver.enable_restitution,
    }
}

fn validation_dto(v: ValidationConfig) -> ValidationDto {
    ValidationDto {
        check_finite: v.check_finite,
        max_penetration: v.max_penetration,
        max_mass_ratio: v.max_mass_ratio,
        max_coordinate: v.max_coordinate,
        max_velocity: v.max_velocity,
        max_angular_velocity: v.max_angular_velocity,
        log_warnings: v.log_warnings,
        auto_correct: v.auto_correct,
    }
}

fn gravity_config_dto_2(c: &super::gravity::GravityConfig2) -> GravityConfig2Dto {
    GravityConfig2Dto {
        mode: c.mode as u32,
        integrator: c.integrator as u32,
        big_g: c.big_g,
        softening: c.softening,
        attractor: v2(c.attractor),
        attractor_mu: c.attractor_mu,
        theta: c.theta,
    }
}

fn gravity_config_dto_3(c: &super::gravity::GravityConfig3) -> GravityConfig3Dto {
    GravityConfig3Dto {
        mode: c.mode as u32,
        integrator: c.integrator as u32,
        big_g: c.big_g,
        softening: c.softening,
        attractor: v3(c.attractor),
        attractor_mu: c.attractor_mu,
        theta: c.theta,
    }
}

fn ambient_config_dto_2(c: &super::ambient::AmbientConfig2) -> AmbientConfig2Dto {
    AmbientConfig2Dto {
        spring: Spring2Dto {
            enabled: c.spring.enabled,
            anchor: v2(c.spring.anchor),
            stiffness: c.spring.stiffness,
            damping: c.spring.damping,
            max_displacement: c.spring.max_displacement,
        },
        drag: Drag2Dto {
            enabled: c.drag.enabled,
            linear: c.drag.linear,
            quadratic: c.drag.quadratic,
            air_density: c.drag.air_density,
            angular: c.drag.angular,
            wind: v2(c.drag.wind),
        },
        turbulence: Turbulence2Dto {
            enabled: c.turbulence.enabled,
            amplitude: c.turbulence.amplitude,
            frequency: c.turbulence.frequency,
            time_scale: c.turbulence.time_scale,
            seed: c.turbulence.seed,
            octaves: c.turbulence.octaves,
            divergence_free: c.turbulence.divergence_free,
        },
    }
}

fn ambient_config_dto_3(c: &super::ambient::AmbientConfig3) -> AmbientConfig3Dto {
    AmbientConfig3Dto {
        spring: Spring3Dto {
            enabled: c.spring.enabled,
            anchor: v3(c.spring.anchor),
            stiffness: c.spring.stiffness,
            damping: c.spring.damping,
            max_displacement: c.spring.max_displacement,
        },
        drag: Drag3Dto {
            enabled: c.drag.enabled,
            linear: c.drag.linear,
            quadratic: c.drag.quadratic,
            air_density: c.drag.air_density,
            angular: c.drag.angular,
            wind: v3(c.drag.wind),
        },
        turbulence: Turbulence3Dto {
            enabled: c.turbulence.enabled,
            amplitude: c.turbulence.amplitude,
            frequency: c.turbulence.frequency,
            time_scale: c.turbulence.time_scale,
            seed: c.turbulence.seed,
            octaves: c.turbulence.octaves,
            divergence_free: c.turbulence.divergence_free,
        },
    }
}

/// Validated ambient config from a DTO. All floats finite; the
/// coefficient fields must be non-negative. Anything else is `Corrupt`.
fn ambient_config_from_2(
    c: &AmbientConfig2Dto,
) -> Result<super::ambient::AmbientConfig2, SnapshotError> {
    use super::ambient::{AmbientConfig2, DragConfig2, SpringConfig2, TurbulenceConfig2};
    finite(c.spring.anchor.x, "ambient.spring.anchor.x")?;
    finite(c.spring.anchor.y, "ambient.spring.anchor.y")?;
    finite(c.spring.stiffness, "ambient.spring.stiffness")?;
    finite(c.spring.damping, "ambient.spring.damping")?;
    finite(c.spring.max_displacement, "ambient.spring.max_displacement")?;
    if c.spring.stiffness < 0.0 || c.spring.damping < 0.0 || c.spring.max_displacement < 0.0 {
        return corrupt("ambient.spring coefficients must be >= 0");
    }
    finite(c.drag.linear, "ambient.drag.linear")?;
    finite(c.drag.quadratic, "ambient.drag.quadratic")?;
    finite(c.drag.air_density, "ambient.drag.air_density")?;
    finite(c.drag.angular, "ambient.drag.angular")?;
    finite(c.drag.wind.x, "ambient.drag.wind.x")?;
    finite(c.drag.wind.y, "ambient.drag.wind.y")?;
    if c.drag.linear < 0.0
        || c.drag.quadratic < 0.0
        || c.drag.air_density < 0.0
        || c.drag.angular < 0.0
    {
        return corrupt("ambient.drag coefficients must be >= 0");
    }
    finite(c.turbulence.amplitude, "ambient.turbulence.amplitude")?;
    finite(c.turbulence.frequency, "ambient.turbulence.frequency")?;
    finite(c.turbulence.time_scale, "ambient.turbulence.time_scale")?;
    if c.turbulence.amplitude < 0.0 {
        return corrupt("ambient.turbulence.amplitude must be >= 0");
    }
    Ok(AmbientConfig2 {
        spring: SpringConfig2 {
            enabled: c.spring.enabled,
            anchor: Vec2::new(c.spring.anchor.x, c.spring.anchor.y),
            stiffness: c.spring.stiffness,
            damping: c.spring.damping,
            max_displacement: c.spring.max_displacement,
        },
        drag: DragConfig2 {
            enabled: c.drag.enabled,
            linear: c.drag.linear,
            quadratic: c.drag.quadratic,
            air_density: c.drag.air_density,
            angular: c.drag.angular,
            wind: Vec2::new(c.drag.wind.x, c.drag.wind.y),
        },
        turbulence: TurbulenceConfig2 {
            enabled: c.turbulence.enabled,
            amplitude: c.turbulence.amplitude,
            frequency: c.turbulence.frequency,
            time_scale: c.turbulence.time_scale,
            seed: c.turbulence.seed,
            octaves: c.turbulence.octaves,
            divergence_free: c.turbulence.divergence_free,
        },
    })
}

/// 3D counterpart of [`ambient_config_from_2`].
fn ambient_config_from_3(
    c: &AmbientConfig3Dto,
) -> Result<super::ambient::AmbientConfig3, SnapshotError> {
    use super::ambient::{AmbientConfig3, DragConfig3, SpringConfig3, TurbulenceConfig3};
    finite(c.spring.anchor.x, "ambient.spring.anchor.x")?;
    finite(c.spring.anchor.y, "ambient.spring.anchor.y")?;
    finite(c.spring.anchor.z, "ambient.spring.anchor.z")?;
    finite(c.spring.stiffness, "ambient.spring.stiffness")?;
    finite(c.spring.damping, "ambient.spring.damping")?;
    finite(c.spring.max_displacement, "ambient.spring.max_displacement")?;
    if c.spring.stiffness < 0.0 || c.spring.damping < 0.0 || c.spring.max_displacement < 0.0 {
        return corrupt("ambient.spring coefficients must be >= 0");
    }
    finite(c.drag.linear, "ambient.drag.linear")?;
    finite(c.drag.quadratic, "ambient.drag.quadratic")?;
    finite(c.drag.air_density, "ambient.drag.air_density")?;
    finite(c.drag.angular, "ambient.drag.angular")?;
    finite(c.drag.wind.x, "ambient.drag.wind.x")?;
    finite(c.drag.wind.y, "ambient.drag.wind.y")?;
    finite(c.drag.wind.z, "ambient.drag.wind.z")?;
    if c.drag.linear < 0.0
        || c.drag.quadratic < 0.0
        || c.drag.air_density < 0.0
        || c.drag.angular < 0.0
    {
        return corrupt("ambient.drag coefficients must be >= 0");
    }
    finite(c.turbulence.amplitude, "ambient.turbulence.amplitude")?;
    finite(c.turbulence.frequency, "ambient.turbulence.frequency")?;
    finite(c.turbulence.time_scale, "ambient.turbulence.time_scale")?;
    if c.turbulence.amplitude < 0.0 {
        return corrupt("ambient.turbulence.amplitude must be >= 0");
    }
    Ok(AmbientConfig3 {
        spring: SpringConfig3 {
            enabled: c.spring.enabled,
            anchor: Vec3::new(c.spring.anchor.x, c.spring.anchor.y, c.spring.anchor.z),
            stiffness: c.spring.stiffness,
            damping: c.spring.damping,
            max_displacement: c.spring.max_displacement,
        },
        drag: DragConfig3 {
            enabled: c.drag.enabled,
            linear: c.drag.linear,
            quadratic: c.drag.quadratic,
            air_density: c.drag.air_density,
            angular: c.drag.angular,
            wind: Vec3::new(c.drag.wind.x, c.drag.wind.y, c.drag.wind.z),
        },
        turbulence: TurbulenceConfig3 {
            enabled: c.turbulence.enabled,
            amplitude: c.turbulence.amplitude,
            frequency: c.turbulence.frequency,
            time_scale: c.turbulence.time_scale,
            seed: c.turbulence.seed,
            octaves: c.turbulence.octaves,
            divergence_free: c.turbulence.divergence_free,
        },
    })
}

/// Validated gravity config from a DTO: enum tags must be known,
/// floats finite, softening non-negative. Anything else is `Corrupt`,
/// never a silent default (the serde `Default` only fills fields
/// *absent* from older files, which are default by construction).
fn gravity_config_from_2(
    c: &GravityConfig2Dto,
) -> Result<super::gravity::GravityConfig2, SnapshotError> {
    use super::gravity::{GravityConfig2, GravityMode, Integrator};
    let mode = match c.mode {
        0 => GravityMode::Uniform,
        1 => GravityMode::Central,
        2 => GravityMode::NBody,
        3 => GravityMode::BarnesHut,
        _ => return corrupt(format!("unknown gravity mode {}", c.mode)),
    };
    let integrator = match c.integrator {
        0 => Integrator::SemiImplicitEuler,
        1 => Integrator::VelocityVerlet,
        _ => return corrupt(format!("unknown integrator {}", c.integrator)),
    };
    finite(c.big_g, "gravity.big_g")?;
    finite(c.softening, "gravity.softening")?;
    if c.softening < 0.0 {
        return corrupt("gravity.softening must be >= 0");
    }
    finite(c.attractor.x, "gravity.attractor.x")?;
    finite(c.attractor.y, "gravity.attractor.y")?;
    finite(c.attractor_mu, "gravity.attractor_mu")?;
    finite(c.theta, "gravity.theta")?;
    if c.theta < 0.0 {
        return corrupt("gravity.theta must be >= 0");
    }
    Ok(GravityConfig2 {
        mode,
        integrator,
        big_g: c.big_g,
        softening: c.softening,
        attractor: Vec2::new(c.attractor.x, c.attractor.y),
        attractor_mu: c.attractor_mu,
        theta: c.theta,
    })
}

/// 3D counterpart of [`gravity_config_from_2`].
fn gravity_config_from_3(
    c: &GravityConfig3Dto,
) -> Result<super::gravity::GravityConfig3, SnapshotError> {
    use super::gravity::{GravityConfig3, GravityMode, Integrator};
    let mode = match c.mode {
        0 => GravityMode::Uniform,
        1 => GravityMode::Central,
        2 => GravityMode::NBody,
        3 => GravityMode::BarnesHut,
        _ => return corrupt(format!("unknown gravity mode {}", c.mode)),
    };
    let integrator = match c.integrator {
        0 => Integrator::SemiImplicitEuler,
        1 => Integrator::VelocityVerlet,
        _ => return corrupt(format!("unknown integrator {}", c.integrator)),
    };
    finite(c.big_g, "gravity.big_g")?;
    finite(c.softening, "gravity.softening")?;
    if c.softening < 0.0 {
        return corrupt("gravity.softening must be >= 0");
    }
    finite(c.attractor.x, "gravity.attractor.x")?;
    finite(c.attractor.y, "gravity.attractor.y")?;
    finite(c.attractor.z, "gravity.attractor.z")?;
    finite(c.attractor_mu, "gravity.attractor_mu")?;
    finite(c.theta, "gravity.theta")?;
    if c.theta < 0.0 {
        return corrupt("gravity.theta must be >= 0");
    }
    Ok(GravityConfig3 {
        mode,
        integrator,
        big_g: c.big_g,
        softening: c.softening,
        attractor: Vec3::new(c.attractor.x, c.attractor.y, c.attractor.z),
        attractor_mu: c.attractor_mu,
        theta: c.theta,
    })
}

fn body2_dto(b: &RigidBody2) -> Body2Dto {
    let r = b.transform.rotation;
    Body2Dto {
        id: b.id as u32,
        is_static: b.is_static,
        is_sleeping: b.is_sleeping,
        linear_damping: b.linear_damping,
        angular_damping: b.angular_damping,
        material: mat_dto(b.material),
        shape: b.shape.as_ref().map(shape2_dto),
        position: v2(b.transform.position),
        rotation: [r.x_axis.x, r.x_axis.y, r.y_axis.x, r.y_axis.y],
        velocity: v2(b.velocity),
        angular_velocity: b.angular_velocity,
        gravity_scale: b.gravity_scale,
        ambient_scale: b.ambient_scale,
        temperature: b.temperature,
    }
}

fn body3_dto(b: &RigidBody3) -> Body3Dto {
    let q = b.transform.rotation;
    Body3Dto {
        id: b.id as u32,
        is_static: b.is_static,
        is_sleeping: b.is_sleeping,
        linear_damping: b.linear_damping,
        angular_damping: b.angular_damping,
        material: mat_dto(b.material),
        shape: b.shape.as_ref().map(shape3_dto),
        position: v3(b.transform.position),
        rotation: [q.x, q.y, q.z, q.w],
        velocity: v3(b.velocity),
        angular_velocity: v3(b.angular_velocity),
        gravity_scale: b.gravity_scale,
        ambient_scale: b.ambient_scale,
        temperature: b.temperature,
    }
}

fn joint_dto(j: &Joint2) -> Joint2Dto {
    match j {
        Joint2::Distance(d) => Joint2Dto::Distance {
            id: d.id as u32,
            body_a: d.body_a as u32,
            body_b: d.body_b as u32,
            anchor_a: v2(d.anchor_a),
            anchor_b: v2(d.anchor_b),
            rest_length: d.rest_length,
            impulse: d.impulse,
        },
        Joint2::Hinge(h) => Joint2Dto::Hinge {
            id: h.id as u32,
            body_a: h.body_a as u32,
            body_b: h.body_b as u32,
            anchor_a: v2(h.anchor_a),
            anchor_b: v2(h.anchor_b),
            limit_min: h.limit_min,
            limit_max: h.limit_max,
            impulse: v2(h.impulse),
            angular_impulse: h.angular_impulse,
        },
        Joint2::Prismatic(p) => Joint2Dto::Prismatic {
            id: p.id as u32,
            body_a: p.body_a as u32,
            body_b: p.body_b as u32,
            anchor_a: v2(p.anchor_a),
            anchor_b: v2(p.anchor_b),
            axis_a: v2(p.axis_a),
            limit_min: p.limit_min,
            limit_max: p.limit_max,
            ref_angle: p.ref_angle,
            perp_impulse: p.perp_impulse,
            angular_impulse: p.angular_impulse,
            limit_impulse: p.limit_impulse,
        },
        Joint2::Fixed(f) => Joint2Dto::Fixed {
            id: f.id as u32,
            body_a: f.body_a as u32,
            body_b: f.body_b as u32,
            anchor_a: v2(f.anchor_a),
            anchor_b: v2(f.anchor_b),
            ref_angle: f.ref_angle,
            impulse: v2(f.impulse),
            angular_impulse: f.angular_impulse,
        },
    }
}

fn world2_dto(w: &World2, mode: LoadMode) -> World2Dto {
    let mut contacts: Vec<Contact2Dto> = w
        .contact_cache
        .iter()
        .map(|((a, b), entries)| Contact2Dto {
            body_a: *a as u32,
            body_b: *b as u32,
            entries: entries
                .iter()
                .map(|e| ContactEntry2Dto {
                    point: v2(e.point),
                    normal: v2(e.normal),
                    normal_impulse: e.normal_impulse,
                    tangent_impulse: e.tangent_impulse,
                })
                .collect(),
        })
        .collect();
    // Canonical order for deterministic bytes.
    contacts.sort_by_key(|c| (c.body_a, c.body_b));

    let (joints, contacts) = match mode {
        LoadMode::Exact => (w.joints.iter().map(joint_dto).collect(), contacts),
        LoadMode::Fresh => {
            // Cold solver state: impulses zeroed, latched angles cleared.
            let joints = w
                .joints
                .iter()
                .map(|j| match j {
                    Joint2::Distance(d) => Joint2Dto::Distance {
                        id: d.id as u32,
                        body_a: d.body_a as u32,
                        body_b: d.body_b as u32,
                        anchor_a: v2(d.anchor_a),
                        anchor_b: v2(d.anchor_b),
                        rest_length: d.rest_length,
                        impulse: 0.0,
                    },
                    Joint2::Hinge(h) => Joint2Dto::Hinge {
                        id: h.id as u32,
                        body_a: h.body_a as u32,
                        body_b: h.body_b as u32,
                        anchor_a: v2(h.anchor_a),
                        anchor_b: v2(h.anchor_b),
                        limit_min: h.limit_min,
                        limit_max: h.limit_max,
                        impulse: v2(Vec2::ZERO),
                        angular_impulse: 0.0,
                    },
                    Joint2::Prismatic(p) => Joint2Dto::Prismatic {
                        id: p.id as u32,
                        body_a: p.body_a as u32,
                        body_b: p.body_b as u32,
                        anchor_a: v2(p.anchor_a),
                        anchor_b: v2(p.anchor_b),
                        axis_a: v2(p.axis_a),
                        limit_min: p.limit_min,
                        limit_max: p.limit_max,
                        ref_angle: None,
                        perp_impulse: 0.0,
                        angular_impulse: 0.0,
                        limit_impulse: 0.0,
                    },
                    Joint2::Fixed(f) => Joint2Dto::Fixed {
                        id: f.id as u32,
                        body_a: f.body_a as u32,
                        body_b: f.body_b as u32,
                        anchor_a: v2(f.anchor_a),
                        anchor_b: v2(f.anchor_b),
                        ref_angle: None,
                        impulse: v2(Vec2::ZERO),
                        angular_impulse: 0.0,
                    },
                })
                .collect();
            (joints, Vec::new())
        }
    };

    // Conditional schema: default worlds stay v3 (byte-identical to
    // pre-gravity files); gravity-feature worlds use v4; anything using
    // ambient features (or a non-unit ambient scale) uses v5; anything
    // with a non-default CCD config uses v6; anything with non-default
    // solver flags uses v7; anything with thermal active uses v8.
    let has_ambient = w.ambient.is_active()
        || w.bodies.iter().any(|b| b.ambient_scale != 1.0)
        || w.fluid.config.gamma != 1.0
        || w.fluid.config.artificial_viscosity != 0.0
        || w.fluid.config.surface_tension != 0.0
        || w.fluid.config.cfl != 0.0
        || w.fluid.config.max_substeps != 1
        || w.fluid.config.adaptive_resolution
        || w.fluid.config.wall_friction != 0.0
        || w.fluid.config.phase_count > 1
        || w.fluid.config.boundary_particles;
    let has_ccd = !w.ccd.is_default();
    let has_solver_flags = !w.solver.enable_friction || !w.solver.enable_restitution;
    let has_gravity =
        !w.gravity_config.is_default() || w.bodies.iter().any(|b| b.gravity_scale != 1.0);
    let has_thermal = w.thermal.config.is_active();
    let has_broadphase_mode = w.broadphase_mode != phieng_broadphase::BroadphaseMode::SpatialHash;
    // Schema v9: any non-default collision-tier config needs the v9
    // block (not just `enabled`): a disabled config with an explicit
    // ceiling is still a user choice and must round-trip exactly.
    let has_collision_tier = w.collision_tier.enabled
        || w.collision_tier.max_tier != super::collision_tier::CollisionTier::Tier5_GJK_EPA;
    let schema = if has_thermal || has_broadphase_mode || has_collision_tier {
        SNAPSHOT_SCHEMA_VERSION
    } else if has_solver_flags {
        SNAPSHOT_SCHEMA_SOLVER_FLAGS
    } else if has_ccd {
        SNAPSHOT_SCHEMA_CCD
    } else if has_ambient {
        SNAPSHOT_SCHEMA_AMBIENT
    } else if has_gravity {
        SNAPSHOT_SCHEMA_GRAVITY
    } else {
        SNAPSHOT_SCHEMA_MIN_READABLE
    };

    World2Dto {
        schema,
        gravity: v2(w.gravity),
        gravity_config: gravity_config_dto_2(&w.gravity_config),
        ambient: ambient_config_dto_2(&w.ambient),
        time: if has_ambient { w.time } else { 0.0 },
        ccd: ccd_config_dto(&w.ccd),
        thermal: thermal_config_dto(&w.thermal.config),
        broadphase_mode: broadphase_mode_discriminant(&w.broadphase_mode),
        collision_tier: collision_tier_dto(&w.collision_tier),
        dt: w.dt,
        linear_sleep_threshold: w.linear_sleep_threshold,
        angular_sleep_threshold: w.angular_sleep_threshold,
        solver: solver_dto_2(w),
        validation: validation_dto(w.validation),
        next_body_id: w.next_body_id as u32,
        next_joint_id: w.next_joint_id as u32,
        cell_size: w.broadphase.cell_size(),
        bodies: w.bodies.iter().map(body2_dto).collect(),
        joints,
        contacts,
        fluid: fluid2_dto(&w.fluid),
    }
}

fn quat_arr(q: Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

fn joint3_dto(j: &Joint3) -> Joint3Dto {
    match j {
        Joint3::Distance(d) => Joint3Dto::Distance3 {
            id: d.id as u32,
            body_a: d.body_a as u32,
            body_b: d.body_b as u32,
            anchor_a: v3(d.anchor_a),
            anchor_b: v3(d.anchor_b),
            rest_length: d.rest_length,
            impulse: d.impulse,
        },
        Joint3::Hinge(h) => Joint3Dto::Hinge3 {
            id: h.id as u32,
            body_a: h.body_a as u32,
            body_b: h.body_b as u32,
            anchor_a: v3(h.anchor_a),
            anchor_b: v3(h.anchor_b),
            axis_a: v3(h.axis_a),
            axis_b: v3(h.axis_b),
            limit_min: h.limit_min,
            limit_max: h.limit_max,
            impulse: v3(h.impulse),
            align_u1: h.align_impulse.x,
            align_u2: h.align_impulse.y,
            twist_impulse: h.twist_impulse,
        },
        Joint3::Prismatic(p) => Joint3Dto::Prismatic3 {
            id: p.id as u32,
            body_a: p.body_a as u32,
            body_b: p.body_b as u32,
            anchor_a: v3(p.anchor_a),
            anchor_b: v3(p.anchor_b),
            axis_a: v3(p.axis_a),
            limit_min: p.limit_min,
            limit_max: p.limit_max,
            ref_orientation: p.ref_orientation.map(quat_arr),
            perp_u1: p.perp_impulse.x,
            perp_u2: p.perp_impulse.y,
            angular_impulse: v3(p.angular_impulse),
            limit_impulse: p.limit_impulse,
        },
        Joint3::Fixed(f) => Joint3Dto::Fixed3 {
            id: f.id as u32,
            body_a: f.body_a as u32,
            body_b: f.body_b as u32,
            anchor_a: v3(f.anchor_a),
            anchor_b: v3(f.anchor_b),
            ref_orientation: f.ref_orientation.map(quat_arr),
            impulse: v3(f.impulse),
            angular_impulse: v3(f.angular_impulse),
        },
    }
}

fn world3_dto(w: &World3, mode: LoadMode) -> World3Dto {
    let mut contacts: Vec<Contact3Dto> = w
        .contact_cache
        .iter()
        .map(|((a, b), entries)| Contact3Dto {
            body_a: *a as u32,
            body_b: *b as u32,
            entries: entries
                .iter()
                .map(|e| ContactEntry3Dto {
                    point: v3(e.point),
                    normal: v3(e.normal),
                    normal_impulse: e.normal_impulse,
                    tangent1_impulse: e.tangent1_impulse,
                    tangent2_impulse: e.tangent2_impulse,
                })
                .collect(),
        })
        .collect();
    contacts.sort_by_key(|c| (c.body_a, c.body_b));
    let (joints, contacts) = match mode {
        LoadMode::Exact => (w.joints.iter().map(joint3_dto).collect(), contacts),
        LoadMode::Fresh => {
            // Cold solver state: impulses zeroed, latched orientations cleared.
            let joints = w
                .joints
                .iter()
                .map(|j| match j {
                    Joint3::Distance(d) => Joint3Dto::Distance3 {
                        id: d.id as u32,
                        body_a: d.body_a as u32,
                        body_b: d.body_b as u32,
                        anchor_a: v3(d.anchor_a),
                        anchor_b: v3(d.anchor_b),
                        rest_length: d.rest_length,
                        impulse: 0.0,
                    },
                    Joint3::Hinge(h) => Joint3Dto::Hinge3 {
                        id: h.id as u32,
                        body_a: h.body_a as u32,
                        body_b: h.body_b as u32,
                        anchor_a: v3(h.anchor_a),
                        anchor_b: v3(h.anchor_b),
                        axis_a: v3(h.axis_a),
                        axis_b: v3(h.axis_b),
                        limit_min: h.limit_min,
                        limit_max: h.limit_max,
                        impulse: v3(Vec3::ZERO),
                        align_u1: 0.0,
                        align_u2: 0.0,
                        twist_impulse: 0.0,
                    },
                    Joint3::Prismatic(p) => Joint3Dto::Prismatic3 {
                        id: p.id as u32,
                        body_a: p.body_a as u32,
                        body_b: p.body_b as u32,
                        anchor_a: v3(p.anchor_a),
                        anchor_b: v3(p.anchor_b),
                        axis_a: v3(p.axis_a),
                        limit_min: p.limit_min,
                        limit_max: p.limit_max,
                        ref_orientation: None,
                        perp_u1: 0.0,
                        perp_u2: 0.0,
                        angular_impulse: v3(Vec3::ZERO),
                        limit_impulse: 0.0,
                    },
                    Joint3::Fixed(f) => Joint3Dto::Fixed3 {
                        id: f.id as u32,
                        body_a: f.body_a as u32,
                        body_b: f.body_b as u32,
                        anchor_a: v3(f.anchor_a),
                        anchor_b: v3(f.anchor_b),
                        ref_orientation: None,
                        impulse: v3(Vec3::ZERO),
                        angular_impulse: v3(Vec3::ZERO),
                    },
                })
                .collect();
            (joints, Vec::new())
        }
    };

    // Conditional schema: see `world2_dto`.
    let has_ambient = w.ambient.is_active()
        || w.bodies.iter().any(|b| b.ambient_scale != 1.0)
        || w.fluid.config.gamma != 1.0
        || w.fluid.config.artificial_viscosity != 0.0
        || w.fluid.config.surface_tension != 0.0
        || w.fluid.config.cfl != 0.0
        || w.fluid.config.max_substeps != 1
        || w.fluid.config.adaptive_resolution
        || w.fluid.config.wall_friction != 0.0
        || w.fluid.config.phase_count > 1
        || w.fluid.config.boundary_particles;
    let has_ccd = !w.ccd.is_default();
    let has_solver_flags = !w.solver.enable_friction || !w.solver.enable_restitution;
    let has_gravity =
        !w.gravity_config.is_default() || w.bodies.iter().any(|b| b.gravity_scale != 1.0);
    let has_thermal = w.thermal.config.is_active();
    let has_broadphase_mode = w.broadphase_mode != phieng_broadphase::BroadphaseMode::SpatialHash;
    // Schema v9: any non-default collision-tier config needs the v9
    // block (not just `enabled`): a disabled config with an explicit
    // ceiling is still a user choice and must round-trip exactly.
    let has_collision_tier = w.collision_tier.enabled
        || w.collision_tier.max_tier != super::collision_tier::CollisionTier::Tier5_GJK_EPA;
    let schema = if has_thermal || has_broadphase_mode || has_collision_tier {
        SNAPSHOT_SCHEMA_VERSION
    } else if has_solver_flags {
        SNAPSHOT_SCHEMA_SOLVER_FLAGS
    } else if has_ccd {
        SNAPSHOT_SCHEMA_CCD
    } else if has_ambient {
        SNAPSHOT_SCHEMA_AMBIENT
    } else if has_gravity {
        SNAPSHOT_SCHEMA_GRAVITY
    } else {
        SNAPSHOT_SCHEMA_MIN_READABLE
    };

    World3Dto {
        schema,
        gravity: v3(w.gravity),
        gravity_config: gravity_config_dto_3(&w.gravity_config),
        ambient: ambient_config_dto_3(&w.ambient),
        time: if has_ambient { w.time } else { 0.0 },
        ccd: ccd_config_dto(&w.ccd),
        thermal: thermal_config_dto(&w.thermal.config),
        broadphase_mode: broadphase_mode_discriminant(&w.broadphase_mode),
        collision_tier: collision_tier_dto(&w.collision_tier),
        dt: w.dt,
        linear_sleep_threshold: w.linear_sleep_threshold,
        angular_sleep_threshold: w.angular_sleep_threshold,
        solver: solver_dto_3(w),
        validation: validation_dto(w.validation),
        next_body_id: w.next_body_id as u32,
        next_joint_id: w.next_joint_id as u32,
        bodies: w.bodies.iter().map(body3_dto).collect(),
        joints,
        contacts,
        fluid: fluid3_dto(&w.fluid),
    }
}

// ============================================================================
// DTO -> world (load, both modes)
// ============================================================================

fn check_schema(schema: u32) -> Result<(), SnapshotError> {
    if (SNAPSHOT_SCHEMA_MIN_READABLE..=SNAPSHOT_SCHEMA_VERSION).contains(&schema) {
        Ok(())
    } else {
        Err(SnapshotError::Version(schema))
    }
}

fn material_from(m: &MaterialDto) -> Result<PhysicsMaterial, SnapshotError> {
    finite(m.friction, "material.friction")?;
    finite(m.restitution, "material.restitution")?;
    finite(m.density, "material.density")?;
    finite(m.specific_heat, "material.specific_heat")?;
    finite(m.thermal_conductivity, "material.thermal_conductivity")?;
    finite(m.emissivity, "material.emissivity")?;
    // Reuse the sanitizing constructor so load matches creation policy,
    // then apply thermal properties (defaults for pre-v8 files).
    let mut m_out = PhysicsMaterial::new(m.friction, m.restitution, m.density);
    m_out.specific_heat = m.specific_heat;
    m_out.thermal_conductivity = m.thermal_conductivity;
    m_out.emissivity = m.emissivity;
    Ok(m_out)
}

fn shape2_from(s: &Shape2Dto) -> Result<Shape2, SnapshotError> {
    use super::body::Shape2 as S;
    match s {
        Shape2Dto::Circle { radius } => {
            finite(*radius, "circle.radius")?;
            Ok(S::Circle { radius: *radius })
        }
        Shape2Dto::Box { half_extents } => {
            finite(half_extents.x, "box.half_extents.x")?;
            finite(half_extents.y, "box.half_extents.y")?;
            Ok(S::Box {
                half_extents: Vec2::new(half_extents.x, half_extents.y),
            })
        }
        Shape2Dto::Polygon { vertices } => {
            if vertices.len() > 1024 {
                return corrupt("polygon vertex count exceeds 1024");
            }
            let mut vs = Vec::with_capacity(vertices.len());
            for v in vertices {
                finite(v.x, "polygon.vertex.x")?;
                finite(v.y, "polygon.vertex.y")?;
                vs.push(Vec2::new(v.x, v.y));
            }
            Ok(S::Polygon { vertices: vs })
        }
    }
}

fn shape3_from(s: &Shape3Dto) -> Result<Shape3, SnapshotError> {
    use super::body::Shape3 as S;
    match s {
        Shape3Dto::Sphere { radius } => {
            finite(*radius, "sphere.radius")?;
            Ok(S::Sphere { radius: *radius })
        }
        Shape3Dto::Box { half_extents } => {
            finite(half_extents.x, "box.half_extents.x")?;
            finite(half_extents.y, "box.half_extents.y")?;
            finite(half_extents.z, "box.half_extents.z")?;
            Ok(S::Box {
                half_extents: Vec3::new(half_extents.x, half_extents.y, half_extents.z),
            })
        }
        Shape3Dto::Capsule {
            radius,
            half_length,
        } => {
            finite(*radius, "capsule.radius")?;
            finite(*half_length, "capsule.half_length")?;
            Ok(S::Capsule {
                radius: *radius,
                half_length: *half_length,
            })
        }
        Shape3Dto::ConvexHull { vertices } => {
            if vertices.len() > 4096 {
                return corrupt("hull vertex count exceeds 4096");
            }
            let mut vs = Vec::with_capacity(vertices.len());
            for v in vertices {
                finite(v.x, "hull.vertex.x")?;
                finite(v.y, "hull.vertex.y")?;
                finite(v.z, "hull.vertex.z")?;
                vs.push(Vec3::new(v.x, v.y, v.z));
            }
            Ok(S::ConvexHull { vertices: vs })
        }
        Shape3Dto::ConvexMesh { vertices, indices } => {
            if vertices.len() > 4096 {
                return corrupt("mesh vertex count exceeds 4096");
            }
            // 4096 triangles (3 indices each) mirrors the hull vertex cap.
            if indices.len() > 12288 {
                return corrupt("mesh index count exceeds 12288");
            }
            let mut vs = Vec::with_capacity(vertices.len());
            for v in vertices {
                finite(v.x, "mesh.vertex.x")?;
                finite(v.y, "mesh.vertex.y")?;
                finite(v.z, "mesh.vertex.z")?;
                vs.push(Vec3::new(v.x, v.y, v.z));
            }
            // Full topological validation runs in `set_shape` on load.
            Ok(S::ConvexMesh {
                vertices: vs,
                indices: indices.clone(),
            })
        }
    }
}

fn apply_fluid2(w: &mut World2, f: &Fluid2Dto) -> Result<(), SnapshotError> {
    use super::fluid::{FluidConfig2, Particle2, MAX_PARTICLES_2D};
    let c = &f.config;
    finite(c.particle_mass, "fluid.particle_mass")?;
    finite(c.rest_density, "fluid.rest_density")?;
    finite(c.smoothing_length, "fluid.smoothing_length")?;
    finite(c.stiffness, "fluid.stiffness")?;
    finite(c.gamma, "fluid.gamma")?;
    finite(c.artificial_viscosity, "fluid.artificial_viscosity")?;
    finite(c.surface_tension, "fluid.surface_tension")?;
    finite(c.cfl, "fluid.cfl")?;
    finite(c.viscosity, "fluid.viscosity")?;
    finite(c.drag, "fluid.drag")?;
    finite(c.max_velocity, "fluid.max_velocity")?;
    finite(c.particle_radius, "fluid.particle_radius")?;
    if f.particles.len() > MAX_PARTICLES_2D {
        return corrupt("fluid particle count exceeds MAX_PARTICLES_2D");
    }
    w.fluid.config = FluidConfig2 {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
        phase_count: c.phases.as_ref().map(|p| p.count.clamp(1, 4)).unwrap_or(1),
        phase_rest_density: c
            .phases
            .as_ref()
            .map(|p| p.rest_density)
            .unwrap_or([1.0; 4]),
        phase_mass: c
            .phases
            .as_ref()
            .map(|p| p.mass)
            .unwrap_or([c.particle_mass; 4]),
    };
    w.fluid.clear();
    for p in &f.particles {
        finite(p.position.x, "fluid.position.x")?;
        finite(p.position.y, "fluid.position.y")?;
        finite(p.velocity.x, "fluid.velocity.x")?;
        finite(p.velocity.y, "fluid.velocity.y")?;
        // Density/pressure are recomputed deterministically on the next
        // step, so only positions and velocities persist.
        w.fluid.particles.push(Particle2 {
            position: Vec2::new(p.position.x, p.position.y),
            velocity: Vec2::new(p.velocity.x, p.velocity.y),
            density: c.rest_density,
            pressure: 0.0,
            mass: if p.mass > 0.0 {
                p.mass
            } else {
                c.particle_mass.max(0.0)
            },
            phase: p.phase,
        });
    }
    Ok(())
}

fn apply_fluid3(w: &mut World3, f: &Fluid3Dto) -> Result<(), SnapshotError> {
    use super::fluid3::{FluidConfig3, Particle3, MAX_PARTICLES_3D};
    let c = &f.config;
    finite(c.particle_mass, "fluid.particle_mass")?;
    finite(c.rest_density, "fluid.rest_density")?;
    finite(c.smoothing_length, "fluid.smoothing_length")?;
    finite(c.stiffness, "fluid.stiffness")?;
    finite(c.gamma, "fluid.gamma")?;
    finite(c.artificial_viscosity, "fluid.artificial_viscosity")?;
    finite(c.surface_tension, "fluid.surface_tension")?;
    finite(c.cfl, "fluid.cfl")?;
    finite(c.viscosity, "fluid.viscosity")?;
    finite(c.drag, "fluid.drag")?;
    finite(c.max_velocity, "fluid.max_velocity")?;
    finite(c.particle_radius, "fluid.particle_radius")?;
    if f.particles.len() > MAX_PARTICLES_3D {
        return corrupt("fluid particle count exceeds MAX_PARTICLES_3D");
    }
    w.fluid.config = FluidConfig3 {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
        phase_count: c.phases.as_ref().map(|p| p.count.clamp(1, 4)).unwrap_or(1),
        phase_rest_density: c
            .phases
            .as_ref()
            .map(|p| p.rest_density)
            .unwrap_or([1.0; 4]),
        phase_mass: c
            .phases
            .as_ref()
            .map(|p| p.mass)
            .unwrap_or([c.particle_mass; 4]),
    };
    w.fluid.clear();
    for p in &f.particles {
        finite(p.position.x, "fluid.position.x")?;
        finite(p.position.y, "fluid.position.y")?;
        finite(p.position.z, "fluid.position.z")?;
        finite(p.velocity.x, "fluid.velocity.x")?;
        finite(p.velocity.y, "fluid.velocity.y")?;
        finite(p.velocity.z, "fluid.velocity.z")?;
        w.fluid.particles.push(Particle3 {
            position: Vec3::new(p.position.x, p.position.y, p.position.z),
            velocity: Vec3::new(p.velocity.x, p.velocity.y, p.velocity.z),
            density: c.rest_density,
            pressure: 0.0,
            mass: if p.mass > 0.0 {
                p.mass
            } else {
                c.particle_mass.max(0.0)
            },
            phase: p.phase,
        });
    }
    Ok(())
}

fn apply_solver_2(w: &mut World2, s: &SolverDto) -> Result<(), SnapshotError> {
    finite(s.slop, "solver.slop")?;
    finite(s.baumgarte, "solver.baumgarte")?;
    finite(s.early_exit_tolerance, "solver.early_exit_tolerance")?;
    if s.iterations > 1024 {
        return corrupt("solver.iterations exceeds 1024");
    }
    w.solver.iterations = s.iterations as usize;
    w.solver.slop = s.slop;
    w.solver.baumgarte = s.baumgarte;
    w.solver.early_exit_tolerance = s.early_exit_tolerance;
    w.solver.enable_friction = s.friction;
    w.solver.enable_restitution = s.restitution;
    Ok(())
}

fn apply_solver_3(w: &mut World3, s: &SolverDto) -> Result<(), SnapshotError> {
    finite(s.slop, "solver.slop")?;
    finite(s.baumgarte, "solver.baumgarte")?;
    finite(s.early_exit_tolerance, "solver.early_exit_tolerance")?;
    if s.iterations > 1024 {
        return corrupt("solver.iterations exceeds 1024");
    }
    w.solver.iterations = s.iterations as usize;
    w.solver.slop = s.slop;
    w.solver.baumgarte = s.baumgarte;
    w.solver.early_exit_tolerance = s.early_exit_tolerance;
    w.solver.enable_friction = s.friction;
    w.solver.enable_restitution = s.restitution;
    Ok(())
}

fn apply_validation(w: &mut ValidationConfig, v: &ValidationDto) -> Result<(), SnapshotError> {
    finite(v.max_penetration, "validation.max_penetration")?;
    finite(v.max_mass_ratio, "validation.max_mass_ratio")?;
    finite(v.max_coordinate, "validation.max_coordinate")?;
    finite(v.max_velocity, "validation.max_velocity")?;
    finite(v.max_angular_velocity, "validation.max_angular_velocity")?;
    w.check_finite = v.check_finite;
    w.max_penetration = v.max_penetration;
    w.max_mass_ratio = v.max_mass_ratio;
    w.max_coordinate = v.max_coordinate;
    w.max_velocity = v.max_velocity;
    w.max_angular_velocity = v.max_angular_velocity;
    w.log_warnings = v.log_warnings;
    w.auto_correct = v.auto_correct;
    Ok(())
}

fn load_body2(w: &mut World2, b: &Body2Dto) -> Result<(), SnapshotError> {
    finite(b.position.x, "body.position.x")?;
    finite(b.position.y, "body.position.y")?;
    for (i, r) in b.rotation.iter().enumerate() {
        finite(*r, &format!("body.rotation[{i}]"))?;
    }
    finite(b.velocity.x, "body.velocity.x")?;
    finite(b.velocity.y, "body.velocity.y")?;
    finite(b.angular_velocity, "body.angular_velocity")?;
    finite(b.linear_damping, "body.linear_damping")?;
    finite(b.angular_damping, "body.angular_damping")?;
    finite(b.gravity_scale, "body.gravity_scale")?;
    let material = material_from(&b.material)?;

    // id == index invariant: snapshot bodies must arrive in order.
    if b.id as usize != w.bodies.len() {
        return corrupt(format!(
            "body id out of order (got {}, want {})",
            b.id,
            w.bodies.len()
        ));
    }

    let shape = match &b.shape {
        Some(s) => Some(shape2_from(s)?),
        None => None,
    };

    // Rebuild through the validated constructors so mass/inertia match
    // creation exactly; then restore exact dynamic state.
    let mut body = if b.is_static {
        RigidBody2::static_body(w.next_body_id)
    } else {
        RigidBody2::new(w.next_body_id)
    };
    body.transform.position = Vec2::new(b.position.x, b.position.y);
    if let Some(shape) = shape {
        body.set_shape(shape, material.density)?;
        body.set_material(material)?;
    } else {
        // Shapeless bodies carry the material verbatim (mass stays default).
        body.material = material;
    }
    body.transform.rotation = Mat2::from_cols(
        Vec2::new(b.rotation[0], b.rotation[1]),
        Vec2::new(b.rotation[2], b.rotation[3]),
    );
    body.velocity = Vec2::new(b.velocity.x, b.velocity.y);
    body.angular_velocity = b.angular_velocity;
    body.linear_damping = b.linear_damping;
    body.angular_damping = b.angular_damping;
    body.gravity_scale = b.gravity_scale;
    body.ambient_scale = b.ambient_scale;
    body.temperature = b.temperature;
    body.is_sleeping = b.is_sleeping;
    w.add_body(body)?;
    Ok(())
}

fn load_body3(w: &mut World3, b: &Body3Dto) -> Result<(), SnapshotError> {
    finite(b.position.x, "body.position.x")?;
    finite(b.position.y, "body.position.y")?;
    finite(b.position.z, "body.position.z")?;
    for (i, r) in b.rotation.iter().enumerate() {
        finite(*r, &format!("body.rotation[{i}]"))?;
    }
    finite(b.velocity.x, "body.velocity.x")?;
    finite(b.velocity.y, "body.velocity.y")?;
    finite(b.velocity.z, "body.velocity.z")?;
    finite(b.angular_velocity.x, "body.angular_velocity.x")?;
    finite(b.angular_velocity.y, "body.angular_velocity.y")?;
    finite(b.angular_velocity.z, "body.angular_velocity.z")?;
    finite(b.linear_damping, "body.linear_damping")?;
    finite(b.angular_damping, "body.angular_damping")?;
    finite(b.gravity_scale, "body.gravity_scale")?;
    let material = material_from(&b.material)?;

    if b.id as usize != w.bodies.len() {
        return corrupt(format!(
            "body id out of order (got {}, want {})",
            b.id,
            w.bodies.len()
        ));
    }

    let shape = match &b.shape {
        Some(s) => Some(shape3_from(s)?),
        None => None,
    };

    let mut body = if b.is_static {
        RigidBody3::static_body(w.next_body_id)
    } else {
        RigidBody3::new(w.next_body_id)
    };
    body.transform.position = Vec3::new(b.position.x, b.position.y, b.position.z);
    if let Some(shape) = shape {
        body.set_shape(shape, material.density)?;
        body.set_material(material)?;
    } else {
        body.material = material;
    }
    body.transform.rotation =
        Quat::from_xyzw(b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]).normalize();
    body.velocity = Vec3::new(b.velocity.x, b.velocity.y, b.velocity.z);
    body.angular_velocity = Vec3::new(
        b.angular_velocity.x,
        b.angular_velocity.y,
        b.angular_velocity.z,
    );
    body.linear_damping = b.linear_damping;
    body.angular_damping = b.angular_damping;
    body.gravity_scale = b.gravity_scale;
    body.ambient_scale = b.ambient_scale;
    body.temperature = b.temperature;
    body.is_sleeping = b.is_sleeping;
    w.add_body(body)?;
    Ok(())
}

fn load_joint2(w: &mut World2, j: &Joint2Dto, mode: LoadMode) -> Result<(), SnapshotError> {
    use super::joint::Joint2 as J;
    let n = w.bodies.len();
    let fresh = mode == LoadMode::Fresh;
    let joint = match j {
        Joint2Dto::Distance {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            impulse,
        } => {
            let mut j = make_distance_joint(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                Vec2::new(
                    finite(anchor_a.x, "joint.anchor_a.x")?,
                    finite(anchor_a.y, "joint.anchor_a.y")?,
                ),
                Vec2::new(
                    finite(anchor_b.x, "joint.anchor_b.x")?,
                    finite(anchor_b.y, "joint.anchor_b.y")?,
                ),
                finite(*rest_length, "joint.rest_length")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Distance(d) = &mut j {
                    d.impulse = finite(*impulse, "joint.impulse")?;
                }
            }
            j
        }
        Joint2Dto::Hinge {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            limit_min,
            limit_max,
            impulse,
            angular_impulse,
        } => {
            let mut j = make_hinge_joint(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                Vec2::new(
                    finite(anchor_a.x, "joint.anchor_a.x")?,
                    finite(anchor_a.y, "joint.anchor_a.y")?,
                ),
                Vec2::new(
                    finite(anchor_b.x, "joint.anchor_b.x")?,
                    finite(anchor_b.y, "joint.anchor_b.y")?,
                ),
                finite(*limit_min, "joint.limit_min")?,
                finite(*limit_max, "joint.limit_max")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Hinge(h) = &mut j {
                    h.impulse = Vec2::new(
                        finite(impulse.x, "joint.impulse.x")?,
                        finite(impulse.y, "joint.impulse.y")?,
                    );
                    h.angular_impulse = finite(*angular_impulse, "joint.angular_impulse")?;
                }
            }
            j
        }
        Joint2Dto::Prismatic {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            limit_min,
            limit_max,
            ref_angle,
            perp_impulse,
            angular_impulse,
            limit_impulse,
        } => {
            let mut j = make_prismatic_joint(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                Vec2::new(
                    finite(anchor_a.x, "joint.anchor_a.x")?,
                    finite(anchor_a.y, "joint.anchor_a.y")?,
                ),
                Vec2::new(
                    finite(anchor_b.x, "joint.anchor_b.x")?,
                    finite(anchor_b.y, "joint.anchor_b.y")?,
                ),
                Vec2::new(
                    finite(axis_a.x, "joint.axis.x")?,
                    finite(axis_a.y, "joint.axis.y")?,
                ),
                finite(*limit_min, "joint.limit_min")?,
                finite(*limit_max, "joint.limit_max")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Prismatic(p) = &mut j {
                    p.ref_angle = match ref_angle {
                        Some(a) => Some(finite(*a, "joint.ref_angle")?),
                        None => None,
                    };
                    p.perp_impulse = finite(*perp_impulse, "joint.perp_impulse")?;
                    p.angular_impulse = finite(*angular_impulse, "joint.angular_impulse")?;
                    p.limit_impulse = finite(*limit_impulse, "joint.limit_impulse")?;
                }
            }
            j
        }
        Joint2Dto::Fixed {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            ref_angle,
            impulse,
            angular_impulse,
        } => {
            let mut j = make_fixed_joint(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                Vec2::new(
                    finite(anchor_a.x, "joint.anchor_a.x")?,
                    finite(anchor_a.y, "joint.anchor_a.y")?,
                ),
                Vec2::new(
                    finite(anchor_b.x, "joint.anchor_b.x")?,
                    finite(anchor_b.y, "joint.anchor_b.y")?,
                ),
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Fixed(f) = &mut j {
                    f.ref_angle = match ref_angle {
                        Some(a) => Some(finite(*a, "joint.ref_angle")?),
                        None => None,
                    };
                    f.impulse = Vec2::new(
                        finite(impulse.x, "joint.impulse.x")?,
                        finite(impulse.y, "joint.impulse.y")?,
                    );
                    f.angular_impulse = finite(*angular_impulse, "joint.angular_impulse")?;
                }
            }
            j
        }
    };
    // Joint ids must arrive in order (id == index invariant).
    if joint.id() != w.joints.len() {
        return corrupt(format!(
            "joint id out of order (got {}, want {})",
            joint.id(),
            w.joints.len()
        ));
    }
    w.add_joint(joint)?;
    Ok(())
}

fn v3_from(d: &Vec3Dto, what: &str) -> Result<Vec3, SnapshotError> {
    Ok(Vec3::new(
        finite(d.x, &format!("{what}.x"))?,
        finite(d.y, &format!("{what}.y"))?,
        finite(d.z, &format!("{what}.z"))?,
    ))
}

fn quat_from(a: &[f32; 4], what: &str) -> Result<Quat, SnapshotError> {
    let q = Quat::from_xyzw(
        finite(a[0], &format!("{what}.x"))?,
        finite(a[1], &format!("{what}.y"))?,
        finite(a[2], &format!("{what}.z"))?,
        finite(a[3], &format!("{what}.w"))?,
    );
    if q.length_squared() <= 1e-12 {
        return corrupt(format!("{what} is a zero quaternion"));
    }
    Ok(q.normalize())
}

fn load_joint3(w: &mut World3, j: &Joint3Dto, mode: LoadMode) -> Result<(), SnapshotError> {
    use super::joint3::Joint3 as J;
    let n = w.bodies.len();
    let fresh = mode == LoadMode::Fresh;
    let joint = match j {
        Joint3Dto::Distance3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            impulse,
        } => {
            let mut j = make_distance_joint_3(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                v3_from(anchor_a, "joint.anchor_a")?,
                v3_from(anchor_b, "joint.anchor_b")?,
                finite(*rest_length, "joint.rest_length")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Distance(d) = &mut j {
                    d.impulse = finite(*impulse, "joint.impulse")?;
                }
            }
            j
        }
        Joint3Dto::Hinge3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            limit_min,
            limit_max,
            impulse,
            align_u1,
            align_u2,
            twist_impulse,
        } => {
            let mut j = make_hinge_joint_3(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                v3_from(anchor_a, "joint.anchor_a")?,
                v3_from(anchor_b, "joint.anchor_b")?,
                v3_from(axis_a, "joint.axis_a")?,
                v3_from(axis_b, "joint.axis_b")?,
                finite(*limit_min, "joint.limit_min")?,
                finite(*limit_max, "joint.limit_max")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Hinge(h) = &mut j {
                    h.impulse = v3_from(impulse, "joint.impulse")?;
                    h.align_impulse = Vec2Pair {
                        x: finite(*align_u1, "joint.align_u1")?,
                        y: finite(*align_u2, "joint.align_u2")?,
                    };
                    h.twist_impulse = finite(*twist_impulse, "joint.twist_impulse")?;
                }
            }
            j
        }
        Joint3Dto::Prismatic3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            limit_min,
            limit_max,
            ref_orientation,
            perp_u1,
            perp_u2,
            angular_impulse,
            limit_impulse,
        } => {
            let mut j = make_prismatic_joint_3(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                v3_from(anchor_a, "joint.anchor_a")?,
                v3_from(anchor_b, "joint.anchor_b")?,
                v3_from(axis_a, "joint.axis_a")?,
                finite(*limit_min, "joint.limit_min")?,
                finite(*limit_max, "joint.limit_max")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Prismatic(p) = &mut j {
                    p.ref_orientation = match ref_orientation {
                        Some(a) => Some(quat_from(a, "joint.ref_orientation")?),
                        None => None,
                    };
                    p.perp_impulse = Vec2Pair {
                        x: finite(*perp_u1, "joint.perp_u1")?,
                        y: finite(*perp_u2, "joint.perp_u2")?,
                    };
                    p.angular_impulse = v3_from(angular_impulse, "joint.angular_impulse")?;
                    p.limit_impulse = finite(*limit_impulse, "joint.limit_impulse")?;
                }
            }
            j
        }
        Joint3Dto::Fixed3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            ref_orientation,
            impulse,
            angular_impulse,
        } => {
            let mut j = make_fixed_joint_3(
                *id as usize,
                *body_a as usize,
                *body_b as usize,
                v3_from(anchor_a, "joint.anchor_a")?,
                v3_from(anchor_b, "joint.anchor_b")?,
                0.0,
                n,
            )?;
            if !fresh {
                if let J::Fixed(f) = &mut j {
                    f.ref_orientation = match ref_orientation {
                        Some(a) => Some(quat_from(a, "joint.ref_orientation")?),
                        None => None,
                    };
                    f.impulse = v3_from(impulse, "joint.impulse")?;
                    f.angular_impulse = v3_from(angular_impulse, "joint.angular_impulse")?;
                }
            }
            j
        }
    };
    // Joint ids must arrive in order (id == index invariant).
    if joint.id() != w.joints.len() {
        return corrupt(format!(
            "joint id out of order (got {}, want {})",
            joint.id(),
            w.joints.len()
        ));
    }
    w.add_joint(joint)?;
    Ok(())
}

fn load_contact2(w: &mut World2, c: &Contact2Dto, mode: LoadMode) -> Result<(), SnapshotError> {
    if mode == LoadMode::Fresh {
        return Ok(());
    }
    let (a, b) = (c.body_a as usize, c.body_b as usize);
    if a >= w.bodies.len() || b >= w.bodies.len() {
        return corrupt("contact pair references unknown body");
    }
    // Keys are stored verbatim (the engine itself does not canonicalize:
    // circle/box dispatch can emit (circle, box) in either index order).
    // Only self-pairs are impossible and rejected.
    if a == b {
        return corrupt("contact pair must reference two distinct bodies");
    }
    if c.entries.len() > 16 {
        return corrupt("contact entry count exceeds 16");
    }
    let mut entries = Vec::with_capacity(c.entries.len());
    for e in &c.entries {
        finite(e.point.x, "contact.point.x")?;
        finite(e.point.y, "contact.point.y")?;
        finite(e.normal.x, "contact.normal.x")?;
        finite(e.normal.y, "contact.normal.y")?;
        finite(e.normal_impulse, "contact.normal_impulse")?;
        finite(e.tangent_impulse, "contact.tangent_impulse")?;
        entries.push(CachedContact {
            point: Vec2::new(e.point.x, e.point.y),
            normal: Vec2::new(e.normal.x, e.normal.y),
            normal_impulse: e.normal_impulse,
            tangent_impulse: e.tangent_impulse,
        });
    }
    w.contact_cache.insert((a, b), entries);
    Ok(())
}

fn load_contact3(w: &mut World3, c: &Contact3Dto, mode: LoadMode) -> Result<(), SnapshotError> {
    if mode == LoadMode::Fresh {
        return Ok(());
    }
    let (a, b) = (c.body_a as usize, c.body_b as usize);
    if a >= w.bodies.len() || b >= w.bodies.len() {
        return corrupt("contact pair references unknown body");
    }
    // Stored verbatim like 2D (see load_contact2): only self-pairs rejected.
    if a == b {
        return corrupt("contact pair must reference two distinct bodies");
    }
    if c.entries.len() > 16 {
        return corrupt("contact entry count exceeds 16");
    }
    let mut entries = Vec::with_capacity(c.entries.len());
    for e in &c.entries {
        finite(e.point.x, "contact.point.x")?;
        finite(e.point.y, "contact.point.y")?;
        finite(e.point.z, "contact.point.z")?;
        finite(e.normal.x, "contact.normal.x")?;
        finite(e.normal.y, "contact.normal.y")?;
        finite(e.normal.z, "contact.normal.z")?;
        finite(e.normal_impulse, "contact.normal_impulse")?;
        finite(e.tangent1_impulse, "contact.tangent1_impulse")?;
        finite(e.tangent2_impulse, "contact.tangent2_impulse")?;
        entries.push(CachedContact3D {
            point: Vec3::new(e.point.x, e.point.y, e.point.z),
            normal: Vec3::new(e.normal.x, e.normal.y, e.normal.z),
            normal_impulse: e.normal_impulse,
            tangent1_impulse: e.tangent1_impulse,
            tangent2_impulse: e.tangent2_impulse,
        });
    }
    w.contact_cache.insert((a, b), entries);
    Ok(())
}

// ============================================================================
// Public JSON API
// ============================================================================

/// Serialize a 2D world to JSON. `mode` selects exact vs fresh content.
pub fn world2_to_json(world: &World2, mode: LoadMode) -> Result<String, SnapshotError> {
    serde_json::to_string_pretty(&world2_dto(world, mode))
        .map_err(|e| SnapshotError::Parse(e.to_string()))
}

/// Deserialize a 2D world from JSON.
pub fn world2_from_json(text: &str, mode: LoadMode) -> Result<World2, SnapshotError> {
    let dto: World2Dto =
        serde_json::from_str(text).map_err(|e| SnapshotError::Parse(e.to_string()))?;
    world2_from_dto(&dto, mode)
}

/// Serialize a 3D world to JSON.
pub fn world3_to_json(world: &World3, mode: LoadMode) -> Result<String, SnapshotError> {
    serde_json::to_string_pretty(&world3_dto(world, mode))
        .map_err(|e| SnapshotError::Parse(e.to_string()))
}

/// Deserialize a 3D world from JSON.
pub fn world3_from_json(text: &str, mode: LoadMode) -> Result<World3, SnapshotError> {
    let dto: World3Dto =
        serde_json::from_str(text).map_err(|e| SnapshotError::Parse(e.to_string()))?;
    world3_from_dto(&dto, mode)
}

fn world2_from_dto(dto: &World2Dto, mode: LoadMode) -> Result<World2, SnapshotError> {
    check_schema(dto.schema)?;
    if dto.bodies.len() > 1_000_000 || dto.joints.len() > 1_000_000 {
        return corrupt("body/joint count exceeds 1,000,000");
    }
    finite(dto.gravity.x, "gravity.x")?;
    finite(dto.gravity.y, "gravity.y")?;
    finite(dto.dt, "dt")?;
    finite(dto.linear_sleep_threshold, "linear_sleep_threshold")?;
    finite(dto.angular_sleep_threshold, "angular_sleep_threshold")?;
    finite(dto.cell_size, "cell_size")?;

    let mut w = World2::new();
    w.gravity = Vec2::new(dto.gravity.x, dto.gravity.y);
    w.gravity_config = gravity_config_from_2(&dto.gravity_config)?;
    finite(dto.time, "time")?;
    w.ambient = ambient_config_from_2(&dto.ambient)?;
    w.time = dto.time;
    finite(dto.ccd.max_movement, "ccd.max_movement")?;
    if dto.ccd.max_movement <= 0.0 {
        return corrupt("ccd.max_movement must be positive");
    }
    if dto.ccd.max_substeps < 1 {
        return corrupt("ccd.max_substeps must be >= 1");
    }
    w.ccd = ccd_config_from_dto(&dto.ccd);
    // Schema v9: collision-tier ceiling (older files default to full
    // fidelity).
    w.collision_tier = collision_tier_from_dto(&dto.collision_tier)?;
    w.dt = dto.dt;
    w.linear_sleep_threshold = dto.linear_sleep_threshold;
    w.angular_sleep_threshold = dto.angular_sleep_threshold;
    apply_solver_2(&mut w, &dto.solver)?;
    apply_validation(&mut w.validation, &dto.validation)?;
    // Schema v8: broad-phase backend tag. Older payloads omit the field
    // (serde default = grid tag) and always load the grid backend.
    let broadphase_mode = broadphase_mode_from_tag(dto.broadphase_mode)?;
    w.broadphase_mode = broadphase_mode;
    w.broadphase = phieng_broadphase::Broadphase2::new(broadphase_mode, dto.cell_size);

    for b in &dto.bodies {
        load_body2(&mut w, b)?;
    }
    if dto.next_body_id as usize != w.bodies.len() {
        return corrupt("next_body_id disagrees with body count");
    }
    for j in &dto.joints {
        load_joint2(&mut w, j, mode)?;
    }
    if dto.next_joint_id as usize != w.joints.len() {
        return corrupt("next_joint_id disagrees with joint count");
    }
    for c in &dto.contacts {
        load_contact2(&mut w, c, mode)?;
    }
    apply_fluid2(&mut w, &dto.fluid)?;
    // Thermal config: v8+ carries the block; older payloads use defaults.
    if dto.schema >= SNAPSHOT_SCHEMA_THERMAL {
        w.thermal.config = thermal_config_from_dto(&dto.thermal);
    }
    Ok(w)
}

fn world3_from_dto(dto: &World3Dto, mode: LoadMode) -> Result<World3, SnapshotError> {
    check_schema(dto.schema)?;
    if dto.bodies.len() > 1_000_000 || dto.joints.len() > 1_000_000 {
        return corrupt("body/joint count exceeds 1,000,000");
    }
    finite(dto.gravity.x, "gravity.x")?;
    finite(dto.gravity.y, "gravity.y")?;
    finite(dto.gravity.z, "gravity.z")?;
    finite(dto.dt, "dt")?;
    finite(dto.linear_sleep_threshold, "linear_sleep_threshold")?;
    finite(dto.angular_sleep_threshold, "angular_sleep_threshold")?;

    let mut w = World3::new();
    w.gravity = Vec3::new(dto.gravity.x, dto.gravity.y, dto.gravity.z);
    w.gravity_config = gravity_config_from_3(&dto.gravity_config)?;
    finite(dto.time, "time")?;
    w.ambient = ambient_config_from_3(&dto.ambient)?;
    w.time = dto.time;
    finite(dto.ccd.max_movement, "ccd.max_movement")?;
    if dto.ccd.max_movement <= 0.0 {
        return corrupt("ccd.max_movement must be positive");
    }
    if dto.ccd.max_substeps < 1 {
        return corrupt("ccd.max_substeps must be >= 1");
    }
    w.ccd = ccd_config_from_dto(&dto.ccd);
    // Schema v9: collision-tier ceiling (older files default to full
    // fidelity).
    w.collision_tier = collision_tier_from_dto(&dto.collision_tier)?;
    w.dt = dto.dt;
    w.linear_sleep_threshold = dto.linear_sleep_threshold;
    w.angular_sleep_threshold = dto.angular_sleep_threshold;
    apply_solver_3(&mut w, &dto.solver)?;
    apply_validation(&mut w.validation, &dto.validation)?;
    // Schema v8: broad-phase backend tag (see `world2_from_dto`).
    let broadphase_mode = broadphase_mode_from_tag(dto.broadphase_mode)?;
    w.broadphase_mode = broadphase_mode;
    w.broadphase = phieng_broadphase::Broadphase3::new(broadphase_mode);

    for b in &dto.bodies {
        load_body3(&mut w, b)?;
    }
    if dto.next_body_id as usize != w.bodies.len() {
        return corrupt("next_body_id disagrees with body count");
    }
    for j in &dto.joints {
        load_joint3(&mut w, j, mode)?;
    }
    if dto.next_joint_id as usize != w.joints.len() {
        return corrupt("next_joint_id disagrees with joint count");
    }
    for c in &dto.contacts {
        load_contact3(&mut w, c, mode)?;
    }
    apply_fluid3(&mut w, &dto.fluid)?;
    // Thermal config: v8+ carries the block; older payloads use defaults.
    if dto.schema >= SNAPSHOT_SCHEMA_THERMAL {
        w.thermal.config = thermal_config_from_dto(&dto.thermal);
    }
    Ok(w)
}

// ============================================================================
// Public binary API (little-endian envelope + CRC32)
// ============================================================================

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn rest(&self) -> usize {
        self.b.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        if self.rest() < n {
            return Err(SnapshotError::Parse("binary payload truncated".into()));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, SnapshotError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_bits(self.read_u32()?))
    }
}

fn encode_envelope(kind: u8, version: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(13 + payload.len());
    out.extend_from_slice(&BINARY_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    out
}

/// Strict envelope check: magic, kind, length, CRC must match, and the
/// schema version must be one this build can read
/// (`[MIN_READABLE, VERSION]`). Returns the file's version with the
/// payload so body/config decoding can branch on it.
fn decode_envelope(bytes: &[u8], want_kind: u8) -> Result<(u32, Vec<u8>), SnapshotError> {
    if bytes.len() < 13 {
        return Err(SnapshotError::Parse("binary snapshot too short".into()));
    }
    if bytes[0..4] != BINARY_MAGIC {
        return Err(SnapshotError::Parse("bad snapshot magic".into()));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if !(SNAPSHOT_SCHEMA_MIN_READABLE..=SNAPSHOT_SCHEMA_VERSION).contains(&version) {
        return Err(SnapshotError::Version(version));
    }
    if bytes[8] != want_kind {
        return Err(SnapshotError::Corrupt("snapshot kind mismatch".into()));
    }
    let len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
    if bytes.len() != 13 + len + 4 {
        return Err(SnapshotError::Parse(
            "binary snapshot length mismatch".into(),
        ));
    }
    let payload = &bytes[13..13 + len];
    let want = u32::from_le_bytes([
        bytes[13 + len],
        bytes[13 + len + 1],
        bytes[13 + len + 2],
        bytes[13 + len + 3],
    ]);
    if crc32fast::hash(payload) != want {
        return Err(SnapshotError::Corrupt("snapshot CRC mismatch".into()));
    }
    Ok((version, payload.to_vec()))
}

fn encode_solver(out: &mut Vec<u8>, s: &SolverDto, version: u32) {
    write_u32(out, s.iterations);
    write_f32(out, s.slop);
    write_f32(out, s.baumgarte);
    write_f32(out, s.early_exit_tolerance);
    // Schema v7+: friction/restitution switches.
    if version >= SNAPSHOT_SCHEMA_SOLVER_FLAGS {
        write_u8(out, s.friction as u8);
        write_u8(out, s.restitution as u8);
    }
}

/// Schema v4 gravity config block (2D): mode/integrator tags,
/// G, softening, attractor, mu. Only written/read for v4 payloads;
/// v3 payloads carry no gravity section (defaults on load).
fn encode_gravity_config_2(out: &mut Vec<u8>, g: &GravityConfig2Dto) {
    write_u32(out, g.mode);
    write_u32(out, g.integrator);
    write_f32(out, g.big_g);
    write_f32(out, g.softening);
    write_f32(out, g.attractor.x);
    write_f32(out, g.attractor.y);
    write_f32(out, g.attractor_mu);
    write_f32(out, g.theta);
}

fn decode_gravity_config_2(r: &mut Reader) -> Result<GravityConfig2Dto, SnapshotError> {
    Ok(GravityConfig2Dto {
        mode: r.read_u32()?,
        integrator: r.read_u32()?,
        big_g: r.read_f32()?,
        softening: r.read_f32()?,
        attractor: Vec2Dto {
            x: r.read_f32()?,
            y: r.read_f32()?,
        },
        attractor_mu: r.read_f32()?,
        theta: r.read_f32()?,
    })
}

/// 3D counterpart of [`encode_gravity_config_2`].
fn encode_gravity_config_3(out: &mut Vec<u8>, g: &GravityConfig3Dto) {
    write_u32(out, g.mode);
    write_u32(out, g.integrator);
    write_f32(out, g.big_g);
    write_f32(out, g.softening);
    write_f32(out, g.attractor.x);
    write_f32(out, g.attractor.y);
    write_f32(out, g.attractor.z);
    write_f32(out, g.attractor_mu);
    write_f32(out, g.theta);
}

fn decode_gravity_config_3(r: &mut Reader) -> Result<GravityConfig3Dto, SnapshotError> {
    Ok(GravityConfig3Dto {
        mode: r.read_u32()?,
        integrator: r.read_u32()?,
        big_g: r.read_f32()?,
        softening: r.read_f32()?,
        attractor: Vec3Dto {
            x: r.read_f32()?,
            y: r.read_f32()?,
            z: r.read_f32()?,
        },
        attractor_mu: r.read_f32()?,
        theta: r.read_f32()?,
    })
}

fn encode_ambient_config_2(out: &mut Vec<u8>, a: &AmbientConfig2Dto) {
    write_u8(out, a.spring.enabled as u8);
    write_f32(out, a.spring.anchor.x);
    write_f32(out, a.spring.anchor.y);
    write_f32(out, a.spring.stiffness);
    write_f32(out, a.spring.damping);
    write_f32(out, a.spring.max_displacement);
    write_u8(out, a.drag.enabled as u8);
    write_f32(out, a.drag.linear);
    write_f32(out, a.drag.quadratic);
    write_f32(out, a.drag.air_density);
    write_f32(out, a.drag.angular);
    write_f32(out, a.drag.wind.x);
    write_f32(out, a.drag.wind.y);
    write_u8(out, a.turbulence.enabled as u8);
    write_f32(out, a.turbulence.amplitude);
    write_f32(out, a.turbulence.frequency);
    write_f32(out, a.turbulence.time_scale);
    write_u32(out, a.turbulence.seed);
    write_u8(out, a.turbulence.octaves);
    write_u8(out, a.turbulence.divergence_free as u8);
}

fn decode_ambient_config_2(r: &mut Reader) -> Result<AmbientConfig2Dto, SnapshotError> {
    Ok(AmbientConfig2Dto {
        spring: Spring2Dto {
            enabled: r.read_u8()? != 0,
            anchor: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
            stiffness: r.read_f32()?,
            damping: r.read_f32()?,
            max_displacement: r.read_f32()?,
        },
        drag: Drag2Dto {
            enabled: r.read_u8()? != 0,
            linear: r.read_f32()?,
            quadratic: r.read_f32()?,
            air_density: r.read_f32()?,
            angular: r.read_f32()?,
            wind: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
        },
        turbulence: Turbulence2Dto {
            enabled: r.read_u8()? != 0,
            amplitude: r.read_f32()?,
            frequency: r.read_f32()?,
            time_scale: r.read_f32()?,
            seed: r.read_u32()?,
            octaves: r.read_u8()?,
            divergence_free: r.read_u8()? != 0,
        },
    })
}

fn encode_ambient_config_3(out: &mut Vec<u8>, a: &AmbientConfig3Dto) {
    write_u8(out, a.spring.enabled as u8);
    write_f32(out, a.spring.anchor.x);
    write_f32(out, a.spring.anchor.y);
    write_f32(out, a.spring.anchor.z);
    write_f32(out, a.spring.stiffness);
    write_f32(out, a.spring.damping);
    write_f32(out, a.spring.max_displacement);
    write_u8(out, a.drag.enabled as u8);
    write_f32(out, a.drag.linear);
    write_f32(out, a.drag.quadratic);
    write_f32(out, a.drag.air_density);
    write_f32(out, a.drag.angular);
    write_f32(out, a.drag.wind.x);
    write_f32(out, a.drag.wind.y);
    write_f32(out, a.drag.wind.z);
    write_u8(out, a.turbulence.enabled as u8);
    write_f32(out, a.turbulence.amplitude);
    write_f32(out, a.turbulence.frequency);
    write_f32(out, a.turbulence.time_scale);
    write_u32(out, a.turbulence.seed);
    write_u8(out, a.turbulence.octaves);
    write_u8(out, a.turbulence.divergence_free as u8);
}

fn decode_ambient_config_3(r: &mut Reader) -> Result<AmbientConfig3Dto, SnapshotError> {
    Ok(AmbientConfig3Dto {
        spring: Spring3Dto {
            enabled: r.read_u8()? != 0,
            anchor: Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            },
            stiffness: r.read_f32()?,
            damping: r.read_f32()?,
            max_displacement: r.read_f32()?,
        },
        drag: Drag3Dto {
            enabled: r.read_u8()? != 0,
            linear: r.read_f32()?,
            quadratic: r.read_f32()?,
            air_density: r.read_f32()?,
            angular: r.read_f32()?,
            wind: Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            },
        },
        turbulence: Turbulence3Dto {
            enabled: r.read_u8()? != 0,
            amplitude: r.read_f32()?,
            frequency: r.read_f32()?,
            time_scale: r.read_f32()?,
            seed: r.read_u32()?,
            octaves: r.read_u8()?,
            divergence_free: r.read_u8()? != 0,
        },
    })
}

fn decode_solver(r: &mut Reader, version: u32) -> Result<SolverDto, SnapshotError> {
    Ok(SolverDto {
        iterations: r.read_u32()?,
        slop: r.read_f32()?,
        baumgarte: r.read_f32()?,
        early_exit_tolerance: r.read_f32()?,
        // Schema v7+: friction/restitution switches.
        friction: if version >= SNAPSHOT_SCHEMA_SOLVER_FLAGS {
            r.read_u8()? != 0
        } else {
            true
        },
        restitution: if version >= SNAPSHOT_SCHEMA_SOLVER_FLAGS {
            r.read_u8()? != 0
        } else {
            true
        },
    })
}

fn encode_validation(out: &mut Vec<u8>, v: &ValidationDto) {
    write_u8(out, v.check_finite as u8);
    write_f32(out, v.max_penetration);
    write_f32(out, v.max_mass_ratio);
    write_f32(out, v.max_coordinate);
    write_f32(out, v.max_velocity);
    write_f32(out, v.max_angular_velocity);
    write_u8(out, v.log_warnings as u8);
    write_u8(out, v.auto_correct as u8);
}

fn decode_validation(r: &mut Reader) -> Result<ValidationDto, SnapshotError> {
    Ok(ValidationDto {
        check_finite: r.read_u8()? != 0,
        max_penetration: r.read_f32()?,
        max_mass_ratio: r.read_f32()?,
        max_coordinate: r.read_f32()?,
        max_velocity: r.read_f32()?,
        max_angular_velocity: r.read_f32()?,
        log_warnings: r.read_u8()? != 0,
        auto_correct: r.read_u8()? != 0,
    })
}

fn encode_shape2(out: &mut Vec<u8>, s: &Shape2Dto) {
    match s {
        Shape2Dto::Circle { radius } => {
            write_u8(out, 0);
            write_f32(out, *radius);
        }
        Shape2Dto::Box { half_extents } => {
            write_u8(out, 1);
            write_f32(out, half_extents.x);
            write_f32(out, half_extents.y);
        }
        Shape2Dto::Polygon { vertices } => {
            write_u8(out, 2);
            write_u32(out, vertices.len() as u32);
            for v in vertices {
                write_f32(out, v.x);
                write_f32(out, v.y);
            }
        }
    }
}

fn decode_shape2(r: &mut Reader) -> Result<Option<Shape2Dto>, SnapshotError> {
    // None is encoded as tag 255 (bodies may be shapeless).
    let tag = r.read_u8()?;
    match tag {
        255 => Ok(None),
        0 => Ok(Some(Shape2Dto::Circle {
            radius: r.read_f32()?,
        })),
        1 => Ok(Some(Shape2Dto::Box {
            half_extents: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
        })),
        2 => {
            let n = r.read_u32()? as usize;
            if n > 1024 {
                return Err(SnapshotError::Parse(
                    "polygon vertex count exceeds 1024".into(),
                ));
            }
            let mut vertices = Vec::with_capacity(n);
            for _ in 0..n {
                vertices.push(Vec2Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                });
            }
            Ok(Some(Shape2Dto::Polygon { vertices }))
        }
        _ => Err(SnapshotError::Parse("unknown 2D shape tag".into())),
    }
}

fn encode_shape3(out: &mut Vec<u8>, s: &Shape3Dto) {
    match s {
        Shape3Dto::Sphere { radius } => {
            write_u8(out, 0);
            write_f32(out, *radius);
        }
        Shape3Dto::Box { half_extents } => {
            write_u8(out, 1);
            write_f32(out, half_extents.x);
            write_f32(out, half_extents.y);
            write_f32(out, half_extents.z);
        }
        Shape3Dto::Capsule {
            radius,
            half_length,
        } => {
            write_u8(out, 2);
            write_f32(out, *radius);
            write_f32(out, *half_length);
        }
        Shape3Dto::ConvexHull { vertices } => {
            write_u8(out, 3);
            write_u32(out, vertices.len() as u32);
            for v in vertices {
                write_f32(out, v.x);
                write_f32(out, v.y);
                write_f32(out, v.z);
            }
        }
        Shape3Dto::ConvexMesh { vertices, indices } => {
            write_u8(out, 4);
            write_u32(out, vertices.len() as u32);
            for v in vertices {
                write_f32(out, v.x);
                write_f32(out, v.y);
                write_f32(out, v.z);
            }
            write_u32(out, (indices.len() / 3) as u32);
            for i in indices {
                write_u32(out, *i);
            }
        }
    }
}

fn decode_shape3(r: &mut Reader) -> Result<Option<Shape3Dto>, SnapshotError> {
    let tag = r.read_u8()?;
    match tag {
        255 => Ok(None),
        0 => Ok(Some(Shape3Dto::Sphere {
            radius: r.read_f32()?,
        })),
        1 => Ok(Some(Shape3Dto::Box {
            half_extents: Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            },
        })),
        2 => Ok(Some(Shape3Dto::Capsule {
            radius: r.read_f32()?,
            half_length: r.read_f32()?,
        })),
        3 => {
            let n = r.read_u32()? as usize;
            if n > 4096 {
                return Err(SnapshotError::Parse(
                    "hull vertex count exceeds 4096".into(),
                ));
            }
            let mut vertices = Vec::with_capacity(n);
            for _ in 0..n {
                vertices.push(Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                });
            }
            Ok(Some(Shape3Dto::ConvexHull { vertices }))
        }
        4 => {
            let n = r.read_u32()? as usize;
            if n > 4096 {
                return Err(SnapshotError::Parse(
                    "mesh vertex count exceeds 4096".into(),
                ));
            }
            let mut vertices = Vec::with_capacity(n);
            for _ in 0..n {
                vertices.push(Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                });
            }
            let t = r.read_u32()? as usize;
            if t > 4096 {
                return Err(SnapshotError::Parse(
                    "mesh triangle count exceeds 4096".into(),
                ));
            }
            let mut indices = Vec::with_capacity(t * 3);
            for _ in 0..t * 3 {
                indices.push(r.read_u32()?);
            }
            Ok(Some(Shape3Dto::ConvexMesh { vertices, indices }))
        }
        _ => Err(SnapshotError::Parse("unknown 3D shape tag".into())),
    }
}

/// Serialize a 2D world to the versioned binary envelope.
pub fn world2_to_binary(world: &World2, mode: LoadMode) -> Result<Vec<u8>, SnapshotError> {
    let dto = world2_dto(world, mode);
    let mut p = Vec::new();
    write_f32(&mut p, dto.gravity.x);
    write_f32(&mut p, dto.gravity.y);
    write_f32(&mut p, dto.dt);
    write_f32(&mut p, dto.linear_sleep_threshold);
    write_f32(&mut p, dto.angular_sleep_threshold);
    encode_solver(&mut p, &dto.solver, dto.schema);
    encode_validation(&mut p, &dto.validation);
    // Schema v4: gravity config block. v3 payloads omit it (strict:
    // a v4-tagged payload without it fails truncation, never misparses).
    let gravity_v4 = dto.schema >= SNAPSHOT_SCHEMA_GRAVITY;
    if gravity_v4 {
        encode_gravity_config_2(&mut p, &dto.gravity_config);
    }
    // Schema v5: ambient config block + clock.
    let ambient_v5 = dto.schema >= SNAPSHOT_SCHEMA_AMBIENT;
    if ambient_v5 {
        encode_ambient_config_2(&mut p, &dto.ambient);
        write_f32(&mut p, dto.time);
    }
    // Schema v6: CCD config block.
    if dto.schema >= SNAPSHOT_SCHEMA_CCD {
        write_u8(&mut p, dto.ccd.enabled as u8);
        write_u8(&mut p, dto.ccd.tier as u8);
        write_f32(&mut p, dto.ccd.max_movement);
        write_u32(&mut p, dto.ccd.max_substeps);
    }
    // Schema v8: thermal config block.
    let thermal_v8 = dto.schema >= SNAPSHOT_SCHEMA_THERMAL;
    if thermal_v8 {
        write_u8(&mut p, dto.thermal.enabled as u8);
        write_f32(&mut p, dto.thermal.ambient_temperature);
        write_f32(&mut p, dto.thermal.stefan_boltzmann);
        write_f32(&mut p, dto.thermal.convection_coefficient);
        write_u8(&mut p, dto.thermal.radiation_enabled as u8);
        write_u8(&mut p, dto.thermal.conduction_enabled as u8);
        write_u8(&mut p, dto.thermal.convection_enabled as u8);
        write_f32(&mut p, dto.thermal.max_temperature);
        write_f32(&mut p, dto.thermal.min_temperature);
        write_f32(&mut p, dto.thermal.radiation_radius_factor);
        write_f32(&mut p, dto.thermal.max_temperature_rate);
        // Schema v8: broad-phase backend tag. Older payloads have neither
        // this nor the thermal block; strictness is free because the whole
        // block is version-gated.
        write_u32(&mut p, dto.broadphase_mode);
    }
    // Schema v9: collision-tier ceiling (version-gated, so v8 payloads
    // keep decoding unchanged).
    if dto.schema >= SNAPSHOT_SCHEMA_COLLISION_TIER {
        write_u8(&mut p, dto.collision_tier.enabled as u8);
        write_u32(&mut p, dto.collision_tier.max_tier);
    }
    write_u32(&mut p, dto.next_body_id);
    write_u32(&mut p, dto.next_joint_id);
    write_f32(&mut p, dto.cell_size);
    write_u32(&mut p, dto.bodies.len() as u32);
    for b in &dto.bodies {
        write_u32(&mut p, b.id);
        write_u8(&mut p, (b.is_static as u8) | ((b.is_sleeping as u8) << 1));
        write_f32(&mut p, b.linear_damping);
        write_f32(&mut p, b.angular_damping);
        write_f32(&mut p, b.material.friction);
        write_f32(&mut p, b.material.restitution);
        write_f32(&mut p, b.material.density);
        // Schema v8: material thermal properties.
        if thermal_v8 {
            write_f32(&mut p, b.material.specific_heat);
            write_f32(&mut p, b.material.thermal_conductivity);
            write_f32(&mut p, b.material.emissivity);
        }
        match &b.shape {
            None => write_u8(&mut p, 255),
            Some(s) => encode_shape2(&mut p, s),
        }
        write_f32(&mut p, b.position.x);
        write_f32(&mut p, b.position.y);
        for r in b.rotation {
            write_f32(&mut p, r);
        }
        write_f32(&mut p, b.velocity.x);
        write_f32(&mut p, b.velocity.y);
        write_f32(&mut p, b.angular_velocity);
        if gravity_v4 {
            write_f32(&mut p, b.gravity_scale);
        }
        if ambient_v5 {
            write_f32(&mut p, b.ambient_scale);
        }
        // Schema v8: per-body temperature.
        if thermal_v8 {
            write_f32(&mut p, b.temperature);
        }
    }
    write_u32(&mut p, dto.joints.len() as u32);
    for j in &dto.joints {
        encode_joint2(&mut p, j);
    }
    write_u32(&mut p, dto.contacts.len() as u32);
    for c in &dto.contacts {
        write_u32(&mut p, c.body_a);
        write_u32(&mut p, c.body_b);
        write_u32(&mut p, c.entries.len() as u32);
        for e in &c.entries {
            write_f32(&mut p, e.point.x);
            write_f32(&mut p, e.point.y);
            write_f32(&mut p, e.normal.x);
            write_f32(&mut p, e.normal.y);
            write_f32(&mut p, e.normal_impulse);
            write_f32(&mut p, e.tangent_impulse);
        }
    }
    encode_fluid2(&mut p, &dto.fluid, dto.schema);
    Ok(encode_envelope(KIND_WORLD2, dto.schema, &p))
}

/// Version-gated gravity decode: v4+ payloads carry the config block,
/// older payloads resolve to defaults (same values a v3 file implies).
fn decode_opt_gravity_2(r: &mut Reader, version: u32) -> Result<GravityConfig2Dto, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_GRAVITY {
        decode_gravity_config_2(r)
    } else {
        Ok(GravityConfig2Dto::default())
    }
}

/// 3D counterpart of [`decode_opt_gravity_2`].
fn decode_opt_gravity_3(r: &mut Reader, version: u32) -> Result<GravityConfig3Dto, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_GRAVITY {
        decode_gravity_config_3(r)
    } else {
        Ok(GravityConfig3Dto::default())
    }
}

/// Version-gated per-body gravity scale: v4+ records carry a trailing
/// scale float per body; older records imply full coupling (1.0).
fn decode_opt_scale(r: &mut Reader, version: u32) -> Result<f32, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_GRAVITY {
        r.read_f32()
    } else {
        Ok(1.0)
    }
}

/// Version-gated per-body ambient scale: v5+ records carry a trailing
/// scale float per body; older records imply full coupling (1.0).
fn decode_opt_ambient_scale(r: &mut Reader, version: u32) -> Result<f32, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        r.read_f32()
    } else {
        Ok(1.0)
    }
}

/// Version-gated ambient config: v5 payloads carry the block; older
/// payloads resolve to a default config.
fn decode_opt_ambient_2(r: &mut Reader, version: u32) -> Result<AmbientConfig2Dto, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        decode_ambient_config_2(r)
    } else {
        Ok(AmbientConfig2Dto::default())
    }
}

/// 3D counterpart of [`decode_opt_ambient_2`].
fn decode_opt_ambient_3(r: &mut Reader, version: u32) -> Result<AmbientConfig3Dto, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        decode_ambient_config_3(r)
    } else {
        Ok(AmbientConfig3Dto::default())
    }
}

/// Version-gated simulation clock: v5 payloads carry it; older payloads
/// imply time 0.0.
fn decode_opt_time(r: &mut Reader, version: u32) -> Result<f32, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        r.read_f32()
    } else {
        Ok(0.0)
    }
}

/// Version-gated CCD config: v6 payloads carry the block; older
/// payloads resolve to a default config.
fn decode_opt_ccd(r: &mut Reader, version: u32) -> Result<CcdConfigDto, SnapshotError> {
    if version >= SNAPSHOT_SCHEMA_CCD {
        Ok(CcdConfigDto {
            enabled: r.read_u8()? != 0,
            tier: r.read_u8()?.into(),
            max_movement: r.read_f32()?,
            max_substeps: r.read_u32()?,
        })
    } else {
        Ok(CcdConfigDto::default())
    }
}

fn read_rho4(r: &mut Reader) -> Result<[f32; 4], SnapshotError> {
    let mut a = [0.0f32; 4];
    for v in a.iter_mut() {
        *v = r.read_f32()?;
    }
    Ok(a)
}

fn read_mass4(r: &mut Reader) -> Result<[f32; 4], SnapshotError> {
    let mut a = [0.0f32; 4];
    for v in a.iter_mut() {
        *v = r.read_f32()?;
    }
    Ok(a)
}

fn encode_fluid_config2(out: &mut Vec<u8>, c: &FluidConfig2Dto, version: u32) {
    write_f32(out, c.particle_mass);
    write_f32(out, c.rest_density);
    write_f32(out, c.smoothing_length);
    write_f32(out, c.stiffness);
    // Schema v5: EOS/viscosity/surface-tension block.
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        write_f32(out, c.gamma);
        write_f32(out, c.artificial_viscosity);
        write_f32(out, c.surface_tension);
        write_f32(out, c.cfl);
        write_u32(out, c.max_substeps);
        write_u8(out, c.adaptive_resolution as u8);
        write_f32(out, c.split_density_ratio);
        write_f32(out, c.merge_density_ratio);
        write_u8(out, c.boundary_particles as u8);
        write_f32(out, c.wall_friction);
        // Per-phase definitions (schema v5). `None` (single-phase) writes
        // count=1 + the default arrays so the layout is always fixed-size.
        match c.phases.as_ref() {
            Some(ph) => {
                write_u8(out, ph.count);
                for &v in ph.rest_density.iter() {
                    write_f32(out, v);
                }
                for &v in ph.mass.iter() {
                    write_f32(out, v);
                }
            }
            None => {
                write_u8(out, 1);
                for _ in 0..4 {
                    write_f32(out, 1.0);
                }
                for _ in 0..4 {
                    write_f32(out, c.particle_mass);
                }
            }
        }
    }
    write_f32(out, c.viscosity);
    write_f32(out, c.drag);
    write_f32(out, c.max_velocity);
    write_f32(out, c.particle_radius);
    write_u8(out, c.enabled as u8);
}

fn decode_fluid_config2(r: &mut Reader, version: u32) -> Result<FluidConfig2Dto, SnapshotError> {
    let particle_mass = r.read_f32()?;
    let rest_density = r.read_f32()?;
    let smoothing_length = r.read_f32()?;
    let stiffness = r.read_f32()?;
    let (
        gamma,
        artificial_viscosity,
        surface_tension,
        cfl,
        max_substeps,
        adaptive_resolution,
        split_density_ratio,
        merge_density_ratio,
        boundary_particles,
        wall_friction,
    ) = if version >= SNAPSHOT_SCHEMA_AMBIENT {
        (
            r.read_f32()?,
            r.read_f32()?,
            r.read_f32()?,
            r.read_f32()?,
            r.read_u32()?,
            r.read_u8()? != 0,
            r.read_f32()?,
            r.read_f32()?,
            r.read_u8()? != 0,
            r.read_f32()?,
        )
    } else {
        (
            default_gamma(),
            0.0,
            0.0,
            0.0,
            default_max_substeps(),
            false,
            default_split_ratio(),
            default_merge_ratio(),
            false,
            0.0,
        )
    };
    let (phase_count, phase_rest_density, phase_mass) = if version >= SNAPSHOT_SCHEMA_AMBIENT {
        let count = r.read_u8()?;
        let mut rho = [1.0f32; 4];
        for v in rho.iter_mut() {
            *v = r.read_f32()?;
        }
        let mut m = [particle_mass; 4];
        for v in m.iter_mut() {
            *v = r.read_f32()?;
        }
        (count, rho, m)
    } else {
        (1u8, [1.0f32; 4], [particle_mass; 4])
    };
    let phases = if phase_count > 1 {
        Some(PhasesDto {
            count: phase_count,
            rest_density: phase_rest_density,
            mass: phase_mass,
        })
    } else {
        None
    };
    Ok(FluidConfig2Dto {
        particle_mass,
        rest_density,
        smoothing_length,
        stiffness,
        gamma,
        artificial_viscosity,
        surface_tension,
        cfl,
        max_substeps,
        adaptive_resolution,
        split_density_ratio,
        merge_density_ratio,
        boundary_particles,
        wall_friction,
        phases,
        viscosity: r.read_f32()?,
        drag: r.read_f32()?,
        max_velocity: r.read_f32()?,
        particle_radius: r.read_f32()?,
        enabled: match r.read_u8()? {
            0 => false,
            1 => true,
            _ => return Err(SnapshotError::Parse("bad fluid enabled flag".into())),
        },
    })
}

fn encode_fluid2(out: &mut Vec<u8>, f: &Fluid2Dto, version: u32) {
    encode_fluid_config2(out, &f.config, version);
    write_u32(out, f.particles.len() as u32);
    for p in &f.particles {
        write_f32(out, p.position.x);
        write_f32(out, p.position.y);
        write_f32(out, p.velocity.x);
        write_f32(out, p.velocity.y);
        if version >= SNAPSHOT_SCHEMA_AMBIENT {
            write_f32(out, p.mass);
            write_u8(out, p.phase);
        }
    }
}

fn decode_fluid2(r: &mut Reader, version: u32) -> Result<Fluid2Dto, SnapshotError> {
    let config = decode_fluid_config2(r, version)?;
    let n = r.read_u32()? as usize;
    if n > super::fluid::MAX_PARTICLES_2D {
        return Err(SnapshotError::Parse(
            "fluid particle count exceeds MAX_PARTICLES_2D".into(),
        ));
    }
    let mut particles = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        particles.push(FluidParticle2Dto {
            position: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
            velocity: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
            mass: if version >= SNAPSHOT_SCHEMA_AMBIENT {
                r.read_f32()?
            } else {
                0.0
            },
            phase: if version >= SNAPSHOT_SCHEMA_AMBIENT {
                r.read_u8()?
            } else {
                0
            },
        });
    }
    Ok(Fluid2Dto { config, particles })
}

fn encode_fluid_config3(out: &mut Vec<u8>, c: &FluidConfig3Dto, version: u32) {
    write_f32(out, c.particle_mass);
    write_f32(out, c.rest_density);
    write_f32(out, c.smoothing_length);
    write_f32(out, c.stiffness);
    if version >= SNAPSHOT_SCHEMA_AMBIENT {
        write_f32(out, c.gamma);
        write_f32(out, c.artificial_viscosity);
        write_f32(out, c.surface_tension);
        write_f32(out, c.cfl);
        write_u32(out, c.max_substeps);
        write_u8(out, c.adaptive_resolution as u8);
        write_f32(out, c.split_density_ratio);
        write_f32(out, c.merge_density_ratio);
        write_u8(out, c.boundary_particles as u8);
        write_f32(out, c.wall_friction);
        // Per-phase definitions (schema v5). `None` (single-phase) writes
        // count=1 + the default arrays so the layout is always fixed-size.
        match c.phases.as_ref() {
            Some(ph) => {
                write_u8(out, ph.count);
                for &v in ph.rest_density.iter() {
                    write_f32(out, v);
                }
                for &v in ph.mass.iter() {
                    write_f32(out, v);
                }
            }
            None => {
                write_u8(out, 1);
                for _ in 0..4 {
                    write_f32(out, 1.0);
                }
                for _ in 0..4 {
                    write_f32(out, c.particle_mass);
                }
            }
        }
    }
    write_f32(out, c.viscosity);
    write_f32(out, c.drag);
    write_f32(out, c.max_velocity);
    write_f32(out, c.particle_radius);
    write_u8(out, c.enabled as u8);
}

fn decode_fluid_config3(r: &mut Reader, version: u32) -> Result<FluidConfig3Dto, SnapshotError> {
    let particle_mass = r.read_f32()?;
    let rest_density = r.read_f32()?;
    let smoothing_length = r.read_f32()?;
    let stiffness = r.read_f32()?;
    let (
        gamma,
        artificial_viscosity,
        surface_tension,
        cfl,
        max_substeps,
        adaptive_resolution,
        split_density_ratio,
        merge_density_ratio,
        boundary_particles,
        wall_friction,
        phase_count,
        phase_rest_density,
        phase_mass,
    ) = if version >= SNAPSHOT_SCHEMA_AMBIENT {
        (
            r.read_f32()?,
            r.read_f32()?,
            r.read_f32()?,
            r.read_f32()?,
            r.read_u32()?,
            r.read_u8()? != 0,
            r.read_f32()?,
            r.read_f32()?,
            r.read_u8()? != 0,
            r.read_f32()?,
            r.read_u8()?,
            read_rho4(r)?,
            read_mass4(r)?,
        )
    } else {
        (
            default_gamma(),
            0.0,
            0.0,
            0.0,
            default_max_substeps(),
            false,
            default_split_ratio(),
            default_merge_ratio(),
            false,
            0.0,
            1u8,
            [1.0; 4],
            [particle_mass; 4],
        )
    };
    let phases = if phase_count > 1 {
        Some(PhasesDto {
            count: phase_count,
            rest_density: phase_rest_density,
            mass: phase_mass,
        })
    } else {
        None
    };
    Ok(FluidConfig3Dto {
        particle_mass,
        rest_density,
        smoothing_length,
        stiffness,
        gamma,
        artificial_viscosity,
        surface_tension,
        cfl,
        max_substeps,
        adaptive_resolution,
        split_density_ratio,
        merge_density_ratio,
        boundary_particles,
        wall_friction,
        phases,
        viscosity: r.read_f32()?,
        drag: r.read_f32()?,
        max_velocity: r.read_f32()?,
        particle_radius: r.read_f32()?,
        enabled: match r.read_u8()? {
            0 => false,
            1 => true,
            _ => return Err(SnapshotError::Parse("bad fluid enabled flag".into())),
        },
    })
}

fn encode_fluid3(out: &mut Vec<u8>, f: &Fluid3Dto, version: u32) {
    encode_fluid_config3(out, &f.config, version);
    write_u32(out, f.particles.len() as u32);
    for p in &f.particles {
        write_f32(out, p.position.x);
        write_f32(out, p.position.y);
        write_f32(out, p.position.z);
        write_f32(out, p.velocity.x);
        write_f32(out, p.velocity.y);
        write_f32(out, p.velocity.z);
        if version >= SNAPSHOT_SCHEMA_AMBIENT {
            write_f32(out, p.mass);
            write_u8(out, p.phase);
        }
    }
}

fn decode_fluid3(r: &mut Reader, version: u32) -> Result<Fluid3Dto, SnapshotError> {
    let config = decode_fluid_config3(r, version)?;
    let n = r.read_u32()? as usize;
    if n > super::fluid3::MAX_PARTICLES_3D {
        return Err(SnapshotError::Parse(
            "fluid particle count exceeds MAX_PARTICLES_3D".into(),
        ));
    }
    let mut particles = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        particles.push(FluidParticle3Dto {
            position: Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            },
            velocity: Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            },
            mass: if version >= SNAPSHOT_SCHEMA_AMBIENT {
                r.read_f32()?
            } else {
                0.0
            },
            phase: if version >= SNAPSHOT_SCHEMA_AMBIENT {
                r.read_u8()?
            } else {
                0
            },
        });
    }
    Ok(Fluid3Dto { config, particles })
}

fn encode_joint2(out: &mut Vec<u8>, j: &Joint2Dto) {
    match j {
        Joint2Dto::Distance {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            impulse,
        } => {
            write_u8(out, 0);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, *rest_length);
            write_f32(out, *impulse);
        }
        Joint2Dto::Hinge {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            limit_min,
            limit_max,
            impulse,
            angular_impulse,
        } => {
            write_u8(out, 1);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, *limit_min);
            write_f32(out, *limit_max);
            write_f32(out, impulse.x);
            write_f32(out, impulse.y);
            write_f32(out, *angular_impulse);
        }
        Joint2Dto::Prismatic {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            limit_min,
            limit_max,
            ref_angle,
            perp_impulse,
            angular_impulse,
            limit_impulse,
        } => {
            write_u8(out, 2);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, axis_a.x);
            write_f32(out, axis_a.y);
            write_f32(out, *limit_min);
            write_f32(out, *limit_max);
            match ref_angle {
                Some(a) => {
                    write_u8(out, 1);
                    write_f32(out, *a);
                }
                None => write_u8(out, 0),
            }
            write_f32(out, *perp_impulse);
            write_f32(out, *angular_impulse);
            write_f32(out, *limit_impulse);
        }
        Joint2Dto::Fixed {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            ref_angle,
            impulse,
            angular_impulse,
        } => {
            write_u8(out, 3);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            match ref_angle {
                Some(a) => {
                    write_u8(out, 1);
                    write_f32(out, *a);
                }
                None => write_u8(out, 0),
            }
            write_f32(out, impulse.x);
            write_f32(out, impulse.y);
            write_f32(out, *angular_impulse);
        }
    }
}

fn decode_joint2(r: &mut Reader) -> Result<Joint2Dto, SnapshotError> {
    let tag = r.read_u8()?;
    let id = r.read_u32()?;
    let body_a = r.read_u32()?;
    let body_b = r.read_u32()?;
    let anchor_a = Vec2Dto {
        x: r.read_f32()?,
        y: r.read_f32()?,
    };
    let anchor_b = Vec2Dto {
        x: r.read_f32()?,
        y: r.read_f32()?,
    };
    match tag {
        0 => Ok(Joint2Dto::Distance {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length: r.read_f32()?,
            impulse: r.read_f32()?,
        }),
        1 => Ok(Joint2Dto::Hinge {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            limit_min: r.read_f32()?,
            limit_max: r.read_f32()?,
            impulse: Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            },
            angular_impulse: r.read_f32()?,
        }),
        2 => {
            let axis_a = Vec2Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
            };
            let limit_min = r.read_f32()?;
            let limit_max = r.read_f32()?;
            let ref_angle = match r.read_u8()? {
                0 => None,
                1 => Some(r.read_f32()?),
                _ => return Err(SnapshotError::Parse("bad ref_angle flag".into())),
            };
            Ok(Joint2Dto::Prismatic {
                id,
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                axis_a,
                limit_min,
                limit_max,
                ref_angle,
                perp_impulse: r.read_f32()?,
                angular_impulse: r.read_f32()?,
                limit_impulse: r.read_f32()?,
            })
        }
        3 => {
            let ref_angle = match r.read_u8()? {
                0 => None,
                1 => Some(r.read_f32()?),
                _ => return Err(SnapshotError::Parse("bad ref_angle flag".into())),
            };
            Ok(Joint2Dto::Fixed {
                id,
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                ref_angle,
                impulse: Vec2Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                },
                angular_impulse: r.read_f32()?,
            })
        }
        _ => Err(SnapshotError::Parse("unknown joint tag".into())),
    }
}

fn encode_joint3(out: &mut Vec<u8>, j: &Joint3Dto) {
    match j {
        Joint3Dto::Distance3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            impulse,
        } => {
            write_u8(out, 0);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_a.z);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, anchor_b.z);
            write_f32(out, *rest_length);
            write_f32(out, *impulse);
        }
        Joint3Dto::Hinge3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            limit_min,
            limit_max,
            impulse,
            align_u1,
            align_u2,
            twist_impulse,
        } => {
            write_u8(out, 1);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_a.z);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, anchor_b.z);
            write_f32(out, axis_a.x);
            write_f32(out, axis_a.y);
            write_f32(out, axis_a.z);
            write_f32(out, axis_b.x);
            write_f32(out, axis_b.y);
            write_f32(out, axis_b.z);
            write_f32(out, *limit_min);
            write_f32(out, *limit_max);
            write_f32(out, impulse.x);
            write_f32(out, impulse.y);
            write_f32(out, impulse.z);
            write_f32(out, *align_u1);
            write_f32(out, *align_u2);
            write_f32(out, *twist_impulse);
        }
        Joint3Dto::Prismatic3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            limit_min,
            limit_max,
            ref_orientation,
            perp_u1,
            perp_u2,
            angular_impulse,
            limit_impulse,
        } => {
            write_u8(out, 2);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_a.z);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, anchor_b.z);
            write_f32(out, axis_a.x);
            write_f32(out, axis_a.y);
            write_f32(out, axis_a.z);
            write_f32(out, *limit_min);
            write_f32(out, *limit_max);
            match ref_orientation {
                Some(a) => {
                    write_u8(out, 1);
                    write_f32(out, a[0]);
                    write_f32(out, a[1]);
                    write_f32(out, a[2]);
                    write_f32(out, a[3]);
                }
                None => write_u8(out, 0),
            }
            write_f32(out, *perp_u1);
            write_f32(out, *perp_u2);
            write_f32(out, angular_impulse.x);
            write_f32(out, angular_impulse.y);
            write_f32(out, angular_impulse.z);
            write_f32(out, *limit_impulse);
        }
        Joint3Dto::Fixed3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            ref_orientation,
            impulse,
            angular_impulse,
        } => {
            write_u8(out, 3);
            write_u32(out, *id);
            write_u32(out, *body_a);
            write_u32(out, *body_b);
            write_f32(out, anchor_a.x);
            write_f32(out, anchor_a.y);
            write_f32(out, anchor_a.z);
            write_f32(out, anchor_b.x);
            write_f32(out, anchor_b.y);
            write_f32(out, anchor_b.z);
            match ref_orientation {
                Some(a) => {
                    write_u8(out, 1);
                    write_f32(out, a[0]);
                    write_f32(out, a[1]);
                    write_f32(out, a[2]);
                    write_f32(out, a[3]);
                }
                None => write_u8(out, 0),
            }
            write_f32(out, impulse.x);
            write_f32(out, impulse.y);
            write_f32(out, impulse.z);
            write_f32(out, angular_impulse.x);
            write_f32(out, angular_impulse.y);
            write_f32(out, angular_impulse.z);
        }
    }
}

fn decode_joint3(r: &mut Reader) -> Result<Joint3Dto, SnapshotError> {
    let tag = r.read_u8()?;
    let id = r.read_u32()?;
    let body_a = r.read_u32()?;
    let body_b = r.read_u32()?;
    let anchor_a = Vec3Dto {
        x: r.read_f32()?,
        y: r.read_f32()?,
        z: r.read_f32()?,
    };
    let anchor_b = Vec3Dto {
        x: r.read_f32()?,
        y: r.read_f32()?,
        z: r.read_f32()?,
    };
    match tag {
        0 => Ok(Joint3Dto::Distance3 {
            id,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length: r.read_f32()?,
            impulse: r.read_f32()?,
        }),
        1 => {
            let axis_a = Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            };
            let axis_b = Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            };
            let limit_min = r.read_f32()?;
            let limit_max = r.read_f32()?;
            Ok(Joint3Dto::Hinge3 {
                id,
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                axis_a,
                axis_b,
                limit_min,
                limit_max,
                impulse: Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                },
                align_u1: r.read_f32()?,
                align_u2: r.read_f32()?,
                twist_impulse: r.read_f32()?,
            })
        }
        2 => {
            let axis_a = Vec3Dto {
                x: r.read_f32()?,
                y: r.read_f32()?,
                z: r.read_f32()?,
            };
            let limit_min = r.read_f32()?;
            let limit_max = r.read_f32()?;
            let ref_orientation = match r.read_u8()? {
                0 => None,
                1 => Some([r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?]),
                _ => return Err(SnapshotError::Parse("bad ref_orientation flag".into())),
            };
            Ok(Joint3Dto::Prismatic3 {
                id,
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                axis_a,
                limit_min,
                limit_max,
                ref_orientation,
                perp_u1: r.read_f32()?,
                perp_u2: r.read_f32()?,
                angular_impulse: Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                },
                limit_impulse: r.read_f32()?,
            })
        }
        3 => {
            let ref_orientation = match r.read_u8()? {
                0 => None,
                1 => Some([r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?]),
                _ => return Err(SnapshotError::Parse("bad ref_orientation flag".into())),
            };
            Ok(Joint3Dto::Fixed3 {
                id,
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                ref_orientation,
                impulse: Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                },
                angular_impulse: Vec3Dto {
                    x: r.read_f32()?,
                    y: r.read_f32()?,
                    z: r.read_f32()?,
                },
            })
        }
        _ => Err(SnapshotError::Parse("unknown 3D joint tag".into())),
    }
}

/// Deserialize a 2D world from the versioned binary envelope.
pub fn world2_from_binary(bytes: &[u8], mode: LoadMode) -> Result<World2, SnapshotError> {
    let (version, payload) = decode_envelope(bytes, KIND_WORLD2)?;
    let mut r = Reader {
        b: &payload,
        pos: 0,
    };
    let dto = World2Dto {
        schema: version,
        gravity: Vec2Dto {
            x: r.read_f32()?,
            y: r.read_f32()?,
        },
        dt: r.read_f32()?,
        linear_sleep_threshold: r.read_f32()?,
        angular_sleep_threshold: r.read_f32()?,
        solver: decode_solver(&mut r, version)?,
        validation: decode_validation(&mut r)?,
        gravity_config: decode_opt_gravity_2(&mut r, version)?,
        ambient: decode_opt_ambient_2(&mut r, version)?,
        time: decode_opt_time(&mut r, version)?,
        ccd: decode_opt_ccd(&mut r, version)?,
        // Schema v8: thermal config block.
        thermal: if version >= SNAPSHOT_SCHEMA_THERMAL {
            ThermalConfigDto {
                enabled: r.read_u8()? != 0,
                ambient_temperature: r.read_f32()?,
                stefan_boltzmann: r.read_f32()?,
                convection_coefficient: r.read_f32()?,
                radiation_enabled: r.read_u8()? != 0,
                conduction_enabled: r.read_u8()? != 0,
                convection_enabled: r.read_u8()? != 0,
                max_temperature: r.read_f32()?,
                min_temperature: r.read_f32()?,
                radiation_radius_factor: r.read_f32()?,
                max_temperature_rate: r.read_f32()?,
            }
        } else {
            ThermalConfigDto::default()
        },
        // Schema v8: broad-phase backend tag (shares the thermal block's
        // version gate, so older payloads stay untouched).
        broadphase_mode: if version >= SNAPSHOT_SCHEMA_THERMAL {
            r.read_u32()?
        } else {
            default_broadphase_mode_tag()
        },
        // Schema v9: collision-tier ceiling.
        collision_tier: if version >= SNAPSHOT_SCHEMA_COLLISION_TIER {
            CollisionTierConfigDto {
                enabled: r.read_u8()? != 0,
                max_tier: r.read_u32()?,
            }
        } else {
            CollisionTierConfigDto::default()
        },
        next_body_id: r.read_u32()?,
        next_joint_id: r.read_u32()?,
        cell_size: r.read_f32()?,
        bodies: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse("body count exceeds 1,000,000".into()));
            }
            let mut v = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                let id = r.read_u32()?;
                let flags = r.read_u8()?;
                if flags & !0b11 != 0 {
                    return Err(SnapshotError::Parse("bad body flags".into()));
                }
                let body = Body2Dto {
                    id,
                    is_static: flags & 0b01 != 0,
                    is_sleeping: flags & 0b10 != 0,
                    linear_damping: r.read_f32()?,
                    angular_damping: r.read_f32()?,
                    material: MaterialDto {
                        friction: r.read_f32()?,
                        restitution: r.read_f32()?,
                        density: r.read_f32()?,
                        // Schema v8: material thermal properties.
                        specific_heat: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_specific_heat()
                        },
                        thermal_conductivity: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_thermal_conductivity()
                        },
                        emissivity: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_emissivity()
                        },
                    },
                    shape: decode_shape2(&mut r)?,
                    position: Vec2Dto {
                        x: r.read_f32()?,
                        y: r.read_f32()?,
                    },
                    rotation: [r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?],
                    velocity: Vec2Dto {
                        x: r.read_f32()?,
                        y: r.read_f32()?,
                    },
                    angular_velocity: r.read_f32()?,
                    gravity_scale: decode_opt_scale(&mut r, version)?,
                    ambient_scale: decode_opt_ambient_scale(&mut r, version)?,
                    // Schema v8: per-body temperature.
                    temperature: if version >= SNAPSHOT_SCHEMA_THERMAL {
                        r.read_f32()?
                    } else {
                        default_temperature()
                    },
                };
                v.push(body);
            }
            v
        },
        joints: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse("joint count exceeds 1,000,000".into()));
            }
            let mut v = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                v.push(decode_joint2(&mut r)?);
            }
            v
        },
        contacts: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse(
                    "contact pair count exceeds 1,000,000".into(),
                ));
            }
            let mut v = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                let body_a = r.read_u32()?;
                let body_b = r.read_u32()?;
                let m = r.read_u32()? as usize;
                if m > 16 {
                    return Err(SnapshotError::Parse(
                        "contact entry count exceeds 16".into(),
                    ));
                }
                let mut entries = Vec::with_capacity(m);
                for _ in 0..m {
                    entries.push(ContactEntry2Dto {
                        point: Vec2Dto {
                            x: r.read_f32()?,
                            y: r.read_f32()?,
                        },
                        normal: Vec2Dto {
                            x: r.read_f32()?,
                            y: r.read_f32()?,
                        },
                        normal_impulse: r.read_f32()?,
                        tangent_impulse: r.read_f32()?,
                    });
                }
                v.push(Contact2Dto {
                    body_a,
                    body_b,
                    entries,
                });
            }
            v
        },
        fluid: decode_fluid2(&mut r, version)?,
    };
    if r.pos != payload.len() {
        return Err(SnapshotError::Parse(
            "trailing bytes in snapshot payload".into(),
        ));
    }
    world2_from_dto(&dto, mode)
}

/// Serialize a 3D world to the versioned binary envelope.
pub fn world3_to_binary(world: &World3, mode: LoadMode) -> Result<Vec<u8>, SnapshotError> {
    let dto = world3_dto(world, mode);
    let mut p = Vec::new();
    write_f32(&mut p, dto.gravity.x);
    write_f32(&mut p, dto.gravity.y);
    write_f32(&mut p, dto.gravity.z);
    write_f32(&mut p, dto.dt);
    write_f32(&mut p, dto.linear_sleep_threshold);
    write_f32(&mut p, dto.angular_sleep_threshold);
    encode_solver(&mut p, &dto.solver, dto.schema);
    encode_validation(&mut p, &dto.validation);
    // Schema v4: gravity config block (see `world2_to_binary`).
    let gravity_v4 = dto.schema >= SNAPSHOT_SCHEMA_GRAVITY;
    if gravity_v4 {
        encode_gravity_config_3(&mut p, &dto.gravity_config);
    }
    // Schema v5: ambient config block + clock (see `world2_to_binary`).
    let ambient_v5 = dto.schema >= SNAPSHOT_SCHEMA_AMBIENT;
    if ambient_v5 {
        encode_ambient_config_3(&mut p, &dto.ambient);
        write_f32(&mut p, dto.time);
    }
    // Schema v6: CCD config block.
    if dto.schema >= SNAPSHOT_SCHEMA_CCD {
        write_u8(&mut p, dto.ccd.enabled as u8);
        write_u8(&mut p, dto.ccd.tier as u8);
        write_f32(&mut p, dto.ccd.max_movement);
        write_u32(&mut p, dto.ccd.max_substeps);
    }
    // Schema v8: thermal config block.
    let thermal_v8 = dto.schema >= SNAPSHOT_SCHEMA_THERMAL;
    if thermal_v8 {
        write_u8(&mut p, dto.thermal.enabled as u8);
        write_f32(&mut p, dto.thermal.ambient_temperature);
        write_f32(&mut p, dto.thermal.stefan_boltzmann);
        write_f32(&mut p, dto.thermal.convection_coefficient);
        write_u8(&mut p, dto.thermal.radiation_enabled as u8);
        write_u8(&mut p, dto.thermal.conduction_enabled as u8);
        write_u8(&mut p, dto.thermal.convection_enabled as u8);
        write_f32(&mut p, dto.thermal.max_temperature);
        write_f32(&mut p, dto.thermal.min_temperature);
        write_f32(&mut p, dto.thermal.radiation_radius_factor);
        write_f32(&mut p, dto.thermal.max_temperature_rate);
        // Schema v8: broad-phase backend tag (see `world2_to_binary`).
        write_u32(&mut p, dto.broadphase_mode);
    }
    // Schema v9: collision-tier ceiling (see `world2_to_binary`).
    if dto.schema >= SNAPSHOT_SCHEMA_COLLISION_TIER {
        write_u8(&mut p, dto.collision_tier.enabled as u8);
        write_u32(&mut p, dto.collision_tier.max_tier);
    }
    write_u32(&mut p, dto.next_body_id);
    write_u32(&mut p, dto.next_joint_id);
    write_u32(&mut p, dto.bodies.len() as u32);
    for b in &dto.bodies {
        write_u32(&mut p, b.id);
        write_u8(&mut p, (b.is_static as u8) | ((b.is_sleeping as u8) << 1));
        write_f32(&mut p, b.linear_damping);
        write_f32(&mut p, b.angular_damping);
        write_f32(&mut p, b.material.friction);
        write_f32(&mut p, b.material.restitution);
        write_f32(&mut p, b.material.density);
        // Schema v8: material thermal properties.
        if thermal_v8 {
            write_f32(&mut p, b.material.specific_heat);
            write_f32(&mut p, b.material.thermal_conductivity);
            write_f32(&mut p, b.material.emissivity);
        }
        match &b.shape {
            None => write_u8(&mut p, 255),
            Some(s) => encode_shape3(&mut p, s),
        }
        write_f32(&mut p, b.position.x);
        write_f32(&mut p, b.position.y);
        write_f32(&mut p, b.position.z);
        for r in b.rotation {
            write_f32(&mut p, r);
        }
        write_f32(&mut p, b.velocity.x);
        write_f32(&mut p, b.velocity.y);
        write_f32(&mut p, b.velocity.z);
        write_f32(&mut p, b.angular_velocity.x);
        write_f32(&mut p, b.angular_velocity.y);
        write_f32(&mut p, b.angular_velocity.z);
        if gravity_v4 {
            write_f32(&mut p, b.gravity_scale);
        }
        if ambient_v5 {
            write_f32(&mut p, b.ambient_scale);
        }
        // Schema v8: per-body temperature.
        if thermal_v8 {
            write_f32(&mut p, b.temperature);
        }
    }
    write_u32(&mut p, dto.joints.len() as u32);
    for j in &dto.joints {
        encode_joint3(&mut p, j);
    }
    write_u32(&mut p, dto.contacts.len() as u32);
    for c in &dto.contacts {
        write_u32(&mut p, c.body_a);
        write_u32(&mut p, c.body_b);
        write_u32(&mut p, c.entries.len() as u32);
        for e in &c.entries {
            write_f32(&mut p, e.point.x);
            write_f32(&mut p, e.point.y);
            write_f32(&mut p, e.point.z);
            write_f32(&mut p, e.normal.x);
            write_f32(&mut p, e.normal.y);
            write_f32(&mut p, e.normal.z);
            write_f32(&mut p, e.normal_impulse);
            write_f32(&mut p, e.tangent1_impulse);
            write_f32(&mut p, e.tangent2_impulse);
        }
    }
    encode_fluid3(&mut p, &dto.fluid, dto.schema);
    Ok(encode_envelope(KIND_WORLD3, dto.schema, &p))
}

/// Deserialize a 3D world from the versioned binary envelope.
pub fn world3_from_binary(bytes: &[u8], mode: LoadMode) -> Result<World3, SnapshotError> {
    let (version, payload) = decode_envelope(bytes, KIND_WORLD3)?;
    let mut r = Reader {
        b: &payload,
        pos: 0,
    };
    let dto = World3Dto {
        schema: version,
        gravity: Vec3Dto {
            x: r.read_f32()?,
            y: r.read_f32()?,
            z: r.read_f32()?,
        },
        dt: r.read_f32()?,
        linear_sleep_threshold: r.read_f32()?,
        angular_sleep_threshold: r.read_f32()?,
        solver: decode_solver(&mut r, version)?,
        validation: decode_validation(&mut r)?,
        gravity_config: decode_opt_gravity_3(&mut r, version)?,
        ambient: decode_opt_ambient_3(&mut r, version)?,
        time: decode_opt_time(&mut r, version)?,
        ccd: decode_opt_ccd(&mut r, version)?,
        // Schema v8: thermal config block.
        thermal: if version >= SNAPSHOT_SCHEMA_THERMAL {
            ThermalConfigDto {
                enabled: r.read_u8()? != 0,
                ambient_temperature: r.read_f32()?,
                stefan_boltzmann: r.read_f32()?,
                convection_coefficient: r.read_f32()?,
                radiation_enabled: r.read_u8()? != 0,
                conduction_enabled: r.read_u8()? != 0,
                convection_enabled: r.read_u8()? != 0,
                max_temperature: r.read_f32()?,
                min_temperature: r.read_f32()?,
                radiation_radius_factor: r.read_f32()?,
                max_temperature_rate: r.read_f32()?,
            }
        } else {
            ThermalConfigDto::default()
        },
        // Schema v8: broad-phase backend tag (see `world2_from_binary`).
        broadphase_mode: if version >= SNAPSHOT_SCHEMA_THERMAL {
            r.read_u32()?
        } else {
            default_broadphase_mode_tag()
        },
        // Schema v9: collision-tier ceiling.
        collision_tier: if version >= SNAPSHOT_SCHEMA_COLLISION_TIER {
            CollisionTierConfigDto {
                enabled: r.read_u8()? != 0,
                max_tier: r.read_u32()?,
            }
        } else {
            CollisionTierConfigDto::default()
        },
        next_body_id: r.read_u32()?,
        next_joint_id: r.read_u32()?,
        bodies: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse("body count exceeds 1,000,000".into()));
            }
            let mut v = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                let id = r.read_u32()?;
                let flags = r.read_u8()?;
                if flags & !0b11 != 0 {
                    return Err(SnapshotError::Parse("bad body flags".into()));
                }
                v.push(Body3Dto {
                    id,
                    is_static: flags & 0b01 != 0,
                    is_sleeping: flags & 0b10 != 0,
                    linear_damping: r.read_f32()?,
                    angular_damping: r.read_f32()?,
                    material: MaterialDto {
                        friction: r.read_f32()?,
                        restitution: r.read_f32()?,
                        density: r.read_f32()?,
                        // Schema v8: material thermal properties.
                        specific_heat: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_specific_heat()
                        },
                        thermal_conductivity: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_thermal_conductivity()
                        },
                        emissivity: if version >= SNAPSHOT_SCHEMA_THERMAL {
                            r.read_f32()?
                        } else {
                            default_emissivity()
                        },
                    },
                    shape: decode_shape3(&mut r)?,
                    position: Vec3Dto {
                        x: r.read_f32()?,
                        y: r.read_f32()?,
                        z: r.read_f32()?,
                    },
                    rotation: [r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?],
                    velocity: Vec3Dto {
                        x: r.read_f32()?,
                        y: r.read_f32()?,
                        z: r.read_f32()?,
                    },
                    angular_velocity: Vec3Dto {
                        x: r.read_f32()?,
                        y: r.read_f32()?,
                        z: r.read_f32()?,
                    },
                    gravity_scale: decode_opt_scale(&mut r, version)?,
                    ambient_scale: decode_opt_ambient_scale(&mut r, version)?,
                    // Schema v8: per-body temperature.
                    temperature: if version >= SNAPSHOT_SCHEMA_THERMAL {
                        r.read_f32()?
                    } else {
                        default_temperature()
                    },
                });
            }
            v
        },
        joints: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse("joint count exceeds 1,000,000".into()));
            }
            let mut v = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                v.push(decode_joint3(&mut r)?);
            }
            v
        },
        contacts: {
            let n = r.read_u32()? as usize;
            if n > 1_000_000 {
                return Err(SnapshotError::Parse(
                    "contact pair count exceeds 1,000,000".into(),
                ));
            }
            let mut v = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                let body_a = r.read_u32()?;
                let body_b = r.read_u32()?;
                let m = r.read_u32()? as usize;
                if m > 16 {
                    return Err(SnapshotError::Parse(
                        "contact entry count exceeds 16".into(),
                    ));
                }
                let mut entries = Vec::with_capacity(m);
                for _ in 0..m {
                    entries.push(ContactEntry3Dto {
                        point: Vec3Dto {
                            x: r.read_f32()?,
                            y: r.read_f32()?,
                            z: r.read_f32()?,
                        },
                        normal: Vec3Dto {
                            x: r.read_f32()?,
                            y: r.read_f32()?,
                            z: r.read_f32()?,
                        },
                        normal_impulse: r.read_f32()?,
                        tangent1_impulse: r.read_f32()?,
                        tangent2_impulse: r.read_f32()?,
                    });
                }
                v.push(Contact3Dto {
                    body_a,
                    body_b,
                    entries,
                });
            }
            v
        },
        fluid: decode_fluid3(&mut r, version)?,
    };
    if r.pos != payload.len() {
        return Err(SnapshotError::Parse(
            "trailing bytes in snapshot payload".into(),
        ));
    }
    world3_from_dto(&dto, mode)
}

// ============================================================================
// File helpers (thin IO shell over the pure encode/decode above)
// ============================================================================

fn write_file(path: &std::path::Path, bytes: &[u8]) -> Result<(), SnapshotError> {
    std::fs::write(path, bytes).map_err(|e| SnapshotError::Io(e.to_string()))
}

fn read_file(path: &std::path::Path) -> Result<Vec<u8>, SnapshotError> {
    std::fs::read(path).map_err(|e| SnapshotError::Io(e.to_string()))
}

/// Save a 2D world to a file (JSON when `format == FORMAT_JSON`,
/// binary when `format == FORMAT_BINARY`).
pub fn world2_save_file(
    world: &World2,
    path: &std::path::Path,
    format: u32,
    mode: LoadMode,
) -> Result<(), SnapshotError> {
    match format {
        FORMAT_JSON => write_file(path, world2_to_json(world, mode)?.as_bytes()),
        FORMAT_BINARY => write_file(path, &world2_to_binary(world, mode)?),
        _ => Err(SnapshotError::Corrupt("unknown snapshot format".into())),
    }
}

/// Load a 2D world from a file. The format is detected from the JSON
/// envelope: files starting with `{` parse as JSON, otherwise binary.
pub fn world2_load_file(path: &std::path::Path, mode: LoadMode) -> Result<World2, SnapshotError> {
    let bytes = read_file(path)?;
    let trimmed = bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .copied()
        .unwrap_or(0);
    if trimmed == b'{' {
        let text = std::str::from_utf8(&bytes).map_err(|e| SnapshotError::Parse(e.to_string()))?;
        world2_from_json(text, mode)
    } else {
        world2_from_binary(&bytes, mode)
    }
}

/// Save a 3D world to a file (JSON when `format == FORMAT_JSON`,
/// binary when `format == FORMAT_BINARY`).
pub fn world3_save_file(
    world: &World3,
    path: &std::path::Path,
    format: u32,
    mode: LoadMode,
) -> Result<(), SnapshotError> {
    match format {
        FORMAT_JSON => write_file(path, world3_to_json(world, mode)?.as_bytes()),
        FORMAT_BINARY => write_file(path, &world3_to_binary(world, mode)?),
        _ => Err(SnapshotError::Corrupt("unknown snapshot format".into())),
    }
}

/// Load a 3D world from a file (format auto-detected like 2D).
pub fn world3_load_file(path: &std::path::Path, mode: LoadMode) -> Result<World3, SnapshotError> {
    let bytes = read_file(path)?;
    let trimmed = bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .copied()
        .unwrap_or(0);
    if trimmed == b'{' {
        let text = std::str::from_utf8(&bytes).map_err(|e| SnapshotError::Parse(e.to_string()))?;
        world3_from_json(text, mode)
    } else {
        world3_from_binary(&bytes, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Shape2;
    use glam::Vec2;

    fn scene2() -> World2 {
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
        w.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.add_dynamic_body(
            Vec2::new(1.0, 3.0),
            Shape2::Box {
                half_extents: Vec2::new(0.4, 0.4),
            },
            1.5,
        )
        .expect("valid shape");
        w
    }

    fn hash_world2(w: &World2) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for body in &w.bodies {
            for v in [
                body.transform.position.x,
                body.transform.position.y,
                body.velocity.x,
                body.velocity.y,
                body.angular_velocity,
                body.rotation_angle(),
            ] {
                h ^= v.to_bits() as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    #[test]
    fn test_snapshot_json_exact_roundtrip() {
        let mut w = scene2();
        for _ in 0..120 {
            w.step(1.0 / 60.0);
        }
        let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
        let r = world2_from_json(&text, LoadMode::Exact).expect("deserializes");
        assert_eq!(hash_world2(&w), hash_world2(&r));
        assert_eq!(w.bodies.len(), r.bodies.len());
        assert_eq!(w.next_body_id, r.next_body_id);
    }

    #[test]
    fn test_snapshot_binary_exact_roundtrip() {
        let mut w = scene2();
        for _ in 0..120 {
            w.step(1.0 / 60.0);
        }
        let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
        let r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
        assert_eq!(hash_world2(&w), hash_world2(&r));
    }

    fn assert_fluid2_eq(a: &World2, b: &World2) {
        assert_eq!(a.fluid.config.enabled, b.fluid.config.enabled);
        for (field, x, y) in [
            (
                "particle_mass",
                a.fluid.config.particle_mass,
                b.fluid.config.particle_mass,
            ),
            (
                "rest_density",
                a.fluid.config.rest_density,
                b.fluid.config.rest_density,
            ),
            (
                "smoothing_length",
                a.fluid.config.smoothing_length,
                b.fluid.config.smoothing_length,
            ),
            (
                "stiffness",
                a.fluid.config.stiffness,
                b.fluid.config.stiffness,
            ),
            (
                "viscosity",
                a.fluid.config.viscosity,
                b.fluid.config.viscosity,
            ),
            ("drag", a.fluid.config.drag, b.fluid.config.drag),
            (
                "max_velocity",
                a.fluid.config.max_velocity,
                b.fluid.config.max_velocity,
            ),
            (
                "particle_radius",
                a.fluid.config.particle_radius,
                b.fluid.config.particle_radius,
            ),
        ] {
            assert_eq!(x.to_bits(), y.to_bits(), "fluid config {field} drifted");
        }
        assert_eq!(a.fluid.particles.len(), b.fluid.particles.len());
        for (pa, pb) in a.fluid.particles.iter().zip(b.fluid.particles.iter()) {
            assert_eq!(pa.position.x.to_bits(), pb.position.x.to_bits());
            assert_eq!(pa.position.y.to_bits(), pb.position.y.to_bits());
            assert_eq!(pa.velocity.x.to_bits(), pb.velocity.x.to_bits());
            assert_eq!(pa.velocity.y.to_bits(), pb.velocity.y.to_bits());
        }
    }

    #[test]
    fn test_snapshot_fluid_expansion_roundtrip() {
        use crate::body::Shape2;
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
        w.fluid
            .add_particles(&[Vec2::new(0.0, 0.0), Vec2::new(0.25, 0.0)]);
        w.fluid.config.gamma = 7.0;
        w.fluid.config.stiffness = 15.0;
        w.fluid.config.artificial_viscosity = 1.2;
        w.fluid.config.surface_tension = 0.7;
        w.fluid.config.wall_friction = 0.4;
        for _ in 0..10 {
            w.step(1.0 / 60.0);
        }
        // JSON + binary round-trips preserve the new fluid config (v5).
        let r = world2_from_json(
            &world2_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        assert_eq!(r.fluid.config.gamma, 7.0);
        assert_eq!(r.fluid.config.artificial_viscosity, 1.2);
        assert_eq!(r.fluid.config.surface_tension, 0.7);
        assert_eq!(r.fluid.config.wall_friction, 0.4);
        let b = world2_from_binary(
            &world2_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert_eq!(b.fluid.config.gamma, 7.0);
        assert_eq!(b.fluid.config.artificial_viscosity, 1.2);
        assert_eq!(b.fluid.config.surface_tension, 0.7);
        assert_eq!(b.fluid.config.wall_friction, 0.4);
    }

    #[test]
    fn test_snapshot_fluid_roundtrip_2d() {
        let mut w = World2::new();
        w.add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
        let pts: Vec<Vec2> = (0..3)
            .flat_map(|iy| (0..4).map(move |ix| Vec2::new(ix as f32 * 0.25, iy as f32 * 0.25)))
            .collect();
        assert_eq!(w.fluid.add_particles(&pts), 12);
        for _ in 0..60 {
            w.step(1.0 / 60.0);
        }
        // JSON and binary round-trips preserve fluid state bit-exactly.
        let r = world2_from_json(
            &world2_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        assert_fluid2_eq(&w, &r);
        let b = world2_from_binary(
            &world2_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert_fluid2_eq(&w, &b);
        // Continuation: identical futures after a mid-run save.
        let mut direct = w;
        let mut r2 = world2_from_json(
            &world2_to_json(&direct, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        for _ in 0..60 {
            direct.step(1.0 / 60.0);
            r2.step(1.0 / 60.0);
        }
        assert_fluid2_eq(&direct, &r2);
        assert_eq!(hash_world2(&direct), hash_world2(&r2));
    }

    #[test]
    fn test_snapshot_ambient_roundtrip_2d() {
        use crate::ambient::{AmbientConfig2, DragConfig2, SpringConfig2, TurbulenceConfig2};
        let mut w = World2::new();
        let id = w
            .add_dynamic_body(Vec2::new(1.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].ambient_scale = 0.25;
        w.ambient = AmbientConfig2 {
            spring: SpringConfig2 {
                enabled: true,
                anchor: Vec2::new(0.5, -0.5),
                stiffness: 12.0,
                damping: 1.5,
                max_displacement: 8.0,
            },
            drag: DragConfig2 {
                enabled: true,
                linear: 0.3,
                quadratic: 0.7,
                air_density: 1.2,
                angular: 0.05,
                wind: Vec2::new(1.0, 0.0),
            },
            turbulence: TurbulenceConfig2 {
                enabled: true,
                amplitude: 2.0,
                frequency: 0.6,
                time_scale: 1.3,
                seed: 12345,
                octaves: 4,
                divergence_free: true,
            },
        };
        for _ in 0..30 {
            w.step(1.0 / 60.0);
        }
        // v5 schema is selected when ambient is active.
        assert!(w.time > 0.0);

        let r = world2_from_json(
            &world2_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        assert_eq!(w.ambient, r.ambient);
        assert_eq!(w.bodies[id].ambient_scale, r.bodies[id].ambient_scale);
        assert!(
            (w.time - r.time).abs() < 1e-4,
            "json time {} vs {}",
            w.time,
            r.time
        );

        let b = world2_from_binary(
            &world2_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert_eq!(w.ambient, b.ambient);
        assert_eq!(w.bodies[id].ambient_scale, b.bodies[id].ambient_scale);
        assert!(
            (w.time - b.time).abs() < 1e-4,
            "bin time {} vs {}",
            w.time,
            b.time
        );

        // Continuation parity: identical futures after a mid-run save.
        let mut direct = w;
        let mut r2 = world2_from_binary(
            &world2_to_binary(&direct, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        for _ in 0..60 {
            direct.step(1.0 / 60.0);
            r2.step(1.0 / 60.0);
        }
        assert_eq!(hash_world2(&direct), hash_world2(&r2));
    }

    #[test]
    fn test_snapshot_ambient_roundtrip_3d() {
        use crate::ambient::{AmbientConfig3, SpringConfig3, TurbulenceConfig3};
        let mut w = World3::new();
        let id = w
            .add_dynamic_body(
                Vec3::new(1.0, 2.0, 3.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        w.bodies[id].ambient_scale = 2.0;
        w.ambient = AmbientConfig3 {
            spring: SpringConfig3 {
                enabled: true,
                anchor: Vec3::new(0.0, 1.0, 0.0),
                stiffness: 9.0,
                damping: 0.5,
                max_displacement: 0.0,
            },
            drag: Default::default(),
            turbulence: TurbulenceConfig3 {
                enabled: true,
                amplitude: 1.5,
                frequency: 0.4,
                time_scale: 0.9,
                seed: 777,
                octaves: 3,
                divergence_free: false,
            },
        };
        for _ in 0..20 {
            w.step(1.0 / 60.0);
        }
        let b = world3_from_binary(
            &world3_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert_eq!(w.ambient, b.ambient);
        assert_eq!(w.bodies[id].ambient_scale, b.bodies[id].ambient_scale);
        assert!((w.time - b.time).abs() < 1e-4);
    }

    #[test]
    fn test_snapshot_fluid_roundtrip_3d() {
        use crate::body::Shape3;
        use glam::Vec3;
        let mut w = World3::new();
        w.add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
        )
        .expect("valid shape");
        let pts: Vec<Vec3> = (0..2)
            .flat_map(|iz| {
                (0..3).flat_map(move |iy| {
                    (0..3).map(move |ix| {
                        Vec3::new(ix as f32 * 0.25, iy as f32 * 0.25, iz as f32 * 0.25)
                    })
                })
            })
            .collect();
        assert_eq!(w.fluid.add_particles(&pts), 18);
        for _ in 0..60 {
            w.step(1.0 / 60.0);
        }
        let check = |a: &World3, b: &World3| {
            assert_eq!(a.fluid.particles.len(), b.fluid.particles.len());
            for (pa, pb) in a.fluid.particles.iter().zip(b.fluid.particles.iter()) {
                assert_eq!(pa.position.x.to_bits(), pb.position.x.to_bits());
                assert_eq!(pa.position.y.to_bits(), pb.position.y.to_bits());
                assert_eq!(pa.position.z.to_bits(), pb.position.z.to_bits());
                assert_eq!(pa.velocity.x.to_bits(), pb.velocity.x.to_bits());
                assert_eq!(pa.velocity.y.to_bits(), pb.velocity.y.to_bits());
                assert_eq!(pa.velocity.z.to_bits(), pb.velocity.z.to_bits());
            }
        };
        let r = world3_from_json(
            &world3_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        check(&w, &r);
        let b = world3_from_binary(
            &world3_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        check(&w, &b);
        let mut direct = w;
        let mut r2 = world3_from_json(
            &world3_to_json(&direct, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        for _ in 0..60 {
            direct.step(1.0 / 60.0);
            r2.step(1.0 / 60.0);
        }
        check(&direct, &r2);
    }

    #[test]
    fn test_snapshot_json_binary_equivalent() {
        let mut w = scene2();
        for _ in 0..60 {
            w.step(1.0 / 60.0);
        }
        let a = world2_from_json(
            &world2_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        let b = world2_from_binary(
            &world2_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert_eq!(hash_world2(&a), hash_world2(&b));
    }

    #[test]
    fn test_snapshot_convex_mesh_roundtrip() {
        use crate::body::Shape3;
        use glam::Vec3;
        // Centered unit-box mesh (outward wound).
        let (hx, hy, hz) = (0.5f32, 0.5f32, 0.5f32);
        let vertices = vec![
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
        ];
        let indices = vec![
            1, 2, 6, 1, 6, 5, // +x
            0, 4, 7, 0, 7, 3, // -x
            3, 7, 6, 3, 6, 2, // +y
            0, 1, 5, 0, 5, 4, // -y
            4, 5, 6, 4, 6, 7, // +z
            0, 3, 2, 0, 2, 1, // -z
        ];
        let mut w = World3::new();
        w.add_dynamic_body(
            Vec3::new(0.0, 2.0, 0.0),
            Shape3::ConvexMesh { vertices, indices },
            1.0,
        )
        .expect("valid mesh");
        let r = world3_from_json(
            &world3_to_json(&w, LoadMode::Exact).expect("json"),
            LoadMode::Exact,
        )
        .expect("json load");
        assert_eq!(w.bodies.len(), r.bodies.len());
        assert!((w.bodies[0].mass - r.bodies[0].mass).abs() < 1e-6);
        assert!(
            (w.bodies[0].inertia.x_axis - r.bodies[0].inertia.x_axis).length() < 1e-4,
            "inertia drifted across JSON round-trip"
        );
        let b = world3_from_binary(
            &world3_to_binary(&w, LoadMode::Exact).expect("bin"),
            LoadMode::Exact,
        )
        .expect("bin load");
        assert!((w.bodies[0].mass - b.bodies[0].mass).abs() < 1e-6);
        assert!(
            (w.bodies[0].inertia.x_axis - b.bodies[0].inertia.x_axis).length() < 1e-4,
            "inertia drifted across binary round-trip"
        );
    }

    #[test]
    fn test_snapshot_continuation_matches() {
        // Save at frame N, step M more both ways: identical futures.
        let mut w = scene2();
        for _ in 0..60 {
            w.step(1.0 / 60.0);
        }
        let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
        let mut r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
        for _ in 0..60 {
            w.step(1.0 / 60.0);
            r.step(1.0 / 60.0);
        }
        assert_eq!(hash_world2(&w), hash_world2(&r));
    }

    #[test]
    fn test_snapshot_fresh_mode_converges() {
        let mut w = scene2();
        for _ in 0..120 {
            w.step(1.0 / 60.0);
        }
        let bytes = world2_to_binary(&w, LoadMode::Fresh).expect("serializes");
        let mut r = world2_from_binary(&bytes, LoadMode::Fresh).expect("deserializes");
        assert!(r.contact_cache.is_empty());
        for _ in 0..120 {
            r.step(1.0 / 60.0);
        }
        for body in &r.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.velocity.x.is_finite());
        }
    }

    #[test]
    fn test_snapshot_rejects_hostile_input() {
        assert!(matches!(
            world2_from_json("{nope", LoadMode::Exact),
            Err(SnapshotError::Parse(_))
        ));
        assert!(matches!(
            world2_from_binary(b"junk", LoadMode::Exact),
            Err(SnapshotError::Parse(_))
        ));
        // Wrong version.
        let mut bytes = world2_to_binary(&scene2(), LoadMode::Exact).expect("serializes");
        bytes[4] = 99;
        // Fix the CRC so the failure is specifically the version gate.
        let len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
        let crc = crc32fast::hash(&bytes[13..13 + len]).to_le_bytes();
        bytes[13 + len..13 + len + 4].copy_from_slice(&crc);
        assert!(matches!(
            world2_from_binary(&bytes, LoadMode::Exact),
            Err(SnapshotError::Version(99))
        ));
        // Corrupt a payload byte (CRC catches it).
        let mut bytes = world2_to_binary(&scene2(), LoadMode::Exact).expect("serializes");
        bytes[20] ^= 0xFF;
        assert!(matches!(
            world2_from_binary(&bytes, LoadMode::Exact),
            Err(SnapshotError::Corrupt(_))
        ));
        // Truncate.
        let bytes = world2_to_binary(&scene2(), LoadMode::Exact).expect("serializes");
        assert!(matches!(
            world2_from_binary(&bytes[..bytes.len() / 2], LoadMode::Exact),
            Err(SnapshotError::Parse(_))
        ));
        // Hostile fluid sections: over-cap particle count and bad flag.
        fn fluid_bytes(count: u32, enabled: u8) -> Vec<u8> {
            let mut b = Vec::new();
            for _ in 0..8 {
                b.extend_from_slice(&1.0f32.to_le_bytes());
            }
            b.push(enabled);
            b.extend_from_slice(&count.to_le_bytes());
            b
        }
        let mut r = Reader {
            b: &fluid_bytes(5000, 1),
            pos: 0,
        };
        assert!(matches!(
            decode_fluid2(&mut r, SNAPSHOT_SCHEMA_GRAVITY),
            Err(SnapshotError::Parse(_))
        ));
        let mut r = Reader {
            b: &fluid_bytes(0, 2),
            pos: 0,
        };
        assert!(matches!(
            decode_fluid2(&mut r, SNAPSHOT_SCHEMA_GRAVITY),
            Err(SnapshotError::Parse(_))
        ));
        let mut r = Reader {
            b: &fluid_bytes(5000, 1),
            pos: 0,
        };
        // Same caps apply in 3D (fewer fields per particle, same guard).
        assert!(matches!(
            decode_fluid3(&mut r, SNAPSHOT_SCHEMA_GRAVITY),
            Err(SnapshotError::Parse(_))
        ));
    }

    #[test]
    fn test_snapshot_broadphase_mode_roundtrip_2d() {
        for mode in [
            phieng_broadphase::BroadphaseMode::BruteForce,
            phieng_broadphase::BroadphaseMode::SpatialHash,
            phieng_broadphase::BroadphaseMode::DynamicAABBTree,
            phieng_broadphase::BroadphaseMode::Hybrid,
        ] {
            let mut w = scene2();
            w.broadphase_mode = mode;
            w.broadphase = phieng_broadphase::Broadphase2::new(mode, 1.0);

            for fmt in [LoadMode::Exact, LoadMode::Fresh] {
                let text = world2_to_json(&w, fmt).expect("serializes");
                let back = world2_from_json(&text, fmt).expect("deserializes");
                assert_eq!(
                    back.broadphase_mode, mode,
                    "JSON must preserve mode {:?}",
                    mode
                );

                let bytes = world2_to_binary(&w, fmt).expect("serializes");
                let back = world2_from_binary(&bytes, fmt).expect("deserializes");
                assert_eq!(
                    back.broadphase_mode, mode,
                    "binary must preserve mode {:?}",
                    mode
                );
            }
        }
    }

    /// Schema v9: the collision-tier ceiling round-trips through JSON and
    /// binary, and a default world still writes the older schema.
    #[test]
    fn test_snapshot_collision_tier_roundtrip() {
        use crate::collision_tier::{CollisionTier, CollisionTierConfig};

        for tier in [
            CollisionTier::Tier1_AABBOnly,
            CollisionTier::Tier2_SphereAABB,
            CollisionTier::Tier3_SphereSphere,
            CollisionTier::Tier4_BoxBox,
            CollisionTier::Tier5_GJK_EPA,
        ] {
            for enabled in [true, false] {
                let mut w = scene2();
                w.collision_tier = CollisionTierConfig {
                    enabled,
                    max_tier: tier,
                };

                for fmt in [LoadMode::Exact, LoadMode::Fresh] {
                    let text = world2_to_json(&w, fmt).expect("serializes");
                    let back = world2_from_json(&text, fmt).expect("deserializes");
                    assert_eq!(back.collision_tier.enabled, enabled);
                    assert_eq!(back.collision_tier.max_tier, tier);

                    let bytes = world2_to_binary(&w, fmt).expect("serializes");
                    let back = world2_from_binary(&bytes, fmt).expect("deserializes");
                    assert_eq!(back.collision_tier.enabled, enabled);
                    assert_eq!(back.collision_tier.max_tier, tier);
                }
            }
        }
    }

    /// A default world keeps writing its old schema (byte-identical
    /// output), so existing files are untouched by the tier addition.
    #[test]
    fn test_snapshot_default_world_keeps_old_schema() {
        let w = scene2();
        let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
        assert!(
            text.contains("\"schema\": 3"),
            "default world must stay on its old schema, got: {}",
            text.lines().next().unwrap_or("")
        );
        assert!(
            !text.contains("collision_tier"),
            "default world must not carry a collision_tier key"
        );
    }

    /// A v8 payload (no collision-tier block) still loads, defaulting to
    /// full fidelity (Tier5) — not to a derived-zero Tier1.
    #[test]
    fn test_snapshot_v8_payload_loads() {
        let mut w = scene2();
        // Force the v8 schema by keeping every new field at its default.
        w.broadphase_mode = phieng_broadphase::BroadphaseMode::SpatialHash;
        w.thermal.config = crate::thermal::ThermalConfig2D::default();
        w.collision_tier = crate::collision_tier::CollisionTierConfig::default();
        // A world with only thermal + broadphase fields set writes v8.
        w.broadphase_mode = phieng_broadphase::BroadphaseMode::BruteForce;
        let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
        assert!(
            text.contains("\"schema\": 9"),
            "non-default world writes v9"
        );

        // A v8-shaped document (tier key absent) must load as full
        // fidelity.
        let stripped = text.replace(
            "\"collision_tier\": {\n      \"enabled\": true,\n      \"max_tier\": 1\n    },",
            "",
        );
        let back = world2_from_json(&stripped, LoadMode::Exact).expect("deserializes");
        assert!(!back.collision_tier.enabled);
        assert_eq!(
            back.collision_tier.max_tier,
            crate::collision_tier::CollisionTier::Tier5_GJK_EPA,
            "an absent tier block must default to full fidelity, not Tier1"
        );
    }

    #[test]
    fn test_snapshot_broadphase_mode_roundtrip_3d() {
        for mode in [
            phieng_broadphase::BroadphaseMode::BruteForce,
            phieng_broadphase::BroadphaseMode::DynamicAABBTree,
            phieng_broadphase::BroadphaseMode::Hybrid,
        ] {
            let mut w = World3::new();
            w.broadphase_mode = mode;
            w.broadphase = phieng_broadphase::Broadphase3::new(mode);

            for fmt in [LoadMode::Exact, LoadMode::Fresh] {
                let text = world3_to_json(&w, fmt).expect("serializes");
                let back = world3_from_json(&text, fmt).expect("deserializes");
                assert_eq!(
                    back.broadphase_mode, mode,
                    "JSON must preserve 3D mode {:?}",
                    mode
                );

                let bytes = world3_to_binary(&w, fmt).expect("serializes");
                let back = world3_from_binary(&bytes, fmt).expect("deserializes");
                assert_eq!(
                    back.broadphase_mode, mode,
                    "binary must preserve 3D mode {:?}",
                    mode
                );
            }
        }
    }

    #[test]
    fn test_snapshot_default_world_omits_mode_key() {
        // Default worlds keep their old schema and carry no mode key.
        let w = scene2();
        let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
        assert!(text.contains("\"schema\": 3"), "default scene stays v3");
        assert!(
            !text.contains("broadphase_mode"),
            "default worlds must not carry a mode key"
        );
    }

    #[test]
    fn test_snapshot_rejects_unknown_mode_tag() {
        // Unknown mode tags are Corrupt, never a silent default. JSON and
        // binary both funnel through `world2_from_dto`, so the JSON path
        // pins the validation for both.
        assert!(matches!(
            broadphase_mode_from_tag(99),
            Err(SnapshotError::Corrupt(_))
        ));

        let hacked: World2Dto = {
            let dto = world2_dto(&scene2(), LoadMode::Exact);
            World2Dto {
                broadphase_mode: 99,
                ..dto
            }
        };
        let hacked_text = serde_json::to_string(&hacked).expect("serializes");
        assert!(matches!(
            world2_from_json(&hacked_text, LoadMode::Exact),
            Err(SnapshotError::Corrupt(_))
        ));
    }
}
