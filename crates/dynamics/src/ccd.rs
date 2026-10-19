//! Continuous collision detection (CCD) via adaptive substepping.
//!
//! When enabled, [`World2::step`](crate::World2::step) /
//! [`World3::step`](crate::World3::step) measure the fastest effective
//! body speed (linear plus angular sweep) and split the step into enough
//! substeps that no body travels more than [`CcdConfig::max_movement`]
//! per substep. Each substep runs the full pipeline (gravity, ambient,
//! fluid, broadphase, narrowphase, solve, integrate), so fast bodies can
//! no longer tunnel through thin walls within a single frame.
//!
//! CCD *tiers* control how aggressively tunneling is prevented:
//! - **Off**: CCD disabled entirely (legacy behavior).
//! - **SweptAABB**: Substep only body pairs whose swept AABBs overlap
//!   during the step — cheap broadphase culling for dense scenes where
//!   most bodies never get near each other.
//! - **FullGJK**: Full substepping on every body — maximum safety, highest
//!   cost (current legacy behavior when CCD is enabled).
//!
//! The default config is a complete no-op (one substep, bit-identical
//! trajectories). Non-default configs persist through snapshots at
//! schema v6 and cross the FFI bridge.

use glam::{Vec2, Vec3};

/// CCD tier: controls how aggressively tunneling is prevented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CcdTier {
    /// CCD disabled entirely (default).
    #[default]
    Off,
    /// Substep only pairs whose swept AABBs overlap during the step.
    /// Cheap broadphase culling: fast movers are caught, distant bodies
    /// pay no per-substep cost.
    SweptAABB,
    /// Full substepping on every body — maximum safety, highest cost.
    FullGJK,
}

/// CCD configuration, shared by 2D and 3D worlds. Dimension-agnostic:
/// only speeds, distances, and counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CcdConfig {
    /// Master switch. `false` (default) = single substep, legacy behavior.
    pub enabled: bool,
    /// CCD tier: which pairs/bodies receive substepping.
    pub tier: CcdTier,
    /// Maximum distance any body may travel within one substep. Must be
    /// smaller than the thinnest collider the scene relies on; the
    /// default (`0.1`) suits decimeter-scale scenes at 60 Hz.
    pub max_movement: f32,
    /// Hard cap on substeps per world step (cost guard). Must be `>= 1`.
    pub max_substeps: u32,
}

impl Default for CcdConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tier: CcdTier::Off,
            max_movement: 0.1,
            max_substeps: 8,
        }
    }
}

impl CcdConfig {
    /// True for the default (legacy, single-substep) configuration.
    /// Snapshot schema selection uses this: default configs never
    /// trigger a schema bump.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// True when substepping can actually engage.
    pub fn is_active(&self) -> bool {
        self.enabled
            && self.tier != CcdTier::Off
            && self.max_substeps > 1
            && self.max_movement.is_finite()
            && self.max_movement > 0.0
    }

    /// Number of substeps for a step of length `dt` given the peak
    /// effective body speed. Always `1` when inactive, on non-positive
    /// `dt`, or on non-finite/non-positive speed.
    pub fn substeps(&self, dt: f32, peak_speed: f32) -> u32 {
        if !self.is_active() || dt <= 0.0 || !peak_speed.is_finite() || peak_speed <= 0.0 {
            return 1;
        }
        let max = self.max_substeps.max(1);
        let need = (peak_speed * dt / self.max_movement).ceil() as i64;
        need.clamp(1, max as i64) as u32
    }
}

/// Map a [`SolverQuality`] preset to a CCD tier:
/// - `Ultra` → FullGJK (maximum safety)
/// - `High` → SweptAABB (good balance)
/// - `Medium` / `Low` / `Minimal` → Off (cost-sensitive)
pub fn ccd_tier_from_quality(q: crate::solver::SolverQuality) -> CcdTier {
    use crate::solver::SolverQuality as Q;
    match q {
        Q::Ultra => CcdTier::FullGJK,
        Q::High => CcdTier::SweptAABB,
        Q::Medium => CcdTier::Off,
        Q::Low => CcdTier::Off,
        Q::Minimal => CcdTier::Off,
    }
}

/// Fast swept-AABB overlap test between two bodies over one step.
///
/// Approximates each body's swept volume by extruding its AABB along its
/// velocity vector (plus angular sweep at extents). Returns true if the
/// swept volumes overlap — meaning the pair *might* tunnel and should
/// receive substepping.
#[allow(clippy::too_many_arguments)]
pub fn swept_aabb_overlap(
    pos_a: Vec2,
    vel_a: Vec2,
    ang_a: f32,
    extent_a: f32,
    pos_b: Vec2,
    vel_b: Vec2,
    ang_b: f32,
    extent_b: f32,
    dt: f32,
) -> bool {
    // Sweep each body's center by its velocity.
    let a_end = pos_a + vel_a * dt;
    let b_end = pos_b + vel_b * dt;

    // Conservative swept AABB: union of start and end AABBs, expanded
    // by angular sweep (rotation at extents).
    let ext_a = extent_a.max(0.0) + ang_a.abs() * extent_a * dt;
    let ext_b = extent_b.max(0.0) + ang_b.abs() * extent_b * dt;

    let a_min = pos_a.min(a_end) - Vec2::splat(ext_a);
    let a_max = pos_a.max(a_end) + Vec2::splat(ext_a);
    let b_min = pos_b.min(b_end) - Vec2::splat(ext_b);
    let b_max = pos_b.max(b_end) + Vec2::splat(ext_b);

    a_min.x <= b_max.x && a_max.x >= b_min.x && a_min.y <= b_max.y && a_max.y >= b_min.y
}

/// 3D swept-AABB overlap: same idea with Vec3.
#[allow(clippy::too_many_arguments)]
pub fn swept_aabb_overlap_3(
    pos_a: Vec3,
    vel_a: Vec3,
    ang_a: f32,
    extent_a: f32,
    pos_b: Vec3,
    vel_b: Vec3,
    ang_b: f32,
    extent_b: f32,
    dt: f32,
) -> bool {
    let a_end = pos_a + vel_a * dt;
    let b_end = pos_b + vel_b * dt;

    let ext_a = extent_a.max(0.0) + ang_a.abs() * extent_a * dt;
    let ext_b = extent_b.max(0.0) + ang_b.abs() * extent_b * dt;

    let a_min = pos_a.min(a_end) - Vec3::splat(ext_a);
    let a_max = pos_a.max(a_end) + Vec3::splat(ext_a);
    let b_min = pos_b.min(b_end) - Vec3::splat(ext_b);
    let b_max = pos_b.max(b_end) + Vec3::splat(ext_b);

    a_min.x <= b_max.x
        && a_max.x >= b_min.x
        && a_min.y <= b_max.y
        && a_max.y >= b_min.y
        && a_min.z <= b_max.z
        && a_max.z >= b_min.z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccd_config_defaults_single_step() {
        let cfg = CcdConfig::default();
        assert!(cfg.is_default());
        assert!(!cfg.is_active());
        assert_eq!(cfg.substeps(1.0 / 60.0, 100.0), 1);
    }

    #[test]
    fn ccd_substep_count_scales_with_speed() {
        let cfg = CcdConfig {
            enabled: true,
            tier: CcdTier::FullGJK,
            max_movement: 0.1,
            max_substeps: 8,
        };
        assert!(cfg.is_active());
        // 5 units/s at 60 Hz = 0.083/step < 0.1 -> exactly 1 substep.
        assert_eq!(cfg.substeps(1.0 / 60.0, 5.0), 1);
        // 11 units/s -> 0.183/step -> 2 substeps.
        assert_eq!(cfg.substeps(1.0 / 60.0, 11.0), 2);
        // Extreme speed clamps at the cap instead of exploding cost.
        assert_eq!(cfg.substeps(1.0 / 60.0, 1.0e6), 8);
    }

    #[test]
    fn ccd_rejects_degenerate_inputs() {
        let cfg = CcdConfig {
            enabled: true,
            tier: CcdTier::FullGJK,
            max_movement: 0.1,
            max_substeps: 8,
        };
        assert_eq!(cfg.substeps(0.0, 10.0), 1);
        assert_eq!(cfg.substeps(-1.0, 10.0), 1);
        assert_eq!(cfg.substeps(1.0 / 60.0, f32::NAN), 1);
        assert_eq!(cfg.substeps(1.0 / 60.0, f32::INFINITY), 1);
        let off = CcdConfig {
            enabled: true,
            tier: CcdTier::Off,
            max_movement: 0.1,
            max_substeps: 8,
        };
        assert!(!off.is_active());
        let one = CcdConfig {
            enabled: true,
            tier: CcdTier::FullGJK,
            max_movement: 0.1,
            max_substeps: 1,
        };
        assert!(!one.is_active());
    }

    #[test]
    fn swept_aabb_overlap_detects_approach() {
        // Two bodies 4 units apart closing at 120 units/s each: they
        // travel ~2 units per 1/60 s and their swept volumes overlap.
        let hit = swept_aabb_overlap(
            Vec2::new(-2.0, 0.0),
            Vec2::new(120.0, 0.0),
            0.0,
            0.5,
            Vec2::new(2.0, 0.0),
            Vec2::new(-120.0, 0.0),
            0.0,
            0.5,
            1.0 / 60.0,
        );
        assert!(hit);
    }

    #[test]
    fn swept_aabb_overlap_ignores_slow_far_bodies() {
        // Same positions but slow: swept volumes never meet.
        let hit = swept_aabb_overlap(
            Vec2::new(-2.0, 0.0),
            Vec2::new(1.0, 0.0),
            0.0,
            0.5,
            Vec2::new(2.0, 0.0),
            Vec2::new(-1.0, 0.0),
            0.0,
            0.5,
            1.0 / 60.0,
        );
        assert!(!hit);
    }

    #[test]
    fn swept_aabb_no_overlap_when_separated() {
        // Two bodies moving apart: swept AABBs do not overlap.
        let hit = swept_aabb_overlap(
            Vec2::new(-2.0, 0.0),
            Vec2::new(-2.0, 0.0),
            0.0,
            0.5,
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 0.0),
            0.0,
            0.5,
            1.0 / 60.0,
        );
        assert!(!hit);
    }

    #[test]
    fn solver_quality_maps_to_tier() {
        use crate::solver::SolverQuality;
        assert_eq!(
            ccd_tier_from_quality(SolverQuality::Ultra),
            CcdTier::FullGJK
        );
        assert_eq!(
            ccd_tier_from_quality(SolverQuality::High),
            CcdTier::SweptAABB
        );
        assert_eq!(ccd_tier_from_quality(SolverQuality::Medium), CcdTier::Off);
        assert_eq!(ccd_tier_from_quality(SolverQuality::Low), CcdTier::Off);
        assert_eq!(ccd_tier_from_quality(SolverQuality::Minimal), CcdTier::Off);
    }
}
