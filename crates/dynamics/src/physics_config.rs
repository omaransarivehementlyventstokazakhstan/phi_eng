//! Unified physics configuration facade.
//!
//! Brings together every subsystem tunable (solver quality, collision tier,
//! broadphase mode, sleep, CCD, thermal) under one struct so a caller can
//! configure a world in one shot instead of touching each subsystem
//! separately. Defaults match the pre-tier-system behavior exactly:
//! `SolverQuality::High`, `CollisionTier::Tier5_GJK_EPA` (inactive),
//! `BroadphaseMode::SpatialHash`, sleep on, CCD off, thermal off.
//!
//! Per-subsystem structs remain the source of truth; this facade only
//! *applies* them. Worlds are not required to use it — direct field
//! assignment still works and is bit-identical.

use crate::ccd::CcdConfig;
use crate::collision_tier::{CollisionTier, CollisionTierConfig};
use crate::solver::SolverQuality;
use crate::thermal::ThermalConfig2D;
use phieng_broadphase::BroadphaseMode;

/// Unified physics configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsConfig {
    pub solver_quality: SolverQuality,
    pub collision_tier: CollisionTier,
    pub broadphase_mode: BroadphaseMode,
    pub enable_sleep: bool,
    pub linear_sleep_threshold: f32,
    pub angular_sleep_threshold: f32,
    pub ccd: CcdConfig,
    pub thermal: ThermalConfig2D,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            solver_quality: SolverQuality::High,
            collision_tier: CollisionTier::Tier5_GJK_EPA,
            broadphase_mode: BroadphaseMode::SpatialHash,
            enable_sleep: true,
            linear_sleep_threshold: phieng_math::LINEAR_SLEEP_THRESHOLD,
            angular_sleep_threshold: phieng_math::ANGULAR_SLEEP_THRESHOLD,
            ccd: CcdConfig::default(),
            thermal: ThermalConfig2D::default(),
        }
    }
}

impl PhysicsConfig {
    /// Auto-detect a sensible config for a low-end device (1-2 cores,
    /// 512MB-1GB RAM, no SIMD). Mirrors the optimization plan table.
    pub fn low_end() -> Self {
        Self {
            solver_quality: SolverQuality::Low,
            collision_tier: CollisionTier::Tier3_SphereSphere,
            broadphase_mode: BroadphaseMode::Hybrid,
            enable_sleep: true,
            linear_sleep_threshold: 0.05,
            angular_sleep_threshold: 0.05,
            ccd: CcdConfig::default(),
            thermal: ThermalConfig2D::default(),
        }
    }

    /// High-fidelity config for desktop / unconstrained devices.
    pub fn high_fidelity() -> Self {
        Self {
            solver_quality: SolverQuality::Ultra,
            collision_tier: CollisionTier::Tier5_GJK_EPA,
            broadphase_mode: BroadphaseMode::SpatialHash,
            enable_sleep: true,
            linear_sleep_threshold: 0.05,
            angular_sleep_threshold: 0.05,
            ccd: CcdConfig::default(),
            thermal: ThermalConfig2D::default(),
        }
    }

    /// Apply this config to a 2D world. Idempotent and deterministic.
    pub fn apply_to_2d(&self, w: &mut crate::World2) {
        w.solver.apply_quality(self.solver_quality);
        w.collision_tier = CollisionTierConfig {
            enabled: self.collision_tier != CollisionTier::Tier5_GJK_EPA,
            max_tier: self.collision_tier,
        };
        w.broadphase_mode = self.broadphase_mode;
        w.broadphase = phieng_broadphase::Broadphase2::new(self.broadphase_mode, 1.0);
        w.linear_sleep_threshold = if self.enable_sleep {
            self.linear_sleep_threshold
        } else {
            0.0
        };
        w.angular_sleep_threshold = if self.enable_sleep {
            self.angular_sleep_threshold
        } else {
            0.0
        };
        w.ccd = self.ccd;
        w.thermal.config = self.thermal;
    }

    /// Apply this config to a 3D world. Idempotent and deterministic.
    pub fn apply_to_3d(&self, w: &mut crate::World3) {
        w.solver.apply_quality(self.solver_quality);
        w.collision_tier = CollisionTierConfig {
            enabled: self.collision_tier != CollisionTier::Tier5_GJK_EPA,
            max_tier: self.collision_tier,
        };
        w.broadphase_mode = self.broadphase_mode;
        w.broadphase = phieng_broadphase::Broadphase3::new(self.broadphase_mode);
        w.linear_sleep_threshold = if self.enable_sleep {
            self.linear_sleep_threshold
        } else {
            0.0
        };
        w.angular_sleep_threshold = if self.enable_sleep {
            self.angular_sleep_threshold
        } else {
            0.0
        };
        w.ccd = self.ccd;
        w.thermal.config = self.thermal;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{World2, World3};

    #[test]
    fn default_matches_pre_tier_system() {
        let cfg = PhysicsConfig::default();
        assert_eq!(cfg.solver_quality, crate::solver::SolverQuality::High);
        assert_eq!(cfg.collision_tier, CollisionTier::Tier5_GJK_EPA);
        assert_eq!(cfg.broadphase_mode, BroadphaseMode::SpatialHash);
        assert!(cfg.enable_sleep);
        assert!(!cfg.ccd.is_active());
        assert!(!cfg.thermal.is_active());
    }

    #[test]
    fn apply_to_2d_round_trips() {
        let mut w = World2::new();
        let cfg = PhysicsConfig::low_end();
        cfg.apply_to_2d(&mut w);
        assert!(!w.solver.enable_friction);
        assert!(w.collision_tier.is_active());
        assert_eq!(w.broadphase_mode, BroadphaseMode::Hybrid);
        assert_eq!(w.linear_sleep_threshold, 0.05);
    }

    #[test]
    fn apply_to_3d_round_trips() {
        let mut w = World3::new();
        let cfg = PhysicsConfig::high_fidelity();
        cfg.apply_to_3d(&mut w);
        assert!(w.solver.enable_friction);
        assert!(!w.collision_tier.is_active());
        assert_eq!(w.broadphase_mode, BroadphaseMode::SpatialHash);
    }

    #[test]
    fn low_end_and_high_fidelity_differ() {
        assert_ne!(PhysicsConfig::low_end(), PhysicsConfig::high_fidelity());
    }
}
