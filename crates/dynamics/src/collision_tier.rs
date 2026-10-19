//! Collision detection tier system.
//!
//! A configurable early-exit ladder for the narrowphase: every shape-pair
//! test is assigned the cheapest tier that can separate it, and a world
//! carries a `max_tier` ceiling. Pairs whose required tier exceeds the
//! ceiling are *skipped* (treated as non-colliding) — never guessed, just
//! ignored — so low-end devices pay only the cost they can afford.
//!
//! Tiers (cheapest → most expensive):
//! - `Tier1_AABBOnly`: broadphase AABB overlap only (no narrowphase work).
//! - `Tier2_SphereAABB`: analytic sphere-vs-box (sphere center to box
//!   closest point).
//! - `Tier3_SphereSphere`: analytic sphere-vs-sphere (2D: circle-circle).
//! - `Tier4_BoxBox`: separating-axis test (SAT, OBB-OBB in 3D, SAT for
//!   2D polygons/boxes).
//! - `Tier5_GJK_EPA`: general convex-vs-convex (GJK/EPA, hulls/meshes).
//!
//! Default is `Tier5` (full fidelity) so existing behavior is unchanged.

use super::body::{Shape2, Shape3};

/// Narrowphase collision tier.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum CollisionTier {
    #[default]
    Tier1_AABBOnly = 0,
    Tier2_SphereAABB = 1,
    Tier3_SphereSphere = 2,
    Tier4_BoxBox = 3,
    Tier5_GJK_EPA = 4,
}

impl CollisionTier {
    /// Map a `SolverQuality` preset to its collision ceiling. Mirrors the
    /// optimization plan table: Minimal only does AABB, Low skips GJK/EPA,
    /// Medium skips GJK/EPA, High does SAT, Ultra does everything.
    pub fn from_solver_quality(q: crate::solver::SolverQuality) -> Self {
        use crate::solver::SolverQuality as Q;
        match q {
            Q::Minimal => CollisionTier::Tier1_AABBOnly,
            Q::Low => CollisionTier::Tier3_SphereSphere,
            Q::Medium => CollisionTier::Tier4_BoxBox,
            Q::High => CollisionTier::Tier4_BoxBox,
            Q::Ultra => CollisionTier::Tier5_GJK_EPA,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CollisionTier::Tier1_AABBOnly => "Tier1_AABBOnly",
            CollisionTier::Tier2_SphereAABB => "Tier2_SphereAABB",
            CollisionTier::Tier3_SphereSphere => "Tier3_SphereSphere",
            CollisionTier::Tier4_BoxBox => "Tier4_BoxBox",
            CollisionTier::Tier5_GJK_EPA => "Tier5_GJK_EPA",
        }
    }
}

/// Per-world collision-tier configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionTierConfig {
    /// Master switch. When false the tier ceiling is ignored (full fidelity),
    /// exactly like `ccd`/`thermal` being disabled.
    pub enabled: bool,
    /// Highest tier the narrowphase is allowed to run. Pairs whose required
    /// tier exceeds this are skipped (no collision generated).
    pub max_tier: CollisionTier,
}

impl Default for CollisionTierConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_tier: CollisionTier::Tier5_GJK_EPA,
        }
    }
}

impl CollisionTierConfig {
    pub fn is_active(&self) -> bool {
        self.enabled
    }
}

/// Required tier for a 2D shape pair. The narrowphase skips the pair when
/// this exceeds the configured ceiling.
pub fn required_tier_2d(a: &Shape2, b: &Shape2) -> CollisionTier {
    use Shape2 as S;
    match (a, b) {
        // Circle vs Circle: analytic sphere-sphere.
        (S::Circle { .. }, S::Circle { .. }) => CollisionTier::Tier3_SphereSphere,
        // Circle vs Box / Box vs Circle: sphere-vs-AABB.
        (S::Circle { .. }, S::Box { .. }) | (S::Box { .. }, S::Circle { .. }) => {
            CollisionTier::Tier2_SphereAABB
        }
        // Box vs Box: SAT.
        (S::Box { .. }, S::Box { .. }) => CollisionTier::Tier4_BoxBox,
        // Anything involving a polygon: full SAT / convex.
        (S::Polygon { .. }, _) | (_, S::Polygon { .. }) => CollisionTier::Tier4_BoxBox,
    }
}

/// Required tier for a 3D shape pair.
pub fn required_tier_3d(a: &Shape3, b: &Shape3) -> CollisionTier {
    use Shape3 as S;
    match (a, b) {
        (S::Sphere { .. }, S::Sphere { .. }) => CollisionTier::Tier3_SphereSphere,
        (S::Sphere { .. }, S::Box { .. }) | (S::Box { .. }, S::Sphere { .. }) => {
            CollisionTier::Tier2_SphereAABB
        }
        (S::Sphere { .. }, S::Capsule { .. }) | (S::Capsule { .. }, S::Sphere { .. }) => {
            CollisionTier::Tier2_SphereAABB
        }
        (S::Box { .. }, S::Box { .. }) => CollisionTier::Tier4_BoxBox,
        (S::Box { .. }, S::Capsule { .. }) | (S::Capsule { .. }, S::Box { .. }) => {
            CollisionTier::Tier4_BoxBox
        }
        (S::Capsule { .. }, S::Capsule { .. }) => CollisionTier::Tier4_BoxBox,
        // Convex hulls / meshes require GJK/EPA.
        (S::ConvexHull { .. }, _) | (_, S::ConvexHull { .. }) => CollisionTier::Tier5_GJK_EPA,
        (S::ConvexMesh { .. }, _) | (_, S::ConvexMesh { .. }) => CollisionTier::Tier5_GJK_EPA,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{Shape2, Shape3};

    #[test]
    fn tier_ordering_is_monotonic() {
        assert!(CollisionTier::Tier1_AABBOnly < CollisionTier::Tier2_SphereAABB);
        assert!(CollisionTier::Tier2_SphereAABB < CollisionTier::Tier3_SphereSphere);
        assert!(CollisionTier::Tier3_SphereSphere < CollisionTier::Tier4_BoxBox);
        assert!(CollisionTier::Tier4_BoxBox < CollisionTier::Tier5_GJK_EPA);
    }

    #[test]
    fn default_config_is_inactive() {
        let c = CollisionTierConfig::default();
        assert!(!c.is_active());
        assert_eq!(c.max_tier, CollisionTier::Tier5_GJK_EPA);
    }

    #[test]
    fn required_tier_2d_classifies() {
        assert_eq!(
            required_tier_2d(
                &Shape2::Circle { radius: 1.0 },
                &Shape2::Circle { radius: 1.0 }
            ),
            CollisionTier::Tier3_SphereSphere
        );
        assert_eq!(
            required_tier_2d(
                &Shape2::Circle { radius: 1.0 },
                &Shape2::Box {
                    half_extents: glam::Vec2::new(1.0, 1.0)
                }
            ),
            CollisionTier::Tier2_SphereAABB
        );
        assert_eq!(
            required_tier_2d(
                &Shape2::Box {
                    half_extents: glam::Vec2::new(1.0, 1.0)
                },
                &Shape2::Box {
                    half_extents: glam::Vec2::new(1.0, 1.0)
                }
            ),
            CollisionTier::Tier4_BoxBox
        );
    }

    #[test]
    fn required_tier_3d_classifies() {
        assert_eq!(
            required_tier_3d(
                &Shape3::Sphere { radius: 1.0 },
                &Shape3::Sphere { radius: 1.0 }
            ),
            CollisionTier::Tier3_SphereSphere
        );
        assert_eq!(
            required_tier_3d(
                &Shape3::Sphere { radius: 1.0 },
                &Shape3::Box {
                    half_extents: glam::Vec3::new(1.0, 1.0, 1.0)
                }
            ),
            CollisionTier::Tier2_SphereAABB
        );
        assert_eq!(
            required_tier_3d(
                &Shape3::ConvexMesh {
                    vertices: vec![glam::Vec3::ZERO],
                    indices: vec![0, 0, 0]
                },
                &Shape3::Sphere { radius: 1.0 }
            ),
            CollisionTier::Tier5_GJK_EPA
        );
    }

    #[test]
    fn solver_quality_maps_to_tier() {
        use crate::solver::SolverQuality;
        assert_eq!(
            CollisionTier::from_solver_quality(SolverQuality::Minimal),
            CollisionTier::Tier1_AABBOnly
        );
        assert_eq!(
            CollisionTier::from_solver_quality(SolverQuality::Ultra),
            CollisionTier::Tier5_GJK_EPA
        );
    }
}
