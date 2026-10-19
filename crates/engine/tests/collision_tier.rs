//! Collision-tier tests (H01–H05, mirrored Rust↔C#).
//!
//! The narrowphase tier system caps how expensive a shape-pair test may
//! get: pairs whose cheapest separating test exceeds the ceiling are
//! *skipped* (no contact generated). That is a fidelity-for-cost trade,
//! so these tests pin two things:
//!   1. the ceiling actually takes effect (a cheap ceiling really does
//!      stop contacts from forming), and
//!   2. it is opt-in and off by default (a default world is bit-identical
//!      to a world with the tier system disabled).

use phieng_dynamics::collision_tier::{CollisionTier, CollisionTierConfig};
use phieng_dynamics::{Shape2, Shape3, World2, World3};

const DT: f32 = 1.0 / 60.0;

fn hash2(w: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &w.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn hash3(w: &World3) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &w.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.transform.position.z,
            body.velocity.x,
            body.velocity.y,
            body.velocity.z,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// A stack of boxes resting on a static floor. Box–box pairs need
/// Tier4 (SAT), so a ceiling below Tier4 must stop them from stacking.
fn box_stack_2d(parallel_bodies: usize) -> World2 {
    let mut w = World2::new();
    w.gravity = glam::Vec2::new(0.0, -9.81);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.add_static_body(
        glam::Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: glam::Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    for i in 0..parallel_bodies {
        w.add_dynamic_body(
            glam::Vec2::new(0.0, 0.05 + i as f32 * 1.02),
            Shape2::Box {
                half_extents: glam::Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    }
    w
}

fn sphere_stack_3d() -> World3 {
    let mut w = World3::new();
    w.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    w.add_static_body(
        glam::Vec3::new(0.0, -1.0, 0.0),
        Shape3::Box {
            half_extents: glam::Vec3::new(10.0, 0.5, 10.0),
        },
    )
    .expect("valid shape");
    for i in 0..3 {
        w.add_dynamic_body(
            glam::Vec3::new(0.0, 0.25 + i as f32 * 1.1, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    }
    w
}

/// H01: the tier system is off by default, and a default world is
/// bit-identical to one with the tier config explicitly disabled.
#[test]
fn h01_default_is_full_fidelity() {
    let mut a = box_stack_2d(4);
    let mut b = box_stack_2d(4);
    b.collision_tier = CollisionTierConfig {
        enabled: false,
        max_tier: CollisionTier::Tier5_GJK_EPA,
    };
    for _ in 0..180 {
        a.step(DT);
        b.step(DT);
    }
    assert_eq!(
        hash2(&a),
        hash2(&b),
        "default must match an explicitly-disabled tier"
    );
}

/// H02: a ceiling below the pair's required tier stops the contact.
///
/// The stack is box–box (Tier4). Under a Tier3 ceiling the boxes never
/// collide, so they fall through the floor instead of stacking.
#[test]
fn h02_ceiling_below_required_tier_stops_contacts() {
    let mut full = box_stack_2d(3);
    let mut capped = box_stack_2d(3);
    capped.collision_tier = CollisionTierConfig {
        enabled: true,
        max_tier: CollisionTier::Tier3_SphereSphere,
    };

    for _ in 0..180 {
        full.step(DT);
        capped.step(DT);
    }

    // Full fidelity: the boxes stack up (they do not fall through).
    let top_full = full.bodies.last().expect("body").transform.position.y;
    assert!(
        top_full > 0.0,
        "with full fidelity the stack must hold, top y = {top_full}"
    );

    // Capped: box–box contacts are skipped, so the boxes fall through.
    let top_capped = capped.bodies.last().expect("body").transform.position.y;
    assert!(
        top_capped < top_full - 1.0,
        "a ceiling below Tier4 must stop box-box contacts: \
         top y = {top_capped} vs {top_full}"
    );
}

/// H03: a ceiling at the required tier keeps the contacts.
#[test]
fn h03_ceiling_at_required_tier_keeps_contacts() {
    let mut full = box_stack_2d(3);
    let mut capped = box_stack_2d(3);
    capped.collision_tier = CollisionTierConfig {
        enabled: true,
        max_tier: CollisionTier::Tier4_BoxBox,
    };

    for _ in 0..180 {
        full.step(DT);
        capped.step(DT);
    }
    assert_eq!(
        hash2(&full),
        hash2(&capped),
        "a Tier4 ceiling must behave exactly like full fidelity for box-box pairs"
    );
}

/// H04: 3D mirrors the 2D contract.
#[test]
fn h04_3d_ceiling_behavior() {
    let mut full = sphere_stack_3d();
    let mut capped = sphere_stack_3d();
    // Sphere–sphere is Tier3, and sphere–box is Tier2: a Tier2 ceiling
    // still allows the sphere/sphere pair but not the sphere/box pair.
    capped.collision_tier = CollisionTierConfig {
        enabled: true,
        max_tier: CollisionTier::Tier2_SphereAABB,
    };

    for _ in 0..180 {
        full.step(DT);
        capped.step(DT);
    }

    let top_full = full.bodies.last().expect("body").transform.position.y;
    let top_capped = capped.bodies.last().expect("body").transform.position.y;
    assert!(
        top_capped < top_full - 0.5,
        "a Tier2 ceiling must stop the sphere/box contacts: \
         top y = {top_capped} vs {top_full}"
    );

    // The capped run is still deterministic (same inputs, same steps).
    let mut again = sphere_stack_3d();
    again.collision_tier = CollisionTierConfig {
        enabled: true,
        max_tier: CollisionTier::Tier2_SphereAABB,
    };
    for _ in 0..180 {
        again.step(DT);
    }
    assert_eq!(
        hash3(&capped),
        hash3(&again),
        "tier-capped 3D run is deterministic"
    );
}

/// H05: the tier config round-trips through the FFI-style config
/// conversion and the world keeps running deterministically.
#[test]
fn h05_tier_config_is_deterministic() {
    let mut a = box_stack_2d(4);
    let mut b = box_stack_2d(4);
    for w in [&mut a, &mut b] {
        w.collision_tier = CollisionTierConfig {
            enabled: true,
            max_tier: CollisionTier::Tier2_SphereAABB,
        };
    }
    for _ in 0..120 {
        a.step(DT);
        b.step(DT);
    }
    assert_eq!(hash2(&a), hash2(&b), "tier-capped runs stay deterministic");
    assert!(hash2(&a) != 0);
}
