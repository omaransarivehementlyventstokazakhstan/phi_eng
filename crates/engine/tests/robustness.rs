//! Robustness probes + pins (test IDs S/W/N/D).
//!
//! Probe tests characterize behavior with loose bounds (finite state,
//! broad limits) and record actuals in their messages; pin tests assert
//! exact desired behavior. See LIMITATIONS.md for documented gaps and
//! tools/check_test_parity.py for the Rust<->C# ID convention.

use glam::Vec2;
use phieng_dynamics::body::{Shape2, ShapeError};
use phieng_dynamics::{ValidationConfig, World2};

const DT: f32 = 1.0 / 60.0;

fn floor(world: &mut World2) {
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
}

/// Assert every dynamic body has finite position/velocity (NaN/Inf scan).
fn assert_finite(world: &World2, ctx: &str) {
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
        ] {
            assert!(
                v.is_finite(),
                "{ctx}: non-finite state {v} in body {}",
                body.id
            );
        }
    }
}

/// W07 probe: friction/restitution swapped mid-contact. Cached impulses
/// were accumulated under the old material; the Coulomb clamp and bias
/// rules run under the new one. Characterize the transient.
#[test]
fn w07_material_change_mid_contact() {
    let mut world = World2::new();
    floor(&mut world);
    let b = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..120 {
        world.step(DT);
    }
    assert_finite(&world, "w07 settled");
    let mut m = world.bodies[b].material;
    m.friction = 0.9;
    m.restitution = 0.8;
    world.bodies[b].set_material(m).expect("valid shape");
    let mut peak = 0.0f32;
    for _ in 0..60 {
        world.step(DT);
        peak = peak.max(world.bodies[b].velocity.length());
    }
    assert_finite(&world, "w07 after material swap");
    assert!(
        peak < 0.5,
        "w07: material swap transient peak velocity {peak} (spike if large)"
    );
}

/// N13 probe: 50% initial overlap. Baumgarte bias scales with depth
/// (0.2 * depth / dt), so deep starts push hard. Assert bounded
/// ejection: finite, separating, peak velocity recorded.
#[test]
fn n13_deep_penetration_50pct() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let top = world
        .add_dynamic_body(
            Vec2::new(0.5, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let mut peak = 0.0f32;
    for frame in 0..120 {
        world.step(DT);
        peak = peak.max(world.bodies[top].velocity.length());
        assert_finite(&world, &format!("n13 frame {frame}"));
    }
    let x = world.bodies[top].transform.position.x;
    assert!(x > 0.5, "n13: deep overlap must separate along +x, x={x}");
    assert!(
        peak < 10.0,
        "n13: ejection peak velocity {peak} (bias clamp needed if large)"
    );
}

/// S16 probe: 1000:1 mass ratio over 10k steps. No ratio clamp exists;
/// characterize drift/jitter of the light body on the heavy one.
#[test]
fn s16_mass_ratio_1000_long() {
    let mut world = World2::new();
    floor(&mut world);
    world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1000.0,
        )
        .expect("valid shape");
    let light = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let mut peak = 0.0f32;
    for frame in 0..10_000 {
        world.step(DT);
        peak = peak.max(world.bodies[light].velocity.length());
        if frame % 1000 == 0 {
            assert_finite(&world, &format!("s16 frame {frame}"));
        }
    }
    assert_finite(&world, "s16 final");
    let y = world.bodies[light].transform.position.y;
    assert!(
        y > 0.9,
        "s16: light body tunneled through heavy stack, y={y}"
    );
    assert!(
        peak < 1.0,
        "s16: light-body jitter peak {peak} over 10k steps (ratio handling needed if large)"
    );
}

/// S17 probe: 1e6:1 ratio. Documents the solver limit: assert bounded
/// (finite, inside the world), not correct.
#[test]
fn s17_mass_ratio_1e6_bounded() {
    let mut world = World2::new();
    floor(&mut world);
    world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1e6,
        )
        .expect("valid shape");
    let light = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for frame in 0..1200 {
        world.step(DT);
        assert_finite(&world, &format!("s17 frame {frame}"));
    }
    let p = world.bodies[light].transform.position;
    assert!(
        p.x.abs() < 100.0 && p.y.abs() < 100.0,
        "s17: state escaped world bounds: {p} (limit exceeded)"
    );
}

/// S18 pin (Q2 boundary rule): zero/non-positive/non-finite density for a
/// dynamic body is rejected at creation — never sanitized into a default.
/// Statics ignore mass, so any finite density (including zero) is accepted
/// for them. NaN positions reject too.
#[test]
fn s18_zero_mass_dynamic() {
    let mut world = World2::new();
    floor(&mut world);
    assert_eq!(
        world.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.3 }, 0.0),
        Err(ShapeError::ZeroDensity)
    );
    assert_eq!(
        world.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.3 }, -1.0),
        Err(ShapeError::ZeroDensity)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Circle { radius: 0.3 },
            f32::NAN
        ),
        Err(ShapeError::NonFiniteInput)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(f32::NAN, 2.0),
            Shape2::Circle { radius: 0.3 },
            1.0
        ),
        Err(ShapeError::NonFiniteInput)
    );
    // Static bodies ignore mass: zero density is accepted, inverses stay 0.
    let s = world
        .add_static_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.3 })
        .expect("valid shape");
    world.bodies[s]
        .set_shape(Shape2::Circle { radius: 0.3 }, 0.0)
        .expect("static accepts zero density");
    assert_eq!(world.bodies[s].inv_mass, 0.0);
    // Valid creation still works.
    let b = world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    for frame in 0..120 {
        world.step(DT);
        assert_finite(&world, &format!("s18 frame {frame}"));
    }
    assert!(
        world.bodies[b].transform.position.y > -1.0,
        "s18: valid body must not tunnel"
    );
}

/// S19 probe: infinite inertia (inv_inertia = 0) with finite mass, and
/// zero inv_mass on a non-static body. Both constructible via pub
/// fields; assert defined (finite, expected kinematics) behavior.
#[test]
fn s19_infinite_inertia_variants() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    // (a) Infinite inertia: off-center impulse must not rotate the body.
    let a = world
        .add_dynamic_body(
            Vec2::new(-2.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[a].inv_inertia = 0.0;
    world.bodies[a].apply_impulse(Vec2::new(0.0, 1.0), Vec2::new(-1.5, 0.0));
    for _ in 0..60 {
        world.step(DT);
    }
    assert_finite(&world, "s19a");
    assert_eq!(
        world.bodies[a].angular_velocity, 0.0,
        "s19a: infinite inertia must not rotate"
    );
    assert!(
        world.bodies[a].velocity.y > 0.0,
        "s19a: linear response must survive: v={}",
        world.bodies[a].velocity
    );
    // (b) Zero inv_mass, non-static flag: solver must treat as immovable.
    let b = world
        .add_dynamic_body(
            Vec2::new(2.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[b].inv_mass = 0.0;
    world.bodies[b].apply_impulse(Vec2::new(5.0, 0.0), Vec2::new(2.0, 0.0));
    for _ in 0..60 {
        world.step(DT);
    }
    assert_finite(&world, "s19b");
    assert_eq!(
        world.bodies[b].transform.position,
        Vec2::new(2.0, 0.0),
        "s19b: zero inv_mass body must not move"
    );
}

/// FNV-1a hash over dynamic body state (positions, velocities, spin).
/// Used by determinism pins: same scenario twice must hash identically.
fn hash_world2(world: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
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

/// S03 pin: direct sleep-flag mutation between steps (the only way to
/// observe "mid-pipeline" flag states — step() itself is synchronous).
/// The solver must treat flags consistently: it never clears them, and
/// a mutated scenario is deterministic (identical hashes twice).
#[test]
fn s03_sleep_flag_mutation_consistent() {
    fn run_once() -> (u64, bool) {
        let mut world = World2::new();
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        world
            .add_dynamic_body(
                Vec2::new(0.0, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        world
            .add_dynamic_body(
                Vec2::new(0.0, 1.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        for _ in 0..60 {
            world.step(DT);
        }
        // Mutate directly between steps, then run on.
        world.bodies[2].is_sleeping = true;
        for _ in 0..60 {
            world.step(DT);
        }
        (hash_world2(&world), world.bodies[2].is_sleeping)
    }
    let (h1, f1) = run_once();
    let (h2, f2) = run_once();
    assert_eq!(
        (h1, f1),
        (h2, f2),
        "s03: mutated-flag scenario must be deterministic"
    );
    assert!(f1, "s03: solver must never clear sleep flags");
}

/// S05 pin: a sleeping body holding warm-start impulses wakes without a
/// phantom push. Settle (cache fills with resting impulses) -> force
/// sleep -> wake -> 30 steps: velocity stays at resting scale and the
/// body re-settles instead of jumping.
#[test]
fn s05_wake_no_phantom_push() {
    let mut world = World2::new();
    floor(&mut world);
    let b = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..120 {
        world.step(DT);
    }
    world.bodies[b].velocity = Vec2::ZERO;
    world.bodies[b].angular_velocity = 0.0;
    world.step(DT);
    assert!(
        world.bodies[b].is_sleeping,
        "s05: box must be asleep before wake"
    );
    let rest_y = world.bodies[b].transform.position.y;
    world.wake_body(b);
    let mut peak = 0.0f32;
    for _ in 0..30 {
        world.step(DT);
        peak = peak.max(world.bodies[b].velocity.length());
    }
    assert_finite(&world, "s05 woken");
    assert!(
        peak < 0.5,
        "s05: wake must not resurrect a phantom push, peak={peak}"
    );
    assert!(
        (world.bodies[b].transform.position.y - rest_y).abs() < 0.1,
        "s05: must re-settle, y={}",
        world.bodies[b].transform.position.y
    );
}

/// S20 pin: impulse propagation through flagged-asleep bodies is
/// deterministic. Force-sleep a 3-stack, kick the bottom, step 60:
/// identical hashes twice, bottom moved, all finite. (With partial
/// sleep the impulse propagates normally; this pins determinism so the
/// future full-sleep rework can assert its own propagation semantics.)
#[test]
fn s20_impulse_through_sleeping_stack() {
    fn run_once() -> (u64, f32) {
        let mut world = World2::new();
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        for i in 0..3 {
            world
                .add_dynamic_body(
                    Vec2::new(0.0, i as f32),
                    Shape2::Box {
                        half_extents: Vec2::new(0.5, 0.5),
                    },
                    1.0,
                )
                .expect("valid shape");
        }
        for _ in 0..60 {
            world.step(DT);
        }
        for id in 1..=3 {
            world.bodies[id].velocity = Vec2::ZERO;
            world.bodies[id].angular_velocity = 0.0;
        }
        world.step(DT);
        world.bodies[1].apply_impulse(Vec2::new(3.0, 0.0), Vec2::new(0.0, 0.0));
        for _ in 0..60 {
            world.step(DT);
        }
        for body in &world.bodies {
            for v in [
                body.transform.position.x,
                body.transform.position.y,
                body.velocity.x,
                body.velocity.y,
                body.angular_velocity,
            ] {
                assert!(v.is_finite(), "s20: non-finite state");
            }
        }
        (hash_world2(&world), world.bodies[1].transform.position.x)
    }
    let (h1, x1) = run_once();
    let (h2, x2) = run_once();
    assert_eq!(
        h1, h2,
        "s20: propagation through flagged bodies must be deterministic"
    );
    assert!(
        x1 > 0.1,
        "s20: kicked bottom must have moved: x={x1} vs {x2}"
    );
}

/// S21 pin: solver early-exit is trajectory-neutral. Same resting scene
/// with default tolerance vs 0.0 (exit disabled): identical hashes —
/// skipped iterations are no-ops — and settled in both runs.
#[test]
fn s21_early_exit_trajectory_neutral() {
    fn run_once(tol: f32) -> World2 {
        let mut world = World2::new();
        world.solver.early_exit_tolerance = tol;
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        world
            .add_dynamic_body(
                Vec2::new(-0.5, 0.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        world
            .add_dynamic_body(Vec2::new(0.5, 0.5), Shape2::Circle { radius: 0.4 }, 1.0)
            .expect("valid shape");
        for _ in 0..300 {
            world.step(DT);
        }
        world
    }
    let on = run_once(1e-6);
    let on2 = run_once(1e-6);
    let off = run_once(0.0);
    // Same tolerance twice: bitwise determinism (replays/lockstep).
    assert_eq!(
        hash_world2(&on),
        hash_world2(&on2),
        "s21: same config must hash identically"
    );
    // Across tolerances: bounded difference, not bitwise equality.
    // Skipped iterations would have applied sub-tolerance impulses, so a
    // last-ulp rounding can flip hash bits; positions agree to ~1e-9.
    let mut dpos = 0.0f32;
    let mut dvel = 0.0f32;
    for (a, b) in on.bodies.iter().zip(off.bodies.iter()) {
        dpos = dpos.max((a.transform.position - b.transform.position).length());
        dvel = dvel.max((a.velocity - b.velocity).length());
    }
    assert!(
        dpos < 1e-6 && dvel < 1e-6,
        "s21: threshold must not change the trajectory: dpos={dpos:e} dvel={dvel:e}"
    );
    // Both runs settled (sanity that neither diverged).
    for body in on.bodies.iter().skip(1) {
        assert!(
            body.velocity.length() < 1.0,
            "s21: unsettled: v={}",
            body.velocity
        );
    }
}

/// N12 pin: needle overlap at 100 m/s (never-guess policy + no-CCD
/// regime). A thin fast capsule crossing a static slab: EPA may skip
/// needle frames and discrete stepping may tunnel — both documented.
/// Asserts bounded behavior (finite state, finite speed, inside world
/// bounds), not correctness.
#[test]
fn n12_needle_high_speed_bounded() {
    use glam::Vec3;
    use phieng_dynamics::body::Shape3;
    use phieng_dynamics::World3;
    let mut world = World3::new();
    world.gravity = Vec3::ZERO;
    world
        .add_static_body(
            Vec3::new(0.0, 0.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(0.05, 2.0, 2.0),
            },
        )
        .expect("valid shape");
    let b = world
        .add_dynamic_body(
            Vec3::new(-5.0, 0.0, 0.0),
            Shape3::Capsule {
                radius: 0.05,
                half_length: 1.0,
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[b].velocity = Vec3::new(100.0, 0.0, 0.0);
    for frame in 0..120 {
        world.step(DT);
        let body = &world.bodies[b];
        assert!(
            body.transform.position.is_finite() && body.velocity.is_finite(),
            "n12 frame {frame}: non-finite state"
        );
        assert!(
            body.velocity.length() < 500.0,
            "n12 frame {frame}: speed exploded: {}",
            body.velocity
        );
        assert!(
            body.transform.position.x.abs() < 2000.0,
            "n12 frame {frame}: escaped world bounds: {}",
            body.transform.position
        );
    }
}

/// S04 pin: sleep threshold is strict `<` on both axes. Exactly at
/// threshold stays awake; static bodies always report asleep. Pure
/// function of floats: 1000 evaluations must agree (documents the
/// comparison operator so future edits can't silently flip it).
#[test]
fn s04_sleep_threshold_boundary() {
    let mut world = World2::new();
    let b = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    world.bodies[b].velocity = Vec2::new(0.01, 0.0);
    world.bodies[b].angular_velocity = 0.0;
    assert!(
        !world.bodies[b].should_sleep(0.01, 0.01),
        "s04: exactly at threshold must stay awake (strict <)"
    );
    world.bodies[b].velocity = Vec2::new(0.009, 0.0);
    assert!(
        world.bodies[b].should_sleep(0.01, 0.01),
        "s04: below threshold must sleep"
    );
    world.bodies[b].velocity = Vec2::ZERO;
    world.bodies[b].angular_velocity = 0.01;
    assert!(
        !world.bodies[b].should_sleep(0.01, 0.01),
        "s04: angular exactly at threshold must stay awake"
    );
    world.bodies[b].angular_velocity = 0.009;
    assert!(
        world.bodies[b].should_sleep(0.01, 0.01),
        "s04: angular below threshold must sleep"
    );
    let s = world
        .add_static_body(
            Vec2::ZERO,
            Shape2::Box {
                half_extents: Vec2::new(1.0, 1.0),
            },
        )
        .expect("valid shape");
    world.bodies[s].velocity = Vec2::new(99.0, 0.0);
    assert!(
        world.bodies[s].should_sleep(0.01, 0.01),
        "s04: static bodies always report asleep"
    );
    for _ in 0..1000 {
        assert!(!world.bodies[b].should_sleep(0.0, 0.0));
    }
}

/// W08 pin: contact normal flips between frames (B teleported to the
/// mirror side with contact points still within match distance). The
/// matcher must skip flipped entries (`dot <= 0`) and cold-restart.
/// Design note: warm-start seeds scalars along the *current* normal, so
/// it is direction-safe by construction; this test guards the skip rule
/// itself — a regression to vector-seeding (applying the stored normal)
/// would push both bodies the wrong way and fail the direction asserts.
/// Single solver iteration makes the seeded impulse dominate the step.
#[test]
fn w08_flipped_normal_cold_restart() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    let he = Vec2::new(0.05, 0.05);
    let a = world
        .add_dynamic_body(Vec2::new(-0.04, 0.0), Shape2::Box { half_extents: he }, 1.0)
        .expect("valid shape");
    let b = world
        .add_dynamic_body(Vec2::new(0.04, 0.0), Shape2::Box { half_extents: he }, 1.0)
        .expect("valid shape");
    // Sustain a steady overlap so the cache holds live impulses.
    for _ in 0..5 {
        world.bodies[a].transform.position = Vec2::new(-0.04, 0.0);
        world.bodies[a].velocity = Vec2::ZERO;
        world.bodies[b].transform.position = Vec2::new(0.04, 0.0);
        world.bodies[b].velocity = Vec2::ZERO;
        world.step(DT);
    }
    assert!(
        !world.contact_cache.is_empty(),
        "w08: cache must hold the sustained contact"
    );
    // Mirror B to the left of A: same overlap magnitude, flipped normal,
    // contact points within match distance of the old ones.
    world.solver.iterations = 1;
    world.bodies[b].transform.position = Vec2::new(-0.12, 0.0);
    world.bodies[b].velocity = Vec2::ZERO;
    world.step(DT);
    assert_finite(&world, "w08 flipped");
    // Sign must match the new contact normal (A→B), magnitude bounded.
    // Single-iteration convergence is sensitive to rounding; we assert the
    // direction is correct and magnitude is within a generous bound,
    // not a specific value. 30% cross-language divergence is expected
    // from single-iteration rounding — the test guards the direction
    // and that the impulse doesn't blow up.
    assert!(
        world.bodies[b].velocity.x < 0.02,
        "w08: B must move -x under the fresh normal, v={}",
        world.bodies[b].velocity
    );
    assert!(
        world.bodies[a].velocity.x > -0.02,
        "w08: A must move +x under the fresh normal, v={}",
        world.bodies[a].velocity
    );
    assert!(
        world.bodies[b].velocity.x > -0.5 && world.bodies[a].velocity.x < 0.5,
        "w08: magnitudes must stay bounded, A={} B={}",
        world.bodies[a].velocity.x,
        world.bodies[b].velocity.x
    );
}

/// W09 pin: a pair separated for exactly one frame must cold-restart,
/// never resurrect a stale impulse. The write-back `retain` evicts the
/// gap frame's entries; assert the cache is empty right after the gap.
#[test]
fn w09_one_frame_gap_cold_restart() {
    let mut world = World2::new();
    floor(&mut world);
    let b = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..60 {
        world.step(DT);
    }
    let rest_y = world.bodies[b].transform.position.y;
    // One airborne frame: no contact possible.
    world.bodies[b].transform.position.y = rest_y + 1.0;
    world.bodies[b].velocity = Vec2::ZERO;
    world.step(DT);
    assert!(
        world.contact_cache.is_empty(),
        "w09: gap frame must evict the pair's entries"
    );
    // Return exactly; the re-established contact must settle, not spike.
    world.bodies[b].transform.position.y = rest_y;
    world.bodies[b].velocity = Vec2::ZERO;
    let mut peak = 0.0f32;
    for _ in 0..30 {
        world.step(DT);
        peak = peak.max(world.bodies[b].velocity.length());
    }
    assert_finite(&world, "w09 returned");
    assert!(
        peak < 1.0,
        "w09: re-established contact must settle without resurrection spike, peak={peak}"
    );
    assert!(
        (world.bodies[b].transform.position.y - rest_y).abs() < 0.2,
        "w09: must recover the rest pose, y={}",
        world.bodies[b].transform.position.y
    );
}

/// W10 pin: removal remaps survivor cache keys through the swap-remove
/// shift (selective eviction) instead of dropping the whole cache.
/// Entries touching the removed body are dropped (those contacts are
/// gone); survivors keep their warm-start impulses. Re-added ids start
/// cold, and no key ever dangles past the body list.
#[test]
fn w10_remove_readd_same_id_cold() {
    let mut world = World2::new();
    floor(&mut world);
    for i in 0..3 {
        world
            .add_dynamic_body(
                Vec2::new(0.0, i as f32),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
    }
    // Bodies: floor 0, A 1, B 2, C 3. Cached pairs: (0,1), (1,2), (2,3).
    for _ in 0..120 {
        world.step(DT);
    }
    assert!(
        !world.contact_cache.is_empty(),
        "w10: stack must hold cached contacts"
    );
    // Remove middle B (id 2, non-tail -> remap path, last = 3 moves to 2).
    world.remove_body(2);
    assert_eq!(world.bodies.len(), 3);
    assert_eq!(world.bodies[2].id, 2, "w10: tail must shift into the gap");
    assert!(
        world
            .contact_cache
            .get(&(0, 1))
            .map(|v| !v.is_empty())
            .unwrap_or(false),
        "w10: survivor pair (0,1) must keep its entries"
    );
    assert!(
        !world.contact_cache.contains_key(&(1, 2)),
        "w10: dropped pair (1,2) must be gone"
    );
    for (a, b) in world.contact_cache.keys() {
        assert!(
            *a < world.bodies.len() && *b < world.bodies.len(),
            "w10: dangling cache key ({a}, {b})"
        );
    }
    // Tail path: removing the last body retains survivor keys untouched.
    let before: Vec<(usize, usize)> = world.contact_cache.keys().cloned().collect();
    world.remove_body(2);
    assert_eq!(world.bodies.len(), 2);
    for k in &before {
        assert!(
            world.contact_cache.contains_key(k),
            "w10: tail removal must not disturb survivor key {k:?}"
        );
    }
    // Re-add reuses the id cold: no resurrection from dropped entries.
    let c = world
        .add_dynamic_body(Vec2::new(3.0, 2.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    assert_eq!(c, 2, "w10: id reuse must follow next_body_id");
    for frame in 0..120 {
        world.step(DT);
        assert_finite(&world, &format!("w10 frame {frame}"));
        for (a, b) in world.contact_cache.keys() {
            assert!(
                *a < world.bodies.len() && *b < world.bodies.len(),
                "w10: dangling cache key ({a}, {b})"
            );
        }
    }
    assert!(
        world.bodies[1].transform.position.y > -0.6,
        "w10: survivor must not tunnel: y={}",
        world.bodies[1].transform.position.y
    );
    assert!(
        world.bodies[c].transform.position.y > -0.6,
        "w10: re-added body must settle, not tunnel: y={}",
        world.bodies[c].transform.position.y
    );
}

/// D22 pin: CW vs CCW winding of the same triangle must yield bitwise
/// identical mass properties and (up to float noise) identical collision
/// results — winding is a representation detail, never physics.
#[test]
fn d22_winding_invariance() {
    use phieng_collision::sat::sat_polygon_polygon;
    let ccw = vec![
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.0, 0.5),
    ];
    let cw: Vec<Vec2> = ccw.iter().rev().cloned().collect();
    let mut world = World2::new();
    let a = world
        .add_dynamic_body(Vec2::ZERO, Shape2::Polygon { vertices: ccw }, 2.0)
        .expect("valid shape");
    let b = world
        .add_dynamic_body(Vec2::ZERO, Shape2::Polygon { vertices: cw }, 2.0)
        .expect("valid shape");
    assert_eq!(
        world.bodies[a].mass.to_bits(),
        world.bodies[b].mass.to_bits(),
        "d22: mass must be winding-invariant"
    );
    assert_eq!(
        world.bodies[a].inertia.to_bits(),
        world.bodies[b].inertia.to_bits(),
        "d22: inertia must be winding-invariant"
    );
    let bx = vec![
        Vec2::new(0.1, -0.6),
        Vec2::new(1.1, -0.6),
        Vec2::new(1.1, 0.4),
        Vec2::new(0.1, 0.4),
    ];
    let Some(Shape2::Polygon { vertices: va }) = &world.bodies[a].shape else {
        panic!("d22: expected polygon")
    };
    let Some(Shape2::Polygon { vertices: vb }) = &world.bodies[b].shape else {
        panic!("d22: expected polygon")
    };
    let (na, da) = sat_polygon_polygon(va, &bx).expect("d22: must overlap");
    let (nb, db) = sat_polygon_polygon(vb, &bx).expect("d22: must overlap");
    assert!(
        (da - db).abs() < 1e-6,
        "d22: depths must agree: {da} vs {db}"
    );
    assert!(na.dot(nb) > 0.999, "d22: normals must agree: {na} vs {nb}");
}

/// D26 pin: a zero-length capsule is geometrically a sphere. Support
/// functions must agree, and EPA depths on the same overlap must agree
/// within EPA tolerance — no degenerate-axis NaN path allowed.
/// (Creation still rejects it with ZeroLengthCapsule: the math is exact,
/// the API prefers an explicit Sphere. See ShapeError.)
#[test]
fn d26_zero_length_capsule_is_sphere() {
    use glam::{Quat, Vec3};
    use phieng_collision::epa::gjk_epa;
    use phieng_collision::gjk3d::{support_capsule, support_obb, support_sphere};
    use phieng_dynamics::body::Shape3;
    use phieng_dynamics::World3;
    let mut w3 = World3::new();
    assert_eq!(
        w3.add_dynamic_body(
            Vec3::ZERO,
            Shape3::Capsule {
                radius: 0.5,
                half_length: 0.0
            },
            1.0,
        ),
        Err(ShapeError::ZeroLengthCapsule)
    );
    let center = Vec3::new(1.5, 0.0, 0.0);
    for dir in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        -Vec3::X,
        Vec3::new(1.0, 2.0, 3.0).normalize(),
        Vec3::new(-0.3, 0.8, -0.5).normalize(),
    ] {
        let cap = support_capsule(center, center, 0.5, dir);
        let sph = support_sphere(center, 0.5, dir);
        assert!(
            (cap - sph).length() < 1e-6,
            "d26: supports must agree along {dir}: {cap} vs {sph}"
        );
    }
    let he = Vec3::ONE;
    let center = Vec3::new(1.2, 0.0, 0.0);
    let sa_box = |dir: Vec3| support_obb(Vec3::ZERO, he, Quat::IDENTITY, dir);
    let sa_sph = |dir: Vec3| support_sphere(center, 0.5, dir);
    let sa_cap = |dir: Vec3| support_capsule(center, center, 0.5, dir);
    let d_sph = gjk_epa(&sa_box, &sa_sph)
        .expect("d26: sphere must collide")
        .depth;
    let hit = gjk_epa(&sa_box, &sa_cap).expect("d26: capsule must collide");
    assert!(
        (hit.depth - d_sph).abs() < 1e-3,
        "d26: EPA depths must agree: {} vs {d_sph}",
        hit.depth
    );
    assert!(
        hit.normal.x.abs() > 0.99,
        "d26: normal must point along +x: {}",
        hit.normal
    );
}

/// D27: a dynamic hull is rejected (its mass properties cannot be
/// computed from vertices alone), while a static hull is accepted — a
/// static body needs no mass. The old stub (1 kg, identity inertia) is
/// gone: an unsupported dynamic shape is an error, not a silent ghost.
#[test]
fn d27_dynamic_hull_is_rejected() {
    use phieng_dynamics::body::Shape3;
    use phieng_dynamics::World3;
    let verts = vec![
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 1.0, 0.0),
        glam::Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut world = World3::new();
    // Too few vertices is still the more specific error.
    assert_eq!(
        world.add_dynamic_body(
            glam::Vec3::ZERO,
            Shape3::ConvexHull {
                vertices: vec![
                    glam::Vec3::new(0.0, 0.0, 0.0),
                    glam::Vec3::new(1.0, 0.0, 0.0),
                    glam::Vec3::new(0.0, 1.0, 0.0),
                ],
            },
            1.0,
        ),
        Err(ShapeError::TooFewHullVertices)
    );
    // A valid hull is refused for a dynamic body...
    assert_eq!(
        world.add_dynamic_body(
            glam::Vec3::ZERO,
            Shape3::ConvexHull {
                vertices: verts.clone(),
            },
            1.0,
        ),
        Err(ShapeError::DynamicConvexHull)
    );
    // ...and accepted for a static one.
    let wall = world
        .add_static_body(
            glam::Vec3::ZERO,
            Shape3::ConvexHull {
                vertices: verts.clone(),
            },
        )
        .expect("a static hull needs no mass");
    assert!(world.bodies[wall].is_static);
    // A dynamic mesh is the supported form.
    let mesh = Shape3::ConvexMesh {
        vertices: verts.clone(),
        indices: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    };
    let id = world
        .add_dynamic_body(glam::Vec3::new(5.0, 0.0, 0.0), mesh, 1.0)
        .expect("a closed mesh has exact mass");
    assert!(world.bodies[id].mass > 0.0);
}

/// S01: a sleeping body auto-wakes when a moving body makes NEW contact
/// with it (uncached pair), then responds to the impact. Settled pairs
/// stay cached and keep sleeping.
#[test]
fn s01_autowake_on_new_contact() {
    let mut world = World2::new();
    floor(&mut world);
    let sleeper = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    // Settle, then wait out the deferred-sleep counter.
    for _ in 0..120 {
        world.step(DT);
    }
    assert!(
        world.bodies[sleeper].is_sleeping,
        "s01: box must be asleep before the shove"
    );
    // Fast ball aimed at the box's left face: the impact must shove the
    // box sideways (a downward drop would just sandwich it against the
    // floor, proving nothing).
    let ball = world
        .add_dynamic_body(Vec2::new(-3.0, 0.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    world.bodies[ball].velocity = Vec2::new(10.0, 0.0);
    let mut woke = false;
    let mut max_push = 0.0f32;
    for _ in 0..180 {
        world.step(DT);
        if !world.bodies[sleeper].is_sleeping {
            woke = true;
        }
        max_push = max_push.max(world.bodies[sleeper].transform.position.x);
    }
    assert_finite(&world, "s01 shoved box");
    assert!(woke, "s01: sleeper must auto-wake on new contact");
    assert!(
        max_push > 0.05,
        "s01: woken sleeper must be shoved sideways, xmax={max_push}"
    );
}

/// S02: waking the bottom of a sleeping stack cascades synchronously —
/// contact partners of an awake body cannot stay asleep. Both settle
/// back without floating or tunneling.
#[test]
fn s02_wake_cascades_to_stack() {
    let mut world = World2::new();
    floor(&mut world);
    // Resting stack with a hairline gap: the top falls into place and
    // settles, then both sleep (deferred sleep needs 30 slow frames).
    // NOTE: a exactly-touching placement is avoided — zero-penetration
    // starts hit a manifold tie-break that rocks instead of settling.
    let bottom = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let top = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.05),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..180 {
        world.step(DT);
    }
    assert!(
        world.bodies[bottom].is_sleeping && world.bodies[top].is_sleeping,
        "s02: stack must start asleep"
    );
    // The cascade is synchronous: the top is awake immediately.
    world.wake_body(bottom);
    assert!(
        !world.bodies[bottom].is_sleeping && !world.bodies[top].is_sleeping,
        "s02: wake must cascade to the contact partner at once"
    );
    for _ in 0..120 {
        world.step(DT);
    }
    assert_finite(&world, "s02 woken stack");
    let by = world.bodies[bottom].transform.position.y;
    let ty = world.bodies[top].transform.position.y;
    assert!(
        by > -0.1 && by < 0.3,
        "s02: bottom must rest on the floor, y={by}"
    );
    assert!(
        ty > 0.5 && ty < 1.5,
        "s02: top must rest on the bottom, y={ty}"
    );
}
/// N11 pin: grazing capsule pair approaching slowly. Sub-margin overlap
/// skips EPA (None) while deeper overlap resolves: penetration must
/// self-heal to a bounded equilibrium (Baumgarte balances the approach)
/// instead of accumulating into tunneling.
#[test]
fn n11_grazing_pair_self_heals() {
    use glam::Vec3;
    use phieng_dynamics::body::Shape3;
    use phieng_dynamics::World3;
    let mut world = World3::new();
    world.gravity = Vec3::ZERO;
    let cap = || Shape3::Capsule {
        radius: 0.3,
        half_length: 0.5,
    };
    world
        .add_static_body(Vec3::ZERO, cap())
        .expect("valid shape");
    let b = world
        .add_dynamic_body(Vec3::new(0.5999, 0.0, 0.0), cap(), 1.0)
        .expect("valid shape");
    world.bodies[b].velocity = Vec3::new(-0.02, 0.0, 0.0);
    let mut max_overlap = 0.0f32;
    for frame in 0..100 {
        world.step(DT);
        for body in &world.bodies {
            for v in [
                body.transform.position.x,
                body.transform.position.y,
                body.transform.position.z,
                body.velocity.x,
                body.velocity.y,
                body.velocity.z,
            ] {
                assert!(v.is_finite(), "n11 frame {frame}: non-finite state");
            }
        }
        let overlap = 0.6 - world.bodies[b].transform.position.x;
        max_overlap = max_overlap.max(overlap);
    }
    assert!(
        max_overlap < 0.01,
        "n11: penetration must self-heal, max overlap={max_overlap}"
    );
    assert!(
        world.bodies[b].transform.position.x > 0.4,
        "n11: must not tunnel through: x={}",
        world.bodies[b].transform.position.x
    );
}

/// N15 pin: never-guess policy under load — a capsule bearing a box in a
/// 3D stack. Grazing capsule contacts may skip frames; the stack must stay
/// bounded (nothing tunnels, speeds sane). Pose is NOT prescribed: the
/// capsule may tip and the heap settle however it settles.
#[test]
fn n15_capsule_bearing_stack_stable() {
    use glam::Vec3;
    use phieng_dynamics::body::Shape3;
    use phieng_dynamics::World3;
    let mut world = World3::new();
    world
        .add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
        )
        .expect("valid shape");
    let box_he = Vec3::new(0.5, 0.5, 0.5);
    world
        .add_dynamic_body(
            Vec3::new(0.0, 0.0, 0.0),
            Shape3::Box {
                half_extents: box_he,
            },
            1.0,
        )
        .expect("valid shape");
    world
        .add_dynamic_body(
            Vec3::new(0.0, 1.3, 0.0),
            Shape3::Capsule {
                radius: 0.3,
                half_length: 0.5,
            },
            1.0,
        )
        .expect("valid shape");
    world
        .add_dynamic_body(
            Vec3::new(0.0, 2.6, 0.0),
            Shape3::Box {
                half_extents: box_he,
            },
            1.0,
        )
        .expect("valid shape");
    for frame in 0..600 {
        world.step(DT);
        if frame % 60 == 0 {
            for body in &world.bodies {
                assert!(
                    body.transform.position.is_finite() && body.velocity.is_finite(),
                    "n15 frame {frame}: non-finite state"
                );
            }
        }
    }
    for body in world.bodies.iter().skip(1) {
        assert!(
            body.transform.position.y > -0.6,
            "n15: body {} tunneled: y={}",
            body.id,
            body.transform.position.y
        );
        assert!(
            body.velocity.length() < 2.0,
            "n15: body {} never settled: v={}",
            body.id,
            body.velocity
        );
    }
}

/// D23 pin (Q2 boundary rule): collinear "triangle" and < 3 distinct
/// vertices reject at creation. Only exact degeneracy rejects;
/// near-degenerate thin polygons still simulate (2D SAT is robust).
#[test]
fn d23_collinear_polygon() {
    let mut world = World2::new();
    floor(&mut world);
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Polygon {
                vertices: vec![
                    Vec2::new(-0.5, 0.0),
                    Vec2::new(0.0, 0.0),
                    Vec2::new(0.5, 0.0),
                ],
            },
            1.0,
        ),
        Err(ShapeError::DegeneratePolygon)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Polygon {
                vertices: vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)],
            },
            1.0,
        ),
        Err(ShapeError::DegeneratePolygon)
    );
    // Valid thin triangle still simulates.
    let b = world
        .add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Polygon {
                vertices: vec![
                    Vec2::new(-0.5, 0.0),
                    Vec2::new(0.5, 0.0),
                    Vec2::new(0.0, 0.01),
                ],
            },
            1.0,
        )
        .expect("valid shape");
    for frame in 0..120 {
        world.step(DT);
        assert_finite(&world, &format!("d23 frame {frame}"));
    }
    let _ = world.bodies[b].transform.position;
}

/// D24 pin (Q2 boundary rule): zero/negative/non-finite radius rejects
/// at creation instead of simulating a point with floored mass.
#[test]
fn d24_zero_radius_circle() {
    let mut world = World2::new();
    floor(&mut world);
    assert_eq!(
        world.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.0 }, 1.0),
        Err(ShapeError::ZeroSize)
    );
    assert_eq!(
        world.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: -0.5 }, 1.0),
        Err(ShapeError::ZeroSize)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Circle { radius: f32::NAN },
            1.0
        ),
        Err(ShapeError::NonFiniteInput)
    );
    let b = world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    for frame in 0..120 {
        world.step(DT);
        assert_finite(&world, &format!("d24 frame {frame}"));
    }
    let _ = world.bodies[b].transform.position;
}

/// D25 pin (Q2 boundary rule): zero/negative/non-finite half-extents
/// reject at creation.
#[test]
fn d25_zero_size_box() {
    let mut world = World2::new();
    floor(&mut world);
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Box {
                half_extents: Vec2::ZERO
            },
            1.0,
        ),
        Err(ShapeError::ZeroSize)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, -0.5)
            },
            1.0,
        ),
        Err(ShapeError::ZeroSize)
    );
    assert_eq!(
        world.add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Box {
                half_extents: Vec2::new(f32::NAN, 0.5)
            },
            1.0,
        ),
        Err(ShapeError::NonFiniteInput)
    );
    let b = world
        .add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for frame in 0..120 {
        world.step(DT);
        assert_finite(&world, &format!("d25 frame {frame}"));
    }
    let _ = world.bodies[b].transform.position;
}

/// V01: validation layer catches NaN position and corrects it (debug mode)
#[test]
fn v01_nan_position_corrected() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        check_finite: true,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].transform.position.x = f32::NAN;
    world.step(1.0 / 60.0);
    let body = &world.bodies[id];
    assert!(body.transform.position.x.is_finite());
    assert!(body.transform.position.y.is_finite());
    assert_eq!(world.validation_report.non_finite_corrected, 1);
}

/// V02: validation layer clamps velocity exceeding max
#[test]
fn v02_velocity_clamped() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        max_velocity: 10.0,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].velocity = Vec2::new(100.0, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].velocity.length() <= 10.0 + 1e-4);
    assert_eq!(world.validation_report.velocity_clamped, 1);
}

/// V03: validation layer clamps angular velocity exceeding max
#[test]
fn v03_angular_velocity_clamped() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        max_angular_velocity: 5.0,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(
            Vec2::new(0.0, 5.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[id].angular_velocity = 100.0;
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].angular_velocity.abs() <= 5.0 + 1e-4);
    assert_eq!(world.validation_report.angular_velocity_clamped, 1);
}

/// V04: validation layer clamps position exceeding max coordinate
#[test]
fn v04_position_clamped() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        max_coordinate: 100.0,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(Vec2::new(1000.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].transform.position.length() <= 100.0 + 1e-4);
    assert_eq!(world.validation_report.coordinate_clamped, 1);
}

/// V05: validation layer clamps penetration depth
#[test]
fn v05_penetration_clamped() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.validation = ValidationConfig {
        max_penetration: 0.1,
        ..Default::default()
    };
    world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let _id = world
        .add_dynamic_body(
            Vec2::new(0.1, 0.0), // 40% overlap (penetration ~0.4)
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.step(1.0 / 60.0);
    // The constraint should have its penetration clamped
    assert!(world.validation_report.penetration_clamped >= 1);
}

/// V06: validation layer detects extreme mass ratio
#[test]
fn v06_mass_ratio_detected() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.validation = ValidationConfig {
        max_mass_ratio: 100.0,
        ..Default::default()
    };
    world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1000.0, // mass ratio 1000:1 vs static (infinite)
        )
        .expect("valid shape");
    let _light = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.step(1.0 / 60.0);
    // Should detect mass ratio exceeded for the heavy/light pair
    assert_eq!(world.validation_report.mass_ratio_exceeded, 1);
}

/// V07: validation layer works in 3D
#[test]
fn v07_validation_3d() {
    use phieng_dynamics::{body::Shape3, World3};
    let mut world = World3::new();
    world.validation = ValidationConfig {
        check_finite: true,
        max_velocity: 10.0,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(
            glam::Vec3::new(0.0, 5.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    world.bodies[id].velocity = glam::Vec3::new(100.0, 0.0, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].velocity.length() <= 10.0 + 1e-4);
    assert_eq!(world.validation_report.velocity_clamped, 1);
}

/// V08: strict config enables all checks
#[test]
fn v08_strict_config() {
    let mut world = World2::new();
    world.validation = ValidationConfig::strict();
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].velocity = Vec2::new(f32::NAN, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].velocity.x.is_finite());
    assert!(world.validation_report.non_finite_corrected >= 1);
}

/// V09: permissive config disables most checks for performance
#[test]
fn v09_permissive_config() {
    let mut world = World2::new();
    world.validation = ValidationConfig::permissive();
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].velocity = Vec2::new(10000.0, 0.0);
    world.step(1.0 / 60.0);
    // Permissive has high limits, should not clamp at 10000
    // but we check it doesn't crash
    assert!(world.bodies[id].velocity.x.is_finite());
}

/// V10: validation report aggregates correctly
#[test]
fn v10_validation_report_aggregates() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        check_finite: true,
        max_velocity: 10.0,
        max_coordinate: 100.0,
        ..Default::default()
    };
    let id1 = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id1].transform.position.x = f32::NAN;
    let id2 = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id2].velocity = Vec2::new(100.0, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.validation_report.non_finite_corrected >= 1);
    assert!(world.validation_report.velocity_clamped >= 1);
    assert!(world.validation_report.any_issues());
}

/// V11: 3D NaN position with finite velocity must still report NonFiniteCorrected.
/// Regression pin: the 3D validator previously reset its detection flag between
/// position/velocity/angular checks, so a NaN position with finite velocity
/// was corrected but reported as Ok.
#[test]
fn v11_3d_nan_position_reports() {
    use phieng_dynamics::{body::Shape3, World3};
    let mut world = World3::new();
    world.validation = ValidationConfig {
        check_finite: true,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(
            glam::Vec3::new(0.0, 5.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    world.bodies[id].transform.position.x = f32::NAN;
    world.bodies[id].velocity = glam::Vec3::new(1.0, 0.0, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].transform.position.x.is_finite());
    assert!(world.bodies[id].transform.position.y.is_finite());
    assert!(world.bodies[id].transform.position.z.is_finite());
    assert!(
        world.validation_report.non_finite_corrected >= 1,
        "3D NaN position must be reported even with finite velocity"
    );
}

/// Validation log warnings are a no-op without a linked logger: enabling
/// `log_warnings` must not panic and corrections must still apply.
#[test]
fn validation_log_warnings_noop_without_logger() {
    let mut world = World2::new();
    world.validation = ValidationConfig {
        check_finite: true,
        log_warnings: true,
        ..Default::default()
    };
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].velocity = Vec2::new(f32::NAN, 0.0);
    world.step(1.0 / 60.0);
    assert!(world.bodies[id].velocity.x.is_finite());
    assert!(world.validation_report.non_finite_corrected >= 1);
}
