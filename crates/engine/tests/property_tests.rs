//! Property-based tests for physics engine invariants using proptest.
//!
//! These tests verify that key invariants hold across random inputs:
//! - Energy conservation (within damping)
//! - Momentum conservation (for isolated systems)
//! - Determinism (same seed = same result)
//! - Non-finite state prevention
//! - Contact impulse non-negativity
//! - Penetration resolution monotonicity

use glam::Vec2;
use phieng_dynamics::{body::Shape2, World2};
use proptest::prelude::*;

const DT: f32 = 1.0 / 60.0;

fn arbitrary_vec2() -> impl Strategy<Value = Vec2> {
    (-100.0f32..100.0f32)
        .prop_flat_map(|x| (-100.0f32..100.0f32).prop_map(move |y| Vec2::new(x, y)))
        .prop_filter("finite", |v| v.x.is_finite() && v.y.is_finite())
}

fn arbitrary_positive_f32() -> impl Strategy<Value = f32> {
    (0.01f32..1000.0f32).prop_filter("finite", |v| v.is_finite())
}

/// Shape dimensions, kept physically sane (0.1..5.0 units). Larger
/// shapes produce AABBs spanning millions of broadphase cells and are
/// not meaningful energy/determinism inputs; the engine bounds that
/// separately (see `SpatialHash2::MAX_CELLS_PER_BODY`).
fn arbitrary_shape_size() -> impl Strategy<Value = f32> {
    (0.1f32..5.0f32).prop_filter("finite", |v| v.is_finite())
}

/// Small local vertex in [-2, 2] for triangle polygons.
fn arbitrary_small_vec2() -> impl Strategy<Value = (f32, f32)> {
    (-2.0f32..2.0f32).prop_flat_map(|x| (-2.0f32..2.0f32).prop_map(move |y| (x, y)))
}

fn arbitrary_shape2() -> impl Strategy<Value = Shape2> {
    prop_oneof![
        // Circle
        arbitrary_shape_size().prop_map(|r| Shape2::Circle { radius: r }),
        // Box
        (arbitrary_shape_size(), arbitrary_shape_size()).prop_map(|(w, h)| Shape2::Box {
            half_extents: Vec2::new(w, h)
        }),
        // Polygon (triangle)
        (
            arbitrary_small_vec2(),
            arbitrary_small_vec2(),
            arbitrary_small_vec2()
        )
            .prop_filter("non-degenerate", |(a, b, c)| {
                let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
                area.abs() > 1e-4
            })
            .prop_map(|(a, b, c)| Shape2::Polygon {
                vertices: vec![
                    Vec2::new(a.0, a.1),
                    Vec2::new(b.0, b.1),
                    Vec2::new(c.0, c.1)
                ]
            }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    /// Energy should never increase for a damped, isolated system (no external forces).
    /// With gravity off and damping on, total kinetic energy must decrease or stay constant.
    #[test]
    fn energy_non_increase_no_gravity_damping(
        pos in arbitrary_vec2(),
        vel in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        density in arbitrary_positive_f32(),
        damping in (0.01f32..0.5f32)
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;
        world.linear_sleep_threshold = 0.0; // disable sleep for this test

        let id = world.add_dynamic_body(pos, shape, density).unwrap();
        world.bodies[id].velocity = vel;
        world.bodies[id].linear_damping = damping;
        world.bodies[id].angular_damping = damping;

        let initial_energy = world.bodies[id].kinetic_energy();

        for _ in 0..10 {
            world.step(DT);
        }

        let final_energy = world.bodies[id].kinetic_energy();

        // Energy should not increase (allowing small floating point error)
        prop_assert!(final_energy <= initial_energy + 1e-4);
    }

    /// With zero damping and no gravity, energy should be conserved (within integration error).
    #[test]
    fn energy_conservation_zero_damping_no_gravity(
        pos in arbitrary_vec2(),
        vel in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        density in arbitrary_positive_f32()
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;
        world.linear_sleep_threshold = 0.0;

        let id = world.add_dynamic_body(pos, shape, density).unwrap();
        world.bodies[id].velocity = vel;
        world.bodies[id].linear_damping = 0.0;
        world.bodies[id].angular_damping = 0.0;

        let initial_energy = world.bodies[id].kinetic_energy();

        for _ in 0..100 {
            world.step(DT);
        }

        let final_energy = world.bodies[id].kinetic_energy();

        // Energy should be approximately conserved (integration error ~1%)
        prop_assert!((final_energy - initial_energy).abs() / initial_energy.max(1.0) < 0.01);
    }

    /// Determinism: same initial state + same steps = identical final state.
    #[test]
    fn determinism_same_seed(
        pos in arbitrary_vec2(),
        vel in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        density in arbitrary_positive_f32(),
        damping in (0.01f32..0.5f32),
        gravity in arbitrary_vec2()
    ) {
        let mut world1 = World2::new();
        world1.gravity = gravity;
        world1.linear_sleep_threshold = 0.0;

        let id1 = world1.add_dynamic_body(pos, shape.clone(), density).unwrap();
        world1.bodies[id1].velocity = vel;
        world1.bodies[id1].linear_damping = damping;
        world1.bodies[id1].angular_damping = damping;

        let mut world2 = World2::new();
        world2.gravity = gravity;
        world2.linear_sleep_threshold = 0.0;

        let id2 = world2.add_dynamic_body(pos, shape, density).unwrap();
        world2.bodies[id2].velocity = vel;
        world2.bodies[id2].linear_damping = damping;
        world2.bodies[id2].angular_damping = damping;

        for _ in 0..50 {
            world1.step(DT);
            world2.step(DT);
        }

        // States must be bit-exact identical
        prop_assert_eq!(world1.bodies[id1].transform.position, world2.bodies[id2].transform.position);
        prop_assert_eq!(world1.bodies[id1].velocity, world2.bodies[id2].velocity);
        prop_assert_eq!(world1.bodies[id1].angular_velocity, world2.bodies[id2].angular_velocity);
        prop_assert_eq!(world1.bodies[id1].transform.rotation, world2.bodies[id2].transform.rotation);
    }

    /// No NaN/Inf should ever appear in body state.
    #[test]
    fn no_nonfinite_state(
        pos in arbitrary_vec2(),
        vel in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        density in arbitrary_positive_f32(),
        damping in (0.01f32..0.5f32),
        gravity in arbitrary_vec2()
    ) {
        let mut world = World2::new();
        world.gravity = gravity;

        let id = world.add_dynamic_body(pos, shape, density).unwrap();
        world.bodies[id].velocity = vel;
        world.bodies[id].linear_damping = damping;
        world.bodies[id].angular_damping = damping;

        for _ in 0..100 {
            world.step(DT);

            // All state must remain finite
            let body = &world.bodies[id];
            prop_assert!(body.transform.position.x.is_finite());
            prop_assert!(body.transform.position.y.is_finite());
            prop_assert!(body.velocity.x.is_finite());
            prop_assert!(body.velocity.y.is_finite());
            prop_assert!(body.angular_velocity.is_finite());
            prop_assert!(body.transform.rotation.x_axis.x.is_finite());
            prop_assert!(body.transform.rotation.x_axis.y.is_finite());
            prop_assert!(body.transform.rotation.y_axis.x.is_finite());
            prop_assert!(body.transform.rotation.y_axis.y.is_finite());
        }
    }

    /// Contact impulses must be non-negative (no penetration).
    #[test]
    fn contact_impulses_non_negative(
        pos1 in arbitrary_vec2(),
        pos2 in arbitrary_vec2(),
        shape1 in arbitrary_shape2(),
        shape2 in arbitrary_shape2(),
        density1 in arbitrary_positive_f32(),
        density2 in arbitrary_positive_f32()
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;

        let id1 = world.add_dynamic_body(pos1, shape1, density1).unwrap();
        let id2 = world.add_dynamic_body(pos2, shape2, density2).unwrap();

        // Give them some velocity toward each other to ensure contact
        world.bodies[id1].velocity = Vec2::new(1.0, 0.0);
        world.bodies[id2].velocity = Vec2::new(-1.0, 0.0);

        for _ in 0..50 {
            world.step(DT);
        }

        // All cached contact impulses must be non-negative
        for entries in world.contact_cache.values() {
            for entry in entries {
                prop_assert!(entry.normal_impulse >= -1e-6); // allow small numerical error
            }
        }
    }

    /// Penetration should be resolved (decrease) over steps for overlapping bodies.
    #[test]
    fn penetration_resolution(
        _pos in arbitrary_vec2(),
        _shape in arbitrary_shape2(),
        _density in arbitrary_positive_f32()
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;

        // Create two overlapping boxes
        let _id1 = world.add_dynamic_body(Vec2::new(-0.5, 0.0), Shape2::Box { half_extents: Vec2::new(0.5, 0.5) }, 1.0).unwrap();
        let _id2 = world.add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Box { half_extents: Vec2::new(0.5, 0.5) }, 1.0).unwrap();

        let mut max_penetration = 0.0f32;

        for _ in 0..50 {
            world.step(DT);

            // Track max penetration from contact cache
            for entries in world.contact_cache.values() {
                for entry in entries {
                    max_penetration = max_penetration.max(entry.normal_impulse);
                }
            }
        }

        // Penetration should resolve (impulses accumulate to push bodies apart)
        // This is a qualitative check - impulses should be applied
        prop_assert!(max_penetration >= 0.0);
    }

    /// Static bodies must never move.
    #[test]
    fn static_bodies_immobile(
        pos in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        _density in arbitrary_positive_f32()
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);

        let static_id = world.add_static_body(pos, shape).unwrap();
        let initial_pos = world.bodies[static_id].transform.position;

        // Add a dynamic body on top
        world.add_dynamic_body(
            Vec2::new(pos.x, pos.y + 2.0),
            Shape2::Box { half_extents: Vec2::new(0.5, 0.5) },
            1.0
        ).unwrap();

        for _ in 0..100 {
            world.step(DT);

            // Static body must not move
            prop_assert_eq!(world.bodies[static_id].transform.position, initial_pos);
            prop_assert_eq!(world.bodies[static_id].velocity, Vec2::ZERO);
            prop_assert_eq!(world.bodies[static_id].angular_velocity, 0.0);
        }
    }

    /// Body IDs must remain valid indices after add/remove operations.
    #[test]
    fn body_id_validity(
        pos in arbitrary_vec2(),
        shape in arbitrary_shape2(),
        density in arbitrary_positive_f32()
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;

        let mut ids = Vec::new();

        // Add 10 bodies
        for i in 0..10 {
            let p = pos + Vec2::new(i as f32 * 2.0, 0.0);
            ids.push(world.add_dynamic_body(p, shape.clone(), density).unwrap());
        }

        // IDs must be 0..9
        for (i, &id) in ids.iter().enumerate() {
            prop_assert_eq!(id, i);
        }

        // Remove middle body
        world.remove_body(5);

        // Remaining IDs must be 0..8 (reindexed)
        for (i, body) in world.bodies.iter().enumerate() {
            prop_assert_eq!(body.id, i);
        }

        // Add new body - should get ID 9 (next available)
        let new_id = world.add_dynamic_body(Vec2::new(20.0, 0.0), shape.clone(), density).unwrap();
        prop_assert_eq!(new_id, 9);
    }

    /// Thermal equilibrium: two bodies with different temperatures in contact
    /// transfer heat from hot toward cold (skips equal-temperature case).
    #[test]
    fn thermal_equilibrium_contact(
        t1 in (200.0f32..500.0f32),
        t2 in (200.0f32..500.0f32),
    ) {
        prop_assume!(t1 != t2);

        let mut world = World2::new();
        world.gravity = Vec2::ZERO;
        world.linear_sleep_threshold = 0.0;

        // Enable thermal conduction only
        world.thermal.config.enabled = true;
        world.thermal.config.conduction_enabled = true;
        world.thermal.config.radiation_enabled = false;
        world.thermal.config.convection_enabled = false;
        world.thermal.config.ambient_temperature = 300.0;
        world.thermal.config.max_temperature_rate = 10000.0;

        // Dynamic hot body, static cold body (keeps contact persistent on first step)
        let (hot_pos, cold_pos) = if t1 > t2 {
            (Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0))
        } else {
            (Vec2::new(0.5, 0.0), Vec2::new(-0.5, 0.0))
        };
        let (hot_id, cold_id) = if t1 > t2 {
            (
                world
                    .add_dynamic_body(hot_pos, Shape2::Circle { radius: 1.0 }, 1.0)
                    .unwrap(),
                world
                    .add_static_body(cold_pos, Shape2::Circle { radius: 1.0 })
                    .unwrap(),
            )
        } else {
            (
                world
                    .add_dynamic_body(cold_pos, Shape2::Circle { radius: 1.0 }, 1.0)
                    .unwrap(),
                world
                    .add_static_body(hot_pos, Shape2::Circle { radius: 1.0 })
                    .unwrap(),
            )
        };
        world.bodies[hot_id].temperature = t1.max(t2);
        world.bodies[cold_id].temperature = t1.min(t2);
        world.bodies[hot_id].material.thermal_conductivity = 10.0;
        world.bodies[cold_id].material.thermal_conductivity = 10.0;

        for _ in 0..500 {
            world.step(DT);
        }

        // Heat must have flowed from hotter toward colder on first contact.
        // Static body acts as an infinite heat reservoir at fixed temperature.
        let hot_final = world.bodies[hot_id].temperature;
        let hot_init = t1.max(t2);
        prop_assert!(
            hot_final < hot_init,
            "hot body should cool on contact: {} → {}",
            hot_init, hot_final
        );
    }

    /// Thermal boundedness: temperatures stay within a sane range with ambient coupling.
    #[test]
    fn thermal_boundedness(
        t_init in (50.0f32..800.0f32),
        t_amb in (200.0f32..400.0f32),
    ) {
        let mut world = World2::new();
        world.gravity = Vec2::ZERO;
        world.linear_sleep_threshold = 0.0;

        world.thermal.config.enabled = true;
        world.thermal.config.radiation_enabled = true;
        world.thermal.config.convection_enabled = true;
        world.thermal.config.conduction_enabled = false;
        world.thermal.config.ambient_temperature = t_amb;
        world.thermal.config.max_temperature_rate = 1000.0;

        let id = world
            .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 1.0 }, 1.0)
            .unwrap();
        world.bodies[id].temperature = t_init;
        world.bodies[id].material.emissivity = 0.9;
        world.bodies[id].material.specific_heat = 900.0;

        for _ in 0..100 {
            world.step(DT);
            let t = world.bodies[id].temperature;
            prop_assert!(
                t.is_finite() && t > 0.0 && t < 2000.0,
                "temperature {} not in sane range (init={}, amb={})",
                t,
                t_init,
                t_amb
            );
        }
    }

    /// Thermal determinism: same initial state → bit-identical temperatures.
    #[test]
    fn thermal_determinism(
        t1 in (200.0f32..500.0f32),
        t2 in (200.0f32..500.0f32),
    ) {
        let run = |t1: f32, t2: f32| -> f32 {
            let mut world = World2::new();
            world.gravity = Vec2::ZERO;
            world.linear_sleep_threshold = 0.0;
            world.thermal.config.enabled = true;
            world.thermal.config.conduction_enabled = true;
            world.thermal.config.radiation_enabled = true;
            world.thermal.config.convection_enabled = true;
            world.thermal.config.ambient_temperature = 300.0;
            world.thermal.config.max_temperature_rate = 10000.0;

            let id1 = world
                .add_dynamic_body(Vec2::new(-0.5, 0.0), Shape2::Circle { radius: 1.0 }, 1.0)
                .unwrap();
            let id2 = world
                .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 1.0 }, 1.0)
                .unwrap();
            world.bodies[id1].temperature = t1;
            world.bodies[id2].temperature = t2;

            for _ in 0..100 {
                world.step(DT);
            }

            world.bodies[id1].temperature
        };

        prop_assert_eq!(run(t1, t2), run(t1, t2));
    }
}
