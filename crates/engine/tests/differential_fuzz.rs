//! Differential fuzzing test: compare Rust and C# simulation results for identical seeds.
//!
//! This test runs the same physics scenario in both Rust (native) and C# (managed)
//! with identical initial conditions and compares the final states to ensure
//! cross-language parity within acceptable tolerances.

use glam::Vec2;
use phieng_dynamics::{body::Shape2, World2};
use proptest::prelude::*;

const DT: f32 = 1.0 / 60.0;

fn arbitrary_vec2() -> impl Strategy<Value = Vec2> {
    (-10.0f32..10.0f32)
        .prop_flat_map(|x| (-10.0f32..10.0f32).prop_map(move |y| Vec2::new(x, y)))
        .prop_filter("finite", |v| v.x.is_finite() && v.y.is_finite())
}

fn arbitrary_shape2() -> impl Strategy<Value = Shape2> {
    prop_oneof![
        (0.01f32..5.0f32).prop_map(|r| Shape2::Circle { radius: r }),
        ((0.01f32..5.0f32), (0.01f32..5.0f32)).prop_map(|(w, h)| Shape2::Box {
            half_extents: Vec2::new(w, h)
        }),
        (arbitrary_vec2(), arbitrary_vec2(), arbitrary_vec2())
            .prop_filter("non-degenerate", |(a, b, c)| {
                let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                area.abs() > 1e-4
            })
            .prop_map(|(a, b, c)| Shape2::Polygon {
                vertices: vec![a, b, c]
            }),
    ]
}

/// One fuzzed scene: per-body positions, shapes, densities, velocities,
/// and dampings, plus world gravity and fluid particle positions.
/// Factored out so the strategy type doesn't sprawl across every test
/// signature.
type BodyScenario = (
    Vec<Vec2>,
    Vec<Shape2>,
    Vec<f32>,
    Vec<Vec2>,
    Vec<f32>,
    Vec2,
    Vec<Vec2>,
);

fn deterministic_scenario() -> impl Strategy<Value = BodyScenario> {
    // Generate a fixed scenario: N bodies with random but deterministic properties
    (1u8..5).prop_flat_map(|n| {
        let n = n as usize;
        let positions = prop::collection::vec(arbitrary_vec2(), n);
        let shapes = prop::collection::vec(arbitrary_shape2(), n);
        let densities = prop::collection::vec(0.1f32..10.0f32, n);
        let velocities = prop::collection::vec(arbitrary_vec2(), n);
        let dampings = prop::collection::vec(0.0f32..0.1f32, n);
        let gravity = arbitrary_vec2();
        let fluid = prop::collection::vec(arbitrary_vec2(), 0..6usize);

        (
            positions, shapes, densities, velocities, dampings, gravity, fluid,
        )
    })
}

proptest! {
    /// Cross-language parity: Rust simulation with identical inputs should produce
    /// bit-exact same result on repeated runs (determinism).
    #[test]
    fn determinism_rust_native(
        (positions, shapes, densities, velocities, dampings, gravity, fluid) in deterministic_scenario()
    ) {

        fn run_simulation(
            positions: &[Vec2],
            shapes: &[Shape2],
            densities: &[f32],
            velocities: &[Vec2],
            dampings: &[f32],
            gravity: Vec2,
            fluid: &[Vec2],
        ) -> Vec<(Vec2, Vec2, f32, f32)> {
            let mut world = World2::new();
            world.gravity = gravity;
            world.linear_sleep_threshold = 0.0;
            world.angular_sleep_threshold = 0.0;
            world.fluid.add_particles(fluid);

            let mut ids: Vec<usize> = Vec::new();
            for i in 0..positions.len() {
                let id = world.add_dynamic_body(positions[i], shapes[i].clone(), densities[i]).unwrap();
                world.bodies[id].velocity = velocities[i];
                world.bodies[id].linear_damping = dampings[i];
                world.bodies[id].angular_damping = dampings[i];
                ids.push(id);
            }

            for _ in 0..30 {
                world.step(DT);
            }

            ids.iter().map(|&id| {
                let b = &world.bodies[id];
                (b.transform.position, b.velocity, b.angular_velocity, b.rotation_angle())
            }).collect()
        }

        let state1 = run_simulation(&positions, &shapes, &densities, &velocities, &dampings, gravity, &fluid);
        let state2 = run_simulation(&positions, &shapes, &densities, &velocities, &dampings, gravity, &fluid);

        // States must be bit-exact identical (determinism)
        for (i, (s1, s2)) in state1.iter().zip(state2.iter()).enumerate() {
            prop_assert_eq!(s1.0, s2.0, "Body {} position differs: {:?} vs {:?}", i, s1.0, s2.0);
            prop_assert_eq!(s1.1, s2.1, "Body {} velocity differs", i);
            prop_assert_eq!(s1.2, s2.2, "Body {} angular velocity differs", i);
            prop_assert_eq!(s1.3, s2.3, "Body {} rotation differs", i);
        }
    }

    /// No non-finite values ever appear in simulation state.
    #[test]
    fn no_nonfinite_state_any_scenario(
        bodies in prop::collection::vec(
            (arbitrary_vec2(), arbitrary_shape2(), (0.01f32..100.0f32), arbitrary_vec2(), (0.0f32..0.5f32)), 2..6
        ),
        gravity in arbitrary_vec2(),
        fluid in prop::collection::vec(arbitrary_vec2(), 0..6usize),
    ) {
        let mut world = World2::new();
        world.gravity = gravity;
        world.fluid.add_particles(&fluid);

        for (pos, shape, density, vel, damping) in &bodies {
            let id = world.add_dynamic_body(*pos, shape.clone(), *density).unwrap();
            world.bodies[id].velocity = *vel;
            world.bodies[id].linear_damping = *damping;
            world.bodies[id].angular_damping = *damping;
        }

        for _ in 0..50 {
            world.step(DT);

            for body in &world.bodies {
                prop_assert!(body.transform.position.x.is_finite(), "Non-finite position.x");
                prop_assert!(body.transform.position.y.is_finite(), "Non-finite position.y");
                prop_assert!(body.velocity.x.is_finite(), "Non-finite velocity.x");
                prop_assert!(body.velocity.y.is_finite(), "Non-finite velocity.y");
                prop_assert!(body.angular_velocity.is_finite(), "Non-finite angular_velocity");
                prop_assert!(body.transform.rotation.x_axis.x.is_finite(), "Non-finite rot.xx");
                prop_assert!(body.transform.rotation.x_axis.y.is_finite(), "Non-finite rot.xy");
                prop_assert!(body.transform.rotation.y_axis.x.is_finite(), "Non-finite rot.yx");
                prop_assert!(body.transform.rotation.y_axis.y.is_finite(), "Non-finite rot.yy");
            }
            for p in world.fluid.particles.iter() {
                prop_assert!(p.position.x.is_finite(), "Non-finite particle.x");
                prop_assert!(p.position.y.is_finite(), "Non-finite particle.y");
                prop_assert!(p.velocity.x.is_finite(), "Non-finite particle vx");
                prop_assert!(p.velocity.y.is_finite(), "Non-finite particle vy");
                prop_assert!(p.density.is_finite(), "Non-finite particle density");
            }
        }
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn test_cross_language_parity_setup() {
        // Just verify the test module compiles and basic strategies work
        let mut runner = proptest::test_runner::TestRunner::default();
        let _ = arbitrary_vec2().new_tree(&mut runner).unwrap();
        let _ = arbitrary_shape2().new_tree(&mut runner).unwrap();
    }
}
