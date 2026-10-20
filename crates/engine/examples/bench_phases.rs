//! Phase-level profiling companion to `bench_large`.
//!
//! `bench_large` reports the *total* step time; this example reports
//! *where* that time goes, using the `phase-timing` feature of the
//! dynamics crate. The instrumentation is observational only: it reads a
//! clock and accumulates elapsed time, so the trajectory hashes are
//! identical with the feature on and off.
//!
//! Run with:
//!
//! ```text
//! cargo run -p phieng --example bench_phases --release --features phase-timing
//! ```
//!
//! Without the feature the example prints how to enable it and exits —
//! it still compiles, because `cargo test --workspace` builds every
//! example.

#[cfg(feature = "phase-timing")]
use phieng_dynamics::body::{Shape2, Shape3};
#[cfg(feature = "phase-timing")]
use phieng_dynamics::{World2, World3};

/// The profiling body. Only compiled with the `phase-timing` feature.
#[cfg(feature = "phase-timing")]
mod profiling {
    use super::*;
    use glam::{Vec2, Vec3};
    use phieng_dynamics::PhaseTimings;

    const DT: f32 = 1.0 / 60.0;

    /// Print one phase row, in the order the step pipeline runs them.
    fn print_phases(label: &str, timings: &PhaseTimings, wall_ms: f64, scene: &str) {
        let total = timings.total_ms();
        println!("--- {label} ---");
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "gravity",
            timings.gravity / timings.steps as f64,
            100.0 * timings.gravity / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "fluid",
            timings.fluid / timings.steps as f64,
            100.0 * timings.fluid / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "bp:rebuild",
            timings.broadphase_rebuild / timings.steps as f64,
            100.0 * timings.broadphase_rebuild / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "bp:pairs",
            timings.broadphase_pairs / timings.steps as f64,
            100.0 * timings.broadphase_pairs / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "narrowphase",
            timings.narrowphase / timings.steps as f64,
            100.0 * timings.narrowphase / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "cache",
            timings.cache / timings.steps as f64,
            100.0 * timings.cache / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "solver",
            timings.solver / timings.steps as f64,
            100.0 * timings.solver / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "integrate",
            timings.integrate / timings.steps as f64,
            100.0 * timings.integrate / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "thermal",
            timings.thermal / timings.steps as f64,
            100.0 * timings.thermal / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  {:>5.1}%",
            "sleep",
            timings.sleep / timings.steps as f64,
            100.0 * timings.sleep / total
        );
        println!(
            "  {:<12} {:>9.4} ms/step  (accounted {:.1}% of the wall clock)",
            "TOTAL",
            timings.ms_per_step(),
            100.0 * total / wall_ms
        );
        println!("  scene: {scene}");
        println!();
    }

    /// The 2D pile: 1500 dynamic bodies over a static floor.
    fn pile_2d() {
        const BODIES: usize = 1500;
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(30.0, 0.5),
                },
            )
            .expect("valid shape");
        for i in 0..BODIES {
            let x = -14.0 + (i as f32 % 30.0) * 0.95;
            let y = 2.0 + (i as f32 / 30.0) * 0.75;
            if i % 3 == 0 {
                world
                    .add_dynamic_body(
                        Vec2::new(x, y),
                        Shape2::Box {
                            half_extents: Vec2::new(0.3, 0.3),
                        },
                        1.0,
                    )
                    .expect("valid shape");
            } else {
                world
                    .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.3 }, 1.0)
                    .expect("valid shape");
            }
        }
        // The same protocol as `bench_large`: warm up, then the same
        // solver sweep, so the phase split describes the same scene the
        // CI gate measures (a fully settled pile).
        for _ in 0..900 {
            world.step(DT);
        }
        for iters in [1usize, 8] {
            world.solver.iterations = iters;
            for _ in 0..30 {
                world.step(DT);
            }
            for _ in 0..5 {
                for _ in 0..300 {
                    world.step(DT);
                }
            }
        }
        world.solver.iterations = 8;
        world.reset_phase_timings();
        let start = std::time::Instant::now();
        for _ in 0..300 {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pairs = world.contact_cache.len();
        let contacts: usize = world.contact_cache.values().map(|l| l.len()).sum();
        print_phases(
            "pile 2D (1500 bodies)",
            world.phase_timings(),
            wall_ms,
            &format!(
                "{pairs} cached pairs, {contacts} cached contacts, {} raw broadphase pairs, \
{} grid cells for {} bodies",
                world.broadphase_pair_count(),
                world.broadphase_cell_count(),
                world.bodies.len()
            ),
        );
    }

    /// The 3D pile: 400 dynamic bodies over a static floor.
    fn pile_3d() {
        const BODIES: usize = 400;
        let mut world = World3::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        world
            .add_static_body(
                Vec3::new(0.0, -1.0, 0.0),
                Shape3::Box {
                    half_extents: Vec3::new(30.0, 0.5, 30.0),
                },
            )
            .expect("valid shape");
        for i in 0..BODIES {
            let x = -9.0 + (i as f32 % 20.0) * 0.95;
            let z = -4.5 + ((i / 20) % 10) as f32 * 0.95;
            let y = 2.0 + (i / 200) as f32 * 0.75;
            if i % 3 == 0 {
                world
                    .add_dynamic_body(
                        Vec3::new(x, y, z),
                        Shape3::Box {
                            half_extents: Vec3::new(0.3, 0.3, 0.3),
                        },
                        1.0,
                    )
                    .expect("valid shape");
            } else {
                world
                    .add_dynamic_body(Vec3::new(x, y, z), Shape3::Sphere { radius: 0.3 }, 1.0)
                    .expect("valid shape");
            }
        }
        for _ in 0..600 {
            world.step(DT);
        }
        world.reset_phase_timings();
        let start = std::time::Instant::now();
        for _ in 0..300 {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pairs = world.contact_cache.len();
        let contacts: usize = world.contact_cache.values().map(|l| l.len()).sum();
        print_phases(
            "pile 3D (400 bodies)",
            world.phase_timings(),
            wall_ms,
            &format!(
                "{pairs} cached pairs, {contacts} cached contacts, {} raw broadphase pairs, \
{} grid cells for {} bodies",
                world.broadphase_pair_count(),
                world.broadphase_cell_count(),
                world.bodies.len()
            ),
        );
    }

    /// Eight independent stacks: the island scene.
    fn islands() {
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(60.0, 0.5),
                },
            )
            .expect("valid shape");
        for s in 0..8 {
            for i in 0..5 {
                world
                    .add_dynamic_body(
                        Vec2::new(-21.0 + s as f32 * 6.0, 0.05 + i as f32 * 1.02),
                        Shape2::Box {
                            half_extents: Vec2::new(0.5, 0.5),
                        },
                        1.0,
                    )
                    .expect("valid shape");
            }
        }
        for _ in 0..60 {
            world.step(DT);
        }
        world.reset_phase_timings();
        let start = std::time::Instant::now();
        for _ in 0..300 {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pairs = world.contact_cache.len();
        let contacts: usize = world.contact_cache.values().map(|l| l.len()).sum();
        print_phases(
            "islands (40 bodies, 8 stacks)",
            world.phase_timings(),
            wall_ms,
            &format!(
                "{pairs} cached pairs, {contacts} cached contacts, {} raw broadphase pairs, \
{} grid cells for {} bodies",
                world.broadphase_pair_count(),
                world.broadphase_cell_count(),
                world.bodies.len()
            ),
        );
    }

    /// A static-heavy scene: 400 overlapping static boxes plus 20 dynamic
    /// bodies. The broadphase generates a lot of static-static pairs.
    fn statics() {
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        for i in 0..400 {
            world
                .add_static_body(
                    Vec2::new(-5.0 + (i % 20) as f32 * 0.5, (i / 20) as f32 * 0.5),
                    Shape2::Box {
                        half_extents: Vec2::new(0.9, 0.9),
                    },
                )
                .expect("valid shape");
        }
        for i in 0..20 {
            world
                .add_dynamic_body(
                    Vec2::new(-5.0 + i as f32 * 0.5, 12.0),
                    Shape2::Circle { radius: 0.3 },
                    1.0,
                )
                .expect("valid shape");
        }
        for _ in 0..60 {
            world.step(DT);
        }
        world.reset_phase_timings();
        let start = std::time::Instant::now();
        for _ in 0..60 {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pairs = world.contact_cache.len();
        print_phases(
            "statics (420 bodies)",
            world.phase_timings(),
            wall_ms,
            &format!(
                "{pairs} cached pairs, {} raw broadphase pairs, {} grid cells for {} bodies",
                world.broadphase_pair_count(),
                world.broadphase_cell_count(),
                world.bodies.len()
            ),
        );
    }

    /// The SPH fluid: 96 particles in a tank.
    fn fluid() {
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world
            .add_static_body(
                Vec2::new(2.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(3.0, 0.5),
                },
            )
            .expect("valid shape");
        world
            .add_static_body(
                Vec2::new(-1.0, 2.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 3.5),
                },
            )
            .expect("valid shape");
        world
            .add_static_body(
                Vec2::new(5.0, 2.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 3.5),
                },
            )
            .expect("valid shape");
        let mut positions = Vec::with_capacity(96);
        for i in 0..8 {
            for j in 0..12 {
                positions.push(Vec2::new(-1.0 + i as f32 * 0.25, j as f32 * 0.25));
            }
        }
        world.fluid.add_particles(&positions);
        for _ in 0..60 {
            world.step(DT);
        }
        world.reset_phase_timings();
        let start = std::time::Instant::now();
        for _ in 0..300 {
            world.step(DT);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        print_phases(
            "fluid (96 SPH particles)",
            world.phase_timings(),
            wall_ms,
            "96 SPH particles",
        );
    }

    /// Run every scene.
    pub fn run() {
        pile_2d();
        pile_3d();
        islands();
        statics();
        fluid();
    }
}

fn main() {
    #[cfg(feature = "phase-timing")]
    profiling::run();

    #[cfg(not(feature = "phase-timing"))]
    println!(
        "bench_phases needs the `phase-timing` feature:\n  \
         cargo run -p phieng --example bench_phases --release --features phase-timing"
    );
}
