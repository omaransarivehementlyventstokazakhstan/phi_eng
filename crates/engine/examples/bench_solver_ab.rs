//! Same-binary, same-process A/B of two implementations of one phase.
//!
//! Cross-binary benchmarking on a busy machine cannot resolve small
//! effects: two builds of *identical* source routinely differ by 6-10 %
//! per phase because the code layout moved. That offset never averages
//! out (it is a fixed property of each binary), and the heap layout adds
//! a second one: a variant that allocates a different scratch moves the
//! body's array to a different address and changes its cache-set
//! aliasing, worth several percent in *every* phase.
//!
//! This harness removes all three confounds:
//!   * one binary, so there is no code-layout difference;
//!   * one world, so there is no heap-layout difference;
//!   * the two variants alternate every few steps, so the pile's own
//!     settling drift (the pile keeps moving while the measurement runs,
//!     so whichever variant is measured last looks cheapest) hits both
//!     equally.
//!
//! The variants are selected by the `phase-timing` build's profiling
//! switches, which do not exist in the shipped build.
//!
//! Run with:
//!
//! ```text
//! cargo run -p phieng --example bench_solver_ab --release --features phase-timing
//! ```

#[cfg(feature = "phase-timing")]
use phieng_dynamics::body::Shape2;
#[cfg(feature = "phase-timing")]
use phieng_dynamics::World2;

/// The A/B harness. Only compiled with the `phase-timing` feature.
#[cfg(feature = "phase-timing")]
mod harness {
    use super::*;
    use glam::Vec2;
    use phieng_dynamics::PhaseTimings;

    const DT: f32 = 1.0 / 60.0;
    /// Steps per measured segment. Short enough that the settling drift
    /// over one segment is negligible, long enough that the timing is
    /// not dominated by the clock's own resolution.
    const SEGMENT_STEPS: usize = 20;
    /// Measured segments. The two variants alternate every segment, so
    /// each covers half of them.
    const SEGMENTS: usize = 40;

    /// Which implementation a measured segment uses.
    ///
    /// Every variant leaves the trajectory bit-identical -- that is the
    /// point of the switches -- so the world stays in lockstep whichever
    /// variant runs, and the two cover the same span of the simulation.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Variant {
        /// The solver's packed body view (the shipped one).
        PackedSolver,
        /// The solver's direct fat-body view.
        DirectSolver,
        /// The broad-phase grid's pair ownership rule (the shipped one).
        OwnedPairs,
        /// The broad-phase grid's old behaviour: the pair is generated
        /// in every cell the two bodies share.
        UnownedPairs,
        /// The broad-phase grid's static-cell skip (the shipped one).
        SkipStaticCells,
        /// The broad-phase grid's old behaviour: every candidate pair in
        /// every cell is walked, and the static-static ones are dropped
        /// one by one.
        WalkStaticCells,
    }

    impl Variant {
        /// Label used in the report.
        fn name(self) -> &'static str {
            match self {
                Variant::PackedSolver => "packed",
                Variant::DirectSolver => "direct",
                Variant::OwnedPairs => "owned",
                Variant::UnownedPairs => "unowned",
                Variant::SkipStaticCells => "skip",
                Variant::WalkStaticCells => "walk",
            }
        }

        /// Select this variant in the world.
        fn apply(self, world: &mut World2) {
            match self {
                Variant::PackedSolver | Variant::OwnedPairs | Variant::SkipStaticCells => {
                    world.solver.packed = true;
                    world.set_broadphase_owned_pairs(true);
                    world.set_broadphase_skip_static_cells(true);
                }
                Variant::DirectSolver => {
                    world.solver.packed = false;
                    world.set_broadphase_owned_pairs(true);
                    world.set_broadphase_skip_static_cells(true);
                }
                Variant::UnownedPairs => {
                    world.solver.packed = true;
                    world.set_broadphase_owned_pairs(false);
                    world.set_broadphase_skip_static_cells(true);
                }
                Variant::WalkStaticCells => {
                    world.solver.packed = true;
                    world.set_broadphase_owned_pairs(true);
                    world.set_broadphase_skip_static_cells(false);
                }
            }
        }
    }

    /// FNV-1a over the bodies' state, the same hash `bench_large` pins.
    fn final_hash(world: &World2) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for body in &world.bodies {
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

    /// Build the pile scene, identical for every variant.
    ///
    /// The body count is a parameter so the same harness can be run at
    /// several scene sizes: the packed view's whole point is that its
    /// working set is 32 bytes per body, so the wider the scene, the
    /// more the direct view's fat bodies should hurt.
    ///
    /// `shape_mix` selects the shapes: 0 is the benchmark's mix (every
    /// third body a box), 1 is all circles, 2 is all boxes. The mix
    /// decides which narrowphase path each pair takes, so the three
    /// scenes price the circle-circle, circle-box and box-box paths.
    fn build_pile(bodies: usize, shape_mix: u8) -> World2 {
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
        for i in 0..bodies {
            let x = -14.0 + (i as f32 % 30.0) * 0.95;
            let y = 2.0 + (i as f32 / 30.0) * 0.75;
            let is_box = match shape_mix {
                1 => false,
                2 => true,
                _ => i % 3 == 0,
            };
            if is_box {
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
        world
    }

    /// The static-heavy scene: 400 static boxes and 20 dynamic circles.
    ///
    /// The statics overlap heavily and each spans several cells, so the
    /// grid holds a quadratic number of static-static candidate pairs —
    /// none of which can ever produce an impulse.
    fn build_statics() -> World2 {
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
        world
    }

    /// Phase names, in pipeline order.
    const PHASES: [&str; 10] = [
        "gravity",
        "fluid",
        "bp:rebuild",
        "bp:pairs",
        "narrowphase",
        "cache",
        "solver",
        "integrate",
        "thermal",
        "sleep",
    ];

    /// Pull one phase's milliseconds per step out of a timing block.
    fn phase_ms(timings: &PhaseTimings, name: &str) -> f64 {
        let steps = timings.steps as f64;
        match name {
            "gravity" => timings.gravity / steps,
            "fluid" => timings.fluid / steps,
            "bp:rebuild" => timings.broadphase_rebuild / steps,
            "bp:pairs" => timings.broadphase_pairs / steps,
            "narrowphase" => timings.narrowphase / steps,
            "cache" => timings.cache / steps,
            "solver" => timings.solver / steps,
            "integrate" => timings.integrate / steps,
            "thermal" => timings.thermal / steps,
            "sleep" => timings.sleep / steps,
            _ => 0.0,
        }
    }

    /// Settle the pile the way `bench_large` and `bench_phases` do.
    fn settle(world: &mut World2) {
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
    }

    /// The definitive A/B: ONE world, ONE heap layout, the two variants
    /// alternating segment by segment.
    ///
    /// `scene` picks the world: 0 is the 1500-body pile, 1 is the
    /// static-heavy scene.
    pub fn run_alternating(a: Variant, b: Variant, bodies: usize, shape_mix: u8, scene: u8) {
        let mut world = if scene == 1 {
            build_statics()
        } else {
            build_pile(bodies, shape_mix)
        };
        if scene == 1 {
            for _ in 0..60 {
                world.step(DT);
            }
        } else {
            settle(&mut world);
        }

        // Any scratch a variant allocates must already be allocated
        // before the measured segments, or the first segment of that
        // variant would change the heap layout the other is measured
        // against.
        for variant in [a, b] {
            variant.apply(&mut world);
            world.step(DT);
        }

        let mut acc_a: Vec<f64> = vec![0.0; PHASES.len()];
        let mut acc_b: Vec<f64> = vec![0.0; PHASES.len()];
        let mut total_a = 0.0;
        let mut total_b = 0.0;

        for segment in 0..SEGMENTS {
            let is_a = segment % 2 == 0;
            let variant = if is_a { a } else { b };
            variant.apply(&mut world);
            world.reset_phase_timings();
            let start = std::time::Instant::now();
            for _ in 0..SEGMENT_STEPS {
                world.step(DT);
            }
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let timings = *world.phase_timings();
            let steps = SEGMENT_STEPS as f64;
            for (i, name) in PHASES.iter().enumerate() {
                let ms = phase_ms(&timings, name) * steps;
                if is_a {
                    acc_a[i] += ms;
                } else {
                    acc_b[i] += ms;
                }
            }
            if is_a {
                total_a += wall_ms;
            } else {
                total_b += wall_ms;
            }
        }

        println!("final hash {:#x}", final_hash(&world));
        println!(
            "\n{:<14} {:>10} {:>10} {:>9}   ({} vs {})",
            "phase",
            a.name(),
            b.name(),
            "delta",
            a.name(),
            b.name()
        );
        for (i, name) in PHASES.iter().enumerate() {
            let va = acc_a[i];
            let vb = acc_b[i];
            let delta = 100.0 * (va - vb) / vb;
            println!("{name:<14} {va:>10.4} {vb:>10.4} {delta:>8.1}%");
        }
        let total_delta = 100.0 * (total_a - total_b) / total_b;
        println!(
            "{:<14} {:>10.4} {:>10.4} {:>8.1}%  (wall clock)",
            "TOTAL", total_a, total_b, total_delta
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `bench_solver_ab [comparison] [bodies]`. The body count defaults
    // to the 1500 of `bench_large`; larger scenes show how the two
    // variants scale.
    #[cfg(feature = "phase-timing")]
    {
        use harness::Variant;
        let comparison = args.first().map(String::as_str).unwrap_or("solver");
        let bodies: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
        let shape_mix: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let (a, b) = match comparison {
            "pairs" => (Variant::OwnedPairs, Variant::UnownedPairs),
            "statics" => (Variant::SkipStaticCells, Variant::WalkStaticCells),
            _ => (Variant::PackedSolver, Variant::DirectSolver),
        };
        let scene: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
        harness::run_alternating(a, b, bodies, shape_mix, scene);
    }
    #[cfg(not(feature = "phase-timing"))]
    {
        let _ = args;
        eprintln!("enable the `phase-timing` feature to run this example");
    }
}
