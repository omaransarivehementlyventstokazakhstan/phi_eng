//! Same-binary, same-process A/B of the solver's body view.
//!
//! Cross-binary benchmarking on a busy machine cannot resolve small
//! effects: two builds of *identical* source routinely differ by 6-10 %
//! per phase because the code layout moved, and that offset never
//! averages out (it is a fixed property of each binary). This example
//! removes both confounds Ã¢â‚¬â€ the two variants live in one binary, and
//! they are measured in the same process, back to back.
//!
//! The variants are the solver's packed body view (the default) and the
//! direct fat-body view, selected by `Solver::packed`. The switch only
//! exists in the `phase-timing` build, so the shipped default build has
//! exactly one path.
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
    /// Bodies in the 2D pile. Matches `bench_large` and `bench_phases`.
    const BODIES: usize = 1500;
    /// Steps per measurement block. Long enough that the first (cold)
    /// steps do not dominate the block's average.
    const BLOCK: usize = 300;
    /// Measurement blocks. The reported number per phase is the minimum
    /// over the blocks, which is the least perturbed sample.
    const BLOCKS: usize = 7;
    /// Steps per measured segment. Short enough that the settling drift
    /// over one segment is negligible, long enough that the timing is
    /// not dominated by the clock's own resolution.
    const SEGMENT_STEPS: usize = 20;
    /// Measured segments. The two variants alternate every segment, so
    /// each covers half of them.
    const SEGMENTS: usize = 40;

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
    fn build_pile(packed: bool) -> World2 {
        let mut world = World2::new();
        world.gravity = Vec2::new(0.0, -9.81);
        world.solver.packed = packed;
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

    /// Measure ONE variant in this process, alone, with a warm cache.
    ///
    /// The interleaved mode above is the fairest comparison of the two
    /// paths, but it evicts each world's bodies from cache on every step,
    /// which is not how the engine runs. This mode measures one variant
    /// in a dedicated process instead, and the driver alternates the two
    /// processes: same binary (no layout confound), realistic cache
    /// behaviour, and the order alternation cancels the CPU-boost bias.
    ///
    /// Prints `phase <name> <ms>` lines and a `total <ms>` line.
    pub fn run_solo(packed: bool) {
        let mut world = build_pile(packed);
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

        let mut best: Vec<f64> = vec![f64::MAX; PHASES.len()];
        let mut best_total = f64::MAX;
        for _ in 0..BLOCKS {
            world.reset_phase_timings();
            let start = std::time::Instant::now();
            for _ in 0..BLOCK {
                world.step(DT);
            }
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let timings = *world.phase_timings();
            for (i, name) in PHASES.iter().enumerate() {
                best[i] = best[i].min(phase_ms(&timings, name));
            }
            best_total = best_total.min(wall_ms / BLOCK as f64);
        }
        for (i, name) in PHASES.iter().enumerate() {
            println!("phase {name} {:.6}", best[i]);
        }
        println!("total {best_total:.6}");
    }

    /// The definitive A/B: ONE world, ONE heap layout, the two solver
    /// paths alternating segment by segment.
    ///
    /// Every other comparison has a confound this one does not:
    ///   * two binaries differ in code layout (6-10 % on this machine);
    ///   * two processes differ in heap layout Ã¢â‚¬â€ the packed path
    ///     allocates a 48 KB scratch, which moves the body's array to a
    ///     different address and changes its cache-set aliasing, worth
    ///     several percent in EVERY phase;
    ///   * two interleaved worlds evict each other from cache;
    ///   * long blocks drift, because the pile keeps settling while the
    ///     measurement runs, so whichever variant is measured last looks
    ///     cheapest no matter which code it ran.
    ///
    /// Here the world, its bodies and its heap are identical for both
    /// variants; only the solver's body view changes. The variant
    /// alternates every few steps, so the two variants cover the same
    /// span of the simulation and the settling drift hits both equally.
    /// The reported number per phase is the total over all segments.
    pub fn run_alternating() {
        let mut world = build_pile(true);
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

        // The packed path's scratch must already be allocated before the
        // measured segments, or the first packed segment would change
        // the heap layout the direct segments are measured against.
        world.solver.packed = true;
        world.step(DT);

        let mut acc_packed: Vec<f64> = vec![0.0; PHASES.len()];
        let mut acc_direct: Vec<f64> = vec![0.0; PHASES.len()];
        let mut total_packed = 0.0;
        let mut total_direct = 0.0;
        let mut trend: Vec<f64> = Vec::new();

        for segment in 0..SEGMENTS {
            let is_packed = segment % 2 == 0;
            world.solver.packed = is_packed;
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
                if is_packed {
                    acc_packed[i] += ms;
                } else {
                    acc_direct[i] += ms;
                }
            }
            trend.push(wall_ms / steps);
            if is_packed {
                total_packed += wall_ms;
            } else {
                total_direct += wall_ms;
            }
        }

        println!("final hash {:#x}", final_hash(&world));
        // The per-segment trend must be flat: if it is not, the scene is
        // still changing and the comparison is not fair.
        let first = trend[0];
        let last = trend[trend.len() - 1];
        println!(
            "per-step trend: first {:.4} ms, last {:.4} ms ({:+.1} %)",
            first,
            last,
            100.0 * (last - first) / first
        );

        println!(
            "\n{:<14} {:>10} {:>10} {:>9}",
            "phase", "packed", "direct", "delta"
        );
        for (i, name) in PHASES.iter().enumerate() {
            let p = acc_packed[i];
            let d = acc_direct[i];
            let delta = 100.0 * (p - d) / d;
            println!("{name:<14} {p:>10.4} {d:>10.4} {delta:>8.1}%");
        }
        let total_delta = 100.0 * (total_packed - total_direct) / total_direct;
        println!(
            "{:<14} {:>10.4} {:>10.4} {:>8.1}%  (wall clock)",
            "TOTAL", total_packed, total_direct, total_delta
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(feature = "phase-timing")]
    match args.first().map(String::as_str) {
        Some("packed") => harness::run_solo(true),
        Some("direct") => harness::run_solo(false),
        Some("alternate") => harness::run_alternating(),
        _ => harness::run_alternating(),
    }
    #[cfg(not(feature = "phase-timing"))]
    {
        let _ = args;
        eprintln!("enable the `phase-timing` feature to run this example");
    }
}
