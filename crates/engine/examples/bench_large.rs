//! Large-scale throughput + allocation benchmark.
//!
//! Builds a world with ~1500 dynamic bodies over a static floor, steps it,
//! and reports wall time plus heap allocations per step (via a counting
//! global allocator). Run with:
//! `cargo run --target x86_64-pc-windows-gnu -p phieng --example bench_large --release`

use glam::{Vec2, Vec3};
use phieng_dynamics::body::{Shape2, Shape3};
use phieng_dynamics::{World2, World3};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct CountingAlloc;

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_counters() {
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

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

fn final_hash3(world: &World3) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.transform.position.z,
            body.velocity.x,
            body.velocity.y,
            body.velocity.z,
            body.angular_velocity.x,
            body.angular_velocity.y,
            body.angular_velocity.z,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Median ms/step over 5 windows of FRAMES steps (world keeps evolving).
fn median_ms<F: FnMut()>(frames: usize, mut step_fn: F) -> f64 {
    let mut samples = [0.0f64; 5];
    for s in samples.iter_mut() {
        let start = Instant::now();
        for _ in 0..frames {
            step_fn();
        }
        *s = start.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[2]
}

/// Pinned trajectory hash of the 2D pile scene.
///
/// The physics is expected to be bit-identical across builds and
/// platforms; this constant is the anchor that proves it. Re-pinning is
/// a deliberate act: if it changes, the physics moved. If the change
/// was intended, update it here AND in README.md, and say so in the
/// CHANGELOG. Never silence it by loosening the comparison.
const HASH_PILE_2D: u64 = 0x4acb6694f3a9917;

/// Pinned trajectory hash of the 8-stack island scene (serial solve).
const HASH_ISLANDS: u64 = 0xd4a3a56f185c13b2;

/// Pinned trajectory hash of the 3D pile scene.
const HASH_PILE_3D: u64 = 0x4aad96014e832f5b;

/// Fail with a message that says what to do about it.
fn check_hash(label: &str, got: u64, expected: u64) {
    if got != expected {
        panic!(
            "{label} trajectory hash changed: got {got:#x}, expected {expected:#x}.\n\
             The physics moved. If that was intended, re-pin the constant in \
             bench_large.rs AND in README.md, and note it in the CHANGELOG; \
             otherwise this is a regression."
        );
    }
}

fn main() {
    const BODIES: usize = 1500;
    const FRAMES: usize = 300;
    // `bench_large no-exit` disables solver early-exit to isolate its
    // effect on physics trajectory and allocation behavior.
    let no_exit = std::env::args().any(|a| a == "no-exit");

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

    if no_exit {
        world.solver.early_exit_tolerance = 0.0;
    }

    // Warm up (grow scratch buffers, settle contacts into a pile).
    let dt = 1.0 / 60.0;
    for _ in 0..900 {
        world.step(dt);
    }

    // Solver sweep first: separates solver cost from broad/narrowphase.
    // (Each setting re-settles briefly so iteration counts compare fairly.)
    for iters in [1usize, 8] {
        world.solver.iterations = iters;
        for _ in 0..30 {
            world.step(dt);
        }
        let mut samples = [0.0f64; 5];
        for s in samples.iter_mut() {
            let start = Instant::now();
            for _ in 0..FRAMES {
                world.step(dt);
            }
            *s = start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("iters={iters}: median {:.3} ms/step (of 5)", samples[2]);
    }
    world.solver.iterations = 8;

    // Allocation + contact census over one measured window.
    reset_counters();
    let start = Instant::now();
    let mut pair_total = 0usize;
    let mut contact_total = 0usize;
    for _ in 0..FRAMES {
        world.step(dt);
        pair_total += world.contact_cache.len();
        for list in world.contact_cache.values() {
            contact_total += list.len();
        }
    }
    let elapsed = start.elapsed();
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);

    let pile_hash = final_hash(&world);
    check_hash("pile 2D", pile_hash, HASH_PILE_2D);

    // With the `phase-timing` feature enabled, report where the step
    // time goes. The feature is off by default, so the CI bench-smoke
    // job is unaffected.
    #[cfg(feature = "phase-timing")]
    {
        let t = world.phase_timings();
        let steps = t.steps.max(1) as f64;
        println!("[phases] pile 2D");
        println!("[phases]   gravity     {:>8.4} ms/step", t.gravity / steps);
        println!("[phases]   fluid       {:>8.4} ms/step", t.fluid / steps);
        println!(
            "[phases]   bp:aabb     {:>8.4} ms/step",
            t.broadphase_aabb / steps
        );
        println!(
            "[phases]   bp:rebuild  {:>8.4} ms/step",
            t.broadphase_rebuild / steps
        );
        println!(
            "[phases]   bp:pairs    {:>8.4} ms/step",
            t.broadphase_pairs / steps
        );
        println!(
            "[phases]   narrowphase {:>8.4} ms/step",
            t.narrowphase / steps
        );
        println!("[phases]   cache       {:>8.4} ms/step", t.cache / steps);
        println!("[phases]   solver      {:>8.4} ms/step", t.solver / steps);
        println!(
            "[phases]   integrate   {:>8.4} ms/step",
            t.integrate / steps
        );
        println!("[phases]   thermal     {:>8.4} ms/step", t.thermal / steps);
        println!("[phases]   sleep       {:>8.4} ms/step", t.sleep / steps);
        println!("[phases]   TOTAL       {:>8.4} ms/step", t.ms_per_step());
    }

    println!("bodies: {}", world.bodies.len());
    println!("frames: {FRAMES}");
    println!(
        "final hash: {pile_hash:#x} (early-exit {})",
        if no_exit { "OFF" } else { "ON" }
    );
    println!("avg pairs/step: {:.0}", pair_total as f64 / FRAMES as f64);
    println!(
        "avg contacts/step: {:.0}",
        contact_total as f64 / FRAMES as f64
    );
    println!("total: {:.2?}", elapsed);
    println!(
        "per step: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0 / FRAMES as f64
    );
    println!("allocs/step: {:.1}", allocs as f64 / FRAMES as f64);
    println!("bytes/step: {:.0}", bytes as f64 / FRAMES as f64);

    // Steady-state probe: one box resting on the floor. Pairs, cache
    // and scratch are all stable here, so this measures the engine's
    // irreducible per-step allocation floor.
    let mut calm = World2::new();
    calm.gravity = Vec2::new(0.0, -9.81);
    calm.add_static_body(
        Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    calm.add_dynamic_body(
        Vec2::new(0.0, 0.2),
        Shape2::Box {
            half_extents: Vec2::new(0.5, 0.5),
        },
        1.0,
    )
    .expect("valid shape");
    for _ in 0..600 {
        calm.step(dt);
    }
    reset_counters();
    for _ in 0..FRAMES {
        calm.step(dt);
    }
    let calm_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let calm_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    println!(
        "resting allocs/step: {:.2}",
        calm_allocs as f64 / FRAMES as f64
    );
    println!(
        "resting bytes/step: {:.1}",
        calm_bytes as f64 / FRAMES as f64
    );

    // Bisect: same world but the box held high (no pairs, no contacts).
    let mut air = World2::new();
    air.gravity = Vec2::new(0.0, 0.0);
    air.add_static_body(
        Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    air.add_dynamic_body(
        Vec2::new(0.0, 50.0),
        Shape2::Box {
            half_extents: Vec2::new(0.5, 0.5),
        },
        1.0,
    )
    .expect("valid shape");
    for _ in 0..120 {
        air.step(dt);
    }
    reset_counters();
    for _ in 0..FRAMES {
        air.step(dt);
    }
    println!(
        "airborne allocs/step: {:.2}",
        ALLOC_COUNT.load(Ordering::Relaxed) as f64 / FRAMES as f64
    );

    // Bisect: resting contact but solver doing no work.
    calm.solver.iterations = 0;
    reset_counters();
    for _ in 0..FRAMES {
        calm.step(dt);
    }
    println!(
        "resting-no-solve allocs/step: {:.2}",
        ALLOC_COUNT.load(Ordering::Relaxed) as f64 / FRAMES as f64
    );

    // Bisect: totally empty world (no bodies at all).
    let mut empty = World2::new();
    for _ in 0..60 {
        empty.step(dt);
    }
    reset_counters();
    for _ in 0..FRAMES {
        empty.step(dt);
    }
    println!(
        "empty allocs/step: {:.2}",
        ALLOC_COUNT.load(Ordering::Relaxed) as f64 / FRAMES as f64
    );

    // Islands: 8 independent 5-box stacks (8 islands). Same world run
    // serial and parallel; the trajectories must be bit-identical and
    // the timing ratio is the parallel-scaling signal. Best-of-5
    // medians; machine noise is ~±40%, so read speedup as a trend and
    // identity as a boolean.
    fn build_islands() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::new(0.0, -9.81);
        w.add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(60.0, 0.5),
            },
        )
        .expect("valid shape");
        for s in 0..8 {
            let x = -21.0 + s as f32 * 6.0;
            for i in 0..5 {
                w.add_dynamic_body(
                    Vec2::new(x, 0.05 + i as f32 * 1.02),
                    Shape2::Box {
                        half_extents: Vec2::new(0.5, 0.5),
                    },
                    1.0,
                )
                .expect("valid shape");
            }
        }
        w
    }
    let mut serial = build_islands();
    for _ in 0..60 {
        serial.step(dt);
    }
    let serial_ms = median_ms(FRAMES, || serial.step(dt));
    let serial_hash = final_hash(&serial);
    let mut parallel = build_islands();
    parallel.solver.parallel = true;
    for _ in 0..60 {
        parallel.step(dt);
    }
    let parallel_ms = median_ms(FRAMES, || parallel.step(dt));
    let parallel_hash = final_hash(&parallel);
    // Allocation probe for the parallel path: the island partition and
    // the per-island index lists are rebuilt every step, so this is the
    // number of heap allocations the parallel solver costs per step.
    reset_counters();
    for _ in 0..FRAMES {
        parallel.step(dt);
    }
    let island_allocs = ALLOC_COUNT.load(Ordering::Relaxed);

    // The same probe at scale: the island partition is O(bodies), so the
    // per-step allocation count must not scale with the body count.
    fn build_pile(n: usize) -> World2 {
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
        for i in 0..n {
            let x = -14.0 + (i as f32 % 30.0) * 0.95;
            let y = 2.0 + (i as f32 / 30.0) * 0.75;
            world
                .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.3 }, 1.0)
                .expect("valid shape");
        }
        world
    }
    let mut big_parallel = build_pile(1500);
    big_parallel.solver.parallel = true;
    for _ in 0..900 {
        big_parallel.step(dt);
    }
    reset_counters();
    for _ in 0..FRAMES {
        big_parallel.step(dt);
    }
    let big_parallel_allocs = ALLOC_COUNT.load(Ordering::Relaxed);

    // Control: the identical world with the serial solver. The
    // difference between the two numbers is what the parallel path
    // costs.
    let mut big_serial = build_pile(1500);
    for _ in 0..900 {
        big_serial.step(dt);
    }
    reset_counters();
    for _ in 0..FRAMES {
        big_serial.step(dt);
    }
    let big_serial_allocs = ALLOC_COUNT.load(Ordering::Relaxed);

    check_hash("islands (serial)", serial_hash, HASH_ISLANDS);
    check_hash("islands (parallel)", parallel_hash, HASH_ISLANDS);

    println!(
        "[islands] serial={serial_ms:.4} ms/step parallel={parallel_ms:.4} ms/step \
         speedup={:.2}x identical={} serial={serial_hash:#x} parallel={parallel_hash:#x} \
         parallel_allocs/step={:.1} pile_parallel_allocs/step={:.1} pile_serial_allocs/step={:.1}",
        serial_ms / parallel_ms,
        serial_hash == parallel_hash,
        island_allocs as f64 / FRAMES as f64,
        big_parallel_allocs as f64 / FRAMES as f64,
        big_serial_allocs as f64 / FRAMES as f64,
    );

    // Direct probe of the island partition build: it is the only part of
    // the parallel path that scales with the body count.
    {
        use phieng_dynamics::island::IslandScratch;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..1500usize {
            pairs.push((i, (i + 1) % 1500));
        }
        let mut scratch = IslandScratch::new();
        // Warm up: grow every buffer to its steady-state capacity.
        for _ in 0..5 {
            scratch.build(1500, &pairs, &pairs);
        }
        reset_counters();
        for _ in 0..FRAMES {
            scratch.build(1500, &pairs, &pairs);
        }
        let build_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        let build_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
        println!(
            "[islands] build_allocs/step={:.1} build_bytes/alloc={:.0}",
            build_allocs as f64 / FRAMES as f64,
            if build_allocs > 0 {
                build_bytes as f64 / build_allocs as f64
            } else {
                0.0
            },
        );
    }

    // Barnes-Hut gravity: the tree is rebuilt every kick, so its arena,
    // per-leaf index buffers and scratch must all be recycled.
    {
        let mut bh = build_pile(1500);
        bh.gravity_config.mode = phieng_dynamics::gravity::GravityMode::BarnesHut;
        for _ in 0..60 {
            bh.step(dt);
        }
        reset_counters();
        for _ in 0..FRAMES {
            bh.step(dt);
        }
        let bh_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        println!(
            "[barnes-hut] allocs/step={:.1}",
            bh_allocs as f64 / FRAMES as f64
        );
    }

    // Static-heavy scene: a dense grid of static boxes (plus a few
    // dynamic bodies falling onto them). The static colliders alone
    // generate a large number of broadphase pairs. The main pile scene
    // has a single static floor, so it cannot show the cost of a
    // static-dominated scene.
    {
        const STATICS: usize = 400;
        let mut stat = World2::new();
        stat.gravity = Vec2::new(0.0, -9.81);
        for i in 0..STATICS {
            // Deliberately dense: the boxes overlap heavily, so the
            // broadphase emits a large number of static-static pairs.
            // That is the case where the pair list, not the narrowphase,
            // dominates the cost.
            let x = -5.0 + (i as f32 % 20.0) * 0.5;
            let y = 0.0 + (i as f32 / 20.0) * 0.5;
            stat.add_static_body(
                Vec2::new(x, y),
                Shape2::Box {
                    half_extents: Vec2::new(0.9, 0.9),
                },
            )
            .expect("valid shape");
        }
        // A few dynamic bodies so the scene is not entirely static.
        for i in 0..20 {
            stat.add_dynamic_body(
                Vec2::new(-10.0 + i as f32 * 1.0, 12.0),
                Shape2::Circle { radius: 0.3 },
                1.0,
            )
            .expect("valid shape");
        }
        for _ in 0..60 {
            stat.step(dt);
        }
        // The dense static scene is much heavier per step than the pile
        // (every static box overlaps its neighbours), so it uses fewer
        // frames per sample to keep the benchmark's runtime sane.
        const STATIC_FRAMES: usize = 60;
        let mut stat_pairs = 0usize;
        let stat_ms = median_ms(STATIC_FRAMES, || {
            stat.step(dt);
            stat_pairs += stat.contact_cache.len();
        });
        println!(
            "[statics] bodies={} per step: {stat_ms:.3} ms (median of 5) avg contacts/step: {:.0}",
            stat.bodies.len(),
            stat_pairs as f64 / (STATIC_FRAMES * 5) as f64
        );
    }

    // 3D pile: 400 bodies (20 x 10 x 2 grid of boxes/spheres) over a
    // static floor. 3D was never benched at scale; this is the control
    // column for any future RigidBody3/Solver3D layout work.
    let mut pile3 = World3::new();
    pile3.gravity = Vec3::new(0.0, -9.81, 0.0);
    pile3
        .add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(30.0, 0.5, 30.0),
            },
        )
        .expect("valid shape");
    for i in 0..400 {
        let x = -9.0 + (i % 20) as f32 * 0.95;
        let z = -4.5 + ((i / 20) % 10) as f32 * 0.95;
        let y = 2.0 + (i / 200) as f32 * 0.75;
        let shape = if i % 3 == 0 {
            Shape3::Box {
                half_extents: Vec3::new(0.3, 0.3, 0.3),
            }
        } else {
            Shape3::Sphere { radius: 0.3 }
        };
        pile3
            .add_dynamic_body(Vec3::new(x, y, z), shape, 1.0)
            .expect("valid shape");
    }
    for _ in 0..600 {
        pile3.step(dt);
    }
    for iters in [1usize, 8] {
        pile3.solver.iterations = iters;
        for _ in 0..30 {
            pile3.step(dt);
        }
        let ms = median_ms(FRAMES, || pile3.step(dt));
        println!("[pile3d] iters={iters}: median {ms:.3} ms/step (of 5)");
    }
    pile3.solver.iterations = 8;
    reset_counters();
    let mut pair3_total = 0usize;
    let mut contact3_total = 0usize;
    for _ in 0..FRAMES {
        pile3.step(dt);
        pair3_total += pile3.contact_cache.len();
        for list in pile3.contact_cache.values() {
            contact3_total += list.len();
        }
    }
    let pile3_hash = final_hash3(&pile3);
    check_hash("pile 3D", pile3_hash, HASH_PILE_3D);

    println!("bodies3d: {}", pile3.bodies.len());
    println!("final hash3d: {pile3_hash:#x}");
    println!(
        "avg pairs3d/step: {:.0}",
        pair3_total as f64 / FRAMES as f64
    );
    println!(
        "avg contacts3d/step: {:.0}",
        contact3_total as f64 / FRAMES as f64
    );
    println!(
        "allocs3d/step: {:.1}",
        ALLOC_COUNT.load(Ordering::Relaxed) as f64 / FRAMES as f64
    );
    println!(
        "bytes3d/step: {:.0}",
        ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / FRAMES as f64
    );
}
