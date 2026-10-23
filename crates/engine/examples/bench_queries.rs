//! Query prefilter A/B: the world's prefiltered query against the same
//! query written as a brute-force loop over every body.
//!
//! The point is not just speed. Both arms must agree on every answer —
//! the prefilter is an optimisation only if it is invisible. The
//! harness asserts that on every probe, then reports the speed ratio.
//!
//! Run with: `cargo run --release -p phieng --example bench_queries`

use std::time::Instant;

use glam::Vec2;
use phieng_dynamics::body::QueryFilter;
use phieng_dynamics::shape_query::{closest_point2, point_in_shape2, shapes_overlap2};
use phieng_dynamics::{Shape2, World2};

/// A grid of static circles plus a few walls, so queries hit often
/// enough that the prefilter still has to do real work.
fn scene(n_side: usize) -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let step = 1.0f32;
    for iy in 0..n_side {
        for ix in 0..n_side {
            let p = Vec2::new(ix as f32 * step, iy as f32 * step);
            w.add_static_body(p, Shape2::Circle { radius: 0.45 })
                .expect("valid shape");
        }
    }
    w
}

fn brute_point_test(world: &World2, point: Vec2) -> Vec<u32> {
    let mut out = Vec::new();
    for body in &world.bodies {
        let Some(shape) = body.shape.as_ref() else {
            continue;
        };
        if point_in_shape2(
            shape,
            body.transform.position,
            body.transform.rotation,
            point,
        ) {
            out.push(body.id as u32);
        }
    }
    out
}

fn brute_closest(world: &World2, point: Vec2, max: f32) -> Option<(u32, f32)> {
    let mut best: Option<(u32, f32)> = None;
    for body in &world.bodies {
        let Some(shape) = body.shape.as_ref() else {
            continue;
        };
        let Some(hit) = closest_point2(
            shape,
            body.transform.position,
            body.transform.rotation,
            point,
        ) else {
            continue;
        };
        if !hit.inside && hit.distance > max {
            continue;
        }
        let nearer = match best {
            Some((_, d)) => hit.distance < d,
            None => true,
        };
        if nearer {
            best = Some((body.id as u32, hit.distance));
        }
    }
    best
}

fn brute_overlap(world: &World2, shape: &Shape2, position: Vec2) -> Vec<u32> {
    let mut out = Vec::new();
    let id = glam::Mat2::IDENTITY;
    for body in &world.bodies {
        let Some(other) = body.shape.as_ref() else {
            continue;
        };
        if shapes_overlap2(
            shape,
            position,
            id,
            other,
            body.transform.position,
            body.transform.rotation,
        ) {
            out.push(body.id as u32);
        }
    }
    out
}

fn main() {
    let world = scene(100);
    let all = QueryFilter::everything();
    println!("bodies: {}", world.bodies.len());

    // Deterministic probes across the grid.
    let probes: Vec<Vec2> = (0..200)
        .map(|k| {
            let f = k as f32 * 0.37;
            Vec2::new(f % 99.0, (f * 1.7) % 99.0)
        })
        .collect();

    // Correctness first: identical answers on every probe.
    let mut checksum = 0u64;
    for p in &probes {
        let fast = world.point_test(*p, &all);
        let slow = brute_point_test(&world, *p);
        assert_eq!(fast, slow, "point test diverged at {p:?}");
        checksum = checksum.wrapping_add(fast.len() as u64);
        let f2 = world.closest_point(*p, &all, 1000.0);
        let s2 = brute_closest(&world, *p, 1000.0);
        assert_eq!(
            f2.map(|h| (h.body, h.distance)),
            s2,
            "closest point diverged at {p:?}"
        );
    }
    let probe_shape = Shape2::Circle { radius: 0.6 };
    for p in probes.iter().take(50) {
        let fast = world.overlap_shape(&probe_shape, *p, glam::Mat2::IDENTITY, &all);
        let slow = brute_overlap(&world, &probe_shape, *p);
        assert_eq!(fast, slow, "overlap diverged at {p:?}");
    }
    println!("answers identical on every probe (checksum {checksum})");

    // Timing: alternate the arms so drift hits both equally.
    let (mut t_fast, mut t_slow) = (0.0f64, 0.0f64);
    for round in 0..5 {
        let start = Instant::now();
        let mut sink = 0usize;
        for p in &probes {
            sink += world.point_test(*p, &all).len();
        }
        t_fast += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);

        let start = Instant::now();
        let mut sink = 0usize;
        for p in &probes {
            sink += brute_point_test(&world, *p).len();
        }
        t_slow += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);
        println!("round {round}: point test done");
    }
    println!(
        "point test: prefiltered {:.3} ms, brute {:.3} ms, {:.1}x",
        t_fast * 1e3,
        t_slow * 1e3,
        t_slow / t_fast
    );

    let (mut c_fast, mut c_slow) = (0.0f64, 0.0f64);
    for _ in 0..5 {
        let start = Instant::now();
        let mut sink = 0.0f32;
        for p in &probes {
            if let Some(h) = world.closest_point(*p, &all, 1000.0) {
                sink += h.distance;
            }
        }
        c_fast += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);

        let start = Instant::now();
        let mut sink = 0.0f32;
        for p in &probes {
            if let Some((_, d)) = brute_closest(&world, *p, 1000.0) {
                sink += d;
            }
        }
        c_slow += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);
    }
    println!(
        "closest:    prefiltered {:.3} ms, brute {:.3} ms, {:.1}x",
        c_fast * 1e3,
        c_slow * 1e3,
        c_slow / c_fast
    );

    let (mut o_fast, mut o_slow) = (0.0f64, 0.0f64);
    let id = glam::Mat2::IDENTITY;
    for _ in 0..5 {
        let start = Instant::now();
        let mut sink = 0usize;
        for p in probes.iter().take(50) {
            sink += world.overlap_shape(&probe_shape, *p, id, &all).len();
        }
        o_fast += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);

        let start = Instant::now();
        let mut sink = 0usize;
        for p in probes.iter().take(50) {
            sink += brute_overlap(&world, &probe_shape, *p).len();
        }
        o_slow += start.elapsed().as_secs_f64();
        std::hint::black_box(sink);
    }
    println!(
        "overlap:    prefiltered {:.3} ms, brute {:.3} ms, {:.1}x",
        o_fast * 1e3,
        o_slow * 1e3,
        o_slow / o_fast
    );
}
