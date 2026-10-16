//! N-body exact vs Barnes-Hut crossover benchmark.
//!
//! Builds 2D worlds with N dynamic bodies on a deterministic grid,
//! steps each gravity mode (exact pairwise, BH theta 0.5/1.0), and
//! reports wall time per step plus the trajectory hash. The two modes
//! are approximate-vs-exact by design, so hashes are expected to
//! DIFFER between modes — the gate is determinism within a mode
//! (re-run and compare), not cross-mode equality. Run with:
//! `cargo run -p phieng --example bench_gravity --release`

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::{GravityMode, World2};
use std::time::Instant;

fn scene(n: usize, mode: GravityMode, theta: f32) -> World2 {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.gravity_config.mode = mode;
    w.gravity_config.big_g = 1.0;
    w.gravity_config.softening = 0.5;
    w.gravity_config.theta = theta;
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    let side = (n as f32).sqrt().ceil() as usize;
    for i in 0..n {
        let x = (i % side) as f32 * 3.0 - side as f32 * 1.5;
        let y = (i / side) as f32 * 3.0 - side as f32 * 1.5;
        let id = w
            .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].linear_damping = 0.0;
        w.bodies[id].angular_damping = 0.0;
        w.bodies[id].velocity = Vec2::new(0.05 * (i % 7) as f32, -0.03 * (i % 5) as f32);
    }
    w
}

fn hash_world(w: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &w.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn bench(n: usize, mode: GravityMode, theta: f32, steps: usize) -> (f64, u64) {
    let mut w = scene(n, mode, theta);
    // Warm-up (allocator, branch predictors) outside the window.
    for _ in 0..5 {
        w.step(1.0 / 60.0);
    }
    let start = Instant::now();
    for _ in 0..steps {
        w.step(1.0 / 60.0);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    (ms, hash_world(&w))
}

fn main() {
    let steps = 30;
    for n in [200usize, 500, 1000, 2000] {
        let (exact_ms, exact_hash) = bench(n, GravityMode::NBody, 0.5, steps);
        let (bh05_ms, bh05_hash) = bench(n, GravityMode::BarnesHut, 0.5, steps);
        let (bh10_ms, bh10_hash) = bench(n, GravityMode::BarnesHut, 1.0, steps);
        println!(
            "n={n}: exact={exact_ms:.2}ms/step hash={exact_hash:#x} | \
             bh0.5={bh05_ms:.2}ms/step hash={bh05_hash:#x} | \
             bh1.0={bh10_ms:.2}ms/step hash={bh10_hash:#x}"
        );
    }
    println!(
        "note: exact vs BH hashes differ by design (approximation); re-run to verify determinism"
    );
}
