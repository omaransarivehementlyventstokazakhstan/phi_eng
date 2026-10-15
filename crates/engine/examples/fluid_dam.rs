//! hello-fluid: a 5-minute 2D SPH dam break.
//!
//! Builds a tank, fills one corner with fluid particles, steps the
//! world, and prints collapse readings. Run with:
//! `cargo run -p phieng --example fluid_dam --release`

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::World2;
use std::time::Instant;

fn main() {
    let mut world = World2::new();
    // Tank: floor plus both walls (the dam sits in the left corner).
    world
        .add_static_body(
            Vec2::new(2.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(3.0, 0.5),
            },
        )
        .expect("valid shape");
    for x in [-1.0f32, 5.0] {
        world
            .add_static_body(
                Vec2::new(x, 2.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 3.5),
                },
            )
            .expect("valid shape");
    }
    // Water column: 8 x 12 lattice lying against the wall.
    let mut pts = Vec::new();
    for iy in 0..12 {
        for ix in 0..8 {
            pts.push(Vec2::new(ix as f32 * 0.25, iy as f32 * 0.25));
        }
    }
    println!("seeded {} particles", world.fluid.add_particles(&pts));

    let dt = 1.0 / 60.0;
    let mut peak = 0.0f32;
    let mut max_x = 0.0f32;
    let start = Instant::now();
    for _ in 0..600 {
        world.step(dt);
        for p in world.fluid.particles.iter() {
            peak = peak.max(p.velocity.length());
            max_x = max_x.max(p.position.x);
        }
    }
    println!(
        "600 steps in {:.1} ms ({:.4} ms/step)",
        start.elapsed().as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0 / 600.0
    );
    println!("collapse reach: max_x={max_x:.2} (started < 1.75)");
    println!("peak particle speed: {peak:.2} m/s");
    println!("done: the column should have slumped rightward and stayed in the tank.");
}
