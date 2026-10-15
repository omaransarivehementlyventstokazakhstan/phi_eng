use glam::Vec2;
use phieng::*;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::World2;

fn main() {
    println!("Physics Engine Demo - 2D");
    println!("========================");
    println!();

    // Create a new physics world
    let mut world = World2::new();

    // Set gravity
    world.gravity = Vec2::new(0.0, -9.81);

    // Add a static floor
    let floor_id = world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    println!("Added floor at y=-1.0 (id: {})", floor_id);

    // Add a static wall on the left
    let wall_id = world
        .add_static_body(
            Vec2::new(-5.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 5.0),
            },
        )
        .expect("valid shape");
    println!("Added left wall at x=-5.0 (id: {})", wall_id);

    // Add a static wall on the right
    let wall2_id = world
        .add_static_body(
            Vec2::new(5.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 5.0),
            },
        )
        .expect("valid shape");
    println!("Added right wall at x=5.0 (id: {})", wall2_id);

    // Add some dynamic balls
    println!();
    println!("Dropping balls...");
    for i in 0..10 {
        let x = -4.0 + (i as f32 * 0.8);
        let y = 5.0 + (i as f32 * 0.5);
        let ball_id = world
            .add_dynamic_body(
                Vec2::new(x, y),
                Shape2::Circle { radius: 0.3 },
                1.0, // density
            )
            .expect("valid shape");
        println!(
            "  Ball {} at ({:.1}, {:.1}) (id: {})",
            ball_id, x, y, ball_id
        );
    }

    // Add a dynamic box
    let box_id = world
        .add_dynamic_body(
            Vec2::new(0.0, 3.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    println!("  Box at (0.0, 3.0) (id: {})", box_id);

    // Simulate for 60 frames (1 second at 60 FPS)
    println!();
    println!("Simulating for 1 second (60 frames)...");
    let dt = 1.0 / 60.0;
    for frame in 0..60 {
        world.step(dt);

        // Print state every 15 frames
        if frame % 15 == 0 {
            println!("  Frame {}: ", frame);
            for body in world.bodies.iter() {
                if !body.is_static {
                    println!(
                        "    Body {}: pos=({:.2}, {:.2}), vel=({:.2}, {:.2})",
                        body.id,
                        body.transform.position.x,
                        body.transform.position.y,
                        body.velocity.x,
                        body.velocity.y
                    );
                }
            }
        }
    }

    println!();
    println!("Final state:");
    for body in world.bodies.iter() {
        println!(
            "  Body {}: pos=({:.2}, {:.2}), vel=({:.2}, {:.2}), static={}",
            body.id,
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.is_static
        );
    }

    println!();
    println!("Demo complete!");
}
