use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::World2;

#[test]
fn test_basic_simulation() {
    let mut world = World2::new();
    world.gravity = Vec2::new(0.0, -9.81);

    // Add floor
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");

    // Add ball
    let ball_id = world
        .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");

    // Simulate
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    // Ball should have fallen
    let ball = &world.bodies[ball_id];
    assert!(ball.transform.position.y < 5.0, "Ball should have fallen");
}

#[test]
fn test_static_body() {
    let mut world = World2::new();
    let id = world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(1.0, 1.0),
            },
        )
        .expect("valid shape");

    let body = &world.bodies[id];
    assert!(body.is_static);
    assert_eq!(body.inv_mass, 0.0);
}

#[test]
fn test_dynamic_body_creation() {
    let mut world = World2::new();
    let id = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");

    let body = &world.bodies[id];
    assert!(!body.is_static);
    assert!(body.inv_mass > 0.0);
    assert!(body.shape.is_some());
}

/// Phase 1 regression: a box dropped on a static floor must bounce and
/// settle on top of the floor — never tunnel through it.
#[test]
fn test_box_rests_on_floor() {
    let mut world = World2::new();
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let box_id = world
        .add_dynamic_body(
            Vec2::new(0.0, 3.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");

    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step(dt);
    }

    let b = &world.bodies[box_id];
    // Floor top is at y=-0.5; box half-height 0.5 (+rotation slack) means the
    // center must stay well above the floor center. Tunneling would put it
    // at/below y=-1.
    assert!(
        b.transform.position.y > -0.5,
        "box tunneled through floor: y={}",
        b.transform.position.y
    );
    // Should have settled (bounce decayed via restitution + damping).
    assert!(
        b.velocity.length() < 1.0,
        "box should have settled: v={}",
        b.velocity
    );
}

/// Phase 1 regression: two overlapping boxes must produce contact points
/// (exercises SAT + clipping through the full pipeline).
#[test]
fn test_overlapping_boxes_separate() {
    use phieng_collision::sat::{polygon_polygon_contact_points, sat_polygon_polygon};

    let a = vec![
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
    ];
    let b: Vec<Vec2> = a.iter().map(|v| *v + Vec2::new(0.5, 0.2)).collect();

    let (normal, pen) = sat_polygon_polygon(&a, &b).expect("boxes overlap");
    assert!(pen > 0.0);
    let contacts = polygon_polygon_contact_points(&a, &b, normal);
    assert!(
        !contacts.is_empty(),
        "overlapping boxes must yield contact points"
    );

    // Through the world pipeline the pair must push apart along +x
    // (the least-penetration axis), not tunnel. Slight overlap keeps the
    // Baumgarte response gentle; deep initial overlap pops by design.
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let top = world
        .add_dynamic_body(
            Vec2::new(0.95, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }
    let t = &world.bodies[top];
    assert!(
        t.transform.position.x > 0.95,
        "box should separate along +x: x={}",
        t.transform.position.x
    );
    assert!(
        t.transform.position.y > -1.0,
        "box sank through static box: y={}",
        t.transform.position.y
    );
}

/// Build an inclined ramp (static box rotated by `angle`) with a dynamic box
/// resting on its top surface. Returns the dynamic box id and its start pos.
fn incline_world(angle: f32, mu: f32) -> (World2, usize, Vec2) {
    use phieng_math::Transform2;

    let mut world = World2::new();
    // Static ramp: wide box rotated about its center.
    let ramp_id = world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(5.0, 0.5),
            },
        )
        .expect("valid shape");
    world.bodies[ramp_id].transform = Transform2::from_position_angle(Vec2::new(0.0, 0.0), angle);
    world.bodies[ramp_id].material.friction = mu;

    // Surface frame: outward normal and downslope tangent.
    let rot = glam::Mat2::from_angle(angle);
    let normal = rot * Vec2::Y;
    let start = rot * Vec2::new(0.0, 0.5) + normal * 0.51;
    let box_id = world
        .add_dynamic_body(
            start,
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[box_id].transform = Transform2::from_position_angle(start, angle);
    world.bodies[box_id].material.friction = mu;
    world.bodies[box_id].material.restitution = 0.0;
    (world, box_id, start)
}

/// Phase 2: high friction must hold a box on a 25-degree incline
/// (tan(25deg) ~= 0.47 < mu=0.9), while zero friction must let it slide.
#[test]
fn test_incline_sticks_with_friction() {
    let angle = 25.0_f32.to_radians();
    let (mut world, box_id, start) = incline_world(angle, 0.9);
    let dt = 1.0 / 60.0;
    for _ in 0..300 {
        world.step(dt);
    }
    let b = &world.bodies[box_id];
    let slid = (b.transform.position - start).length();
    assert!(
        slid < 0.6,
        "high friction should hold the box on the incline, slid={slid}"
    );
    assert!(
        b.transform.position.y > -2.0,
        "box tunneled through ramp: y={}",
        b.transform.position.y
    );
}

#[test]
fn test_incline_slides_without_friction() {
    let angle = 25.0_f32.to_radians();
    let (mut world, box_id, start) = incline_world(angle, 0.0);
    let dt = 1.0 / 60.0;
    for _ in 0..300 {
        world.step(dt);
    }
    let b = &world.bodies[box_id];
    let slid = (b.transform.position - start).length();
    assert!(
        slid > 2.0,
        "zero friction should let the box slide, slid={slid}"
    );
}

/// Phase 2: a 3-box stack must settle without tunneling or jitter explosion.
#[test]
fn test_stack_stabilizes() {
    let mut world = World2::new();
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let mut ids = Vec::new();
    for i in 0..3 {
        ids.push(
            world
                .add_dynamic_body(
                    Vec2::new(0.0, 0.05 + i as f32 * 1.02),
                    Shape2::Box {
                        half_extents: Vec2::new(0.5, 0.5),
                    },
                    1.0,
                )
                .expect("valid shape"),
        );
    }
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step(dt);
    }
    for (level, id) in ids.iter().enumerate() {
        let b = &world.bodies[*id];
        // Floor top -0.5; level k rests near y = k (box height 1). Tunneling
        // would put a box far below its shelf.
        assert!(
            b.transform.position.y > level as f32 - 0.75,
            "stack level {level} tunneled: y={}",
            b.transform.position.y
        );
        assert!(
            b.velocity.length() < 1.0,
            "stack level {level} jittering: v={}",
            b.velocity
        );
    }
}

/// Phase 2: warm-start cache must carry impulses across frames for a
/// resting contact.
#[test]
fn test_warm_start_cache_carries_impulse() {
    let mut world = World2::new();
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    // Slight initial overlap so contact exists on frame 0.
    let box_id = world
        .add_dynamic_body(
            Vec2::new(0.0, -0.05),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[box_id].material.restitution = 0.0;

    world.step(1.0 / 60.0);
    let entries = world.contact_cache.get(&(0, box_id));
    assert!(
        entries.map(|v| !v.is_empty()).unwrap_or(false),
        "resting contact must be cached"
    );
    let total: f32 = entries.unwrap().iter().map(|e| e.normal_impulse).sum();
    assert!(
        total > 0.0,
        "cached normal impulse must be positive, got {total}"
    );
}

/// Drop a ball from a fixed height and measure the rebound apex.
/// Restitution combines with max, so a bouncy ball (e=0.9) on a dead
/// floor (e=0) must bounce to ~e^2 of the drop height, while a dead
/// ball (e=0) must (nearly) stick.
fn bounce_apex(ball_restitution: f32) -> f32 {
    use phieng_dynamics::PhysicsMaterial;

    let mut world = World2::new();
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    let ball = world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[ball]
        .set_material(PhysicsMaterial::new(0.4, ball_restitution, 1.0))
        .expect("valid shape");

    // Impact happens ~frame 38; scan the post-impact window for the apex.
    // (The window must exclude the pre-impact fall, whose heights would
    // otherwise dwarf small rebounds.)
    let mut apex = f32::MIN;
    for frame in 0..240 {
        world.step(1.0 / 60.0);
        if frame >= 45 {
            apex = apex.max(world.bodies[ball].transform.position.y);
        }
    }
    apex
}

/// Phase 2 (materials): restitution end-to-end through the pipeline.
#[test]
fn test_bouncy_ball_rebounds() {
    // Drop height to contact = 2.0; e=0.9 rebounds to ~0.81*2.0 = 1.62.
    let apex = bounce_apex(0.9);
    assert!(
        (0.8..2.0).contains(&apex),
        "bouncy ball should rebound high, apex={apex}"
    );
}

#[test]
fn test_dead_ball_sticks() {
    let apex = bounce_apex(0.0);
    assert!(apex < 0.6, "dead ball should not rebound, apex={apex}");
}

/// Determinism: the same scene stepped twice in one binary must produce
/// bit-identical final state (positions, velocities, angles). This is the
/// guarantee replays and lockstep netcode build on. If this ever fails,
/// something nondeterministic (iteration order, uninitialized memory,
/// time-dependent logic) crept into the pipeline.
fn deterministic_scene_hash() -> u64 {
    let mut world = World2::new();
    world.gravity = Vec2::new(0.0, -9.81);
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    // Mixed shapes with lateral offsets so contacts involve rotation,
    // friction, warm starting, and multi-point manifolds.
    for i in 0..6 {
        let x = -3.0 + i as f32 * 1.2;
        if i % 2 == 0 {
            world
                .add_dynamic_body(
                    Vec2::new(x, 2.0 + i as f32 * 0.7),
                    Shape2::Box {
                        half_extents: Vec2::new(0.4, 0.4),
                    },
                    1.0,
                )
                .expect("valid shape");
        } else {
            world
                .add_dynamic_body(
                    Vec2::new(x + 0.3, 2.5 + i as f32 * 0.7),
                    Shape2::Circle { radius: 0.35 },
                    1.5,
                )
                .expect("valid shape");
        }
    }
    let dt = 1.0 / 60.0;
    for _ in 0..300 {
        world.step(dt);
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
            body.rotation_angle(),
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[test]
fn test_determinism_bit_exact() {
    let a = deterministic_scene_hash();
    let b = deterministic_scene_hash();
    assert_eq!(a, b, "same scene must hash identically: {a:#x} vs {b:#x}");
}
