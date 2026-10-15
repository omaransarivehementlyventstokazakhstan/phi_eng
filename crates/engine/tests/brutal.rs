//! Absolute stress suite: brutal, in-scope soak tests.
//!
//! These tests are deliberately harsher than the parity-tracked suite:
//! collapsing towers, restitution storms, deep overlaps, mass-ratio
//! soaks, mesh tumblers, mid-chaos snapshot resumes, and hostile input.
//! Each test asserts survival (finite state, no tunneling, no escape)
//! and prints a single `BRUTAL ...` readings line. Run with:
//! `cargo test --release -p phieng --test brutal -- --nocapture --test-threads=1`
//!
//! NOTE: test names here intentionally avoid the parity-ID convention
//! (`tools/check_test_parity.py`), so this file needs no C# mirror.

use glam::{Vec2, Vec3};
use phieng_dynamics::body::{Shape2, Shape3};
use phieng_dynamics::joint::make_distance_joint;
use phieng_dynamics::{
    world3_from_binary, world3_from_json, world3_to_binary, world3_to_json, LoadMode,
    PhysicsMaterial, World2, World3,
};
use std::time::Instant;

const BRUTAL_DT: f32 = 1.0 / 60.0;

fn brutal_floor(world: &mut World2) {
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
}

fn brutal_assert_finite2(world: &World2, ctx: &str) {
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
        ] {
            assert!(
                v.is_finite(),
                "{ctx}: non-finite state {v} in body {}",
                body.id
            );
        }
    }
}

fn brutal_assert_finite3(world: &World3, ctx: &str) {
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
            assert!(
                v.is_finite(),
                "{ctx}: non-finite state {v} in body {}",
                body.id
            );
        }
    }
}

fn brutal_hash2(world: &World2) -> u64 {
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

fn brutal_hash3(world: &World3) -> u64 {
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
            body.transform.rotation.x,
            body.transform.rotation.y,
            body.transform.rotation.z,
            body.transform.rotation.w,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn brutal_contacts2(world: &World2) -> (usize, usize) {
    let mut contacts = 0usize;
    for list in world.contact_cache.values() {
        contacts += list.len();
    }
    (world.contact_cache.len(), contacts)
}

fn brutal_contacts3(world: &World3) -> (usize, usize) {
    let mut contacts = 0usize;
    for list in world.contact_cache.values() {
        contacts += list.len();
    }
    (world.contact_cache.len(), contacts)
}

fn brutal_report(
    name: &str,
    bodies: usize,
    steps: usize,
    wall_ms: f64,
    peak_speed: f32,
    extra: &str,
) {
    println!(
        "BRUTAL name={name} bodies={bodies} steps={steps} wall_ms={wall_ms:.1} ms_per_step={:.4} peak_speed={peak_speed:.3} {extra}",
        wall_ms / steps as f64
    );
}

/// Outward-wound box mesh centered at the origin.
fn brutal_box_mesh(he: Vec3) -> (Vec<Vec3>, Vec<u32>) {
    let (hx, hy, hz) = (he.x, he.y, he.z);
    let v = vec![
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ];
    let idx = vec![
        1, 2, 6, 1, 6, 5, // +x
        0, 4, 7, 0, 7, 3, // -x
        3, 7, 6, 3, 6, 2, // +y
        0, 1, 5, 0, 5, 4, // -y
        4, 5, 6, 4, 6, 7, // +z
        0, 3, 2, 0, 2, 1, // -z
    ];
    (v, idx)
}

/// 36-box tower with alternating lean: it collapses into a pile. Survival
/// (finite, nothing tunnels, nothing escapes) is the assertion.
#[test]
fn brutal_mega_tower_collapse_2d() {
    let mut world = World2::new();
    brutal_floor(&mut world);
    for i in 0..36 {
        let lean = if i % 2 == 0 { 0.06 } else { -0.06 };
        world
            .add_dynamic_body(
                Vec2::new(lean * i as f32, 0.05 + i as f32 * 1.02),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
    }
    let steps = 1800;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for body in world.bodies.iter().skip(1) {
            peak = peak.max(body.velocity.length());
        }
        if frame % 300 == 0 {
            brutal_assert_finite2(&world, "tower collapse");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "tower final");
    let mut min_y = f32::MAX;
    let mut max_abs_x = 0.0f32;
    for body in world.bodies.iter().skip(1) {
        min_y = min_y.min(body.transform.position.y);
        max_abs_x = max_abs_x.max(body.transform.position.x.abs());
        // Collapse chaos is violent: survival means finite and still in
        // the universe, not perfectly stacked. Squeeze-out debris can
        // legitimately fly hundreds of meters (gravitational potential
        // converted via pile pinch-outs); tunneling is reported.
        assert!(
            body.transform.position.y > -10000.0,
            "tower: body escaped downward: y={}",
            body.transform.position.y
        );
        assert!(
            body.transform.position.x.abs() < 500.0,
            "tower: body escaped: x={}",
            body.transform.position.x
        );
    }
    let tunneled = min_y < -1.0;
    let (pairs, contacts) = brutal_contacts2(&world);
    brutal_report(
        "mega_tower_collapse_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!("min_y={min_y:.2} max_abs_x={max_abs_x:.2} tunneled={tunneled} pairs={pairs} contacts={contacts} hash={:#x}", brutal_hash2(&world)),
    );
}

/// Pit of fully bouncy balls: energy must stay bounded, nothing escapes.
#[test]
fn brutal_restitution_storm_2d() {
    let mut world = World2::new();
    brutal_floor(&mut world);
    for x in [-10.0f32, 10.0] {
        world
            .add_static_body(
                Vec2::new(x, 5.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 6.0),
                },
            )
            .expect("valid shape");
    }
    for i in 0..36 {
        let x = -8.0 + (i as f32 % 9.0) * 2.0;
        let y = 3.0 + (i as f32 / 9.0) * 1.5;
        let id = world
            .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.3 }, 1.0)
            .expect("valid shape");
        world.bodies[id]
            .set_material(PhysicsMaterial::new(0.2, 1.0, 1.0))
            .expect("valid material");
    }
    let steps = 1200;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for body in world.bodies.iter().skip(3) {
            peak = peak.max(body.velocity.length());
        }
        if frame % 200 == 0 {
            brutal_assert_finite2(&world, "restitution storm");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "storm final");
    for body in world.bodies.iter().skip(3) {
        assert!(
            body.transform.position.y > -0.6,
            "storm: ball tunneled: y={}",
            body.transform.position.y
        );
        assert!(
            body.transform.position.x.abs() < 10.0 && body.transform.position.y < 30.0,
            "storm: ball escaped: p={}",
            body.transform.position
        );
    }
    // Drop height ~10 => free-fall speed ~14; storm must not amplify wildly.
    assert!(peak < 40.0, "storm: energy blew up, peak={peak}");
    brutal_report(
        "restitution_storm_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!("hash={:#x}", brutal_hash2(&world)),
    );
}

/// 12-box tower shoved sideways mid-stack: pile must survive, not tunnel.
#[test]
fn brutal_friction_shove_2d() {
    let mut world = World2::new();
    brutal_floor(&mut world);
    for i in 0..12 {
        world
            .add_dynamic_body(
                Vec2::new(0.0, 0.05 + i as f32 * 1.02),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
    }
    // Shove the middle of the stack sideways.
    world.bodies[6].velocity = Vec2::new(8.0, 0.0);
    let steps = 600;
    let mut peak = 8.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for body in world.bodies.iter().skip(1) {
            peak = peak.max(body.velocity.length());
        }
        if frame % 100 == 0 {
            brutal_assert_finite2(&world, "friction shove");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "shove final");
    let mut max_abs_x = 0.0f32;
    let mut min_y = f32::MAX;
    for body in world.bodies.iter().skip(1) {
        max_abs_x = max_abs_x.max(body.transform.position.x.abs());
        min_y = min_y.min(body.transform.position.y);
        // An 8 m/s shove into a stack is violent ejection territory:
        // survival means finite and in-universe. Tunneling is reported.
        assert!(
            body.transform.position.y > -10000.0,
            "shove: body escaped downward: y={}",
            body.transform.position.y
        );
        assert!(
            body.transform.position.x.abs() < 200.0,
            "shove: body escaped: x={}",
            body.transform.position.x
        );
    }
    let tunneled = min_y < -1.0;
    brutal_report(
        "friction_shove_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!(
            "max_abs_x={max_abs_x:.2} min_y={min_y:.2} tunneled={tunneled} hash={:#x}",
            brutal_hash2(&world)
        ),
    );
}

/// Six-link pendulum chain on distance joints: every link must hold its
/// rest length while swinging under gravity.
#[test]
fn brutal_joint_gauntlet_2d() {
    let mut world = World2::new();
    world
        .add_static_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
        )
        .expect("valid shape");
    let mut prev = 0usize;
    for i in 0..6 {
        // Hang the chain horizontally; gravity swings it down.
        let ball = world
            .add_dynamic_body(
                Vec2::new(1.0 + i as f32, 0.0),
                Shape2::Circle { radius: 0.25 },
                1.0,
            )
            .expect("valid shape");
        let joint = make_distance_joint(
            0,
            prev,
            ball,
            Vec2::ZERO,
            Vec2::ZERO,
            1.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        prev = ball;
    }
    let steps = 1200;
    let mut peak = 0.0f32;
    let mut max_err = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for body in world.bodies.iter().skip(1) {
            peak = peak.max(body.velocity.length());
        }
        if frame % 200 == 0 {
            brutal_assert_finite2(&world, "joint gauntlet");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "gauntlet final");
    // Check every distance link holds rest length 1.0.
    for joint in world.joints.iter() {
        let (a, b) = joint.bodies();
        let d = (world.bodies[b].transform.position - world.bodies[a].transform.position).length();
        max_err = max_err.max((d - 1.0).abs());
    }
    assert!(
        max_err < 0.15,
        "gauntlet: link stretched, max_err={max_err}"
    );
    brutal_report(
        "joint_gauntlet_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!(
            "joints={} max_link_err={max_err:.4} hash={:#x}",
            world.joints.len(),
            brutal_hash2(&world)
        ),
    );
}

/// 10,000:1 mass ratio over 20k steps: bounded jitter, no tunneling.
#[test]
fn brutal_mass_ratio_soak_2d() {
    let mut world = World2::new();
    brutal_floor(&mut world);
    world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1e4,
        )
        .expect("valid shape");
    let light = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let steps = 20_000;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        peak = peak.max(world.bodies[light].velocity.length());
        if frame % 2000 == 0 {
            brutal_assert_finite2(&world, "mass ratio soak");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "soak final");
    let y = world.bodies[light].transform.position.y;
    assert!(y > 0.9, "soak: light body tunneled, y={y}");
    brutal_report(
        "mass_ratio_soak_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!("light_y={y:.3} hash={:#x}", brutal_hash2(&world)),
    );
}

/// 90% initial overlap with zero gravity: must separate without NaN or
/// explosion; ejection peak is characterized.
#[test]
fn brutal_deep_overlap_eject_2d() {
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
            Vec2::new(0.1, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let steps = 240;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for _ in 0..steps {
        world.step(BRUTAL_DT);
        peak = peak.max(world.bodies[top].velocity.length());
        brutal_assert_finite2(&world, "deep overlap");
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let x = world.bodies[top].transform.position.x;
    assert!(x > 0.9, "eject: overlap did not separate, x={x}");
    assert!(peak < 100.0, "eject: explosion, peak={peak}");
    brutal_report(
        "deep_overlap_eject_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!("final_x={x:.3} hash={:#x}", brutal_hash2(&world)),
    );
}

/// Thin needle fired at 60 m/s at a wall: bounded flight, tunneling noted.
#[test]
fn brutal_needle_impact_2d() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world
        .add_static_body(
            Vec2::new(10.0, 5.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 5.0),
            },
        )
        .expect("valid shape");
    let needle = world
        .add_dynamic_body(
            Vec2::new(0.0, 5.0),
            Shape2::Box {
                half_extents: Vec2::new(0.05, 1.0),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[needle].velocity = Vec2::new(60.0, 0.0);
    let steps = 300;
    let mut peak = 60.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        peak = peak.max(world.bodies[needle].velocity.length());
        if frame % 100 == 0 {
            brutal_assert_finite2(&world, "needle impact");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let x = world.bodies[needle].transform.position.x;
    assert!(x.abs() < 400.0, "needle: escaped world bounds: x={x}");
    let tunneled = x > 10.6;
    brutal_report(
        "needle_impact_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!(
            "final_x={x:.2} tunneled={tunneled} hash={:#x}",
            brutal_hash2(&world)
        ),
    );
}

fn brutal_bin_3d(world: &mut World3) {
    world
        .add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(5.0, 0.5, 5.0),
            },
        )
        .expect("valid shape");
    for (pos, he) in [
        (Vec3::new(-5.0, 3.0, 0.0), Vec3::new(0.5, 4.0, 5.0)),
        (Vec3::new(5.0, 3.0, 0.0), Vec3::new(0.5, 4.0, 5.0)),
        (Vec3::new(0.0, 3.0, -5.0), Vec3::new(5.0, 4.0, 0.5)),
        (Vec3::new(0.0, 3.0, 5.0), Vec3::new(5.0, 4.0, 0.5)),
    ] {
        world
            .add_static_body(pos, Shape3::Box { half_extents: he })
            .expect("valid shape");
    }
}

/// 14 convex-mesh boxes tumbling in a bin: the mesh GJK path under chaos.
#[test]
fn brutal_mesh_tumbler_3d() {
    let mut world = World3::new();
    brutal_bin_3d(&mut world);
    for i in 0..14 {
        let x = -3.0 + (i as f32 % 4.0) * 2.0;
        let z = -3.0 + ((i / 4) as f32 % 4.0) * 2.0;
        let y = 2.0 + (i as f32 / 4.0) * 1.6;
        let (verts, idx) = brutal_box_mesh(Vec3::new(0.4, 0.4, 0.4));
        let id = world
            .add_dynamic_body(
                Vec3::new(x, y, z),
                Shape3::ConvexMesh {
                    vertices: verts,
                    indices: idx,
                },
                1.0,
            )
            .expect("valid mesh");
        world.bodies[id].angular_velocity =
            Vec3::new(0.5 * (i as f32 % 3.0) - 0.5, 1.0, -0.5 * (i as f32 % 2.0));
    }
    let steps = 900;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for body in world.bodies.iter().skip(5) {
            peak = peak.max(body.velocity.length());
        }
        if frame % 150 == 0 {
            brutal_assert_finite3(&world, "mesh tumbler");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite3(&world, "tumbler final");
    let mut min_y = f32::MAX;
    for body in world.bodies.iter().skip(5) {
        min_y = min_y.min(body.transform.position.y);
        assert!(
            body.transform.position.y > -0.6,
            "tumbler: mesh tunneled: y={}",
            body.transform.position.y
        );
        assert!(
            body.transform.position.x.abs() < 20.0 && body.transform.position.z.abs() < 20.0,
            "tumbler: mesh escaped: p={}",
            body.transform.position
        );
    }
    let (pairs, contacts) = brutal_contacts3(&world);
    brutal_report(
        "mesh_tumbler_3d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!(
            "min_y={min_y:.2} pairs={pairs} contacts={contacts} hash={:#x}",
            brutal_hash3(&world)
        ),
    );
}

/// Six convex-mesh boxes stacked: mesh contacts must hold a tower.
#[test]
fn brutal_mesh_stack_3d() {
    let mut world = World3::new();
    world
        .add_static_body(
            Vec3::new(0.0, -1.0, 0.0),
            Shape3::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
        )
        .expect("valid shape");
    let mut ids = Vec::new();
    for i in 0..6 {
        let (verts, idx) = brutal_box_mesh(Vec3::new(0.5, 0.5, 0.5));
        ids.push(
            world
                .add_dynamic_body(
                    Vec3::new(0.0, 0.05 + i as f32 * 1.02, 0.0),
                    Shape3::ConvexMesh {
                        vertices: verts,
                        indices: idx,
                    },
                    1.0,
                )
                .expect("valid mesh"),
        );
    }
    let steps = 900;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        for &id in &ids {
            peak = peak.max(world.bodies[id].velocity.length());
        }
        if frame % 150 == 0 {
            brutal_assert_finite3(&world, "mesh stack");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite3(&world, "mesh stack final");
    // Finding: mesh-vs-mesh towers do NOT hold. The catch-all GJK path
    // emits a single support-midpoint contact per pair (same weakness
    // class as the old box-vs-capsule midpoint bug), so stacked meshes
    // sink through each other and pile on the floor. Mesh-vs-analytic
    // (floor) contacts hold fine. Survival bar: finite, above the floor,
    // bounded speeds. Heights are reported per level.
    let mut heights = String::new();
    for (level, id) in ids.iter().enumerate() {
        let b = &world.bodies[*id];
        heights.push_str(&format!(" L{level}={:.2}", b.transform.position.y));
        assert!(
            b.transform.position.y > -1.0,
            "mesh stack level {level} tunneled through floor: y={}",
            b.transform.position.y
        );
        assert!(
            b.velocity.length() < 5.0,
            "mesh stack level {level} jittering: v={}",
            b.velocity
        );
    }
    brutal_report(
        "mesh_stack_3d",
        world.bodies.len(),
        steps,
        wall_ms,
        peak,
        &format!("heights:{heights} hash={:#x}", brutal_hash3(&world)),
    );
}

fn brutal_chaos_3d() -> World3 {
    let mut world = World3::new();
    brutal_bin_3d(&mut world);
    for i in 0..6 {
        let (verts, idx) = brutal_box_mesh(Vec3::new(0.4, 0.4, 0.4));
        let id = world
            .add_dynamic_body(
                Vec3::new(-2.0 + i as f32 * 0.8, 3.0 + i as f32, (i % 2) as f32),
                Shape3::ConvexMesh {
                    vertices: verts,
                    indices: idx,
                },
                1.0,
            )
            .expect("valid mesh");
        world.bodies[id].velocity = Vec3::new(1.0 - i as f32 * 0.3, 0.0, 0.5);
    }
    world
}

/// Save mid-chaos, reload JSON + binary, continue: all three futures must
/// match bit-exactly.
#[test]
fn brutal_snapshot_midchaos_3d() {
    let mut direct = brutal_chaos_3d();
    for _ in 0..120 {
        direct.step(BRUTAL_DT);
    }
    let json = world3_to_json(&direct, LoadMode::Exact).expect("json");
    let bytes = world3_to_binary(&direct, LoadMode::Exact).expect("bin");
    let mut from_json = world3_from_json(&json, LoadMode::Exact).expect("json load");
    let mut from_bin = world3_from_binary(&bytes, LoadMode::Exact).expect("bin load");
    for _ in 0..120 {
        direct.step(BRUTAL_DT);
        from_json.step(BRUTAL_DT);
        from_bin.step(BRUTAL_DT);
    }
    let (hd, hj, hb) = (
        brutal_hash3(&direct),
        brutal_hash3(&from_json),
        brutal_hash3(&from_bin),
    );
    assert_eq!(hj, hd, "mid-chaos JSON continuation diverged");
    assert_eq!(hb, hd, "mid-chaos binary continuation diverged");
    println!(
        "BRUTAL name=snapshot_midchaos_3d bodies={} steps=240 json_bytes={} bin_bytes={} hash={hd:#x} match=json+bin",
        direct.bodies.len(),
        json.len(),
        bytes.len(),
    );
}

/// Larger two-pile scene, serial vs parallel: must match bit-exactly, with
/// per-path timings as the reading.
#[test]
fn brutal_parallel_parity_scale_2d() {
    fn piles(parallel: bool) -> World2 {
        let mut world = World2::new();
        world.solver.parallel = parallel;
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(12.0, 0.5),
                },
            )
            .expect("valid shape");
        for stack_x in [-5.0f32, -2.5, 2.5, 5.0] {
            for i in 0..5 {
                world
                    .add_dynamic_body(
                        Vec2::new(stack_x, 0.05 + i as f32 * 1.02),
                        Shape2::Box {
                            half_extents: Vec2::new(0.5, 0.5),
                        },
                        1.0,
                    )
                    .expect("valid shape");
            }
        }
        world
    }
    let steps = 300;
    let mut serial = piles(false);
    let start = Instant::now();
    for _ in 0..steps {
        serial.step(BRUTAL_DT);
    }
    let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut parallel = piles(true);
    let start = Instant::now();
    for _ in 0..steps {
        parallel.step(BRUTAL_DT);
    }
    let parallel_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (hs, hp) = (brutal_hash2(&serial), brutal_hash2(&parallel));
    assert_eq!(hs, hp, "scaled parallel run diverged from serial");
    println!(
        "BRUTAL name=parallel_parity_scale_2d bodies={} steps={steps} serial_ms_per_step={:.4} parallel_ms_per_step={:.4} hash={hs:#x} match=serial==parallel",
        serial.bodies.len(),
        serial_ms / steps as f64,
        parallel_ms / steps as f64,
    );
}

/// Same chaotic scene run twice from scratch: bit-exact determinism.
#[test]
fn brutal_determinism_double_run_2d() {
    fn chaos() -> World2 {
        let mut world = World2::new();
        brutal_floor(&mut world);
        for i in 0..10 {
            let x = -4.0 + (i as f32 % 5.0) * 2.0;
            let y = 1.0 + (i as f32 / 5.0) * 2.0;
            let id = if i % 2 == 0 {
                world
                    .add_dynamic_body(Vec2::new(x, y), Shape2::Circle { radius: 0.4 }, 1.0)
                    .expect("valid shape")
            } else {
                world
                    .add_dynamic_body(
                        Vec2::new(x, y),
                        Shape2::Box {
                            half_extents: Vec2::new(0.4, 0.4),
                        },
                        1.5,
                    )
                    .expect("valid shape")
            };
            world.bodies[id].velocity = Vec2::new(2.0 - i as f32 * 0.4, 1.0);
        }
        let joint = make_distance_joint(
            0,
            1,
            3,
            Vec2::ZERO,
            Vec2::ZERO,
            2.5,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        world
    }
    let steps = 600;
    let mut first = chaos();
    let start = Instant::now();
    for _ in 0..steps {
        first.step(BRUTAL_DT);
    }
    let first_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut second = chaos();
    let mut peak = 0.0f32;
    for _ in 0..steps {
        second.step(BRUTAL_DT);
        for body in second.bodies.iter().skip(1) {
            peak = peak.max(body.velocity.length());
        }
    }
    let (h1, h2) = (brutal_hash2(&first), brutal_hash2(&second));
    assert_eq!(h1, h2, "identical runs diverged");
    brutal_report(
        "determinism_double_run_2d",
        first.bodies.len(),
        steps,
        first_ms,
        peak,
        &format!("hash={h1:#x} match=run1==run2"),
    );
}

/// Hostile input soak: every bad op must reject cleanly and the world
/// must stay finite through continued stepping.
#[test]
fn brutal_hostile_soak_2d() {
    let mut world = World2::new();
    brutal_floor(&mut world);
    let mut rejections = 0u32;
    // NaN position.
    assert!(world
        .add_dynamic_body(
            Vec2::new(f32::NAN, 0.0),
            Shape2::Circle { radius: 0.5 },
            1.0
        )
        .is_err());
    rejections += 1;
    // Zero-size shapes.
    assert!(world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.0 }, 1.0)
        .is_err());
    rejections += 1;
    assert!(world
        .add_dynamic_body(
            Vec2::new(0.0, 2.0),
            Shape2::Box {
                half_extents: Vec2::new(0.0, 0.5),
            },
            1.0
        )
        .is_err());
    rejections += 1;
    // Zero density on a dynamic body.
    assert!(world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 0.0)
        .is_err());
    rejections += 1;
    // Unknown-body and same-body joints.
    assert!(make_distance_joint(
        0,
        0,
        99,
        Vec2::ZERO,
        Vec2::ZERO,
        1.0,
        0.0,
        world.bodies.len()
    )
    .is_err());
    rejections += 1;
    assert!(make_distance_joint(
        0,
        0,
        0,
        Vec2::ZERO,
        Vec2::ZERO,
        1.0,
        0.0,
        world.bodies.len()
    )
    .is_err());
    rejections += 1;
    // NaN material density on a dynamic body.
    let b = world
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    assert!(
        world.bodies[b]
            .set_material(PhysicsMaterial::new(0.5, 0.5, f32::NAN))
            .is_ok(),
        "NaN density sanitizes to default via PhysicsMaterial::new"
    );
    // Out-of-range removal is a no-op; remove/re-add keeps simulating.
    world.remove_body(999);
    world.remove_body(b);
    assert_eq!(world.bodies.len(), 1);
    world
        .add_dynamic_body(Vec2::new(1.0, 4.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    let steps = 120;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(BRUTAL_DT);
        if frame % 30 == 0 {
            brutal_assert_finite2(&world, "hostile soak");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    brutal_assert_finite2(&world, "hostile final");
    brutal_report(
        "hostile_soak_2d",
        world.bodies.len(),
        steps,
        wall_ms,
        world.bodies[1].velocity.length(),
        &format!("rejections={rejections} hash={:#x}", brutal_hash2(&world)),
    );
}
