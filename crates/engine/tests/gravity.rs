//! Gravitational physics tests (no parity IDs — these pin Rust-core
//! gravity behavior; the mirrored G block lands with the C#
//! implementation in a later phase, like brutal.rs stays exempt).
//!
//! Covered: uniform free-fall known answer, central circular-orbit
//! stability, N-body momentum conservation + determinism, softening
//! close-encounter safety, immune/tiny probes, Verlet-vs-Euler energy
//! behavior, and schema-v4 snapshot round-trips (2D + 3D).

use glam::{Vec2, Vec3};
use phieng_dynamics::body::{Shape2, Shape3};
use phieng_dynamics::{
    world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, world3_from_binary,
    world3_from_json, world3_to_binary, world3_to_json, GravityMode, Integrator, LoadMode, World2,
    World3,
};

const DT: f32 = 1.0 / 60.0;

fn no_damping_no_sleep_2d(w: &mut World2) {
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn no_damping_no_sleep_3d(w: &mut World3) {
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn assert_worlds_equal_2d(a: &World2, b: &World2) {
    assert_eq!(a.bodies.len(), b.bodies.len());
    for (x, y) in a.bodies.iter().zip(b.bodies.iter()) {
        assert_eq!(x.transform.position, y.transform.position);
        assert_eq!(x.velocity, y.velocity);
        assert_eq!(x.angular_velocity, y.angular_velocity);
        assert_eq!(x.transform.rotation, y.transform.rotation);
    }
}

fn assert_worlds_equal_3d(a: &World3, b: &World3) {
    assert_eq!(a.bodies.len(), b.bodies.len());
    for (x, y) in a.bodies.iter().zip(b.bodies.iter()) {
        assert_eq!(x.transform.position, y.transform.position);
        assert_eq!(x.velocity, y.velocity);
        assert_eq!(x.angular_velocity, y.angular_velocity);
        assert_eq!(x.transform.rotation, y.transform.rotation);
    }
}

fn total_momentum_2d(w: &World2) -> Vec2 {
    w.bodies
        .iter()
        .map(|b| b.velocity * b.mass)
        .fold(Vec2::ZERO, |a, v| a + v)
}

fn total_momentum_3d(w: &World3) -> Vec3 {
    w.bodies
        .iter()
        .map(|b| b.velocity * b.mass)
        .fold(Vec3::ZERO, |a, v| a + v)
}

fn specific_energy_2d(pos: Vec2, vel: Vec2, mu: f32) -> f32 {
    0.5 * vel.length_squared() - mu / pos.length().max(1e-6)
}

/// Uniform field, zero damping: 60 steps at g = -10 must give v = -10
/// and the exact semi-implicit Euler displacement, bit-deterministically.
#[test]
fn gravity_uniform_free_fall_known_answer() {
    fn scene() -> (World2, usize) {
        let mut w = World2::new();
        w.gravity = Vec2::new(0.0, -10.0);
        let id = w
            .add_dynamic_body(Vec2::new(0.0, 5.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        no_damping_no_sleep_2d(&mut w);
        (w, id)
    }
    let (mut a, id_a) = scene();
    let (mut b, _) = scene();
    for _ in 0..60 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    let ball = &a.bodies[id_a];
    assert!(
        (ball.velocity.y + 10.0).abs() < 1e-4,
        "v={}",
        ball.velocity.y
    );
    // Semi-implicit Euler: p = 5 - 10*dt*(1+...+60)/60 = 5 - 61/12.
    let expect = 5.0 - 61.0 / 12.0;
    assert!(
        (ball.transform.position.y - expect).abs() < 1e-3,
        "y={}",
        ball.transform.position.y
    );
}

/// Central attractor, circular orbit: mu = 100 at the origin, probe at
/// r = 10 with v = sqrt(mu/r). After ~1 period the radius must hold
/// (bounded, no crash into the center / escape) and two identical runs
/// must agree bit-exactly.
#[test]
fn gravity_central_circular_orbit_stable() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(Vec2::new(10.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].velocity = Vec2::new(0.0, v);
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..1200 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    let r = a.bodies[0].transform.position.length();
    assert!(
        (r - 10.0).abs() / 10.0 < 0.15,
        "orbit decayed/escaped: r={r}"
    );
    assert!(
        a.bodies[0].transform.position.x.is_finite()
            && a.bodies[0].transform.position.y.is_finite(),
        "non-finite orbit state"
    );
}

/// Full N-body: symmetric +-F application conserves total momentum to
/// float rounding, deterministically.
#[test]
fn gravity_nbody_momentum_conserved() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        let specs = [
            (Vec2::new(-20.0, 0.0), Vec2::new(0.5, 0.0), 0.5),
            (Vec2::new(15.0, 5.0), Vec2::new(0.0, 0.5), 0.7),
            (Vec2::new(0.0, -15.0), Vec2::new(-0.3, 0.2), 1.0),
        ];
        for (p, v, r) in specs {
            let id = w
                .add_dynamic_body(p, Shape2::Circle { radius: r }, 1.0)
                .expect("valid shape");
            w.bodies[id].velocity = v;
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let p0 = total_momentum_2d(&a);
    let mut b = scene();
    for _ in 0..600 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    let p1 = total_momentum_2d(&a);
    assert!(
        (p1 - p0).length() < 1e-2,
        "momentum drifted: {p0:?} -> {p1:?}"
    );
    for body in &a.bodies {
        assert!(body.transform.position.x.is_finite());
        assert!(body.velocity.x.is_finite());
    }
}

/// Coincident bodies with zero and nonzero softening: the skip guard
/// must hold — no NaN/Inf, finite state, deterministic.
#[test]
fn gravity_softening_coincident_no_nan() {
    for eps in [0.0f32, 0.5] {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = eps;
        for _ in 0..2 {
            w.add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 }, 1.0)
                .expect("valid shape");
        }
        no_damping_no_sleep_2d(&mut w);
        for _ in 0..120 {
            w.step(DT);
        }
        for body in &w.bodies {
            assert!(body.transform.position.x.is_finite());
            assert!(body.transform.position.y.is_finite());
            assert!(body.velocity.x.is_finite());
            assert!(body.velocity.y.is_finite());
        }
    }
}

/// Immune probe (scale 0) never moves; tiny-density probe follows the
/// field. Both stay finite.
#[test]
fn gravity_probe_immune_and_light() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.gravity_config.mode = GravityMode::NBody;
    w.gravity_config.big_g = 1.0;
    w.gravity_config.softening = 0.1;
    let planet = w
        .add_static_body(
            Vec2::ZERO,
            Shape2::Box {
                half_extents: Vec2::new(5.0, 5.0),
            },
        )
        .expect("valid shape");
    // Static 10x10 box at unit density: mass 100, a real attractor.
    assert!(w.bodies[planet].mass > 50.0);
    let immune = w
        .add_dynamic_body(Vec2::new(60.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    w.bodies[immune].gravity_scale = 0.0;
    let probe = w
        .add_dynamic_body(Vec2::new(-60.0, 0.0), Shape2::Circle { radius: 0.5 }, 1e-7)
        .expect("valid shape");
    for _ in 0..120 {
        w.step(DT);
    }
    // Immune body bit-identical to its start.
    assert_eq!(w.bodies[immune].transform.position, Vec2::new(60.0, 0.0));
    assert_eq!(w.bodies[immune].velocity, Vec2::ZERO);
    // Light probe fell toward the planet.
    assert!(
        w.bodies[probe].transform.position.x > -60.0,
        "probe did not fall: {:?}",
        w.bodies[probe].transform.position
    );
    assert!(w.bodies[probe].velocity.x.is_finite());
}

/// Velocity-Verlet must beat semi-implicit Euler on orbital energy
/// drift (order-of-accuracy separation), with both deterministic.
#[test]
fn gravity_verlet_beats_euler_on_orbit() {
    fn scene(verlet: bool) -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        if verlet {
            w.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(Vec2::new(10.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].velocity = Vec2::new(0.0, v);
        no_damping_no_sleep_2d(&mut w);
        w
    }
    fn max_drift(verlet: bool) -> f32 {
        let mut w = scene(verlet);
        let b = &w.bodies[0];
        let e0 = specific_energy_2d(b.transform.position, b.velocity, 100.0);
        let mut worst = 0.0f32;
        for step in 0..1200 {
            w.step(DT);
            if step % 60 == 0 {
                let b = &w.bodies[0];
                let e = specific_energy_2d(b.transform.position, b.velocity, 100.0);
                worst = worst.max((e - e0).abs() / e0.abs());
            }
        }
        // Determinism: identical twin agrees bit-exactly.
        let mut twin = scene(verlet);
        for _ in 0..1200 {
            twin.step(DT);
        }
        assert_worlds_equal_2d(&w, &twin);
        worst
    }
    let euler = max_drift(false);
    let verlet = max_drift(true);
    assert!(
        verlet < euler * 0.5,
        "verlet drift {verlet} not clearly better than euler {euler}"
    );
}

/// Schema v4 round-trip: N-body + Verlet + non-unit scales save to
/// `"schema": 4`, reload into identical config, and continue
/// bit-identical through JSON and binary Exact loads. Default worlds
/// stay `"schema": 3` with no gravity keys (byte-compat).
#[test]
fn gravity_snapshot_v4_roundtrip() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.integrator = Integrator::VelocityVerlet;
        w.gravity_config.big_g = 2.0;
        w.gravity_config.softening = 0.1;
        let specs = [
            (Vec2::new(-8.0, 0.0), 1.0),
            (Vec2::new(8.0, 0.0), 0.5),
            (Vec2::new(0.0, 8.0), 2.0),
        ];
        for (p, s) in specs {
            let id = w
                .add_dynamic_body(p, Shape2::Circle { radius: 0.5 }, 1.0)
                .expect("valid shape");
            w.bodies[id].gravity_scale = s;
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut direct = scene();
    for _ in 0..60 {
        direct.step(DT);
    }
    let json = world2_to_json(&direct, LoadMode::Exact).expect("serializes");
    assert!(
        json.contains("\"schema\": 4"),
        "gravity world must write schema 4"
    );
    let bytes = world2_to_binary(&direct, LoadMode::Exact).expect("serializes");
    let mut from_json = world2_from_json(&json, LoadMode::Exact).expect("json loads");
    let mut from_bin = world2_from_binary(&bytes, LoadMode::Exact).expect("bin loads");
    assert_eq!(from_json.gravity_config.mode, GravityMode::NBody);
    assert_eq!(
        from_json.gravity_config.integrator,
        Integrator::VelocityVerlet
    );
    assert_eq!(from_json.gravity_config.big_g, 2.0);
    assert_eq!(from_bin.gravity_config.softening, 0.1);
    assert_eq!(from_json.bodies[1].gravity_scale, 0.5);
    assert_eq!(from_bin.bodies[2].gravity_scale, 2.0);
    for _ in 0..60 {
        direct.step(DT);
        from_json.step(DT);
        from_bin.step(DT);
    }
    assert_worlds_equal_2d(&direct, &from_json);
    assert_worlds_equal_2d(&direct, &from_bin);

    // Default worlds stay schema 3 with no gravity keys.
    let mut plain = World2::new();
    plain
        .add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    let plain_json = world2_to_json(&plain, LoadMode::Exact).expect("serializes");
    assert!(plain_json.contains("\"schema\": 3"), "default stays v3");
    assert!(!plain_json.contains("gravity_config"), "v3 omits gravity");
    assert!(!plain_json.contains("gravity_scale"), "v3 omits scales");
    let plain_back = world2_from_json(&plain_json, LoadMode::Exact).expect("v3 loads");
    assert!(plain_back.gravity_config.is_default());
    assert_eq!(plain_back.bodies[0].gravity_scale, 1.0);
}

/// Hand-corrupted gravity fields are rejected, never defaulted:
/// unknown mode tag and negative softening must fail load.
#[test]
fn gravity_snapshot_rejects_bad_config() {
    let mut w = World2::new();
    w.gravity_config.mode = GravityMode::NBody;
    let json = world2_to_json(&w, LoadMode::Exact).expect("serializes");
    let bad_mode = json.replacen("\"mode\": 2", "\"mode\": 99", 1);
    assert!(world2_from_json(&bad_mode, LoadMode::Exact).is_err());
    let bad_soft = json.replacen("\"softening\": 0.0", "\"softening\": -1.0", 1);
    assert!(world2_from_json(&bad_soft, LoadMode::Exact).is_err());
}

// ---------------------------------------------------------------------------
// 3D mirrors.
// ---------------------------------------------------------------------------

/// 3D central orbit in the xy-plane: radius holds, bit-deterministic.
#[test]
fn gravity3_central_orbit_stable() {
    fn scene() -> World3 {
        let mut w = World3::new();
        w.gravity = Vec3::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(
                Vec3::new(10.0, 0.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        w.bodies[id].velocity = Vec3::new(0.0, v, 0.0);
        no_damping_no_sleep_3d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..1200 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_3d(&a, &b);
    let p = a.bodies[0].transform.position;
    let r = Vec2::new(p.x, p.y).length();
    assert!((r - 10.0).abs() / 10.0 < 0.15, "orbit decayed: r={r}");
    assert!(p.z.abs() < 0.5, "orbit left its plane: z={}", p.z);
    assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
}

/// 3D N-body: momentum conserved, deterministic.
#[test]
fn gravity3_nbody_momentum_conserved() {
    fn scene() -> World3 {
        let mut w = World3::new();
        w.gravity = Vec3::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        let specs = [
            (Vec3::new(-20.0, 0.0, 1.0), Vec3::new(0.5, 0.0, 0.0), 0.5),
            (Vec3::new(15.0, 5.0, -1.0), Vec3::new(0.0, 0.5, 0.1), 0.8),
        ];
        for (p, v, r) in specs {
            let id = w
                .add_dynamic_body(p, Shape3::Sphere { radius: r }, 1.0)
                .expect("valid shape");
            w.bodies[id].velocity = v;
        }
        no_damping_no_sleep_3d(&mut w);
        w
    }
    let mut a = scene();
    let p0 = total_momentum_3d(&a);
    let mut b = scene();
    for _ in 0..300 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_3d(&a, &b);
    let p1 = total_momentum_3d(&a);
    assert!(
        (p1 - p0).length() < 1e-2,
        "momentum drifted: {p0:?} -> {p1:?}"
    );
}

/// 3D schema v4 round-trip with config + continuation equality.
#[test]
fn gravity3_snapshot_v4_roundtrip() {
    let mut direct = World3::new();
    direct.gravity = Vec3::ZERO;
    direct.gravity_config.mode = GravityMode::NBody;
    direct.gravity_config.integrator = Integrator::VelocityVerlet;
    direct.gravity_config.big_g = 1.5;
    direct.gravity_config.softening = 0.2;
    for (p, s) in [
        (Vec3::new(-6.0, 0.0, 0.0), 1.0),
        (Vec3::new(6.0, 0.0, 0.0), 0.25),
    ] {
        let id = direct
            .add_dynamic_body(p, Shape3::Sphere { radius: 0.5 }, 1.0)
            .expect("valid shape");
        direct.bodies[id].gravity_scale = s;
    }
    no_damping_no_sleep_3d(&mut direct);
    for _ in 0..60 {
        direct.step(DT);
    }
    let json = world3_to_json(&direct, LoadMode::Exact).expect("serializes");
    assert!(json.contains("\"schema\": 4"));
    let bytes = world3_to_binary(&direct, LoadMode::Exact).expect("serializes");
    let mut from_json = world3_from_json(&json, LoadMode::Exact).expect("json loads");
    let mut from_bin = world3_from_binary(&bytes, LoadMode::Exact).expect("bin loads");
    assert_eq!(from_bin.gravity_config.mode, GravityMode::NBody);
    assert_eq!(
        from_json.gravity_config.integrator,
        Integrator::VelocityVerlet
    );
    assert_eq!(from_json.bodies[1].gravity_scale, 0.25);
    for _ in 0..60 {
        direct.step(DT);
        from_json.step(DT);
        from_bin.step(DT);
    }
    assert_worlds_equal_3d(&direct, &from_json);
    assert_worlds_equal_3d(&direct, &from_bin);
}

// ---------------------------------------------------------------------------
// Barnes-Hut mode (approximate by design: same-mode runs are
// bit-identical, but exact and BH trajectories never match).
// ---------------------------------------------------------------------------

/// BH determinism: 50 bodies, fresh trees every kick, two identical
/// runs agree bit-exactly.
#[test]
fn gravity_bh_deterministic() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::BarnesHut;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        w.gravity_config.theta = 0.5;
        for i in 0..50 {
            let id = w
                .add_dynamic_body(
                    Vec2::new(
                        (i as f32 * 1.7) % 40.0 - 20.0,
                        (i as f32 * 2.3) % 40.0 - 20.0,
                    ),
                    Shape2::Circle { radius: 0.5 },
                    1.0,
                )
                .expect("valid shape");
            w.bodies[id].velocity = Vec2::new(0.1 * i as f32, -0.05 * i as f32);
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..100 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    for body in &a.bodies {
        assert!(body.transform.position.x.is_finite());
        assert!(body.velocity.x.is_finite());
    }
}

/// BH orbit sanity: static planet + probes stay bound (loose radius
/// bound — the tree approximates), deterministically.
#[test]
fn gravity_bh_orbit_bounded() {
    fn scene() -> (World2, Vec<usize>) {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::BarnesHut;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        w.gravity_config.theta = 0.5;
        // Static 10x10 planet, mass 100.
        w.add_static_body(
            Vec2::ZERO,
            Shape2::Box {
                half_extents: Vec2::new(5.0, 5.0),
            },
        )
        .expect("valid shape");
        let mut probes = Vec::new();
        // Circular-orbit velocity for r = 30, mu = 100.
        let v = (100.0f32 / 30.0).sqrt();
        for angle in [0.0f32, 2.1, 4.2] {
            let p = Vec2::new(angle.cos(), angle.sin()) * 30.0;
            let id = w
                .add_dynamic_body(p, Shape2::Circle { radius: 0.5 }, 1.0)
                .expect("valid shape");
            let tangent = Vec2::new(-angle.sin(), angle.cos());
            w.bodies[id].velocity = tangent * v;
            probes.push(id);
        }
        no_damping_no_sleep_2d(&mut w);
        (w, probes)
    }
    let (mut a, probes_a) = scene();
    let (mut b, _) = scene();
    for _ in 0..600 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    for &id in &probes_a {
        let r = a.bodies[id].transform.position.length();
        assert!(
            (r - 30.0).abs() / 30.0 < 0.5,
            "probe escaped/crashed: r={r}"
        );
    }
}

/// BH snapshot round-trip: theta + scales persist, continuation is
/// bit-identical through JSON and binary Exact loads.
#[test]
fn gravity_bh_snapshot_roundtrip() {
    let mut direct = World2::new();
    direct.gravity = Vec2::ZERO;
    direct.gravity_config.mode = GravityMode::BarnesHut;
    direct.gravity_config.big_g = 2.0;
    direct.gravity_config.softening = 0.1;
    direct.gravity_config.theta = 0.7;
    for (p, s) in [
        (Vec2::new(-8.0, 0.0), 1.0),
        (Vec2::new(8.0, 0.0), 0.5),
        (Vec2::new(0.0, 8.0), 2.0),
    ] {
        let id = direct
            .add_dynamic_body(p, Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        direct.bodies[id].gravity_scale = s;
    }
    no_damping_no_sleep_2d(&mut direct);
    for _ in 0..60 {
        direct.step(DT);
    }
    let json = world2_to_json(&direct, LoadMode::Exact).expect("serializes");
    assert!(json.contains("\"schema\": 4"));
    assert!(json.contains("\"theta\": 0.7"));
    let bytes = world2_to_binary(&direct, LoadMode::Exact).expect("serializes");
    let mut from_json = world2_from_json(&json, LoadMode::Exact).expect("json loads");
    let mut from_bin = world2_from_binary(&bytes, LoadMode::Exact).expect("bin loads");
    assert_eq!(from_json.gravity_config.mode, GravityMode::BarnesHut);
    assert_eq!(from_bin.gravity_config.theta, 0.7);
    assert_eq!(from_json.bodies[2].gravity_scale, 2.0);
    for _ in 0..60 {
        direct.step(DT);
        from_json.step(DT);
        from_bin.step(DT);
    }
    assert_worlds_equal_2d(&direct, &from_json);
    assert_worlds_equal_2d(&direct, &from_bin);
}

/// Empty BH world steps without panic.
#[test]
fn gravity_bh_empty_world() {
    let mut w = World2::new();
    w.gravity_config.mode = GravityMode::BarnesHut;
    for _ in 0..10 {
        w.step(DT);
    }
    assert_eq!(w.bodies.len(), 0);
}

/// 3D BH determinism.
#[test]
fn gravity3_bh_deterministic() {
    fn scene() -> World3 {
        let mut w = World3::new();
        w.gravity = Vec3::ZERO;
        w.gravity_config.mode = GravityMode::BarnesHut;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        w.gravity_config.theta = 0.5;
        for i in 0..30 {
            let id = w
                .add_dynamic_body(
                    Vec3::new(
                        (i as f32 * 1.7) % 20.0 - 10.0,
                        (i as f32 * 2.3) % 20.0 - 10.0,
                        (i as f32 * 3.1) % 20.0 - 10.0,
                    ),
                    Shape3::Sphere { radius: 0.5 },
                    1.0,
                )
                .expect("valid shape");
            w.bodies[id].velocity = Vec3::new(0.1 * i as f32, 0.0, -0.05 * i as f32);
        }
        no_damping_no_sleep_3d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..60 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_3d(&a, &b);
}

// ---------------------------------------------------------------------------
// G parity block (mirrored by G01-G10 in tests/PhiEng.Tests/GravityTests.cs;
// enforced by tools/check_test_parity.py). Cross-language bit-equality is
// NOT claimed — each side asserts run-twice determinism plus
// tolerance-bounded physics behavior on mirrored scenes.
// ---------------------------------------------------------------------------

/// G01: central circular orbit holds its radius, deterministically.
#[test]
fn g01_central_orbit_stable() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(Vec2::new(10.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].velocity = Vec2::new(0.0, v);
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..600 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    let r = a.bodies[0].transform.position.length();
    assert!((r - 10.0).abs() / 10.0 < 0.10, "orbit decayed: r={r}");
}

/// G02: N-body momentum conservation + determinism.
#[test]
fn g02_nbody_deterministic() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        for (p, r) in [
            (Vec2::new(-20.0, 0.0), 0.5),
            (Vec2::new(15.0, 5.0), 0.7),
            (Vec2::new(0.0, -15.0), 1.0),
        ] {
            w.add_dynamic_body(p, Shape2::Circle { radius: r }, 1.0)
                .expect("valid shape");
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let p0 = total_momentum_2d(&a);
    let mut b = scene();
    for _ in 0..300 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
    let p1 = total_momentum_2d(&a);
    assert!((p1 - p0).length() < 1e-2, "momentum drifted");
}

/// G03: Verlet energy drift is clearly smaller than Euler's.
#[test]
fn g03_verlet_energy_order() {
    fn max_drift(verlet: bool) -> f32 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        if verlet {
            w.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(Vec2::new(10.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        w.bodies[id].velocity = Vec2::new(0.0, v);
        no_damping_no_sleep_2d(&mut w);
        let b = &w.bodies[id];
        let e0 = specific_energy_2d(b.transform.position, b.velocity, 100.0);
        let mut worst = 0.0f32;
        for step in 0..600 {
            w.step(DT);
            if step % 60 == 0 {
                let b = &w.bodies[id];
                let e = specific_energy_2d(b.transform.position, b.velocity, 100.0);
                worst = worst.max((e - e0).abs() / e0.abs());
            }
        }
        worst
    }
    assert!(max_drift(true) < max_drift(false) * 0.5);
}

/// G04: coincident bodies stay finite (softening skip guard).
#[test]
fn g04_softening_coincident_safe() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.gravity_config.mode = GravityMode::NBody;
    w.gravity_config.big_g = 1.0;
    w.gravity_config.softening = 0.0;
    for _ in 0..2 {
        w.add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
    }
    no_damping_no_sleep_2d(&mut w);
    for _ in 0..60 {
        w.step(DT);
    }
    for body in &w.bodies {
        assert!(body.transform.position.x.is_finite());
        assert!(body.velocity.x.is_finite());
    }
}

/// G05: immune probe never moves; tiny-density probe follows the field.
#[test]
fn g05_probe_exempt() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    w.gravity_config.mode = GravityMode::NBody;
    w.gravity_config.big_g = 1.0;
    w.gravity_config.softening = 0.1;
    w.add_static_body(
        Vec2::ZERO,
        Shape2::Box {
            half_extents: Vec2::new(5.0, 5.0),
        },
    )
    .expect("valid shape");
    let immune = w
        .add_dynamic_body(Vec2::new(60.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    w.bodies[immune].gravity_scale = 0.0;
    let probe = w
        .add_dynamic_body(Vec2::new(-60.0, 0.0), Shape2::Circle { radius: 0.5 }, 1e-7)
        .expect("valid shape");
    for _ in 0..120 {
        w.step(DT);
    }
    assert_eq!(w.bodies[immune].transform.position, Vec2::new(60.0, 0.0));
    assert!(w.bodies[probe].transform.position.x > -60.0);
}

/// G06: N-body + Verlet + scales snapshot round-trip with continuation.
#[test]
fn g06_snapshot_roundtrip() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.integrator = Integrator::VelocityVerlet;
        w.gravity_config.big_g = 2.0;
        w.gravity_config.softening = 0.1;
        for (p, s) in [(Vec2::new(-8.0, 0.0), 1.0), (Vec2::new(8.0, 0.0), 0.5)] {
            let id = w
                .add_dynamic_body(p, Shape2::Circle { radius: 0.5 }, 1.0)
                .expect("valid shape");
            w.bodies[id].gravity_scale = s;
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut direct = scene();
    for _ in 0..30 {
        direct.step(DT);
    }
    let json = world2_to_json(&direct, LoadMode::Exact).expect("serializes");
    let bytes = world2_to_binary(&direct, LoadMode::Exact).expect("serializes");
    let mut from_json = world2_from_json(&json, LoadMode::Exact).expect("json loads");
    let mut from_bin = world2_from_binary(&bytes, LoadMode::Exact).expect("bin loads");
    assert_eq!(from_json.gravity_config.mode, GravityMode::NBody);
    assert_eq!(from_bin.bodies[1].gravity_scale, 0.5);
    for _ in 0..30 {
        direct.step(DT);
        from_json.step(DT);
        from_bin.step(DT);
    }
    assert_worlds_equal_2d(&direct, &from_json);
    assert_worlds_equal_2d(&direct, &from_bin);
}

/// G07: Barnes-Hut determinism on a 20-body scene.
#[test]
fn g07_bh_deterministic() {
    fn scene() -> World2 {
        let mut w = World2::new();
        w.gravity = Vec2::ZERO;
        w.gravity_config.mode = GravityMode::BarnesHut;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        w.gravity_config.theta = 0.5;
        for i in 0..20 {
            w.add_dynamic_body(
                Vec2::new(
                    (i as f32 * 1.7) % 20.0 - 10.0,
                    (i as f32 * 2.3) % 20.0 - 10.0,
                ),
                Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        }
        no_damping_no_sleep_2d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..60 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_2d(&a, &b);
}

/// G08: 3D central orbit holds its radius, deterministically.
#[test]
fn g08_central_orbit_stable_3d() {
    fn scene() -> World3 {
        let mut w = World3::new();
        w.gravity = Vec3::ZERO;
        w.gravity_config.mode = GravityMode::Central;
        w.gravity_config.attractor_mu = 100.0;
        let v = (100.0f32 / 10.0).sqrt();
        let id = w
            .add_dynamic_body(
                Vec3::new(10.0, 0.0, 0.0),
                Shape3::Sphere { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        w.bodies[id].velocity = Vec3::new(0.0, v, 0.0);
        no_damping_no_sleep_3d(&mut w);
        w
    }
    let mut a = scene();
    let mut b = scene();
    for _ in 0..600 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_3d(&a, &b);
    let p = a.bodies[0].transform.position;
    let r = Vec2::new(p.x, p.y).length();
    assert!((r - 10.0).abs() / 10.0 < 0.10, "orbit decayed: r={r}");
}

/// G09: 3D N-body momentum conservation + determinism.
#[test]
fn g09_nbody_deterministic_3d() {
    fn scene() -> World3 {
        let mut w = World3::new();
        w.gravity = Vec3::ZERO;
        w.gravity_config.mode = GravityMode::NBody;
        w.gravity_config.big_g = 1.0;
        w.gravity_config.softening = 0.1;
        for (p, r) in [
            (Vec3::new(-20.0, 0.0, 1.0), 0.5),
            (Vec3::new(15.0, 5.0, -1.0), 0.8),
        ] {
            w.add_dynamic_body(p, Shape3::Sphere { radius: r }, 1.0)
                .expect("valid shape");
        }
        no_damping_no_sleep_3d(&mut w);
        w
    }
    let mut a = scene();
    let p0 = total_momentum_3d(&a);
    let mut b = scene();
    for _ in 0..300 {
        a.step(DT);
        b.step(DT);
    }
    assert_worlds_equal_3d(&a, &b);
    let p1 = total_momentum_3d(&a);
    assert!((p1 - p0).length() < 1e-2, "momentum drifted");
}

/// G10: 3D N-body + Verlet + scales snapshot round-trip.
#[test]
fn g10_snapshot_roundtrip_3d() {
    let mut direct = World3::new();
    direct.gravity = Vec3::ZERO;
    direct.gravity_config.mode = GravityMode::NBody;
    direct.gravity_config.integrator = Integrator::VelocityVerlet;
    direct.gravity_config.big_g = 1.5;
    direct.gravity_config.softening = 0.2;
    for (p, s) in [
        (Vec3::new(-6.0, 0.0, 0.0), 1.0),
        (Vec3::new(6.0, 0.0, 0.0), 0.25),
    ] {
        let id = direct
            .add_dynamic_body(p, Shape3::Sphere { radius: 0.5 }, 1.0)
            .expect("valid shape");
        direct.bodies[id].gravity_scale = s;
    }
    no_damping_no_sleep_3d(&mut direct);
    for _ in 0..30 {
        direct.step(DT);
    }
    let json = world3_to_json(&direct, LoadMode::Exact).expect("serializes");
    let bytes = world3_to_binary(&direct, LoadMode::Exact).expect("serializes");
    let mut from_json = world3_from_json(&json, LoadMode::Exact).expect("json loads");
    let mut from_bin = world3_from_binary(&bytes, LoadMode::Exact).expect("bin loads");
    assert_eq!(from_json.gravity_config.mode, GravityMode::NBody);
    assert_eq!(from_bin.bodies[1].gravity_scale, 0.25);
    for _ in 0..30 {
        direct.step(DT);
        from_json.step(DT);
        from_bin.step(DT);
    }
    assert_worlds_equal_3d(&direct, &from_json);
    assert_worlds_equal_3d(&direct, &from_bin);
}
