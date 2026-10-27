//! Electromagnetics integration tests (EM block).
//!
//! Every test here has a `EMxx_` prefix and a mirror in
//! `tests/PhiEng.Tests/ElectroTests.cs`; `tools/check_test_parity.py`
//! enforces the pairing.

use glam::{Vec2, Vec3};
use phieng::phieng_electromagnetics::{ElectroConfig2, ElectroConfig3, ElectroMode, ElectroState2};
use phieng::{Shape2, Shape3, World2, World3};

fn pairwise2() -> ElectroConfig2 {
    ElectroConfig2 {
        enabled: true,
        mode: ElectroMode::Pairwise,
        ..Default::default()
    }
}

fn charged_world2(charges: &[(f32, f32)]) -> (World2, Vec<usize>) {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let mut ids = Vec::new();
    for &(x, q) in charges {
        let id = w
            .add_dynamic_body(Vec2::new(x, 0.0), Shape2::Circle { radius: 0.05 }, 1.0)
            .unwrap();
        w.bodies[id].linear_damping = 0.0;
        w.bodies[id].angular_damping = 0.0;
        w.bodies[id].electro.charge = q;
        ids.push(id);
    }
    w.electro.configure(pairwise2()).unwrap();
    (w, ids)
}

fn step(w: &mut World2, n: usize) {
    for _ in 0..n {
        w.step(1.0 / 60.0);
    }
}

#[test]
fn em01_like_charges_repel() {
    let (mut w, ids) = charged_world2(&[(-1.0, 1.0), (1.0, 1.0)]);
    w.step(1.0 / 60.0);
    assert!(
        w.bodies[ids[0]].velocity.x < 0.0,
        "left must be pushed left"
    );
    assert!(
        w.bodies[ids[1]].velocity.x > 0.0,
        "right must be pushed right"
    );
}

#[test]
fn em02_opposite_charges_attract() {
    let (mut w, ids) = charged_world2(&[(-1.0, 1.0), (1.0, -1.0)]);
    w.step(1.0 / 60.0);
    assert!(
        w.bodies[ids[0]].velocity.x > 0.0,
        "left must be pulled right"
    );
    assert!(
        w.bodies[ids[1]].velocity.x < 0.0,
        "right must be pulled left"
    );
}

#[test]
fn em03_uniform_field_accelerates_by_q_over_m() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let id = w
        .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.1 }, 1.0)
        .unwrap();
    w.bodies[id].linear_damping = 0.0;
    w.bodies[id].electro.charge = 2.0;
    let mut cfg = pairwise2();
    cfg.uniform_e = Vec2::new(3.0, 0.0);
    w.electro.configure(cfg).unwrap();
    w.step(1.0 / 60.0);
    let expected = 2.0 * 3.0 * w.bodies[id].inv_mass / 60.0;
    assert!(
        (w.bodies[id].velocity.x - expected).abs() < 1e-5,
        "vx = {}, expected {expected}",
        w.bodies[id].velocity.x
    );
}

#[test]
fn em04_charge_scale_zero_is_immune() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let id = w
        .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.1 }, 1.0)
        .unwrap();
    w.bodies[id].linear_damping = 0.0;
    w.bodies[id].electro.charge = 1.0;
    w.bodies[id].electro.charge_scale = 0.0;
    let mut cfg = pairwise2();
    cfg.uniform_e = Vec2::new(5.0, 0.0);
    w.electro.configure(cfg).unwrap();
    step(&mut w, 10);
    assert_eq!(w.bodies[id].velocity, Vec2::ZERO);
}

#[test]
fn em05_static_body_exerts_but_does_not_move() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let fixed = w
        .add_static_body(Vec2::ZERO, Shape2::Circle { radius: 0.1 })
        .unwrap();
    w.bodies[fixed].electro.charge = 1.0;
    let free = w
        .add_dynamic_body(Vec2::new(1.0, 0.0), Shape2::Circle { radius: 0.1 }, 1.0)
        .unwrap();
    w.bodies[free].linear_damping = 0.0;
    w.bodies[free].electro.charge = 1.0;
    w.electro.configure(pairwise2()).unwrap();
    step(&mut w, 10);
    assert_eq!(w.bodies[fixed].transform.position, Vec2::ZERO);
    assert!(w.bodies[free].transform.position.x > 1.0);
}

#[test]
fn em06_enabled_with_zero_charge_is_bit_identical() {
    let mut a = World2::new();
    let mut b = World2::new();
    for w in [&mut a, &mut b] {
        w.add_dynamic_body(Vec2::new(0.0, 3.0), Shape2::Circle { radius: 0.2 }, 1.0)
            .unwrap();
    }
    b.electro.configure(pairwise2()).unwrap();
    for _ in 0..60 {
        a.step(1.0 / 60.0);
        b.step(1.0 / 60.0);
    }
    assert_eq!(
        a.bodies[0].transform.position,
        b.bodies[0].transform.position
    );
    assert_eq!(a.bodies[0].velocity, b.bodies[0].velocity);
}

#[test]
fn em07_field_mode_pushes_like_charges_apart() {
    let mut cfg = pairwise2();
    cfg.mode = ElectroMode::Field;
    cfg.field.nx = 48;
    cfg.field.ny = 48;
    cfg.field.dx = 0.1;
    cfg.field.origin = Vec2::splat(-2.4);
    cfg.field.iterations = 120;
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let mut ids = Vec::new();
    for x in [-0.5f32, 0.5] {
        let id = w
            .add_dynamic_body(Vec2::new(x, 0.0), Shape2::Circle { radius: 0.05 }, 1.0)
            .unwrap();
        w.bodies[id].linear_damping = 0.0;
        w.bodies[id].electro.charge = 1.0;
        ids.push(id);
    }
    w.electro.configure(cfg).unwrap();
    step(&mut w, 20);
    assert!(w.bodies[ids[0]].transform.position.x < -0.5);
    assert!(w.bodies[ids[1]].transform.position.x > 0.5);
}

#[test]
fn em08_grounded_conductor_drains() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let id = w
        .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.1 }, 1.0)
        .unwrap();
    w.bodies[id].electro = ElectroState2 {
        charge: 5.0,
        is_conductor: true,
        is_grounded: true,
        resistance: 0.0,
        ..Default::default()
    };
    let mut cfg = pairwise2();
    cfg.mode = ElectroMode::Off;
    w.electro.configure(cfg).unwrap();
    w.step(1.0 / 60.0);
    assert_eq!(w.bodies[id].electro.charge, 0.0);
}

#[test]
fn em09_three_d_opposite_charges_attract() {
    let mut w = World3::new();
    w.gravity = Vec3::ZERO;
    let mut ids = Vec::new();
    for (x, q) in [(-1.0f32, 1.0f32), (1.0, -1.0)] {
        let id = w
            .add_dynamic_body(Vec3::new(x, 0.0, 0.0), Shape3::Sphere { radius: 0.05 }, 1.0)
            .unwrap();
        w.bodies[id].linear_damping = 0.0;
        w.bodies[id].electro.charge = q;
        w.bodies[id].is_sensor = true;
        ids.push(id);
    }
    w.electro
        .configure(ElectroConfig3 {
            enabled: true,
            mode: ElectroMode::Pairwise,
            ..Default::default()
        })
        .unwrap();
    w.step(1.0 / 60.0);
    assert!(
        w.bodies[ids[0]].velocity.x > 0.0,
        "3D left must be pulled right"
    );
    assert!(
        w.bodies[ids[1]].velocity.x < 0.0,
        "3D right must be pulled left"
    );
}

#[test]
fn em10_field_and_potential_query_are_analytic() {
    let mut w = World2::new();
    w.gravity = Vec2::ZERO;
    let id = w
        .add_dynamic_body(Vec2::ZERO, Shape2::Circle { radius: 0.1 }, 1.0)
        .unwrap();
    w.bodies[id].electro.charge = 1.0;
    w.electro.configure(pairwise2()).unwrap();
    // k = 1, q = 1, at (2, 0): E = (0.25, 0), V = 0.5.
    let e = phieng::electro::electric_field_at_2(&w.bodies, &w.electro, Vec2::new(2.0, 0.0));
    assert!((e - Vec2::new(0.25, 0.0)).length() < 1e-6, "E = {e:?}");
    let v = phieng::electro::electric_potential_at_2(&w.bodies, &w.electro, Vec2::new(2.0, 0.0));
    assert!((v - 0.5).abs() < 1e-6, "V = {v}");
}

#[test]
fn em11_ffi_config_and_body_round_trip() {
    use phieng::ffi::{
        phieng_world_add_body, phieng_world_free, phieng_world_get_body_charge,
        phieng_world_get_body_electro_state, phieng_world_get_electro_config, phieng_world_new,
        phieng_world_set_body_charge, phieng_world_set_body_electro_state,
        phieng_world_set_electro_config, ElectroBodyFfi, ElectroConfigFfi2, SHAPE_CIRCLE,
    };
    unsafe {
        let world = phieng_world_new();
        let id =
            phieng_world_add_body(world, 0.0, 0.0, SHAPE_CIRCLE, 0.5, 0.0, 1.0, 0, 0.5, 0.0) as u32;
        let cfg = ElectroConfigFfi2 {
            mode: 1,
            boundary: 0,
            enabled: 1,
            coulomb_k: 1.0,
            softening: 0.0,
            ex: 2.0,
            ey: 0.0,
            bz: 0.0,
            permittivity: 1.0,
            permeability: 1.0,
            max_acceleration: 1.0e30,
            nx: 64,
            ny: 64,
            dx: 0.1,
            ox: -3.2,
            oy: -3.2,
            iterations: 40,
            omega: 1.8,
            eps_r: 1.0,
            snapshot_field: 1,
        };
        assert_eq!(phieng_world_set_electro_config(world, &cfg), 1);
        let mut got_cfg = ElectroConfigFfi2::default();
        assert_eq!(phieng_world_get_electro_config(world, &mut got_cfg), 1);
        assert_eq!(got_cfg.mode, 1);
        assert_eq!(got_cfg.ex, 2.0);

        let state = ElectroBodyFfi {
            charge: 3.0,
            charge_scale: 1.0,
            conductivity: 0.0,
            permittivity: 1.0,
            permeability: 1.0,
            magnetic_moment_x: 0.0,
            magnetic_moment_y: 0.0,
            magnetic_moment_z: 0.0,
            capacitance: 1.0,
            resistance: 0.0,
            flags: 0,
        };
        assert_eq!(phieng_world_set_body_electro_state(world, id, &state), 1);
        let mut got = ElectroBodyFfi::default();
        assert_eq!(phieng_world_get_body_electro_state(world, id, &mut got), 1);
        assert_eq!(got.charge, 3.0);

        assert_eq!(phieng_world_set_body_charge(world, id, 4.0), 1);
        let mut q = 0.0f32;
        assert_eq!(phieng_world_get_body_charge(world, id, &mut q), 1);
        assert_eq!(q, 4.0);
        assert_eq!(phieng_world_set_body_charge(world, id, f32::NAN), 0);

        phieng_world_free(world);
    }
}

#[test]
fn em12_snapshot_round_trips_the_block() {
    use phieng::snapshot::{world2_from_json, world2_to_json, LoadMode};
    let (w, ids) = charged_world2(&[(-1.0, 2.0), (1.0, -3.0)]);
    let text = world2_to_json(&w, LoadMode::Exact).unwrap();
    let mut loaded = world2_from_json(&text, LoadMode::Exact).unwrap();
    for &id in &ids {
        assert_eq!(
            loaded.bodies[id].electro.charge,
            w.bodies[id].electro.charge
        );
    }
    assert_eq!(loaded.electro.config, w.electro.config);
    // The trajectory continues exactly.
    let mut original = w;
    for _ in 0..30 {
        original.step(1.0 / 60.0);
        loaded.step(1.0 / 60.0);
    }
    for &id in &ids {
        assert_eq!(
            loaded.bodies[id].transform.position,
            original.bodies[id].transform.position
        );
    }
}
