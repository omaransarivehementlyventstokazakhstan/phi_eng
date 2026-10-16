//! Ambient force tests (A01–A05): spring/SHM, air drag, and turbulence.
//! These embed shared parity IDs mirrored by
//! `tests/PhiEng.Tests/AmbientTests.cs` (enforced by
//! `tools/check_test_parity.py`), so the managed and Rust engines are
//! tested against the same contract.

use glam::Vec2;
use phieng_dynamics::ambient::{AmbientConfig2, DragConfig2, SpringConfig2, TurbulenceConfig2};
use phieng_dynamics::{Shape2, World2};

const DT: f32 = 1.0 / 240.0;

fn clean(w: &mut World2) {
    w.gravity = Vec2::ZERO;
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

/// A01: an undamped spring produces simple-harmonic oscillation about
/// the anchor — the body crosses the anchor and returns near its start
/// after one period.
#[test]
fn a01_spring_shm_period() {
    let mut w = World2::new();
    let id = w
        .add_dynamic_body(Vec2::new(1.0, 0.0), Shape2::Circle { radius: 0.2 }, 1.0)
        .expect("valid shape");
    clean(&mut w);
    w.ambient = AmbientConfig2 {
        spring: SpringConfig2 {
            enabled: true,
            anchor: Vec2::ZERO,
            stiffness: 40.0,
            damping: 0.0,
            max_displacement: 0.0,
        },
        ..Default::default()
    };
    let omega = (40.0 / w.bodies[id].mass).sqrt();
    let period = 2.0 * std::f32::consts::PI / omega;
    let steps = (period / DT).round() as usize;
    let mut crossed = false;
    for _ in 0..steps {
        w.step(DT);
        if w.bodies[id].transform.position.x < 0.0 {
            crossed = true;
        }
    }
    assert!(crossed, "spring never crossed the anchor");
    assert!(
        (w.bodies[id].transform.position.x - 1.0).abs() < 0.05,
        "SHM did not close a period: x={}",
        w.bodies[id].transform.position.x
    );
}

/// A02: linear drag + gravity reaches terminal velocity `v = g*m/linear`.
#[test]
fn a02_drag_terminal_velocity() {
    let mut w = World2::new();
    let id = w
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    clean(&mut w);
    w.gravity = Vec2::new(0.0, -9.81);
    let linear = 4.0f32;
    w.ambient = AmbientConfig2 {
        drag: DragConfig2 {
            enabled: true,
            linear,
            quadratic: 0.0,
            air_density: 1.0,
            angular: 0.0,
            wind: Vec2::ZERO,
        },
        ..Default::default()
    };
    for _ in 0..4000 {
        w.step(DT);
    }
    let expected = -9.81 * w.bodies[id].mass / linear;
    assert!(
        (w.bodies[id].velocity.y - expected).abs() < 0.2,
        "terminal velocity off: {} want {}",
        w.bodies[id].velocity.y,
        expected
    );
}

/// A03: turbulence is deterministic — identical setups produce
/// bit-identical velocities.
#[test]
fn a03_turbulence_deterministic() {
    fn run() -> Vec2 {
        let mut w = World2::new();
        let id = w
            .add_dynamic_body(Vec2::new(1.0, 2.0), Shape2::Circle { radius: 0.3 }, 1.0)
            .expect("valid shape");
        clean(&mut w);
        w.ambient = AmbientConfig2 {
            turbulence: TurbulenceConfig2 {
                enabled: true,
                amplitude: 5.0,
                frequency: 0.5,
                time_scale: 1.0,
                seed: 42,
                octaves: 4,
                divergence_free: true,
            },
            ..Default::default()
        };
        for _ in 0..200 {
            w.step(1.0 / 60.0);
        }
        w.bodies[id].velocity
    }
    let a = run();
    let b = run();
    assert_eq!(a.x.to_bits(), b.x.to_bits());
    assert_eq!(a.y.to_bits(), b.y.to_bits());
}

/// A04: a default ambient config is inert — free fall is exactly `g*t`.
#[test]
fn a04_ambient_default_noop() {
    let mut w = World2::new();
    let id = w
        .add_dynamic_body(Vec2::new(0.0, 10.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    clean(&mut w);
    w.gravity = Vec2::new(0.0, -9.81);
    for _ in 0..240 {
        w.step(1.0 / 60.0);
    }
    assert!(
        (w.bodies[id].velocity.y - (-9.81 * 4.0)).abs() < 1e-2,
        "default ambient perturbed free fall: {}",
        w.bodies[id].velocity.y
    );
}

/// A05: `ambient_scale == 0` makes a body immune to ambient forces.
#[test]
fn a05_ambient_scale_immune() {
    let mut w = World2::new();
    let id = w
        .add_dynamic_body(Vec2::new(3.0, 0.0), Shape2::Circle { radius: 0.3 }, 1.0)
        .expect("valid shape");
    clean(&mut w);
    w.bodies[id].ambient_scale = 0.0;
    w.ambient = AmbientConfig2 {
        spring: SpringConfig2 {
            enabled: true,
            anchor: Vec2::ZERO,
            stiffness: 500.0,
            damping: 0.5,
            max_displacement: 0.0,
        },
        ..Default::default()
    };
    for _ in 0..60 {
        w.step(1.0 / 60.0);
    }
    assert_eq!(w.bodies[id].velocity, Vec2::ZERO, "immune body moved");
}
