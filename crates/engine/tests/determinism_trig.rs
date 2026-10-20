//! Deterministic trigonometry: the cross-platform anchor (R01-R03).
//!
//! Mirrors `tests/PhiEng.Tests/DeterministicTrigTests.cs` (enforced by
//! `tools/check_test_parity.py`).
//!
//! The platform math library is not portable: `sinf`, `cosf` and
//! `atan2f` disagree between glibc (Linux) and the Microsoft CRT
//! (Windows) in the last ULP, and the same platform can disagree with
//! itself when the compiler rewrites an inlined call. In a rigid-body
//! simulation that single ULP is enough to send a trajectory in two
//! different directions.
//!
//! These tests pin *exact bit patterns* of the deterministic helpers and
//! of the engine's own rotation integration. Because the helpers never
//! call the platform libm, the pins are the same on every platform -
//! that is the property under test. A change here means the arithmetic
//! changed, and every pinned trajectory hash in the repo is invalidated.

use phieng_math::trig::{atan2, compute_cos_sin};

/// R01: the deterministic sine and cosine, pinned to exact bit patterns.
#[test]
fn r01_deterministic_cos_sin_is_portable() {
    let cases: [(f32, u32, u32); 8] = [
        (0.0, 0x3f800000, 0x00000000),
        (0.5, 0x3f60a940, 0x3ef57744),
        (1.0, 0x3f0a5140, 0x3f576aa4),
        (-1.0, 0x3f0a5140, 0xbf576aa4),
        (std::f32::consts::FRAC_PI_2, 0xb33bbd2e, 0x3f800000),
        (std::f32::consts::PI, 0xbf800000, 0xb3bbbd2e),
        (3.0, 0xbf7d7026, 0x3e1081c3),
        (-7.25, 0x3f11637b, 0xbf52b56e),
    ];
    for (angle, cos_bits, sin_bits) in cases {
        let (cos, sin) = compute_cos_sin(angle);
        assert_eq!(
            (cos.to_bits(), sin.to_bits()),
            (cos_bits, sin_bits),
            "compute_cos_sin({angle}) is not portable"
        );
    }
}

/// R02: the deterministic `atan2`, pinned to exact bit patterns.
#[test]
fn r02_deterministic_atan2_is_portable() {
    let cases: [(f32, f32, u32); 8] = [
        (0.0, 1.0, 0x00000000),
        (1.0, 0.0, 0x3fc90fdb),
        (-1.0, 0.0, 0xbfc90fdb),
        (0.0, -1.0, 0x40490fdb),
        (1.0, 1.0, 0x3f490fdb),
        (-3.5, 2.25, 0xbf7fdc89),
        (0.125, -0.75, 0x403e7e0f),
        (-1e-3, 1e-3, 0xbf490fdb),
    ];
    for (y, x, bits) in cases {
        assert_eq!(
            atan2(y, x).to_bits(),
            bits,
            "atan2({y}, {x}) is not portable"
        );
    }
}

/// R03: the engine's own rotation integration is portable.
///
/// This is the test that would have caught the original CI failure: it
/// pins the rotation matrix a body ends up with after a fixed number of
/// integration steps. The pins are identical on every platform, so the
/// test can only pass if the engine really is free of the platform math
/// library.
#[test]
fn r03_body_rotation_integration_is_portable() {
    use glam::Vec2;
    use phieng_dynamics::body::Shape2;
    use phieng_dynamics::World2;

    const DT: f32 = 1.0 / 60.0;

    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    let id = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");

    {
        let body = &mut world.bodies[id];
        body.angular_velocity = 1.75;
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
    }

    for _ in 0..10 {
        world.step(DT);
    }

    let body = &world.bodies[id];
    let rot = body.transform.rotation;
    assert_eq!(
        (
            rot.x_axis.x.to_bits(),
            rot.x_axis.y.to_bits(),
            rot.y_axis.x.to_bits(),
            rot.y_axis.y.to_bits()
        ),
        (
            0x3f753026, // x_axis.x
            0x3e93399d, // x_axis.y
            0xbe93399d, // y_axis.x
            0x3f753026, // y_axis.y
        ),
        "the 2D rotation integration is not portable"
    );
}

/// R04: the deterministic exponential, pinned to exact bit patterns.
#[test]
fn r04_deterministic_exp_is_portable() {
    use phieng_math::exponential::exp;

    let cases: [(f32, u32); 9] = [
        (0.0, 0x3f800000),
        (0.5, 0x3fd3094c),
        (1.0, 0x402df854),
        (-1.0, 0x3ebc5ab2),
        (2.5, 0x4142eb7f),
        (-10.0, 0x383e6bce),
        (20.0, 0x4de75844),
        (0.001, 0x3f8020c9),
        (100.0, 0x7f800000), // overflows to infinity
    ];
    for (x, bits) in cases {
        assert_eq!(exp(x).to_bits(), bits, "exp({x}) is not portable");
    }
}

/// R05: the deterministic logarithm, pinned to exact bit patterns.
#[test]
fn r05_deterministic_ln_is_portable() {
    use phieng_math::exponential::ln;

    let cases: [(f32, u32); 8] = [
        (1.0, 0x00000000),
        (2.0, 0x3f317218),
        (0.5, 0xbf317218),
        (10.0, 0x40135d8e),
        (0.001, 0xc0dd0c55),
        (1e-6, 0xc15d0c55),
        (3.5, 0x3fa05a89),
        (100.0, 0x40935d8e),
    ];
    for (x, bits) in cases {
        assert_eq!(ln(x).to_bits(), bits, "ln({x}) is not portable");
    }
}

/// R06: the deterministic helpers track the platform math library.
///
/// The tolerance is in ULPs of the result and is far looser than the
/// actual agreement (which is ~0.5 ULP, i.e. correctly rounded). The
/// point is not the accuracy — it is that the deterministic version is
/// *close enough* that replacing the platform call does not change the
/// character of the physics.
#[test]
fn r06_exponential_matches_platform_libm() {
    use phieng_math::exponential::{exp, ln};

    fn ulp_diff(a: f32, b: f32) -> f64 {
        if a == b {
            return 0.0;
        }
        let (a, b) = (a as f64, b as f64);
        let scale = a.abs().max(b.abs()).max(f32::MIN_POSITIVE as f64);
        (a - b).abs() / (scale * f32::EPSILON as f64)
    }

    let mut worst_exp = 0.0f64;
    let mut worst_ln = 0.0f64;
    for i in -60000..=60000 {
        let x = (i as f32) * 0.001;
        worst_exp = worst_exp.max(ulp_diff(exp(x), (x as f64).exp() as f32));
    }
    for i in 1..20000 {
        let x = (i as f32) * 1e-3;
        worst_ln = worst_ln.max(ulp_diff(ln(x), (x as f64).ln() as f32));
    }
    assert!(
        worst_exp <= 2.0,
        "exp deviates by {worst_exp} ULP from the platform libm"
    );
    assert!(
        worst_ln <= 2.0,
        "ln deviates by {worst_ln} ULP from the platform libm"
    );
}

/// R07: `exp` and `ln` are mutual inverses.
///
/// The nuclear subsystem turns a half-life into a decay constant and
/// decays an activity with it; this is the property that makes the
/// round trip meaningful.
#[test]
fn r07_exponential_and_logarithm_are_mutual_inverses() {
    use phieng_math::exponential::{exp, ln};

    for i in -2000..=2000 {
        let x = (i as f32) * 0.01;
        let back = ln(exp(x));
        assert!(
            (back - x).abs() <= x.abs() * 1e-5 + 1e-6,
            "ln(exp({x})) = {back}"
        );
    }
}
