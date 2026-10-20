//! Radiation model expansion (U01-U12).
//!
//! Mirrors `tests/PhiEng.Tests/RadiationTests.cs` (enforced by
//! `tools/check_test_parity.py`).
//!
//! The radiation model gained three opt-in options, all of which default
//! to the historical behaviour so existing trajectories stay
//! bit-identical:
//!
//! * a symmetric (order-independent) view-factor model,
//! * a sky/ground split for the environment radiation,
//! * a partially transparent occluder,
//!
//! plus a fix to the occlusion test, which used to let radiation leak
//! through gaps next to the two exchanging bodies.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::thermal::ViewFactorModel;
use phieng_dynamics::World2;

const DT: f32 = 1.0 / 60.0;

/// The ambient temperature used by the scenes.
///
/// It is deliberately equal to the hot body's temperature, so the hot
/// body's *environment* exchange is zero and the only heat it moves is
/// the pair exchange under test.
const AMBIENT: f32 = 500.0;

/// The hot body's temperature.
const HOT: f32 = 500.0;

/// The cold body's temperature.
const COLD: f32 = 300.0;

/// The hot body's radius.
const R_HOT: f32 = 0.5;

/// The cold body's radius. Equal to [`R_HOT`] unless a test asks for
/// unequal bodies.
const R_COLD: f32 = 0.5;

/// A hot body at `x_hot` and a cold body at `x_cold`, exchanging
/// radiation with nothing else in the scene.
///
/// Gravity is off and the sleep thresholds are zero, so the only thing
/// that changes over a step is the temperature.
fn scene(x_hot: f32, x_cold: f32, r_cold: f32) -> (World2, usize, usize) {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;

    let cfg = &mut world.thermal.config;
    cfg.enabled = true;
    cfg.radiation_enabled = true;
    cfg.conduction_enabled = false;
    cfg.convection_enabled = false;
    cfg.ambient_temperature = AMBIENT;
    // A large rate cap so the exchange is not clamped away.
    cfg.max_temperature_rate = 1e9;

    let hot = world
        .add_dynamic_body(Vec2::new(x_hot, 0.0), Shape2::Circle { radius: R_HOT }, 1.0)
        .expect("valid shape");
    let cold = world
        .add_dynamic_body(
            Vec2::new(x_cold, 0.0),
            Shape2::Circle { radius: r_cold },
            1.0,
        )
        .expect("valid shape");
    world.bodies[hot].temperature = HOT;
    world.bodies[cold].temperature = COLD;
    (world, hot, cold)
}

/// The heat the pair exchange moved into the hot body in one step.
///
/// The ambient is at the hot body's temperature, so the hot body's
/// environment exchange cancels out and this is the pair exchange
/// alone.
fn pair_exchange(r_cold: f32, setup: impl Fn(&mut World2)) -> f32 {
    let (mut world, hot, _) = scene(-1.0, 1.0, r_cold);
    setup(&mut world);
    let before = world.bodies[hot].temperature;
    world.step(DT);
    world.bodies[hot].temperature - before
}

/// An occluding body between the pair, at the hot body's temperature so
/// that it exchanges no heat with it.
fn add_occluder(world: &mut World2, at: Vec2) {
    let id = world
        .add_dynamic_body(at, Shape2::Circle { radius: 0.4 }, 1.0)
        .expect("valid shape");
    world.bodies[id].temperature = HOT;
}

/// The heat capacity of a circle body of the given radius.
fn heat_capacity(radius: f32) -> f32 {
    let mass = std::f32::consts::PI * radius * radius; // density 1.0
    mass * 900.0 // the default specific heat
}

/// U01: the legacy view-factor model is the default, and the pair
/// exchange matches the historical formula.
#[test]
fn u01_legacy_view_factor_model_is_the_default() {
    let (world, _, _) = scene(-1.0, 1.0, R_COLD);
    assert_eq!(
        world.thermal.config.view_factor_model,
        ViewFactorModel::Legacy,
        "the radiation model must default to the legacy behaviour"
    );

    // The historical formula, computed by hand:
    //   eps_eff = 1 / (1/eps_i + 1/eps_j - 1)
    //   F       = (r_j / d)^2
    //   Q       = eps_eff * sigma * A * F * (T_i^4 - T_j^4)
    // and the temperature change is Q / (m * c_p) * dt.
    let sigma = world.thermal.config.stefan_boltzmann;
    let eps = world.bodies[0].material.emissivity.clamp(0.0, 1.0);
    let eps_eff = 1.0 / (1.0 / eps + 1.0 / eps - 1.0);
    let d = 2.0f32;
    let view_factor = (R_COLD / d).powi(2);
    // The 2D pair term uses the characteristic length (the radius).
    let area = R_COLD;
    let q = eps_eff * sigma * area * view_factor * (HOT.powi(4) - COLD.powi(4));
    // Mirror the engine's arithmetic exactly: the hot body *loses* the
    // heat, the delta is `(flux / heat_capacity) * dt`, and the *sum* is
    // rounded to f32. At 500 K the f32 spacing is 3e-5, so the sum is
    // what the test has to compare against.
    let expected = (HOT + (-q / heat_capacity(R_HOT)) * DT) - HOT;

    let measured = pair_exchange(R_COLD, |_| {});
    assert_eq!(
        measured.to_bits(),
        expected.to_bits(),
        "the legacy pair exchange should be {expected}, measured {measured}"
    );
}

/// U02: the symmetric model agrees with the legacy model for two
/// identical bodies.
#[test]
fn u02_symmetric_model_matches_legacy_for_equal_bodies() {
    let legacy = pair_exchange(R_COLD, |_| {});
    let symmetric = pair_exchange(R_COLD, |w| {
        w.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
    });

    assert!(
        (symmetric - legacy).abs() <= legacy.abs() * 1e-3 + 1e-9,
        "the symmetric model should match the legacy one for equal bodies: \
         legacy {legacy}, symmetric {symmetric}"
    );
}

/// U03: the symmetric model does not depend on which body of the pair
/// the loop visits first.
#[test]
fn u03_symmetric_model_is_order_independent() {
    // Hot on the left, then hot on the right: the exchange magnitude
    // must be the same.
    let a = pair_exchange(R_COLD, |w| {
        w.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
    });

    let (mut world, hot, _) = scene(1.0, -1.0, R_COLD);
    world.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
    let before = world.bodies[hot].temperature;
    world.step(DT);
    let b = world.bodies[hot].temperature - before;

    assert!(
        (a - b).abs() <= a.abs() * 1e-3 + 1e-9,
        "the symmetric exchange must not depend on the body order: {a} vs {b}"
    );
}

/// U04: the symmetric model is a *different* model for unequal bodies.
#[test]
fn u04_symmetric_model_differs_from_legacy_for_unequal_bodies() {
    let legacy = pair_exchange(0.25, |_| {});
    let symmetric = pair_exchange(0.25, |w| {
        w.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
    });

    assert!(
        (symmetric - legacy).abs() > legacy.abs() * 1e-3,
        "the symmetric model must actually change the exchange for unequal \
         bodies: legacy {legacy}, symmetric {symmetric}"
    );
}

/// U05: a sky/ground split changes the environment exchange.
#[test]
fn u05_sky_ground_split_changes_the_equilibrium() {
    // A body in a cold sky and a warm ground.
    let (mut world, body, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.sky_temperature = 100.0;
    world.thermal.config.ground_temperature = 400.0;
    let before = world.bodies[body].temperature;
    world.step(DT);
    let split = world.bodies[body].temperature - before;

    // The same scene with a uniform environment.
    let (mut world, body, _) = scene(-1.0, 1.0, R_COLD);
    let before = world.bodies[body].temperature;
    world.step(DT);
    let uniform = world.bodies[body].temperature - before;

    assert!(
        (split - uniform).abs() > uniform.abs() * 1e-3,
        "the sky/ground split must change the exchange: \
         split {split}, uniform {uniform}"
    );
}

/// U06: a sky and ground equal to the ambient are bit-identical to the
/// historical single-temperature exchange.
#[test]
fn u06_equal_sky_and_ground_match_the_historical_exchange() {
    let (mut world, body, _) = scene(-1.0, 1.0, R_COLD);
    let baseline_before = world.bodies[body].temperature;
    world.step(DT);
    let baseline = world.bodies[body].temperature - baseline_before;

    let (mut world, body, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.sky_temperature = AMBIENT;
    world.thermal.config.ground_temperature = AMBIENT;
    let before = world.bodies[body].temperature;
    world.step(DT);
    let explicit = world.bodies[body].temperature - before;

    assert_eq!(
        baseline.to_bits(),
        explicit.to_bits(),
        "an explicit sky/ground equal to the ambient must be bit-identical \
         to the historical exchange"
    );
}

/// U07: an opaque body between the pair blocks the exchange.
#[test]
fn u07_opaque_occluder_blocks_the_exchange() {
    let unoccluded = pair_exchange(R_COLD, |_| {});

    let (mut world, hot, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.radiation_occlusion = true;
    add_occluder(&mut world, Vec2::new(0.0, 0.0));
    let before = world.bodies[hot].temperature;
    world.step(DT);
    let occluded = world.bodies[hot].temperature - before;

    assert!(
        occluded.abs() < unoccluded.abs() * 0.5,
        "an opaque occluder must block most of the exchange: \
         unoccluded {unoccluded}, occluded {occluded}"
    );
}

/// U08: an occluder *next to* one of the exchanging bodies blocks the
/// exchange too.
///
/// This is the regression test for the removed midpoint heuristic: the
/// old code exempted blockers that sat close to an endpoint, so
/// radiation leaked through the gap.
#[test]
fn u08_off_axis_occluder_also_blocks() {
    let unoccluded = pair_exchange(R_COLD, |_| {});

    // The blocker sits just next to the hot body, off the exact line.
    let (mut world, hot, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.radiation_occlusion = true;
    add_occluder(&mut world, Vec2::new(-0.35, 0.35));
    let before = world.bodies[hot].temperature;
    world.step(DT);
    let occluded = world.bodies[hot].temperature - before;

    assert!(
        occluded.abs() < unoccluded.abs() * 0.5,
        "an off-axis occluder next to an endpoint must block the exchange: \
         unoccluded {unoccluded}, occluded {occluded}"
    );
}

/// U09: a partially transparent occluder attenuates the exchange
/// instead of blocking it completely.
#[test]
fn u09_partial_occluder_attenuates_the_exchange() {
    let unoccluded = pair_exchange(R_COLD, |_| {});

    let (mut world, hot, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.radiation_occlusion = true;
    world.thermal.config.occluder_transmittance = 0.5;
    add_occluder(&mut world, Vec2::new(0.0, 0.0));
    let before = world.bodies[hot].temperature;
    world.step(DT);
    let attenuated = world.bodies[hot].temperature - before;

    assert!(
        attenuated.abs() < unoccluded.abs() * 0.9 && attenuated.abs() > unoccluded.abs() * 0.1,
        "a half-transparent occluder must attenuate, not block: \
         unoccluded {unoccluded}, attenuated {attenuated}"
    );
}

/// U10: a fully transparent occluder does not block at all.
#[test]
fn u10_fully_transparent_occluder_does_not_block() {
    let unoccluded = pair_exchange(R_COLD, |_| {});

    let (mut world, hot, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.radiation_occlusion = true;
    world.thermal.config.occluder_transmittance = 1.0;
    add_occluder(&mut world, Vec2::new(0.0, 0.0));
    let before = world.bodies[hot].temperature;
    world.step(DT);
    let transmitted = world.bodies[hot].temperature - before;

    assert!(
        (transmitted - unoccluded).abs() <= unoccluded.abs() * 1e-3 + 1e-9,
        "a fully transparent occluder must not block: \
         unoccluded {unoccluded}, transmitted {transmitted}"
    );
}

/// U11: the radiation model is deterministic, and the parallel solver
/// lands on the same trajectory.
#[test]
fn u11_radiation_is_deterministic() {
    let run = |parallel: bool| -> f32 {
        let (mut world, _, _) = scene(-1.0, 1.0, R_COLD);
        world.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
        world.thermal.config.sky_temperature = 150.0;
        world.thermal.config.ground_temperature = 320.0;
        world.solver.parallel = parallel;
        for _ in 0..60 {
            world.step(DT);
        }
        world.bodies[0].temperature
    };

    let a = run(false);
    let b = run(false);
    assert_eq!(a.to_bits(), b.to_bits(), "two identical runs must agree");

    let p = run(true);
    assert_eq!(
        a.to_bits(),
        p.to_bits(),
        "the parallel solver must land on the serial trajectory"
    );
}

/// U12: the radiation options survive a snapshot round-trip (JSON and
/// binary).
#[test]
fn u12_radiation_options_round_trip_through_a_snapshot() {
    use phieng_dynamics::snapshot::{
        world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, LoadMode,
    };

    let (mut world, _, _) = scene(-1.0, 1.0, R_COLD);
    world.thermal.config.view_factor_model = ViewFactorModel::Symmetric;
    world.thermal.config.sky_temperature = 123.5;
    world.thermal.config.ground_temperature = 287.25;
    world.thermal.config.occluder_transmittance = 0.375;

    let json = world2_to_json(&world, LoadMode::Exact).expect("serializable");
    let restored = world2_from_json(&json, LoadMode::Exact).expect("loadable");
    assert_eq!(
        restored.thermal.config.view_factor_model,
        ViewFactorModel::Symmetric
    );
    assert_eq!(restored.thermal.config.sky_temperature, 123.5);
    assert_eq!(restored.thermal.config.ground_temperature, 287.25);
    assert_eq!(restored.thermal.config.occluder_transmittance, 0.375);

    let binary = world2_to_binary(&world, LoadMode::Exact).expect("serializable");
    let restored = world2_from_binary(&binary, LoadMode::Exact).expect("loadable");
    assert_eq!(
        restored.thermal.config.view_factor_model,
        ViewFactorModel::Symmetric
    );
    assert_eq!(restored.thermal.config.sky_temperature, 123.5);
    assert_eq!(restored.thermal.config.ground_temperature, 287.25);
    assert_eq!(restored.thermal.config.occluder_transmittance, 0.375);
}
