//! Nuclear physics (Y01-Y20).
//!
//! Mirrors `tests/PhiEng.Tests/NuclearTests.cs` (enforced by
//! `tools/check_test_parity.py`).
//!
//! The nuclear subsystem is a game-scale model: decay, activation,
//! fission and a rate-based chain reaction. It is off by default, so a
//! default world is bit-identical to a pre-nuclear one.

use glam::{Vec2, Vec3};
use phieng_dynamics::body::Shape2;
use phieng_dynamics::nuclear::{Isotope, NuclearConfig};
use phieng_dynamics::World2;

const DT: f32 = 1.0 / 60.0;

/// A nuclide index. Index 0 is the "no nuclide" placeholder.
const NO_NUCLIDE: u16 = 0;
/// A short-lived nuclide (half-life in nuclear seconds).
const SHORT: u16 = 1;
/// A stable nuclide.
const STABLE: u16 = 2;
/// A fissile nuclide.
const FISSILE: u16 = 3;

/// A configuration with a small nuclide table.
fn config() -> NuclearConfig {
    NuclearConfig {
        enabled: true,
        // Scaled so the fission effects are far above the f32 noise floor.
        fission_energy: 1.0e6,
        fuel_per_fission: 1.0e-3,
        isotopes: vec![
            // 0: no nuclide.
            Isotope::default(),
            // 1: a short-lived nuclide that decays in 10 nuclear seconds.
            Isotope {
                half_life: 10.0,
                decay_energy: 1.0,
                ionization_fraction: 0.5,
                daughter: STABLE,
                ..Isotope::default()
            },
            // 2: stable.
            Isotope {
                half_life: 0.0,
                decay_energy: 0.0,
                ..Isotope::default()
            },
            // 3: fissile.
            Isotope {
                half_life: 0.0,
                decay_energy: 0.0,
                ionization_fraction: 0.0,
                daughter: NO_NUCLIDE,
                absorption_cross_section: 1.0,
                fission_cross_section: 1.0,
                neutron_yield: 2.0,
                is_fissile: true,
            },
        ],
        ..NuclearConfig::default()
    }
}

/// A world with one dynamic body carrying `nuclide`.
fn world_with(nuclide: u16, activity: f32) -> (World2, usize) {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    world.nuclear.config = config();

    let id = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[id].nuclear.isotope = nuclide;
    world.bodies[id].nuclear.activity = activity;
    (world, id)
}

/// Y01: a disabled nuclear config is a no-op.
#[test]
fn y01_disabled_config_is_a_noop() {
    let (mut world, id) = world_with(SHORT, 10.0);
    world.nuclear.config.enabled = false;

    let before_activity = world.bodies[id].nuclear.activity;
    let before_temperature = world.bodies[id].temperature;
    world.step(DT);

    assert_eq!(
        world.bodies[id].nuclear.activity.to_bits(),
        before_activity.to_bits(),
        "a disabled nuclear config must not decay anything"
    );
    assert_eq!(
        world.bodies[id].temperature.to_bits(),
        before_temperature.to_bits(),
        "a disabled nuclear config must not heat anything"
    );
}

/// Y02: the activity halves after one half-life.
#[test]
fn y02_activity_halves_after_a_half_life() {
    let mut cfg = config();
    cfg.time_scale = 60.0; // one second of simulation is one minute of nuclear time
    let (mut world, id) = world_with(SHORT, 8.0);
    world.nuclear.config = cfg;

    // 60 steps of 1/60 s is one simulation second, which with the time
    // scale is 60 nuclear seconds: exactly six half-lives of a
    // 10-nuclear-second nuclide, so the activity is 8 / 2^6.
    for _ in 0..60 {
        world.step(DT);
    }
    let activity = world.bodies[id].nuclear.activity;
    let expected = 8.0 / 64.0;
    assert!(
        (activity - expected).abs() <= expected * 1e-3,
        "after six half-lives the activity should be {expected}, got {activity}"
    );
}

/// Y03: the decay energy heats the body.
#[test]
fn y03_decay_heats_the_body() {
    let (mut world, id) = world_with(SHORT, 10.0);
    let before = world.bodies[id].temperature;
    world.step(DT);
    let after = world.bodies[id].temperature;

    assert!(
        after > before,
        "a decaying body must get hotter: {before} -> {after}"
    );
}

/// Y04: the temperature respects the thermal clamps.
#[test]
fn y04_temperature_respects_the_thermal_clamps() {
    let (mut world, id) = world_with(SHORT, 1e9);
    world.thermal.config.max_temperature = 400.0;
    world.thermal.config.min_temperature = 0.0;

    for _ in 0..120 {
        world.step(DT);
    }
    let temperature = world.bodies[id].temperature;
    assert!(
        (0.0..=400.0).contains(&temperature),
        "the temperature must stay inside the thermal clamps, got {temperature}"
    );
}

/// Y05: the activity is clamped at `max_activity`.
#[test]
fn y05_activity_is_clamped() {
    let (mut world, id) = world_with(SHORT, 1e30);
    world.nuclear.config.max_activity = 100.0;
    world.step(DT);
    assert!(
        world.bodies[id].nuclear.activity <= 100.0,
        "the activity must be clamped, got {}",
        world.bodies[id].nuclear.activity
    );
}

/// Y06: the energy budget balances: the heat plus the ionizing
/// radiation equals the energy the nuclide released.
#[test]
fn y06_the_energy_budget_balances() {
    let (mut world, id) = world_with(SHORT, 10.0);
    let activity = world.bodies[id].nuclear.activity;
    let dt_nuclear = world.nuclear.config.time_scale * DT;
    let expected = activity * 1.0 * dt_nuclear;

    world.step(DT);
    let heat = world.nuclear.scratch.heat_at(id);
    let radiation = world.nuclear.scratch.radiation_at(id);
    let total = heat + radiation;

    assert!(
        (total - expected).abs() <= expected * 1e-4,
        "the released energy must be split into heat and radiation: \
         expected {expected}, got {total} (heat {heat}, radiation {radiation})"
    );
}

/// Y07: a stable nuclide never decays.
#[test]
fn y07_a_stable_nuclide_never_decays() {
    let (mut world, id) = world_with(STABLE, 10.0);
    world.step(DT);
    assert_eq!(
        world.bodies[id].nuclear.activity.to_bits(),
        0.0f32.to_bits(),
        "a stable nuclide has no activity"
    );
}

/// Y08: a body with no nuclide takes no part.
#[test]
fn y08_a_body_without_a_nuclide_is_untouched() {
    let (mut world, id) = world_with(NO_NUCLIDE, 10.0);
    let before = world.bodies[id].nuclear.activity;
    world.step(DT);
    assert_eq!(
        world.bodies[id].nuclear.activity.to_bits(),
        before.to_bits(),
        "a body without a nuclide must be untouched"
    );
}

/// Y09: a fission consumes fuel.
#[test]
fn y09_a_fission_consumes_fuel() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1.0;
    world.nuclear.config.ambient_flux = 1.0;

    world.step(DT);
    assert!(
        world.bodies[id].nuclear.fuel < 1.0,
        "a fission must consume fuel, got {}",
        world.bodies[id].nuclear.fuel
    );
}

/// Y10: a fission releases heat.
#[test]
fn y10_a_fission_releases_heat() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1.0;
    world.nuclear.config.ambient_flux = 1.0;

    let before = world.bodies[id].temperature;
    world.step(DT);
    assert!(
        world.bodies[id].temperature > before,
        "a fission must release heat"
    );
}

/// Y11: a chain reaction grows when the fuel is above the threshold.
#[test]
fn y11_a_supercritical_chain_reaction_grows() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1.0;
    world.nuclear.config.ambient_flux = 1.0;
    world.nuclear.config.fission_rate = 10.0;
    world.nuclear.config.fuel_per_fission = 1e-9;

    // Two bodies close together: the neutrons one produces feed the
    // other, so the reaction is supercritical.
    let other = world
        .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 0.2 }, 1.0)
        .expect("valid shape");
    world.bodies[other].nuclear.isotope = FISSILE;
    world.bodies[other].nuclear.fuel = 1.0;

    let mut total_heat = 0.0f32;
    for _ in 0..30 {
        world.step(DT);
        total_heat += world.nuclear.scratch.heat_at(id);
    }

    assert!(
        total_heat > 0.0,
        "a supercritical chain reaction must produce heat"
    );
}

/// Y12: a subcritical chain reaction dies out.
#[test]
fn y12_a_subcritical_chain_reaction_dies_out() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1.0;
    // No ambient flux and no neighbours: nothing sustains the reaction.
    world.nuclear.config.ambient_flux = 0.0;

    let mut total_heat = 0.0f32;
    for _ in 0..30 {
        world.step(DT);
        total_heat += world.nuclear.scratch.heat_at(id);
    }

    assert!(
        total_heat <= 0.0,
        "a subcritical chain reaction must not produce heat, got {total_heat}"
    );
}

/// Y13: the delayed neutrons hold some of the release back.
#[test]
fn y13_delayed_neutrons_hold_the_release_back() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1.0;
    world.nuclear.config.ambient_flux = 1.0;
    world.nuclear.config.delayed_neutron_fraction = 1.0;
    world.nuclear.config.delayed_neutron_delay = 100.0;

    world.step(DT);
    assert!(
        world.bodies[id].nuclear.delayed_neutrons > 0.0,
        "the delayed neutrons must be held back"
    );
    assert!(
        world.bodies[id].nuclear.neutrons <= 0.0,
        "no prompt neutrons may be released when every neutron is delayed"
    );
}

/// Y14: the fuel is never negative.
#[test]
fn y14_the_fuel_is_never_negative() {
    let (mut world, id) = world_with(FISSILE, 0.0);
    world.bodies[id].nuclear.fuel = 1e-6;
    world.nuclear.config.ambient_flux = 1e6;
    world.nuclear.config.fission_rate = 1e6;

    for _ in 0..60 {
        world.step(DT);
        assert!(
            world.bodies[id].nuclear.fuel >= 0.0,
            "the fuel must not go negative, got {}",
            world.bodies[id].nuclear.fuel
        );
    }
}

/// Y15: the 3D mirror decays too.
#[test]
fn y15_3d_decay_heats_the_body() {
    use phieng_dynamics::body::Shape3;

    let mut world = phieng_dynamics::World3::new();
    world.gravity = Vec3::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    world.nuclear.config = config();

    let id = world
        .add_dynamic_body(
            Vec3::new(0.0, 0.0, 0.0),
            Shape3::Sphere { radius: 0.5 },
            1.0,
        )
        .expect("valid shape");
    world.bodies[id].nuclear.isotope = SHORT;
    world.bodies[id].nuclear.activity = 10.0;

    let before = world.bodies[id].temperature;
    world.step(DT);
    assert!(
        world.bodies[id].temperature > before,
        "a decaying 3D body must get hotter"
    );
}

/// Y16: the nuclear phase is deterministic.
#[test]
fn y16_the_nuclear_phase_is_deterministic() {
    let run = || -> (u32, u32) {
        let (mut world, id) = world_with(SHORT, 10.0);
        world.nuclear.config.ambient_flux = 1.0;
        for _ in 0..120 {
            world.step(DT);
        }
        (
            world.bodies[id].nuclear.activity.to_bits(),
            world.bodies[id].temperature.to_bits(),
        )
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "two identical nuclear runs must agree");
}

/// Y17: the parallel solver lands on the same trajectory.
#[test]
fn y17_parallel_lands_on_the_same_trajectory() {
    let run = |parallel: bool| -> (u32, u32) {
        let (mut world, id) = world_with(SHORT, 10.0);
        world.nuclear.config.ambient_flux = 1.0;
        world.solver.parallel = parallel;
        for _ in 0..120 {
            world.step(DT);
        }
        (
            world.bodies[id].nuclear.activity.to_bits(),
            world.bodies[id].temperature.to_bits(),
        )
    };

    let serial = run(false);
    let parallel = run(true);
    assert_eq!(
        serial, parallel,
        "the parallel solver must land on the serial trajectory"
    );
}

/// Y18: a static body acts as an infinite-capacity reservoir.
#[test]
fn y18_a_static_body_does_not_heat_up() {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    world.nuclear.config = config();

    let id = world
        .add_static_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 })
        .expect("valid shape");
    world.bodies[id].nuclear.isotope = SHORT;
    world.bodies[id].nuclear.activity = 10.0;

    let before = world.bodies[id].temperature;
    world.step(DT);
    assert_eq!(
        world.bodies[id].temperature.to_bits(),
        before.to_bits(),
        "a static body must not change temperature"
    );
}

/// Y19: the nuclide table must be usable through the configuration.
#[test]
fn y19_the_nuclide_table_is_usable() {
    let cfg = config();
    assert_eq!(cfg.isotope(NO_NUCLIDE).map(|i| i.half_life), Some(0.0));
    assert_eq!(cfg.isotope(SHORT).map(|i| i.half_life), Some(10.0));
    assert!(
        cfg.isotope(99).is_none(),
        "an unknown nuclide must be absent"
    );
    assert!(cfg.is_active(), "the configuration must be active");
}

/// Y20: the decay constant is `ln(2) / half_life`.
#[test]
fn y20_the_decay_constant_is_ln2_over_the_half_life() {
    use phieng_dynamics::nuclear::decay_constant;

    let expected = std::f32::consts::LN_2 / 10.0;
    assert_eq!(
        decay_constant(10.0).to_bits(),
        expected.to_bits(),
        "the decay constant must be ln(2) / half-life"
    );
    assert_eq!(
        decay_constant(0.0),
        0.0,
        "a stable nuclide decays at zero rate"
    );
    assert_eq!(
        decay_constant(-1.0),
        0.0,
        "a stable nuclide decays at zero rate"
    );
}

/// Y21: the ionizing radiation is deposited in the nearby bodies.
#[test]
fn y21_the_dose_is_deposited_in_the_nearby_bodies() {
    let (mut world, source) = world_with(SHORT, 10.0);
    let other = world
        .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 0.2 }, 1.0)
        .expect("valid shape");
    world.nuclear.config.dose_range = 2.0;

    world.step(DT);
    let dose = world.bodies[other].nuclear.dose;
    assert!(
        dose > 0.0,
        "the nearby body must receive a dose, got {dose}"
    );
    assert_eq!(
        world.bodies[source].nuclear.dose.to_bits(),
        0.0f32.to_bits(),
        "a body must not be irradiated by its own radiation"
    );
}

/// Y22: the accumulated dose is clamped.
#[test]
fn y22_the_dose_is_clamped() {
    let (mut world, _) = world_with(SHORT, 1e30);
    world
        .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 0.2 }, 1.0)
        .expect("valid shape");
    world.nuclear.config.dose_range = 2.0;
    world.nuclear.config.max_dose = 1.0;

    world.step(DT);
    let dose = world.bodies[1].nuclear.dose;
    assert!(dose <= 1.0, "the dose must be clamped, got {dose}");
}

/// Y23: the nuclear state round-trips through a snapshot.
#[test]
fn y23_the_nuclear_state_round_trips_through_a_snapshot() {
    use phieng_dynamics::snapshot::{
        world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, LoadMode,
    };

    let (mut world, id) = world_with(SHORT, 12.5);
    world.bodies[id].nuclear.fuel = 0.25;
    world.bodies[id].nuclear.neutrons = 3.0;
    world.bodies[id].nuclear.dose = 1.5;
    world.nuclear.config.time_scale = 7.0;
    world.nuclear.config.ambient_flux = 0.25;

    let json = world2_to_json(&world, LoadMode::Exact).expect("serializable");
    let restored = world2_from_json(&json, LoadMode::Exact).expect("loadable");
    assert_eq!(
        restored.bodies[id].nuclear.isotope,
        world.bodies[id].nuclear.isotope
    );
    assert_eq!(
        restored.bodies[id].nuclear.activity.to_bits(),
        world.bodies[id].nuclear.activity.to_bits()
    );
    assert_eq!(
        restored.bodies[id].nuclear.fuel.to_bits(),
        world.bodies[id].nuclear.fuel.to_bits()
    );
    assert_eq!(
        restored.bodies[id].nuclear.dose.to_bits(),
        world.bodies[id].nuclear.dose.to_bits()
    );
    assert_eq!(restored.nuclear.config.time_scale, 7.0);
    assert_eq!(restored.nuclear.config.ambient_flux, 0.25);
    assert_eq!(
        restored.nuclear.config.isotopes.len(),
        world.nuclear.config.isotopes.len()
    );

    let binary = world2_to_binary(&world, LoadMode::Exact).expect("serializable");
    let restored = world2_from_binary(&binary, LoadMode::Exact).expect("loadable");
    assert_eq!(
        restored.bodies[id].nuclear.activity.to_bits(),
        world.bodies[id].nuclear.activity.to_bits()
    );
    assert_eq!(
        restored.bodies[id].nuclear.dose.to_bits(),
        world.bodies[id].nuclear.dose.to_bits()
    );
    assert_eq!(restored.nuclear.config.time_scale, 7.0);
    assert_eq!(
        restored.nuclear.config.isotopes.len(),
        world.nuclear.config.isotopes.len()
    );
}
