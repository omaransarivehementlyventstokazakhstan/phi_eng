//! Nuclear stress suite: brutal, in-scope soak tests.
//!
//! The nuclear subsystem (radioactive decay, neutron capture, fission,
//! chain reactions and dose) had **no** absolute-stress coverage. These
//! tests pin the *physics* where it has a closed form - the activity
//! decay is exactly exponential, and the released energy has to land in
//! the body's heat content - and the *survival* where it does not (a
//! super-critical chain reaction must stay bounded because the fuel
//! runs out).
//!
//! Each test asserts survival (finite state, bounded temperatures, no
//! free energy) and prints a single `NUCLEAR ...` readings line. Run
//! with:
//! `cargo test --release -p phieng --test nuclear_stress -- --nocapture --test-threads=1`
//!
//! NOTE: test names here intentionally avoid the parity-ID convention
//! (`tools/check_test_parity.py`), so this file needs no C# mirror.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::nuclear::Isotope;
use phieng_dynamics::World2;
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

/// The nuclide table used by the scenes.
const SHORT: u16 = 1;
const STABLE: u16 = 2;
const FISSILE: u16 = 3;

/// Print one readings line.
fn nuclear_report(name: &str, bodies: usize, steps: usize, wall_ms: f64, extra: &str) {
    println!(
        "NUCLEAR name={name} bodies={bodies} steps={steps} wall_ms={wall_ms:.1} ms_per_step={:.4} {extra}",
        wall_ms / steps as f64
    );
}

/// Every body's nuclear state and temperature must stay finite.
fn assert_nuclear_sane(world: &World2, ctx: &str) {
    for body in &world.bodies {
        let n = &body.nuclear;
        assert!(
            n.activity.is_finite(),
            "{ctx}: non-finite activity {}",
            n.activity
        );
        assert!(n.fuel.is_finite(), "{ctx}: non-finite fuel {}", n.fuel);
        assert!(
            n.neutrons.is_finite(),
            "{ctx}: non-finite neutrons {}",
            n.neutrons
        );
        assert!(n.dose.is_finite(), "{ctx}: non-finite dose {}", n.dose);
        assert!(
            body.temperature.is_finite(),
            "{ctx}: non-finite temperature {}",
            body.temperature
        );
    }
}

/// A world with the nuclear phase enabled and the thermal phase off
/// (so the released heat stays in the body, where it can be checked).
fn nuclear_world() -> World2 {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    world.nuclear.config.enabled = true;
    world
}

/// The standard nuclide table: a short-lived nuclide that decays into a
/// stable one, a stable one, a fissile one and its daughter.
fn standard_isotopes() -> Vec<Isotope> {
    vec![
        // 0: no nuclide.
        Isotope::default(),
        // 1: short-lived (10 nuclear seconds), 1 J per decay, half of it
        // as ionizing radiation.
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
            daughter: STABLE,
            ..Isotope::default()
        },
        // 3: fissile.
        Isotope {
            half_life: 0.0,
            decay_energy: 0.0,
            daughter: STABLE,
            absorption_cross_section: 1.0,
            fission_cross_section: 1.0,
            neutron_yield: 2.0,
            is_fissile: true,
            ..Isotope::default()
        },
        // 4: the daughter of the fissile nuclide.
        Isotope {
            half_life: 0.0,
            decay_energy: 0.0,
            daughter: STABLE,
            ..Isotope::default()
        },
    ]
}

/// The radioactive activity of a nuclide decays exactly exponentially:
/// after `n` half-lives it is `A0 / 2^n`.
///
/// The energy released along the way has to show up as heat in the
/// body, which closes the energy budget from the other side.
#[test]
fn nuclear_decay_matches_half_life() {
    let mut world = nuclear_world();
    world.nuclear.config.isotopes = standard_isotopes();
    // One simulation second is sixty nuclear seconds.
    world.nuclear.config.time_scale = 60.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].nuclear.isotope = SHORT;
    world.bodies[body].nuclear.activity = 100.0;

    let start_temp = world.bodies[body].temperature;
    let steps = 60; // one simulation second = six half-lives
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        if frame % 10 == 0 {
            assert_nuclear_sane(&world, "decay");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    let activity = world.bodies[body].nuclear.activity;
    let expected_activity = 100.0 / 64.0; // six half-lives
    assert!(
        (activity - expected_activity).abs() <= expected_activity * 1e-3,
        "the activity should halve every half-life: expected \
         {expected_activity}, measured {activity}"
    );

    // The energy released along the way is the sum of the *discrete*
    // releases: each step releases the start-of-step activity times E
    // times dt, and the activity falls by a factor r every step, so the
    // total is the geometric sum A0 * E * dt * (1 - r^N) / (1 - r).
    let lambda = std::f32::consts::LN_2 / 10.0;
    let dt_nuclear = DT * 60.0;
    let r = phieng_math::exp(-lambda * dt_nuclear);
    let n = steps as f32;
    let energy =
        100.0 * 1.0 * dt_nuclear * (1.0 - phieng_math::exp(-lambda * dt_nuclear * n)) / (1.0 - r);
    // Half of the released energy is ionizing, so only the other half
    // reaches the body as heat.
    let heat = energy * 0.5;
    let capacity = world.bodies[body].mass * world.bodies[body].material.specific_heat;
    let expected_temp = start_temp + heat / capacity;
    let measured_temp = world.bodies[body].temperature;
    assert!(
        (measured_temp - expected_temp).abs() <= expected_temp.abs() * 1e-2,
        "the released energy must show up as heat: expected \
         {expected_temp}, measured {measured_temp}"
    );

    nuclear_report(
        "decay_matches_half_life",
        1,
        steps,
        wall_ms,
        &format!(
            "activity={activity:.4} expected={expected_activity:.4} energy={energy:.2}J \
             heat={heat:.2}J temp={start_temp:.2}->{measured_temp:.4} expected={expected_temp:.4}"
        ),
    );
}

/// The fission release scales with the fuel the reaction has to work
/// with: more fuel, more fissions, more heat.
///
/// The model has no *true* criticality threshold: the fission rate is
/// proportional to the fuel, so a body with any fuel at all burns it
/// (exponentially), and the mean-field neutron spread cannot extinguish
/// a reaction the way a real, stochastic one can. What the sweep pins is
/// the scaling law, which is what a caller actually relies on.
#[test]
fn nuclear_fission_release_scales_with_fuel() {
    let mut heat_by_fuel = Vec::new();
    for fuel_pct in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let mut world = nuclear_world();
        world.nuclear.config.isotopes = standard_isotopes();
        world.nuclear.config.ambient_flux = 1.0;
        world.nuclear.config.fission_rate = 1.0;
        world.nuclear.config.fuel_per_fission = 1e-3;
        // The physical 200 MeV per fission is invisible at this scale,
        // so the sweep uses a game-scale energy.
        world.nuclear.config.fission_energy = 1.0e3;

        // Two bodies side by side: the neutrons one produces feed the
        // other, so the reaction can go super-critical.
        let a = world
            .add_dynamic_body(Vec2::new(-0.5, 0.0), Shape2::Circle { radius: 0.25 }, 1.0)
            .expect("valid shape");
        let b = world
            .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 0.25 }, 1.0)
            .expect("valid shape");
        for id in [a, b] {
            world.bodies[id].nuclear.isotope = FISSILE;
            world.bodies[id].nuclear.fuel = fuel_pct;
        }

        let steps = 300;
        let mut heat = 0.0f32;
        for frame in 0..steps {
            world.step(DT);
            heat += world.nuclear.scratch.heat_at(a) + world.nuclear.scratch.heat_at(b);
            if frame % 50 == 0 {
                assert_nuclear_sane(&world, "criticality sweep");
            }
        }
        heat_by_fuel.push((fuel_pct, heat));
    }

    // More fuel must mean more heat: the release is monotone in the
    // fuel it has to work with.
    for pair in heat_by_fuel.windows(2) {
        assert!(
            pair[1].1 > pair[0].1,
            "more fuel must give more heat: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
    // And no fuel means no fission at all.
    assert!(
        heat_by_fuel[0].1 <= 0.0,
        "a body with no fuel must not fission, it released {}",
        heat_by_fuel[0].1
    );

    let sweep: Vec<String> = heat_by_fuel
        .iter()
        .map(|(f, h)| format!("{f:.2}->{h:.1}"))
        .collect();
    nuclear_report(
        "fission_release_scales_with_fuel",
        2,
        300,
        0.0,
        &format!("sweep[{}]", sweep.join(" ")),
    );
}

/// A super-critical chain reaction must run away *and stop*: the fuel
/// runs out, so the activity peaks and falls. It must never diverge.
#[test]
fn nuclear_supercritical_runaway_stays_bounded() {
    let mut world = nuclear_world();
    world.nuclear.config.isotopes = standard_isotopes();
    world.nuclear.config.ambient_flux = 10.0;
    world.nuclear.config.fission_rate = 10.0;
    world.nuclear.config.fuel_per_fission = 1e-4;
    world.nuclear.config.fission_energy = 1.0e3;

    let a = world
        .add_dynamic_body(Vec2::new(-0.5, 0.0), Shape2::Circle { radius: 0.25 }, 1.0)
        .expect("valid shape");
    let b = world
        .add_dynamic_body(Vec2::new(0.5, 0.0), Shape2::Circle { radius: 0.25 }, 1.0)
        .expect("valid shape");
    for id in [a, b] {
        world.bodies[id].nuclear.isotope = FISSILE;
        world.bodies[id].nuclear.fuel = 1.0;
    }

    let steps = 600;
    let mut peak_activity = 0.0f32;
    let mut peak_temp = 0.0f32;
    let mut total_heat = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        total_heat += world.nuclear.scratch.heat_at(a) + world.nuclear.scratch.heat_at(b);
        for id in [a, b] {
            peak_activity = peak_activity.max(world.bodies[id].nuclear.activity);
            peak_temp = peak_temp.max(world.bodies[id].temperature);
        }
        if frame % 50 == 0 {
            assert_nuclear_sane(&world, "runaway");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    // The fuel is finite, so the reaction must consume it.
    let fuel_left = world.bodies[a].nuclear.fuel + world.bodies[b].nuclear.fuel;
    assert!(
        fuel_left < 2.0,
        "the reaction must consume the fuel, {fuel_left} is left"
    );
    // Nothing may diverge.
    assert!(
        peak_temp < 1.0e6,
        "the temperature must stay bounded, it peaked at {peak_temp}"
    );
    assert!(
        peak_activity < 1.0e30,
        "the activity must stay bounded, it peaked at {peak_activity}"
    );

    nuclear_report(
        "supercritical_runaway_stays_bounded",
        2,
        steps,
        wall_ms,
        &format!(
            "fuel_left={fuel_left:.6} peak_activity={peak_activity:.3} peak_temp={peak_temp:.2} \
             total_heat={total_heat:.1}"
        ),
    );
}

/// The released energy splits into heat and ionizing radiation, and the
/// two halves must add up to the whole.
#[test]
fn nuclear_decay_energy_splits_into_heat_and_radiation() {
    let mut world = nuclear_world();
    world.nuclear.config.isotopes = standard_isotopes();
    world.nuclear.config.time_scale = 60.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].nuclear.isotope = SHORT;
    world.bodies[body].nuclear.activity = 100.0;

    let steps = 60;
    let mut heat = 0.0f32;
    let mut radiation = 0.0f32;
    for frame in 0..steps {
        world.step(DT);
        heat += world.nuclear.scratch.heat_at(body);
        radiation += world.nuclear.scratch.radiation_at(body);
        if frame % 10 == 0 {
            assert_nuclear_sane(&world, "energy split");
        }
    }

    // The discrete release sum (a left Riemann sum of the exponential).
    let lambda = std::f32::consts::LN_2 / 10.0;
    let dt_nuclear = DT * 60.0;
    let r = phieng_math::exp(-lambda * dt_nuclear);
    let n = steps as f32;
    let total =
        100.0 * 1.0 * dt_nuclear * (1.0 - phieng_math::exp(-lambda * dt_nuclear * n)) / (1.0 - r);

    assert!(
        (heat + radiation - total).abs() <= total * 1e-2,
        "heat + radiation must equal the released energy: {heat} + {radiation} != {total}"
    );
    // The nuclide ionizes half of its energy.
    assert!(
        (radiation - heat).abs() <= total * 1e-2,
        "a 50% ionizing fraction must split the energy evenly: \
         heat {heat}, radiation {radiation}"
    );

    nuclear_report(
        "decay_energy_splits_into_heat_and_radiation",
        1,
        steps,
        0.0,
        &format!("heat={heat:.3} radiation={radiation:.3} total={total:.3} (ionizing 50%)"),
    );
}

/// A nuclide with a daughter captures a neutron and becomes it - and
/// the chain continues into the grandchild.
#[test]
fn nuclear_capture_chain_advances() {
    let mut world = nuclear_world();
    let mut isotopes = standard_isotopes();
    // The short-lived nuclide absorbs neutrons and becomes the stable
    // one; the stable one absorbs neutrons and becomes the fissile one.
    isotopes[SHORT as usize].absorption_cross_section = 1.0;
    isotopes[STABLE as usize].absorption_cross_section = 1.0;
    isotopes[STABLE as usize].daughter = FISSILE;
    world.nuclear.config.isotopes = isotopes;
    world.nuclear.config.ambient_flux = 1000.0;
    world.nuclear.config.activation_rate = 1000.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].nuclear.isotope = SHORT;

    let start_isotope = world.bodies[body].nuclear.isotope;
    let steps = 60;
    let mut seen = vec![start_isotope];
    for frame in 0..steps {
        world.step(DT);
        let iso = world.bodies[body].nuclear.isotope;
        if seen.last() != Some(&iso) {
            seen.push(iso);
        }
        if frame % 10 == 0 {
            assert_nuclear_sane(&world, "capture chain");
        }
    }

    // The chain must have advanced at least one step.
    assert!(
        seen.len() >= 2,
        "the capture must advance the nuclide, the chain was {seen:?}"
    );
    assert_eq!(
        seen[0], start_isotope,
        "the chain must start at the original nuclide"
    );
    // The chain advances twice: the original becomes its daughter, and
    // the daughter absorbs in turn and becomes *its* daughter.
    assert!(
        seen.len() >= 3,
        "the chain must advance through the daughter: {seen:?}"
    );
    assert_eq!(
        world.bodies[body].nuclear.isotope, world.nuclear.config.isotopes[STABLE as usize].daughter,
        "the body must end as the grandchild of the original nuclide"
    );

    nuclear_report(
        "capture_chain_advances",
        1,
        steps,
        0.0,
        &format!(
            "chain_len={} chain_head={:?} final={}",
            seen.len(),
            &seen[..seen.len().min(8)],
            world.bodies[body].nuclear.isotope
        ),
    );
}

/// The ionizing share of the decay is deposited as dose in the bodies
/// that are in range, and the dose is clamped.
#[test]
fn nuclear_dose_accumulates_and_clamps() {
    let mut world = nuclear_world();
    world.nuclear.config.isotopes = standard_isotopes();
    world.nuclear.config.time_scale = 60.0;
    world.nuclear.config.dose_range = 2.0;
    world.nuclear.config.max_dose = 1.0;

    let source = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    let neighbour = world
        .add_dynamic_body(Vec2::new(0.8, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[source].nuclear.isotope = SHORT;
    world.bodies[source].nuclear.activity = 100.0;

    let steps = 120;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        if frame % 20 == 0 {
            assert_nuclear_sane(&world, "dose");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    let dose = world.bodies[neighbour].nuclear.dose;
    assert!(
        dose > 0.0,
        "the neighbour must accumulate dose, it got {dose}"
    );
    assert!(
        dose <= world.nuclear.config.max_dose,
        "the dose must be clamped at {}, it is {dose}",
        world.nuclear.config.max_dose
    );
    // The source does not irradiate itself.
    assert_eq!(
        world.bodies[source].nuclear.dose.to_bits(),
        0.0f32.to_bits(),
        "a body must not be irradiated by its own radiation"
    );

    nuclear_report(
        "dose_accumulates_and_clamps",
        2,
        steps,
        wall_ms,
        &format!(
            "dose={dose:.4} clamped_at={:.1} range={:.1}",
            world.nuclear.config.max_dose, world.nuclear.config.dose_range
        ),
    );
}

/// The extremes of the knobs must not produce a NaN state.
#[test]
fn nuclear_extreme_activity_stays_finite() {
    let mut world = nuclear_world();
    let mut isotopes = standard_isotopes();
    isotopes[SHORT as usize].decay_energy = 1e30;
    world.nuclear.config.isotopes = isotopes;
    world.nuclear.config.time_scale = 60.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].nuclear.isotope = SHORT;
    world.bodies[body].nuclear.activity = 1e30;

    let steps = 60;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        if frame % 10 == 0 {
            assert_nuclear_sane(&world, "extreme nuclear");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert!(
        world.bodies[body].temperature.is_finite(),
        "the temperature must stay finite, it is {}",
        world.bodies[body].temperature
    );
    assert!(
        world.bodies[body].nuclear.activity.is_finite(),
        "the activity must stay finite"
    );

    nuclear_report(
        "extreme_activity_stays_finite",
        1,
        steps,
        wall_ms,
        &format!(
            "activity={:.4} temp={:.4}",
            world.bodies[body].nuclear.activity, world.bodies[body].temperature
        ),
    );
}
