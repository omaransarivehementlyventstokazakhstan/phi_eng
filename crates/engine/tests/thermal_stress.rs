//! Thermal & radiation stress suite: brutal, in-scope soak tests.
//!
//! The thermal subsystem (conduction, radiation, convection, thermal
//! expansion) had **no** absolute-stress coverage at all. These tests pin
//! the *physics* where it has a closed form - the convective and the
//! conductive exchange are exponentials - and the *survival* where it
//! does not (the radiative exchange is stiff at game scale: a 5000 K body
//! radiates ~10^8 W, which would empty it in a single step without the
//! rate clamp).
//!
//! Each test asserts survival (finite state, bounded temperatures, no
//! energy invented) and prints a single `THERMAL ...` readings line.
//! Run with:
//! `cargo test --release -p phieng --test thermal_stress -- --nocapture --test-threads=1`
//!
//! NOTE: test names here intentionally avoid the parity-ID convention
//! (`tools/check_test_parity.py`), so this file needs no C# mirror.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::World2;
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

/// A rate cap high enough that the clamp never masks the physics.
/// The default (100 K/s) dominates every fast exchange.
const NO_RATE_CAP: f32 = 1.0e9;

/// Print one readings line.
fn thermal_report(name: &str, bodies: usize, steps: usize, wall_ms: f64, extra: &str) {
    println!(
        "THERMAL name={name} bodies={bodies} steps={steps} wall_ms={wall_ms:.1} ms_per_step={:.4} {extra}",
        wall_ms / steps as f64
    );
}

/// Every temperature in the world must stay finite and inside the
/// configured clamps.
fn assert_thermal_sane(world: &World2, ctx: &str) {
    let (lo, hi) = (
        world.thermal.config.min_temperature,
        world.thermal.config.max_temperature,
    );
    for body in &world.bodies {
        let t = body.temperature;
        assert!(t.is_finite(), "{ctx}: non-finite temperature {t}");
        assert!(
            t >= lo - 1e-3 && t <= hi + 1e-3,
            "{ctx}: temperature {t} outside the clamps [{lo}, {hi}]"
        );
    }
}

/// The total heat content of the world, in joules (per unit mass).
fn total_heat(world: &World2) -> f64 {
    world
        .bodies
        .iter()
        .map(|b| b.mass as f64 * b.material.specific_heat as f64 * b.temperature as f64)
        .sum()
}

/// A world with the thermal phase enabled and everything else quiet.
fn thermal_world() -> World2 {
    let mut world = World2::new();
    world.gravity = Vec2::ZERO;
    world.linear_sleep_threshold = 0.0;
    world.angular_sleep_threshold = 0.0;
    world.thermal.config.enabled = true;
    world.thermal.config.max_temperature_rate = NO_RATE_CAP;
    world
}

/// The least-squares slope of `ln|y|` against `x`, for a series that
/// should be an exponential.
fn log_slope(xs: &[f32], ys: &[f32]) -> f64 {
    let pts: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(_, y)| y.abs() > 1e-6)
        .map(|(x, y)| (*x as f64, (*y as f64).abs().ln()))
        .collect();
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    num / den
}

/// The thermal exchange between two touching bodies is an exponential
/// decay of the temperature *difference*, and it moves no heat out of
/// the pair.
///
/// The cold body is deliberately heavy: it acts as a near-constant
/// reservoir (its heat capacity is 1000x the hot body's), so the hot
/// body's temperature decays exponentially toward it and the pair's
/// total heat is conserved to within the reservoir's own exchange with
/// the floor.
#[test]
fn thermal_conduction_exchange_is_exponential() {
    let mut world = thermal_world();
    world.thermal.config.conduction_enabled = true;
    world.thermal.config.radiation_enabled = false;
    world.thermal.config.convection_enabled = false;

    // A static floor at the reservoir's temperature: it exchanges nothing
    // with the reservoir.
    world
        .add_static_body(
            Vec2::new(0.0, -1.5),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    world.bodies[0].temperature = 300.0;

    // The cold, heavy reservoir.
    let cold = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(1.0, 0.5),
            },
            1000.0,
        )
        .expect("valid shape");
    // The hot, light body resting on it.
    let hot = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[cold].temperature = 300.0;
    world.bodies[hot].temperature = 400.0;

    let heat0 = total_heat(&world);
    let steps = 600;
    let mut times = Vec::with_capacity(steps);
    let mut gaps = Vec::with_capacity(steps);
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        times.push((frame + 1) as f32 * DT);
        gaps.push(world.bodies[hot].temperature - world.bodies[cold].temperature);
        if frame % 100 == 0 {
            assert_thermal_sane(&world, "conduction");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    // The gap must shrink to a small fraction of its start.
    let first = gaps[0].abs();
    let last = gaps[gaps.len() - 1].abs();
    assert!(
        last < first * 0.25,
        "the temperature gap should collapse: {first} -> {last}"
    );

    // An exponential decay is a straight line in log space.
    let half = steps / 2;
    let slope_a = log_slope(&times[..half], &gaps[..half]);
    let slope_b = log_slope(&times[half..], &gaps[half..]);
    assert!(
        (slope_a - slope_b).abs() <= slope_a.abs() * 0.35 + 1e-3,
        "the decay should be exponential: slopes {slope_a} vs {slope_b}"
    );
    let tau = -1.0 / slope_a;

    // The pair exchanges heat with itself: no energy is created.
    let heat1 = total_heat(&world);
    assert!(
        (heat1 - heat0).abs() < 2e-3 * heat0.abs().max(1.0),
        "the exchange must conserve the pair's heat: {heat0} -> {heat1}"
    );

    thermal_report(
        "conduction_exchange_is_exponential",
        3,
        steps,
        wall_ms,
        &format!(
            "gap={first:.2}->{last:.4} tau={tau:.3}s slope={slope_a:.4} heat={heat0:.1}->{heat1:.1}"
        ),
    );
}

/// An isolated body cools by Newtonian convection, which is an
/// exponential approach to the ambient with a known time constant.
#[test]
fn thermal_convective_cooling_matches_analytic() {
    let mut world = thermal_world();
    world.thermal.config.convection_enabled = true;
    world.thermal.config.radiation_enabled = false;
    world.thermal.config.convection_coefficient = 10.0;
    world.thermal.config.ambient_temperature = 300.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].temperature = 600.0;

    // The analytic constants come from the body itself, not from a
    // re-implementation of the model.
    let area = world.bodies[body]
        .shape
        .as_ref()
        .expect("a shape")
        .surface_area();
    let mass = world.bodies[body].mass;
    let capacity = world.bodies[body].material.specific_heat;
    let h = world.thermal.config.convection_coefficient;
    let ambient = world.thermal.config.ambient_temperature;
    let tau = mass * capacity / (h * area);

    let steps = 600;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        if frame % 100 == 0 {
            assert_thermal_sane(&world, "convection");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    let t = steps as f32 * DT;
    let expected = ambient + (600.0 - ambient) * phieng_math::exp(-t / tau);
    let measured = world.bodies[body].temperature;

    assert!(
        (measured - expected).abs() <= expected.abs() * 0.02 + 0.5,
        "convective cooling should follow the analytic curve: \
         expected {expected}, measured {measured} (tau={tau} s)"
    );

    thermal_report(
        "convective_cooling_matches_analytic",
        1,
        steps,
        wall_ms,
        &format!(
            "tau={tau:.3}s expected={expected:.3} measured={measured:.3} err={:.4}",
            (measured - expected).abs()
        ),
    );
}

/// The radiative exchange between two bodies is antisymmetric, and the
/// pair never gains heat from nowhere.
///
/// The scene isolates the *pair* exchange: the ambient sits at the cold
/// body's temperature, so the cold body's exchange with the ambient is
/// exactly zero and its whole temperature change is the pair exchange.
///
/// The antisymmetry is checked by symmetry rather than by arithmetic:
/// the same two bodies are exchanged left-for-right, and the cold body
/// must gain exactly the same heat either way. That only holds for the
/// symmetric view-factor model - the legacy one is order-dependent,
/// which is the whole reason the symmetric model exists.
#[test]
fn thermal_radiative_exchange_is_antisymmetric() {
    let run = |hot_left: bool| -> (f32, f32) {
        let mut world = thermal_world();
        world.thermal.config.radiation_enabled = true;
        world.thermal.config.conduction_enabled = false;
        world.thermal.config.convection_enabled = false;
        world.thermal.config.ambient_temperature = 320.0;
        world.thermal.config.view_factor_model =
            phieng_dynamics::thermal::ViewFactorModel::Symmetric;

        let (hot_x, cold_x) = if hot_left { (-1.0, 1.0) } else { (1.0, -1.0) };
        let hot = world
            .add_dynamic_body(Vec2::new(hot_x, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        let cold = world
            .add_dynamic_body(Vec2::new(cold_x, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[hot].temperature = 900.0;
        world.bodies[cold].temperature = 320.0;

        let heat0 = total_heat(&world);
        world.step(DT);
        let heat1 = total_heat(&world);
        assert!(
            heat1 <= heat0 + 1e-2,
            "the exchange must never create heat: {heat0} -> {heat1}"
        );
        (
            900.0 - world.bodies[hot].temperature,
            world.bodies[cold].temperature - 320.0,
        )
    };

    let (hot_loss_a, cold_gain_a) = run(true);
    let (hot_loss_b, cold_gain_b) = run(false);

    // The cold body warms: the pair exchange reaches it.
    assert!(
        cold_gain_a > 0.0,
        "the cold body must gain heat from the hot one, it gained {cold_gain_a}"
    );
    // The hot body loses more than the cold one gains: it also radiates
    // to the ambient.
    assert!(
        hot_loss_a > cold_gain_a,
        "the hot body must also radiate to the ambient: lost {hot_loss_a}, \
         the cold body gained {cold_gain_a}"
    );
    // Antisymmetry: swapping the bodies must not change the exchange.
    assert!(
        (cold_gain_a - cold_gain_b).abs() <= cold_gain_a.abs() * 1e-3,
        "the exchange must be antisymmetric: {cold_gain_a} vs {cold_gain_b}"
    );
    assert!(
        (hot_loss_a - hot_loss_b).abs() <= hot_loss_a.abs() * 1e-3,
        "the exchange must be antisymmetric: {hot_loss_a} vs {hot_loss_b}"
    );

    thermal_report(
        "radiative_exchange_is_antisymmetric",
        2,
        1,
        0.0,
        &format!(
            "hot_left cold_gain={cold_gain_a:.6} hot_loss={hot_loss_a:.6} | \
             hot_right cold_gain={cold_gain_b:.6} hot_loss={hot_loss_b:.6}"
        ),
    );
}

/// The radiative exchange is *stiff*: a very hot body radiates ~10^8 W
/// and would empty itself in a single step. The rate clamp is what
/// keeps that survivable.
#[test]
fn thermal_radiation_is_stiff_but_bounded() {
    let mut world = thermal_world();
    world.thermal.config.radiation_enabled = true;
    world.thermal.config.conduction_enabled = false;
    world.thermal.config.convection_enabled = false;
    world.thermal.config.ambient_temperature = 1.0;
    // The *default* rate cap - the point of the test is that it holds.
    world.thermal.config.max_temperature_rate = 100.0;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    world.bodies[body].temperature = 5000.0;

    let steps = 600;
    let mut peak = 5000.0f32;
    let mut previous = 5000.0f32;
    let mut monotone = true;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        let t = world.bodies[body].temperature;
        if t > previous {
            monotone = false;
        }
        previous = t;
        peak = peak.max(t);
        if frame % 50 == 0 {
            assert_thermal_sane(&world, "stiff radiation");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert!(monotone, "a radiating body must never heat up");
    assert!(
        world.bodies[body].temperature < 5000.0,
        "the body must have cooled"
    );
    // The clamp allows 100 K/s, so 10 s of simulation can shed at most
    // 1000 K.
    assert!(
        5000.0 - world.bodies[body].temperature <= 100.0 * (steps as f32 * DT) + 1.0,
        "the rate clamp must bound the loss"
    );

    thermal_report(
        "radiation_is_stiff_but_bounded",
        1,
        steps,
        wall_ms,
        &format!(
            "final_t={:.2} lost={:.2} cap={:.0}K peak={peak:.1}",
            world.bodies[body].temperature,
            5000.0 - world.bodies[body].temperature,
            100.0 * steps as f32 * DT
        ),
    );
}

/// Thermal expansion scales the shape by `1 + alpha * (T - T_ref)`, and
/// the factor is applied *relative to the authored shape*, so it never
/// compounds.
#[test]
fn thermal_expansion_matches_formula() {
    let mut world = thermal_world();
    world.thermal.config.radiation_enabled = false;
    world.thermal.config.convection_enabled = false;
    world.thermal.config.expansion_coefficient = 1.0e-3;
    world.thermal.config.expansion_reference_temperature = 293.15;

    let body = world
        .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 1.0 }, 1.0)
        .expect("valid shape");
    world.bodies[body].temperature = 1000.0;

    let radius0 = 1.0f32;
    let expected = radius0 * (1.0 + 1.0e-3 * (1000.0 - 293.15));

    let steps = 200;
    let start = Instant::now();
    for _ in 0..steps {
        world.step(DT);
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    let radius = match world.bodies[body].shape.as_ref().expect("a shape") {
        Shape2::Circle { radius } => *radius,
        _ => unreachable!("the shape is a circle"),
    };

    assert!(
        (radius - expected).abs() <= expected * 1e-3,
        "the shape should scale by 1 + alpha * (T - T_ref): \
         expected {expected}, measured {radius}"
    );
    // Not compounded: 200 steps must not have produced
    // radius0 * (1 + alpha * dT)^200. The compounded value overflows f32
    // long before 200 steps, so it is computed in f64.
    let compounded: f64 = 1.0f64 * (1.0f64 + 1.0e-3 * (1000.0 - 293.15)).powi(200);
    assert!(
        (radius as f64 - compounded).abs() > compounded * 0.1,
        "the expansion must not compound: radius {radius} vs compounded {compounded}"
    );

    thermal_report(
        "expansion_matches_formula",
        1,
        steps,
        wall_ms,
        &format!("radius={radius:.6} expected={expected:.6} compounded_would_be={compounded:.3e}"),
    );
}

/// The symmetric view-factor model agrees with the legacy one for two
/// identical bodies, and differs for two different ones.
///
/// The scene isolates the *pair* exchange: the ambient sits at the cold
/// body's temperature, so the cold body's exchange with the ambient is
/// exactly zero and its whole temperature change is the pair exchange.
#[test]
fn thermal_symmetric_view_factor_matches_legacy() {
    let run = |symmetric: bool, r_cold: f32| -> f32 {
        let mut world = thermal_world();
        world.thermal.config.radiation_enabled = true;
        world.thermal.config.conduction_enabled = false;
        world.thermal.config.convection_enabled = false;
        world.thermal.config.ambient_temperature = 320.0;
        if symmetric {
            world.thermal.config.view_factor_model =
                phieng_dynamics::thermal::ViewFactorModel::Symmetric;
        }
        let hot = world
            .add_dynamic_body(Vec2::new(-1.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        let cold = world
            .add_dynamic_body(Vec2::new(1.0, 0.0), Shape2::Circle { radius: r_cold }, 1.0)
            .expect("valid shape");
        world.bodies[hot].temperature = 900.0;
        world.bodies[cold].temperature = 320.0;
        let before = world.bodies[cold].temperature;
        world.step(DT);
        world.bodies[cold].temperature - before
    };

    // Equal bodies: the two models must agree.
    let legacy_eq = run(false, 0.5);
    let sym_eq = run(true, 0.5);
    assert!(
        (sym_eq - legacy_eq).abs() <= legacy_eq.abs() * 1e-2 + 1e-6,
        "the symmetric model must match the legacy one for equal bodies: \
         {legacy_eq} vs {sym_eq}"
    );

    // Different bodies: they must differ, or the model does nothing.
    let legacy_neq = run(false, 0.25);
    let sym_neq = run(true, 0.25);
    assert!(
        (sym_neq - legacy_neq).abs() > legacy_neq.abs() * 1e-2,
        "the symmetric model must change the exchange for unequal bodies: \
         legacy {legacy_neq}, symmetric {sym_neq}"
    );

    thermal_report(
        "symmetric_view_factor_matches_legacy",
        2,
        1,
        0.0,
        &format!(
            "equal_body legacy={legacy_eq:.6} symmetric={sym_eq:.6} | \
             unequal legacy={legacy_neq:.6} symmetric={sym_neq:.6}"
        ),
    );
}

/// A cold sky and a warm ground settle the body somewhere different
/// from a uniform ambient.
#[test]
fn thermal_sky_ground_split_changes_equilibrium() {
    let run = |sky: f32, ground: f32| -> f32 {
        let mut world = thermal_world();
        world.thermal.config.radiation_enabled = true;
        world.thermal.config.conduction_enabled = false;
        world.thermal.config.convection_enabled = false;
        world.thermal.config.ambient_temperature = 300.0;
        world.thermal.config.sky_temperature = sky;
        world.thermal.config.ground_temperature = ground;
        let body = world
            .add_dynamic_body(Vec2::new(0.0, 0.0), Shape2::Circle { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[body].temperature = 300.0;
        for _ in 0..600 {
            world.step(DT);
        }
        world.bodies[body].temperature
    };

    let uniform = run(0.0, 0.0);
    let split = run(100.0, 400.0);
    let cold_sky = run(100.0, 100.0);

    assert!(
        (split - uniform).abs() > 1e-3,
        "a sky/ground split must change the equilibrium: \
         uniform {uniform}, split {split}"
    );
    // The split equilibrium sits between the sky and the ground
    // (T^4 is the average of the two), so a warm ground over a cold
    // sky warms the body past a uniform ambient.
    assert!(
        split > uniform,
        "a warm ground over a cold sky must warm the body past a uniform \
         ambient: uniform {uniform}, split {split}"
    );
    assert!(
        cold_sky < uniform,
        "a uniformly cold sky must cool the body: cold sky {cold_sky}, \
         uniform {uniform}"
    );

    thermal_report(
        "sky_ground_split_changes_equilibrium",
        1,
        600,
        0.0,
        &format!("uniform={uniform:.4} split(100/400)={split:.4} cold_sky(100/100)={cold_sky:.4}"),
    );
}

/// A hot body stacked on a cold one exchanges heat until they
/// converge, and the pair's total heat is unchanged.
///
/// The cold body is heavy (1000x the heat capacity) so it acts as a
/// near-constant reservoir, and the floor sits at its temperature so the
/// reservoir exchanges nothing with the ground.
#[test]
fn thermal_shock_converges_without_explosion() {
    let mut world = thermal_world();
    world.thermal.config.conduction_enabled = true;
    world.thermal.config.radiation_enabled = false;
    world.thermal.config.convection_enabled = false;

    world
        .add_static_body(
            Vec2::new(0.0, -1.5),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    world.bodies[0].temperature = 300.0;

    let cold = world
        .add_dynamic_body(
            Vec2::new(0.0, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(1.0, 0.5),
            },
            1000.0,
        )
        .expect("valid shape");
    let hot = world
        .add_dynamic_body(
            Vec2::new(0.0, 1.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[cold].temperature = 300.0;
    world.bodies[hot].temperature = 5000.0;

    let heat0 = total_heat(&world);
    let steps = 1200;
    let mut peak = 0.0f32;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        for body in &world.bodies {
            peak = peak.max(body.velocity.length());
        }
        if frame % 100 == 0 {
            assert_thermal_sane(&world, "thermal shock");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    let gap = (world.bodies[hot].temperature - world.bodies[cold].temperature).abs();
    assert!(gap < 500.0, "the shock must relax: the gap is still {gap}");
    let heat1 = total_heat(&world);
    assert!(
        (heat1 - heat0).abs() < 2e-3 * heat0.abs(),
        "the pair must conserve its heat: {heat0} -> {heat1}"
    );

    thermal_report(
        "shock_converges_without_explosion",
        3,
        steps,
        wall_ms,
        &format!(
            "final_hot={:.2} final_cold={:.2} gap={gap:.4} heat={heat0:.1}->{heat1:.1} peak_speed={peak:.4}",
            world.bodies[hot].temperature,
            world.bodies[cold].temperature
        ),
    );
}

/// The extremes of every knob at once must still leave a finite state.
#[test]
fn thermal_extreme_values_stay_finite() {
    let mut world = thermal_world();
    world.thermal.config.radiation_enabled = true;
    world.thermal.config.conduction_enabled = true;
    world.thermal.config.convection_enabled = true;
    world.thermal.config.max_temperature = 5000.0;
    world.thermal.config.min_temperature = 0.0;
    world.thermal.config.expansion_coefficient = 1.0e-2;
    world.thermal.config.expansion_reference_temperature = 0.0;
    world.thermal.config.forced_convection_coefficient = 1000.0;
    world.thermal.config.ambient_temperature = 5000.0;

    let hot = world
        .add_dynamic_body(
            Vec2::new(-0.49, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    let cold = world
        .add_dynamic_body(
            Vec2::new(0.49, 0.0),
            Shape2::Box {
                half_extents: Vec2::new(0.5, 0.5),
            },
            1.0,
        )
        .expect("valid shape");
    world.bodies[hot].temperature = 5000.0;
    world.bodies[cold].temperature = 0.0;

    let steps = 600;
    let start = Instant::now();
    for frame in 0..steps {
        world.step(DT);
        if frame % 50 == 0 {
            assert_thermal_sane(&world, "extreme thermal");
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Every shape must still be finite and non-degenerate after the
    // expansion has run at the maximum coefficient.
    for body in &world.bodies {
        if let Some(shape) = body.shape.as_ref() {
            match shape {
                Shape2::Box { half_extents } => {
                    assert!(half_extents.x.is_finite() && half_extents.x > 0.0);
                    assert!(half_extents.y.is_finite() && half_extents.y > 0.0);
                }
                Shape2::Circle { radius } => {
                    assert!(radius.is_finite() && *radius > 0.0);
                }
                Shape2::Polygon { vertices } => {
                    for v in vertices {
                        assert!(v.x.is_finite() && v.y.is_finite());
                    }
                }
            }
        }
    }

    thermal_report(
        "extreme_values_stay_finite",
        2,
        steps,
        wall_ms,
        &format!(
            "final_hot={:.2} final_cold={:.2} expansion={:.4}",
            world.bodies[hot].temperature,
            world.bodies[cold].temperature,
            world.thermal.config.expansion_coefficient
        ),
    );
}
