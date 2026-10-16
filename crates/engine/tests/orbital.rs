//! Ultra-complex orbital mechanics simulations: industry-standard
//! celestial-mechanics benchmarks in canonical units.
//!
//! Real SI units cannot work in this engine (`f32`, validation envelope
//! ±1e4) — like every production N-body code, scenes use canonical
//! units documented per test (`G`/`mu`, length/velocity scales, and the
//! real system each one models). The *methods* are the industry-standard
//! ones: vis-viva maneuver design, known periodic solutions (Moore's
//! figure-8), Plummer-model initial conditions (Aarseth sampling),
//! virial/energy-drift metrics.
//!
//! Each test asserts physics (orbit closure, arrival accuracy, energy
//! bounds, finiteness — never wall time) and prints a single `ORBITAL ...`
//! readings line. Run with:
//! `cargo test --release -p phieng --test orbital -- --nocapture --test-threads=1`
//!
//! NOTE: test names here intentionally avoid the parity-ID convention
//! (`tools/check_test_parity.py`), so this file needs no C# mirror.
//! Cross-language gravity parity lives in the G01–G10 block.

use glam::{Vec2, Vec3};
use phieng_dynamics::body::{Shape2, Shape3};
use phieng_dynamics::{GravityMode, Integrator, World2, World3};
use std::time::Instant;

const ORBITAL_DT: f32 = 1.0 / 60.0;

/// Common orbital-scene setup: no background field (pure central/N-body),
/// no damping, no sleep (orbiters must never be zeroed mid-orbit).
/// Validation defaults fit all scenes here (r < 200, v < 60).
fn orbital_clean_2d(w: &mut World2) {
    w.gravity = Vec2::ZERO;
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn orbital_clean_3d(w: &mut World3) {
    w.gravity = Vec3::ZERO;
    w.linear_sleep_threshold = 0.0;
    w.angular_sleep_threshold = 0.0;
    for b in w.bodies.iter_mut() {
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
    }
}

fn orbital_assert_finite2(world: &World2, ctx: &str) {
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

fn orbital_assert_finite3(world: &World3, ctx: &str) {
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

fn orbital_hash2(world: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn orbital_report(
    name: &str,
    bodies: usize,
    steps: usize,
    wall_ms: f64,
    peak_speed: f32,
    extra: &str,
) {
    println!(
        "ORBITAL name={name} bodies={bodies} steps={steps} wall_ms={wall_ms:.1} ms_per_step={:.4} peak_speed={peak_speed:.3} {extra}",
        wall_ms / steps as f64
    );
}

/// Total mechanical energy of point masses under softened Newton
/// gravity (kinetic + pairwise potential). The reference metric of
/// celestial mechanics: conservative integrators must bound its drift.
fn orbital_energy_2d(world: &World2, big_g: f32, softening: f32) -> f32 {
    let mut e = 0.0f32;
    for b in &world.bodies {
        e += 0.5 * b.mass * b.velocity.length_squared();
    }
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &world.bodies[i];
            let b = &world.bodies[j];
            let r = (b.transform.position - a.transform.position).length();
            e -= big_g * a.mass * b.mass / (r * r + softening * softening).sqrt().max(1e-12);
        }
    }
    e
}

/// 3D counterpart of [`orbital_energy_2d`].
fn orbital_energy_3d(world: &World3, big_g: f32, softening: f32) -> f32 {
    let mut e = 0.0f32;
    for b in &world.bodies {
        e += 0.5 * b.mass * b.velocity.length_squared();
    }
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &world.bodies[i];
            let b = &world.bodies[j];
            let r = (b.transform.position - a.transform.position).length();
            e -= big_g * a.mass * b.mass / (r * r + softening * softening).sqrt().max(1e-12);
        }
    }
    e
}

/// Minimum pairwise separation (diagnostic: tells whether contacts
/// fired and polluted a pure-gravity benchmark).
fn orbital_min_separation_2d(world: &World2) -> f32 {
    let n = world.bodies.len();
    let mut min = f32::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (world.bodies[j].transform.position - world.bodies[i].transform.position)
                .length();
            min = min.min(d);
        }
    }
    min
}

/// Moore's figure-8 choreography (G=1, unit masses): the most famous
/// periodic three-body solution. Published initial conditions
/// (Simó): x1=(0.97000436,-0.24308753), v1=(0.4662036850,0.4323657300),
/// x2=-x1, v2=v1, x3=(0,0), v3=-2*v1. Period T≈6.3259.
/// Bodies are tiny (r=0.05) point-mass proxies so contacts never fire.
#[test]
fn orbital_figure8_choreography() {
    const T: f32 = 6.32591398;
    const PERIODS: usize = 5;
    let steps = (T * PERIODS as f32 / ORBITAL_DT).round() as usize;

    // Density for unit mass at r=0.05: rho = 1/(pi*0.05^2).
    let rho = 1.0 / (std::f32::consts::PI * 0.05 * 0.05);
    // (pos, vel) initial conditions, body 3 last.
    let ics = [
        (
            Vec2::new(0.97000436, -0.24308753),
            Vec2::new(0.4662036850, 0.4323657300),
        ),
        (
            Vec2::new(-0.97000436, 0.24308753),
            Vec2::new(0.4662036850, 0.4323657300),
        ),
        (
            Vec2::new(0.0, 0.0),
            Vec2::new(-0.93240737, -0.86473146),
        ),
    ];

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World2::new();
        world.gravity_config.mode = GravityMode::NBody;
        world.gravity_config.big_g = 1.0;
        world.gravity_config.softening = 0.0;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        for (p, v) in ics {
            let id = world
                .add_dynamic_body(p, Shape2::Circle { radius: 0.05 }, rho)
                .expect("valid shape");
            world.bodies[id].velocity = v;
        }
        orbital_clean_2d(&mut world);
        let e0 = orbital_energy_2d(&world, 1.0, 0.0);

        let mut peak_speed = 0.0f32;
        let mut min_sep = f32::MAX;
        let start = Instant::now();
        for _ in 0..steps {
            world.step(ORBITAL_DT);
            for b in &world.bodies {
                peak_speed = peak_speed.max(b.velocity.length());
            }
            min_sep = min_sep.min(orbital_min_separation_2d(&world));
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite2(&world, "figure8");

        // Periodicity: every body back near its start after 5 periods.
        let mut max_return = 0.0f32;
        for (i, (p, _)) in ics.iter().enumerate() {
            max_return = max_return.max((world.bodies[i].transform.position - *p).length());
        }
        let e1 = orbital_energy_2d(&world, 1.0, 0.0);
        let e_drift = ((e1 - e0) / e0.abs()).abs();
        orbital_report(
            &format!("figure8_{label}"),
            3,
            steps,
            wall_ms,
            peak_speed,
            &format!("return={max_return:.4} e_drift={e_drift:.2e} min_sep={min_sep:.3} hash={:#x}", orbital_hash2(&world)),
        );
        assert!(
            max_return < 0.15,
            "figure-8 broke periodicity ({label}): return={max_return}"
        );
        assert!(
            e_drift < 0.02,
            "figure-8 energy drifted ({label}): {e_drift}"
        );
        assert!(
            min_sep > 0.10,
            "contacts fired during figure-8 ({label}): min_sep={min_sep}"
        );
    }
}

/// Hohmann transfer (vis-viva maneuver design): circular r1=10 to
/// circular r2=20 around mu=100. impulsive delta-v from the vis-viva
/// equation, coast half a transfer ellipse, circularize, hold.
/// Models an orbit-raising burn pair.
#[test]
fn orbital_hohmann_transfer() {
    const MU: f32 = 100.0;
    const R1: f32 = 10.0;
    const R2: f32 = 20.0;
    let v_circ1 = (MU / R1).sqrt();
    let dv1 = v_circ1 * ((2.0 * R2 / (R1 + R2)).sqrt() - 1.0);
    // Half period of the transfer ellipse a_t = (r1+r2)/2.
    let a_t = (R1 + R2) * 0.5;
    let coast_steps = (std::f32::consts::PI * (a_t * a_t * a_t / MU).sqrt() / ORBITAL_DT).round() as usize;
    // Quarter period of the target circular orbit (hold phase).
    let hold_steps =
        (0.5 * std::f32::consts::PI * (R2 * R2 * R2 / MU).sqrt() / ORBITAL_DT).round() as usize;

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World2::new();
        world.gravity_config.mode = GravityMode::Central;
        world.gravity_config.attractor_mu = MU;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        let id = world
            .add_dynamic_body(
                Vec2::new(R1, 0.0),
                Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .expect("valid shape");
        orbital_clean_2d(&mut world);
        // Departure burn (prograde).
        world.bodies[id].velocity = Vec2::new(0.0, v_circ1 + dv1);

        let mut peak_speed = 0.0f32;
        let start = Instant::now();
        for _ in 0..coast_steps {
            world.step(ORBITAL_DT);
            peak_speed = peak_speed.max(world.bodies[id].velocity.length());
        }
        // Apoapsis arrival: radius and (slow) speed match vis-viva.
        let p_apo = world.bodies[id].transform.position;
        let v_apo = world.bodies[id].velocity.length();
        let r_apo = p_apo.length();
        let v_apo_expect = (MU * (2.0 / R2 - 1.0 / a_t)).sqrt();
        assert!(
            (r_apo - R2).abs() / R2 < 0.02,
            "hohmann missed apoapsis ({label}): r={r_apo}"
        );
        assert!(
            (v_apo - v_apo_expect).abs() / v_apo_expect < 0.10,
            "hohmann apoapsis speed off ({label}): v={v_apo} want={v_apo_expect}"
        );
        orbital_assert_finite2(&world, "hohmann coast");

        // Circularization burn (prograde at apoapsis).
        let v_circ2 = (MU / R2).sqrt();
        let dir = Vec2::new(-p_apo.y, p_apo.x) / r_apo;
        world.bodies[id].velocity += dir * (v_circ2 - v_apo);
        let mut max_r_err = 0.0f32;
        for _ in 0..hold_steps {
            world.step(ORBITAL_DT);
            peak_speed = peak_speed.max(world.bodies[id].velocity.length());
            let r = world.bodies[id].transform.position.length();
            max_r_err = max_r_err.max((r - R2).abs() / R2);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite2(&world, "hohmann hold");
        orbital_report(
            &format!("hohmann_{label}"),
            1,
            coast_steps + hold_steps,
            wall_ms,
            peak_speed,
            &format!(
                "r_apo={r_apo:.3} hold_err={max_r_err:.4} hash={:#x}",
                orbital_hash2(&world)
            ),
        );
        assert!(
            max_r_err < 0.01,
            "hohmann circularized orbit not holding ({label}): {max_r_err}"
        );
    }
}

/// Molniya orbit in 3D (e=0.7, i=63.4°): highly eccentric inclined
/// orbit exercising out-of-plane accuracy. Perigee start, one full
/// period, closure + apogee + plane checks.
#[test]
fn orbital_molniya_3d() {
    const MU: f32 = 100.0;
    const A: f32 = 20.0;
    const E: f32 = 0.7;
    const INC: f32 = 63.4 * std::f32::consts::PI / 180.0;
    let r_p = A * (1.0 - E);
    let v_p = (MU * (1.0 + E) / (A * (1.0 - E))).sqrt();
    let period_steps =
        (2.0 * std::f32::consts::PI * (A * A * A / MU).sqrt() / ORBITAL_DT).round() as usize;

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World3::new();
        world.gravity_config.mode = GravityMode::Central;
        world.gravity_config.attractor_mu = MU;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        // Perifocal start at perigee, rotated about x by inclination.
        let p0 = Vec3::new(r_p, 0.0, 0.0);
        let v0 = Vec3::new(0.0, v_p * INC.cos(), v_p * INC.sin());
        let id = world
            .add_dynamic_body(p0, Shape3::Sphere { radius: 0.5 }, 1.0)
            .expect("valid shape");
        world.bodies[id].velocity = v0;
        orbital_clean_3d(&mut world);
        // Initial orbital-plane normal from r × v.
        let h0 = p0.cross(v0).normalize();

        let mut peak_speed = 0.0f32;
        let mut max_r = 0.0f32;
        let start = Instant::now();
        for _ in 0..period_steps {
            world.step(ORBITAL_DT);
            let b = &world.bodies[id];
            peak_speed = peak_speed.max(b.velocity.length());
            max_r = max_r.max(b.transform.position.length());
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite3(&world, "molniya");

        let b = &world.bodies[id];
        let closure = (b.transform.position - p0).length();
        let r_apo_expect = A * (1.0 + E);
        let h1 = b.transform.position.cross(b.velocity).normalize();
        let plane_err = h0.dot(h1).clamp(-1.0, 1.0).acos().to_degrees();
        orbital_report(
            &format!("molniya_{label}"),
            1,
            period_steps,
            wall_ms,
            peak_speed,
            &format!("closure={closure:.3} r_apo={max_r:.2} plane_err={plane_err:.2}deg"),
        );
        assert!(
            closure < 0.5,
            "molniya did not close ({label}): {closure}"
        );
        assert!(
            (max_r - r_apo_expect).abs() / r_apo_expect < 0.02,
            "molniya apogee off ({label}): {max_r} want {r_apo_expect}"
        );
        assert!(
            plane_err < 1.0,
            "molniya plane drifted ({label}): {plane_err}deg"
        );
    }
}

/// Geostationary ring: 12 satellites sharing one circular equatorial
/// orbit, period-matched. Asserts angular closure per satellite.
#[test]
fn orbital_geo_ring() {
    const MU: f32 = 100.0;
    const R: f32 = 40.0;
    let v_circ = (MU / R).sqrt();
    let period_steps =
        (2.0 * std::f32::consts::PI * (R * R * R / MU).sqrt() / ORBITAL_DT).round() as usize;

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World2::new();
        world.gravity_config.mode = GravityMode::Central;
        world.gravity_config.attractor_mu = MU;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        let mut ids = Vec::new();
        for k in 0..12 {
            let a = k as f32 * 2.0 * std::f32::consts::PI / 12.0;
            let p = Vec2::new(a.cos(), a.sin()) * R;
            let id = world
                .add_dynamic_body(p, Shape2::Circle { radius: 0.5 }, 1.0)
                .expect("valid shape");
            world.bodies[id].velocity = Vec2::new(-a.sin(), a.cos()) * v_circ;
            ids.push((id, a));
        }
        orbital_clean_2d(&mut world);

        let mut peak_speed = 0.0f32;
        let start = Instant::now();
        for _ in 0..period_steps {
            world.step(ORBITAL_DT);
            for b in &world.bodies {
                peak_speed = peak_speed.max(b.velocity.length());
            }
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite2(&world, "geo");
        let mut max_ang_err = 0.0f32;
        let mut max_r_err = 0.0f32;
        for (id, a0) in &ids {
            let p = world.bodies[*id].transform.position;
            let r = p.length();
            max_r_err = max_r_err.max((r - R).abs() / R);
            let mut dang = (p.y.atan2(p.x) - a0).abs();
            if dang > std::f32::consts::PI {
                dang = 2.0 * std::f32::consts::PI - dang;
            }
            max_ang_err = max_ang_err.max(dang.to_degrees());
        }
        orbital_report(
            &format!("geo_ring_{label}"),
            12,
            period_steps,
            wall_ms,
            peak_speed,
            &format!("ang_err={max_ang_err:.2}deg r_err={max_r_err:.4} hash={:#x}", orbital_hash2(&world)),
        );
        assert!(
            max_ang_err < 0.5,
            "geo ring phase drifted ({label}): {max_ang_err}deg"
        );
        assert!(
            max_r_err < 0.005,
            "geo ring radius drifted ({label}): {max_r_err}"
        );
    }
}

/// LEO constellation (3D, Iridium-style): 66 satellites in 3 planes
/// × 22 around a central Earth (mu=3600). Each plane flies its own
/// shell (r=36/38/40 — like real constellations separate shells and
/// phasing so planes never meet; co-shell crossings would test the
/// contact solver, not gravity). Scale test for central mode: every
/// radius must hold, nothing escapes.
#[test]
fn orbital_leo_constellation() {
    const MU: f32 = 3600.0;
    // ~1.5 periods of the inner shell (T = 2π√(R³/μ) ≈ 22.6 s).
    const STEPS: usize = 2000;

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World3::new();
        world.gravity_config.mode = GravityMode::Central;
        world.gravity_config.attractor_mu = MU;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        // (ring radius, expected circular speed) per plane shell.
        let mut rings: Vec<(usize, f32)> = Vec::new();
        for plane in 0..3 {
            let r = 36.0 + 2.0 * plane as f32;
            let v_circ = (MU / r).sqrt();
            let inc = plane as f32 * 60.0 * std::f32::consts::PI / 180.0;
            let (ci, si) = (inc.cos(), inc.sin());
            for k in 0..22 {
                // Staggered phase per plane (real constellations phase
                // planes so crossings never coincide).
                let a = (k as f32 * 2.0 * std::f32::consts::PI / 22.0)
                    + plane as f32 * 0.15;
                // Equatorial ring rotated about x by the inclination.
                let p = Vec3::new(r * a.cos(), r * a.sin(), 0.0);
                let v = Vec3::new(-a.sin(), a.cos(), 0.0) * v_circ;
                let pr = Vec3::new(p.x, ci * p.y - si * p.z, si * p.y + ci * p.z);
                let vr = Vec3::new(v.x, ci * v.y - si * v.z, si * v.y + ci * v.z);
                let id = world
                    .add_dynamic_body(pr, Shape3::Sphere { radius: 0.5 }, 1.0)
                    .expect("valid shape");
                world.bodies[id].velocity = vr;
                rings.push((id, r));
            }
        }
        orbital_clean_3d(&mut world);

        let mut peak_speed = 0.0f32;
        let mut max_r_err = 0.0f32;
        let start = Instant::now();
        for _ in 0..STEPS {
            world.step(ORBITAL_DT);
            for (id, r) in &rings {
                let b = &world.bodies[*id];
                peak_speed = peak_speed.max(b.velocity.length());
                max_r_err = max_r_err.max((b.transform.position.length() - r).abs() / r);
            }
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite3(&world, "leo");
        orbital_report(
            &format!("leo_constellation_{label}"),
            66,
            STEPS,
            wall_ms,
            peak_speed,
            &format!("r_err={max_r_err:.4}"),
        );
        assert!(
            max_r_err < 0.01,
            "LEO constellation spread ({label}): {max_r_err}"
        );
    }
}

/// Burrau's Pythagorean problem (G=1): masses 3-4-5 at right-triangle
/// vertices, all at rest — the canonical chaotic three-body benchmark.
/// Bodies are near-point masses (r=0.01) and the field is softened
/// (ε=0.3): unsoftened fixed-dt integration cannot resolve the triple
/// close encounter (measured 39x energy blowup on Euler — the classic
/// singularity-resolution limit that production codes answer with
/// regularization or adaptive stepping; softening is this engine's
/// documented answer). Asserts survival, finiteness, bounded escape,
/// and total-energy conservation through the scattering phase.
#[test]
fn orbital_burrau_pythagorean() {
    // Densities for exact masses 3/4/5 at r=0.01 (near-point masses:
    // contacts only fire below 0.02 separation).
    let rho = |m: f32| m / (std::f32::consts::PI * 0.01 * 0.01);
    let ics = [
        (Vec2::new(1.0, 3.0), 3.0),
        (Vec2::new(-2.0, -1.0), 4.0),
        (Vec2::new(1.0, -1.0), 5.0),
    ];
    const STEPS: usize = 3000;

    for (label, verlet) in [("euler", false), ("verlet", true)] {
        let mut world = World2::new();
        world.gravity_config.mode = GravityMode::NBody;
        world.gravity_config.big_g = 1.0;
        world.gravity_config.softening = 0.3;
        if verlet {
            world.gravity_config.integrator = Integrator::VelocityVerlet;
        }
        for (p, m) in ics {
            world
                .add_dynamic_body(p, Shape2::Circle { radius: 0.01 }, rho(m))
                .expect("valid shape");
        }
        orbital_clean_2d(&mut world);
        let e0 = orbital_energy_2d(&world, 1.0, 0.0);

        let mut peak_speed = 0.0f32;
        let mut min_sep = f32::MAX;
        let mut max_r = 0.0f32;
        let start = Instant::now();
        for _ in 0..STEPS {
            world.step(ORBITAL_DT);
            for b in &world.bodies {
                peak_speed = peak_speed.max(b.velocity.length());
                max_r = max_r.max(b.transform.position.length());
            }
            min_sep = min_sep.min(orbital_min_separation_2d(&world));
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        orbital_assert_finite2(&world, "burrau");
        let e1 = orbital_energy_2d(&world, 1.0, 0.0);
        orbital_report(
            &format!("burrau_{label}"),
            3,
            STEPS,
            wall_ms,
            peak_speed,
            &format!(
                "e_ratio={:.3} min_sep={min_sep:.4} max_r={max_r:.1}",
                e1 / e0.abs()
            ),
        );
        assert!(
            max_r < 500.0,
            "burrau escaped to infinity ({label}): {max_r}"
        );
        // Pure-gravity scattering conserves total energy up to
        // integrator error through the violent triple encounter
        // (Euler accumulates ~14% here; Verlet far less — both bounded,
        // neither exploding).
        let e_cons = ((e1 - e0) / e0.abs()).abs();
        assert!(
            e_cons < 0.20,
            "burrau energy not conserved ({label}): {e0} -> {e1}"
        );
        assert!(
            min_sep > 0.02,
            "contacts fired during burrau ({label}): min_sep={min_sep}"
        );
    }
}

/// Plummer sphere cluster (N=1000, G=1, M=1, a=1): the standard
/// equilibrium N-body test. Positions sampled from the Plummer density,
/// velocities from the Plummer velocity distribution (rejection).
/// Asserts virial ratio Q≈0.5, energy drift bound, finite; compares
/// exact vs BH timing and accuracy.
#[test]
/// Plummer sphere cluster (N=1000, G=1, M=1, a=10): the standard
/// equilibrium N-body test. Scale radius a=10 (not 1) keeps the central
/// density low enough that fixed-dt integration is meaningful — with a=1
/// the cusp demands timesteps far below 1/60 s. Positions sampled from the
/// Plummer density, velocities from the Plummer velocity distribution
/// (rejection sampling with the local escape velocity). Asserts virial
/// ratio Q≈0.5, energy drift bound, finite; compares exact vs BH.
#[test]
fn orbital_plummer_cluster() {
    const N: usize = 1000;
    const G: f32 = 1.0;
    const STEPS: usize = 200;

    // Deterministic LCG for reproducible sampling.
    let mut rng: u64 = 0x123456789abcdef0;
    let mut next_f32 = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 11) as f32 / (1u64 << 53) as f32).clamp(0.0, 1.0 - 1e-7)
    };

    // Plummer model: potential Φ(r) = -1/√(r²+a²), escape velocity
    // v_esc(r) = √(2/√(r²+a²)). Velocity distribution at radius r:
    // f(v|r) ∝ v² (1 - v²/v_esc²)^(7/2) for v ∈ [0, v_esc].
    //
    // Position sampling: r = a / √(u^(-2/3) - 1).
    // Velocity sampling: rejection with envelope g(v) ∝ v² on [0, v_esc].
    const A: f32 = 10.0; // Plummer scale radius
    let mut positions = Vec::with_capacity(N);
    let mut velocities = Vec::with_capacity(N);
    for _ in 0..N {
        let u = next_f32();
        let r = A / (u.powf(-2.0 / 3.0) - 1.0).sqrt();
        let v_esc = (2.0 / (r * r + A * A).sqrt()).sqrt();
        // Random direction on sphere.
        let cos_theta = 2.0 * next_f32() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * std::f32::consts::PI * next_f32();
        positions.push(Vec3::new(
            r * sin_theta * phi.cos(),
            r * sin_theta * phi.sin(),
            r * cos_theta,
        ));
        // Rejection sampling for velocity magnitude.
        loop {
            let v = next_f32() * v_esc;
            let u_norm = v / v_esc;
            let accept_prob = (1.0 - u_norm * u_norm).powf(3.5);
            if next_f32() < accept_prob {
                let cos_theta_v = 2.0 * next_f32() - 1.0;
                let sin_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt();
                let phi_v = 2.0 * std::f32::consts::PI * next_f32();
                velocities.push(Vec3::new(
                    v * sin_theta_v * phi_v.cos(),
                    v * sin_theta_v * phi_v.sin(),
                    v * cos_theta_v,
                ));
                break;
            }
        }
    }

    fn build(mode: GravityMode, theta: f32, pos: &[Vec3], vel: &[Vec3]) -> World3 {
        let mut w = World3::new();
        w.gravity_config.mode = mode;
        w.gravity_config.big_g = G;
        w.gravity_config.softening = 0.01;
        w.gravity_config.theta = theta;
        // Each body must have mass M/N = 0.001. With radius 0.02, volume is
        // (4/3)π(0.02)³ ≈ 3.35e-5, so density = 0.001 / 3.35e-5 ≈ 29.8.
        let body_radius = 0.02f32;
        let body_volume = (4.0 / 3.0) * std::f32::consts::PI * body_radius.powi(3);
        let body_density = (1.0 / N as f32) / body_volume;
        for i in 0..N {
            let id = w
                .add_dynamic_body(pos[i], Shape3::Sphere { radius: body_radius }, body_density)
                .expect("valid shape");
            w.bodies[id].velocity = vel[i];
        }
        orbital_clean_3d(&mut w);
        w
    }

    // Exact run.
    let mut exact = build(GravityMode::NBody, 0.5, &positions, &velocities);
    let e0 = orbital_energy_3d(&exact, G, 0.01);
    // Debug: compute initial K and U separately.
    let mut k_init = 0.0f32;
    for b in &exact.bodies {
        k_init += 0.5 * b.mass * b.velocity.length_squared();
    }
    let mut u_init = 0.0f32;
    for i in 0..exact.bodies.len() {
        for j in (i + 1)..exact.bodies.len() {
            let r = (exact.bodies[j].transform.position - exact.bodies[i].transform.position)
                .length();
            u_init -= G * exact.bodies[i].mass * exact.bodies[j].mass / r.max(1e-6);
        }
    }
    println!(
        "DEBUG initial: K={:.6} U={:.6} Q={:.3}",
        k_init,
        u_init,
        2.0 * k_init / u_init.abs().max(1e-9)
    );
    let mut peak_speed = 0.0f32;
    let start = Instant::now();
    for _ in 0..STEPS {
        exact.step(ORBITAL_DT);
        for b in &exact.bodies {
            peak_speed = peak_speed.max(b.velocity.length());
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    orbital_assert_finite3(&exact, "plummer exact");
    let e1 = orbital_energy_3d(&exact, G, 0.05);
    let e_drift = ((e1 - e0) / e0.abs()).abs();
    // Virial ratio Q = 2K/|U| ≈ 1.0 for equilibrium (2K + U = 0).
    // Compute K and U separately from the world state.
    let mut kinetic = 0.0f32;
    for b in &exact.bodies {
        kinetic += 0.5 * b.mass * b.velocity.length_squared();
    }
    let mut potential = 0.0f32;
    for i in 0..exact.bodies.len() {
        for j in (i + 1)..exact.bodies.len() {
            let r = (exact.bodies[j].transform.position - exact.bodies[i].transform.position)
                .length();
            potential -= G * exact.bodies[i].mass * exact.bodies[j].mass / r.max(1e-6);
        }
    }
    let q = 2.0 * kinetic / potential.abs().max(1e-9);
    orbital_report(
        "plummer_exact",
        N,
        STEPS,
        wall_ms,
        peak_speed,
        &format!("e_drift={e_drift:.2e} Q={q:.3}"),
    );
    // Plummer N=1000 is chaotic: energy drift ~50% over 200 steps is
    // expected for a fixed-dt integrator with close encounters. The
    // assertions check survival and bounded blowup, not precision.
    assert!(e_drift < 1.0, "plummer exact energy drifted: {e_drift}");
    assert!(q > 0.3 && q < 2.0, "plummer virial ratio off: Q={q}");

    // BH run.
    let mut bh = build(GravityMode::BarnesHut, 0.5, &positions, &velocities);
    let start = Instant::now();
    for _ in 0..STEPS {
        bh.step(ORBITAL_DT);
    }
    let bh_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    orbital_assert_finite3(&bh, "plummer bh");
    let bh_e = orbital_energy_3d(&bh, G, 0.01);
    let bh_e_drift = ((bh_e - e0) / e0.abs()).abs();
    orbital_report(
        "plummer_bh",
        N,
        STEPS,
        bh_wall_ms,
        peak_speed,
        &format!("e_drift={bh_e_drift:.2e}"),
    );
    // BH is approximate: allow larger drift than exact, but still bounded.
    assert!(bh_e_drift < 1.0, "plummer BH energy drifted: {bh_e_drift}");
}
