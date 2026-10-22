//! The game-facing quantum layer (M49–M56, mirrored Rust↔C#).
//!
//! These tests assert what a game actually consumes: Born-rule sampling
//! that follows the wavefunction, the Bohmian velocity a driven body
//! surfs, the Madelung fluid's dispersion against the analytic
//! frequency, the classical limit that recovers the ordinary fluid,
//! and the density/phase feed a renderer reads.

use phieng_math::complex::Complex32;
use phieng_quantum::field::{QuantumConfig, QuantumField};
use phieng_quantum::hamiltonian::Boundary1D;
use phieng_quantum::madelung::{quantum_potential, MadelungScratch, QuantumFluid1D};

/// M49: Born-rule sampling follows the wavefunction.
///
/// A flat six-outcome state must fill its bins evenly, and the seeded
/// stream pins the exact counts: the same seed gives the same shots on
/// every platform and in both languages.
#[test]
fn m49_the_sampling_follows_the_born_rule() {
    let mut field = QuantumField::new(QuantumConfig {
        enabled: true,
        grid_size: 8,
        ..Default::default()
    });
    assert_eq!(field.state().dim(), 6, "the Dirichlet ends are not drawn");
    for a in field.state_mut().amplitudes_mut().iter_mut() {
        *a = Complex32::new(1.0, 0.0);
    }
    field.state_mut().normalize();

    let mut bins = [0usize; 6];
    let mut one = [0.0f32; 1];
    for _ in 0..10000 {
        field.sample(&mut one);
        bins[one[0] as usize] += 1;
    }
    assert_eq!(
        bins,
        [1671, 1643, 1625, 1660, 1661, 1740],
        "the seeded shots are pinned"
    );
    let total: usize = bins.iter().sum();
    assert_eq!(total, 10000);
}

/// M50: the same seed gives the same shots, a different seed different
/// ones.
#[test]
fn m50_the_sampling_is_reproducible() {
    let draws = |seed: u64| {
        let mut field = QuantumField::new(QuantumConfig {
            enabled: true,
            grid_size: 8,
            seed,
            ..Default::default()
        });
        let mut out = [0.0f32; 64];
        field.sample(&mut out);
        out
    };
    let a = draws(1);
    let b = draws(1);
    assert_eq!(a, b, "the same seed must give the same shots");
    let c = draws(2);
    assert_ne!(a, c, "a different seed must give different shots");
}

/// M51: the Bohmian velocity vanishes for a stationary state and
/// follows the phase gradient otherwise.
#[test]
fn m51_the_bohmian_velocity() {
    // The harmonic ground state is real: it has no current anywhere.
    let field = QuantumField::new(QuantumConfig {
        enabled: true,
        grid_size: 128,
        ..Default::default()
    });
    let mut worst = 0.0f32;
    for i in 0..128 {
        let x = -10.0 + i as f32 * (20.0 / 127.0);
        worst = worst.max(field.velocity_at(x).abs());
    }
    assert!(
        worst < 1e-9,
        "a stationary state must not flow (max|v| = {worst:e})"
    );

    // A constant-amplitude state with a linear phase on a periodic grid
    // flows at exactly the discrete phase gradient: sin(Δ)/dx.
    let mut flow = QuantumField::new(QuantumConfig {
        enabled: true,
        grid_size: 64,
        domain_min: -5.0,
        domain_max: 5.0,
        boundary: Boundary1D::Periodic,
        ..Default::default()
    });
    let n = flow.state().dim();
    assert_eq!(n, 64);
    for (j, a) in flow.state_mut().amplitudes_mut().iter_mut().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * 2.0 * j as f64 / n as f64;
        let s = 1.0 / (n as f32).sqrt();
        *a = Complex32::new(s * (phase.cos() as f32), s * (phase.sin() as f32));
    }
    // The discrete-exact flow: sin(Δ)/dx with Δ the phase step.
    let dx = 10.0 / 63.0;
    let delta = std::f32::consts::PI / 16.0;
    let want = delta.sin() / dx;
    let mut worst = 0.0f32;
    for j in 0..n {
        let x = -5.0 + j as f32 * dx;
        worst = worst.max((flow.velocity_at(x) - want).abs());
    }
    assert!(
        worst < 1e-5,
        "the flow must be the discrete phase gradient (worst {worst:e})"
    );
}

/// M52: a driven body lands on the grid with the flow velocity, and the
/// drive is deterministic.
#[test]
fn m52_a_driven_body_surfs_the_wavefunction() {
    use phieng_dynamics::World2;

    let drive = |seed: u64| {
        let mut world = World2::new();
        world.gravity = glam::Vec2::new(0.0, 0.0);
        let id = world
            .add_dynamic_body(
                glam::Vec2::new(0.0, 3.0),
                phieng_dynamics::Shape2::Circle { radius: 0.5 },
                1.0,
            )
            .expect("a valid body");
        world.quantum.configure(QuantumConfig {
            enabled: true,
            grid_size: 32,
            seed,
            ..Default::default()
        });
        assert!(
            world.drive_body_by_quantum(id),
            "the drive must accept a live body"
        );
        (
            world.bodies[id].transform.position.x,
            world.bodies[id].velocity.x,
        )
    };
    let (x0, v0) = drive(7);
    let (x1, v1) = drive(7);
    assert_eq!(
        (x0.to_bits(), v0.to_bits()),
        (x1.to_bits(), v1.to_bits()),
        "the same seed must drive the same placement"
    );
    // The grid is [-10, 10]: the body must land on it.
    assert!(
        (-10.0..=10.0).contains(&x0),
        "the body must land on the grid (x = {x0})"
    );
    assert!(v0.is_finite(), "the driven velocity must be finite");

    // A field that is off, or a body that is not there, refuses.
    let mut world = World2::new();
    let id = world
        .add_dynamic_body(
            glam::Vec2::new(0.0, 0.0),
            phieng_dynamics::Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("a valid body");
    assert!(
        !world.drive_body_by_quantum(id),
        "a disabled field must refuse the drive"
    );
    world.quantum.configure(QuantumConfig {
        enabled: true,
        ..Default::default()
    });
    assert!(
        !world.drive_body_by_quantum(9999),
        "an unknown body must be refused"
    );
}

/// M53: the Madelung fluid oscillates at the analytic frequency.
///
/// A small sinusoidal density perturbation of a uniform fluid is a
/// standing wave at ω = ℏk²/2m. The test finds the first zero
/// down-crossing of the density at the origin and reads the frequency
/// off it.
#[test]
fn m53_the_madelung_fluid_disperses() {
    let l = 2.0 * std::f32::consts::PI;
    let n = 64usize;
    let dx = l / n as f32;
    let mut fluid = QuantumFluid1D::new(n, dx, 1.0, 1.0, 1.0);
    let mut rho0 = vec![0.0f32; n];
    for (i, slot) in rho0.iter_mut().enumerate().take(n) {
        *slot = 1.0 + 0.01 * (i as f32 * dx).cos();
    }
    fluid.set_density(&rho0);
    let m0 = fluid.mass_total();

    let dt = 0.0005f32;
    let mut scratch = MadelungScratch::new(n);
    let mut t = 0.0f32;
    let mut crossing = 0.0f32;
    for _ in 0..20000 {
        fluid.step(dt, &mut scratch);
        t += dt;
        // The first crossing below the mean after the start.
        if t > 0.5 && fluid.rho()[0] < 1.0 {
            crossing = t;
            break;
        }
    }
    assert!(crossing > 0.0, "the wave must cross the mean");
    // The crossing is a quarter period: ω = π / (2 t).
    let omega = std::f32::consts::PI / (2.0 * crossing);
    let relative = (omega - 0.5).abs() / 0.5;
    assert!(
        relative < 0.05,
        "the measured frequency {omega} must be within 5% of 0.5"
    );

    // The mass survives the run.
    let relative_mass = (fluid.mass_total() - m0).abs() / m0;
    assert!(
        relative_mass < 1e-5,
        "the mass must be conserved (drift {relative_mass:e})"
    );
}

/// M54: the classical limit recovers the ordinary fluid.
///
/// With ℏ = 0 the quantum potential is exactly zero and a still fluid
/// stays still: nothing moves, nothing appears.
#[test]
fn m54_the_classical_limit_recovers_the_ordinary_fluid() {
    let l = 2.0 * std::f32::consts::PI;
    let n = 64usize;
    let dx = l / n as f32;
    let mut fluid = QuantumFluid1D::new(n, dx, 0.0, 1.0, 1.0);
    let mut rho0 = vec![0.0f32; n];
    for (i, slot) in rho0.iter_mut().enumerate().take(n) {
        *slot = 1.0 + 0.01 * (i as f32 * dx).cos();
    }
    fluid.set_density(&rho0);

    // The quantum potential of the same bump is exactly zero.
    let mut q = vec![0.0f32; n];
    assert!(quantum_potential(fluid.rho(), dx, 0.0, 1.0, &mut q));
    for v in q.iter() {
        assert!(v.abs() < 1e-12, "Q must vanish with hbar");
    }

    let mut scratch = MadelungScratch::new(n);
    for _ in 0..2000 {
        assert!(fluid.step(0.0005, &mut scratch));
    }
    for (i, r) in fluid.rho().iter().enumerate() {
        assert!(
            (r - rho0[i]).abs() < 1e-9,
            "a still classical fluid must stay still"
        );
        assert!(
            fluid.vel()[i].abs() < 1e-12,
            "no velocity may appear from nothing"
        );
    }
}

/// M55: the density sums to one and the ground state's phase is flat.
#[test]
fn m55_the_visualisation_feed() {
    let field = QuantumField::new(QuantumConfig {
        enabled: true,
        grid_size: 128,
        ..Default::default()
    });
    let n = field.state().dim();
    let mut density = vec![0.0f32; n];
    let mut phase = vec![0.0f32; n];
    assert_eq!(field.density(&mut density), n);
    assert_eq!(field.phase(&mut phase), n);

    let total: f32 = density.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "the density must sum to one ({total})"
    );
    for d in density.iter() {
        assert!(*d >= 0.0, "a density must not be negative");
    }

    // The ground state is real up to a global sign: its phase is one
    // value everywhere.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in phase.iter() {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    assert!(
        hi - lo < 1e-3,
        "the ground state's phase must be flat (range {})",
        hi - lo
    );
}

/// M56: the game-facing surface round-trips through the native bridge.
#[test]
fn m56_the_ffi_round_trip() {
    use phieng::ffi::{
        phieng_world_free, phieng_world_new, phieng_world_quantum_density,
        phieng_world_quantum_drive_body, phieng_world_quantum_force_at, phieng_world_quantum_phase,
        QuantumConfigFfi,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());

        // A field that is off refuses the drive and reports no force.
        assert_eq!(
            phieng_world_quantum_drive_body(world, 0),
            0,
            "a disabled field must refuse the drive"
        );
        assert_eq!(phieng_world_quantum_force_at(world, 0.0), 0.0);

        // Switch the field on through the FFI struct.
        let cfg = QuantumConfigFfi {
            enabled: 1,
            method: 0,
            boundary: 0,
            grid_size: 32,
            domain_min: -5.0,
            domain_max: 5.0,
            mass: 1.0,
            hbar: 1.0,
            omega: 2.0,
            seed: 12345,
        };
        // A body to drive.
        let id = phieng::ffi::phieng_world_add_body(world, 0.0, 0.0, 0, 0.5, 0.0, 1.0, 1, 0.5, 0.0);
        assert!(id >= 0, "the body must be added");
        // Configure through the dedicated setter.
        assert_eq!(
            phieng::ffi::phieng_world_quantum_config(world, &cfg),
            1,
            "the config must be accepted"
        );
        assert_eq!(
            phieng_world_quantum_drive_body(world, id as u32),
            1,
            "the drive must accept a live body"
        );
        let force = phieng_world_quantum_force_at(world, 1.0);
        assert!(force.is_finite(), "the force must be finite");

        // The density: a null array asks for the count alone.
        let count = phieng_world_quantum_density(world, std::ptr::null_mut(), 0);
        assert_eq!(count, 30, "the Dirichlet ends are not drawn");
        let mut density = vec![0.0f32; count as usize];
        assert_eq!(
            phieng_world_quantum_density(world, density.as_mut_ptr(), count),
            count
        );
        let total: f32 = density.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "the density must sum to one ({total})"
        );

        // The phase: same convention.
        let count = phieng_world_quantum_phase(world, std::ptr::null_mut(), 0);
        assert_eq!(count, 30);
        let mut phase = vec![0.0f32; count as usize];
        assert_eq!(
            phieng_world_quantum_phase(world, phase.as_mut_ptr(), count),
            count
        );

        phieng_world_free(world);
    }
}
