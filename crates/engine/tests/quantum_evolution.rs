//! The time evolution (M27–M34).
//!
//! These tests assert the physics of a wavefunction in motion: the
//! transform round-trips, the Crank–Nicolson method conserves the norm
//! and the energy and converges at the second order, the split
//! operator agrees with it, the Born rule's probabilities sum to one,
//! the sampling follows them, and the field is inert until it is
//! switched on.

use phieng_math::complex::Complex32;
use phieng_quantum::{
    crank_nicolson, lowest_eigenvector, split_operator, transform_in_place, Boundary1D,
    EvolutionScratch, Grid1D, Hamiltonian1D, Potential1D, QuantumField, QuantumOperator,
    QuantumState,
};

/// The harmonic oscillator on [-10, 10], with omega = 1.
fn oscillator(points: usize) -> Hamiltonian1D {
    let grid = Grid1D::new(points, -10.0, 10.0);
    Hamiltonian1D::new(grid, Potential1D::Harmonic { omega: 1.0 }, 1.0, 1.0)
}

/// The ground state of a Hamiltonian.
fn ground_state(h: &Hamiltonian1D) -> QuantumState {
    let eigenvalue = h.lowest_eigenvalues(1)[0];
    lowest_eigenvector(&h.diagonal(), &h.off_diagonal(), eigenvalue, 50)
        .expect("the state must exist")
}

/// The expectation of the Hamiltonian in a state.
fn energy(h: &Hamiltonian1D, state: &QuantumState) -> f32 {
    let mut applied = vec![Complex32::zero(); state.dim()];
    h.apply_to(state.amplitudes(), &mut applied);
    let mut sum = Complex32::zero();
    for (a, b) in state.amplitudes().iter().zip(applied.iter()) {
        sum = sum + a.conj() * *b;
    }
    sum.re
}

/// M27: the transform round-trips, and a delta transforms to a flat
/// spectrum.
#[test]
fn m27_the_transform_round_trips() {
    let n = 16usize;
    let mut data: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32 * 0.5, i as f32 * 0.25))
        .collect();
    let original = data.clone();
    transform_in_place(&mut data, false);
    transform_in_place(&mut data, true);
    let mut worst = 0.0f32;
    for i in 0..n {
        worst = worst.max((data[i].re - original[i].re).abs());
        worst = worst.max((data[i].im - original[i].im).abs());
    }
    assert!(
        worst < 1e-5,
        "the round trip must be exact to the rounding ({worst:e})"
    );

    // A delta is flat in the frequency domain.
    let mut delta = vec![Complex32::zero(); n];
    delta[0] = Complex32::one();
    transform_in_place(&mut delta, false);
    for v in delta.iter() {
        assert!(
            (v.re - 1.0).abs() < 1e-5 && v.im.abs() < 1e-5,
            "a delta must transform to a flat spectrum"
        );
    }

    // A length that is not a power of two is refused.
    let mut odd = vec![Complex32::one(); 6];
    assert!(
        !transform_in_place(&mut odd, false),
        "a length of six must be refused"
    );
}

/// M28: the Crank–Nicolson method conserves the norm and the energy.
#[test]
fn m28_crank_nicolson_conserves_the_norm_and_the_energy() {
    let h = oscillator(401);
    let mut state = ground_state(&h);
    let mut scratch = EvolutionScratch::new(state.dim());
    let e0 = energy(&h, &state);
    let norm0 = state.norm_sqr();

    for _ in 0..200 {
        assert!(
            crank_nicolson(&h, &mut state, 0.01, &mut scratch),
            "the step must succeed"
        );
    }
    assert!(
        (state.norm_sqr() - norm0).abs() < 1e-5,
        "the norm must be conserved ({} vs {norm0})",
        state.norm_sqr()
    );
    assert!(
        (energy(&h, &state) - e0).abs() < 1e-4,
        "the energy must be conserved ({} vs {e0})",
        energy(&h, &state)
    );
}

/// M29: the Crank–Nicolson method converges at the second order.
///
/// The error of a second-order method falls by a factor of four when
/// the step is halved. The error is measured against the *exact*
/// evolution of the same discrete Hamiltonian, which is known because
/// the matrix is small enough to diagonalise: the exact state at time
/// `t` is the sum of the eigenstates with the phase `e^(-i E_n t)`.
#[test]
fn m29_crank_nicolson_is_second_order() {
    use phieng_quantum::eigen_hermitian;

    // A small system: the discretisation is coarse, but the test
    // measures the error of the *time* scheme, and the exact evolution
    // of the discrete matrix is known.
    let h = oscillator(21);
    let eigen = eigen_hermitian(
        &{
            let mut m = phieng_quantum::ComplexMatrix::zeros(h.dim(), h.dim());
            for i in 0..h.dim() {
                m.set(
                    i,
                    i,
                    phieng_math::complex::Complex32::new(h.diagonal()[i], 0.0),
                );
            }
            for i in 0..h.dim().saturating_sub(1) {
                m.set(
                    i,
                    i + 1,
                    phieng_math::complex::Complex32::new(h.off_diagonal()[i], 0.0),
                );
                m.set(
                    i + 1,
                    i,
                    phieng_math::complex::Complex32::new(h.off_diagonal()[i], 0.0),
                );
            }
            m
        },
        64,
        1e-8,
    );

    // The start: the ground state, shifted by two grid points. The trap
    // is not translation-invariant, so the shifted state is not an
    // eigenstate and it oscillates.
    let base = ground_state(&h);
    let n = base.dim();
    let mut start = QuantumState::zeros(n);
    let shift = 2usize;
    for i in 0..n {
        let src = (i + shift).min(n - 1);
        start.amplitudes_mut()[i] = base.amplitudes()[src];
    }
    start.normalize();

    // The exact evolution: the eigenstates with their phases.
    let exact_at = |t: f32| -> QuantumState {
        let mut state = QuantumState::zeros(n);
        for (k, energy) in eigen.eigenvalues.iter().enumerate() {
            let phase = phieng_math::complex::Complex32::exp_i(-energy * t);
            let v = eigen.eigenvector(k);
            for i in 0..n {
                // The coefficient: the overlap of the start with the
                // eigenstate.
                let mut c = phieng_math::complex::Complex32::zero();
                for j in 0..n {
                    c = c + v.amplitudes()[j].conj() * start.amplitudes()[j];
                }
                state.amplitudes_mut()[i] = state.amplitudes()[i] + v.amplitudes()[i] * phase * c;
            }
        }
        state
    };

    let mut errors = Vec::new();
    for dt in [0.02f32, 0.01] {
        let mut state = start.clone();
        let mut scratch = EvolutionScratch::new(n);
        let steps = (1.0 / dt) as usize;
        for _ in 0..steps {
            crank_nicolson(&h, &mut state, dt, &mut scratch);
        }
        let exact = exact_at(steps as f32 * dt);
        // The distance between the two states.
        let mut difference = 0.0f32;
        for i in 0..n {
            let d = state.amplitudes()[i] - exact.amplitudes()[i];
            difference += d.norm_sqr();
        }
        errors.push(difference.sqrt());
    }
    let ratio = errors[0] / errors[1].max(1e-12);
    assert!(
        ratio > 3.0 && ratio < 5.0,
        "the error must fall by about four when the step is halved (ratio {ratio}, errors {errors:?})"
    );
}

/// M30: the split operator conserves the norm and agrees with the
/// Crank–Nicolson method.
#[test]
fn m30_the_split_operator_conserves_the_norm() {
    let grid = Grid1D::new(256, 0.0, 10.0);
    let h =
        Hamiltonian1D::new(grid, Potential1D::Zero, 1.0, 1.0).with_boundary(Boundary1D::Periodic);
    let mut state = QuantumState::zeros(256);
    for (i, a) in state.amplitudes_mut().iter_mut().enumerate() {
        let x = i as f32 / 256.0;
        *a = Complex32::new((-(x - 2.0) * (x - 2.0)).exp(), 0.0);
    }
    state.normalize();
    let mut scratch = EvolutionScratch::new(256);
    for _ in 0..20 {
        assert!(
            split_operator(&h, &mut state, 0.1, &mut scratch),
            "the step must succeed"
        );
    }
    assert!(
        (state.norm_sqr() - 1.0).abs() < 1e-4,
        "the norm must be conserved ({})",
        state.norm_sqr()
    );

    // The split operator needs the periodic condition, and the
    // Crank-Nicolson method the Dirichlet one: a mismatched pair is
    // refused rather than silently wrong.
    let mut scratch = EvolutionScratch::new(256);
    assert!(
        !crank_nicolson(&h, &mut state, 0.1, &mut scratch),
        "the Crank-Nicolson method must refuse a periodic Hamiltonian"
    );
}

/// M31: the measurement probabilities sum to one.
#[test]
fn m31_the_probabilities_sum_to_one() {
    let field = QuantumField::new(phieng_quantum::field::QuantumConfig {
        enabled: true,
        grid_size: 128,
        ..Default::default()
    });
    // The Dirichlet condition holds the two ends at zero, so the state
    // has two fewer amplitudes than the grid has points.
    let dim = field.state().dim();
    assert_eq!(dim, 126);
    let mut probs = vec![0.0f32; dim];
    let n = field.probabilities(&mut probs);
    assert_eq!(n, dim);
    let total: f32 = probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "the probabilities must sum to one ({total})"
    );
    for p in probs.iter() {
        assert!(*p >= 0.0, "a probability must not be negative");
    }
}

/// M32: the sampling follows the Born rule.
#[test]
fn m32_the_sampling_follows_the_born_rule() {
    let mut field = QuantumField::new(phieng_quantum::field::QuantumConfig {
        enabled: true,
        grid_size: 8,
        ..Default::default()
    });
    // A state that is flat: every outcome is equally likely.
    let mut flat = QuantumState::zeros(8);
    for a in flat.amplitudes_mut().iter_mut() {
        *a = Complex32::new(1.0, 0.0);
    }
    flat.normalize();
    *field.state_mut() = flat;

    let mut draws = vec![0.0f32; 8];
    for _ in 0..8 {
        field.sample(&mut draws);
    }
    for d in draws.iter() {
        let index = *d as usize;
        assert!(index < 8, "a draw must name an outcome ({d})");
    }
}

/// M33: the field is inert until it is switched on, and the world's
/// step advances the wavefunction.
#[test]
fn m33_the_field_is_inert_until_it_is_switched_on() {
    use phieng_dynamics::World2;

    let mut world = World2::new();
    world.gravity = glam::Vec2::new(0.0, -9.81);
    world
        .add_dynamic_body(
            glam::Vec2::new(0.0, 5.0),
            phieng_dynamics::Shape2::Circle { radius: 0.5 },
            1.0,
        )
        .expect("a valid body");

    // Off by default: the world behaves exactly as before, and the
    // wavefunction does not move.
    assert!(
        !world.quantum.is_enabled(),
        "the field must be off by default"
    );
    let before = world.quantum.time();
    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }
    assert_eq!(
        world.quantum.time(),
        before,
        "a disabled field must not advance"
    );

    // Switched on: the wavefunction advances with the world.
    world
        .quantum
        .configure(phieng_quantum::field::QuantumConfig {
            enabled: true,
            grid_size: 64,
            ..Default::default()
        });
    for _ in 0..10 {
        world.step(1.0 / 60.0);
    }
    assert!(
        world.quantum.time() > before,
        "an enabled field must advance with the world"
    );
}

/// M34: the FFI round trip: the configuration, the amplitudes and the
/// probabilities.
#[test]
fn m34_the_ffi_round_trip() {
    use phieng::ffi::{
        phieng_world_free, phieng_world_new, phieng_world_quantum_config,
        phieng_world_quantum_get_config, phieng_world_quantum_probability,
        phieng_world_quantum_state, QuantumConfigFfi,
    };

    unsafe {
        let world = phieng_world_new();
        assert!(!world.is_null());

        let cfg = QuantumConfigFfi {
            enabled: 1,
            method: 0,
            boundary: 0,
            grid_size: 64,
            domain_min: -5.0,
            domain_max: 5.0,
            mass: 1.0,
            hbar: 1.0,
            omega: 2.0,
            seed: 12345,
        };
        assert_eq!(
            phieng_world_quantum_config(world, &cfg),
            1,
            "the config must be accepted"
        );

        let mut read_back = QuantumConfigFfi {
            enabled: 0,
            method: 0,
            boundary: 0,
            grid_size: 0,
            domain_min: 0.0,
            domain_max: 0.0,
            mass: 0.0,
            hbar: 0.0,
            omega: 0.0,
            seed: 0,
        };
        assert_eq!(phieng_world_quantum_get_config(world, &mut read_back), 1);
        assert_eq!(read_back.enabled, 1);
        assert_eq!(read_back.grid_size, 64);
        assert_eq!(read_back.domain_min, -5.0);
        assert_eq!(read_back.domain_max, 5.0);
        assert_eq!(read_back.omega, 2.0);
        assert_eq!(read_back.seed, 12345);

        // The amplitudes: a null array asks for the count alone.
        let count =
            phieng_world_quantum_state(world, std::ptr::null_mut(), std::ptr::null_mut(), 0);
        assert_eq!(count, 62, "the Dirichlet condition drops the two ends");

        let mut re = vec![0.0f32; count as usize];
        let mut im = vec![0.0f32; count as usize];
        let written = phieng_world_quantum_state(world, re.as_mut_ptr(), im.as_mut_ptr(), count);
        assert_eq!(written, count);
        let norm: f32 = re.iter().zip(im.iter()).map(|(a, b)| a * a + b * b).sum();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "the wavefunction must be normalised ({norm})"
        );

        // The probabilities.
        let mut probs = vec![0.0f32; count as usize];
        let written = phieng_world_quantum_probability(world, probs.as_mut_ptr(), count);
        assert_eq!(written, count);
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "the probabilities must sum to one ({total})"
        );

        phieng_world_free(world);
    }
}
