//! The stationary one-dimensional Schrödinger equation (M17–M26).
//!
//! These tests assert the *physics*, not the code: the infinite well's
//! `E_n = n²π²/2`, the harmonic oscillator's `E_n = (n + ½)ℏω`, a
//! ground state that is normalised and symmetric, a Sturm count that
//! agrees with the spectrum, a tridiagonal solve that reproduces a
//! known solution, a double well whose ground state sits below the
//! barrier, and the soft Coulomb potential's hydrogenic ground state.
//!
//! The grid is 401 points unless a test says otherwise; the spacing is
//! then `1/400`, and the discretisation error is second order in it —
//! which is why the well's first level is already within 3e-5 of the
//! continuum value.

use phieng_math::complex::Complex32;
use phieng_quantum::{
    lowest_eigenvector, sturm_count, tridiagonal_eigenvalues, tridiagonal_lowest_eigenvalues,
    tridiagonal_solve, Boundary1D, Grid1D, Hamiltonian1D, Potential1D, QuantumOperator,
    QuantumState,
};

/// The infinite square well on [0, 1], with hbar = m = 1.
fn well(points: usize) -> Hamiltonian1D {
    let grid = Grid1D::new(points, 0.0, 1.0);
    Hamiltonian1D::new(grid, Potential1D::Zero, 1.0, 1.0)
}

/// The harmonic oscillator on [-10, 10], with omega = 1.
fn harmonic_oscillator(points: usize) -> Hamiltonian1D {
    let grid = Grid1D::new(points, -10.0, 10.0);
    Hamiltonian1D::new(grid, Potential1D::Harmonic { omega: 1.0 }, 1.0, 1.0)
}

/// The index of the largest amplitude of a state.
fn peak(state: &QuantumState) -> usize {
    let mut best = 0;
    for i in 1..state.dim() {
        if state.amplitudes()[i].re.abs() > state.amplitudes()[best].re.abs() {
            best = i;
        }
    }
    best
}

/// M17: the infinite square well's spectrum is `E_n = n²π²/2`.
#[test]
fn m17_the_infinite_well_spectrum() {
    let h = well(401);
    let values = tridiagonal_eigenvalues(&h.diagonal(), &h.off_diagonal());
    assert_eq!(values.len(), 399, "the interior points are the unknowns");

    for n in 1..=4usize {
        let exact = (n * n) as f32 * std::f32::consts::PI * std::f32::consts::PI / 2.0;
        let relative = (values[n - 1] - exact).abs() / exact;
        assert!(
            relative < 1e-3,
            "E_{n} = {} must be within 1e-3 of {exact} (off by {relative:.3e})",
            values[n - 1]
        );
    }

    // The spectrum rises, and the gap grows like n.
    for n in 1..values.len() {
        assert!(values[n] > values[n - 1], "the spectrum must rise");
    }
    assert!(values[1] - values[0] > values[0], "the gap must grow");
}

/// M18: the harmonic oscillator's spectrum is `E_n = (n + ½)ℏω`.
#[test]
fn m18_the_harmonic_oscillator_spectrum() {
    let h = harmonic_oscillator(401);
    let values = tridiagonal_lowest_eigenvalues(&h.diagonal(), &h.off_diagonal(), 4);
    assert_eq!(values.len(), 4);

    for (n, value) in values.iter().enumerate() {
        let exact = n as f32 + 0.5;
        let relative = (value - exact).abs() / exact;
        assert!(
            relative < 1e-3,
            "E_{n} = {value} must be within 1e-3 of {exact} (off by {relative:.3e})"
        );
    }

    // The levels are evenly spaced: the gap is hbar omega.
    for n in 1..values.len() {
        assert!(
            (values[n] - values[n - 1] - 1.0).abs() < 1e-3,
            "the gap must be one"
        );
    }
}

/// M19: the ground state is normalised and symmetric.
#[test]
fn m19_the_ground_state_is_normalised_and_symmetric() {
    let h = well(401);
    let (energy, state) = h.ground_state(200, 1e-9);
    let exact = std::f32::consts::PI * std::f32::consts::PI / 2.0;
    let relative = (energy - exact).abs() / exact;
    assert!(
        relative < 1e-3,
        "the ground state energy {energy} must be within 1e-3 of {exact} (off by {relative:.3e})"
    );
    assert!(state.is_normalized(1e-5), "the state must be normalised");

    let n = state.dim();
    for i in 0..n {
        let asymmetry = (state.amplitudes()[i].re - state.amplitudes()[n - 1 - i].re).abs();
        assert!(asymmetry < 1e-5, "the state must be symmetric (i = {i})");
    }
}

/// M20: the harmonic oscillator's ground state peaks at the centre.
#[test]
fn m20_the_harmonic_ground_state_peaks_at_the_centre() {
    let h = harmonic_oscillator(401);
    let (energy, state) = h.ground_state(200, 1e-9);
    assert!(
        (energy - 0.5).abs() < 1e-3,
        "the ground state energy {energy} must be 0.5 to 1e-3"
    );

    // The unknowns are the interior points of a 401 point grid, so the
    // centre is the 200th of them: index 199.
    let centre = state.dim() / 2;
    assert_eq!(centre, 199);
    let p = peak(&state);
    assert!(
        (p as i64 - centre as i64).abs() <= 1,
        "the wavefunction must peak at the centre (peak at {p})"
    );
}

/// M21: the Sturm count is the number of eigenvalues below the query point.
#[test]
fn m21_the_sturm_count_counts_the_eigenvalues_below() {
    let h = harmonic_oscillator(401);
    let d = h.diagonal();
    let e = h.off_diagonal();
    let spectrum = tridiagonal_lowest_eigenvalues(&d, &e, 4);

    // Below the lowest eigenvalue there is nothing.
    assert_eq!(sturm_count(&d, &e, spectrum[0] - 0.01), 0);
    // Just above it there is exactly one.
    assert_eq!(sturm_count(&d, &e, spectrum[0] + 0.01), 1);

    // The count at a point is the number of eigenvalues below it.
    for (k, value) in spectrum.iter().enumerate() {
        assert_eq!(
            sturm_count(&d, &e, *value - 0.01),
            k,
            "the count at {value}"
        );
        assert_eq!(
            sturm_count(&d, &e, *value + 0.01),
            k + 1,
            "the count at {value}"
        );
    }

    // And the count never decreases as the query point rises.
    let mut last = 0usize;
    for i in 0..=400 {
        let x = -20.0 + i as f32 * 0.1;
        let count = sturm_count(&d, &e, x);
        assert!(count >= last, "the count must not decrease at {x}");
        last = count;
    }
}

/// M22: the tridiagonal solve reproduces the known solution.
#[test]
fn m22_the_tridiagonal_solve_reproduces_the_solution() {
    let diagonal = [4.0f32, 5.0, 6.0, 7.0];
    let off_diagonal = [-1.0f32, -1.0, -1.0];
    let rhs = [2.0f32, 6.0, 12.0, 25.0];
    let mut solution = [0.0f32; 4];

    assert!(
        tridiagonal_solve(&diagonal, &off_diagonal, &rhs, &mut solution),
        "the solve must succeed"
    );
    for (i, value) in solution.iter().enumerate() {
        assert!(
            (value - (i + 1) as f32).abs() < 1e-5,
            "x[{i}] = {value} must be {}",
            i + 1
        );
    }

    // The buffers must be long enough, or the solve refuses.
    assert!(
        !tridiagonal_solve(&diagonal, &off_diagonal, &[0.0, 0.0], &mut solution),
        "a short right-hand side must be refused"
    );
    assert!(
        !tridiagonal_solve(&diagonal, &off_diagonal, &rhs, &mut [0.0, 0.0]),
        "a short output buffer must be refused"
    );
}

/// M23: the lowest eigenvector's expectation value is the eigenvalue.
#[test]
fn m23_the_lowest_eigenvector_has_the_eigenvalue_as_its_expectation() {
    let h = well(401);
    let d = h.diagonal();
    let e = h.off_diagonal();
    let eigenvalue = tridiagonal_lowest_eigenvalues(&d, &e, 1)[0];

    let state = lowest_eigenvector(&d, &e, eigenvalue, 50).expect("the iteration must converge");
    assert!(
        state.is_normalized(1e-5),
        "the eigenvector must be normalised"
    );

    let expectation = state.expectation(&h);
    assert!(
        (expectation.re - eigenvalue).abs() <= 1e-3 * eigenvalue,
        "the expectation {} must be the eigenvalue {eigenvalue}",
        expectation.re
    );
    assert!(
        expectation.im.abs() < 1e-5,
        "the expectation of a Hermitian operator must be real"
    );
}

/// M24: the double well's ground state sits below the barrier.
#[test]
fn m24_the_double_well_ground_state_is_below_the_barrier() {
    const BARRIER: f32 = 5.0;
    let grid = Grid1D::new(401, -8.0, 8.0);
    let h = Hamiltonian1D::new(
        grid.clone(),
        Potential1D::DoubleWell {
            center: 0.0,
            separation: 2.0,
            depth: 1.0,
            barrier: BARRIER,
        },
        1.0,
        1.0,
    );

    let (energy, state) = h.ground_state(200, 1e-9);
    assert!(
        energy < BARRIER,
        "the ground state energy {energy} must be below the barrier"
    );

    // The weight inside the barrier must be a small fraction of the
    // weight inside a well: the state is in a well, not over both.
    let mut well_mass = 0.0f64;
    let mut barrier_mass = 0.0f64;
    for i in 0..state.dim() {
        let x = grid.x_at(i + 1);
        let density = (state.amplitudes()[i].re * state.amplitudes()[i].re) as f64;
        if x.abs() >= 1.5 && x.abs() <= 3.0 {
            well_mass += density;
        }
        if x.abs() <= 1.0 {
            barrier_mass += density;
        }
    }
    assert!(well_mass > 0.0, "the state must live in a well");
    assert!(
        barrier_mass < 0.05 * well_mass,
        "the state must be localised: well {well_mass}, barrier {barrier_mass}"
    );
}

/// M25: the soft Coulomb potential's ground state is the hydrogenic one.
#[test]
fn m25_the_soft_coulomb_ground_state_is_the_hydrogenic_one() {
    let grid = Grid1D::new(2001, -20.0, 20.0);
    let h = Hamiltonian1D::new(
        grid.clone(),
        Potential1D::SoftCoulomb {
            strength: 1.0,
            softening: 1.4,
        },
        1.0,
        1.0,
    );

    let (energy, _state) = h.ground_state(100, 1e-9);
    let relative = (energy + 0.5).abs() / 0.5;
    assert!(
        relative <= 0.02,
        "the ground state energy {energy} must be within 2% of -0.5 (off by {relative:.2}%)"
    );
}

/// M26: the grid's spacing, positions and endpoints.
#[test]
fn m26_the_grid_spacing_and_positions() {
    let grid = Grid1D::new(401, -1.0, 1.0);
    assert_eq!(grid.n(), 401);
    assert_eq!(grid.x_min(), -1.0);
    assert_eq!(grid.x_max(), 1.0);
    assert_eq!(grid.dx(), 2.0 / 400.0);

    let positions = grid.positions();
    assert_eq!(positions.len(), 401);
    assert_eq!(positions[0], -1.0);
    assert_eq!(positions[400], 1.0);
    assert_eq!(positions[200], 0.0);
    for i in 1..positions.len() {
        assert!(
            positions[i] > positions[i - 1],
            "the positions must rise (i = {i})"
        );
    }

    // The periodic boundary keeps every point; the Dirichlet one holds
    // the two ends at zero and drops them from the unknowns.
    let h = Hamiltonian1D::new(Grid1D::new(401, -1.0, 1.0), Potential1D::Zero, 1.0, 1.0);
    assert_eq!(
        h.unknowns(),
        399,
        "the Dirichlet condition drops the two ends"
    );
    let periodic = h.clone().with_boundary(Boundary1D::Periodic);
    assert_eq!(
        periodic.unknowns(),
        401,
        "the periodic condition keeps every point"
    );

    // A state of the right size can be built and normalised.
    let mut state = QuantumState::zeros(periodic.dim());
    for (i, a) in state.amplitudes_mut().iter_mut().enumerate() {
        *a = Complex32::new(1.0, 0.0);
        let _ = i;
    }
    assert!(state.normalize());
    assert!(state.is_normalized(1e-5));
}
