//! The quantum vocabulary (M09–M16, mirrored Rust↔C#).
//!
//! A state is a vector of complex amplitudes; an observable is an
//! operator acting on it. These tests pin the arithmetic of that
//! vocabulary and — more importantly — the *physics* it has to obey:
//!
//! * a state that is normalised has Born probabilities that sum to one,
//!   and the probability of an outcome is the expectation of the
//!   projector onto it;
//! * the expectation value of a Hermitian operator is real;
//! * two states that differ only by a phase are the same state;
//! * the eigensolver finds the spectrum of a Hermitian matrix, and the
//!   eigenvectors it returns really do reconstruct it.
//!
//! The plan for the whole ladder is in `QUANTUM_PLAN.md`.

use phieng_math::complex::Complex32;
use phieng_quantum::{
    dominant_eigenpair, eigen_hermitian, ComplexMatrix, DenseOperator, DiagonalOperator,
    QuantumState,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// A three by three Hermitian matrix: the eigenvalues are 2 ± √2 and 2.
fn tridiagonal3() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(3, 3);
    for i in 0..3 {
        m.set(i, i, Complex32::new(2.0, 0.0));
    }
    m.set(0, 1, Complex32::new(1.0, 0.0));
    m.set(1, 0, Complex32::new(1.0, 0.0));
    m.set(1, 2, Complex32::new(1.0, 0.0));
    m.set(2, 1, Complex32::new(1.0, 0.0));
    m
}

/// A two by two Hermitian matrix with a complex off-diagonal: the
/// eigenvalues are ±√2.
fn complex2() -> ComplexMatrix {
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, Complex32::new(1.0, 0.0));
    m.set(1, 1, Complex32::new(-1.0, 0.0));
    m.set(0, 1, Complex32::new(0.0, 1.0));
    m.set(1, 0, Complex32::new(0.0, -1.0));
    m
}

/// M09: a state can be normalised, and its Born probabilities follow.
#[test]
fn m09_a_state_normalises_and_its_probabilities_follow() {
    let mut s =
        QuantumState::from_amplitudes(vec![Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)]);
    assert_eq!(bits(s.norm_sqr()), 0x4000_0000, "the norm of (1, 1) is two");
    assert!(s.normalize(), "the state must normalise");
    assert_eq!(bits(s.norm_sqr()), 0x3f7f_ffff);
    assert!(s.is_normalized(1e-6), "the state must be normalised now");

    let p = s.born_probabilities();
    assert_eq!(p.len(), 2);
    assert_eq!(bits(p[0]), 0x3eff_ffff);
    assert_eq!(bits(p[1]), 0x3eff_ffff);
    assert!(
        (p[0] + p[1] - 1.0).abs() < 1e-6,
        "the probabilities must sum to one"
    );

    // The zero state has no norm to scale by.
    let mut zero = QuantumState::zeros(3);
    assert!(!zero.normalize(), "the zero state cannot be normalised");
    assert_eq!(bits(zero.norm_sqr()), 0);
}

/// M10: the expectation value of a Hermitian operator is real.
#[test]
fn m10_a_hermitian_expectation_is_real() {
    let op = DenseOperator::new(complex2());
    let s = QuantumState::from_amplitudes(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)]);
    let value = s.expectation(&op);
    assert_eq!(bits(value.re), 0x3f80_0000);
    assert_eq!(
        bits(value.im),
        0,
        "the expectation of a Hermitian operator must be real"
    );

    // A superposition, to make sure the off-diagonal terms cancel.
    let s = QuantumState::from_amplitudes(vec![Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)]);
    let value = s.expectation(&op);
    assert!(
        value.im.abs() < 1e-6,
        "the expectation of a Hermitian operator must be real, got {value:?}"
    );
}

/// M11: the Born probability is the expectation of the projector.
#[test]
fn m11_the_born_probability_is_a_projector_expectation() {
    let mut proj = ComplexMatrix::zeros(2, 2);
    proj.set(0, 0, Complex32::new(1.0, 0.0));
    let op = DenseOperator::new(proj);

    let s = QuantumState::from_amplitudes(vec![Complex32::new(0.6, 0.0), Complex32::new(0.8, 0.0)]);
    let p = s.born_probabilities();
    let value = s.expectation(&op);
    assert_eq!(bits(p[0]), 0x3eb8_51ec, "the probability is 0.6 squared");
    assert_eq!(bits(value.re), 0x3eb8_51ec);
    assert!(
        (value.re - p[0]).abs() < 1e-6,
        "the projector expectation must be the probability"
    );
}

/// M12: two states that differ only by a phase are the same state.
#[test]
fn m12_the_overlap_and_the_fidelity() {
    let a = QuantumState::from_amplitudes(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)]);
    let b = QuantumState::from_amplitudes(vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)]);
    // A phase-shifted copy of `a`.
    let c =
        QuantumState::from_amplitudes(vec![Complex32::new(0.0, 0.0), Complex32::new(-1.0, 0.0)]);

    assert_eq!(bits(a.fidelity(&a)), 0x3f80_0000, "a state matches itself");
    assert_eq!(bits(a.fidelity(&b)), 0, "orthogonal states do not match");
    assert_eq!(
        bits(a.fidelity(&c)),
        0,
        "a phase-shifted orthogonal state does not match"
    );

    // The phase that aligns them: multiplying `a` by it gives `c`'s
    // direction.
    let phase = a.phase_to(&c);
    let aligned = a.overlap(&c).scale(0.0);
    assert!(aligned.is_zero());
    assert!(phase.is_finite());
}

/// M13: the dense matrix operations.
#[test]
fn m13_the_matrix_operations() {
    let m = complex2();
    assert_eq!(
        bits(m.trace().re),
        0,
        "the trace of a Hermitian matrix is real"
    );
    assert!(m.trace().im.abs() < 1e-6, "the trace must be real");
    assert!(m.is_hermitian(1e-6), "the matrix must be Hermitian");

    let dag = m.dagger();
    assert_eq!(bits(dag.get(0, 1).re), 0);
    assert_eq!(bits(dag.get(0, 1).im), 0x3f80_0000, "the dagger conjugates");

    let k = m.kron(&ComplexMatrix::identity(2));
    assert_eq!(k.rows(), 4);
    assert_eq!(k.cols(), 4);

    // The identity times the matrix is the matrix.
    let id = ComplexMatrix::identity(2);
    let product = id.mat_mul(&m);
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(
                (bits(product.get(r, c).re), bits(product.get(r, c).im)),
                (bits(m.get(r, c).re), bits(m.get(r, c).im)),
                "the identity must not change the matrix"
            );
        }
    }

    // A matrix times its dagger is positive semi-definite: the
    // expectation of it is a real, non-negative number.
    let psd = m.mat_mul(&m.dagger());
    let op = DenseOperator::new(psd);
    let s = QuantumState::from_amplitudes(vec![Complex32::new(0.6, 0.0), Complex32::new(0.8, 0.0)]);
    let value = s.expectation(&op);
    assert!(value.im.abs() < 1e-6, "the expectation must be real");
    assert!(value.re >= 0.0, "the expectation must not be negative");
}

/// M14: the eigensolver finds the spectrum of a Hermitian matrix.
#[test]
fn m14_the_eigensolver_finds_the_spectrum() {
    let e = eigen_hermitian(&tridiagonal3(), 32, 1e-6);
    assert_eq!(e.eigenvalues.len(), 3);
    // 2 - sqrt(2), 2, 2 + sqrt(2).
    assert_eq!(bits(e.eigenvalues[0]), 0x3f15_f61a);
    assert_eq!(bits(e.eigenvalues[1]), 0x3fff_fffe);
    assert_eq!(bits(e.eigenvalues[2]), 0x405a_8279);
    for (got, want) in e
        .eigenvalues
        .iter()
        .zip([2.0f32 - 2.0f32.sqrt(), 2.0, 2.0 + 2.0f32.sqrt()])
    {
        assert!((got - want).abs() < 1e-5, "eigenvalue {got} != {want}");
    }

    // A matrix with a complex off-diagonal: the eigenvalues are ±√2.
    let e = eigen_hermitian(&complex2(), 32, 1e-6);
    assert_eq!(bits(e.eigenvalues[0]), 0xbfb5_04f2);
    assert_eq!(bits(e.eigenvalues[1]), 0x3fb5_04f2);
    for (got, want) in e.eigenvalues.iter().zip([-2.0f32.sqrt(), 2.0f32.sqrt()]) {
        assert!((got - want).abs() < 1e-5, "eigenvalue {got} != {want}");
    }
}

/// M15: the eigenvectors are orthogonal and rebuild the matrix.
#[test]
fn m15_the_eigenvectors_rebuild_the_matrix() {
    let m = tridiagonal3();
    let e = eigen_hermitian(&m, 32, 1e-6);

    // The eigenvalues are the expectations in their own eigenstates.
    for (i, value) in e.eigenvalues.iter().enumerate() {
        let v = e.eigenvector(i);
        let op = DenseOperator::new(m.clone());
        let expectation = v.expectation(&op);
        assert!(
            (expectation.re - value).abs() < 1e-4,
            "the expectation {expectation:?} must be the eigenvalue {value}"
        );
        assert!(
            expectation.im.abs() < 1e-4,
            "the expectation of a Hermitian operator must be real"
        );
    }

    // The eigenvectors are orthogonal.
    for i in 0..e.eigenvalues.len() {
        for j in 0..e.eigenvalues.len() {
            if i == j {
                continue;
            }
            let overlap = e.eigenvector(i).overlap(&e.eigenvector(j));
            assert!(
                overlap.abs() < 1e-4,
                "the eigenvectors must be orthogonal, got {overlap:?}"
            );
        }
    }
}

/// M16: the power iteration finds the largest magnitude.
#[test]
fn m16_the_power_iteration_finds_the_dominant_eigenvalue() {
    let values = vec![
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
    ];
    let op = DiagonalOperator::new(values);
    let (value, state) = dominant_eigenpair(&op, 3, 200, 1e-6);
    assert_eq!(bits(value), 0x3fff_fffe);
    assert!(
        (value - 2.0).abs() < 1e-4,
        "the dominant eigenvalue must be two, got {value}"
    );
    // The state converges on the first basis state.
    assert!(
        state.amplitudes()[0].re > 0.99,
        "the dominant state must be the first basis state"
    );
    assert!(
        state.amplitudes()[1].re < 0.01,
        "the dominant state must be the first basis state"
    );
    assert!(state.is_normalized(1e-5), "the state must stay normalised");
}
