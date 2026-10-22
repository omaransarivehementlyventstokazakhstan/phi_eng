//! Entanglement and quantum information (M65–M72, mirrored Rust↔C#).
//!
//! These tests assert the physics of correlated systems: the Bell
//! states and their stabilizers, the reduced state of half a pair, the
//! von Neumann entropy, the CHSH game with its Tsirelson bound, the
//! noisy channels, and the Lindblad master equation.

use phieng_math::complex::Complex32;
use phieng_quantum::{
    amplitude_damping_kraus, bell_state, chsh_win_count, dephasing_kraus, depolarizing_kraus,
    lindblad_step, BellPair, ComplexMatrix, DenseOperator, DensityMatrix, QuantumState,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// M65: the Bell states are normalized, orthogonal, and stabilized by
/// XX and ZZ.
#[test]
fn m65_the_bell_states() {
    let states = [
        bell_state(BellPair::PhiPlus),
        bell_state(BellPair::PhiMinus),
        bell_state(BellPair::PsiPlus),
        bell_state(BellPair::PsiMinus),
    ];
    for s in states.iter() {
        assert!(
            (s.norm_sqr() - 1.0).abs() < 1e-6,
            "every Bell state must be normalized"
        );
    }
    for (i, a) in states.iter().enumerate() {
        for (j, b) in states.iter().enumerate() {
            let fidelity = a.fidelity(b);
            if i == j {
                assert!(
                    (fidelity - 1.0).abs() < 1e-6,
                    "a Bell state must match itself"
                );
            } else {
                assert_eq!(fidelity, 0.0, "distinct Bell states must be orthogonal");
            }
        }
    }

    // |Φ+> is stabilized by XX and ZZ: both expectations are +1.
    let phi = bell_state(BellPair::PhiPlus);
    let mut xx = ComplexMatrix::zeros(4, 4);
    for (r, c, v) in [(0, 3, 1.0f32), (1, 2, 1.0), (2, 1, 1.0), (3, 0, 1.0)] {
        xx.set(r, c, Complex32::new(v, 0.0));
    }
    let mut zz = ComplexMatrix::zeros(4, 4);
    for (r, v) in [(0, 1.0f32), (1, -1.0), (2, -1.0), (3, 1.0)] {
        zz.set(r, r, Complex32::new(v, 0.0));
    }
    let op_xx = DenseOperator::new(xx);
    let op_zz = DenseOperator::new(zz);
    assert!(
        (phi.expectation(&op_xx).re - 1.0).abs() < 1e-5,
        "XX must stabilize |Φ+>"
    );
    assert!(
        (phi.expectation(&op_zz).re - 1.0).abs() < 1e-5,
        "ZZ must stabilize |Φ+>"
    );
}

/// M66: half a Bell pair is maximally mixed.
#[test]
fn m66_half_a_bell_pair_is_maximally_mixed() {
    let bell = bell_state(BellPair::PhiPlus);
    let reduced = DensityMatrix::from_pure(&bell).partial_trace(&[0], 2);
    assert_eq!(reduced.dim(), 2);
    let m = reduced.matrix();
    // Each diagonal is one half, up to the rounding of the amplitudes.
    assert_eq!(bits(m.get(0, 0).re), 0x3eff_ffff);
    assert_eq!(bits(m.get(1, 1).re), 0x3eff_ffff);
    assert_eq!(bits(m.get(0, 1).re), 0);
    assert_eq!(bits(m.get(0, 1).im), 0);
    assert!(
        (reduced.purity() - 0.5).abs() < 1e-6,
        "half a pair must have purity one half"
    );

    // Half a product state stays pure.
    let product = QuantumState::from_amplitudes(vec![
        Complex32::one(),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::zero(),
    ]);
    let reduced = DensityMatrix::from_pure(&product).partial_trace(&[0], 2);
    assert!(
        (reduced.purity() - 1.0).abs() < 1e-6,
        "half a product state must stay pure"
    );
}

/// M67: the von Neumann entropy is zero for pure states and one bit
/// for half a pair.
#[test]
fn m67_the_von_neumann_entropy() {
    let pure = DensityMatrix::from_pure(&QuantumState::from_amplitudes(vec![
        Complex32::one(),
        Complex32::zero(),
    ]));
    assert_eq!(pure.von_neumann_entropy(), 0.0);

    let bell = bell_state(BellPair::PhiPlus);
    let reduced = DensityMatrix::from_pure(&bell).partial_trace(&[0], 2);
    let entropy = reduced.von_neumann_entropy();
    let ln2 = phieng_math::exponential::ln(2.0);
    assert!(
        (entropy - ln2).abs() < 1e-5,
        "half a pair must carry one bit of entropy (got {entropy})"
    );

    // The maximally mixed qubit carries one bit too.
    let mixed = DensityMatrix::maximally_mixed(2);
    assert!(
        (mixed.von_neumann_entropy() - ln2).abs() < 1e-5,
        "the maximally mixed qubit must carry one bit"
    );
}

/// M68: the CHSH game beats every classical strategy.
#[test]
fn m68_the_chsh_game() {
    // Two thousand rounds on a fixed seed: the count is pinned, not
    // just bounded.
    assert_eq!(chsh_win_count(2000, 77), 1696);
    // Deterministic twice over.
    assert_eq!(chsh_win_count(2000, 77), 1696);
    // A different seed plays a different game.
    assert_ne!(chsh_win_count(2000, 78), 1696);

    // The rate clears the classical bound of 3/4 with room to spare.
    let rate = chsh_win_count(2000, 77) as f32 / 2000.0;
    assert!(
        rate > 0.75,
        "the quantum strategy must beat the classical bound (rate {rate})"
    );
    assert!(
        (rate - 0.8536).abs() < 0.02,
        "the rate must sit at the Tsirelson value cos²(pi/8) (rate {rate})"
    );
}

/// M69: the noisy channels do what they say.
#[test]
fn m69_the_noisy_channels() {
    let zero = DensityMatrix::from_pure(&QuantumState::from_amplitudes(vec![
        Complex32::one(),
        Complex32::zero(),
    ]));
    let one = DensityMatrix::from_pure(&QuantumState::from_amplitudes(vec![
        Complex32::zero(),
        Complex32::one(),
    ]));

    // Full depolarisation is white noise, whatever went in.
    let white = zero.apply_channel(&depolarizing_kraus(1.0));
    assert!(
        (white.matrix().get(0, 0).re - 0.5).abs() < 1e-6,
        "full depolarisation must halve the population"
    );
    assert!(
        (white.matrix().get(1, 1).re - 0.5).abs() < 1e-6,
        "full depolarisation must halve the population"
    );
    // No depolarisation is the identity.
    let same = zero.apply_channel(&depolarizing_kraus(0.0));
    assert!(
        (same.matrix().get(0, 0).re - 1.0).abs() < 1e-6,
        "zero depolarisation must change nothing"
    );

    // Full amplitude damping sends |1> to |0>, and spares |0>.
    let decayed = one.apply_channel(&amplitude_damping_kraus(1.0));
    assert!(
        (decayed.matrix().get(0, 0).re - 1.0).abs() < 1e-6,
        "full damping must decay |1> to |0>"
    );
    let spared = zero.apply_channel(&amplitude_damping_kraus(1.0));
    assert!(
        (spared.matrix().get(0, 0).re - 1.0).abs() < 1e-6,
        "damping must spare |0>"
    );

    // Full dephasing kills the coherence and keeps the populations.
    let plus = DensityMatrix::from_pure(&{
        let mut s = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::one()]);
        s.normalize();
        s
    });
    let dephased = plus.apply_channel(&dephasing_kraus(0.5));
    assert!(
        dephased.matrix().get(0, 1).norm_sqr() < 1e-12,
        "full dephasing must kill the coherence"
    );
    assert!(
        (dephased.matrix().get(0, 0).re - 0.5).abs() < 1e-6,
        "dephasing must keep the populations"
    );
}

/// M70: the Lindblad step matches the hand computation and preserves
/// the trace.
#[test]
fn m70_the_lindblad_step() {
    // H = 0, one jump Z at rate 1, rho = |+><+|, dt = 0.01: the
    // off-diagonal decays by exactly the rate times the step.
    let mut plus = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::one()]);
    plus.normalize();
    let mut rho = DensityMatrix::from_pure(&plus);
    let h = ComplexMatrix::zeros(2, 2);
    let z = phieng_quantum::z_gate();
    assert!(lindblad_step(&mut rho, &h, &[(1.0, z)], 0.01));
    let m = rho.matrix();
    assert_eq!(bits(m.get(0, 0).re), 0x3eff_ffff);
    assert_eq!(bits(m.get(0, 1).re), 0x3efa_e147);
    assert_eq!(bits(m.get(1, 0).re), 0x3efa_e147);
    assert_eq!(bits(m.get(1, 1).re), 0x3eff_ffff);
    // The trace is preserved up to the rounding: the two terms that
    // cancel are summed in a different order.
    assert!(
        (rho.trace().re - 1.0).abs() < 1e-6,
        "the trace must be preserved"
    );

    // A hundred steps: the trace is still one, and the coherence has
    // decayed to e^-1 within the Euler error.
    for _ in 0..99 {
        assert!(lindblad_step(
            &mut rho,
            &h,
            &[(1.0, phieng_quantum::z_gate())],
            0.01
        ));
    }
    assert!(
        (rho.trace().re - 1.0).abs() < 1e-5,
        "the trace must survive a hundred steps"
    );
    let coherence = rho.matrix().get(0, 1).re;
    // A Z jump dephases at twice its rate: LρL† and -{L†L,ρ}/2 move the
    // off-diagonal the same way. A hundred steps at dt = 0.01 land near
    // 0.5 e^-2, and the Euler error is a few percent.
    let decayed = 0.5 * (-2.0f32).exp();
    assert!(
        (coherence - decayed).abs() < 0.03 * decayed.abs().max(1e-6),
        "the coherence must decay to e^-2 within 3% (got {coherence})"
    );

    // Dimension mismatches refuse rather than guess.
    let mut rho = DensityMatrix::maximally_mixed(2);
    let big = ComplexMatrix::zeros(3, 3);
    assert!(!lindblad_step(&mut rho, &big, &[], 0.01));
}

/// M71: mixed states interpolate between pure and white noise.
#[test]
fn m71_mixed_states() {
    // A pure state has purity one.
    let pure = DensityMatrix::from_pure(&QuantumState::from_amplitudes(vec![
        Complex32::new(0.6, 0.0),
        Complex32::new(0.8, 0.0),
    ]));
    assert!((pure.purity() - 1.0).abs() < 1e-6, "a pure state is pure");
    assert!(pure.is_valid(1e-6), "a pure state is valid");

    // White noise over d levels has purity 1/d.
    let white = DensityMatrix::maximally_mixed(4);
    assert!(
        (white.purity() - 0.25).abs() < 1e-6,
        "white noise over four levels has purity one quarter"
    );
    assert!(white.is_valid(1e-6), "white noise is valid");

    // A non-Hermitian matrix is not a state.
    let mut bad = ComplexMatrix::zeros(2, 2);
    bad.set(0, 0, Complex32::new(0.5, 0.0));
    bad.set(1, 1, Complex32::new(0.5, 0.0));
    bad.set(0, 1, Complex32::new(0.25, 0.0));
    assert!(
        !DensityMatrix::from_matrix_unchecked(bad).is_valid(1e-6),
        "a non-Hermitian matrix must fail validation"
    );
}

/// M72: the partial trace over three qubits and the diagonal rule.
#[test]
fn m72_partial_traces_and_probabilities() {
    // |010>: tracing out qubits 0 and 2 leaves |1> on qubit 1.
    let mut amps = vec![Complex32::zero(); 8];
    amps[2] = Complex32::one();
    let s = QuantumState::from_amplitudes(amps);
    let middle = DensityMatrix::from_pure(&s).partial_trace(&[1], 3);
    assert_eq!(middle.dim(), 2);
    assert!(
        (middle.matrix().get(1, 1).re - 1.0).abs() < 1e-6,
        "the middle qubit of |010> must be |1>"
    );
    assert!(middle.matrix().get(0, 0).re.abs() < 1e-6);

    // Keeping two qubits of |000> + |111> over sqrt(2): the survivor is
    // half |00> and half |11>. The amplitude is the correctly rounded
    // 1/sqrt(2), not a decimal literal.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let ghz = QuantumState::from_amplitudes(vec![
        Complex32::new(s, 0.0),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::zero(),
        Complex32::new(s, 0.0),
    ]);
    let pair = DensityMatrix::from_pure(&ghz).partial_trace(&[0, 1], 3);
    assert!(
        (pair.matrix().get(0, 0).re - 0.5).abs() < 1e-5,
        "the surviving pair must be half |00>"
    );
    assert!(
        (pair.matrix().get(3, 3).re - 0.5).abs() < 1e-5,
        "the surviving pair must be half |11>"
    );

    // The diagonal is the Born rule, in either language of the API.
    let probs = DensityMatrix::from_pure(&ghz).probabilities();
    assert_eq!(probs.len(), 8);
    assert!(
        (probs[0] + probs[7] - 1.0).abs() < 1e-6,
        "the diagonal must carry the whole probability"
    );
}
