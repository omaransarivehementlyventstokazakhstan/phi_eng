//! The gates and circuits (M39–M48, mirrored Rust↔C#).
//!
//! These tests assert the *physics* of quantum computation: the gate
//! identities, the entangling gates' truth tables, the Bloch sphere,
//! the measurement rule, and the textbook algorithms — Deutsch-Jozsa,
//! Bernstein-Vazirani, Grover, the quantum Fourier transform,
//! teleportation and superdense coding.
//!
//! Qubit 0 is the least significant bit of the basis index
//! (little-endian): every test here and in the managed mirror uses
//! that order.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;
use phieng_quantum::{
    apply_cnot, apply_controlled_phase, apply_qft, apply_single, apply_swap, apply_toffoli,
    bernstein_vazirani, bloch, cnot_gate, cz_gate, deutsch_jozsa, fredkin_gate, grover_iterations,
    grover_probability, h_gate, measure, rx_gate, ry_gate, rz_gate, s_gate, superdense_send,
    swap_gate, t_gate, teleport_fidelity, toffoli_gate, u3_gate, x_gate, y_gate, z_gate,
    QuantumCircuit, QuantumOperator, QuantumState, TruthTable,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// M39: the single-qubit gates square to what they should.
#[test]
fn m39_the_single_qubit_gate_identities() {
    let id = phieng_quantum::ComplexMatrix::identity(2);
    let close = |m: &phieng_quantum::ComplexMatrix| {
        let mut worst = 0.0f32;
        for r in 0..2 {
            for c in 0..2 {
                let d = m.get(r, c) - id.get(r, c);
                worst = worst.max(d.re.abs()).max(d.im.abs());
            }
        }
        worst
    };

    // H² = X² = Y² = Z² = I.
    assert!(
        close(&h_gate().mat_mul(&h_gate())) < 1e-6,
        "H squared must be I"
    );
    assert!(
        close(&x_gate().mat_mul(&x_gate())) < 1e-6,
        "X squared must be I"
    );
    assert!(
        close(&y_gate().mat_mul(&y_gate())) < 1e-6,
        "Y squared must be I"
    );
    assert!(
        close(&z_gate().mat_mul(&z_gate())) < 1e-6,
        "Z squared must be I"
    );

    // S² = Z and T⁴ = Z (so T⁸ = I).
    let s2 = s_gate().mat_mul(&s_gate());
    let mut worst = 0.0f32;
    for r in 0..2 {
        for c in 0..2 {
            let d = s2.get(r, c) - z_gate().get(r, c);
            worst = worst.max(d.re.abs()).max(d.im.abs());
        }
    }
    assert!(worst < 1e-6, "S squared must be Z");
    let t2 = t_gate().mat_mul(&t_gate());
    let t4 = t2.mat_mul(&t2);
    let mut worst = 0.0f32;
    for r in 0..2 {
        for c in 0..2 {
            let d = t4.get(r, c) - z_gate().get(r, c);
            worst = worst.max(d.re.abs()).max(d.im.abs());
        }
    }
    assert!(worst < 1e-6, "T to the fourth must be Z");
    let t8 = t4.mat_mul(&t4);
    assert!(close(&t8) < 1e-6, "T to the eighth must be I");

    // The Hadamard's entries are exactly ±1/√2.
    assert_eq!(bits(h_gate().get(0, 0).re), 0x3f35_04f3);
    assert_eq!(bits(h_gate().get(1, 0).re), 0x3f35_04f3);
    assert_eq!(bits(h_gate().get(1, 1).re), 0xbf35_04f3);
    assert_eq!(bits(h_gate().get(0, 0).im), 0);

    // A zero rotation is the identity, whatever the axis.
    for g in [rx_gate(0.0), ry_gate(0.0), rz_gate(0.0)] {
        assert!(close(&g) < 1e-6, "a zero rotation must be I");
    }
    // RX(π) is X up to a global phase: it swaps the populations.
    let mut s = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()]);
    assert!(apply_single(&mut s, &rx_gate(std::f32::consts::PI), 0));
    assert!(
        (s.amplitudes()[0].norm_sqr()) < 1e-6 && (s.amplitudes()[1].norm_sqr() - 1.0).abs() < 1e-6,
        "RX(pi) must flip |0> to |1>"
    );

    // U3 contains the other rotations: U3(π/2, 0, π) is H up to a phase.
    let u = u3_gate(std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::PI);
    let mut a = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()]);
    let mut b = a.clone();
    assert!(apply_single(&mut a, &u, 0));
    assert!(apply_single(&mut b, &h_gate(), 0));
    assert!(
        a.fidelity(&b) > 1.0 - 1e-6,
        "U3(pi/2, 0, pi) must be H up to a phase"
    );
}

/// M40: the entangling gates' truth tables.
#[test]
fn m40_the_entangling_gates_truth_tables() {
    // CNOT with control qubit 1 and target qubit 0: |10> <-> |11>.
    for (input, output) in [(0usize, 0usize), (1, 1), (2, 3), (3, 2)] {
        let mut s = QuantumState::basis(4, input);
        assert!(apply_cnot(&mut s, 1, 0));
        let probs = s.born_probabilities();
        for (i, p) in probs.iter().enumerate() {
            let want = if i == output { 1.0 } else { 0.0 };
            assert!(
                (p - want).abs() < 1e-6,
                "CNOT|{input:02b}> must be |{output:02b}>"
            );
        }
    }
    // The matrix says the same.
    let cnot = cnot_gate();
    assert_eq!(bits(cnot.get(2, 3).re), 0x3f80_0000);
    assert_eq!(bits(cnot.get(3, 2).re), 0x3f80_0000);

    // CZ flips the phase of |11> and nothing else.
    let mut s = QuantumState::from_amplitudes(vec![
        Complex32::one(),
        Complex32::one(),
        Complex32::one(),
        Complex32::one(),
    ]);
    s.normalize();
    // It is a 4x4 gate, so the single-qubit path refuses it.
    let mut t = s.clone();
    assert!(
        !apply_single(&mut t, &cz_gate(), 0),
        "a 4x4 gate must not apply as a single-qubit gate"
    );
    let op = phieng_quantum::DenseOperator::new(cz_gate());
    let mut out = vec![Complex32::zero(); 4];
    let mut u = s.clone();
    op.apply_to(u.amplitudes(), &mut out);
    u = QuantumState::from_amplitudes(out);
    assert!(
        (u.amplitudes()[3] + s.amplitudes()[3]).norm_sqr() < 1e-12,
        "CZ must flip |11>"
    );
    assert!(
        (u.amplitudes()[0] - s.amplitudes()[0]).norm_sqr() < 1e-12,
        "CZ must spare |00>"
    );

    // SWAP exchanges |01> and |10>.
    let mut s = QuantumState::basis(4, 1);
    assert!(apply_swap(&mut s, 0, 1));
    assert!(
        (s.amplitudes()[2] - Complex32::one()).norm_sqr() < 1e-12,
        "SWAP|01> must be |10>"
    );
    let swap = swap_gate();
    assert_eq!(bits(swap.get(1, 2).re), 0x3f80_0000);

    // Toffoli flips the target only when both controls are set.
    for input in 0..8usize {
        let mut s = QuantumState::basis(8, input);
        assert!(apply_toffoli(&mut s, 2, 1, 0));
        let want = if input == 6 {
            7
        } else if input == 7 {
            6
        } else {
            input
        };
        let probs = s.born_probabilities();
        for (i, p) in probs.iter().enumerate() {
            let w = if i == want { 1.0 } else { 0.0 };
            assert!((p - w).abs() < 1e-6, "Toffoli|{input}> must be |{want}>");
        }
    }

    // A controlled phase touches only the doubly-set amplitude.
    let mut s = QuantumState::from_amplitudes(vec![
        Complex32::one(),
        Complex32::one(),
        Complex32::one(),
        Complex32::one(),
    ]);
    s.normalize();
    assert!(apply_controlled_phase(
        &mut s,
        0,
        1,
        std::f32::consts::FRAC_PI_2
    ));
    let p = std::f32::consts::FRAC_PI_2;
    let want = Complex32::exp_i(p);
    let got = s.amplitudes()[3] * Complex32::new(2.0, 0.0);
    assert!(
        (got - want).norm_sqr() < 1e-10,
        "the controlled phase must be e^(i pi/2) on |11>"
    );

    // Fredkin swaps its targets only when the control is set.
    let fredkin = fredkin_gate();
    assert_eq!(bits(fredkin.get(5, 6).re), 0x3f80_0000);
    assert_eq!(bits(fredkin.get(6, 5).re), 0x3f80_0000);
    assert_eq!(bits(fredkin.get(5, 5).re), 0);
    // The Toffoli matrix says the same as the indexed version.
    let toffoli = toffoli_gate();
    assert_eq!(bits(toffoli.get(6, 7).re), 0x3f80_0000);
    assert_eq!(bits(toffoli.get(7, 6).re), 0x3f80_0000);
    assert_eq!(bits(toffoli.get(6, 6).re), 0);
}

/// M41: the Bloch sphere.
#[test]
fn m41_the_bloch_sphere() {
    let zero = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()]);
    assert_eq!(bloch(&zero), Some((0.0, 0.0, 1.0)));

    let one = QuantumState::from_amplitudes(vec![Complex32::zero(), Complex32::one()]);
    assert_eq!(bloch(&one), Some((0.0, 0.0, -1.0)));

    // |+> = H|0> sits on the equator, on the x axis.
    let mut plus = zero.clone();
    assert!(apply_single(&mut plus, &h_gate(), 0));
    let (x, y, z) = bloch(&plus).expect("one qubit has Bloch coordinates");
    assert!((x - 1.0).abs() < 1e-6, "x = {x}");
    assert!(y.abs() < 1e-6, "y = {y}");
    assert!(z.abs() < 1e-6, "z = {z}");

    // Every pure state sits on the sphere.
    let mut rng = DeterministicRng::new(11);
    for _ in 0..16 {
        let mut s = QuantumState::zeros(2);
        for a in s.amplitudes_mut().iter_mut() {
            *a = Complex32::new(rng.next_f32_signed(), rng.next_f32_signed());
        }
        s.normalize();
        let (x, y, z) = bloch(&s).expect("one qubit");
        assert!(
            (x * x + y * y + z * z - 1.0).abs() < 1e-5,
            "pure states live on the sphere"
        );
    }

    // Two qubits have no Bloch vector.
    assert_eq!(bloch(&QuantumState::zeros(4)), None);
}

/// M42: the measurement rule.
#[test]
fn m42_the_measurement_rule() {
    // |+> measures 0 or 1 with equal probability, and the draw is
    // reproducible: the same seed gives the same shots.
    let shots = |seed: u64| {
        let mut rng = DeterministicRng::new(seed);
        let mut outcomes = Vec::new();
        for _ in 0..16 {
            let mut s = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::one()]);
            s.normalize();
            outcomes.push(measure(&mut s, 0, &mut rng).expect("qubit 0 exists"));
        }
        outcomes
    };
    let a = shots(1);
    let b = shots(1);
    assert_eq!(a, b, "the same seed must give the same shots");
    let ones = a.iter().filter(|b| **b).count();
    assert!(
        (4..=12).contains(&ones),
        "sixteen fair shots must not be degenerate ({ones} ones)"
    );
    let c = shots(2);
    assert_ne!(a, c, "a different seed must give different shots");

    // The collapse: after the measurement the state is the branch that
    // was measured, renormalised.
    let mut rng = DeterministicRng::new(3);
    let mut s =
        QuantumState::from_amplitudes(vec![Complex32::new(0.6, 0.0), Complex32::new(0.8, 0.0)]);
    let outcome = measure(&mut s, 0, &mut rng).expect("qubit 0 exists");
    let branch = if outcome { 1 } else { 0 };
    assert_eq!(bits(s.amplitudes()[branch].norm_sqr()), 0x3f80_0000);
    assert_eq!(bits(s.amplitudes()[1 - branch].norm_sqr()), 0);

    // An out-of-range qubit refuses.
    let mut s = QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()]);
    assert_eq!(measure(&mut s, 4, &mut rng), None);
}

/// M43: Deutsch-Jozsa finds constant oracles in one query.
#[test]
fn m43_deutsch_jozsa() {
    assert!(
        deutsch_jozsa(3, &TruthTable::constant(3, false), 1),
        "a constant-zero oracle must read constant"
    );
    assert!(
        deutsch_jozsa(3, &TruthTable::constant(3, true), 2),
        "a constant-one oracle must read constant"
    );
    assert!(
        !deutsch_jozsa(3, &TruthTable::first_bit(3), 3),
        "the first-bit oracle is balanced and must read balanced"
    );
    assert!(
        !deutsch_jozsa(1, &TruthTable::first_bit(1), 4),
        "one input bit is enough for a balanced oracle"
    );
}

/// M44: Bernstein-Vazirani recovers the secret in one query.
#[test]
fn m44_bernstein_vazirani() {
    for secret in [0u32, 1, 5, 21, 255] {
        let n = if secret == 0 {
            1
        } else {
            32 - secret.leading_zeros() as usize
        };
        let n = n.clamp(1, 8);
        let recovered = bernstein_vazirani(secret & ((1 << n) - 1), n, 7);
        assert_eq!(
            recovered,
            secret & ((1 << n) - 1),
            "the secret {secret} must come back"
        );
    }
}

/// M45: Grover finds the marked item.
#[test]
fn m45_grover_finds_the_marked_item() {
    assert_eq!(grover_iterations(2), 1, "four items need one iteration");
    assert_eq!(grover_iterations(3), 2, "eight items need two iterations");

    // Four items: one iteration lands exactly on the target.
    let p = grover_probability(2, 2);
    assert!(
        p >= 1.0 - 1.0 / 4.0,
        "four items must succeed with probability >= 3/4 (got {p})"
    );
    assert!(
        (p - 1.0).abs() < 1e-4,
        "four items land exactly on the target (got {p})"
    );

    // Eight items: two iterations clear 7/8.
    let p = grover_probability(3, 5);
    assert!(
        p >= 1.0 - 1.0 / 8.0,
        "eight items must succeed with probability >= 7/8 (got {p})"
    );

    // An out-of-range target is a zero probability, not a panic.
    assert_eq!(grover_probability(2, 9), 0.0);
}

/// M46: the Fourier transform and its inverse reconstruct the input.
#[test]
fn m46_the_fourier_transform_round_trips() {
    for basis in 0..8usize {
        let mut circuit = QuantumCircuit::new(3, 1);
        circuit.state_mut().amplitudes_mut()[0] = Complex32::zero();
        circuit.state_mut().amplitudes_mut()[basis] = Complex32::one();
        let start = circuit.state().clone();
        apply_qft(&mut circuit, false);
        // A basis state transforms to a flat spectrum.
        for a in circuit.state().amplitudes().iter() {
            assert!(
                (a.norm_sqr() - 1.0 / 8.0).abs() < 1e-5,
                "a basis state must transform flat"
            );
        }
        apply_qft(&mut circuit, true);
        assert!(
            circuit.state().fidelity(&start) > 1.0 - 1e-5,
            "the inverse must reconstruct basis state {basis}"
        );
    }
}

/// M47: teleportation delivers the input state with fidelity one.
#[test]
fn m47_teleportation_delivers_the_state() {
    let inputs = [
        QuantumState::from_amplitudes(vec![Complex32::one(), Complex32::zero()]),
        QuantumState::from_amplitudes(vec![Complex32::zero(), Complex32::one()]),
        QuantumState::from_amplitudes(vec![Complex32::new(0.6, 0.0), Complex32::new(0.8, 0.0)]),
    ];
    for (k, input) in inputs.iter().enumerate() {
        // Every seed: the fidelity is one whatever Alice measures.
        for seed in [1u64, 2, 3] {
            let fidelity = teleport_fidelity(input, seed);
            assert!(
                fidelity > 1.0 - 1e-5,
                "teleportation {k} with seed {seed} must deliver fidelity one (got {fidelity})"
            );
        }
    }
    assert_eq!(teleport_fidelity(&QuantumState::zeros(4), 1), 0.0);
}

/// M48: superdense coding carries two bits on one qubit.
#[test]
fn m48_superdense_coding_carries_two_bits() {
    for b0 in [false, true] {
        for b1 in [false, true] {
            assert_eq!(
                superdense_send(b0, b1),
                (b0, b1),
                "the bits ({b0}, {b1}) must round-trip"
            );
        }
    }
}
