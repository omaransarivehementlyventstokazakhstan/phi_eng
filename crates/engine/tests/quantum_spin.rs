//! Spin and particles (M57–M64, mirrored Rust↔C#).
//!
//! These tests assert the physics of the spin-1/2 degree of freedom and
//! of occupancy itself: the Pauli algebra, the Stern-Gerlach
//! measurement, the Rabi flop and the Larmor precession, the bosonic
//! ladder, the fermionic anticommutation (which *is* the Pauli
//! exclusion principle), and the Fermi gas at zero temperature.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;
use phieng_quantum::{
    anticommutator, apply_single, bernstein_vazirani, boson_annihilation, boson_creation,
    commutator, fermi_gas_energy, fermion_annihilation, fermion_creation, measure_sz,
    number_operator, pauli_x, pauli_y, pauli_z, rabi_evolve, spin_down, spin_up, spin_vector,
    up_probability, zeeman_hamiltonian, QuantumOperator, QuantumState,
};

fn close(a: &phieng_quantum::ComplexMatrix, b: &phieng_quantum::ComplexMatrix) -> f32 {
    let mut worst = 0.0f32;
    for r in 0..a.rows().max(b.rows()) {
        for c in 0..a.cols().max(b.cols()) {
            let d = a.get(r, c) - b.get(r, c);
            worst = worst.max(d.re.abs()).max(d.im.abs());
        }
    }
    worst
}

/// M57: the Pauli matrices square to I, anticommute, and cycle.
#[test]
fn m57_the_pauli_algebra() {
    let id = phieng_quantum::ComplexMatrix::identity(2);
    for p in [pauli_x(), pauli_y(), pauli_z()] {
        assert!(
            close(&p.mat_mul(&p), &id) < 1e-6,
            "every Pauli matrix squares to I"
        );
        assert_eq!(p.trace(), Complex32::zero(), "the trace must vanish");
    }

    // sigma_x sigma_y = i sigma_z, cyclically.
    let xy = pauli_x().mat_mul(&pauli_y());
    let z = pauli_z();
    for r in 0..2 {
        for c in 0..2 {
            let want = z.get(r, c) * Complex32::i();
            let d = xy.get(r, c) - want;
            assert!(
                d.re.abs() < 1e-6 && d.im.abs() < 1e-6,
                "sigma_x sigma_y must be i sigma_z"
            );
        }
    }

    // Distinct Paulis anticommute.
    let anti = anticommutator(&pauli_x(), &pauli_y());
    assert!(
        close(&anti, &phieng_quantum::ComplexMatrix::zeros(2, 2)) < 1e-6,
        "distinct Paulis must anticommute"
    );

    // The determinant is -1.
    let x = pauli_x();
    let det = x.get(0, 0) * x.get(1, 1) - x.get(0, 1) * x.get(1, 0);
    assert!(
        (det.re + 1.0).abs() < 1e-6 && det.im.abs() < 1e-6,
        "det(sigma_x) must be -1"
    );
}

/// M58: Stern-Gerlach measures spin up with the Born probability.
#[test]
fn m58_stern_gerlach() {
    // Spin up is always up; spin down never is.
    let mut rng = DeterministicRng::new(5);
    let mut up = spin_up();
    for _ in 0..8 {
        assert_eq!(
            measure_sz(&mut up, &mut rng),
            Some(true),
            "spin up must measure up"
        );
    }
    let mut down = spin_down();
    for _ in 0..8 {
        assert_eq!(
            measure_sz(&mut down, &mut rng),
            Some(false),
            "spin down must measure down"
        );
    }

    // A superposition measures up with |alpha|²: (0.6, 0.8) goes up
    // 36% of the time, and the seeded stream pins the shots.
    assert!(
        (up_probability(&QuantumState::from_amplitudes(vec![
            Complex32::new(0.6, 0.0),
            Complex32::new(0.8, 0.0)
        ])) - 0.36)
            .abs()
            < 1e-6,
        "the up probability of (0.6, 0.8) must be 0.36"
    );
    let shots = |seed: u64| {
        let mut rng = DeterministicRng::new(seed);
        let mut outcomes = Vec::new();
        for _ in 0..32 {
            let mut s = QuantumState::from_amplitudes(vec![
                Complex32::new(0.6, 0.0),
                Complex32::new(0.8, 0.0),
            ]);
            outcomes.push(measure_sz(&mut s, &mut rng));
        }
        outcomes
    };
    assert_eq!(shots(9), shots(9), "the same seed must give the same shots");
    let mixed = shots(9);
    assert!(
        mixed.contains(&Some(true)) && mixed.contains(&Some(false)),
        "a superposition must measure both ways"
    );

    // Not one spin, not a Stern-Gerlach experiment.
    assert_eq!(measure_sz(&mut QuantumState::zeros(4), &mut rng), None);
}

/// M59: the Rabi flop swaps the populations on a pi pulse.
#[test]
fn m59_the_rabi_flop() {
    let omega = 1.5f32;
    // A pi pulse: all the way over.
    let mut s = spin_up();
    assert!(rabi_evolve(&mut s, omega, std::f32::consts::PI / omega));
    assert!(
        (s.amplitudes()[1].norm_sqr() - 1.0).abs() < 1e-5,
        "a pi pulse must flip spin up to spin down"
    );
    // A 2 pi pulse: all the way back.
    assert!(rabi_evolve(&mut s, omega, std::f32::consts::PI / omega));
    assert!(
        (s.amplitudes()[0].norm_sqr() - 1.0).abs() < 1e-5,
        "a second pi pulse must flip it back"
    );
    // Halfway: fifty-fifty.
    let mut s = spin_up();
    let half_pi = std::f32::consts::FRAC_PI_2 / omega;
    assert!(rabi_evolve(&mut s, omega, half_pi));
    assert!(
        (up_probability(&s) - 0.5).abs() < 1e-5,
        "a pi/2 pulse must split the populations"
    );
    // Not one spin, not a Rabi flop.
    assert!(!rabi_evolve(&mut QuantumState::zeros(4), omega, 1.0));
}

/// M60: a uniform field splits the energies and precesses the spin.
#[test]
fn m60_the_zeeman_splitting_and_larmor_precession() {
    // Along z: the energies are ∓mu B / 2.
    let h = zeeman_hamiltonian([0.0, 0.0, 2.0], 1.0);
    let e = phieng_quantum::eigen_hermitian(&h, 32, 1e-8);
    assert!(
        (e.eigenvalues[0] + 1.0).abs() < 1e-5,
        "the lower Zeeman level must be -mu B / 2"
    );
    assert!(
        (e.eigenvalues[1] - 1.0).abs() < 1e-5,
        "the upper Zeeman level must be +mu B / 2"
    );

    // Larmor precession: |+x> under a z field rotates in the xy plane
    // at the Larmor frequency mu B, so after a quarter period it points
    // along +y.
    let mu_b = 2.0f32;
    let mut s = spin_up();
    assert!(apply_single(&mut s, &phieng_quantum::h_gate(), 0));
    let quarter = std::f32::consts::FRAC_PI_2 / mu_b;
    let mut evolved = s.clone();
    // The propagator is the Z rotation by mu B t.
    assert!(apply_single(
        &mut evolved,
        &phieng_quantum::rz_gate(mu_b * quarter),
        0
    ));
    let (x, y, z) = spin_vector(&evolved).expect("one spin");
    assert!(
        x.abs() < 1e-5,
        "after a quarter period x must vanish (x = {x})"
    );
    assert!(
        (y - 0.5).abs() < 1e-5,
        "after a quarter period the spin must point along +y (y = {y})"
    );
    assert!(
        z.abs() < 1e-5,
        "the precession must stay equatorial (z = {z})"
    );

    // The spin vector of |up> is the north pole, halved units and all.
    assert_eq!(spin_vector(&spin_up()), Some((0.0, 0.0, 0.5)));
    assert_eq!(spin_vector(&QuantumState::zeros(4)), None);
}

/// M61: the bosonic ladder climbs and descends by square roots.
#[test]
fn m61_the_bosonic_ladder() {
    let cutoff = 5usize;
    let a = boson_annihilation(cutoff);
    let adag = boson_creation(cutoff);

    // a|n> = sqrt(n) |n-1>, checked by matrix action.
    for n in 1..cutoff {
        let mut s = QuantumState::basis(cutoff, n);
        let op = phieng_quantum::DenseOperator::new(a.clone());
        let mut out = vec![Complex32::zero(); cutoff];
        op.apply_to(s.amplitudes(), &mut out);
        s = QuantumState::from_amplitudes(out);
        let want = (n as f32).sqrt();
        assert!(
            (s.amplitudes()[n - 1].re - want).abs() < 1e-6,
            "a|{n}> must be sqrt({n})|{}>",
            n - 1
        );
        for (i, amp) in s.amplitudes().iter().enumerate() {
            if i != n - 1 {
                assert_eq!(
                    amp.norm_sqr(),
                    0.0,
                    "a|{n}> must have nothing outside |{}>",
                    n - 1
                );
            }
        }
    }
    // The vacuum annihilates to nothing.
    let mut s = QuantumState::basis(cutoff, 0);
    let op = phieng_quantum::DenseOperator::new(a.clone());
    let mut out = vec![Complex32::zero(); cutoff];
    op.apply_to(s.amplitudes(), &mut out);
    s = QuantumState::from_amplitudes(out);
    assert_eq!(s.norm_sqr(), 0.0, "a|0> must be the zero vector");

    // The number operator counts: N|3> = 3|3>.
    let n_op = number_operator(cutoff);
    assert_eq!(n_op.get(3, 3), Complex32::new(3.0, 0.0));
    assert_eq!(n_op.get(2, 3), Complex32::zero());

    // Creation is the dagger of annihilation, entry for entry.
    let dagger = a.dagger();
    for r in 0..cutoff {
        for c in 0..cutoff {
            assert_eq!(
                (adag.get(r, c).re.to_bits(), adag.get(r, c).im.to_bits()),
                (dagger.get(r, c).re.to_bits(), dagger.get(r, c).im.to_bits()),
                "creation must be the dagger"
            );
        }
    }
}

/// M62: fermions anticommute, which is the Pauli principle.
#[test]
fn m62_the_fermionic_anticommutation() {
    let a = fermion_annihilation();
    let adag = fermion_creation();
    let id = phieng_quantum::ComplexMatrix::identity(2);
    let empty = phieng_quantum::ComplexMatrix::zeros(2, 2);

    // {a, a†} = I.
    assert!(
        close(&anticommutator(&a, &adag), &id) < 1e-6,
        "creation and annihilation must anticommute to I"
    );
    // a² = 0 and (a†)² = 0: no double occupancy, in either direction.
    assert!(
        close(&a.mat_mul(&a), &empty) < 1e-6,
        "annihilating twice must give zero"
    );
    assert!(
        close(&adag.mat_mul(&adag), &empty) < 1e-6,
        "creating twice must give zero"
    );

    // The exclusion principle as a state: creating on |1> vanishes.
    let mut s = QuantumState::from_amplitudes(vec![Complex32::zero(), Complex32::one()]);
    let op = phieng_quantum::DenseOperator::new(adag.clone());
    let mut out = vec![Complex32::zero(); 2];
    op.apply_to(s.amplitudes(), &mut out);
    s = QuantumState::from_amplitudes(out);
    assert_eq!(
        s.norm_sqr(),
        0.0,
        "a second fermion in the mode must vanish"
    );

    // [a, a†] is not I for fermions (it is diag(1, -1)): the bosonic
    // relation fails exactly where it should.
    let comm = commutator(&a, &adag);
    assert!(
        (comm.get(0, 0).re - 1.0).abs() < 1e-6,
        "the fermionic commutator must be diag(1, -1)"
    );
    assert!((comm.get(1, 1).re + 1.0).abs() < 1e-6);
}

/// M63: the Fermi gas fills the lowest levels first.
#[test]
fn m63_the_fermi_gas() {
    // Four levels, two particles: the two lowest.
    assert_eq!(
        fermi_gas_energy(&[0.5, 1.5, 2.5, 3.5], 2),
        2.0,
        "two fermions take the two lowest levels"
    );
    // Unsorted input fills the same way.
    assert_eq!(
        fermi_gas_energy(&[3.5, 0.5, 2.5, 1.5], 2),
        2.0,
        "the filling must not depend on the order"
    );
    // Nobody home, or nowhere to go.
    assert_eq!(fermi_gas_energy(&[0.5, 1.5], 0), 0.0);
    assert_eq!(fermi_gas_energy(&[0.5, 1.5], 3), 0.0);
    // The harmonic ladder: 0.5 + 1.5 + 2.5 for three fermions.
    assert_eq!(
        fermi_gas_energy(&[0.5, 1.5, 2.5, 3.5], 3),
        4.5,
        "three fermions in a harmonic trap"
    );
}

/// M64: a spin rotation carries |up> to |down> up to a phase.
#[test]
fn m64_spin_rotations() {
    // RY(π) takes |up> to |down>, exactly up to a sign.
    let mut s = spin_up();
    assert!(apply_single(
        &mut s,
        &phieng_quantum::ry_gate(std::f32::consts::PI),
        0
    ));
    assert!(
        s.fidelity(&spin_down()) > 1.0 - 1e-6,
        "RY(pi)|up> must be |down> up to a phase"
    );

    // The Bernstein-Vazirani oracle still works when the diffusion is
    // written with spin operators: a smoke test that the two vocabularies
    // compose (X is sigma_x, Z is sigma_z).
    assert_eq!(
        bernstein_vazirani(0b101, 3, 7),
        0b101,
        "the gate vocabulary must compose"
    );
}
