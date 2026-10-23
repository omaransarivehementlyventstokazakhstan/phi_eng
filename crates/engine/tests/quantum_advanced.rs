//! Many-body and variational methods (M81–M88, mirrored Rust↔C#).
//!
//! These tests close the ladder: the tight-binding band (M81) and the
//! SSH edge modes (M82), tunnelling through a barrier by the transfer
//! matrix (M83), the Hubbard dimer exactly and by Hartree-Fock (M84),
//! the entanglement spectrum that DMRG truncates (M85), the
//! parameter-shift gradient (M86), the VQE loop it drives (M87), and
//! the Berry phase of a qubit loop (M88).

use phieng_math::complex::Complex32;
use phieng_quantum::{
    apply_single, berry_phase, descend, eigen_hermitian, entanglement_spectrum,
    hubbard_dimer_exact, hubbard_dimer_rhf, parameter_shift_gradient, rx_gate, ry_gate, ssh_chain,
    tfi_chain, tight_binding_chain, transmission, DenseOperator, QuantumState,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// M81: the periodic chain carries the cosine band from -2t to +2t.
#[test]
fn m81_the_chain_carries_the_cosine_band() {
    let e = eigen_hermitian(&tight_binding_chain(32, 1.0, true), 64, 1e-8).eigenvalues;
    assert_eq!(e.len(), 32);
    assert_eq!(bits(e[0]), 0xbfff_fffb);
    assert!((e[0] + 2.0).abs() < 1e-5, "the band bottom sits at -2t");
    let width = e[31] - e[0];
    assert_eq!(bits(width), 0x407f_fff5);
    assert!((width - 4.0).abs() < 1e-5, "the bandwidth is 4t");
}

/// M82: the topological SSH chain shelters two near-zero edge modes;
/// the trivial one leaves the gap empty.
#[test]
fn m82_the_ssh_chain_shelters_edge_modes() {
    let e = eigen_hermitian(&ssh_chain(16, 0.5, 1.0), 100, 1e-8).eigenvalues;
    assert_eq!(e.len(), 32);
    // The middle pair: split only by their overlap across the chain.
    assert_eq!(bits(e[15]), 0xb73f_dd73);
    assert_eq!(bits(e[16]), 0x3740_8aaa);
    assert!(e[15].abs() < 1e-3 && e[16].abs() < 1e-3);
    // The bulk gap stays near 2(w - v).
    let gap = e[17] - e[14];
    assert_eq!(bits(gap), 0x3f85_7290);
    assert!((gap - 1.0).abs() / 1.0 < 5e-2);
    // A mid-gap state lives on the ends: uniform would put 4/32 here.
    let solved = eigen_hermitian(&ssh_chain(16, 0.5, 1.0), 100, 1e-8);
    let mut ends = 0.0f32;
    for i in [0usize, 1, 30, 31] {
        ends += solved.eigenvectors.get(i, 15).norm_sqr();
    }
    assert!(
        ends > 0.5,
        "the edge mode must live on the ends, got {ends}"
    );
    // The trivial phase (v > w) has no states near zero.
    let t = eigen_hermitian(&ssh_chain(16, 1.0, 0.5), 100, 1e-8).eigenvalues;
    let min_abs = t.iter().map(|x| x.abs()).fold(f32::INFINITY, f32::min);
    assert_eq!(bits(min_abs), 0x3f03_d80e);
    assert!(min_abs > 0.2, "the trivial chain stays gapped");
}

/// M83: the transfer matrix flies over free space, tunnels under the
/// barrier, crosses resonances at one, and stops at zero energy.
#[test]
fn m83_the_transfer_matrix_tunnels_and_flies() {
    // Free space transmits everything.
    let free = transmission(1.0, 0.0, 3.0, 1.0, 1.0);
    assert_eq!(bits(free), 0x3f80_0001);
    assert!((free - 1.0).abs() < 1e-5);
    // Tunnelling matches the analytic barrier formula.
    let t = transmission(1.0, 2.0, 3.0, 1.0, 1.0);
    assert_eq!(bits(t), 0x3a58_6d1f);
    let kappa_width = 2.0f32.sqrt() * 3.0;
    let analytic = 1.0 / (1.0 + kappa_width.sinh().powi(2));
    assert!((t - analytic).abs() / analytic < 1e-4);
    // At k'a = pi the barrier turns transparent.
    let resonant = 2.0 + std::f32::consts::PI.powi(2) / 18.0;
    let r = transmission(resonant, 2.0, 3.0, 1.0, 1.0);
    assert_eq!(bits(r), 0x3f80_0000);
    assert!((r - 1.0).abs() < 1e-6);
    // Zero energy does not propagate.
    assert_eq!(transmission(0.0, 2.0, 3.0, 1.0, 1.0), 0.0);
    assert_eq!(transmission(-1.0, 2.0, 3.0, 1.0, 1.0), 0.0);
}

/// M84: Hartree-Fock on the Hubbard dimer is variational — above the
/// exact answer by the correlation energy it cannot see.
#[test]
fn m84_hartree_fock_is_variational_on_the_dimer() {
    let exact = hubbard_dimer_exact(1.0, 2.0);
    assert_eq!(bits(exact[0]), 0xbf9e_3779);
    assert!((exact[0] - (1.0 - 5.0f32.sqrt())).abs() < 1e-6);
    let (rhf, density) = hubbard_dimer_rhf(1.0, 2.0, 50);
    assert_eq!(bits(rhf), 0xbf80_0000);
    assert!(
        rhf >= exact[0] - 1e-6,
        "mean-field must stay above the exact ground state"
    );
    assert!(
        rhf - exact[0] > 0.1,
        "mean-field must miss correlation energy"
    );
    // The dimer stays symmetric: one electron per site.
    assert!((density.get(0, 0).re - 1.0).abs() < 1e-6);
    assert!((density.get(1, 1).re - 1.0).abs() < 1e-6);
}

/// M85: half of the Ising ground state is entangled, and keeping two
/// Schmidt states drops almost nothing — the DMRG truncation.
#[test]
fn m85_the_entanglement_spectrum_truncates_cleanly() {
    let solved = eigen_hermitian(&tfi_chain(4, 1.0, 1.0), 100, 1e-8);
    let mut ground = QuantumState::zeros(16);
    for (i, a) in ground.amplitudes_mut().iter_mut().enumerate() {
        *a = solved.eigenvectors.get(i, 0);
    }
    let spectrum = entanglement_spectrum(&ground, &[0, 1], 4);
    assert_eq!(spectrum.len(), 4);
    assert_eq!(bits(spectrum[0]), 0x3f6a_ea49);
    assert_eq!(bits(spectrum[1]), 0x3da8_9780);
    assert_eq!(bits(spectrum[2]), 0x3821_f4f7);
    assert_eq!(bits(spectrum[3]), 0x366a_4083);
    // A spectrum, largest first, summing to one.
    for pair in spectrum.windows(2) {
        assert!(pair[0] >= pair[1]);
    }
    let sum: f32 = spectrum.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Entangled (no product state), yet two states nearly suffice.
    assert!(spectrum[0] < 0.99);
    assert!(spectrum[2] + spectrum[3] < 1e-3);
}

/// M86: the parameter shift differentiates a Pauli rotation exactly.
#[test]
fn m86_the_parameter_shift_is_exact() {
    let z = DenseOperator::new({
        let mut m = phieng_quantum::ComplexMatrix::zeros(2, 2);
        m.set(0, 0, Complex32::new(1.0, 0.0));
        m.set(1, 1, Complex32::new(-1.0, 0.0));
        m
    });
    let expectation = |theta: f32| {
        let mut s = QuantumState::basis(2, 0);
        apply_single(&mut s, &rx_gate(theta), 0);
        s.expectation(&z).re
    };
    let gradient = parameter_shift_gradient(&expectation, 0.7);
    assert_eq!(bits(gradient), 0xbf24_eb72);
    assert!((gradient + 0.7f32.sin()).abs() < 1e-6);
}

/// M87: the VQE loop descends to the exact ground energy.
#[test]
fn m87_the_vqe_loop_finds_the_ground_state() {
    let h = DenseOperator::new({
        let mut m = phieng_quantum::ComplexMatrix::zeros(2, 2);
        m.set(0, 1, Complex32::new(1.0, 0.0));
        m.set(1, 0, Complex32::new(1.0, 0.0));
        m.set(0, 0, Complex32::new(0.5, 0.0));
        m.set(1, 1, Complex32::new(-0.5, 0.0));
        m
    });
    let energy = |p: &[f32]| {
        let mut s = QuantumState::basis(2, 0);
        apply_single(&mut s, &rx_gate(p[0]), 0);
        apply_single(&mut s, &ry_gate(p[1]), 0);
        s.expectation(&h).re
    };
    let gradient = |p: &[f32]| {
        vec![
            parameter_shift_gradient(&|t| energy(&[t, p[1]]), p[0]),
            parameter_shift_gradient(&|t| energy(&[p[0], t]), p[1]),
        ]
    };
    let (_, found) = descend(&[0.3, -0.2], 200, 0.3, &energy, &gradient);
    assert_eq!(bits(found), 0xbf8f_1bbd);
    assert!((found + 1.25f32.sqrt()).abs() < 1e-3);
}

/// M88: a qubit looped around a cone picks up minus half the solid
/// angle.
#[test]
fn m88_the_qubit_loop_picks_up_the_berry_phase() {
    let gamma = berry_phase(std::f32::consts::PI / 3.0, 64);
    assert_eq!(bits(gamma), 0xbfc8_f0d8);
    assert!((gamma + std::f32::consts::PI / 2.0).abs() < 2e-3);
}
