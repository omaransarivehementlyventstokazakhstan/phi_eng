//! Three-dimensional grids, separable Hamiltonians, and the
//! Aharonov-Bohm ring (M73–M80, mirrored Rust↔C#).
//!
//! These tests lift the one-dimensional machinery onto three-dimensional
//! grids: the grid bookkeeping (M73), the separable spectrum of the cubic
//! box (M74) and of the harmonic trap (M75), the tensor-product ground
//! state (M76), the matrix-free-versus-dense cross-check (M77), and the
//! magnetic ring — zero flux (M78), quarter flux (M79), and flux
//! periodicity (M80).

use phieng_math::complex::Complex32;
use phieng_quantum::{
    eigen_hermitian, Grid1D, Grid3D, Hamiltonian1D, Hamiltonian3D, Potential1D, QuantumState,
    RingHamiltonian,
};

fn bits(v: f32) -> u32 {
    v.to_bits()
}

fn box_1d(n: usize) -> Hamiltonian1D {
    Hamiltonian1D::new(Grid1D::new(n, 0.0, 1.0), Potential1D::Zero, 1.0, 1.0)
}

fn trap_1d(n: usize) -> Hamiltonian1D {
    Hamiltonian1D::new(
        Grid1D::new(n, -6.0, 6.0),
        Potential1D::Harmonic { omega: 1.0 },
        1.0,
        1.0,
    )
}

/// M73: the three-dimensional grid maps cells to unknowns and back.
#[test]
fn m73_the_grid_maps_cells_to_unknowns_and_back() {
    let grid = Grid3D::new([6, 6, 6], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    // Four interior points per axis: 64 unknowns.
    assert_eq!(grid.dirichlet_unknowns(), 64);
    assert_eq!(grid.shape(), (6, 6, 6));
    // The flat index runs x-fastest and round-trips through coordinates.
    assert_eq!(grid.index(4, 3, 2), Some(4 + 6 * (3 + 6 * 2)));
    assert_eq!(grid.coords(4 + 6 * (3 + 6 * 2)), Some((4, 3, 2)));
    for flat in 0..grid.n() {
        let (ix, iy, iz) = grid.coords(flat).expect("every cell has coordinates");
        assert_eq!(grid.index(ix, iy, iz), Some(flat));
    }
    // Unknown zero is the first interior cell; the boundary owns none.
    assert_eq!(grid.unknown_to_grid(0), Some((1, 1, 1)));
    assert_eq!(grid.grid_to_unknown(1, 1, 1), Some(0));
    assert_eq!(grid.grid_to_unknown(0, 1, 1), None);
    assert_eq!(grid.grid_to_unknown(5, 5, 5), None);
    assert_eq!(grid.unknown_to_grid(64), None);
    // A non-cubic grid multiplies its interior extents.
    let thin = Grid3D::new([4, 5, 6], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert_eq!(thin.dirichlet_unknowns(), 2 * 3 * 4);
    // The node positions tile the box from the corner.
    let p = grid.position(0);
    assert_eq!(
        (bits(p.0), bits(p.1), bits(p.2)),
        (bits(0.0), bits(0.0), bits(0.0))
    );
    let q = grid.position(grid.index(1, 1, 1).expect("interior exists"));
    assert_eq!(
        (bits(q.0), bits(q.1), bits(q.2)),
        (bits(0.2), bits(0.2), bits(0.2))
    );
}

/// M74: the cubic box spectrum is the sum of three particle-in-a-box
/// spectra, with the first excited level triply degenerate.
#[test]
fn m74_the_cubic_box_sums_three_wells() {
    let h = Hamiltonian3D::separable(
        Grid3D::new([81, 81, 81], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        box_1d(81),
        box_1d(81),
        box_1d(81),
    )
    .expect("identical axes must fit");
    let e = h.separable_energies(4);
    assert_eq!(e.len(), 4);
    let exact = std::f32::consts::PI.powi(2) / 2.0;
    assert_eq!(bits(e[0]), 0x416c_d70e);
    assert!((e[0] - 3.0 * exact).abs() / (3.0 * exact) < 1e-3);
    // The (1,1,2) permutations: bit-identical, near 6 * pi^2 / 2.
    assert_eq!(bits(e[1]), bits(e[2]));
    assert_eq!(bits(e[2]), bits(e[3]));
    assert!((e[1] - 6.0 * exact).abs() / (6.0 * exact) < 1e-3);
}

/// M75: the harmonic trap ladders add the same way, with the same
/// triple degeneracy at the first excited level.
#[test]
fn m75_the_harmonic_trap_ladders_add() {
    let h = Hamiltonian3D::separable(
        Grid3D::new([151, 151, 151], [-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]),
        trap_1d(151),
        trap_1d(151),
        trap_1d(151),
    )
    .expect("identical axes must fit");
    let e = h.separable_energies(4);
    assert_eq!(bits(e[0]), 0x3fbf_ec68);
    assert!((e[0] - 1.5).abs() / 1.5 < 1e-3);
    assert_eq!(bits(e[1]), bits(e[2]));
    assert_eq!(bits(e[2]), bits(e[3]));
    assert!((e[1] - 2.5).abs() / 2.5 < 1e-3);
}

/// M76: the separable ground state is normalised and carries the
/// separable ground energy.
#[test]
fn m76_the_ground_state_is_a_normalised_product() {
    let h = Hamiltonian3D::separable(
        Grid3D::new([21, 21, 21], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        box_1d(21),
        box_1d(21),
        box_1d(21),
    )
    .expect("identical axes must fit");
    let (energy, state) = h.separable_ground_state();
    assert_eq!(bits(energy), 0x416c_6244);
    let mut norm = 0.0f32;
    for a in state.amplitudes().iter() {
        norm += a.norm_sqr();
    }
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "the product must be normalised, got {norm}"
    );
    assert_eq!(bits(energy), bits(h.separable_energies(1)[0]));
}

/// M77: the matrix-free operator agrees with its dense matrix, which is
/// exactly Hermitian, and the tensor ground state matches the dense
/// diagonalisation.
#[test]
fn m77_the_operator_matches_its_dense_matrix() {
    let h = Hamiltonian3D::separable(
        Grid3D::new([4, 4, 4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        box_1d(4),
        box_1d(4),
        box_1d(4),
    )
    .expect("identical axes must fit");
    let dense = h.dense_matrix();
    assert_eq!((dense.rows(), dense.cols()), (8, 8));
    // Exactly Hermitian: the columns come from one symmetric stencil.
    for r in 0..8 {
        for c in 0..8 {
            assert_eq!(dense.get(r, c).conj(), dense.get(c, r));
        }
    }
    // Matrix-free versus dense on the all-ones vector.
    let full = vec![Complex32::one(); 8];
    let mut via_apply = vec![Complex32::zero(); 8];
    h.apply_to(&full, &mut via_apply);
    let mut worst = 0.0f32;
    for (r, expected) in via_apply.iter().enumerate() {
        let mut s = Complex32::zero();
        for (c, f) in full.iter().enumerate() {
            s = s + dense.get(r, c) * *f;
        }
        let d = *expected - s;
        worst = worst.max(d.re.abs()).max(d.im.abs());
    }
    assert!(
        worst < 1e-5,
        "matrix-free versus dense disagrees by {worst}"
    );
    // The tensor product is the dense ground state.
    let (_, tensor) = h.separable_ground_state();
    let eigen = eigen_hermitian(&dense, 64, 1e-8);
    let mut amps = vec![Complex32::zero(); 8];
    for (i, a) in amps.iter_mut().enumerate() {
        *a = eigen.eigenvectors.get(i, 0);
    }
    let mut dense_state = QuantumState::zeros(8);
    dense_state.amplitudes_mut().copy_from_slice(&amps);
    assert!(
        tensor.fidelity(&dense_state) > 1.0 - 1e-6,
        "the tensor product must be the dense ground state"
    );
}

/// M78: the ring at zero flux keeps the free spectrum, and the link
/// phase is the quarter-turn-free Peierls factor of one.
#[test]
fn m78_the_ring_at_zero_flux_is_free() {
    let ring = RingHamiltonian::new(64, 1.0, 1.0, 1.0, 0.0);
    // The zero-flux link phase is exactly one.
    assert_eq!(ring.link_phase(), Complex32::new(1.0, 0.0));
    let e = ring.eigenvalues();
    assert_eq!(e.len(), 64);
    assert_eq!(bits(e[0]), 0xb543_1663);
    assert!(e[0].abs() < 1e-3, "the zero-flux ground state sits at zero");
    let c = ring.hopping();
    let first = 2.0 * c * (1.0 - (2.0 * std::f32::consts::PI / 64.0).cos());
    assert!((e[1] - first).abs() < 1e-3);
}

/// M79: a quarter flux quantum lifts the ground state to the
/// Aharonov-Bohm value 2c(1 - cos(pi/128)).
#[test]
fn m79_a_quarter_flux_lifts_the_ground_state() {
    let ring = RingHamiltonian::new(64, 1.0, 1.0, 1.0, 0.25);
    // The quarter-flux link phase is cos(pi/128) to the bit.
    assert_eq!(bits(ring.link_phase().re), 0x3f7f_ec43);
    let e = ring.eigenvalues();
    assert_eq!(bits(e[0]), 0x3cff_feb0);
    let formula = 2.0 * ring.hopping() * (1.0 - (2.0 * std::f32::consts::PI * -0.25 / 64.0).cos());
    assert!((e[0] - formula).abs() < 1e-3);
}

/// M80: threading one more flux quantum changes nothing — the spectrum
/// is periodic in the flux.
#[test]
fn m80_the_spectrum_is_periodic_in_the_flux() {
    let a = RingHamiltonian::new(64, 1.0, 1.0, 1.0, 0.3).eigenvalues();
    let b = RingHamiltonian::new(64, 1.0, 1.0, 1.0, 1.3).eigenvalues();
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 0.01, "flux periodicity breaks by {worst}");
}
