//! Entanglement: the Bell states and the CHSH game.
//!
//! Two qubits can correlate more strongly than any classical system:
//! measure one half of a Bell pair and the other half is decided, no
//! matter how far apart they are. The CHSH game turns that slogan into
//! a number: a classical strategy wins at most 3/4 of the rounds, and
//! the quantum strategy wins cos²(pi/8) ≈ 0.854 of them. The gap
//! between the two is the Tsirelson bound, and it is the sharpest
//! experimental line between the classical and the quantum world.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

use crate::gates;
use crate::state::QuantumState;

/// Which Bell state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BellPair {
    /// `(|00> + |11>) / sqrt(2)`.
    PhiPlus,
    /// `(|00> - |11>) / sqrt(2)`.
    PhiMinus,
    /// `(|01> + |10>) / sqrt(2)`.
    PsiPlus,
    /// `(|01> - |10>) / sqrt(2)`.
    PsiMinus,
}

/// The Bell state: maximally entangled, perfectly correlated.
pub fn bell_state(which: BellPair) -> QuantumState {
    let s = crate::gates::inv_sqrt_2();
    let (a, b, c, d) = match which {
        BellPair::PhiPlus => (s, 0.0, 0.0, s),
        BellPair::PhiMinus => (s, 0.0, 0.0, -s),
        BellPair::PsiPlus => (0.0, s, s, 0.0),
        BellPair::PsiMinus => (0.0, s, -s, 0.0),
    };
    QuantumState::from_amplitudes(vec![
        Complex32::new(a, 0.0),
        Complex32::new(b, 0.0),
        Complex32::new(c, 0.0),
        Complex32::new(d, 0.0),
    ])
}

/// One round of the CHSH game, played optimally on a fresh Bell pair.
///
/// The referee draws the settings `a` and `b`; Alice measures qubit 0
/// in the Z basis (`a` false) or the X basis (`a` true), and Bob
/// measures qubit 1 in the `(Z + X) / sqrt(2)` basis (`b` false) or the
/// `(Z - X) / sqrt(2)` basis (`b` true). They win the round when
/// `x ^ y == (a && b)`: a classical strategy wins at most 3/4 of the
/// rounds, and this one wins cos²(pi/8) ≈ 0.854 of them.
fn chsh_round(rng: &mut DeterministicRng) -> bool {
    let a = rng.next_f32() < 0.5;
    let b = rng.next_f32() < 0.5;
    let mut state = bell_state(BellPair::PhiPlus);
    // Alice's measurement basis.
    if a {
        gates::apply_single(&mut state, &gates::h_gate(), 0);
    }
    // Bob's measurement basis: RY(-/+pi/4) turns the tilted axis into Z.
    let angle = if b {
        std::f32::consts::FRAC_PI_4
    } else {
        -std::f32::consts::FRAC_PI_4
    };
    gates::apply_single(&mut state, &gates::ry_gate(angle), 1);
    let x = gates::measure(&mut state, 0, rng).unwrap_or(false);
    let y = gates::measure(&mut state, 1, rng).unwrap_or(false);
    (x ^ y) == (a && b)
}

/// Plays the CHSH game for `rounds` rounds with the given seed and
/// returns the number of rounds won.
///
/// The stream is seeded, so the same seed gives the same count on
/// every platform and in both languages: the count is pinned, not just
/// bounded.
pub fn chsh_win_count(rounds: usize, seed: u64) -> usize {
    let mut rng = DeterministicRng::new(seed);
    let mut wins = 0usize;
    for _ in 0..rounds {
        if chsh_round(&mut rng) {
            wins += 1;
        }
    }
    wins
}
