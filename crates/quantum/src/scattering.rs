//! Scattering off a rectangular barrier: the transfer matrix.
//!
//! The wave is matched plane wave to plane wave — an interface matrix
//! at each wall, a propagation matrix inside — multiplied into one
//! transfer matrix `M` with `T = 1/|M_11|^2`. Below the barrier the
//! wavenumber turns imaginary and the propagation decays: the same
//! code tunnels and flies, because the complex exponential does both.
//! The managed mirror multiplies the same matrices in the same order.

use phieng_math::complex::Complex32;

use crate::matrix::ComplexMatrix;

/// The interface matrix from wavenumber `k1` to `k2`: continuity of
/// the wave and its slope, solved for the amplitudes past the wall in
/// terms of the ones before it.
fn interface(k1: Complex32, k2: Complex32) -> ComplexMatrix {
    let ratio = k1 / k2;
    let one = Complex32::one();
    let half = Complex32::new(0.5, 0.0);
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, half * (one + ratio));
    m.set(0, 1, half * (one - ratio));
    m.set(1, 0, half * (one - ratio));
    m.set(1, 1, half * (one + ratio));
    m
}

/// The propagation matrix over `length` at wavenumber `k`: each
/// direction picks up its own phase.
///
/// The exponent is `i k L`, not `k L`: the wavenumber is not the
/// argument of the exponential, `i` times it is. For a real `k` this
/// is the oscillation; for an imaginary one the forward wave decays
/// and the backward one grows, and writing the conjugate instead
/// would decay them both and halve the tunnelling.
fn propagate(k: Complex32, length: f32) -> ComplexMatrix {
    // i * k * L and -i * k * L, as complex arguments.
    let fwd = Complex32::new(-k.im * length, k.re * length);
    let bwd = Complex32::new(k.im * length, -k.re * length);
    let mut m = ComplexMatrix::zeros(2, 2);
    m.set(0, 0, fwd.exp());
    m.set(1, 1, bwd.exp());
    m
}

/// The transmission probability through a rectangular barrier of
/// height `potential` and `width`, at `energy`, for a particle of
/// `mass` (with hbar `hbar`).
///
/// Zero energy does not propagate, so it transmits nothing; anything
/// else tunnels below the barrier top and flies above it, with the
/// resonant energies crossing at exactly one.
pub fn transmission(energy: f32, potential: f32, width: f32, mass: f32, hbar: f32) -> f32 {
    if energy <= 0.0 {
        return 0.0;
    }
    let m = mass.max(f32::MIN_POSITIVE);
    let h = hbar.max(f32::MIN_POSITIVE);
    // The wavenumbers outside and inside; inside turns imaginary below
    // the barrier top, which is what makes the wave decay there.
    let k0 = Complex32::new(2.0 * m * energy, 0.0).sqrt().scale(1.0 / h);
    let ki = Complex32::new(2.0 * m * (energy - potential), 0.0)
        .sqrt()
        .scale(1.0 / h);
    let total = interface(k0, ki)
        .mat_mul(&propagate(ki, width))
        .mat_mul(&interface(ki, k0));
    let m11 = total.get(1, 1);
    1.0 / m11.norm_sqr()
}
