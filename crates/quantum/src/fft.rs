//! The fast Fourier transform.
//!
//! # The guarantee
//!
//! The transform is the textbook iterative Cooley-Tukey: a bit
//! reversal permutation followed by `log2(n)` butterfly stages. The
//! twiddle factors are built from the deterministic trigonometry, never
//! from the platform libm, and the order of every operation is fixed —
//! so the transform gives the same bits on every platform and in both
//! languages.
//!
//! The twiddle angle is computed in `f64` and narrowed once, because
//! the angle of the last stage of a long transform loses its low bits
//! in `f32`: at `2^-20` of a radian the phase is already wrong in the
//! seventh digit, and a thousand stages compound it.
//!
//! # The size
//!
//! The length must be a power of two. A transform of any other length
//! is refused rather than approximated: a radix-3 or radix-5 stage
//! would be a second implementation to keep in step, and the quantum
//! grids are powers of two by construction (the doubling of the
//! resolution is how a wavefunction simulation is refined).

use phieng_math::complex::Complex32;
use phieng_math::trig::compute_cos_sin;

/// Whether a length is a power of two.
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && n & (n - 1) == 0
}

/// The transform, in place.
///
/// `inverse` selects the direction: the forward transform has no
/// scaling, the inverse divides by the length at the end. The two are
/// each other's inverse up to that scaling, so
/// `inverse(forward(x)) == x / n`.
///
/// Returns false when the length is not a power of two or is zero; the
/// buffer is then undefined.
pub fn transform_in_place(data: &mut [Complex32], inverse: bool) -> bool {
    let n = data.len();
    if !is_power_of_two(n) {
        return false;
    }
    if n == 1 {
        return true;
    }

    // The bit-reversal permutation: index i goes to the index whose
    // bits are i's bits reversed.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // The butterfly stages, from the smallest to the largest.
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    let mut len = 2usize;
    while len <= n {
        // The twiddle of the stage: exp(sign * 2 pi i / len).
        let angle = (sign * 2.0 * std::f64::consts::PI / len as f64) as f32;
        let (c, s) = compute_cos_sin(angle);
        let w = Complex32::new(c, s);
        let half = len / 2;
        let mut base = 0usize;
        while base < n {
            // The twiddle of this butterfly: w^j, built by multiplication
            // so the table is never materialised.
            let mut wn = Complex32::one();
            for k in 0..half {
                let u = data[base + k];
                let v = data[base + k + half] * wn;
                data[base + k] = u + v;
                data[base + k + half] = u - v;
                wn = wn * w;
            }
            base += len;
        }
        len <<= 1;
    }

    if inverse {
        // The inverse divides by the length.
        let scale = 1.0f32 / n as f32;
        for value in data.iter_mut() {
            *value = value.scale(scale);
        }
    }
    true
}

/// The transform, into a separate buffer.
///
/// `out` must be at least as long as `data`; a shorter buffer is left
/// alone and the function returns false.
pub fn transform(data: &[Complex32], inverse: bool, out: &mut [Complex32]) -> bool {
    if out.len() < data.len() {
        return false;
    }
    out[..data.len()].copy_from_slice(data);
    transform_in_place(&mut out[..data.len()], inverse)
}
