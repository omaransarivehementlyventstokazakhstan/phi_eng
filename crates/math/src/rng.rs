//! A seeded, reproducible random number generator.
//!
//! # Why this module exists
//!
//! Some parts of a simulation have to be random and still have to be
//! reproducible: sampling a measurement outcome from a wavefunction
//! (the Born rule) has to give the same answer for the same seed, or a
//! saved game cannot be replayed and a pinned trajectory hash cannot
//! exist. The engine had no general random number generator at all —
//! the noise module has its own hash-based value noise, and the
//! collision crate has a throwaway generator in its tests.
//!
//! # The algorithm
//!
//! **xoshiro256++**, seeded with **splitmix64**. The state is four
//! `u32`s, the output is a `u32`, and the mixing is exact integer
//! arithmetic — no floating point, no libm, nothing that can differ
//! between platforms. The same seed therefore gives the same stream
//! everywhere.
//!
//! The managed C# mirror (`src/PhiEng.Math/DeterministicRng.cs`)
//! implements the same generator in the same order, so the two engines
//! draw the same numbers.
//!
//! # The floating point mapping
//!
//! [`DeterministicRng::next_f32`] takes the top 24 bits of a `u32` and
//! scales them by `2^-24`, which gives a value in `[0, 1)` with every
//! bit pattern exactly representable — no rounding, so the mapping is
//! exact in both languages.

use crate::exponential::ln;
use crate::trig::compute_cos_sin;

/// The golden ratio's odd multiple, the splitmix64 increment.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
/// The splitmix64 finalizer's first multiplier.
const SPLITMIX_MUL_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// The splitmix64 finalizer's second multiplier.
const SPLITMIX_MUL_B: u64 = 0x94D0_49BB_1331_11EB;
/// `2^-24`, the scale of the 24-bit float mapping.
const TWO_POW_MINUS_24: f32 = 1.0 / 16_777_216.0;

/// A seeded, reproducible random number generator.
///
/// xoshiro256++: four `u32` words of state, a `u32` of output, exact
/// integer arithmetic throughout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicRng {
    /// The generator's state.
    s: [u32; 4],
}

impl DeterministicRng {
    /// A generator seeded with `seed`.
    ///
    /// The seed is expanded through splitmix64 into the four state
    /// words, so even neighbouring seeds give unrelated streams.
    ///
    /// # Examples
    ///
    /// ```
    /// use phieng_math::rng::DeterministicRng;
    ///
    /// let mut a = DeterministicRng::new(1);
    /// let mut b = DeterministicRng::new(1);
    /// assert_eq!(a.next_u32(), b.next_u32());
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u32; 4];
        for word in s.iter_mut() {
            let v = splitmix64(&mut sm);
            *word = ((v >> 32) as u32) ^ (v as u32);
        }
        Self { s }
    }

    /// A generator from an exact state.
    ///
    /// The state can be saved and restored, so a stream can be resumed
    /// where it left off.
    pub fn from_state(s: [u32; 4]) -> Self {
        Self { s }
    }

    /// The generator's exact state.
    pub fn state(&self) -> [u32; 4] {
        self.s
    }

    /// The next `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// The next value in `[0, 1)`, with 24 bits of resolution.
    ///
    /// The mapping is exact: every value is a multiple of `2^-24`, so
    /// no rounding happens and the two engines agree bit-for-bit.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * TWO_POW_MINUS_24
    }

    /// The next value in `[-1, 1)`, with 24 bits of resolution.
    #[inline]
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// The next value in `[0, bound)`, without modulo bias.
    ///
    /// Returns zero for a bound of zero. A draw that would land outside
    /// the range is rejected and retaken, so the distribution is exact.
    pub fn next_below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        let zone = u32::MAX - (u32::MAX % bound);
        loop {
            let v = self.next_u32();
            if v < zone {
                return v % bound;
            }
        }
    }

    /// The next value from the standard normal distribution.
    ///
    /// The Box–Muller transform, with the logarithm and the trigonometry
    /// from the deterministic helpers, so the result is reproducible
    /// bit-for-bit. One draw gives one value: the transform's second
    /// output is discarded rather than cached, which keeps the stream
    /// position a simple function of the number of draws.
    pub fn next_standard_normal(&mut self) -> f32 {
        // u1 must be positive for the logarithm; the generator can
        // produce exactly zero, so it is nudged off it.
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let r = (-2.0 * ln(u1)).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let (c, _s) = compute_cos_sin(theta);
        r * c
    }
}

/// One step of splitmix64: the seed expander.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL_A);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL_B);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::DeterministicRng;

    /// The generator's pinned stream.
    ///
    /// These exact values are produced by the implementation above and
    /// must be identical on every platform. If this test fails, the
    /// generator changed, and every pinned quantum result in the repo is
    /// invalidated.
    #[test]
    fn rng_pinned_bit_patterns() {
        let mut rng = DeterministicRng::new(1);
        let expected: [u32; 8] = [
            0x7c64370d, 0x62ffbc74, 0x058b01af, 0x248ebb73, 0x96863623, 0xfc417b82, 0x591b1abf,
            0x7359a0d8,
        ];
        for want in expected {
            assert_eq!(rng.next_u32(), want, "the stream is not portable");
        }
    }
}
