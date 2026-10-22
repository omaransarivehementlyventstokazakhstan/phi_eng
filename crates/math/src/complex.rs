//! Complex arithmetic in `f32`.
//!
//! # Why this module exists
//!
//! Quantum mechanics is arithmetic over the complex numbers: a state
//! is a vector of complex amplitudes, an observable is a complex
//! matrix, and the evolution is an exponential of a complex operator.
//! The engine had no complex type at all, so this is the layer
//! everything quantum stands on.
//!
//! # The guarantee
//!
//! The complex transcendental functions here are built **only** from the
//! deterministic helpers in this crate — [`crate::trig::compute_cos_sin`],
//! [`crate::trig::atan2`], [`crate::exponential::exp`] and
//! [`crate::exponential::ln`] — and never from the platform libm. The
//! arithmetic itself uses only IEEE-754-exact operations (add, subtract,
//! multiply, divide, square root and the round-to-nearest-even
//! conversion to `f32`), in a fixed evaluation order.
//!
//! The managed C# mirror (`src/PhiEng.Math/Complex32.cs`) implements the
//! same arithmetic in the same order, so the two engines agree
//! bit-for-bit here — the same contract the rigid core relies on.
//!
//! # Precision
//!
//! The public API is `f32` (the simulation is `f32` throughout). A
//! complex transcendental composes two `f32` helpers, so it carries up
//! to two rounding steps rather than the single step of the scalar
//! helpers: the result is accurate to a couple of ULP, not to half an
//! ULP. That is below the noise floor of any `f32` simulation, and it
//! does not change the order of a scheme (a constant relative error
//! does not).
//!
//! Two places would overflow a naive formula and are written the stable
//! way instead, identically in both languages:
//!
//! * **The modulus.** `sqrt(re*re + im*im)` overflows above `|z| ~ 2^38`
//!   and underflows below `|z| ~ 2^-38`. [`Complex32::abs`] scales by
//!   the larger component first.
//! * **The division.** `(ac+bd)/(c²+d²)` overflows for large `c, d`.
//!   the division uses Smith's algorithm, which divides by the
//!   larger of the two and never forms the square.

use crate::exponential::{exp, ln};
use crate::trig::{atan2, compute_cos_sin};

/// A complex number with `f32` components.
///
/// The layout is `#[repr(C)]` so it can cross the C ABI unchanged, and
/// the size is asserted at compile time: the managed mirror declares
/// the same two fields in the same order, and a layout change that
/// broke one side would break the other silently.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex32 {
    /// The real part.
    pub re: f32,
    /// The imaginary part.
    pub im: f32,
}

// Guard the cross-language layout contract: the managed mirror's
// `Complex32` must have the same size and the same field order.
const _: () = assert!(std::mem::size_of::<Complex32>() == 8);

impl Complex32 {
    /// The complex number `re + im * i`.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Zero.
    #[inline]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// One.
    #[inline]
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// The imaginary unit.
    #[inline]
    pub const fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    /// The product with a real number.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    /// The conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// The reciprocal.
    #[inline]
    pub fn inv(self) -> Self {
        Self::one() / self
    }

    /// The squared modulus, `re*re + im*im`.
    ///
    /// Overflows above `|z| ~ 2^19`; use [`Complex32::abs`] when the
    /// magnitude itself is wanted.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// The modulus, `|z|`.
    ///
    /// The larger component is factored out first, so the result is
    /// finite for every finite input and does not lose its low bits for
    /// a very small or a very large number.
    #[inline]
    pub fn abs(self) -> f32 {
        let (re, im) = (self.re, self.im);
        let scale = re.abs().max(im.abs());
        if scale == 0.0 {
            return 0.0;
        }
        let (a, b) = (re / scale, im / scale);
        scale * (a * a + b * b).sqrt()
    }

    /// Whether both components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Whether both components are zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    /// The number `r * (cos theta + i sin theta)`.
    ///
    /// The phase comes from the deterministic trigonometry, so the
    /// result is bit-identical on every platform.
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        let (c, s) = compute_cos_sin(theta);
        Self::new(r * c, r * s)
    }

    /// The polar form `(r, theta)`, with `theta` in `(-pi, pi]`.
    #[inline]
    pub fn to_polar(self) -> (f32, f32) {
        (self.abs(), atan2(self.im, self.re))
    }

    /// The phase factor `cos(theta) + i sin(theta)`.
    ///
    /// This is the workhorse: every gate, every Fourier twiddle and
    /// every time-evolution step is a product of phase factors.
    #[inline]
    pub fn exp_i(theta: f32) -> Self {
        let (c, s) = compute_cos_sin(theta);
        Self::new(c, s)
    }

    /// The complex exponential, `e^z`.
    ///
    /// `e^(re) * (cos im + i sin im)`, with both factors from the
    /// deterministic helpers.
    #[inline]
    pub fn exp(self) -> Self {
        let m = exp(self.re);
        let (c, s) = compute_cos_sin(self.im);
        Self::new(m * c, m * s)
    }

    /// The principal logarithm, `ln z`.
    ///
    /// `ln|z| + i arg(z)`, with the argument in `(-pi, pi]` — the
    /// principal branch, so the branch cut is the negative real axis.
    /// The logarithm of zero is `-inf` with a zero argument.
    #[inline]
    pub fn ln(self) -> Self {
        let r = self.abs();
        Self::new(ln(r), atan2(self.im, self.re))
    }

    /// The principal square root.
    ///
    /// The root with a non-negative real part. The naive
    /// `sqrt((r+re)/2)` loses its low bits when the real part is
    /// negative and the imaginary part is small, so there the accurate
    /// `|im| / (2u)` identity is used instead.
    #[inline]
    pub fn sqrt(self) -> Self {
        let r = self.abs();
        if r == 0.0 {
            return Self::zero();
        }
        let mut t = ((r + self.re) * 0.5).sqrt();
        let u = ((r - self.re) * 0.5).sqrt();
        if self.re < 0.0 && u > 0.0 {
            // t*u = |im|/2 exactly, and t is the cancelled half here.
            t = self.im.abs() / (2.0 * u);
        }
        let y = if self.im < 0.0 { -u } else { u };
        Self::new(t, y)
    }

    /// The complex sine.
    ///
    /// `sin z = (e^(iz) - e^(-iz)) / (2i)`, with `e^(-iz)` the
    /// conjugate of `e^(iz)`. The imaginary part is written as a
    /// subtraction from zero rather than a negation, so a real argument
    /// gives `+0.0` there and not `-0.0` — the two compare equal but
    /// have different bit patterns, and a bit pattern is what the
    /// pinned hashes see.
    #[inline]
    pub fn sin(self) -> Self {
        let e = Self::new(-self.im, self.re).exp();
        let d = e - e.conj();
        Self::new(d.im * 0.5, 0.0 - d.re * 0.5)
    }

    /// The complex cosine.
    ///
    /// `cos z = (e^(iz) + e^(-iz)) / 2`.
    #[inline]
    pub fn cos(self) -> Self {
        let e = Self::new(-self.im, self.re).exp();
        let s = e + e.conj();
        Self::new(s.re * 0.5, s.im * 0.5)
    }
}

impl std::ops::Add for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Div for Complex32 {
    type Output = Self;
    /// Smith's algorithm: the divisor is scaled by its larger component
    /// first, so the intermediate never overflows and never underflows
    /// to a spurious zero. Dividing by exactly zero gives a NaN, which
    /// is the safe answer: an infinity in an amplitude would poison
    /// the state it lands in.
    ///
    /// The two branches differ in more than the ratio. With `r = d/c`
    /// the denominator is `(c² + d²)/c`, so the numerator has to be
    /// `(a + b r) / (b - a r)`; with `r = c/d` the denominator is
    /// `(c² + d²)/d`, so the numerator has to be `(a r + b) / (b r - a)`.
    /// Swapping the two gives a quotient that is close to the right
    /// answer in one of the two branches and wrong in the other — and
    /// wrong in a way that is easy to miss, because the wrong branch
    /// still returns a finite number.
    #[inline]
    fn div(self, o: Self) -> Self {
        let (c, d) = (o.re, o.im);
        if c.abs() >= d.abs() {
            if c == 0.0 {
                // Both components are zero: the quotient is a NaN.
                return Self::new(f32::NAN, f32::NAN);
            }
            let r = d / c;
            let den = c + d * r;
            Self::new((self.re + self.im * r) / den, (self.im - self.re * r) / den)
        } else {
            let r = c / d;
            let den = c * r + d;
            Self::new((self.re * r + self.im) / den, (self.im * r - self.re) / den)
        }
    }
}

impl std::ops::Neg for Complex32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Mul<f32> for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Complex32::scale(self, s)
    }
}

#[cfg(test)]
mod tests {
    use super::Complex32;

    /// The complex arithmetic's pinned bit patterns.
    ///
    /// These exact bit patterns are produced by the implementation above
    /// and must be identical on every platform — that is the whole point
    /// of the module. If this test fails, the arithmetic changed (a
    /// different evaluation order, a different compiler contraction, or
    /// a different libm), and every pinned quantum result in the repo is
    /// invalidated.
    #[test]
    fn complex_pinned_bit_patterns() {
        let a = Complex32::new(1.5, -2.25);
        let b = Complex32::new(-0.75, 3.5);
        assert_eq!(
            ((a + b).re.to_bits(), (a + b).im.to_bits()),
            (0x3f400000, 0x3fa00000),
            "the sum is not portable"
        );
        assert_eq!(
            ((a - b).re.to_bits(), (a - b).im.to_bits()),
            (0x40100000, 0xc0b80000),
            "the difference is not portable"
        );
        assert_eq!(
            ((a * b).re.to_bits(), (a * b).im.to_bits()),
            (0x40d80000, 0x40de0000),
            "the product is not portable"
        );
        assert_eq!(
            ((a / b).re.to_bits(), (a / b).im.to_bits()),
            (0xbf33d30b, 0xbe8e5c68),
            "the quotient is not portable"
        );
        assert_eq!(a.abs().to_bits(), 0x402d1104, "the modulus is not portable");
        assert_eq!(
            (a.inv().re.to_bits(), a.inv().im.to_bits()),
            (0x3e520d21, 0x3e9d89d9),
            "the reciprocal is not portable"
        );

        // The transcendentals.
        // (z, exp, ln, sqrt), each a pair of bit patterns.
        type Case = (Complex32, (u32, u32), (u32, u32), (u32, u32));
        let cases: [Case; 5] = [
            (
                Complex32::new(1.0, 0.0),
                (0x402df854, 0x0),
                (0x0, 0x0),
                (0x3f800000, 0x0),
            ),
            (
                Complex32::new(0.0, 1.0),
                (0x3f0a5140, 0x3f576aa4),
                (0x0, 0x3fc90fdb),
                (0x3f3504f3, 0x3f3504f3),
            ),
            (
                Complex32::new(2.0, -3.5),
                (0xc0dd6ccc, 0x4025e2a0),
                (0x3fb2701d, 0xbf869c79),
                (0x3fde46e1, 0xbf80fe06),
            ),
            (
                Complex32::new(-0.75, -0.125),
                (0x3eeff6f4, 0xbd7138eb),
                (0xbe8c4774, 0xc03e7e0f),
                (0x3d934b6b, 0xbf5e7739),
            ),
            (
                Complex32::new(3.0, std::f32::consts::FRAC_PI_4),
                (0x41633dee, 0x41633dee),
                (0x3f90dd76, 0x3e831959),
                (0x3fdf9009, 0x3e663c13),
            ),
        ];
        for (z, exp, ln, sqrt) in cases {
            let e = z.exp();
            assert_eq!(
                (e.re.to_bits(), e.im.to_bits()),
                exp,
                "exp({z:?}) is not portable"
            );
            let l = z.ln();
            assert_eq!(
                (l.re.to_bits(), l.im.to_bits()),
                ln,
                "ln({z:?}) is not portable"
            );
            let s = z.sqrt();
            assert_eq!(
                (s.re.to_bits(), s.im.to_bits()),
                sqrt,
                "sqrt({z:?}) is not portable"
            );
        }

        // The phase factor.
        let phase = Complex32::exp_i(1.0);
        assert_eq!(
            (phase.re.to_bits(), phase.im.to_bits()),
            (0x3f0a5140, 0x3f576aa4),
            "exp_i(1.0) is not portable"
        );
    }
}
