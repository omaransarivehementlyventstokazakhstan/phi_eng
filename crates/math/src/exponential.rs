//! Deterministic exponential and logarithm.
//!
//! # Why this module exists
//!
//! The platform libm is **not portable**. `expf` and `logf` disagree
//! between glibc (Linux) and the Microsoft CRT (Windows) in the last
//! ULP, exactly like `sinf`, `cosf` and `atan2f` do — see
//! [`crate::trig`] for the full story and the measurement that proved
//! it.
//!
//! That matters here because the nuclear subsystem decays quantities
//! exponentially (`activity *= exp(-lambda * dt)`). A last-ULP
//! difference in `exp` would make a half-life drift, and a chain
//! reaction is a *feedback loop*: the tiniest difference decides
//! whether the reaction is subcritical or supercritical.
//!
//! # The guarantee
//!
//! Like [`crate::trig`], the helpers below use only operations that
//! IEEE-754 specifies exactly — add, subtract, multiply, divide, the
//! extraction of an exponent from a bit pattern, and the
//! round-to-nearest-even conversion to `f32` — with a fixed evaluation
//! order. They never call the platform libm, so they return
//! **bit-identical results on every platform**.
//!
//! The managed C# mirror (`src/PhiEng.Math/DeterministicScalar.cs`)
//! implements the same arithmetic in the same order.
//!
//! # Precision
//!
//! The public API is `f32`; the evaluation runs in `f64` scratch and is
//! rounded once at the end, which makes the result the correctly
//! rounded `f32` value for all practical inputs.

/// `ln(2)`, rounded to `f64`.
const LN2: f64 = std::f64::consts::LN_2;

/// `ln(2)` split into a high part with the low 30 mantissa bits
/// cleared and the residual.
///
/// `k * LN2_HI` is then exactly representable in `f64` for
/// `|k| < 2^22`; the `f32` range only ever needs `|k| <= 129`.
const LN2_HI: f64 = 0.6931470632553101;
/// The residual `ln(2) - LN2_HI`.
const LN2_LO: f64 = 1.1730463522763301e-07;

/// `sqrt(2)`, rounded to `f64`. Used to recentre the logarithm's
/// mantissa into `[1/sqrt(2), sqrt(2))`.
const SQRT2: f64 = std::f64::consts::SQRT_2;

// Taylor coefficients `1/n!` for `exp(r)` on `[-ln(2)/2, ln(2)/2]`.
// The truncation error at the limit is ~2e-13.
const EXP_C1: f64 = 1.0;
const EXP_C2: f64 = 0.5;
const EXP_C3: f64 = 0.16666666666666666;
const EXP_C4: f64 = 0.041666666666666664;
const EXP_C5: f64 = 0.008333333333333333;
const EXP_C6: f64 = 0.001388888888888889;
const EXP_C7: f64 = 0.0001984126984126984;
const EXP_C8: f64 = 2.48015873015873e-05;
const EXP_C9: f64 = 2.7557319223985893e-06;
const EXP_C10: f64 = 2.755731922398589e-07;

// Taylor coefficients `1/(2n+1)` for `atanh(u)` with
// `|u| <= 1 - sqrt(2)/2 ~ 0.1716`. The truncation error at the limit is
// ~1.6e-17.
const ATANH_C1: f64 = 1.0;
const ATANH_C2: f64 = 0.3333333333333333;
const ATANH_C3: f64 = 0.2;
const ATANH_C4: f64 = 0.14285714285714285;
const ATANH_C5: f64 = 0.1111111111111111;
const ATANH_C6: f64 = 0.09090909090909091;
const ATANH_C7: f64 = 0.07692307692307693;
const ATANH_C8: f64 = 0.06666666666666667;
const ATANH_C9: f64 = 0.058823529411764705;
const ATANH_C10: f64 = 0.05263157894736842;

/// Largest argument whose exponential still fits in `f32`.
/// `exp(88.8) > f32::MAX`.
const EXP_OVERFLOW: f64 = 89.0;
/// Smallest argument whose exponential still fits in `f32` as a
/// subnormal. `exp(-104) ~ 1e-45`.
const EXP_UNDERFLOW: f64 = -104.0;

/// Deterministic `exp(x)` — the natural exponential of `x`.
///
/// The result is bit-identical on every platform — see the
/// [module docs](self).
///
/// # Examples
///
/// ```
/// use phieng_math::exponential::exp;
///
/// assert_eq!(exp(0.0), 1.0);
/// ```
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == f32::NEG_INFINITY {
        return 0.0;
    }

    let xd = x as f64;
    if xd > EXP_OVERFLOW {
        return f32::INFINITY;
    }
    if xd < EXP_UNDERFLOW {
        return 0.0;
    }

    // Reduce: x = k*ln(2) + r with |r| <= ln(2)/2.
    let k = (xd / LN2).round();
    let r = (xd - k * LN2_HI) - k * LN2_LO;

    // exp(r) by Taylor series (Horner).
    let mut p = EXP_C10;
    p = EXP_C9 + r * p;
    p = EXP_C8 + r * p;
    p = EXP_C7 + r * p;
    p = EXP_C6 + r * p;
    p = EXP_C5 + r * p;
    p = EXP_C4 + r * p;
    p = EXP_C3 + r * p;
    p = EXP_C2 + r * p;
    p = EXP_C1 + r * p;
    let e = 1.0 + r * p;

    (e * pow2(k as i32)) as f32
}

/// Deterministic `ln(x)` — the natural logarithm of `x`.
///
/// Returns `NaN` for negative input and `-inf` for zero, matching the
/// usual libm conventions. The result is bit-identical on every
/// platform — see the [module docs](self).
///
/// # Examples
///
/// ```
/// use phieng_math::exponential::ln;
///
/// assert_eq!(ln(1.0), 0.0);
/// ```
pub fn ln(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }

    let xd = x as f64;
    // Decompose x = m * 2^e with m in [1, 2).
    let bits = xd.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);

    // Recentre m into [1/sqrt(2), sqrt(2)) so that
    // u = (m - 1) / (m + 1) stays small.
    if m > SQRT2 {
        m *= 0.5;
        e += 1;
    }

    // ln(m) = 2 * atanh(u), u = (m - 1) / (m + 1).
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let mut p = ATANH_C10;
    p = ATANH_C9 + u2 * p;
    p = ATANH_C8 + u2 * p;
    p = ATANH_C7 + u2 * p;
    p = ATANH_C6 + u2 * p;
    p = ATANH_C5 + u2 * p;
    p = ATANH_C4 + u2 * p;
    p = ATANH_C3 + u2 * p;
    p = ATANH_C2 + u2 * p;
    p = ATANH_C1 + u2 * p;
    let ln_m = 2.0 * u * p;

    (e as f64 * LN2 + ln_m) as f32
}

/// `2^k` for an integer `k`, constructed exactly from the exponent
/// field instead of calling `pow`, which is a libm call.
fn pow2(k: i32) -> f64 {
    // The `f32` argument range of `exp` keeps |k| inside [-150, 129],
    // well within the exactly representable window [-1022, 1023].
    let clamped = k.clamp(-1022, 1023);
    let biased = (clamped + 1023) as u64;
    f64::from_bits(biased << 52)
}

/// Distance between two `f32` values expressed in units in the last
/// place of the larger magnitude.
#[cfg(test)]
fn ulp_diff(a: f32, b: f32) -> f64 {
    let a = a as f64;
    let b = b as f64;
    if a == b {
        return 0.0;
    }
    let scale = a.abs().max(b.abs()).max(f32::MIN_POSITIVE as f64);
    (a - b).abs() / (scale * f32::EPSILON as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deterministic helpers must track the platform libm closely
    /// enough that the physics does not change character. The tolerance
    /// is in ULPs of the result and is far looser than the actual
    /// agreement (which is ~0.5 ULP, i.e. correctly rounded).
    ///
    /// The comparison is always against the libm evaluated at the *same*
    /// `f32` input.
    #[test]
    fn exponential_matches_libm_within_tolerance() {
        let mut worst_exp = 0.0f64;
        let mut worst_ln = 0.0f64;

        for i in -60000..=60000 {
            let x = (i as f32) * 0.001;
            worst_exp = worst_exp.max(ulp_diff(exp(x), (x as f64).exp() as f32));
        }
        // Large magnitudes, where the range reduction does the work.
        for i in -2000..=2000 {
            let x = (i as f32) * 0.05;
            worst_exp = worst_exp.max(ulp_diff(exp(x), (x as f64).exp() as f32));
        }
        // Around the powers of two, where the mantissa is recentred.
        for k in -120..=120 {
            let x = (1.0f32) * (2.0f32).powi(k);
            if x > 0.0 && x.is_finite() {
                worst_exp = worst_exp.max(ulp_diff(exp(x), (x as f64).exp() as f32));
                worst_ln = worst_ln.max(ulp_diff(ln(x), (x as f64).ln() as f32));
            }
        }
        for i in -60000..=60000 {
            let x = (i as f32) * 0.001 + 0.5;
            if x > 0.0 {
                worst_ln = worst_ln.max(ulp_diff(ln(x), (x as f64).ln() as f32));
            }
        }
        for i in 1..20000 {
            let x = (i as f32) * 1e-3;
            worst_ln = worst_ln.max(ulp_diff(ln(x), (x as f64).ln() as f32));
        }

        assert!(
            worst_exp <= 2.0,
            "exp deviates by {worst_exp} ULP from the platform libm"
        );
        assert!(
            worst_ln <= 2.0,
            "ln deviates by {worst_ln} ULP from the platform libm"
        );
    }

    /// Special values must behave like the usual libm conventions.
    #[test]
    fn exponential_special_cases() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert!(exp(f32::NAN).is_nan());
        assert!(exp(100.0).is_infinite());
        assert_eq!(exp(-200.0), 0.0);

        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!(ln(f32::NAN).is_nan());
    }

    /// `exp` and `ln` must be mutual inverses to within the `f32`
    /// rounding step — the property the nuclear subsystem relies on
    /// when it turns a half-life into a decay constant and back.
    #[test]
    fn exponential_round_trips() {
        for i in -2000..=2000 {
            let x = (i as f32) * 0.01;
            let back = ln(exp(x));
            assert!(
                (back - x).abs() <= x.abs() * 1e-5 + 1e-6,
                "ln(exp({x})) = {back}"
            );
        }
    }
}
