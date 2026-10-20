//! Deterministic trigonometry.
//!
//! # Why this module exists
//!
//! The platform libm is **not portable**. `sinf`, `cosf` and `atan2f`
//! disagree between glibc (Linux) and the Microsoft CRT (Windows) in the
//! last ULP, and the *same* platform can disagree with itself: LLVM may
//! rewrite an inlined `sin_cos` into separate `sin`/`cos` calls, or fuse
//! it differently depending on the surrounding code.
//!
//! That is enough to break a rigid-body trajectory. Measured on the
//! canonical 2D determinism scene, the rotation matrix of one body
//! diverged by 1 ULP at step 45 (from `atan2` + `Mat2::from_angle`), and
//! the contact solver amplified that into a completely different
//! trajectory by step 240 — with *identical* inputs on both sides:
//!
//! ```text
//! call #44 (identical inputs):  m00=3f7fffe3 m10=3af21c75 w=4033ec0c
//!                               -> newang=3d3859d4   (Windows AND Linux)
//! call #45 (result differs):    m10=bd3849e5 (Windows) vs bd3849e6 (Linux)
//! ```
//!
//! The 3D scene survived only because it barely rotates; that is luck,
//! not a guarantee.
//!
//! # The guarantee
//!
//! The helpers below use only operations that IEEE-754 specifies
//! exactly — add, subtract, multiply, divide, square root and the
//! round-to-nearest-even conversion to `f32` — with a fixed evaluation
//! order. They never call the platform libm. Consequently they return
//! **bit-identical results on every platform**, which is what makes the
//! absolute trajectory hashes in `crates/engine/tests/determinism_hash.rs`
//! portable anchors rather than per-OS constants.
//!
//! The managed C# mirror (`src/PhiEng.Math/DeterministicTrig.cs`)
//! implements the same arithmetic in the same order, so the two engines
//! agree bit-for-bit here as well — unlike the platform `MathF`/`Math`
//! functions they replace.
//!
//! # Precision
//!
//! The public API is `f32` (the simulation is `f32` throughout), but the
//! evaluation is carried out in `f64` and rounded once at the end. The
//! `f64` scratch keeps the range reduction and the polynomial error far
//! below the `f32` rounding step, so for the angles a simulation
//! actually uses the result is the correctly rounded `f32` value - at
//! least as accurate as the platform libm it replaces.
//!
//! Past about 1e7 radians the *input* has already lost its low bits,
//! and no implementation can recover them: the reduction becomes
//! accurate only to the precision of the `f32` angle itself. That is a
//! property of the input, not of this implementation, and it matches
//! what the platform libm does with the same `f32`.
//!
//! `f64` is safe here because it is *scratch*: no `f64` value is ever
//! stored in the simulation state, so nothing can accumulate in double
//! precision across frames.

/// `pi`, rounded to `f64`.
const PI: f64 = std::f64::consts::PI;
/// `pi / 4`, rounded to `f64`.
const QUARTER_PI: f64 = std::f64::consts::FRAC_PI_4;
/// `tan(pi / 8) = sqrt(2) - 1`, rounded to `f64`.
///
/// Used to split the `atan` argument reduction so the Taylor series
/// only ever sees `|z| <= tan(pi/8)`.
const TAN_PI_8: f64 = 0.41421356237309515;
/// `2 / pi`, rounded to `f64`.
const INVPIO2: f64 = std::f64::consts::FRAC_2_PI;

/// `pi / 2` split into a high part with the low 32 mantissa bits
/// cleared and the residual.
///
/// `n * PIO2_HI` is then exactly representable in `f64` for
/// `|n| < 2^20`, and the subtraction `x - n * PIO2_HI` is exact
/// (Sterbenz), so the range reduction carries no error of its own.
const PIO2_HI: f64 = 1.570796012878418;
/// The residual `pi / 2 - PIO2_HI`.
const PIO2_LO: f64 = 3.13916478589249e-07;

/// `pi` rounded to `f32`.
const PI_F32: f32 = std::f32::consts::PI;
/// `pi / 2` rounded to `f32`.
const HALF_PI_F32: f32 = std::f32::consts::FRAC_PI_2;

// Taylor coefficients for `sin(r) = r * (1 + r^2 * p(r^2))` on
// `[-pi/4, pi/4]`. Truncation error at `r = pi/4` is ~7.5e-12.
const SIN_C1: f64 = -0.16666666666666666;
const SIN_C2: f64 = 0.008333333333333333;
const SIN_C3: f64 = -0.0001984126984126984;
const SIN_C4: f64 = 2.7557319223985893e-06;
const SIN_C5: f64 = -2.505210838544172e-08;

// Taylor coefficients for `cos(r) = 1 + r^2 * q(r^2)` on
// `[-pi/4, pi/4]`. Truncation error at `r = pi/4` is ~4.2e-13.
const COS_C1: f64 = -0.5;
const COS_C2: f64 = 0.041666666666666664;
const COS_C3: f64 = -0.001388888888888889;
const COS_C4: f64 = 2.48015873015873e-05;
const COS_C5: f64 = -2.755731922398589e-07;
const COS_C6: f64 = 2.08767569878681e-09;

// Taylor coefficients for `atan(z) = z * (1 + z^2 * p(z^2))`, used only
// for `|z| <= tan(pi/8)`. Truncation error at the limit is ~4.4e-10.
const ATAN_C1: f64 = -0.3333333333333333;
const ATAN_C2: f64 = 0.2;
const ATAN_C3: f64 = -0.14285714285714285;
const ATAN_C4: f64 = 0.1111111111111111;
const ATAN_C5: f64 = -0.09090909090909091;
const ATAN_C6: f64 = 0.07692307692307693;
const ATAN_C7: f64 = -0.06666666666666667;
const ATAN_C8: f64 = 0.058823529411764705;
const ATAN_C9: f64 = -0.05263157894736842;

/// Deterministic `sin` and `cos` of `angle` (radians).
///
/// Returns `(cos, sin)`, matching the `sin_cos` convention. The result
/// is bit-identical on every platform — see the [module docs](self).
///
/// # Examples
///
/// ```
/// use phieng_math::trig::compute_cos_sin;
///
/// let (c, s) = compute_cos_sin(0.0);
/// assert_eq!((c, s), (1.0, 0.0));
/// ```
pub fn compute_cos_sin(angle: f32) -> (f32, f32) {
    if !angle.is_finite() {
        return (f32::NAN, f32::NAN);
    }

    // Reduce to r in [-pi/4, pi/4] with a quadrant index.
    let a = angle as f64;
    let n = (a * INVPIO2).round();
    let r = (a - n * PIO2_HI) - n * PIO2_LO;

    // sin(r) = r * (1 + z * p(z)), cos(r) = 1 + z * q(z), z = r^2.
    let z = r * r;
    let mut p = SIN_C5;
    p = SIN_C4 + z * p;
    p = SIN_C3 + z * p;
    p = SIN_C2 + z * p;
    p = SIN_C1 + z * p;
    let s = r * (1.0 + z * p);
    let mut q = COS_C6;
    q = COS_C5 + z * q;
    q = COS_C4 + z * q;
    q = COS_C3 + z * q;
    q = COS_C2 + z * q;
    q = COS_C1 + z * q;
    let c = 1.0 + z * q;

    // Quadrant: n is an exact integer, so the remainder is exact too.
    let rem = n % 4.0;
    let quadrant = if rem < 0.0 { rem + 4.0 } else { rem };
    match quadrant {
        1.0 => (-s as f32, c as f32),
        2.0 => (-c as f32, -s as f32),
        3.0 => (s as f32, -c as f32),
        _ => (c as f32, s as f32),
    }
}

/// Deterministic `atan2(y, x)` — the angle of the vector `(x, y)` in
/// radians, in `(-pi, pi]`.
///
/// The result is bit-identical on every platform — see the
/// [module docs](self).
///
/// # Examples
///
/// ```
/// use phieng_math::trig::atan2;
///
/// assert_eq!(atan2(0.0, 1.0), 0.0);
/// assert_eq!(atan2(1.0, 0.0), std::f32::consts::FRAC_PI_2);
/// ```
pub fn atan2(y: f32, x: f32) -> f32 {
    if !y.is_finite() || !x.is_finite() {
        return f32::NAN;
    }

    let y_negative = y.is_sign_negative();

    // Degenerate axes, matching the usual IEEE-754 conventions.
    if y == 0.0 {
        return if x > 0.0 {
            // IEEE-754: atan2(+-0, +x) is +-0, preserving the sign of
            // the zero. The platform libm does the same.
            if y_negative {
                -0.0
            } else {
                0.0
            }
        } else if x < 0.0 {
            if y_negative {
                -PI_F32
            } else {
                PI_F32
            }
        } else if y_negative {
            -0.0
        } else {
            0.0
        };
    }
    if x == 0.0 {
        return if y > 0.0 { HALF_PI_F32 } else { -HALF_PI_F32 };
    }

    let (yd, xd) = (y as f64, x as f64);
    let ax = xd.abs();
    let ay = yd.abs();

    // Scale first so the squares cannot overflow, then use the
    // half-angle identity atan2(y, x) = 2 * atan(y / (r + |x|)) with
    // r = hypot(x, y). The scale cancels in the ratio.
    let scale = ax.max(ay);
    let sx = ax / scale;
    let sy = yd / scale;
    let r = (sx * sx + sy * sy).sqrt();
    let t = sy / (r + sx); // in (-1, 1], carries the sign of y

    let mut angle = 2.0 * atan_unit(t);
    if xd < 0.0 {
        // Left half-plane: reflect across the y axis.
        angle = if t >= 0.0 { PI - angle } else { -PI - angle };
    }
    angle as f32
}

/// `atan` of a value already known to lie in `[-1, 1]`.
fn atan_unit(t: f64) -> f64 {
    if t > TAN_PI_8 {
        // atan(t) = pi/4 + atan((t - 1) / (t + 1))
        QUARTER_PI + atan_poly((t - 1.0) / (t + 1.0))
    } else if t < -TAN_PI_8 {
        // atan(t) = -pi/4 + atan((t + 1) / (1 - t))
        -QUARTER_PI + atan_poly((t + 1.0) / (1.0 - t))
    } else {
        atan_poly(t)
    }
}

/// `atan(z)` for `|z| <= tan(pi/8)`.
fn atan_poly(z: f64) -> f64 {
    let z2 = z * z;
    let mut p = ATAN_C9;
    p = ATAN_C8 + z2 * p;
    p = ATAN_C7 + z2 * p;
    p = ATAN_C6 + z2 * p;
    p = ATAN_C5 + z2 * p;
    p = ATAN_C4 + z2 * p;
    p = ATAN_C3 + z2 * p;
    p = ATAN_C2 + z2 * p;
    p = ATAN_C1 + z2 * p;
    z * (1.0 + z2 * p)
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
    /// is in ULPs of the result, and is far looser than the actual
    /// agreement (which is ~0.5 ULP, i.e. correctly rounded).
    ///
    /// The comparison is always against the libm evaluated at the *same*
    /// `f32` input: rounding the angle to `f32` first and comparing
    /// against a `f64` evaluation would measure the input rounding, not
    /// the implementation.
    #[test]
    fn trig_matches_libm_within_tolerance() {
        let mut worst_cos = 0.0f64;
        let mut worst_sin = 0.0f64;
        let mut worst_atan = 0.0f64;

        // A dense sweep over several turns.
        for i in -60000..=60000 {
            let a = (i as f32) * 0.000_1;
            let (c, s) = compute_cos_sin(a);
            let want_c = (a as f64).cos() as f32;
            let want_s = (a as f64).sin() as f32;
            worst_cos = worst_cos.max(ulp_diff(c, want_c));
            worst_sin = worst_sin.max(ulp_diff(s, want_s));
        }
        // Large angles, where the range reduction does the work.
        for i in -20000..=20000 {
            let a = (i as f32) * 7.0;
            let (c, s) = compute_cos_sin(a);
            let want_c = (a as f64).cos() as f32;
            let want_s = (a as f64).sin() as f32;
            worst_cos = worst_cos.max(ulp_diff(c, want_c));
            worst_sin = worst_sin.max(ulp_diff(s, want_s));
        }
        // Right around the quadrant boundaries, where the reduction is
        // most delicate.
        for k in 0..64 {
            let base = (k as f32) * std::f32::consts::FRAC_PI_2;
            for d in -4i32..=4 {
                let a = base + (d as f32) * f32::EPSILON * base.abs().max(1.0);
                let (c, s) = compute_cos_sin(a);
                let want_c = (a as f64).cos() as f32;
                let want_s = (a as f64).sin() as f32;
                worst_cos = worst_cos.max(ulp_diff(c, want_c));
                worst_sin = worst_sin.max(ulp_diff(s, want_s));
            }
        }

        for i in -20000..=20000 {
            let y = (i as f32) * 0.01;
            for x in [1.0f32, -1.0, 0.5, -0.25, 3.0, -7.5, 1e-4, -1e-4] {
                let got = atan2(y, x);
                let want = ((y as f64).atan2(x as f64)) as f32;
                worst_atan = worst_atan.max(ulp_diff(got, want));
            }
        }
        // Near the axes, where atan2 has branch boundaries.
        for i in 1..2000 {
            let e = (i as f32) * 1e-7;
            for (y, x) in [(e, 1.0f32), (-e, 1.0), (1.0, e), (1.0, -e), (-1.0, e)] {
                let got = atan2(y, x);
                let want = ((y as f64).atan2(x as f64)) as f32;
                worst_atan = worst_atan.max(ulp_diff(got, want));
            }
        }

        assert!(
            worst_cos <= 2.0,
            "cos deviates by {worst_cos} ULP from the platform libm"
        );
        assert!(
            worst_sin <= 2.0,
            "sin deviates by {worst_sin} ULP from the platform libm"
        );
        assert!(
            worst_atan <= 4.0,
            "atan2 deviates by {worst_atan} ULP from the platform libm"
        );
    }

    /// Special values must behave like the usual libm conventions.
    #[test]
    fn trig_special_cases() {
        assert_eq!(compute_cos_sin(0.0), (1.0, 0.0));
        assert_eq!(compute_cos_sin(-0.0), (1.0, -0.0));

        let (c, s) = compute_cos_sin(std::f32::consts::FRAC_PI_2);
        assert!(c.abs() < 1e-6, "cos(pi/2) = {c}");
        assert!((s - 1.0).abs() < 1e-6, "sin(pi/2) = {s}");

        let (c, s) = compute_cos_sin(std::f32::consts::PI);
        assert!((c + 1.0).abs() < 1e-6, "cos(pi) = {c}");
        assert!(s.abs() < 1e-6, "sin(pi) = {s}");

        // Non-finite input stays non-finite (never panics, never wraps).
        assert!(compute_cos_sin(f32::INFINITY).0.is_nan());
        assert!(compute_cos_sin(f32::NEG_INFINITY).1.is_nan());
        assert!(compute_cos_sin(f32::NAN).0.is_nan());

        assert_eq!(atan2(0.0, 1.0).to_bits(), 0.0f32.to_bits());
        // The sign of a zero must survive: `0.0 == -0.0` is true, so the
        // comparison has to be on the bits.
        assert_eq!(atan2(-0.0, 1.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(atan2(1.0, 0.0), std::f32::consts::FRAC_PI_2);
        assert_eq!(atan2(-1.0, 0.0), -std::f32::consts::FRAC_PI_2);
        assert_eq!(atan2(0.0, -1.0), std::f32::consts::PI);
        assert_eq!(atan2(-0.0, -1.0), -std::f32::consts::PI);
        assert!(atan2(f32::INFINITY, 1.0).is_nan());
        assert!(atan2(1.0, f32::NAN).is_nan());
    }

    /// Round-trip: recovering the angle from a rotation matrix and
    /// rebuilding it must land back on the same matrix. This is exactly
    /// the operation the 2D body rotation integration performs.
    ///
    /// The round trip is *not* bit-exact: the recovered angle is an
    /// `f32`, so it carries up to half an ULP of the angle, which shows
    /// up in the rebuilt matrix. That quantisation is inherent to the
    /// angle representation and is the behaviour the engine has always
    /// had; the point of this test is that it is now the *same*
    /// quantisation on every platform.
    ///
    /// The bound is absolute rather than relative: near `cos = 0` a
    /// relative measure is meaningless, but the *absolute* error stays
    /// at the `f32` noise floor (~1e-7) because the angle error is what
    /// drives it.
    #[test]
    fn trig_round_trips_through_a_rotation_matrix() {
        let mut worst = 0.0f32;
        for i in -500..=500 {
            let a = (i as f32) * 0.01;
            let (c, s) = compute_cos_sin(a);
            // Matrix convention: x_axis = (cos, sin), y_axis = (-sin, cos).
            // The engine recovers the angle as atan2(-y_axis.x, x_axis.x),
            // i.e. atan2(sin, cos) here.
            let recovered = atan2(s, c);
            let (c2, s2) = compute_cos_sin(recovered);
            worst = worst.max((c2 - c).abs()).max((s2 - s).abs());
        }
        assert!(
            worst <= 1e-6,
            "rotation round-trip drifts by {worst} (angle quantisation)"
        );
    }

    /// Cross-platform anchor.
    ///
    /// These exact bit patterns are produced by the implementation
    /// above and must be identical on every platform — that is the
    /// whole point of the module. If this test fails, the arithmetic
    /// changed (a different evaluation order, a different compiler
    /// contraction, or a different libm), and every pinned trajectory
    /// hash in the repo is invalidated.
    #[test]
    fn trig_pinned_bit_patterns() {
        let cos_sin_cases: [(f32, u32, u32); 8] = [
            (0.0, 0x3f800000, 0x00000000),
            (0.5, 0x3f60a940, 0x3ef57744),
            (1.0, 0x3f0a5140, 0x3f576aa4),
            (-1.0, 0x3f0a5140, 0xbf576aa4),
            (std::f32::consts::FRAC_PI_2, 0xb33bbd2e, 0x3f800000),
            (std::f32::consts::PI, 0xbf800000, 0xb3bbbd2e),
            (3.0, 0xbf7d7026, 0x3e1081c3),
            (-7.25, 0x3f11637b, 0xbf52b56e),
        ];
        for (angle, cos_bits, sin_bits) in cos_sin_cases {
            let (c, s) = compute_cos_sin(angle);
            assert_eq!(
                (c.to_bits(), s.to_bits()),
                (cos_bits, sin_bits),
                "compute_cos_sin({angle}) is not portable"
            );
        }

        let atan_cases: [(f32, f32, u32); 8] = [
            (0.0, 1.0, 0x00000000),
            (1.0, 0.0, 0x3fc90fdb),
            (-1.0, 0.0, 0xbfc90fdb),
            (0.0, -1.0, 0x40490fdb),
            (1.0, 1.0, 0x3f490fdb),
            (-3.5, 2.25, 0xbf7fdc89),
            (0.125, -0.75, 0x403e7e0f),
            (-1e-3, 1e-3, 0xbf490fdb),
        ];
        for (y, x, bits) in atan_cases {
            assert_eq!(
                atan2(y, x).to_bits(),
                bits,
                "atan2({y}, {x}) is not portable"
            );
        }
    }
}
