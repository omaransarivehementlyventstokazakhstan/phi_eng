namespace PhiEng.Math;

/// <summary>
/// Deterministic exponential and logarithm.
/// </summary>
/// <remarks>
/// <para>
/// The platform math library is <b>not portable</b>: <c>MathF.Exp</c>
/// and <c>MathF.Log</c> disagree between the Microsoft CRT (Windows)
/// and the Unix libm (Linux) in the last ULP, exactly like
/// <c>MathF.Sin</c>, <c>MathF.Cos</c> and <c>MathF.Atan2</c> do.
/// </para>
/// <para>
/// That matters because the nuclear subsystem decays quantities
/// exponentially (<c>activity *= exp(-lambda * dt)</c>). A last-ULP
/// difference in <c>exp</c> would make a half-life drift, and a chain
/// reaction is a <i>feedback loop</i>: the tiniest difference decides
/// whether the reaction is subcritical or supercritical.
/// </para>
/// <para>
/// The helpers below use only operations that IEEE-754 specifies
/// exactly - add, subtract, multiply, divide, the extraction of an
/// exponent from a bit pattern, and the round-to-nearest-even
/// conversion back to <see cref="float"/> - with a fixed evaluation
/// order. They never call the platform math library, so they return
/// <b>bit-identical results on every platform</b>. This mirrors
/// <c>phieng-math</c>'s <c>exponential</c> module operation for
/// operation, so the Rust and C# engines agree bit-for-bit here.
/// </para>
/// <para>
/// The public API is <see cref="float"/>; the evaluation runs in
/// <see cref="double"/> scratch and is rounded once at the end, which
/// makes the result the correctly rounded <see cref="float"/> value for
/// all practical inputs.
/// </para>
/// </remarks>
public static class DeterministicScalar
{
    /// <summary>ln(2), rounded to <see cref="double"/>.</summary>
    private const double LN2 = 0.6931471805599453;

    /// <summary>
    /// ln(2) split into a high part with the low 30 mantissa bits
    /// cleared and the residual. <c>k * LN2_HI</c> is then exactly
    /// representable in <see cref="double"/> for <c>|k| &lt; 2^22</c>.
    /// </summary>
    private const double LN2_HI = 0.6931470632553101;

    /// <summary>The residual <c>ln(2) - LN2_HI</c>.</summary>
    private const double LN2_LO = 1.1730463522763301e-07;

    /// <summary>sqrt(2), rounded to <see cref="double"/>.</summary>
    private const double SQRT2 = 1.4142135623730951;

    // Taylor coefficients 1/n! for exp(r) on [-ln(2)/2, ln(2)/2].
    private const double EXP_C1 = 1.0;
    private const double EXP_C2 = 0.5;
    private const double EXP_C3 = 0.16666666666666666;
    private const double EXP_C4 = 0.041666666666666664;
    private const double EXP_C5 = 0.008333333333333333;
    private const double EXP_C6 = 0.001388888888888889;
    private const double EXP_C7 = 0.0001984126984126984;
    private const double EXP_C8 = 2.48015873015873e-05;
    private const double EXP_C9 = 2.7557319223985893e-06;
    private const double EXP_C10 = 2.755731922398589e-07;

    // Taylor coefficients 1/(2n+1) for atanh(u) with
    // |u| <= 1 - sqrt(2)/2 ~ 0.1716.
    private const double ATANH_C1 = 1.0;
    private const double ATANH_C2 = 0.3333333333333333;
    private const double ATANH_C3 = 0.2;
    private const double ATANH_C4 = 0.14285714285714285;
    private const double ATANH_C5 = 0.1111111111111111;
    private const double ATANH_C6 = 0.09090909090909091;
    private const double ATANH_C7 = 0.07692307692307693;
    private const double ATANH_C8 = 0.06666666666666667;
    private const double ATANH_C9 = 0.058823529411764705;
    private const double ATANH_C10 = 0.05263157894736842;

    /// <summary>Largest argument whose exponential still fits in <see cref="float"/>.</summary>
    private const double EXP_OVERFLOW = 89.0;

    /// <summary>Smallest argument whose exponential still fits in <see cref="float"/>.</summary>
    private const double EXP_UNDERFLOW = -104.0;

    /// <summary>
    /// Deterministic <c>exp(x)</c> - the natural exponential of <paramref name="x"/>.
    /// </summary>
    /// <param name="x">The argument.</param>
    /// <returns>The exponential, correctly rounded to <see cref="float"/>.</returns>
    public static float Exp(float x)
    {
        if (float.IsNaN(x))
        {
            return float.NaN;
        }
        if (x == float.PositiveInfinity)
        {
            return float.PositiveInfinity;
        }
        if (x == float.NegativeInfinity)
        {
            return 0.0f;
        }

        double xd = x;
        if (xd > EXP_OVERFLOW)
        {
            return float.PositiveInfinity;
        }
        if (xd < EXP_UNDERFLOW)
        {
            return 0.0f;
        }

        // Reduce: x = k*ln(2) + r with |r| <= ln(2)/2.
        // MidpointRounding.AwayFromZero matches Rust's f64::round();
        // the default Math.Round would round halves to even.
        double k = System.Math.Round(xd / LN2, MidpointRounding.AwayFromZero);
        double r = (xd - k * LN2_HI) - k * LN2_LO;

        // exp(r) by Taylor series (Horner).
        double p = EXP_C10;
        p = EXP_C9 + r * p;
        p = EXP_C8 + r * p;
        p = EXP_C7 + r * p;
        p = EXP_C6 + r * p;
        p = EXP_C5 + r * p;
        p = EXP_C4 + r * p;
        p = EXP_C3 + r * p;
        p = EXP_C2 + r * p;
        p = EXP_C1 + r * p;
        double e = 1.0 + r * p;

        return (float)(e * Pow2((int)k));
    }

    /// <summary>
    /// Deterministic <c>ln(x)</c> - the natural logarithm of <paramref name="x"/>.
    /// </summary>
    /// <param name="x">The argument; must not be negative.</param>
    /// <returns>The natural logarithm, or NaN for a negative argument.</returns>
    public static float Ln(float x)
    {
        if (float.IsNaN(x))
        {
            return float.NaN;
        }
        if (x < 0.0f)
        {
            return float.NaN;
        }
        if (x == 0.0f)
        {
            return float.NegativeInfinity;
        }
        if (x == float.PositiveInfinity)
        {
            return float.PositiveInfinity;
        }

        double xd = x;

        // Decompose x = m * 2^e with m in [1, 2).
        long bits = BitConverter.DoubleToInt64Bits(xd);
        int e = (int)((bits >> 52) & 0x7ff) - 1023;
        double m = BitConverter.Int64BitsToDouble((bits & 0x000f_ffff_ffff_ffffL) | 0x3ff0_0000_0000_0000L);

        // Recentre m into [1/sqrt(2), sqrt(2)) so that
        // u = (m - 1) / (m + 1) stays small.
        if (m > SQRT2)
        {
            m *= 0.5;
            e += 1;
        }

        // ln(m) = 2 * atanh(u), u = (m - 1) / (m + 1).
        double u = (m - 1.0) / (m + 1.0);
        double u2 = u * u;
        double p = ATANH_C10;
        p = ATANH_C9 + u2 * p;
        p = ATANH_C8 + u2 * p;
        p = ATANH_C7 + u2 * p;
        p = ATANH_C6 + u2 * p;
        p = ATANH_C5 + u2 * p;
        p = ATANH_C4 + u2 * p;
        p = ATANH_C3 + u2 * p;
        p = ATANH_C2 + u2 * p;
        p = ATANH_C1 + u2 * p;
        double lnM = 2.0 * u * p;

        return (float)(e * LN2 + lnM);
    }

    /// <summary>
    /// 2^k for an integer k, constructed exactly from the exponent field
    /// instead of calling <c>Math.Pow</c>, which is a platform math
    /// library call.
    /// </summary>
    private static double Pow2(int k)
    {
        // The float argument range of Exp keeps |k| inside [-150, 129],
        // well within the exactly representable window [-1022, 1023].
        int clamped = k < -1022 ? -1022 : (k > 1023 ? 1023 : k);
        long biased = (long)(clamped + 1023);
        return BitConverter.Int64BitsToDouble(biased << 52);
    }
}
