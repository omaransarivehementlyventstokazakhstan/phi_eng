namespace PhiEng.Math;

/// <summary>
/// Deterministic trigonometry.
/// </summary>
/// <remarks>
/// <para>
/// The platform math library is <b>not portable</b>. <c>MathF.Sin</c>,
/// <c>MathF.Cos</c> and <c>MathF.Atan2</c> disagree between the
/// Microsoft CRT (Windows) and the Unix libm (Linux) in the last ULP,
/// and the <i>same</i> platform can disagree with itself: the JIT may
/// rewrite an inlined call differently depending on the surrounding
/// code.
/// </para>
/// <para>
/// That is enough to break a rigid-body trajectory. Measured on the
/// canonical 2D determinism scene, one body's rotation matrix diverged
/// by 1 ULP at step 45 (from <c>Atan2</c> plus the rotation-matrix
/// rebuild), and the contact solver amplified that into a completely
/// different trajectory by step 240 - with <i>identical</i> inputs on
/// both sides.
/// </para>
/// <para>
/// The helpers below use only operations that IEEE-754 specifies
/// exactly - add, subtract, multiply, divide, square root and the
/// round-to-nearest-even conversion back to <see cref="float"/> - with a
/// fixed evaluation order. They never call the platform math library,
/// so they return <b>bit-identical results on every platform</b>. This
/// mirrors <c>phieng-math</c>'s <c>trig</c> module operation for
/// operation, so the Rust and C# engines agree bit-for-bit here.
/// </para>
/// <para>
/// The public API is <see cref="float"/> (the simulation is
/// <c>float</c> throughout), but the evaluation is carried out in
/// <see cref="double"/> and rounded once at the end. The
/// <see cref="double"/> scratch keeps the range reduction and the
/// polynomial error far below the <see cref="float"/> rounding step, so
/// the result is the correctly rounded <see cref="float"/> value for all
/// practical inputs - at least as accurate as the platform library it
/// replaces. <see cref="double"/> is safe here because it is
/// <i>scratch</i>: no <see cref="double"/> value is ever stored in the
/// simulation state, so nothing can accumulate in double precision
/// across frames.
/// </para>
/// </remarks>
public static class DeterministicTrig
{
    /// <summary>pi, rounded to <see cref="double"/>.</summary>
    private const double PI = 3.141592653589793;

    /// <summary>pi / 4, rounded to <see cref="double"/>.</summary>
    private const double QUARTER_PI = 0.7853981633974483;

    /// <summary>tan(pi / 8) = sqrt(2) - 1, rounded to <see cref="double"/>.</summary>
    private const double TAN_PI_8 = 0.41421356237309515;

    /// <summary>2 / pi, rounded to <see cref="double"/>.</summary>
    private const double INVPIO2 = 0.6366197723675814;

    /// <summary>
    /// pi / 2 split into a high part with the low 32 mantissa bits
    /// cleared and the residual. <c>n * PIO2_HI</c> is then exactly
    /// representable in <see cref="double"/> for <c>|n| &lt; 2^20</c>.
    /// </summary>
    private const double PIO2_HI = 1.570796012878418;

    /// <summary>The residual <c>pi / 2 - PIO2_HI</c>.</summary>
    private const double PIO2_LO = 3.13916478589249e-07;

    /// <summary>pi, rounded to <see cref="float"/>.</summary>
    private const float PI_F32 = 3.1415927f;

    /// <summary>pi / 2, rounded to <see cref="float"/>.</summary>
    private const float HALF_PI_F32 = 1.5707964f;

    // Taylor coefficients for sin(r) = r * (1 + r^2 * p(r^2)) on
    // [-pi/4, pi/4]. Truncation error at r = pi/4 is ~7.5e-12.
    private const double SIN_C1 = -0.16666666666666666;
    private const double SIN_C2 = 0.008333333333333333;
    private const double SIN_C3 = -0.0001984126984126984;
    private const double SIN_C4 = 2.7557319223985893e-06;
    private const double SIN_C5 = -2.505210838544172e-08;

    // Taylor coefficients for cos(r) = 1 + r^2 * q(r^2) on
    // [-pi/4, pi/4]. Truncation error at r = pi/4 is ~4.2e-13.
    private const double COS_C1 = -0.5;
    private const double COS_C2 = 0.041666666666666664;
    private const double COS_C3 = -0.001388888888888889;
    private const double COS_C4 = 2.48015873015873e-05;
    private const double COS_C5 = -2.755731922398589e-07;
    private const double COS_C6 = 2.08767569878681e-09;

    // Taylor coefficients for atan(z) = z * (1 + z^2 * p(z^2)), used only
    // for |z| <= tan(pi/8). Truncation error at the limit is ~4.4e-10.
    private const double ATAN_C1 = -0.3333333333333333;
    private const double ATAN_C2 = 0.2;
    private const double ATAN_C3 = -0.14285714285714285;
    private const double ATAN_C4 = 0.1111111111111111;
    private const double ATAN_C5 = -0.09090909090909091;
    private const double ATAN_C6 = 0.07692307692307693;
    private const double ATAN_C7 = -0.06666666666666667;
    private const double ATAN_C8 = 0.058823529411764705;
    private const double ATAN_C9 = -0.05263157894736842;

    /// <summary>
    /// Deterministic sine and cosine of <paramref name="angle"/> (radians).
    /// </summary>
    /// <param name="angle">The angle in radians.</param>
    /// <returns><c>(cos, sin)</c>, matching the <c>sin_cos</c> convention.</returns>
    public static (float Cos, float Sin) ComputeCosSin(float angle)
    {
        if (!float.IsFinite(angle))
        {
            return (float.NaN, float.NaN);
        }

        // Reduce to r in [-pi/4, pi/4] with a quadrant index.
        double a = angle;
        // MidpointRounding.AwayFromZero matches Rust's f64::round();
        // the default Math.Round would round halves to even.
        double n = System.Math.Round(a * INVPIO2, MidpointRounding.AwayFromZero);
        double r = (a - n * PIO2_HI) - n * PIO2_LO;

        // sin(r) = r * (1 + z * p(z)), cos(r) = 1 + z * q(z), z = r^2.
        double z = r * r;
        double p = SIN_C5;
        p = SIN_C4 + z * p;
        p = SIN_C3 + z * p;
        p = SIN_C2 + z * p;
        p = SIN_C1 + z * p;
        double s = r * (1.0 + z * p);
        double q = COS_C6;
        q = COS_C5 + z * q;
        q = COS_C4 + z * q;
        q = COS_C3 + z * q;
        q = COS_C2 + z * q;
        q = COS_C1 + z * q;
        double c = 1.0 + z * q;

        // Quadrant: n is an exact integer, so the remainder is exact too.
        double rem = n % 4.0;
        double quadrant = rem < 0.0 ? rem + 4.0 : rem;
        return quadrant switch
        {
            1.0 => (-(float)s, (float)c),
            2.0 => (-(float)c, -(float)s),
            3.0 => ((float)s, -(float)c),
            _ => ((float)c, (float)s),
        };
    }

    /// <summary>
    /// Deterministic <c>atan2(y, x)</c> - the angle of the vector
    /// <c>(x, y)</c> in radians, in <c>(-pi, pi]</c>.
    /// </summary>
    /// <param name="y">The y component.</param>
    /// <param name="x">The x component.</param>
    /// <returns>The angle in radians.</returns>
    public static float Atan2(float y, float x)
    {
        if (!float.IsFinite(y) || !float.IsFinite(x))
        {
            return float.NaN;
        }

        bool yNegative = float.IsNegative(y);

        // Degenerate axes, matching the usual IEEE-754 conventions.
        if (y == 0.0f)
        {
            if (x > 0.0f)
            {
                // IEEE-754: atan2(+-0, +x) is +-0, preserving the sign
                // of the zero. The platform libm does the same.
                return yNegative ? -0.0f : 0.0f;
            }
            if (x < 0.0f)
            {
                return yNegative ? -PI_F32 : PI_F32;
            }
            return yNegative ? -0.0f : 0.0f;
        }
        if (x == 0.0f)
        {
            return y > 0.0f ? HALF_PI_F32 : -HALF_PI_F32;
        }

        double yd = y;
        double xd = x;
        double ax = System.Math.Abs(xd);
        double ay = System.Math.Abs(yd);

        // Scale first so the squares cannot overflow, then use the
        // half-angle identity atan2(y, x) = 2 * atan(y / (r + |x|)) with
        // r = hypot(x, y). The scale cancels in the ratio.
        double scale = System.Math.Max(ax, ay);
        double sx = ax / scale;
        double sy = yd / scale;
        double r = System.Math.Sqrt(sx * sx + sy * sy);
        double t = sy / (r + sx); // in (-1, 1], carries the sign of y

        double angle = 2.0 * AtanUnit(t);
        if (xd < 0.0)
        {
            // Left half-plane: reflect across the y axis.
            angle = t >= 0.0 ? PI - angle : -PI - angle;
        }
        return (float)angle;
    }

    /// <summary>atan of a value already known to lie in [-1, 1].</summary>
    private static double AtanUnit(double t)
    {
        if (t > TAN_PI_8)
        {
            // atan(t) = pi/4 + atan((t - 1) / (t + 1))
            return QUARTER_PI + AtanPoly((t - 1.0) / (t + 1.0));
        }
        if (t < -TAN_PI_8)
        {
            // atan(t) = -pi/4 + atan((t + 1) / (1 - t))
            return -QUARTER_PI + AtanPoly((t + 1.0) / (1.0 - t));
        }
        return AtanPoly(t);
    }

    /// <summary>atan(z) for |z| &lt;= tan(pi/8).</summary>
    private static double AtanPoly(double z)
    {
        double z2 = z * z;
        double p = ATAN_C9;
        p = ATAN_C8 + z2 * p;
        p = ATAN_C7 + z2 * p;
        p = ATAN_C6 + z2 * p;
        p = ATAN_C5 + z2 * p;
        p = ATAN_C4 + z2 * p;
        p = ATAN_C3 + z2 * p;
        p = ATAN_C2 + z2 * p;
        p = ATAN_C1 + z2 * p;
        return z * (1.0 + z2 * p);
    }
}
