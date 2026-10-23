using System.Runtime.InteropServices;

namespace PhiEng.Math;

/// <summary>
/// A complex number with <see cref="float"/> components.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors <c>phieng-math</c>'s <c>complex</c> module operation for
/// operation, so the Rust and C# engines agree bit-for-bit here — the
/// same contract the deterministic trigonometry relies on.
/// </para>
/// <para>
/// The complex transcendental functions are built only from
/// <see cref="DeterministicTrig"/> and <see cref="DeterministicScalar"/>
/// and never from the platform math library. Two places would overflow a
/// naive formula and are written the stable way in both languages: the
/// modulus (scaled before the square root) and the division (Smith's
/// algorithm).
/// </para>
/// <para>
/// The layout matches the Rust <c>#[repr(C)]</c> struct field for field:
/// the real part first, then the imaginary part.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
public struct Complex32 : IEquatable<Complex32>
{
    /// <summary>The real part.</summary>
    public float Re;

    /// <summary>The imaginary part.</summary>
    public float Im;

    /// <summary>The complex number <paramref name="re"/> + <paramref name="im"/> * i.</summary>
    public Complex32(float re, float im)
    {
        Re = re;
        Im = im;
    }

    /// <summary>Zero.</summary>
    public static Complex32 Zero => new(0.0f, 0.0f);

    /// <summary>One.</summary>
    public static Complex32 One => new(1.0f, 0.0f);

    /// <summary>The imaginary unit.</summary>
    public static Complex32 I => new(0.0f, 1.0f);

    /// <summary>The sum.</summary>
    public static Complex32 Add(Complex32 a, Complex32 b) => new(a.Re + b.Re, a.Im + b.Im);

    /// <summary>The difference.</summary>
    public static Complex32 Subtract(Complex32 a, Complex32 b) => new(a.Re - b.Re, a.Im - b.Im);

    /// <summary>The product.</summary>
    public static Complex32 Multiply(Complex32 a, Complex32 b) =>
        new(a.Re * b.Re - a.Im * b.Im, a.Re * b.Im + a.Im * b.Re);

    /// <summary>
    /// The quotient, by Smith's algorithm: the divisor is scaled by its
    /// larger component first, so the intermediate never overflows and
    /// never underflows to a spurious zero.
    /// </summary>
    public static Complex32 Divide(Complex32 a, Complex32 b)
    {
        float c = b.Re;
        float d = b.Im;
        if (MathF.Abs(c) >= MathF.Abs(d))
        {
            if (c == 0.0f)
            {
                // Both components are zero: the quotient is a NaN.
                return new Complex32(float.NaN, float.NaN);
            }
            // r = d/c and the denominator is (c^2 + d^2)/c, so the
            // quotient's numerator is multiplied by c and divided by
            // (c^2 + d^2) - without ever forming that square.
            float r = d / c;
            float den = c + d * r;
            return new Complex32((a.Re + a.Im * r) / den, (a.Im - a.Re * r) / den);
        }
        else
        {
            float r = c / d;
            float den = c * r + d;
            return new Complex32((a.Re * r + a.Im) / den, (a.Im * r - a.Re) / den);
        }
    }

    /// <summary>The product with a real number.</summary>
    public static Complex32 Scale(Complex32 a, float s) => new(a.Re * s, a.Im * s);

    /// <summary>The negation.</summary>
    public static Complex32 Negate(Complex32 a) => new(-a.Re, -a.Im);

    /// <summary>The conjugate.</summary>
    public static Complex32 Conjugate(Complex32 a) => new(a.Re, -a.Im);

    /// <summary>The conjugate.</summary>
    public Complex32 Conjugate() => new(Re, -Im);

    /// <summary>The product with a real number.</summary>
    public Complex32 Scale(float s) => new(Re * s, Im * s);

    /// <summary>The reciprocal.</summary>
    public static Complex32 Inverse(Complex32 a) => Divide(One, a);

    /// <summary>The squared modulus, re*re + im*im.</summary>
    public static float NormSqr(Complex32 a) => a.Re * a.Re + a.Im * a.Im;

    /// <summary>
    /// The modulus. The larger component is factored out first, so the
    /// result is finite for every finite input.
    /// </summary>
    public static float Abs(Complex32 a)
    {
        float re = a.Re;
        float im = a.Im;
        float scale = MathF.Max(MathF.Abs(re), MathF.Abs(im));
        if (scale == 0.0f)
        {
            return 0.0f;
        }
        float x = re / scale;
        float y = im / scale;
        return scale * MathF.Sqrt(x * x + y * y);
    }

    /// <summary>Whether both components are finite.</summary>
    public static bool IsFinite(Complex32 a) => float.IsFinite(a.Re) && float.IsFinite(a.Im);

    /// <summary>Whether both components are zero.</summary>
    public static bool IsZero(Complex32 a) => a.Re == 0.0f && a.Im == 0.0f;

    /// <summary>The number r * (cos theta + i sin theta).</summary>
    public static Complex32 FromPolar(float r, float theta)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(theta);
        return new Complex32(r * c, r * s);
    }

    /// <summary>The polar form (r, theta), with theta in (-pi, pi].</summary>
    public static (float R, float Theta) ToPolar(Complex32 a) =>
        (Abs(a), DeterministicTrig.Atan2(a.Im, a.Re));

    /// <summary>The phase factor cos(theta) + i sin(theta).</summary>
    public static Complex32 ExpI(float theta)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(theta);
        return new Complex32(c, s);
    }

    /// <summary>The complex exponential, e^z.</summary>
    public static Complex32 Exp(Complex32 a)
    {
        float m = DeterministicScalar.Exp(a.Re);
        var (c, s) = DeterministicTrig.ComputeCosSin(a.Im);
        return new Complex32(m * c, m * s);
    }

    /// <summary>The principal logarithm, ln z, with the argument in (-pi, pi].</summary>
    public static Complex32 Ln(Complex32 a) =>
        new(DeterministicScalar.Ln(Abs(a)), DeterministicTrig.Atan2(a.Im, a.Re));

    /// <summary>
    /// The principal square root: the root with a non-negative real part.
    /// </summary>
    public static Complex32 Sqrt(Complex32 a)
    {
        float r = Abs(a);
        if (r == 0.0f)
        {
            return Zero;
        }
        float t = MathF.Sqrt((r + a.Re) * 0.5f);
        float u = MathF.Sqrt((r - a.Re) * 0.5f);
        if (a.Re < 0.0f && u > 0.0f)
        {
            // t*u = |im|/2 exactly, and t is the cancelled half here.
            t = MathF.Abs(a.Im) / (2.0f * u);
        }
        float y = a.Im < 0.0f ? -u : u;
        return new Complex32(t, y);
    }

    /// <summary>The complex sine.</summary>
    /// <remarks>
    /// <c>sin z = (e^(iz) - e^(-iz)) / (2i)</c>. Note <c>e^(-iz)</c> is
    /// *not* the conjugate of <c>e^(iz)</c> for a complex <c>z</c>; using
    /// the conjugate collapses the result to a real number (the bug this
    /// used to have). The imaginary part is written as a subtraction from
    /// zero rather than a negation, so a real argument gives <c>+0.0</c>
    /// there and not <c>-0.0</c>.
    /// </remarks>
    public static Complex32 Sin(Complex32 a)
    {
        var e = Exp(new Complex32(-a.Im, a.Re));
        var m = Exp(new Complex32(a.Im, -a.Re));
        var d = Subtract(e, m);
        return new Complex32(d.Im * 0.5f, 0.0f - d.Re * 0.5f);
    }

    /// <summary>The complex cosine. See <see cref="Sin"/>.</summary>
    public static Complex32 Cos(Complex32 a)
    {
        var e = Exp(new Complex32(-a.Im, a.Re));
        var m = Exp(new Complex32(a.Im, -a.Re));
        var s = Add(e, m);
        return new Complex32(s.Re * 0.5f, s.Im * 0.5f);
    }

    /// <summary>The sum.</summary>
    public static Complex32 operator +(Complex32 a, Complex32 b) => Add(a, b);

    /// <summary>The difference.</summary>
    public static Complex32 operator -(Complex32 a, Complex32 b) => Subtract(a, b);

    /// <summary>The product.</summary>
    public static Complex32 operator *(Complex32 a, Complex32 b) => Multiply(a, b);

    /// <summary>The quotient.</summary>
    public static Complex32 operator /(Complex32 a, Complex32 b) => Divide(a, b);

    /// <summary>The negation.</summary>
    public static Complex32 operator -(Complex32 a) => Negate(a);

    /// <summary>The product with a real number.</summary>
    public static Complex32 operator *(Complex32 a, float s) => Scale(a, s);

    /// <inheritdoc/>
    public bool Equals(Complex32 other) => Re.Equals(other.Re) && Im.Equals(other.Im);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Complex32 other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Re, Im);

    /// <summary>Whether the two numbers are equal, component for component.</summary>
    public static bool operator ==(Complex32 a, Complex32 b) => a.Equals(b);

    /// <summary>Whether the two numbers differ.</summary>
    public static bool operator !=(Complex32 a, Complex32 b) => !a.Equals(b);

    /// <inheritdoc/>
    public override string ToString() => $"({Re}, {Im})";
}
