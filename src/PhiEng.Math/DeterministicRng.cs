namespace PhiEng.Math;

/// <summary>
/// A seeded, reproducible random number generator.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors <c>phieng-math</c>'s <c>rng</c> module operation for
/// operation, so the Rust and C# engines draw the same numbers: the
/// state is four <see cref="uint"/> words, the output is a
/// <see cref="uint"/>, and the mixing is exact integer arithmetic — no
/// floating point, nothing that can differ between platforms.
/// </para>
/// <para>
/// The algorithm is xoshiro256++, seeded with splitmix64.
/// </para>
/// </remarks>
public sealed class DeterministicRng
{
    /// <summary>The golden ratio's odd multiple, the splitmix64 increment.</summary>
    private const ulong SplitmixIncrement = 0x9E3779B97F4A7C15UL;

    /// <summary>The splitmix64 finalizer's first multiplier.</summary>
    private const ulong SplitmixMulA = 0xBF58476D1CE4E5B9UL;

    /// <summary>The splitmix64 finalizer's second multiplier.</summary>
    private const ulong SplitmixMulB = 0x94D049BB133111EBUL;

    /// <summary>2^-24, the scale of the 24-bit float mapping.</summary>
    private const float TwoPowMinus24 = 1.0f / 16777216.0f;

    /// <summary>
    /// The smallest positive normal <see cref="float"/>, 0x00800000.
    /// </summary>
    /// <remarks>
    /// This is Rust's <c>f32::MIN_POSITIVE</c>, not .NET's
    /// <see cref="float.Epsilon"/> (which is the smallest subnormal, a
    /// different number). The two languages must clamp to the same
    /// value, or the normal deviate from the standard normal.
    /// </remarks>
    private const float MinPositive = 1.17549435e-38f;

    private uint _s0;
    private uint _s1;
    private uint _s2;
    private uint _s3;

    /// <summary>A generator seeded with <paramref name="seed"/>.</summary>
    /// <remarks>
    /// The seed is expanded through splitmix64 into the four state
    /// words, so even neighbouring seeds give unrelated streams.
    /// </remarks>
    public DeterministicRng(ulong seed)
    {
        ulong sm = seed;
        _s0 = Mix(ref sm);
        _s1 = Mix(ref sm);
        _s2 = Mix(ref sm);
        _s3 = Mix(ref sm);
    }

    /// <summary>A generator from an exact state.</summary>
    public DeterministicRng(uint s0, uint s1, uint s2, uint s3)
    {
        _s0 = s0;
        _s1 = s1;
        _s2 = s2;
        _s3 = s3;
    }

    /// <summary>The generator's exact state.</summary>
    public (uint S0, uint S1, uint S2, uint S3) State => (_s0, _s1, _s2, _s3);

    /// <summary>The next <see cref="uint"/>.</summary>
    public uint NextUInt32()
    {
        uint result = unchecked(_s0 + _s3);
        result = unchecked((result << 23) | (result >> 9));
        result = unchecked(result + _s0);

        uint t = unchecked(_s1 << 17);
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = unchecked((_s3 << 45) | (_s3 >> 19));
        return result;
    }

    /// <summary>The next value in [0, 1), with 24 bits of resolution.</summary>
    /// <remarks>
    /// The mapping is exact: every value is a multiple of 2^-24, so no
    /// rounding happens and the two engines agree bit-for-bit.
    /// </remarks>
    public float NextFloat() => (NextUInt32() >> 8) * TwoPowMinus24;

    /// <summary>The next value in [-1, 1), with 24 bits of resolution.</summary>
    public float NextFloatSigned() => NextFloat() * 2.0f - 1.0f;

    /// <summary>The next value in [0, bound), without modulo bias.</summary>
    /// <remarks>Returns zero for a bound of zero.</remarks>
    public uint NextBelow(uint bound)
    {
        if (bound == 0)
        {
            return 0;
        }
        uint zone = uint.MaxValue - (uint.MaxValue % bound);
        while (true)
        {
            uint v = NextUInt32();
            if (v < zone)
            {
                return v % bound;
            }
        }
    }

    /// <summary>The next value from the standard normal distribution.</summary>
    /// <remarks>
    /// The Box–Muller transform, with the logarithm and the trigonometry
    /// from the deterministic helpers, so the result is reproducible
    /// bit-for-bit. One draw gives one value: the transform's second
    /// output is discarded rather than cached, which keeps the stream
    /// position a simple function of the number of draws.
    /// </remarks>
    public float NextStandardNormal()
    {
        // u1 must be positive for the logarithm; the generator can
        // produce exactly zero, so it is nudged off it.
        float u1 = MathF.Max(NextFloat(), MinPositive);
        float u2 = NextFloat();
        float r = MathF.Sqrt(-2.0f * DeterministicScalar.Ln(u1));
        float theta = 2.0f * MathF.PI * u2;
        var (c, _s) = DeterministicTrig.ComputeCosSin(theta);
        return r * c;
    }

    /// <summary>One step of splitmix64: the seed expander.</summary>
    private static uint Mix(ref ulong state)
    {
        state = unchecked(state + SplitmixIncrement);
        ulong z = state;
        z = unchecked((z ^ (z >> 30)) * SplitmixMulA);
        z = unchecked((z ^ (z >> 27)) * SplitmixMulB);
        z ^= z >> 31;
        return (uint)((z >> 32) ^ z);
    }
}
