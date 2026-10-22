using Xunit;
using PhiEng.Math;

namespace PhiEng.Tests;

/// <summary>
/// The quantum math layer (M01-M08). Mirrors
/// <c>crates/engine/tests/quantum_math.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// Quantum mechanics is arithmetic over the complex numbers, and a
/// measurement is a random draw. These tests pin both, bit for bit, so
/// the Rust core and the managed mirror cannot drift apart: a sampled
/// measurement has to be the same draw on every platform and in both
/// languages.
/// </remarks>
public class QuantumMathTests
{
    /// <summary>The bit pattern of a float, as the pins are written.</summary>
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>The two bit patterns of a complex number.</summary>
    private static (uint Re, uint Im) CBits(Complex32 c) =>
        (BitConverter.SingleToUInt32Bits(c.Re), BitConverter.SingleToUInt32Bits(c.Im));

    /// <summary>
    /// M01: the complex arithmetic's pinned bit patterns.
    /// </summary>
    /// <remarks>
    /// The operands are <c>a = 1.5 - 2.25i</c> and <c>b = -0.75 + 3.5i</c>.
    /// </remarks>
    [Fact]
    public void M01_ComplexArithmeticIsPortable()
    {
        var a = new Complex32(1.5f, -2.25f);
        var b = new Complex32(-0.75f, 3.5f);
        Assert.Equal((0x3f400000u, 0x3fa00000u), CBits(Complex32.Add(a, b)));
        Assert.Equal((0x40100000u, 0xc0b80000u), CBits(Complex32.Subtract(a, b)));
        Assert.Equal((0x40d80000u, 0x40de0000u), CBits(Complex32.Multiply(a, b)));
        Assert.Equal((0xbf33d30bu, 0xbe8e5c68u), CBits(Complex32.Divide(a, b)));
        Assert.Equal((0x40700000u, 0xc0b40000u), CBits(Complex32.Scale(a, 2.5f)));
        Assert.Equal((0xbfc00000u, 0x40100000u), CBits(Complex32.Negate(a)));
        Assert.Equal((0x3fc00000u, 0x40100000u), CBits(Complex32.Conjugate(a)));
        Assert.Equal((0x3e520d21u, 0x3e9d89d9u), CBits(Complex32.Inverse(a)));
        Assert.Equal(0x402d1104u, Bits(Complex32.Abs(a)));
        Assert.Equal(0x40ea0000u, Bits(Complex32.NormSqr(a)));
        var (r, theta) = Complex32.ToPolar(a);
        Assert.Equal(0x402d1104u, Bits(r));
        Assert.Equal(0xbf7b985fu, Bits(theta));
    }

    /// <summary>
    /// M02: the complex transcendentals' pinned bit patterns.
    /// </summary>
    [Fact]
    public void M02_ComplexTranscendentalsArePortable()
    {
        (Complex32 Z, (uint Re, uint Im) Exp, (uint Re, uint Im) Ln,
            (uint Re, uint Im) Sqrt, (uint Re, uint Im) Sin)[] cases =
        {
            (new Complex32(0.0f, 0.0f),
                (0x3f800000u, 0x00000000u), (0xff800000u, 0x00000000u),
                (0x00000000u, 0x00000000u), (0x00000000u, 0x00000000u)),
            (new Complex32(1.0f, 0.0f),
                (0x402df854u, 0x00000000u), (0x00000000u, 0x00000000u),
                (0x3f800000u, 0x00000000u), (0x3f576aa4u, 0x00000000u)),
            (new Complex32(0.0f, 1.0f),
                (0x3f0a5140u, 0x3f576aa4u), (0x00000000u, 0x3fc90fdbu),
                (0x3f3504f3u, 0x3f3504f3u), (0x00000000u, 0x00000000u)),
            (new Complex32(0.5f, 0.25f),
                (0x3fcc79c8u, 0x3ed0d84au), (0xbf14e220u, 0x3eed6338u),
                (0x3f3a48cdu, 0x3e2fe733u), (0x3ebf2b45u, 0x00000000u)),
            (new Complex32(-1.25f, 2.0f),
                (0xbdf42dd9u, 0x3e8562a0u), (0x3f5ba773u, 0x40084805u),
                (0x3f3e961eu, 0x3fabeebau), (0xbe038371u, 0x00000000u)),
            (new Complex32(2.0f, -3.5f),
                (0xc0dd6cccu, 0x4025e2a0u), (0x3fb2701du, 0xbf869c79u),
                (0x3fde46e1u, 0xbf80fe06u), (0x41f0e4f4u, 0x00000000u)),
            (new Complex32(-0.75f, -0.125f),
                (0x3eeff6f4u, 0xbd7138ebu), (0xbe8c4774u, 0xc03e7e0fu),
                (0x3d934b6bu, 0xbf5e7739u), (0xbf45bbdfu, 0x00000000u)),
            (new Complex32(3.0f, 0.7853981633974483f),
                (0x41633deeu, 0x41633deeu), (0x3f90dd76u, 0x3e831959u),
                (0x3fdf9009u, 0x3e663c13u), (0x3d83c5bcu, 0x00000000u)),
        };

        foreach (var (z, exp, ln, sqrt, sin) in cases)
        {
            Assert.Equal(exp, CBits(Complex32.Exp(z)));
            Assert.Equal(ln, CBits(Complex32.Ln(z)));
            Assert.Equal(sqrt, CBits(Complex32.Sqrt(z)));
            Assert.Equal(sin, CBits(Complex32.Sin(z)));
        }

        // The phase factor: the workhorse of every gate and Fourier twiddle.
        (float Theta, (uint Re, uint Im) Expected)[] phases =
        {
            (0.0f, (0x3f800000u, 0x00000000u)),
            (0.5f, (0x3f60a940u, 0x3ef57744u)),
            (1.0f, (0x3f0a5140u, 0x3f576aa4u)),
            (-0.75f, (0x3f3b4ff6u, 0xbf2e7fe1u)),
            (MathF.PI, (0xbf800000u, 0xb3bbbd2eu)),
            (2.5f, (0xbf4d17bfu, 0x3f193578u)),
        };
        foreach (var (theta, expected) in phases)
        {
            Assert.Equal(expected, CBits(Complex32.ExpI(theta)));
        }
    }

    /// <summary>
    /// M03: the edge cases, pinned.
    /// </summary>
    /// <remarks>
    /// The two places a naive formula breaks are pinned here: the
    /// modulus (which overflows if the components are squared first) and
    /// the division (which overflows if the divisor's square is formed).
    /// </remarks>
    [Fact]
    public void M03_ComplexEdgeCasesArePortable()
    {
        var zero = Complex32.Zero;
        // A NaN's bit pattern is not pinned: the payload of a NaN is not
        // a contract, and the two languages spell "not a number"
        // differently. What is pinned is that the answer is not a number.
        Assert.True(float.IsNaN(Complex32.Divide(zero, zero).Re));
        Assert.True(float.IsNaN(Complex32.Divide(zero, zero).Im));
        Assert.True(float.IsNaN(Complex32.Divide(Complex32.One, zero).Re));
        Assert.True(float.IsNaN(Complex32.Divide(Complex32.One, zero).Im));

        // A huge and a tiny magnitude: the scaled modulus survives both.
        Assert.Equal(0x718ecc90u, Bits(Complex32.Abs(new Complex32(1e30f, 1e30f))));
        Assert.Equal(0x0de57822u, Bits(Complex32.Abs(new Complex32(1e-30f, 1e-30f))));

        // The principal square root, including the branch cases.
        Assert.Equal((0x00000000u, 0x40000000u), CBits(Complex32.Sqrt(new Complex32(-4.0f, 0.0f))));
        Assert.Equal((0x40000000u, 0x3f800000u), CBits(Complex32.Sqrt(new Complex32(3.0f, 4.0f))));
        Assert.Equal((0x3f800000u, 0x40000000u), CBits(Complex32.Sqrt(new Complex32(-3.0f, 4.0f))));
        Assert.Equal((0x40000000u, 0xbf800000u), CBits(Complex32.Sqrt(new Complex32(3.0f, -4.0f))));

        // The squared modulus does overflow - that is documented, and the
        // division is the one that must not.
        Assert.Equal(0x7f800000u, Bits(Complex32.NormSqr(new Complex32(1e20f, 1e20f))));
        Assert.Equal(
            (0x3f800000u, 0x00000000u),
            CBits(Complex32.Divide(
                new Complex32(1e20f, 1e20f),
                new Complex32(1e20f, 1e20f))));
    }

    /// <summary>
    /// M04: the random number generator's pinned stream.
    /// </summary>
    [Fact]
    public void M04_RngStreamIsPortable()
    {
        var rng = new DeterministicRng(1);
        uint[] expected =
        {
            0x7c64370du, 0x62ffbc74u, 0x058b01afu, 0x248ebb73u, 0x96863623u, 0xfc417b82u, 0x591b1abfu,
            0x7359a0d8u, 0x2d73df81u, 0x7a18ef6fu, 0x40ce785du, 0x8ac166f9u, 0xa54ec268u, 0x7397865au,
            0x47ebe3eeu, 0x1ce01314u,
        };
        foreach (var want in expected)
        {
            Assert.Equal(want, rng.NextUInt32());
        }
        Assert.Equal((3659518936u, 1389279675u, 1908592355u, 2094870070u), rng.State);
    }

    /// <summary>
    /// M05: the generator's seeding.
    /// </summary>
    /// <remarks>
    /// The same seed must give the same stream, and two different seeds
    /// must not.
    /// </remarks>
    [Fact]
    public void M05_RngSeedingIsReproducible()
    {
        var a = new DeterministicRng(42);
        var b = new DeterministicRng(42);
        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(a.NextUInt32(), b.NextUInt32());
        }

        var c = new DeterministicRng(43);
        var d = new DeterministicRng(42);
        bool differs = false;
        for (int i = 0; i < 32; i++)
        {
            if (c.NextUInt32() != d.NextUInt32())
            {
                differs = true;
            }
        }
        Assert.True(differs, "two different seeds must give different streams");
    }

    /// <summary>
    /// M06: the generator's float mapping, pinned.
    /// </summary>
    [Fact]
    public void M06_RngFloatMappingIsPortable()
    {
        var rng = new DeterministicRng(12345);
        uint[] expected =
        {
            0x3f623f08u, 0x3b785100u, 0x3f4ed462u, 0x3f6c6baau, 0x3eaa0ad8u, 0x3e39a1acu, 0x3f27aa48u,
            0x3eed9958u,
        };
        foreach (var want in expected)
        {
            float v = rng.NextFloat();
            Assert.Equal(want, Bits(v));
            Assert.True(v >= 0.0f && v < 1.0f, "the draw must be in [0, 1)");
        }
    }

    /// <summary>
    /// M07: the generator's normal deviate, pinned.
    /// </summary>
    [Fact]
    public void M07_RngNormalIsPortable()
    {
        var rng = new DeterministicRng(7);
        uint[] expected =
        {
            0xbf59ff96u, 0x3e7acd55u, 0x3e4ce768u, 0x40185864u, 0xbf364ea7u, 0x3e51d25du,
        };
        foreach (var want in expected)
        {
            Assert.Equal(want, Bits(rng.NextStandardNormal()));
        }
    }

    /// <summary>
    /// M08: a stream can be saved and resumed exactly.
    /// </summary>
    /// <remarks>
    /// A saved game has to continue with the same numbers it would have
    /// drawn without the save.
    /// </remarks>
    [Fact]
    public void M08_RngStateResumesTheStream()
    {
        var rng = new DeterministicRng(99);
        for (int i = 0; i < 5; i++)
        {
            rng.NextUInt32();
        }
        var saved = rng.State;
        var expected = new uint[8];
        for (int i = 0; i < 8; i++)
        {
            expected[i] = rng.NextUInt32();
        }

        var resumed = new DeterministicRng(saved.S0, saved.S1, saved.S2, saved.S3);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(expected[i], resumed.NextUInt32());
        }
    }
}
