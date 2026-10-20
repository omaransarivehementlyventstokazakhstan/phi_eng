using System.Numerics;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Math;

namespace PhiEng.Tests;

/// <summary>
/// Deterministic trigonometry tests (R01-R03). Mirrors
/// <c>crates/engine/tests/determinism_trig.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// The platform math library is not portable: <c>MathF.Sin</c>,
/// <c>MathF.Cos</c> and <c>MathF.Atan2</c> disagree between the
/// Microsoft CRT (Windows) and the Unix libm (Linux) in the last ULP,
/// and the same platform can disagree with itself when the JIT rewrites
/// an inlined call. In a rigid-body simulation that single ULP is
/// enough to send a trajectory in two different directions.
/// </remarks>
public class DeterministicTrigTests
{
    /// <summary>
    /// R01: the deterministic sine and cosine, pinned to exact bit patterns.
    /// </summary>
    [Fact]
    public void R01_DeterministicCosSinIsPortable()
    {
        (float Angle, uint Cos, uint Sin)[] cases =
        {
            (0.0f, 0x3f800000, 0x00000000),
            (0.5f, 0x3f60a940, 0x3ef57744),
            (1.0f, 0x3f0a5140, 0x3f576aa4),
            (-1.0f, 0x3f0a5140, 0xbf576aa4),
            (MathF.PI / 2.0f, 0xb33bbd2e, 0x3f800000),
            (MathF.PI, 0xbf800000, 0xb3bbbd2e),
            (3.0f, 0xbf7d7026, 0x3e1081c3),
            (-7.25f, 0x3f11637b, 0xbf52b56e),
        };
        foreach (var (angle, cosBits, sinBits) in cases)
        {
            var (cos, sin) = DeterministicTrig.ComputeCosSin(angle);
            Assert.Equal((cosBits, sinBits), (BitConverter.SingleToUInt32Bits(cos), BitConverter.SingleToUInt32Bits(sin)));
        }
    }

    /// <summary>
    /// R02: the deterministic <c>atan2</c>, pinned to exact bit patterns.
    /// </summary>
    [Fact]
    public void R02_DeterministicAtan2IsPortable()
    {
        (float Y, float X, uint Bits)[] cases =
        {
            (0.0f, 1.0f, 0x00000000),
            (1.0f, 0.0f, 0x3fc90fdb),
            (-1.0f, 0.0f, 0xbfc90fdb),
            (0.0f, -1.0f, 0x40490fdb),
            (1.0f, 1.0f, 0x3f490fdb),
            (-3.5f, 2.25f, 0xbf7fdc89),
            (0.125f, -0.75f, 0x403e7e0f),
            (-1e-3f, 1e-3f, 0xbf490fdb),
        };
        foreach (var (y, x, bits) in cases)
        {
            Assert.Equal(bits, BitConverter.SingleToUInt32Bits(DeterministicTrig.Atan2(y, x)));
        }
    }

    /// <summary>
    /// R03: the engine's own rotation integration is portable.
    /// </summary>
    /// <remarks>
    /// This is the test that would have caught the original CI failure:
    /// it pins the rotation matrix a body ends up with after a fixed
    /// number of integration steps. The pins are identical on every
    /// platform, so the test can only pass if the engine really is free
    /// of the platform math library.
    /// </remarks>
    [Fact]
    public void R03_BodyRotationIntegrationIsPortable()
    {
        const float Dt = 1.0f / 60.0f;

        var world = new World2
        {
            Gravity = Vector2.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };
        int id = world.AddDynamicBody(
            Vector2.Zero,
            new BoxShape2(new Vector2(0.5f, 0.5f)),
            1.0f);

        var body = world.Bodies[id];
        body.AngularVelocity = 1.75f;
        body.LinearDamping = 0.0f;
        body.AngularDamping = 0.0f;

        for (int i = 0; i < 10; i++)
        {
            world.Step(Dt);
        }

        Matrix3x2 rot = world.Bodies[id].Transform.Rotation;
        Assert.Equal(
            (
                0x3f753026u, // x_axis.x (M11)
                0x3e93399du, // x_axis.y (M12)
                0xbe93399du, // y_axis.x (M21)
                0x3f753026u  // y_axis.y (M22)
            ),
            (
                BitConverter.SingleToUInt32Bits(rot.M11),
                BitConverter.SingleToUInt32Bits(rot.M12),
                BitConverter.SingleToUInt32Bits(rot.M21),
                BitConverter.SingleToUInt32Bits(rot.M22)
            ));
    }

    /// <summary>
    /// R04: the deterministic exponential, pinned to exact bit patterns.
    /// </summary>
    [Fact]
    public void R04_DeterministicExpIsPortable()
    {
        (float X, uint Bits)[] cases =
        {
            (0.0f, 0x3f800000),
            (0.5f, 0x3fd3094c),
            (1.0f, 0x402df854),
            (-1.0f, 0x3ebc5ab2),
            (2.5f, 0x4142eb7f),
            (-10.0f, 0x383e6bce),
            (20.0f, 0x4de75844),
            (0.001f, 0x3f8020c9),
            (100.0f, 0x7f800000), // overflows to infinity
        };
        foreach (var (x, bits) in cases)
        {
            Assert.Equal(bits, BitConverter.SingleToUInt32Bits(DeterministicScalar.Exp(x)));
        }
    }

    /// <summary>
    /// R05: the deterministic logarithm, pinned to exact bit patterns.
    /// </summary>
    [Fact]
    public void R05_DeterministicLnIsPortable()
    {
        (float X, uint Bits)[] cases =
        {
            (1.0f, 0x00000000),
            (2.0f, 0x3f317218),
            (0.5f, 0xbf317218),
            (10.0f, 0x40135d8e),
            (0.001f, 0xc0dd0c55),
            (1e-6f, 0xc15d0c55),
            (3.5f, 0x3fa05a89),
            (100.0f, 0x40935d8e),
        };
        foreach (var (x, bits) in cases)
        {
            Assert.Equal(bits, BitConverter.SingleToUInt32Bits(DeterministicScalar.Ln(x)));
        }
    }

    /// <summary>
    /// R06: the deterministic helpers track the platform math library.
    /// </summary>
    /// <remarks>
    /// The tolerance is in ULPs of the result and is far looser than the
    /// actual agreement (which is ~0.5 ULP, i.e. correctly rounded). The
    /// point is not the accuracy - it is that the deterministic version
    /// is <i>close enough</i> that replacing the platform call does not
    /// change the character of the physics.
    /// </remarks>
    [Fact]
    public void R06_ExponentialMatchesPlatformLibm()
    {
        static double UlpDiff(float a, float b)
        {
            if (a == b)
            {
                return 0.0;
            }
            double ad = a;
            double bd = b;
            double scale = System.Math.Max(System.Math.Max(System.Math.Abs(ad), System.Math.Abs(bd)), 1.17549435e-38);
            // 1.1920929e-7 is the f32 machine epsilon (2^-23), matching
            // Rust's f32::EPSILON. C#'s float.Epsilon is the smallest
            // *subnormal*, not the machine epsilon.
            return System.Math.Abs(ad - bd) / (scale * 1.1920929e-7);
        }

        double worstExp = 0.0;
        double worstLn = 0.0;
        for (int i = -60000; i <= 60000; i++)
        {
            float x = i * 0.001f;
            worstExp = System.Math.Max(worstExp, UlpDiff(DeterministicScalar.Exp(x), MathF.Exp(x)));
        }
        for (int i = 1; i < 20000; i++)
        {
            float x = i * 1e-3f;
            worstLn = System.Math.Max(worstLn, UlpDiff(DeterministicScalar.Ln(x), MathF.Log(x)));
        }
        Assert.True(worstExp <= 2.0, $"exp deviates by {worstExp} ULP from the platform libm");
        Assert.True(worstLn <= 2.0, $"ln deviates by {worstLn} ULP from the platform libm");
    }

    /// <summary>
    /// R07: <c>exp</c> and <c>ln</c> are mutual inverses.
    /// </summary>
    /// <remarks>
    /// The nuclear subsystem turns a half-life into a decay constant and
    /// decays an activity with it; this is the property that makes the
    /// round trip meaningful.
    /// </remarks>
    [Fact]
    public void R07_ExponentialAndLogarithmAreMutualInverses()
    {
        for (int i = -2000; i <= 2000; i++)
        {
            float x = i * 0.01f;
            float back = DeterministicScalar.Ln(DeterministicScalar.Exp(x));
            Assert.True(
                MathF.Abs(back - x) <= MathF.Abs(x) * 1e-5f + 1e-6f,
                $"ln(exp({x})) = {back}");
        }
    }
}
