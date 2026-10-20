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
}
