using System.Numerics;
using Xunit;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Native write-surface tests (F41–F46): the bridge could read a body's
/// state but not change it. These pin that a body can be placed, spun,
/// pushed and woken, and that the bodies can be enumerated. Mirrors
/// <c>crates/engine/tests/ffi_write_surface.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class FfiWriteSurfaceTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>F41: a body can be placed, and the world sees where it went.</summary>
    [Fact]
    public void F41_SetBodyTransformMovesTheBody()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        w.SetBodyTransform(id, new Vector2(3.0f, -2.0f), 0.5f);

        var (pos, _) = w.GetBodyTransform(id);
        Assert.Equal(3.0f, pos.X);
        Assert.Equal(-2.0f, pos.Y);
    }

    /// <summary>F42: a body can be spun, and the spin survives a step.</summary>
    [Fact]
    public void F42_SetBodyAngularVelocity()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        w.SetBodyAngularVelocity(id, 2.5f);
        Assert.Equal(2.5f, w.GetBodyAngularVelocity(id));

        w.Step(Dt);
        // The spin survives the step: the body's angular damping is
        // 0.01 per second, so it eases off a little and no more.
        Assert.True(System.Math.Abs(w.GetBodyAngularVelocity(id) - 2.5f) < 0.01f);
    }

    /// <summary>F43: an impulse moves a body, and a force does too.</summary>
    [Fact]
    public void F43_ApplyImpulseAndForce()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);

        w.ApplyImpulse(id, new Vector2(2.0f, 0.0f));
        w.Step(Dt);
        float x = w.GetBodyTransform(id).Position.X;
        Assert.True(x > 0.0f, $"the impulse must have moved the body (x={x})");

        w.ApplyForce(id, new Vector2(10.0f, 0.0f), 1.0f);
        w.Step(Dt);
        float after = w.GetBodyTransform(id).Position.X;
        Assert.True(after > x, $"the force must have moved the body further (x={after})");
    }

    /// <summary>F44: a body can be woken, and an unknown id is refused.</summary>
    [Fact]
    public void F44_WakeBody()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        w.WakeBody(id);
        Assert.Throws<ArgumentOutOfRangeException>(() => w.WakeBody(9999));
    }

    /// <summary>F45: the bodies can be enumerated.</summary>
    [Fact]
    public void F45_BodyEnumeration()
    {
        using var w = new PhiNativeWorld();
        int a = w.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        int b = w.AddDynamicBall(new Vector2(2.0f, 5.0f), 0.5f, 1.0f);
        Assert.Equal(a, w.GetBodyIdAt(0));
        Assert.Equal(b, w.GetBodyIdAt(1));
        Assert.Equal(-1, w.GetBodyIdAt(2));
    }

    /// <summary>F46: the write surface refuses a bad body id and a bad value.</summary>
    [Fact]
    public void F46_WriteSurfaceRejectsBadInput()
    {
        using var w = new PhiNativeWorld();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => w.SetBodyTransform(9999, Vector2.Zero, 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => w.SetBodyTransform(0, new Vector2(float.NaN, 0.0f), 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => w.ApplyImpulse(0, new Vector2(float.PositiveInfinity, 0.0f)));
    }
}
