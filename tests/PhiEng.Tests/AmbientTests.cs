using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Ambient force tests (A01–A05): spring/SHM, air drag, turbulence.
/// Mirrored by the Rust core tests in
/// <c>crates/engine/tests/ambient.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class AmbientTests
{
    private const float Dt = 1.0f / 240.0f;

    private static World2 Clean()
    {
        var w = new World2
        {
            Gravity = Vector2.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };
        return w;
    }

    private static void NoDamping(World2 w)
    {
        foreach (var b in w.Bodies)
        {
            b.LinearDamping = 0.0f;
            b.AngularDamping = 0.0f;
        }
    }

    /// <summary>A01: undamped spring oscillates about the anchor and closes a period.</summary>
    [Fact]
    public void A01_SpringShmPeriod()
    {
        var w = Clean();
        int id = w.AddDynamicBody(new Vector2(1.0f, 0.0f), new CircleShape(0.2f), 1.0f);
        NoDamping(w);
        w.Ambient = new AmbientConfig2
        {
            Spring = new SpringConfig2
            {
                Enabled = true,
                Anchor = Vector2.Zero,
                Stiffness = 40.0f,
                Damping = 0.0f,
                MaxDisplacement = 0.0f,
            },
            Drag = DragConfig2.Default,
            Turbulence = TurbulenceConfig2.Default,
        };
        float omega = MathF.Sqrt(40.0f / w.Bodies[id].Mass);
        float period = 2.0f * MathF.PI / omega;
        int steps = (int)MathF.Round(period / Dt);
        bool crossed = false;
        for (int i = 0; i < steps; i++)
        {
            w.Step(Dt);
            if (w.Bodies[id].Transform.Position.X < 0.0f) crossed = true;
        }
        Assert.True(crossed, "spring never crossed the anchor");
        Assert.True(MathF.Abs(w.Bodies[id].Transform.Position.X - 1.0f) < 0.05f,
            $"SHM did not close a period: x={w.Bodies[id].Transform.Position.X}");
    }

    /// <summary>A02: linear drag + gravity reaches terminal velocity g*m/linear.</summary>
    [Fact]
    public void A02_DragTerminalVelocity()
    {
        var w = Clean();
        w.Gravity = new Vector2(0.0f, -9.81f);
        int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.3f), 1.0f);
        NoDamping(w);
        const float linear = 4.0f;
        w.Ambient = new AmbientConfig2
        {
            Spring = SpringConfig2.Default,
            Drag = new DragConfig2
            {
                Enabled = true,
                Linear = linear,
                Quadratic = 0.0f,
                AirDensity = 1.0f,
                Angular = 0.0f,
                Wind = Vector2.Zero,
            },
            Turbulence = TurbulenceConfig2.Default,
        };
        for (int i = 0; i < 4000; i++) w.Step(Dt);
        float expected = -9.81f * w.Bodies[id].Mass / linear;
        Assert.True(MathF.Abs(w.Bodies[id].Velocity.Y - expected) < 0.2f,
            $"terminal velocity off: {w.Bodies[id].Velocity.Y} want {expected}");
    }

    /// <summary>A03: turbulence is deterministic across identical runs.</summary>
    [Fact]
    public void A03_TurbulenceDeterministic()
    {
        static Vector2 Run()
        {
            var w = Clean();
            int id = w.AddDynamicBody(new Vector2(1.0f, 2.0f), new CircleShape(0.3f), 1.0f);
            NoDamping(w);
            w.Ambient = new AmbientConfig2
            {
                Spring = SpringConfig2.Default,
                Drag = DragConfig2.Default,
                Turbulence = new TurbulenceConfig2
                {
                    Enabled = true,
                    Amplitude = 5.0f,
                    Frequency = 0.5f,
                    TimeScale = 1.0f,
                    Seed = 42,
                    Octaves = 4,
                    DivergenceFree = true,
                },
            };
            for (int i = 0; i < 200; i++) w.Step(1.0f / 60.0f);
            return w.Bodies[id].Velocity;
        }
        var a = Run();
        var b = Run();
        Assert.Equal(a.X, b.X);
        Assert.Equal(a.Y, b.Y);
    }

    /// <summary>A04: a default ambient config is inert (exact free fall).</summary>
    [Fact]
    public void A04_AmbientDefaultNoop()
    {
        var w = Clean();
        w.Gravity = new Vector2(0.0f, -9.81f);
        int id = w.AddDynamicBody(new Vector2(0.0f, 10.0f), new CircleShape(0.3f), 1.0f);
        NoDamping(w);
        for (int i = 0; i < 240; i++) w.Step(1.0f / 60.0f);
        Assert.True(MathF.Abs(w.Bodies[id].Velocity.Y - (-9.81f * 4.0f)) < 1e-2f,
            $"default ambient perturbed free fall: {w.Bodies[id].Velocity.Y}");
    }

    /// <summary>A05: ambient_scale == 0 makes a body immune to ambient forces.</summary>
    [Fact]
    public void A05_AmbientScaleImmune()
    {
        var w = Clean();
        int id = w.AddDynamicBody(new Vector2(3.0f, 0.0f), new CircleShape(0.3f), 1.0f);
        NoDamping(w);
        w.Bodies[id].AmbientScale = 0.0f;
        w.Ambient = new AmbientConfig2
        {
            Spring = new SpringConfig2
            {
                Enabled = true,
                Anchor = Vector2.Zero,
                Stiffness = 500.0f,
                Damping = 0.5f,
                MaxDisplacement = 0.0f,
            },
            Drag = DragConfig2.Default,
            Turbulence = TurbulenceConfig2.Default,
        };
        for (int i = 0; i < 60; i++) w.Step(1.0f / 60.0f);
        Assert.Equal(Vector2.Zero, w.Bodies[id].Velocity);
    }
}
