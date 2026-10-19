using System.Numerics;
using Xunit;
using PhiEng.Native;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Thermal subsystem tests (T01–T16): conduction, radiation, convection,
/// clamping, and determinism. Mirrored by the Rust core tests in
/// <c>crates/dynamics/src/thermal.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class ThermalTests
{
    private const float Dt = 1.0f / 60.0f;

    private static PhiNativeWorld Clean()
    {
        return new PhiNativeWorld();
    }

    /// <summary>T01: conduction between two contacting bodies reaches equilibrium.</summary>
    [Fact]
    public void T01_ConductionReachesEquilibrium()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 200.0f,
            MaxTemperatureRate = 10000.0f,
        });
        // Two overlapping circles (slightly offset): narrowphase generates contact → conduction.
        int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, true); // static cold reservoir
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 10.0f, 0.9f);
        w.SetBodyThermalMaterial(b1, 900.0f, 10.0f, 0.9f);

        for (int i = 0; i < 500; i++)
            w.Step(Dt);

        var t0 = w.GetBodyTemperature(b0);
        var t1 = w.GetBodyTemperature(b1);
        // Conduction must have transferred heat from hot (400) toward cold (200).
        Assert.True(t0 < 400.0f && t0 > 200.0f, $"Hot body cooled: T0={t0}");
        Assert.True(t1 >= 200.0f, $"Cold body warmed or held: T1={t1}");
    }

    /// <summary>T02: conduction rate scales with conductivity.</summary>
    [Fact]
    public void T02_ConductionScalesWithConductivity()
    {
        float Run(float k)
        {
            using var w = Clean();
            w.SetThermalConfig(new ThermalConfig2
            {
                Enabled = true,
                ConductionEnabled = true,
                RadiationEnabled = false,
                ConvectionEnabled = false,
                MaxTemperatureRate = 10000.0f,
            });
            int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
            int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, true);
            w.SetBodyTemperature(b0, 400.0f);
            w.SetBodyTemperature(b1, 200.0f);
            w.SetBodyThermalMaterial(b0, 900.0f, k, 0.9f);
            w.SetBodyThermalMaterial(b1, 900.0f, k, 0.9f);

            for (int i = 0; i < 60; i++)
                w.Step(Dt);

            return MathF.Abs(w.GetBodyTemperature(b0) - w.GetBodyTemperature(b1));
        }

        var diffLow = Run(1.0f);
        var diffHigh = Run(100.0f);
        // Higher conductivity → more heat transferred → smaller final difference.
        Assert.True(diffHigh < diffLow,
            $"Higher conductivity should equilibrate faster: low={diffLow}, high={diffHigh}");
    }

    /// <summary>T03: zero conductivity means no heat flow.</summary>
    [Fact]
    public void T03_ZeroConductivityNoHeatFlow()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 0.0f, 0.9f);
        w.SetBodyThermalMaterial(b1, 900.0f, 0.0f, 0.9f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.Equal(400.0f, w.GetBodyTemperature(b0));
        Assert.Equal(200.0f, w.GetBodyTemperature(b1));
    }

    /// <summary>T09: temperature clamped at max.</summary>
    [Fact]
    public void T09_TemperatureClampedAtMax()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 5000.0f,
            MaxTemperature = 5000.0f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 4999.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 1.0f, 0.9f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b0) <= 5000.0f);
    }

    /// <summary>T10: temperature clamped at min (absolute zero).</summary>
    [Fact]
    public void T10_TemperatureClampedAtMin()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 0.001f,
            MinTemperature = 0.0f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 1.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 1.0f, 0.9f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b0) >= 0.0f);
    }

    /// <summary>T11: static body acts as a thermal reservoir.</summary>
    [Fact]
    public void T11_StaticBodyIsReservoir()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, true); // static
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 10.0f, 0.9f);
        w.SetBodyThermalMaterial(b1, 900.0f, 10.0f, 0.9f);

        for (int i = 0; i < 500; i++)
            w.Step(Dt);

        Assert.Equal(200.0f, w.GetBodyTemperature(b1), 5);
        Assert.True(w.GetBodyTemperature(b0) > 200.0f);
        Assert.True(w.GetBodyTemperature(b0) < 400.0f);
    }

    /// <summary>T12: disabled config is a no-op.</summary>
    [Fact]
    public void T12_DisabledConfigIsNoop()
    {
        using var w = Clean();
        // ThermalConfig2 defaults to Enabled = false
        w.SetThermalConfig(new ThermalConfig2());
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.Equal(400.0f, w.GetBodyTemperature(b0));
        Assert.Equal(200.0f, w.GetBodyTemperature(b1));
    }

    /// <summary>T13: conduction disabled means no heat flow through contact.</summary>
    [Fact]
    public void T13_ConductionDisabled()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.Equal(400.0f, w.GetBodyTemperature(b0));
        Assert.Equal(200.0f, w.GetBodyTemperature(b1));
    }

    /// <summary>T16: same initial conditions produce bit-identical temperatures.</summary>
    [Fact]
    public void T16_Determinism()
    {
        float Run()
        {
            using var w = Clean();
            w.SetThermalConfig(new ThermalConfig2
            {
                Enabled = true,
                ConductionEnabled = true,
                RadiationEnabled = true,
                ConvectionEnabled = true,
                MaxTemperatureRate = 10000.0f,
            });
            int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
            int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, true);
            w.SetBodyTemperature(b0, 400.0f);
            w.SetBodyTemperature(b1, 200.0f);
            w.SetBodyThermalMaterial(b0, 900.0f, 10.0f, 0.9f);
            w.SetBodyThermalMaterial(b1, 900.0f, 10.0f, 0.9f);

            for (int i = 0; i < 100; i++)
                w.Step(Dt);

            return w.GetBodyTemperature(b0);
        }

        Assert.Equal(Run(), Run());
    }

    /// <summary>T04: hotter than ambient, net radiative loss cools the body.</summary>
    [Fact]
    public void T04_RadiationCoolsHotBody()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 1.0f, 0.9f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b0) < 500.0f);
    }

    /// <summary>T05: colder than ambient, net radiative gain warms the body.</summary>
    [Fact]
    public void T05_RadiationWarmsColdBody()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 200.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 1.0f, 0.9f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b0) > 200.0f);
    }

    /// <summary>T06: convection approaches ambient monotonically without overshoot.</summary>
    [Fact]
    public void T06_ConvectionMonotoneToAmbient()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = false,
            ConductionEnabled = false,
            ConvectionEnabled = true,
            AmbientTemperature = 300.0f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 1.0f, 0.0f);

        float prev = 500.0f;
        for (int i = 0; i < 600; i++)
        {
            w.Step(Dt);
            float t = w.GetBodyTemperature(b0);
            Assert.True(t <= prev + 1e-3f, "convection must not heat a hot body");
            Assert.True(t >= 300.0f - 1e-3f, "must not cross below ambient");
            prev = t;
        }
        Assert.True(prev < 500.0f && prev > 300.0f, $"final temp {prev}");
    }

    /// <summary>T07: nearby bodies exchange radiative heat body-to-body.</summary>
    [Fact]
    public void T07_NearbyBodiesExchangeRadiation()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            RadiationRadiusFactor = 5.0f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(-0.5f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);

        for (int i = 0; i < 200; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b1) > 200.0f, "cold body must gain heat nearby");
        Assert.True(w.GetBodyTemperature(b0) < 400.0f, "hot body must lose heat nearby");
    }

    /// <summary>T08: beyond the cutoff, body-to-body exchange stops.</summary>
    [Fact]
    public void T08_RadiationCutoffStopsExchange()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = true,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            RadiationRadiusFactor = 5.0f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(-500.0f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddCircleBody(new Vector2(500.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);

        for (int i = 0; i < 200; i++)
            w.Step(Dt);

        float cold = w.GetBodyTemperature(b1);
        Assert.True(cold > 200.0f, "cold body still sees ambient");
        Assert.True(cold <= 293.15f + 1e-3f, $"cold body must not gain from distant body, got {cold}");
    }

    /// <summary>T14: 3D conduction moves heat hot-to-cold through a contact.</summary>
    [Fact]
    public void T14_Conduction3DMovesHeat()
    {
        using var w = new PhiNativeWorld3();
        w.SetThermalConfig(new ThermalConfig3
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddSphereBody(new Vector3(-0.5f, 0.0f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddSphereBody(new Vector3(0.5f, 0.0f, 0.0f), 1.0f, 1.0f, true);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);
        w.SetBodyThermalMaterial(b0, 900.0f, 10.0f, 0.9f);
        w.SetBodyThermalMaterial(b1, 900.0f, 10.0f, 0.9f);

        for (int i = 0; i < 500; i++)
            w.Step(Dt);

        Assert.True(w.GetBodyTemperature(b0) < 400.0f);
        Assert.True(w.GetBodyTemperature(b1) >= 200.0f);
    }

    /// <summary>T15: 3D determinism mirrors the 2D contract.</summary>
    [Fact]
    public void T15_Determinism3D()
    {
        float Run()
        {
            using var w = new PhiNativeWorld3();
            w.SetThermalConfig(new ThermalConfig3
            {
                Enabled = true,
                RadiationEnabled = true,
                MaxTemperatureRate = 10000.0f,
            });
            int b0 = w.AddSphereBody(new Vector3(-0.5f, 0.0f, 0.0f), 1.0f, 1.0f, false);
            int b1 = w.AddSphereBody(new Vector3(0.5f, 0.0f, 0.0f), 1.0f, 1.0f, true);
            w.SetBodyTemperature(b0, 400.0f);
            w.SetBodyTemperature(b1, 200.0f);

            for (int i = 0; i < 100; i++)
                w.Step(Dt);

            return w.GetBodyTemperature(b0);
        }

        Assert.Equal(Run(), Run());
    }

    /// <summary>T17: 3D disabled config is a no-op.</summary>
    [Fact]
    public void T17_DisabledConfig3DIsNoop()
    {
        using var w = new PhiNativeWorld3();
        w.SetThermalConfig(new ThermalConfig3()); // disabled by default
        int b0 = w.AddSphereBody(new Vector3(-0.5f, 0.0f, 0.0f), 1.0f, 1.0f, false);
        int b1 = w.AddSphereBody(new Vector3(0.5f, 0.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 400.0f);
        w.SetBodyTemperature(b1, 200.0f);

        for (int i = 0; i < 100; i++)
            w.Step(Dt);

        Assert.Equal(400.0f, w.GetBodyTemperature(b0));
        Assert.Equal(200.0f, w.GetBodyTemperature(b1));
    }
}
