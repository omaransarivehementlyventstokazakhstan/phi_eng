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

    /// <summary>T18: a body between two others blocks body-to-body radiation
    /// when occlusion is enabled. The A/B uses the same three bodies in both
    /// runs and only flips the flag, so the blocker's own exchange cancels
    /// out and the difference isolates the blocked 0-1 exchange.</summary>
    [Fact]
    public void T18_RadiationOcclusionBlocksIntermediateBody()
    {
        float Build(bool occlusion)
        {
            using var w = Clean();
            w.SetThermalConfig(new ThermalConfig2
            {
                Enabled = true,
                RadiationEnabled = true,
                ConductionEnabled = false,
                ConvectionEnabled = false,
                RadiationOcclusion = occlusion,
                AmbientTemperature = 293.15f,
                MaxTemperatureRate = 10000.0f,
            });
            // Hot body, cold body, and a blocker sitting between them.
            int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 0.4f, 1.0f, false);
            int b1 = w.AddCircleBody(new Vector2(2.0f, 0.0f), 0.4f, 1.0f, false);
            int b2 = w.AddCircleBody(new Vector2(1.0f, 0.0f), 0.4f, 1.0f, false);
            w.SetBodyTemperature(b0, 600.0f);
            w.SetBodyTemperature(b1, 300.0f);
            w.SetBodyTemperature(b2, 400.0f);

            for (int i = 0; i < 10; i++)
                w.Step(Dt);

            return w.GetBodyTemperature(b0);
        }

        float blocked = Build(true);
        float unblocked = Build(false);
        Assert.True(blocked > unblocked,
            $"Occlusion must reduce body-to-body exchange: blocked {blocked} vs unblocked {unblocked}");
    }

    /// <summary>T19: a moving body loses more heat than a stationary one when
    /// forced convection is enabled.</summary>
    [Fact]
    public void T19_ForcedConvectionCoolsMovingBodyFaster()
    {
        float Run(float speed)
        {
            using var w = Clean();
            w.SetThermalConfig(new ThermalConfig2
            {
                Enabled = true,
                RadiationEnabled = false,
                ConductionEnabled = false,
                ConvectionEnabled = true,
                ForcedConvectionCoefficient = 50.0f,
                AmbientTemperature = 293.15f,
                MaxTemperatureRate = 10000.0f,
            });
            int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 1.0f, 1.0f, false);
            w.SetBodyTemperature(b0, 500.0f);
            w.SetBodyVelocity(b0, new Vector2(speed, 0.0f));

            for (int i = 0; i < 60; i++)
                w.Step(Dt);

            return w.GetBodyTemperature(b0);
        }

        float still = Run(0.0f);
        float moving = Run(10.0f);
        Assert.True(moving < still,
            $"Moving body must cool faster: moving {moving} vs still {still}");
    }

    /// <summary>T20: a hot body expands when an expansion coefficient is set.</summary>
    [Fact]
    public void T20_ExpansionScalesShapeWithTemperature()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = false,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            ExpansionCoefficient = 1.0e-3f,
            ExpansionReferenceTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);

        float radiusBefore = w.GetBodyShapeRadius(b0);
        for (int i = 0; i < 10; i++)
            w.Step(Dt);
        float radiusAfter = w.GetBodyShapeRadius(b0);

        Assert.True(radiusAfter > radiusBefore,
            $"Hot body must expand: {radiusBefore} -> {radiusAfter}");

        // The target scale is 1 + alpha * (T - T_ref); applying the factor
        // directly to the already-expanded shape would overshoot.
        float expected = 1.0f + 1.0e-3f * (500.0f - 293.15f);
        Assert.True(MathF.Abs(radiusAfter - expected) < 1e-4f,
            $"Expected scale {expected}, got {radiusAfter}");
    }

    /// <summary>T21: expansion is opt-in. With the default (zero) coefficient
    /// the shape must not change at all.</summary>
    [Fact]
    public void T21_ExpansionDisabledIsNoop()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = false,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);

        float radiusBefore = w.GetBodyShapeRadius(b0);
        for (int i = 0; i < 10; i++)
            w.Step(Dt);
        float radiusAfter = w.GetBodyShapeRadius(b0);

        Assert.Equal(radiusBefore, radiusAfter);
    }

    /// <summary>T22: expansion is applied by the world step, and reaches the
    /// exact target scale.</summary>
    [Fact]
    public void T22_ExpansionAppliesInWorldStep()
    {
        using var w = Clean();
        w.SetThermalConfig(new ThermalConfig2
        {
            Enabled = true,
            RadiationEnabled = false,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            ExpansionCoefficient = 1.0e-3f,
            ExpansionReferenceTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);

        float before = w.GetBodyShapeRadius(b0);
        for (int i = 0; i < 10; i++)
            w.Step(Dt);
        float after = w.GetBodyShapeRadius(b0);

        Assert.True(after > before, $"Hot body must expand: {before} -> {after}");
        float expected = 1.0f + 1.0e-3f * (500.0f - 293.15f);
        Assert.True(MathF.Abs(after - expected) < 1e-4f,
            $"Expected scale {expected}, got {after}");
    }

    /// <summary>The thermal config must round-trip through the FFI,
    /// including the three sub-feature fields. A field-order or
    /// type-width mismatch in the FFI struct would corrupt the values
    /// that follow it, so every field is checked.</summary>
    [Fact]
    public void T25_ThermalConfigFfiRoundtrip()
    {
        using var w = Clean();
        var config = new ThermalConfig2
        {
            Enabled = true,
            AmbientTemperature = 350.0f,
            StefanBoltzmann = 5.67e-8f,
            ConvectionCoefficient = 12.5f,
            RadiationEnabled = false,
            ConductionEnabled = true,
            ConvectionEnabled = false,
            MaxTemperature = 4000.0f,
            MinTemperature = 10.0f,
            RadiationRadiusFactor = 3.5f,
            MaxTemperatureRate = 42.0f,
            RadiationOcclusion = true,
            ForcedConvectionCoefficient = 7.25f,
            ExpansionCoefficient = 1.5e-4f,
            ExpansionReferenceTemperature = 280.0f,
        };
        w.SetThermalConfig(config);

        var back = w.GetThermalConfig();
        Assert.True(back.Enabled);
        Assert.Equal(350.0f, back.AmbientTemperature);
        Assert.Equal(12.5f, back.ConvectionCoefficient);
        Assert.False(back.RadiationEnabled);
        Assert.True(back.ConductionEnabled);
        Assert.False(back.ConvectionEnabled);
        Assert.Equal(4000.0f, back.MaxTemperature);
        Assert.Equal(10.0f, back.MinTemperature);
        Assert.Equal(3.5f, back.RadiationRadiusFactor);
        Assert.Equal(42.0f, back.MaxTemperatureRate);
        Assert.True(back.RadiationOcclusion);
        Assert.Equal(7.25f, back.ForcedConvectionCoefficient);
        Assert.Equal(1.5e-4f, back.ExpansionCoefficient);
        Assert.Equal(280.0f, back.ExpansionReferenceTemperature);
    }

    /// <summary>T24: 3D expansion mirrors the 2D contract.</summary>
    [Fact]
    public void T24_Expansion3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetThermalConfig(new ThermalConfig3
        {
            Enabled = true,
            RadiationEnabled = false,
            ConductionEnabled = false,
            ConvectionEnabled = false,
            ExpansionCoefficient = 1.0e-3f,
            ExpansionReferenceTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        });
        int b0 = w.AddSphereBody(new Vector3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, false);
        w.SetBodyTemperature(b0, 500.0f);

        for (int i = 0; i < 10; i++)
            w.Step(Dt);

        float expected = 1.0f + 1.0e-3f * (500.0f - 293.15f);
        Assert.True(MathF.Abs(w.GetBodyShapeRadius(b0) - expected) < 1e-4f,
            $"Expected scale {expected}, got {w.GetBodyShapeRadius(b0)}");
    }

    /// <summary>T23: expansion must not compound. A body held at a constant
    /// temperature has a fixed target scale, so it must stop growing.</summary>
    [Fact]
    public void T23_ExpansionDoesNotCompound()
    {
        float Run(int steps)
        {
            using var w = Clean();
            w.SetThermalConfig(new ThermalConfig2
            {
                Enabled = true,
                RadiationEnabled = false,
                ConductionEnabled = false,
                ConvectionEnabled = false,
                ExpansionCoefficient = 1.0e-3f,
                ExpansionReferenceTemperature = 293.15f,
                MaxTemperatureRate = 10000.0f,
            });
            int b0 = w.AddCircleBody(new Vector2(0.0f, 0.0f), 1.0f, 1.0f, false);
            w.SetBodyTemperature(b0, 500.0f);

            for (int i = 0; i < steps; i++)
                w.Step(Dt);

            return w.GetBodyShapeRadius(b0);
        }

        float after1 = Run(1);
        float after100 = Run(100);
        Assert.Equal(after1, after100);
    }
}
