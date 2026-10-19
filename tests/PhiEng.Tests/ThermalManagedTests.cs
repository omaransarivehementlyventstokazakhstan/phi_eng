using System.Numerics;
using PhiEng.Dynamics;
using PhiEng.Native;
using Xunit;

namespace PhiEng.Tests;

/// <summary>
/// Managed thermal mirror tests (T26-T37).
///
/// Every test here runs the SAME scenario through the pure-managed
/// <see cref="World2"/>/<see cref="World3"/> (which now steps the
/// thermal subsystem) and checks the same outcome the Rust core
/// produces, so the two engines are pinned to the same behaviour.
/// Mirrored by the Rust core tests in
/// <c>crates/dynamics/src/thermal.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class ThermalManagedTests
{
    private const float Dt = 1.0f / 60.0f;

    private static World2 Clean() => new World2();

    /// <summary>T26: conduction between two contacting bodies reaches equilibrium.</summary>
    [Fact]
    public void T26_ManagedConductionReachesEquilibrium()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            AmbientTemperature = 200.0f,
            MaxTemperatureRate = 10000.0f,
        };
        // Overlapping circles: the narrowphase generates a contact, which
        // is what drives conduction.
        w.AddStaticBody(new Vector2(0.5f, 0.0f), new CircleShape(1.0f));
        w.AddDynamicBody(new Vector2(-0.5f, 0.0f), new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 200.0f;
        w.Bodies[1].Temperature = 400.0f;
        foreach (var b in w.Bodies)
        {
            b.Material.SpecificHeat = 900.0f;
            b.Material.ThermalConductivity = 10.0f;
        }

        for (int i = 0; i < 500; i++) w.Step(Dt);

        float hot = w.Bodies[1].Temperature;
        float cold = w.Bodies[0].Temperature;
        Assert.True(hot < 400.0f && hot > 200.0f, $"Hot body must cool: {hot}");
        Assert.True(cold >= 200.0f, $"Cold body must warm or hold: {cold}");
    }

    /// <summary>T27: a hot body radiates heat away.</summary>
    [Fact]
    public void T27_ManagedRadiationCoolsHotBody()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = true,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 500.0f;

        for (int i = 0; i < 100; i++) w.Step(Dt);

        Assert.True(w.Bodies[0].Temperature < 500.0f,
            $"Hot body must radiate heat away: {w.Bodies[0].Temperature}");
    }

    /// <summary>T28: a cold body absorbs radiation from the environment.</summary>
    [Fact]
    public void T28_ManagedRadiationWarmsColdBody()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = true,
            ConvectionEnabled = false,
            AmbientTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 200.0f;

        for (int i = 0; i < 100; i++) w.Step(Dt);

        Assert.True(w.Bodies[0].Temperature > 200.0f,
            $"Cold body must absorb radiation: {w.Bodies[0].Temperature}");
    }

    /// <summary>T29: the temperature is clamped to the configured maximum.</summary>
    [Fact]
    public void T29_ManagedTemperatureClampedAtMax()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = false,
            ConvectionEnabled = true,
            AmbientTemperature = 1000.0f,
            MaxTemperature = 400.0f,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 100.0f;

        for (int i = 0; i < 200; i++) w.Step(Dt);

        Assert.True(w.Bodies[0].Temperature <= 400.0f + 1e-3f,
            $"Temperature must respect the cap: {w.Bodies[0].Temperature}");
    }

    /// <summary>T30: a static body is an infinite-capacity thermal reservoir.</summary>
    [Fact]
    public void T30_ManagedStaticBodyIsReservoir()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = true,
            RadiationEnabled = false,
            ConvectionEnabled = true,
            AmbientTemperature = 200.0f,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddStaticBody(new Vector2(0.5f, 0.0f), new CircleShape(1.0f));
        w.AddDynamicBody(new Vector2(-0.5f, 0.0f), new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 200.0f;
        w.Bodies[1].Temperature = 400.0f;

        for (int i = 0; i < 300; i++) w.Step(Dt);

        Assert.Equal(200.0f, w.Bodies[0].Temperature);
        Assert.True(w.Bodies[1].Temperature < 400.0f,
            $"The dynamic body must cool: {w.Bodies[1].Temperature}");
    }

    /// <summary>T31: a disabled thermal config is a no-op.</summary>
    [Fact]
    public void T31_ManagedDisabledConfigIsNoop()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2 { Enabled = false };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 400.0f;

        for (int i = 0; i < 100; i++) w.Step(Dt);

        Assert.Equal(400.0f, w.Bodies[0].Temperature);
    }

    /// <summary>T32: the managed thermal phase is deterministic.</summary>
    [Fact]
    public void T32_ManagedThermalDeterminism()
    {
        float Run()
        {
            var w = Clean();
            w.Thermal = new ThermalConfig2
            {
                Enabled = true,
                AmbientTemperature = 250.0f,
                MaxTemperatureRate = 10000.0f,
            };
            w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
            w.Bodies[0].Temperature = 400.0f;
            for (int i = 0; i < 100; i++) w.Step(Dt);
            return w.Bodies[0].Temperature;
        }

        Assert.Equal(Run(), Run());
    }

    /// <summary>T33: radiation occlusion blocks the exchange between two
    /// bodies when a third body lies between them.</summary>
    [Fact]
    public void T33_ManagedRadiationOcclusionBlocksExchange()
    {
        float Build(bool occlusion)
        {
            var w = Clean();
            w.Thermal = new ThermalConfig2
            {
                Enabled = true,
                ConductionEnabled = false,
                RadiationEnabled = true,
                ConvectionEnabled = false,
                RadiationOcclusion = occlusion,
                AmbientTemperature = 293.15f,
                MaxTemperatureRate = 10000.0f,
            };
            w.AddDynamicBody(new Vector2(0.0f, 0.0f), new CircleShape(0.4f), 1.0f);
            w.AddDynamicBody(new Vector2(2.0f, 0.0f), new CircleShape(0.4f), 1.0f);
            w.AddDynamicBody(new Vector2(1.0f, 0.0f), new CircleShape(0.4f), 1.0f);
            w.Bodies[0].Temperature = 600.0f;
            w.Bodies[1].Temperature = 300.0f;
            w.Bodies[2].Temperature = 400.0f;

            for (int i = 0; i < 10; i++) w.Step(Dt);
            return w.Bodies[0].Temperature;
        }

        float blocked = Build(true);
        float unblocked = Build(false);
        Assert.True(blocked > unblocked,
            $"Occlusion must reduce body-to-body exchange: blocked {blocked} vs unblocked {unblocked}");
    }

    /// <summary>T34: forced convection cools a moving body faster.</summary>
    [Fact]
    public void T34_ManagedForcedConvectionCoolsMovingBodyFaster()
    {
        float Run(float speed)
        {
            var w = Clean();
            w.Thermal = new ThermalConfig2
            {
                Enabled = true,
                ConductionEnabled = false,
                RadiationEnabled = false,
                ConvectionEnabled = true,
                ForcedConvectionCoefficient = 50.0f,
                AmbientTemperature = 293.15f,
                MaxTemperatureRate = 10000.0f,
            };
            w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
            w.Bodies[0].Temperature = 500.0f;
            w.Bodies[0].Velocity = new Vector2(speed, 0.0f);

            for (int i = 0; i < 60; i++) w.Step(Dt);
            return w.Bodies[0].Temperature;
        }

        float still = Run(0.0f);
        float moving = Run(10.0f);
        Assert.True(moving < still,
            $"Moving body must cool faster: moving {moving} vs still {still}");
    }

    /// <summary>T35: thermal expansion scales a hot body's shape to the
    /// target factor, without compounding.</summary>
    [Fact]
    public void T35_ManagedExpansionScalesShape()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            ExpansionCoefficient = 1.0e-3f,
            ExpansionReferenceTemperature = 293.15f,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 500.0f;

        float radiusBefore = ((CircleShape)w.Bodies[0].Shape!).Radius;
        for (int i = 0; i < 10; i++) w.Step(Dt);
        float radiusAfter = ((CircleShape)w.Bodies[0].Shape!).Radius;

        Assert.True(radiusAfter > radiusBefore,
            $"Hot body must expand: {radiusBefore} -> {radiusAfter}");
        float expected = 1.0f + 1.0e-3f * (500.0f - 293.15f);
        Assert.True(MathF.Abs(radiusAfter - expected) < 1e-4f,
            $"Expected scale {expected}, got {radiusAfter}");
    }

    /// <summary>T36: expansion is opt-in — with the default (zero)
    /// coefficient the shape must not change.</summary>
    [Fact]
    public void T36_ManagedExpansionDisabledIsNoop()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            ConductionEnabled = false,
            RadiationEnabled = false,
            ConvectionEnabled = false,
            MaxTemperatureRate = 10000.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 500.0f;

        float radiusBefore = ((CircleShape)w.Bodies[0].Shape!).Radius;
        for (int i = 0; i < 10; i++) w.Step(Dt);
        float radiusAfter = ((CircleShape)w.Bodies[0].Shape!).Radius;

        Assert.Equal(radiusBefore, radiusAfter);
    }

    /// <summary>T37: the managed thermal state round-trips through a
    /// snapshot (per-body temperature, material thermal properties, the
    /// config and the applied expansion factor).</summary>
    [Fact]
    public void T37_ManagedThermalSnapshotRoundtrip()
    {
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            AmbientTemperature = 320.0f,
            ConvectionCoefficient = 12.5f,
            RadiationOcclusion = true,
            ForcedConvectionCoefficient = 7.25f,
            ExpansionCoefficient = 1.5e-4f,
            ExpansionReferenceTemperature = 280.0f,
            MaxTemperatureRate = 42.0f,
        };
        w.AddDynamicBody(Vector2.Zero, new CircleShape(1.0f), 1.0f);
        w.Bodies[0].Temperature = 456.0f;
        w.Bodies[0].ExpansionScale = 1.25f;
        w.Bodies[0].Material.SpecificHeat = 1234.0f;
        w.Bodies[0].Material.ThermalConductivity = 5.5f;
        w.Bodies[0].Material.Emissivity = 0.42f;

        byte[] bin = Snapshot.World2ToBinary(w, SnapshotLoadMode.Exact);
        var back = Snapshot.World2FromBinary(bin, SnapshotLoadMode.Exact);

        Assert.True(back.Thermal.Enabled);
        Assert.Equal(320.0f, back.Thermal.AmbientTemperature);
        Assert.Equal(12.5f, back.Thermal.ConvectionCoefficient);
        Assert.True(back.Thermal.RadiationOcclusion);
        Assert.Equal(7.25f, back.Thermal.ForcedConvectionCoefficient);
        Assert.Equal(1.5e-4f, back.Thermal.ExpansionCoefficient);
        Assert.Equal(280.0f, back.Thermal.ExpansionReferenceTemperature);
        Assert.Equal(42.0f, back.Thermal.MaxTemperatureRate);

        Assert.Equal(456.0f, back.Bodies[0].Temperature);
        Assert.Equal(1.25f, back.Bodies[0].ExpansionScale);
        Assert.Equal(1234.0f, back.Bodies[0].Material.SpecificHeat);
        Assert.Equal(5.5f, back.Bodies[0].Material.ThermalConductivity);
        Assert.Equal(0.42f, back.Bodies[0].Material.Emissivity);
    }

    /// <summary>
    /// A file written by the MANAGED engine must load in the Rust core
    /// (and vice versa). The two engines share the "PHE1" envelope magic
    /// and the schema version space, so the binary layout must agree
    /// byte-for-byte — otherwise one engine would silently mis-parse the
    /// other's payload.
    /// </summary>
    [Fact]
    public void F40_ManagedFileLoadsInRustCore()
    {
        if (!OperatingSystem.IsWindows()) return;
        var w = Clean();
        w.Thermal = new ThermalConfig2
        {
            Enabled = true,
            AmbientTemperature = 320.0f,
            ExpansionCoefficient = 1.0e-3f,
        };
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int ball = w.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        w.Bodies[ball].Temperature = 512.0f;
        w.Bodies[ball].ExpansionScale = 1.1f;

        string path = Path.Combine(Path.GetTempPath(), "phieng_managed_thermal.bin");
        try
        {
            File.WriteAllBytes(path, Snapshot.World2ToBinary(w, SnapshotLoadMode.Exact));

            // The Rust core must accept the managed payload.
            using var native = PhiNativeWorld.Load(path, 0);
            Assert.Equal(w.Bodies.Count, native.BodyCount);
            var (npos, _) = native.GetBodyTransform(ball);
            var mpos = w.Bodies[ball].Transform.Position;
            Assert.Equal(
                BitConverter.SingleToInt32Bits(mpos.X),
                BitConverter.SingleToInt32Bits(npos.X));
            Assert.Equal(
                BitConverter.SingleToInt32Bits(mpos.Y),
                BitConverter.SingleToInt32Bits(npos.Y));
            // The thermal state must survive the crossing too.
            Assert.Equal(512.0f, native.GetBodyTemperature(ball));
        }
        finally
        {
            try { File.Delete(path); } catch { }
        }
    }
}
