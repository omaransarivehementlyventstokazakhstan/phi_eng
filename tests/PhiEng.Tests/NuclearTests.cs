using System.Numerics;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Nuclear physics (Y01-Y20). Mirrors
/// <c>crates/engine/tests/nuclear.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// The nuclear subsystem is a game-scale model: decay, activation,
/// fission and a rate-based chain reaction. It is off by default, so a
/// default world is bit-identical to a pre-nuclear one.
/// </remarks>
public class NuclearTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>A nuclide index. Index 0 is the "no nuclide" placeholder.</summary>
    private const ushort NoNuclide = 0;

    /// <summary>A short-lived nuclide (half-life in nuclear seconds).</summary>
    private const ushort Short = 1;

    /// <summary>A stable nuclide.</summary>
    private const ushort Stable = 2;

    /// <summary>A fissile nuclide.</summary>
    private const ushort Fissile = 3;

    /// <summary>A configuration with a small nuclide table.</summary>
    private static NuclearConfig Config()
    {
        var cfg = new NuclearConfig
        {
            Enabled = true,
            // Scaled so the fission effects are far above the float
            // noise floor.
            FissionEnergy = 1.0e6f,
            FuelPerFission = 1.0e-3f,
        };
        cfg.Isotopes = new List<Isotope>
        {
            // 0: no nuclide.
            new Isotope(),
            // 1: a short-lived nuclide that decays in 10 nuclear seconds.
            new Isotope
            {
                HalfLife = 10.0f,
                DecayEnergy = 1.0f,
                IonizationFraction = 0.5f,
                Daughter = Stable,
            },
            // 2: stable.
            new Isotope { HalfLife = 0.0f, DecayEnergy = 0.0f },
            // 3: fissile.
            new Isotope
            {
                HalfLife = 0.0f,
                DecayEnergy = 0.0f,
                IonizationFraction = 0.0f,
                Daughter = NoNuclide,
                AbsorptionCrossSection = 1.0f,
                FissionCrossSection = 1.0f,
                NeutronYield = 2.0f,
                IsFissile = true,
            },
        };
        return cfg;
    }

    /// <summary>A world with one dynamic body carrying a nuclide.</summary>
    private static (World2 World, int Body) WorldWith(ushort nuclide, float activity)
    {
        var world = new World2
        {
            Gravity = Vector2.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };
        world.Nuclear.Config = Config();

        int id = world.AddDynamicBody(Vector2.Zero, new CircleShape(0.5f), 1.0f);
        world.Bodies[id].Nuclear.Isotope = nuclide;
        world.Bodies[id].Nuclear.Activity = activity;
        return (world, id);
    }

    /// <summary>Y01: a disabled nuclear config is a no-op.</summary>
    [Fact]
    public void Y01_DisabledConfigIsANoop()
    {
        var (world, id) = WorldWith(Short, 10.0f);
        world.Nuclear.Config.Enabled = false;

        float beforeActivity = world.Bodies[id].Nuclear.Activity;
        float beforeTemperature = world.Bodies[id].Temperature;
        world.Step(Dt);

        Assert.True(
            BitConverter.SingleToUInt32Bits(beforeActivity)
            == BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
            "a disabled nuclear config must not decay anything");
        Assert.True(
            BitConverter.SingleToUInt32Bits(beforeTemperature)
            == BitConverter.SingleToUInt32Bits(world.Bodies[id].Temperature),
            "a disabled nuclear config must not heat anything");
    }

    /// <summary>Y02: the activity halves after one half-life.</summary>
    [Fact]
    public void Y02_ActivityHalvesAfterAHalfLife()
    {
        var (world, id) = WorldWith(Short, 8.0f);
        world.Nuclear.Config.TimeScale = 60.0f;

        // 60 steps of 1/60 s is one simulation second, which with the time
        // scale is 60 nuclear seconds: exactly six half-lives of a
        // 10-nuclear-second nuclide, so the activity is 8 / 2^6.
        for (int i = 0; i < 60; i++)
        {
            world.Step(Dt);
        }
        float activity = world.Bodies[id].Nuclear.Activity;
        float expected = 8.0f / 64.0f;
        Assert.True(
            MathF.Abs(activity - expected) <= expected * 1e-3f,
            $"after six half-lives the activity should be {expected}, got {activity}");
    }

    /// <summary>Y03: the decay energy heats the body.</summary>
    [Fact]
    public void Y03_DecayHeatsTheBody()
    {
        var (world, id) = WorldWith(Short, 10.0f);
        float before = world.Bodies[id].Temperature;
        world.Step(Dt);
        Assert.True(
            world.Bodies[id].Temperature > before,
            $"a decaying body must get hotter: {before} -> {world.Bodies[id].Temperature}");
    }

    /// <summary>Y04: the temperature respects the thermal clamps.</summary>
    [Fact]
    public void Y04_TemperatureRespectsTheThermalClamps()
    {
        var (world, id) = WorldWith(Short, 1e9f);
        world.Thermal.MaxTemperature = 400.0f;
        world.Thermal.MinTemperature = 0.0f;

        for (int i = 0; i < 120; i++)
        {
            world.Step(Dt);
        }
        float temperature = world.Bodies[id].Temperature;
        Assert.True(
            temperature <= 400.0f && temperature >= 0.0f,
            $"the temperature must stay inside the thermal clamps, got {temperature}");
    }

    /// <summary>Y05: the activity is clamped at <c>max_activity</c>.</summary>
    [Fact]
    public void Y05_ActivityIsClamped()
    {
        var (world, id) = WorldWith(Short, 1e30f);
        world.Nuclear.Config.MaxActivity = 100.0f;
        world.Step(Dt);
        Assert.True(
            world.Bodies[id].Nuclear.Activity <= 100.0f,
            $"the activity must be clamped, got {world.Bodies[id].Nuclear.Activity}");
    }

    /// <summary>Y06: the energy budget balances.</summary>
    [Fact]
    public void Y06_TheEnergyBudgetBalances()
    {
        var (world, id) = WorldWith(Short, 10.0f);
        float activity = world.Bodies[id].Nuclear.Activity;
        float dtNuclear = world.Nuclear.Config.TimeScale * Dt;
        float expected = activity * 1.0f * dtNuclear;

        world.Step(Dt);
        float heat = world.Nuclear.Scratch.HeatAt(id);
        float radiation = world.Nuclear.Scratch.RadiationAt(id);
        float total = heat + radiation;

        Assert.True(
            MathF.Abs(total - expected) <= expected * 1e-4f,
            $"the released energy must be split into heat and radiation: "
            + $"expected {expected}, got {total} (heat {heat}, radiation {radiation})");
    }

    /// <summary>Y07: a stable nuclide never decays.</summary>
    [Fact]
    public void Y07_AStableNuclideNeverDecays()
    {
        var (world, id) = WorldWith(Stable, 10.0f);
        world.Step(Dt);
        Assert.True(
            BitConverter.SingleToUInt32Bits(0.0f)
            == BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
            "a stable nuclide has no activity");
    }

    /// <summary>Y08: a body with no nuclide takes no part.</summary>
    [Fact]
    public void Y08_ABodyWithoutANuclideIsUntouched()
    {
        var (world, id) = WorldWith(NoNuclide, 10.0f);
        float before = world.Bodies[id].Nuclear.Activity;
        world.Step(Dt);
        Assert.True(
            BitConverter.SingleToUInt32Bits(before)
            == BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
            "a body without a nuclide must be untouched");
    }

    /// <summary>Y09: a fission consumes fuel.</summary>
    [Fact]
    public void Y09_AFissionConsumesFuel()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1.0f;

        world.Step(Dt);
        Assert.True(
            world.Bodies[id].Nuclear.Fuel < 1.0f,
            $"a fission must consume fuel, got {world.Bodies[id].Nuclear.Fuel}");
    }

    /// <summary>Y10: a fission releases heat.</summary>
    [Fact]
    public void Y10_AFissionReleasesHeat()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1.0f;

        float before = world.Bodies[id].Temperature;
        world.Step(Dt);
        Assert.True(world.Bodies[id].Temperature > before, "a fission must release heat");
    }

    /// <summary>Y11: a chain reaction grows when the fuel is above the threshold.</summary>
    [Fact]
    public void Y11_ASupercriticalChainReactionGrows()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1.0f;
        world.Nuclear.Config.FissionRate = 10.0f;
        world.Nuclear.Config.FuelPerFission = 1e-9f;

        // Two bodies close together: the neutrons one produces feed the
        // other, so the reaction is supercritical.
        int other = world.AddDynamicBody(new Vector2(0.5f, 0.0f), new CircleShape(0.2f), 1.0f);
        world.Bodies[other].Nuclear.Isotope = Fissile;
        world.Bodies[other].Nuclear.Fuel = 1.0f;

        float totalHeat = 0.0f;
        for (int i = 0; i < 30; i++)
        {
            world.Step(Dt);
            totalHeat += world.Nuclear.Scratch.HeatAt(id);
        }

        Assert.True(totalHeat > 0.0f, "a supercritical chain reaction must produce heat");
    }

    /// <summary>Y12: a subcritical chain reaction dies out.</summary>
    [Fact]
    public void Y12_ASubcriticalChainReactionDiesOut()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        // No ambient flux and no neighbours: nothing sustains the reaction.
        world.Nuclear.Config.AmbientFlux = 0.0f;

        float totalHeat = 0.0f;
        for (int i = 0; i < 30; i++)
        {
            world.Step(Dt);
            totalHeat += world.Nuclear.Scratch.HeatAt(id);
        }

        Assert.True(
            totalHeat <= 0.0f,
            $"a subcritical chain reaction must not produce heat, got {totalHeat}");
    }

    /// <summary>Y13: the delayed neutrons hold some of the release back.</summary>
    [Fact]
    public void Y13_DelayedNeutronsHoldTheReleaseBack()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1.0f;
        world.Nuclear.Config.DelayedNeutronFraction = 1.0f;
        world.Nuclear.Config.DelayedNeutronDelay = 100.0f;

        world.Step(Dt);
        Assert.True(
            world.Bodies[id].Nuclear.DelayedNeutrons > 0.0f,
            "the delayed neutrons must be held back");
        Assert.True(
            world.Bodies[id].Nuclear.Neutrons <= 0.0f,
            "no prompt neutrons may be released when every neutron is delayed");
    }

    /// <summary>Y14: the fuel is never negative.</summary>
    [Fact]
    public void Y14_TheFuelIsNeverNegative()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1e-6f;
        world.Nuclear.Config.AmbientFlux = 1e6f;
        world.Nuclear.Config.FissionRate = 1e6f;

        for (int i = 0; i < 60; i++)
        {
            world.Step(Dt);
            Assert.True(
                world.Bodies[id].Nuclear.Fuel >= 0.0f,
                $"the fuel must not go negative, got {world.Bodies[id].Nuclear.Fuel}");
        }
    }

    /// <summary>Y15: the 3D mirror decays too.</summary>
    [Fact]
    public void Y15_3dDecayHeatsTheBody()
    {
        var world = new World3
        {
            Gravity = Vector3.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };
        world.Nuclear.Config = Config();

        int id = world.AddDynamicBody(Vector3.Zero, new SphereShape3(0.5f), 1.0f);
        world.Bodies[id].Nuclear.Isotope = Short;
        world.Bodies[id].Nuclear.Activity = 10.0f;

        float before = world.Bodies[id].Temperature;
        world.Step(Dt);
        Assert.True(
            world.Bodies[id].Temperature > before,
            "a decaying 3D body must get hotter");
    }

    /// <summary>Y16: the nuclear phase is deterministic.</summary>
    [Fact]
    public void Y16_TheNuclearPhaseIsDeterministic()
    {
        (uint Activity, uint Temperature) Run()
        {
            var (world, id) = WorldWith(Short, 10.0f);
            world.Nuclear.Config.AmbientFlux = 1.0f;
            for (int i = 0; i < 120; i++)
            {
                world.Step(Dt);
            }
            return (
                BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
                BitConverter.SingleToUInt32Bits(world.Bodies[id].Temperature));
        }

        Assert.Equal(Run(), Run());
    }

    /// <summary>Y17: the parallel solver lands on the same trajectory.</summary>
    [Fact]
    public void Y17_ParallelLandsOnTheSameTrajectory()
    {
        (uint Activity, uint Temperature) Run(bool parallel)
        {
            var (world, id) = WorldWith(Short, 10.0f);
            world.Nuclear.Config.AmbientFlux = 1.0f;
            world.Solver.Parallel = parallel;
            for (int i = 0; i < 120; i++)
            {
                world.Step(Dt);
            }
            return (
                BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
                BitConverter.SingleToUInt32Bits(world.Bodies[id].Temperature));
        }

        Assert.Equal(Run(false), Run(true));
    }

    /// <summary>Y18: a static body acts as an infinite-capacity reservoir.</summary>
    [Fact]
    public void Y18_AStaticBodyDoesNotHeatUp()
    {
        var world = new World2
        {
            Gravity = Vector2.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };
        world.Nuclear.Config = Config();

        int id = world.AddStaticBody(Vector2.Zero, new CircleShape(0.5f));
        world.Bodies[id].Nuclear.Isotope = Short;
        world.Bodies[id].Nuclear.Activity = 10.0f;

        float before = world.Bodies[id].Temperature;
        world.Step(Dt);
        Assert.True(
            BitConverter.SingleToUInt32Bits(before)
            == BitConverter.SingleToUInt32Bits(world.Bodies[id].Temperature),
            "a static body must not change temperature");
    }

    /// <summary>Y19: the nuclide table must be usable through the configuration.</summary>
    [Fact]
    public void Y19_TheNuclideTableIsUsable()
    {
        var cfg = Config();
        Assert.Equal(0.0f, cfg.GetIsotope(NoNuclide)!.HalfLife);
        Assert.Equal(10.0f, cfg.GetIsotope(Short)!.HalfLife);
        Assert.Null(cfg.GetIsotope(99));
        Assert.True(cfg.IsActive(), "the configuration must be active");
    }

    /// <summary>Y20: the decay constant is <c>ln(2) / half_life</c>.</summary>
    [Fact]
    public void Y20_TheDecayConstantIsLn2OverTheHalfLife()
    {
        float expected = 0.6931472f / 10.0f;
        Assert.True(
            BitConverter.SingleToUInt32Bits(expected)
            == BitConverter.SingleToUInt32Bits(Isotope.DecayConstantOf(10.0f)),
            "the decay constant must be ln(2) / half-life");
        Assert.Equal(0.0f, Isotope.DecayConstantOf(0.0f));
        Assert.Equal(0.0f, Isotope.DecayConstantOf(-1.0f));
    }

    /// <summary>Y21: the ionizing radiation is deposited in the nearby bodies.</summary>
    [Fact]
    public void Y21_TheDoseIsDepositedInTheNearbyBodies()
    {
        var (world, source) = WorldWith(Short, 10.0f);
        int other = world.AddDynamicBody(new Vector2(0.5f, 0.0f), new CircleShape(0.2f), 1.0f);
        world.Nuclear.Config.DoseRange = 2.0f;

        world.Step(Dt);
        float dose = world.Bodies[other].Nuclear.Dose;
        Assert.True(dose > 0.0f, $"the nearby body must receive a dose, got {dose}");
        Assert.True(
            BitConverter.SingleToUInt32Bits(0.0f)
            == BitConverter.SingleToUInt32Bits(world.Bodies[source].Nuclear.Dose),
            "a body must not be irradiated by its own radiation");
    }

    /// <summary>Y22: the accumulated dose is clamped.</summary>
    [Fact]
    public void Y22_TheDoseIsClamped()
    {
        var (world, _) = WorldWith(Short, 1e30f);
        world.AddDynamicBody(new Vector2(0.5f, 0.0f), new CircleShape(0.2f), 1.0f);
        world.Nuclear.Config.DoseRange = 2.0f;
        world.Nuclear.Config.MaxDose = 1.0f;

        world.Step(Dt);
        float dose = world.Bodies[1].Nuclear.Dose;
        Assert.True(dose <= 1.0f, $"the dose must be clamped, got {dose}");
    }

    /// <summary>Y23: the nuclear state round-trips through a snapshot.</summary>
    [Fact]
    public void Y23_TheNuclearStateRoundTripsThroughASnapshot()
    {
        var (world, id) = WorldWith(Short, 12.5f);
        world.Bodies[id].Nuclear.Fuel = 0.25f;
        world.Bodies[id].Nuclear.Neutrons = 3.0f;
        world.Bodies[id].Nuclear.Dose = 1.5f;
        world.Nuclear.Config.TimeScale = 7.0f;
        world.Nuclear.Config.AmbientFlux = 0.25f;

        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        var restored = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.Equal(world.Bodies[id].Nuclear.Isotope, restored.Bodies[id].Nuclear.Isotope);
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
            BitConverter.SingleToUInt32Bits(restored.Bodies[id].Nuclear.Activity));
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Fuel),
            BitConverter.SingleToUInt32Bits(restored.Bodies[id].Nuclear.Fuel));
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Dose),
            BitConverter.SingleToUInt32Bits(restored.Bodies[id].Nuclear.Dose));
        Assert.Equal(7.0f, restored.Nuclear.Config.TimeScale);
        Assert.Equal(0.25f, restored.Nuclear.Config.AmbientFlux);
        Assert.Equal(
            world.Nuclear.Config.Isotopes.Count,
            restored.Nuclear.Config.Isotopes.Count);

        byte[] binary = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var restoredBinary = Snapshot.World2FromBinary(binary, SnapshotLoadMode.Exact);
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Activity),
            BitConverter.SingleToUInt32Bits(restoredBinary.Bodies[id].Nuclear.Activity));
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[id].Nuclear.Dose),
            BitConverter.SingleToUInt32Bits(restoredBinary.Bodies[id].Nuclear.Dose));
        Assert.Equal(7.0f, restoredBinary.Nuclear.Config.TimeScale);
        Assert.Equal(
            world.Nuclear.Config.Isotopes.Count,
            restoredBinary.Nuclear.Config.Isotopes.Count);
    }

    /// <summary>
    /// Y24: the nuclear config and the per-body state round-trip through
    /// the native FFI.
    /// </summary>
    [Fact]
    public void Y24_NuclearConfigFfiRoundtrip()
    {
        using var w = new PhiNativeWorld();

        var config = new NuclearConfig
        {
            Enabled = true,
            TimeScale = 3.0f,
            FissionEnergy = 2.5e-11f,
            AmbientFlux = 0.75f,
            FissionRate = 2.0f,
            FuelPerFission = 1e-5f,
            NeutronRange = 4.5f,
            DoseRange = 6.5f,
            DelayedNeutronFraction = 0.0125f,
            DelayedNeutronDelay = 0.75f,
            MaxActivity = 1e20f,
            DosePerJoule = 2.5f,
            MaxDose = 1e25f,
        };
        config.Isotopes = new List<Isotope>
        {
            new Isotope(),
            new Isotope
            {
                HalfLife = 3600.0f,
                DecayEnergy = 1.5e-13f,
                IonizationFraction = 0.25f,
                AbsorptionCrossSection = 3.0f,
                FissionCrossSection = 4.0f,
                NeutronYield = 2.5f,
            },
        };
        w.SetNuclearConfig(config);

        var restored = w.GetNuclearConfig();
        Assert.Equal(3.0f, restored.TimeScale);
        Assert.Equal(0.75f, restored.AmbientFlux);
        Assert.Equal(4.5f, restored.NeutronRange);
        Assert.Equal(6.5f, restored.DoseRange);
        Assert.Equal(0.0125f, restored.DelayedNeutronFraction);
        Assert.True(restored.Enabled);

        int body = w.AddCircleBody(Vector2.Zero, 0.5f, 1.0f, false);
        Assert.True(body >= 0);
        w.SetBodyNuclearState((uint)body, new NuclearState
        {
            Isotope = 1,
            Activity = 12.5f,
            Fuel = 0.75f,
            Neutrons = 3.0f,
            DelayedNeutrons = 2.0f,
            Dose = 0.125f,
        });

        var state = w.GetBodyNuclearState((uint)body);
        Assert.Equal((ushort)1, state.Isotope);
        Assert.Equal(12.5f, state.Activity);
        Assert.Equal(0.75f, state.Fuel);
        Assert.Equal(3.0f, state.Neutrons);
        Assert.Equal(2.0f, state.DelayedNeutrons);
        Assert.Equal(0.125f, state.Dose);
    }

    /// <summary>
    /// Y25: an extreme activity must not poison the body's temperature.
    /// </summary>
    [Fact]
    public void Y25_ExtremeActivityDoesNotPoisonTheTemperature()
    {
        var (world, id) = WorldWith(Short, 1e30f);
        // A decay energy that overflows when multiplied by the activity.
        world.Nuclear.Config.Isotopes[Short].DecayEnergy = 1e30f;

        world.Step(Dt);
        float temperature = world.Bodies[id].Temperature;
        Assert.True(float.IsFinite(temperature), $"the temperature must stay finite, got {temperature}");
        Assert.True(
            temperature >= world.Thermal.MinTemperature,
            $"the temperature must not fall below the clamp, got {temperature}");
    }

    /// <summary>
    /// Y26: the fissions cannot exceed the fuel actually present.
    /// </summary>
    [Fact]
    public void Y26_FissionCannotExceedTheAvailableFuel()
    {
        var (world, id) = WorldWith(Fissile, 0.0f);
        world.Bodies[id].Nuclear.Fuel = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1.0f;
        world.Nuclear.Config.FissionRate = 1.0f;
        // One fission consumes the whole fuel.
        world.Nuclear.Config.FuelPerFission = 1.0f;
        world.Nuclear.Config.FissionEnergy = 1.0e6f;

        // One unit of fuel buys exactly one fission, so the whole run can
        // release at most 1e6 J - no matter how many steps it takes.
        float totalHeat = 0.0f;
        for (int i = 0; i < 240; i++)
        {
            world.Step(Dt);
            totalHeat += world.Nuclear.Scratch.HeatAt(id);
        }
        float fuel = world.Bodies[id].Nuclear.Fuel;
        Assert.True(fuel >= 0.0f, $"the fuel must not go negative, got {fuel}");
        Assert.True(totalHeat <= 1.0e6f * 1.001f, $"the heat must not exceed the fuel, got {totalHeat}");
        Assert.True(totalHeat > 0.0f, $"the fission must have released heat, got {totalHeat}");
    }

    /// <summary>
    /// Y27: a zero-length step must not re-apply the previous step's heat.
    /// </summary>
    [Fact]
    public void Y27_AZeroLengthStepDoesNotReapplyTheHeat()
    {
        var (world, id) = WorldWith(Short, 1e6f);
        world.Step(Dt);
        float afterOne = world.Bodies[id].Temperature;
        Assert.True(afterOne > 293.15f, "the decay must heat the body");

        world.Step(0.0f);
        float afterZero = world.Bodies[id].Temperature;
        Assert.True(
            BitConverter.SingleToUInt32Bits(afterOne)
            == BitConverter.SingleToUInt32Bits(afterZero),
            "a zero-length step must not re-apply the previous heat");
    }

    /// <summary>
    /// Y28: a nuclide with a daughter captures a neutron and becomes it.
    /// </summary>
    [Fact]
    public void Y28_ANuclideWithADaughterCapturesANeutron()
    {
        var (world, id) = WorldWith(Short, 0.0f);
        // The short nuclide absorbs neutrons and decays into the stable one.
        world.Nuclear.Config.Isotopes[Short].AbsorptionCrossSection = 1.0f;
        world.Nuclear.Config.AmbientFlux = 1000.0f;
        world.Nuclear.Config.ActivationRate = 1000.0f;

        ushort startIsotope = world.Bodies[id].Nuclear.Isotope;
        for (int i = 0; i < 10; i++)
        {
            world.Step(Dt);
        }
        Assert.True(
            world.Nuclear.Config.Isotopes[startIsotope].Daughter
            == world.Bodies[id].Nuclear.Isotope,
            "the body must become its daughter after capturing a neutron");
    }
}
