using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Radiation model expansion (U01-U12). Mirrors
/// <c>crates/engine/tests/radiation.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// The radiation model gained three opt-in options, all of which default
/// to the historical behaviour so existing trajectories stay
/// bit-identical: a symmetric (order-independent) view-factor model, a
/// sky/ground split for the environment radiation, and a partially
/// transparent occluder - plus a fix to the occlusion test, which used
/// to let radiation leak through gaps next to the two exchanging bodies.
/// </remarks>
public class RadiationTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>
    /// The ambient temperature used by the scenes. It is deliberately
    /// equal to the hot body's temperature, so the hot body's
    /// <i>environment</i> exchange is zero and the only heat it moves is
    /// the pair exchange under test.
    /// </summary>
    private const float Ambient = 500.0f;

    private const float Hot = 500.0f;
    private const float Cold = 300.0f;
    private const float RHot = 0.5f;
    private const float RCold = 0.5f;

    /// <summary>
    /// A hot body at <paramref name="xHot"/> and a cold body at
    /// <paramref name="xCold"/>, exchanging radiation with nothing else.
    /// </summary>
    private static (World2 World, int Hot, int Cold) Scene(float xHot, float xCold, float rCold)
    {
        var world = new World2
        {
            Gravity = Vector2.Zero,
            LinearSleepThreshold = 0.0f,
            AngularSleepThreshold = 0.0f,
        };

        var cfg = world.Thermal;
        cfg.Enabled = true;
        cfg.RadiationEnabled = true;
        cfg.ConductionEnabled = false;
        cfg.ConvectionEnabled = false;
        cfg.AmbientTemperature = Ambient;
        // A large rate cap so the exchange is not clamped away.
        cfg.MaxTemperatureRate = 1e9f;

        int hot = world.AddDynamicBody(new Vector2(xHot, 0.0f), new CircleShape(RHot), 1.0f);
        int cold = world.AddDynamicBody(new Vector2(xCold, 0.0f), new CircleShape(rCold), 1.0f);
        world.Bodies[hot].Temperature = Hot;
        world.Bodies[cold].Temperature = Cold;
        return (world, hot, cold);
    }

    /// <summary>
    /// The heat the pair exchange moved into the hot body in one step.
    /// </summary>
    private static float PairExchange(float rCold, Action<World2>? setup = null)
    {
        var (world, hot, _) = Scene(-1.0f, 1.0f, rCold);
        setup?.Invoke(world);
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        return world.Bodies[hot].Temperature - before;
    }

    /// <summary>
    /// An occluding body between the pair, at the hot body's temperature
    /// so that it exchanges no heat with it.
    /// </summary>
    private static void AddOccluder(World2 world, Vector2 at)
    {
        int id = world.AddDynamicBody(at, new CircleShape(0.4f), 1.0f);
        world.Bodies[id].Temperature = Hot;
    }

    /// <summary>The heat capacity of a circle body of the given radius.</summary>
    private static float HeatCapacity(float radius)
    {
        float mass = MathF.PI * radius * radius; // density 1.0
        return mass * 900.0f; // the default specific heat
    }

    /// <summary>
    /// U01: the legacy view-factor model is the default, and the pair
    /// exchange matches the historical formula.
    /// </summary>
    [Fact]
    public void U01_LegacyViewFactorModelIsTheDefault()
    {
        var (world, _, _) = Scene(-1.0f, 1.0f, RCold);
        Assert.Equal(ViewFactorModel.Legacy, world.Thermal.ViewFactorModel);

        // The historical formula, computed by hand:
        //   eps_eff = 1 / (1/eps_i + 1/eps_j - 1)
        //   F       = (r_j / d)^2
        //   Q       = eps_eff * sigma * A * F * (T_i^4 - T_j^4)
        // and the temperature change is Q / (m * c_p) * dt.
        float sigma = world.Thermal.StefanBoltzmann;
        float eps = System.Math.Clamp(world.Bodies[0].Material.Emissivity, 0.0f, 1.0f);
        float epsEff = 1.0f / (1.0f / eps + 1.0f / eps - 1.0f);
        float d = 2.0f;
        float viewFactor = (RCold / d) * (RCold / d);
        // The 2D pair term uses the characteristic length (the radius).
        float area = RCold;
        float q = epsEff * sigma * area * viewFactor
            * (MathF.Pow(Hot, 4.0f) - MathF.Pow(Cold, 4.0f));
        // Mirror the engine's arithmetic exactly: the hot body *loses*
        // the heat and the sum is rounded to float.
        float expected = (Hot + (-q / HeatCapacity(RHot)) * Dt) - Hot;

        float measured = PairExchange(RCold);
        Assert.True(
            MathF.Abs(measured - expected) <= MathF.Abs(expected) * 1e-3f + 1e-9f,
            $"the legacy pair exchange should be {expected}, measured {measured}");
    }

    /// <summary>
    /// U02: the symmetric model agrees with the legacy model for two
    /// identical bodies.
    /// </summary>
    [Fact]
    public void U02_SymmetricModelMatchesLegacyForEqualBodies()
    {
        float legacy = PairExchange(RCold);
        float symmetric = PairExchange(RCold, w =>
            w.Thermal.ViewFactorModel = ViewFactorModel.Symmetric);

        Assert.True(
            MathF.Abs(symmetric - legacy) <= MathF.Abs(legacy) * 1e-3f + 1e-9f,
            $"the symmetric model should match the legacy one for equal bodies: "
            + $"legacy {legacy}, symmetric {symmetric}");
    }

    /// <summary>
    /// U03: the symmetric model does not depend on which body of the
    /// pair the loop visits first.
    /// </summary>
    [Fact]
    public void U03_SymmetricModelIsOrderIndependent()
    {
        float a = PairExchange(RCold, w =>
            w.Thermal.ViewFactorModel = ViewFactorModel.Symmetric);

        var (world, hot, _) = Scene(1.0f, -1.0f, RCold);
        world.Thermal.ViewFactorModel = ViewFactorModel.Symmetric;
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        float b = world.Bodies[hot].Temperature - before;

        Assert.True(
            MathF.Abs(a - b) <= MathF.Abs(a) * 1e-3f + 1e-9f,
            $"the symmetric exchange must not depend on the body order: {a} vs {b}");
    }

    /// <summary>
    /// U04: the symmetric model is a <i>different</i> model for unequal
    /// bodies.
    /// </summary>
    [Fact]
    public void U04_SymmetricModelDiffersFromLegacyForUnequalBodies()
    {
        float legacy = PairExchange(0.25f);
        float symmetric = PairExchange(0.25f, w =>
            w.Thermal.ViewFactorModel = ViewFactorModel.Symmetric);

        Assert.True(
            MathF.Abs(symmetric - legacy) > MathF.Abs(legacy) * 1e-3f,
            $"the symmetric model must actually change the exchange for unequal "
            + $"bodies: legacy {legacy}, symmetric {symmetric}");
    }

    /// <summary>
    /// U05: a sky/ground split changes the environment exchange.
    /// </summary>
    [Fact]
    public void U05_SkyGroundSplitChangesTheEquilibrium()
    {
        var (splitWorld, splitBody, _) = Scene(-1.0f, 1.0f, RCold);
        splitWorld.Thermal.SkyTemperature = 100.0f;
        splitWorld.Thermal.GroundTemperature = 400.0f;
        float before = splitWorld.Bodies[splitBody].Temperature;
        splitWorld.Step(Dt);
        float split = splitWorld.Bodies[splitBody].Temperature - before;

        var (uniformWorld, uniformBody, _) = Scene(-1.0f, 1.0f, RCold);
        float beforeUniform = uniformWorld.Bodies[uniformBody].Temperature;
        uniformWorld.Step(Dt);
        float uniform = uniformWorld.Bodies[uniformBody].Temperature - beforeUniform;

        Assert.True(
            MathF.Abs(split - uniform) > MathF.Abs(uniform) * 1e-3f,
            $"the sky/ground split must change the exchange: "
            + $"split {split}, uniform {uniform}");
    }

    /// <summary>
    /// U06: a sky and ground equal to the ambient are bit-identical to
    /// the historical single-temperature exchange.
    /// </summary>
    [Fact]
    public void U06_EqualSkyAndGroundMatchTheHistoricalExchange()
    {
        var (baselineWorld, baselineBody, _) = Scene(-1.0f, 1.0f, RCold);
        float before = baselineWorld.Bodies[baselineBody].Temperature;
        baselineWorld.Step(Dt);
        float baseline = baselineWorld.Bodies[baselineBody].Temperature - before;

        var (explicitWorld, explicitBody, _) = Scene(-1.0f, 1.0f, RCold);
        explicitWorld.Thermal.SkyTemperature = Ambient;
        explicitWorld.Thermal.GroundTemperature = Ambient;
        float beforeExplicit = explicitWorld.Bodies[explicitBody].Temperature;
        explicitWorld.Step(Dt);
        float explicitValue = explicitWorld.Bodies[explicitBody].Temperature - beforeExplicit;

        Assert.True(
            BitConverter.SingleToUInt32Bits(baseline)
            == BitConverter.SingleToUInt32Bits(explicitValue),
            "an explicit sky/ground equal to the ambient must be bit-identical "
            + "to the historical exchange");
    }

    /// <summary>
    /// U07: an opaque body between the pair blocks the exchange.
    /// </summary>
    [Fact]
    public void U07_OpaqueOccluderBlocksTheExchange()
    {
        float unoccluded = PairExchange(RCold);

        var (world, hot, _) = Scene(-1.0f, 1.0f, RCold);
        world.Thermal.RadiationOcclusion = true;
        AddOccluder(world, new Vector2(0.0f, 0.0f));
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        float occluded = world.Bodies[hot].Temperature - before;

        Assert.True(
            MathF.Abs(occluded) < MathF.Abs(unoccluded) * 0.5f,
            $"an opaque occluder must block most of the exchange: "
            + $"unoccluded {unoccluded}, occluded {occluded}");
    }

    /// <summary>
    /// U08: an occluder <i>next to</i> one of the exchanging bodies
    /// blocks the exchange too.
    /// </summary>
    /// <remarks>
    /// This is the regression test for the removed midpoint heuristic:
    /// the old code exempted blockers that sat close to an endpoint, so
    /// radiation leaked through the gap.
    /// </remarks>
    [Fact]
    public void U08_OffAxisOccluderAlsoBlocks()
    {
        float unoccluded = PairExchange(RCold);

        // The blocker sits just next to the hot body, off the exact line.
        var (world, hot, _) = Scene(-1.0f, 1.0f, RCold);
        world.Thermal.RadiationOcclusion = true;
        AddOccluder(world, new Vector2(-0.35f, 0.35f));
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        float occluded = world.Bodies[hot].Temperature - before;

        Assert.True(
            MathF.Abs(occluded) < MathF.Abs(unoccluded) * 0.5f,
            $"an off-axis occluder next to an endpoint must block the exchange: "
            + $"unoccluded {unoccluded}, occluded {occluded}");
    }

    /// <summary>
    /// U09: a partially transparent occluder attenuates the exchange
    /// instead of blocking it completely.
    /// </summary>
    [Fact]
    public void U09_PartialOccluderAttenuatesTheExchange()
    {
        float unoccluded = PairExchange(RCold);

        var (world, hot, _) = Scene(-1.0f, 1.0f, RCold);
        world.Thermal.RadiationOcclusion = true;
        world.Thermal.OccluderTransmittance = 0.5f;
        AddOccluder(world, new Vector2(0.0f, 0.0f));
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        float attenuated = world.Bodies[hot].Temperature - before;

        Assert.True(
            MathF.Abs(attenuated) < MathF.Abs(unoccluded) * 0.9f
            && MathF.Abs(attenuated) > MathF.Abs(unoccluded) * 0.1f,
            $"a half-transparent occluder must attenuate, not block: "
            + $"unoccluded {unoccluded}, attenuated {attenuated}");
    }

    /// <summary>
    /// U10: a fully transparent occluder does not block at all.
    /// </summary>
    [Fact]
    public void U10_FullyTransparentOccluderDoesNotBlock()
    {
        float unoccluded = PairExchange(RCold);

        var (world, hot, _) = Scene(-1.0f, 1.0f, RCold);
        world.Thermal.RadiationOcclusion = true;
        world.Thermal.OccluderTransmittance = 1.0f;
        AddOccluder(world, new Vector2(0.0f, 0.0f));
        float before = world.Bodies[hot].Temperature;
        world.Step(Dt);
        float transmitted = world.Bodies[hot].Temperature - before;

        Assert.True(
            MathF.Abs(transmitted - unoccluded) <= MathF.Abs(unoccluded) * 1e-3f + 1e-9f,
            $"a fully transparent occluder must not block: "
            + $"unoccluded {unoccluded}, transmitted {transmitted}");
    }

    /// <summary>
    /// U11: the radiation model is deterministic, and the parallel solver
    /// lands on the same trajectory.
    /// </summary>
    [Fact]
    public void U11_RadiationIsDeterministic()
    {
        float Run(bool parallel)
        {
            var (world, _, _) = Scene(-1.0f, 1.0f, RCold);
            world.Thermal.ViewFactorModel = ViewFactorModel.Symmetric;
            world.Thermal.SkyTemperature = 150.0f;
            world.Thermal.GroundTemperature = 320.0f;
            world.Solver.Parallel = parallel;
            for (int i = 0; i < 60; i++)
            {
                world.Step(Dt);
            }
            return world.Bodies[0].Temperature;
        }

        float a = Run(false);
        float b = Run(false);
        Assert.True(
            BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(b),
            "two identical runs must agree");

        float p = Run(true);
        Assert.True(
            BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(p),
            "the parallel solver must land on the serial trajectory");
    }

    /// <summary>
    /// U12: the radiation options survive a snapshot round-trip (JSON and
    /// binary).
    /// </summary>
    [Fact]
    public void U12_RadiationOptionsRoundTripThroughASnapshot()
    {
        var (world, _, _) = Scene(-1.0f, 1.0f, RCold);
        world.Thermal.ViewFactorModel = ViewFactorModel.Symmetric;
        world.Thermal.SkyTemperature = 123.5f;
        world.Thermal.GroundTemperature = 287.25f;
        world.Thermal.OccluderTransmittance = 0.375f;

        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        var restored = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.Equal(ViewFactorModel.Symmetric, restored.Thermal.ViewFactorModel);
        Assert.Equal(123.5f, restored.Thermal.SkyTemperature);
        Assert.Equal(287.25f, restored.Thermal.GroundTemperature);
        Assert.Equal(0.375f, restored.Thermal.OccluderTransmittance);

        byte[] binary = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var restoredBinary = Snapshot.World2FromBinary(binary, SnapshotLoadMode.Exact);
        Assert.Equal(ViewFactorModel.Symmetric, restoredBinary.Thermal.ViewFactorModel);
        Assert.Equal(123.5f, restoredBinary.Thermal.SkyTemperature);
        Assert.Equal(287.25f, restoredBinary.Thermal.GroundTemperature);
        Assert.Equal(0.375f, restoredBinary.Thermal.OccluderTransmittance);
    }
}
