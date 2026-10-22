using System.Numerics;
using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;
using PhiEng.Dynamics;
using PhiEng.Native;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// The game-facing quantum layer (M49–M56). Mirrors
/// <c>crates/engine/tests/quantum_game.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// These tests assert what a game actually consumes: Born-rule sampling
/// that follows the wavefunction, the Bohmian velocity a driven body
/// surfs, the Madelung fluid's dispersion against the analytic
/// frequency, the classical limit that recovers the ordinary fluid,
/// and the density/phase feed a renderer reads.
/// </remarks>
public class QuantumGameTests
{
    /// <summary>The bit pattern of a float, as the pins are written.</summary>
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>
    /// M49: Born-rule sampling follows the wavefunction.
    /// </summary>
    /// <remarks>
    /// A flat six-outcome state must fill its bins evenly, and the
    /// seeded stream pins the exact counts: the same seed gives the
    /// same shots on every platform and in both languages.
    /// </remarks>
    [Fact]
    public void M49_TheSamplingFollowsTheBornRule()
    {
        var config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 8;
        var field = new QuantumField(config);
        Assert.Equal(6, field.State.Dim);
        for (int i = 0; i < field.State.Dim; i++)
        {
            field.State.Amplitudes[i] = Complex32.One;
        }
        field.State.Normalize();

        int[] bins = new int[6];
        var one = new float[1];
        for (int i = 0; i < 10000; i++)
        {
            field.Sample(one);
            bins[(int)one[0]]++;
        }
        Assert.Equal(new[] { 1671, 1643, 1625, 1660, 1661, 1740 }, bins);
        int total = 0;
        foreach (int b in bins)
        {
            total += b;
        }
        Assert.Equal(10000, total);
    }

    /// <summary>
    /// M50: the same seed gives the same shots, a different seed
    /// different ones.
    /// </summary>
    [Fact]
    public void M50_TheSamplingIsReproducible()
    {
        static float[] Draws(ulong seed)
        {
            var config = QuantumConfig.Default;
            config.Enabled = true;
            config.GridSize = 8;
            config.Seed = seed;
            var field = new QuantumField(config);
            var output = new float[64];
            field.Sample(output);
            return output;
        }
        Assert.Equal(Draws(1), Draws(1));
        Assert.NotEqual(Draws(1), Draws(2));
    }

    /// <summary>
    /// M51: the Bohmian velocity vanishes for a stationary state and
    /// follows the phase gradient otherwise.
    /// </summary>
    [Fact]
    public void M51_TheBohmianVelocity()
    {
        // The harmonic ground state is real: it has no current anywhere.
        var config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 128;
        var field = new QuantumField(config);
        float worst = 0.0f;
        for (int i = 0; i < 128; i++)
        {
            float x = -10.0f + i * (20.0f / 127.0f);
            worst = MathF.Max(worst, MathF.Abs(field.VelocityAt(x)));
        }
        Assert.True(worst < 1e-9f, $"a stationary state must not flow (max|v| = {worst:E})");

        // A constant-amplitude state with a linear phase on a periodic
        // grid flows at exactly the discrete phase gradient: sin(Δ)/dx.
        var flowConfig = QuantumConfig.Default;
        flowConfig.Enabled = true;
        flowConfig.GridSize = 64;
        flowConfig.DomainMin = -5.0f;
        flowConfig.DomainMax = 5.0f;
        flowConfig.Boundary = Boundary1D.Periodic;
        var flow = new QuantumField(flowConfig);
        int n = flow.State.Dim;
        Assert.Equal(64, n);
        for (int j = 0; j < n; j++)
        {
            double phase = 2.0 * System.Math.PI * 2.0 * j / n;
            float s = 1.0f / MathF.Sqrt(n);
            flow.State.Amplitudes[j] = new Complex32(
                s * (float)System.Math.Cos(phase), s * (float)System.Math.Sin(phase));
        }
        // The discrete-exact flow: sin(Δ)/dx with Δ the phase step.
        float dx = 10.0f / 63.0f;
        float delta = MathF.PI / 16.0f;
        float want = MathF.Sin(delta) / dx;
        float worstDv = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float x = -5.0f + j * dx;
            worstDv = MathF.Max(worstDv, MathF.Abs(flow.VelocityAt(x) - want));
        }
        Assert.True(worstDv < 1e-5f, $"the flow must be the discrete phase gradient (worst {worstDv:E})");
    }

    /// <summary>
    /// M52: a driven body lands on the grid with the flow velocity, and
    /// the drive is deterministic.
    /// </summary>
    [Fact]
    public void M52_ADrivenBodySurfsTheWavefunction()
    {
        static (float X, float V) Drive(ulong seed)
        {
            var world = new World2();
            world.Gravity = new Vector2(0.0f, 0.0f);
            int id = world.AddDynamicBody(new Vector2(0.0f, 3.0f), new CircleShape(0.5f), 1.0f);
            var config = QuantumConfig.Default;
            config.Enabled = true;
            config.GridSize = 32;
            config.Seed = seed;
            world.Quantum.Configure(config);
            Assert.True(world.DriveBodyByQuantum(id), "the drive must accept a live body");
            return (world.Bodies[id].Transform.Position.X, world.Bodies[id].Velocity.X);
        }
        var first = Drive(7);
        var second = Drive(7);
        Assert.Equal((Bits(first.X), Bits(first.V)), (Bits(second.X), Bits(second.V)));
        Assert.True(
            first.X >= -10.0f && first.X <= 10.0f,
            $"the body must land on the grid (x = {first.X})");
        Assert.True(float.IsFinite(first.V), "the driven velocity must be finite");

        // A field that is off, or a body that is not there, refuses.
        var world = new World2();
        int id = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new CircleShape(0.5f), 1.0f);
        Assert.False(world.DriveBodyByQuantum(id), "a disabled field must refuse the drive");
        var config = QuantumConfig.Default;
        config.Enabled = true;
        world.Quantum.Configure(config);
        Assert.False(world.DriveBodyByQuantum(9999), "an unknown body must be refused");
    }

    /// <summary>
    /// M53: the Madelung fluid oscillates at the analytic frequency.
    /// </summary>
    /// <remarks>
    /// A small sinusoidal density perturbation of a uniform fluid is a
    /// standing wave at ω = ℏk²/2m. The test finds the first zero
    /// down-crossing of the density at the origin and reads the
    /// frequency off it.
    /// </remarks>
    [Fact]
    public void M53_TheMadelungFluidDisperses()
    {
        float l = 2.0f * MathF.PI;
        int n = 64;
        float dx = l / n;
        var fluid = new QuantumFluid1D(n, dx, 1.0f, 1.0f, 1.0f);
        var rho0 = new float[n];
        for (int i = 0; i < n; i++)
        {
            rho0[i] = 1.0f + 0.01f * MathF.Cos(i * dx);
        }
        fluid.SetDensity(rho0);
        float m0 = fluid.MassTotal();

        float dt = 0.0005f;
        var scratch = new MadelungScratch(n);
        float t = 0.0f;
        float crossing = 0.0f;
        for (int step = 0; step < 20000; step++)
        {
            fluid.Step(dt, scratch);
            t += dt;
            // The first crossing below the mean after the start.
            if (t > 0.5f && fluid.Rho[0] < 1.0f)
            {
                crossing = t;
                break;
            }
        }
        Assert.True(crossing > 0.0f, "the wave must cross the mean");
        // The crossing is a quarter period: ω = π / (2 t).
        float omega = MathF.PI / (2.0f * crossing);
        float relative = MathF.Abs(omega - 0.5f) / 0.5f;
        Assert.True(
            relative < 0.05f,
            $"the measured frequency {omega} must be within 5% of 0.5");

        // The mass survives the run.
        float relativeMass = MathF.Abs(fluid.MassTotal() - m0) / m0;
        Assert.True(
            relativeMass < 1e-5f,
            $"the mass must be conserved (drift {relativeMass:E})");
    }

    /// <summary>
    /// M54: the classical limit recovers the ordinary fluid.
    /// </summary>
    /// <remarks>
    /// With ℏ = 0 the quantum potential is exactly zero and a still
    /// fluid stays still: nothing moves, nothing appears.
    /// </remarks>
    [Fact]
    public void M54_TheClassicalLimitRecoversTheOrdinaryFluid()
    {
        float l = 2.0f * MathF.PI;
        int n = 64;
        float dx = l / n;
        var fluid = new QuantumFluid1D(n, dx, 0.0f, 1.0f, 1.0f);
        var rho0 = new float[n];
        for (int i = 0; i < n; i++)
        {
            rho0[i] = 1.0f + 0.01f * MathF.Cos(i * dx);
        }
        fluid.SetDensity(rho0);

        // The quantum potential of the same bump is exactly zero.
        var q = new float[n];
        Assert.True(Madelung.QuantumPotential(fluid.Rho, dx, 0.0f, 1.0f, q));
        foreach (float v in q)
        {
            Assert.True(MathF.Abs(v) < 1e-12f, "Q must vanish with hbar");
        }

        var scratch = new MadelungScratch(n);
        for (int step = 0; step < 2000; step++)
        {
            Assert.True(fluid.Step(0.0005f, scratch));
        }
        for (int i = 0; i < n; i++)
        {
            Assert.True(
                MathF.Abs(fluid.Rho[i] - rho0[i]) < 1e-9f,
                "a still classical fluid must stay still");
            Assert.True(
                MathF.Abs(fluid.Vel[i]) < 1e-12f,
                "no velocity may appear from nothing");
        }
    }

    /// <summary>
    /// M55: the density sums to one and the ground state's phase is flat.
    /// </summary>
    [Fact]
    public void M55_TheVisualisationFeed()
    {
        var config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 128;
        var field = new QuantumField(config);
        int n = field.State.Dim;
        var density = new float[n];
        var phase = new float[n];
        Assert.Equal(n, field.Density(density));
        Assert.Equal(n, field.Phase(phase));

        float total = 0.0f;
        foreach (float d in density)
        {
            total += d;
        }
        Assert.True(MathF.Abs(total - 1.0f) < 1e-5f, $"the density must sum to one ({total})");
        foreach (float d in density)
        {
            Assert.True(d >= 0.0f, "a density must not be negative");
        }

        // The ground state is real up to a global sign: its phase is one
        // value everywhere.
        float lo = float.PositiveInfinity;
        float hi = float.NegativeInfinity;
        foreach (float p in phase)
        {
            lo = MathF.Min(lo, p);
            hi = MathF.Max(hi, p);
        }
        Assert.True(hi - lo < 1e-3f, $"the ground state's phase must be flat (range {hi - lo})");
    }

    /// <summary>
    /// M56: the game-facing surface through the native bridge.
    /// </summary>
    [Fact]
    public void M56_TheNativeBridgeRoundTrip()
    {
        using var world = new PhiNativeWorld();

        // A field that is off refuses the drive and reports no force.
        Assert.False(world.QuantumDriveBody(0), "a disabled field must refuse the drive");
        Assert.Equal(0.0f, world.QuantumForceAt(0.0f));

        // Switch the field on through the native config.
        var cfg = new NativeMethods.QuantumConfigFfi
        {
            Enabled = 1,
            Method = 0,
            Boundary = 0,
            GridSize = 32,
            DomainMin = -5.0f,
            DomainMax = 5.0f,
            Mass = 1.0f,
            Hbar = 1.0f,
            Omega = 2.0f,
            Seed = 12345,
        };
        // A body to drive.
        int id = world.AddCircleBody(new Vector2(0.0f, 0.0f), 0.5f, 1.0f, false);
        Assert.True(id >= 0, "the body must be added");
        world.SetQuantumConfig(ToManaged(cfg));
        Assert.True(world.QuantumDriveBody((uint)id), "the drive must accept a live body");
        float force = world.QuantumForceAt(1.0f);
        Assert.True(float.IsFinite(force), "the force must be finite");

        // The density: the count, then the values.
        float[] density = world.QuantumDensity();
        Assert.Equal(30, density.Length);
        float total = 0.0f;
        foreach (float d in density)
        {
            total += d;
        }
        Assert.True(MathF.Abs(total - 1.0f) < 1e-4f, $"the density must sum to one ({total})");

        // The phase: the same convention.
        float[] phase = world.QuantumPhase();
        Assert.Equal(30, phase.Length);
    }

    private static QuantumConfig ToManaged(NativeMethods.QuantumConfigFfi ffi) => new()
    {
        Enabled = ffi.Enabled != 0,
        Method = (EvolutionMethod)ffi.Method,
        Boundary = (Boundary1D)ffi.Boundary,
        GridSize = (int)ffi.GridSize,
        DomainMin = ffi.DomainMin,
        DomainMax = ffi.DomainMax,
        Mass = ffi.Mass,
        Hbar = ffi.Hbar,
        Omega = ffi.Omega,
        Seed = ffi.Seed,
    };
}
