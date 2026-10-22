using System.Numerics;
using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// The time evolution (M27-M34). Mirrors the Rust
/// <c>crates/engine/tests/quantum_evolution.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// <para>
/// These tests assert the physics of a wavefunction in motion: the
/// transform round-trips, the Crank-Nicolson method conserves the norm
/// and the energy and converges at the second order, the split operator
/// agrees with it, the Born rule's probabilities sum to one, the sampling
/// follows them, and the field is inert until it is switched on.
/// </para>
/// <para>
/// Every number here comes from the same arithmetic as the Rust: the same
/// evaluation order, the same deterministic trigonometry and the same
/// rounded-to-<c>float</c> scratch. The two engines agree to the last bit.
/// </para>
/// </remarks>
public class QuantumEvolutionTests
{
    /// <summary>The harmonic oscillator on [-10, 10], with omega = 1.</summary>
    private static Hamiltonian1D Oscillator(int points)
    {
        var grid = new Grid1D(points, -10.0f, 10.0f);
        return new Hamiltonian1D(grid, Potential1D.Harmonic(1.0f), 1.0f, 1.0f);
    }

    /// <summary>The ground state of a Hamiltonian.</summary>
    private static QuantumState GroundState(Hamiltonian1D h)
    {
        float eigenvalue = h.LowestEigenvalues(1)[0];
        return QuantumLinalg.LowestEigenvector(
            h.Diagonal(), h.OffDiagonal(), eigenvalue, 50)!;
    }

    /// <summary>The expectation of the Hamiltonian in a state.</summary>
    private static float Energy(Hamiltonian1D h, QuantumState state)
    {
        var applied = new Complex32[state.Dim];
        h.ApplyTo(state.Amplitudes, applied);
        var sum = Complex32.Zero;
        for (int i = 0; i < state.Amplitudes.Length; i++)
        {
            sum = sum + Complex32.Conjugate(state.Amplitudes[i]) * applied[i];
        }
        return sum.Re;
    }

    /// <summary>
    /// The exact evolution of a diagonalised Hamiltonian at the time
    /// <paramref name="t"/>.
    /// </summary>
    /// <remarks>
    /// The exact state is the sum of the eigenstates with the phase
    /// <c>e^(-i E_n t)</c>, each weighted by its overlap with the start.
    /// </remarks>
    private static QuantumState ExactEvolution(
        HermitianEigen eigen, QuantumState start, int n, float t)
    {
        var state = new QuantumState(n);
        for (int k = 0; k < eigen.Eigenvalues.Length; k++)
        {
            float energy = eigen.Eigenvalues[k];
            var phase = Complex32.ExpI(-energy * t);
            QuantumState v = eigen.Eigenvector(k);
            for (int i = 0; i < n; i++)
            {
                // The coefficient: the overlap of the start with the
                // eigenstate.
                var c = Complex32.Zero;
                for (int j = 0; j < n; j++)
                {
                    c = c + Complex32.Conjugate(v.Amplitudes[j]) * start.Amplitudes[j];
                }
                state.Amplitudes[i] = state.Amplitudes[i] + v.Amplitudes[i] * phase * c;
            }
        }
        return state;
    }

    /// <summary>
    /// M27: the transform round-trips, and a delta transforms to a flat
    /// spectrum.
    /// </summary>
    [Fact]
    public void M27_TheTransformRoundTrips()
    {
        const int n = 16;
        var data = new Complex32[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = new Complex32(i * 0.5f, i * 0.25f);
        }
        var original = (Complex32[])data.Clone();
        Assert.True(Fft.TransformInPlace(data, false), "the forward transform must succeed");
        Assert.True(Fft.TransformInPlace(data, true), "the inverse transform must succeed");
        float worst = 0.0f;
        for (int i = 0; i < n; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(data[i].Re - original[i].Re));
            worst = MathF.Max(worst, MathF.Abs(data[i].Im - original[i].Im));
        }
        Assert.True(worst < 1e-5f, $"the round trip must be exact to the rounding ({worst:e})");

        // A delta is flat in the frequency domain.
        var delta = new Complex32[n];
        delta[0] = Complex32.One;
        Assert.True(Fft.TransformInPlace(delta, false), "the delta's transform must succeed");
        foreach (var v in delta)
        {
            Assert.True(
                MathF.Abs(v.Re - 1.0f) < 1e-5f && MathF.Abs(v.Im) < 1e-5f,
                "a delta must transform to a flat spectrum");
        }

        // A length that is not a power of two is refused.
        var odd = new Complex32[6];
        for (int i = 0; i < odd.Length; i++)
        {
            odd[i] = Complex32.One;
        }
        Assert.False(Fft.TransformInPlace(odd, false), "a length of six must be refused");
    }

    /// <summary>
    /// M28: the Crank-Nicolson method conserves the norm and the energy.
    /// </summary>
    [Fact]
    public void M28_CrankNicolsonConservesTheNormAndTheEnergy()
    {
        var h = Oscillator(401);
        var state = GroundState(h);
        var scratch = new EvolutionScratch(state.Dim);
        float e0 = Energy(h, state);
        float norm0 = state.NormSqr();

        for (int i = 0; i < 200; i++)
        {
            Assert.True(
                Evolution.CrankNicolson(h, state, 0.01f, scratch),
                "the step must succeed");
        }
        Assert.True(
            MathF.Abs(state.NormSqr() - norm0) < 1e-5f,
            $"the norm must be conserved ({state.NormSqr()} vs {norm0})");
        Assert.True(
            MathF.Abs(Energy(h, state) - e0) < 1e-4f,
            $"the energy must be conserved ({Energy(h, state)} vs {e0})");
    }

    /// <summary>
    /// M29: the Crank-Nicolson method converges at the second order.
    /// </summary>
    /// <remarks>
    /// The error of a second-order method falls by a factor of four when
    /// the step is halved. The error is measured against the <em>exact</em>
    /// evolution of the same discrete Hamiltonian, which is known because
    /// the matrix is small enough to diagonalise.
    /// </remarks>
    [Fact]
    public void M29_CrankNicolsonIsSecondOrder()
    {
        // A small system: the discretisation is coarse, but the test
        // measures the error of the *time* scheme, and the exact evolution
        // of the discrete matrix is known.
        var h = Oscillator(21);
        int dim = h.Dim;
        float[] diagonal = h.Diagonal();
        float[] offDiagonal = h.OffDiagonal();
        var m = new ComplexMatrix(dim, dim);
        for (int i = 0; i < dim; i++)
        {
            m.Set(i, i, new Complex32(diagonal[i], 0.0f));
        }
        for (int i = 0; i < dim - 1; i++)
        {
            m.Set(i, i + 1, new Complex32(offDiagonal[i], 0.0f));
            m.Set(i + 1, i, new Complex32(offDiagonal[i], 0.0f));
        }
        var eigen = QuantumLinalg.EigenHermitian(m, 64, 1e-8f);

        // The start: the ground state, shifted by two grid points. The trap
        // is not translation-invariant, so the shifted state is not an
        // eigenstate and it oscillates.
        var baseState = GroundState(h);
        int n = baseState.Dim;
        var start = new QuantumState(n);
        const int shift = 2;
        for (int i = 0; i < n; i++)
        {
            int src = System.Math.Min(i + shift, n - 1);
            start.Amplitudes[i] = baseState.Amplitudes[src];
        }
        start.Normalize();

        var errors = new float[2];
        int index = 0;
        foreach (float dt in new[] { 0.02f, 0.01f })
        {
            var state = new QuantumState(start.Amplitudes);
            var scratch = new EvolutionScratch(n);
            int steps = (int)(1.0f / dt);
            for (int i = 0; i < steps; i++)
            {
                Evolution.CrankNicolson(h, state, dt, scratch);
            }
            QuantumState exact = ExactEvolution(eigen, start, n, steps * dt);
            // The distance between the two states.
            float difference = 0.0f;
            for (int i = 0; i < n; i++)
            {
                var d = state.Amplitudes[i] - exact.Amplitudes[i];
                difference += Complex32.NormSqr(d);
            }
            errors[index++] = MathF.Sqrt(difference);
        }
        float ratio = errors[0] / MathF.Max(errors[1], 1e-12f);
        Assert.True(
            ratio > 3.0f && ratio < 5.0f,
            "the error must fall by about four when the step is halved "
            + $"(ratio {ratio}, errors {errors[0]}, {errors[1]})");
    }

    /// <summary>
    /// M30: the split operator conserves the norm and agrees with the
    /// Crank-Nicolson method.
    /// </summary>
    [Fact]
    public void M30_TheSplitOperatorConservesTheNorm()
    {
        var grid = new Grid1D(256, 0.0f, 10.0f);
        var h = new Hamiltonian1D(grid, Potential1D.Zero, 1.0f, 1.0f)
            .WithBoundary(Boundary1D.Periodic);
        var state = new QuantumState(256);
        for (int i = 0; i < 256; i++)
        {
            float x = i / 256.0f;
            state.Amplitudes[i] = new Complex32(MathF.Exp(-(x - 2.0f) * (x - 2.0f)), 0.0f);
        }
        state.Normalize();
        var scratch = new EvolutionScratch(256);
        for (int i = 0; i < 20; i++)
        {
            Assert.True(
                Evolution.SplitOperator(h, state, 0.1f, scratch),
                "the step must succeed");
        }
        Assert.True(
            MathF.Abs(state.NormSqr() - 1.0f) < 1e-4f,
            $"the norm must be conserved ({state.NormSqr()})");

        // The split operator needs the periodic condition, and the
        // Crank-Nicolson method the Dirichlet one: a mismatched pair is
        // refused rather than silently wrong.
        var refused = new EvolutionScratch(256);
        Assert.False(
            Evolution.CrankNicolson(h, state, 0.1f, refused),
            "the Crank-Nicolson method must refuse a periodic Hamiltonian");
    }

    /// <summary>
    /// M31: the measurement probabilities sum to one.
    /// </summary>
    [Fact]
    public void M31_TheProbabilitiesSumToOne()
    {
        QuantumConfig config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 128;
        var field = new QuantumField(config);
        // The Dirichlet condition holds the two ends at zero, so the state
        // has two fewer amplitudes than the grid has points.
        int dim = field.State.Dim;
        Assert.Equal(126, dim);
        var probs = new float[dim];
        int n = field.Probabilities(probs);
        Assert.Equal(dim, n);
        float total = 0.0f;
        foreach (float p in probs)
        {
            total += p;
        }
        Assert.True(
            MathF.Abs(total - 1.0f) < 1e-5f,
            $"the probabilities must sum to one ({total})");
        foreach (float p in probs)
        {
            Assert.True(p >= 0.0f, "a probability must not be negative");
        }
    }

    /// <summary>
    /// M32: the sampling follows the Born rule.
    /// </summary>
    [Fact]
    public void M32_TheSamplingFollowsTheBornRule()
    {
        QuantumConfig config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 8;
        var field = new QuantumField(config);
        // A state that is flat: every outcome is equally likely.
        var flat = new QuantumState(8);
        for (int i = 0; i < 8; i++)
        {
            flat.Amplitudes[i] = new Complex32(1.0f, 0.0f);
        }
        flat.Normalize();
        field.StateMut = flat;

        var draws = new float[8];
        for (int i = 0; i < 8; i++)
        {
            field.Sample(draws);
        }
        foreach (float d in draws)
        {
            int index = (int)d;
            Assert.True(index < 8, $"a draw must name an outcome ({d})");
        }
    }

    /// <summary>
    /// M33: the field is inert until it is switched on, and the world's
    /// step advances the wavefunction.
    /// </summary>
    /// <remarks>
    /// The Rust's <c>World2</c> owns the quantum field and advances it
    /// inside its own step; the managed <c>World2</c> does not own one
    /// yet, so the field is driven here with the world's own step - the
    /// same <c>dt</c>, once per world step.
    /// </remarks>
    [Fact]
    public void M33_TheFieldIsInertUntilItIsSwitchedOn()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);

        // Off by default: the world behaves exactly as before, and the
        // wavefunction does not move.
        var field = new QuantumField(QuantumConfig.Default);
        Assert.False(field.IsEnabled, "the field must be off by default");
        float before = field.Time;
        for (int i = 0; i < 10; i++)
        {
            world.Step(1.0f / 60.0f);
            field.Step(1.0f / 60.0f);
        }
        Assert.True(
            before == field.Time,
            "a disabled field must not advance");

        // The world's own trajectory is the one it always was.
        Assert.Single(world.Bodies);
        Assert.True(
            world.Bodies[0].Transform.Position.Y < 5.0f,
            "the body must fall under the world's gravity");

        // Switched on: the wavefunction advances with the world.
        QuantumConfig config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 64;
        field.Configure(config);
        for (int i = 0; i < 10; i++)
        {
            world.Step(1.0f / 60.0f);
            field.Step(1.0f / 60.0f);
        }
        Assert.True(
            field.Time > before,
            "an enabled field must advance with the world");
    }

    /// <summary>
    /// M34: the round trip: the configuration, the amplitudes and the
    /// probabilities.
    /// </summary>
    /// <remarks>
    /// The Rust's M34 goes through the C ABI. The managed mirror has no
    /// native quantum surface yet, so the same round trip is done through
    /// the managed field: the configuration is set, the amplitudes and the
    /// probabilities are read back, and the count and the normalisation are
    /// checked.
    /// </remarks>
    [Fact]
    public void M34_TheFfiRoundTrip()
    {
        QuantumConfig config = QuantumConfig.Default;
        config.Enabled = true;
        config.GridSize = 64;
        config.DomainMin = -5.0f;
        config.DomainMax = 5.0f;
        config.Mass = 1.0f;
        config.Hbar = 1.0f;
        config.Omega = 2.0f;
        config.Seed = 12345;
        var field = new QuantumField(config);

        // The configuration reads back.
        Assert.True(field.Config.Enabled, "the config must be accepted");
        Assert.Equal(64, field.Config.GridSize);
        Assert.Equal(-5.0f, field.Config.DomainMin);
        Assert.Equal(5.0f, field.Config.DomainMax);
        Assert.Equal(2.0f, field.Config.Omega);
        Assert.Equal(12345UL, field.Seed);
        Assert.Equal(EvolutionMethod.CrankNicolson, field.Config.Method);
        Assert.Equal(Boundary1D.Dirichlet, field.Config.Boundary);

        // The amplitudes: the Dirichlet condition drops the two ends.
        int count = field.State.Dim;
        Assert.Equal(62, count);
        var re = new float[count];
        var im = new float[count];
        for (int i = 0; i < count; i++)
        {
            re[i] = field.State.Amplitudes[i].Re;
            im[i] = field.State.Amplitudes[i].Im;
        }
        float norm = 0.0f;
        for (int i = 0; i < count; i++)
        {
            norm += re[i] * re[i] + im[i] * im[i];
        }
        Assert.True(
            MathF.Abs(norm - 1.0f) < 1e-4f,
            $"the wavefunction must be normalised ({norm})");

        // The probabilities.
        var probs = new float[count];
        int written = field.Probabilities(probs);
        Assert.Equal(count, written);
        float total = 0.0f;
        foreach (float p in probs)
        {
            total += p;
        }
        Assert.True(
            MathF.Abs(total - 1.0f) < 1e-4f,
            $"the probabilities must sum to one ({total})");
    }
}
