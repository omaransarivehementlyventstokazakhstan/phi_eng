using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// Entanglement and quantum information (M65–M72). Mirrors
/// <c>crates/engine/tests/quantum_info.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class QuantumInfoTests
{
    /// <summary>The bit pattern of a float, as the pins are written.</summary>
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>
    /// M65: the Bell states are normalized, orthogonal, and stabilized
    /// by XX and ZZ.
    /// </summary>
    [Fact]
    public void M65_TheBellStates()
    {
        var states = new[]
        {
            QuantumEntanglement.BellState(BellPair.PhiPlus),
            QuantumEntanglement.BellState(BellPair.PhiMinus),
            QuantumEntanglement.BellState(BellPair.PsiPlus),
            QuantumEntanglement.BellState(BellPair.PsiMinus),
        };
        foreach (var s in states)
        {
            Assert.True(MathF.Abs(s.NormSqr() - 1.0f) < 1e-6f, "every Bell state must be normalized");
        }
        for (int i = 0; i < states.Length; i++)
        {
            for (int j = 0; j < states.Length; j++)
            {
                float fidelity = states[i].Fidelity(states[j]);
                if (i == j)
                {
                    Assert.True(MathF.Abs(fidelity - 1.0f) < 1e-6f, "a Bell state must match itself");
                }
                else
                {
                    Assert.Equal(0.0f, fidelity);
                }
            }
        }

        // |Φ+> is stabilized by XX and ZZ: both expectations are +1.
        var phi = QuantumEntanglement.BellState(BellPair.PhiPlus);
        var xx = new ComplexMatrix(4, 4);
        (int R, int C)[] xxEntries = { (0, 3), (1, 2), (2, 1), (3, 0) };
        foreach (var (r, c) in xxEntries)
        {
            xx.Set(r, c, Complex32.One);
        }
        var zz = new ComplexMatrix(4, 4);
        (int R, int C, float V)[] zzEntries = { (0, 0, 1.0f), (1, 1, -1.0f), (2, 2, -1.0f), (3, 3, 1.0f) };
        foreach (var (r, c, v) in zzEntries)
        {
            zz.Set(r, c, new Complex32(v, 0.0f));
        }
        var opXx = QuantumOperators.Dense(xx);
        var opZz = QuantumOperators.Dense(zz);
        Assert.True(
            MathF.Abs(phi.Expectation(opXx).Re - 1.0f) < 1e-5f,
            "XX must stabilize |Φ+>");
        Assert.True(
            MathF.Abs(phi.Expectation(opZz).Re - 1.0f) < 1e-5f,
            "ZZ must stabilize |Φ+>");
    }

    /// <summary>
    /// M66: half a Bell pair is maximally mixed.
    /// </summary>
    [Fact]
    public void M66_HalfABellPairIsMaximallyMixed()
    {
        var bell = QuantumEntanglement.BellState(BellPair.PhiPlus);
        var reduced = DensityMatrix.FromPure(bell).PartialTrace(new[] { 0 }, 2);
        Assert.Equal(2, reduced.Dim);
        var m = reduced.Matrix;
        // Each diagonal is one half, up to the rounding of the amplitudes.
        Assert.Equal(0x3effffffu, Bits(m.Get(0, 0).Re));
        Assert.Equal(0x3effffffu, Bits(m.Get(1, 1).Re));
        Assert.Equal(0u, Bits(m.Get(0, 1).Re));
        Assert.Equal(0u, Bits(m.Get(0, 1).Im));
        Assert.True(
            MathF.Abs(reduced.Purity() - 0.5f) < 1e-6f,
            "half a pair must have purity one half");

        // Half a product state stays pure.
        var product = new QuantumState(new[]
        {
            Complex32.One, Complex32.Zero, Complex32.Zero, Complex32.Zero,
        });
        var reducedProduct = DensityMatrix.FromPure(product).PartialTrace(new[] { 0 }, 2);
        Assert.True(
            MathF.Abs(reducedProduct.Purity() - 1.0f) < 1e-6f,
            "half a product state must stay pure");
    }

    /// <summary>
    /// M67: the von Neumann entropy is zero for pure states and one bit
    /// for half a pair.
    /// </summary>
    [Fact]
    public void M67_TheVonNeumannEntropy()
    {
        var pure = DensityMatrix.FromPure(
            new QuantumState(new[] { Complex32.One, Complex32.Zero }));
        Assert.Equal(0.0f, pure.VonNeumannEntropy());

        var bell = QuantumEntanglement.BellState(BellPair.PhiPlus);
        var reduced = DensityMatrix.FromPure(bell).PartialTrace(new[] { 0 }, 2);
        float entropy = reduced.VonNeumannEntropy();
        float ln2 = DeterministicScalar.Ln(2.0f);
        Assert.True(
            MathF.Abs(entropy - ln2) < 1e-5f,
            $"half a pair must carry one bit of entropy (got {entropy})");

        // The maximally mixed qubit carries one bit too.
        var mixed = DensityMatrix.MaximallyMixed(2);
        Assert.True(
            MathF.Abs(mixed.VonNeumannEntropy() - ln2) < 1e-5f,
            "the maximally mixed qubit must carry one bit");
    }

    /// <summary>
    /// M68: the CHSH game beats every classical strategy.
    /// </summary>
    [Fact]
    public void M68_TheChshGame()
    {
        // Two thousand rounds on a fixed seed: the count is pinned, not
        // just bounded.
        Assert.Equal(1696, QuantumEntanglement.ChshWinCount(2000, 77));
        // Deterministic twice over.
        Assert.Equal(1696, QuantumEntanglement.ChshWinCount(2000, 77));
        // A different seed plays a different game.
        Assert.NotEqual(1696, QuantumEntanglement.ChshWinCount(2000, 78));

        // The rate clears the classical bound of 3/4 with room to spare.
        float rate = QuantumEntanglement.ChshWinCount(2000, 77) / 2000.0f;
        Assert.True(rate > 0.75f, $"the quantum strategy must beat the classical bound (rate {rate})");
        Assert.True(
            MathF.Abs(rate - 0.8536f) < 0.02f,
            $"the rate must sit at the Tsirelson value cos²(pi/8) (rate {rate})");
    }

    /// <summary>
    /// M69: the noisy channels do what they say.
    /// </summary>
    [Fact]
    public void M69_TheNoisyChannels()
    {
        var zero = DensityMatrix.FromPure(
            new QuantumState(new[] { Complex32.One, Complex32.Zero }));
        var one = DensityMatrix.FromPure(
            new QuantumState(new[] { Complex32.Zero, Complex32.One }));

        // Full depolarisation is white noise, whatever went in.
        var white = zero.ApplyChannel(QuantumChannels.Depolarizing(1.0f));
        Assert.True(
            MathF.Abs(white.Matrix.Get(0, 0).Re - 0.5f) < 1e-6f,
            "full depolarisation must halve the population");
        Assert.True(
            MathF.Abs(white.Matrix.Get(1, 1).Re - 0.5f) < 1e-6f,
            "full depolarisation must halve the population");
        // No depolarisation is the identity.
        var same = zero.ApplyChannel(QuantumChannels.Depolarizing(0.0f));
        Assert.True(
            MathF.Abs(same.Matrix.Get(0, 0).Re - 1.0f) < 1e-6f,
            "zero depolarisation must change nothing");

        // Full amplitude damping sends |1> to |0>, and spares |0>.
        var decayed = one.ApplyChannel(QuantumChannels.AmplitudeDamping(1.0f));
        Assert.True(
            MathF.Abs(decayed.Matrix.Get(0, 0).Re - 1.0f) < 1e-6f,
            "full damping must decay |1> to |0>");
        var spared = zero.ApplyChannel(QuantumChannels.AmplitudeDamping(1.0f));
        Assert.True(
            MathF.Abs(spared.Matrix.Get(0, 0).Re - 1.0f) < 1e-6f,
            "damping must spare |0>");

        // Full dephasing kills the coherence and keeps the populations.
        var plusState = new QuantumState(new[] { Complex32.One, Complex32.One });
        plusState.Normalize();
        var plus = DensityMatrix.FromPure(plusState);
        var dephased = plus.ApplyChannel(QuantumChannels.Dephasing(0.5f));
        Assert.True(
            Complex32.NormSqr(dephased.Matrix.Get(0, 1)) < 1e-12f,
            "full dephasing must kill the coherence");
        Assert.True(
            MathF.Abs(dephased.Matrix.Get(0, 0).Re - 0.5f) < 1e-6f,
            "dephasing must keep the populations");
    }

    /// <summary>
    /// M70: the Lindblad step matches the hand computation and preserves
    /// the trace.
    /// </summary>
    [Fact]
    public void M70_TheLindbladStep()
    {
        // H = 0, one jump Z at rate 1, rho = |+><+|, dt = 0.01: the
        // off-diagonal decays by exactly the rate times the step.
        var plus = new QuantumState(new[] { Complex32.One, Complex32.One });
        plus.Normalize();
        var rho = DensityMatrix.FromPure(plus);
        var h = new ComplexMatrix(2, 2);
        var z = QuantumGates.Z();
        Assert.True(QuantumChannels.LindbladStep(
            rho, h, new[] { (1.0f, z) }, 0.01f));
        var m = rho.Matrix;
        Assert.Equal(0x3effffffu, Bits(m.Get(0, 0).Re));
        Assert.Equal(0x3efae147u, Bits(m.Get(0, 1).Re));
        Assert.Equal(0x3efae147u, Bits(m.Get(1, 0).Re));
        Assert.Equal(0x3effffffu, Bits(m.Get(1, 1).Re));
        // The trace is preserved up to the rounding: the two terms that
        // cancel are summed in a different order.
        Assert.True(
            MathF.Abs(rho.Trace().Re - 1.0f) < 1e-6f,
            "the trace must be preserved");

        // A hundred steps: the trace is still one, and the coherence has
        // decayed to e^-2 within the Euler error. A Z jump dephases at
        // twice its rate: LρL† and -{L†L,ρ}/2 move the off-diagonal the
        // same way.
        for (int i = 0; i < 99; i++)
        {
            Assert.True(QuantumChannels.LindbladStep(
                rho, h, new[] { (1.0f, QuantumGates.Z()) }, 0.01f));
        }
        Assert.True(
            MathF.Abs(rho.Trace().Re - 1.0f) < 1e-5f,
            "the trace must survive a hundred steps");
        float coherence = rho.Matrix.Get(0, 1).Re;
        float decayed = 0.5f * MathF.Exp(-2.0f);
        Assert.True(
            MathF.Abs(coherence - decayed) < 0.03f * MathF.Max(MathF.Abs(decayed), 1e-6f),
            $"the coherence must decay to e^-2 within 3% (got {coherence})");
    }

    /// <summary>
    /// M71: mixed states interpolate between pure and white noise.
    /// </summary>
    [Fact]
    public void M71_MixedStates()
    {
        // A pure state has purity one.
        var pure = DensityMatrix.FromPure(new QuantumState(new[]
        {
            new Complex32(0.6f, 0.0f),
            new Complex32(0.8f, 0.0f),
        }));
        Assert.True(MathF.Abs(pure.Purity() - 1.0f) < 1e-6f, "a pure state is pure");
        Assert.True(pure.IsValid(1e-6f), "a pure state is valid");

        // White noise over d levels has purity 1/d.
        var white = DensityMatrix.MaximallyMixed(4);
        Assert.True(
            MathF.Abs(white.Purity() - 0.25f) < 1e-6f,
            "white noise over four levels has purity one quarter");
        Assert.True(white.IsValid(1e-6f), "white noise is valid");

        // A non-Hermitian matrix is not a state.
        var bad = new ComplexMatrix(2, 2);
        bad.Set(0, 0, new Complex32(0.5f, 0.0f));
        bad.Set(1, 1, new Complex32(0.5f, 0.0f));
        bad.Set(0, 1, new Complex32(0.25f, 0.0f));
        Assert.False(
            DensityMatrix.FromMatrixUnchecked(bad).IsValid(1e-6f),
            "a non-Hermitian matrix must fail validation");
    }

    /// <summary>
    /// M72: the partial trace over three qubits and the diagonal rule.
    /// </summary>
    [Fact]
    public void M72_PartialTracesAndProbabilities()
    {
        // |010>: tracing out qubits 0 and 2 leaves |1> on qubit 1.
        var amps = new Complex32[8];
        amps[2] = Complex32.One;
        var s = new QuantumState(amps);
        var middle = DensityMatrix.FromPure(s).PartialTrace(new[] { 1 }, 3);
        Assert.Equal(2, middle.Dim);
        Assert.True(
            MathF.Abs(middle.Matrix.Get(1, 1).Re - 1.0f) < 1e-6f,
            "the middle qubit of |010> must be |1>");
        Assert.True(MathF.Abs(middle.Matrix.Get(0, 0).Re) < 1e-6f);

        // Keeping two qubits of |000> + |111> over sqrt(2): the survivor
        // is half |00> and half |11>. The amplitude is the correctly
        // rounded 1/sqrt(2), written as bits like the Rust.
        float inv_sqrt_2 = BitConverter.Int32BitsToSingle(0x3F3504F3);
        var ghzAmps = new Complex32[8];
        ghzAmps[0] = new Complex32(inv_sqrt_2, 0.0f);
        ghzAmps[7] = new Complex32(inv_sqrt_2, 0.0f);
        var ghz = new QuantumState(ghzAmps);
        var pair = DensityMatrix.FromPure(ghz).PartialTrace(new[] { 0, 1 }, 3);
        Assert.True(
            MathF.Abs(pair.Matrix.Get(0, 0).Re - 0.5f) < 1e-5f,
            "the surviving pair must be half |00>");
        Assert.True(
            MathF.Abs(pair.Matrix.Get(3, 3).Re - 0.5f) < 1e-5f,
            "the surviving pair must be half |11>");

        // The diagonal is the Born rule, in either language of the API.
        float[] probs = DensityMatrix.FromPure(ghz).Probabilities();
        Assert.Equal(8, probs.Length);
        Assert.True(
            MathF.Abs(probs[0] + probs[7] - 1.0f) < 1e-6f,
            "the diagonal must carry the whole probability");
    }
}
