using System.Numerics;
using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// The gates and circuits (M39–M48). Mirrors
/// <c>crates/engine/tests/quantum_gates.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// Qubit 0 is the least significant bit of the basis index
/// (little-endian): every test here and in the Rust core uses that
/// order.
/// </remarks>
public class QuantumGatesTests
{
    /// <summary>The bit pattern of a float, as the pins are written.</summary>
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>
    /// M39: the single-qubit gates square to what they should.
    /// </summary>
    [Fact]
    public void M39_TheSingleQubitGateIdentities()
    {
        var id = ComplexMatrix.Identity(2);
        float Close(ComplexMatrix m)
        {
            float worst = 0.0f;
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    var d = m.Get(r, c) - id.Get(r, c);
                    worst = MathF.Max(worst, MathF.Max(MathF.Abs(d.Re), MathF.Abs(d.Im)));
                }
            }
            return worst;
        }

        // H² = X² = Y² = Z² = I.
        Assert.True(Close(QuantumGates.H().MatMul(QuantumGates.H())) < 1e-6f, "H squared must be I");
        Assert.True(Close(QuantumGates.X().MatMul(QuantumGates.X())) < 1e-6f, "X squared must be I");
        Assert.True(Close(QuantumGates.Y().MatMul(QuantumGates.Y())) < 1e-6f, "Y squared must be I");
        Assert.True(Close(QuantumGates.Z().MatMul(QuantumGates.Z())) < 1e-6f, "Z squared must be I");

        // S² = Z and T⁴ = Z (so T⁸ = I).
        var s2 = QuantumGates.S().MatMul(QuantumGates.S());
        var z = QuantumGates.Z();
        float worst = 0.0f;
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                var d = s2.Get(r, c) - z.Get(r, c);
                worst = MathF.Max(worst, MathF.Max(MathF.Abs(d.Re), MathF.Abs(d.Im)));
            }
        }
        Assert.True(worst < 1e-6f, "S squared must be Z");
        var t2 = QuantumGates.T().MatMul(QuantumGates.T());
        var t4 = t2.MatMul(t2);
        worst = 0.0f;
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                var d = t4.Get(r, c) - z.Get(r, c);
                worst = MathF.Max(worst, MathF.Max(MathF.Abs(d.Re), MathF.Abs(d.Im)));
            }
        }
        Assert.True(worst < 1e-6f, "T to the fourth must be Z");
        Assert.True(Close(t4.MatMul(t4)) < 1e-6f, "T to the eighth must be I");

        // The Hadamard's entries are exactly ±1/√2.
        Assert.Equal(0x3f3504f3u, Bits(QuantumGates.H().Get(0, 0).Re));
        Assert.Equal(0x3f3504f3u, Bits(QuantumGates.H().Get(1, 0).Re));
        Assert.Equal(0xbf3504f3u, Bits(QuantumGates.H().Get(1, 1).Re));
        Assert.Equal(0u, Bits(QuantumGates.H().Get(0, 0).Im));

        // A zero rotation is the identity, whatever the axis.
        foreach (var g in new[] { QuantumGates.Rx(0.0f), QuantumGates.Ry(0.0f), QuantumGates.Rz(0.0f) })
        {
            Assert.True(Close(g) < 1e-6f, "a zero rotation must be I");
        }
        // RX(π) is X up to a global phase: it swaps the populations.
        var s = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        Assert.True(QuantumGates.ApplySingle(s, QuantumGates.Rx(MathF.PI), 0));
        Assert.True(
            Complex32.NormSqr(s.Amplitudes[0]) < 1e-6f
                && MathF.Abs(Complex32.NormSqr(s.Amplitudes[1]) - 1.0f) < 1e-6f,
            "RX(pi) must flip |0> to |1>");

        // U3 contains the other rotations: U3(π/2, 0, π) is H up to a phase.
        var u = QuantumGates.U3(MathF.PI / 2.0f, 0.0f, MathF.PI);
        var a = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        var b = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        Assert.True(QuantumGates.ApplySingle(a, u, 0));
        Assert.True(QuantumGates.ApplySingle(b, QuantumGates.H(), 0));
        Assert.True(a.Fidelity(b) > 1.0f - 1e-6f, "U3(pi/2, 0, pi) must be H up to a phase");
    }

    /// <summary>
    /// M40: the entangling gates' truth tables.
    /// </summary>
    [Fact]
    public void M40_TheEntanglingGatesTruthTables()
    {
        // CNOT with control qubit 1 and target qubit 0: |10> <-> |11>.
        (int Input, int Output)[] cnotCases = { (0, 0), (1, 1), (2, 3), (3, 2) };
        foreach (var (input, output) in cnotCases)
        {
            var s = QuantumState.Basis(4, input);
            Assert.True(QuantumGates.ApplyCnot(s, 1, 0));
            var probs = s.BornProbabilities();
            for (int i = 0; i < probs.Length; i++)
            {
                float want = i == output ? 1.0f : 0.0f;
                Assert.True(
                    MathF.Abs(probs[i] - want) < 1e-6f,
                    $"CNOT|{input} ({Convert.ToString(input, 2)})> must be |{output}>");
            }
        }
        // The matrix says the same.
        var cnot = QuantumGates.Cnot();
        Assert.Equal(0x3f800000u, Bits(cnot.Get(2, 3).Re));
        Assert.Equal(0x3f800000u, Bits(cnot.Get(3, 2).Re));

        // CZ flips the phase of |11> and nothing else. It is a 4x4 gate,
        // so the single-qubit path refuses it.
        var s2 = new QuantumState(new[] { Complex32.One, Complex32.One, Complex32.One, Complex32.One });
        s2.Normalize();
        var t = new QuantumState(new[] { Complex32.One, Complex32.One, Complex32.One, Complex32.One });
        t.Normalize();
        Assert.False(
            QuantumGates.ApplySingle(t, QuantumGates.Cz(), 0),
            "a 4x4 gate must not apply as a single-qubit gate");
        var op = QuantumOperators.Dense(QuantumGates.Cz());
        var dense = new Complex32[4];
        op.ApplyTo(s2.Amplitudes, dense);
        var u = new QuantumState(dense);
        Assert.True(
            Complex32.NormSqr(u.Amplitudes[3] + s2.Amplitudes[3]) < 1e-12f,
            "CZ must flip |11>");
        Assert.True(
            Complex32.NormSqr(u.Amplitudes[0] - s2.Amplitudes[0]) < 1e-12f,
            "CZ must spare |00>");

        // SWAP exchanges |01> and |10>.
        var sw = QuantumState.Basis(4, 1);
        Assert.True(QuantumGates.ApplySwap(sw, 0, 1));
        Assert.True(
            Complex32.NormSqr(sw.Amplitudes[2] - Complex32.One) < 1e-12f,
            "SWAP|01> must be |10>");
        var swap = QuantumGates.Swap();
        Assert.Equal(0x3f800000u, Bits(swap.Get(1, 2).Re));

        // Toffoli flips the target only when both controls are set.
        for (int input = 0; input < 8; input++)
        {
            var ts = QuantumState.Basis(8, input);
            Assert.True(QuantumGates.ApplyToffoli(ts, 2, 1, 0));
            int want = input == 6 ? 7 : (input == 7 ? 6 : input);
            var probs = ts.BornProbabilities();
            for (int i = 0; i < probs.Length; i++)
            {
                float w = i == want ? 1.0f : 0.0f;
                Assert.True(MathF.Abs(probs[i] - w) < 1e-6f, $"Toffoli|{input}> must be |{want}>");
            }
        }

        // A controlled phase touches only the doubly-set amplitude.
        var cp = new QuantumState(new[] { Complex32.One, Complex32.One, Complex32.One, Complex32.One });
        cp.Normalize();
        Assert.True(QuantumGates.ApplyControlledPhase(cp, 0, 1, MathF.PI / 2.0f));
        var wantPhase = Complex32.ExpI(MathF.PI / 2.0f);
        var got = cp.Amplitudes[3] * 2.0f;
        Assert.True(
            Complex32.NormSqr(got - wantPhase) < 1e-10f,
            "the controlled phase must be e^(i pi/2) on |11>");

        // Fredkin swaps its targets only when the control is set.
        var fredkin = QuantumGates.Fredkin();
        Assert.Equal(0x3f800000u, Bits(fredkin.Get(5, 6).Re));
        Assert.Equal(0x3f800000u, Bits(fredkin.Get(6, 5).Re));
        Assert.Equal(0u, Bits(fredkin.Get(5, 5).Re));
        // The Toffoli matrix says the same as the indexed version.
        var toffoli = QuantumGates.Toffoli();
        Assert.Equal(0x3f800000u, Bits(toffoli.Get(6, 7).Re));
        Assert.Equal(0x3f800000u, Bits(toffoli.Get(7, 6).Re));
        Assert.Equal(0u, Bits(toffoli.Get(6, 6).Re));
    }

    /// <summary>
    /// M41: the Bloch sphere.
    /// </summary>
    [Fact]
    public void M41_TheBlochSphere()
    {
        var zero = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        Assert.Equal((0.0f, 0.0f, 1.0f), QuantumGates.Bloch(zero));

        var one = new QuantumState(new[] { Complex32.Zero, Complex32.One });
        Assert.Equal((0.0f, 0.0f, -1.0f), QuantumGates.Bloch(one));

        // |+> = H|0> sits on the equator, on the x axis.
        var plus = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        Assert.True(QuantumGates.ApplySingle(plus, QuantumGates.H(), 0));
        var (x, y, z) = QuantumGates.Bloch(plus)!.Value;
        Assert.True(MathF.Abs(x - 1.0f) < 1e-6f, $"x = {x}");
        Assert.True(MathF.Abs(y) < 1e-6f, $"y = {y}");
        Assert.True(MathF.Abs(z) < 1e-6f, $"z = {z}");

        // Every pure state sits on the sphere.
        var rng = new DeterministicRng(11);
        for (int k = 0; k < 16; k++)
        {
            var s = new QuantumState(2);
            for (int i = 0; i < 2; i++)
            {
                s.Amplitudes[i] = new Complex32(rng.NextFloatSigned(), rng.NextFloatSigned());
            }
            s.Normalize();
            var b = QuantumGates.Bloch(s)!.Value;
            Assert.True(
                MathF.Abs(b.X * b.X + b.Y * b.Y + b.Z * b.Z - 1.0f) < 1e-5f,
                "pure states live on the sphere");
        }

        // Two qubits have no Bloch vector.
        Assert.Null(QuantumGates.Bloch(new QuantumState(4)));
    }

    /// <summary>
    /// M42: the measurement rule.
    /// </summary>
    [Fact]
    public void M42_TheMeasurementRule()
    {
        // |+> measures 0 or 1 with equal probability, and the draw is
        // reproducible: the same seed gives the same shots.
        static bool[] Shots(ulong seed)
        {
            var rng = new DeterministicRng(seed);
            var outcomes = new bool[16];
            for (int i = 0; i < 16; i++)
            {
                var st = new QuantumState(new[] { Complex32.One, Complex32.One });
                st.Normalize();
                outcomes[i] = QuantumGates.Measure(st, 0, rng) ?? false;
            }
            return outcomes;
        }
        var a = Shots(1);
        var b = Shots(1);
        Assert.Equal(a, b);
        int ones = 0;
        foreach (var o in a)
        {
            if (o)
            {
                ones++;
            }
        }
        Assert.True(ones >= 4 && ones <= 12, $"sixteen fair shots must not be degenerate ({ones} ones)");
        Assert.False(a.SequenceEqual(Shots(2)), "a different seed must give different shots");

        // The collapse: after the measurement the state is the branch
        // that was measured, renormalised.
        var rng = new DeterministicRng(3);
        var s = new QuantumState(new[] { new Complex32(0.6f, 0.0f), new Complex32(0.8f, 0.0f) });
        bool outcome = QuantumGates.Measure(s, 0, rng) ?? false;
        int branch = outcome ? 1 : 0;
        Assert.Equal(0x3f800000u, Bits(Complex32.NormSqr(s.Amplitudes[branch])));
        Assert.Equal(0u, Bits(Complex32.NormSqr(s.Amplitudes[1 - branch])));

        // An out-of-range qubit refuses.
        var t = new QuantumState(new[] { Complex32.One, Complex32.Zero });
        Assert.Null(QuantumGates.Measure(t, 4, rng));
    }

    /// <summary>
    /// M43: Deutsch-Jozsa finds constant oracles in one query.
    /// </summary>
    [Fact]
    public void M43_DeutschJozsa()
    {
        Assert.True(
            QuantumAlgorithms.DeutschJozsa(3, TruthTable.Constant(3, false), 1),
            "a constant-zero oracle must read constant");
        Assert.True(
            QuantumAlgorithms.DeutschJozsa(3, TruthTable.Constant(3, true), 2),
            "a constant-one oracle must read constant");
        Assert.False(
            QuantumAlgorithms.DeutschJozsa(3, TruthTable.FirstBit(3), 3),
            "the first-bit oracle is balanced and must read balanced");
        Assert.False(
            QuantumAlgorithms.DeutschJozsa(1, TruthTable.FirstBit(1), 4),
            "one input bit is enough for a balanced oracle");
    }

    /// <summary>
    /// M44: Bernstein-Vazirani recovers the secret in one query.
    /// </summary>
    [Fact]
    public void M44_BernsteinVazirani()
    {
        foreach (uint secret in new uint[] { 0, 1, 5, 21, 255 })
        {
            int n = secret == 0 ? 1 : 32 - BitOperations.LeadingZeroCount(secret);
            n = System.Math.Clamp(n, 1, 8);
            uint mask = ((1u << n) - 1);
            Assert.Equal(
                secret & mask,
                QuantumAlgorithms.BernsteinVazirani(secret & mask, n, 7));
        }
    }

    /// <summary>
    /// M45: Grover finds the marked item.
    /// </summary>
    [Fact]
    public void M45_GroverFindsTheMarkedItem()
    {
        Assert.Equal(1, QuantumAlgorithms.GroverIterations(2));
        Assert.Equal(2, QuantumAlgorithms.GroverIterations(3));

        // Four items: one iteration lands exactly on the target.
        float p = QuantumAlgorithms.GroverProbability(2, 2);
        Assert.True(p >= 1.0f - 1.0f / 4.0f, $"four items must succeed with probability >= 3/4 (got {p})");
        Assert.True(MathF.Abs(p - 1.0f) < 1e-4f, $"four items land exactly on the target (got {p})");

        // Eight items: two iterations clear 7/8.
        p = QuantumAlgorithms.GroverProbability(3, 5);
        Assert.True(p >= 1.0f - 1.0f / 8.0f, $"eight items must succeed with probability >= 7/8 (got {p})");

        // An out-of-range target is a zero probability, not a panic.
        Assert.Equal(0.0f, QuantumAlgorithms.GroverProbability(2, 9));
    }

    /// <summary>
    /// M46: the Fourier transform and its inverse reconstruct the input.
    /// </summary>
    [Fact]
    public void M46_TheFourierTransformRoundTrips()
    {
        for (int basis = 0; basis < 8; basis++)
        {
            var circuit = new QuantumCircuit(3, 1);
            circuit.State.Amplitudes[0] = Complex32.Zero;
            circuit.State.Amplitudes[basis] = Complex32.One;
            var start = new QuantumState(circuit.State.Amplitudes);
            QuantumAlgorithms.ApplyQft(circuit, false);
            // A basis state transforms to a flat spectrum.
            foreach (var amp in circuit.State.Amplitudes)
            {
                Assert.True(
                    MathF.Abs(Complex32.NormSqr(amp) - 1.0f / 8.0f) < 1e-5f,
                    "a basis state must transform flat");
            }
            QuantumAlgorithms.ApplyQft(circuit, true);
            Assert.True(
                circuit.State.Fidelity(start) > 1.0f - 1e-5f,
                $"the inverse must reconstruct basis state {basis}");
        }
    }

    /// <summary>
    /// M47: teleportation delivers the input state with fidelity one.
    /// </summary>
    [Fact]
    public void M47_TeleportationDeliversTheState()
    {
        var inputs = new[]
        {
            new QuantumState(new[] { Complex32.One, Complex32.Zero }),
            new QuantumState(new[] { Complex32.Zero, Complex32.One }),
            new QuantumState(new[] { new Complex32(0.6f, 0.0f), new Complex32(0.8f, 0.0f) }),
        };
        for (int k = 0; k < inputs.Length; k++)
        {
            // Every seed: the fidelity is one whatever Alice measures.
            foreach (ulong seed in new ulong[] { 1, 2, 3 })
            {
                float fidelity = QuantumAlgorithms.TeleportFidelity(inputs[k], seed);
                Assert.True(
                    fidelity > 1.0f - 1e-5f,
                    $"teleportation {k} with seed {seed} must deliver fidelity one (got {fidelity})");
            }
        }
        Assert.Equal(0.0f, QuantumAlgorithms.TeleportFidelity(new QuantumState(4), 1));
    }

    /// <summary>
    /// M48: superdense coding carries two bits on one qubit.
    /// </summary>
    [Fact]
    public void M48_SuperdenseCodingCarriesTwoBits()
    {
        foreach (bool b0 in new[] { false, true })
        {
            foreach (bool b1 in new[] { false, true })
            {
                Assert.Equal(
                    (b0, b1),
                    QuantumAlgorithms.SuperdenseSend(b0, b1));
            }
        }
    }
}
