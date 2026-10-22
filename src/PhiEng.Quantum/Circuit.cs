using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// A truth table for a Boolean oracle on <c>n</c> inputs.
/// </summary>
/// <remarks>
/// The table lists <c>f(x)</c> for every <c>x</c> in ascending order. An
/// oracle is data, not code, so the two engines cannot disagree on what
/// it computes: the same table in, the same unitary out.
/// </remarks>
public sealed class TruthTable
{
    private readonly int _inputs;
    private readonly bool[] _table;

    private TruthTable(int inputs, bool[] table)
    {
        _inputs = inputs;
        _table = table;
    }

    /// <summary>A table of 2^n constant values.</summary>
    public static TruthTable Constant(int nInputs, bool value)
    {
        int len = 1 << System.Math.Min(nInputs, 20);
        var table = new bool[len];
        for (int i = 0; i < len; i++)
        {
            table[i] = value;
        }
        return new TruthTable(nInputs, table);
    }

    /// <summary>The balanced oracle f(x) = x_0.</summary>
    public static TruthTable FirstBit(int nInputs)
    {
        int len = 1 << System.Math.Min(nInputs, 20);
        var table = new bool[len];
        for (int x = 0; x < len; x++)
        {
            table[x] = (x & 1) != 0;
        }
        return new TruthTable(nInputs, table);
    }

    /// <summary>The Bernstein-Vazirani oracle f(x) = parity(secret &amp; x).</summary>
    public static TruthTable DotProduct(int nInputs, uint secret)
    {
        int len = 1 << System.Math.Min(nInputs, 20);
        var table = new bool[len];
        for (int x = 0; x < len; x++)
        {
            table[x] = (System.Numerics.BitOperations.PopCount((uint)x & secret) & 1) != 0;
        }
        return new TruthTable(nInputs, table);
    }

    /// <summary>The number of input bits.</summary>
    public int Inputs => _inputs;

    /// <summary>The oracle's value on <paramref name="x"/>.</summary>
    public bool Value(int x) => (x >= 0 && x < _table.Length) && _table[x];
}

/// <summary>
/// A register of <c>n</c> qubits with a seeded measurement stream.
/// </summary>
public sealed class QuantumCircuit
{
    private readonly int _qubits;
    private readonly QuantumState _state;
    private readonly DeterministicRng _rng;

    /// <summary>A circuit of <paramref name="n"/> qubits in |0...0&gt;, drawing from <paramref name="seed"/>.</summary>
    public QuantumCircuit(int n, ulong seed)
    {
        int dim = 1 << System.Math.Min(n, 20);
        var amps = new Complex32[dim];
        amps[0] = Complex32.One;
        _qubits = n;
        _state = new QuantumState(amps);
        _rng = new DeterministicRng(seed);
    }

    /// <summary>The number of qubits.</summary>
    public int Qubits => _qubits;

    /// <summary>The register's state.</summary>
    public QuantumState State => _state;

    /// <summary>Applies a 2 by 2 gate to one qubit.</summary>
    public bool ApplySingle(ComplexMatrix gate, int qubit) =>
        QuantumGates.ApplySingle(_state, gate, qubit);

    /// <summary>Applies a CNOT.</summary>
    public bool ApplyCnot(int control, int target) =>
        QuantumGates.ApplyCnot(_state, control, target);

    /// <summary>Applies a Toffoli.</summary>
    public bool ApplyToffoli(int a, int b, int target) =>
        QuantumGates.ApplyToffoli(_state, a, b, target);

    /// <summary>Applies a SWAP.</summary>
    public bool ApplySwap(int a, int b) =>
        QuantumGates.ApplySwap(_state, a, b);

    /// <summary>Applies a controlled phase.</summary>
    public bool ApplyControlledPhase(int control, int target, float theta) =>
        QuantumGates.ApplyControlledPhase(_state, control, target, theta);

    /// <summary>
    /// Applies a bit-flip oracle: |x&gt;|y&gt; becomes |x&gt;|y ^ f(x)&gt;,
    /// with the ancilla the highest qubit.
    /// </summary>
    public bool ApplyOracle(TruthTable oracle)
    {
        int n = oracle.Inputs;
        if (n > 20 || _qubits < n + 1)
        {
            return false;
        }
        int ancilla = 1 << n;
        int dim = _state.Dim;
        var amps = _state.Amplitudes;
        for (int x = 0; x < (1 << n); x++)
        {
            if (!oracle.Value(x))
            {
                continue;
            }
            int j = x | ancilla;
            if (j >= dim)
            {
                return false;
            }
            (amps[x], amps[j]) = (amps[j], amps[x]);
        }
        return true;
    }

    /// <summary>Measures one qubit, collapsing the register.</summary>
    public bool? Measure(int qubit) => QuantumGates.Measure(_state, qubit, _rng);

    /// <summary>Measures every qubit, lowest first.</summary>
    public bool[] MeasureAll()
    {
        var result = new bool[_qubits];
        for (int q = 0; q < _qubits; q++)
        {
            result[q] = Measure(q) ?? false;
        }
        return result;
    }

    /// <summary>The probability of each basis state.</summary>
    public float[] Probabilities() => _state.BornProbabilities();
}

/// <summary>The textbook algorithms, each a pure function of its inputs.</summary>
public static class QuantumAlgorithms
{
    /// <summary>
    /// The Deutsch-Jozsa problem: is the oracle constant or balanced?
    /// Returns true for a constant oracle.
    /// </summary>
    public static bool DeutschJozsa(int nInputs, TruthTable oracle, ulong seed)
    {
        var circuit = new QuantumCircuit(nInputs + 1, seed);
        int ancilla = nInputs;
        // The ancilla starts in |1>.
        circuit.State.Amplitudes[0] = Complex32.Zero;
        circuit.State.Amplitudes[1 << ancilla] = Complex32.One;
        // Hadamards everywhere.
        var h = QuantumGates.H();
        for (int q = 0; q <= nInputs; q++)
        {
            circuit.ApplySingle(h, q);
        }
        circuit.ApplyOracle(oracle);
        for (int q = 0; q < nInputs; q++)
        {
            circuit.ApplySingle(h, q);
        }
        // Constant oracles leave the inputs in |0...0>.
        bool[] outcomes = circuit.MeasureAll();
        for (int q = 0; q < nInputs; q++)
        {
            if (outcomes[q])
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// The Bernstein-Vazirani problem: recover the secret string with one
    /// query.
    /// </summary>
    public static uint BernsteinVazirani(uint secret, int nInputs, ulong seed)
    {
        var oracle = TruthTable.DotProduct(nInputs, secret);
        var circuit = new QuantumCircuit(nInputs + 1, seed);
        int ancilla = nInputs;
        circuit.State.Amplitudes[0] = Complex32.Zero;
        circuit.State.Amplitudes[1 << ancilla] = Complex32.One;
        var h = QuantumGates.H();
        for (int q = 0; q <= nInputs; q++)
        {
            circuit.ApplySingle(h, q);
        }
        circuit.ApplyOracle(oracle);
        for (int q = 0; q < nInputs; q++)
        {
            circuit.ApplySingle(h, q);
        }
        uint recovered = 0;
        for (int q = 0; q < nInputs; q++)
        {
            if (circuit.Measure(q) ?? false)
            {
                recovered |= 1u << q;
            }
        }
        return recovered;
    }

    /// <summary>
    /// The number of Grover iterations for <paramref name="nQubits"/>:
    /// floor(pi/4 sqrt(N)).
    /// </summary>
    /// <remarks>
    /// The floor (not the ceiling) is what lands on the target: for four
    /// items one iteration rotates the state exactly onto the marked
    /// item, and the ceiling would overshoot it back down.
    /// </remarks>
    public static int GroverIterations(int nQubits)
    {
        int n = 1 << System.Math.Min(nQubits, 20);
        return (int)((System.Math.PI / 4.0) * System.Math.Sqrt(n));
    }

    /// <summary>
    /// The Grover search: the probability of the marked item after the
    /// prescribed number of iterations.
    /// </summary>
    public static float GroverProbability(int nQubits, int target)
    {
        int dim = 1 << System.Math.Min(nQubits, 20);
        if (target < 0 || target >= dim)
        {
            return 0.0f;
        }
        var circuit = new QuantumCircuit(nQubits, 1);
        var h = QuantumGates.H();
        for (int q = 0; q < nQubits; q++)
        {
            circuit.ApplySingle(h, q);
        }
        int iterations = System.Math.Max(1, GroverIterations(nQubits));
        for (int k = 0; k < iterations; k++)
        {
            // The phase oracle: the target changes sign.
            var amps = circuit.State.Amplitudes;
            amps[target] = -amps[target];
            // The diffusion: reflect about the mean.
            var mean = Complex32.Zero;
            foreach (var a in amps)
            {
                mean = mean + a;
            }
            mean = mean * (1.0f / dim);
            for (int i = 0; i < dim; i++)
            {
                amps[i] = mean * 2.0f - amps[i];
            }
        }
        return Complex32.NormSqr(circuit.State.Amplitudes[target]);
    }

    /// <summary>The quantum Fourier transform, in place.</summary>
    public static void ApplyQft(QuantumCircuit circuit, bool inverse)
    {
        int n = circuit.Qubits;
        var h = QuantumGates.H();
        if (!inverse)
        {
            for (int j = n - 1; j >= 0; j--)
            {
                circuit.ApplySingle(h, j);
                for (int k = j - 1; k >= 0; k--)
                {
                    int power = j - k + 1;
                    float angle = 2.0f * MathF.PI / (1u << System.Math.Min(power, 30));
                    circuit.ApplyControlledPhase(k, j, angle);
                }
            }
            for (int j = 0; j < n / 2; j++)
            {
                circuit.ApplySwap(j, n - 1 - j);
            }
        }
        else
        {
            // The adjoint in reverse order: F = S U, so F' = U' S, and
            // the swaps - their own inverse - come first.
            for (int j = 0; j < n / 2; j++)
            {
                circuit.ApplySwap(j, n - 1 - j);
            }
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < j; k++)
                {
                    int power = j - k + 1;
                    float angle = -2.0f * MathF.PI / (1u << System.Math.Min(power, 30));
                    circuit.ApplyControlledPhase(k, j, angle);
                }
                circuit.ApplySingle(h, j);
            }
        }
    }

    /// <summary>
    /// The teleportation protocol: the fidelity of Bob's qubit with the
    /// input state after Alice's measurements and the corrections.
    /// </summary>
    public static float TeleportFidelity(QuantumState input, ulong seed)
    {
        if (input.Dim != 2)
        {
            return 0.0f;
        }
        // Qubit 0 carries the input, qubits 1 and 2 share a Bell pair.
        var circuit = new QuantumCircuit(3, seed);
        circuit.State.Amplitudes[0] = Complex32.Zero;
        circuit.State.Amplitudes[1] = Complex32.Zero;
        circuit.State.Amplitudes[0] = input.Amplitudes[0];
        circuit.State.Amplitudes[1] = input.Amplitudes[1];
        var h = QuantumGates.H();
        circuit.ApplySingle(h, 1);
        circuit.ApplyCnot(1, 2);
        // Alice's Bell measurement.
        circuit.ApplyCnot(0, 1);
        circuit.ApplySingle(h, 0);
        bool m0 = circuit.Measure(0) ?? false;
        bool m1 = circuit.Measure(1) ?? false;
        // The corrections on Bob's qubit.
        if (m1)
        {
            circuit.ApplySingle(QuantumGates.X(), 2);
        }
        if (m0)
        {
            circuit.ApplySingle(QuantumGates.Z(), 2);
        }
        // Bob's qubit, read off the collapsed state: Alice's two qubits
        // are fixed, so the amplitudes at the two indices that differ
        // only in qubit 2 are his state up to the known branch.
        int bottom = (m0 ? 1 : 0) | ((m1 ? 1 : 0) << 1);
        var amps = circuit.State.Amplitudes;
        var bob = new QuantumState(new[] { amps[bottom], amps[bottom | 4] });
        bob.Normalize();
        return bob.Fidelity(input);
    }

    /// <summary>
    /// The superdense coding protocol: two classical bits travel on one
    /// qubit plus a shared Bell pair. Returns the two bits Bob decodes.
    /// </summary>
    public static (bool B0, bool B1) SuperdenseSend(bool bit0, bool bit1)
    {
        var circuit = new QuantumCircuit(2, 1);
        var h = QuantumGates.H();
        circuit.ApplySingle(h, 0);
        circuit.ApplyCnot(0, 1);
        // Alice encodes: X for the first bit, Z for the second.
        if (bit0)
        {
            circuit.ApplySingle(QuantumGates.X(), 0);
        }
        if (bit1)
        {
            circuit.ApplySingle(QuantumGates.Z(), 0);
        }
        // Bob decodes: CNOT, Hadamard, measure. The first measured bit
        // carries the Z-encoded bit and the second the X-encoded one, so
        // the pair comes back swapped relative to the encoding order.
        circuit.ApplyCnot(0, 1);
        circuit.ApplySingle(h, 0);
        bool m0 = circuit.Measure(0) ?? false;
        bool m1 = circuit.Measure(1) ?? false;
        return (m1, m0);
    }
}
