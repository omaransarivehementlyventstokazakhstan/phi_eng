using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The quantum gates: the matrices, their action on a register, the
/// Bloch sphere and the measurement.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::gates</c> module operation
/// for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// Qubit 0 is the least significant bit of the basis index
/// (little-endian): the amplitude of <c>|q(n-1) ... q(1) q(0)&gt;</c>
/// lives at <c>Amplitudes[index]</c> with
/// <c>index = q(0) + 2 q(1) + ...</c>.
/// </para>
/// <para>
/// A gate is never embedded in the full 2^n by 2^n matrix: the
/// amplitudes that differ only in the gate's qubits are visited in
/// ascending index order and the small matrix is applied to each pair
/// in place.
/// </para>
/// </remarks>
public static class QuantumGates
{
    /// <summary>
    /// 1 / sqrt(2), the Hadamard normalisation, as exact bits.
    /// </summary>
    /// <remarks>
    /// The correctly rounded <see cref="float"/> value, written as bits
    /// so the two engines cannot disagree on how it was computed.
    /// </remarks>
    public static float InvSqrt2 => BitConverter.Int32BitsToSingle(0x3F3504F3);

    /// <summary>The Hadamard gate.</summary>
    public static ComplexMatrix H()
    {
        float s = InvSqrt2;
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, new Complex32(s, 0.0f));
        m.Set(0, 1, new Complex32(s, 0.0f));
        m.Set(1, 0, new Complex32(s, 0.0f));
        m.Set(1, 1, new Complex32(-s, 0.0f));
        return m;
    }

    /// <summary>The Pauli X gate (the bit flip).</summary>
    public static ComplexMatrix X()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 1, Complex32.One);
        m.Set(1, 0, Complex32.One);
        return m;
    }

    /// <summary>The Pauli Y gate.</summary>
    public static ComplexMatrix Y()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 1, new Complex32(0.0f, -1.0f));
        m.Set(1, 0, new Complex32(0.0f, 1.0f));
        return m;
    }

    /// <summary>The Pauli Z gate (the phase flip).</summary>
    public static ComplexMatrix Z()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, Complex32.One);
        m.Set(1, 1, new Complex32(-1.0f, 0.0f));
        return m;
    }

    /// <summary>The S gate (the square root of Z).</summary>
    public static ComplexMatrix S()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, Complex32.One);
        m.Set(1, 1, Complex32.I);
        return m;
    }

    /// <summary>The T gate (the fourth root of Z).</summary>
    public static ComplexMatrix T()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, Complex32.One);
        m.Set(1, 1, Complex32.ExpI(MathF.PI / 4.0f));
        return m;
    }

    /// <summary>The X rotation by <paramref name="theta"/>.</summary>
    public static ComplexMatrix Rx(float theta)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(theta * 0.5f);
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, new Complex32(c, 0.0f));
        m.Set(0, 1, new Complex32(0.0f, -s));
        m.Set(1, 0, new Complex32(0.0f, -s));
        m.Set(1, 1, new Complex32(c, 0.0f));
        return m;
    }

    /// <summary>The Y rotation by <paramref name="theta"/>.</summary>
    public static ComplexMatrix Ry(float theta)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(theta * 0.5f);
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, new Complex32(c, 0.0f));
        m.Set(0, 1, new Complex32(-s, 0.0f));
        m.Set(1, 0, new Complex32(s, 0.0f));
        m.Set(1, 1, new Complex32(c, 0.0f));
        return m;
    }

    /// <summary>The Z rotation by <paramref name="theta"/>.</summary>
    public static ComplexMatrix Rz(float theta)
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, Complex32.ExpI(-theta * 0.5f));
        m.Set(1, 1, Complex32.ExpI(theta * 0.5f));
        return m;
    }

    /// <summary>The general single-qubit gate U3.</summary>
    public static ComplexMatrix U3(float theta, float phi, float lambda)
    {
        var (ct, st) = DeterministicTrig.ComputeCosSin(theta * 0.5f);
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, new Complex32(ct, 0.0f));
        m.Set(0, 1, Complex32.ExpI(lambda) * -st);
        m.Set(1, 0, Complex32.ExpI(phi) * st);
        m.Set(1, 1, Complex32.ExpI(phi + lambda) * ct);
        return m;
    }

    /// <summary>The controlled-NOT: flips the target where the control is set.</summary>
    public static ComplexMatrix Cnot()
    {
        var m = new ComplexMatrix(4, 4);
        m.Set(0, 0, Complex32.One);
        m.Set(1, 1, Complex32.One);
        m.Set(2, 3, Complex32.One);
        m.Set(3, 2, Complex32.One);
        return m;
    }

    /// <summary>The controlled-Z: a phase flip where both qubits are set.</summary>
    public static ComplexMatrix Cz()
    {
        var m = ComplexMatrix.Identity(4);
        m.Set(3, 3, new Complex32(-1.0f, 0.0f));
        return m;
    }

    /// <summary>The SWAP gate: exchanges the two qubits.</summary>
    public static ComplexMatrix Swap()
    {
        var m = new ComplexMatrix(4, 4);
        m.Set(0, 0, Complex32.One);
        m.Set(1, 2, Complex32.One);
        m.Set(2, 1, Complex32.One);
        m.Set(3, 3, Complex32.One);
        return m;
    }

    /// <summary>The Toffoli gate: flips the target where both controls are set.</summary>
    public static ComplexMatrix Toffoli()
    {
        var m = ComplexMatrix.Identity(8);
        m.Set(6, 6, Complex32.Zero);
        m.Set(7, 7, Complex32.Zero);
        m.Set(6, 7, Complex32.One);
        m.Set(7, 6, Complex32.One);
        return m;
    }

    /// <summary>The Fredkin gate: exchanges the targets where the control is set.</summary>
    public static ComplexMatrix Fredkin()
    {
        var m = ComplexMatrix.Identity(8);
        m.Set(5, 5, Complex32.Zero);
        m.Set(6, 6, Complex32.Zero);
        m.Set(5, 6, Complex32.One);
        m.Set(6, 5, Complex32.One);
        return m;
    }

    /// <summary>Applies a 2 by 2 gate to one qubit of a register.</summary>
    /// <returns>False when the qubit is out of range or the gate is not 2 by 2.</returns>
    public static bool ApplySingle(QuantumState state, ComplexMatrix gate, int qubit)
    {
        int dim = state.Dim;
        if (gate.Rows != 2 || gate.Cols != 2)
        {
            return false;
        }
        // The register must hold a whole number of qubits: its length
        // is a power of two, and the qubit must be one of them.
        if (dim == 0 || (dim & (dim - 1)) != 0)
        {
            return false;
        }
        if (qubit < 0 || qubit >= 32 || (1 << qubit) >= dim)
        {
            return false;
        }
        int mask = 1 << qubit;
        var g00 = gate.Get(0, 0);
        var g01 = gate.Get(0, 1);
        var g10 = gate.Get(1, 0);
        var g11 = gate.Get(1, 1);
        var amps = state.Amplitudes;
        for (int i = 0; i < dim; i++)
        {
            if ((i & mask) != 0)
            {
                continue;
            }
            int j = i | mask;
            if (j >= dim)
            {
                continue;
            }
            var a = amps[i];
            var b = amps[j];
            amps[i] = g00 * a + g01 * b;
            amps[j] = g10 * a + g11 * b;
        }
        return true;
    }

    /// <summary>Applies a controlled phase: the amplitudes where both qubits are set gain e^(i theta).</summary>
    /// <returns>False when either qubit is out of range.</returns>
    public static bool ApplyControlledPhase(QuantumState state, int control, int target, float theta)
    {
        int dim = state.Dim;
        if (control < 0 || control >= 32 || target < 0 || target >= 32
            || (1 << control) >= dim || (1 << target) >= dim)
        {
            return false;
        }
        var phase = Complex32.ExpI(theta);
        int mask = (1 << control) | (1 << target);
        var amps = state.Amplitudes;
        for (int i = 0; i < dim; i++)
        {
            if ((i & mask) == mask)
            {
                amps[i] = amps[i] * phase;
            }
        }
        return true;
    }

    /// <summary>Applies a CNOT with the given control and target.</summary>
    /// <returns>False when either qubit is out of range.</returns>
    public static bool ApplyCnot(QuantumState state, int control, int target)
    {
        int dim = state.Dim;
        if (control < 0 || control >= 32 || target < 0 || target >= 32
            || (1 << control) >= dim || (1 << target) >= dim || control == target)
        {
            return false;
        }
        int cmask = 1 << control;
        int tmask = 1 << target;
        var amps = state.Amplitudes;
        for (int i = 0; i < dim; i++)
        {
            if ((i & cmask) == 0 || (i & tmask) != 0)
            {
                continue;
            }
            int j = i | tmask;
            (amps[i], amps[j]) = (amps[j], amps[i]);
        }
        return true;
    }

    /// <summary>Applies a Toffoli with two controls and a target.</summary>
    /// <returns>False when any qubit is out of range or two coincide.</returns>
    public static bool ApplyToffoli(QuantumState state, int controlA, int controlB, int target)
    {
        int dim = state.Dim;
        if (controlA == controlB || controlA == target || controlB == target)
        {
            return false;
        }
        foreach (int q in new[] { controlA, controlB, target })
        {
            if (q < 0 || q >= 32 || (1 << q) >= dim)
            {
                return false;
            }
        }
        int amask = 1 << controlA;
        int bmask = 1 << controlB;
        int tmask = 1 << target;
        var amps = state.Amplitudes;
        for (int i = 0; i < dim; i++)
        {
            if ((i & amask) == 0 || (i & bmask) == 0 || (i & tmask) != 0)
            {
                continue;
            }
            int j = i | tmask;
            (amps[i], amps[j]) = (amps[j], amps[i]);
        }
        return true;
    }

    /// <summary>Applies a SWAP of two qubits.</summary>
    /// <returns>False when either qubit is out of range or the two coincide.</returns>
    public static bool ApplySwap(QuantumState state, int a, int b)
    {
        int dim = state.Dim;
        if (a == b)
        {
            return false;
        }
        foreach (int q in new[] { a, b })
        {
            if (q < 0 || q >= 32 || (1 << q) >= dim)
            {
                return false;
            }
        }
        int amask = 1 << a;
        int bmask = 1 << b;
        var amps = state.Amplitudes;
        // Exactly one of the two bits set: the pair straddles the swap.
        for (int i = 0; i < dim; i++)
        {
            bool hasA = (i & amask) != 0;
            bool hasB = (i & bmask) != 0;
            if (hasA == hasB || !hasA)
            {
                continue;
            }
            int j = (i & ~amask) | bmask;
            (amps[i], amps[j]) = (amps[j], amps[i]);
        }
        return true;
    }

    /// <summary>The Bloch-sphere coordinates (x, y, z) of a single-qubit state.</summary>
    /// <returns>Null when the state is not one qubit.</returns>
    public static (float X, float Y, float Z)? Bloch(QuantumState state)
    {
        if (state.Dim != 2)
        {
            return null;
        }
        var a0 = state.Amplitudes[0];
        var a1 = state.Amplitudes[1];
        var cross = Complex32.Conjugate(a0) * a1;
        return (
            2.0f * cross.Re,
            2.0f * cross.Im,
            Complex32.NormSqr(a0) - Complex32.NormSqr(a1));
    }

    /// <summary>Measures one qubit of a register, collapsing the state onto the measured branch.</summary>
    /// <returns>The outcome, or null when the qubit is out of range.</returns>
    public static bool? Measure(QuantumState state, int qubit, DeterministicRng rng)
    {
        int dim = state.Dim;
        if (qubit < 0 || qubit >= 32 || (1 << qubit) >= dim)
        {
            return null;
        }
        int mask = 1 << qubit;
        float p0 = 0.0f;
        var amps = state.Amplitudes;
        for (int i = 0; i < dim; i++)
        {
            if ((i & mask) == 0)
            {
                p0 += Complex32.NormSqr(amps[i]);
            }
        }
        bool outcome = rng.NextFloat() >= p0;
        float p = outcome ? 1.0f - p0 : p0;
        if (!float.IsFinite(p) || p <= 0.0f)
        {
            return null;
        }
        // Collapse onto the measured branch and renormalise.
        float inv = 1.0f / MathF.Sqrt(p);
        for (int i = 0; i < dim; i++)
        {
            bool set = (i & mask) != 0;
            if (set == outcome)
            {
                amps[i] = Complex32.Scale(amps[i], inv);
            }
            else
            {
                amps[i] = Complex32.Zero;
            }
        }
        return outcome;
    }
}
