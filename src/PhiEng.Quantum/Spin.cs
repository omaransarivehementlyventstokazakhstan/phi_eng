using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Spin-1/2: the Pauli algebra, the Stern-Gerlach measurement, and
/// precession in a magnetic field.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::spin</c> module operation
/// for operation, so the two engines agree bit-for-bit. A spin-1/2
/// particle lives in a two-dimensional Hilbert space, so a spinor
/// <em>is</em> a single-qubit state. Natural units throughout
/// (<c>hbar = 1</c>).
/// </remarks>
public static class QuantumSpin
{
    /// <summary>The Pauli X matrix.</summary>
    public static ComplexMatrix PauliX() => QuantumGates.X();

    /// <summary>The Pauli Y matrix.</summary>
    public static ComplexMatrix PauliY() => QuantumGates.Y();

    /// <summary>The Pauli Z matrix.</summary>
    public static ComplexMatrix PauliZ() => QuantumGates.Z();

    /// <summary>The spin-up state: the +1 eigenstate of S_z.</summary>
    public static QuantumState SpinUp() =>
        new(new[] { Complex32.One, Complex32.Zero });

    /// <summary>The spin-down state: the -1 eigenstate of S_z.</summary>
    public static QuantumState SpinDown() =>
        new(new[] { Complex32.Zero, Complex32.One });

    /// <summary>
    /// Measures S_z of a spinor: true for up, false for down.
    /// </summary>
    /// <returns>Null when the state is not one spin.</returns>
    public static bool? MeasureSz(QuantumState state, DeterministicRng rng)
    {
        if (state.Dim != 2)
        {
            return null;
        }
        // The register reports true for |1>; spin up is |0>.
        bool? one = QuantumGates.Measure(state, 0, rng);
        return one == null ? null : (bool?)!one.Value;
    }

    /// <summary>The probability of measuring spin up.</summary>
    public static float UpProbability(QuantumState state)
    {
        if (state.Dim != 2)
        {
            return 0.0f;
        }
        return Complex32.NormSqr(state.Amplitudes[0]);
    }

    /// <summary>
    /// The Zeeman Hamiltonian of a uniform field: H = -mu (B . sigma) / 2.
    /// </summary>
    /// <remarks>
    /// With the field along z the energies are ∓mu B / 2: the splitting
    /// is the Larmor frequency itself in natural units.
    /// </remarks>
    public static ComplexMatrix ZeemanHamiltonian(float[] field, float moment)
    {
        var h = new ComplexMatrix(2, 2);
        var terms = new[]
        {
            (QuantumGates.X(), field[0]),
            (QuantumGates.Y(), field[1]),
            (QuantumGates.Z(), field[2]),
        };
        foreach (var (pauli, component) in terms)
        {
            var scaled = pauli.Scale(-moment * component * 0.5f);
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    h.Set(r, c, h.Get(r, c) + scaled.Get(r, c));
                }
            }
        }
        return h;
    }

    /// <summary>
    /// Evolves a spinor under a transverse field: the Rabi flop.
    /// </summary>
    /// <remarks>
    /// The Hamiltonian is (omega / 2) sigma_x, so the exact propagator
    /// is the X rotation by omega t: starting from spin up, the down
    /// probability is sin²(omega t / 2).
    /// </remarks>
    public static bool RabiEvolve(QuantumState state, float omega, float t)
    {
        if (state.Dim != 2)
        {
            return false;
        }
        return QuantumGates.ApplySingle(state, QuantumGates.Rx(omega * t), 0);
    }

    /// <summary>
    /// The spin vector (&lt;S_x&gt;, &lt;S_y&gt;, &lt;S_z&gt;) of a spinor,
    /// in units of hbar = 1.
    /// </summary>
    /// <returns>Null when the state is not one spin.</returns>
    public static (float X, float Y, float Z)? SpinVector(QuantumState state)
    {
        var bloch = QuantumGates.Bloch(state);
        if (bloch == null)
        {
            return null;
        }
        return (bloch.Value.X * 0.5f, bloch.Value.Y * 0.5f, bloch.Value.Z * 0.5f);
    }
}
