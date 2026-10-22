using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Second quantization: Fock states, ladder operators, and the Fermi
/// gas.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::fock</c> module operation
/// for operation, so the two engines agree bit-for-bit. Bosons stack
/// without limit (up to the truncation); fermions refuse double
/// occupancy, which is the Pauli exclusion principle as a one-line
/// theorem.
/// </remarks>
public static class QuantumFock
{
    /// <summary>
    /// The bosonic annihilation operator in a <paramref name="cutoff"/>-
    /// dimensional Fock space: a|n&gt; = sqrt(n) |n - 1&gt;.
    /// </summary>
    public static ComplexMatrix BosonAnnihilation(int cutoff)
    {
        var m = new ComplexMatrix(cutoff, cutoff);
        for (int n = 1; n < cutoff; n++)
        {
            m.Set(n - 1, n, new Complex32(MathF.Sqrt(n), 0.0f));
        }
        return m;
    }

    /// <summary>The bosonic creation operator: the dagger of the annihilation.</summary>
    public static ComplexMatrix BosonCreation(int cutoff) =>
        BosonAnnihilation(cutoff).Dagger();

    /// <summary>The number operator N = a†a: diagonal with 0, 1, 2, ....</summary>
    public static ComplexMatrix NumberOperator(int cutoff)
    {
        var m = new ComplexMatrix(cutoff, cutoff);
        for (int n = 0; n < cutoff; n++)
        {
            m.Set(n, n, new Complex32(n, 0.0f));
        }
        return m;
    }

    /// <summary>
    /// The fermionic annihilation operator for one mode, in the
    /// (|0&gt;, |1&gt;) basis: a|1&gt; = |0&gt;, and a|0&gt; = 0.
    /// </summary>
    public static ComplexMatrix FermionAnnihilation()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 1, Complex32.One);
        return m;
    }

    /// <summary>The fermionic creation operator: the dagger of the annihilation.</summary>
    public static ComplexMatrix FermionCreation() =>
        FermionAnnihilation().Dagger();

    /// <summary>The anticommutator {a, b} = ab + ba.</summary>
    public static ComplexMatrix Anticommutator(ComplexMatrix a, ComplexMatrix b) =>
        a.MatMul(b).Add(b.MatMul(a));

    /// <summary>The commutator [a, b] = ab - ba.</summary>
    public static ComplexMatrix Commutator(ComplexMatrix a, ComplexMatrix b) =>
        a.MatMul(b).Sub(b.MatMul(a));

    /// <summary>
    /// The ground-state energy of spinless fermions filling
    /// single-particle levels: at zero temperature the particles occupy
    /// the lowest levels first.
    /// </summary>
    /// <returns>
    /// 0.0 when asked for no particles or for more particles than
    /// levels: the extra fermions have nowhere to go, and inventing
    /// levels for them would be guessing.
    /// </returns>
    public static float FermiGasEnergy(float[] levels, int nParticles)
    {
        if (nParticles <= 0 || nParticles > levels.Length)
        {
            return 0.0f;
        }
        // The levels arrive sorted in every call site that matters; the
        // selection below takes the smallest entries however they
        // arrive. OrderBy is stable, like the Rust sort_by.
        float sum = 0.0f;
        int taken = 0;
        foreach (float level in levels.OrderBy(v => v))
        {
            if (taken >= nParticles)
            {
                break;
            }
            sum += level;
            taken++;
        }
        return sum;
    }
}
