using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Small many-body problems: the Hubbard dimer and the entanglement
/// spectrum.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::manybody</c> module entry
/// for entry, so the two engines agree bit-for-bit.
/// </remarks>
public static class QuantumManyBody
{
    /// <summary>The singlet sector of the Hubbard dimer: two sites, two electrons, spectrum ascending.</summary>
    /// <remarks>
    /// Either electron can hop, so each double occupancy couples to
    /// the split singlet with <c>sqrt(2) t</c>.
    /// </remarks>
    public static float[] HubbardDimerExact(float hopping, float u)
    {
        var t = new Complex32(-hopping * MathF.Sqrt(2.0f), 0.0f);
        var uu = new Complex32(u, 0.0f);
        var m = new ComplexMatrix(3, 3);
        m.Set(0, 0, uu);
        m.Set(1, 1, uu);
        m.Set(0, 2, t);
        m.Set(2, 0, t);
        m.Set(1, 2, t);
        m.Set(2, 1, t);
        return QuantumLinalg.EigenHermitian(m, 32, 1e-8f).Eigenvalues;
    }

    /// <summary>The restricted Hartree-Fock energy of the Hubbard dimer, by self-consistent iteration, with the converged density beside it.</summary>
    /// <remarks>
    /// The Fock matrix is <c>F_ii = U n_i / 2</c> with <c>-t</c>
    /// between the sites; its lowest eigenvector, doubly occupied, is
    /// the new density, mixed half-and-half with the old one. The
    /// energy is variational, never below the exact one.
    /// </remarks>
    public static (float Energy, ComplexMatrix Density) HubbardDimerRhf(float hopping, float u, int iterations)
    {
        // The bonding orbital, doubly occupied: uniform unit density.
        var p = new ComplexMatrix(2, 2);
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                p.Set(r, c, Complex32.One);
            }
        }
        var hop = new Complex32(-hopping, 0.0f);
        for (int it = 0; it < iterations; it++)
        {
            var f = new ComplexMatrix(2, 2);
            f.Set(0, 0, new Complex32(u * p.Get(0, 0).Re / 2.0f, 0.0f));
            f.Set(1, 1, new Complex32(u * p.Get(1, 1).Re / 2.0f, 0.0f));
            f.Set(0, 1, hop);
            f.Set(1, 0, hop);
            var solved = QuantumLinalg.EigenHermitian(f, 32, 1e-8f);
            var c0 = solved.Eigenvectors.Get(0, 0);
            var c1 = solved.Eigenvectors.Get(1, 0);
            var two = new Complex32(2.0f, 0.0f);
            var half = new Complex32(0.5f, 0.0f);
            var q00 = two * c0 * Complex32.Conjugate(c0);
            var q01 = two * c0 * Complex32.Conjugate(c1);
            var q11 = two * c1 * Complex32.Conjugate(c1);
            p.Set(0, 0, half * (p.Get(0, 0) + q00));
            p.Set(0, 1, half * (p.Get(0, 1) + q01));
            p.Set(1, 0, half * (p.Get(1, 0) + Complex32.Conjugate(q01)));
            p.Set(1, 1, half * (p.Get(1, 1) + q11));
        }
        float eOne = (p.Get(0, 1) + p.Get(1, 0)).Re * -hopping;
        float eInt = u * (p.Get(0, 0).Re * p.Get(0, 0).Re + p.Get(1, 1).Re * p.Get(1, 1).Re) / 4.0f;
        return (eOne + eInt, p);
    }

    /// <summary>A transverse-field Ising chain: ferromagnetic <paramref name="j"/> along z, field <paramref name="h"/> along x, open ends.</summary>
    public static ComplexMatrix TfiChain(int n, float j, float h)
    {
        n = System.Math.Max(n, 1);
        var id = ComplexMatrix.Identity(2);
        var full = new ComplexMatrix(1 << n, 1 << n);
        // The bonds: -J Z_i Z_{i+1}.
        for (int i = 0; i < n - 1; i++)
        {
            var term = ComplexMatrix.FromDiagonal(new[] { Complex32.One });
            for (int site = 0; site < n; site++)
            {
                var factor = (site == i || site == i + 1) ? QuantumSpin.PauliZ() : id;
                term = term.Kron(factor);
            }
            var scaled = term.Scale(-j);
            for (int r = 0; r < full.Rows; r++)
            {
                for (int c = 0; c < full.Cols; c++)
                {
                    full.Set(r, c, full.Get(r, c) + scaled.Get(r, c));
                }
            }
        }
        // The field: -h X_i.
        for (int i = 0; i < n; i++)
        {
            var term = ComplexMatrix.FromDiagonal(new[] { Complex32.One });
            for (int site = 0; site < n; site++)
            {
                var factor = site == i ? QuantumSpin.PauliX() : id;
                term = term.Kron(factor);
            }
            var scaled = term.Scale(-h);
            for (int r = 0; r < full.Rows; r++)
            {
                for (int c = 0; c < full.Cols; c++)
                {
                    full.Set(r, c, full.Get(r, c) + scaled.Get(r, c));
                }
            }
        }
        return full;
    }

    /// <summary>The entanglement spectrum across a cut: the eigenvalues of the reduced density matrix, largest first.</summary>
    /// <remarks>
    /// The truncation step of the density-matrix renormalisation
    /// group: the spectrum tells which states of the kept half
    /// matter, and the weight it drops is the error of keeping only a
    /// few.
    /// </remarks>
    public static float[] EntanglementSpectrum(QuantumState state, int[] keep, int totalQubits)
    {
        var rho = DensityMatrix.FromPure(state);
        float[] spectrum = rho.PartialTrace(keep, totalQubits).Eigenvalues();
        System.Array.Sort(spectrum);
        System.Array.Reverse(spectrum);
        return spectrum;
    }
}
