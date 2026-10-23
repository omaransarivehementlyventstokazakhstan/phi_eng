using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Tight-binding lattices: bands, gaps and edge modes.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::lattice</c> module entry
/// for entry, so the two engines agree bit-for-bit.
/// </remarks>
public static class QuantumLattice
{
    /// <summary>A chain of <paramref name="n"/> sites with uniform hopping <paramref name="hopping"/>.</summary>
    /// <remarks>
    /// Open ends by default; <paramref name="periodic"/> closes the
    /// ring. The diagonal is zero: the on-site energy is the zero of
    /// energy here.
    /// </remarks>
    public static ComplexMatrix TightBindingChain(int n, float hopping, bool periodic)
    {
        n = System.Math.Max(n, 2);
        var m = new ComplexMatrix(n, n);
        var t = new Complex32(-hopping, 0.0f);
        for (int i = 0; i < n - 1; i++)
        {
            m.Set(i, i + 1, t);
            m.Set(i + 1, i, t);
        }
        if (periodic && n >= 3)
        {
            m.Set(0, n - 1, t);
            m.Set(n - 1, 0, t);
        }
        return m;
    }

    /// <summary>An SSH chain of <paramref name="cells"/> unit cells: hopping <paramref name="v"/> inside the cell, <paramref name="w"/> between cells, open ends.</summary>
    /// <remarks>
    /// For <c>v &lt; w</c> the chain is topological and shelters two
    /// near-zero edge modes; for <c>v &gt; w</c> it is an ordinary
    /// gapped insulator.
    /// </remarks>
    public static ComplexMatrix SshChain(int cells, float v, float w)
    {
        cells = System.Math.Max(cells, 1);
        int n = 2 * cells;
        var m = new ComplexMatrix(n, n);
        var a = new Complex32(-v, 0.0f);
        var b = new Complex32(-w, 0.0f);
        for (int j = 0; j < cells; j++)
        {
            m.Set(2 * j, 2 * j + 1, a);
            m.Set(2 * j + 1, 2 * j, a);
        }
        for (int j = 0; j < cells - 1; j++)
        {
            m.Set(2 * j + 1, 2 * j + 2, b);
            m.Set(2 * j + 2, 2 * j + 1, b);
        }
        return m;
    }
}
