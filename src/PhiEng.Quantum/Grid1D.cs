namespace PhiEng.Quantum;

/// <summary>
/// A uniform grid of <c>n</c> points between <c>x_min</c> and <c>x_max</c>.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::grid::Grid1D</c> operation for
/// operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The spacing is <c>(x_max - x_min) / (n - 1)</c>: the two ends are
/// <em>points</em>, not the edges of a cell, so the grid includes both
/// boundary values.
/// </para>
/// </remarks>
public sealed class Grid1D
{
    private readonly int _n;
    private readonly float _xMin;
    private readonly float _xMax;

    /// <summary>A grid of <paramref name="n"/> points from <paramref name="xMin"/> to <paramref name="xMax"/>.</summary>
    /// <remarks>
    /// A grid of fewer than two points has no spacing and is clamped to
    /// two: a one-point grid is a degenerate case the solvers cannot use,
    /// and refusing to be built is better than dividing by zero later.
    /// </remarks>
    public Grid1D(int n, float xMin, float xMax)
    {
        _n = System.Math.Max(n, 2);
        _xMin = xMin;
        _xMax = xMax;
    }

    /// <summary>The number of points.</summary>
    public int N => _n;

    /// <summary>The left end.</summary>
    public float XMin => _xMin;

    /// <summary>The right end.</summary>
    public float XMax => _xMax;

    /// <summary>The spacing between the points.</summary>
    public float Dx() => (_xMax - _xMin) / (_n - 1);

    /// <summary>The position of the <paramref name="index"/>-th point.</summary>
    /// <remarks>
    /// An index outside the grid returns the nearest end. The Rust index
    /// is unsigned, so a negative index has no counterpart there; it
    /// returns the left end here, which is the same rule.
    /// </remarks>
    public float XAt(int index)
    {
        if (index < 0)
        {
            return _xMin;
        }
        if (index >= _n)
        {
            return _xMax;
        }
        return _xMin + index * Dx();
    }

    /// <summary>Every point's position.</summary>
    public float[] Positions()
    {
        var positions = new float[_n];
        for (int i = 0; i < _n; i++)
        {
            positions[i] = XAt(i);
        }
        return positions;
    }
}
