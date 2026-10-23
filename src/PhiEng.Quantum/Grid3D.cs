namespace PhiEng.Quantum;

/// <summary>
/// A uniform three-dimensional grid.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::grid3::Grid3D</c> operation
/// for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// Points are laid out with x fastest: the index of
/// <c>(ix, iy, iz)</c> is <c>ix + nx * (iy + ny * iz)</c>.
/// </para>
/// </remarks>
public sealed class Grid3D
{
    private readonly int _nx;
    private readonly int _ny;
    private readonly int _nz;
    private readonly float _xMin;
    private readonly float _xMax;
    private readonly float _yMin;
    private readonly float _yMax;
    private readonly float _zMin;
    private readonly float _zMax;

    /// <summary>A grid over the three intervals.</summary>
    /// <remarks>
    /// An axis with fewer than two points is clamped to two: a one-point
    /// axis has no spacing, and refusing to be built is better than
    /// dividing by zero later.
    /// </remarks>
    public Grid3D(
        int nx, int ny, int nz,
        float xMin, float yMin, float zMin,
        float xMax, float yMax, float zMax)
    {
        _nx = System.Math.Max(nx, 2);
        _ny = System.Math.Max(ny, 2);
        _nz = System.Math.Max(nz, 2);
        _xMin = xMin;
        _xMax = xMax;
        _yMin = yMin;
        _yMax = yMax;
        _zMin = zMin;
        _zMax = zMax;
    }

    /// <summary>The point counts along each axis.</summary>
    public (int Nx, int Ny, int Nz) Shape() => (_nx, _ny, _nz);

    /// <summary>The total number of points.</summary>
    public int N() => _nx * _ny * _nz;

    /// <summary>The spacings along each axis.</summary>
    public (float Dx, float Dy, float Dz) Spacings() => (
        (_xMax - _xMin) / (_nx - 1),
        (_yMax - _yMin) / (_ny - 1),
        (_zMax - _zMin) / (_nz - 1));

    /// <summary>The index of <c>(ix, iy, iz)</c>, or null outside the grid.</summary>
    public int? Index(int ix, int iy, int iz)
    {
        if (ix >= _nx || iy >= _ny || iz >= _nz)
        {
            return null;
        }
        return ix + _nx * (iy + _ny * iz);
    }

    /// <summary>The coordinates of an index: the inverse of <see cref="Index"/>.</summary>
    public (int Ix, int Iy, int Iz)? Coords(int index)
    {
        if (index >= N())
        {
            return null;
        }
        int ix = index % _nx;
        int iy = (index / _nx) % _ny;
        int iz = index / (_nx * _ny);
        return (ix, iy, iz);
    }

    /// <summary>The position of an index.</summary>
    /// <remarks>
    /// An index outside the grid returns the far corner.
    /// </remarks>
    public (float X, float Y, float Z) Position(int index)
    {
        var (dx, dy, dz) = Spacings();
        var c = Coords(index);
        if (c == null)
        {
            return (_xMax, _yMax, _zMax);
        }
        return (
            _xMin + c.Value.Ix * dx,
            _yMin + c.Value.Iy * dy,
            _zMin + c.Value.Iz * dz);
    }

    /// <summary>
    /// The number of interior unknowns under the Dirichlet condition:
    /// the ends are held at zero on every axis.
    /// </summary>
    public int DirichletUnknowns() => (_nx - 2) * (_ny - 2) * (_nz - 2);

    /// <summary>The grid point of the <paramref name="u"/>-th Dirichlet unknown.</summary>
    /// <remarks>
    /// Unknowns run in the same x-fastest order over the interior points
    /// of each axis. Null past the last one.
    /// </remarks>
    public (int Ix, int Iy, int Iz)? UnknownToGrid(int u)
    {
        int mx = _nx - 2;
        int my = _ny - 2;
        int mz = _nz - 2;
        if (u >= mx * my * mz)
        {
            return null;
        }
        int ix = u % mx;
        int iy = (u / mx) % my;
        int iz = u / (mx * my);
        return (ix + 1, iy + 1, iz + 1);
    }

    /// <summary>
    /// The unknown index of an interior grid point, or null on the
    /// boundary: the inverse of <see cref="UnknownToGrid"/>.
    /// </summary>
    public int? GridToUnknown(int ix, int iy, int iz)
    {
        if (ix == 0 || iy == 0 || iz == 0)
        {
            return null;
        }
        if (ix >= _nx - 1 || iy >= _ny - 1 || iz >= _nz - 1)
        {
            return null;
        }
        int mx = _nx - 2;
        int my = _ny - 2;
        return (ix - 1) + mx * ((iy - 1) + my * (iz - 1));
    }
}
