using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// A separable Hamiltonian on a <see cref="Grid3D"/>: three
/// one-dimensional Hamiltonians plus an optional non-separable remainder
/// sampled on the full grid.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::hamiltonian3::Hamiltonian3D</c>
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// Everything here assumes the Dirichlet condition on all three axes:
/// the ends are held at zero, so the unknowns are the interior points
/// and every spectrum below comes from the tridiagonal solvers.
/// </para>
/// </remarks>
public sealed class Hamiltonian3D
{
    private readonly Grid3D _grid;
    private readonly Hamiltonian1D _hx;
    private readonly Hamiltonian1D _hy;
    private readonly Hamiltonian1D _hz;
    private readonly float[] _nonseparable;

    private Hamiltonian3D(Grid3D grid, Hamiltonian1D hx, Hamiltonian1D hy, Hamiltonian1D hz)
    {
        _grid = grid;
        _hx = hx;
        _hy = hy;
        _hz = hz;
        _nonseparable = new float[grid.N()];
    }

    /// <summary>A separable Hamiltonian from three one-dimensional problems.</summary>
    /// <remarks>
    /// Null when the pieces do not fit: the grids must match the axes,
    /// the spacings must agree exactly, and all three conditions must be
    /// Dirichlet.
    /// </remarks>
    public static Hamiltonian3D? Separable(Grid3D grid, Hamiltonian1D hx, Hamiltonian1D hy, Hamiltonian1D hz)
    {
        var (nx, ny, nz) = grid.Shape();
        var (dx, dy, dz) = grid.Spacings();
        static bool Fits(Hamiltonian1D h, int n, float d) =>
            h.Grid.N == n && h.Grid.Dx() == d && h.Boundary == Boundary1D.Dirichlet;
        if (!Fits(hx, nx, dx) || !Fits(hy, ny, dy) || !Fits(hz, nz, dz))
        {
            return null;
        }
        return new Hamiltonian3D(grid, hx, hy, hz);
    }

    /// <summary>Adds a non-separable remainder, sampled on the full grid in x-fastest order.</summary>
    /// <remarks>
    /// False, changing nothing, when the slice is short.
    /// </remarks>
    public bool WithNonseparable(float[] values)
    {
        if (values.Length < _grid.N())
        {
            return false;
        }
        System.Array.Copy(values, _nonseparable, _grid.N());
        return true;
    }

    /// <summary>The grid.</summary>
    public Grid3D Grid => _grid;

    /// <summary>The number of unknowns: the interior points.</summary>
    public int Unknowns() => _grid.DirichletUnknowns();

    /// <summary>The lowest <paramref name="count"/> energies: sums of one-dimensional energies, lowest first.</summary>
    /// <remarks>
    /// Each axis contributes its lowest few levels (<c>count + 2</c>, at
    /// least six); every combination is an energy of the separable
    /// problem, and the lowest <paramref name="count"/> of them are
    /// returned. The non-separable remainder is not included.
    /// <para>
    /// The three axis energies are added smallest-first: float addition
    /// is commutative but not associative, so the permutations of one
    /// degenerate level would otherwise land on different last-bit
    /// roundings. Sorting the three summands makes every permutation of
    /// the same multiset bit-identical, which is what the physical
    /// permutation symmetry demands.
    /// </para>
    /// </remarks>
    public float[] SeparableEnergies(int count)
    {
        int take = System.Math.Min(
            _hx.Unknowns() > 0 ? _hx.Unknowns() : 1,
            System.Math.Max(6, count + 2));
        float[] ex = QuantumLinalg.TridiagonalLowestEigenvalues(_hx.Diagonal(), _hx.OffDiagonal(), take);
        float[] ey = QuantumLinalg.TridiagonalLowestEigenvalues(_hy.Diagonal(), _hy.OffDiagonal(), take);
        float[] ez = QuantumLinalg.TridiagonalLowestEigenvalues(_hz.Diagonal(), _hz.OffDiagonal(), take);
        var sums = new System.Collections.Generic.List<float>(ex.Length * ey.Length * ez.Length);
        foreach (float x in ex)
        {
            foreach (float y in ey)
            {
                foreach (float z in ez)
                {
                    float[] parts = { x, y, z };
                    System.Array.Sort(parts);
                    sums.Add((parts[0] + parts[1]) + parts[2]);
                }
            }
        }
        float[] all = sums.ToArray();
        System.Array.Sort(all);
        int keep = System.Math.Min(count, all.Length);
        float[] result = new float[keep];
        System.Array.Copy(all, result, keep);
        return result;
    }

    /// <summary>The ground state of the separable problem: the tensor product of the three one-dimensional ground states.</summary>
    /// <remarks>
    /// The unknown <c>u</c> at interior point <c>(ix, iy, iz)</c> carries
    /// <c>psix[ix-1] * psiy[iy-1] * psiz[iz-1]</c>. The energy is the sum
    /// of the three one-dimensional bisection energies, exactly as the
    /// Rust core reports it (not the Rayleigh quotient).
    /// </remarks>
    public (float Energy, QuantumState State) SeparableGroundState()
    {
        static (float, QuantumState) Ground(Hamiltonian1D h)
        {
            float[] spectrum = h.LowestEigenvalues(1);
            float e0 = spectrum.Length > 0 ? spectrum[0] : 0.0f;
            QuantumState? state = QuantumLinalg.LowestEigenvector(
                h.Diagonal(), h.OffDiagonal(), e0, 50);
            if (state == null)
            {
                var fallback = new QuantumState(h.Unknowns());
                for (int i = 0; i < fallback.Amplitudes.Length; i++)
                {
                    fallback.Amplitudes[i] = Complex32.One;
                }
                fallback.Normalize();
                state = fallback;
            }
            return (e0, state);
        }
        var (ex, sx) = Ground(_hx);
        var (ey, sy) = Ground(_hy);
        var (ez, sz) = Ground(_hz);
        int n = Unknowns();
        var amps = new Complex32[n];
        for (int u = 0; u < n; u++)
        {
            var g = _grid.UnknownToGrid(u);
            if (g == null)
            {
                continue;
            }
            amps[u] = sx.Amplitudes[g.Value.Ix - 1]
                * sy.Amplitudes[g.Value.Iy - 1]
                * sz.Amplitudes[g.Value.Iz - 1];
        }
        var result = new QuantumState(amps);
        result.Normalize();
        return (ex + ey + ez, result);
    }

    /// <summary>Applies the operator to a state, into a separate buffer.</summary>
    /// <remarks>
    /// Three one-dimensional stencils plus the sampled potentials. The
    /// ends read as zero. False for a short buffer.
    /// </remarks>
    public bool ApplyTo(Complex32[] psi, Complex32[] outState)
    {
        int n = Unknowns();
        if (psi.Length < n || outState.Length < n)
        {
            return false;
        }
        var (nx, ny, nz) = _grid.Shape();
        _ = nz;
        int mx = nx - 2;
        int my = ny - 2;
        int mz = _grid.Shape().Nz - 2;
        float cx = _hx.KineticCoefficient();
        float cy = _hy.KineticCoefficient();
        float cz = _hz.KineticCoefficient();
        float[] vx = _hx.Potential;
        float[] vy = _hy.Potential;
        float[] vz = _hz.Potential;
        for (int u = 0; u < n; u++)
        {
            int ix = u % mx;
            int iy = (u / mx) % my;
            int iz = u / (mx * my);
            Complex32 xm = ix == 0 ? Complex32.Zero : psi[u - 1];
            Complex32 xp = ix + 1 >= mx ? Complex32.Zero : psi[u + 1];
            Complex32 ym = iy == 0 ? Complex32.Zero : psi[u - mx];
            Complex32 yp = iy + 1 >= my ? Complex32.Zero : psi[u + mx];
            int mxy = mx * my;
            Complex32 zm = iz == 0 ? Complex32.Zero : psi[u - mxy];
            Complex32 zp = iz + 1 >= mz ? Complex32.Zero : psi[u + mxy];
            Complex32 centre = psi[u];
            Complex32 lap =
                Complex32.Scale(xm - Complex32.Scale(centre, 2.0f) + xp, cx)
                + Complex32.Scale(ym - Complex32.Scale(centre, 2.0f) + yp, cy)
                + Complex32.Scale(zm - Complex32.Scale(centre, 2.0f) + zp, cz);
            int gi = (ix + 1) + nx * ((iy + 1) + ny * (iz + 1));
            int clamped = gi < _nonseparable.Length - 1 ? gi : _nonseparable.Length - 1;
            float potential = vx[ix + 1] + vy[iy + 1] + vz[iz + 1] + _nonseparable[clamped];
            outState[u] = Complex32.Scale(centre, potential) - lap;
        }
        return true;
    }

    /// <summary>The dense matrix of the operator, for small grids.</summary>
    /// <remarks>
    /// Column <c>j</c> is the operator applied to the <c>j</c>-th basis
    /// state: exact, and exactly as expensive as it sounds.
    /// </remarks>
    public ComplexMatrix DenseMatrix()
    {
        int n = Unknowns();
        var m = new ComplexMatrix(n, n);
        if (n == 0)
        {
            return m;
        }
        var basis = new Complex32[n];
        var column = new Complex32[n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                basis[i] = Complex32.Zero;
            }
            basis[j] = Complex32.One;
            if (!ApplyTo(basis, column))
            {
                continue;
            }
            for (int i = 0; i < n; i++)
            {
                m.Set(i, j, column[i]);
            }
        }
        return m;
    }
}
