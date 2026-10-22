using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The boundary condition of a one-dimensional Hamiltonian.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::hamiltonian::Boundary1D</c>
/// enum.
/// </remarks>
public enum Boundary1D
{
    /// <summary>
    /// The wavefunction vanishes at both ends: the infinite well.
    /// </summary>
    /// <remarks>
    /// The unknowns are the <em>interior</em> points; the two ends are
    /// held at zero and are not part of the state.
    /// </remarks>
    Dirichlet,

    /// <summary>
    /// The wavefunction wraps around: the ring.
    /// </summary>
    /// <remarks>
    /// Every point is an unknown, and the stencil reaches around.
    /// </remarks>
    Periodic,
}

/// <summary>
/// The Hamiltonian of a particle on a line: a kinetic term and a
/// potential.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::hamiltonian::Hamiltonian1D</c>
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The kinetic term is the three-point finite difference
/// <c>-(hbar^2 / 2m) * (psi[i+1] - 2 psi[i] + psi[i-1]) / dx^2</c>, which
/// is second-order accurate in the spacing, and the whole operator is
/// <em>matrix-free</em>: it is applied as a stencil, so a thousand-point
/// Hamiltonian costs three operations per point and never materialises a
/// matrix.
/// </para>
/// </remarks>
public sealed class Hamiltonian1D : IQuantumOperator
{
    private readonly Grid1D _grid;
    private readonly float _mass;
    private readonly float _hbar;
    private readonly float[] _potential;
    private readonly Boundary1D _boundary;

    /// <summary>A Hamiltonian on <paramref name="grid"/> with the given potential.</summary>
    /// <remarks>
    /// The mass and the reduced Planck constant are one unless the
    /// caller says otherwise - the natural units of the field.
    /// </remarks>
    public Hamiltonian1D(Grid1D grid, Potential1D potential, float mass, float hbar)
    {
        _grid = grid;
        _mass = mass;
        _hbar = hbar;
        _potential = potential.Sample(grid);
        _boundary = Boundary1D.Dirichlet;
    }

    private Hamiltonian1D(
        Grid1D grid,
        float mass,
        float hbar,
        float[] potential,
        Boundary1D boundary)
    {
        _grid = grid;
        _mass = mass;
        _hbar = hbar;
        _potential = potential;
        _boundary = boundary;
    }

    /// <summary>The same Hamiltonian with a different boundary condition.</summary>
    public Hamiltonian1D WithBoundary(Boundary1D boundary) =>
        new(_grid, _mass, _hbar, _potential, boundary);

    /// <summary>The grid the Hamiltonian lives on.</summary>
    public Grid1D Grid => _grid;

    /// <summary>The potential sampled on the grid.</summary>
    public float[] Potential => _potential;

    /// <summary>The boundary condition.</summary>
    public Boundary1D Boundary => _boundary;

    /// <summary>The mass.</summary>
    public float Mass => _mass;

    /// <summary>The reduced Planck constant.</summary>
    public float Hbar => _hbar;

    /// <summary>The kinetic coefficient, <c>hbar^2 / (2 m dx^2)</c>.</summary>
    public float KineticCoefficient()
    {
        float dx = _grid.Dx();
        return _hbar * _hbar / (2.0f * _mass * dx * dx);
    }

    /// <summary>The number of unknowns.</summary>
    /// <remarks>
    /// The Dirichlet condition holds the two ends at zero, so they are
    /// not unknowns; the periodic condition makes every point one.
    /// </remarks>
    public int Unknowns()
    {
        switch (_boundary)
        {
            case Boundary1D.Dirichlet:
                return _grid.N < 2 ? 0 : _grid.N - 2;
            case Boundary1D.Periodic:
                return _grid.N;
            default:
                return 0;
        }
    }

    /// <summary>The diagonal of the Hamiltonian matrix.</summary>
    /// <remarks>
    /// The diagonal is <c>hbar^2 / (m dx^2) + V[i]</c>: the two kinetic
    /// terms of the stencil collected, plus the potential.
    /// <para>
    /// For the Dirichlet condition the unknowns are the interior points,
    /// so the entries are the interior's; for the periodic condition
    /// every point is an unknown.
    /// </para>
    /// </remarks>
    public float[] Diagonal()
    {
        float c = KineticCoefficient();
        int n = Unknowns();
        var d = new float[n];
        switch (_boundary)
        {
            case Boundary1D.Dirichlet:
                // The unknowns are the interior points 1..n-2.
                for (int i = 1; i < (_grid.N - 1); i++)
                {
                    d[i - 1] = 2.0f * c + _potential[i];
                }
                break;
            case Boundary1D.Periodic:
                for (int i = 0; i < n; i++)
                {
                    d[i] = 2.0f * c + _potential[i];
                }
                break;
            default:
                break;
        }
        return d;
    }

    /// <summary>The off-diagonal of the Hamiltonian matrix, <c>H[i][i+1]</c>.</summary>
    /// <remarks>
    /// The off-diagonal is <c>-c</c> everywhere. The periodic case also
    /// closes the ring with <c>H[n-1][0] = -c</c>, which makes the
    /// matrix cyclic rather than tridiagonal; the tridiagonal solvers
    /// here do not see that entry, so they solve the Dirichlet problem.
    /// </remarks>
    public float[] OffDiagonal()
    {
        float c = KineticCoefficient();
        int n = Unknowns();
        var e = new float[n < 1 ? 0 : n - 1];
        for (int i = 0; i < e.Length; i++)
        {
            e[i] = -c;
        }
        return e;
    }

    /// <summary>An energy at or above the top of the spectrum.</summary>
    /// <remarks>
    /// The Gershgorin bound: every eigenvalue of the matrix lies within
    /// one of the rows' discs, so the largest row's upper edge is an
    /// upper bound for the whole spectrum. It is the shift the
    /// imaginary-time relaxation needs.
    /// </remarks>
    public float SpectralUpperBound()
    {
        float c = KineticCoefficient();
        float[] d = Diagonal();
        float[] e = OffDiagonal();
        float upper = float.NegativeInfinity;
        for (int i = 0; i < d.Length; i++)
        {
            float radius = 0.0f;
            if (i > 0)
            {
                radius += i - 1 < e.Length ? MathF.Abs(e[i - 1]) : 0.0f;
            }
            if (i + 1 < d.Length)
            {
                radius += i < e.Length ? MathF.Abs(e[i]) : 0.0f;
            }
            float candidate = d[i] + radius;
            upper = candidate > upper ? candidate : upper;
        }
        return upper + c;
    }

    /// <summary>The lowest eigenvalues of the Hamiltonian.</summary>
    /// <remarks>
    /// The spectrum of the discrete Hamiltonian, by the Sturm bisection.
    /// Only the lowest <paramref name="count"/> are found: the whole
    /// spectrum of a thousand-point Hamiltonian costs a hundred times
    /// more, and the ground state needs the first two.
    /// </remarks>
    public float[] LowestEigenvalues(int count) =>
        QuantumLinalg.TridiagonalLowestEigenvalues(Diagonal(), OffDiagonal(), count);

    /// <summary>The ground state of the Hamiltonian and its energy.</summary>
    /// <remarks>
    /// The state is found by the inverse iteration, which needs a shift
    /// just below the ground state's energy; that shift comes from the
    /// spectrum, so the caller does not have to guess it.
    /// <para>
    /// The energy returned is the energy of the state the iteration
    /// produced.
    /// </para>
    /// </remarks>
    public (float Energy, QuantumState State) GroundState(int iterations, float tol)
    {
        float[] spectrum = LowestEigenvalues(1);
        float e0 = spectrum.Length > 0 ? spectrum[0] : 0.0f;
        QuantumState? state = QuantumLinalg.LowestEigenvector(
            Diagonal(), OffDiagonal(), e0, System.Math.Max(iterations, 1));
        if (state == null)
        {
            return (e0, InitialState());
        }

        // The energy of the state the iteration produced.
        var applied = new Complex32[Unknowns()];
        ApplyTo(state.Amplitudes, applied);
        var energy = Complex32.Zero;
        int pairs = System.Math.Min(state.Amplitudes.Length, applied.Length);
        for (int i = 0; i < pairs; i++)
        {
            energy = energy + Complex32.Conjugate(state.Amplitudes[i]) * applied[i];
        }
        _ = tol;
        return (energy.Re, state);
    }

    /// <summary>The state of the ground state: a real, positive wavefunction.</summary>
    /// <remarks>
    /// A constant one is a poor starting point for the inverse iteration
    /// - it has no overlap with the odd-parity states, and they would
    /// stay exactly zero forever. A gentle ramp from the middle has an
    /// overlap with everything, so every state can be found.
    /// </remarks>
    public QuantumState InitialState()
    {
        int n = Unknowns();
        var amps = new Complex32[n];
        for (int i = 0; i < n; i++)
        {
            float x = i / (float)(System.Math.Max(n, 2) - 1);
            amps[i] = new Complex32(x, 0.0f);
        }
        var s = new QuantumState(amps);
        s.Normalize();
        return s;
    }

    /// <inheritdoc/>
    public int Dim => Unknowns();

    /// <inheritdoc/>
    public void ApplyTo(Complex32[] psi, Complex32[] outState)
    {
        float c = KineticCoefficient();
        int n = Unknowns();
        if (psi.Length < n || outState.Length < n)
        {
            return;
        }
        switch (_boundary)
        {
            case Boundary1D.Dirichlet:
                // The unknowns are the interior points; the two ends are
                // held at zero and are not part of the state.
                for (int i = 0; i < n; i++)
                {
                    Complex32 left = i == 0 ? Complex32.Zero : psi[i - 1];
                    Complex32 right = i + 1 >= n ? Complex32.Zero : psi[i + 1];
                    Complex32 lap = right - Complex32.Scale(psi[i], 2.0f) + left;
                    // The unknown i is the grid point i+1.
                    outState[i] = Complex32.Scale(psi[i], _potential[i + 1]) - Complex32.Scale(lap, c);
                }
                break;
            case Boundary1D.Periodic:
                for (int i = 0; i < n; i++)
                {
                    Complex32 left = psi[(i + n - 1) % n];
                    Complex32 right = psi[(i + 1) % n];
                    Complex32 lap = right - Complex32.Scale(psi[i], 2.0f) + left;
                    outState[i] = Complex32.Scale(psi[i], _potential[i]) - Complex32.Scale(lap, c);
                }
                break;
            default:
                break;
        }
    }
}
