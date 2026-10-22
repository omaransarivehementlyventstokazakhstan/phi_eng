using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// How a wavefunction is configured.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::field::QuantumConfig</c> struct
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The defaults are the ones a world starts with: the field is off, and
/// the grid is a small harmonic trap.
/// </para>
/// </remarks>
public sealed class QuantumConfig
{
    /// <summary>
    /// Whether the field evolves at all.
    /// </summary>
    /// <remarks>
    /// Off by default: a world that never asks for a wavefunction must
    /// behave exactly as it did before the field existed, down to the last
    /// bit of its trajectory.
    /// </remarks>
    public bool Enabled { get; set; }

    /// <summary>The number of grid points.</summary>
    public int GridSize { get; set; }

    /// <summary>The left end of the grid.</summary>
    public float DomainMin { get; set; }

    /// <summary>The right end of the grid.</summary>
    public float DomainMax { get; set; }

    /// <summary>The mass of the particle.</summary>
    public float Mass { get; set; }

    /// <summary>The reduced Planck constant.</summary>
    public float Hbar { get; set; }

    /// <summary>The frequency of the harmonic trap the particle sits in.</summary>
    public float Omega { get; set; }

    /// <summary>How the wavefunction is advanced.</summary>
    public EvolutionMethod Method { get; set; }

    /// <summary>The boundary condition.</summary>
    /// <remarks>
    /// The Crank-Nicolson method needs the Dirichlet condition and the
    /// split operator needs the periodic one; a mismatched pair is refused
    /// rather than silently wrong.
    /// </remarks>
    public Boundary1D Boundary { get; set; }

    /// <summary>The seed of the measurement stream.</summary>
    public ulong Seed { get; set; }

    /// <summary>The configuration a world starts with.</summary>
    /// <remarks>
    /// The field is off, the grid is 256 points on [-10, 10], the particle
    /// has unit mass and unit reduced Planck constant, the trap's
    /// frequency is one, the evolution is Crank-Nicolson under the
    /// Dirichlet condition, and the measurement stream is seeded with one.
    /// </remarks>
    public static QuantumConfig Default => new()
    {
        Enabled = false,
        GridSize = 256,
        DomainMin = -10.0f,
        DomainMax = 10.0f,
        Mass = 1.0f,
        Hbar = 1.0f,
        Omega = 1.0f,
        Method = EvolutionMethod.CrankNicolson,
        Boundary = Boundary1D.Dirichlet,
        Seed = 1,
    };
}

/// <summary>
/// A wavefunction on a line, attached to a world.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::field::QuantumField</c> struct
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The field owns the grid, the Hamiltonian, the state and the scratch the
/// evolution needs, so a step allocates nothing.
/// </para>
/// </remarks>
public sealed class QuantumField
{
    private QuantumConfig _config;
    private Grid1D _grid;
    private Hamiltonian1D _hamiltonian;
    private QuantumState _state;
    private EvolutionScratch _scratch;
    private DeterministicRng _rng;

    /// <summary>The cumulative probability of the measurement stream.</summary>
    /// <summary>
    /// The smallest positive normal <see cref="float"/>, 0x00800000.
    /// </summary>
    /// <remarks>
    /// This is Rust's <c>f32::MIN_POSITIVE</c>, not .NET's
    /// <see cref="float.Epsilon"/> (which is the smallest subnormal, a
    /// different number). The two languages must clamp to the same
    /// value.
    /// </remarks>
    private const float MinPositive = 1.17549435e-38f;

    private float[] _cumulative;

    /// <summary>The time the wavefunction has been evolved for.</summary>
    private float _time;

    /// <summary>A field with the given configuration.</summary>
    /// <remarks>
    /// The wavefunction starts in the ground state of the trap, so a field
    /// that is stepped without being excited does not move - a stationary
    /// state is the easiest thing to check.
    /// </remarks>
    public QuantumField(QuantumConfig config)
    {
        _config = config;
        _grid = new Grid1D(config.GridSize, config.DomainMin, config.DomainMax);
        _hamiltonian = new Hamiltonian1D(
            _grid,
            Potential1D.Harmonic(config.Omega),
            config.Mass,
            config.Hbar).WithBoundary(config.Boundary);
        _state = GroundStateOf(_hamiltonian);
        _scratch = new EvolutionScratch(_state.Dim);
        _rng = new DeterministicRng(config.Seed);
        _cumulative = new float[_state.Dim];
        _time = 0.0f;
    }

    /// <summary>A field with the given configuration.</summary>
    public static QuantumField New(QuantumConfig config) => new(config);

    /// <summary>A field with the default configuration.</summary>
    public static QuantumField New() => new(QuantumConfig.Default);

    /// <summary>
    /// The ground state of a Hamiltonian, or the ramp if the solver cannot
    /// find one.
    /// </summary>
    private static QuantumState GroundStateOf(Hamiltonian1D h)
    {
        float[] spectrum = h.LowestEigenvalues(1);
        float eigenvalue = spectrum.Length > 0 ? spectrum[0] : 0.0f;
        return QuantumLinalg.LowestEigenvector(
            h.Diagonal(), h.OffDiagonal(), eigenvalue, 50) ?? h.InitialState();
    }

    /// <summary>The field's configuration.</summary>
    public QuantumConfig Config => _config;

    /// <summary>Whether the field is on.</summary>
    public bool IsEnabled => _config.Enabled;

    /// <summary>The grid the wavefunction lives on.</summary>
    public Grid1D Grid => _grid;

    /// <summary>The wavefunction.</summary>
    /// <remarks>
    /// The state is the live one, not a copy: the amplitudes are the
    /// field's own.
    /// </remarks>
    public QuantumState State => _state;

    /// <summary>
    /// The wavefunction, mutable: a caller may put a state of its own in
    /// the field.
    /// </summary>
    /// <remarks>
    /// The state is the live one, and the setter replaces it outright - the
    /// Rust's <c>*field.state_mut() = state</c>.
    /// </remarks>
    public QuantumState StateMut
    {
        get => _state;
        set => _state = value;
    }

    /// <summary>The time the wavefunction has been evolved for.</summary>
    public float Time => _time;

    /// <summary>Sets the field's clock.</summary>
    /// <remarks>
    /// The clock is structural state - it tells how long the
    /// wavefunction has been evolved - so the snapshot loader restores
    /// it exactly rather than resetting it.
    /// </remarks>
    public void SetTime(float time)
    {
        _time = time;
    }

    /// <summary>The measurement stream's seed.</summary>
    public ulong Seed => _config.Seed;

    /// <summary>
    /// Replaces the configuration and rebuilds the field.
    /// </summary>
    /// <remarks>
    /// The wavefunction starts again in the ground state: a new grid is a
    /// new wavefunction.
    /// </remarks>
    public void Configure(QuantumConfig config)
    {
        _config = config;
        _grid = new Grid1D(config.GridSize, config.DomainMin, config.DomainMax);
        _hamiltonian = new Hamiltonian1D(
            _grid,
            Potential1D.Harmonic(config.Omega),
            config.Mass,
            config.Hbar).WithBoundary(config.Boundary);
        _state = GroundStateOf(_hamiltonian);
        _scratch = new EvolutionScratch(_state.Dim);
        _rng = new DeterministicRng(config.Seed);
        _cumulative = new float[_state.Dim];
        _time = 0.0f;
    }

    /// <summary>
    /// The probability of each outcome: the squared modulus of each
    /// amplitude.
    /// </summary>
    /// <param name="output">The destination of the probabilities.</param>
    /// <returns>
    /// The number of probabilities written: the state's dimension, or the
    /// output's length if that is shorter.
    /// </returns>
    /// <remarks>
    /// The probabilities of a normalised state sum to one.
    /// </remarks>
    public int Probabilities(float[] output)
    {
        int n = System.Math.Min(_state.Dim, output.Length);
        for (int i = 0; i < n; i++)
        {
            output[i] = Complex32.NormSqr(_state.Amplitudes[i]);
        }
        return n;
    }

    /// <summary>
    /// Draws outcomes from the Born rule.
    /// </summary>
    /// <param name="output">The destination of the draws.</param>
    /// <returns>
    /// The number of draws written: the state's dimension, or the output's
    /// length if that is shorter.
    /// </returns>
    /// <remarks>
    /// Each draw is an outcome with the state's probability, so the
    /// distribution of the draws is the probability distribution. The
    /// stream is seeded, so the same seed gives the same draws - a
    /// measurement can be replayed.
    /// </remarks>
    public int Sample(float[] output)
    {
        int dim = _state.Dim;
        int n = System.Math.Min(dim, output.Length);
        // The cumulative distribution is built over the whole state, not
        // the output: a short output asks for fewer draws, not a
        // narrower distribution. The Rust draws the same way, so the
        // streams agree draw for draw.
        if (dim == 0 || _cumulative.Length < dim)
        {
            return 0;
        }
        float total = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            total += Complex32.NormSqr(_state.Amplitudes[i]);
            _cumulative[i] = total;
        }
        if (!float.IsFinite(total) || total <= 0.0f)
        {
            return 0;
        }
        // The draws: the inverse-CDF method. Each draw is the first
        // outcome whose cumulative probability passes it.
        for (int i = 0; i < n; i++)
        {
            float u = _rng.NextFloat() * total;
            int index = 0;
            for (int j = 0; j < dim; j++)
            {
                if (u < _cumulative[j])
                {
                    index = j;
                    break;
                }
                index = j;
            }
            output[i] = index;
        }
        return n;
    }

    /// <summary>
    /// Draws a position on the grid from the Born rule.
    /// </summary>
    /// <returns>
    /// The grid point the drawn outcome names, or null when there is
    /// nothing to draw. For the Dirichlet condition unknown
    /// <c>i</c> is grid point <c>i + 1</c> (the ends are held at zero
    /// and never drawn); for the periodic condition it is grid point
    /// <c>i</c>.
    /// </returns>
    public float? SamplePosition()
    {
        int? index = DrawIndex();
        if (index == null)
        {
            return null;
        }
        return GridPosition(index.Value);
    }

    /// <summary>The grid position an unknown index names.</summary>
    private float GridPosition(int index)
    {
        return _config.Boundary == Boundary1D.Dirichlet
            ? _grid.XAt(index + 1)
            : _grid.XAt(index);
    }

    /// <summary>Draws one outcome index from the Born rule.</summary>
    private int? DrawIndex()
    {
        int n = _state.Dim;
        if (n == 0 || _cumulative.Length < n)
        {
            return null;
        }
        float total = 0.0f;
        for (int i = 0; i < n; i++)
        {
            total += Complex32.NormSqr(_state.Amplitudes[i]);
            _cumulative[i] = total;
        }
        if (!float.IsFinite(total) || total <= 0.0f)
        {
            return null;
        }
        float u = _rng.NextFloat() * total;
        int found = 0;
        for (int i = 0; i < n; i++)
        {
            if (u < _cumulative[i])
            {
                found = i;
                break;
            }
            found = i;
        }
        return found;
    }

    /// <summary>The unknown index nearest to a position, clamped to the state.</summary>
    private int? NearestIndex(float x)
    {
        int n = _state.Dim;
        if (n == 0 || !float.IsFinite(x))
        {
            return null;
        }
        float dx = _grid.Dx();
        if (!float.IsFinite(dx) || dx <= 0.0f)
        {
            return null;
        }
        float q = (x - _grid.XMin) / dx;
        // A quotient this large is beyond any grid anyone will build;
        // clamping keeps the float-to-integer cast below in range, where
        // the two engines disagree on out-of-range casts.
        long p = (long)MathF.Floor(System.Math.Clamp(q, -9.0e15f, 9.0e15f) + 0.5f);
        int point = _config.Boundary == Boundary1D.Dirichlet
            // The unknowns are the interior points 1..=n.
            ? (int)System.Math.Clamp(p, 1L, n) - 1
            : (int)(((p % n) + n) % n);
        return System.Math.Min(point, n - 1);
    }

    /// <summary>
    /// The Bohmian velocity at a position: v = J / rho with the
    /// probability current J = (hbar / m) Im(psi* dpsi/dx).
    /// </summary>
    /// <remarks>
    /// A stationary state has no current anywhere, so it reports zero;
    /// a state with a phase gradient reports the flow. Central
    /// differences on the three-point stencil around the nearest grid
    /// point; where the density is at the floor the velocity is zero
    /// rather than infinite.
    /// </remarks>
    public float VelocityAt(float x)
    {
        int? maybe = NearestIndex(x);
        if (maybe == null)
        {
            return 0.0f;
        }
        int i = maybe.Value;
        int n = _state.Dim;
        var amps = _state.Amplitudes;
        Complex32 At(int k)
        {
            if (_config.Boundary == Boundary1D.Dirichlet)
            {
                return k >= n ? Complex32.Zero : amps[k];
            }
            return amps[k % n];
        }
        Complex32 left = i == 0
            ? (_config.Boundary == Boundary1D.Dirichlet ? Complex32.Zero : At(n - 1))
            : At(i - 1);
        Complex32 centre = At(i);
        Complex32 right = i + 1 >= n
            ? (_config.Boundary == Boundary1D.Dirichlet ? Complex32.Zero : At(0))
            : At(i + 1);
        float rho = Complex32.NormSqr(centre);
        if (rho <= Madelung.DensityFloor)
        {
            return 0.0f;
        }
        float dx = _grid.Dx();
        var deriv = (right - left) * (1.0f / (2.0f * dx));
        float current = _config.Hbar / MathF.Max(_config.Mass, MinPositive)
            * (Complex32.Conjugate(centre) * deriv).Im;
        return current / rho;
    }

    /// <summary>
    /// The force a classical particle feels at a position:
    /// F = -d(V + Q)/dx, the classical potential plus the quantum
    /// potential.
    /// </summary>
    /// <remarks>
    /// The quantum potential is evaluated point by point from the
    /// wavefunction's density, so no buffer is needed; where the
    /// density is at the floor the quantum term is zero. This is the
    /// Bohmian force: a game that integrates it moves bodies the way
    /// the wavefunction pushes.
    /// </remarks>
    public float ForceAt(float x)
    {
        int? maybe = NearestIndex(x);
        if (maybe == null)
        {
            return 0.0f;
        }
        int i = maybe.Value;
        int n = _state.Dim;
        float dx = _grid.Dx();
        // The density and the classical potential on the five-point
        // footprint the central difference of Q needs.
        int lo = System.Math.Max(0, i - 2);
        var rho = new float[5];
        var pot = new float[5];
        for (int k = 0; k < 5; k++)
        {
            (rho[k], pot[k]) = DensityAndPotentialAt(lo + k, n);
        }
        // The quantum potential at the three central points.
        var q = new float[3];
        for (int k = 0; k < 3; k++)
        {
            q[k] = PointQuantumPotential(rho[k], rho[k + 1], rho[k + 2], dx, _config.Hbar, _config.Mass);
        }
        // The classical potential at the three central points.
        var v = new float[3];
        for (int k = 0; k < 3; k++)
        {
            v[k] = PointPotential(lo + 1 + k, n);
        }
        return -((v[2] + q[2]) - (v[0] + q[0])) / (2.0f * dx);
    }

    /// <summary>
    /// The density and the classical potential at unknown <paramref name="i"/>.
    /// </summary>
    /// <remarks>
    /// Out-of-range indices are vacuum with the potential of the
    /// nearest end for the Dirichlet condition, and wrap for the
    /// periodic one.
    /// </remarks>
    private (float Rho, float Pot) DensityAndPotentialAt(int i, int n)
    {
        if (_config.Boundary == Boundary1D.Dirichlet)
        {
            if (i >= n)
            {
                return (0.0f, PointPotential(System.Math.Max(0, n - 1), n));
            }
            return (Complex32.NormSqr(_state.Amplitudes[i]), PointPotential(i, n));
        }
        int k = i % n;
        return (Complex32.NormSqr(_state.Amplitudes[k]), PointPotential(k, n));
    }

    /// <summary>The classical potential at unknown <paramref name="i"/>, clamped to the state.</summary>
    private float PointPotential(int i, int n)
    {
        float[] potential = _hamiltonian.Potential;
        if (n == 0 || potential.Length == 0)
        {
            return 0.0f;
        }
        int k = System.Math.Min(i, n - 1);
        k = System.Math.Min(k, potential.Length - 1);
        if (_config.Boundary == Boundary1D.Dirichlet)
        {
            return k + 1 < potential.Length ? potential[k + 1] : 0.0f;
        }
        return potential[k];
    }

    /// <summary>The quantum potential of three consecutive densities.</summary>
    private static float PointQuantumPotential(
        float s0, float s1, float s2, float dx, float hbar, float mass)
    {
        float centre = MathF.Sqrt(MathF.Max(s1, 0.0f));
        if (centre <= Madelung.DensityFloor)
        {
            return 0.0f;
        }
        float left = MathF.Sqrt(MathF.Max(s0, 0.0f));
        float right = MathF.Sqrt(MathF.Max(s2, 0.0f));
        float lap = (left - centre * 2.0f + right) / (dx * dx);
        return -(hbar * hbar) / (2.0f * MathF.Max(mass, MinPositive)) * lap / centre;
    }

    /// <summary>
    /// The density on the grid: the squared modulus of each amplitude.
    /// </summary>
    /// <param name="output">The destination of the density.</param>
    /// <returns>The number of entries written.</returns>
    public int Density(float[] output)
    {
        int n = System.Math.Min(_state.Dim, output.Length);
        for (int i = 0; i < n; i++)
        {
            output[i] = Complex32.NormSqr(_state.Amplitudes[i]);
        }
        return n;
    }

    /// <summary>
    /// The phase on the grid: the argument of each amplitude, in
    /// (-pi, pi], from the deterministic arctangent.
    /// </summary>
    /// <param name="output">The destination of the phase.</param>
    /// <returns>The number of entries written.</returns>
    public int Phase(float[] output)
    {
        int n = System.Math.Min(_state.Dim, output.Length);
        for (int i = 0; i < n; i++)
        {
            output[i] = DeterministicTrig.Atan2(_state.Amplitudes[i].Im, _state.Amplitudes[i].Re);
        }
        return n;
    }

    /// <summary>
    /// Advances the wavefunction by one step.
    /// </summary>
    /// <param name="dt">The step.</param>
    /// <returns>
    /// False when the configuration and the method disagree - the
    /// Crank-Nicolson method needs the Dirichlet condition and the split
    /// operator the periodic one - in which case the wavefunction is left
    /// alone. A disabled field is not an error: the step succeeds and
    /// nothing moves.
    /// </returns>
    public bool Step(float dt)
    {
        if (!_config.Enabled)
        {
            return true;
        }
        bool advanced;
        switch (_config.Method)
        {
            case EvolutionMethod.CrankNicolson:
                advanced = Evolution.CrankNicolson(_hamiltonian, _state, dt, _scratch);
                break;
            case EvolutionMethod.SplitOperator:
                advanced = Evolution.SplitOperator(_hamiltonian, _state, dt, _scratch);
                break;
            default:
                advanced = false;
                break;
        }
        if (advanced)
        {
            _time += dt;
        }
        return advanced;
    }

    /// <summary>
    /// The energy of the wavefunction: the expectation of the Hamiltonian.
    /// </summary>
    public float Energy()
    {
        int n = _state.Dim;
        var applied = new Complex32[n];
        _hamiltonian.ApplyTo(_state.Amplitudes, applied);
        var sum = Complex32.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = sum + Complex32.Conjugate(_state.Amplitudes[i]) * applied[i];
        }
        return sum.Re;
    }
}
