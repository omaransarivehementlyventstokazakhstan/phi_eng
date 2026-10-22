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
        int n = System.Math.Min(_state.Dim, output.Length);
        // The cumulative probability of the outcomes. The cumulative array
        // is as long as the state was when the field was built, so a state
        // put in by the caller can be longer: the Rust zips the two, and
        // the zip stops at the shorter one.
        int c = System.Math.Min(_cumulative.Length, n);
        float total = 0.0f;
        for (int i = 0; i < c; i++)
        {
            total += Complex32.NormSqr(_state.Amplitudes[i]);
            _cumulative[i] = total;
        }
        // The draws: the inverse-CDF method. Each draw is the first
        // outcome whose cumulative probability passes it.
        for (int i = 0; i < n; i++)
        {
            float u = _rng.NextFloat() * total;
            int index = 0;
            for (int j = 0; j < c; j++)
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
