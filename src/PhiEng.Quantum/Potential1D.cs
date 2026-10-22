namespace PhiEng.Quantum;

/// <summary>The kind of a one-dimensional potential.</summary>
/// <remarks>
/// This mirrors the variants of the Rust
/// <c>phieng_quantum::potential::Potential1D</c> enum.
/// </remarks>
public enum PotentialKind
{
    /// <summary>No potential at all: the free particle.</summary>
    Zero,

    /// <summary>A constant everywhere.</summary>
    Constant,

    /// <summary>The harmonic trap, half m omega squared x squared.</summary>
    Harmonic,

    /// <summary>A square well: a flat bottom of <c>depth</c> between the walls.</summary>
    Box,

    /// <summary>A step: <c>height</c> on the right of <c>position</c>, zero on the left.</summary>
    Step,

    /// <summary>Two wells separated by a barrier: the tunnelling system.</summary>
    DoubleWell,

    /// <summary>The soft Coulomb potential, minus k over the root of x squared plus a squared.</summary>
    SoftCoulomb,

    /// <summary>A potential given point by point.</summary>
    Custom,
}

/// <summary>
/// The potential energy along a line.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::potential::Potential1D</c>
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The units are the engine's: the mass and the reduced Planck constant
/// are one unless the caller says otherwise, so the energies come out in
/// the same units as the grid's length.
/// </para>
/// </remarks>
public sealed class Potential1D
{
    /// <summary>The kind of the potential.</summary>
    public PotentialKind Kind { get; init; }

    /// <summary>The constant value, for the constant potential.</summary>
    public float Value { get; init; }

    /// <summary>The angular frequency, for the harmonic trap.</summary>
    public float Omega { get; init; }

    /// <summary>The centre of the well, for the box and the double well.</summary>
    public float Center { get; init; }

    /// <summary>Half the width of the well, for the box.</summary>
    public float HalfWidth { get; init; }

    /// <summary>The depth of the well, for the box and the double well.</summary>
    public float Depth { get; init; }

    /// <summary>Where the step is, for the step potential.</summary>
    public float Position { get; init; }

    /// <summary>The height of the step, for the step potential.</summary>
    public float Height { get; init; }

    /// <summary>Half the distance between the two wells, for the double well.</summary>
    public float Separation { get; init; }

    /// <summary>The height of the barrier, for the double well.</summary>
    public float Barrier { get; init; }

    /// <summary>The strength of the attraction, for the soft Coulomb potential.</summary>
    public float Strength { get; init; }

    /// <summary>The softening length, for the soft Coulomb potential.</summary>
    public float Softening { get; init; }

    /// <summary>The point-by-point values, for the custom potential.</summary>
    public float[] CustomValues { get; init; } = Array.Empty<float>();

    /// <summary>No potential at all: the free particle.</summary>
    public static Potential1D Zero => new() { Kind = PotentialKind.Zero };

    /// <summary>A constant everywhere.</summary>
    public static Potential1D Constant(float value) =>
        new() { Kind = PotentialKind.Constant, Value = value };

    /// <summary>The harmonic trap, half m omega squared x squared, centred on the origin.</summary>
    /// <remarks>
    /// The energies of a harmonic oscillator on an infinite line are
    /// <c>(n + 1/2) hbar omega</c>.
    /// </remarks>
    public static Potential1D Harmonic(float omega) =>
        new() { Kind = PotentialKind.Harmonic, Omega = omega };

    /// <summary>A square well: a flat bottom of <paramref name="depth"/> between the walls.</summary>
    public static Potential1D Box(float center, float halfWidth, float depth) =>
        new()
        {
            Kind = PotentialKind.Box,
            Center = center,
            HalfWidth = halfWidth,
            Depth = depth,
        };

    /// <summary>A step: <paramref name="height"/> on the right of <paramref name="position"/>, zero on the left.</summary>
    public static Potential1D Step(float position, float height) =>
        new() { Kind = PotentialKind.Step, Position = position, Height = height };

    /// <summary>Two wells separated by a barrier: the tunnelling system.</summary>
    public static Potential1D DoubleWell(float center, float separation, float depth, float barrier) =>
        new()
        {
            Kind = PotentialKind.DoubleWell,
            Center = center,
            Separation = separation,
            Depth = depth,
            Barrier = barrier,
        };

    /// <summary>The soft Coulomb potential, minus k over the root of x squared plus a squared.</summary>
    /// <remarks>
    /// The softening keeps the potential finite at the origin, so the
    /// discretisation never sees a singularity. It is the standard way to
    /// make the hydrogen atom solvable on a grid.
    /// </remarks>
    public static Potential1D SoftCoulomb(float strength, float softening) =>
        new()
        {
            Kind = PotentialKind.SoftCoulomb,
            Strength = strength,
            Softening = softening,
        };

    /// <summary>A potential given point by point.</summary>
    /// <remarks>
    /// The values are taken at the grid's points; a shorter list is
    /// padded with zeros, a longer one is truncated.
    /// </remarks>
    public static Potential1D Custom(float[] values) =>
        new() { Kind = PotentialKind.Custom, CustomValues = values };

    /// <summary>The potential energy at <paramref name="x"/>.</summary>
    public float ValueAt(float x)
    {
        switch (Kind)
        {
            case PotentialKind.Zero:
                return 0.0f;
            case PotentialKind.Constant:
                return Value;
            case PotentialKind.Harmonic:
                return 0.5f * Omega * Omega * x * x;
            case PotentialKind.Box:
                return MathF.Abs(x - Center) <= HalfWidth ? -Depth : 0.0f;
            case PotentialKind.Step:
                return x >= Position ? Height : 0.0f;
            case PotentialKind.DoubleWell:
            {
                // Two wells of half-width `separation / 2`, with a
                // barrier of the same width between them.
                float half = Separation * 0.5f;
                bool inWell = MathF.Abs(x - (Center - Separation)) <= half
                    || MathF.Abs(x - (Center + Separation)) <= half;
                if (inWell)
                {
                    return -Depth;
                }
                if (MathF.Abs(x - Center) <= half)
                {
                    return Barrier;
                }
                return 0.0f;
            }
            case PotentialKind.SoftCoulomb:
                return -Strength / MathF.Sqrt(x * x + Softening * Softening);
            case PotentialKind.Custom:
                // The custom potential is sampled, not evaluated: its
                // values are indexed by the grid's points, so there is no
                // way to answer for an arbitrary `x`. Use Sample for it.
                return 0.0f;
            default:
                return 0.0f;
        }
    }

    /// <summary>The potential sampled on a grid.</summary>
    /// <remarks>
    /// The custom potential is the exception: its values are already
    /// sampled, and they are used as they are.
    /// </remarks>
    public float[] Sample(Grid1D grid)
    {
        if (Kind == PotentialKind.Custom)
        {
            var custom = new float[grid.N];
            int copied = System.Math.Min(CustomValues.Length, grid.N);
            Array.Copy(CustomValues, custom, copied);
            return custom;
        }
        var values = new float[grid.N];
        for (int i = 0; i < grid.N; i++)
        {
            values[i] = ValueAt(grid.XAt(i));
        }
        return values;
    }
}
