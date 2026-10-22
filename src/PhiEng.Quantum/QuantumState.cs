using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// A state in a finite-dimensional Hilbert space: a vector of complex
/// amplitudes.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::state::QuantumState</c>
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The state is <em>not</em> required to be normalised - a state that
/// has just come out of a solver may carry a scale - but everything
/// that reads a probability expects a normalised one.
/// </para>
/// </remarks>
public sealed class QuantumState
{
    private readonly Complex32[] _amps;

    /// <summary>The zero state of a space of <paramref name="dim"/> dimensions.</summary>
    public QuantumState(int dim)
    {
        _amps = new Complex32[dim];
    }

    /// <summary>The state with the given amplitudes.</summary>
    public QuantumState(Complex32[] amplitudes)
    {
        _amps = (Complex32[])amplitudes.Clone();
    }

    /// <summary>The <paramref name="index"/>-th basis state: one in that place, zero elsewhere.</summary>
    public static QuantumState Basis(int dim, int index)
    {
        var s = new QuantumState(dim);
        if (index >= 0 && index < dim)
        {
            s._amps[index] = Complex32.One;
        }
        return s;
    }

    /// <summary>The dimension of the space.</summary>
    public int Dim => _amps.Length;

    /// <summary>The amplitudes.</summary>
    public Complex32[] Amplitudes => _amps;

    /// <summary>The squared norm, the inner product of the state with itself.</summary>
    public float NormSqr()
    {
        float sum = 0.0f;
        foreach (var a in _amps)
        {
            sum += Complex32.NormSqr(a);
        }
        return sum;
    }

    /// <summary>The norm.</summary>
    public float Norm() => MathF.Sqrt(NormSqr());

    /// <summary>Scale the state to unit norm.</summary>
    /// <returns>
    /// False, changing nothing, when the state has no norm to scale by -
    /// a zero state cannot be normalised.
    /// </returns>
    public bool Normalize()
    {
        float n = Norm();
        if (!float.IsFinite(n) || n <= 0.0f)
        {
            return false;
        }
        float inv = 1.0f / n;
        for (int i = 0; i < _amps.Length; i++)
        {
            _amps[i] = Complex32.Scale(_amps[i], inv);
        }
        return true;
    }

    /// <summary>Whether the state is normalised, to within <paramref name="tol"/>.</summary>
    public bool IsNormalized(float tol) => MathF.Abs(NormSqr() - 1.0f) <= tol;

    /// <summary>The Born probabilities: the squared modulus of each amplitude.</summary>
    public float[] BornProbabilities()
    {
        var p = new float[_amps.Length];
        for (int i = 0; i < _amps.Length; i++)
        {
            p[i] = Complex32.NormSqr(_amps[i]);
        }
        return p;
    }

    /// <summary>The expectation value of an observable.</summary>
    public Complex32 Expectation(IQuantumOperator op)
    {
        int n = System.Math.Min(op.Dim, Dim);
        var applied = new Complex32[n];
        op.ApplyTo(_amps, applied);
        var sum = Complex32.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = sum + Complex32.Conjugate(_amps[i]) * applied[i];
        }
        return sum;
    }

    /// <summary>The overlap of two states.</summary>
    public Complex32 Overlap(QuantumState other)
    {
        int n = System.Math.Min(Dim, other.Dim);
        var sum = Complex32.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = sum + Complex32.Conjugate(_amps[i]) * other._amps[i];
        }
        return sum;
    }

    /// <summary>
    /// The fidelity: the squared magnitude of the overlap, the
    /// probability of measuring one state as the other.
    /// </summary>
    public float Fidelity(QuantumState other) => Complex32.NormSqr(Overlap(other));

    /// <summary>The phase that aligns this state with <paramref name="other"/>.</summary>
    public Complex32 PhaseTo(QuantumState other)
    {
        var o = other.Overlap(this);
        if (Complex32.IsZero(o))
        {
            return Complex32.One;
        }
        return Complex32.Inverse(o);
    }
}
