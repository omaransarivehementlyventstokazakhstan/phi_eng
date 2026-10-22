using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Something that acts on a quantum state.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::operator::QuantumOperator</c>
/// trait. The trait is <em>matrix-free</em> on purpose: the Hamiltonian
/// of a wavefunction on a grid is a banded operator, and materialising
/// it as a dense matrix would cost a million entries to compute what a
/// three-point stencil computes in three operations.
/// </para>
/// </remarks>
public interface IQuantumOperator
{
    /// <summary>The dimension of the space this operator acts on.</summary>
    int Dim { get; }

    /// <summary>
    /// Apply the operator to a state, into a separate buffer.
    /// </summary>
    /// <remarks>
    /// The application writes into a separate buffer rather than in
    /// place. A stencil operator reads its neighbours while it writes,
    /// so it cannot work in place without a copy - and a copy that
    /// allocates on every call is a per-frame allocation, which this
    /// engine does not have. The two-buffer form lets the caller own
    /// the scratch.
    /// </remarks>
    void ApplyTo(Complex32[] psi, Complex32[] outState);
}

/// <summary>
/// The operators that ship with the mirror: the dense one, the diagonal
/// one and the identity.
/// </summary>
public static class QuantumOperators
{
    /// <summary>A dense operator: a matrix, applied as a matrix.</summary>
    public static IQuantumOperator Dense(ComplexMatrix matrix) => new DenseOperator(matrix);

    /// <summary>A diagonal operator: a vector of eigenvalues.</summary>
    public static IQuantumOperator Diagonal(Complex32[] values) => new DiagonalOperator(values);

    /// <summary>The identity operator.</summary>
    public static IQuantumOperator Identity(int dim) => new IdentityOperator(dim);
}

/// <summary>A dense operator: a matrix, applied as a matrix.</summary>
public sealed class DenseOperator : IQuantumOperator
{
    private readonly ComplexMatrix _matrix;

    /// <summary>The operator of <paramref name="matrix"/>.</summary>
    public DenseOperator(ComplexMatrix matrix)
    {
        _matrix = matrix;
    }

    /// <summary>The operator's matrix.</summary>
    public ComplexMatrix Matrix => _matrix;

    /// <inheritdoc/>
    public int Dim => _matrix.Rows;

    /// <inheritdoc/>
    public void ApplyTo(Complex32[] psi, Complex32[] outState)
    {
        int n = System.Math.Min(_matrix.Rows, System.Math.Min(psi.Length, outState.Length));
        if (n == 0)
        {
            return;
        }
        _matrix.MatVec(psi, outState);
    }
}

/// <summary>A diagonal operator: a vector of eigenvalues.</summary>
public sealed class DiagonalOperator : IQuantumOperator
{
    private readonly Complex32[] _values;

    /// <summary>The operator with <paramref name="values"/> on its diagonal.</summary>
    public DiagonalOperator(Complex32[] values)
    {
        _values = values;
    }

    /// <summary>The operator's diagonal.</summary>
    public Complex32[] Values => _values;

    /// <inheritdoc/>
    public int Dim => _values.Length;

    /// <inheritdoc/>
    public void ApplyTo(Complex32[] psi, Complex32[] outState)
    {
        int n = System.Math.Min(_values.Length, System.Math.Min(psi.Length, outState.Length));
        for (int i = 0; i < n; i++)
        {
            outState[i] = _values[i] * psi[i];
        }
    }
}

/// <summary>The identity operator.</summary>
public sealed class IdentityOperator : IQuantumOperator
{
    /// <summary>The identity on a space of <paramref name="dim"/> dimensions.</summary>
    public IdentityOperator(int dim)
    {
        Dim = dim;
    }

    /// <inheritdoc/>
    public int Dim { get; }

    /// <inheritdoc/>
    public void ApplyTo(Complex32[] psi, Complex32[] outState)
    {
        int n = System.Math.Min(Dim, System.Math.Min(psi.Length, outState.Length));
        Array.Copy(psi, outState, n);
    }
}
