using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// A dense complex matrix, stored row by row.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::matrix::ComplexMatrix</c>
/// operation for operation, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The dense representation is for the <em>small</em> systems: a qubit
/// pair, a gate, a handful of basis states. The operators of a
/// wavefunction on a grid are never dense.
/// </para>
/// </remarks>
public sealed class ComplexMatrix
{
    private readonly Complex32[] _data;

    /// <summary>A matrix of <paramref name="rows"/> by <paramref name="cols"/> zeros.</summary>
    public ComplexMatrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        _data = new Complex32[rows * cols];
    }

    /// <summary>The <paramref name="n"/> by <paramref name="n"/> identity.</summary>
    public static ComplexMatrix Identity(int n)
    {
        var m = new ComplexMatrix(n, n);
        for (int i = 0; i < n; i++)
        {
            m.Set(i, i, Complex32.One);
        }
        return m;
    }

    /// <summary>The matrix with <paramref name="values"/> on its diagonal.</summary>
    public static ComplexMatrix FromDiagonal(Complex32[] values)
    {
        int n = values.Length;
        var m = new ComplexMatrix(n, n);
        for (int i = 0; i < n; i++)
        {
            m.Set(i, i, values[i]);
        }
        return m;
    }

    /// <summary>The number of rows.</summary>
    public int Rows { get; }

    /// <summary>The number of columns.</summary>
    public int Cols { get; }

    /// <summary>The matrix's entries, row by row.</summary>
    public Complex32[] Data => _data;

    /// <summary>The entry at (row, col).</summary>
    /// <remarks>
    /// Out-of-range indices return zero rather than throwing: a matrix is
    /// data, and a bad index should not take a simulation down.
    /// </remarks>
    public Complex32 Get(int row, int col)
    {
        if (row >= Rows || col >= Cols)
        {
            return Complex32.Zero;
        }
        return _data[row * Cols + col];
    }

    /// <summary>The entry at (row, col).</summary>
    /// <remarks>Out-of-range indices are ignored.</remarks>
    public void Set(int row, int col, Complex32 value)
    {
        if (row >= Rows || col >= Cols)
        {
            return;
        }
        _data[row * Cols + col] = value;
    }

    /// <summary><c>out = self * psi</c>.</summary>
    public void MatVec(Complex32[] psi, Complex32[] outState)
    {
        if (outState.Length < Rows)
        {
            return;
        }
        for (int r = 0; r < Rows; r++)
        {
            int rowStart = r * Cols;
            var sum = Complex32.Zero;
            for (int c = 0; c < Cols; c++)
            {
                sum = sum + _data[rowStart + c] * psi[c];
            }
            outState[r] = sum;
        }
    }

    /// <summary>The matrix product <c>self * other</c>.</summary>
    /// <remarks>
    /// The dimensions must agree; a mismatch returns an empty matrix
    /// rather than throwing.
    /// </remarks>
    public ComplexMatrix MatMul(ComplexMatrix other)
    {
        if (Cols != other.Rows)
        {
            return new ComplexMatrix(0, 0);
        }
        var result = new ComplexMatrix(Rows, other.Cols);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < other.Cols; c++)
            {
                var sum = Complex32.Zero;
                for (int k = 0; k < Cols; k++)
                {
                    sum = sum + Get(r, k) * other.Get(k, c);
                }
                result.Set(r, c, sum);
            }
        }
        return result;
    }

    /// <summary>The conjugate transpose, <c>self^dagger</c>.</summary>
    public ComplexMatrix Dagger()
    {
        var result = new ComplexMatrix(Cols, Rows);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                result.Set(c, r, Get(r, c).Conjugate());
            }
        }
        return result;
    }

    /// <summary>The trace.</summary>
    public Complex32 Trace()
    {
        int n = System.Math.Min(Rows, Cols);
        var sum = Complex32.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = sum + Get(i, i);
        }
        return sum;
    }

    /// <summary>The sum of every entry's squared modulus.</summary>
    public float FrobeniusSqr()
    {
        float sum = 0.0f;
        foreach (var v in _data)
        {
            sum += Complex32.NormSqr(v);
        }
        return sum;
    }

    /// <summary>Whether the matrix is Hermitian, to within <paramref name="tol"/>.</summary>
    public bool IsHermitian(float tol)
    {
        if (Rows != Cols)
        {
            return false;
        }
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                var a = Get(r, c);
                var b = Complex32.Conjugate(Get(c, r));
                if (MathF.Abs(a.Re - b.Re) > tol || MathF.Abs(a.Im - b.Im) > tol)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>The tensor product, <c>self (x) other</c>.</summary>
    public ComplexMatrix Kron(ComplexMatrix other)
    {
        var result = new ComplexMatrix(Rows * other.Rows, Cols * other.Cols);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                var a = Get(r, c);
                for (int r2 = 0; r2 < other.Rows; r2++)
                {
                    for (int c2 = 0; c2 < other.Cols; c2++)
                    {
                        result.Set(
                            r * other.Rows + r2,
                            c * other.Cols + c2,
                            a * other.Get(r2, c2));
                    }
                }
            }
        }
        return result;
    }

    /// <summary>The sum.</summary>
    public ComplexMatrix Add(ComplexMatrix other)
    {
        if (Rows != other.Rows || Cols != other.Cols)
        {
            return new ComplexMatrix(0, 0);
        }
        var result = new ComplexMatrix(Rows, Cols);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = _data[i] + other._data[i];
        }
        return result;
    }

    /// <summary>The difference.</summary>
    public ComplexMatrix Sub(ComplexMatrix other)
    {
        if (Rows != other.Rows || Cols != other.Cols)
        {
            return new ComplexMatrix(0, 0);
        }
        var result = new ComplexMatrix(Rows, Cols);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = _data[i] - other._data[i];
        }
        return result;
    }

    /// <summary>The product with a real number.</summary>
    public ComplexMatrix Scale(float s)
    {
        var result = new ComplexMatrix(Rows, Cols);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = Complex32.Scale(_data[i], s);
        }
        return result;
    }
}
