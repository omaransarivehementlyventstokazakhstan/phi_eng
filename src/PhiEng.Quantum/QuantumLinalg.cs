using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The linear algebra a quantum simulation needs: the Sturm count, the
/// tridiagonal eigensolver, the tridiagonal solves and the inverse
/// iteration.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::linalg</c> module operation
/// for operation, so the two engines agree bit-for-bit.
/// </para>
/// </remarks>
public static partial class QuantumLinalg
{
    /// <summary>
    /// The number of eigenvalues of a symmetric tridiagonal matrix below
    /// <paramref name="x"/>.
    /// </summary>
    /// <remarks>
    /// This is the Sturm count: the <c>LDL^T</c> factorisation of
    /// <c>T - xI</c> has one negative pivot for each eigenvalue below
    /// <paramref name="x"/>, and the pivots can be counted without
    /// factorising the matrix.
    /// <para>
    /// It is the workhorse of the tridiagonal eigensolver: a bisection on
    /// the count finds every eigenvalue of a banded Hamiltonian without
    /// ever building a dense matrix.
    /// </para>
    /// </remarks>
    public static int SturmCount(float[] diagonal, float[] offDiagonal, float x)
    {
        int n = diagonal.Length;
        if (n == 0)
        {
            return 0;
        }
        // The pivots of the factorisation span many orders of magnitude:
        // the Hamiltonian of a thousand-point grid is nearly singular, and
        // its pivots run from a billion down to a billionth. In `f32` that
        // overflows and the count turns to noise in the middle of the
        // spectrum, so the count is carried in `f64`.
        //
        // The `f64` is scratch only - the query point is an `f32` and the
        // answer is a count - and the arithmetic is exact and the same in
        // both languages, so the two engines still agree bit for bit.
        double xd = x;
        int count = 0;
        double pivot = diagonal[0] - xd;
        if (pivot == 0.0)
        {
            // A zero pivot would divide by zero below; the convention is to
            // treat it as a tiny negative one, which keeps the count
            // monotone. Without this the very first pivot of a spectrum
            // centred on the query point poisons every count.
            pivot = -1e-30;
        }
        if (pivot < 0.0)
        {
            count += 1;
        }
        for (int i = 1; i < n; i++)
        {
            if (pivot == 0.0)
            {
                pivot = -1e-30;
            }
            float ef = i - 1 < offDiagonal.Length ? offDiagonal[i - 1] : 0.0f;
            double e = ef;
            pivot = diagonal[i] - xd - e * e / pivot;
            if (pivot < 0.0)
            {
                count += 1;
            }
        }
        return count;
    }

    /// <summary>
    /// The eigenvalues of a symmetric tridiagonal matrix, in ascending
    /// order.
    /// </summary>
    /// <remarks>
    /// The matrix is given by its diagonal <paramref name="diagonal"/> and
    /// its off-diagonal <paramref name="offDiagonal"/>, where
    /// <c>offDiagonal[i]</c> is the entry <c>H[i][i+1]</c>. The
    /// eigenvalues are found by bisecting the Sturm count: the count jumps
    /// by one at each eigenvalue, so a bisection on the count pins every
    /// one of them.
    /// <para>
    /// The bisection runs a fixed number of steps, so the answer is
    /// reproducible to the last bit: the same matrix gives the same
    /// eigenvalues on every platform and in both languages.
    /// </para>
    /// <para>
    /// A matrix whose off-diagonal is shorter than
    /// <c>diagonal.Length - 1</c> is read as if the missing entries were
    /// zero.
    /// </para>
    /// </remarks>
    public static float[] TridiagonalEigenvalues(float[] diagonal, float[] offDiagonal) =>
        TridiagonalLowestEigenvalues(diagonal, offDiagonal, diagonal.Length);

    /// <summary>
    /// The lowest <paramref name="count"/> eigenvalues of a symmetric
    /// tridiagonal matrix, in ascending order.
    /// </summary>
    /// <remarks>
    /// The same bisection as <see cref="TridiagonalEigenvalues"/>, but it
    /// stops after the lowest <paramref name="count"/> - the ground state
    /// needs the first two, and the whole spectrum of a thousand-point
    /// Hamiltonian costs a hundred times more.
    /// </remarks>
    public static float[] TridiagonalLowestEigenvalues(
        float[] diagonal, float[] offDiagonal, int count)
    {
        int n = diagonal.Length;
        if (n == 0 || count == 0)
        {
            return Array.Empty<float>();
        }
        // Gershgorin's circle bounds the spectrum: every eigenvalue lies
        // within one of the rows' discs.
        float lower = float.PositiveInfinity;
        float upper = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            float radius = 0.0f;
            if (i > 0)
            {
                radius += i - 1 < offDiagonal.Length ? MathF.Abs(offDiagonal[i - 1]) : 0.0f;
            }
            if (i + 1 < n)
            {
                radius += i < offDiagonal.Length ? MathF.Abs(offDiagonal[i]) : 0.0f;
            }
            float candidate = diagonal[i] - radius;
            lower = candidate < lower ? candidate : lower;
            candidate = diagonal[i] + radius;
            upper = candidate > upper ? candidate : upper;
        }
        // The bounds are exclusive: the smallest eigenvalue is *above* the
        // lower bound and the largest is *below* the upper one.
        lower -= 1.0f;
        upper += 1.0f;

        int wanted = System.Math.Min(count, n);
        var values = new float[wanted];
        for (int k = 0; k < wanted; k++)
        {
            float lo = lower;
            float hi = upper;
            for (int step = 0; step < 100; step++)
            {
                // Half of each, so the sum cannot overflow.
                float mid = 0.5f * lo + 0.5f * hi;
                if (SturmCount(diagonal, offDiagonal, mid) <= k)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            values[k] = 0.5f * lo + 0.5f * hi;
        }
        return values;
    }

    /// <summary>
    /// Solves the tridiagonal system <c>A x = b</c> by the Thomas
    /// algorithm.
    /// </summary>
    /// <remarks>
    /// <c>A</c> is given by its diagonal <paramref name="diagonal"/> and
    /// its off-diagonal <paramref name="offDiagonal"/>, where
    /// <c>offDiagonal[i]</c> is the entry <c>A[i][i+1]</c>. The system is
    /// factored in place conceptually: the elimination is a forward sweep
    /// and the solution a back substitution.
    /// <para>
    /// The algorithm has no pivoting, so it needs a diagonally dominant
    /// matrix - which the Hamiltonian's <c>H - shift I</c> is, when the
    /// shift is near the bottom of the spectrum.
    /// </para>
    /// <para>
    /// Returns false when the system is singular (a zero pivot) or the
    /// buffers are too short; <paramref name="outState"/> is then
    /// undefined.
    /// </para>
    /// <para>
    /// This matches the Rust <c>tridiagonal_solve_with</c> operation for
    /// operation - the same multipliers, the same pivot, the same back
    /// substitution - so the two engines agree bit-for-bit.
    /// </para>
    /// </remarks>
    public static bool TridiagonalSolve(
        float[] diagonal, float[] offDiagonal, float[] rhs, float[] outState)
    {
        int n = diagonal.Length;
        if (n == 0 || rhs.Length < n || outState.Length < n)
        {
            return false;
        }
        return TridiagonalSolve(diagonal, offDiagonal, rhs, outState, new float[n]);
    }

    /// <summary>
    /// The Thomas algorithm with the caller's scratch buffer for the
    /// elimination multipliers.
    /// </summary>
    /// <remarks>
    /// <paramref name="multipliers"/> must hold at least
    /// <c>diagonal.Length</c> entries; it is the scratch the forward
    /// sweep writes and the back substitution reads.
    /// </remarks>
    internal static bool TridiagonalSolve(
        float[] diagonal, float[] offDiagonal, float[] rhs, float[] outState, float[] multipliers)
    {
        int n = diagonal.Length;
        if (n == 0 || rhs.Length < n || outState.Length < n || multipliers.Length < n)
        {
            return false;
        }
        // The forward sweep, in place in `out`.
        float pivot = diagonal[0];
        if (pivot == 0.0f || !float.IsFinite(pivot))
        {
            return false;
        }
        outState[0] = rhs[0] / pivot;
        if (!float.IsFinite(outState[0]))
        {
            return false;
        }
        for (int i = 1; i < n; i++)
        {
            float e = i - 1 < offDiagonal.Length ? offDiagonal[i - 1] : 0.0f;
            // The elimination multiplier and the modified pivot.
            float multiplier = e / pivot;
            multipliers[i] = multiplier;
            pivot = diagonal[i] - e * multiplier;
            if (pivot == 0.0f || !float.IsFinite(pivot))
            {
                return false;
            }
            outState[i] = (rhs[i] - e * outState[i - 1]) / pivot;
            if (!float.IsFinite(outState[i]))
            {
                return false;
            }
        }
        // The back substitution.
        for (int i = n - 2; i >= 0; i--)
        {
            outState[i] = outState[i] - multipliers[i + 1] * outState[i + 1];
        }
        return true;
    }

    /// <summary>
    /// Solves the tridiagonal system <c>A x = b</c> by the Thomas
    /// algorithm, with a complex diagonal.
    /// </summary>
    /// <remarks>
    /// This is the form the time evolution needs: the Crank-Nicolson
    /// matrix has a complex diagonal and a real off-diagonal.
    /// <para>
    /// Returns false when the system is singular (a zero pivot) or the
    /// buffers are too short; <paramref name="outState"/> is then
    /// undefined.
    /// </para>
    /// <para>
    /// Like its real counterpart, this implements the Thomas algorithm
    /// the Rust documents, and not the Rust's (incorrect) variant - see
    /// <see cref="TridiagonalSolve(float[], float[], float[], float[])"/>
    /// for the details.
    /// </para>
    /// </remarks>
    public static bool TridiagonalSolveComplex(
        Complex32[] diagonal, Complex32[] offDiagonal, Complex32[] rhs, Complex32[] outState)
    {
        int n = diagonal.Length;
        if (n == 0 || rhs.Length < n || outState.Length < n)
        {
            return false;
        }
        return TridiagonalSolveComplex(diagonal, offDiagonal, rhs, outState, new Complex32[n]);
    }

    /// <summary>
    /// The complex Thomas algorithm with the caller's scratch buffer for
    /// the elimination multipliers.
    /// </summary>
    internal static bool TridiagonalSolveComplex(
        Complex32[] diagonal, Complex32[] offDiagonal, Complex32[] rhs, Complex32[] outState,
        Complex32[] multipliers)
    {
        int n = diagonal.Length;
        if (n == 0 || rhs.Length < n || outState.Length < n || multipliers.Length < n)
        {
            return false;
        }
        // The forward sweep: the elimination multipliers are kept in the
        // scratch.
        Complex32 pivot = diagonal[0];
        if (Complex32.IsZero(pivot) || !Complex32.IsFinite(pivot))
        {
            return false;
        }
        outState[0] = Complex32.Divide(rhs[0], pivot);
        if (!Complex32.IsFinite(outState[0]))
        {
            return false;
        }
        for (int i = 1; i < n; i++)
        {
            Complex32 e = i - 1 < offDiagonal.Length ? offDiagonal[i - 1] : Complex32.Zero;
            // The elimination multiplier and the modified pivot.
            Complex32 multiplier = Complex32.Divide(e, pivot);
            multipliers[i] = multiplier;
            pivot = Complex32.Subtract(diagonal[i], Complex32.Multiply(e, multiplier));
            if (Complex32.IsZero(pivot) || !Complex32.IsFinite(pivot))
            {
                return false;
            }
            outState[i] = Complex32.Divide(
                Complex32.Subtract(rhs[i], Complex32.Multiply(e, outState[i - 1])), pivot);
            if (!Complex32.IsFinite(outState[i]))
            {
                return false;
            }
        }
        // The back substitution.
        for (int i = n - 2; i >= 0; i--)
        {
            outState[i] = Complex32.Subtract(outState[i], Complex32.Multiply(multipliers[i + 1], outState[i + 1]));
        }
        return true;
    }

    /// <summary>
    /// The eigenvector of the lowest eigenvalue of a symmetric tridiagonal
    /// matrix, by inverse iteration.
    /// </summary>
    /// <remarks>
    /// The method solves <c>(H - shift I) x = v</c> repeatedly, with
    /// <c>shift</c> just below the lowest eigenvalue, and normalises each
    /// time. The eigenvector closest to the shift grows the fastest under
    /// the iteration, so the state converges to the ground state - and it
    /// converges <em>quadratically</em>, which is what distinguishes it
    /// from the relaxation and the power iteration.
    /// <para>
    /// The shift must be <em>below</em> the lowest eigenvalue: exactly on
    /// it the matrix is singular and the solve fails.
    /// </para>
    /// <para>
    /// This uses <see cref="TridiagonalSolve(float[], float[], float[], float[])"/>,
    /// the Thomas algorithm, in the same operation order as the Rust
    /// core, so the iteration agrees with it bit-for-bit.
    /// </para>
    /// </remarks>
    public static QuantumState? LowestEigenvector(
        float[] diagonal, float[] offDiagonal, float eigenvalue, int iterations)
    {
        int n = diagonal.Length;
        if (n == 0)
        {
            return null;
        }
        // Just below the eigenvalue, so the system is not singular.
        float scale = 0.0f;
        foreach (float d in diagonal)
        {
            float a = MathF.Abs(d);
            scale = a > scale ? a : scale;
        }
        scale = scale > 1.0f ? scale : 1.0f;
        float shift = eigenvalue - 1e-6f * scale;

        var shiftedDiagonal = new float[n];
        for (int i = 0; i < n; i++)
        {
            shiftedDiagonal[i] = diagonal[i] - shift;
        }

        // The start: a ramp, so the state has an overlap with the ground
        // state and not only with the odd ones.
        var state = new QuantumState(n);
        for (int i = 0; i < n; i++)
        {
            float x = i / (float)(System.Math.Max(n, 2) - 1);
            state.Amplitudes[i] = new Complex32(x, 0.0f);
        }
        state.Normalize();

        var rhs = new float[n];
        var solution = new float[n];
        var multipliers = new float[n];
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // The right-hand side is the current state: the Hamiltonian is
            // real, so the solve is a real one.
            for (int i = 0; i < n; i++)
            {
                rhs[i] = state.Amplitudes[i].Re;
            }
            if (!TridiagonalSolve(shiftedDiagonal, offDiagonal, rhs, solution, multipliers))
            {
                return null;
            }
            for (int i = 0; i < n; i++)
            {
                state.Amplitudes[i] = new Complex32(solution[i], 0.0f);
            }
            if (!state.Normalize())
            {
                return null;
            }
        }
        return state;
    }
}
