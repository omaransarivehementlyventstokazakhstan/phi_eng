using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The result of a Hermitian eigendecomposition.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::linalg::HermitianEigen</c>.
/// </remarks>
public sealed class HermitianEigen
{
    /// <summary>The eigenvalues, in ascending order.</summary>
    public float[] Eigenvalues { get; init; } = Array.Empty<float>();

    /// <summary>
    /// The eigenvectors, one per column, in the same order as the
    /// eigenvalues.
    /// </summary>
    public ComplexMatrix Eigenvectors { get; init; } = new(0, 0);

    /// <summary>The eigenvector of the <paramref name="index"/>-th eigenvalue, normalised.</summary>
    public QuantumState Eigenvector(int index)
    {
        int n = Eigenvectors.Cols;
        var amps = new Complex32[n];
        for (int r = 0; r < Eigenvectors.Rows; r++)
        {
            amps[r] = Eigenvectors.Get(r, index);
        }
        var s = new QuantumState(amps);
        s.Normalize();
        return s;
    }
}

/// <summary>
/// The linear algebra a quantum simulation needs.
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
    /// The Hermitian eigendecomposition of a matrix, by the cyclic Jacobi
    /// method.
    /// </summary>
    /// <remarks>
    /// The Jacobi method rotates the matrix until it is diagonal, and the
    /// rotations are accumulated into the eigenvectors. It is chosen for
    /// two reasons: it is unconditionally stable, and it is exactly
    /// reproducible - a fixed sweep order, a fixed pivot rule, no
    /// randomness.
    /// </remarks>
    public static HermitianEigen EigenHermitian(ComplexMatrix a, int maxSweeps, float tol)
    {
        int n = a.Rows;
        var m = new ComplexMatrix(n, a.Cols);
        Array.Copy(a.Data, m.Data, a.Data.Length);
        var v = ComplexMatrix.Identity(n);
        if (n == 0)
        {
            return new HermitianEigen { Eigenvalues = Array.Empty<float>(), Eigenvectors = v };
        }

        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            // The squared weight of the off-diagonal: the measure of how
            // far the matrix is from diagonal.
            float off = 0.0f;
            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    off += Complex32.NormSqr(m.Get(p, q));
                }
            }
            if (off <= tol * tol)
            {
                break;
            }

            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    var apq = m.Get(p, q);
                    float r = Complex32.Abs(apq);
                    if (!float.IsFinite(r) || r <= 0.0f)
                    {
                        continue;
                    }
                    // The phase of the entry and the rotation that zeroes
                    // it: tan(2 theta) = 2|a_pq| / (a_qq - a_pp).
                    //
                    // The rotation is G = [[c, s*conj(phi)], [-s*phi, c]]
                    // on the (p, q) plane, so that A' = G^H A G has a zero
                    // (p, q) entry. Writing it the other way round (with
                    // the signs of s swapped) would need the opposite
                    // angle.
                    var phase = Complex32.Scale(apq, 1.0f / r);
                    float app = m.Get(p, p).Re;
                    float aqq = m.Get(q, q).Re;
                    float theta = 0.5f * DeterministicTrig.Atan2(2.0f * r, aqq - app);
                    var (c, s) = DeterministicTrig.ComputeCosSin(theta);
                    var gpp = new Complex32(c, 0.0f);
                    var gpq = Complex32.Scale(Complex32.Conjugate(phase), s);
                    var gqp = Complex32.Scale(phase, -s);
                    var gqq = new Complex32(c, 0.0f);

                    // A <- A G: the columns p and q are mixed.
                    for (int k = 0; k < n; k++)
                    {
                        var akp = m.Get(k, p);
                        var akq = m.Get(k, q);
                        m.Set(k, p, gpp * akp + gqp * akq);
                        m.Set(k, q, gpq * akp + gqq * akq);
                    }
                    // A <- G^H A: the rows p and q are mixed. The row of
                    // G^H is the conjugate of the *column* of G, so the
                    // row p takes gqp's conjugate and the row q takes
                    // gpq's.
                    for (int j = 0; j < n; j++)
                    {
                        var apj = m.Get(p, j);
                        var aqj = m.Get(q, j);
                        m.Set(p, j, Complex32.Conjugate(gpp) * apj + Complex32.Conjugate(gqp) * aqj);
                        m.Set(q, j, Complex32.Conjugate(gpq) * apj + Complex32.Conjugate(gqq) * aqj);
                    }
                    // V <- V G: the same column mixing, on the eigenvectors.
                    for (int k = 0; k < v.Rows; k++)
                    {
                        var vkp = v.Get(k, p);
                        var vkq = v.Get(k, q);
                        v.Set(k, p, gpp * vkp + gqp * vkq);
                        v.Set(k, q, gpq * vkp + gqq * vkq);
                    }
                }
            }
        }

        // The eigenvalues are the diagonal now; sort them ascending and
        // carry the eigenvectors along. The sort is stable, so a
        // degenerate spectrum keeps the order the algorithm produced.
        var diag = new float[n];
        for (int i = 0; i < n; i++)
        {
            diag[i] = m.Get(i, i).Re;
        }
        var order = Enumerable.Range(0, n).OrderBy(i => diag[i]).ToArray();

        var eigenvalues = new float[n];
        var eigenvectors = new ComplexMatrix(n, n);
        for (int newIndex = 0; newIndex < n; newIndex++)
        {
            int oldIndex = order[newIndex];
            eigenvalues[newIndex] = diag[oldIndex];
            for (int r = 0; r < n; r++)
            {
                eigenvectors.Set(r, newIndex, v.Get(r, oldIndex));
            }
        }

        return new HermitianEigen
        {
            Eigenvalues = eigenvalues,
            Eigenvectors = eigenvectors,
        };
    }

    /// <summary>
    /// The eigenpair of the largest magnitude, by the power iteration.
    /// </summary>
    /// <remarks>
    /// The iteration starts from a fixed vector - a superposition with
    /// every amplitude equal - and repeatedly applies the operator and
    /// normalises. It converges to the eigenvector of the
    /// largest-magnitude eigenvalue, which for the Hamiltonian is the
    /// <em>highest</em> energy state.
    /// </remarks>
    public static (float Value, QuantumState State) DominantEigenpair(
        IQuantumOperator op, int dim, int iterations, float tol)
    {
        var state = new QuantumState(dim);
        for (int i = 0; i < state.Dim; i++)
        {
            state.Amplitudes[i] = Complex32.One;
        }
        state.Normalize();

        float last = 0.0f;
        var applied = new Complex32[dim];
        for (int i = 0; i < iterations; i++)
        {
            op.ApplyTo(state.Amplitudes, applied);
            Array.Copy(applied, state.Amplitudes, dim);
            state.Normalize();
            // The Rayleigh quotient: the expectation of the operator in
            // the current state.
            float value = state.Expectation(op).Re;
            if (MathF.Abs(value - last) <= tol)
            {
                return (value, state);
            }
            last = value;
        }
        return (last, state);
    }
}
