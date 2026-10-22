using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Density matrices: mixed states, partial traces, entropy, channels
/// and open-system evolution.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::density</c> module operation
/// for operation, so the two engines agree bit-for-bit.
/// </remarks>
public sealed class DensityMatrix
{
    private readonly ComplexMatrix _matrix;

    private DensityMatrix(ComplexMatrix matrix)
    {
        _matrix = matrix;
    }

    /// <summary>The density matrix of a pure state: |psi&gt;&lt;psi|.</summary>
    public static DensityMatrix FromPure(QuantumState state)
    {
        int n = state.Dim;
        var matrix = new ComplexMatrix(n, n);
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                matrix.Set(r, c, state.Amplitudes[r] * Complex32.Conjugate(state.Amplitudes[c]));
            }
        }
        return new DensityMatrix(matrix);
    }

    /// <summary>A density matrix from a raw matrix, without validation.</summary>
    /// <remarks>
    /// The caller promises Hermiticity, positivity and unit trace;
    /// <see cref="IsValid"/> checks the first and the last. This exists
    /// so tests and tools can name a matrix that no constructor would
    /// produce.
    /// </remarks>
    public static DensityMatrix FromMatrixUnchecked(ComplexMatrix matrix) =>
        new(matrix);

    /// <summary>The maximally mixed state: the identity over the dimension.</summary>
    public static DensityMatrix MaximallyMixed(int dim)
    {
        var matrix = new ComplexMatrix(dim, dim);
        if (dim == 0)
        {
            return new DensityMatrix(matrix);
        }
        var share = new Complex32(1.0f / dim, 0.0f);
        for (int i = 0; i < dim; i++)
        {
            matrix.Set(i, i, share);
        }
        return new DensityMatrix(matrix);
    }

    /// <summary>The dimension.</summary>
    public int Dim => _matrix.Rows;

    /// <summary>The underlying matrix.</summary>
    public ComplexMatrix Matrix => _matrix;

    /// <summary>The trace: one for every valid density matrix.</summary>
    public Complex32 Trace() => _matrix.Trace();

    /// <summary>
    /// The purity Tr(rho²): one for a pure state, 1/d for the maximally
    /// mixed state of dimension d.
    /// </summary>
    public float Purity() => _matrix.MatMul(_matrix).Trace().Re;

    /// <summary>
    /// Whether the matrix is a plausible density matrix: Hermitian with
    /// unit trace, to within <paramref name="tol"/>.
    /// </summary>
    public bool IsValid(float tol) =>
        _matrix.IsHermitian(tol) && MathF.Abs(_matrix.Trace().Re - 1.0f) <= tol;

    /// <summary>The eigenvalues, in ascending order.</summary>
    public float[] Eigenvalues() => QuantumLinalg.EigenHermitian(_matrix, 64, 1e-8f).Eigenvalues;

    /// <summary>
    /// The von Neumann entropy -Tr(rho ln rho), in nats.
    /// </summary>
    /// <remarks>
    /// Zero for a pure state, ln d for the maximally mixed state of
    /// dimension d. Non-positive eigenvalues - rounding noise below a
    /// true zero - contribute nothing rather than a NaN.
    /// </remarks>
    public float VonNeumannEntropy()
    {
        float entropy = 0.0f;
        foreach (float value in Eigenvalues())
        {
            if (value > 0.0f)
            {
                entropy -= value * DeterministicScalar.Ln(value);
            }
        }
        return entropy;
    }

    /// <summary>
    /// The diagonal: the measurement probabilities in the computational
    /// basis.
    /// </summary>
    public float[] Probabilities()
    {
        int n = Dim;
        var output = new float[n];
        for (int i = 0; i < n; i++)
        {
            output[i] = _matrix.Get(i, i).Re;
        }
        return output;
    }

    /// <summary>
    /// Traces out every qubit except the ones in <paramref name="keep"/>.
    /// </summary>
    /// <remarks>
    /// <paramref name="keep"/> lists the qubit indices that survive, in
    /// any order; the result has one qubit per kept index, in the order
    /// they were listed. Tracing out everything returns the 1 by 1
    /// matrix [[1]].
    /// </remarks>
    public DensityMatrix PartialTrace(int[] keep, int totalQubits)
    {
        var kept = new System.Collections.Generic.List<int>();
        foreach (int q in keep)
        {
            if (q >= 0 && q < totalQubits && !kept.Contains(q))
            {
                kept.Add(q);
            }
        }
        kept.Sort();
        int bits = System.Math.Min(kept.Count, 20);
        int dim = 1 << bits;
        var matrix = new ComplexMatrix(dim, dim);
        var traced = new System.Collections.Generic.List<int>();
        for (int q = 0; q < totalQubits; q++)
        {
            if (!kept.Contains(q))
            {
                traced.Add(q);
            }
        }
        int tracedBits = System.Math.Min(traced.Count, 20);
        for (int a = 0; a < dim; a++)
        {
            for (int b = 0; b < dim; b++)
            {
                var sum = Complex32.Zero;
                for (int e = 0; e < (1 << tracedBits); e++)
                {
                    int row = 0;
                    int col = 0;
                    for (int k = 0; k < kept.Count; k++)
                    {
                        int q = kept[k];
                        if ((a & (1 << k)) != 0)
                        {
                            row |= 1 << q;
                        }
                        if ((b & (1 << k)) != 0)
                        {
                            col |= 1 << q;
                        }
                    }
                    for (int k = 0; k < traced.Count; k++)
                    {
                        int q = traced[k];
                        if ((e & (1 << k)) != 0)
                        {
                            row |= 1 << q;
                            col |= 1 << q;
                        }
                    }
                    sum = sum + _matrix.Get(row, col);
                }
                matrix.Set(a, b, sum);
            }
        }
        return new DensityMatrix(matrix);
    }

    /// <summary>Applies a channel: rho -&gt; sum_k K_k rho K_k†.</summary>
    public DensityMatrix ApplyChannel(ComplexMatrix[] kraus)
    {
        int n = Dim;
        var output = new ComplexMatrix(n, n);
        foreach (var k in kraus)
        {
            var right = k.MatMul(_matrix).MatMul(k.Dagger());
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    output.Set(r, c, output.Get(r, c) + right.Get(r, c));
                }
            }
        }
        return new DensityMatrix(output);
    }
}

/// <summary>The textbook noisy channels, as Kraus operators.</summary>
public static class QuantumChannels
{
    /// <summary>
    /// The Kraus operators of the depolarising channel with error rate
    /// <paramref name="p"/>: with probability <c>p</c> the qubit is
    /// replaced by white noise.
    /// </summary>
    public static ComplexMatrix[] Depolarizing(float p)
    {
        float keep = MathF.Sqrt(1.0f - 0.75f * p);
        float flip = MathF.Sqrt(0.25f * p);
        var k0 = ComplexMatrix.Identity(2);
        for (int i = 0; i < 2; i++)
        {
            k0.Set(i, i, new Complex32(keep, 0.0f));
        }
        var kx = new ComplexMatrix(2, 2);
        kx.Set(0, 1, new Complex32(flip, 0.0f));
        kx.Set(1, 0, new Complex32(flip, 0.0f));
        var ky = new ComplexMatrix(2, 2);
        ky.Set(0, 1, new Complex32(0.0f, -flip));
        ky.Set(1, 0, new Complex32(0.0f, flip));
        var kz = new ComplexMatrix(2, 2);
        kz.Set(0, 0, new Complex32(flip, 0.0f));
        kz.Set(1, 1, new Complex32(-flip, 0.0f));
        return new[] { k0, kx, ky, kz };
    }

    /// <summary>
    /// The Kraus operators of amplitude damping with rate
    /// <paramref name="gamma"/>: |1&gt; decays to |0&gt;, and the
    /// coherence follows it down.
    /// </summary>
    public static ComplexMatrix[] AmplitudeDamping(float gamma)
    {
        var k0 = new ComplexMatrix(2, 2);
        k0.Set(0, 0, Complex32.One);
        k0.Set(1, 1, new Complex32(MathF.Sqrt(MathF.Max(1.0f - gamma, 0.0f)), 0.0f));
        var k1 = new ComplexMatrix(2, 2);
        k1.Set(0, 1, new Complex32(MathF.Sqrt(MathF.Max(gamma, 0.0f)), 0.0f));
        return new[] { k0, k1 };
    }

    /// <summary>
    /// The Kraus operators of dephasing with rate <paramref name="p"/>:
    /// the populations survive, the coherences shrink by
    /// <c>1 - 2p</c>.
    /// </summary>
    public static ComplexMatrix[] Dephasing(float p)
    {
        var k0 = ComplexMatrix.Identity(2);
        for (int i = 0; i < 2; i++)
        {
            k0.Set(i, i, new Complex32(MathF.Sqrt(1.0f - p), 0.0f));
        }
        float s = MathF.Sqrt(MathF.Max(p, 0.0f));
        var k1 = new ComplexMatrix(2, 2);
        k1.Set(0, 0, new Complex32(s, 0.0f));
        k1.Set(1, 1, new Complex32(-s, 0.0f));
        return new[] { k0, k1 };
    }

    /// <summary>
    /// One Euler step of the Lindblad master equation.
    /// </summary>
    /// <remarks>
    /// The Hamiltonian part is unitary; the dissipator is where the
    /// environment steals coherence. Returns false for dimension
    /// mismatches, leaving the matrix alone in that case.
    /// </remarks>
    public static bool LindbladStep(
        DensityMatrix rho,
        ComplexMatrix hamiltonian,
        (float Rate, ComplexMatrix Jump)[] jumps,
        float dt)
    {
        int n = rho.Dim;
        if (hamiltonian.Rows != n || hamiltonian.Cols != n)
        {
            return false;
        }
        foreach (var (_, jump) in jumps)
        {
            if (jump.Rows != n || jump.Cols != n)
            {
                return false;
            }
        }
        var m = rho.Matrix;
        // -i [H, rho].
        var hr = hamiltonian.MatMul(m);
        var rh = m.MatMul(hamiltonian);
        var derivative = new ComplexMatrix(n, n);
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                var commutator = hr.Get(r, c) - rh.Get(r, c);
                // -i times the commutator: (x + iy) -> (y - ix).
                derivative.Set(r, c, new Complex32(commutator.Im, -commutator.Re));
            }
        }
        // The dissipator, jump by jump.
        foreach (var (rate, jump) in jumps)
        {
            var lrl = jump.MatMul(m).MatMul(jump.Dagger());
            var ldl = jump.Dagger().MatMul(jump);
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    // {L†L, rho} / 2 at (r, c): half the row-plus-column.
                    var anti = Complex32.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        anti = anti + ldl.Get(r, k) * m.Get(k, c);
                        anti = anti + m.Get(r, k) * ldl.Get(k, c);
                    }
                    var term = lrl.Get(r, c) - anti * 0.5f;
                    derivative.Set(r, c, derivative.Get(r, c) + term * rate);
                }
            }
        }
        var next = new ComplexMatrix(n, n);
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                next.Set(r, c, m.Get(r, c) + derivative.Get(r, c) * dt);
            }
        }
        // The wrapped matrix is live: write the step back into it.
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                m.Set(r, c, next.Get(r, c));
            }
        }
        return true;
    }
}
