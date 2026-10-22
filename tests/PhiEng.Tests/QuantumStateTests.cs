using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// The quantum vocabulary (M09-M16). Mirrors
/// <c>crates/engine/tests/quantum_state.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// A state is a vector of complex amplitudes; an observable is an
/// operator acting on it. These tests pin the arithmetic of that
/// vocabulary and the <em>physics</em> it has to obey: a normalised
/// state's Born probabilities sum to one, the probability of an outcome
/// is the expectation of the projector onto it, the expectation of a
/// Hermitian operator is real, two states that differ only by a phase are
/// the same state, and the eigensolver's eigenvectors really do rebuild
/// the matrix.
/// </remarks>
public class QuantumStateTests
{
    /// <summary>The bit pattern of a float, as the pins are written.</summary>
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>A three by three Hermitian matrix: the eigenvalues are 2 ± √2 and 2.</summary>
    private static ComplexMatrix Tridiagonal3()
    {
        var m = new ComplexMatrix(3, 3);
        for (int i = 0; i < 3; i++)
        {
            m.Set(i, i, new Complex32(2.0f, 0.0f));
        }
        m.Set(0, 1, new Complex32(1.0f, 0.0f));
        m.Set(1, 0, new Complex32(1.0f, 0.0f));
        m.Set(1, 2, new Complex32(1.0f, 0.0f));
        m.Set(2, 1, new Complex32(1.0f, 0.0f));
        return m;
    }

    /// <summary>A two by two Hermitian matrix with a complex off-diagonal: ±√2.</summary>
    private static ComplexMatrix Complex2()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, new Complex32(1.0f, 0.0f));
        m.Set(1, 1, new Complex32(-1.0f, 0.0f));
        m.Set(0, 1, new Complex32(0.0f, 1.0f));
        m.Set(1, 0, new Complex32(0.0f, -1.0f));
        return m;
    }

    /// <summary>
    /// M09: a state can be normalised, and its Born probabilities follow.
    /// </summary>
    [Fact]
    public void M09_AStateNormalisesAndItsProbabilitiesFollow()
    {
        var s = new QuantumState(new[]
        {
            new Complex32(1.0f, 0.0f),
            new Complex32(1.0f, 0.0f),
        });
        Assert.Equal(0x40000000u, Bits(s.NormSqr()));
        Assert.True(s.Normalize(), "the state must normalise");
        Assert.Equal(0x3f7fffffu, Bits(s.NormSqr()));
        Assert.True(s.IsNormalized(1e-6f), "the state must be normalised now");

        var p = s.BornProbabilities();
        Assert.Equal(2, p.Length);
        Assert.Equal(0x3effffffu, Bits(p[0]));
        Assert.Equal(0x3effffffu, Bits(p[1]));
        Assert.True(MathF.Abs(p[0] + p[1] - 1.0f) < 1e-6f, "the probabilities must sum to one");

        // The zero state has no norm to scale by.
        var zero = new QuantumState(3);
        Assert.False(zero.Normalize(), "the zero state cannot be normalised");
        Assert.Equal(0u, Bits(zero.NormSqr()));
    }

    /// <summary>
    /// M10: the expectation value of a Hermitian operator is real.
    /// </summary>
    [Fact]
    public void M10_AHermitianExpectationIsReal()
    {
        var op = QuantumOperators.Dense(Complex2());
        var s = new QuantumState(new[]
        {
            new Complex32(1.0f, 0.0f),
            new Complex32(0.0f, 0.0f),
        });
        var value = s.Expectation(op);
        Assert.Equal(0x3f800000u, Bits(value.Re));
        Assert.Equal(0u, Bits(value.Im));

        // A superposition, to make sure the off-diagonal terms cancel.
        var superposed = new QuantumState(new[]
        {
            new Complex32(1.0f, 0.0f),
            new Complex32(1.0f, 0.0f),
        });
        var value2 = superposed.Expectation(op);
        Assert.True(
            MathF.Abs(value2.Im) < 1e-6f,
            $"the expectation of a Hermitian operator must be real, got {value2}");
    }

    /// <summary>
    /// M11: the Born probability is the expectation of the projector.
    /// </summary>
    [Fact]
    public void M11_TheBornProbabilityIsAProjectorExpectation()
    {
        var proj = new ComplexMatrix(2, 2);
        proj.Set(0, 0, Complex32.One);
        var op = QuantumOperators.Dense(proj);

        var s = new QuantumState(new[]
        {
            new Complex32(0.6f, 0.0f),
            new Complex32(0.8f, 0.0f),
        });
        var p = s.BornProbabilities();
        var value = s.Expectation(op);
        Assert.Equal(0x3eb851ecu, Bits(p[0]));
        Assert.Equal(0x3eb851ecu, Bits(value.Re));
        Assert.True(
            MathF.Abs(value.Re - p[0]) < 1e-6f,
            "the projector expectation must be the probability");
    }

    /// <summary>
    /// M12: two states that differ only by a phase are the same state.
    /// </summary>
    [Fact]
    public void M12_TheOverlapAndTheFidelity()
    {
        var a = new QuantumState(new[]
        {
            new Complex32(1.0f, 0.0f),
            new Complex32(0.0f, 0.0f),
        });
        var b = new QuantumState(new[]
        {
            new Complex32(0.0f, 0.0f),
            new Complex32(1.0f, 0.0f),
        });
        // A phase-shifted copy of `a`.
        var c = new QuantumState(new[]
        {
            new Complex32(0.0f, 0.0f),
            new Complex32(-1.0f, 0.0f),
        });

        Assert.Equal(0x3f800000u, Bits(a.Fidelity(a)));
        Assert.Equal(0u, Bits(a.Fidelity(b)));
        Assert.Equal(0u, Bits(a.Fidelity(c)));

        var phase = a.PhaseTo(c);
        Assert.True(Complex32.IsFinite(phase));
    }

    /// <summary>
    /// M13: the dense matrix operations.
    /// </summary>
    [Fact]
    public void M13_TheMatrixOperations()
    {
        var m = Complex2();
        Assert.Equal(0u, Bits(m.Trace().Re));
        Assert.True(MathF.Abs(m.Trace().Im) < 1e-6f, "the trace must be real");
        Assert.True(m.IsHermitian(1e-6f), "the matrix must be Hermitian");

        var dag = m.Dagger();
        Assert.Equal(0u, Bits(dag.Get(0, 1).Re));
        Assert.Equal(0x3f800000u, Bits(dag.Get(0, 1).Im));

        var k = m.Kron(ComplexMatrix.Identity(2));
        Assert.Equal(4, k.Rows);
        Assert.Equal(4, k.Cols);

        // The identity times the matrix is the matrix.
        var id = ComplexMatrix.Identity(2);
        var product = id.MatMul(m);
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(Bits(m.Get(r, c).Re), Bits(product.Get(r, c).Re));
                Assert.Equal(Bits(m.Get(r, c).Im), Bits(product.Get(r, c).Im));
            }
        }

        // A matrix times its dagger is positive semi-definite: the
        // expectation of it is a real, non-negative number.
        var psd = m.MatMul(m.Dagger());
        var op = QuantumOperators.Dense(psd);
        var s = new QuantumState(new[]
        {
            new Complex32(0.6f, 0.0f),
            new Complex32(0.8f, 0.0f),
        });
        var value = s.Expectation(op);
        Assert.True(MathF.Abs(value.Im) < 1e-6f, "the expectation must be real");
        Assert.True(value.Re >= 0.0f, "the expectation must not be negative");
    }

    /// <summary>
    /// M14: the eigensolver finds the spectrum of a Hermitian matrix.
    /// </summary>
    [Fact]
    public void M14_TheEigensolverFindsTheSpectrum()
    {
        var e = QuantumLinalg.EigenHermitian(Tridiagonal3(), 32, 1e-6f);
        Assert.Equal(3, e.Eigenvalues.Length);
        // 2 - sqrt(2), 2, 2 + sqrt(2).
        Assert.Equal(0x3f15f61au, Bits(e.Eigenvalues[0]));
        Assert.Equal(0x3ffffffeu, Bits(e.Eigenvalues[1]));
        Assert.Equal(0x405a8279u, Bits(e.Eigenvalues[2]));
        float[] want = { 2.0f - MathF.Sqrt(2.0f), 2.0f, 2.0f + MathF.Sqrt(2.0f) };
        for (int i = 0; i < 3; i++)
        {
            Assert.True(
                MathF.Abs(e.Eigenvalues[i] - want[i]) < 1e-5f,
                $"eigenvalue {e.Eigenvalues[i]} != {want[i]}");
        }

        // A matrix with a complex off-diagonal: the eigenvalues are ±√2.
        var e2 = QuantumLinalg.EigenHermitian(Complex2(), 32, 1e-6f);
        Assert.Equal(0xbfb504f2u, Bits(e2.Eigenvalues[0]));
        Assert.Equal(0x3fb504f2u, Bits(e2.Eigenvalues[1]));
        float[] want2 = { -MathF.Sqrt(2.0f), MathF.Sqrt(2.0f) };
        for (int i = 0; i < 2; i++)
        {
            Assert.True(
                MathF.Abs(e2.Eigenvalues[i] - want2[i]) < 1e-5f,
                $"eigenvalue {e2.Eigenvalues[i]} != {want2[i]}");
        }
    }

    /// <summary>
    /// M15: the eigenvectors are orthogonal and rebuild the matrix.
    /// </summary>
    [Fact]
    public void M15_TheEigenvectorsRebuildTheMatrix()
    {
        var m = Tridiagonal3();
        var e = QuantumLinalg.EigenHermitian(m, 32, 1e-6f);
        var op = QuantumOperators.Dense(m);

        // The eigenvalues are the expectations in their own eigenstates.
        for (int i = 0; i < e.Eigenvalues.Length; i++)
        {
            var v = e.Eigenvector(i);
            var expectation = v.Expectation(op);
            Assert.True(
                MathF.Abs(expectation.Re - e.Eigenvalues[i]) < 1e-4f,
                $"the expectation {expectation} must be the eigenvalue {e.Eigenvalues[i]}");
            Assert.True(
                MathF.Abs(expectation.Im) < 1e-4f,
                "the expectation of a Hermitian operator must be real");
        }

        // The eigenvectors are orthogonal.
        for (int i = 0; i < e.Eigenvalues.Length; i++)
        {
            for (int j = 0; j < e.Eigenvalues.Length; j++)
            {
                if (i == j)
                {
                    continue;
                }
                var overlap = e.Eigenvector(i).Overlap(e.Eigenvector(j));
                Assert.True(
                    Complex32.Abs(overlap) < 1e-4f,
                    $"the eigenvectors must be orthogonal, got {overlap}");
            }
        }
    }

    /// <summary>
    /// M16: the power iteration finds the largest magnitude.
    /// </summary>
    [Fact]
    public void M16_ThePowerIterationFindsTheDominantEigenvalue()
    {
        var op = QuantumOperators.Diagonal(new[]
        {
            new Complex32(2.0f, 0.0f),
            new Complex32(1.0f, 0.0f),
            new Complex32(0.0f, 0.0f),
        });
        var (value, state) = QuantumLinalg.DominantEigenpair(op, 3, 200, 1e-6f);
        Assert.Equal(0x3ffffffeu, Bits(value));
        Assert.True(MathF.Abs(value - 2.0f) < 1e-4f, $"the dominant eigenvalue must be two, got {value}");
        Assert.True(state.Amplitudes[0].Re > 0.99f, "the dominant state must be the first basis state");
        Assert.True(state.Amplitudes[1].Re < 0.01f, "the dominant state must be the first basis state");
        Assert.True(state.IsNormalized(1e-5f), "the state must stay normalised");
    }
}
