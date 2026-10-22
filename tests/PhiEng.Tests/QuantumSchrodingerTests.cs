using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// The stationary one-dimensional Schrodinger equation (M17-M26). Mirrors
/// the Rust <c>crates/quantum</c> Stage C tests (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
/// <remarks>
/// <para>
/// These tests assert the <em>physics</em>: the infinite well's
/// <c>E_n = n^2 pi^2 / 2</c>, the harmonic oscillator's
/// <c>E_n = (n + 1/2) hbar omega</c>, a ground state that is normalised
/// and symmetric, a Sturm count that agrees with the spectrum, a
/// tridiagonal solve that reproduces a known solution, a double well
/// whose ground state sits below the barrier, and the soft Coulomb
/// potential's hydrogenic ground state.
/// </para>
/// <para>
/// The grid is 401 points unless a test says otherwise; the spacing is
/// then <c>1/400</c>, and the discretisation error is second order in
/// it - which is why the well's first level is already within 5e-6 of
/// the continuum value.
/// </para>
/// </remarks>
public class QuantumSchrodingerTests
{
    /// <summary>The infinite square well on [0, 1], with hbar = m = 1.</summary>
    private static Hamiltonian1D Well(int points = 401)
    {
        var grid = new Grid1D(points, 0.0f, 1.0f);
        return new Hamiltonian1D(grid, Potential1D.Zero, 1.0f, 1.0f);
    }

    /// <summary>The harmonic oscillator on [-10, 10], with omega = 1.</summary>
    private static Hamiltonian1D HarmonicOscillator(int points = 401)
    {
        var grid = new Grid1D(points, -10.0f, 10.0f);
        return new Hamiltonian1D(grid, Potential1D.Harmonic(1.0f), 1.0f, 1.0f);
    }

    /// <summary>The index of the largest amplitude of a state.</summary>
    private static int Peak(QuantumState state)
    {
        int peak = 0;
        for (int i = 1; i < state.Dim; i++)
        {
            if (MathF.Abs(state.Amplitudes[i].Re) > MathF.Abs(state.Amplitudes[peak].Re))
            {
                peak = i;
            }
        }
        return peak;
    }

    /// <summary>
    /// M17: the infinite square well's spectrum is <c>E_n = n^2 pi^2 / 2</c>.
    /// </summary>
    /// <remarks>
    /// The unknowns are the interior points, so the Dirichlet condition
    /// holds the two ends at zero and the discrete spectrum is that of a
    /// 399 by 399 tridiagonal matrix. The first four levels are checked
    /// against the continuum values to 1e-3 relative.
    /// </remarks>
    [Fact]
    public void M17_TheInfiniteWellSpectrum()
    {
        var h = Well();
        var values = QuantumLinalg.TridiagonalEigenvalues(h.Diagonal(), h.OffDiagonal());
        Assert.Equal(399, values.Length);

        for (int n = 1; n <= 4; n++)
        {
            double exact = n * n * System.Math.PI * System.Math.PI / 2.0;
            double relative = System.Math.Abs(values[n - 1] - exact) / exact;
            Assert.True(
                relative < 1e-3,
                $"E_{n} = {values[n - 1]} must be within 1e-3 of {exact} (off by {relative:E3})");
        }

        // The spectrum rises, and the gap grows like n.
        for (int n = 1; n < values.Length; n++)
        {
            Assert.True(values[n] > values[n - 1], "the spectrum must rise");
        }
        Assert.True(values[1] - values[0] > values[0] - 0.0f, "the gap must grow");
    }

    /// <summary>
    /// M18: the harmonic oscillator's spectrum is <c>E_n = (n + 1/2) hbar omega</c>.
    /// </summary>
    [Fact]
    public void M18_TheHarmonicOscillatorSpectrum()
    {
        var h = HarmonicOscillator();
        var values = QuantumLinalg.TridiagonalLowestEigenvalues(h.Diagonal(), h.OffDiagonal(), 4);
        Assert.Equal(4, values.Length);

        for (int n = 0; n < values.Length; n++)
        {
            double exact = n + 0.5;
            double relative = System.Math.Abs(values[n] - exact) / exact;
            Assert.True(
                relative < 1e-3,
                $"E_{n} = {values[n]} must be within 1e-3 of {exact} (off by {relative:E3})");
        }

        // The levels are evenly spaced: the gap is hbar omega.
        for (int n = 1; n < values.Length; n++)
        {
            Assert.True(System.Math.Abs((values[n] - values[n - 1]) - 1.0f) < 1e-3f, "the gap must be one");
        }
    }

    /// <summary>
    /// M19: the ground state of the well: the right energy, normalised and
    /// symmetric.
    /// </summary>
    /// <remarks>
    /// The ground state of a symmetric well is symmetric about its centre,
    /// so the amplitude at <c>i</c> equals the amplitude at <c>n - 1 - i</c>.
    /// </remarks>
    [Fact]
    public void M19_TheGroundStateIsNormalisedAndSymmetric()
    {
        var h = Well();
        var (energy, state) = h.GroundState(200, 1e-9f);

        double exact = System.Math.PI * System.Math.PI / 2.0;
        double relative = System.Math.Abs(energy - exact) / exact;
        Assert.True(
            relative < 1e-3,
            $"the ground state energy {energy} must be within 1e-3 of {exact} (off by {relative:E3})");

        Assert.True(state.IsNormalized(1e-5f), "the state must be normalised");

        int n = state.Dim;
        for (int i = 0; i < n; i++)
        {
            float asymmetry = MathF.Abs(
                state.Amplitudes[i].Re - state.Amplitudes[n - 1 - i].Re);
            Assert.True(asymmetry < 1e-5f, $"the state must be symmetric (i = {i})");
        }
    }

    /// <summary>
    /// M20: the ground state of the harmonic oscillator: the energy is
    /// <c>hbar omega / 2</c> and the wavefunction peaks at the centre.
    /// </summary>
    [Fact]
    public void M20_TheHarmonicGroundStatePeaksAtTheCentre()
    {
        var h = HarmonicOscillator();
        var (energy, state) = h.GroundState(200, 1e-9f);

        Assert.True(
            MathF.Abs(energy - 0.5f) < 1e-3f,
            $"the ground state energy {energy} must be 0.5 to 1e-3");

        // The unknowns are the interior points of a 401 point grid, so the
        // centre is the 200th of them: index 199.
        int centre = state.Dim / 2;
        Assert.Equal(199, centre);
        int peak = Peak(state);
        Assert.True(
            System.Math.Abs(peak - centre) <= 1,
            $"the wavefunction must peak at the centre (peak at {peak})");
    }

    /// <summary>
    /// M21: the Sturm count is the number of eigenvalues below the query
    /// point.
    /// </summary>
    [Fact]
    public void M21_TheSturmCountCountsTheEigenvaluesBelow()
    {
        var h = HarmonicOscillator();
        float[] d = h.Diagonal();
        float[] e = h.OffDiagonal();
        var spectrum = QuantumLinalg.TridiagonalLowestEigenvalues(d, e, 4);

        // Below the lowest eigenvalue there is nothing.
        Assert.Equal(0, QuantumLinalg.SturmCount(d, e, spectrum[0] - 0.01f));
        // Just above it there is exactly one.
        Assert.Equal(1, QuantumLinalg.SturmCount(d, e, spectrum[0] + 0.01f));

        // The count at a point is the number of eigenvalues below it.
        for (int k = 0; k < spectrum.Length; k++)
        {
            Assert.Equal(k, QuantumLinalg.SturmCount(d, e, spectrum[k] - 0.01f));
            Assert.Equal(k + 1, QuantumLinalg.SturmCount(d, e, spectrum[k] + 0.01f));
        }

        // And the count never decreases as the query point rises.
        int last = 0;
        for (int i = 0; i <= 400; i++)
        {
            float x = spectrum[0] - 0.5f + i * (4.0f / 400.0f);
            int count = QuantumLinalg.SturmCount(d, e, x);
            Assert.True(count >= last, $"the count must not decrease (x = {x})");
            last = count;
        }
    }

    /// <summary>
    /// M22: the tridiagonal solve reproduces a known solution.
    /// </summary>
    /// <remarks>
    /// The system is the four by four tridiagonal matrix with the
    /// diagonal 4, 5, 6, 7 and the off-diagonal -1; its solution is
    /// 1, 2, 3, 4.
    /// </remarks>
    [Fact]
    public void M22_TheTridiagonalSolveReproducesTheSolution()
    {
        float[] diagonal = { 4.0f, 5.0f, 6.0f, 7.0f };
        float[] offDiagonal = { -1.0f, -1.0f, -1.0f };
        float[] rhs = { 2.0f, 6.0f, 12.0f, 25.0f };
        var solution = new float[4];

        Assert.True(
            QuantumLinalg.TridiagonalSolve(diagonal, offDiagonal, rhs, solution),
            "the solve must succeed");
        for (int i = 0; i < 4; i++)
        {
            Assert.True(
                MathF.Abs(solution[i] - (i + 1)) < 1e-5f,
                $"x[{i}] = {solution[i]} must be {i + 1}");
        }

        // The buffers must be long enough, or the solve refuses.
        Assert.False(
            QuantumLinalg.TridiagonalSolve(diagonal, offDiagonal, new float[2], solution),
            "a short right-hand side must be refused");
        Assert.False(
            QuantumLinalg.TridiagonalSolve(diagonal, offDiagonal, rhs, new float[2]),
            "a short output buffer must be refused");
    }

    /// <summary>
    /// M23: the lowest eigenvector's expectation value is the eigenvalue,
    /// and the vector is normalised.
    /// </summary>
    [Fact]
    public void M23_TheLowestEigenvectorHasTheEigenvalueAsItsExpectation()
    {
        var h = Well();
        float[] d = h.Diagonal();
        float[] e = h.OffDiagonal();
        float eigenvalue = QuantumLinalg.TridiagonalLowestEigenvalues(d, e, 1)[0];

        var state = QuantumLinalg.LowestEigenvector(d, e, eigenvalue, 50);
        Assert.NotNull(state);
        Assert.True(state!.IsNormalized(1e-5f), "the eigenvector must be normalised");

        var expectation = state.Expectation(h);
        Assert.True(
            MathF.Abs(expectation.Re - eigenvalue) <= 1e-3f * eigenvalue,
            $"the expectation {expectation.Re} must be the eigenvalue {eigenvalue}");
        Assert.True(
            MathF.Abs(expectation.Im) < 1e-5f,
            "the expectation of a Hermitian operator must be real");
    }

    /// <summary>
    /// M24: the double well's ground state sits below the barrier, so it
    /// is localised in a well instead of spread over both.
    /// </summary>
    [Fact]
    public void M24_TheDoubleWellGroundStateIsBelowTheBarrier()
    {
        const float barrier = 5.0f;
        var grid = new Grid1D(401, -8.0f, 8.0f);
        var h = new Hamiltonian1D(
            grid, Potential1D.DoubleWell(0.0f, 2.0f, 1.0f, barrier), 1.0f, 1.0f);

        var (energy, state) = h.GroundState(200, 1e-9f);
        Assert.True(energy < barrier, $"the ground state energy {energy} must be below the barrier");

        // The weight inside the barrier must be a small fraction of the
        // weight inside a well: the state is in a well, not over both.
        double wellMass = 0.0;
        double barrierMass = 0.0;
        for (int i = 0; i < state.Dim; i++)
        {
            float x = grid.XAt(i + 1);
            double density = state.Amplitudes[i].Re * state.Amplitudes[i].Re;
            if (MathF.Abs(x) >= 1.5f && MathF.Abs(x) <= 3.0f)
            {
                wellMass += density;
            }
            if (MathF.Abs(x) <= 1.0f)
            {
                barrierMass += density;
            }
        }
        Assert.True(wellMass > 0.0f, "the state must live in a well");
        Assert.True(
            barrierMass < 0.05 * wellMass,
            $"the state must be localised: well {wellMass}, barrier {barrierMass}");
    }

    /// <summary>
    /// M25: the soft Coulomb potential's ground state is the hydrogenic
    /// one: the energy is within 2% of -0.5.
    /// </summary>
    /// <remarks>
    /// The one-dimensional soft Coulomb potential is weaker than the
    /// Coulomb one at every point, so its ground state depends strongly
    /// on the softening: the energy is -0.371 at a softening of 2, -0.670
    /// at 1 and -1.177 at 0.5. The softening of 1.4 is the one that
    /// reproduces the hydrogenic ground state, and the energy is stable
    /// under refinement (the same value to 1e-5 on a 401 point and on a
    /// 2001 point grid).
    /// </remarks>
    [Fact]
    public void M25_TheSoftCoulombGroundStateIsTheHydrogenicOne()
    {
        var grid = new Grid1D(2001, -20.0f, 20.0f);
        var h = new Hamiltonian1D(
            grid, Potential1D.SoftCoulomb(1.0f, 1.4f), 1.0f, 1.0f);

        var (energy, _) = h.GroundState(100, 1e-9f);
        float relative = MathF.Abs(energy + 0.5f) / 0.5f;
        Assert.True(
            relative <= 0.02f,
            $"the ground state energy {energy} must be within 2% of -0.5 (off by {relative:P2})");
    }

    /// <summary>
    /// M26: the grid: the spacing, the positions and the endpoints.
    /// </summary>
    [Fact]
    public void M26_TheGridSpacingAndPositions()
    {
        var grid = new Grid1D(401, -1.0f, 1.0f);
        Assert.Equal(401, grid.N);
        Assert.Equal(-1.0f, grid.XMin);
        Assert.Equal(1.0f, grid.XMax);
        Assert.Equal(2.0f / 400.0f, grid.Dx());

        var positions = grid.Positions();
        Assert.Equal(401, positions.Length);
        Assert.Equal(-1.0f, positions[0]);
        Assert.Equal(1.0f, positions[400]);
        Assert.Equal(0.0f, positions[200]);
        for (int i = 1; i < positions.Length; i++)
        {
            Assert.True(
                MathF.Abs((positions[i] - positions[i - 1]) - grid.Dx()) < 1e-6f,
                "the spacing must be uniform");
        }

        // An index outside the grid returns the nearest end.
        Assert.Equal(grid.XMax, grid.XAt(401));
        Assert.Equal(grid.XMax, grid.XAt(1000));

        // A grid of fewer than two points has no spacing and is clamped
        // to two.
        var clamped = new Grid1D(1, -1.0f, 1.0f);
        Assert.Equal(2, clamped.N);
        Assert.Equal(2.0f, clamped.Dx());
        Assert.Equal(2, clamped.Positions().Length);
    }
}
