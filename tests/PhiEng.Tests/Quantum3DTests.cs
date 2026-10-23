using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// Three-dimensional grids, separable Hamiltonians, and the
/// Aharonov-Bohm ring (M73–M80). Mirrors
/// <c>crates/engine/tests/quantum_3d.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class Quantum3DTests
{
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    private static Hamiltonian1D Box1D(int n) =>
        new(new Grid1D(n, 0.0f, 1.0f), Potential1D.Zero, 1.0f, 1.0f);

    private static Hamiltonian1D Trap1D(int n) =>
        new(new Grid1D(n, -6.0f, 6.0f), Potential1D.Harmonic(1.0f), 1.0f, 1.0f);

    /// <summary>
    /// M73: the three-dimensional grid maps cells to unknowns and back.
    /// </summary>
    [Fact]
    public void M73_TheGridMapsCellsToUnknownsAndBack()
    {
        var grid = new Grid3D(6, 6, 6, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        // Four interior points per axis: 64 unknowns.
        Assert.Equal(64, grid.DirichletUnknowns());
        Assert.Equal((6, 6, 6), grid.Shape());
        // The flat index runs x-fastest and round-trips through coordinates.
        Assert.Equal(4 + 6 * (3 + 6 * 2), grid.Index(4, 3, 2));
        Assert.Equal((4, 3, 2), grid.Coords(4 + 6 * (3 + 6 * 2)));
        for (int flat = 0; flat < grid.N(); flat++)
        {
            var c = grid.Coords(flat);
            Assert.NotNull(c);
            Assert.Equal(flat, grid.Index(c.Value.Ix, c.Value.Iy, c.Value.Iz));
        }
        // Unknown zero is the first interior cell; the boundary owns none.
        Assert.Equal((1, 1, 1), grid.UnknownToGrid(0));
        Assert.Equal(0, grid.GridToUnknown(1, 1, 1));
        Assert.Null(grid.GridToUnknown(0, 1, 1));
        Assert.Null(grid.GridToUnknown(5, 5, 5));
        Assert.Null(grid.UnknownToGrid(64));
        // A non-cubic grid multiplies its interior extents.
        var thin = new Grid3D(4, 5, 6, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        Assert.Equal(2 * 3 * 4, thin.DirichletUnknowns());
        // The node positions tile the box from the corner.
        var p = grid.Position(0);
        Assert.Equal((Bits(0.0f), Bits(0.0f), Bits(0.0f)), (Bits(p.X), Bits(p.Y), Bits(p.Z)));
        var q = grid.Position(grid.Index(1, 1, 1)!.Value);
        Assert.Equal((Bits(0.2f), Bits(0.2f), Bits(0.2f)), (Bits(q.X), Bits(q.Y), Bits(q.Z)));
    }

    /// <summary>
    /// M74: the cubic box spectrum is the sum of three particle-in-a-box
    /// spectra, with the first excited level triply degenerate.
    /// </summary>
    [Fact]
    public void M74_TheCubicBoxSumsThreeWells()
    {
        var h = Hamiltonian3D.Separable(
            new Grid3D(81, 81, 81, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
            Box1D(81), Box1D(81), Box1D(81));
        Assert.NotNull(h);
        float[] e = h!.SeparableEnergies(4);
        Assert.Equal(4, e.Length);
        float exact = MathF.PI * MathF.PI / 2.0f;
        Assert.Equal(0x416cd70eu, Bits(e[0]));
        Assert.True(MathF.Abs(e[0] - 3.0f * exact) / (3.0f * exact) < 1e-3f);
        // The (1,1,2) permutations: bit-identical, near 6 * pi^2 / 2.
        Assert.Equal(Bits(e[1]), Bits(e[2]));
        Assert.Equal(Bits(e[2]), Bits(e[3]));
        Assert.True(MathF.Abs(e[1] - 6.0f * exact) / (6.0f * exact) < 1e-3f);
    }

    /// <summary>
    /// M75: the harmonic trap ladders add the same way, with the same
    /// triple degeneracy at the first excited level.
    /// </summary>
    [Fact]
    public void M75_TheHarmonicTrapLaddersAdd()
    {
        var h = Hamiltonian3D.Separable(
            new Grid3D(151, 151, 151, -6.0f, -6.0f, -6.0f, 6.0f, 6.0f, 6.0f),
            Trap1D(151), Trap1D(151), Trap1D(151));
        Assert.NotNull(h);
        float[] e = h!.SeparableEnergies(4);
        Assert.Equal(0x3fbfec68u, Bits(e[0]));
        Assert.True(MathF.Abs(e[0] - 1.5f) / 1.5f < 1e-3f);
        Assert.Equal(Bits(e[1]), Bits(e[2]));
        Assert.Equal(Bits(e[2]), Bits(e[3]));
        Assert.True(MathF.Abs(e[1] - 2.5f) / 2.5f < 1e-3f);
    }

    /// <summary>
    /// M76: the separable ground state is normalised and carries the
    /// separable ground energy.
    /// </summary>
    [Fact]
    public void M76_TheGroundStateIsANormalisedProduct()
    {
        var h = Hamiltonian3D.Separable(
            new Grid3D(21, 21, 21, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
            Box1D(21), Box1D(21), Box1D(21));
        Assert.NotNull(h);
        var (energy, state) = h!.SeparableGroundState();
        Assert.Equal(0x416c6244u, Bits(energy));
        float norm = 0.0f;
        foreach (var a in state.Amplitudes)
        {
            norm += Complex32.NormSqr(a);
        }
        Assert.True(MathF.Abs(norm - 1.0f) < 1e-6f, $"the product must be normalised, got {norm}");
        Assert.Equal(Bits(energy), Bits(h.SeparableEnergies(1)[0]));
    }

    /// <summary>
    /// M77: the matrix-free operator agrees with its dense matrix, which
    /// is exactly Hermitian, and the tensor ground state matches the
    /// dense diagonalisation.
    /// </summary>
    [Fact]
    public void M77_TheOperatorMatchesItsDenseMatrix()
    {
        var h = Hamiltonian3D.Separable(
            new Grid3D(4, 4, 4, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
            Box1D(4), Box1D(4), Box1D(4));
        Assert.NotNull(h);
        ComplexMatrix dense = h!.DenseMatrix();
        Assert.Equal((8, 8), (dense.Rows, dense.Cols));
        // Exactly Hermitian: the columns come from one symmetric stencil.
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                Assert.Equal(Complex32.Conjugate(dense.Get(r, c)), dense.Get(c, r));
            }
        }
        // Matrix-free versus dense on the all-ones vector.
        var full = new Complex32[8];
        for (int i = 0; i < 8; i++)
        {
            full[i] = Complex32.One;
        }
        var viaApply = new Complex32[8];
        Assert.True(h.ApplyTo(full, viaApply));
        float worst = 0.0f;
        for (int r = 0; r < 8; r++)
        {
            var s = Complex32.Zero;
            for (int c = 0; c < 8; c++)
            {
                s += dense.Get(r, c) * full[c];
            }
            var d = viaApply[r] - s;
            worst = MathF.Max(worst, MathF.Max(MathF.Abs(d.Re), MathF.Abs(d.Im)));
        }
        Assert.True(worst < 1e-5f, $"matrix-free versus dense disagrees by {worst}");
        // The tensor product is the dense ground state.
        var (_, tensor) = h.SeparableGroundState();
        var eigen = QuantumLinalg.EigenHermitian(dense, 64, 1e-8f);
        var amps = new Complex32[8];
        for (int i = 0; i < 8; i++)
        {
            amps[i] = eigen.Eigenvectors.Get(i, 0);
        }
        var denseState = new QuantumState(amps);
        Assert.True(
            tensor.Fidelity(denseState) > 1.0f - 1e-6f,
            "the tensor product must be the dense ground state");
    }

    /// <summary>
    /// M78: the ring at zero flux keeps the free spectrum, and the link
    /// phase is exactly one.
    /// </summary>
    [Fact]
    public void M78_TheRingAtZeroFluxIsFree()
    {
        var ring = new RingHamiltonian(64, 1.0f, 1.0f, 1.0f, 0.0f);
        Assert.Equal(new Complex32(1.0f, 0.0f), ring.LinkPhase());
        float[] e = ring.Eigenvalues();
        Assert.Equal(64, e.Length);
        Assert.Equal(0xb5431663u, Bits(e[0]));
        Assert.True(MathF.Abs(e[0]) < 1e-3f, "the zero-flux ground state sits at zero");
        float c = ring.Hopping;
        float first = 2.0f * c * (1.0f - MathF.Cos(2.0f * MathF.PI / 64.0f));
        Assert.True(MathF.Abs(e[1] - first) < 1e-3f);
    }

    /// <summary>
    /// M79: a quarter flux quantum lifts the ground state to the
    /// Aharonov-Bohm value 2c(1 - cos(pi/128)).
    /// </summary>
    [Fact]
    public void M79_AQuarterFluxLiftsTheGroundState()
    {
        var ring = new RingHamiltonian(64, 1.0f, 1.0f, 1.0f, 0.25f);
        // The quarter-flux link phase is cos(pi/128) to the bit.
        Assert.Equal(0x3f7fec43u, Bits(ring.LinkPhase().Re));
        float[] e = ring.Eigenvalues();
        Assert.Equal(0x3cfffeb0u, Bits(e[0]));
        float formula = 2.0f * ring.Hopping * (1.0f - MathF.Cos(2.0f * MathF.PI * -0.25f / 64.0f));
        Assert.True(MathF.Abs(e[0] - formula) < 1e-3f);
    }

    /// <summary>
    /// M80: threading one more flux quantum changes nothing — the
    /// spectrum is periodic in the flux.
    /// </summary>
    [Fact]
    public void M80_TheSpectrumIsPeriodicInTheFlux()
    {
        float[] a = new RingHamiltonian(64, 1.0f, 1.0f, 1.0f, 0.3f).Eigenvalues();
        float[] b = new RingHamiltonian(64, 1.0f, 1.0f, 1.0f, 1.3f).Eigenvalues();
        Assert.Equal(a.Length, b.Length);
        float worst = 0.0f;
        for (int i = 0; i < a.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        }
        Assert.True(worst < 0.01f, $"flux periodicity breaks by {worst}");
    }
}
