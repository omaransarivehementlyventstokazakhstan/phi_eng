using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// Many-body and variational methods (M81–M88). Mirrors
/// <c>crates/engine/tests/quantum_advanced.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class QuantumAdvancedTests
{
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>
    /// M81: the periodic chain carries the cosine band from -2t to +2t.
    /// </summary>
    [Fact]
    public void M81_TheChainCarriesTheCosineBand()
    {
        float[] e = QuantumLinalg.EigenHermitian(QuantumLattice.TightBindingChain(32, 1.0f, true), 64, 1e-8f).Eigenvalues;
        Assert.Equal(32, e.Length);
        Assert.Equal(0xbffffffbu, Bits(e[0]));
        Assert.True(MathF.Abs(e[0] + 2.0f) < 1e-5f, "the band bottom sits at -2t");
        float width = e[31] - e[0];
        Assert.Equal(0x407ffff5u, Bits(width));
        Assert.True(MathF.Abs(width - 4.0f) < 1e-5f, "the bandwidth is 4t");
    }

    /// <summary>
    /// M82: the topological SSH chain shelters two near-zero edge
    /// modes; the trivial one leaves the gap empty.
    /// </summary>
    [Fact]
    public void M82_TheSshChainSheltersEdgeModes()
    {
        float[] e = QuantumLinalg.EigenHermitian(QuantumLattice.SshChain(16, 0.5f, 1.0f), 100, 1e-8f).Eigenvalues;
        Assert.Equal(32, e.Length);
        // The middle pair: split only by their overlap across the chain.
        Assert.Equal(0xb73fdd73u, Bits(e[15]));
        Assert.Equal(0x37408aaau, Bits(e[16]));
        Assert.True(MathF.Abs(e[15]) < 1e-3f && MathF.Abs(e[16]) < 1e-3f);
        // The bulk gap stays near 2(w - v).
        float gap = e[17] - e[14];
        Assert.Equal(0x3f857290u, Bits(gap));
        Assert.True(MathF.Abs(gap - 1.0f) / 1.0f < 5e-2f);
        // A mid-gap state lives on the ends: uniform would put 4/32 here.
        var solved = QuantumLinalg.EigenHermitian(QuantumLattice.SshChain(16, 0.5f, 1.0f), 100, 1e-8f);
        float ends = 0.0f;
        foreach (int i in new[] { 0, 1, 30, 31 })
        {
            ends += Complex32.NormSqr(solved.Eigenvectors.Get(i, 15));
        }
        Assert.True(ends > 0.5f, $"the edge mode must live on the ends, got {ends}");
        // The trivial phase (v > w) has no states near zero.
        float[] t = QuantumLinalg.EigenHermitian(QuantumLattice.SshChain(16, 1.0f, 0.5f), 100, 1e-8f).Eigenvalues;
        float minAbs = float.PositiveInfinity;
        foreach (float x in t)
        {
            minAbs = MathF.Min(minAbs, MathF.Abs(x));
        }
        Assert.Equal(0x3f03d80eu, Bits(minAbs));
        Assert.True(minAbs > 0.2f, "the trivial chain stays gapped");
    }

    /// <summary>
    /// M83: the transfer matrix flies over free space, tunnels under
    /// the barrier, crosses resonances at one, and stops at zero
    /// energy.
    /// </summary>
    [Fact]
    public void M83_TheTransferMatrixTunnelsAndFlies()
    {
        // Free space transmits everything.
        float free = QuantumScattering.Transmission(1.0f, 0.0f, 3.0f, 1.0f, 1.0f);
        Assert.Equal(0x3f800001u, Bits(free));
        Assert.True(MathF.Abs(free - 1.0f) < 1e-5f);
        // Tunnelling matches the analytic barrier formula.
        float t = QuantumScattering.Transmission(1.0f, 2.0f, 3.0f, 1.0f, 1.0f);
        Assert.Equal(0x3a586d1fu, Bits(t));
        float kappaWidth = MathF.Sqrt(2.0f) * 3.0f;
        float sinh = (MathF.Exp(kappaWidth) - MathF.Exp(-kappaWidth)) / 2.0f;
        float analytic = 1.0f / (1.0f + sinh * sinh);
        Assert.True(MathF.Abs(t - analytic) / analytic < 1e-4f);
        // At k'a = pi the barrier turns transparent.
        float resonant = 2.0f + MathF.PI * MathF.PI / 18.0f;
        float r = QuantumScattering.Transmission(resonant, 2.0f, 3.0f, 1.0f, 1.0f);
        Assert.Equal(0x3f800000u, Bits(r));
        Assert.True(MathF.Abs(r - 1.0f) < 1e-6f);
        // Zero energy does not propagate.
        Assert.Equal(0.0f, QuantumScattering.Transmission(0.0f, 2.0f, 3.0f, 1.0f, 1.0f));
        Assert.Equal(0.0f, QuantumScattering.Transmission(-1.0f, 2.0f, 3.0f, 1.0f, 1.0f));
    }

    /// <summary>
    /// M84: Hartree-Fock on the Hubbard dimer is variational — above
    /// the exact answer by the correlation energy it cannot see.
    /// </summary>
    [Fact]
    public void M84_HartreeFockIsVariationalOnTheDimer()
    {
        float[] exact = QuantumManyBody.HubbardDimerExact(1.0f, 2.0f);
        Assert.Equal(0xbf9e3779u, Bits(exact[0]));
        Assert.True(MathF.Abs(exact[0] - (1.0f - MathF.Sqrt(5.0f))) < 1e-6f);
        var (rhf, density) = QuantumManyBody.HubbardDimerRhf(1.0f, 2.0f, 50);
        Assert.Equal(0xbf800000u, Bits(rhf));
        Assert.True(
            rhf >= exact[0] - 1e-6f,
            "mean-field must stay above the exact ground state");
        Assert.True(
            rhf - exact[0] > 0.1f,
            "mean-field must miss correlation energy");
        // The dimer stays symmetric: one electron per site.
        Assert.True(MathF.Abs(density.Get(0, 0).Re - 1.0f) < 1e-6f);
        Assert.True(MathF.Abs(density.Get(1, 1).Re - 1.0f) < 1e-6f);
    }

    /// <summary>
    /// M85: half of the Ising ground state is entangled, and keeping
    /// two Schmidt states drops almost nothing — the DMRG truncation.
    /// </summary>
    [Fact]
    public void M85_TheEntanglementSpectrumTruncatesCleanly()
    {
        var solved = QuantumLinalg.EigenHermitian(QuantumManyBody.TfiChain(4, 1.0f, 1.0f), 100, 1e-8f);
        var amps = new Complex32[16];
        for (int i = 0; i < 16; i++)
        {
            amps[i] = solved.Eigenvectors.Get(i, 0);
        }
        var ground = new QuantumState(amps);
        float[] spectrum = QuantumManyBody.EntanglementSpectrum(ground, new[] { 0, 1 }, 4);
        Assert.Equal(4, spectrum.Length);
        Assert.Equal(0x3f6aea49u, Bits(spectrum[0]));
        Assert.Equal(0x3da89780u, Bits(spectrum[1]));
        Assert.Equal(0x3821f4f7u, Bits(spectrum[2]));
        Assert.Equal(0x366a4083u, Bits(spectrum[3]));
        // A spectrum, largest first, summing to one.
        for (int i = 0; i + 1 < spectrum.Length; i++)
        {
            Assert.True(spectrum[i] >= spectrum[i + 1]);
        }
        float sum = 0.0f;
        foreach (float x in spectrum)
        {
            sum += x;
        }
        Assert.True(MathF.Abs(sum - 1.0f) < 1e-5f);
        // Entangled (no product state), yet two states nearly suffice.
        Assert.True(spectrum[0] < 0.99f);
        Assert.True(spectrum[2] + spectrum[3] < 1e-3f);
    }

    /// <summary>
    /// M86: the parameter shift differentiates a Pauli rotation
    /// exactly.
    /// </summary>
    [Fact]
    public void M86_TheParameterShiftIsExact()
    {
        var z = new ComplexMatrix(2, 2);
        z.Set(0, 0, new Complex32(1.0f, 0.0f));
        z.Set(1, 1, new Complex32(-1.0f, 0.0f));
        var zop = QuantumOperators.Dense(z);
        float Expectation(float theta)
        {
            var s = QuantumState.Basis(2, 0);
            QuantumGates.ApplySingle(s, QuantumGates.Rx(theta), 0);
            return s.Expectation(zop).Re;
        }
        float gradient = QuantumVariational.ParameterShiftGradient(Expectation, 0.7f);
        Assert.Equal(0xbf24eb72u, Bits(gradient));
        Assert.True(MathF.Abs(gradient + MathF.Sin(0.7f)) < 1e-6f);
    }

    /// <summary>
    /// M87: the VQE loop descends to the exact ground energy.
    /// </summary>
    [Fact]
    public void M87_TheVqeLoopFindsTheGroundState()
    {
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 1, new Complex32(1.0f, 0.0f));
        m.Set(1, 0, new Complex32(1.0f, 0.0f));
        m.Set(0, 0, new Complex32(0.5f, 0.0f));
        m.Set(1, 1, new Complex32(-0.5f, 0.0f));
        var h = QuantumOperators.Dense(m);
        float Energy(float[] p)
        {
            var s = QuantumState.Basis(2, 0);
            QuantumGates.ApplySingle(s, QuantumGates.Rx(p[0]), 0);
            QuantumGates.ApplySingle(s, QuantumGates.Ry(p[1]), 0);
            return s.Expectation(h).Re;
        }
        float[] Gradient(float[] p) => new[]
        {
            QuantumVariational.ParameterShiftGradient(t => Energy(new[] { t, p[1] }), p[0]),
            QuantumVariational.ParameterShiftGradient(t => Energy(new[] { p[0], t }), p[1]),
        };
        var (_, found) = QuantumVariational.Descend(new[] { 0.3f, -0.2f }, 200, 0.3f, Energy, Gradient);
        Assert.Equal(0xbf8f1bbdu, Bits(found));
        Assert.True(MathF.Abs(found + MathF.Sqrt(1.25f)) < 1e-3f);
    }

    /// <summary>
    /// M88: a qubit looped around a cone picks up minus half the solid
    /// angle.
    /// </summary>
    [Fact]
    public void M88_TheQubitLoopPicksUpTheBerryPhase()
    {
        float gamma = RingHamiltonian.BerryPhase(MathF.PI / 3.0f, 64);
        Assert.Equal(0xbfc8f0d8u, Bits(gamma));
        Assert.True(MathF.Abs(gamma + MathF.PI / 2.0f) < 2e-3f);
    }
}
