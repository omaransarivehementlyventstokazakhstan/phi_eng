using Xunit;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// Spin and particles (M57–M64). Mirrors
/// <c>crates/engine/tests/quantum_spin.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class QuantumSpinTests
{
    /// <summary>The worst entry-wise distance of two matrices.</summary>
    private static float Close(ComplexMatrix a, ComplexMatrix b)
    {
        int rows = System.Math.Max(a.Rows, b.Rows);
        int cols = System.Math.Max(a.Cols, b.Cols);
        float worst = 0.0f;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var d = a.Get(r, c) - b.Get(r, c);
                worst = MathF.Max(worst, MathF.Max(MathF.Abs(d.Re), MathF.Abs(d.Im)));
            }
        }
        return worst;
    }

    /// <summary>
    /// M57: the Pauli matrices square to I, anticommute, and cycle.
    /// </summary>
    [Fact]
    public void M57_ThePauliAlgebra()
    {
        var id = ComplexMatrix.Identity(2);
        foreach (var p in new[] { QuantumSpin.PauliX(), QuantumSpin.PauliY(), QuantumSpin.PauliZ() })
        {
            Assert.True(Close(p.MatMul(p), id) < 1e-6f, "every Pauli matrix squares to I");
            Assert.Equal(Complex32.Zero, p.Trace());
        }

        // sigma_x sigma_y = i sigma_z, cyclically.
        var xy = QuantumSpin.PauliX().MatMul(QuantumSpin.PauliY());
        var z = QuantumSpin.PauliZ();
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                var want = z.Get(r, c) * Complex32.I;
                var d = xy.Get(r, c) - want;
                Assert.True(
                    MathF.Abs(d.Re) < 1e-6f && MathF.Abs(d.Im) < 1e-6f,
                    "sigma_x sigma_y must be i sigma_z");
            }
        }

        // Distinct Paulis anticommute.
        var anti = QuantumFock.Anticommutator(QuantumSpin.PauliX(), QuantumSpin.PauliY());
        Assert.True(
            Close(anti, new ComplexMatrix(2, 2)) < 1e-6f,
            "distinct Paulis must anticommute");

        // The determinant is -1.
        var x = QuantumSpin.PauliX();
        var det = x.Get(0, 0) * x.Get(1, 1) - x.Get(0, 1) * x.Get(1, 0);
        Assert.True(MathF.Abs(det.Re + 1.0f) < 1e-6f, "det(sigma_x) must be -1");
        Assert.True(MathF.Abs(det.Im) < 1e-6f, "det(sigma_x) must be real");
    }

    /// <summary>
    /// M58: Stern-Gerlach measures spin up with the Born probability.
    /// </summary>
    [Fact]
    public void M58_SternGerlach()
    {
        // Spin up is always up; spin down never is.
        var rng = new DeterministicRng(5);
        var up = QuantumSpin.SpinUp();
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(true, QuantumSpin.MeasureSz(up, rng));
        }
        var down = QuantumSpin.SpinDown();
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(false, QuantumSpin.MeasureSz(down, rng));
        }

        // A superposition measures up with |alpha|²: (0.6, 0.8) goes up
        // 36% of the time, and the seeded stream pins the shots.
        Assert.True(
            MathF.Abs(QuantumSpin.UpProbability(new QuantumState(new[]
            {
                new Complex32(0.6f, 0.0f),
                new Complex32(0.8f, 0.0f),
            })) - 0.36f) < 1e-6f,
            "the up probability of (0.6, 0.8) must be 0.36");
        static bool?[] Shots(ulong seed)
        {
            var r = new DeterministicRng(seed);
            var outcomes = new bool?[32];
            for (int i = 0; i < 32; i++)
            {
                var s = new QuantumState(new[]
                {
                    new Complex32(0.6f, 0.0f),
                    new Complex32(0.8f, 0.0f),
                });
                outcomes[i] = QuantumSpin.MeasureSz(s, r);
            }
            return outcomes;
        }
        Assert.Equal(Shots(9), Shots(9));
        var mixed = Shots(9);
        Assert.Contains(true, mixed);
        Assert.Contains(false, mixed);

        // Not one spin, not a Stern-Gerlach experiment.
        Assert.Null(QuantumSpin.MeasureSz(new QuantumState(4), rng));
    }

    /// <summary>
    /// M59: the Rabi flop swaps the populations on a pi pulse.
    /// </summary>
    [Fact]
    public void M59_TheRabiFlop()
    {
        float omega = 1.5f;
        // A pi pulse: all the way over.
        var s = QuantumSpin.SpinUp();
        Assert.True(QuantumSpin.RabiEvolve(s, omega, MathF.PI / omega));
        Assert.True(
            MathF.Abs(Complex32.NormSqr(s.Amplitudes[1]) - 1.0f) < 1e-5f,
            "a pi pulse must flip spin up to spin down");
        // A 2 pi pulse: all the way back.
        Assert.True(QuantumSpin.RabiEvolve(s, omega, MathF.PI / omega));
        Assert.True(
            MathF.Abs(Complex32.NormSqr(s.Amplitudes[0]) - 1.0f) < 1e-5f,
            "a second pi pulse must flip it back");
        // Halfway: fifty-fifty.
        var half = QuantumSpin.SpinUp();
        float halfPi = (MathF.PI / 2.0f) / omega;
        Assert.True(QuantumSpin.RabiEvolve(half, omega, halfPi));
        Assert.True(
            MathF.Abs(QuantumSpin.UpProbability(half) - 0.5f) < 1e-5f,
            "a pi/2 pulse must split the populations");
        // Not one spin, not a Rabi flop.
        Assert.False(QuantumSpin.RabiEvolve(new QuantumState(4), omega, 1.0f));
    }

    /// <summary>
    /// M60: a uniform field splits the energies and precesses the spin.
    /// </summary>
    [Fact]
    public void M60_TheZeemanSplittingAndLarmorPrecession()
    {
        // Along z: the energies are ∓mu B / 2.
        var h = QuantumSpin.ZeemanHamiltonian(new[] { 0.0f, 0.0f, 2.0f }, 1.0f);
        var e = QuantumLinalg.EigenHermitian(h, 32, 1e-8f);
        Assert.True(
            MathF.Abs(e.Eigenvalues[0] + 1.0f) < 1e-5f,
            "the lower Zeeman level must be -mu B / 2");
        Assert.True(
            MathF.Abs(e.Eigenvalues[1] - 1.0f) < 1e-5f,
            "the upper Zeeman level must be +mu B / 2");

        // Larmor precession: |+x> under a z field rotates in the xy
        // plane at the Larmor frequency mu B, so after a quarter period
        // it points along +y.
        float muB = 2.0f;
        var s = QuantumSpin.SpinUp();
        Assert.True(QuantumGates.ApplySingle(s, QuantumGates.H(), 0));
        float quarter = (MathF.PI / 2.0f) / muB;
        var evolved = new QuantumState(new[] { s.Amplitudes[0], s.Amplitudes[1] });
        Assert.True(QuantumGates.ApplySingle(evolved, QuantumGates.Rz(muB * quarter), 0));
        var (x, y, z) = QuantumSpin.SpinVector(evolved)!.Value;
        Assert.True(MathF.Abs(x) < 1e-5f, $"after a quarter period x must vanish (x = {x})");
        Assert.True(
            MathF.Abs(y - 0.5f) < 1e-5f,
            $"after a quarter period the spin must point along +y (y = {y})");
        Assert.True(MathF.Abs(z) < 1e-5f, $"the precession must stay equatorial (z = {z})");

        // The spin vector of |up> is the north pole, halved units and all.
        var north = QuantumSpin.SpinVector(QuantumSpin.SpinUp())!.Value;
        Assert.True(
            north == (0.0f, 0.0f, 0.5f),
            "the spin vector of |up> must be the north pole halved");
        Assert.Null(QuantumSpin.SpinVector(new QuantumState(4)));
    }

    /// <summary>
    /// M61: the bosonic ladder climbs and descends by square roots.
    /// </summary>
    [Fact]
    public void M61_TheBosonicLadder()
    {
        int cutoff = 5;
        var a = QuantumFock.BosonAnnihilation(cutoff);
        var adag = QuantumFock.BosonCreation(cutoff);

        // a|n> = sqrt(n) |n-1>, checked by matrix action.
        for (int n = 1; n < cutoff; n++)
        {
            var s = QuantumState.Basis(cutoff, n);
            var op = QuantumOperators.Dense(a);
            var output = new Complex32[cutoff];
            op.ApplyTo(s.Amplitudes, output);
            s = new QuantumState(output);
            float want = MathF.Sqrt(n);
            Assert.True(
                MathF.Abs(s.Amplitudes[n - 1].Re - want) < 1e-6f,
                $"a|{n}> must be sqrt({n})|{n - 1}>");
            for (int i = 0; i < cutoff; i++)
            {
                if (i != n - 1)
                {
                    Assert.True(
                        Complex32.NormSqr(s.Amplitudes[i]) == 0.0f,
                        $"a|{n}> must have nothing outside |{n - 1}>");
                }
            }
        }
        // The vacuum annihilates to nothing.
        var vacuum = QuantumState.Basis(cutoff, 0);
        {
            var op = QuantumOperators.Dense(a);
            var output = new Complex32[cutoff];
            op.ApplyTo(vacuum.Amplitudes, output);
            vacuum = new QuantumState(output);
        }
        Assert.Equal(0.0f, vacuum.NormSqr());

        // The number operator counts: N|3> = 3|3>.
        var nOp = QuantumFock.NumberOperator(cutoff);
        Assert.Equal(new Complex32(3.0f, 0.0f), nOp.Get(3, 3));
        Assert.Equal(Complex32.Zero, nOp.Get(2, 3));

        // Creation is the dagger of annihilation, entry for entry.
        var dagger = a.Dagger();
        for (int r = 0; r < cutoff; r++)
        {
            for (int c = 0; c < cutoff; c++)
            {
                Assert.Equal(adag.Get(r, c), dagger.Get(r, c));
            }
        }
    }

    /// <summary>
    /// M62: fermions anticommute, which is the Pauli principle.
    /// </summary>
    [Fact]
    public void M62_TheFermionicAnticommutation()
    {
        var a = QuantumFock.FermionAnnihilation();
        var adag = QuantumFock.FermionCreation();
        var id = ComplexMatrix.Identity(2);
        var empty = new ComplexMatrix(2, 2);

        // {a, a†} = I.
        Assert.True(
            Close(QuantumFock.Anticommutator(a, adag), id) < 1e-6f,
            "creation and annihilation must anticommute to I");
        // a² = 0 and (a†)² = 0: no double occupancy, in either direction.
        Assert.True(
            Close(a.MatMul(a), empty) < 1e-6f,
            "annihilating twice must give zero");
        Assert.True(
            Close(adag.MatMul(adag), empty) < 1e-6f,
            "creating twice must give zero");

        // The exclusion principle as a state: creating on |1> vanishes.
        var s = new QuantumState(new[] { Complex32.Zero, Complex32.One });
        var op = QuantumOperators.Dense(adag);
        var output = new Complex32[2];
        op.ApplyTo(s.Amplitudes, output);
        s = new QuantumState(output);
        Assert.Equal(0.0f, s.NormSqr());

        // [a, a†] is not I for fermions (it is diag(1, -1)): the bosonic
        // relation fails exactly where it should.
        var comm = QuantumFock.Commutator(a, adag);
        Assert.True(MathF.Abs(comm.Get(0, 0).Re - 1.0f) < 1e-6f);
        Assert.True(MathF.Abs(comm.Get(1, 1).Re + 1.0f) < 1e-6f);
    }

    /// <summary>
    /// M63: the Fermi gas fills the lowest levels first.
    /// </summary>
    [Fact]
    public void M63_TheFermiGas()
    {
        // Four levels, two particles: the two lowest.
        Assert.Equal(
            2.0f,
            QuantumFock.FermiGasEnergy(new[] { 0.5f, 1.5f, 2.5f, 3.5f }, 2));
        // Unsorted input fills the same way.
        Assert.Equal(
            2.0f,
            QuantumFock.FermiGasEnergy(new[] { 3.5f, 0.5f, 2.5f, 1.5f }, 2));
        // Nobody home, or nowhere to go.
        Assert.Equal(0.0f, QuantumFock.FermiGasEnergy(new[] { 0.5f, 1.5f }, 0));
        Assert.Equal(0.0f, QuantumFock.FermiGasEnergy(new[] { 0.5f, 1.5f }, 3));
        // The harmonic ladder: 0.5 + 1.5 + 2.5 for three fermions.
        Assert.Equal(
            4.5f,
            QuantumFock.FermiGasEnergy(new[] { 0.5f, 1.5f, 2.5f, 3.5f }, 3));
    }

    /// <summary>
    /// M64: a spin rotation carries |up&gt; to |down&gt; up to a phase.
    /// </summary>
    [Fact]
    public void M64_SpinRotations()
    {
        // RY(π) takes |up> to |down>, exactly up to a sign.
        var s = QuantumSpin.SpinUp();
        Assert.True(QuantumGates.ApplySingle(s, QuantumGates.Ry(MathF.PI), 0));
        Assert.True(
            s.Fidelity(QuantumSpin.SpinDown()) > 1.0f - 1e-6f,
            "RY(pi)|up> must be |down> up to a phase");

        // The Bernstein-Vazirani oracle still works when the diffusion is
        // written with spin operators: a smoke test that the two
        // vocabularies compose (X is sigma_x, Z is sigma_z).
        Assert.Equal(0b101u, QuantumAlgorithms.BernsteinVazirani(0b101, 3, 7));
    }
}
