using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// Scattering off a rectangular barrier: the transfer matrix.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::scattering</c> module entry
/// for entry, so the two engines agree bit-for-bit.
/// </remarks>
public static class QuantumScattering
{
    /// <summary>The interface matrix from wavenumber <paramref name="k1"/> to <paramref name="k2"/>.</summary>
    private static ComplexMatrix Interface(Complex32 k1, Complex32 k2)
    {
        var ratio = Complex32.Divide(k1, k2);
        var one = Complex32.One;
        var half = new Complex32(0.5f, 0.0f);
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, half * (one + ratio));
        m.Set(0, 1, half * (one - ratio));
        m.Set(1, 0, half * (one - ratio));
        m.Set(1, 1, half * (one + ratio));
        return m;
    }

    /// <summary>The propagation matrix over <paramref name="length"/> at wavenumber <paramref name="k"/>.</summary>
    /// <remarks>
    /// The exponent is <c>i k L</c>, not <c>k L</c>: for an imaginary
    /// wavenumber the forward wave decays and the backward one grows,
    /// and writing the conjugate instead would decay them both.
    /// </remarks>
    private static ComplexMatrix Propagate(Complex32 k, float length)
    {
        var fwd = new Complex32(-k.Im * length, k.Re * length);
        var bwd = new Complex32(k.Im * length, -k.Re * length);
        var m = new ComplexMatrix(2, 2);
        m.Set(0, 0, Complex32.Exp(fwd));
        m.Set(1, 1, Complex32.Exp(bwd));
        return m;
    }

    /// <summary>The transmission probability through a rectangular barrier of height <paramref name="potential"/> and <paramref name="width"/>, at <paramref name="energy"/>, for a particle of <paramref name="mass"/> (with hbar <paramref name="hbar"/>).</summary>
    /// <remarks>
    /// Zero energy does not propagate, so it transmits nothing;
    /// anything else tunnels below the barrier top and flies above it,
    /// with the resonant energies crossing at exactly one.
    /// </remarks>
    public static float Transmission(float energy, float potential, float width, float mass, float hbar)
    {
        if (energy <= 0.0f)
        {
            return 0.0f;
        }
        float m = MathF.Max(mass, 1.17549435e-38f);
        float h = MathF.Max(hbar, 1.17549435e-38f);
        var k0 = Complex32.Scale(
            Complex32.Sqrt(new Complex32(2.0f * m * energy, 0.0f)), 1.0f / h);
        var ki = Complex32.Scale(
            Complex32.Sqrt(new Complex32(2.0f * m * (energy - potential), 0.0f)), 1.0f / h);
        ComplexMatrix total = Interface(k0, ki).MatMul(Propagate(ki, width)).MatMul(Interface(ki, k0));
        var m11 = total.Get(1, 1);
        return 1.0f / Complex32.NormSqr(m11);
    }
}
