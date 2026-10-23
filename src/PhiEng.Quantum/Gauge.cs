using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// A charged particle on a ring threaded by magnetic flux: the
/// Aharonov-Bohm effect.
/// </summary>
/// <remarks>
/// <para>
/// This mirrors the Rust <c>phieng_quantum::gauge::RingHamiltonian</c>
/// entry for entry, so the two engines agree bit-for-bit.
/// </para>
/// <para>
/// The flux never touches the particle — the magnetic field vanishes on
/// the ring itself — and still the spectrum moves with it. The vector
/// potential enters the hopping amplitudes as Peierls phases: hopping
/// one way around the ring picks up <c>e^(-i theta)</c> per link and the
/// other way <c>e^(+i theta)</c>, with <c>theta = 2 pi f / N</c> for
/// <c>N</c> links and flux <c>f</c> in units of the flux quantum.
/// </para>
/// <para>
/// The energies are <c>E_m = 2c (1 - cos(2 pi (m - f) / N))</c>: the
/// whole spectrum slides with the flux and returns to itself every
/// quantum.
/// </para>
/// </remarks>
public sealed class RingHamiltonian
{
    private readonly int _n;
    private readonly float _hopping;
    private readonly float _flux;

    /// <summary>A ring of <paramref name="n"/> points of radius <paramref name="radius"/>, threaded by <paramref name="flux"/> flux quanta.</summary>
    /// <remarks>
    /// The hopping is the finite-difference kinetic coefficient
    /// <c>hbar^2 / (2 m dx^2)</c> with <c>dx = 2 pi R / n</c>.
    /// </remarks>
    public RingHamiltonian(int n, float radius, float mass, float hbar, float flux)
    {
        _n = System.Math.Max(n, 3);
        float dx = 2.0f * MathF.PI * radius / _n;
        _hopping = hbar * hbar / (2.0f * MathF.Max(mass, 1.17549435e-38f) * dx * dx);
        _flux = flux;
    }

    /// <summary>The number of points.</summary>
    public int N => _n;

    /// <summary>The hopping amplitude.</summary>
    public float Hopping => _hopping;

    /// <summary>The flux in flux quanta.</summary>
    public float Flux => _flux;

    /// <summary>The Peierls phase per link: <c>e^(-i 2 pi f / N)</c>.</summary>
    public Complex32 LinkPhase()
    {
        float theta = 2.0f * MathF.PI * _flux / _n;
        return Complex32.ExpI(-theta);
    }

    /// <summary>The Hamiltonian matrix: <c>2c</c> on the diagonal and the phased hopping on the ring, closed around the end.</summary>
    public ComplexMatrix Matrix()
    {
        var m = new ComplexMatrix(_n, _n);
        var phase = LinkPhase();
        for (int j = 0; j < _n; j++)
        {
            m.Set(j, j, new Complex32(2.0f * _hopping, 0.0f));
            int next = (j + 1) % _n;
            // Hopping forward picks up the phase; hopping back picks up
            // its conjugate, so the matrix stays Hermitian.
            m.Set(j, next, Complex32.Scale(phase, -_hopping));
            m.Set(next, j, Complex32.Scale(Complex32.Conjugate(phase), -_hopping));
        }
        return m;
    }

    /// <summary>The spectrum, in ascending order.</summary>
    public float[] Eigenvalues() => QuantumLinalg.EigenHermitian(Matrix(), 64, 1e-8f).Eigenvalues;
}
