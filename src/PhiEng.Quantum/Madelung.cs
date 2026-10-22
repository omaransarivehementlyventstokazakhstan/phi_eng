using PhiEng.Math;

namespace PhiEng.Quantum;

/// <summary>
/// The Madelung fluid: the hydrodynamic face of a wavefunction.
/// </summary>
/// <remarks>
/// This mirrors the Rust <c>phieng_quantum::madelung</c> module
/// operation for operation, so the two engines agree bit-for-bit.
/// </remarks>
public static class Madelung
{
    /// <summary>
    /// The floor under a density before it is square-rooted.
    /// </summary>
    public const float DensityFloor = 1e-12f;

    /// <summary>
    /// The quantum potential Q = -(hbar^2 / 2m) (lap s) / s with
    /// s = sqrt(rho), on a periodic grid.
    /// </summary>
    /// <returns>False when the buffers disagree or the parameters are unusable.</returns>
    public static bool QuantumPotential(
        float[] rho, float dx, float hbar, float mass, float[] output)
    {
        int n = rho.Length;
        if (output.Length < n || n == 0)
        {
            return false;
        }
        if (!float.IsFinite(dx) || dx <= 0.0f || !float.IsFinite(hbar) || hbar < 0.0f)
        {
            return false;
        }
        if (!float.IsFinite(mass) || mass <= 0.0f)
        {
            return false;
        }
        float coeff = -(hbar * hbar) / (2.0f * mass);
        for (int i = 0; i < n; i++)
        {
            float sHere = MathF.Sqrt(MathF.Max(rho[i], 0.0f));
            if (sHere <= DensityFloor)
            {
                output[i] = 0.0f;
                continue;
            }
            float sLeft = MathF.Sqrt(MathF.Max(rho[(i + n - 1) % n], 0.0f));
            float sRight = MathF.Sqrt(MathF.Max(rho[(i + 1) % n], 0.0f));
            float lap = (sLeft - sHere * 2.0f + sRight) / (dx * dx);
            output[i] = coeff * lap / sHere;
        }
        return true;
    }
}

/// <summary>
/// A one-dimensional Madelung fluid on a periodic grid.
/// </summary>
public sealed class QuantumFluid1D
{
    private readonly int _n;
    private readonly float _dx;
    private readonly float _hbar;
    private readonly float _mass;
    private readonly float[] _rho;
    private readonly float[] _vel;

    /// <summary>
    /// A fluid of <paramref name="n"/> points with spacing
    /// <paramref name="dx"/>, uniform density <paramref name="rho0"/>
    /// and zero velocity.
    /// </summary>
    public QuantumFluid1D(int n, float dx, float hbar, float mass, float rho0)
    {
        _n = n;
        _dx = dx;
        _hbar = hbar;
        _mass = mass;
        _rho = new float[n];
        _vel = new float[n];
        float r = MathF.Max(rho0, 0.0f);
        for (int i = 0; i < n; i++)
        {
            _rho[i] = r;
        }
    }

    /// <summary>The number of points.</summary>
    public int N => _n;

    /// <summary>The spacing.</summary>
    public float Dx => _dx;

    /// <summary>The density.</summary>
    public float[] Rho => _rho;

    /// <summary>The velocity.</summary>
    public float[] Vel => _vel;

    /// <summary>
    /// Replaces the density. Returns false when the slice is short; a
    /// negative entry is clamped to zero rather than admitted.
    /// </summary>
    public bool SetDensity(float[] values)
    {
        if (values.Length < _n)
        {
            return false;
        }
        for (int i = 0; i < _n; i++)
        {
            _rho[i] = MathF.Max(values[i], 0.0f);
        }
        return true;
    }

    /// <summary>
    /// Replaces the velocity. Returns false when the slice is short.
    /// </summary>
    public bool SetVelocity(float[] values)
    {
        if (values.Length < _n)
        {
            return false;
        }
        Array.Copy(values, _vel, _n);
        return true;
    }

    /// <summary>The total mass: the density integrated over the grid.</summary>
    public float MassTotal()
    {
        float sum = 0.0f;
        foreach (float r in _rho)
        {
            sum += r;
        }
        return sum * _dx;
    }

    /// <summary>
    /// Advances the fluid by one explicit Euler step.
    /// </summary>
    /// <returns>
    /// False for a non-positive step, a mismatched scratch, or unusable
    /// parameters, leaving the fluid alone in that case.
    /// </returns>
    public bool Step(float dt, MadelungScratch scratch)
    {
        int n = _n;
        if (n == 0
            || !float.IsFinite(dt)
            || dt <= 0.0f
            || scratch.RhoNext.Length < n
            || scratch.VelNext.Length < n
            || scratch.Potential.Length < n)
        {
            return false;
        }
        if (!Madelung.QuantumPotential(_rho, _dx, _hbar, _mass, scratch.Potential))
        {
            return false;
        }
        float dx = _dx;
        for (int i = 0; i < n; i++)
        {
            int left = (i + n - 1) % n;
            int right = (i + 1) % n;
            float fluxLeft = _rho[left] * _vel[left];
            float fluxRight = _rho[right] * _vel[right];
            float nextRho = _rho[i] - dt * (fluxRight - fluxLeft) / (2.0f * dx);
            scratch.RhoNext[i] = MathF.Max(nextRho, 0.0f);
            float advect = _vel[i] * (_vel[right] - _vel[left]) / (2.0f * dx);
            float push = (scratch.Potential[right] - scratch.Potential[left]) / (2.0f * dx);
            scratch.VelNext[i] = _vel[i] - dt * (advect + push);
        }
        Array.Copy(scratch.RhoNext, _rho, n);
        Array.Copy(scratch.VelNext, _vel, n);
        return true;
    }
}

/// <summary>
/// Scratch for a fluid step, so that stepping allocates nothing.
/// </summary>
public sealed class MadelungScratch
{
    /// <summary>The next density.</summary>
    public readonly float[] RhoNext;

    /// <summary>The next velocity.</summary>
    public readonly float[] VelNext;

    /// <summary>The quantum potential of the current density.</summary>
    public readonly float[] Potential;

    /// <summary>Scratch for a fluid of <paramref name="n"/> points.</summary>
    public MadelungScratch(int n)
    {
        RhoNext = new float[n];
        VelNext = new float[n];
        Potential = new float[n];
    }
}
