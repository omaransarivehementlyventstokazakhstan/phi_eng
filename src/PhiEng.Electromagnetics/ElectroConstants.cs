using System.Numerics;

namespace PhiEng.Electromagnetics;

/// <summary>
/// Shared constants. The defaults are game-scale, exactly like gravity's
/// <c>big_g = 1</c>; the SI values are provided for callers that want
/// physical units. Mirrors <c>phieng-electromagnetics::constants</c>.
/// </summary>
public static class ElectroConstants
{
    /// <summary>Coulomb's constant in engine units.</summary>
    public const float CoulombK = 1.0f;
    /// <summary>The speed of light in engine units (cells per second).</summary>
    public const float LightSpeed = 1.0f;
    /// <summary>Coulomb's constant in SI units.</summary>
    public const float SiCoulombK = 8.9875517923e9f;
    /// <summary>The speed of light in SI units (m/s).</summary>
    public const float SiLightSpeed = 299792458.0f;
    /// <summary>The largest charge magnitude the engine accepts.</summary>
    public const float MaxCharge = 1.0e15f;
    /// <summary>The largest acceleration the kick may apply.</summary>
    public const float MaxAcceleration = 1.0e30f;
    /// <summary>The smallest material factor the grid accepts.</summary>
    public const float MinMaterial = 1.0e-6f;
    /// <summary>The largest grid cell count (in memory and in a snapshot).</summary>
    public const int MaxFieldCells = 262_144;

    /// <summary>Vacuum permittivity implied by a Coulomb constant.</summary>
    public static float Eps0(float coulombK) =>
        1.0f / (4.0f * MathF.PI * coulombK);

    /// <summary>Vacuum permeability implied by <paramref name="coulombK"/> and the speed.</summary>
    public static float Mu0(float coulombK, float lightSpeed) =>
        1.0f / (Eps0(coulombK) * lightSpeed * lightSpeed);
}
