namespace PhiEng.Dynamics;

/// <summary>
/// Continuous collision detection config, shared by 2D and 3D worlds.
/// Mirrors Rust <c>phieng_dynamics::ccd::CcdConfig</c>: when enabled,
/// each step splits into substeps so no body travels more than
/// <see cref="MaxMovement"/> per substep. Default is a no-op.
/// </summary>
public struct CcdConfig
{
    /// <summary>Master switch. <c>false</c> (default) = single legacy substep.</summary>
    public bool Enabled;
    /// <summary>Maximum distance any body may travel within one substep.</summary>
    public float MaxMovement;
    /// <summary>Hard cap on substeps per world step (cost guard, &gt;= 1).</summary>
    public int MaxSubsteps;

    /// <summary>Default config: disabled, 0.1 movement, 8 max substeps.</summary>
    public static CcdConfig Default => new() { MaxMovement = 0.1f, MaxSubsteps = 8 };

    /// <summary>True for the default (legacy) configuration.</summary>
    public readonly bool IsDefault() => !Enabled && MaxMovement == 0.1f && MaxSubsteps == 8;

    /// <summary>True when substepping can actually engage.</summary>
    public readonly bool IsActive() => Enabled && MaxSubsteps > 1 && float.IsFinite(MaxMovement) && MaxMovement > 0.0f;

    /// <summary>Number of substeps for a step of length <paramref name="dt"/> given the peak effective body speed.</summary>
    public readonly uint Substeps(float dt, float peakSpeed)
    {
        if (!IsActive() || dt <= 0.0f || !float.IsFinite(peakSpeed) || peakSpeed <= 0.0f)
        {
            return 1;
        }
        long need = (long)System.Math.Ceiling(peakSpeed * dt / MaxMovement);
        return (uint)System.Math.Clamp(need, 1L, (long)System.Math.Max(1, MaxSubsteps));
    }
}
