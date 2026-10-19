namespace PhiEng.Dynamics;

/// <summary>
/// Thermal simulation tunables (2D). Mirrors Rust <c>ThermalConfig2D</c>.
/// Defaults are <see langword="disabled"/> so default trajectories are
/// bit-identical to pre-thermal worlds.
/// </summary>
public class ThermalConfig2
{
    /// <summary>Master switch for thermal stepping.</summary>
    public bool Enabled { get; set; } = false;
    /// <summary>Ambient/environment temperature in Kelvin (default 293.15 K).</summary>
    public float AmbientTemperature { get; set; } = 293.15f;
    /// <summary>Stefan–Boltzmann constant σ in W/(m²·K⁴).</summary>
    public float StefanBoltzmann { get; set; } = 5.67e-8f;
    /// <summary>Convection coefficient h in W/(m²·K).</summary>
    public float ConvectionCoefficient { get; set; } = 10.0f;
    /// <summary>Enable body-to-body + body-to-environment radiation.</summary>
    public bool RadiationEnabled { get; set; } = true;
    /// <summary>Enable contact-driven conduction.</summary>
    public bool ConductionEnabled { get; set; } = true;
    /// <summary>Enable ambient convection.</summary>
    public bool ConvectionEnabled { get; set; } = true;
    /// <summary>Safety cap on temperature in Kelvin.</summary>
    public float MaxTemperature { get; set; } = 5000.0f;
    /// <summary>Safety floor in Kelvin (absolute zero).</summary>
    public float MinTemperature { get; set; } = 0.0f;
    /// <summary>Radiation cutoff as multiplier of combined bounding radii.</summary>
    public float RadiationRadiusFactor { get; set; } = 5.0f;
    /// <summary>Maximum temperature change per step in Kelvin.</summary>
    public float MaxTemperatureRate { get; set; } = 100.0f;
}

/// <summary>
/// Thermal simulation tunables (3D). Same field set as <see cref="ThermalConfig2"/>.
/// </summary>
public class ThermalConfig3 : ThermalConfig2 { }
