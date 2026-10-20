namespace PhiEng.Dynamics;

/// <summary>
/// Thermal simulation tunables (2D). Mirrors Rust <c>ThermalConfig2D</c>.
/// Defaults are <see langword="disabled"/> so default trajectories are
/// bit-identical to pre-thermal worlds.
/// </summary>
public class ThermalConfig2
{
    /// <summary>Default body temperature in Kelvin (293.15 K).</summary>
    public const float DefaultTemperature = 293.15f;

    /// <summary>Stefan-Boltzmann constant sigma in W/(m²·K⁴).</summary>
    public const float DefaultStefanBoltzmann = 5.67e-8f;

    /// <summary>Upper temperature clamp (5000 K).</summary>
    public const float DefaultMaxTemperature = 5000.0f;

    /// <summary>Lower temperature clamp (0 K, absolute zero).</summary>
    public const float DefaultMinTemperature = 0.0f;

    /// <summary>
    /// Minimum conduction distance, guarding against a zero-length
    /// characteristic size.
    /// </summary>
    public const float MinConductionDistance = 1e-4f;

    /// <summary>
    /// Conduction scaling factor that makes thermal dynamics visible on
    /// game timescales (seconds) rather than minutes for realistic
    /// materials. Mirrors Rust <c>CONDUCTION_SCALE</c>.
    /// </summary>
    public const float ConductionScale = 500.0f;

    /// <summary>Master switch for thermal stepping.</summary>
    public bool Enabled { get; set; } = false;
    /// <summary>Ambient/environment temperature in Kelvin (default 293.15 K).</summary>
    public float AmbientTemperature { get; set; } = DefaultTemperature;
    /// <summary>Stefan–Boltzmann constant σ in W/(m²·K⁴).</summary>
    public float StefanBoltzmann { get; set; } = DefaultStefanBoltzmann;
    /// <summary>Convection coefficient h in W/(m²·K).</summary>
    public float ConvectionCoefficient { get; set; } = 10.0f;
    /// <summary>Enable body-to-body + body-to-environment radiation.</summary>
    public bool RadiationEnabled { get; set; } = true;
    /// <summary>Enable contact-driven conduction.</summary>
    public bool ConductionEnabled { get; set; } = true;
    /// <summary>Enable ambient convection.</summary>
    public bool ConvectionEnabled { get; set; } = true;
    /// <summary>Safety cap on temperature in Kelvin.</summary>
    public float MaxTemperature { get; set; } = DefaultMaxTemperature;
    /// <summary>Safety floor in Kelvin (absolute zero).</summary>
    public float MinTemperature { get; set; } = DefaultMinTemperature;
    /// <summary>Radiation cutoff as multiplier of combined bounding radii.</summary>
    public float RadiationRadiusFactor { get; set; } = 5.0f;
    /// <summary>Maximum temperature change per step in Kelvin.</summary>
    public float MaxTemperatureRate { get; set; } = 100.0f;

    /// <summary>
    /// Block body-to-body radiation when another body lies between the pair
    /// (line-of-sight test). Off by default: it costs an O(n) scan per pair
    /// and changes the exchanged heat, so it is opt-in.
    /// </summary>
    public bool RadiationOcclusion { get; set; } = false;

    /// <summary>
    /// Extra convection coefficient per unit of relative speed
    /// (W/(m²·K) per m/s). Models forced convection over a moving
    /// surface. 0 (the default) means free convection only.
    /// </summary>
    public float ForcedConvectionCoefficient { get; set; } = 0.0f;

    /// <summary>
    /// Linear thermal-expansion coefficient (1/K). Scales a body's shape
    /// by <c>1 + alpha * (T - reference)</c>. 0 (the default) means no
    /// expansion, so default trajectories stay bit-identical.
    /// </summary>
    public float ExpansionCoefficient { get; set; } = 0.0f;

    /// <summary>Reference temperature (K) the expansion is measured against.</summary>
    public float ExpansionReferenceTemperature { get; set; } = DefaultTemperature;

    /// <summary>
    /// Exchange model used by the body-to-body radiation. Mirrors the Rust
    /// <c>ViewFactorModel</c>.
    /// </summary>
    public ViewFactorModel ViewFactorModel { get; set; } = ViewFactorModel.Legacy;

    /// <summary>
    /// Sky temperature (K) used by the environment radiation.
    /// Zero or less means "use <see cref="AmbientTemperature"/>.
    /// </summary>
    public float SkyTemperature { get; set; }

    /// <summary>
    /// Ground temperature (K) used by the environment radiation.
    /// Zero or less means "use <see cref="AmbientTemperature"/>.
    /// </summary>
    public float GroundTemperature { get; set; }

    /// <summary>
    /// Fraction of the radiation that passes through an occluding body
    /// (0 = fully opaque, the historical behaviour; 1 = the occluder
    /// does not block at all).
    /// </summary>
    public float OccluderTransmittance { get; set; }

    /// <summary>
    /// True when the thermal phase would do work: the subsystem is
    /// enabled, at least one heat-transfer mechanism is on, and the
    /// ambient temperature is usable. Mirrors Rust
    /// <c>ThermalConfig2D::is_active</c>.
    /// </summary>
    public bool IsActive() =>
        Enabled
        && (RadiationEnabled || ConductionEnabled || ConvectionEnabled)
        && float.IsFinite(AmbientTemperature)
        && AmbientTemperature > 0.0f;

    /// <summary>
    /// True when every field still holds its default value. Used by the
    /// snapshot writer to keep default worlds byte-identical on older
    /// schemas.
    /// </summary>
    public bool IsDefault()
    {
        var d = new ThermalConfig2();
        return Enabled == d.Enabled
            && AmbientTemperature == d.AmbientTemperature
            && StefanBoltzmann == d.StefanBoltzmann
            && ConvectionCoefficient == d.ConvectionCoefficient
            && RadiationEnabled == d.RadiationEnabled
            && ConductionEnabled == d.ConductionEnabled
            && ConvectionEnabled == d.ConvectionEnabled
            && MaxTemperature == d.MaxTemperature
            && MinTemperature == d.MinTemperature
            && RadiationRadiusFactor == d.RadiationRadiusFactor
            && MaxTemperatureRate == d.MaxTemperatureRate
            && RadiationOcclusion == d.RadiationOcclusion
            && ForcedConvectionCoefficient == d.ForcedConvectionCoefficient
            && ExpansionCoefficient == d.ExpansionCoefficient
            && ExpansionReferenceTemperature == d.ExpansionReferenceTemperature
        && ViewFactorModel == d.ViewFactorModel
        && SkyTemperature == d.SkyTemperature
        && GroundTemperature == d.GroundTemperature
        && OccluderTransmittance == d.OccluderTransmittance;
    }
}

/// <summary>
/// Thermal simulation tunables (3D). Same field set as <see cref="ThermalConfig2"/>.
/// </summary>
public class ThermalConfig3 : ThermalConfig2 { }
