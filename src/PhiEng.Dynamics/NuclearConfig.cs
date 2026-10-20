namespace PhiEng.Dynamics;

/// <summary>
/// A nuclide the engine knows about. Mirrors the Rust <c>Isotope</c> in
/// crates/dynamics/src/nuclear.rs.
/// </summary>
public sealed class Isotope
{
    /// <summary>
    /// The half-life in seconds. Zero or less means the nuclide is
    /// stable: it never decays.
    /// </summary>
    public float HalfLife { get; set; }

    /// <summary>The energy released per decay, in joules.</summary>
    public float DecayEnergy { get; set; }

    /// <summary>
    /// The fraction of the decay energy that leaves as ionizing
    /// radiation; the rest becomes heat. Clamped to 0..1.
    /// </summary>
    public float IonizationFraction { get; set; }

    /// <summary>
    /// The nuclide produced by neutron capture; zero means the capture
    /// leaves the nuclide unchanged.
    /// </summary>
    public ushort Daughter { get; set; }

    /// <summary>
    /// The neutron absorption cross-section, in barns
    /// (1 barn = 1e-28 m^2).
    /// </summary>
    public float AbsorptionCrossSection { get; set; }

    /// <summary>The fission cross-section, in barns.</summary>
    public float FissionCrossSection { get; set; }

    /// <summary>The neutrons released per fission.</summary>
    public float NeutronYield { get; set; }

    /// <summary>Whether the nuclide can fission.</summary>
    public bool IsFissile { get; set; }

    /// <summary>A nuclide with no properties: stable and inert.</summary>
    public static Isotope Default => new();

    /// <summary>
    /// The decay constant <c>lambda = ln(2) / half_life</c>, in 1/s. A
    /// stable nuclide (half-life &lt;= 0) has the constant zero.
    /// </summary>
    public float DecayConstant() => DecayConstantOf(HalfLife);

    /// <summary>
    /// The decay constant <c>lambda = ln(2) / half_life</c>, in 1/s.
    /// </summary>
    /// <param name="halfLife">The half-life in seconds.</param>
    public static float DecayConstantOf(float halfLife)
    {
        if (halfLife <= 0.0f || !float.IsFinite(halfLife))
        {
            return 0.0f;
        }
        // ln(2) rounded to float - the same constant the Rust core uses.
        return 0.6931472f / halfLife;
    }
}

/// <summary>
/// The nuclear simulation settings. Mirrors the Rust
/// <c>NuclearConfig</c> in crates/dynamics/src/nuclear.rs.
/// </summary>
public sealed class NuclearConfig
{
    /// <summary>
    /// Master switch. Off by default, so a default world is
    /// bit-identical to a pre-nuclear one.
    /// </summary>
    public bool Enabled { get; set; }

    /// <summary>
    /// The nuclide table. Index zero is the "no nuclide" placeholder;
    /// the real nuclides start at index one.
    /// </summary>
    public List<Isotope> Isotopes { get; set; } = new() { new Isotope() };

    /// <summary>
    /// Nuclear seconds per simulation second. A half-life of years
    /// becomes visible with a large value (10^5 turns a 5730-year
    /// carbon-14 half-life into half a minute of simulation).
    /// </summary>
    public float TimeScale { get; set; } = 1.0f;

    /// <summary>The energy released per fission, in joules.</summary>
    public float FissionEnergy { get; set; } = 3.204e-11f;

    /// <summary>The background neutron flux (a dimensionless rate).</summary>
    public float AmbientFlux { get; set; }

    /// <summary>
    /// The fission rate constant: how fast a unit flux over a unit
    /// cross-section fissions a unit of fuel.
    /// </summary>
    public float FissionRate { get; set; } = 1.0f;

    /// <summary>
    /// The capture rate constant: how fast a unit flux over a unit
    /// absorption cross-section captures a neutron (and turns the nuclide
    /// into its daughter).
    /// </summary>
    public float ActivationRate { get; set; } = 1.0f;

    /// <summary>The fuel consumed per fission.</summary>
    public float FuelPerFission { get; set; } = 1e-6f;

    /// <summary>
    /// The range within which free neutrons are shared, in metres.
    /// </summary>
    public float NeutronRange { get; set; } = 1.0f;

    /// <summary>
    /// The range within which ionizing radiation is deposited, in
    /// metres. Radiation that reaches no body is simply lost.
    /// </summary>
    public float DoseRange { get; set; } = 1.0f;

    /// <summary>The fraction of the fission neutrons that are delayed.</summary>
    public float DelayedNeutronFraction { get; set; } = 0.006f;

    /// <summary>The delay of the delayed neutrons, in nuclear seconds.</summary>
    public float DelayedNeutronDelay { get; set; } = 0.5f;

    /// <summary>The maximum activity, in decays per second.</summary>
    public float MaxActivity { get; set; } = 1e30f;

    /// <summary>The dose accumulated per joule of ionizing radiation.</summary>
    public float DosePerJoule { get; set; } = 1.0f;

    /// <summary>The maximum accumulated dose.</summary>
    public float MaxDose { get; set; } = 1e30f;

    /// <summary>True when the nuclear phase should run.</summary>
    public bool IsActive()
    {
        return Enabled
            && float.IsFinite(TimeScale)
            && TimeScale > 0.0f
            && Isotopes.Count > 1;
    }

    /// <summary>
    /// True when every field still holds its default value. Used by the
    /// snapshot writer to keep default worlds byte-identical on older
    /// schemas.
    /// </summary>
    public bool IsDefault()
    {
        var d = new NuclearConfig();
        if (Enabled != d.Enabled
            || TimeScale != d.TimeScale
            || FissionEnergy != d.FissionEnergy
            || AmbientFlux != d.AmbientFlux
            || FissionRate != d.FissionRate
            || ActivationRate != d.ActivationRate
            || FuelPerFission != d.FuelPerFission
            || NeutronRange != d.NeutronRange
            || DoseRange != d.DoseRange
            || DelayedNeutronFraction != d.DelayedNeutronFraction
            || DelayedNeutronDelay != d.DelayedNeutronDelay
            || MaxActivity != d.MaxActivity
            || DosePerJoule != d.DosePerJoule
            || MaxDose != d.MaxDose
            || Isotopes.Count != d.Isotopes.Count)
        {
            return false;
        }
        for (int i = 0; i < Isotopes.Count; i++)
        {
            var a = Isotopes[i];
            var b = d.Isotopes[i];
            if (a.HalfLife != b.HalfLife
                || a.DecayEnergy != b.DecayEnergy
                || a.IonizationFraction != b.IonizationFraction
                || a.Daughter != b.Daughter
                || a.AbsorptionCrossSection != b.AbsorptionCrossSection
                || a.FissionCrossSection != b.FissionCrossSection
                || a.NeutronYield != b.NeutronYield
                || a.IsFissile != b.IsFissile)
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>The nuclide with the given index, if it exists.</summary>
    /// <param name="index">The nuclide index.</param>
    public Isotope? GetIsotope(ushort index)
    {
        return index < Isotopes.Count ? Isotopes[index] : null;
    }
}

/// <summary>
/// The per-body nuclear state. Mirrors the Rust <c>NuclearState</c>.
/// </summary>
public sealed class NuclearState
{
    /// <summary>
    /// The nuclide index. Zero means "no nuclide": the body is not
    /// radioactive and takes no part in the nuclear phase.
    /// </summary>
    public ushort Isotope { get; set; }

    /// <summary>The activity, in decays per second.</summary>
    public float Activity { get; set; }

    /// <summary>The remaining fissile fuel, 0..1.</summary>
    public float Fuel { get; set; }

    /// <summary>The free neutron population.</summary>
    public float Neutrons { get; set; }

    /// <summary>The delayed neutron population.</summary>
    public float DelayedNeutrons { get; set; }

    /// <summary>The accumulated ionizing dose.</summary>
    public float Dose { get; set; }

    /// <summary>
    /// The accumulated neutron-capture progress. A body with a daughter
    /// nuclide captures a neutron when this reaches one, and then becomes
    /// that daughter.
    /// </summary>
    public float CaptureProgress { get; set; }

    /// <summary>
    /// True when every field still holds its default value.
    /// </summary>
    public bool IsDefault()
    {
        return Isotope == 0
            && Activity == 0.0f
            && Fuel == 0.0f
            && Neutrons == 0.0f
            && DelayedNeutrons == 0.0f
            && Dose == 0.0f
            && CaptureProgress == 0.0f;
    }
}

/// <summary>
/// The per-step nuclear scratch. Mirrors the Rust <c>NuclearScratch</c>.
/// </summary>
public sealed class NuclearScratch
{
    internal float[] Heat = Array.Empty<float>();
    internal float[] Radiation = Array.Empty<float>();
    internal float[] Flux = Array.Empty<float>();
    internal float[] DoseWeights = Array.Empty<float>();

    /// <summary>The heat produced at a body this step, in joules.</summary>
    /// <param name="index">The body index.</param>
    public float HeatAt(int index) => index < Heat.Length ? Heat[index] : 0.0f;

    /// <summary>The ionizing radiation produced at a body this step, in joules.</summary>
    /// <param name="index">The body index.</param>
    public float RadiationAt(int index) => index < Radiation.Length ? Radiation[index] : 0.0f;
}

/// <summary>
/// The per-world nuclear state and configuration. Mirrors the Rust
/// <c>NuclearState2</c>.
/// </summary>
public sealed class NuclearState2
{
    /// <summary>The nuclear configuration.</summary>
    public NuclearConfig Config { get; set; } = new();

    /// <summary>The per-step scratch.</summary>
    public NuclearScratch Scratch { get; } = new();
}
