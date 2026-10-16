using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Spring / simple-harmonic-motion field (2D). Mirrors Rust
/// <c>phieng_dynamics::ambient::SpringConfig2</c>. Disabled by default.
/// </summary>
public struct SpringConfig2
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>World-space rest point.</summary>
    public Vector2 Anchor;
    /// <summary>Hooke stiffness k &gt;= 0.</summary>
    public float Stiffness;
    /// <summary>Viscous damping c &gt;= 0.</summary>
    public float Damping;
    /// <summary>Rest-length leash (0 = unlimited).</summary>
    public float MaxDisplacement;

    /// <summary>No-op field.</summary>
    public static SpringConfig2 Default => new();
}

/// <summary>Aerodynamic drag field (2D). Mirrors Rust <c>DragConfig2</c>.</summary>
public struct DragConfig2
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>Linear (Stokes) coefficient &gt;= 0.</summary>
    public float Linear;
    /// <summary>Quadratic coefficient &gt;= 0.</summary>
    public float Quadratic;
    /// <summary>Air density multiplier &gt;= 0.</summary>
    public float AirDensity;
    /// <summary>Angular drag torque coefficient &gt;= 0.</summary>
    public float Angular;
    /// <summary>Ambient air velocity.</summary>
    public Vector2 Wind;

    /// <summary>No-op field (air density 1.0 keeps the struct at its Rust default).</summary>
    public static DragConfig2 Default => new() { AirDensity = 1.0f };
}

/// <summary>Deterministic turbulence field (2D). Mirrors Rust <c>TurbulenceConfig2</c>.</summary>
public struct TurbulenceConfig2
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>Peak acceleration magnitude &gt;= 0.</summary>
    public float Amplitude;
    /// <summary>Spatial frequency.</summary>
    public float Frequency;
    /// <summary>Temporal rate.</summary>
    public float TimeScale;
    /// <summary>Noise seed.</summary>
    public uint Seed;
    /// <summary>fBm octaves (clamped to 1..=8).</summary>
    public int Octaves;
    /// <summary>Curl-noise (divergence-free) mode.</summary>
    public bool DivergenceFree;

    /// <summary>No-op field (frequency/time-scale 1.0, 3 octaves).</summary>
    public static TurbulenceConfig2 Default => new() { Frequency = 1.0f, TimeScale = 1.0f, Octaves = 3 };
}

/// <summary>Aggregate ambient-force config (2D). Mirrors Rust <c>AmbientConfig2</c>.</summary>
public struct AmbientConfig2
{
    /// <summary>Spring / SHM field.</summary>
    public SpringConfig2 Spring;
    /// <summary>Aerodynamic drag field.</summary>
    public DragConfig2 Drag;
    /// <summary>Turbulence field.</summary>
    public TurbulenceConfig2 Turbulence;

    /// <summary>All sub-fields disabled.</summary>
    public static AmbientConfig2 Default => new()
    {
        Spring = SpringConfig2.Default,
        Drag = DragConfig2.Default,
        Turbulence = TurbulenceConfig2.Default,
    };

    /// <summary>True when at least one sub-field is enabled.</summary>
    public readonly bool IsActive() => Spring.Enabled || Drag.Enabled || Turbulence.Enabled;
}

/// <summary>Spring / SHM field (3D). Mirrors Rust <c>SpringConfig3</c>.</summary>
public struct SpringConfig3
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>World-space rest point.</summary>
    public Vector3 Anchor;
    /// <summary>Hooke stiffness k &gt;= 0.</summary>
    public float Stiffness;
    /// <summary>Viscous damping c &gt;= 0.</summary>
    public float Damping;
    /// <summary>Rest-length leash (0 = unlimited).</summary>
    public float MaxDisplacement;

    /// <summary>No-op field.</summary>
    public static SpringConfig3 Default => new();
}

/// <summary>Aerodynamic drag field (3D). Mirrors Rust <c>DragConfig3</c>.</summary>
public struct DragConfig3
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>Linear (Stokes) coefficient &gt;= 0.</summary>
    public float Linear;
    /// <summary>Quadratic coefficient &gt;= 0.</summary>
    public float Quadratic;
    /// <summary>Air density multiplier &gt;= 0.</summary>
    public float AirDensity;
    /// <summary>Angular drag torque coefficient &gt;= 0.</summary>
    public float Angular;
    /// <summary>Ambient air velocity.</summary>
    public Vector3 Wind;

    /// <summary>No-op field.</summary>
    public static DragConfig3 Default => new() { AirDensity = 1.0f };
}

/// <summary>Deterministic turbulence field (3D). Mirrors Rust <c>TurbulenceConfig3</c>.</summary>
public struct TurbulenceConfig3
{
    /// <summary>Master switch.</summary>
    public bool Enabled;
    /// <summary>Peak acceleration magnitude &gt;= 0.</summary>
    public float Amplitude;
    /// <summary>Spatial frequency.</summary>
    public float Frequency;
    /// <summary>Temporal rate.</summary>
    public float TimeScale;
    /// <summary>Noise seed.</summary>
    public uint Seed;
    /// <summary>fBm octaves (clamped to 1..=8).</summary>
    public int Octaves;
    /// <summary>Curl-noise (divergence-free) mode.</summary>
    public bool DivergenceFree;

    /// <summary>No-op field.</summary>
    public static TurbulenceConfig3 Default => new() { Frequency = 1.0f, TimeScale = 1.0f, Octaves = 3 };
}

/// <summary>Aggregate ambient-force config (3D). Mirrors Rust <c>AmbientConfig3</c>.</summary>
public struct AmbientConfig3
{
    /// <summary>Spring / SHM field.</summary>
    public SpringConfig3 Spring;
    /// <summary>Aerodynamic drag field.</summary>
    public DragConfig3 Drag;
    /// <summary>Turbulence field.</summary>
    public TurbulenceConfig3 Turbulence;

    /// <summary>All sub-fields disabled.</summary>
    public static AmbientConfig3 Default => new()
    {
        Spring = SpringConfig3.Default,
        Drag = DragConfig3.Default,
        Turbulence = TurbulenceConfig3.Default,
    };

    /// <summary>True when at least one sub-field is enabled.</summary>
    public readonly bool IsActive() => Spring.Enabled || Drag.Enabled || Turbulence.Enabled;
}
