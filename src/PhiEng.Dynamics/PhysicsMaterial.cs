namespace PhiEng.Dynamics;

/// <summary>
/// Surface and mass properties of a rigid body. Mirrors the Rust
/// <c>PhysicsMaterial</c> in crates/dynamics/src/material.rs (keep defaults
/// and combine rules in sync).
/// A reference type: <see cref="RigidBody2.SetMaterial"/> stores a defensive
/// copy so bodies never alias the same instance.
/// </summary>
public class PhysicsMaterial
{
    /// <summary>Fallback friction for non-finite or negative input.</summary>
    public const float DefaultFriction = 0.4f;
    /// <summary>Fallback restitution for non-finite input.</summary>
    public const float DefaultRestitution = 0.1f;
    /// <summary>Fallback density for non-finite or non-positive input.</summary>
    public const float DefaultDensity = 1.0f;
    /// <summary>Fallback specific heat capacity J/(kg·K).</summary>
    public const float DefaultSpecificHeat = 900.0f;
    /// <summary>Fallback thermal conductivity W/(m·K).</summary>
    public const float DefaultThermalConductivity = 1.0f;
    /// <summary>Fallback surface emissivity 0..1.</summary>
    public const float DefaultEmissivity = 0.9f;

    /// <summary>Minimum sane density (guards against zero/negative mass).</summary>
    public const float MinDensity = 1e-6f;

    /// <summary>Sliding resistance, typically 0.0 (frictionless) to 1.0+ (grippy).</summary>
    public float Friction { get; set; }
    /// <summary>Bounciness, 0.0 (dead) to 1.0 (perfectly elastic).</summary>
    public float Restitution { get; set; }
    /// <summary>Mass per unit area (2D); derives mass and inertia.</summary>
    public float Density { get; set; }
    /// <summary>Specific heat capacity J/(kg·K).</summary>
    public float SpecificHeat { get; set; }
    /// <summary>Thermal conductivity W/(m·K).</summary>
    public float ThermalConductivity { get; set; }
    /// <summary>Surface emissivity 0..1.</summary>
    public float Emissivity { get; set; }

    /// <summary>Builds a default material (moderate grip, nearly dead bounce, unit density).</summary>
    public PhysicsMaterial()
        : this(DefaultFriction, DefaultRestitution, DefaultDensity)
    {
    }

    /// <summary>Builds a material, sanitizing inputs (see <see cref="DefaultFriction"/>).</summary>
    public PhysicsMaterial(float friction, float restitution, float density)
    {
        Friction = float.IsFinite(friction) && friction >= 0.0f ? friction : DefaultFriction;
        Restitution = float.IsFinite(restitution) ? MathF.Max(0.0f, MathF.Min(1.0f, restitution)) : DefaultRestitution;
        Density = float.IsFinite(density) && density > 0.0f ? density : DefaultDensity;
        SpecificHeat = DefaultSpecificHeat;
        ThermalConductivity = DefaultThermalConductivity;
        Emissivity = DefaultEmissivity;
    }

    /// <summary>Fresh default instance (never a shared singleton).</summary>
    public static PhysicsMaterial Default => new PhysicsMaterial();

    /// <summary>Defensive copy.</summary>
    public PhysicsMaterial Clone() => new PhysicsMaterial(Friction, Restitution, Density);

    /// <summary>
    /// Combine two friction coefficients with the geometric mean, so a
    /// frictionless surface (mu = 0) always slides regardless of the other
    /// body, and equal materials keep their value.
    /// </summary>
    public static float CombineFriction(float a, float b) =>
        MathF.Sqrt(MathF.Max(0.0f, a) * MathF.Max(0.0f, b));

    /// <summary>
    /// Combine restitution with the maximum, so a bouncy object always
    /// bounces even against a dead surface.
    /// </summary>
    public static float CombineRestitution(float a, float b) => MathF.Max(a, b);
}
