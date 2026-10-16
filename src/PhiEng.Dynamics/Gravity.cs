using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Gravitational force model. Discriminants match the Rust
/// <c>phieng_dynamics::gravity::GravityMode</c> (snapshot v4 file
/// format) — never reorder.
/// </summary>
public enum GravityMode : uint
{
    /// <summary>Classic uniform field (<c>World.Gravity</c>), scaled per body.</summary>
    Uniform = 0,
    /// <summary>Single fixed attractor (<c>Attractor</c>, <c>AttractorMu</c>).</summary>
    Central = 1,
    /// <summary>Full pairwise Newtonian gravity. O(n^2).</summary>
    NBody = 2,
    /// <summary>Barnes-Hut approximation of N-body. O(n log n), approximate by design.</summary>
    BarnesHut = 3,
}

/// <summary>
/// Time-integration scheme for the gravity/free-flight part of the step.
/// Discriminants match Rust (snapshot v4) — never reorder.
/// </summary>
public enum Integrator : uint
{
    /// <summary>Semi-implicit Euler (status quo). Default.</summary>
    SemiImplicitEuler = 0,
    /// <summary>Kick-drift-kick leapfrog. Better orbital energy behavior.</summary>
    VelocityVerlet = 1,
}

/// <summary>
/// Gravity configuration for a 2D world. The uniform
/// <c>World2.Gravity</c> vector always applies as a background field
/// (superposition); this selects the additional force model.
/// </summary>
public struct GravityConfig2
{
    /// <summary>Force model.</summary>
    public GravityMode Mode;
    /// <summary>Integration scheme.</summary>
    public Integrator Integrator;
    /// <summary>Gravitational constant in scene units (default 1.0).</summary>
    public float BigG;
    /// <summary>Softening length, >= 0 (default 0.0).</summary>
    public float Softening;
    /// <summary>Fixed attractor point (Central mode).</summary>
    public Vector2 Attractor;
    /// <summary>G*M of the fixed attractor (Central mode).</summary>
    public float AttractorMu;
    /// <summary>Barnes-Hut opening criterion, >= 0 (default 0.5).</summary>
    public float Theta;

    /// <summary>Legacy behavior: uniform field, Euler, unit G.</summary>
    public static GravityConfig2 Default => new()
    {
        Mode = GravityMode.Uniform,
        Integrator = Integrator.SemiImplicitEuler,
        BigG = 1.0f,
        Softening = 0.0f,
        Attractor = Vector2.Zero,
        AttractorMu = 0.0f,
        Theta = 0.5f,
    };

    /// <summary>True for a freshly constructed config (serializes as schema v3).</summary>
    public readonly bool IsDefault() =>
        Mode == GravityMode.Uniform &&
        Integrator == Integrator.SemiImplicitEuler &&
        BigG == 1.0f &&
        Softening == 0.0f &&
        Attractor == Vector2.Zero &&
        AttractorMu == 0.0f &&
        Theta == 0.5f;
}

/// <summary>
/// Gravity configuration for a 3D world. Same semantics as
/// <see cref="GravityConfig2"/> with a 3D attractor point.
/// </summary>
public struct GravityConfig3
{
    /// <summary>Force model.</summary>
    public GravityMode Mode;
    /// <summary>Integration scheme.</summary>
    public Integrator Integrator;
    /// <summary>Gravitational constant in scene units (default 1.0).</summary>
    public float BigG;
    /// <summary>Softening length, >= 0 (default 0.0).</summary>
    public float Softening;
    /// <summary>Fixed attractor point (Central mode).</summary>
    public Vector3 Attractor;
    /// <summary>G*M of the fixed attractor (Central mode).</summary>
    public float AttractorMu;
    /// <summary>Barnes-Hut opening criterion, >= 0 (default 0.5).</summary>
    public float Theta;

    /// <summary>Legacy behavior: uniform field, Euler, unit G.</summary>
    public static GravityConfig3 Default => new()
    {
        Mode = GravityMode.Uniform,
        Integrator = Integrator.SemiImplicitEuler,
        BigG = 1.0f,
        Softening = 0.0f,
        Attractor = Vector3.Zero,
        AttractorMu = 0.0f,
        Theta = 0.5f,
    };

    /// <summary>True for a freshly constructed config (serializes as schema v3).</summary>
    public readonly bool IsDefault() =>
        Mode == GravityMode.Uniform &&
        Integrator == Integrator.SemiImplicitEuler &&
        BigG == 1.0f &&
        Softening == 0.0f &&
        Attractor == Vector3.Zero &&
        AttractorMu == 0.0f &&
        Theta == 0.5f;
}
