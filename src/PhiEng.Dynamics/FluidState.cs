using System.Collections.Generic;
using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// SPH fluid particle (2D): position and velocity. Density and pressure
/// are recomputed every step and never persisted. Mirrors the Rust
/// <c>Particle2</c> (data only; the C# side does not step fluids yet).
/// </summary>
public struct FluidParticle2
{
    /// <summary>World-space position.</summary>
    public Vector2 Position { get; set; }
    /// <summary>World-space velocity.</summary>
    public Vector2 Velocity { get; set; }
}

/// <summary>
/// SPH fluid tunables (2D). Mirrors Rust <c>FluidConfig2</c>.
/// </summary>
public class FluidConfig2
{
    /// <summary>Uniform particle mass (should approximate rest-density times spacing squared).</summary>
    public float ParticleMass { get; set; } = 0.0625f;
    /// <summary>Rest density of the fluid.</summary>
    public float RestDensity { get; set; } = 1.0f;
    /// <summary>Kernel support radius (covers ~2 particle spacings).</summary>
    public float SmoothingLength { get; set; } = 0.5f;
    /// <summary>Pressure stiffness in the linear equation of state.</summary>
    public float Stiffness { get; set; } = 100.0f;
    /// <summary>XSPH smoothing factor in [0, 1].</summary>
    public float Viscosity { get; set; } = 0.3f;
    /// <summary>Quadratic drag coefficient per submerged sample area.</summary>
    public float Drag { get; set; } = 4.0f;
    /// <summary>Hard speed cap (safety net against ejection chains).</summary>
    public float MaxVelocity { get; set; } = 30.0f;
    /// <summary>Collision radius used against rigid-body shapes.</summary>
    public float ParticleRadius { get; set; } = 0.125f;
    /// <summary>Master switch for fluid stepping.</summary>
    public bool Enabled { get; set; } = true;
}

/// <summary>
/// SPH fluid state (2D): config plus live particles. Data carrier for
/// snapshots and the native bridge; stepping lives in Rust for now.
/// </summary>
public class FluidState2
{
    /// <summary>Hard cap on stored particles (creation-time guard).</summary>
    public const int MaxParticles = 4096;
    /// <summary>Fluid tunables.</summary>
    public FluidConfig2 Config { get; set; } = new();
    /// <summary>Live particles in insertion order.</summary>
    public List<FluidParticle2> Particles { get; } = new();

    /// <summary>Adds particles, capped at <see cref="MaxParticles"/> total. Returns the count added.</summary>
    public int AddParticles(IEnumerable<Vector2> positions)
    {
        int added = 0;
        foreach (var p in positions)
        {
            if (Particles.Count >= MaxParticles) break;
            Particles.Add(new FluidParticle2 { Position = p, Velocity = Vector2.Zero });
            added++;
        }
        return added;
    }

    /// <summary>Removes all particles.</summary>
    public void Clear() => Particles.Clear();
}

/// <summary>
/// SPH fluid particle (3D). Mirrors Rust <c>Particle3</c> (data only).
/// </summary>
public struct FluidParticle3
{
    /// <summary>World-space position.</summary>
    public Vector3 Position { get; set; }
    /// <summary>World-space velocity.</summary>
    public Vector3 Velocity { get; set; }
}

/// <summary>
/// SPH fluid tunables (3D). Mirrors Rust <c>FluidConfig3</c>.
/// </summary>
public class FluidConfig3
{
    /// <summary>Uniform particle mass (calibrated so a settled lattice sums to rest density).</summary>
    public float ParticleMass { get; set; } = 0.0625f;
    /// <summary>Rest density of the fluid.</summary>
    public float RestDensity { get; set; } = 1.0f;
    /// <summary>Kernel support radius (covers ~2 particle spacings).</summary>
    public float SmoothingLength { get; set; } = 0.5f;
    /// <summary>Pressure stiffness in the linear equation of state.</summary>
    public float Stiffness { get; set; } = 100.0f;
    /// <summary>XSPH smoothing factor in [0, 1].</summary>
    public float Viscosity { get; set; } = 0.3f;
    /// <summary>Quadratic drag coefficient per submerged sample area.</summary>
    public float Drag { get; set; } = 4.0f;
    /// <summary>Hard speed cap (safety net against ejection chains).</summary>
    public float MaxVelocity { get; set; } = 30.0f;
    /// <summary>Collision radius used against rigid-body shapes.</summary>
    public float ParticleRadius { get; set; } = 0.125f;
    /// <summary>Master switch for fluid stepping.</summary>
    public bool Enabled { get; set; } = true;
}

/// <summary>
/// SPH fluid state (3D): config plus live particles. Data carrier for
/// snapshots and the native bridge; stepping lives in Rust for now.
/// </summary>
public class FluidState3
{
    /// <summary>Hard cap on stored particles (creation-time guard).</summary>
    public const int MaxParticles = 4096;
    /// <summary>Fluid tunables.</summary>
    public FluidConfig3 Config { get; set; } = new();
    /// <summary>Live particles in insertion order.</summary>
    public List<FluidParticle3> Particles { get; } = new();

    /// <summary>Adds particles, capped at <see cref="MaxParticles"/> total. Returns the count added.</summary>
    public int AddParticles(IEnumerable<Vector3> positions)
    {
        int added = 0;
        foreach (var p in positions)
        {
            if (Particles.Count >= MaxParticles) break;
            Particles.Add(new FluidParticle3 { Position = p, Velocity = Vector3.Zero });
            added++;
        }
        return added;
    }

    /// <summary>Removes all particles.</summary>
    public void Clear() => Particles.Clear();
}
