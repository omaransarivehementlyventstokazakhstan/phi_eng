namespace PhiEng.Broadphase;

/// <summary>
/// Narrowphase collision tier: the cheapest test that can separate a
/// shape pair. Mirrors the Rust <c>CollisionTier</c> in
/// <c>crates/dynamics/src/collision_tier.rs</c>; the discriminant values
/// are part of the FFI contract and must stay in sync (see
/// <c>phieng_world_set_collision_tier</c>).
/// </summary>
public enum CollisionTier
{
    /// <summary>AABB overlap only (no narrowphase work).</summary>
    Tier1_AABBOnly = 0,

    /// <summary>Sphere vs AABB (analytic closest point).</summary>
    Tier2_SphereAABB = 1,

    /// <summary>Sphere vs sphere (analytic; circle-circle in 2D).</summary>
    Tier3_SphereSphere = 2,

    /// <summary>Box vs box (separating-axis test).</summary>
    Tier4_BoxBox = 3,

    /// <summary>General convex vs convex (GJK/EPA).</summary>
    Tier5_GJK_EPA = 4,
}

/// <summary>
/// Per-world collision-tier configuration. Mirrors the Rust
/// <c>CollisionTierConfig</c>: pairs whose required tier exceeds
/// <see cref="MaxTier"/> are skipped (no collision generated), so a lower
/// ceiling trades fidelity for cost. Disabled (the default) means full
/// fidelity, exactly like the CCD and thermal configs.
/// </summary>
public struct CollisionTierConfig
{
    /// <summary>Master switch. <c>false</c> (default) ignores the ceiling.</summary>
    public bool Enabled;

    /// <summary>Highest tier the narrowphase may run.</summary>
    public CollisionTier MaxTier;

    /// <summary>Default config: disabled, full fidelity (Tier5).</summary>
    public static CollisionTierConfig Default => new() { MaxTier = CollisionTier.Tier5_GJK_EPA };

    /// <summary>True when the tier ceiling is actually applied.</summary>
    public readonly bool IsActive() => Enabled && MaxTier != CollisionTier.Tier5_GJK_EPA;
}
