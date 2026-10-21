namespace PhiEng.Dynamics;

/// <summary>
/// The layers a scene query considers.
///
/// A query behaves like a body on <see cref="Group"/>: a body is
/// reported when it accepts the query's group and the query's mask
/// accepts the body's layer — the same rule two bodies use, so a query
/// and a body agree about whether they would collide.
///
/// Mirrors Rust <c>phieng_dynamics::body::QueryFilter</c>.
/// </summary>
public readonly struct QueryFilter
{
    /// <summary>The layer the query pretends to be on.</summary>
    public uint Group { get; }

    /// <summary>The layers the query accepts.</summary>
    public uint Mask { get; }

    /// <summary>A filter that accepts every body, whatever its layer.</summary>
    public static QueryFilter Everything => new(CollisionFilter.DefaultGroup, CollisionFilter.DefaultMask);

    /// <summary>Creates a query filter.</summary>
    public QueryFilter(uint group, uint mask)
    {
        Group = group;
        Mask = mask;
    }

    /// <summary>Whether this filter accepts a body on <paramref name="group"/>
    /// with the mask <paramref name="mask"/>.</summary>
    public bool Accepts(uint group, uint mask)
        => (group & Mask) != 0 && (Group & mask) != 0;

    /// <summary>Whether this filter accepts the given body.</summary>
    public bool Accepts(RigidBody2 body) => Accepts(body.CollisionGroup, body.CollisionMask);

    /// <summary>Whether this filter accepts the given body.</summary>
    public bool Accepts(RigidBody3 body) => Accepts(body.CollisionGroup, body.CollisionMask);
}
