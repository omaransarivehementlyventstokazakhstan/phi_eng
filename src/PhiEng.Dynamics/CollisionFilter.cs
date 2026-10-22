namespace PhiEng.Dynamics;

/// <summary>
/// Collision filtering: the layer a body belongs to and the layers it
/// accepts.
///
/// Two bodies may collide only when each accepts the other's layer, the
/// rule every layered physics engine uses (Box2D, Bullet), with one
/// addition: two bodies on the same group always collide, whatever
/// their masks say. A body that accepts one layer therefore collides
/// only with the bodies on it, and one side is enough to veto the pair.
///
/// Mirrors Rust <c>phieng_dynamics::body::collision_filter_allows</c>.
/// </summary>
public static class CollisionFilter
{
    /// <summary>The layer a body belongs to unless it is configured otherwise.</summary>
    public const uint DefaultGroup = 1;

    /// <summary>The layers a body accepts unless it is configured otherwise: every layer.</summary>
    public const uint DefaultMask = uint.MaxValue;

    /// <summary>Whether two bodies with the given filters may collide.</summary>
    public static bool Allows(uint groupA, uint maskA, uint groupB, uint maskB)
    {
        // The same group overrides the masks.
        if (groupA == groupB && groupA != 0)
        {
            return true;
        }
        return (groupA & maskB) != 0 && (groupB & maskA) != 0;
    }

    /// <summary>Whether two bodies with the given filters may collide.</summary>
    public static bool Allows(RigidBody2 a, RigidBody2 b)
        => Allows(a.CollisionGroup, a.CollisionMask, b.CollisionGroup, b.CollisionMask);

    /// <summary>Whether two bodies with the given filters may collide.</summary>
    public static bool Allows(RigidBody3 a, RigidBody3 b)
        => Allows(a.CollisionGroup, a.CollisionMask, b.CollisionGroup, b.CollisionMask);
}
