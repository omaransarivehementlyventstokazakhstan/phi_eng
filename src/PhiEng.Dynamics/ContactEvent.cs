using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>What happened to a contact between two bodies.</summary>
public enum ContactEventKind
{
    /// <summary>The bodies started touching this step.</summary>
    Began = 0,
    /// <summary>The bodies stopped touching this step.</summary>
    Ended = 1,
}

/// <summary>
/// A contact event: two bodies started or stopped touching.
///
/// The world collects one event per pair per step, and the caller
/// drains them after the step. Mirrors Rust
/// <c>phieng_dynamics::body::ContactEvent2</c>.
/// </summary>
public readonly struct ContactEvent2
{
    /// <summary>The first body of the pair.</summary>
    public uint BodyA { get; }

    /// <summary>The second body of the pair.</summary>
    public uint BodyB { get; }

    /// <summary>Whether the contact began or ended.</summary>
    public ContactEventKind Kind { get; }

    /// <summary>The contact point, in world space, at the moment the event fired.</summary>
    public Vector2 Point { get; }

    /// <summary>The contact normal, in world space, at the moment the event fired.</summary>
    public Vector2 Normal { get; }

    /// <summary>Creates a contact event.</summary>
    public ContactEvent2(uint bodyA, uint bodyB, ContactEventKind kind, Vector2 point, Vector2 normal)
    {
        BodyA = bodyA;
        BodyB = bodyB;
        Kind = kind;
        Point = point;
        Normal = normal;
    }
}

/// <summary>
/// A contact event in 3D. Mirrors Rust
/// <c>phieng_dynamics::body::ContactEvent3</c>.
/// </summary>
public readonly struct ContactEvent3
{
    /// <summary>The first body of the pair.</summary>
    public uint BodyA { get; }

    /// <summary>The second body of the pair.</summary>
    public uint BodyB { get; }

    /// <summary>Whether the contact began or ended.</summary>
    public ContactEventKind Kind { get; }

    /// <summary>The contact point, in world space, at the moment the event fired.</summary>
    public Vector3 Point { get; }

    /// <summary>The contact normal, in world space, at the moment the event fired.</summary>
    public Vector3 Normal { get; }

    /// <summary>Creates a contact event.</summary>
    public ContactEvent3(uint bodyA, uint bodyB, ContactEventKind kind, Vector3 point, Vector3 normal)
    {
        BodyA = bodyA;
        BodyB = bodyB;
        Kind = kind;
        Point = point;
        Normal = normal;
    }
}
