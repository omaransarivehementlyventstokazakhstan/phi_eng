using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>The 3D character's size and feel. Mirrors Rust <c>CharacterConfig3</c>.</summary>
public sealed class CharacterConfig3
{
    /// <summary>The sphere's radius.</summary>
    public float Radius = 0.4f;
    /// <summary>The cosine of the steepest walkable slope (50 degrees by default).</summary>
    public float MaxSlopeCos = 0.6427876f;
    /// <summary>How far below the feet to probe for ground.</summary>
    public float GroundProbe = 0.02f;
    /// <summary>The maximum number of contacts resolved per move.</summary>
    public int MaxSlides = 4;
}

/// <summary>What one move did. Mirrors Rust <c>CharacterMove3</c>.</summary>
public struct CharacterMove3
{
    /// <summary>The translation actually applied.</summary>
    public Vector3 Translation;
    /// <summary>Whether a walkable surface is under the feet.</summary>
    public bool Grounded;
    /// <summary>The ground normal (up when not grounded).</summary>
    public Vector3 GroundNormal;
    /// <summary>The number of contacts the move resolved.</summary>
    public int Contacts;
}

/// <summary>
/// A 3D kinematic character controller: move and slide.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the Rust <c>phieng_dynamics::character3</c> module operation
/// for operation, so the two engines agree bit-for-bit. The character is
/// a sphere that the caller places: the controller advances it by a
/// desired translation, stops at the first surface, and slides the
/// remainder along it. It never applies gravity.
/// </para>
/// <para>
/// The sphere body is deliberate: sphere-versus-shape is what the
/// closest-point query covers analytically (sphere, box, capsule). A
/// capsule body needs a capsule-versus-shape distance query, which does
/// not exist yet. The convex hull and mesh report no closest point, so
/// they are walked through — the same documented gap as the ray cast.
/// </para>
/// </remarks>
public static class Character3
{
    private const float ContactSlackScale = 1e-4f;
    private const float Separation = 1e-4f;

    private static (Vector3 Normal, float Penetration)? BlockingContact(
        World3 world, Vector3 center, float radius, Vector3 delta, QueryFilter filter)
    {
        (Vector3 Normal, float Penetration)? best = null;
        foreach (var body in world.Bodies)
        {
            if (!filter.Accepts(body)) continue;
            if (body.Shape == null) continue;
            var hit = ShapeQuery3.ClosestPoint(
                body.Shape, body.Transform.Position, body.Transform.Rotation, center);
            if (hit == null) continue;
            var h = hit.Value;
            float slack = radius * ContactSlackScale;
            float penetration;
            if (h.Inside)
            {
                penetration = radius + h.Distance;
            }
            else if (h.Distance < radius - slack)
            {
                penetration = radius - h.Distance;
            }
            else
            {
                continue;
            }
            if (Vector3.Dot(delta, h.Normal) >= 0.0f) continue;
            if (best == null || penetration > best.Value.Penetration)
            {
                best = (h.Normal, penetration);
            }
        }
        return best;
    }

    /// <summary>Move a kinematic sphere character by <paramref name="desired"/>, sliding along contacts.</summary>
    public static CharacterMove3 Move(
        World3 world, Vector3 start, CharacterConfig3 config, Vector3 desired, QueryFilter filter)
    {
        float radius = MathF.Max(config.Radius, 1e-4f);
        var position = start;
        var translation = Vector3.Zero;
        int contacts = 0;
        int slides = System.Math.Max(config.MaxSlides, 1);

        float length = desired.Length();
        int steps = length <= radius * 0.5f
            ? 1
            : System.Math.Clamp((int)MathF.Ceiling(length / (radius * 0.5f)), 2, 64);
        var step = desired / steps;

        for (int s = 0; s < steps; s++)
        {
            var remaining = step;
            for (int k = 0; k < slides; k++)
            {
                if (remaining.LengthSquared() <= 1e-12f) break;
                if (BlockingContact(world, position + remaining, radius, remaining, filter) == null)
                {
                    position += remaining;
                    translation += remaining;
                    break;
                }
                float lo = 0.0f;
                float hi = 1.0f;
                for (int i = 0; i < 10; i++)
                {
                    float mid = 0.5f * lo + 0.5f * hi;
                    if (BlockingContact(world, position + remaining * mid, radius, remaining, filter) == null)
                    {
                        lo = mid;
                    }
                    else
                    {
                        hi = mid;
                    }
                }
                var advance = remaining * lo;
                position += advance;
                translation += advance;
                // The advanced fraction is spent: slide only the remainder.
                remaining -= advance;
                contacts++;

                var probe = remaining * (lo + 1e-3f);
                var contact = BlockingContact(world, position + probe, radius, remaining, filter);
                if (contact == null) break;
                var normal = contact.Value.Normal;
                float into = Vector3.Dot(remaining, normal);
                if (into >= 0.0f) continue;
                remaining -= normal * into;
                position += normal * Separation;
                translation += normal * Separation;
            }
        }

        bool grounded = false;
        var groundNormal = Vector3.UnitY;
        var ground = BlockingContact(
            world, position - Vector3.UnitY * config.GroundProbe, radius, -Vector3.UnitY, filter);
        if (ground != null && ground.Value.Normal.Y >= config.MaxSlopeCos)
        {
            grounded = true;
            groundNormal = ground.Value.Normal;
        }

        return new CharacterMove3
        {
            Translation = translation,
            Grounded = grounded,
            GroundNormal = groundNormal,
            Contacts = contacts,
        };
    }
}
