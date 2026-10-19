using System.Numerics;
using Xunit;
using PhiEng.Broadphase;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Collision-tier tests (H01–H05): the narrowphase tier ceiling, its
/// opt-in default, and its effect on contacts. Mirrors
/// <c>crates/engine/tests/collision_tier.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class CollisionTierTests
{
    private const float Dt = 1.0f / 60.0f;

    private static ulong Hash(PhiNativeWorld w, int count)
    {
        ulong h = 0xcbf29ce484222325;
        for (int i = 0; i < count; i++)
        {
            var (pos, _) = w.GetBodyTransform(i);
            foreach (float v in new[] { pos.X, pos.Y, w.GetBodyVelocity(i).X, w.GetBodyVelocity(i).Y })
            {
                h ^= BitConverter.SingleToUInt32Bits(v);
                h *= 0x100000001b3;
            }
        }
        return h;
    }

    /// <summary>A stack of boxes on a static floor. Box-box needs Tier4.</summary>
    private static PhiNativeWorld BoxStack(int bodies)
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        for (int i = 0; i < bodies; i++)
        {
            w.AddBoxBody(new Vector2(0.0f, 0.05f + i * 1.02f), new Vector2(0.5f, 0.5f), 1.0f, false);
        }
        return w;
    }

    /// <summary>H01: the tier system is off by default and matches an explicit disable.</summary>
    [Fact]
    public void H01_DefaultIsFullFidelity()
    {
        using var a = BoxStack(4);
        using var b = BoxStack(4);
        b.SetCollisionTier(new CollisionTierConfig { Enabled = false, MaxTier = CollisionTier.Tier5_GJK_EPA });

        for (int i = 0; i < 180; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        Assert.Equal(Hash(a, 5), Hash(b, 5));
    }

    /// <summary>H02: a ceiling below the pair's required tier stops the contact.</summary>
    [Fact]
    public void H02_CeilingBelowRequiredTierStopsContacts()
    {
        using var full = BoxStack(3);
        using var capped = BoxStack(3);
        capped.SetCollisionTier(new CollisionTierConfig { Enabled = true, MaxTier = CollisionTier.Tier3_SphereSphere });

        for (int i = 0; i < 180; i++)
        {
            full.Step(Dt);
            capped.Step(Dt);
        }

        float topFull = full.GetBodyTransform(full.BodyCount - 1).Position.Y;
        Assert.True(topFull > 0.0f, $"full fidelity must hold the stack, top y = {topFull}");

        float topCapped = capped.GetBodyTransform(capped.BodyCount - 1).Position.Y;
        Assert.True(topCapped < topFull - 1.0f,
            $"a ceiling below Tier4 must stop box-box contacts: top y = {topCapped} vs {topFull}");
    }

    /// <summary>H03: a ceiling at the required tier keeps the contacts.</summary>
    [Fact]
    public void H03_CeilingAtRequiredTierKeepsContacts()
    {
        using var full = BoxStack(3);
        using var capped = BoxStack(3);
        capped.SetCollisionTier(new CollisionTierConfig { Enabled = true, MaxTier = CollisionTier.Tier4_BoxBox });

        for (int i = 0; i < 180; i++)
        {
            full.Step(Dt);
            capped.Step(Dt);
        }
        Assert.Equal(Hash(full, 4), Hash(capped, 4));
    }

    /// <summary>H04: 3D mirrors the 2D contract (sphere/box is Tier2).</summary>
    [Fact]
    public void H04_3DCeilingBehavior()
    {
        using var full = new PhiNativeWorld3();
        using var capped = new PhiNativeWorld3();
        foreach (var w in new[] { full, capped })
        {
            w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
            w.SetSleepThresholds(0.0f, 0.0f);
            w.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
            for (int i = 0; i < 3; i++)
            {
                w.AddSphereBody(new Vector3(0.0f, 0.25f + i * 1.1f, 0.0f), 0.5f, 1.0f, false);
            }
        }
        capped.SetCollisionTier(new CollisionTierConfig { Enabled = true, MaxTier = CollisionTier.Tier2_SphereAABB });

        for (int i = 0; i < 180; i++)
        {
            full.Step(Dt);
            capped.Step(Dt);
        }

        float topFull = full.GetBodyTransform(full.BodyCount - 1).Position.Y;
        float topCapped = capped.GetBodyTransform(capped.BodyCount - 1).Position.Y;
        Assert.True(topCapped < topFull - 0.5f,
            $"a Tier2 ceiling must stop the sphere/box contacts: top y = {topCapped} vs {topFull}");
    }

    /// <summary>H05: tier-capped runs stay deterministic.</summary>
    [Fact]
    public void H05_TierConfigIsDeterministic()
    {
        using var a = BoxStack(4);
        using var b = BoxStack(4);
        foreach (var w in new[] { a, b })
        {
            w.SetCollisionTier(new CollisionTierConfig { Enabled = true, MaxTier = CollisionTier.Tier2_SphereAABB });
        }
        for (int i = 0; i < 120; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        Assert.Equal(Hash(a, 5), Hash(b, 5));
    }
}
