using System.Numerics;
using Xunit;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Collision-filter tests (L01–L06): the per-body layer and mask, and
/// the rule that two bodies may collide only when each accepts the
/// other's layer. Mirrors
/// <c>crates/engine/tests/collision_filter.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class CollisionFilterTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>A ball dropped on a static floor.</summary>
    private static PhiNativeWorld DropBall()
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        w.AddDynamicBall(new Vector2(0.0f, 1.0f), 0.5f, 1.0f);
        return w;
    }

    /// <summary>L01: the default filter accepts every layer.</summary>
    [Fact]
    public void L01_DefaultFilterCollidesWithEverything()
    {
        using var w = DropBall();
        var (group, mask) = w.GetBodyCollisionFilter(1);
        Assert.Equal(1u, group);
        Assert.Equal(uint.MaxValue, mask);

        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
        }
        // The floor's top surface is at y = -0.5 and the ball's radius
        // is 0.5, so a resting ball sits at y = 0.
        float ballY = w.GetBodyTransform(1).Position.Y;
        Assert.True(ballY > -0.2f, $"the ball must rest on the floor (y={ballY})");
    }

    /// <summary>L02: a layer that does not accept the other side rejects the pair.</summary>
    [Fact]
    public void L02_ALayerThatDoesNotAcceptRejectsThePair()
    {
        using var w = DropBall();
        // The floor stays on the default layer; the ball moves to a
        // layer the floor does not accept, and the floor does not
        // accept the ball's layer either.
        w.SetBodyCollisionFilter(1, 2u, 1u);
        w.SetBodyCollisionFilter(0, 1u, 1u);

        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
        }
        float ballY = w.GetBodyTransform(1).Position.Y;
        Assert.True(ballY < -1.0f, $"the filtered ball must fall through (y={ballY})");
    }

    /// <summary>L03: one side is enough to veto the pair.</summary>
    [Fact]
    public void L03_OneSidedVetoStopsThePair()
    {
        using var w = DropBall();
        // The floor still accepts everything; the ball accepts only itself.
        w.SetBodyCollisionFilter(1, 2u, 1u << 2);

        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
        }
        float ballY = w.GetBodyTransform(1).Position.Y;
        Assert.True(ballY < -1.0f, $"the ball's own veto must stop the pair (y={ballY})");
    }

    /// <summary>L04: the same rule in 3D.</summary>
    [Fact]
    public void L04_FilterAppliesIn3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        w.AddDynamicBall(new Vector3(0.0f, 1.0f, 0.0f), 0.5f, 1.0f);
        w.SetBodyCollisionFilter(1, 2u, 1u);
        w.SetBodyCollisionFilter(0, 1u, 1u);

        for (int i = 0; i < 240; i++)
        {
            w.Step(Dt);
        }
        float ballY = w.GetBodyTransform(1).Position.Y;
        Assert.True(ballY < -1.0f, $"the filtered ball must fall through (y={ballY})");
    }

    /// <summary>L05: the filter survives a snapshot round trip.</summary>
    [Fact]
    public void L05_FilterSurvivesASnapshotRoundTrip()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        w.AddDynamicBody(new Vector2(0.0f, 1.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        w.Bodies[1].CollisionGroup = 7u;
        w.Bodies[1].CollisionMask = 0b1010u;

        string json = Dynamics.Snapshot.World2ToJson(w, Dynamics.SnapshotLoadMode.Exact);
        var restored = Dynamics.Snapshot.World2FromJson(json, Dynamics.SnapshotLoadMode.Exact);
        Assert.Equal(7u, restored.Bodies[1].CollisionGroup);
        Assert.Equal(0b1010u, restored.Bodies[1].CollisionMask);

        // A world whose bodies all carry the default filter must load
        // with the default. (The managed writer always writes every
        // key, so the check is on the decoded value; the Rust writer
        // elides the default and pins that in the Rust mirror.)
        var plain = new Dynamics.World2();
        plain.Gravity = new Vector2(0.0f, -9.81f);
        plain.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        plain.AddDynamicBody(new Vector2(0.0f, 1.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        string plainJson = Dynamics.Snapshot.World2ToJson(plain, Dynamics.SnapshotLoadMode.Exact);
        var plainRestored = Dynamics.Snapshot.World2FromJson(plainJson, Dynamics.SnapshotLoadMode.Exact);
        Assert.Equal(Dynamics.CollisionFilter.DefaultGroup, plainRestored.Bodies[1].CollisionGroup);
        Assert.Equal(Dynamics.CollisionFilter.DefaultMask, plainRestored.Bodies[1].CollisionMask);
    }

    /// <summary>L06: the native bridge round-trips the filter.</summary>
    [Fact]
    public void L06_FfiRoundTripsTheFilter()
    {
        using var w = DropBall();
        w.SetBodyCollisionFilter(1, 5u, 0b1100);
        var (group, mask) = w.GetBodyCollisionFilter(1);
        Assert.Equal(5u, group);
        Assert.Equal(0b1100u, mask);

        // A step must not disturb the filter.
        w.Step(Dt);
        (group, mask) = w.GetBodyCollisionFilter(1);
        Assert.Equal(5u, group);
        Assert.Equal(0b1100u, mask);
    }

    /// <summary>
    /// The managed mirror applies the same rule as the native core (the
    /// managed world has its own narrowphase loop).
    /// </summary>
    [Fact]
    public void ManagedMirrorAppliesTheSameRule()
    {
        var resting = ManagedDropBall();
        for (int i = 0; i < 240; i++)
        {
            resting.Step(Dt);
        }
        float y = resting.Bodies[1].Transform.Position.Y;
        Assert.True(y > -0.2f, $"the ball must rest on the floor (y={y})");

        // The same scene with a filter that rejects the pair.
        var filtered = ManagedDropBall();
        filtered.Bodies[1].CollisionGroup = 2u;
        filtered.Bodies[1].CollisionMask = 1u;
        filtered.Bodies[0].CollisionMask = 1u;
        for (int i = 0; i < 240; i++)
        {
            filtered.Step(Dt);
        }
        float fallen = filtered.Bodies[1].Transform.Position.Y;
        Assert.True(fallen < -1.0f, $"the filtered ball must fall through (y={fallen})");
    }

    private static Dynamics.World2 ManagedDropBall()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        w.AddDynamicBody(new Vector2(0.0f, 1.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        return w;
    }
}
