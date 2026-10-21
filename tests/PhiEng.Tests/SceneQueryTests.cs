using System.Numerics;
using Xunit;
using PhiEng.Math;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Scene-query tests (X01–X06): which bodies are near a place. Mirrors
/// <c>crates/engine/tests/scene_query.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class SceneQueryTests
{
    /// <summary>A floor and three balls at known places.</summary>
    private static PhiNativeWorld Scene()
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        foreach (float x in new[] { -5.0f, 0.0f, 5.0f })
        {
            w.AddDynamicBall(new Vector2(x, 1.0f), 0.5f, 1.0f);
        }
        return w;
    }

    /// <summary>X01: the query returns the bodies whose box overlaps, in id order.</summary>
    [Fact]
    public void X01_OverlapQueryReturnsTheBodiesInTheBox()
    {
        using var w = Scene();
        var middle = new AABB2(new Vector2(-1.0f, 0.0f), new Vector2(1.0f, 2.0f));
        uint[] hits = w.OverlapAabb(middle.Min.X, middle.Min.Y, middle.Max.X, middle.Max.Y, 1, uint.MaxValue);
        Assert.Equal(new uint[] { 2 }, hits);

        var all = new AABB2(new Vector2(-10.0f, 0.0f), new Vector2(10.0f, 2.0f));
        hits = w.OverlapAabb(all.Min.X, all.Min.Y, all.Max.X, all.Max.Y, 1, uint.MaxValue);
        Assert.Equal(new uint[] { 1, 2, 3 }, hits);

        var nothing = new AABB2(new Vector2(100.0f, 100.0f), new Vector2(101.0f, 101.0f));
        hits = w.OverlapAabb(nothing.Min.X, nothing.Min.Y, nothing.Max.X, nothing.Max.Y, 1, uint.MaxValue);
        Assert.Empty(hits);
    }

    /// <summary>X02: the collision filter applies to queries.</summary>
    [Fact]
    public void X02_QueryRespectsTheCollisionFilter()
    {
        using var w = Scene();
        // The middle ball moves to a group of its own. A group is one
        // bit (the Box2D convention), so the ball's group is the flag
        // 0x2 and a mask that accepts only that group has the same bit.
        w.SetBodyCollisionFilter(2, 2u, 2u);

        uint[] hits = w.OverlapAabb(-1.0f, 0.0f, 1.0f, 2.0f, 2, 2);
        Assert.Equal(new uint[] { 2 }, hits);

        // A query on the default group does not: the ball accepts only
        // its own group, and the query is not on it. The rule is mutual,
        // so the ball's own veto applies to a query as much as to a body.
        hits = w.OverlapAabb(-1.0f, 0.0f, 1.0f, 2.0f, 1, uint.MaxValue);
        Assert.Empty(hits);
    }

    /// <summary>X03: the query sees the current positions.</summary>
    [Fact]
    public void X03_QuerySeesTheCurrentPositions()
    {
        // The managed world can place a body directly (the native bridge
        // has no body-transform setter yet), which is exactly what this
        // test needs: a body that is somewhere else now, while the
        // broad phase still holds where it was.
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        foreach (float x in new[] { -5.0f, 0.0f, 5.0f })
        {
            w.AddDynamicBody(new Vector2(x, 1.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        }
        var moved = new Transform2 { Position = new Vector2(50.0f, 50.0f) };
        w.Bodies[3].Transform = moved;

        var hits = w.OverlapAabb(
            new AABB2(new Vector2(4.0f, 0.0f), new Vector2(6.0f, 2.0f)),
            Dynamics.QueryFilter.Everything);
        Assert.Empty(hits);

        hits = w.OverlapAabb(
            new AABB2(new Vector2(49.0f, 49.0f), new Vector2(51.0f, 51.0f)),
            Dynamics.QueryFilter.Everything);
        Assert.Equal(new uint[] { 3 }, hits);
    }

    /// <summary>X04: the same query in 3D.</summary>
    [Fact]
    public void X04_OverlapQueryIn3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
        w.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        foreach (float x in new[] { -5.0f, 0.0f, 5.0f })
        {
            w.AddDynamicBall(new Vector3(x, 1.0f, 0.0f), 0.5f, 1.0f);
        }
        uint[] hits = w.OverlapAabb(-1.0f, 0.0f, -1.0f, 1.0f, 2.0f, 1.0f, 1, uint.MaxValue);
        Assert.Equal(new uint[] { 2 }, hits);
    }

    /// <summary>X05: the native bridge answers the same query.</summary>
    [Fact]
    public void X05_FfiOverlapQuery()
    {
        using var w = Scene();
        uint[] hits = w.OverlapAabb(-1.0f, 0.0f, 1.0f, 2.0f, 1, uint.MaxValue);
        Assert.Equal(new uint[] { 2 }, hits);
    }

    /// <summary>X06: the managed mirror answers the same query.</summary>
    [Fact]
    public void X06_ManagedMirrorAnswersTheSameQuery()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        w.AddStaticBody(new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        foreach (float x in new[] { -5.0f, 0.0f, 5.0f })
        {
            w.AddDynamicBody(new Vector2(x, 1.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        }
        var hits = w.OverlapAabb(
            new AABB2(new Vector2(-1.0f, 0.0f), new Vector2(1.0f, 2.0f)),
            Dynamics.QueryFilter.Everything);
        Assert.Equal(new uint[] { 2 }, hits);
    }
}
