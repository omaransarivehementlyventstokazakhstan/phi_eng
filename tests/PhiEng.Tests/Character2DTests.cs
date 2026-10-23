using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// The 2D kinematic character controller (O01–O06). Mirrors
/// <c>crates/engine/tests/character_2d.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class Character2DTests
{
    private static World2 FloorWorld()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(new Vector2(0.0f, -0.5f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        return w;
    }

    private static World2 WallWorld()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(new Vector2(2.5f, 0.0f), new BoxShape2(new Vector2(0.5f, 3.0f)));
        return w;
    }

    private static World2 RampWorld(float angle)
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(new Vector2(2.0f, 0.0f), new BoxShape2(new Vector2(2.0f, 0.3f)));
        var (c, s) = PhiEng.Math.DeterministicTrig.ComputeCosSin(angle);
        var t = w.Bodies[0].Transform;
        t.Rotation = new Matrix3x2(c, s, -s, c, 0.0f, 0.0f);
        w.Bodies[0].Transform = t;
        return w;
    }

    /// <summary>
    /// O01: a free move is applied exactly, with no contacts; a resting
    /// character still reports ground.
    /// </summary>
    [Fact]
    public void O01_AFreeMoveIsExact()
    {
        var w = FloorWorld();
        var cfg = new CharacterConfig2();
        var m = Character2.Move(w, Vector2.Zero, cfg, new Vector2(1.0f, 0.0f), QueryFilter.Everything);
        Assert.Equal(new Vector2(1.0f, 0.0f), m.Translation);
        Assert.Equal(0, m.Contacts);
        Assert.True(m.Grounded, "a character on the floor is grounded");
        Assert.Equal(Vector2.UnitY, m.GroundNormal);

        var air = Character2.Move(
            w, new Vector2(0.0f, 5.0f), cfg, new Vector2(0.0f, -1.0f), QueryFilter.Everything);
        Assert.Equal(new Vector2(0.0f, -1.0f), air.Translation);
        Assert.False(air.Grounded);
    }

    /// <summary>
    /// O02: a head-on move stops at the wall, one radius short of its
    /// face, and never tunnels.
    /// </summary>
    [Fact]
    public void O02_AHeadOnMoveStopsAtTheWall()
    {
        var w = WallWorld();
        var cfg = new CharacterConfig2();
        var m = Character2.Move(w, Vector2.Zero, cfg, new Vector2(5.0f, 0.0f), QueryFilter.Everything);
        Assert.True(MathF.Abs(m.Translation.X - 1.6f) < 2e-3f, $"stopped at {m.Translation.X}");
        Assert.Equal(0.0f, m.Translation.Y);
        Assert.True(m.Contacts > 0, "the wall must register as contacts");
    }

    /// <summary>
    /// O03: a diagonal move into a wall loses only the blocked
    /// component.
    /// </summary>
    [Fact]
    public void O03_ADiagonalMoveSlides()
    {
        var w = WallWorld();
        var cfg = new CharacterConfig2();
        var m = Character2.Move(w, Vector2.Zero, cfg, new Vector2(5.0f, 1.0f), QueryFilter.Everything);
        Assert.True(MathF.Abs(m.Translation.X - 1.6f) < 2e-3f, $"x blocked at {m.Translation.X}");
        Assert.True(MathF.Abs(m.Translation.Y - 1.0f) < 2e-3f, $"y must survive the slide, got {m.Translation.Y}");
    }

    /// <summary>
    /// O04: ground comes from the probe and from the slope rule, not
    /// from merely touching something.
    /// </summary>
    [Fact]
    public void O04_GroundNeedsAWalkableSurface()
    {
        var cfg = new CharacterConfig2();
        var w = FloorWorld();
        var m = Character2.Move(w, Vector2.Zero, cfg, Vector2.Zero, QueryFilter.Everything);
        Assert.True(m.Grounded);
        Assert.Equal(Vector2.UnitY, m.GroundNormal);

        var wall = WallWorld();
        var beside = Character2.Move(wall, new Vector2(1.6f, 0.0f), cfg, Vector2.Zero, QueryFilter.Everything);
        Assert.False(beside.Grounded, "a wall is not ground");

        var ramp = RampWorld(MathF.PI / 3.0f);
        var steep = Character2.Move(ramp, new Vector2(2.0f, 0.5f), cfg, Vector2.Zero, QueryFilter.Everything);
        Assert.False(steep.Grounded, "a 60-degree slope is too steep for ground");
    }

    /// <summary>
    /// O05: the query filter selects what is solid for the character.
    /// </summary>
    [Fact]
    public void O05_TheFilterSelectsSolidBodies()
    {
        var w = WallWorld();
        var cfg = new CharacterConfig2();
        w.Bodies[0].CollisionGroup = 2;
        var through = Character2.Move(
            w, Vector2.Zero, cfg, new Vector2(5.0f, 0.0f), new QueryFilter(1u, 1u));
        Assert.True((through.Translation - new Vector2(5.0f, 0.0f)).Length() < 1e-5f,
            $"a filtered wall must be walked through, got {through.Translation}");
        Assert.Equal(0, through.Contacts);

        var blocked = Character2.Move(
            w, Vector2.Zero, cfg, new Vector2(5.0f, 0.0f), new QueryFilter(1u, 2u));
        Assert.True(MathF.Abs(blocked.Translation.X - 1.6f) < 2e-3f);
    }

    /// <summary>
    /// O06: a walkable ramp is climbed (the slide carries the character
    /// up the normal) and reports ground.
    /// </summary>
    [Fact]
    public void O06_AWalkableRampIsClimbed()
    {
        var w = RampWorld(MathF.PI / 4.0f);
        var cfg = new CharacterConfig2();
        var m = Character2.Move(w, Vector2.Zero, cfg, new Vector2(3.0f, 0.0f), QueryFilter.Everything);
        Assert.True(m.Translation.Y > 0.9f, $"the slide must climb, got {m.Translation.Y}");
        Assert.True(m.Translation.X > 1.9f, $"x must still advance, got {m.Translation.X}");
        Assert.True(m.Grounded, "a 45-degree ramp is walkable");
        Assert.True(m.GroundNormal.Y > 0.65f, "ramp normal should point up");
    }
}
