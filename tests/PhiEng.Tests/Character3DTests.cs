using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// The 3D kinematic character controller (O07–O12). Mirrors
/// <c>crates/engine/tests/character_3d.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class Character3DTests
{
    private static World3 FloorWorld()
    {
        var w = new World3 { Gravity = Vector3.Zero };
        w.AddStaticBody(new Vector3(0.0f, -0.5f, 0.0f), new BoxShape3(new Vector3(10.0f, 0.5f, 10.0f)));
        return w;
    }

    private static World3 WallWorld()
    {
        var w = new World3 { Gravity = Vector3.Zero };
        w.AddStaticBody(new Vector3(2.5f, 1.0f, 0.0f), new BoxShape3(new Vector3(0.5f, 3.0f, 3.0f)));
        return w;
    }

    private static World3 RampWorld(float angle)
    {
        var w = new World3 { Gravity = Vector3.Zero };
        w.AddStaticBody(new Vector3(2.0f, 0.0f, 0.0f), new BoxShape3(new Vector3(2.0f, 0.3f, 2.0f)));
        var b = w.Bodies[0];
        var t = b.Transform;
        t.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, angle);
        b.Transform = t;
        return w;
    }

    /// <summary>
    /// O07: a free move is applied exactly, with no contacts; a resting
    /// character reports ground.
    /// </summary>
    [Fact]
    public void O07_AFreeMoveIsExact()
    {
        var w = FloorWorld();
        var cfg = new CharacterConfig3();
        var m = Character3.Move(w, Vector3.Zero, cfg, new Vector3(1.0f, 0.0f, 0.0f), QueryFilter.Everything);
        Assert.Equal(new Vector3(1.0f, 0.0f, 0.0f), m.Translation);
        Assert.Equal(0, m.Contacts);
        Assert.True(m.Grounded, "a character on the floor is grounded");
        Assert.Equal(Vector3.UnitY, m.GroundNormal);

        var air = Character3.Move(
            w, new Vector3(0.0f, 5.0f, 0.0f), cfg, new Vector3(0.0f, -1.0f, 0.0f), QueryFilter.Everything);
        Assert.Equal(new Vector3(0.0f, -1.0f, 0.0f), air.Translation);
        Assert.False(air.Grounded);
    }

    /// <summary>
    /// O08: a head-on move stops one radius short of the wall face and
    /// never tunnels.
    /// </summary>
    [Fact]
    public void O08_AHeadOnMoveStopsAtTheWall()
    {
        var w = WallWorld();
        var cfg = new CharacterConfig3();
        var m = Character3.Move(w, Vector3.Zero, cfg, new Vector3(5.0f, 0.0f, 0.0f), QueryFilter.Everything);
        Assert.True(MathF.Abs(m.Translation.X - 1.6f) < 2e-3f, $"stopped at {m.Translation.X}");
        Assert.Equal(0.0f, m.Translation.Y);
        Assert.Equal(0.0f, m.Translation.Z);
        Assert.True(m.Contacts > 0);
    }

    /// <summary>
    /// O09: a diagonal move loses only the blocked component; the
    /// other two axes survive exactly.
    /// </summary>
    [Fact]
    public void O09_ADiagonalMoveSlides()
    {
        var w = WallWorld();
        var cfg = new CharacterConfig3();
        var m = Character3.Move(
            w, Vector3.Zero, cfg, new Vector3(5.0f, 1.0f, 0.5f), QueryFilter.Everything);
        Assert.True(MathF.Abs(m.Translation.X - 1.6f) < 2e-3f, $"x blocked at {m.Translation.X}");
        Assert.True(MathF.Abs(m.Translation.Y - 1.0f) < 2e-3f, $"y must survive, got {m.Translation.Y}");
        Assert.True(MathF.Abs(m.Translation.Z - 0.5f) < 2e-3f, $"z must survive, got {m.Translation.Z}");
    }

    /// <summary>
    /// O10: ground needs a walkable surface — a floor grounds, a wall
    /// does not, and a 60-degree ramp is too steep.
    /// </summary>
    [Fact]
    public void O10_GroundNeedsAWalkableSurface()
    {
        var cfg = new CharacterConfig3();
        var floor = Character3.Move(FloorWorld(), Vector3.Zero, cfg, Vector3.Zero, QueryFilter.Everything);
        Assert.True(floor.Grounded);
        Assert.Equal(Vector3.UnitY, floor.GroundNormal);

        var beside = Character3.Move(
            WallWorld(), new Vector3(1.6f, 0.0f, 0.0f), cfg, Vector3.Zero, QueryFilter.Everything);
        Assert.False(beside.Grounded, "a wall is not ground");

        var steep = Character3.Move(
            RampWorld(MathF.PI / 3.0f), new Vector3(2.0f, 0.5f, 0.0f), cfg, Vector3.Zero, QueryFilter.Everything);
        Assert.False(steep.Grounded, "a 60-degree slope is too steep for ground");
    }

    /// <summary>
    /// O11: the filter selects solid bodies, and a capsule body is
    /// solid through its analytic query.
    /// </summary>
    [Fact]
    public void O11_TheFilterAndCapsuleSolidity()
    {
        var cfg = new CharacterConfig3();
        var w = WallWorld();
        w.Bodies[0].CollisionGroup = 2;
        var through = Character3.Move(
            w, Vector3.Zero, cfg, new Vector3(5.0f, 0.0f, 0.0f), new QueryFilter(1u, 1u));
        Assert.True((through.Translation - new Vector3(5.0f, 0.0f, 0.0f)).Length() < 1e-5f);
        Assert.Equal(0, through.Contacts);

        var blocked = Character3.Move(
            w, Vector3.Zero, cfg, new Vector3(5.0f, 0.0f, 0.0f), new QueryFilter(1u, 2u));
        Assert.True(MathF.Abs(blocked.Translation.X - 1.6f) < 2e-3f);

        // A capsule wall: face at x = 1.7, radius 0.3 => 1.4, minus skin.
        var cw = new World3 { Gravity = Vector3.Zero };
        cw.AddStaticBody(new Vector3(2.0f, 0.0f, 0.0f), new CapsuleShape3(0.3f, 1.0f));
        var capsule = Character3.Move(cw, Vector3.Zero, cfg, new Vector3(4.0f, 0.0f, 0.0f), QueryFilter.Everything);
        Assert.True(MathF.Abs(capsule.Translation.X - 1.3f) < 2e-3f, $"capsule stop at {capsule.Translation.X}");
    }

    /// <summary>
    /// O12: a walkable ramp is climbed and grounds; an unsupported
    /// convex hull is walked through (the documented query gap).
    /// </summary>
    [Fact]
    public void O12_RampsClimbAndHullsAreNotSolid()
    {
        var cfg = new CharacterConfig3();
        var m = Character3.Move(
            RampWorld(MathF.PI / 4.0f), Vector3.Zero, cfg, new Vector3(3.0f, 0.0f, 0.0f), QueryFilter.Everything);
        Assert.True(m.Translation.Y > 0.9f, $"the slide must climb, got {m.Translation.Y}");
        Assert.True(m.Grounded, "a 45-degree ramp is walkable");

        var hullWorld = new World3 { Gravity = Vector3.Zero };
        hullWorld.AddStaticBody(new Vector3(2.0f, 0.0f, 0.0f), new ConvexHullShape3(new List<Vector3>
        {
            new(1.0f, -1.0f, -1.0f), new(3.0f, -1.0f, -1.0f),
            new(2.0f, 1.0f, -1.0f), new(2.0f, -1.0f, 1.0f),
        }));
        var through = Character3.Move(
            hullWorld, Vector3.Zero, cfg, new Vector3(4.0f, 0.0f, 0.0f), QueryFilter.Everything);
        Assert.True((through.Translation - new Vector3(4.0f, 0.0f, 0.0f)).Length() < 1e-5f,
            "a hull is walked through (documented gap)");
    }
}
