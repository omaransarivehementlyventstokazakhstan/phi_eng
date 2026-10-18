using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// CCD tests (C01–C04): conservative advancement via adaptive
/// substepping. Mirrors <c>crates/engine/tests/ccd.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class CcdTests
{
    private const float Dt = 1.0f / 60.0f;

    private static CcdConfig CcdOn() => new() { Enabled = true, MaxMovement = 0.05f, MaxSubsteps = 64 };

    private static void Calm(World2 world)
    {
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        foreach (var b in world.Bodies)
        {
            b.LinearDamping = 0.0f;
            b.AngularDamping = 0.0f;
            b.SetMaterial(new PhysicsMaterial(b.Material.Friction, 0.0f, b.Material.Density));
        }
    }

    private static void Calm(World3 world)
    {
        world.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        foreach (var b in world.Bodies)
        {
            b.LinearDamping = 0.0f;
            b.AngularDamping = 0.0f;
            b.SetMaterial(new PhysicsMaterial(b.Material.Friction, 0.0f, b.Material.Density));
        }
    }

    private static (World2 World, int Ball) ThinWall2D(float speedY)
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(5.0f, 0.05f)));
        int ball = world.AddDynamicBody(new Vector2(0.0f, 3.0f), new CircleShape(0.25f), 1.0f);
        world.Bodies[ball].Velocity = new Vector2(0.0f, speedY);
        Calm(world);
        return (world, ball);
    }

    private static (World3 World, int Ball) ThinWall3D(float speedY)
    {
        var world = new World3();
        world.AddStaticBody(new Vector3(0.0f, 0.0f, 0.0f), new BoxShape3(new Vector3(5.0f, 0.05f, 5.0f)));
        int ball = world.AddDynamicBody(new Vector3(0.0f, 3.0f, 0.0f), new SphereShape3(0.25f), 1.0f);
        world.Bodies[ball].Velocity = new Vector3(0.0f, speedY, 0.0f);
        Calm(world);
        return (world, ball);
    }

    /// <summary>C01: a fast ball tunnels through a thin wall with CCD off, but rests on it with CCD on.</summary>
    [Fact]
    public void C01_ThinWall2D()
    {
        var (off, offBall) = ThinWall2D(-60.0f);
        for (int i = 0; i < 120; i++) off.Step(Dt);
        Assert.True(off.Bodies[offBall].Transform.Position.Y < -1.0f,
            $"without CCD the fast ball should tunnel, y={off.Bodies[offBall].Transform.Position.Y}");

        var (on, onBall) = ThinWall2D(-60.0f);
        on.Ccd = CcdOn();
        for (int i = 0; i < 120; i++) on.Step(Dt);
        float y = on.Bodies[onBall].Transform.Position.Y;
        Assert.True(y > 0.2f && y < 0.6f, $"with CCD the ball should rest on the wall, y={y}");
        Assert.True(on.Bodies[onBall].Velocity.Length() < 1.0f, "ball should have settled");
    }

    /// <summary>C02: the 3D mirror — a fast sphere tunnels with CCD off, but is caught with CCD on.</summary>
    [Fact]
    public void C02_ThinWall3D()
    {
        var (off, offBall) = ThinWall3D(-120.0f);
        for (int i = 0; i < 120; i++) off.Step(Dt);
        Assert.True(off.Bodies[offBall].Transform.Position.Y < -1.0f,
            $"without CCD the fast sphere should tunnel, y={off.Bodies[offBall].Transform.Position.Y}");

        var (on, onBall) = ThinWall3D(-120.0f);
        on.Ccd = CcdOn();
        for (int i = 0; i < 120; i++) on.Step(Dt);
        float y = on.Bodies[onBall].Transform.Position.Y;
        Assert.True(y > 0.2f && y < 0.6f, $"with CCD the sphere should rest on the slab, y={y}");
    }

    /// <summary>C03: CCD enabled but below the speed threshold takes exactly one substep — bit-identical to off.</summary>
    [Fact]
    public void C03_DefaultOffBitIdentical()
    {
        World2 Build()
        {
            var world = new World2();
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
            Calm(world);
            return world;
        }

        var off = Build();
        var on = Build();
        on.Ccd = new CcdConfig { Enabled = true, MaxMovement = 10.0f, MaxSubsteps = 8 };
        for (int i = 0; i < 300; i++)
        {
            off.Step(Dt);
            on.Step(Dt);
        }
        Assert.Equal(off.Bodies.Count, on.Bodies.Count);
        for (int i = 0; i < off.Bodies.Count; i++)
        {
            Assert.Equal(off.Bodies[i].Transform.Position, on.Bodies[i].Transform.Position);
            Assert.Equal(off.Bodies[i].Velocity, on.Bodies[i].Velocity);
        }
        Assert.True(System.Math.Abs(off.Time - 5.0f) < 1e-4f, "clock must advance once per step");

        string json = Snapshot.World2ToJson(off, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 3", json);
        var back = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.True(back.Ccd.IsDefault());
    }

    /// <summary>C04: a CCD world round-trips its config through JSON and binary and keeps stopping the fast ball.</summary>
    [Fact]
    public void C04_CcdSnapshotRoundtrip()
    {
        var (world, ball) = ThinWall2D(-60.0f);
        world.Ccd = CcdOn();

        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        Assert.Contains("\"ccd\"", json);
        Assert.Contains("\"schema\": 6", json);
        var rj = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.True(rj.Ccd.IsActive());
        Assert.Equal(0.05f, rj.Ccd.MaxMovement);
        Assert.Equal(64, rj.Ccd.MaxSubsteps);

        byte[] bin = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var rb = Snapshot.World2FromBinary(bin, SnapshotLoadMode.Exact);
        Assert.True(rb.Ccd.IsActive());
        Assert.Equal(rj.Ccd.MaxMovement, rb.Ccd.MaxMovement);
        Assert.Equal(rj.Ccd.MaxSubsteps, rb.Ccd.MaxSubsteps);

        rj.Bodies[ball].Velocity = new Vector2(0.0f, -60.0f);
        rb.Bodies[ball].Velocity = new Vector2(0.0f, -60.0f);
        for (int i = 0; i < 120; i++)
        {
            world.Step(Dt);
            rj.Step(Dt);
            rb.Step(Dt);
        }
        Assert.True(world.Bodies[ball].Transform.Position.Y is var y0 && y0 > 0.2f && y0 < 0.6f, $"live world tunneled, y={y0}");
        Assert.True(rj.Bodies[ball].Transform.Position.Y is var y1 && y1 > 0.2f && y1 < 0.6f, $"json world tunneled, y={y1}");
        Assert.True(rb.Bodies[ball].Transform.Position.Y is var y2 && y2 > 0.2f && y2 < 0.6f, $"binary world tunneled, y={y2}");
    }
}
