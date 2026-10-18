using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Solver quality-tier tests (Q01–Q05): iteration presets plus
/// friction/restitution switches. Mirrored by
/// <c>crates/engine/tests/quality.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class QualityTests
{
    private const float Dt = 1.0f / 60.0f;

    private static void Calm(World2 world)
    {
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        foreach (var b in world.Bodies)
        {
            b.LinearDamping = 0.0f;
            b.AngularDamping = 0.0f;
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
        }
    }

    /// <summary>Q01: quality tiers map to the documented iteration budgets and feature switches.</summary>
    [Fact]
    public void Q01_TierPresets()
    {
        var solver = new Solver();
        Assert.Equal(8, solver.Iterations);
        Assert.True(solver.EnableFriction && solver.EnableRestitution);
        solver.ApplyQuality(SolverQuality.Ultra);
        Assert.Equal((8, true, true), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));
        solver.ApplyQuality(SolverQuality.High);
        Assert.Equal((6, true, true), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));
        solver.ApplyQuality(SolverQuality.Medium);
        Assert.Equal((4, true, true), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));
        solver.ApplyQuality(SolverQuality.Low);
        Assert.Equal((2, false, false), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));
        solver.ApplyQuality(SolverQuality.Minimal);
        Assert.Equal((1, false, false), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));
        solver.ApplyQuality(SolverQuality.Ultra);
        Assert.Equal((8, true, true), (solver.Iterations, solver.EnableFriction, solver.EnableRestitution));

        var solver3 = new Solver3D();
        solver3.ApplyQuality(SolverQuality.Low);
        Assert.Equal((2, false, false), (solver3.Iterations, solver3.EnableFriction, solver3.EnableRestitution));
        solver3.ApplyQuality(SolverQuality.High);
        Assert.Equal((6, true, true), (solver3.Iterations, solver3.EnableFriction, solver3.EnableRestitution));
    }

    private static (World2 World, int Box) SlideScene2D(bool friction)
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -0.5f), new BoxShape2(new Vector2(50.0f, 0.5f)));
        int box = world.AddDynamicBody(new Vector2(0.0f, 0.5f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        Calm(world);
        world.Bodies[box].Velocity = new Vector2(5.0f, 0.0f);
        if (!friction)
        {
            world.Solver.ApplyQuality(SolverQuality.Low);
            // Low also cuts iterations and restitution; restore those so
            // the test isolates friction only.
            world.Solver.Iterations = 8;
            world.Solver.EnableRestitution = true;
        }
        return (world, box);
    }

    /// <summary>Q02: with friction off, a sliding box keeps its speed; with friction on it stops.</summary>
    [Fact]
    public void Q02_FrictionOffSlides()
    {
        var (on, onBox) = SlideScene2D(true);
        for (int i = 0; i < 120; i++) on.Step(Dt);
        Assert.True(System.Math.Abs(on.Bodies[onBox].Velocity.X) < 0.5f,
            $"friction should stop the box, vx={on.Bodies[onBox].Velocity.X}");

        var (off, offBox) = SlideScene2D(false);
        for (int i = 0; i < 120; i++) off.Step(Dt);
        Assert.True(off.Bodies[offBox].Velocity.X > 4.5f,
            $"without friction the box should keep sliding, vx={off.Bodies[offBox].Velocity.X}");
    }

    private static (World2 World, int Ball) BounceScene(bool restitution)
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -0.5f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int ball = world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        Calm(world);
        foreach (var b in world.Bodies)
        {
            b.SetMaterial(new PhysicsMaterial(b.Material.Friction, 1.0f, b.Material.Density));
        }
        if (!restitution)
        {
            world.Solver.EnableRestitution = false;
        }
        return (world, ball);
    }

    /// <summary>Q03: with restitution off, a dropped ball does not bounce.</summary>
    [Fact]
    public void Q03_RestitutionOffNoBounce()
    {
        var (on, onBall) = BounceScene(true);
        bool impacted = false;
        float peak = 0.0f;
        for (int i = 0; i < 120; i++)
        {
            on.Step(Dt);
            float y = on.Bodies[onBall].Transform.Position.Y;
            if (y < 1.0f) impacted = true;
            if (impacted) peak = System.Math.Max(peak, y);
        }
        Assert.True(impacted, "ball never reached the floor");
        Assert.True(peak > 1.5f, $"elastic ball should bounce high, peak={peak}");

        var (off, offBall) = BounceScene(false);
        impacted = false;
        float peakOff = 0.0f;
        for (int i = 0; i < 120; i++)
        {
            off.Step(Dt);
            float y = off.Bodies[offBall].Transform.Position.Y;
            if (y < 1.0f) impacted = true;
            if (impacted) peakOff = System.Math.Max(peakOff, y);
        }
        Assert.True(impacted, "ball never reached the floor");
        Assert.True(peakOff < 1.0f, $"inelastic ball must never leave rest height, peak={peakOff}");
        float rest = off.Bodies[offBall].Transform.Position.Y;
        Assert.True(rest > 0.3f && rest < 0.7f, $"inelastic ball should rest on the floor, y={rest}");
    }

    /// <summary>Q04: the 3D solver honors the friction switch — same slide test.</summary>
    [Fact]
    public void Q04_FrictionOffSlides3D()
    {
        (World3 World, int Box) Build(bool friction)
        {
            var world = new World3();
            world.AddStaticBody(new Vector3(0.0f, -0.5f, 0.0f), new BoxShape3(new Vector3(50.0f, 0.5f, 50.0f)));
            int box = world.AddDynamicBody(new Vector3(0.0f, 0.5f, 0.0f), new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)), 1.0f);
            Calm(world);
            world.Bodies[box].Velocity = new Vector3(5.0f, 0.0f, 0.0f);
            if (!friction)
            {
                world.Solver.ApplyQuality(SolverQuality.Low);
                world.Solver.Iterations = 8;
                world.Solver.EnableRestitution = true;
            }
            return (world, box);
        }

        var (on, onBox) = Build(true);
        for (int i = 0; i < 120; i++) on.Step(Dt);
        Assert.True(System.Math.Abs(on.Bodies[onBox].Velocity.X) < 0.5f,
            $"3D friction should stop the box, vx={on.Bodies[onBox].Velocity.X}");

        var (off, offBox) = Build(false);
        for (int i = 0; i < 120; i++) off.Step(Dt);
        Assert.True(off.Bodies[offBox].Velocity.X > 4.5f,
            $"3D without friction the box should keep sliding, vx={off.Bodies[offBox].Velocity.X}");
    }

    /// <summary>Q05: solver flags round-trip through JSON (schema v7) and a default world stays on its old schema.</summary>
    [Fact]
    public void Q05_SolverFlagsSnapshotRoundtrip()
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 3", json);
        var back = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.True(back.Solver.EnableFriction);
        Assert.True(back.Solver.EnableRestitution);

        world.Solver.ApplyQuality(SolverQuality.Low);
        string flagged = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 7", flagged);
        var rj = Snapshot.World2FromJson(flagged, SnapshotLoadMode.Exact);
        Assert.False(rj.Solver.EnableFriction);
        Assert.False(rj.Solver.EnableRestitution);
        Assert.Equal(2, rj.Solver.Iterations);

        byte[] bin = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var rb = Snapshot.World2FromBinary(bin, SnapshotLoadMode.Exact);
        Assert.False(rb.Solver.EnableFriction);
        Assert.False(rb.Solver.EnableRestitution);
        Assert.Equal(2, rb.Solver.Iterations);
    }
}
