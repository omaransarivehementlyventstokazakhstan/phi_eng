using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Parallel island solver tests (P block): the parallel path must be
/// bit-identical to the serial path. Mirrors Rust parallel_tests.rs in
/// crates/engine/tests/parallel_tests.rs. Parity enforced by
/// tools/check_test_parity.py.
/// </summary>
public class ParallelTests
{
    private const float Dt = 1.0f / 60.0f;

    private static World2 TwoStacks(bool parallel)
    {
        var world = new World2();
        world.Solver.Parallel = parallel;
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        foreach (float stackX in new[] { -3.0f, 3.0f })
        {
            for (int i = 0; i < 3; i++)
            {
                world.AddDynamicBody(
                    new Vector2(stackX, 0.05f + i * 1.02f),
                    new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            }
        }
        return world;
    }

    private static ulong HashWorld(World2 world)
    {
        ulong h = 0xcbf29ce484222325UL;
        foreach (var body in world.Bodies)
        {
            foreach (float v in new[]
            {
                body.Transform.Position.X, body.Transform.Position.Y,
                body.Velocity.X, body.Velocity.Y,
                body.AngularVelocity, body.RotationAngle
            })
            {
                h ^= (ulong)BitConverter.SingleToInt32Bits(v);
                h *= 0x100000001b3UL;
            }
        }
        return h;
    }

    [Fact]
    public void P31_ParallelMatchesSerial()
    {
        var serial = TwoStacks(false);
        var parallel = TwoStacks(true);
        for (int i = 0; i < 300; i++)
        {
            serial.Step(Dt);
            parallel.Step(Dt);
        }
        Assert.Equal(HashWorld(serial), HashWorld(parallel));
    }

    [Fact]
    public void P32_JointConnectedMatchesSerial()
    {
        World2 Build(bool parallel)
        {
            var world = TwoStacks(parallel);
            world.AddJoint(new DistanceJoint(3, 6, Vector2.Zero, Vector2.Zero,
                6.0f, 0.0f, world.Bodies.Count));
            return world;
        }
        var serial = Build(false);
        var parallel = Build(true);
        for (int i = 0; i < 300; i++)
        {
            serial.Step(Dt);
            parallel.Step(Dt);
        }
        Assert.Equal(HashWorld(serial), HashWorld(parallel));
        foreach (int id in new[] { 3, 6 })
        {
            Assert.True(parallel.Bodies[id].Transform.Position.Y > 1.0f,
                $"stack top tunneled: y={parallel.Bodies[id].Transform.Position.Y}");
        }
    }

    [Fact]
    public void P33_SingleIslandFallback()
    {
        World2 Build(bool parallel)
        {
            var world = new World2();
            world.Solver.Parallel = parallel;
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
            world.AddDynamicBody(new Vector2(0.0f, 3.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            return world;
        }
        var serial = Build(false);
        var parallel = Build(true);
        for (int i = 0; i < 300; i++)
        {
            serial.Step(Dt);
            parallel.Step(Dt);
        }
        Assert.Equal(HashWorld(serial), HashWorld(parallel));
    }

    [Fact]
    public void P34_IslandPartitioning()
    {
        var islands = IslandManager.BuildIslands(6,
            new List<(int, int)> { (0, 1), (2, 3) },
            new List<(int, int)> { (4, 5) });
        Assert.Equal(3, islands.Count);
        Assert.Equal(new List<int> { 0, 1 }, islands[0].Bodies);
        Assert.Equal(new List<int> { 2, 3 }, islands[1].Bodies);
        Assert.Equal(new List<int> { 4, 5 }, islands[2].Bodies);
        var map = IslandManager.IslandIndexMap(islands, 6);
        var assigned = IslandManager.AssignConstraints(islands,
            new List<(int, int)> { (0, 1), (2, 3), (1, 0) }, map);
        Assert.Equal(new List<int> { 0, 2 }, assigned[0]);
        Assert.Equal(new List<int> { 1 }, assigned[1]);
        Assert.Empty(assigned[2]);
    }
}
