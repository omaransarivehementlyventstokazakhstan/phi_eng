using System.Numerics;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Bulk FFI transfer tests: the batch creation/read APIs added for
/// per-frame polling must agree with the scalar getters slot-for-slot,
/// reject bad descriptors per-slot without rollback, and surface stale
/// ids as NaN instead of aborting the batch. Test names deliberately
/// avoid the Letter+2digits parity-ID pattern (these are FFI-harness
/// tests, not mirrored robustness cases).
/// </summary>
public class NativeBulkTests
{
    private readonly ITestOutputHelper _output;

    public NativeBulkTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static bool NativeAvailable() => OperatingSystem.IsWindows();

    private void SkipMessage()
        => _output.WriteLine("skipped: staged native library is win-x64 only");

    [Fact]
    public void Bulk_DescriptorLayouts_MatchRust()
    {
        // Guards the cross-language repr(C) contract on the C# side.
        // Rust asserts the same sizes at compile time (ffi.rs).
        Assert.Equal(36, Marshal.SizeOf<BodyDesc2D>());
        Assert.Equal(44, Marshal.SizeOf<BodyDesc3D>());
    }

    [Fact]
    public void Bulk_AddBodies_MatchesScalarIds()
    {
        if (!NativeAvailable())
        {
            SkipMessage();
            return;
        }

        using var world = new PhiNativeWorld();
        var descs = new[]
        {
            BodyDesc2D.Circle(0.0f, 2.0f, 0.5f, 1.0f),
            BodyDesc2D.Box(1.0f, 2.0f, 0.5f, 0.5f, 1.0f),
            // Rejected descriptors: unknown tag, negative density.
            new BodyDesc2D(2.0f, 2.0f, 99, 0.5f, 0.0f, 1.0f, false, 0.5f, 0.0f),
            new BodyDesc2D(3.0f, 2.0f, PhiNativeWorld.ShapeCircle, 0.5f, 0.0f, -2.0f, false, 0.5f, 0.0f),
        };
        int[] ids = world.AddBodies(descs);
        Assert.Equal(4, ids.Length);
        Assert.True(ids[0] >= 0 && ids[1] >= 0 && ids[0] != ids[1]);
        Assert.Equal(-1, ids[2]);
        Assert.Equal(-1, ids[3]);
        Assert.Equal(2, world.BodyCount);

        // Same scene through the scalar path gets identical ids.
        using var scalar = new PhiNativeWorld();
        int s0 = scalar.AddDynamicBall(new Vector2(0.0f, 2.0f), 0.5f, 1.0f);
        int s1 = scalar.AddDynamicBox(new Vector2(1.0f, 2.0f), new Vector2(0.5f, 0.5f), 1.0f);
        Assert.Equal(new[] { s0, s1 }, new[] { ids[0], ids[1] });
    }

    [Fact]
    public void Bulk_GetTransforms_MatchesScalarReads()
    {
        if (!NativeAvailable())
        {
            SkipMessage();
            return;
        }

        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        float dt = 1.0f / 60.0f;
        for (int i = 0; i < 60; i++)
        {
            world.Step(dt);
        }

        var query = new uint[] { 0, (uint)ball };
        var pos = new Vector2[2];
        var rot = new float[2];
        var vel = new Vector2[2];
        int valid = world.GetTransformsBulk(query, pos, rot, vel);
        Assert.Equal(2, valid);

        var (p0, r0) = world.GetBodyTransform(0);
        var (p1, r1) = world.GetBodyTransform(ball);
        var v0 = world.GetBodyVelocity(0);
        var v1 = world.GetBodyVelocity(ball);
        Assert.Equal(p0, pos[0]);
        Assert.Equal(r0, rot[0]);
        Assert.Equal(v0, vel[0]);
        Assert.Equal(p1, pos[1]);
        Assert.Equal(r1, rot[1]);
        Assert.Equal(v1, vel[1]);
        _output.WriteLine($"[bulk] ball pos=({pos[1].X:F3},{pos[1].Y:F3}) — fell from y=5 as expected: {pos[1].Y < 5.0f}");

        // Null velocities: positions still resolve, same count.
        var pos2 = new Vector2[2];
        var rot2 = new float[2];
        int valid2 = world.GetTransformsBulk(query, pos2, rot2, null);
        Assert.Equal(2, valid2);
        Assert.Equal(pos, pos2);
    }

    [Fact]
    public void Bulk_StaleId_WritesNaNWithoutAbort()
    {
        if (!NativeAvailable())
        {
            SkipMessage();
            return;
        }

        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        world.RemoveBody(ball); // ball id is now stale; floor reindexes to 0.

        var query = new uint[] { 0, (uint)ball };
        var pos = new Vector2[2];
        var rot = new float[2];
        var vel = new Vector2[2];
        int valid = world.GetTransformsBulk(query, pos, rot, vel);
        Assert.Equal(1, valid);
        Assert.True(float.IsFinite(pos[0].X));
        Assert.True(float.IsNaN(pos[1].X) && float.IsNaN(pos[1].Y) && float.IsNaN(rot[1]));
        Assert.True(float.IsNaN(vel[1].X) && float.IsNaN(vel[1].Y));
    }

    [Fact]
    public void Bulk_TryGetters_MatchThrowing()
    {
        if (!NativeAvailable())
        {
            SkipMessage();
            return;
        }

        using var world = new PhiNativeWorld();
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);

        Assert.True(world.TryGetBodyTransform(ball, out var pos, out var rot));
        var (p, r) = world.GetBodyTransform(ball);
        Assert.Equal(p, pos);
        Assert.Equal(r, rot);

        Assert.True(world.TryGetBodyVelocity(ball, out var vel));
        Assert.Equal(world.GetBodyVelocity(ball), vel);

        Assert.False(world.TryGetBodyTransform(999, out _, out _));
        Assert.False(world.TryGetBodyVelocity(999, out _));
    }

    [Fact]
    public void Bulk3_AddAndRead_Roundtrip()
    {
        if (!NativeAvailable())
        {
            SkipMessage();
            return;
        }

        using var world = new PhiNativeWorld3();
        var descs = new[]
        {
            new BodyDesc3D(0.0f, 2.0f, 0.0f, PhiNativeWorld3.ShapeSphere,
                0.5f, 0.0f, 0.0f, 1.0f, false, 0.5f, 0.0f),
            new BodyDesc3D(1.0f, 2.0f, 0.0f, PhiNativeWorld3.ShapeBox,
                0.5f, 0.5f, 0.5f, 1.0f, false, 0.5f, 0.0f),
        };
        int[] ids = world.AddBodies(descs);
        Assert.Equal(2, ids.Length);
        Assert.True(ids[0] >= 0 && ids[1] >= 0 && ids[0] != ids[1]);
        Assert.Equal(2, world.BodyCount);

        var query = new uint[] { (uint)ids[1], 999 };
        var pos = new Vector3[2];
        var quat = new Quaternion[2];
        int valid = world.GetTransformsBulk(query, pos, quat, null);
        Assert.Equal(1, valid);

        var (sp, sq) = world.GetBodyTransform(ids[1]);
        Assert.Equal(sp, pos[0]);
        Assert.Equal(sq, quat[0]);
        Assert.True(float.IsNaN(pos[1].X) && float.IsNaN(quat[1].X));

        Assert.True(world.TryGetBodyTransform(ids[0], out _, out _));
        Assert.False(world.TryGetBodyTransform(999, out _, out _));
        Assert.True(world.TryGetBodyVelocity(ids[0], out _));
        Assert.False(world.TryGetBodyVelocity(999, out _));
    }
}
