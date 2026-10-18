using System.Collections.Concurrent;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Cross-language integration tests: C# drives the Rust physics core through
/// the FFI bridge (PhiEng.Native -> phieng.dll) and reads
/// transforms back. The native DLL is staged next to the test binaries by
/// the test project's None/CopyToOutputDirectory item.
/// </summary>
public class NativeInteropTests
{
    private readonly ITestOutputHelper _output;

    public NativeInteropTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private void Log(string line)
    {
        _output.WriteLine(line);
        Console.WriteLine(line);
    }

    private static bool NativeAvailable() => OperatingSystem.IsWindows();

    [Fact]
    public void Native_BallDrop_FallsTowardFloor()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        Assert.Equal(0, world.BodyCount);

        int floor = world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        Assert.Equal(0, floor);
        Assert.Equal(1, ball);
        Assert.Equal(2, world.BodyCount);

        float dt = 1.0f / 60.0f;
        for (int frame = 0; frame <= 120; frame++)
        {
            world.Step(dt);
            if (frame % 30 == 0)
            {
                var (pos, rot) = world.GetBodyTransform(ball);
                var vel = world.GetBodyVelocity(ball);
                Log($"[native] frame {frame}: pos=({pos.X:F3},{pos.Y:F3}) vel=({vel.X:F3},{vel.Y:F3}) rot={rot:F3}");
            }
        }

        var (endPos, _) = world.GetBodyTransform(ball);
        Assert.True(endPos.Y < 5.0f, $"ball should have fallen via native stepping, y={endPos.Y}");
        Assert.True(endPos.Y > -1.0f, $"ball tunneled through native floor, y={endPos.Y}");
    }

    [Fact]
    public void Native_BoxDrop_RestsOnFloor()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int box = world.AddBoxBody(new Vector2(0.0f, 3.0f), new Vector2(0.5f, 0.5f), 1.0f, isStatic: false);

        for (int i = 0; i < 600; i++)
        {
            world.Step(1.0f / 60.0f);
        }

        var (pos, rot) = world.GetBodyTransform(box);
        var vel = world.GetBodyVelocity(box);
        Log($"[native] box rest: pos=({pos.X:F3},{pos.Y:F3}) vel=({vel.X:F3},{vel.Y:F3}) rot={rot:F3}");
        Assert.True(pos.Y > -0.5f, $"native box tunneled through floor: y={pos.Y}");
        Assert.True(vel.Length() < 1.0f, $"native box should have settled: v={vel}");
    }

    [Fact]
    public void Native_UnknownBodyId_Throws()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyTransform(999));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyVelocity(999));
    }

    [Fact]
    public void Native_BodyMaterial_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        int ball = world.AddDynamicBall(new Vector2(0.0f, 1.0f), 0.5f, 1.0f);

        // Defaults come back intact (wrapper defaults mirror engine defaults).
        var created = world.GetBodyMaterial(ball);
        Log($"[native] created material: mu={created.Friction:F3} rho={created.Density:F3} e={created.Restitution:F3}");
        Assert.True(MathF.Abs(created.Friction - 0.4f) < 1e-6f);
        Assert.True(MathF.Abs(created.Density - 1.0f) < 1e-6f);
        Assert.True(MathF.Abs(created.Restitution - 0.1f) < 1e-6f);

        // Modification round-trips; unknown ids throw.
        world.SetBodyMaterial(ball, new Dynamics.PhysicsMaterial(0.9f, 0.8f, 2.0f));
        var updated = world.GetBodyMaterial(ball);
        Log($"[native] updated material: mu={updated.Friction:F3} rho={updated.Density:F3} e={updated.Restitution:F3}");
        Assert.True(MathF.Abs(updated.Friction - 0.9f) < 1e-6f);
        Assert.True(MathF.Abs(updated.Density - 2.0f) < 1e-6f);
        Assert.True(MathF.Abs(updated.Restitution - 0.8f) < 1e-6f);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.SetBodyMaterial(999, new Dynamics.PhysicsMaterial()));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyMaterial(999));
    }

    [Fact]
    public void Native_CcdConfig_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        var cfg = new Dynamics.CcdConfig
        {
            Enabled = true,
            MaxMovement = 0.05f,
            MaxSubsteps = 64,
        };
        world.SetCcdConfig(cfg);
        var back = world.GetCcdConfig();
        Assert.True(back.Enabled);
        Assert.Equal(0.05f, back.MaxMovement);
        Assert.Equal(64, back.MaxSubsteps);

        // Bad configs rejected without mutating the world.
        var badMovement = new Dynamics.CcdConfig { Enabled = true, MaxMovement = 0.0f, MaxSubsteps = 64 };
        Assert.Throws<ArgumentException>(() => world.SetCcdConfig(badMovement));
        var badSteps = new Dynamics.CcdConfig { Enabled = true, MaxMovement = 0.05f, MaxSubsteps = 0 };
        Assert.Throws<ArgumentException>(() => world.SetCcdConfig(badSteps));
        Assert.Equal(0.05f, world.GetCcdConfig().MaxMovement);
        Assert.Equal(64, world.GetCcdConfig().MaxSubsteps);
    }

    [Fact]
    public void Native_FluidConfig_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        var cfg = new Dynamics.FluidConfig2
        {
            ParticleMass = 0.125f,
            RestDensity = 2.0f,
            SmoothingLength = 0.75f,
            Stiffness = 200.0f,
            Gamma = 7.0f,
            ArtificialViscosity = 0.1f,
            SurfaceTension = 0.05f,
            Cfl = 0.4f,
            MaxSubsteps = 4,
            AdaptiveResolution = true,
            SplitDensityRatio = 0.4f,
            MergeDensityRatio = 1.8f,
            BoundaryParticles = true,
            Viscosity = 0.15f,
            Drag = 2.0f,
            WallFriction = 0.3f,
            PhaseCount = 2,
            PhaseRestDensity = new float[] { 1.0f, 2.0f, 1.0f, 1.0f },
            PhaseMass = new float[] { 0.125f, 0.25f, 0.125f, 0.125f },
            MaxVelocity = 50.0f,
            ParticleRadius = 0.25f,
            Enabled = true,
        };
        world.SetFluidConfig(cfg);
        var back = world.GetFluidConfig();
        Assert.Equal(0.125f, back.ParticleMass);
        Assert.Equal(2.0f, back.RestDensity);
        Assert.Equal(7.0f, back.Gamma);
        Assert.Equal(4, back.MaxSubsteps);
        Assert.True(back.AdaptiveResolution);
        Assert.True(back.BoundaryParticles);
        Assert.Equal((byte)2, back.PhaseCount);
        Assert.Equal(2.0f, back.PhaseRestDensity[1]);
        Assert.Equal(0.25f, back.PhaseMass[1]);

        // Bad config rejected without mutating the world.
        var bad = new Dynamics.FluidConfig2
        {
            ParticleMass = 0.125f,
            RestDensity = 2.0f,
            SmoothingLength = 0.75f,
            Stiffness = 200.0f,
            Gamma = 0.5f,
            PhaseCount = 2,
            PhaseRestDensity = new float[] { 1.0f, 2.0f, 1.0f, 1.0f },
            PhaseMass = new float[] { 0.125f, 0.25f, 0.125f, 0.125f },
            MaxSubsteps = 4,
        };
        Assert.Throws<ArgumentException>(() => world.SetFluidConfig(bad));
        Assert.Equal(7.0f, world.GetFluidConfig().Gamma);

        // Two-phase particle insert.
        var added = world.AddFluidParticlesPhase(
            new[] { new Vector2(0.0f, 0.0f), new Vector2(1.0f, 0.0f), new Vector2(2.0f, 0.0f) },
            new byte[] { 0, 1, 0 });
        Assert.Equal(3u, added);
        Assert.Equal(3u, world.FluidCount);
        world.FluidClear();
        Assert.Equal(0u, world.FluidCount);
    }

    [Fact]
    public void Native_RemoveBody_Reindexes()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        Assert.Equal(2, world.BodyCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.RemoveBody(999));
        world.RemoveBody(0);
        Assert.Equal(1, world.BodyCount);
        var (pos, _) = world.GetBodyTransform(0);
        Assert.True(pos.Y > 4.0f, $"reindexed ball should be id 0, y={pos.Y}");
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyTransform(1));
    }

    [Fact]
    public void Native3D_BallDrop_FallsTowardFloor()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld3();
        Assert.Equal(0, world.BodyCount);
        int floor = world.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        int ball = world.AddDynamicBall(new Vector3(0.0f, 5.0f, 0.0f), 0.5f, 1.0f);
        Assert.Equal(0, floor);
        Assert.Equal(1, ball);

        float dt = 1.0f / 60.0f;
        for (int i = 0; i < 120; i++) world.Step(dt);

        int capsule = world.AddCapsuleBody(new Vector3(2.0f, 5.0f, 0.0f), 0.3f, 0.5f, 1.0f, isStatic: false);
        Assert.Equal(2, capsule);

        var (pos, rot) = world.GetBodyTransform(ball);
        var vel = world.GetBodyVelocity(ball);
        Log($"[native3d] ball rest: pos=({pos.X:F3},{pos.Y:F3},{pos.Z:F3}) vel={vel.Length():F3}");
        Assert.True(pos.Y < 5.0f, $"ball should have fallen, y={pos.Y}");
        Assert.True(pos.Y > -0.5f, $"ball tunneled through floor, y={pos.Y}");
        float qlen = MathF.Sqrt(rot.X * rot.X + rot.Y * rot.Y + rot.Z * rot.Z + rot.W * rot.W);
        Assert.True(MathF.Abs(qlen - 1.0f) < 1e-4f, $"quaternion must stay unit, len={qlen}");
    }

    [Fact]
    public void Native3D_BodyMaterial_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld3();
        int ball = world.AddDynamicBall(new Vector3(0.0f, 1.0f, 0.0f), 0.5f, 1.0f);
        var created = world.GetBodyMaterial(ball);
        Assert.True(MathF.Abs(created.Friction - 0.4f) < 1e-6f);
        Assert.True(MathF.Abs(created.Density - 1.0f) < 1e-6f);
        Assert.True(MathF.Abs(created.Restitution - 0.1f) < 1e-6f);

        world.SetBodyMaterial(ball, new Dynamics.PhysicsMaterial(0.9f, 0.8f, 2.0f));
        var updated = world.GetBodyMaterial(ball);
        Assert.True(MathF.Abs(updated.Friction - 0.9f) < 1e-6f);
        Assert.True(MathF.Abs(updated.Density - 2.0f) < 1e-6f);
        Assert.True(MathF.Abs(updated.Restitution - 0.8f) < 1e-6f);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.SetBodyMaterial(999, new Dynamics.PhysicsMaterial()));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyMaterial(999));
    }

    [Fact]
    public void Native3D_FluidConfig_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld3();
        var cfg = new Dynamics.FluidConfig3
        {
            ParticleMass = 0.125f,
            RestDensity = 2.0f,
            SmoothingLength = 0.75f,
            Stiffness = 200.0f,
            Gamma = 7.0f,
            PhaseCount = 2,
            PhaseRestDensity = new float[] { 1.0f, 2.0f, 1.0f, 1.0f },
            PhaseMass = new float[] { 0.125f, 0.25f, 0.125f, 0.125f },
        };
        world.SetFluidConfig(cfg);
        var back = world.GetFluidConfig();
        Assert.Equal((byte)2, back.PhaseCount);
        Assert.Equal(7.0f, back.Gamma);
        var bad = new Dynamics.FluidConfig3
        {
            ParticleMass = 0.0f,
            RestDensity = 2.0f,
            SmoothingLength = 0.75f,
            Stiffness = 200.0f,
            Gamma = 7.0f,
            PhaseCount = 2,
            PhaseRestDensity = new float[] { 1.0f, 2.0f, 1.0f, 1.0f },
            PhaseMass = new float[] { 0.125f, 0.25f, 0.125f, 0.125f },
        };
        Assert.Throws<ArgumentException>(() => world.SetFluidConfig(bad));
        var added = world.AddFluidParticlesPhase(
            new[] { new Vector3(0.0f, 0.0f, 0.0f), new Vector3(1.0f, 0.0f, 0.0f) },
            new byte[] { 1, 0 });
        Assert.Equal(2u, added);
        Assert.Equal(2u, world.FluidCount);
        world.FluidClear();
        Assert.Equal(0u, world.FluidCount);
    }

    [Fact]
    public void Native3D_CcdConfig_Roundtrips()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld3();
        var cfg = new Dynamics.CcdConfig
        {
            Enabled = true,
            MaxMovement = 0.05f,
            MaxSubsteps = 32,
        };
        world.SetCcdConfig(cfg);
        var back = world.GetCcdConfig();
        Assert.True(back.Enabled);
        Assert.Equal(0.05f, back.MaxMovement);
        Assert.Equal(32, back.MaxSubsteps);
        var bad = new Dynamics.CcdConfig { Enabled = true, MaxMovement = -1.0f, MaxSubsteps = 32 };
        Assert.Throws<ArgumentException>(() => world.SetCcdConfig(bad));
        Assert.Equal(0.05f, world.GetCcdConfig().MaxMovement);
    }

    [Fact]
    public void Native3D_RemoveBody_Reindexes()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        using var world = new PhiNativeWorld3();
        world.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        world.AddDynamicBall(new Vector3(0.0f, 5.0f, 0.0f), 0.5f, 1.0f);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.RemoveBody(999));
        world.RemoveBody(0);
        Assert.Equal(1, world.BodyCount);
        var (pos, _) = world.GetBodyTransform(0);
        Assert.True(pos.Y > 4.0f, $"reindexed ball should be id 0, y={pos.Y}");
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyTransform(1));
    }

    [Fact]
    public void Native_Dispose_IsSafeAndFinal()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var world = new PhiNativeWorld();
        world.AddDynamicBall(new Vector2(0.0f, 1.0f), 0.5f, 1.0f);
        world.Step(1.0f / 60.0f);
        world.Dispose();
        world.Dispose(); // double-dispose must not crash
        Assert.Throws<ObjectDisposedException>(() => world.Step(1.0f / 60.0f));
        Assert.Throws<ObjectDisposedException>(() => world.GetBodyTransform(0));
    }

    [Fact]
    public void F29_BridgeRejectsUnknownShapeTag()
    {
        // The bridge exposes Circle/Box (2D) and Sphere/Box/Capsule (3D)
        // only: polygon/hull have no tag, and unknown tags must fail
        // cleanly (-1), never UB. See LIMITATIONS.md §FFI shape subset.
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        IntPtr world2d = NativeMethods.phieng_world_new();
        Assert.NotEqual(IntPtr.Zero, world2d);
        try
        {
            Assert.Equal(-1, NativeMethods.phieng_world_add_body(
                world2d, 0, 0, 99u, 1, 1, 1, 0, 0.5f, 0.0f));
            Assert.Equal(-1, NativeMethods.phieng_world_add_body(
                world2d, 0, 0, 2u, 1, 1, 1, 0, 0.5f, 0.0f)); // no tag 2 in 2D
            Assert.Equal(0, NativeMethods.phieng_world_body_count(world2d));
        }
        finally
        {
            NativeMethods.phieng_world_free(world2d);
        }

        IntPtr world3d = NativeMethods.phieng_world3_new();
        Assert.NotEqual(IntPtr.Zero, world3d);
        try
        {
            Assert.Equal(-1, NativeMethods.phieng_world3_add_body(
                world3d, 0, 0, 0, 99u, 1, 1, 1, 1, 0, 0.5f, 0.0f));
            Assert.Equal(0, NativeMethods.phieng_world3_body_count(world3d));
        }
        finally
        {
            NativeMethods.phieng_world3_free(world3d);
        }
    }

    [Fact]
    public async Task F30_DisposeUnderConcurrentStep()
    {
        // The gate lock serializes native calls (Rust step takes &mut) and
        // closes the Dispose/Step check-then-use race. Under hammering,
        // the only benign worker outcome is ObjectDisposedException —
        // never a native crash (use-after-free) or double-free.
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var world = new PhiNativeWorld();
        world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        var errors = new ConcurrentBag<Exception>();
        using var go = new ManualResetEventSlim(false);
        var tasks = new List<Task>();
        for (int i = 0; i < 4; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                go.Wait();
                try
                {
                    for (int s = 0; s < 200; s++) world.Step(1.0f / 60.0f);
                }
                catch (Exception ex) { errors.Add(ex); }
            }));
        }
        tasks.Add(Task.Run(() =>
        {
            go.Wait();
            Thread.Sleep(1);
            for (int d = 0; d < 3; d++) world.Dispose(); // concurrent + repeated dispose
        }));
        go.Set();
        await Task.WhenAll(tasks);
        foreach (var ex in errors)
            Assert.IsType<ObjectDisposedException>(ex);
        Assert.Throws<ObjectDisposedException>(() => world.Step(1.0f / 60.0f));
    }
}
