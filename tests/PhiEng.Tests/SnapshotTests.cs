using System.Numerics;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Snapshot save/load tests, mirroring the Rust snapshot suite in
/// crates/dynamics/src/snapshot.rs and crates/engine/tests/snapshot_tests.rs.
/// Test IDs (E01–E07) are shared across languages; parity is enforced by
/// tools/check_test_parity.py. F34–F35 pin the native save/load bridge
/// (mirrored by f34–f35 in crates/engine/src/ffi.rs).
/// </summary>
public class SnapshotTests
{
    private const float Dt = 1.0f / 60.0f;

    private static World2 Scene()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        world.AddDynamicBody(new Vector2(1.0f, 3.0f), new BoxShape2(new Vector2(0.4f, 0.4f)), 1.5f);
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
    public void E01_JsonExactRoundtrip()
    {
        var world = Scene();
        for (int i = 0; i < 120; i++) world.Step(Dt);
        string text = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        var restored = Snapshot.World2FromJson(text, SnapshotLoadMode.Exact);
        Assert.Equal(HashWorld(world), HashWorld(restored));
        Assert.Equal(world.Bodies.Count, restored.Bodies.Count);
        Assert.Equal(world.NextBodyId, restored.NextBodyId);
    }

    [Fact]
    public void E02_BinaryExactRoundtrip()
    {
        var world = Scene();
        for (int i = 0; i < 120; i++) world.Step(Dt);
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var restored = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.Equal(HashWorld(world), HashWorld(restored));
    }

    [Fact]
    public void E03_ContinuationMatches()
    {
        var world = Scene();
        for (int i = 0; i < 60; i++) world.Step(Dt);
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        var restored = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        for (int i = 0; i < 60; i++)
        {
            world.Step(Dt);
            restored.Step(Dt);
        }
        Assert.Equal(HashWorld(world), HashWorld(restored));
    }

    [Fact]
    public void E04_FreshModeConverges()
    {
        var world = Scene();
        for (int i = 0; i < 120; i++) world.Step(Dt);
        string text = Snapshot.World2ToJson(world, SnapshotLoadMode.Fresh);
        var restored = Snapshot.World2FromJson(text, SnapshotLoadMode.Fresh);
        Assert.Empty(restored.ContactCache);
        for (int i = 0; i < 120; i++) restored.Step(Dt);
        foreach (var body in restored.Bodies)
        {
            Assert.True(float.IsFinite(body.Transform.Position.X));
            Assert.True(float.IsFinite(body.Velocity.X));
        }
    }

    [Fact]
    public void E05_RejectsHostileInput()
    {
        Assert.Throws<ArgumentException>(() => Snapshot.World2FromJson("{nope", SnapshotLoadMode.Exact));
        Assert.Throws<ArgumentException>(() => Snapshot.World2FromBinary(new byte[] { 1, 2, 3 }, SnapshotLoadMode.Exact));
        // Wrong version.
        var world = Scene();
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        bytes[4] = 99;
        Assert.Throws<ArgumentException>(() => Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact));
    }

    [Fact]
    public void E06_JsonBinaryEquivalent()
    {
        var world = Scene();
        for (int i = 0; i < 60; i++) world.Step(Dt);
        var a = Snapshot.World2FromJson(
            Snapshot.World2ToJson(world, SnapshotLoadMode.Exact), SnapshotLoadMode.Exact);
        var b = Snapshot.World2FromBinary(
            Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact), SnapshotLoadMode.Exact);
        Assert.Equal(HashWorld(a), HashWorld(b));
    }

    [Fact]
    public void E07_FileRoundtrip()
    {
        var world = Scene();
        for (int i = 0; i < 60; i++) world.Step(Dt);
        string pj = Path.Combine(Path.GetTempPath(), "phieng_e07.json");
        string pb = Path.Combine(Path.GetTempPath(), "phieng_e07.bin");
        try
        {
            Snapshot.World2SaveFile(world, pj, SnapshotFormat.Json, SnapshotLoadMode.Exact);
            Snapshot.World2SaveFile(world, pb, SnapshotFormat.Binary, SnapshotLoadMode.Exact);
            var rj = Snapshot.World2LoadFile(pj, SnapshotLoadMode.Exact);
            var rb = Snapshot.World2LoadFile(pb, SnapshotLoadMode.Exact);
            Assert.Equal(HashWorld(world), HashWorld(rj));
            Assert.Equal(HashWorld(world), HashWorld(rb));
        }
        finally
        {
            try { File.Delete(pj); } catch { }
            try { File.Delete(pb); } catch { }
        }
    }

    [Fact]
    public void F34_NativeSaveLoadRoundtrip()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        world.AddDistanceJoint(0, ball, new Vector2(0.0f, -1.0f), Vector2.Zero, 6.0f);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        string pb = Path.Combine(Path.GetTempPath(), "phieng_f34.bin");
        string pj = Path.Combine(Path.GetTempPath(), "phieng_f34.json");
        try
        {
            world.Save(pb, 1, 0);
            world.Save(pj, 0, 0);
            using var rb = PhiNativeWorld.Load(pb, 0);
            using var rj = PhiNativeWorld.Load(pj, 0);
            Assert.Equal(world.BodyCount, rb.BodyCount);
            Assert.Equal(world.BodyCount, rj.BodyCount);
            Assert.Equal(world.JointCount, rb.JointCount);
            for (int i = 0; i < 60; i++)
            {
                world.Step(Dt);
                rb.Step(Dt);
            }
            var (pa, _) = world.GetBodyTransform(ball);
            var (pb2, _) = rb.GetBodyTransform(ball);
            Assert.True(MathF.Abs(pa.X - pb2.X) < 1e-4f && MathF.Abs(pa.Y - pb2.Y) < 1e-4f);
        }
        finally
        {
            try { File.Delete(pb); } catch { }
            try { File.Delete(pj); } catch { }
        }
    }

    [Fact]
    public void F36_NativeSavedLoadsManagedBitIdentical()
    {
        // Byte-identical layout proof: a file written by the Rust core
        // must load into the managed engine with identical state bits.
        if (!OperatingSystem.IsWindows()) return;
        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        string pb = Path.Combine(Path.GetTempPath(), "phieng_f36.bin");
        try
        {
            world.Save(pb, 1, 0);
            var managed = Snapshot.World2LoadFile(pb, SnapshotLoadMode.Exact);
            Assert.Equal(world.BodyCount, managed.Bodies.Count);
            var (npos, nrot) = world.GetBodyTransform(ball);
            var mbody = managed.Bodies[ball];
            Assert.Equal(
                BitConverter.SingleToInt32Bits(npos.X),
                BitConverter.SingleToInt32Bits(mbody.Transform.Position.X));
            Assert.Equal(
                BitConverter.SingleToInt32Bits(npos.Y),
                BitConverter.SingleToInt32Bits(mbody.Transform.Position.Y));
        }
        finally
        {
            try { File.Delete(pb); } catch { }
        }
    }

    [Fact]
    public void FluidSnapshotRoundtrip()
    {
        // Fluid state is data-only on the managed side (stepping lives in
        // Rust): JSON + binary must round-trip config and particles exactly.
        var world = Scene();
        world.Fluid.AddParticles(new[]
        {
            new Vector2(0.0f, 0.0f),
            new Vector2(0.25f, 0.0f),
            new Vector2(0.0f, 0.25f),
        });
        world.Fluid.Particles[0] = new FluidParticle2
        {
            Position = world.Fluid.Particles[0].Position,
            Velocity = new Vector2(1.0f, -2.0f),
        };
        // Schema v5 fluid expansion fields must round-trip.
        world.Fluid.Config.Gamma = 7.0f;
        world.Fluid.Config.ArtificialViscosity = 1.2f;
        world.Fluid.Config.SurfaceTension = 0.7f;
        foreach (var mode in new[] { SnapshotLoadMode.Exact, SnapshotLoadMode.Fresh })
        {
            var rj = Snapshot.World2FromJson(
                Snapshot.World2ToJson(world, mode), mode);
            AssertFluidEqual(world, rj);
            var rb = Snapshot.World2FromBinary(
                Snapshot.World2ToBinary(world, mode), mode);
            AssertFluidEqual(world, rb);
        }
    }

    private static void AssertFluidEqual(World2 a, World2 b)
    {
        Assert.Equal(a.Fluid.Config.Enabled, b.Fluid.Config.Enabled);
        Assert.Equal(a.Fluid.Config.ParticleMass, b.Fluid.Config.ParticleMass);
        Assert.Equal(a.Fluid.Config.RestDensity, b.Fluid.Config.RestDensity);
        Assert.Equal(a.Fluid.Config.SmoothingLength, b.Fluid.Config.SmoothingLength);
        Assert.Equal(a.Fluid.Config.Stiffness, b.Fluid.Config.Stiffness);
        Assert.Equal(a.Fluid.Config.Gamma, b.Fluid.Config.Gamma);
        Assert.Equal(a.Fluid.Config.ArtificialViscosity, b.Fluid.Config.ArtificialViscosity);
        Assert.Equal(a.Fluid.Config.SurfaceTension, b.Fluid.Config.SurfaceTension);
        Assert.Equal(a.Fluid.Config.Viscosity, b.Fluid.Config.Viscosity);
        Assert.Equal(a.Fluid.Config.Drag, b.Fluid.Config.Drag);
        Assert.Equal(a.Fluid.Config.MaxVelocity, b.Fluid.Config.MaxVelocity);
        Assert.Equal(a.Fluid.Config.ParticleRadius, b.Fluid.Config.ParticleRadius);
        Assert.Equal(a.Fluid.Particles.Count, b.Fluid.Particles.Count);
        for (int i = 0; i < a.Fluid.Particles.Count; i++)
        {
            Assert.Equal(a.Fluid.Particles[i].Position, b.Fluid.Particles[i].Position);
            Assert.Equal(a.Fluid.Particles[i].Velocity, b.Fluid.Particles[i].Velocity);
        }
    }

    [Fact]
    public void F35_NativeSaveLoadRejects()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var world = new PhiNativeWorld();
        Assert.Throws<IOException>(() => PhiNativeWorld.Load(
            Path.Combine(Path.GetTempPath(), "phieng_nope_missing.bin"), 0));
        string bad = Path.Combine(Path.GetTempPath(), "phieng_f35.bin");
        try
        {
            File.WriteAllBytes(bad, new byte[] { 1, 2, 3, 4 });
            Assert.Throws<IOException>(() => PhiNativeWorld.Load(bad, 0));
            Assert.Throws<IOException>(() => world.Save(bad, 99, 0));
        }
        finally
        {
            try { File.Delete(bad); } catch { }
        }
    }
}
