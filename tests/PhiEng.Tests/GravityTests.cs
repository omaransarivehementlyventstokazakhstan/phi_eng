using System.Numerics;
using Xunit;
using Xunit.Abstractions;
using PhiEng.Native;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Gravity feature tests: FFI config/scale round-trips, managed-vs-native
/// equivalence within tolerances (cross-language bit-equality is NOT
/// claimed), snapshot v4 cross-loading, and managed snapshot round-trips.
/// Test names deliberately avoid the Letter+2digits parity-ID pattern
/// (the mirrored G block lands with the 0.4.0 release, like brutal.rs
/// stays exempt).
/// </summary>
public class GravityTests
{
    private readonly ITestOutputHelper _output;

    public GravityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static bool NativeAvailable() => OperatingSystem.IsWindows();

    private const float Dt = 1.0f / 60.0f;

    // ------------------------------------------------------------------
    // FFI surface.
    // ------------------------------------------------------------------

    [Fact]
    public void Gravity_Config_Roundtrip()
    {
        if (!NativeAvailable()) return;
        using var world = new PhiNativeWorld();
        foreach (var mode in new[] { GravityMode.Uniform, GravityMode.Central, GravityMode.NBody, GravityMode.BarnesHut })
        {
            foreach (var integ in new[] { Integrator.SemiImplicitEuler, Integrator.VelocityVerlet })
            {
                var cfg = new GravityConfig2
                {
                    Mode = mode,
                    Integrator = integ,
                    BigG = 2.0f,
                    Softening = 0.1f,
                    Attractor = new Vector2(1.0f, 2.0f),
                    AttractorMu = 3.0f,
                    Theta = 0.7f,
                };
                world.SetGravityConfig(cfg);
                var back = world.GetGravityConfig();
                Assert.Equal(cfg.Mode, back.Mode);
                Assert.Equal(cfg.Integrator, back.Integrator);
                Assert.Equal(cfg.BigG, back.BigG);
                Assert.Equal(cfg.Softening, back.Softening);
                Assert.Equal(cfg.Attractor, back.Attractor);
                Assert.Equal(cfg.AttractorMu, back.AttractorMu);
                Assert.Equal(cfg.Theta, back.Theta);
                Assert.True(world.TryGetGravityConfig(out var tried));
                Assert.Equal(cfg.Mode, tried.Mode);
            }
        }
        // Unknown tags, NaN G, negative softening/theta: rejected.
        Assert.Throws<ArgumentException>(() => world.SetGravityConfig(new GravityConfig2
        {
            Mode = (GravityMode)99, BigG = 1.0f, Softening = 0.0f,
            Attractor = Vector2.Zero, AttractorMu = 0.0f, Theta = 0.5f,
        }));
        Assert.Throws<ArgumentException>(() => world.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.NBody, Integrator = (Integrator)7, BigG = 1.0f,
            Softening = 0.0f, Attractor = Vector2.Zero, AttractorMu = 0.0f, Theta = 0.5f,
        }));
        Assert.Throws<ArgumentException>(() => world.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.NBody, BigG = float.NaN, Softening = 0.0f,
            Attractor = Vector2.Zero, AttractorMu = 0.0f, Theta = 0.5f,
        }));
        Assert.Throws<ArgumentException>(() => world.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.NBody, BigG = 1.0f, Softening = -1.0f,
            Attractor = Vector2.Zero, AttractorMu = 0.0f, Theta = 0.5f,
        }));
        Assert.Throws<ArgumentException>(() => world.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.BarnesHut, BigG = 1.0f, Softening = 0.0f,
            Attractor = Vector2.Zero, AttractorMu = 0.0f, Theta = -0.5f,
        }));
    }

    [Fact]
    public void SleepThresholds_Roundtrip()
    {
        if (!NativeAvailable()) return;
        using var world = new PhiNativeWorld();
        var (lin, ang) = world.GetSleepThresholds();
        Assert.Equal(0.05f, lin);
        Assert.Equal(0.05f, ang);
        world.SetSleepThresholds(0.0f, 0.0f);
        var (lin2, ang2) = world.GetSleepThresholds();
        Assert.Equal(0.0f, lin2);
        Assert.Equal(0.0f, ang2);
        Assert.Throws<ArgumentException>(() => world.SetSleepThresholds(-1.0f, 0.0f));
        Assert.Throws<ArgumentException>(() => world.SetSleepThresholds(0.0f, float.NaN));
        using var world3 = new PhiNativeWorld3();
        world3.SetSleepThresholds(0.0f, 0.0f);
        var (lin3, _) = world3.GetSleepThresholds();
        Assert.Equal(0.0f, lin3);
        Assert.Throws<ArgumentException>(() => world3.SetSleepThresholds(-1.0f, 0.0f));
    }

    [Fact]
    public void Gravity_Scale_Roundtrip()
    {
        if (!NativeAvailable()) return;
        using var world = new PhiNativeWorld();
        int ball = world.AddDynamicBall(new Vector2(0.0f, 2.0f), 0.5f, 1.0f);
        Assert.Equal(1.0f, world.GetBodyGravityScale(ball));
        foreach (float s in new[] { 0.0f, 0.5f, 2.0f, -1.0f })
        {
            world.SetBodyGravityScale(ball, s);
            Assert.Equal(s, world.GetBodyGravityScale(ball));
            Assert.True(world.TryGetBodyGravityScale(ball, out float back));
            Assert.Equal(s, back);
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => world.SetBodyGravityScale(999, 1.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.SetBodyGravityScale(ball, float.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.GetBodyGravityScale(999));
        Assert.False(world.TryGetBodyGravityScale(999, out _));
    }

    [Fact]
    public void Gravity3_Config_Roundtrip()
    {
        if (!NativeAvailable()) return;
        using var world = new PhiNativeWorld3();
        var cfg = new GravityConfig3
        {
            Mode = GravityMode.BarnesHut,
            Integrator = Integrator.VelocityVerlet,
            BigG = 1.5f,
            Softening = 0.2f,
            Attractor = new Vector3(1.0f, 2.0f, 3.0f),
            AttractorMu = 4.0f,
            Theta = 0.7f,
        };
        world.SetGravityConfig(cfg);
        var back = world.GetGravityConfig();
        Assert.Equal(cfg.Mode, back.Mode);
        Assert.Equal(cfg.Integrator, back.Integrator);
        Assert.Equal(cfg.BigG, back.BigG);
        Assert.Equal(cfg.Softening, back.Softening);
        Assert.Equal(cfg.Attractor, back.Attractor);
        Assert.Equal(cfg.AttractorMu, back.AttractorMu);
        Assert.Equal(cfg.Theta, back.Theta);
        int ball = world.AddDynamicBall(new Vector3(0.0f, 2.0f, 0.0f), 0.5f, 1.0f);
        world.SetBodyGravityScale(ball, 0.0f);
        Assert.Equal(0.0f, world.GetBodyGravityScale(ball));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.SetBodyGravityScale(ball, float.NaN));
    }

    // ------------------------------------------------------------------
    // Managed-vs-native equivalence (tolerances, never bit-equality).
    // ------------------------------------------------------------------

    private static World2 CentralScene(out int probe)
    {
        // NOTE: both scenes start the probe at REST. The FFI surface has
        // no velocity setter, so transverse-velocity orbits cannot be
        // staged natively — orbital dynamics are covered by the core and
        // managed test suites; here both sides fall radially identically.
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.Central,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.0f,
            Attractor = Vector2.Zero,
            AttractorMu = 100.0f,
            Theta = 0.5f,
        };
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        probe = w.AddDynamicBody(new Vector2(10.0f, 0.0f), new CircleShape(0.5f), 1.0f);
        return w;
    }

    private static PhiNativeWorld CentralSceneNative(out int probe)
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, 0.0f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.Central,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.0f,
            Attractor = Vector2.Zero,
            AttractorMu = 100.0f,
            Theta = 0.5f,
        });
        probe = w.AddDynamicBall(new Vector2(10.0f, 0.0f), 0.5f, 1.0f);
        return w;
    }

    [Fact]
    public void Gravity_ManagedMatchesNative_RadialFall()
    {
        if (!NativeAvailable()) return;
        var managed = CentralScene(out int mp);
        using var native = CentralSceneNative(out int np);
        for (int i = 0; i < 120; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }
        Vector2 mpos = managed.Bodies[mp].Transform.Position;
        var (npos, _) = native.GetBodyTransform(np);
        float dx = MathF.Abs(mpos.X - npos.X);
        float dy = MathF.Abs(mpos.Y - npos.Y);
        _output.WriteLine($"[gravity] managed=({mpos.X:F4},{mpos.Y:F4}) native=({npos.X:F4},{npos.Y:F4})");
        Assert.True(mpos.X < 10.0f, "managed probe did not fall");
        Assert.True(dx < 1e-3f && dy < 1e-3f, $"managed/native radial fall diverged: dx={dx}, dy={dy}");
    }

    private static World2 NBodyScene(out int[] ids)
    {
        // NOTE: all bodies start at REST (the FFI surface has no velocity
        // setter): both sides fall identically from identical conditions.
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.1f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        var specs = new[]
        {
            (new Vector2(-20.0f, 0.0f), 0.5f),
            (new Vector2(15.0f, 5.0f), 0.7f),
            (new Vector2(0.0f, -15.0f), 1.0f),
        };
        ids = new int[specs.Length];
        for (int i = 0; i < specs.Length; i++)
        {
            // Managed bodies keep default damping 0.01, matching native.
            ids[i] = w.AddDynamicBody(specs[i].Item1, new CircleShape(specs[i].Item2), 1.0f);
        }
        return w;
    }

    private static PhiNativeWorld NBodySceneNative(out int[] ids)
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, 0.0f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.1f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        });
        var specs = new[]
        {
            (new Vector2(-20.0f, 0.0f), new Vector2(0.5f, 0.0f), 0.5f),
            (new Vector2(15.0f, 5.0f), new Vector2(0.0f, 0.5f), 0.7f),
            (new Vector2(0.0f, -15.0f), new Vector2(-0.3f, 0.2f), 1.0f),
        };
        ids = new int[specs.Length];
        for (int i = 0; i < specs.Length; i++)
        {
            // Native has no damping setter; default 0.01 applies on both
            // sides only if matched — managed defaults are 0.01 too, and
            // sleep is off, so both integrate identically damped.
            ids[i] = w.AddDynamicBall(specs[i].Item1, specs[i].Item3, 1.0f);
        }
        return w;
    }

    [Fact]
    public void Gravity_ManagedMatchesNative_NBody()
    {
        if (!NativeAvailable()) return;
        // Both sides start at rest with default damping 0.01 (the FFI
        // surface has no damping/velocity setters): identical conditions.
        var managed = NBodyScene(out int[] mids);
        using var native = NBodySceneNative(out int[] nids);
        for (int i = 0; i < 60; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }
        var query = new uint[] { (uint)nids[0], (uint)nids[1], (uint)nids[2] };
        var pos = new Vector2[3];
        var rot = new float[3];
        int valid = native.GetTransformsBulk(query, pos, rot, null);
        Assert.Equal(3, valid);
        for (int i = 0; i < 3; i++)
        {
            var mp = managed.Bodies[mids[i]].Transform.Position;
            float dx = MathF.Abs(mp.X - pos[i].X);
            float dy = MathF.Abs(mp.Y - pos[i].Y);
            Assert.True(dx < 1e-2f && dy < 1e-2f, $"body {i} diverged: dx={dx}, dy={dy}");
        }
    }

    [Fact]
    public void Gravity_ManagedMatchesNative_BH()
    {
        if (!NativeAvailable()) return;
        // Well-separated grid: cell open/close decisions sit far from
        // theta, so tree structures agree across languages.
        var managed = new World2();
        managed.Gravity = Vector2.Zero;
        managed.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.BarnesHut,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.5f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        managed.LinearSleepThreshold = 0.0f;
        managed.AngularSleepThreshold = 0.0f;
        var mids = new List<int>();
        for (int i = 0; i < 12; i++)
        {
            float x = (i % 4) * 8.0f - 12.0f;
            float y = (i / 4) * 8.0f - 8.0f;
            int id = managed.AddDynamicBody(new Vector2(x, y), new CircleShape(0.5f), 1.0f);
            mids.Add(id);
        }
        using var native = new PhiNativeWorld();
        native.SetGravity(new Vector2(0.0f, 0.0f));
        native.SetSleepThresholds(0.0f, 0.0f);
        native.SetGravityConfig(managed.GravityConfig);
        var descs = new BodyDesc2D[12];
        for (int i = 0; i < 12; i++)
        {
            float x = (i % 4) * 8.0f - 12.0f;
            float y = (i / 4) * 8.0f - 8.0f;
            descs[i] = BodyDesc2D.Circle(x, y, 0.5f, 1.0f);
        }
        int[] nids = native.AddBodies(descs);
        for (int i = 0; i < 60; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }
        var query = new uint[12];
        for (int i = 0; i < 12; i++) query[i] = (uint)nids[i];
        var pos = new Vector2[12];
        var rot = new float[12];
        Assert.Equal(12, native.GetTransformsBulk(query, pos, rot, null));
        for (int i = 0; i < 12; i++)
        {
            var mp = managed.Bodies[mids[i]].Transform.Position;
            float dx = MathF.Abs(mp.X - pos[i].X);
            float dy = MathF.Abs(mp.Y - pos[i].Y);
            Assert.True(dx < 1e-1f && dy < 1e-1f, $"BH body {i} diverged: dx={dx}, dy={dy}");
        }
    }

    // ------------------------------------------------------------------
    // Snapshot v4 (managed + cross-language).
    // ------------------------------------------------------------------

    [Fact]
    public void Gravity_ManagedSnapshotRoundtrip()
    {
        var w = NBodyScene(out _);
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.VelocityVerlet,
            BigG = 2.0f,
            Softening = 0.1f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        w.Bodies[0].GravityScale = 0.5f;
        for (int i = 0; i < 60; i++) w.Step(Dt);
        string json = Snapshot.World2ToJson(w, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 4", json);
        byte[] bin = Snapshot.World2ToBinary(w, SnapshotLoadMode.Exact);
        var rj = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        var rb = Snapshot.World2FromBinary(bin, SnapshotLoadMode.Exact);
        Assert.Equal(GravityMode.NBody, rj.GravityConfig.Mode);
        Assert.Equal(Integrator.VelocityVerlet, rb.GravityConfig.Integrator);
        Assert.Equal(0.5f, rj.Bodies[0].GravityScale);
        for (int i = 0; i < 60; i++)
        {
            w.Step(Dt);
            rj.Step(Dt);
            rb.Step(Dt);
        }
        for (int i = 0; i < w.Bodies.Count; i++)
        {
            Assert.Equal(w.Bodies[i].Transform.Position, rj.Bodies[i].Transform.Position);
            Assert.Equal(w.Bodies[i].Transform.Position, rb.Bodies[i].Transform.Position);
        }

        // Default worlds stay schema 3. (The C# writer always emits
        // gravity keys; what matters is the version gate plus default
        // values — the Rust writer omits them for byte-identical v3.)
        var plain = new World2();
        plain.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        string pj = Snapshot.World2ToJson(plain, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 3", pj);
        var pjBack = Snapshot.World2FromJson(pj, SnapshotLoadMode.Exact);
        Assert.True(pjBack.GravityConfig.IsDefault());
        Assert.All(pjBack.Bodies, b => Assert.Equal(1.0f, b.GravityScale));
    }

    [Fact]
    public void Gravity_NativeSavedLoadsManaged()
    {
        // Rust-written v4 file must load into the managed engine with
        // identical config, scales, and state bits (F36 pattern for v4).
        if (!NativeAvailable()) return;
        using var world = new PhiNativeWorld();
        world.SetGravity(new Vector2(0.0f, 0.0f));
        world.SetGravityConfig(new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 2.0f,
            Softening = 0.1f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        });
        int a = world.AddDynamicBall(new Vector2(-8.0f, 0.0f), 0.5f, 1.0f);
        int b = world.AddDynamicBall(new Vector2(8.0f, 0.0f), 0.5f, 1.0f);
        world.SetBodyGravityScale(b, 0.5f);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        string pb = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "phieng_grav_v4.bin");
        string pj = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "phieng_grav_v4.json");
        try
        {
            world.Save(pb, 1, 0);
            world.Save(pj, 0, 0);
            var mb = Snapshot.World2LoadFile(pb, SnapshotLoadMode.Exact);
            var mj = Snapshot.World2LoadFile(pj, SnapshotLoadMode.Exact);
            Assert.Equal(GravityMode.NBody, mb.GravityConfig.Mode);
            Assert.Equal(2.0f, mb.GravityConfig.BigG);
            Assert.Equal(0.5f, mb.Bodies[b].GravityScale);
            Assert.Equal(GravityMode.NBody, mj.GravityConfig.Mode);
            var (npos, _) = world.GetBodyTransform(a);
            var mbody = mb.Bodies[a];
            Assert.Equal(
                BitConverter.SingleToInt32Bits(npos.X),
                BitConverter.SingleToInt32Bits(mbody.Transform.Position.X));
            Assert.Equal(
                BitConverter.SingleToInt32Bits(npos.Y),
                BitConverter.SingleToInt32Bits(mbody.Transform.Position.Y));
        }
        finally
        {
            try { System.IO.File.Delete(pb); } catch { }
            try { System.IO.File.Delete(pj); } catch { }
        }
    }

    // ------------------------------------------------------------------
    // G parity block (mirrored by g01-g10 in crates/engine/tests/gravity.rs;
    // enforced by tools/check_test_parity.py). Cross-language bit-equality
    // is NOT claimed — each side asserts run-twice determinism plus
    // tolerance-bounded physics behavior on mirrored scenes.
    // ------------------------------------------------------------------

    private static World2 OrbitScene(out int probe)
    {
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.Central,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.0f,
            Attractor = Vector2.Zero,
            AttractorMu = 100.0f,
            Theta = 0.5f,
        };
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        float v = MathF.Sqrt(100.0f / 10.0f);
        probe = w.AddDynamicBody(new Vector2(10.0f, 0.0f), new CircleShape(0.5f), 1.0f);
        w.Bodies[probe].Velocity = new Vector2(0.0f, v);
        w.Bodies[probe].LinearDamping = 0.0f;
        w.Bodies[probe].AngularDamping = 0.0f;
        return w;
    }

    private static void AssertWorldsEqual(World2 a, World2 b)
    {
        Assert.Equal(a.Bodies.Count, b.Bodies.Count);
        for (int i = 0; i < a.Bodies.Count; i++)
        {
            Assert.Equal(a.Bodies[i].Transform.Position, b.Bodies[i].Transform.Position);
            Assert.Equal(a.Bodies[i].Velocity, b.Bodies[i].Velocity);
            Assert.Equal(a.Bodies[i].AngularVelocity, b.Bodies[i].AngularVelocity);
        }
    }

    private static void AssertWorldsEqual(World3 a, World3 b)
    {
        Assert.Equal(a.Bodies.Count, b.Bodies.Count);
        for (int i = 0; i < a.Bodies.Count; i++)
        {
            Assert.Equal(a.Bodies[i].Transform.Position, b.Bodies[i].Transform.Position);
            Assert.Equal(a.Bodies[i].Velocity, b.Bodies[i].Velocity);
            Assert.Equal(a.Bodies[i].AngularVelocity, b.Bodies[i].AngularVelocity);
        }
    }

    private static Vector2 TotalMomentum(World2 w)
    {
        var p = Vector2.Zero;
        foreach (var b in w.Bodies) p += b.Velocity * b.Mass;
        return p;
    }

    private static float SpecificEnergy(Vector2 pos, Vector2 vel, float mu)
    {
        return 0.5f * vel.LengthSquared() - mu / MathF.Max(pos.Length(), 1e-6f);
    }

    /// <summary>G01: central circular orbit holds its radius, deterministically.</summary>
    [Fact]
    public void G01_CentralOrbitStable()
    {
        var a = OrbitScene(out _);
        var b = OrbitScene(out _);
        for (int i = 0; i < 600; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        AssertWorldsEqual(a, b);
        float r = a.Bodies[0].Transform.Position.Length();
        Assert.True(MathF.Abs(r - 10.0f) / 10.0f < 0.10f, $"orbit decayed: r={r}");
    }

    /// <summary>G02: N-body momentum conservation + determinism.</summary>
    [Fact]
    public void G02_NBodyDeterministic()
    {
        World2 Scene()
        {
            var w = new World2();
            w.Gravity = Vector2.Zero;
            w.GravityConfig = new GravityConfig2
            {
                Mode = GravityMode.NBody,
                Integrator = Integrator.SemiImplicitEuler,
                BigG = 1.0f,
                Softening = 0.1f,
                Attractor = Vector2.Zero,
                AttractorMu = 0.0f,
                Theta = 0.5f,
            };
            w.LinearSleepThreshold = 0.0f;
            w.AngularSleepThreshold = 0.0f;
            foreach (var (p, r) in new[]
            {
                (new Vector2(-20.0f, 0.0f), 0.5f),
                (new Vector2(15.0f, 5.0f), 0.7f),
                (new Vector2(0.0f, -15.0f), 1.0f),
            })
            {
                int id = w.AddDynamicBody(p, new CircleShape(r), 1.0f);
                w.Bodies[id].LinearDamping = 0.0f;
                w.Bodies[id].AngularDamping = 0.0f;
            }
            return w;
        }
        var a = Scene();
        Vector2 p0 = TotalMomentum(a);
        var b = Scene();
        for (int i = 0; i < 300; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        AssertWorldsEqual(a, b);
        Vector2 p1 = TotalMomentum(a);
        Assert.True((p1 - p0).Length() < 1e-2f, "momentum drifted");
    }

    /// <summary>G03: Verlet energy drift is clearly smaller than Euler's.</summary>
    [Fact]
    public void G03_VerletEnergyOrder()
    {
        float MaxDrift(bool verlet)
        {
            var w = OrbitScene(out int id);
            if (verlet)
            {
                var cfg = w.GravityConfig;
                cfg.Integrator = Integrator.VelocityVerlet;
                w.GravityConfig = cfg;
            }
            var bb = w.Bodies[id];
            float e0 = SpecificEnergy(bb.Transform.Position, bb.Velocity, 100.0f);
            float worst = 0.0f;
            for (int step = 0; step < 600; step++)
            {
                w.Step(Dt);
                if (step % 60 == 0)
                {
                    var cur = w.Bodies[id];
                    float e = SpecificEnergy(cur.Transform.Position, cur.Velocity, 100.0f);
                    worst = MathF.Max(worst, MathF.Abs(e - e0) / MathF.Abs(e0));
                }
            }
            return worst;
        }
        Assert.True(MaxDrift(true) < MaxDrift(false) * 0.5f);
    }

    /// <summary>G04: coincident bodies stay finite (softening skip guard).</summary>
    [Fact]
    public void G04_SofteningCoincidentSafe()
    {
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.0f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        for (int i = 0; i < 2; i++)
        {
            int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.5f), 1.0f);
            w.Bodies[id].LinearDamping = 0.0f;
            w.Bodies[id].AngularDamping = 0.0f;
        }
        for (int i = 0; i < 60; i++) w.Step(Dt);
        foreach (var body in w.Bodies)
        {
            Assert.True(float.IsFinite(body.Transform.Position.X));
            Assert.True(float.IsFinite(body.Velocity.X));
        }
    }

    /// <summary>G05: immune probe never moves; tiny-density probe follows the field.</summary>
    [Fact]
    public void G05_ProbeExempt()
    {
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.GravityConfig = new GravityConfig2
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.SemiImplicitEuler,
            BigG = 1.0f,
            Softening = 0.1f,
            Attractor = Vector2.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        w.AddStaticBody(Vector2.Zero, new BoxShape2(new Vector2(5.0f, 5.0f)));
        int immune = w.AddDynamicBody(new Vector2(60.0f, 0.0f), new CircleShape(0.5f), 1.0f);
        w.Bodies[immune].GravityScale = 0.0f;
        int probe = w.AddDynamicBody(new Vector2(-60.0f, 0.0f), new CircleShape(0.5f), 1e-7f);
        for (int i = 0; i < 120; i++) w.Step(Dt);
        Assert.Equal(new Vector2(60.0f, 0.0f), w.Bodies[immune].Transform.Position);
        Assert.True(w.Bodies[probe].Transform.Position.X > -60.0f);
    }

    /// <summary>G06: N-body + Verlet + scales snapshot round-trip with continuation.</summary>
    [Fact]
    public void G06_SnapshotRoundtrip()
    {
        World2 Scene()
        {
            var w = new World2();
            w.Gravity = Vector2.Zero;
            w.GravityConfig = new GravityConfig2
            {
                Mode = GravityMode.NBody,
                Integrator = Integrator.VelocityVerlet,
                BigG = 2.0f,
                Softening = 0.1f,
                Attractor = Vector2.Zero,
                AttractorMu = 0.0f,
                Theta = 0.5f,
            };
            w.LinearSleepThreshold = 0.0f;
            w.AngularSleepThreshold = 0.0f;
            foreach (var (p, s) in new[]
            {
                (new Vector2(-8.0f, 0.0f), 1.0f),
                (new Vector2(8.0f, 0.0f), 0.5f),
            })
            {
                int id = w.AddDynamicBody(p, new CircleShape(0.5f), 1.0f);
                w.Bodies[id].GravityScale = s;
                w.Bodies[id].LinearDamping = 0.0f;
                w.Bodies[id].AngularDamping = 0.0f;
            }
            return w;
        }
        var direct = Scene();
        for (int i = 0; i < 30; i++) direct.Step(Dt);
        string json = Snapshot.World2ToJson(direct, SnapshotLoadMode.Exact);
        byte[] bin = Snapshot.World2ToBinary(direct, SnapshotLoadMode.Exact);
        var fromJson = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        var fromBin = Snapshot.World2FromBinary(bin, SnapshotLoadMode.Exact);
        Assert.Equal(GravityMode.NBody, fromJson.GravityConfig.Mode);
        Assert.Equal(0.5f, fromBin.Bodies[1].GravityScale);
        for (int i = 0; i < 30; i++)
        {
            direct.Step(Dt);
            fromJson.Step(Dt);
            fromBin.Step(Dt);
        }
        AssertWorldsEqual(direct, fromJson);
        AssertWorldsEqual(direct, fromBin);
    }

    /// <summary>G07: Barnes-Hut determinism on a 20-body scene.</summary>
    [Fact]
    public void G07_BhDeterministic()
    {
        World2 Scene()
        {
            var w = new World2();
            w.Gravity = Vector2.Zero;
            w.GravityConfig = new GravityConfig2
            {
                Mode = GravityMode.BarnesHut,
                Integrator = Integrator.SemiImplicitEuler,
                BigG = 1.0f,
                Softening = 0.1f,
                Attractor = Vector2.Zero,
                AttractorMu = 0.0f,
                Theta = 0.5f,
            };
            w.LinearSleepThreshold = 0.0f;
            w.AngularSleepThreshold = 0.0f;
            for (int i = 0; i < 20; i++)
            {
                int id = w.AddDynamicBody(
                    new Vector2((i * 1.7f) % 20.0f - 10.0f, (i * 2.3f) % 20.0f - 10.0f),
                    new CircleShape(0.5f), 1.0f);
                w.Bodies[id].LinearDamping = 0.0f;
                w.Bodies[id].AngularDamping = 0.0f;
            }
            return w;
        }
        var a = Scene();
        var b = Scene();
        for (int i = 0; i < 60; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        AssertWorldsEqual(a, b);
    }

    /// <summary>G08: 3D central orbit holds its radius, deterministically.</summary>
    [Fact]
    public void G08_CentralOrbitStable3D()
    {
        World3 Scene()
        {
            var w = new World3();
            w.Gravity = Vector3.Zero;
            w.GravityConfig = new GravityConfig3
            {
                Mode = GravityMode.Central,
                Integrator = Integrator.SemiImplicitEuler,
                BigG = 1.0f,
                Softening = 0.0f,
                Attractor = Vector3.Zero,
                AttractorMu = 100.0f,
                Theta = 0.5f,
            };
            w.LinearSleepThreshold = 0.0f;
            w.AngularSleepThreshold = 0.0f;
            float v = MathF.Sqrt(100.0f / 10.0f);
            int id = w.AddDynamicBody(new Vector3(10.0f, 0.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
            w.Bodies[id].Velocity = new Vector3(0.0f, v, 0.0f);
            w.Bodies[id].LinearDamping = 0.0f;
            w.Bodies[id].AngularDamping = 0.0f;
            return w;
        }
        var a = Scene();
        var b = Scene();
        for (int i = 0; i < 600; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }
        AssertWorldsEqual(a, b);
        var p = a.Bodies[0].Transform.Position;
        float r = new Vector2(p.X, p.Y).Length();
        Assert.True(MathF.Abs(r - 10.0f) / 10.0f < 0.10f, $"orbit decayed: r={r}");
    }

    /// <summary>G09: 3D N-body momentum conservation + determinism.</summary>
    [Fact]
    public void G09_NBodyDeterministic3D()
    {
        World3 Scene()
        {
            var w = new World3();
            w.Gravity = Vector3.Zero;
            w.GravityConfig = new GravityConfig3
            {
                Mode = GravityMode.NBody,
                Integrator = Integrator.SemiImplicitEuler,
                BigG = 1.0f,
                Softening = 0.1f,
                Attractor = Vector3.Zero,
                AttractorMu = 0.0f,
                Theta = 0.5f,
            };
            w.LinearSleepThreshold = 0.0f;
            w.AngularSleepThreshold = 0.0f;
            foreach (var (p, r) in new[]
            {
                (new Vector3(-20.0f, 0.0f, 1.0f), 0.5f),
                (new Vector3(15.0f, 5.0f, -1.0f), 0.8f),
            })
            {
                int id = w.AddDynamicBody(p, new SphereShape3(r), 1.0f);
                w.Bodies[id].LinearDamping = 0.0f;
                w.Bodies[id].AngularDamping = 0.0f;
            }
            return w;
        }
        var a = Scene();
        Vector3 p0 = Vector3.Zero;
        foreach (var b in a.Bodies) p0 += b.Velocity * b.Mass;
        var b2 = Scene();
        for (int i = 0; i < 300; i++)
        {
            a.Step(Dt);
            b2.Step(Dt);
        }
        AssertWorldsEqual(a, b2);
        Vector3 p1 = Vector3.Zero;
        foreach (var b in a.Bodies) p1 += b.Velocity * b.Mass;
        Assert.True((p1 - p0).Length() < 1e-2f, "momentum drifted");
    }

    /// <summary>G10: 3D N-body + Verlet + scales snapshot round-trip.</summary>
    [Fact]
    public void G10_SnapshotRoundtrip3D()
    {
        var direct = new World3();
        direct.Gravity = Vector3.Zero;
        direct.GravityConfig = new GravityConfig3
        {
            Mode = GravityMode.NBody,
            Integrator = Integrator.VelocityVerlet,
            BigG = 1.5f,
            Softening = 0.2f,
            Attractor = Vector3.Zero,
            AttractorMu = 0.0f,
            Theta = 0.5f,
        };
        direct.LinearSleepThreshold = 0.0f;
        direct.AngularSleepThreshold = 0.0f;
        foreach (var (p, s) in new[]
        {
            (new Vector3(-6.0f, 0.0f, 0.0f), 1.0f),
            (new Vector3(6.0f, 0.0f, 0.0f), 0.25f),
        })
        {
            int id = direct.AddDynamicBody(p, new SphereShape3(0.5f), 1.0f);
            direct.Bodies[id].GravityScale = s;
            direct.Bodies[id].LinearDamping = 0.0f;
            direct.Bodies[id].AngularDamping = 0.0f;
        }
        for (int i = 0; i < 30; i++) direct.Step(Dt);
        string json = Snapshot.World3ToJson(direct, SnapshotLoadMode.Exact);
        byte[] bin = Snapshot.World3ToBinary(direct, SnapshotLoadMode.Exact);
        var fromJson = Snapshot.World3FromJson(json, SnapshotLoadMode.Exact);
        var fromBin = Snapshot.World3FromBinary(bin, SnapshotLoadMode.Exact);
        Assert.Equal(GravityMode.NBody, fromJson.GravityConfig.Mode);
        Assert.Equal(0.25f, fromBin.Bodies[1].GravityScale);
        for (int i = 0; i < 30; i++)
        {
            direct.Step(Dt);
            fromJson.Step(Dt);
            fromBin.Step(Dt);
        }
        AssertWorldsEqual(direct, fromJson);
        AssertWorldsEqual(direct, fromBin);
    }
}
