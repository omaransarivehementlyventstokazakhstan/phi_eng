using System.Numerics;
using Xunit;
using PhiEng.Collision;
using PhiEng.Dynamics;
using PhiEng.Math;

namespace PhiEng.Tests;

/// <summary>
/// Robustness probes + pins, mirroring Rust crates/engine/tests/robustness.rs
/// and the N14 EPA probe. Test IDs (S/W/N/D) are shared across languages;
/// parity is enforced by tools/check_test_parity.py.
/// </summary>
public class RobustnessTests
{
    private const float Dt = 1.0f / 60.0f;

    private static void AddFloor(World2 world)
    {
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
    }

    private static void AssertFinite(World2 world, string ctx)
    {
        foreach (var body in world.Bodies)
        {
            foreach (float v in new[]
            {
                body.Transform.Position.X, body.Transform.Position.Y,
                body.Velocity.X, body.Velocity.Y, body.AngularVelocity
            })
                Assert.True(float.IsFinite(v), $"{ctx}: non-finite state {v} in body {body.Id}");
        }
    }

    [Fact]
    public void W07_MaterialChangeMidContact()
    {
        var world = new World2();
        AddFloor(world);
        int b = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        AssertFinite(world, "w07 settled");
        world.Bodies[b].Material.Friction = 0.9f;
        world.Bodies[b].Material.Restitution = 0.8f;
        float peak = 0;
        for (int i = 0; i < 60; i++)
        {
            world.Step(Dt);
            peak = MathF.Max(peak, world.Bodies[b].Velocity.Length());
        }
        AssertFinite(world, "w07 after material swap");
        Assert.True(peak < 0.5f, $"w07: material swap transient peak {peak}");
    }

    [Fact]
    public void N13_DeepPenetration50Pct()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int top = world.AddDynamicBody(new Vector2(0.5f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        float peak = 0;
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            peak = MathF.Max(peak, world.Bodies[top].Velocity.Length());
            AssertFinite(world, $"n13 frame {frame}");
        }
        float x = world.Bodies[top].Transform.Position.X;
        Assert.True(x > 0.5f, $"n13: must separate along +x, x={x}");
        Assert.True(peak < 10.0f, $"n13: ejection peak {peak}");
    }

    [Fact]
    public void N14_AllVisibleGuardStable()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(0.1f, 0, 0), 1.0f, dir);
        var first = GjkEpa3D.GjkEpa(a, b);
        Assert.NotNull(first);
        for (int frame = 0; frame < 60; frame++)
        {
            var hit = GjkEpa3D.GjkEpa(a, b);
            Assert.NotNull(hit);
            Assert.True(Vector3.Dot(hit.Normal, first.Normal) > 0.9f,
                $"n14 frame {frame}: normal flipped: {hit.Normal} vs {first.Normal}");
            Assert.True(MathF.Abs(hit.Depth - first.Depth) <= 0.05f * first.Depth + 1e-4f,
                $"n14 frame {frame}: depth oscillated: {hit.Depth} vs {first.Depth}");
        }
    }

    [Fact]
    public void S16_MassRatio1000Long()
    {
        var world = new World2();
        AddFloor(world);
        world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1000.0f);
        int light = world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        float peak = 0;
        for (int frame = 0; frame < 10_000; frame++)
        {
            world.Step(Dt);
            peak = MathF.Max(peak, world.Bodies[light].Velocity.Length());
            if (frame % 1000 == 0) AssertFinite(world, $"s16 frame {frame}");
        }
        AssertFinite(world, "s16 final");
        float y = world.Bodies[light].Transform.Position.Y;
        Assert.True(y > 0.9f, $"s16: light body tunneled, y={y}");
        Assert.True(peak < 1.0f, $"s16: jitter peak {peak} over 10k steps");
    }

    [Fact]
    public void S17_MassRatio1e6Bounded()
    {
        var world = new World2();
        AddFloor(world);
        world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1e6f);
        int light = world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int frame = 0; frame < 1200; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"s17 frame {frame}");
        }
        var p = world.Bodies[light].Transform.Position;
        Assert.True(MathF.Abs(p.X) < 100.0f && MathF.Abs(p.Y) < 100.0f,
            $"s17: state escaped world bounds: {p}");
    }

    [Fact]
    public void S18_ZeroMassDynamic()
    {
        var world = new World2();
        AddFloor(world);
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f), 0.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f), -1.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f), float.NaN));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(float.NaN, 2.0f), new CircleShape(0.3f), 1.0f));
        // Static bodies ignore mass: zero density accepted, inverses stay 0.
        int s = world.AddStaticBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f));
        world.Bodies[s].SetShape(new CircleShape(0.3f), 0.0f);
        Assert.Equal(0.0f, world.Bodies[s].InvMass);
        int b = world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f), 1.0f);
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"s18 frame {frame}");
        }
        Assert.True(world.Bodies[b].Transform.Position.Y > -1.0f, "s18: valid body must not tunnel");
    }

    [Fact]
    public void S19_InfiniteInertiaVariants()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        int a = world.AddDynamicBody(new Vector2(-2.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Bodies[a].InvInertia = 0.0f;
        world.Bodies[a].ApplyImpulse(new Vector2(0.0f, 1.0f), new Vector2(-1.5f, 0.0f));
        for (int i = 0; i < 60; i++) world.Step(Dt);
        AssertFinite(world, "s19a");
        Assert.Equal(0.0f, world.Bodies[a].AngularVelocity);
        Assert.True(world.Bodies[a].Velocity.Y > 0.0f, $"s19a: linear response must survive: v={world.Bodies[a].Velocity}");

        int b = world.AddDynamicBody(new Vector2(2.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Bodies[b].InvMass = 0.0f;
        world.Bodies[b].ApplyImpulse(new Vector2(5.0f, 0.0f), new Vector2(2.0f, 0.0f));
        for (int i = 0; i < 60; i++) world.Step(Dt);
        AssertFinite(world, "s19b");
        Assert.Equal(new Vector2(2.0f, 0.0f), world.Bodies[b].Transform.Position);
    }

    [Fact]
    public void D23_CollinearPolygon()
    {
        var world = new World2();
        AddFloor(world);
        Assert.Throws<ArgumentException>(() => world.AddDynamicBody(new Vector2(0.0f, 2.0f),
            new PolygonShape(new List<Vector2>
            {
                new Vector2(-0.5f, 0.0f), new Vector2(0.0f, 0.0f), new Vector2(0.5f, 0.0f),
            }), 1.0f));
        Assert.Throws<ArgumentException>(() => world.AddDynamicBody(new Vector2(0.0f, 2.0f),
            new PolygonShape(new List<Vector2>
            {
                new Vector2(-0.5f, 0.0f), new Vector2(0.5f, 0.0f),
            }), 1.0f));
        int b = world.AddDynamicBody(new Vector2(0.0f, 2.0f),
            new PolygonShape(new List<Vector2>
            {
                new Vector2(-0.5f, 0.0f), new Vector2(0.5f, 0.0f), new Vector2(0.0f, 0.01f),
            }), 1.0f);
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"d23 frame {frame}");
        }
    }

    [Fact]
    public void D24_ZeroRadiusCircle()
    {
        var world = new World2();
        AddFloor(world);
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.0f), 1.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(-0.5f), 1.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(float.NaN), 1.0f));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.3f), 1.0f);
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"d24 frame {frame}");
        }
    }

    [Fact]
    public void D25_ZeroSizeBox()
    {
        var world = new World2();
        AddFloor(world);
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new BoxShape2(Vector2.Zero), 1.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new BoxShape2(new Vector2(0.5f, -0.5f)), 1.0f));
        Assert.Throws<ArgumentException>(() =>
            world.AddDynamicBody(new Vector2(0.0f, 2.0f), new BoxShape2(new Vector2(float.NaN, 0.5f)), 1.0f));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"d25 frame {frame}");
        }
    }

    [Fact]
    public void N11_GrazingPairSelfHeals()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        CapsuleShape3 Cap() => new CapsuleShape3(0.3f, 0.5f);
        world.AddStaticBody(Vector3.Zero, Cap());
        int b = world.AddDynamicBody(new Vector3(0.5999f, 0.0f, 0.0f), Cap(), 1.0f);
        var bb = world.Bodies[b];
        bb.Velocity = new Vector3(-0.02f, 0.0f, 0.0f);
        world.Bodies[b] = bb;
        float maxOverlap = 0;
        for (int frame = 0; frame < 100; frame++)
        {
            world.Step(Dt);
            foreach (var body in world.Bodies)
            {
                var p = body.Transform.Position;
                var v = body.Velocity;
                Assert.True(float.IsFinite(p.X + p.Y + p.Z + v.X + v.Y + v.Z),
                    $"n11 frame {frame}: non-finite state");
            }
            maxOverlap = MathF.Max(maxOverlap, 0.6f - world.Bodies[b].Transform.Position.X);
        }
        Assert.True(maxOverlap < 0.01f, $"n11: penetration must self-heal, max overlap={maxOverlap}");
        Assert.True(world.Bodies[b].Transform.Position.X > 0.4f,
            $"n11: must not tunnel: x={world.Bodies[b].Transform.Position.X}");
    }

    [Fact]
    public void N15_CapsuleBearingStackStable()
    {
        var world = new World3();
        world.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new BoxShape3(new Vector3(10.0f, 0.5f, 10.0f)));
        var boxHe = new Vector3(0.5f, 0.5f, 0.5f);
        world.AddDynamicBody(new Vector3(0.0f, 0.0f, 0.0f), new BoxShape3(boxHe), 1.0f);
        world.AddDynamicBody(new Vector3(0.0f, 1.3f, 0.0f), new CapsuleShape3(0.3f, 0.5f), 1.0f);
        world.AddDynamicBody(new Vector3(0.0f, 2.6f, 0.0f), new BoxShape3(boxHe), 1.0f);
        for (int frame = 0; frame < 600; frame++)
        {
            world.Step(Dt);
            if (frame % 60 == 0)
                foreach (var body in world.Bodies)
                    Assert.True(float.IsFinite(body.Transform.Position.X + body.Velocity.X),
                        $"n15 frame {frame}: non-finite state");
        }
        foreach (var body in world.Bodies.Skip(1))
        {
            Assert.True(body.Transform.Position.Y > -0.6f,
                $"n15: body {body.Id} tunneled: y={body.Transform.Position.Y}");
            Assert.True(body.Velocity.Length() < 2.0f,
                $"n15: body {body.Id} never settled: v={body.Velocity}");
        }
    }

    private static ulong HashWorld2(World2 world)
    {
        ulong h = 0xcbf29ce484222325ul;
        foreach (var body in world.Bodies)
        {
            foreach (float v in new[]
            {
                body.Transform.Position.X, body.Transform.Position.Y,
                body.Velocity.X, body.Velocity.Y,
                body.AngularVelocity, body.RotationAngle,
            })
            {
                h ^= BitConverter.SingleToUInt32Bits(v);
                h *= 0x100000001b3ul;
            }
        }
        return h;
    }

    [Fact]
    public void S03_SleepFlagMutationConsistent()
    {
        (ulong Hash, bool Flag) RunOnce()
        {
            var world = new World2();
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
            world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            for (int i = 0; i < 60; i++) world.Step(Dt);
            var top = world.Bodies[2];
            top.IsSleeping = true;
            world.Bodies[2] = top;
            for (int i = 0; i < 60; i++) world.Step(Dt);
            return (HashWorld2(world), world.Bodies[2].IsSleeping);
        }
        var (h1, f1) = RunOnce();
        var (h2, f2) = RunOnce();
        Assert.Equal((h1, f1), (h2, f2));
        Assert.True(f1, "s03: solver must never clear sleep flags");
    }

    [Fact]
    public void S05_WakeNoPhantomPush()
    {
        var world = new World2();
        AddFloor(world);
        int b = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var bd = world.Bodies[b];
        bd.Velocity = Vector2.Zero;
        bd.AngularVelocity = 0.0f;
        world.Bodies[b] = bd;
        world.Step(Dt);
        Assert.True(world.Bodies[b].IsSleeping, "s05: box must be asleep before wake");
        float restY = world.Bodies[b].Transform.Position.Y;
        world.WakeBody(b);
        float peak = 0;
        for (int i = 0; i < 30; i++)
        {
            world.Step(Dt);
            peak = MathF.Max(peak, world.Bodies[b].Velocity.Length());
        }
        AssertFinite(world, "s05 woken");
        Assert.True(peak < 0.5f, $"s05: wake must not resurrect a phantom push, peak={peak}");
        Assert.True(MathF.Abs(world.Bodies[b].Transform.Position.Y - restY) < 0.1f,
            $"s05: must re-settle, y={world.Bodies[b].Transform.Position.Y}");
    }

    [Fact]
    public void S20_ImpulseThroughSleepingStack()
    {
        (ulong Hash, float X) RunOnce()
        {
            var world = new World2();
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
            for (int i = 0; i < 3; i++)
                world.AddDynamicBody(new Vector2(0.0f, i), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            for (int i = 0; i < 60; i++) world.Step(Dt);
            for (int id = 1; id <= 3; id++)
            {
                var bd = world.Bodies[id];
                bd.Velocity = Vector2.Zero;
                bd.AngularVelocity = 0.0f;
                world.Bodies[id] = bd;
            }
            world.Step(Dt);
            world.Bodies[1].ApplyImpulse(new Vector2(3.0f, 0.0f), new Vector2(0.0f, 0.0f));
            for (int i = 0; i < 60; i++) world.Step(Dt);
            foreach (var body in world.Bodies)
                AssertFinite(world, "s20");
            return (HashWorld2(world), world.Bodies[1].Transform.Position.X);
        }
        var (h1, x1) = RunOnce();
        var (h2, x2) = RunOnce();
        Assert.Equal(h1, h2);
        Assert.True(x1 > 0.1f, $"s20: kicked bottom must have moved: x={x1} vs {x2}");
    }

    [Fact]
    public void S21_EarlyExitTrajectoryNeutral()
    {
        World2 RunOnce(float tol)
        {
            var world = new World2();
            world.Solver.EarlyExitTolerance = tol;
            world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
            world.AddDynamicBody(new Vector2(-0.5f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
            world.AddDynamicBody(new Vector2(0.5f, 0.5f), new CircleShape(0.4f), 1.0f);
            for (int i = 0; i < 300; i++) world.Step(Dt);
            return world;
        }
        var on = RunOnce(1e-6f);
        var on2 = RunOnce(1e-6f);
        var off = RunOnce(0.0f);
        Assert.Equal(HashWorld2(on), HashWorld2(on2));
        float dpos = 0, dvel = 0;
        for (int i = 0; i < on.Bodies.Count; i++)
        {
            dpos = MathF.Max(dpos, (on.Bodies[i].Transform.Position - off.Bodies[i].Transform.Position).Length());
            dvel = MathF.Max(dvel, (on.Bodies[i].Velocity - off.Bodies[i].Velocity).Length());
        }
        Assert.True(dpos < 1e-6f && dvel < 1e-6f,
            $"s21: threshold must not change the trajectory: dpos={dpos:e} dvel={dvel:e}");
        foreach (var body in on.Bodies.Skip(1))
            Assert.True(body.Velocity.Length() < 1.0f, $"s21: unsettled: v={body.Velocity}");
    }

    [Fact]
    public void N12_NeedleHighSpeedBounded()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.05f, 2.0f, 2.0f)));
        int b = world.AddDynamicBody(new Vector3(-5.0f, 0.0f, 0.0f), new CapsuleShape3(0.05f, 1.0f), 1.0f);
        var bd = world.Bodies[b];
        bd.Velocity = new Vector3(100.0f, 0.0f, 0.0f);
        world.Bodies[b] = bd;
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            var body = world.Bodies[b];
            var p = body.Transform.Position;
            Assert.True(float.IsFinite(p.X + p.Y + p.Z), $"n12 frame {frame}: non-finite state");
            Assert.True(body.Velocity.Length() < 500.0f, $"n12 frame {frame}: speed exploded: {body.Velocity}");
            Assert.True(MathF.Abs(p.X) < 2000.0f, $"n12 frame {frame}: escaped bounds: {p}");
        }
    }

    [Fact]
    public void S04_SleepThresholdBoundary()
    {
        var world = new World2();
        int b = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.3f), 1.0f);
        var body = world.Bodies[b];
        body.Velocity = new Vector2(0.01f, 0.0f);
        body.AngularVelocity = 0.0f;
        Assert.False(body.ShouldSleep(0.01f, 0.01f), "s04: exactly at threshold stays awake (strict <)");
        body.Velocity = new Vector2(0.009f, 0.0f);
        Assert.True(body.ShouldSleep(0.01f, 0.01f), "s04: below threshold sleeps");
        body.Velocity = Vector2.Zero;
        body.AngularVelocity = 0.01f;
        Assert.False(body.ShouldSleep(0.01f, 0.01f), "s04: angular at threshold stays awake");
        body.AngularVelocity = 0.009f;
        Assert.True(body.ShouldSleep(0.01f, 0.01f), "s04: angular below sleeps");
        for (int i = 0; i < 1000; i++)
            Assert.False(body.ShouldSleep(0.0f, 0.0f));
    }

    [Fact]
    public void W08_FlippedNormalColdRestart()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        var he = new Vector2(0.05f, 0.05f);
        int a = world.AddDynamicBody(new Vector2(-0.04f, 0.0f), new BoxShape2(he), 1.0f);
        int b = world.AddDynamicBody(new Vector2(0.04f, 0.0f), new BoxShape2(he), 1.0f);
        void Place(int id, Vector2 p)
        {
            var bd = world.Bodies[id];
            var t = bd.Transform;
            t.Position = p;
            bd.Transform = t;
            bd.Velocity = Vector2.Zero;
            world.Bodies[id] = bd;
        }
        for (int i = 0; i < 5; i++)
        {
            Place(a, new Vector2(-0.04f, 0.0f));
            Place(b, new Vector2(0.04f, 0.0f));
            world.Step(Dt);
        }
        Assert.NotEmpty(world.ContactCache);
        world.Solver.Iterations = 1;
        Place(b, new Vector2(-0.12f, 0.0f));
        world.Step(Dt);
        AssertFinite(world, "w08 flipped");
        Assert.True(world.Bodies[b].Velocity.X < 0.02f,
            $"w08: B must move -x under the fresh normal, v={world.Bodies[b].Velocity}");
        Assert.True(world.Bodies[a].Velocity.X > -0.02f,
            $"w08: A must move +x under the fresh normal, v={world.Bodies[a].Velocity}");
    }

    [Fact]
    public void W09_OneFrameGapColdRestart()
    {
        var world = new World2();
        AddFloor(world);
        int b = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        float restY = world.Bodies[b].Transform.Position.Y;
        var bd = world.Bodies[b];
        var t = bd.Transform;
        t.Position = new Vector2(0.0f, restY + 1.0f);
        bd.Transform = t;
        bd.Velocity = Vector2.Zero;
        world.Bodies[b] = bd;
        world.Step(Dt);
        Assert.Empty(world.ContactCache);
        bd = world.Bodies[b];
        t = bd.Transform;
        t.Position = new Vector2(0.0f, restY);
        bd.Transform = t;
        bd.Velocity = Vector2.Zero;
        world.Bodies[b] = bd;
        float peak = 0;
        for (int i = 0; i < 30; i++)
        {
            world.Step(Dt);
            peak = MathF.Max(peak, world.Bodies[b].Velocity.Length());
        }
        AssertFinite(world, "w09 returned");
        Assert.True(peak < 1.0f, $"w09: re-established contact must settle, peak={peak}");
        Assert.True(MathF.Abs(world.Bodies[b].Transform.Position.Y - restY) < 0.2f,
            $"w09: must recover rest pose, y={world.Bodies[b].Transform.Position.Y}");
    }

    [Fact]
    public void W10_RemoveReaddSameIdCold()
    {
        var world = new World2();
        AddFloor(world);
        for (int i = 0; i < 3; i++)
            world.AddDynamicBody(new Vector2(0.0f, i), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        // Bodies: floor 0, A 1, B 2, C 3. Cached pairs: (0,1), (1,2), (2,3).
        for (int i = 0; i < 120; i++) world.Step(Dt);
        Assert.NotEmpty(world.ContactCache);
        // Remove middle B (id 2, non-tail -> remap path, last = 3 moves to 2).
        world.RemoveBody(2);
        Assert.Equal(3, world.Bodies.Count);
        Assert.Equal(2, world.Bodies[2].Id);
        Assert.True(world.ContactCache.TryGetValue((0, 1), out var kept) && kept.Count > 0,
            "w10: survivor pair (0,1) must keep its entries");
        Assert.False(world.ContactCache.ContainsKey((1, 2)), "w10: dropped pair (1,2) must be gone");
        foreach (var (ka, kb) in world.ContactCache.Keys)
            Assert.True(ka < world.Bodies.Count && kb < world.Bodies.Count,
                $"w10: dangling cache key ({ka}, {kb})");
        // Tail path: removing the last body retains survivor keys untouched.
        var before = new List<(int, int)>(world.ContactCache.Keys);
        world.RemoveBody(2);
        Assert.Equal(2, world.Bodies.Count);
        foreach (var k in before)
            Assert.True(world.ContactCache.ContainsKey(k), $"w10: tail removal must not disturb {k}");
        int c = world.AddDynamicBody(new Vector2(3.0f, 2.0f), new CircleShape(0.3f), 1.0f);
        Assert.Equal(2, c);
        for (int frame = 0; frame < 120; frame++)
        {
            world.Step(Dt);
            AssertFinite(world, $"w10 frame {frame}");
            foreach (var (ka, kb) in world.ContactCache.Keys)
                Assert.True(ka < world.Bodies.Count && kb < world.Bodies.Count,
                    $"w10: dangling cache key ({ka}, {kb})");
        }
        Assert.True(world.Bodies[1].Transform.Position.Y > -0.6f,
            $"w10: survivor must not tunnel: y={world.Bodies[1].Transform.Position.Y}");
        Assert.True(world.Bodies[c].Transform.Position.Y > -0.6f,
            $"w10: re-added body must settle: y={world.Bodies[c].Transform.Position.Y}");
    }

    [Fact]
    public void D22_WindingInvariance()
    {
        var ccw = new List<Vector2> { new(-0.5f, -0.5f), new(0.5f, -0.5f), new(0.0f, 0.5f) };
        var cw = new List<Vector2>(ccw);
        cw.Reverse();
        var world = new World2();
        int a = world.AddDynamicBody(Vector2.Zero, new PolygonShape(ccw), 2.0f);
        int b = world.AddDynamicBody(Vector2.Zero, new PolygonShape(cw), 2.0f);
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[a].Mass),
            BitConverter.SingleToUInt32Bits(world.Bodies[b].Mass));
        Assert.Equal(
            BitConverter.SingleToUInt32Bits(world.Bodies[a].Inertia),
            BitConverter.SingleToUInt32Bits(world.Bodies[b].Inertia));
        var box = new List<Vector2> { new(0.1f, -0.6f), new(1.1f, -0.6f), new(1.1f, 0.4f), new(0.1f, 0.4f) };
        var ra = SAT.PolygonPolygon(((PolygonShape)world.Bodies[a].Shape!).Vertices, box);
        var rb = SAT.PolygonPolygon(((PolygonShape)world.Bodies[b].Shape!).Vertices, box);
        Assert.NotNull(ra);
        Assert.NotNull(rb);
        Assert.True(MathF.Abs(ra.Value.Penetration - rb.Value.Penetration) < 1e-6f,
            $"d22: depths must agree: {ra.Value.Penetration} vs {rb.Value.Penetration}");
        Assert.True(Vector2.Dot(ra.Value.Normal, rb.Value.Normal) > 0.999f,
            $"d22: normals must agree: {ra.Value.Normal} vs {rb.Value.Normal}");
    }

    [Fact]
    public void D26_ZeroLengthCapsuleIsSphere()
    {
        // Creation rejects it ("use Sphere instead"); the support math
        // itself is exact. Both levels pinned.
        var w3 = new World3();
        Assert.Throws<ArgumentException>(() => w3.AddDynamicBody(
            Vector3.Zero, new CapsuleShape3(0.5f, 0.0f), 1.0f));
        var center = new Vector3(1.5f, 0, 0);
        foreach (var dir in new[]
        {
            Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ, -Vector3.UnitX,
            Vector3.Normalize(new Vector3(1, 2, 3)), Vector3.Normalize(new Vector3(-0.3f, 0.8f, -0.5f)),
        })
        {
            var cap = GjkEpa3D.SupportCapsule(center, center, 0.5f, dir);
            var sph = GjkEpa3D.SupportSphere(center, 0.5f, dir);
            Assert.True((cap - sph).Length() < 1e-6f,
                $"d26: supports must agree along {dir}: {cap} vs {sph}");
        }
        var he = Vector3.One;
        var bc = new Vector3(1.2f, 0, 0);
        Func<Vector3, Vector3> box = dir => GjkEpa3D.SupportObb(Vector3.Zero, he, Quaternion.Identity, dir);
        Func<Vector3, Vector3> sph2 = dir => GjkEpa3D.SupportSphere(bc, 0.5f, dir);
        Func<Vector3, Vector3> cap2 = dir => GjkEpa3D.SupportCapsule(bc, bc, 0.5f, dir);
        var ds = GjkEpa3D.GjkEpa(box, sph2);
        var hit = GjkEpa3D.GjkEpa(box, cap2);
        Assert.NotNull(ds);
        Assert.NotNull(hit);
        Assert.True(MathF.Abs(hit.Depth - ds.Depth) < 1e-3f,
            $"d26: EPA depths must agree: {hit.Depth} vs {ds.Depth}");
        Assert.True(MathF.Abs(hit.Normal.X) > 0.99f, $"d26: normal must point along +x: {hit.Normal}");
    }

    [Fact]
    public void D27_HullMassStubPinned()
    {
        // KNOWN GAP: see LIMITATIONS.md — hull mass is a stub.
        var verts = new List<Vector3> { new(0, 0, 0), new(1, 0, 0), new(0, 1, 0), new(0, 0, 1) };
        var world = new World3();
        Assert.Throws<ArgumentException>(() => world.AddDynamicBody(
            Vector3.Zero,
            new ConvexHullShape3(new List<Vector3> { new(0, 0, 0), new(1, 0, 0), new(0, 1, 0) }),
            1.0f));
        int a = world.AddDynamicBody(Vector3.Zero, new ConvexHullShape3(verts), 1.0f);
        int b = world.AddDynamicBody(new Vector3(5, 0, 0), new ConvexHullShape3(verts), 7.0f);
        Assert.Equal(world.Bodies[a].Mass, world.Bodies[b].Mass);
        Assert.Equal(world.Bodies[a].Inertia.M11, world.Bodies[b].Inertia.M11);
        Assert.Equal(world.Bodies[a].Inertia.M22, world.Bodies[b].Inertia.M22);
        Assert.Equal(world.Bodies[a].Inertia.M33, world.Bodies[b].Inertia.M33);
        for (int i = 0; i < 60; i++) world.Step(Dt);
    }

    [Fact]
    public void S01_KnownLimitation_NoAutowakeOnContact()
    {
        // KNOWN LIMITATION: see LIMITATIONS.md — no auto-wake on contact.
        var world = new World2();
        AddFloor(world);
        int sleeper = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        var bd = world.Bodies[sleeper];
        bd.Velocity = Vector2.Zero;
        bd.AngularVelocity = 0.0f;
        world.Bodies[sleeper] = bd;
        world.Step(Dt);
        Assert.True(world.Bodies[sleeper].IsSleeping, "s01: box must be asleep before the drop");
        world.AddDynamicBody(new Vector2(0.0f, 3.0f), new CircleShape(0.3f), 1.0f);
        for (int i = 0; i < 180; i++) world.Step(Dt);
        AssertFinite(world, "s01 dropped ball");
        Assert.True(world.Bodies[sleeper].IsSleeping, "s01 KNOWN LIMITATION: sleeper still asleep");
        Assert.True((world.Bodies[sleeper].Transform.Position - new Vector2(0.0f, 0.0f)).Length() < 0.05f,
            $"s01: frozen sleeper must not drift: {world.Bodies[sleeper].Transform.Position}");
    }

    [Fact]
    public void S02_KnownLimitation_NoWakeCascade()
    {
        // KNOWN LIMITATION: see LIMITATIONS.md — waking the bottom of a
        // sleeping stack does not cascade; the top keeps its flag while
        // the pipeline (which never skips sleepers) lets it sink.
        var world = new World2();
        AddFloor(world);
        int bottom = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        int top = world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        foreach (int id in new[] { bottom, top })
        {
            var bd = world.Bodies[id];
            bd.Velocity = Vector2.Zero;
            bd.AngularVelocity = 0.0f;
            world.Bodies[id] = bd;
        }
        world.Step(Dt);
        Assert.True(world.Bodies[bottom].IsSleeping && world.Bodies[top].IsSleeping,
            "s02: stack must start asleep");
        world.WakeBody(bottom);
        var moved = world.Bodies[bottom];
        var t = moved.Transform;
        t.Position = new Vector2(0.0f, 0.2f);
        moved.Transform = t;
        moved.Velocity = Vector2.Zero;
        world.Bodies[bottom] = moved;
        for (int i = 0; i < 120; i++) world.Step(Dt);
        AssertFinite(world, "s02 woken stack");
        Assert.True(world.Bodies[bottom].Transform.Position.Y < 0.05f,
            $"s02: woken bottom must have fallen back: y={world.Bodies[bottom].Transform.Position.Y}");
        Assert.True(world.Bodies[top].IsSleeping, "s02 KNOWN LIMITATION: top still asleep");
        Assert.True(world.Bodies[top].Transform.Position.Y < 1.0f,
            $"s02: unfrozen top must have sunk while flagged asleep: y={world.Bodies[top].Transform.Position.Y}");
    }

    [Fact]
    public void V01_NaNPositionCorrected()
    {
        var world = new World2();
        world.Validation = new ValidationConfig { CheckFinite = true };
        int id = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        var body = world.Bodies[id];
        var t = body.Transform;
        t.Position = new Vector2(float.NaN, 5.0f);
        body.Transform = t;
        world.Bodies[id] = body;
        world.Step(Dt);
        Assert.True(float.IsFinite(world.Bodies[id].Transform.Position.X));
        Assert.True(float.IsFinite(world.Bodies[id].Transform.Position.Y));
        Assert.True(world.ValidationReport.NonFiniteCorrected >= 1);
    }

    [Fact]
    public void V02_VelocityClamped()
    {
        var world = new World2();
        world.Validation = new ValidationConfig { MaxVelocity = 10.0f };
        int id = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        world.Bodies[id].Velocity = new Vector2(100.0f, 0.0f);
        world.Step(Dt);
        Assert.True(world.Bodies[id].Velocity.Length() <= 10.0f + 1e-4f);
        Assert.Equal(1, world.ValidationReport.VelocityClamped);
    }

    [Fact]
    public void V03_AngularVelocityClamped()
    {
        var world = new World2();
        world.Validation = new ValidationConfig { MaxAngularVelocity = 5.0f };
        int id = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Bodies[id].AngularVelocity = 100.0f;
        world.Step(Dt);
        Assert.True(MathF.Abs(world.Bodies[id].AngularVelocity) <= 5.0f + 1e-4f);
        Assert.Equal(1, world.ValidationReport.AngularVelocityClamped);
    }

    [Fact]
    public void V04_PositionClamped()
    {
        var world = new World2();
        world.Validation = new ValidationConfig { MaxCoordinate = 100.0f };
        int id = world.AddDynamicBody(new Vector2(1000.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        world.Step(Dt);
        Assert.True(world.Bodies[id].Transform.Position.Length() <= 100.0f + 1e-4f);
        Assert.Equal(1, world.ValidationReport.CoordinateClamped);
    }

    [Fact]
    public void V05_PenetrationClamped()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.Validation = new ValidationConfig { MaxPenetration = 0.1f };
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.1f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Step(Dt);
        Assert.True(world.ValidationReport.PenetrationClamped >= 1);
    }

    [Fact]
    public void V06_MassRatioDetected()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.Validation = new ValidationConfig { MaxMassRatio = 100.0f };
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1000.0f);
        world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Step(Dt);
        Assert.Equal(1, world.ValidationReport.MassRatioExceeded);
    }

    [Fact]
    public void V07_Validation3D()
    {
        var world = new World3();
        world.Validation = new ValidationConfig { CheckFinite = true, MaxVelocity = 10.0f };
        int id = world.AddDynamicBody(new Vector3(0.0f, 5.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        world.Bodies[id].Velocity = new Vector3(100.0f, 0.0f, 0.0f);
        world.Step(Dt);
        Assert.True(world.Bodies[id].Velocity.Length() <= 10.0f + 1e-4f);
        Assert.Equal(1, world.ValidationReport.VelocityClamped);
    }

    [Fact]
    public void V08_StrictConfig()
    {
        var world = new World2();
        world.Validation = ValidationConfig.Strict;
        int id = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        world.Bodies[id].Velocity = new Vector2(float.NaN, 0.0f);
        world.Step(Dt);
        Assert.True(float.IsFinite(world.Bodies[id].Velocity.X));
        Assert.True(world.ValidationReport.NonFiniteCorrected >= 1);
    }

    [Fact]
    public void V09_PermissiveConfig()
    {
        var world = new World2();
        world.Validation = ValidationConfig.Permissive;
        int id = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        world.Bodies[id].Velocity = new Vector2(10000.0f, 0.0f);
        world.Step(Dt);
        Assert.True(float.IsFinite(world.Bodies[id].Velocity.X));
    }

    [Fact]
    public void V10_ValidationReportAggregates()
    {
        var world = new World2();
        world.Validation = new ValidationConfig { CheckFinite = true, MaxVelocity = 10.0f, MaxCoordinate = 100.0f };
        int id1 = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        var body = world.Bodies[id1];
        var t = body.Transform;
        t.Position = new Vector2(float.NaN, 5.0f);
        body.Transform = t;
        world.Bodies[id1] = body;
        int id2 = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        world.Bodies[id2].Velocity = new Vector2(100.0f, 0.0f);
        world.Step(Dt);
        Assert.True(world.ValidationReport.NonFiniteCorrected >= 1);
        Assert.True(world.ValidationReport.VelocityClamped >= 1);
        Assert.True(world.ValidationReport.AnyIssues);
    }

    [Fact]
    public void V11_Validation3DNaNPositionReports()
    {
        // Regression pin: the 3D validator previously reset its detection flag
        // between position/velocity/angular checks, so a NaN position with
        // finite velocity was corrected but reported as Ok.
        var world = new World3();
        world.Validation = new ValidationConfig { CheckFinite = true };
        int id = world.AddDynamicBody(new Vector3(0.0f, 5.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        var body = world.Bodies[id];
        var t = body.Transform;
        t.Position = new Vector3(float.NaN, 5.0f, 0.0f);
        body.Transform = t;
        body.Velocity = new Vector3(1.0f, 0.0f, 0.0f);
        world.Bodies[id] = body;
        world.Step(Dt);
        Assert.True(float.IsFinite(world.Bodies[id].Transform.Position.X));
        Assert.True(float.IsFinite(world.Bodies[id].Transform.Position.Y));
        Assert.True(float.IsFinite(world.Bodies[id].Transform.Position.Z));
        Assert.True(world.ValidationReport.NonFiniteCorrected >= 1);
    }
}
