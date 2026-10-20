using System.Numerics;
using Xunit;
using PhiEng.Collision;
using PhiEng.Dynamics;
using PhiEng.Math;

namespace PhiEng.Tests;

/// <summary>
/// Pipeline-level tests: friction on inclines, stack stability, warm-start cache.
/// Mirrors the Rust tests in crates/engine/tests/integration.rs.
/// </summary>
public class PhiTests
{
    private const float Dt = 1.0f / 60.0f;

    [Fact]
    public void Box_RestsOnFloor()
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int boxId = world.AddDynamicBody(
            new Vector2(0.0f, 3.0f),
            new BoxShape2(new Vector2(0.5f, 0.5f)),
            1.0f);

        for (int i = 0; i < 600; i++)
        {
            world.Step(Dt);
        }

        var b = world.Bodies[boxId];
        Assert.True(b.Transform.Position.Y > -0.5f, $"box tunneled through floor: y={b.Transform.Position.Y}");
        Assert.True(b.Velocity.Length() < 1.0f, $"box should have settled: v={b.Velocity}");
    }

    [Fact]
    public void OverlappingBoxes_Separate()
    {
        var a = new List<Vector2>
        {
            new Vector2(-0.5f, -0.5f),
            new Vector2(0.5f, -0.5f),
            new Vector2(0.5f, 0.5f),
            new Vector2(-0.5f, 0.5f)
        };
        var b = a.Select(v => v + new Vector2(0.5f, 0.2f)).ToList();

        var sat = SAT.PolygonPolygon(a, b);
        Assert.NotNull(sat);
        Assert.True(sat.Value.Penetration > 0.0f);
        var contacts = SAT.PolygonPolygonContactPoints(a, b, sat.Value.Normal);
        Assert.NotEmpty(contacts);

        // Through the world pipeline the pair must push apart along +x
        // (the least-penetration axis), not tunnel.
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int top = world.AddDynamicBody(
            new Vector2(0.95f, 0.0f),
            new BoxShape2(new Vector2(0.5f, 0.5f)),
            1.0f);
        for (int i = 0; i < 120; i++)
        {
            world.Step(Dt);
        }
        var t = world.Bodies[top];
        Assert.True(t.Transform.Position.X > 0.95f, $"box should separate along +x: x={t.Transform.Position.X}");
        Assert.True(t.Transform.Position.Y > -1.0f, $"box sank through static box: y={t.Transform.Position.Y}");
    }

    private static (World2 World, int BoxId, Vector2 Start) InclineWorld(float angle, float mu)
    {
        var world = new World2();
        // Static ramp: wide box rotated about its center.
        int rampId = world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(5.0f, 0.5f)));
        world.Bodies[rampId].Transform = new Transform2(new Vector2(0.0f, 0.0f), angle);
        world.Bodies[rampId].Material.Friction = mu;

        // Surface frame: outward normal and start point on the top face.
        var frame = new Transform2(Vector2.Zero, angle);
        Vector2 normal = frame.RotateVector(Vector2.UnitY);
        Vector2 start = frame.TransformPoint(new Vector2(0.0f, 0.5f)) + normal * 0.51f;
        int boxId = world.AddDynamicBody(start, new BoxShape2(new Vector2(0.5f, 0.5f)), 1.0f);
        world.Bodies[boxId].Transform = new Transform2(start, angle);
        world.Bodies[boxId].Material.Friction = mu;
        world.Bodies[boxId].Material.Restitution = 0.0f;
        return (world, boxId, start);
    }

    [Fact]
    public void Incline_SticksWithFriction()
    {
        // tan(25deg) ~= 0.47 < mu=0.9, so the box must hold.
        float angle = 25.0f * MathF.PI / 180.0f;
        var (world, boxId, start) = InclineWorld(angle, 0.9f);
        for (int i = 0; i < 300; i++)
        {
            world.Step(Dt);
        }
        var body = world.Bodies[boxId];
        float slid = (body.Transform.Position - start).Length();
        Assert.True(slid < 0.6f, $"high friction should hold the box on the incline, slid={slid}");
        Assert.True(body.Transform.Position.Y > -2.0f, $"box tunneled through ramp: y={body.Transform.Position.Y}");
    }

    [Fact]
    public void Incline_SlidesWithoutFriction()
    {
        float angle = 25.0f * MathF.PI / 180.0f;
        var (world, boxId, start) = InclineWorld(angle, 0.0f);
        for (int i = 0; i < 300; i++)
        {
            world.Step(Dt);
        }
        var body = world.Bodies[boxId];
        float slid = (body.Transform.Position - start).Length();
        Assert.True(slid > 2.0f, $"zero friction should let the box slide, slid={slid}");
    }

    [Fact]
    public void Stack_Stabilizes()
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        var ids = new List<int>();
        for (int i = 0; i < 3; i++)
        {
            ids.Add(world.AddDynamicBody(
                new Vector2(0.0f, 0.05f + i * 1.02f),
                new BoxShape2(new Vector2(0.5f, 0.5f)),
                1.0f));
        }
        for (int i = 0; i < 600; i++)
        {
            world.Step(Dt);
        }
        for (int level = 0; level < ids.Count; level++)
        {
            var body = world.Bodies[ids[level]];
            Assert.True(
                body.Transform.Position.Y > level - 0.75f,
                $"stack level {level} tunneled: y={body.Transform.Position.Y}");
            Assert.True(
                body.Velocity.Length() < 1.0f,
                $"stack level {level} jittering: v={body.Velocity}");
        }
    }

    [Fact]
    public void WarmStart_CacheCarriesImpulse()
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        // Slight initial overlap so contact exists on frame 0.
        int boxId = world.AddDynamicBody(
            new Vector2(0.0f, -0.05f),
            new BoxShape2(new Vector2(0.5f, 0.5f)),
            1.0f);
        world.Bodies[boxId].Material.Restitution = 0.0f;

        world.Step(Dt);
        Assert.True(
            world.ContactCache.TryGetValue((0, boxId), out var entries) && entries.Count > 0,
            "resting contact must be cached");
        float total = entries!.Sum(e => e.NormalImpulse);
        Assert.True(total > 0.0f, $"cached normal impulse must be positive, got {total}");
    }

    [Fact]
    public void CircleInsidePolygon_PushesOutNearestFace()
    {
        // Circle center inside the box, near the top face: must eject UP
        // through the nearest face (pen = r + gap), not down through the box.
        var box = new List<Vector2>
        {
            new Vector2(-1.0f, -1.0f),
            new Vector2(1.0f, -1.0f),
            new Vector2(1.0f, 1.0f),
            new Vector2(-1.0f, 1.0f)
        };
        var hit = SAT.CirclePolygon(new Vector2(0.0f, 0.9f), 0.5f, box);
        Assert.NotNull(hit);
        Assert.True(hit.Value.Normal.Y > 0.9f, $"normal must point up, got {hit.Value.Normal}");
        Assert.True(MathF.Abs(hit.Value.Normal.X) < 1e-4f);
        Assert.True(MathF.Abs(hit.Value.Penetration - 0.6f) < 1e-4f, $"pen={hit.Value.Penetration}");
    }

    [Fact]
    public void CirclePolygonCornerGrazing_Separates()
    {
        // Regression test (mirrors the Rust corner-grazing case): a
        // circle sitting past a box corner, close enough to overlap
        // every face-normal projection yet separated along the corner
        // axis, must report no contact.
        var center = new Vector2(-5.874491f, 1.321865f);
        var xAxis = new Vector2(0.84011143f, -0.54241383f);
        var yAxis = new Vector2(0.54241383f, 0.84011143f);
        float he = 0.3f;
        var local = new[]
        {
            new Vector2(-he, -he),
            new Vector2(he, -he),
            new Vector2(he, he),
            new Vector2(-he, he),
        };
        var worldVerts = new List<Vector2>();
        foreach (var v in local)
        {
            worldVerts.Add(new Vector2(
                xAxis.X * v.X + yAxis.X * v.Y + center.X,
                xAxis.Y * v.X + yAxis.Y * v.Y + center.Y));
        }
        var circle = new Vector2(-5.15f, 1.4447408f);
        float best = float.MaxValue;
        foreach (var v in worldVerts)
        {
            best = MathF.Min(best, (v - circle).Length());
        }
        Assert.True(best > 0.3f, $"test geometry must be separated, gap is {best - 0.3f}");
        var hit = SAT.CirclePolygon(circle, 0.3f, worldVerts);
        Assert.Null(hit);
    }

    [Fact]
    public void CirclePolygonTrueCornerOverlap_Detected()
    {
        // Positive control: push the same circle into genuine overlap
        // and the contact must be reported.
        var center = new Vector2(-5.874491f, 1.321865f);
        var xAxis = new Vector2(0.84011143f, -0.54241383f);
        var yAxis = new Vector2(0.54241383f, 0.84011143f);
        float he = 0.3f;
        var local = new[]
        {
            new Vector2(-he, -he),
            new Vector2(he, -he),
            new Vector2(he, he),
            new Vector2(-he, he),
        };
        var worldVerts = new List<Vector2>();
        foreach (var v in local)
        {
            worldVerts.Add(new Vector2(
                xAxis.X * v.X + yAxis.X * v.Y + center.X,
                xAxis.Y * v.X + yAxis.Y * v.Y + center.Y));
        }
        var circle = new Vector2(-5.15f, 1.4447408f);
        Vector2 nearest = worldVerts[0];
        float best = (nearest - circle).LengthSquared();
        for (int i = 1; i < worldVerts.Count; i++)
        {
            float d = (worldVerts[i] - circle).LengthSquared();
            if (d < best)
            {
                best = d;
                nearest = worldVerts[i];
            }
        }
        Vector2 dir = circle - nearest;
        float len = dir.Length();
        Assert.True(len > 1e-6f);
        dir /= len;
        var moved = circle - dir * 0.0616f;
        var hit = SAT.CirclePolygon(moved, 0.3f, worldVerts);
        Assert.NotNull(hit);
    }

    [Fact]
    public void Rotation_IntegratesWithoutPingPong()
    {
        var body = new RigidBody2(0);
        body.AngularVelocity = 1.0f;
        float prev = 0.0f;
        for (int i = 0; i < 10; i++)
        {
            body.Integrate(Dt);
            float a = body.RotationAngle;
            Assert.True(a > prev, $"angle must advance monotonically, prev={prev} now={a}");
            prev = a;
        }
        Assert.True(MathF.Abs(prev - 10.0f * Dt) < 1e-3f, $"angle={prev}");
    }

    [Fact]
    public void ApplyForce_ScalesWithDt()
    {
        var body = new RigidBody2(0);
        body.SetShape(new CircleShape(1.0f), 1.0f);
        float mass = body.Mass;
        body.ApplyForce(new Vector2(10.0f, 0.0f), Vector2.Zero, 0.5f);
        Assert.True(MathF.Abs(body.Velocity.X - 10.0f / mass * 0.5f) < 1e-5f, $"vx={body.Velocity.X}");

        var wall = RigidBody2.StaticBody(1);
        wall.ApplyForce(new Vector2(10.0f, 0.0f), Vector2.Zero, 1.0f);
        Assert.Equal(Vector2.Zero, wall.Velocity);
    }

    private static ulong DeterministicSceneHash()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        for (int i = 0; i < 6; i++)
        {
            float x = -3.0f + i * 1.2f;
            if (i % 2 == 0)
                world.AddDynamicBody(new Vector2(x, 2.0f + i * 0.7f), new BoxShape2(new Vector2(0.4f, 0.4f)), 1.0f);
            else
                world.AddDynamicBody(new Vector2(x + 0.3f, 2.5f + i * 0.7f), new CircleShape(0.35f), 1.5f);
        }
        for (int i = 0; i < 300; i++) world.Step(Dt);
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

    /// <summary>
    /// Determinism: same scene twice must be bit-identical. Mirrors the
    /// Rust <c>test_determinism_bit_exact</c>. Cross-language equality is
    /// NOT asserted (transcendentals may differ); see README.
    /// </summary>
    [Fact]
    public void Determinism_BitExact()
    {
        ulong a = DeterministicSceneHash();
        ulong b = DeterministicSceneHash();
        Assert.Equal(a, b);
    }

    [Fact]
    public void RemoveBody_KeepsIdIndexInvariant()
    {
        var world = new World2();
        int a = world.AddDynamicBody(new Vector2(-2.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        int b = world.AddDynamicBody(new Vector2(0.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        int c = world.AddDynamicBody(new Vector2(2.0f, 5.0f), new CircleShape(0.5f), 1.0f);
        Assert.Equal((0, 1, 2), (a, b, c));
        world.RemoveBody(1);
        Assert.True(world.Bodies.Count == 2);
        Assert.Equal(0, world.Bodies[0].Id);
        Assert.Equal(1, world.Bodies[1].Id);
        Assert.Equal(2.0f, world.Bodies[1].Transform.Position.X);
        world.RemoveBody(99); // no-op
        Assert.True(world.Bodies.Count == 2);
        int d = world.AddDynamicBody(new Vector2(0.0f, 8.0f), new CircleShape(0.5f), 1.0f);
        Assert.Equal(2, d);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        for (int i = 0; i < 120; i++) world.Step(Dt);
        Assert.True(world.Bodies[0].Transform.Position.Y < 5.0f);
    }

    [Fact]
    public void RemoveBody3D_KeepsIdIndexInvariant()
    {
        var world = new World3();
        world.AddDynamicBody(new Vector3(-2.0f, 5.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        world.AddDynamicBody(new Vector3(2.0f, 5.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        world.RemoveBody(0);
        Assert.True(world.Bodies.Count == 1);
        Assert.Equal(0, world.Bodies[0].Id);
        Assert.Equal(2.0f, world.Bodies[0].Transform.Position.X);
        int d = world.AddDynamicBody(new Vector3(0.0f, 8.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        Assert.Equal(1, d);
        world.Step(Dt);
        Assert.True(world.Bodies[1].Transform.Position.Y < 8.0f);
    }
}
