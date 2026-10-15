using System.Numerics;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Math;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Joint constraint tests, mirroring the Rust joint suite in
/// crates/dynamics/src/joint.rs. Test IDs (J31–J44) are shared across
/// languages; parity is enforced by tools/check_test_parity.py.
/// F31–F33 pin the native joint bridge (mirrored by f31–f33 in
/// crates/engine/src/ffi.rs).
/// </summary>
public class JointTests
{
    private const float Dt = 1.0f / 60.0f;

    private static (World2 World, int Ball) BallOnRod()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int ball = world.AddDynamicBody(new Vector2(0.0f, -2.0f), new CircleShape(0.3f), 1.0f);
        return (world, ball);
    }

    private static void AssertFinite(World2 world)
    {
        foreach (var body in world.Bodies)
        {
            Assert.True(float.IsFinite(body.Transform.Position.X));
            Assert.True(float.IsFinite(body.Velocity.X));
        }
    }

    [Fact]
    public void J31_DistanceRejectsBadInput()
    {
        var world = new World2();
        world.AddStaticBody(Vector2.Zero, new CircleShape(0.5f));
        world.AddStaticBody(Vector2.UnitX, new CircleShape(0.5f));
        int n = world.Bodies.Count;
        var a = Vector2.Zero;
        var b = Vector2.UnitX;
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 99, a, b, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 0, a, b, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 1, new Vector2(float.NaN, 0.0f), b, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 1, a, b, -1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 1, a, b, float.PositiveInfinity, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new DistanceJoint(0, 1, a, b, 1.0f, 0.5f, n));
    }

    [Fact]
    public void J32_DistanceHoldsRestLength()
    {
        var (world, ball) = BallOnRod();
        world.AddJoint(new DistanceJoint(0, ball, Vector2.Zero, Vector2.Zero, 2.0f, 0.0f, world.Bodies.Count));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[ball].Transform.Position;
        float dist = (p - Vector2.Zero).Length();
        Assert.True(MathF.Abs(dist - 2.0f) < 0.05f, $"rod must hold rest length, dist={dist}, pos={p}");
        Assert.True(MathF.Abs(p.X) < 0.1f, $"pendulum hangs straight down, x={p.X}");
    }

    [Fact]
    public void J33_DistancePullsBack()
    {
        var (world, ball) = BallOnRod();
        world.AddJoint(new DistanceJoint(0, ball, Vector2.Zero, Vector2.Zero, 2.0f, 0.0f, world.Bodies.Count));
        var moved = world.Bodies[ball];
        var t = moved.Transform;
        t.Position = new Vector2(1.0f, -1.5f);
        moved.Transform = t;
        moved.Velocity = Vector2.Zero;
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var p = world.Bodies[ball].Transform.Position;
        Assert.True(MathF.Abs(p.X) < 1.0f, $"displaced pendulum must swing back, x={p.X}");
        AssertFinite(world);
    }

    [Fact]
    public void J34_JointRemovedWithBody()
    {
        var (world, ball) = BallOnRod();
        world.AddJoint(new DistanceJoint(0, ball, Vector2.Zero, Vector2.Zero, 2.0f, 0.0f, world.Bodies.Count));
        Assert.Equal(1, world.Joints.Count);
        world.RemoveBody(ball);
        Assert.Empty(world.Joints);
        for (int i = 0; i < 10; i++) world.Step(Dt);
    }

    [Fact]
    public void J35_HingeRejectsBadInput()
    {
        var world = new World2();
        world.AddStaticBody(Vector2.Zero, new CircleShape(0.5f));
        world.AddStaticBody(Vector2.UnitX, new CircleShape(0.5f));
        int n = world.Bodies.Count;
        var a = Vector2.Zero;
        var b = Vector2.UnitX;
        Assert.Throws<ArgumentException>(() => new HingeJoint(0, 99, a, b, 0.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new HingeJoint(1, 1, a, b, 0.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new HingeJoint(0, 1, new Vector2(float.NaN, 0.0f), b, 0.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new HingeJoint(0, 1, a, b, float.NaN, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new HingeJoint(0, 1, a, b, 0.0f, 1.0f, 0.5f, n));
        // Inverted bounds are the valid free-hinge convention.
        var free = new HingeJoint(0, 1, a, b, 1.0f, -1.0f, 0.0f, n);
        Assert.True(free.LimitMin > free.LimitMax);
    }

    private static (World2 World, int Bob) HingeWorld()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int bob = world.AddDynamicBody(new Vector2(2.0f, 0.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        return (world, bob);
    }

    [Fact]
    public void J36_HingePullsToAnchor()
    {
        var (world, bob) = HingeWorld();
        world.AddJoint(new HingeJoint(0, bob, Vector2.Zero, Vector2.Zero, 1.0f, -1.0f, 0.0f, world.Bodies.Count));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[bob].Transform.Position;
        Assert.True(p.Length() < 0.1f, $"hinge must pull the bob to the anchor, pos={p}");
        AssertFinite(world);
    }

    [Fact]
    public void J37_HingeFreeRotation()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.AddStaticBody(new Vector2(0.0f, -2.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int bob = world.AddDynamicBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        world.Bodies[bob].LinearDamping = 0.0f;
        world.Bodies[bob].AngularDamping = 0.0f;
        world.AddJoint(new HingeJoint(0, bob, new Vector2(0.0f, 2.0f), Vector2.Zero, 1.0f, -1.0f, 0.0f, world.Bodies.Count));
        world.Bodies[bob].AngularVelocity = 2.0f;
        for (int i = 0; i < 60; i++) world.Step(Dt);
        float angle = world.Bodies[bob].RotationAngle;
        Assert.True(MathF.Abs(angle - 2.0f) < 0.2f, $"free hinge must not brake spin, angle={angle}");
        Assert.True(world.Bodies[bob].Transform.Position.Length() < 0.05f, "center-pinned bob must not drift");
    }

    [Fact]
    public void J38_HingeLimitsEnforced()
    {
        var (world, bob) = HingeWorld();
        world.AddJoint(new HingeJoint(0, bob, Vector2.Zero, Vector2.Zero, -0.2f, 0.2f, 0.0f, world.Bodies.Count));
        world.Bodies[bob].AngularVelocity = 10.0f;
        for (int i = 0; i < 300; i++) world.Step(Dt);
        float rel = Wrap(world.Bodies[bob].RotationAngle - world.Bodies[0].RotationAngle);
        Assert.True(rel >= -0.35f && rel <= 0.35f, $"hinge limits must catch the windup, rel={rel}");
        AssertFinite(world);
    }

    [Fact]
    public void J39_PrismaticRejectsBadInput()
    {
        var world = new World2();
        world.AddStaticBody(Vector2.Zero, new CircleShape(0.5f));
        world.AddStaticBody(Vector2.UnitX, new CircleShape(0.5f));
        int n = world.Bodies.Count;
        var a = Vector2.Zero;
        var b = Vector2.UnitX;
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 99, a, b, Vector2.UnitX, -1.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(1, 1, a, b, Vector2.UnitX, -1.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 1, new Vector2(float.NaN, 0.0f), b, Vector2.UnitX, -1.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 1, a, b, Vector2.Zero, -1.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 1, a, b, new Vector2(1.0f, float.NaN), -1.0f, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 1, a, b, Vector2.UnitX, float.NaN, 1.0f, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new PrismaticJoint(0, 1, a, b, Vector2.UnitX, -1.0f, 1.0f, 0.5f, n));
        var free = new PrismaticJoint(0, 1, a, b, Vector2.UnitX, 1.0f, -1.0f, 0.0f, n);
        Assert.True(free.LimitMin > free.LimitMax);
    }

    [Fact]
    public void J40_PrismaticSlidesHoldsRail()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int slider = world.AddDynamicBody(new Vector2(0.0f, 1.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        world.AddJoint(new PrismaticJoint(0, slider, new Vector2(0.0f, 2.0f), Vector2.Zero, Vector2.UnitX, 1.0f, -1.0f, 0.0f, world.Bodies.Count));
        world.Bodies[slider].Velocity = new Vector2(5.0f, 0.0f);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var p = world.Bodies[slider].Transform.Position;
        Assert.True(p.X > 1.0f, $"slider must glide along the rail, x={p.X}");
        Assert.True(MathF.Abs(p.Y - 1.0f) < 0.15f, $"rail must hold height against gravity, y={p.Y}");
        AssertFinite(world);
    }

    [Fact]
    public void J41_PrismaticLimitsEnforced()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        world.AddStaticBody(new Vector2(0.0f, 0.0f), new BoxShape2(new Vector2(0.5f, 0.5f)));
        int slider = world.AddDynamicBody(new Vector2(3.0f, 0.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        world.AddJoint(new PrismaticJoint(0, slider, Vector2.Zero, Vector2.Zero, Vector2.UnitX, -1.0f, 1.0f, 0.0f, world.Bodies.Count));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[slider].Transform.Position;
        Assert.True(p.X < 1.2f, $"travel limit must pull the slider back, x={p.X}");
        Assert.True(MathF.Abs(p.Y) < 0.15f, $"rail must hold lateral position, y={p.Y}");
    }

    [Fact]
    public void J42_FixedRejectsBadInput()
    {
        var world = new World2();
        world.AddStaticBody(Vector2.Zero, new CircleShape(0.5f));
        world.AddStaticBody(Vector2.UnitX, new CircleShape(0.5f));
        int n = world.Bodies.Count;
        Assert.Throws<ArgumentException>(() => new FixedJoint(0, 99, Vector2.Zero, Vector2.UnitX, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new FixedJoint(1, 1, Vector2.Zero, Vector2.UnitX, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new FixedJoint(0, 1, new Vector2(0.0f, float.NaN), Vector2.UnitX, 0.0f, n));
        Assert.Throws<ArgumentException>(() => new FixedJoint(0, 1, Vector2.Zero, Vector2.UnitX, 0.5f, n));
    }

    [Fact]
    public void J43_FixedHoldsFormation()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        int a = world.AddDynamicBody(new Vector2(-0.5f, 5.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        int b = world.AddDynamicBody(new Vector2(0.5f, 5.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        world.AddJoint(new FixedJoint(a, b, new Vector2(1.0f, 0.0f), Vector2.Zero, 0.0f, world.Bodies.Count));
        for (int i = 0; i < 180; i++) world.Step(Dt);
        var pa = world.Bodies[a].Transform.Position;
        var pb = world.Bodies[b].Transform.Position;
        float gap = (pb - pa).Length();
        Assert.True(MathF.Abs(gap - 1.0f) < 0.1f, $"weld must hold the formation, gap={gap}");
        AssertFinite(world);
    }

    [Fact]
    public void J44_FixedResistsRotation()
    {
        var world = new World2();
        world.Gravity = Vector2.Zero;
        int a = world.AddDynamicBody(new Vector2(-0.5f, 0.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        int b = world.AddDynamicBody(new Vector2(0.5f, 0.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);
        world.Bodies[a].LinearDamping = 0.0f;
        world.Bodies[a].AngularDamping = 0.0f;
        world.Bodies[b].LinearDamping = 0.0f;
        world.Bodies[b].AngularDamping = 0.0f;
        world.AddJoint(new FixedJoint(a, b, new Vector2(1.0f, 0.0f), Vector2.Zero, 0.0f, world.Bodies.Count));
        world.Bodies[a].AngularVelocity = 5.0f;
        for (int i = 0; i < 120; i++) world.Step(Dt);
        float rel = Wrap(world.Bodies[b].RotationAngle - world.Bodies[a].RotationAngle);
        Assert.True(MathF.Abs(rel) < 0.3f, $"weld must keep co-rotation, rel={rel}");
    }

    [Fact]
    public void F31_NativeJointNullSafety()
    {
        using var world = new PhiNativeWorld();
        Assert.Equal(0, world.JointCount);
        Assert.Throws<ArgumentException>(() => world.AddJoint(99, 0, 1, "{}"));
        Assert.Throws<ArgumentOutOfRangeException>(() => world.RemoveJoint(999));
    }

    [Fact]
    public void F32_NativeJointRejectsBadJson()
    {
        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        Assert.Throws<ArgumentException>(() => world.AddJoint(0, 0, ball, "{nope"));
        Assert.Throws<ArgumentException>(() => world.AddJoint(0, 0, ball, "{\"anchorA\":[0,0],\"anchorB\":[0,0]}"));
        Assert.Throws<ArgumentException>(() => world.AddJoint(0, 0, 999, "{\"anchorA\":[0,0],\"anchorB\":[0,0],\"restLength\":1.0}"));
        Assert.Throws<ArgumentException>(() => world.AddJoint(0, ball, ball, "{\"anchorA\":[0,0],\"anchorB\":[0,0],\"restLength\":1.0}"));
        Assert.Throws<ArgumentException>(() => world.AddJoint(0, 0, ball, "{\"anchorA\":[0,0],\"anchorB\":[0,0],\"restLength\":-1.0}"));
        Assert.Equal(0, world.JointCount);
    }

    [Fact]
    public void F33_NativeJointRoundtrip()
    {
        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        int j = world.AddDistanceJoint(0, ball, new Vector2(0.0f, -1.0f), Vector2.Zero, 6.0f);
        Assert.Equal(0, j);
        int h = world.AddHingeJoint(0, ball, Vector2.Zero, Vector2.Zero, 1.0f, -1.0f);
        Assert.Equal(1, h);
        Assert.Equal(2, world.JointCount);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var (pos, _) = world.GetBodyTransform(ball);
        Assert.True(float.IsFinite(pos.X) && float.IsFinite(pos.Y));
        world.RemoveJoint(0);
        Assert.Equal(1, world.JointCount);
        world.RemoveJoint(0);
        Assert.Equal(0, world.JointCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.RemoveJoint(999));
    }

    private static float Wrap(float a)
    {
        a %= MathF.Tau;
        if (a > MathF.PI) a -= MathF.Tau;
        else if (a < -MathF.PI) a += MathF.Tau;
        return a;
    }
}
