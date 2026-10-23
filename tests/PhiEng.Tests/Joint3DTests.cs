using System.Numerics;
using System.Text;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Math;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// 3D joint tests (K block) + native 3D joint bridge (F37-F39).
/// Mirrors Rust joint3.rs tests (k31-k44) and ffi.rs (f37-f39).
/// Parity enforced by tools/check_test_parity.py.
/// </summary>
public class Joint3DTests
{
    private const float Dt = 1.0f / 60.0f;

    private static World3 TwoStatics()
    {
        var world = new World3();
        world.AddStaticBody(Vector3.Zero, new SphereShape3(0.5f));
        world.AddStaticBody(Vector3.UnitX, new SphereShape3(0.5f));
        return world;
    }

    private static World3 WorldWithBallOnRod()
    {
        var world = new World3();
        world.Gravity = new Vector3(0, -9.81f, 0);
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        world.AddDynamicBody(new Vector3(0, -2, 0), new SphereShape3(0.3f), 1.0f);
        return world;
    }

    private static World3 WorldWithHinge()
    {
        var world = new World3();
        world.Gravity = new Vector3(0, -9.81f, 0);
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        world.AddDynamicBody(new Vector3(2, 0, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        return world;
    }

    private static float TwistOf(World3 world, int a, int b, Vector3 axis)
    {
        Quaternion qa = world.Bodies[a].Transform.Rotation;
        Quaternion qb = world.Bodies[b].Transform.Rotation;
        return Joint3Math.TwistAbout(Joint3Math.RelativeQuat(qa, qb), axis);
    }

    private static float OrientationErr(World3 world, int a, int b)
    {
        Quaternion qa = world.Bodies[a].Transform.Rotation;
        Quaternion qb = world.Bodies[b].Transform.Rotation;
        return Joint3Math.OrientationError(Joint3Math.RelativeQuat(qa, qb)).Length();
    }

    [Fact]
    public void K31_Distance3RejectsBadInput()
    {
        var world = TwoStatics();
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeDistanceJoint3(
            0, 0, 99, Vector3.Zero, Vector3.UnitX, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeDistanceJoint3(
            0, 0, 0, Vector3.Zero, Vector3.UnitX, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeDistanceJoint3(
            0, 0, 1, new Vector3(float.NaN, 0, 0), Vector3.UnitX, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeDistanceJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, -1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeDistanceJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, float.PositiveInfinity, 0, 2));
    }

    [Fact]
    public void K32_Distance3HoldsRestLength()
    {
        var world = WorldWithBallOnRod();
        int ball = 1;
        world.AddJoint(Joint3.MakeDistance(new DistanceJoint3
        {
            BodyA = 0, BodyB = ball,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero, RestLength = 2,
        }));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[ball].Transform.Position;
        float dist = p.Length();
        Assert.True(MathF.Abs(dist - 2.0f) < 0.05f, $"rod must hold rest length, dist={dist}, pos={p}");
        Assert.True(MathF.Abs(p.X) < 0.1f && MathF.Abs(p.Z) < 0.1f, $"pendulum hangs straight down, pos={p}");
    }

    [Fact]
    public void K33_Distance3PullsBack()
    {
        var world = WorldWithBallOnRod();
        int ball = 1;
        world.AddJoint(Joint3.MakeDistance(new DistanceJoint3
        {
            BodyA = 0, BodyB = ball,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero, RestLength = 2,
        }));
        world.Bodies[ball].Transform = new Transform3(new Vector3(1, -1.5f, 0.5f), Quaternion.Identity);
        world.Bodies[ball].Velocity = Vector3.Zero;
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var p = world.Bodies[ball].Transform.Position;
        Assert.True(MathF.Abs(p.X) < 1.0f, $"displaced pendulum swings back, pos={p}");
        foreach (var b in world.Bodies)
        {
            Assert.True(float.IsFinite(b.Transform.Position.X));
            Assert.True(float.IsFinite(b.Velocity.X));
        }
    }

    [Fact]
    public void K34_Joint3RemovedWithBody()
    {
        var world = WorldWithBallOnRod();
        int ball = 1;
        world.AddJoint(Joint3.MakeDistance(new DistanceJoint3
        {
            BodyA = 0, BodyB = ball,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero, RestLength = 2,
        }));
        Assert.Single(world.Joints);
        world.RemoveBody(ball);
        Assert.Empty(world.Joints);
        for (int i = 0; i < 10; i++) world.Step(Dt);
    }

    [Fact]
    public void K35_Hinge3RejectsBadInput()
    {
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeHingeJoint3(
            0, 0, 99, Vector3.Zero, Vector3.UnitX, Vector3.UnitZ, Vector3.UnitZ, 0, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeHingeJoint3(
            0, 1, 1, Vector3.Zero, Vector3.UnitX, Vector3.UnitZ, Vector3.UnitZ, 0, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeHingeJoint3(
            0, 0, 1, new Vector3(float.NaN, 0, 0), Vector3.UnitX, Vector3.UnitZ, Vector3.UnitZ, 0, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeHingeJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, Vector3.Zero, Vector3.UnitZ, 0, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeHingeJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, Vector3.UnitZ, Vector3.UnitZ, float.NaN, 1, 0, 2));
    }

    [Fact]
    public void K36_Hinge3PullsToAnchor()
    {
        var world = WorldWithHinge();
        int bob = 1;
        world.AddJoint(Joint3.MakeHinge(new HingeJoint3
        {
            BodyA = 0, BodyB = bob,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitZ, AxisB = Vector3.UnitZ, LimitMin = 1, LimitMax = -1,
        }));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[bob].Transform.Position;
        Assert.True(p.Length() < 0.15f, $"hinge pulls bob to anchor, pos={p}");
        foreach (var b in world.Bodies)
        {
            Assert.True(float.IsFinite(b.Transform.Position.X));
            Assert.True(float.IsFinite(b.Velocity.X));
        }
    }

    [Fact]
    public void K37_Hinge3FreeSpin()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        world.AddStaticBody(new Vector3(0, -2, 0), new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        int bob = world.AddDynamicBody(Vector3.Zero, new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        world.Bodies[bob].LinearDamping = 0;
        world.Bodies[bob].AngularDamping = 0;
        world.AddJoint(Joint3.MakeHinge(new HingeJoint3
        {
            BodyA = 0, BodyB = bob,
            AnchorA = new Vector3(0, 2, 0), AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitZ, AxisB = Vector3.UnitZ, LimitMin = 1, LimitMax = -1,
        }));
        world.Bodies[bob].AngularVelocity = new Vector3(0, 0, 2);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        float tw = TwistOf(world, 0, bob, Vector3.UnitZ);
        Assert.True(MathF.Abs(tw - 2.0f) < 0.3f, $"free hinge must not brake twist, twist={tw}");
        Assert.True(world.Bodies[bob].Transform.Position.Length() < 0.05f, "center-pinned bob must not drift");
    }

    [Fact]
    public void K38_Hinge3TwistLimitsEnforced()
    {
        var world = WorldWithHinge();
        int bob = 1;
        world.AddJoint(Joint3.MakeHinge(new HingeJoint3
        {
            BodyA = 0, BodyB = bob,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitZ, AxisB = Vector3.UnitZ, LimitMin = -0.2f, LimitMax = 0.2f,
        }));
        world.Bodies[bob].AngularVelocity = new Vector3(0, 0, 10);
        for (int i = 0; i < 300; i++) world.Step(Dt);
        float tw = TwistOf(world, 0, bob, Vector3.UnitZ);
        Assert.True(tw >= -0.4f && tw <= 0.4f, $"hinge twist limits catch windup, twist={tw}");
        foreach (var b in world.Bodies)
        {
            Assert.True(float.IsFinite(b.Transform.Position.X));
            Assert.True(float.IsFinite(b.AngularVelocity.X));
        }
    }

    [Fact]
    public void K39_Prismatic3RejectsBadInput()
    {
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 99, Vector3.Zero, Vector3.UnitX, Vector3.UnitX, -1, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 1, 1, Vector3.Zero, Vector3.UnitX, Vector3.UnitX, -1, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 1, new Vector3(float.NaN, 0, 0), Vector3.UnitX, Vector3.UnitX, -1, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, Vector3.Zero, -1, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, Vector3.UnitX, float.NaN, 1, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, Vector3.UnitX, -1, 1, 0.5f, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakePrismaticJoint3(
            0, 0, 1, Vector3.Zero, Vector3.UnitX, new Vector3(1, float.NaN, 0), -1, 1, 0, 2));
    }

    [Fact]
    public void K40_Prismatic3SlidesHoldsRail()
    {
        var world = new World3();
        world.Gravity = new Vector3(0, -9.81f, 0);
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        int slider = world.AddDynamicBody(Vector3.Zero, new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        world.AddJoint(Joint3.MakePrismatic(new PrismaticJoint3
        {
            BodyA = 0, BodyB = slider,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitX, LimitMin = 1, LimitMax = -1,
        }));
        world.Bodies[slider].Velocity = new Vector3(5, 0, 0);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var p = world.Bodies[slider].Transform.Position;
        Assert.True(p.X > 1.0f, $"slider glides along rail, pos={p}");
        Assert.True(MathF.Abs(p.Y) < 0.15f, $"rail holds height, pos={p}");
        Assert.True(MathF.Abs(p.Z) < 0.15f, $"rail holds lateral Z, pos={p}");
    }

    [Fact]
    public void K41_Prismatic3LimitsEnforced()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        int slider = world.AddDynamicBody(new Vector3(3, 0, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        world.AddJoint(Joint3.MakePrismatic(new PrismaticJoint3
        {
            BodyA = 0, BodyB = slider,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitX, LimitMin = -1, LimitMax = 1,
        }));
        for (int i = 0; i < 300; i++) world.Step(Dt);
        var p = world.Bodies[slider].Transform.Position;
        Assert.True(p.X < 1.2f, $"travel limit pulls slider back, pos={p}");
        Assert.True(MathF.Abs(p.Y) < 0.15f && MathF.Abs(p.Z) < 0.15f, $"rail holds perpendicular plane, pos={p}");
    }

    [Fact]
    public void K42_Fixed3RejectsBadInput()
    {
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeFixedJoint3(
            0, 0, 99, Vector3.Zero, Vector3.UnitX, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeFixedJoint3(
            0, 1, 1, Vector3.Zero, Vector3.UnitX, 0, 2));
        Assert.Throws<ArgumentException>(() => Joint3Math.MakeFixedJoint3(
            0, 0, 1, new Vector3(0, float.NaN, 0), Vector3.UnitX, 0, 2));
    }

    [Fact]
    public void K43_Fixed3HoldsFormation()
    {
        var world = new World3();
        world.Gravity = new Vector3(0, -9.81f, 0);
        int a = world.AddDynamicBody(new Vector3(-0.5f, 5, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        int b = world.AddDynamicBody(new Vector3(0.5f, 5, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        world.AddJoint(Joint3.MakeFixed(new FixedJoint3
        {
            BodyA = a, BodyB = b,
            AnchorA = new Vector3(1, 0, 0), AnchorB = Vector3.Zero,
        }));
        for (int i = 0; i < 180; i++) world.Step(Dt);
        float gap = (world.Bodies[b].Transform.Position - world.Bodies[a].Transform.Position).Length();
        Assert.True(MathF.Abs(gap - 1.0f) < 0.15f, $"weld holds 1-unit formation, gap={gap}");
        foreach (var body in world.Bodies)
        {
            Assert.True(float.IsFinite(body.Transform.Position.X));
            Assert.True(float.IsFinite(body.AngularVelocity.X));
        }
    }

    [Fact]
    public void K44_Fixed3ResistsRotation()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        int a = world.AddDynamicBody(new Vector3(-0.5f, 0, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        int b = world.AddDynamicBody(new Vector3(0.5f, 0, 0), new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        world.Bodies[a].LinearDamping = 0;
        world.Bodies[a].AngularDamping = 0;
        world.Bodies[b].LinearDamping = 0;
        world.Bodies[b].AngularDamping = 0;
        world.AddJoint(Joint3.MakeFixed(new FixedJoint3
        {
            BodyA = a, BodyB = b,
            AnchorA = new Vector3(1, 0, 0), AnchorB = Vector3.Zero,
        }));
        world.Bodies[a].AngularVelocity = new Vector3(0, 0, 5);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        float err = OrientationErr(world, a, b);
        Assert.True(err < 0.4f, $"weld keeps bodies co-rotating, err={err}");
    }

    // ---- F37-F39: native 3D joint bridge ----

    private static PhiNativeWorld3 BallDrop3()
    {
        var world = new PhiNativeWorld3();
        int floor = world.AddBoxBody(new Vector3(0, -1, 0), new Vector3(10, 0.5f, 10), 1.0f, true, 0.5f, 0);
        Assert.Equal(0, floor);
        int ball = world.AddSphereBody(new Vector3(0, 5, 0), 0.5f, 1.0f, false, 0.5f, 0);
        Assert.Equal(1, ball);
        return world;
    }

    [Fact]
    public unsafe void F37_Joint3NullSafety()
    {
        // Null world and null/empty/oversize blobs are clean -1.
        Assert.Equal(-1, NativeMethods.phieng_world3_add_joint(
            IntPtr.Zero, NativeMethods.JOINT3_DISTANCE, 0, 1, (byte*)null, 1));
        using var world = BallDrop3();
        Assert.Equal(-1, world.AddJoint(0, 0, 1, ""));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, new string('x', 64 * 1024 + 1)));
        Assert.Equal(0, world.JointCount);
    }

    [Fact]
    public unsafe void F38_Joint3RejectsBadInput()
    {
        using var world = BallDrop3();
        Assert.Equal(-1, world.AddJoint(99, 0, 1, "{\"anchorA\":[0,0,0]}"));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, "{nope"));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0]}"));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, "{\"anchorA\":[0,0],\"anchorB\":[0,0,0],\"restLength\":1.0}"));
        Assert.Equal(-1, world.AddJoint(0, 0, 999, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0],\"restLength\":1.0}"));
        Assert.Equal(-1, world.AddJoint(0, 1, 1, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0],\"restLength\":1.0}"));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0],\"restLength\":-1.0}"));
        Assert.Equal(-1, world.AddJoint(0, 0, 1, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0],\"restLength\":1.0,\"compliance\":0.5}"));
        Assert.Equal(-1, world.AddJoint((int)NativeMethods.JOINT3_HINGE, 0, 1, "{\"anchorA\":[0,0,0],\"anchorB\":[0,0,0],\"axisA\":[0,0,0],\"axisB\":[0,0,1],\"limitMin\":1.0,\"limitMax\":-1.0}"));
        Assert.Equal(0, world.JointCount);
    }

[Fact]
    public unsafe void F39_Joint3RoundtripRemoval()
    {
        using var world = BallDrop3();
        int j0 = world.AddJoint((int)NativeMethods.JOINT3_DISTANCE, 0, 1,
            "{\"anchorA\":[0.0,-1.0,0.0],\"anchorB\":[0.0,0.0,0.0],\"restLength\":6.0,\"compliance\":0.0}");
        Assert.Equal(0, j0);
        Assert.Equal(1, world.JointCount);
        int h = world.AddJoint((int)NativeMethods.JOINT3_HINGE, 0, 1,
            "{\"anchorA\":[0.0,0.0,0.0],\"anchorB\":[0.0,0.0,0.0],\"axisA\":[0.0,0.0,1.0],\"axisB\":[0.0,0.0,1.0],\"limitMin\":1.0,\"limitMax\":-1.0}");
        Assert.Equal(1, h);
        int p = world.AddJoint((int)NativeMethods.JOINT3_PRISMATIC, 0, 1,
            "{\"anchorA\":[0.0,0.0,0.0],\"anchorB\":[0.0,0.0,0.0],\"axis\":[1.0,0.0,0.0],\"limitMin\":-1.0,\"limitMax\":1.0}");
        Assert.Equal(2, p);
        int f = world.AddJoint((int)NativeMethods.JOINT3_FIXED, 0, 1,
            "{\"anchorA\":[0.0,0.0,0.0],\"anchorB\":[0.0,0.0,0.0]}");
        Assert.Equal(3, f);
        Assert.Equal(4, world.JointCount);

        for (int i = 0; i < 120; i++) world.Step(Dt);

        var (pos, _) = world.GetBodyTransform(1);
        Assert.True(float.IsFinite(pos.X) && float.IsFinite(pos.Y) && float.IsFinite(pos.Z));

        world.RemoveJoint(1);
        Assert.Equal(3, world.JointCount);
        world.RemoveJoint(1);
        Assert.Equal(2, world.JointCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => world.RemoveJoint(999));
        world.RemoveBody(1);
        Assert.Equal(0, world.JointCount);
    }

    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    private static uint JsonSchema(string text)
    {
        const string marker = "\"schema\": ";
        int at = text.IndexOf(marker, StringComparison.Ordinal);
        if (at < 0) throw new InvalidOperationException("the JSON carries no schema");
        string rest = text[(at + marker.Length)..];
        int end = 0;
        while (end < rest.Length && char.IsAsciiDigit(rest[end])) end++;
        return uint.Parse(rest[..end]);
    }

    private static (World3 World, int Arm) MotorPendulum3(float speed, float torque, float gravityY)
    {
        var world = new World3();
        world.Gravity = new Vector3(0.0f, gravityY, 0.0f);
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        world.AddStaticBody(Vector3.Zero, new SphereShape3(0.2f));
        int arm = world.AddDynamicBody(new Vector3(0.0f, -1.0f, 0.0f), new BoxShape3(new Vector3(0.1f, 1.0f, 0.1f)), 1.0f);
        var joint = Joint3.MakeHinge(new HingeJoint3
        {
            BodyA = 0, BodyB = arm,
            AnchorA = Vector3.Zero, AnchorB = new Vector3(0.0f, 1.0f, 0.0f),
            AxisA = Vector3.UnitY, AxisB = Vector3.UnitY,
            LimitMin = 1.0f, LimitMax = -1.0f,
        });
        joint.SetHingeMotor(speed, torque);
        world.AddJoint(joint);
        return (world, arm);
    }

    /// <summary>
    /// K45: a budgeted twist motor spins its arm up to the target rate.
    /// </summary>
    [Fact]
    public void K45_TwistMotorReachesTargetRate()
    {
        var (world, arm) = MotorPendulum3(2.0f, 1000.0f, 0.0f);
        for (int i = 0; i < 300; i++) world.Step(Dt);
        float w = world.Bodies[arm].AngularVelocity.Y;
        Assert.Equal(0x3ffff514u, Bits(w));
        Assert.True(MathF.Abs(w - 2.0f) < 1e-2f, $"motor must reach 2 rad/s, got {w}");
    }

    /// <summary>
    /// K46: a weak lift motor slips downward with its impulse saturated
    /// at the budget, and bad tuning is rejected.
    /// </summary>
    [Fact]
    public void K46_WeakLiftMotorSlipsWithSaturatedImpulse()
    {
        var world = new World3();
        world.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(0.2f, 5.0f, 0.2f)));
        world.Bodies[0].IsSensor = true;
        int slider = world.AddDynamicBody(Vector3.Zero, new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        var joint = Joint3.MakePrismatic(new PrismaticJoint3
        {
            BodyA = 0, BodyB = slider,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitY, LimitMin = -10.0f, LimitMax = 10.0f,
        });
        joint.SetPrismaticMotor(2.0f, 0.5f);
        world.AddJoint(joint);
        Assert.Throws<ArgumentException>(() => world.Joints[0].SetPrismaticMotor(float.NaN, 1.0f));
        Assert.Throws<ArgumentException>(() => world.Joints[0].SetPrismaticMotor(1.0f, -1.0f));
        for (int i = 0; i < 600; i++) world.Step(Dt);
        float y = world.Bodies[slider].Transform.Position.Y;
        float imp = world.Joints[0].Prismatic!.Value.MotorImpulse;
        Assert.Equal(0xc1200002u, Bits(y));
        Assert.Equal(0x3c088889u, Bits(imp));
        Assert.True(y < -5.0f, $"starved lift must sink to its stop, got {y}");
        Assert.True(MathF.Abs(imp - 0.5f * Dt) < 1e-9f, $"impulse must sit on the budget {imp}");
    }

    /// <summary>
    /// K47: a prismatic motor extends its slider at the target speed.
    /// </summary>
    [Fact]
    public void K47_PrismaticMotorExtendsAtTargetSpeed()
    {
        var world = new World3();
        world.Gravity = Vector3.Zero;
        world.LinearSleepThreshold = 0.0f;
        world.AngularSleepThreshold = 0.0f;
        world.AddStaticBody(Vector3.Zero, new BoxShape3(new Vector3(5.0f, 0.2f, 0.2f)));
        int slider = world.AddDynamicBody(Vector3.Zero, new BoxShape3(new Vector3(0.3f, 0.3f, 0.3f)), 1.0f);
        var joint = Joint3.MakePrismatic(new PrismaticJoint3
        {
            BodyA = 0, BodyB = slider,
            AnchorA = Vector3.Zero, AnchorB = Vector3.Zero,
            AxisA = Vector3.UnitX, LimitMin = -10.0f, LimitMax = 10.0f,
        });
        joint.SetPrismaticMotor(2.0f, 10000.0f);
        world.AddJoint(joint);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        var p = world.Bodies[slider].Transform.Position;
        Assert.Equal(0x407ff522u, Bits(p.X));
        Assert.True(MathF.Abs(p.X - 4.0f) < 0.05f, $"slider must extend at 2 m/s, got {p}");
        Assert.True(MathF.Abs(p.Y) < 0.05f && MathF.Abs(p.Z) < 0.05f, $"slider must stay on the rail, got {p}");
    }

    /// <summary>
    /// K48: motors round-trip through save/load (schema v16), and an
    /// unmotorized world stays on its old schema with motors off.
    /// </summary>
    [Fact]
    public void K48_MotorsPersistThroughSaveLoad()
    {
        var (world, arm) = MotorPendulum3(2.0f, 1000.0f, 0.0f);
        for (int i = 0; i < 60; i++) world.Step(Dt);
        string text = Snapshot.World3ToJson(world, SnapshotLoadMode.Exact);
        Assert.Equal(Snapshot.SchemaVersion, JsonSchema(text));
        Assert.Contains("motor_speed", text);
        byte[] bytes = Snapshot.World3ToBinary(world, SnapshotLoadMode.Exact);
        var back = Snapshot.World3FromBinary(bytes, SnapshotLoadMode.Exact);
        var h = back.Joints[0].Hinge!.Value;
        Assert.Equal(Bits(2.0f), Bits(h.MotorSpeed));
        Assert.Equal(Bits(1000.0f), Bits(h.MotorMaxTorque));
        for (int i = 0; i < 60; i++)
        {
            world.Step(Dt);
            back.Step(Dt);
        }
        var (a, b) = (world.Bodies[arm], back.Bodies[arm]);
        Assert.Equal(
            (Bits(a.Velocity.X), Bits(a.Velocity.Y), Bits(a.Velocity.Z), Bits(a.AngularVelocity.Y)),
            (Bits(b.Velocity.X), Bits(b.Velocity.Y), Bits(b.Velocity.Z), Bits(b.AngularVelocity.Y)));

        var (plain, _) = MotorPendulum3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 60; i++) plain.Step(Dt);
        string text2 = Snapshot.World3ToJson(plain, SnapshotLoadMode.Exact);
        Assert.True(JsonSchema(text2) < Snapshot.SchemaVersion, "no motors means no v16");
        Assert.DoesNotContain("motor_speed", text2);
        byte[] bytes2 = Snapshot.World3ToBinary(plain, SnapshotLoadMode.Exact);
        var back2 = Snapshot.World3FromBinary(bytes2, SnapshotLoadMode.Exact);
        var h2 = back2.Joints[0].Hinge!.Value;
        Assert.Equal(0.0f, h2.MotorSpeed);
        Assert.Equal(0.0f, h2.MotorMaxTorque);
        Assert.Equal(0.0f, h2.MotorImpulse);
    }
}
