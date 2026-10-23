using Xunit;
using System.Numerics;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Suspension and arcade vehicles (I01–I08). Mirrors
/// <c>crates/engine/tests/vehicle.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class VehicleTests
{
    private const float Dt = 1.0f / 60.0f;

    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <summary>A 1200 kg 3D car over a long plate, sleeping disabled.</summary>
    private static (World3 World, int Chassis, VehicleConfig3 Config) Car3()
    {
        var w = new World3();
        w.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        w.CollectContactEvents = true;
        w.AddStaticBody(new Vector3(0.0f, -0.5f, 0.0f), new BoxShape3(new Vector3(500.0f, 0.5f, 500.0f)));
        int chassis = w.AddDynamicBody(
            new Vector3(0.0f, 1.0f, 0.0f), new BoxShape3(new Vector3(0.9f, 0.35f, 2.0f)), 240.0f);
        w.Bodies[0].Material.Restitution = 0.0f;
        w.Bodies[chassis].Material.Restitution = 0.0f;
        var cfg = new VehicleConfig3();
        (float X, float Z, bool Steer)[] offs =
        {
            (0.8f, 1.2f, true), (-0.8f, 1.2f, true),
            (0.8f, -1.2f, false), (-0.8f, -1.2f, false),
        };
        for (int k = 0; k < 4; k++)
        {
            cfg.Wheels[k] = new WheelMount3
            {
                Offset = new Vector3(offs[k].X, -0.35f, offs[k].Z),
                Steer = offs[k].Steer,
                Drive = true,
            };
        }
        cfg.WheelCount = 4;
        return (w, chassis, cfg);
    }

    /// <summary>A 1200 kg 2D car in a gravity-free top-down world.</summary>
    private static (World2 World, int Chassis, VehicleConfig2 Config) Car2()
    {
        var w = new World2();
        w.Gravity = Vector2.Zero;
        w.LinearSleepThreshold = 0.0f;
        w.AngularSleepThreshold = 0.0f;
        int chassis = w.AddDynamicBody(
            Vector2.Zero, new BoxShape2(new Vector2(1.0f, 0.5f)), 600.0f);
        var cfg = new VehicleConfig2();
        (float X, float Y, bool Steer, bool Drive)[] offs =
        {
            (1.2f, 0.5f, true, false), (1.2f, -0.5f, true, false),
            (-1.2f, 0.5f, false, true), (-1.2f, -0.5f, false, true),
        };
        for (int k = 0; k < 4; k++)
        {
            cfg.Wheels[k] = new WheelMount2
            {
                Offset = new Vector2(offs[k].X, offs[k].Y),
                Steer = offs[k].Steer,
                Drive = offs[k].Drive,
            };
        }
        cfg.WheelCount = 4;
        return (w, chassis, cfg);
    }

    /// <summary>
    /// I01: the suspension settles at the analytic static compression.
    /// </summary>
    [Fact]
    public void I01_TheSuspensionSettlesAtItsStaticCompression()
    {
        var (w, chassis, cfg) = Car3();
        var input = new VehicleInput3();
        var tele = new WheelTelemetry3[8];
        for (int s = 0; s < 600; s++)
        {
            Assert.Equal(4, Vehicle3.StepVehicle3(w, chassis, cfg, input, Dt, tele));
            w.Step(Dt);
        }
        float m = w.Bodies[chassis].Mass;
        float x0 = m * 9.81f / (4.0f * cfg.SpringK * cfg.RestLength);
        Assert.Equal(0x3ec541a8u, Bits(tele[0].Compression01));
        Assert.True(MathF.Abs(tele[0].Compression01 - x0) / x0 < 1e-3f);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(tele[i].InContact);
            Assert.True(MathF.Abs(tele[i].Compression01 - x0) / x0 < 1e-3f);
        }
    }

    /// <summary>
    /// I02: full throttle drives the car straight up to speed.
    /// </summary>
    [Fact]
    public void I02_FullThrottleDrivesStraight()
    {
        var (w, chassis, cfg) = Car3();
        var input = new VehicleInput3 { Throttle = 1.0f };
        var tele = new WheelTelemetry3[8];
        for (int s = 0; s < 600; s++)
        {
            Vehicle3.StepVehicle3(w, chassis, cfg, input, Dt, tele);
            w.Step(Dt);
        }
        var v = w.Bodies[chassis].Velocity;
        var p = w.Bodies[chassis].Transform.Position;
        Assert.Equal(0x422e9cddu, Bits(v.Length()));
        Assert.True(v.Length() > 30.0f, "full throttle must reach speed");
        Assert.True(MathF.Abs(v.Z - v.Length()) < 1e-3f);
        Assert.True(MathF.Abs(p.X) < 0.5f, "no steering means no sideways drift");
        Assert.True(p.Y > 0.5f, "the car stays on its wheels");
    }

    /// <summary>
    /// I03: steering turns the car right, toward +X.
    /// </summary>
    [Fact]
    public void I03_SteeringTurnsRight()
    {
        var (w, chassis, cfg) = Car3();
        var input = new VehicleInput3 { Throttle = 0.6f, Steering = 1.0f };
        var tele = new WheelTelemetry3[8];
        for (int s = 0; s < 120; s++)
        {
            Vehicle3.StepVehicle3(w, chassis, cfg, input, Dt, tele);
            w.Step(Dt);
        }
        var fwd = PhiEng.Math.DeterministicRotation.RotateVector(
            w.Bodies[chassis].Transform.Rotation, Vector3.UnitZ);
        float yaw = w.Bodies[chassis].AngularVelocity.Y;
        Assert.Equal(0x3f746fcbu, Bits(fwd.X));
        Assert.Equal(0x3f8b0642u, Bits(yaw));
        Assert.True(fwd.X > 0.5f, "full lock must turn hard right");
        Assert.True(yaw > 0.5f, "still yawing right");
    }

    /// <summary>
    /// I04: the brake stops the car.
    /// </summary>
    [Fact]
    public void I04_TheBrakeStopsTheCar()
    {
        var (w, chassis, cfg) = Car3();
        var go = new VehicleInput3 { Throttle = 1.0f };
        var tele = new WheelTelemetry3[8];
        for (int s = 0; s < 400; s++)
        {
            Vehicle3.StepVehicle3(w, chassis, cfg, go, Dt, tele);
            w.Step(Dt);
        }
        Assert.True(w.Bodies[chassis].Velocity.Length() > 10.0f);
        // Half brake: full lock dives the nose onto the
        // ground, and the test is about braking, not belly grazes.
        var stop = new VehicleInput3 { Brake = 0.5f };
        int evTotal = 0;
        for (int s = 0; s < 400; s++)
        {
            Vehicle3.StepVehicle3(w, chassis, cfg, stop, Dt, tele);
            w.Step(Dt);
            evTotal += w.ContactEvents.Count;
        }
        float v1 = w.Bodies[chassis].Velocity.Length();
        Assert.Equal(0x3d987e50u, Bits(v1));
        // Contact-free: a belly graze would hand the stop to the
        // contact solver, which resolves sub-penetration touches
        // last-bit differently across mirrors (documented follow-up).
        Assert.Equal(0, evTotal);
        Assert.True(v1 < 0.6f, $"the brake must (nearly) stop the car, got {v1}");
    }

    /// <summary>
    /// I05: airborne wheels push nothing and report the miss.
    /// </summary>
    [Fact]
    public void I05_AirborneWheelsPushNothing()
    {
        var (w, chassis, cfg) = Car3();
        var t = w.Bodies[chassis].Transform;
        t.Position = new Vector3(t.Position.X, 50.0f, t.Position.Z);
        w.Bodies[chassis].Transform = t;
        var input = new VehicleInput3 { Throttle = 1.0f };
        var tele = new WheelTelemetry3[8];
        var before = w.Bodies[chassis].Velocity;
        Assert.Equal(4, Vehicle3.StepVehicle3(w, chassis, cfg, input, Dt, tele));
        Assert.Equal(before, w.Bodies[chassis].Velocity);
        for (int i = 0; i < 4; i++)
        {
            Assert.False(tele[i].InContact);
            Assert.Equal(0.0f, tele[i].Compression01);
            Assert.Equal(0.0f, tele[i].NormalLoad);
            Assert.Equal(0.0f, tele[i].ForceLong);
            Assert.Equal(0.0f, tele[i].ForceLat);
        }
    }

    /// <summary>
    /// I06: the 2D arcade car drives along its heading.
    /// </summary>
    [Fact]
    public void I06_TheArcadeCarDrives()
    {
        var (w, chassis, cfg) = Car2();
        var input = new VehicleInput2 { Throttle = 1.0f };
        var tele = new WheelTelemetry2[8];
        for (int s = 0; s < 600; s++)
        {
            Assert.Equal(4, Vehicle2.StepVehicle2(w, chassis, cfg, input, Dt, tele));
            w.Step(Dt);
        }
        var v = w.Bodies[chassis].Velocity;
        Assert.Equal(0x42282963u, Bits(v.Length()));
        Assert.True(v.Length() > 30.0f);
        Assert.True(MathF.Abs(v.X - v.Length()) < 1e-3f, "straight along +X");
    }

    /// <summary>
    /// I07: lateral grip kills a slide; the handbrake lets it live
    /// while stopping the car.
    /// </summary>
    [Fact]
    public void I07_GripKillsSlidesAndTheHandbrakeStealsGrip()
    {
        (float Lat, float Speed) Shove(float handbrake)
        {
            var (w, chassis, cfg) = Car2();
            w.Bodies[chassis].Velocity = new Vector2(6.0f, 4.0f);
            var input = new VehicleInput2 { Handbrake = handbrake };
            var tele = new WheelTelemetry2[8];
            for (int s = 0; s < 120; s++)
            {
                Vehicle2.StepVehicle2(w, chassis, cfg, input, Dt, tele);
                w.Step(Dt);
            }
            var b = w.Bodies[chassis];
            var fwd = b.Transform.RotateVector(Vector2.UnitX);
            var lat = new Vector2(-fwd.Y, fwd.X);
            return (Vector2.Dot(b.Velocity, lat), b.Velocity.Length());
        }
        var (latPlain, speedPlain) = Shove(0.0f);
        var (latHb, speedHb) = Shove(1.0f);
        Assert.Equal(0x3e866a1cu, Bits(latPlain));
        Assert.Equal(0x3f878691u, Bits(latHb));
        Assert.True(MathF.Abs(latPlain) < 0.5f, "grip must kill the slide");
        Assert.True(
            MathF.Abs(latHb) > MathF.Abs(latPlain),
            "no rear grip means more slide");
        Assert.True(speedHb < speedPlain, "the handbrake must slow the car");
    }

    /// <summary>
    /// I08: a drop from height bottoms the bump stops and recovers.
    /// </summary>
    [Fact]
    public void I08_BottomingCapsTheLanding()
    {
        var (w, chassis, cfg) = Car3();
        var t = w.Bodies[chassis].Transform;
        t.Position = new Vector3(t.Position.X, 5.0f, t.Position.Z);
        w.Bodies[chassis].Transform = t;
        var input = new VehicleInput3();
        var tele = new WheelTelemetry3[8];
        float maxComp = 0.0f;
        float maxForce = 0.0f;
        int evTotal = 0;
        for (int s = 0; s < 600; s++)
        {
            Vehicle3.StepVehicle3(w, chassis, cfg, input, Dt, tele);
            w.Step(Dt);
            evTotal += w.ContactEvents.Count;
            for (int i = 0; i < 4; i++)
            {
                maxComp = MathF.Max(maxComp, tele[i].Compression01);
                maxForce = MathF.Max(maxForce, tele[i].NormalLoad);
            }
        }
        float y = w.Bodies[chassis].Transform.Position.Y;
        Assert.Equal(0x3f652900u, Bits(y));
        Assert.Equal(0x46bc8629u, Bits(maxForce));
        Assert.Equal(1.0f, maxComp);
        // Contact-free, like I04: the pins cover suspension only.
        Assert.Equal(0, evTotal);
        Assert.True(
            maxForce <= cfg.MaxSuspensionForce,
            "the bump stop caps the force");
        Assert.True(y > 0.5f && y < 1.5f, "recovered, not bounced to orbit");
    }

    /// <summary>
    /// The 3D vehicle step crosses the native bridge with the same
    /// bits: settling 600 steps natively pins the static compression.
    /// </summary>
    [Fact]
    public void VehicleStep_NativeBridge3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
        w.AddStaticBody(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(500.0f, 0.5f, 500.0f));
        int chassis = w.AddBoxBody(
            new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.9f, 0.35f, 2.0f), 240.0f, isStatic: false);
        var cfg = new VehicleConfig3();
        (float X, float Z, bool Steer)[] offs =
        {
            (0.8f, 1.2f, true), (-0.8f, 1.2f, true),
            (0.8f, -1.2f, false), (-0.8f, -1.2f, false),
        };
        for (int k = 0; k < 4; k++)
        {
            cfg.Wheels[k] = new WheelMount3
            {
                Offset = new Vector3(offs[k].X, -0.35f, offs[k].Z),
                Steer = offs[k].Steer,
                Drive = true,
            };
        }
        cfg.WheelCount = 4;
        var input = new VehicleInput3();
        var tele = new WheelTelemetry3[8];
        for (int s = 0; s < 600; s++)
        {
            Assert.Equal(4, w.StepVehicle3(chassis, cfg, input, Dt, tele));
            w.Step(Dt);
        }
        Assert.Equal(0x3ec541a8u, Bits(tele[0].Compression01));
    }

    /// <summary>
    /// The 2D vehicle step crosses the native bridge with the same
    /// bits: full throttle pins the top speed.
    /// </summary>
    [Fact]
    public void VehicleStep_NativeBridge2D()
    {
        using var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, 0.0f));
        int chassis = w.AddBoxBody(Vector2.Zero, new Vector2(1.0f, 0.5f), 600.0f, isStatic: false);
        var cfg = new VehicleConfig2();
        (float X, float Y, bool Steer, bool Drive)[] offs =
        {
            (1.2f, 0.5f, true, false), (1.2f, -0.5f, true, false),
            (-1.2f, 0.5f, false, true), (-1.2f, -0.5f, false, true),
        };
        for (int k = 0; k < 4; k++)
        {
            cfg.Wheels[k] = new WheelMount2
            {
                Offset = new Vector2(offs[k].X, offs[k].Y),
                Steer = offs[k].Steer,
                Drive = offs[k].Drive,
            };
        }
        cfg.WheelCount = 4;
        var input = new VehicleInput2 { Throttle = 1.0f };
        var tele = new WheelTelemetry2[8];
        for (int s = 0; s < 600; s++)
        {
            Assert.Equal(4, w.StepVehicle2(chassis, cfg, input, Dt, tele));
            w.Step(Dt);
        }
        Assert.True(tele[0].InContact);
        Assert.Equal(0.0f, tele[0].Compression01);
        Assert.Equal(0x42282963u, Bits(w.GetBodyVelocity(chassis).Length()));
    }
}
