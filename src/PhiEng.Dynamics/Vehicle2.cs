using System.Numerics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>One 2D wheel's mounting: where it sits and what it does.</summary>
public struct WheelMount2
{
    /// <summary>The wheel in the chassis body frame (x forward, y left).</summary>
    public Vector2 Offset;
    /// <summary>Whether the input steering yaws this wheel.</summary>
    public bool Steer;
    /// <summary>Whether the throttle/brake/handbrake drive this wheel.</summary>
    public bool Drive;
}

/// <summary>A 2D arcade car's tuning. Mirrors the Rust <c>VehicleConfig2</c> field for field.</summary>
public sealed class VehicleConfig2
{
    /// <summary>The most wheels one vehicle may carry.</summary>
    public const int MaxWheels = 8;

    /// <summary>The wheels, in order; only the first <see cref="WheelCount"/> are read.</summary>
    public WheelMount2[] Wheels = new WheelMount2[MaxWheels];
    /// <summary>How many wheels are mounted (clamped to <see cref="MaxWheels"/>).</summary>
    public int WheelCount;
    /// <summary>The total engine force (N), split across the driven wheels.</summary>
    public float EngineForce = 6000.0f;
    /// <summary>The total brake force (N), split across every wheel.</summary>
    public float BrakeForce = 9000.0f;
    /// <summary>The lateral tire stiffness (N·s/m): the grip that kills slide.</summary>
    public float GripLateral = 400.0f;
    /// <summary>The friction-circle coefficient.</summary>
    public float FrictionMu = 1.1f;
    /// <summary>The quadratic air drag (N·s²/m²) at the centre of mass.</summary>
    public float DragK = 0.7f;
    /// <summary>The rolling resistance (per unit normal load).</summary>
    public float RollingResist = 0.02f;
    /// <summary>The maximum steer angle (radians) at full steering input.</summary>
    public float MaxSteer = 0.5f;
    /// <summary>The wheel radius (m): the visual spin's divisor only.</summary>
    public float WheelRadius = 0.33f;
}

/// <summary>The 2D driver's hands, every step. Mirrors the Rust <c>VehicleInput2</c>.</summary>
public struct VehicleInput2
{
    /// <summary>Throttle, -1..1 (reverse included).</summary>
    public float Throttle;
    /// <summary>Brake, 0..1.</summary>
    public float Brake;
    /// <summary>Steering, -1..1 (positive turns right, clockwise).</summary>
    public float Steering;
    /// <summary>Handbrake, 0..1.</summary>
    public float Handbrake;
}

/// <summary>What one 2D wheel did this step. Mirrors the Rust <c>WheelTelemetry2</c>.</summary>
public struct WheelTelemetry2
{
    /// <summary>Always true top-down: there is no suspension to leave ground.</summary>
    public bool InContact;
    /// <summary>Always zero: no suspension, no travel.</summary>
    public float Compression01;
    /// <summary>The static normal load (N).</summary>
    public float NormalLoad;
    /// <summary>The longitudinal tire force (N, along the wheel forward).</summary>
    public float ForceLong;
    /// <summary>The lateral tire force (N, along the wheel left).</summary>
    public float ForceLat;
    /// <summary>The rolling spin (rad/s, visual only): v_long / R.</summary>
    public float Spin;
}

/// <summary>
/// Arcade cars in 2D: a top-down driver with tire grip and a handbrake.
/// Mirrors the Rust <c>vehicle</c> module operation for operation, so
/// the two engines agree bit-for-bit.
/// </summary>
public static class Vehicle2
{
    /// <summary>A smooth sign with a dead zone.</summary>
    private static float SmoothSign(float v) =>
        MathF.Max(-1.0f, MathF.Min(1.0f, v / 0.5f));

    /// <summary>Rotate (<paramref name="x"/>, <paramref name="y"/>) by <paramref name="angle"/> with the deterministic trigonometry.</summary>
    private static (float X, float Y) Rotate2(float x, float y, float angle)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(angle);
        return (x * c - y * s, x * s + y * c);
    }

    /// <summary>
    /// Run one arcade-vehicle step on a chassis. Call it before
    /// <c>World2.Step</c> with the same <paramref name="dt"/>.
    /// Returns the wheels processed (0 for an unknown chassis, no
    /// wheels, or a non-positive <paramref name="dt"/>).
    /// </summary>
    public static int StepVehicle2(
        World2 world, int chassis, VehicleConfig2 config, VehicleInput2 input,
        float dt, WheelTelemetry2[] telemetry)
    {
        if (!float.IsFinite(dt) || dt <= 0.0f) return 0;
        int count = config.WheelCount;
        if (count > VehicleConfig2.MaxWheels) count = VehicleConfig2.MaxWheels;
        if (count > telemetry.Length) count = telemetry.Length;
        if (count <= 0) return 0;
        if (chassis < 0 || chassis >= world.Bodies.Count) return 0;
        var snap = world.Bodies[chassis];
        if (snap.Id != chassis) return 0;

        float throttle = MathF.Max(-1.0f, MathF.Min(1.0f, input.Throttle));
        float brake = MathF.Max(0.0f, MathF.Min(1.0f, input.Brake));
        float steering = MathF.Max(-1.0f, MathF.Min(1.0f, input.Steering));
        float handbrake = MathF.Max(0.0f, MathF.Min(1.0f, input.Handbrake));

        float radius = MathF.Max(config.WheelRadius, 1e-3f);
        // Earth-normalized nominal load: the top-down tire has no
        // suspension to weigh anything with.
        float nominal = MathF.Max(snap.Mass * 9.81f / MathF.Max(count, 1), 1e-6f);
        int driven = 0;
        for (int i = 0; i < count; i++)
        {
            if (config.Wheels[i].Drive) driven++;
        }
        float engineEach = driven == 0 ? 0.0f : config.EngineForce / driven;
        float brakeEach = config.BrakeForce / count;

        Vector2 fwd0 = snap.Transform.RotateVector(Vector2.UnitX);
        var tireForce = new Vector2[VehicleConfig2.MaxWheels];
        var tirePoint = new Vector2[VehicleConfig2.MaxWheels];

        for (int i = 0; i < count; i++)
        {
            var mount = config.Wheels[i];
            Vector2 hard = snap.Transform.Position + snap.Transform.RotateVector(mount.Offset);
            // Positive steering turns right (clockwise on screen).
            float steerAngle = mount.Steer ? -steering * config.MaxSteer : 0.0f;
            var (fx, fy) = Rotate2(fwd0.X, fwd0.Y, steerAngle);
            var wheelFwd = new Vector2(fx, fy);
            var wheelLat = new Vector2(-wheelFwd.Y, wheelFwd.X);
            Vector2 r = hard - snap.Transform.Position;
            Vector2 pointVel = snap.Velocity + snap.AngularVelocity * new Vector2(-r.Y, r.X);
            float vLong = Vector2.Dot(pointVel, wheelFwd);
            float vLat = Vector2.Dot(pointVel, wheelLat);
            float forceLong = 0.0f;
            if (mount.Drive)
            {
                forceLong += throttle * engineEach;
                forceLong -= handbrake * config.BrakeForce * 0.75f * SmoothSign(vLong);
            }
            forceLong -= brake * brakeEach * SmoothSign(vLong);
            forceLong -= config.RollingResist * nominal * SmoothSign(vLong);
            float grip = config.GripLateral;
            if (!mount.Steer)
            {
                grip *= 1.0f - 0.9f * handbrake;
            }
            float forceLat = -grip * vLat;
            float cap = config.FrictionMu * nominal;
            float total = MathF.Sqrt(forceLong * forceLong + forceLat * forceLat);
            if (total > cap)
            {
                float s = cap / total;
                forceLong *= s;
                forceLat *= s;
            }
            tireForce[i] = wheelFwd * forceLong + wheelLat * forceLat;
            tirePoint[i] = hard;
            telemetry[i] = new WheelTelemetry2
            {
                InContact = true,
                Compression01 = 0.0f,
                NormalLoad = nominal,
                ForceLong = forceLong,
                ForceLat = forceLat,
                Spin = vLong / radius,
            };
        }

        var chassisBody = world.Bodies[chassis];
        for (int i = 0; i < count; i++)
        {
            chassisBody.ApplyForce(tireForce[i], tirePoint[i], dt);
        }

        Vector2 v = chassisBody.Velocity;
        float speed = v.Length();
        if (speed > 0.0f)
        {
            chassisBody.ApplyForce(v * (-config.DragK * speed), chassisBody.Transform.Position, dt);
        }
        return count;
    }
}
