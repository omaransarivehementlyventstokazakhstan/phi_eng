using System.Numerics;
using PhiEng.Math;
using PhiEng.Collision;

namespace PhiEng.Dynamics;

/// <summary>One 3D wheel's mounting: where it hangs and what it does.</summary>
public struct WheelMount3
{
    /// <summary>The hardpoint in the chassis body frame (x right, y up, z forward).</summary>
    public Vector3 Offset;
    /// <summary>Whether the input steering yaws this wheel.</summary>
    public bool Steer;
    /// <summary>Whether the throttle/brake/handbrake drive this wheel.</summary>
    public bool Drive;
}

/// <summary>A 3D suspension vehicle's tuning. Mirrors the Rust <c>VehicleConfig3</c> field for field.</summary>
public sealed class VehicleConfig3
{
    /// <summary>The most wheels one vehicle may carry.</summary>
    public const int MaxWheels = 8;

    /// <summary>The wheels, in order; only the first <see cref="WheelCount"/> are read.</summary>
    public WheelMount3[] Wheels = new WheelMount3[MaxWheels];
    /// <summary>How many wheels are mounted (clamped to <see cref="MaxWheels"/>).</summary>
    public int WheelCount;
    /// <summary>The suspension rest length (axle below the hardpoint, metres).</summary>
    public float RestLength = 0.35f;
    /// <summary>The suspension spring rate (N/m).</summary>
    public float SpringK = 22000.0f;
    /// <summary>The suspension damping rate (N·s/m).</summary>
    public float DampingC = 2600.0f;
    /// <summary>The suspension force cap (N): the bottoming stop.</summary>
    public float MaxSuspensionForce = 30000.0f;
    /// <summary>The wheel radius (m).</summary>
    public float WheelRadius = 0.33f;
    /// <summary>The total engine force (N), split across the driven wheels.</summary>
    public float EngineForce = 9000.0f;
    /// <summary>The total brake force (N), split across every wheel.</summary>
    public float BrakeForce = 14000.0f;
    /// <summary>The lateral tire stiffness (N·s/m per unit load).</summary>
    public float GripLateral = 600.0f;
    /// <summary>The friction-circle coefficient.</summary>
    public float FrictionMu = 1.1f;
    /// <summary>The quadratic air drag (N·s²/m²) at the centre of mass.</summary>
    public float DragK = 0.9f;
    /// <summary>The rolling resistance (per unit normal load).</summary>
    public float RollingResist = 0.02f;
    /// <summary>The maximum steer angle (radians) at full steering input.</summary>
    public float MaxSteer = 0.55f;
}

/// <summary>The 3D driver's hands, every step. Mirrors the Rust <c>VehicleInput3</c>.</summary>
public struct VehicleInput3
{
    /// <summary>Throttle, -1..1 (reverse included).</summary>
    public float Throttle;
    /// <summary>Brake, 0..1.</summary>
    public float Brake;
    /// <summary>Steering, -1..1 (positive turns right, toward +X).</summary>
    public float Steering;
    /// <summary>Handbrake, 0..1.</summary>
    public float Handbrake;
}

/// <summary>What one 3D wheel did this step. Mirrors the Rust <c>WheelTelemetry3</c>.</summary>
public struct WheelTelemetry3
{
    /// <summary>Whether the suspension ray touched ground.</summary>
    public bool InContact;
    /// <summary>The suspension compression, 0 at full droop, 1 bottomed.</summary>
    public float Compression01;
    /// <summary>The suspension force (N): the tire's normal load.</summary>
    public float NormalLoad;
    /// <summary>The longitudinal tire force (N, along the wheel forward).</summary>
    public float ForceLong;
    /// <summary>The lateral tire force (N, along the wheel lateral).</summary>
    public float ForceLat;
    /// <summary>The rolling spin (rad/s, visual only): v_long / R.</summary>
    public float Spin;
}

/// <summary>
/// Raycast suspension vehicles in 3D: an arcade/sim-cade car.
/// Mirrors the Rust <c>vehicle3</c> module operation for operation, so
/// the two engines agree bit-for-bit.
/// </summary>
public static class Vehicle3
{
    /// <summary>A smooth sign with a dead zone.</summary>
    private static float SmoothSign(float v) =>
        MathF.Max(-1.0f, MathF.Min(1.0f, v / 0.5f));

    /// <summary>Rotate <paramref name="v"/> around the unit axis <paramref name="up"/> by <paramref name="angle"/> (Rodrigues, deterministic trigonometry).</summary>
    private static Vector3 RotateAroundUp(Vector3 v, Vector3 up, float angle)
    {
        var (c, s) = DeterministicTrig.ComputeCosSin(angle);
        return v * c + Vector3.Cross(up, v) * s + up * (Vector3.Dot(up, v) * (1.0f - c));
    }

    /// <summary>The rigid-body velocity at a world point.</summary>
    private static Vector3 VelocityAtPoint(Vector3 linear, Vector3 angular, Vector3 centre, Vector3 point) =>
        linear + Vector3.Cross(angular, point - centre);

    /// <summary>A suspension ray that refuses to see the chassis: the world's raycast loop, minus one body.</summary>
    private static (uint Body, float T, Vector3 Point, Vector3 Normal)? SuspensionRay(
        World3 world, int chassis, Vector3 origin, Vector3 direction, float maxT)
    {
        (uint Body, float T, Vector3 Point, Vector3 Normal)? best = null;
        var filter = QueryFilter.Everything;
        foreach (var body in world.Bodies)
        {
            if (body.Id == chassis) continue;
            if (!filter.Accepts(body)) continue;
            if (body.IsSensor) continue;
            if (body.Shape == null) continue;
            Vector3 point;
            Vector3 normal;
            float t;
            bool hit;
            switch (body.Shape)
            {
                case SphereShape3 sphere:
                    hit = Query.RaySphere(
                        origin, direction, body.Transform.Position, sphere.Radius,
                        out t, out point, out normal);
                    break;
                case BoxShape3 box:
                    hit = Query.RayObb(
                        origin, direction, body.Transform.Position, body.Transform.Rotation,
                        box.HalfExtents, out t, out point, out normal);
                    break;
                case CapsuleShape3 capsule:
                    var axis = body.Transform.RotateVector(Vector3.UnitY) * capsule.HalfLength;
                    hit = Query.RayCapsule(
                        origin, direction,
                        body.Transform.Position + axis, body.Transform.Position - axis,
                        capsule.Radius, out t, out point, out normal);
                    break;
                default:
                    hit = false;
                    t = 0.0f;
                    point = Vector3.Zero;
                    normal = Vector3.Zero;
                    break;
            }
            if (!hit || t > maxT) continue;
            if (best == null || t < best.Value.T)
            {
                best = ((uint)body.Id, t, point, normal);
            }
        }
        return best;
    }

    /// <summary>
    /// Run one suspension-vehicle step on a chassis. Call it before
    /// <c>World3.Step</c> with the same <paramref name="dt"/>.
    /// Returns the wheels processed (0 for an unknown chassis, no
    /// wheels, or a non-positive <paramref name="dt"/>).
    /// </summary>
    public static int StepVehicle3(
        World3 world, int chassis, VehicleConfig3 config, VehicleInput3 input,
        float dt, WheelTelemetry3[] telemetry)
    {
        if (!float.IsFinite(dt) || dt <= 0.0f) return 0;
        int count = config.WheelCount;
        if (count > VehicleConfig3.MaxWheels) count = VehicleConfig3.MaxWheels;
        if (count > telemetry.Length) count = telemetry.Length;
        if (count <= 0) return 0;
        if (chassis < 0 || chassis >= world.Bodies.Count) return 0;
        var snap = world.Bodies[chassis];
        if (snap.Id != chassis) return 0;

        float throttle = MathF.Max(-1.0f, MathF.Min(1.0f, input.Throttle));
        float brake = MathF.Max(0.0f, MathF.Min(1.0f, input.Brake));
        float steering = MathF.Max(-1.0f, MathF.Min(1.0f, input.Steering));
        float handbrake = MathF.Max(0.0f, MathF.Min(1.0f, input.Handbrake));

        float rest = MathF.Max(config.RestLength, 1e-3f);
        float radius = MathF.Max(config.WheelRadius, 1e-3f);
        float nominal = MathF.Max(snap.Mass * world.Gravity.Length() / MathF.Max(count, 1), 1e-6f);
        int driven = 0;
        for (int i = 0; i < count; i++)
        {
            if (config.Wheels[i].Drive) driven++;
        }
        float engineEach = driven == 0 ? 0.0f : config.EngineForce / driven;
        float brakeEach = config.BrakeForce / count;

        Vector3 fwd0 = DeterministicRotation.RotateVector(snap.Transform.Rotation, Vector3.UnitZ);
        Vector3 up0 = DeterministicRotation.RotateVector(snap.Transform.Rotation, Vector3.UnitY);
        var suspForce = new Vector3[VehicleConfig3.MaxWheels];
        var suspPoint = new Vector3[VehicleConfig3.MaxWheels];
        var tireForce = new Vector3[VehicleConfig3.MaxWheels];
        var tirePoint = new Vector3[VehicleConfig3.MaxWheels];
        var hasSusp = new bool[VehicleConfig3.MaxWheels];
        var hasTire = new bool[VehicleConfig3.MaxWheels];

        for (int i = 0; i < count; i++)
        {
            var mount = config.Wheels[i];
            var tele = new WheelTelemetry3();
            Vector3 hard = snap.Transform.Position + DeterministicRotation.RotateVector(snap.Transform.Rotation, mount.Offset);
            Vector3 dir = -up0;
            var hit = SuspensionRay(world, chassis, hard, dir, rest + radius);
            if (hit == null)
            {
                telemetry[i] = tele;
                continue;
            }
            float axleHeight = hit.Value.T - radius;
            float compression = MathF.Max(0.0f, MathF.Min(rest, rest - axleHeight));
            tele.InContact = true;
            tele.Compression01 = compression / rest;
            Vector3 chassisVel = VelocityAtPoint(
                snap.Velocity, snap.AngularVelocity, snap.Transform.Position, hard);
            var gb = world.Bodies[(int)hit.Value.Body];
            Vector3 groundVel = VelocityAtPoint(
                gb.Velocity, gb.AngularVelocity, gb.Transform.Position, hit.Value.Point);
            float closing = Vector3.Dot(chassisVel - groundVel, dir);
            float spring = MathF.Max(0.0f, MathF.Min(
                config.MaxSuspensionForce,
                config.SpringK * compression + config.DampingC * closing));
            tele.NormalLoad = spring;
            suspForce[i] = up0 * spring;
            suspPoint[i] = hard;
            hasSusp[i] = true;

            float steerAngle = mount.Steer ? steering * config.MaxSteer : 0.0f;
            Vector3 wheelFwd = RotateAroundUp(fwd0, up0, steerAngle);
            Vector3 wheelLat = Vector3.Cross(up0, wheelFwd);
            Vector3 contactVel = VelocityAtPoint(
                snap.Velocity, snap.AngularVelocity, snap.Transform.Position, hit.Value.Point);
            float vLong = Vector3.Dot(contactVel, wheelFwd);
            float vLat = Vector3.Dot(contactVel, wheelLat);
            float load = MathF.Max(0.0f, MathF.Min(1.0f, spring / nominal));
            float forceLong = 0.0f;
            if (mount.Drive)
            {
                forceLong += throttle * engineEach * load;
                forceLong -= handbrake * config.BrakeForce * 0.75f * load * SmoothSign(vLong);
            }
            forceLong -= brake * brakeEach * load * SmoothSign(vLong);
            forceLong -= config.RollingResist * spring * SmoothSign(vLong);
            float grip = config.GripLateral;
            if (mount.Drive)
            {
                grip *= 1.0f - 0.9f * handbrake;
            }
            float forceLat = -grip * vLat * (spring / nominal);
            float cap = config.FrictionMu * spring;
            float total = MathF.Sqrt(forceLong * forceLong + forceLat * forceLat);
            if (total > cap)
            {
                float s = cap / total;
                forceLong *= s;
                forceLat *= s;
            }
            tele.ForceLong = forceLong;
            tele.ForceLat = forceLat;
            tele.Spin = vLong / radius;
            tireForce[i] = wheelFwd * forceLong + wheelLat * forceLat;
            tirePoint[i] = hit.Value.Point;
            hasTire[i] = true;
            telemetry[i] = tele;
        }

        var chassisBody = world.Bodies[chassis];
        for (int i = 0; i < count; i++)
        {
            if (hasSusp[i]) chassisBody.ApplyForce(suspForce[i], suspPoint[i], dt);
            if (hasTire[i]) chassisBody.ApplyForce(tireForce[i], tirePoint[i], dt);
        }

        Vector3 v = chassisBody.Velocity;
        float speed = v.Length();
        if (speed > 0.0f)
        {
            chassisBody.ApplyForce(v * (-config.DragK * speed), chassisBody.Transform.Position, dt);
        }
        return count;
    }
}
