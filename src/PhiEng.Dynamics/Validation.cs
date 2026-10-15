using System.Numerics;

namespace PhiEng.Dynamics;

/// <summary>
/// Validation configuration controlling runtime checks.
/// Mirrors Rust <c>ValidationConfig</c>.
/// </summary>
public class ValidationConfig
{
    /// <summary>Check for NaN/Inf in positions, velocities every step.</summary>
    public bool CheckFinite { get; set; }

    /// <summary>Clamp penetration depth to this maximum (prevents explosion from deep overlap).</summary>
    public float MaxPenetration { get; set; } = 1.0f;

    /// <summary>Maximum allowed mass ratio (heavier/lighter). Pairs exceeding this get warning. 0 = disabled.</summary>
    public float MaxMassRatio { get; set; } = 1000.0f;

    /// <summary>Maximum allowed coordinate magnitude. Bodies exceeding this are clamped.</summary>
    public float MaxCoordinate { get; set; } = 1e4f;

    /// <summary>Maximum allowed velocity magnitude.</summary>
    public float MaxVelocity { get; set; } = 1e3f;

    /// <summary>Maximum allowed angular velocity (rad/s).</summary>
    public float MaxAngularVelocity { get; set; } = 1e3f;

    /// <summary>If true, log warnings for detected issues.</summary>
    public bool LogWarnings { get; set; }

    /// <summary>If true, automatically correct detected issues (clamp, zero, etc.).</summary>
    public bool AutoCorrect { get; set; } = true;

    /// <summary>Default configuration.</summary>
    public static ValidationConfig Default => new();

    /// <summary>Strict config for debugging — checks everything, tight limits.</summary>
    public static ValidationConfig Strict => new()
    {
        CheckFinite = true,
        MaxPenetration = 0.5f,
        MaxMassRatio = 100.0f,
        MaxCoordinate = 1e3f,
        MaxVelocity = 1e2f,
        MaxAngularVelocity = 1e2f,
        LogWarnings = true,
        AutoCorrect = true,
    };

    /// <summary>Permissive config for production — minimal overhead, only catastrophic checks.</summary>
    public static ValidationConfig Permissive => new()
    {
        CheckFinite = false,
        MaxPenetration = 10.0f,
        MaxMassRatio = 0.0f,
        MaxCoordinate = 1e5f,
        MaxVelocity = 1e4f,
        MaxAngularVelocity = 1e4f,
        LogWarnings = false,
        AutoCorrect = true,
    };
}

/// <summary>
/// Validation result with optional correction info.
/// Mirrors Rust <c>ValidationResult</c>.
/// </summary>
public enum ValidationResult
{
    /// <summary>No issues detected.</summary>
    Ok,
    /// <summary>Non-finite value detected and corrected.</summary>
    NonFiniteCorrected,
    /// <summary>Penetration depth exceeded max and was clamped.</summary>
    PenetrationClamped,
    /// <summary>Mass ratio exceeded threshold (logged, not corrected).</summary>
    MassRatioExceeded,
    /// <summary>Coordinate magnitude exceeded max and was clamped.</summary>
    CoordinateClamped,
    /// <summary>Velocity magnitude exceeded max and was clamped.</summary>
    VelocityClamped,
    /// <summary>Angular velocity exceeded max and was clamped.</summary>
    AngularVelocityClamped,
}

/// <summary>
/// Aggregate validation results for reporting.
/// Mirrors Rust <c>ValidationReport</c>.
/// </summary>
public class ValidationReport
{
    /// <summary>Count of non-finite values detected and corrected.</summary>
    public int NonFiniteCorrected { get; set; }
    /// <summary>Count of penetration depths clamped to the maximum.</summary>
    public int PenetrationClamped { get; set; }
    /// <summary>Count of pairs exceeding the mass-ratio threshold.</summary>
    public int MassRatioExceeded { get; set; }
    /// <summary>Count of positions clamped to the coordinate bound.</summary>
    public int CoordinateClamped { get; set; }
    /// <summary>Count of velocities clamped to the speed bound.</summary>
    public int VelocityClamped { get; set; }
    /// <summary>Count of angular velocities clamped to the spin bound.</summary>
    public int AngularVelocityClamped { get; set; }

    /// <summary>Folds one per-body/per-constraint result into the totals.</summary>
    public void Merge(ValidationResult result)
    {
        switch (result)
        {
            case ValidationResult.Ok: break;
            case ValidationResult.NonFiniteCorrected: NonFiniteCorrected++; break;
            case ValidationResult.PenetrationClamped: PenetrationClamped++; break;
            case ValidationResult.MassRatioExceeded: MassRatioExceeded++; break;
            case ValidationResult.CoordinateClamped: CoordinateClamped++; break;
            case ValidationResult.VelocityClamped: VelocityClamped++; break;
            case ValidationResult.AngularVelocityClamped: AngularVelocityClamped++; break;
        }
    }

    /// <summary>True when any counter is non-zero.</summary>
    public bool AnyIssues => NonFiniteCorrected > 0
        || PenetrationClamped > 0
        || MassRatioExceeded > 0
        || CoordinateClamped > 0
        || VelocityClamped > 0
        || AngularVelocityClamped > 0;
}

/// <summary>
/// Static validation helpers for 2D bodies and constraints.
/// </summary>
public static class Validation2D
{
    /// <summary>Validate and optionally correct a single body's state.</summary>
    public static ValidationResult ValidateBody(RigidBody2 body, ValidationConfig config)
    {
        var result = ValidationResult.Ok;

if (config.CheckFinite)
        {
            var hasNonFinite = false;
            
            // Check and fix position
            var pos = body.Transform.Position;
            if (!float.IsFinite(pos.X))
            {
                pos.X = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(pos.Y))
            {
                pos.Y = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite)
            {
                var t = body.Transform;
                t.Position = pos;
                body.Transform = t;
            }

            // Check and fix velocity
            var vel = body.Velocity;
            if (!float.IsFinite(vel.X))
            {
                vel.X = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(vel.Y))
            {
                vel.Y = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite)
            {
                body.Velocity = vel;
            }

            // Check and fix angular velocity
            if (!float.IsFinite(body.AngularVelocity))
            {
                body.AngularVelocity = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite) result = ValidationResult.NonFiniteCorrected;
        }

        // Clamp position
        var posLenSq = body.Transform.Position.LengthSquared();
        var maxPos = config.MaxCoordinate;
        if (posLenSq > maxPos * maxPos)
        {
            var scale = maxPos / MathF.Sqrt(posLenSq);
            var t = body.Transform;
            t.Position *= scale;
            body.Transform = t;
            result = ValidationResult.CoordinateClamped;
        }

        // Clamp linear velocity
        var velLenSq = body.Velocity.LengthSquared();
        var maxVel = config.MaxVelocity;
        if (velLenSq > maxVel * maxVel)
        {
            var scale = maxVel / MathF.Sqrt(velLenSq);
            body.Velocity *= scale;
            result = ValidationResult.VelocityClamped;
        }

        // Clamp angular velocity
        if (MathF.Abs(body.AngularVelocity) > config.MaxAngularVelocity)
        {
            body.AngularVelocity = MathF.Sign(body.AngularVelocity) * config.MaxAngularVelocity;
            result = ValidationResult.AngularVelocityClamped;
        }

        return result;
    }

    /// <summary>Validate contact constraints: clamp penetration, check mass ratios.</summary>
    public static ValidationResult ValidateConstraints(
        List<RigidBody2> bodies,
        List<ContactConstraint> constraints,
        ValidationConfig config)
    {
        var result = ValidationResult.Ok;

        for (int i = 0; i < constraints.Count; i++)
        {
            var c = constraints[i];
            var modified = false;

            // Clamp penetration
            if (c.Penetration > config.MaxPenetration)
            {
                c.Penetration = config.MaxPenetration;
                modified = true;
                result = ValidationResult.PenetrationClamped;
            }
            if (c.Penetration < 0.0f)
            {
                c.Penetration = 0.0f;
                modified = true;
                result = ValidationResult.PenetrationClamped;
            }

            // Check mass ratio without divisions (mirrors Rust): heavy/light
            // mass ratio equals light/heavy inverse-mass ratio. Static pairs
            // (zero inverse mass) never report, exactly as before.
            if (config.MaxMassRatio > 0.0f)
            {
                float ia = bodies[c.BodyA].InvMass;
                float ib = bodies[c.BodyB].InvMass;
                if (ia > 0.0f && ib > 0.0f)
                {
                    float hi = ia > ib ? ia : ib;
                    float lo = ia > ib ? ib : ia;
                    if (hi > config.MaxMassRatio * lo)
                    {
                        result = ValidationResult.MassRatioExceeded;
                    }
                }
            }

            if (modified)
            {
                constraints[i] = c;
            }
        }

        return result;
    }
}

/// <summary>
/// Static validation helpers for 3D bodies and constraints.
/// </summary>
public static class Validation3D
{
    /// <summary>Validate and optionally correct a single body's state.</summary>
    public static ValidationResult ValidateBody(RigidBody3 body, ValidationConfig config)
    {
        var result = ValidationResult.Ok;

        if (config.CheckFinite)
        {
            var hasNonFinite = false;
            
            // Check and fix position
            var pos = body.Transform.Position;
            if (!float.IsFinite(pos.X))
            {
                pos.X = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(pos.Y))
            {
                pos.Y = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(pos.Z))
            {
                pos.Z = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite)
            {
                var t = body.Transform;
                t.Position = pos;
                body.Transform = t;
            }

            // Check and fix velocity
            var vel = body.Velocity;
            if (!float.IsFinite(vel.X))
            {
                vel.X = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(vel.Y))
            {
                vel.Y = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(vel.Z))
            {
                vel.Z = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite)
            {
                body.Velocity = vel;
            }

            // Check and fix angular velocity
            var angVel = body.AngularVelocity;
            if (!float.IsFinite(angVel.X))
            {
                angVel.X = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(angVel.Y))
            {
                angVel.Y = 0.0f;
                hasNonFinite = true;
            }
            if (!float.IsFinite(angVel.Z))
            {
                angVel.Z = 0.0f;
                hasNonFinite = true;
            }
            if (hasNonFinite)
            {
                body.AngularVelocity = angVel;
            }
            if (hasNonFinite) result = ValidationResult.NonFiniteCorrected;
        }

        var posLenSq = body.Transform.Position.LengthSquared();
        var maxPos = config.MaxCoordinate;
        if (posLenSq > maxPos * maxPos)
        {
            var scale = maxPos / MathF.Sqrt(posLenSq);
            var t = body.Transform;
            t.Position *= scale;
            body.Transform = t;
            result = ValidationResult.CoordinateClamped;
        }

        var velLenSq = body.Velocity.LengthSquared();
        var maxVel = config.MaxVelocity;
        if (velLenSq > maxVel * maxVel)
        {
            var scale = maxVel / MathF.Sqrt(velLenSq);
            body.Velocity *= scale;
            result = ValidationResult.VelocityClamped;
        }

        var angVelLen = body.AngularVelocity.Length();
        if (angVelLen > config.MaxAngularVelocity)
        {
            body.AngularVelocity *= config.MaxAngularVelocity / angVelLen;
            result = ValidationResult.AngularVelocityClamped;
        }

        return result;
    }

    /// <summary>Validate 3D contact constraints.</summary>
    public static ValidationResult ValidateConstraints(
        List<RigidBody3> bodies,
        List<ContactConstraint3D> constraints,
        ValidationConfig config)
    {
        var result = ValidationResult.Ok;

        for (int i = 0; i < constraints.Count; i++)
        {
            var c = constraints[i];
            var modified = false;

            if (c.Penetration > config.MaxPenetration)
            {
                c.Penetration = config.MaxPenetration;
                modified = true;
                result = ValidationResult.PenetrationClamped;
            }
            if (c.Penetration < 0.0f)
            {
                c.Penetration = 0.0f;
                modified = true;
                result = ValidationResult.PenetrationClamped;
            }

            if (config.MaxMassRatio > 0.0f)
            {
                float ia = bodies[c.BodyA].InvMass;
                float ib = bodies[c.BodyB].InvMass;
                if (ia > 0.0f && ib > 0.0f)
                {
                    float hi = ia > ib ? ia : ib;
                    float lo = ia > ib ? ib : ia;
                    if (hi > config.MaxMassRatio * lo)
                    {
                        result = ValidationResult.MassRatioExceeded;
                    }
                }
            }

            if (modified)
            {
                constraints[i] = c;
            }
        }

        return result;
    }
}