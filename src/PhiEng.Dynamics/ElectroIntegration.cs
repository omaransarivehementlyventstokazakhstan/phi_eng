using System.Numerics;
using PhiEng.Electromagnetics;
using PhiEng.Math;

namespace PhiEng.Dynamics;

/// <summary>
/// The world-facing electromagnetic phase. Mirrors Rust
/// <c>phieng-dynamics::electro</c>: the configuration, the field grid and
/// the velocity kicks. Pairwise and field modes are mutually exclusive.
/// </summary>
public sealed class ElectroWorldState2
{
    /// <summary>The configuration. Assigning it directly is allowed.</summary>
    public ElectroConfig2 Config { get; set; } = ElectroConfig2.Default;
    /// <summary>The field grid, when the mode is <see cref="ElectroMode.Field"/>.</summary>
    public PoissonGrid2? Poisson { get; internal set; }
    /// <summary>The simulation time the phase has seen.</summary>
    public float Time { get; internal set; }
    /// <summary>Whether the last field solve was refused.</summary>
    public bool LastFieldStalled { get; internal set; }

    internal readonly List<Vector2> Positions = new();
    internal readonly List<float> Charges = new();
    internal readonly List<float> Scales = new();
    internal readonly List<float> InvMasses = new();

    /// <summary>A fresh state with the default (off) configuration.</summary>
    public ElectroWorldState2() { }

    /// <summary>True when the phase should run.</summary>
    public bool IsActive => Config.Enabled;

    /// <summary>Validate and install a configuration.</summary>
    public bool Configure(ElectroConfig2 config)
    {
        if (!config.Validate()) return false;
        Config = config;
        Poisson = null;
        Time = 0.0f;
        LastFieldStalled = false;
        return true;
    }

    internal void EnsureField(List<RigidBody2> bodies)
    {
        if (Poisson is not null && Poisson.Config.Equals(Config.Field)) return;
        var grid = new PoissonGrid2(Config.Field, Config.CoulombK);
        foreach (var body in bodies) grid.DepositCharge(body.Transform.Position, body.Electro.Charge);
        grid.Solve();
        Poisson = grid;
    }
}

/// <summary>The 3D electromagnetic state. See <see cref="ElectroWorldState2"/>.</summary>
public sealed class ElectroWorldState3
{
    /// <summary>The configuration.</summary>
    public ElectroConfig3 Config { get; set; } = ElectroConfig3.Default;
    /// <summary>The field grid, when the mode is <see cref="ElectroMode.Field"/>.</summary>
    public PoissonGrid3? Poisson { get; internal set; }
    /// <summary>The simulation time the phase has seen.</summary>
    public float Time { get; internal set; }
    /// <summary>Whether the last field solve was refused.</summary>
    public bool LastFieldStalled { get; internal set; }

    internal readonly List<Vector3> Positions = new();
    internal readonly List<float> Charges = new();
    internal readonly List<float> Scales = new();
    internal readonly List<float> InvMasses = new();

    /// <summary>A fresh state with the default (off) configuration.</summary>
    public ElectroWorldState3() { }

    /// <summary>True when the phase should run.</summary>
    public bool IsActive => Config.Enabled;

    /// <summary>Validate and install a configuration.</summary>
    public bool Configure(ElectroConfig3 config)
    {
        if (!config.Validate()) return false;
        Config = config;
        Poisson = null;
        Time = 0.0f;
        LastFieldStalled = false;
        return true;
    }

    internal void EnsureField(List<RigidBody3> bodies)
    {
        if (Poisson is not null && Poisson.Config.Equals(Config.Field)) return;
        var grid = new PoissonGrid3(Config.Field, Config.CoulombK);
        foreach (var body in bodies) grid.DepositCharge(body.Transform.Position, body.Electro.Charge);
        grid.Solve();
        Poisson = grid;
    }
}

/// <summary>The electromagnetic phase entry points. Mirrors Rust <c>electro_kick_*</c>.</summary>
public static class ElectroIntegration
{
    private static Vector2 ClampAccel(Vector2 a, float max)
    {
        float l2 = a.LengthSquared();
        if (!float.IsFinite(l2)) return Vector2.Zero;
        return l2 > max * max ? a * (max / MathF.Sqrt(l2)) : a;
    }

    private static Vector3 ClampAccel(Vector3 a, float max)
    {
        float l2 = a.LengthSquared();
        if (!float.IsFinite(l2)) return Vector3.Zero;
        return l2 > max * max ? a * (max / MathF.Sqrt(l2)) : a;
    }

    /// <summary>Apply the electromagnetic velocity kick over <paramref name="h"/>.</summary>
    public static void Kick2(List<RigidBody2> bodies, ElectroWorldState2 state, float h)
    {
        var cfg = state.Config;
        if (!cfg.Enabled || h == 0.0f || bodies.Count == 0) return;
        if (cfg.Mode == ElectroMode.Field) state.EnsureField(bodies);
        float kEff = cfg.CoulombK / cfg.Permittivity;
        float softening = cfg.Softening;
        float max = cfg.MaxAcceleration;

        if (cfg.Mode == ElectroMode.Pairwise)
        {
            int n = bodies.Count;
            var pos = state.Positions;
            var charges = state.Charges;
            var scales = state.Scales;
            var invMasses = state.InvMasses;
            pos.Clear(); charges.Clear(); scales.Clear(); invMasses.Clear();
            foreach (var body in bodies)
            {
                pos.Add(body.Transform.Position);
                charges.Add(body.Electro.Charge);
                scales.Add(body.Electro.ChargeScale);
                invMasses.Add(body.InvMass);
            }
            for (int i = 0; i < n; i++)
            {
                var body = bodies[i];
                if (body.IsStatic || body.IsSleeping) continue;
                float qEff = charges[i] * scales[i];
                Vector2 acc = Vector2.Zero;
                if (qEff != 0.0f)
                {
                    Vector2 p = pos[i];
                    for (int j = 0; j < n; j++)
                    {
                        if (i == j) continue;
                        float qj = charges[j];
                        if (qj == 0.0f) continue;
                        acc += Electrostatics.PairForce2(p, qEff, pos[j], qj, kEff, softening);
                    }
                    acc += cfg.UniformE * qEff;
                    if (cfg.UniformB != 0.0f)
                        acc += Magnetostatics.LorentzAccel2(body.Velocity, qEff, Vector2.Zero, cfg.UniformB);
                }
                Vector2 a = ClampAccel(acc * invMasses[i], max);
                body.Velocity += a * h;
            }
            return;
        }

        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
        {
            foreach (var body in bodies)
            {
                if (body.IsStatic || body.IsSleeping) continue;
                float qEff = body.Electro.Charge * body.Electro.ChargeScale;
                if (qEff == 0.0f) continue;
                Vector2 e = state.Poisson.FieldAt(body.Transform.Position) + cfg.UniformE;
                Vector2 a = Magnetostatics.LorentzAccel2(body.Velocity, qEff, e, cfg.UniformB);
                body.Velocity += ClampAccel(a * body.InvMass, max) * h;
            }
        }
    }

    /// <summary>Advance grounded drains and (in field mode) rebuild the grid.</summary>
    public static void Step2(List<RigidBody2> bodies, ElectroWorldState2 state, float dt)
    {
        if (!state.Config.Enabled) return;
        DrainGrounded2(bodies, dt);
        if (state.Config.Mode == ElectroMode.Field)
        {
            try
            {
                state.EnsureField(bodies);
                var grid = state.Poisson;
                if (grid is not null)
                {
                    grid.ClearSources();
                    foreach (var body in bodies) grid.DepositCharge(body.Transform.Position, body.Electro.Charge);
                    grid.Solve();
                }
            }
            catch (ArgumentException)
            {
                state.LastFieldStalled = true;
                state.Poisson = null;
            }
        }
        state.Time += dt;
    }

    private static void DrainGrounded2(List<RigidBody2> bodies, float dt)
    {
        foreach (var body in bodies)
        {
            var e = body.Electro;
            if (!e.IsConductor || !e.IsGrounded || e.Charge == 0.0f) continue;
            float tau = e.Capacitance * e.Resistance;
            if (float.IsFinite(tau) && tau > 0.0f && float.IsFinite(dt) && dt > 0.0f)
                e.Charge *= DeterministicScalar.Exp(-dt / tau);
            else
                e.Charge = 0.0f;
            body.Electro = e;
        }
    }

    /// <summary>Apply the 3D velocity kick.</summary>
    public static void Kick3(List<RigidBody3> bodies, ElectroWorldState3 state, float h)
    {
        var cfg = state.Config;
        if (!cfg.Enabled || h == 0.0f || bodies.Count == 0) return;
        if (cfg.Mode == ElectroMode.Field) state.EnsureField(bodies);
        float kEff = cfg.CoulombK / cfg.Permittivity;
        float softening = cfg.Softening;
        float max = cfg.MaxAcceleration;

        if (cfg.Mode == ElectroMode.Pairwise)
        {
            int n = bodies.Count;
            var pos = state.Positions;
            var charges = state.Charges;
            var scales = state.Scales;
            var invMasses = state.InvMasses;
            pos.Clear(); charges.Clear(); scales.Clear(); invMasses.Clear();
            foreach (var body in bodies)
            {
                pos.Add(body.Transform.Position);
                charges.Add(body.Electro.Charge);
                scales.Add(body.Electro.ChargeScale);
                invMasses.Add(body.InvMass);
            }
            for (int i = 0; i < n; i++)
            {
                var body = bodies[i];
                if (body.IsStatic || body.IsSleeping) continue;
                float qEff = charges[i] * scales[i];
                Vector3 acc = Vector3.Zero;
                if (qEff != 0.0f)
                {
                    Vector3 p = pos[i];
                    for (int j = 0; j < n; j++)
                    {
                        if (i == j) continue;
                        float qj = charges[j];
                        if (qj == 0.0f) continue;
                        acc += Electrostatics.PairForce3(p, qEff, pos[j], qj, kEff, softening);
                    }
                    acc += cfg.UniformE * qEff;
                    if (cfg.UniformB != Vector3.Zero)
                        acc += Magnetostatics.LorentzAccel3(body.Velocity, qEff, Vector3.Zero, cfg.UniformB);
                }
                Vector3 a = ClampAccel(acc * invMasses[i], max);
                body.Velocity += a * h;
                ApplyDipoleTorque(body, cfg.UniformB, h);
            }
            return;
        }

        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
        {
            foreach (var body in bodies)
            {
                if (body.IsStatic || body.IsSleeping) continue;
                float qEff = body.Electro.Charge * body.Electro.ChargeScale;
                if (qEff != 0.0f)
                {
                    Vector3 e = state.Poisson.FieldAt(body.Transform.Position) + cfg.UniformE;
                    Vector3 a = Magnetostatics.LorentzAccel3(body.Velocity, qEff, e, cfg.UniformB);
                    body.Velocity += ClampAccel(a * body.InvMass, max) * h;
                }
                ApplyDipoleTorque(body, cfg.UniformB, h);
            }
        }
    }

    /// <summary>Advance grounded drains and (in field mode) rebuild the grid.</summary>
    public static void Step3(List<RigidBody3> bodies, ElectroWorldState3 state, float dt)
    {
        if (!state.Config.Enabled) return;
        DrainGrounded3(bodies, dt);
        if (state.Config.Mode == ElectroMode.Field)
        {
            try
            {
                state.EnsureField(bodies);
                var grid = state.Poisson;
                if (grid is not null)
                {
                    grid.ClearSources();
                    foreach (var body in bodies) grid.DepositCharge(body.Transform.Position, body.Electro.Charge);
                    grid.Solve();
                }
            }
            catch (ArgumentException)
            {
                state.LastFieldStalled = true;
                state.Poisson = null;
            }
        }
        state.Time += dt;
    }

    private static void DrainGrounded3(List<RigidBody3> bodies, float dt)
    {
        foreach (var body in bodies)
        {
            var e = body.Electro;
            if (!e.IsConductor || !e.IsGrounded || e.Charge == 0.0f) continue;
            float tau = e.Capacitance * e.Resistance;
            if (float.IsFinite(tau) && tau > 0.0f && float.IsFinite(dt) && dt > 0.0f)
                e.Charge *= DeterministicScalar.Exp(-dt / tau);
            else
                e.Charge = 0.0f;
            body.Electro = e;
        }
    }

    private static void ApplyDipoleTorque(RigidBody3 body, Vector3 b, float h)
    {
        if (b == Vector3.Zero || body.Electro.MagneticMoment == Vector3.Zero) return;
        Vector3 torque = Magnetostatics.DipoleTorque3(body.Electro.MagneticMoment, b);
        body.AngularVelocity += body.WorldInvInertia().TransformVector(torque) * h;
    }

    /// <summary>The electric field at a world point, including the uniform field.</summary>
    public static Vector2 ElectricFieldAt2(List<RigidBody2> bodies, ElectroWorldState2 state, Vector2 p)
    {
        var cfg = state.Config;
        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
            return state.Poisson.FieldAt(p) + cfg.UniformE;
        var pos = new Vector2[bodies.Count];
        var q = new float[bodies.Count];
        for (int i = 0; i < bodies.Count; i++)
        {
            pos[i] = bodies[i].Transform.Position;
            q[i] = bodies[i].Electro.Charge;
        }
        return cfg.UniformE + Electrostatics.FieldAt2(p, pos, q, cfg.CoulombK / cfg.Permittivity, cfg.Softening);
    }

    /// <summary>The electric potential at a world point.</summary>
    public static float ElectricPotentialAt2(List<RigidBody2> bodies, ElectroWorldState2 state, Vector2 p)
    {
        var cfg = state.Config;
        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
            return state.Poisson.PotentialAt(p) - Vector2.Dot(cfg.UniformE, p);
        var pos = new Vector2[bodies.Count];
        var q = new float[bodies.Count];
        for (int i = 0; i < bodies.Count; i++)
        {
            pos[i] = bodies[i].Transform.Position;
            q[i] = bodies[i].Electro.Charge;
        }
        return -Vector2.Dot(cfg.UniformE, p)
            + Electrostatics.PotentialAt2(p, pos, q, cfg.CoulombK / cfg.Permittivity, cfg.Softening);
    }

    /// <summary>The 3D electric field at a world point.</summary>
    public static Vector3 ElectricFieldAt3(List<RigidBody3> bodies, ElectroWorldState3 state, Vector3 p)
    {
        var cfg = state.Config;
        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
            return state.Poisson.FieldAt(p) + cfg.UniformE;
        var pos = new Vector3[bodies.Count];
        var q = new float[bodies.Count];
        for (int i = 0; i < bodies.Count; i++)
        {
            pos[i] = bodies[i].Transform.Position;
            q[i] = bodies[i].Electro.Charge;
        }
        return cfg.UniformE + Electrostatics.FieldAt3(p, pos, q, cfg.CoulombK / cfg.Permittivity, cfg.Softening);
    }

    /// <summary>The 3D electric potential at a world point.</summary>
    public static float ElectricPotentialAt3(List<RigidBody3> bodies, ElectroWorldState3 state, Vector3 p)
    {
        var cfg = state.Config;
        if (cfg.Mode == ElectroMode.Field && state.Poisson is not null)
            return state.Poisson.PotentialAt(p) - Vector3.Dot(cfg.UniformE, p);
        var pos = new Vector3[bodies.Count];
        var q = new float[bodies.Count];
        for (int i = 0; i < bodies.Count; i++)
        {
            pos[i] = bodies[i].Transform.Position;
            q[i] = bodies[i].Electro.Charge;
        }
        return -Vector3.Dot(cfg.UniformE, p)
            + Electrostatics.PotentialAt3(p, pos, q, cfg.CoulombK / cfg.Permittivity, cfg.Softening);
    }
}
