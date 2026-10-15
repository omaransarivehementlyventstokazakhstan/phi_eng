using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Solver-level tests: Coulomb friction cone, accumulation, warm starting.
/// Mirrors the Rust tests in crates/dynamics/src/solver.rs.
/// </summary>
public class SolverTests
{
    private static RigidBody2 DynamicBox(int id, Vector2 pos, Vector2 vel)
    {
        var b = new RigidBody2(id);
        var t = b.Transform;
        t.Position = pos;
        b.Transform = t;
        b.Velocity = vel;
        // Isolate linear response: lock rotation.
        b.InvInertia = 0.0f;
        return b;
    }

    private static ContactConstraint Contact(int a, int b, Vector2 point, Vector2 normal)
    {
        return new ContactConstraint
        {
            BodyA = a,
            BodyB = b,
            Point = point,
            Normal = normal,
            Penetration = 0.01f,
            Friction = 0.5f,
            Restitution = 0.0f,
            NormalImpulse = 0.0f,
            TangentImpulse = 0.0f
        };
    }

    [Fact]
    public void Solver_HasDefaultIterations()
    {
        Assert.Equal(8, new Solver().Iterations);
    }

    [Fact]
    public void EarlyExit_OnRestingContact()
    {
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), Vector2.Zero)
        };
        var constraints = new List<ContactConstraint>
        {
            new ContactConstraint
            {
                BodyA = 0, BodyB = 1,
                Point = new Vector2(0.0f, 0.0f), Normal = Vector2.UnitY,
                Penetration = 0.001f, Friction = 0.5f, Restitution = 0.0f,
            }
        };
        int ran = new Solver().Solve(bodies, constraints, 1.0f / 60.0f);
        Assert.Equal(1, ran);
        Assert.Equal(0.0f, constraints[0].NormalImpulse);
    }

    [Fact]
    public void NoEarlyExit_WhileResolving()
    {
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), Vector2.Zero)
        };
        var constraints = new List<ContactConstraint>
        {
            new ContactConstraint
            {
                BodyA = 0, BodyB = 1,
                Point = new Vector2(0.0f, 0.0f), Normal = Vector2.UnitY,
                Penetration = 0.5f, Friction = 0.5f, Restitution = 0.0f,
            }
        };
        int ran = new Solver().Solve(bodies, constraints, 1.0f / 60.0f);
        Assert.True(ran > 1, $"resolving contact must iterate, ran={ran}");
        Assert.True(constraints[0].NormalImpulse > 0.0f);
    }

    [Fact]
    public void Friction_KillsTangentialVelocity()
    {
        // Static floor (A) + slow-sliding dynamic box (B). One frame of
        // friction (mu=0.5) can erase 0.5*jn of tangential velocity.
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), new Vector2(0.4f, -1.0f))
        };
        var constraints = new List<ContactConstraint>
        {
            Contact(0, 1, new Vector2(0.0f, 0.0f), Vector2.UnitY)
        };

        new Solver().Solve(bodies, constraints, 1.0f / 60.0f);

        Assert.True(
            MathF.Abs(bodies[1].Velocity.X) < 0.05f,
            $"friction should stop sliding, vx={bodies[1].Velocity.X}");
        var c = constraints[0];
        Assert.True(c.NormalImpulse >= 0.0f);
        Assert.True(
            MathF.Abs(c.TangentImpulse) <= c.Friction * c.NormalImpulse + 1e-4f,
            $"|jt|={MathF.Abs(c.TangentImpulse)} > mu*jn={c.Friction * c.NormalImpulse}");
    }

    [Fact]
    public void ZeroFriction_PreservesSliding()
    {
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), new Vector2(3.0f, -1.0f))
        };
        var constraint = Contact(0, 1, new Vector2(0.0f, 0.0f), Vector2.UnitY);
        constraint.Friction = 0.0f;
        var constraints = new List<ContactConstraint> { constraint };

        new Solver().Solve(bodies, constraints, 1.0f / 60.0f);

        Assert.True(
            MathF.Abs(bodies[1].Velocity.X - 3.0f) < 1e-4f,
            $"vx should be preserved, got {bodies[1].Velocity.X}");
        Assert.Equal(0.0f, constraints[0].TangentImpulse);
    }

    [Fact]
    public void FrictionCone_Saturates()
    {
        // Fast slider, low mu: tangent impulse must saturate at mu * jn.
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), new Vector2(20.0f, -1.0f))
        };
        var constraint = Contact(0, 1, new Vector2(0.0f, 0.0f), Vector2.UnitY);
        constraint.Friction = 0.1f;
        var constraints = new List<ContactConstraint> { constraint };

        new Solver().Solve(bodies, constraints, 1.0f / 60.0f);

        var c = constraints[0];
        float maxF = c.Friction * c.NormalImpulse;
        Assert.True(
            MathF.Abs(MathF.Abs(c.TangentImpulse) - maxF) < 1e-3f,
            $"cone should saturate: |jt|={MathF.Abs(c.TangentImpulse)} max={maxF}");
        Assert.True(bodies[1].Velocity.X > 10.0f);
    }

    [Fact]
    public void WarmStart_AppliesCachedImpulse()
    {
        // Pre-loaded impulses must move velocities before any iteration runs
        // (iterations=0 proves warm start alone acts).
        var bodies = new List<RigidBody2>
        {
            RigidBody2.StaticBody(0),
            DynamicBox(1, new Vector2(0.0f, 0.5f), Vector2.Zero)
        };
        var constraint = Contact(0, 1, new Vector2(0.0f, 0.0f), Vector2.UnitY);
        constraint.NormalImpulse = 2.0f;
        constraint.TangentImpulse = 1.0f;
        var constraints = new List<ContactConstraint> { constraint };

        var solver = new Solver { Iterations = 0 };
        solver.Solve(bodies, constraints, 1.0f / 60.0f);

        // P = n*2 + t*1 with t = (-n.Y, n.X) = (-1, 0); mass 1 so dv = P.
        Assert.True(MathF.Abs(bodies[1].Velocity.X - -1.0f) < 1e-5f, $"vx={bodies[1].Velocity.X}");
        Assert.True(MathF.Abs(bodies[1].Velocity.Y - 2.0f) < 1e-5f, $"vy={bodies[1].Velocity.Y}");
    }

    [Fact]
    public void Rotation_IntegratesWithoutPingPong()
    {
        // Regression: angle extraction must return +angle, or bodies
        // oscillate between +a and -a every frame.
        var body = new RigidBody2(0);
        body.AngularVelocity = 1.0f;
        float dt = 1.0f / 60.0f;
        float prev = 0.0f;
        for (int i = 0; i < 10; i++)
        {
            body.Integrate(dt);
            float a = body.RotationAngle;
            Assert.True(a > prev, $"angle must advance monotonically, prev={prev} now={a}");
            prev = a;
        }
        Assert.True(MathF.Abs(prev - 10.0f * dt) < 1e-3f, $"angle={prev}");
    }
}
