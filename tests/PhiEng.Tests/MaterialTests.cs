using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Material tests: defaults, combining rules, body attachment, and a
/// restitution bounce end-to-end. Mirrors the Rust material tests in
/// crates/dynamics/src/material.rs and the bounce tests in
/// crates/engine/tests/integration.rs.
/// </summary>
public class MaterialTests
{
    private const float Dt = 1.0f / 60.0f;

    [Fact]
    public void Material_HasSensibleDefaults()
    {
        var m = PhysicsMaterial.Default;
        Assert.Equal(0.4f, m.Friction);
        Assert.Equal(0.1f, m.Restitution);
        Assert.Equal(1.0f, m.Density);
    }

    [Fact]
    public void Material_ConstructorSanitizes()
    {
        var m = new PhysicsMaterial(float.NaN, 5.0f, -2.0f);
        Assert.Equal(PhysicsMaterial.DefaultFriction, m.Friction);
        Assert.Equal(1.0f, m.Restitution);
        Assert.Equal(PhysicsMaterial.DefaultDensity, m.Density);

        var ok = new PhysicsMaterial(0.8f, 0.6f, 2.0f);
        Assert.Equal(0.8f, ok.Friction);
        Assert.Equal(0.6f, ok.Restitution);
        Assert.Equal(2.0f, ok.Density);
    }

    [Fact]
    public void CombineFriction_IsGeometricMean()
    {
        Assert.True(MathF.Abs(PhysicsMaterial.CombineFriction(0.9f, 0.9f) - 0.9f) < 1e-6f);
        Assert.True(MathF.Abs(PhysicsMaterial.CombineFriction(1.0f, 0.25f) - 0.5f) < 1e-6f);
        // Frictionless side always wins: the pair must slide.
        Assert.Equal(0.0f, PhysicsMaterial.CombineFriction(0.0f, 0.9f));
    }

    [Fact]
    public void CombineRestitution_IsMax()
    {
        Assert.Equal(0.9f, PhysicsMaterial.CombineRestitution(0.9f, 0.0f));
        Assert.Equal(0.5f, PhysicsMaterial.CombineRestitution(0.2f, 0.5f));
    }

    [Fact]
    public void SetMaterial_RecomputesMass()
    {
        var body = new RigidBody2(0);
        body.SetShape(new CircleShape(1.0f), 1.0f);
        float baseMass = body.Mass;

        body.SetMaterial(new PhysicsMaterial(0.8f, 0.6f, 2.0f));
        Assert.Equal(0.8f, body.Material.Friction);
        Assert.Equal(0.6f, body.Material.Restitution);
        Assert.Equal(2.0f, body.Material.Density);
        Assert.True(MathF.Abs(body.Mass - baseMass * 2.0f) < 1e-4f, $"mass={body.Mass}");
        Assert.True(MathF.Abs(body.InvMass - 1.0f / body.Mass) < 1e-6f);

        // The stored material is a copy: later user-side mutation aliases nothing.
        var shared = new PhysicsMaterial(0.5f, 0.5f, 1.0f);
        body.SetMaterial(shared);
        shared.Friction = 0.0f;
        Assert.Equal(0.5f, body.Material.Friction);
    }

    [Fact]
    public void PolygonMass_MatchesBoxAndWinding()
    {
        var ccw = new List<Vector2>
        {
            new Vector2(-0.5f, -0.5f),
            new Vector2(0.5f, -0.5f),
            new Vector2(0.5f, 0.5f),
            new Vector2(-0.5f, 0.5f),
        };
        var cw = new List<Vector2>(ccw);
        cw.Reverse();

        var a = new RigidBody2(0);
        a.SetShape(new PolygonShape(ccw), 2.0f);
        var b = new RigidBody2(1);
        b.SetShape(new PolygonShape(cw), 2.0f);

        foreach (var body in new[] { a, b })
        {
            Assert.True(MathF.Abs(body.Mass - 2.0f) < 1e-4f, $"mass={body.Mass}");
            Assert.True(MathF.Abs(body.Inertia - 2.0f * 2.0f / 12.0f) < 1e-4f, $"inertia={body.Inertia}");
        }
        Assert.True(MathF.Abs(a.Mass - b.Mass) < 1e-6f);
    }

    [Fact]
    public void SetShape_StoresDensityInMaterial()
    {
        var body = new RigidBody2(0);
        body.SetShape(new BoxShape2(new Vector2(0.5f, 0.5f)), 3.0f);
        Assert.Equal(3.0f, body.Material.Density);
        Assert.True(body.Mass > 0.0f);
    }

    private static float BounceApex(float ballRestitution)
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int ball = world.AddDynamicBody(
            new Vector2(0.0f, 2.0f),
            new CircleShape(0.5f),
            1.0f);
        world.Bodies[ball].SetMaterial(new PhysicsMaterial(0.4f, ballRestitution, 1.0f));

        // Impact happens ~frame 38; scan the post-impact window for the apex.
        float apex = float.MinValue;
        for (int frame = 0; frame < 240; frame++)
        {
            world.Step(Dt);
            if (frame >= 45)
            {
                apex = MathF.Max(apex, world.Bodies[ball].Transform.Position.Y);
            }
        }
        return apex;
    }

    [Fact]
    public void BouncyBall_Rebounds()
    {
        // Drop height to contact = 2.0; e=0.9 rebounds to ~0.81*2.0 = 1.62.
        float apex = BounceApex(0.9f);
        Assert.True(apex > 0.8f && apex < 2.0f, $"bouncy ball should rebound high, apex={apex}");
    }

    [Fact]
    public void DeadBall_Sticks()
    {
        float apex = BounceApex(0.0f);
        Assert.True(apex < 0.6f, $"dead ball should not rebound, apex={apex}");
    }
}
