using System.Numerics;
using PhiEng.Dynamics;
using PhiEng.Electromagnetics;
using PhiEng.Native;
using Xunit;

namespace PhiEng.Tests;

/// <summary>
/// Electromagnetics integration tests (EM block). Mirrors
/// <c>crates/engine/tests/electro.rs</c>; the parity tool enforces the
/// pairing.
/// </summary>
public class ElectroTests
{
    private static ElectroConfig2 Pairwise()
    {
        var c = ElectroConfig2.Default;
        c.Enabled = true;
        c.Mode = ElectroMode.Pairwise;
        return c;
    }

    private static void SetCharge(RigidBody2 body, float q)
    {
        var e = body.Electro;
        e.Charge = q;
        body.Electro = e;
    }

    private static (World2, int[]) ChargedWorld2((float x, float q)[] charges)
    {
        var w = new World2 { Gravity = Vector2.Zero };
        var ids = new int[charges.Length];
        for (int i = 0; i < charges.Length; i++)
        {
            ids[i] = w.AddDynamicBody(new Vector2(charges[i].x, 0.0f), new CircleShape(0.05f), 1.0f);
            w.Bodies[ids[i]].LinearDamping = 0.0f;
            w.Bodies[ids[i]].AngularDamping = 0.0f;
            SetCharge(w.Bodies[ids[i]], charges[i].q);
        }
        Assert.True(w.Electro.Configure(Pairwise()));
        return (w, ids);
    }

    [Fact]
    public void EM01_LikeChargesRepel()
    {
        var (w, ids) = ChargedWorld2(new[] { (-1.0f, 1.0f), (1.0f, 1.0f) });
        w.Step(1.0f / 60.0f);
        Assert.True(w.Bodies[ids[0]].Velocity.X < 0.0f);
        Assert.True(w.Bodies[ids[1]].Velocity.X > 0.0f);
    }

    [Fact]
    public void EM02_OppositeChargesAttract()
    {
        var (w, ids) = ChargedWorld2(new[] { (-1.0f, 1.0f), (1.0f, -1.0f) });
        w.Step(1.0f / 60.0f);
        Assert.True(w.Bodies[ids[0]].Velocity.X > 0.0f);
        Assert.True(w.Bodies[ids[1]].Velocity.X < 0.0f);
    }

    [Fact]
    public void EM03_UniformFieldAcceleratesByQOverM()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.1f), 1.0f);
        w.Bodies[id].LinearDamping = 0.0f;
        SetCharge(w.Bodies[id], 2.0f);
        var cfg = Pairwise();
        cfg.UniformE = new Vector2(3.0f, 0.0f);
        Assert.True(w.Electro.Configure(cfg));
        w.Step(1.0f / 60.0f);
        float expected = 2.0f * 3.0f * w.Bodies[id].InvMass / 60.0f;
        Assert.InRange(w.Bodies[id].Velocity.X, expected - 1e-5f, expected + 1e-5f);
    }

    [Fact]
    public void EM04_ChargeScaleZeroIsImmune()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.1f), 1.0f);
        w.Bodies[id].LinearDamping = 0.0f;
        var e = w.Bodies[id].Electro;
        e.Charge = 1.0f;
        e.ChargeScale = 0.0f;
        w.Bodies[id].Electro = e;
        var cfg = Pairwise();
        cfg.UniformE = new Vector2(5.0f, 0.0f);
        Assert.True(w.Electro.Configure(cfg));
        for (int i = 0; i < 10; i++) w.Step(1.0f / 60.0f);
        Assert.Equal(Vector2.Zero, w.Bodies[id].Velocity);
    }

    [Fact]
    public void EM05_StaticBodyExertsButDoesNotMove()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        int fixedId = w.AddStaticBody(Vector2.Zero, new CircleShape(0.1f));
        SetCharge(w.Bodies[fixedId], 1.0f);
        int freeId = w.AddDynamicBody(new Vector2(1.0f, 0.0f), new CircleShape(0.1f), 1.0f);
        w.Bodies[freeId].LinearDamping = 0.0f;
        SetCharge(w.Bodies[freeId], 1.0f);
        Assert.True(w.Electro.Configure(Pairwise()));
        for (int i = 0; i < 10; i++) w.Step(1.0f / 60.0f);
        Assert.Equal(Vector2.Zero, w.Bodies[fixedId].Transform.Position);
        Assert.True(w.Bodies[freeId].Transform.Position.X > 1.0f);
    }

    [Fact]
    public void EM06_EnabledWithZeroChargeIsBitIdentical()
    {
        var a = new World2();
        var b = new World2();
        a.AddDynamicBody(new Vector2(0.0f, 3.0f), new CircleShape(0.2f), 1.0f);
        b.AddDynamicBody(new Vector2(0.0f, 3.0f), new CircleShape(0.2f), 1.0f);
        Assert.True(b.Electro.Configure(Pairwise()));
        for (int i = 0; i < 60; i++)
        {
            a.Step(1.0f / 60.0f);
            b.Step(1.0f / 60.0f);
        }
        Assert.Equal(a.Bodies[0].Transform.Position, b.Bodies[0].Transform.Position);
        Assert.Equal(a.Bodies[0].Velocity, b.Bodies[0].Velocity);
    }

    [Fact]
    public void EM07_FieldModePushesLikeChargesApart()
    {
        var cfg = Pairwise();
        cfg.Mode = ElectroMode.Field;
        cfg.Field = new FieldConfig2
        {
            Nx = 48,
            Ny = 48,
            Dx = 0.1f,
            Origin = new Vector2(-2.4f, -2.4f),
            Iterations = 120,
            Omega = 1.8f,
            Boundary = FieldBoundary.Dirichlet,
            EpsR = 1.0f,
            SnapshotField = true,
        };
        var w = new World2 { Gravity = Vector2.Zero };
        var ids = new int[2];
        float[] xs = { -0.5f, 0.5f };
        for (int i = 0; i < 2; i++)
        {
            ids[i] = w.AddDynamicBody(new Vector2(xs[i], 0.0f), new CircleShape(0.05f), 1.0f);
            w.Bodies[ids[i]].LinearDamping = 0.0f;
            SetCharge(w.Bodies[ids[i]], 1.0f);
        }
        Assert.True(w.Electro.Configure(cfg));
        for (int i = 0; i < 20; i++) w.Step(1.0f / 60.0f);
        Assert.True(w.Bodies[ids[0]].Transform.Position.X < -0.5f);
        Assert.True(w.Bodies[ids[1]].Transform.Position.X > 0.5f);
    }

    [Fact]
    public void EM08_GroundedConductorDrains()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.1f), 1.0f);
        var e = w.Bodies[id].Electro;
        e.Charge = 5.0f;
        e.IsConductor = true;
        e.IsGrounded = true;
        e.Resistance = 0.0f;
        w.Bodies[id].Electro = e;
        var cfg = Pairwise();
        cfg.Mode = ElectroMode.Off;
        Assert.True(w.Electro.Configure(cfg));
        w.Step(1.0f / 60.0f);
        Assert.Equal(0.0f, w.Bodies[id].Electro.Charge);
    }

    [Fact]
    public void EM09_ThreeDOppositeChargesAttract()
    {
        var w = new World3 { Gravity = Vector3.Zero };
        var ids = new int[2];
        (float x, float q)[] charges = { (-1.0f, 1.0f), (1.0f, -1.0f) };
        for (int i = 0; i < 2; i++)
        {
            ids[i] = w.AddDynamicBody(new Vector3(charges[i].x, 0.0f, 0.0f), new SphereShape3(0.05f), 1.0f);
            w.Bodies[ids[i]].LinearDamping = 0.0f;
            var e = w.Bodies[ids[i]].Electro;
            e.Charge = charges[i].q;
            w.Bodies[ids[i]].Electro = e;
        }
        var cfg3 = ElectroConfig3.Default;
        cfg3.Enabled = true;
        cfg3.Mode = ElectroMode.Pairwise;
        Assert.True(w.Electro.Configure(cfg3));
        w.Step(1.0f / 60.0f);
        Assert.True(w.Bodies[ids[0]].Velocity.X > 0.0f);
        Assert.True(w.Bodies[ids[1]].Velocity.X < 0.0f);
    }

    [Fact]
    public void EM10_FieldAndPotentialQueryAreAnalytic()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        int id = w.AddDynamicBody(Vector2.Zero, new CircleShape(0.1f), 1.0f);
        SetCharge(w.Bodies[id], 1.0f);
        Assert.True(w.Electro.Configure(Pairwise()));
        Vector2 e = ElectroIntegration.ElectricFieldAt2(w.Bodies, w.Electro, new Vector2(2.0f, 0.0f));
        Assert.True((e - new Vector2(0.25f, 0.0f)).Length() < 1e-6f);
        float v = ElectroIntegration.ElectricPotentialAt2(w.Bodies, w.Electro, new Vector2(2.0f, 0.0f));
        Assert.True(MathF.Abs(v - 0.5f) < 1e-6f);
    }

    [Fact]
    public void EM11_FfiConfigAndBodyRoundTrip()
    {
        using var world = new PhiNativeWorld();
        int id = world.AddCircleBody(Vector2.Zero, 0.5f, 1.0f, false);
        var cfg = ElectroConfig2.Default;
        cfg.Enabled = true;
        cfg.Mode = ElectroMode.Pairwise;
        cfg.UniformE = new Vector2(2.0f, 0.0f);
        cfg.Field = new FieldConfig2
        {
                Nx = 64,
                Ny = 64,
                Dx = 0.1f,
                Origin = new Vector2(-3.2f, -3.2f),
                Iterations = 40,
                Omega = 1.8f,
                Boundary = FieldBoundary.Dirichlet,
                EpsR = 1.0f,
                SnapshotField = true,
        };
        Assert.True(world.SetElectroConfig(cfg));
        var got = world.GetElectroConfig();
        Assert.Equal(ElectroMode.Pairwise, got.Mode);
        Assert.Equal(2.0f, got.UniformE.X);

        var state = ElectroState2.Default;
        state.Charge = 3.0f;
        Assert.True(world.SetBodyElectroState(id, state));
        Assert.Equal(3.0f, world.GetBodyElectroState(id).Charge);

        Assert.True(world.SetBodyCharge(id, 4.0f));
        Assert.Equal(4.0f, world.GetBodyCharge(id));
        Assert.False(world.SetBodyCharge(id, float.NaN));
    }

    [Fact]
    public void EM12_SnapshotRoundTripsTheBlock()
    {
        var (w, ids) = ChargedWorld2(new[] { (-1.0f, 2.0f), (1.0f, -3.0f) });
        string text = Snapshot.World2ToJson(w, SnapshotLoadMode.Exact);
        var loaded = Snapshot.World2FromJson(text, SnapshotLoadMode.Exact);
        foreach (int id in ids)
            Assert.Equal(w.Bodies[id].Electro.Charge, loaded.Bodies[id].Electro.Charge);
        Assert.Equal(w.Electro.Config.Mode, loaded.Electro.Config.Mode);

        byte[] bytes = Snapshot.World2ToBinary(w, SnapshotLoadMode.Exact);
        var fromBinary = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        foreach (int id in ids)
            Assert.Equal(w.Bodies[id].Electro.Charge, fromBinary.Bodies[id].Electro.Charge);

        for (int i = 0; i < 30; i++)
        {
            w.Step(1.0f / 60.0f);
            loaded.Step(1.0f / 60.0f);
        }
        foreach (int id in ids)
            Assert.Equal(w.Bodies[id].Transform.Position, loaded.Bodies[id].Transform.Position);
    }
}


