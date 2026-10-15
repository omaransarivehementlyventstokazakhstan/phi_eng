using System.Numerics;
using Xunit;
using PhiEng.Collision;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// 3D narrowphase + world tests: GJK separation/collision, support functions,
/// EPA penetration, 3D box drop and stack stability.
/// Mirrors the Rust tests in crates/collision/src/{gjk3d,epa}.rs and
/// crates/dynamics/src/world.rs.
/// </summary>
public class GjkEpa3DTests
{
    private const float Dt = 1.0f / 60.0f;

    [Fact]
    public void Gjk_SeparatedSpheres()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(3, 0, 0), 1.0f, dir);
        var (result, _, _) = GjkEpa3D.Gjk(a, b);
        Assert.Equal(GjkResult3D.Separated, result);
    }

    [Fact]
    public void Gjk_CollidingSpheres()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(1, 0, 0), 1.0f, dir);
        var (result, _, _) = GjkEpa3D.Gjk(a, b);
        Assert.Equal(GjkResult3D.Colliding, result);
    }

    [Fact]
    public void Support_Obb_AxisAligned()
    {
        var he = new Vector3(1, 2, 3);
        Assert.Equal(new Vector3(1, 0, 0), GjkEpa3D.SupportObb(Vector3.Zero, he, Quaternion.Identity, Vector3.UnitX));
        Assert.Equal(new Vector3(0, 2, 0), GjkEpa3D.SupportObb(Vector3.Zero, he, Quaternion.Identity, Vector3.UnitY));
        Assert.Equal(new Vector3(0, 0, 3), GjkEpa3D.SupportObb(Vector3.Zero, he, Quaternion.Identity, Vector3.UnitZ));
    }

    [Fact]
    public void Support_Capsule()
    {
        var a = new Vector3(0, -1, 0);
        var b = new Vector3(0, 1, 0);
        Assert.Equal(new Vector3(0.5f, 0, 0), GjkEpa3D.SupportCapsule(a, b, 0.5f, Vector3.UnitX));
        Assert.Equal(new Vector3(0, 1.5f, 0), GjkEpa3D.SupportCapsule(a, b, 0.5f, Vector3.UnitY));
    }

    [Fact]
    public void Epa_SphereSphere_Penetration()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(1.5f, 0, 0), 1.0f, dir);
        var hit = GjkEpa3D.GjkEpa(a, b);
        Assert.NotNull(hit);
        Assert.True(MathF.Abs(hit.Normal.X) > 0.99f, $"normal.x={hit.Normal.X}");
        Assert.True(MathF.Abs(hit.Depth - 0.5f) < 5e-3f, $"depth={hit.Depth}");
    }

    [Fact]
    public void Epa_ObbObb_Penetration()
    {
        var he = new Vector3(1, 1, 1);
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportObb(Vector3.Zero, he, Quaternion.Identity, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportObb(new Vector3(1.5f, 0, 0), he, Quaternion.Identity, dir);
        var hit = GjkEpa3D.GjkEpa(a, b);
        Assert.NotNull(hit);
        Assert.True(MathF.Abs(hit.Normal.X) > 0.99f, $"normal.x={hit.Normal.X}");
        Assert.True(MathF.Abs(hit.Depth - 0.5f) < 5e-3f, $"depth={hit.Depth}");
    }

    [Fact]
    public void OriginInTet_InsideOutsideDegenerate()
    {
        Assert.True(GjkEpa3D.OriginInTet(
            new Vector3(1, 0, 0), new Vector3(0, 1, 0),
            new Vector3(0, 0, 1), new Vector3(-1, -1, -1)));
        Assert.False(GjkEpa3D.OriginInTet(
            new Vector3(6, 0, 0), new Vector3(5, 1, 0),
            new Vector3(5, 0, 1), new Vector3(4, -1, -1)));
        Assert.False(GjkEpa3D.OriginInTet(
            new Vector3(1, 0, 0), new Vector3(0, 1, 0),
            new Vector3(-1, 0, 0), new Vector3(0, -1, 0)));
    }

    [Fact]
    public void Epa_Separated_ReturnsNull()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(5, 0, 0), 1.0f, dir);
        Assert.Null(GjkEpa3D.GjkEpa(a, b));
    }

    [Fact]
    public void Gjk_NoFalseSeparation_RotatedBoxes()
    {
        // Regression mirror of the Rust test: interpenetrating rotated
        // boxes (SAT depth 0.44) that GJK used to misreport as separated.
        var heA = new Vector3(0.58751166f, 1.9327657f, 1.3370173f);
        var heB = new Vector3(0.44751966f, 1.9226546f, 0.4482013f);
        var rotA = new Quaternion(-0.61496f, -0.22940487f, -0.7301727f, -0.1898564f);
        var rotB = new Quaternion(-0.3793444f, -0.05538883f, -0.46066985f, -0.8005081f);
        var centerB = new Vector3(0.8932357f, -0.9307329f, -1.6851844f);
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportObb(Vector3.Zero, heA, rotA, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportObb(centerB, heB, rotB, dir);
        var (result, _, _) = GjkEpa3D.Gjk(a, b);
        Assert.Equal(GjkResult3D.Colliding, result);
        var sat = Sat3D.ObbObb(Vector3.Zero, heA, rotA, centerB, heB, rotB);
        Assert.NotNull(sat);
        var hit = GjkEpa3D.GjkEpa(a, b);
        Assert.NotNull(hit);
        Assert.True(MathF.Abs(hit.Depth - sat.Value.Penetration) <= 0.02f + 0.05f * sat.Value.Penetration,
            $"depth mismatch: epa={hit.Depth} sat={sat.Value.Penetration}");
        Assert.True(Vector3.Dot(hit.Normal, sat.Value.Normal) > 0.7f,
            $"normal flip: epa={hit.Normal} sat={sat.Value.Normal}");
    }

    [Fact]
    public void SeedTetrahedron_StrictAndSeparated()
    {
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportSphere(Vector3.Zero, 1.0f, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportSphere(new Vector3(0.5f, 0, 0), 1.0f, dir);
        var tet = GjkEpa3D.SeedTetrahedron(a, b);
        Assert.NotNull(tet);
        Assert.Equal(4, tet.Count);
        Assert.True(GjkEpa3D.OriginInTetStrict(tet[0], tet[1], tet[2], tet[3], GjkEpa3D.StrictTetMargin));
        Func<Vector3, Vector3> far = dir => GjkEpa3D.SupportSphere(new Vector3(5, 0, 0), 1.0f, dir);
        Assert.Null(GjkEpa3D.SeedTetrahedron(a, far));
    }

    [Fact]
    public void Epa_NeedlePair_SkipsOrAgrees()
    {
        // Mirror of the Rust needle regression: SAT depth 0.003 but the
        // Minkowski inradius is below the seed margin, so EPA must either
        // skip (documented sliver policy) or agree with SAT — never
        // disagree. GJK's boolean must be Colliding unconditionally.
        var heA = new Vector3(1.4051881f, 0.80691856f, 1.9476976f);
        var heB = new Vector3(0.96561944f, 0.74839616f, 0.6716831f);
        var rotA = new Quaternion(0.10756003f, -0.15346055f, 0.48880064f, -0.85202974f);
        var rotB = new Quaternion(-0.7761683f, 0.58606684f, -0.0012072899f, 0.2325661f);
        var centerB = new Vector3(0.029467806f, 2.1654482f, -2.1280866f);
        Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportObb(Vector3.Zero, heA, rotA, dir);
        Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportObb(centerB, heB, rotB, dir);
        var (result, _, _) = GjkEpa3D.Gjk(a, b);
        Assert.Equal(GjkResult3D.Colliding, result);
        var sat = Sat3D.ObbObb(Vector3.Zero, heA, rotA, centerB, heB, rotB);
        Assert.NotNull(sat);
        Assert.True(MathF.Abs(sat.Value.Penetration - 0.0029f) < 0.001f, $"sat pen={sat.Value.Penetration}");
        var hit = GjkEpa3D.GjkEpa(a, b);
        if (hit != null)
        {
            Assert.True(MathF.Abs(hit.Depth - sat.Value.Penetration) <= 0.02f + 0.05f * sat.Value.Penetration,
                $"depth mismatch: epa={hit.Depth} sat={sat.Value.Penetration}");
            Assert.True(Vector3.Dot(hit.Normal, sat.Value.Normal) > 0.7f);
        }
    }

    [Fact]
    public void Epa_Fuzz_ObbVsSatReference()
    {
        // Mirror of the Rust 1000-case fuzz: deterministic xorshift64*
        // (own implementation — stable across .NET versions, unlike
        // System.Random). SAT is exact for boxes; any EPA disagreement
        // outside the grazing margins is a bug.
        ulong state = 0x9E3779B97F4A7C15ul;
        ulong Next() { ulong x = state; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; state = x; return x * 0x2545F4914F6CDD1Dul; }
        float Unit() => (Next() >> 40) * (1.0f / 16777216.0f);
        float Range(float lo, float hi) => lo + (hi - lo) * Unit();
        Vector3 UnitVec()
        {
            float z = Range(-1, 1);
            float t = Range(0, 2 * MathF.PI);
            float r = MathF.Max(0, MathF.Sqrt(1 - z * z));
            return new Vector3(r * MathF.Cos(t), r * MathF.Sin(t), z);
        }
        Quaternion RandRot() => Quaternion.CreateFromAxisAngle(UnitVec(), Range(0, 2 * MathF.PI));
        Vector3 RandHe() => new Vector3(Range(0.3f, 2.0f), Range(0.3f, 2.0f), Range(0.3f, 2.0f));

        int compared = 0, separated = 0, grazing = 0, slivers = 0;
        for (int i = 0; i < 1000; i++)
        {
            var heA = RandHe(); var heB = RandHe();
            var rotA = RandRot(); var rotB = RandRot();
            float offset = i % 2 == 0 ? Range(0, 2.5f) : Range(2.5f, 5.0f);
            var centerB = UnitVec() * offset;
            Func<Vector3, Vector3> a = dir => GjkEpa3D.SupportObb(Vector3.Zero, heA, rotA, dir);
            Func<Vector3, Vector3> b = dir => GjkEpa3D.SupportObb(centerB, heB, rotB, dir);
            var (result, _, dist) = GjkEpa3D.Gjk(a, b);
            if (result == GjkResult3D.Separated)
            {
                if (dist <= 1e-3f) { grazing++; continue; }
                Assert.Null(Sat3D.ObbObb(Vector3.Zero, heA, rotA, centerB, heB, rotB));
                Assert.Null(GjkEpa3D.GjkEpa(a, b));
                separated++;
            }
            else
            {
                var sat = Sat3D.ObbObb(Vector3.Zero, heA, rotA, centerB, heB, rotB);
                Assert.True(sat != null,
                    $"case {i}: GJK collide but SAT separate: heA={heA} heB={heB} rotA={rotA} rotB={rotB} cB={centerB} dist={dist}");
                if (sat.Value.Penetration < 2 * GjkEpa3D.StrictTetMargin) { grazing++; continue; }
                var hit = GjkEpa3D.GjkEpa(a, b);
                if (hit == null)
                {
                    // Needle overlap (Minkowski inradius below the seed
                    // margin): legitimate skip per the documented sliver
                    // policy. Box-needle penetration is inherently small,
                    // so None on a FAT overlap is a seed bug, not a sliver.
                    Assert.True(sat.Value.Penetration < 0.02f,
                        $"case {i}: EPA skip on fat overlap (pen={sat.Value.Penetration}): heA={heA} heB={heB} rotA={rotA} rotB={rotB} cB={centerB}");
                    slivers++;
                    continue;
                }
                Assert.True(Vector3.Dot(hit.Normal, sat.Value.Normal) > 0.7f,
                    $"case {i}: normal flip: epa={hit.Normal} sat={sat.Value.Normal}");
                float tol = 0.02f + 0.05f * sat.Value.Penetration;
                Assert.True(MathF.Abs(hit.Depth - sat.Value.Penetration) <= tol,
                    $"case {i}: depth mismatch: epa={hit.Depth} sat={sat.Value.Penetration}");
                compared++;
            }
        }
        Assert.True(compared > 200 && separated > 200 && slivers <= 20,
            $"weak fuzz coverage: compared={compared} separated={separated} grazing={grazing} slivers={slivers}");
    }

    [Fact]
    public void Solver3D_EffectiveMass_UsesWorldInertia()
    {
        // Non-cube box (he 1,2,3; mass 48) rotated 90 deg about Z: local
        // Y lands on world X, so world inverse inertia along X is 1/160.
        var anchor = RigidBody3.StaticBody(0);
        var b = new RigidBody3(1);
        b.SetShape(new BoxShape3(new Vector3(1.0f, 2.0f, 3.0f)), 1.0f);
        var t = b.Transform;
        t.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI / 2.0f);
        b.Transform = t;
        b.Velocity = new Vector3(0, 0, -1);
        var bodies = new List<RigidBody3> { anchor, b };
        var constraints = new List<ContactConstraint3D>
        {
            new ContactConstraint3D(0, 1, new Vector3(0, 1, 0), Vector3.UnitZ, 0.0f, 0.5f, 0.0f),
        };
        // Single iteration: exactly one lambda application, so the
        // response equals the effective-mass prediction.
        new Solver3D { Iterations = 1 }.Solve(bodies, constraints, Dt);
        float vz = bodies[1].Velocity.Z;
        Assert.True(MathF.Abs(vz - (-1.0f + 0.76923f)) < 0.005f, $"world-frame inertia response wrong: vz={vz}");
        float wx = bodies[1].AngularVelocity.X;
        Assert.True(MathF.Abs(wx - 36.923f / 160.0f) < 0.005f, $"angular response wrong: wx={wx}");
    }

    [Fact]
    public void RigidBody3_Rotation_IntegratesInWorldFrame()
    {
        var body = new RigidBody3(0);
        var t = body.Transform;
        t.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, MathF.PI / 2.0f);
        body.Transform = t;
        body.AngularVelocity = Vector3.UnitZ;
        body.Integrate(0.01f);
        var marker = body.Transform.TransformPoint(Vector3.UnitZ);
        Assert.True(marker.Y > 0.005f, $"marker must swing toward +Y, got {marker}");
        Assert.True(MathF.Abs(marker.X - 1.0f) < 0.01f, $"marker x should stay ~1, got {marker}");
    }

    [Fact]
    public void Solver3D_EarlyExit_OnRestingContact()
    {
        var anchor = RigidBody3.StaticBody(0);
        var b = new RigidBody3(1);
        b.SetShape(new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)), 1.0f);
        var t = b.Transform;
        t.Position = new Vector3(0, 0.5f, 0);
        b.Transform = t;
        var bodies = new List<RigidBody3> { anchor, b };
        var constraints = new List<ContactConstraint3D>
        {
            new ContactConstraint3D(0, 1, new Vector3(0, 0, 0), Vector3.UnitY, 0.001f, 0.5f, 0.0f),
        };
        int ran = new Solver3D().Solve(bodies, constraints, Dt);
        Assert.Equal(1, ran);
    }

    [Fact]
    public void World3_BoxDrop_RestsOnFloor()
    {
        var world = new World3();
        world.AddStaticBody(new Vector3(0, -1, 0), new BoxShape3(new Vector3(10, 0.5f, 10)));
        int box = world.AddDynamicBody(
            new Vector3(0, 3, 0),
            new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)),
            1.0f);
        for (int i = 0; i < 600; i++) world.Step(Dt);
        var b = world.Bodies[box];
        Assert.True(b.Transform.Position.Y > -0.5f, $"box tunneled: y={b.Transform.Position.Y}");
        Assert.True(b.Velocity.Length() < 1.0f, $"box should settle: v={b.Velocity}");
    }

    [Fact]
    public void World3_CapsuleDrop_RestsOnFloor()
    {
        // Regression mirror of the Rust capsule-drop test: the box-side
        // witness fix (closest box-surface point instead of raw support
        // midpoint) lives or dies here. Caught by F28 equivalence.
        var world = new World3();
        world.AddStaticBody(new Vector3(0, -1, 0), new BoxShape3(new Vector3(10, 0.5f, 10)));
        int cap = world.AddDynamicBody(
            new Vector3(0, 5, 0),
            new CapsuleShape3(0.3f, 0.5f),
            1.0f);
        for (int i = 0; i < 600; i++) world.Step(Dt);
        var b = world.Bodies[cap];
        Assert.True(b.Transform.Position.Y > -0.2f, $"capsule tunneled: y={b.Transform.Position.Y}");
        Assert.True(b.Transform.Position.Y < 1.5f, $"capsule never settled: y={b.Transform.Position.Y}");
        Assert.True(b.Velocity.Length() < 1.0f, $"capsule should settle: v={b.Velocity}");
    }

    [Fact]
    public void World3_SphereDrop_Falls()
    {
        var world = new World3();
        world.AddStaticBody(new Vector3(0, -1, 0), new BoxShape3(new Vector3(10, 0.5f, 10)));
        int ball = world.AddDynamicBody(new Vector3(0, 5, 0), new SphereShape3(0.5f), 1.0f);
        for (int i = 0; i < 120; i++) world.Step(Dt);
        Assert.True(world.Bodies[ball].Transform.Position.Y < 5.0f);
    }

    [Fact]
    public void World3_Stack_Stabilizes()
    {
        var world = new World3();
        world.AddStaticBody(new Vector3(0, -1, 0), new BoxShape3(new Vector3(10, 0.5f, 10)));
        var ids = new List<int>();
        for (int i = 0; i < 3; i++)
            ids.Add(world.AddDynamicBody(
                new Vector3(0, 0.05f + i * 1.02f, 0),
                new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)),
                1.0f));
        for (int i = 0; i < 600; i++) world.Step(Dt);
        for (int level = 0; level < ids.Count; level++)
        {
            var b = world.Bodies[ids[level]];
            Assert.True(b.Transform.Position.Y > level - 0.75f, $"level {level} tunneled: y={b.Transform.Position.Y}");
            Assert.True(b.Velocity.Length() < 1.0f, $"level {level} jittering: v={b.Velocity}");
        }
    }
}
