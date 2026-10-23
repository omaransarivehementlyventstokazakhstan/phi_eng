using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Shape-overlap queries (o07–o10). Mirrors
/// <c>crates/engine/tests/shape_overlap.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class ShapeOverlapTests
{
    private static Matrix3x2 Rot(float angle)
    {
        var (c, s) = PhiEng.Math.DeterministicTrig.ComputeCosSin(angle);
        return new Matrix3x2(c, s, -s, c, 0.0f, 0.0f);
    }

    /// <summary>
    /// o07: the 2D overlap matrix — separated, touching, penetrating,
    /// and a small shape wholly inside a large one.
    /// </summary>
    [Fact]
    public void o07_The2DOverlapMatrix()
    {
        var id = Matrix3x2.Identity;
        var circle = new CircleShape(1.0f);
        Assert.False(ShapeQuery2.Overlap(circle, Vector2.Zero, id, circle, new Vector2(2.5f, 0.0f), id));
        Assert.True(ShapeQuery2.Overlap(circle, Vector2.Zero, id, circle, new Vector2(2.0f, 0.0f), id));
        Assert.True(ShapeQuery2.Overlap(circle, Vector2.Zero, id, circle, new Vector2(1.0f, 0.0f), id));

        // Deep containment: SAT alone would call this separated.
        Assert.True(ShapeQuery2.Overlap(
            new CircleShape(0.2f), Vector2.Zero, id,
            new BoxShape2(new Vector2(5.0f, 5.0f)), Vector2.Zero, id));

        var box = new BoxShape2(new Vector2(1.0f, 0.5f));
        var rot = Rot(MathF.PI / 4.0f);
        Assert.False(ShapeQuery2.Overlap(box, Vector2.Zero, id, box, new Vector2(3.0f, 0.0f), rot));
        Assert.True(ShapeQuery2.Overlap(box, Vector2.Zero, id, box, new Vector2(1.4f, 0.0f), rot));

        var tri = new PolygonShape(new List<Vector2> { Vector2.Zero, Vector2.UnitX, Vector2.UnitY });
        Assert.False(ShapeQuery2.Overlap(
            tri, Vector2.Zero, id, new CircleShape(0.2f), new Vector2(2.0f, 2.0f), id));
        Assert.True(ShapeQuery2.Overlap(
            tri, Vector2.Zero, id, new CircleShape(0.05f), new Vector2(0.2f, 0.2f), id));
    }

    /// <summary>
    /// o08: the 3D overlap matrix — spheres, a capsule, and a convex
    /// hull.
    /// </summary>
    [Fact]
    public void o08_The3DOverlapMatrix()
    {
        var id = Quaternion.Identity;
        var sphere = new SphereShape3(1.0f);
        Assert.False(ShapeQuery3.Overlap(sphere, Vector3.Zero, id, sphere, new Vector3(2.5f, 0.0f, 0.0f), id));
        Assert.True(ShapeQuery3.Overlap(sphere, Vector3.Zero, id, sphere, new Vector3(2.0f, 0.0f, 0.0f), id));

        // Deep containment through GJK plus the point fallback.
        Assert.True(ShapeQuery3.Overlap(
            new SphereShape3(0.1f), Vector3.Zero, id,
            new BoxShape3(new Vector3(5.0f, 5.0f, 5.0f)), Vector3.Zero, id));

        var capsule = new CapsuleShape3(0.5f, 1.0f);
        var small = new BoxShape3(new Vector3(0.2f, 0.2f, 0.2f));
        Assert.True(ShapeQuery3.Overlap(capsule, Vector3.Zero, id, small, new Vector3(0.6f, 0.0f, 0.0f), id));
        Assert.False(ShapeQuery3.Overlap(capsule, Vector3.Zero, id, small, new Vector3(2.0f, 0.0f, 0.0f), id));

        var hull = new ConvexHullShape3(new List<Vector3>
        {
            Vector3.Zero, Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ,
        });
        Assert.True(ShapeQuery3.Overlap(
            hull, Vector3.Zero, id, new SphereShape3(0.1f), new Vector3(0.2f, 0.2f, 0.2f), id));
    }

    /// <summary>
    /// o09: a world overlap query returns ids in ascending order,
    /// includes sensors, and honors the filter.
    /// </summary>
    [Fact]
    public void o09_TheWorldOverlapIsOrderedAndFiltered()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(Vector2.Zero, new CircleShape(1.0f));
        w.AddStaticBody(new Vector2(1.5f, 0.0f), new CircleShape(1.0f));
        w.Bodies[1].IsSensor = true;
        w.AddStaticBody(new Vector2(10.0f, 0.0f), new CircleShape(1.0f));

        Assert.Equal(new[] { 0, 1 }, w.OverlapShape(
            new CircleShape(0.5f), Vector2.Zero, Matrix3x2.Identity, QueryFilter.Everything));

        w.Bodies[1].CollisionGroup = 2;
        Assert.Equal(new[] { 1 }, w.OverlapShape(
            new CircleShape(0.5f), Vector2.Zero, Matrix3x2.Identity, new QueryFilter(1u, 2u)));
        Assert.Equal(new[] { 0 }, w.OverlapShape(
            new CircleShape(0.5f), Vector2.Zero, Matrix3x2.Identity, new QueryFilter(1u, 1u)));

        var w3 = new World3 { Gravity = Vector3.Zero };
        w3.AddStaticBody(Vector3.Zero, new SphereShape3(1.0f));
        w3.AddStaticBody(new Vector3(10.0f, 0.0f, 0.0f), new BoxShape3(new Vector3(1.0f, 1.0f, 1.0f)));
        Assert.Equal(new[] { 0 }, w3.OverlapShape(
            new SphereShape3(0.5f), Vector3.Zero, Quaternion.Identity, QueryFilter.Everything));
    }

    /// <summary>
    /// o10: overlap is symmetric — swapping the two shapes never
    /// changes the answer.
    /// </summary>
    [Fact]
    public void o10_OverlapIsSymmetric()
    {
        var id = Matrix3x2.Identity;
        var rot = Rot(0.7f);
        var cases = new (Shape2 A, Vector2 AP, Shape2 B, Vector2 BP)[]
        {
            (new CircleShape(0.5f), Vector2.Zero, new BoxShape2(new Vector2(3.0f, 3.0f)), Vector2.Zero),
            (new CircleShape(1.0f), Vector2.Zero, new CircleShape(1.0f), new Vector2(1.9f, 0.0f)),
            (new PolygonShape(new List<Vector2> { Vector2.Zero, Vector2.UnitX, Vector2.UnitY }),
                Vector2.Zero, new BoxShape2(new Vector2(0.5f, 0.5f)), new Vector2(0.2f, 0.2f)),
            (new BoxShape2(new Vector2(1.0f, 0.2f)), Vector2.Zero,
                new BoxShape2(new Vector2(1.0f, 0.2f)), new Vector2(2.0f, 0.0f)),
            (new BoxShape2(new Vector2(1.0f, 0.2f)), Vector2.Zero,
                new BoxShape2(new Vector2(1.0f, 0.2f)), new Vector2(1.9f, 0.0f)),
        };
        foreach (var (a, ap, b, bp) in cases)
        {
            Assert.Equal(
                ShapeQuery2.Overlap(a, ap, rot, b, bp, id),
                ShapeQuery2.Overlap(b, bp, id, a, ap, rot));
        }

        var q = DeterministicQuat(0.6f);
        var hull = new ConvexHullShape3(new List<Vector3>
        {
            Vector3.Zero, Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ,
        });
        var cases3 = new (Shape3 A, Vector3 AP, Shape3 B, Vector3 BP)[]
        {
            (new SphereShape3(0.1f), Vector3.Zero, new BoxShape3(new Vector3(3.0f, 3.0f, 3.0f)), Vector3.Zero),
            (hull, Vector3.Zero, new SphereShape3(0.2f), new Vector3(0.1f, 0.1f, 0.1f)),
            (new SphereShape3(1.0f), Vector3.Zero, new SphereShape3(1.0f), new Vector3(1.9f, 0.0f, 0.0f)),
            (new CapsuleShape3(0.4f, 0.8f), Vector3.Zero,
                new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)), new Vector3(1.0f, 0.0f, 0.0f)),
        };
        foreach (var (a, ap, b, bp) in cases3)
        {
            Assert.Equal(
                ShapeQuery3.Overlap(a, ap, q, b, bp, Quaternion.Identity),
                ShapeQuery3.Overlap(b, bp, Quaternion.Identity, a, ap, q));
        }
    }

    private static Quaternion DeterministicQuat(float angle)
    {
        var (c, s) = PhiEng.Math.DeterministicTrig.ComputeCosSin(angle * 0.5f);
        return new Quaternion(0.0f, s, 0.0f, c);
    }
}
