using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Point and closest-point shape queries (o01–o06). Mirrors
/// <c>crates/engine/tests/shape_query.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class ShapeQueryTests
{
    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    private static Matrix3x2 Rot90()
    {
        var (c, s) = PhiEng.Math.DeterministicTrig.ComputeCosSin(MathF.PI / 2.0f);
        return new Matrix3x2(c, s, -s, c, 0.0f, 0.0f);
    }

    /// <summary>
    /// o01: containment is exact for a circle, a rotated box and a
    /// polygon, and does not depend on the polygon's winding.
    /// </summary>
    [Fact]
    public void o01_PointContainmentIsExact()
    {
        var circle = new CircleShape(2.0f);
        Assert.False(ShapeQuery2.PointInShape(circle, Vector2.Zero, Matrix3x2.Identity, new Vector2(2.01f, 0.0f)));
        Assert.True(ShapeQuery2.PointInShape(circle, Vector2.Zero, Matrix3x2.Identity, new Vector2(1.99f, 0.0f)));
        Assert.True(ShapeQuery2.PointInShape(circle, Vector2.Zero, Matrix3x2.Identity, new Vector2(2.0f, 0.0f)));

        var box = new BoxShape2(new Vector2(1.0f, 0.5f));
        var rot = Rot90();
        Assert.False(ShapeQuery2.PointInShape(box, Vector2.Zero, rot, new Vector2(0.6f, 0.0f)));
        Assert.True(ShapeQuery2.PointInShape(box, Vector2.Zero, rot, new Vector2(0.0f, 0.9f)));

        var ccw = new PolygonShape(new List<Vector2> { Vector2.Zero, Vector2.UnitX, Vector2.UnitY });
        var cw = new PolygonShape(new List<Vector2> { Vector2.Zero, Vector2.UnitY, Vector2.UnitX });
        var inside = new Vector2(0.2f, 0.2f);
        var outside = new Vector2(0.8f, 0.8f);
        foreach (var s in new[] { ccw, cw })
        {
            Assert.True(ShapeQuery2.PointInShape(s, Vector2.Zero, Matrix3x2.Identity, inside));
            Assert.False(ShapeQuery2.PointInShape(s, Vector2.Zero, Matrix3x2.Identity, outside));
        }
    }

    /// <summary>
    /// o02: the 2D closest point returns the surface point, the outward
    /// normal, and the distance — with the inside convention.
    /// </summary>
    [Fact]
    public void o02_The2DClosestPointIsAnalytic()
    {
        var circle = new CircleShape(2.0f);
        var split = ShapeQuery2.ClosestPoint(circle, Vector2.Zero, Matrix3x2.Identity, new Vector2(5.0f, 0.0f));
        Assert.NotNull(split);
        var outHit = split!.Value;
        Assert.False(outHit.Inside);
        Assert.Equal(0x40400000u, Bits(outHit.Distance));
        Assert.Equal(Bits(2.0f), Bits(outHit.Point.X));
        Assert.Equal(0x00000000u, Bits(outHit.Normal.Y));

        var inside = ShapeQuery2.ClosestPoint(circle, Vector2.Zero, Matrix3x2.Identity, new Vector2(1.0f, 0.0f))!.Value;
        Assert.True(inside.Inside);
        Assert.Equal(0.0f, inside.Distance);
        Assert.Equal(new Vector2(1.0f, 0.0f), inside.Point);
        Assert.Equal(Vector2.UnitX, inside.Normal);

        var box = new BoxShape2(new Vector2(1.0f, 0.5f));
        var rot = Rot90();
        var boxOut = ShapeQuery2.ClosestPoint(box, Vector2.Zero, rot, new Vector2(2.0f, 0.0f))!.Value;
        Assert.False(boxOut.Inside);
        Assert.True(MathF.Abs(boxOut.Distance - 1.5f) < 1e-6f);
        Assert.True(MathF.Abs(boxOut.Point.X - 0.5f) < 1e-6f);

        var boxIn = ShapeQuery2.ClosestPoint(box, Vector2.Zero, rot, new Vector2(0.2f, 0.1f))!.Value;
        Assert.True(boxIn.Inside);
        Assert.Equal(new Vector2(0.2f, 0.1f), boxIn.Point);
        Assert.True((boxIn.Normal - Vector2.UnitX).Length() < 1e-6f);

        var tri = new PolygonShape(new List<Vector2> { Vector2.Zero, Vector2.UnitX, Vector2.UnitY });
        var triOut = ShapeQuery2.ClosestPoint(tri, Vector2.Zero, Matrix3x2.Identity, new Vector2(2.0f, 0.0f))!.Value;
        Assert.False(triOut.Inside);
        Assert.Equal(0x3f800000u, Bits(triOut.Distance));
        Assert.True((triOut.Point - Vector2.UnitX).Length() < 1e-6f);
    }

    /// <summary>
    /// o03: 3D containment and closest points for the sphere, the box
    /// and the capsule (axis along local Y).
    /// </summary>
    [Fact]
    public void o03_The3DQueriesAreAnalytic()
    {
        var q = Quaternion.Identity;
        var sphere = new SphereShape3(2.0f);
        var outHit = ShapeQuery3.ClosestPoint(sphere, Vector3.Zero, q, new Vector3(5.0f, 0.0f, 0.0f))!.Value;
        Assert.False(outHit.Inside);
        Assert.Equal(0x40400000u, Bits(outHit.Distance));
        Assert.True(ShapeQuery3.PointInShape(sphere, Vector3.Zero, q, new Vector3(1.0f, 0.0f, 0.0f)));

        var capsule = new CapsuleShape3(0.5f, 1.0f);
        var top = ShapeQuery3.ClosestPoint(capsule, Vector3.Zero, q, new Vector3(0.0f, 3.0f, 0.0f))!.Value;
        Assert.False(top.Inside);
        Assert.True(MathF.Abs(top.Distance - 1.5f) < 1e-6f);
        Assert.Equal(Bits(1.5f), Bits(top.Point.Y));
        Assert.True((top.Normal - Vector3.UnitY).Length() < 1e-6f);
        Assert.True(ShapeQuery3.PointInShape(capsule, Vector3.Zero, q, new Vector3(0.49f, 0.0f, 0.0f)));
        Assert.False(ShapeQuery3.PointInShape(capsule, Vector3.Zero, q, new Vector3(0.51f, 0.0f, 0.0f)));

        var box = new BoxShape3(new Vector3(1.0f, 2.0f, 3.0f));
        var inside = ShapeQuery3.ClosestPoint(box, Vector3.Zero, q, new Vector3(0.9f, 0.0f, 0.0f))!.Value;
        Assert.True(inside.Inside);
        Assert.Equal(new Vector3(0.9f, 0.0f, 0.0f), inside.Point);
        Assert.True((inside.Normal - Vector3.UnitX).Length() < 1e-6f);
        var side = ShapeQuery3.ClosestPoint(box, Vector3.Zero, q, new Vector3(4.0f, 0.0f, 0.0f))!.Value;
        Assert.True(MathF.Abs(side.Distance - 3.0f) < 1e-6f);
    }

    /// <summary>
    /// o04: the unsupported convex shapes report a miss, matching the
    /// ray cast's documented gap. (The managed shape set has the hull;
    /// the Rust core additionally has the convex mesh, tested there.)
    /// </summary>
    [Fact]
    public void o04_ConvexShapesReportAMiss()
    {
        var q = Quaternion.Identity;
        var hull = new ConvexHullShape3(new List<Vector3>
        {
            Vector3.Zero, Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ,
        });
        Assert.Null(ShapeQuery3.ClosestPoint(hull, Vector3.Zero, q, Vector3.UnitX));
        Assert.False(ShapeQuery3.PointInShape(hull, Vector3.Zero, q, Vector3.Zero));
    }

    /// <summary>
    /// o05: a world point test returns ids in ascending order, includes
    /// sensors, and honors filters.
    /// </summary>
    [Fact]
    public void o05_TheWorldPointTestIsOrderedAndFiltered()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(Vector2.Zero, new CircleShape(1.0f));
        w.AddStaticBody(new Vector2(0.5f, 0.0f), new CircleShape(1.0f));
        w.Bodies[1].IsSensor = true;
        w.AddStaticBody(new Vector2(10.0f, 0.0f), new CircleShape(1.0f));

        Assert.Equal(new[] { 0, 1 }, w.PointTest(Vector2.Zero, QueryFilter.Everything));

        w.Bodies[1].CollisionGroup = 2;
        Assert.Equal(new[] { 1 }, w.PointTest(Vector2.Zero, new QueryFilter(1u, 2u)));
        Assert.Equal(new[] { 0 }, w.PointTest(Vector2.Zero, new QueryFilter(1u, 1u)));
        Assert.Equal(new[] { 0, 1 }, w.PointTest(Vector2.Zero, QueryFilter.Everything));

        var w3 = new World3 { Gravity = Vector3.Zero };
        w3.AddStaticBody(Vector3.Zero, new SphereShape3(1.0f));
        w3.AddStaticBody(new Vector3(4.0f, 0.0f, 0.0f), new BoxShape3(new Vector3(0.5f, 0.5f, 0.5f)));
        Assert.Equal(new[] { 0 }, w3.PointTest(new Vector3(0.2f, 0.0f, 0.0f), QueryFilter.Everything));
    }

    /// <summary>
    /// o06: the world closest point picks the nearest surface, respects
    /// the range, and prefers an inside hit.
    /// </summary>
    [Fact]
    public void o06_TheWorldClosestPointPicksTheNearest()
    {
        var w = new World2 { Gravity = Vector2.Zero };
        w.AddStaticBody(Vector2.Zero, new CircleShape(1.0f));
        w.AddStaticBody(new Vector2(0.5f, 0.0f), new CircleShape(1.0f));

        var hit = w.ClosestPoint(new Vector2(1.8f, 0.0f), QueryFilter.Everything, 5.0f);
        Assert.NotNull(hit);
        Assert.Equal(1, hit!.Value.Body);
        Assert.False(hit.Value.Inside);
        Assert.Equal(0x3e999998u, Bits(hit.Value.Distance));

        Assert.Null(w.ClosestPoint(new Vector2(1.8f, 0.0f), QueryFilter.Everything, 0.2f));

        var inside = w.ClosestPoint(new Vector2(1.4f, 0.0f), QueryFilter.Everything, 5.0f)!.Value;
        Assert.True(inside.Inside);
        Assert.Equal(1, inside.Body);
        Assert.Equal(0.0f, inside.Distance);

        var tie = w.ClosestPoint(new Vector2(0.9f, 0.0f), QueryFilter.Everything, 5.0f)!.Value;
        Assert.Equal(0, tie.Body);

        var w3 = new World3 { Gravity = Vector3.Zero };
        w3.AddStaticBody(Vector3.Zero, new SphereShape3(1.0f));
        var hit3 = w3.ClosestPoint(new Vector3(0.0f, 3.0f, 0.0f), QueryFilter.Everything, 10.0f)!.Value;
        Assert.Equal(0, hit3.Body);
        Assert.Equal(0x40000000u, Bits(hit3.Distance));
        Assert.Equal(Bits(1.0f), Bits(hit3.Point.Y));
    }
}
