using System.Numerics;
using Xunit;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Native query-surface tests (F47–F50). Mirrors
/// <c>crates/engine/tests/ffi_queries.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class NativeQueryTests
{
    private const uint AllGroup = 1u;
    private const uint AllMask = uint.MaxValue;

    /// <summary>
    /// F47: a 2D point test reports every containing body, in id order,
    /// and honors the layer filter.
    /// </summary>
    [Fact]
    public void F47_PointTestReportsContainingBodies()
    {
        using var world = new PhiNativeWorld();
        int a = world.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, isStatic: true);
        int b = world.AddCircleBody(new Vector2(0.5f, 0.0f), 1.0f, 1.0f, isStatic: true);
        int far = world.AddCircleBody(new Vector2(10.0f, 0.0f), 1.0f, 1.0f, isStatic: true);
        Assert.True(a >= 0 && b >= 0 && far >= 0);

        Assert.Equal(new[] { a, b }, world.PointTest(0.0f, 0.0f, AllGroup, AllMask));
        // A filter that accepts only another layer sees nothing.
        Assert.Empty(world.PointTest(0.0f, 0.0f, 1u, 2u));
    }

    /// <summary>
    /// F48: a 3D point test over a sphere and a box.
    /// </summary>
    [Fact]
    public void F48_PointTest3D()
    {
        using var world = new PhiNativeWorld3();
        int sphere = world.AddSphereBody(Vector3.Zero, 1.0f, 1.0f, isStatic: true);
        int box = world.AddBoxBody(
            new Vector3(10.0f, 0.0f, 0.0f), new Vector3(0.5f, 0.5f, 0.5f), 1.0f, isStatic: true);
        Assert.True(sphere >= 0 && box >= 0);

        Assert.Equal(new[] { sphere }, world.PointTest(0.2f, 0.0f, 0.0f, AllGroup, AllMask));
    }

    /// <summary>
    /// F49: a 2D closest-point query fills the result and matches the
    /// Rust query.
    /// </summary>
    [Fact]
    public void F49_ClosestPointReportsTheSurface()
    {
        using var world = new PhiNativeWorld();
        int id = world.AddCircleBody(Vector2.Zero, 2.0f, 1.0f, isStatic: true);

        var hit = world.ClosestPoint(5.0f, 0.0f, 10.0f, AllGroup, AllMask);
        Assert.NotNull(hit);
        Assert.Equal(id, hit!.Value.Body);
        Assert.False(hit.Value.Inside);
        Assert.Equal(3.0f, hit.Value.Distance);
        Assert.Equal(new Vector2(2.0f, 0.0f), hit.Value.Point);
        Assert.Equal(new Vector2(1.0f, 0.0f), hit.Value.Normal);

        // Out of range: no hit.
        Assert.Null(world.ClosestPoint(5.0f, 0.0f, 1.0f, AllGroup, AllMask));
    }

    /// <summary>
    /// F50: a 3D closest-point query, including the inside convention.
    /// </summary>
    [Fact]
    public void F50_ClosestPoint3DAndBounds()
    {
        using var world = new PhiNativeWorld3();
        int id = world.AddSphereBody(Vector3.Zero, 1.0f, 1.0f, isStatic: true);

        var hit = world.ClosestPoint(0.0f, 3.0f, 0.0f, 10.0f, AllGroup, AllMask);
        Assert.NotNull(hit);
        Assert.Equal(id, hit!.Value.Body);
        Assert.Equal(2.0f, hit.Value.Distance);
        Assert.Equal(1.0f, hit.Value.Point.Y);
        Assert.Equal(1.0f, hit.Value.Normal.Y);

        // Inside: distance zero, escape normal.
        var inside = world.ClosestPoint(0.0f, 0.5f, 0.0f, 10.0f, AllGroup, AllMask);
        Assert.NotNull(inside);
        Assert.True(inside!.Value.Inside);
        Assert.Equal(0.0f, inside.Value.Distance);
    }
}
