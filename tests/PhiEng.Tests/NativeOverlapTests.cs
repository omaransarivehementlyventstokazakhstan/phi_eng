using System.Numerics;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Native shape-overlap tests (F51–F53). Mirrors
/// <c>crates/engine/tests/ffi_overlap.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class NativeOverlapTests
{
    private const uint Group = 1u;
    private const uint Mask = uint.MaxValue;

    /// <summary>
    /// F51: a 2D circle query finds the bodies it touches, matching the
    /// managed query.
    /// </summary>
    [Fact]
    public void F51_OverlapShapeReportsTouchingBodies()
    {
        using var world = new PhiNativeWorld();
        int a = world.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, isStatic: true);
        int b = world.AddCircleBody(new Vector2(1.5f, 0.0f), 1.0f, 1.0f, isStatic: true);
        int far = world.AddCircleBody(new Vector2(10.0f, 0.0f), 1.0f, 1.0f, isStatic: true);
        Assert.True(a >= 0 && b >= 0 && far >= 0);

        var shape = new CircleShape(0.6f);
        var hits = world.OverlapShape(shape, Vector2.Zero, 0.0f, Group, Mask);
        Assert.Equal(new[] { a, b }, hits);

        // A filter that accepts only another layer sees nothing.
        Assert.Empty(world.OverlapShape(shape, Vector2.Zero, 0.0f, 1u, 2u));
    }

    /// <summary>
    /// F52: a 3D box query with a rotation, against a sphere and a far
    /// box.
    /// </summary>
    [Fact]
    public void F52_OverlapShape3DWithRotation()
    {
        using var world = new PhiNativeWorld3();
        int sphere = world.AddSphereBody(Vector3.Zero, 1.0f, 1.0f, isStatic: true);
        int far = world.AddBoxBody(
            new Vector3(10.0f, 0.0f, 0.0f), new Vector3(0.5f, 0.5f, 0.5f), 1.0f, isStatic: true);
        Assert.True(sphere >= 0 && far >= 0);

        var shape = new BoxShape3(new Vector3(0.2f, 0.2f, 2.0f));
        float half = 0.70710677f;
        var rotation = new Quaternion(0.0f, half, 0.0f, half);
        var hits = world.OverlapShape(shape, Vector3.Zero, rotation, Group, Mask);
        Assert.Equal(new[] { sphere }, hits);

        // Out of reach: nothing.
        Assert.Empty(world.OverlapShape(shape, new Vector3(50.0f, 0.0f, 0.0f), rotation, Group, Mask));
    }

    /// <summary>
    /// F53: unsupported shapes are refused by the managed serialiser,
    /// and a rejected shape yields no hits through the bridge (the
    /// malformed-JSON cases are covered on the Rust side).
    /// </summary>
    [Fact]
    public void F53_OverlapShapeRejectsBadInput()
    {
        using var world = new PhiNativeWorld();
        world.AddCircleBody(Vector2.Zero, 1.0f, 1.0f, isStatic: true);
        // A negative radius is rejected by the strict parser.
        Assert.Empty(world.OverlapShape(new CircleShape(-1.0f), Vector2.Zero, 0.0f, Group, Mask));

        // The serialiser writes the invariant, round-trippable form...
        Assert.Contains("\"type\":\"circle\"", ShapeJson.From(new CircleShape(0.6f)));
        // ...and refuses a shape kind the bridge does not know.
        Assert.Throws<ArgumentException>(() => ShapeJson.From(new UnknownShape()));
    }
}

/// <summary>A shape the bridge has no JSON form for.</summary>
internal sealed class UnknownShape : Shape2
{
    public UnknownShape() { Type = Shape2Type.Circle; }
}
