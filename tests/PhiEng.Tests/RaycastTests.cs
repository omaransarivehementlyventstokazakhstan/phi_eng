using System.Numerics;
using Xunit;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Ray-cast tests (X07–X16): a ray cast answers "what does this ray
/// hit first?". Mirrors <c>crates/engine/tests/raycast.rs</c> (enforced
/// by <c>tools/check_test_parity.py</c>).
/// </summary>
public class RaycastTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>A floor with a ball beside the ray and a box further on.</summary>
    private static (PhiNativeWorld World, int Floor, int Ball) Scene()
    {
        var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        int floor = w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = w.AddDynamicBall(new Vector2(5.0f, 2.0f), 0.5f, 1.0f);
        return (w, floor, ball);
    }

    /// <summary>X07: the ray finds the body in front of it.</summary>
    [Fact]
    public void X07_RaycastFindsTheBodyInFront()
    {
        var (w, floor, _) = Scene();
        using (w)
        {
            var hit = w.Raycast(new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
            Assert.NotNull(hit);
            Assert.Equal((uint)floor, hit!.Value.Body);
            Assert.True(System.Math.Abs(hit.Value.T - 10.5f) < 1e-4f, $"t={hit.Value.T}");
            Assert.True(System.Math.Abs(hit.Value.Point.Y + 0.5f) < 1e-4f);
        }
    }

    /// <summary>X08: a ray that misses everything hits nothing.</summary>
    [Fact]
    public void X08_RaycastMisses()
    {
        var (w, _, _) = Scene();
        using (w)
        {
            var hit = w.Raycast(new Vector2(100.0f, 100.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
            Assert.Null(hit);
        }
    }

    /// <summary>X09: the collision filter applies to a ray cast.</summary>
    [Fact]
    public void X09_RaycastRespectsTheCollisionFilter()
    {
        var (w, floor, _) = Scene();
        using (w)
        {
            // The floor moves to a group the query does not accept. A
            // group is one bit, so the floor's group is the flag 0x2 and
            // a mask that accepts it holds that same bit.
            w.SetBodyCollisionFilter(floor, 2u, 2u);
            var hit = w.Raycast(new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
            Assert.Null(hit);

            // A query on the floor's own group finds it.
            var found = w.Raycast(new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f, 2, 2);
            Assert.NotNull(found);
        }
    }

    /// <summary>X10: the nearest of two bodies wins.</summary>
    [Fact]
    public void X10_RaycastReturnsTheNearestBody()
    {
        using var w = new PhiNativeWorld();
        w.SetGravity(new Vector2(0.0f, -9.81f));
        int far = w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int near = w.AddStaticBody(new Vector2(0.0f, 4.0f), new Vector2(1.0f, 0.5f));
        var hit = w.Raycast(new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
        Assert.NotNull(hit);
        Assert.Equal((uint)near, hit!.Value.Body);
        Assert.NotEqual((uint)far, hit.Value.Body);
    }

    /// <summary>X11: the same in 3D, against a sphere and a box.</summary>
    [Fact]
    public void X11_RaycastIn3D()
    {
        using var w = new PhiNativeWorld3();
        w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
        int box = w.AddStaticBody(new Vector3(0.0f, 0.0f, -10.0f), new Vector3(10.0f, 0.5f, 10.0f));
        int sphere = w.AddDynamicBall(new Vector3(0.0f, 0.0f, 5.0f), 0.5f, 1.0f);
        var hit = w.Raycast(
            new Vector3(0.0f, 0.0f, 10.0f), new Vector3(0.0f, 0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
        Assert.NotNull(hit);
        Assert.Equal((uint)sphere, hit!.Value.Body);
        Assert.NotEqual((uint)box, hit.Value.Body);
        Assert.True(System.Math.Abs(hit.Value.T - 4.5f) < 1e-4f, $"t={hit.Value.T}");
        Assert.True(System.Math.Abs(hit.Value.Point.Z - 5.5f) < 1e-4f);
    }

    /// <summary>X12: the native bridge answers the same ray cast.</summary>
    [Fact]
    public void X12_FfiRaycast()
    {
        using var w = new PhiNativeWorld();
        int id = w.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(5.0f, 0.5f));
        var hit = w.Raycast(new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
        Assert.NotNull(hit);
        Assert.Equal((uint)id, hit!.Value.Body);
        Assert.True(System.Math.Abs(hit.Value.T - 10.5f) < 1e-3f, $"t={hit.Value.T}");

        // A ray that misses.
        var miss = w.Raycast(new Vector2(100.0f, 100.0f), new Vector2(0.0f, -1.0f), 100.0f, 1, uint.MaxValue);
        Assert.Null(miss);
    }

    /// <summary>
    /// X13: a ray that starts inside a shape does not hit it.
    ///
    /// The hit is for a ray that <em>enters</em>: a ray that starts
    /// inside would otherwise report the way out, with the normal
    /// pointing away from the ray.
    /// </summary>
    [Fact]
    public void X13_RaycastFromInsideAShapeMisses()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, 0.0f);
        w.AddStaticBody(new Vector2(0.0f, 0.0f), new Dynamics.CircleShape(1.0f));
        var hit = w.Raycast(
            new Vector2(0.0f, 0.0f), new Vector2(1.0f, 0.0f), float.MaxValue,
            Dynamics.QueryFilter.Everything);
        Assert.Null(hit);

        // The same in 3D.
        var w3 = new Dynamics.World3();
        w3.Gravity = new Vector3(0.0f, 0.0f, 0.0f);
        w3.AddStaticBody(new Vector3(0.0f, 0.0f, 0.0f), new Dynamics.SphereShape3(1.0f));
        var hit3 = w3.Raycast(
            new Vector3(0.0f, 0.0f, 0.0f), new Vector3(1.0f, 0.0f, 0.0f), float.MaxValue,
            Dynamics.QueryFilter.Everything);
        Assert.Null(hit3);
    }

    /// <summary>X14: a sensor is not solid: the ray passes through it.</summary>
    [Fact]
    public void X14_RaycastPassesThroughASensor()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, 0.0f);
        int sensor = w.AddStaticBody(
            new Vector2(0.0f, 0.0f), new Dynamics.BoxShape2(new Vector2(1.0f, 1.0f)));
        w.Bodies[sensor].IsSensor = true;
        int floor = w.AddStaticBody(
            new Vector2(0.0f, -5.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));
        var hit = w.Raycast(
            new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), float.MaxValue,
            Dynamics.QueryFilter.Everything);
        Assert.NotNull(hit);
        Assert.Equal((uint)floor, hit!.Value.Body);
    }

    /// <summary>X15: the ray has a range.</summary>
    [Fact]
    public void X15_RaycastRespectsItsRange()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, 0.0f);
        w.AddStaticBody(new Vector2(0.0f, 0.0f), new Dynamics.BoxShape2(new Vector2(1.0f, 1.0f)));
        var origin = new Vector2(0.0f, 10.0f);
        var direction = new Vector2(0.0f, -1.0f);
        Assert.Null(w.Raycast(origin, direction, 5.0f, Dynamics.QueryFilter.Everything));
        Assert.NotNull(w.Raycast(origin, direction, 50.0f, Dynamics.QueryFilter.Everything));
    }

    /// <summary>
    /// X16: two bodies on the same group collide whatever their masks say.
    ///
    /// This is Box2D's rule: the group overrides the masks.
    /// </summary>
    [Fact]
    public void X16_TheSameGroupOverridesTheMasks()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        int a = w.AddDynamicBody(
            new Vector2(0.0f, 0.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        int b = w.AddDynamicBody(
            new Vector2(0.9f, 0.0f), new Dynamics.CircleShape(0.5f), 1.0f);
        // Both on group 2, and neither mask accepts group 2.
        w.Bodies[a].CollisionGroup = 2;
        w.Bodies[a].CollisionMask = 1;
        w.Bodies[b].CollisionGroup = 2;
        w.Bodies[b].CollisionMask = 1;
        var before = w.Bodies[b].Transform.Position;
        for (int i = 0; i < 30; i++)
        {
            w.Step(Dt);
        }
        var after = w.Bodies[b].Transform.Position;
        Assert.True(
            (after - before).Length() > 1e-3f,
            "the two bodies on the same group must push each other apart");
        // The same rule, seen from the pair: the bodies did not sink
        // into each other, they were pushed apart.
        Assert.True(
            (w.Bodies[a].Transform.Position - w.Bodies[b].Transform.Position).Length() > 0.9f,
            "two bodies on the same group must not pass through each other");
    }

    /// <summary>
    /// The managed mirror answers the same ray as the native core (the
    /// managed world has its own narrowphase loop).
    /// </summary>
    [Fact]
    public void ManagedMirrorAnswersTheSameRay()
    {
        var w = new Dynamics.World2();
        w.Gravity = new Vector2(0.0f, -9.81f);
        int floor = w.AddStaticBody(
            new Vector2(0.0f, -1.0f), new Dynamics.BoxShape2(new Vector2(10.0f, 0.5f)));

        var hit = w.Raycast(
            new Vector2(0.0f, 10.0f), new Vector2(0.0f, -1.0f), 100.0f,
            Dynamics.QueryFilter.Everything);
        Assert.NotNull(hit);
        Assert.Equal((uint)floor, hit!.Value.Body);
        Assert.True(System.Math.Abs(hit.Value.T - 10.5f) < 1e-4f, $"t={hit.Value.T}");
    }
}
