using System.Numerics;
using Xunit;
using Xunit.Abstractions;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Cross-language equivalence (F28, C#-only by nature): the identical
/// fixed scene runs through the managed engine and through the Rust core
/// via the FFI bridge, and final states must agree within tolerance.
/// Bit-parity is NOT claimed (different float libraries); behavior parity
/// is. Scenes are gentle (drops and settles, no needles) so the tolerance
/// stays tight enough to catch real divergence.
/// </summary>
public class EquivalenceTests
{
    private const float Dt = 1.0f / 60.0f;
    private readonly ITestOutputHelper _output;

    public EquivalenceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static bool NativeAvailable() => OperatingSystem.IsWindows();

    [Fact]
    public void F28_Equivalence2D()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var managed = new World2();
        managed.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        int mBox = managed.AddDynamicBody(new Vector2(-1.0f, 2.0f), new BoxShape2(new Vector2(0.4f, 0.4f)), 1.0f);
        int mBall = managed.AddDynamicBody(new Vector2(1.0f, 3.0f), new CircleShape(0.35f), 1.5f);
        int mBox2 = managed.AddDynamicBody(new Vector2(0.0f, 5.0f), new BoxShape2(new Vector2(0.3f, 0.3f)), 1.0f);

        using var native = new PhiNativeWorld();
        native.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int nBox = native.AddDynamicBox(new Vector2(-1.0f, 2.0f), new Vector2(0.4f, 0.4f), 1.0f);
        int nBall = native.AddDynamicBall(new Vector2(1.0f, 3.0f), 0.35f, 1.5f);
        int nBox2 = native.AddDynamicBox(new Vector2(0.0f, 5.0f), new Vector2(0.3f, 0.3f), 1.0f);

        for (int i = 0; i < 300; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }

        foreach (var (m, n, name) in new[] { (mBox, nBox, "box"), (mBall, nBall, "ball"), (mBox2, nBox2, "box2") })
        {
            var mp = managed.Bodies[m].Transform.Position;
            var (np, nrot) = native.GetBodyTransform(n);
            var mv = managed.Bodies[m].Velocity;
            var nv = native.GetBodyVelocity(n);
            float dp = (mp - np).Length();
            float dv = (mv - nv).Length();
            float da = MathF.Abs(managed.Bodies[m].RotationAngle - nrot);
            da = MathF.Min(da, 2 * MathF.PI - da); // rotation wraps
            _output.WriteLine($"[f28-2d] {name}: dpos={dp} dvel={dv} dang={da}");
            // Tolerance: 1e-2 pos, 5e-2 vel, 2e-2 angle (radians).
            // Measured: 2D dpos ~1e-8 to 4e-9 (bitwise for box/ball), dvel 0.
            Assert.True(dp < 1e-2f, $"f28-2d {name}: positions diverged (tol 1e-2): {mp} vs {np}");
            Assert.True(dv < 5e-2f, $"f28-2d {name}: velocities diverged (tol 5e-2): {mv} vs {nv}");
            Assert.True(da < 2e-2f, $"f28-2d {name}: angles diverged (tol 2e-2 rad): {managed.Bodies[m].RotationAngle} vs {nrot}");
        }
    }

    [Fact]
    public void F28_Equivalence3D()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var managed = new World3();
        managed.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new BoxShape3(new Vector3(10.0f, 0.5f, 10.0f)));
        int mBox = managed.AddDynamicBody(new Vector3(-1.0f, 2.0f, 0.0f), new BoxShape3(new Vector3(0.4f, 0.4f, 0.4f)), 1.0f);
        int mBall = managed.AddDynamicBody(new Vector3(1.0f, 3.0f, 0.0f), new SphereShape3(0.35f), 1.0f);
        int mCap = managed.AddDynamicBody(new Vector3(0.0f, 5.0f, 0.5f), new CapsuleShape3(0.3f, 0.5f), 1.0f);

        using var native = new PhiNativeWorld3();
        native.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new Vector3(10.0f, 0.5f, 10.0f));
        int nBox = native.AddBoxBody(new Vector3(-1.0f, 2.0f, 0.0f), new Vector3(0.4f, 0.4f, 0.4f), 1.0f, isStatic: false);
        int nBall = native.AddDynamicBall(new Vector3(1.0f, 3.0f, 0.0f), 0.35f, 1.0f);
        int nCap = native.AddCapsuleBody(new Vector3(0.0f, 5.0f, 0.5f), 0.3f, 0.5f, 1.0f, isStatic: false);

        for (int i = 0; i < 300; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }

        foreach (var (m, n, name) in new[] { (mBox, nBox, "box"), (mBall, nBall, "ball"), (mCap, nCap, "capsule") })
        {
            var mp = managed.Bodies[m].Transform.Position;
            var (np, nq) = native.GetBodyTransform(n);
            var mv = managed.Bodies[m].Velocity;
            var nv = native.GetBodyVelocity(n);
            var mq = managed.Bodies[m].Transform.Rotation;
            if (Quaternion.Dot(mq, nq) < 0) nq = -nq; // double cover
            float dp = (mp - np).Length();
            float dv = (mv - nv).Length();
            float dq = (mq - nq).Length();
            _output.WriteLine($"[f28-3d] {name}: dpos={dp} dvel={dv} dquat={dq}");
            // Tolerances: 5e-2 pos, 2e-1 vel, 1e-1 quat (looser for 3D due to capsule EPA).
            // Measured: box/ball dpos ~2e-8, capsule dpos 3.9e-4 (within slop).
            Assert.True(dp < 5e-2f, $"f28-3d {name}: positions diverged (tol 5e-2): {mp} vs {np}");
            Assert.True(dv < 2e-1f, $"f28-3d {name}: velocities diverged (tol 2e-1): {mv} vs {nv}");
            Assert.True(dq < 1e-1f, $"f28-3d {name}: orientations diverged (tol 1e-1): {mq} vs {nq}");
        }
    }

    [Fact]
    public void Fluid_Native_Managed_BehaviorParity()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var pts = new List<Vector2>();
        for (int iy = 0; iy < 6; iy++)
        {
            for (int ix = 0; ix < 8; ix++)
            {
                pts.Add(new Vector2(ix * 0.25f, iy * 0.25f));
            }
        }

        var managed = new World2();
        managed.Gravity = new Vector2(0.0f, -9.81f);
        managed.AddStaticBody(new Vector2(2.0f, -1.0f), new BoxShape2(new Vector2(3.0f, 0.5f)));
        managed.AddStaticBody(new Vector2(-1.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        managed.AddStaticBody(new Vector2(5.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        managed.Fluid.AddParticles(pts);

        using var native = new PhiNativeWorld();
        native.AddStaticBody(new Vector2(2.0f, -1.0f), new Vector2(3.0f, 0.5f));
        // Native wrapper has box-only statics; side walls are thin boxes.
        native.AddStaticBody(new Vector2(-1.0f, 2.0f), new Vector2(0.5f, 3.5f));
        native.AddStaticBody(new Vector2(5.0f, 2.0f), new Vector2(0.5f, 3.5f));
        native.AddFluidParticles(pts);

        for (int i = 0; i < 60; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }

        var got = native.GetFluidParticles(pts.Count);
        Assert.Equal(managed.Fluid.Particles.Count, got.Count);
        for (int i = 0; i < got.Count; i++)
        {
            var mp = managed.Fluid.Particles[i].Position;
            float dp = (mp - got[i].Position).Length();
            var mv = managed.Fluid.Particles[i].Velocity;
            float dv = (mv - got[i].Velocity).Length();
            if (i < 3)
            {
                _output.WriteLine($"[fluid-parity] p{i}: dpos={dp} dvel={dv}");
            }
            Assert.True(dp < 1e-4f, $"fluid particle {i} positions diverged (tol 1e-4): {mp} vs {got[i].Position}");
            Assert.True(dv < 1e-3f, $"fluid particle {i} velocities diverged (tol 1e-3): {mv} vs {got[i].Velocity}");
        }
    }

    [Fact]
    public void Fluid3_Native_Managed_BehaviorParity()
    {
        if (!NativeAvailable())
        {
            _output.WriteLine("skipped: staged native library is win-x64 only");
            return;
        }

        var pts = new List<Vector3>();
        for (int iz = 0; iz < 3; iz++)
        {
            for (int iy = 0; iy < 3; iy++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    pts.Add(new Vector3(ix * 0.25f, iy * 0.25f, iz * 0.25f));
                }
            }
        }

        var managed = new World3();
        managed.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        managed.AddStaticBody(new Vector3(1.0f, -1.0f, 1.0f), new BoxShape3(new Vector3(2.0f, 0.5f, 2.0f)));
        managed.Fluid.AddParticles(pts);

        using var native = new PhiNativeWorld3();
        native.AddStaticBody(new Vector3(1.0f, -1.0f, 1.0f), new Vector3(2.0f, 0.5f, 2.0f));
        native.AddFluidParticles(pts);

        for (int i = 0; i < 30; i++)
        {
            managed.Step(Dt);
            native.Step(Dt);
        }

        var got = native.GetFluidParticles(pts.Count);
        Assert.Equal(managed.Fluid.Particles.Count, got.Count);
        for (int i = 0; i < got.Count; i++)
        {
            var mp = managed.Fluid.Particles[i].Position;
            float dp = (mp - got[i].Position).Length();
            if (i < 3)
            {
                _output.WriteLine($"[fluid3-parity] p{i}: dpos={dp}");
            }
            Assert.True(dp < 1e-4f, $"3D fluid particle {i} positions diverged (tol 1e-4): {mp} vs {got[i].Position}");
        }
    }
}
