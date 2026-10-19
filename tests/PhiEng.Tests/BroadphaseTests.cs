using System.Numerics;
using Xunit;
using PhiEng.Native;
using PhiEng.Broadphase;

namespace PhiEng.Tests;

/// <summary>
/// Broad-phase tier tests (B01–B06): backend parity, Hybrid
/// auto-selection, and mid-run mode switching. Mirrored by the Rust core
/// tests in <c>crates/engine/tests/broadphase.rs</c> (enforced by
/// <c>tools/check_test_parity.py</c>).
/// </summary>
public class BroadphaseTests
{
    private const float Dt = 1.0f / 60.0f;

    /// <summary>
    /// Build an identical scene for every mode: ground plus a 5-row pyramid
    /// of boxes. Both sides of a comparison must come from this builder so
    /// the scenes are provably the same.
    /// </summary>
    private static void BuildPyramid(PhiNativeWorld w)
    {
        w.SetGravity(new Vector2(0.0f, -9.81f));
        w.SetSleepThresholds(0.0f, 0.0f);
        w.AddStaticBody(
            new Vector2(0.0f, -1.0f),
            new Vector2(20.0f, 0.5f));

        for (int row = 0; row < 5; row++)
        {
            int count = 7 - 2 * row;
            for (int c = 0; c < count; c++)
            {
                float y = row * 1.05f;
                float x = (c - (count - 1) / 2) * 1.05f;
                w.AddBoxBody(new Vector2(x, y), new Vector2(0.5f, 0.5f), 1.0f, false);
            }
        }
    }

    private static Vector2[] Fingerprint(PhiNativeWorld w, int count)
    {
        var result = new Vector2[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = w.GetBodyTransform(i).Position;
        }
        return result;
    }

    /// <summary>B01: every 2D backend produces a bit-identical trajectory.</summary>
    [Fact]
    public void B01_All2DModesBitIdentical()
    {
        var modes = new[]
        {
            BroadphaseMode.BruteForce,
            BroadphaseMode.SpatialHash,
            BroadphaseMode.DynamicAABBTree,
        };

        Vector2[]? reference = null;
        foreach (var mode in modes)
        {
            using var w = new PhiNativeWorld();
            BuildPyramid(w);
            w.SetBroadphaseMode(mode);
            for (int i = 0; i < 240; i++) w.Step(Dt);

            var fp = Fingerprint(w, w.BodyCount);
            if (reference == null)
            {
                reference = fp;
            }
            else
            {
                Assert.Equal(reference, fp);
            }
        }
    }

    /// <summary>B02: every 3D backend produces a bit-identical trajectory.</summary>
    [Fact]
    public void B02_All3DModesBitIdentical()
    {
        Vector3[]? reference = null;
        foreach (var mode in new[] { BroadphaseMode.BruteForce, BroadphaseMode.DynamicAABBTree })
        {
            using var w = new PhiNativeWorld3();
            w.SetGravity(new Vector3(0.0f, -9.81f, 0.0f));
            w.SetSleepThresholds(0.0f, 0.0f);
            w.AddStaticBody(
                new Vector3(0.0f, -1.0f, 0.0f),
                new Vector3(10.0f, 0.5f, 10.0f));
            for (int i = 0; i < 8; i++)
            {
                w.AddSphereBody(
                    new Vector3(i * 1.1f - 4.0f, 1.0f + i * 1.1f, 0.0f),
                    0.5f, 1.0f, false);
            }
            w.SetBroadphaseMode(mode);
            for (int i = 0; i < 240; i++) w.Step(Dt);

            var fp = new Vector3[w.BodyCount];
            for (int i = 0; i < w.BodyCount; i++)
            {
                fp[i] = w.GetBodyTransform(i).Position;
            }
            if (reference == null)
            {
                reference = fp;
            }
            else
            {
                Assert.Equal(reference, fp);
            }
        }
    }

    /// <summary>B03: Hybrid resolves from the live body count.</summary>
    [Fact]
    public void B03_HybridMatchesResolvedBackend()
    {
        foreach (int count in new[] { 3, 60, 200 })
        {
            var expected = BroadphaseMode.Hybrid.AutoSelect(count);

            using var hybrid = new PhiNativeWorld();
            BuildPyramid(hybrid);
            hybrid.SetBroadphaseMode(BroadphaseMode.Hybrid);
            for (int i = 0; i < 60; i++) hybrid.Step(Dt);

            using var explicitWorld = new PhiNativeWorld();
            BuildPyramid(explicitWorld);
            explicitWorld.SetBroadphaseMode(expected);
            for (int i = 0; i < 60; i++) explicitWorld.Step(Dt);

            int n = System.Math.Min(hybrid.BodyCount, explicitWorld.BodyCount);
            Assert.Equal(
                Fingerprint(explicitWorld, n),
                Fingerprint(hybrid, n));
        }
    }

    /// <summary>B04: default world keeps the pre-tier-system backend.</summary>
    [Fact]
    public void B04_DefaultIsSpatialHash()
    {
        using var w = new PhiNativeWorld();
        Assert.Equal(BroadphaseMode.SpatialHash, w.GetBroadphaseMode());

        using var w3 = new PhiNativeWorld3();
        Assert.Equal(BroadphaseMode.SpatialHash, w3.GetBroadphaseMode());
    }

    /// <summary>B05: switching mode mid-run preserves the trajectory.</summary>
    [Fact]
    public void B05_ModeSwitchMidRunIsTransparent()
    {
        using var a = new PhiNativeWorld();
        BuildPyramid(a);
        for (int i = 0; i < 120; i++) a.Step(Dt);

        using var b = new PhiNativeWorld();
        BuildPyramid(b);
        for (int i = 0; i < 120; i++) b.Step(Dt);
        b.SetBroadphaseMode(BroadphaseMode.DynamicAABBTree);

        for (int i = 0; i < 120; i++)
        {
            a.Step(Dt);
            b.Step(Dt);
        }

        Assert.Equal(Fingerprint(a, a.BodyCount), Fingerprint(b, b.BodyCount));
    }

    /// <summary>B06: mode round-trips through the FFI boundary.</summary>
    [Fact]
    public void B06_ModeRoundTripsThroughFfi()
    {
        using var w = new PhiNativeWorld();
        foreach (var mode in new[]
        {
            BroadphaseMode.BruteForce,
            BroadphaseMode.SpatialHash,
            BroadphaseMode.DynamicAABBTree,
            BroadphaseMode.Hybrid,
        })
        {
            w.SetBroadphaseMode(mode);
            Assert.Equal(mode, w.GetBroadphaseMode());
            // A step must not throw with any backend live.
            BuildPyramid(w);
            w.Step(Dt);
        }

        using var w3 = new PhiNativeWorld3();
        foreach (var mode in new[] { BroadphaseMode.BruteForce, BroadphaseMode.DynamicAABBTree })
        {
            w3.SetBroadphaseMode(mode);
            Assert.Equal(mode, w3.GetBroadphaseMode());
        }
    }
}
