using System.Collections.Generic;
using System.Numerics;
using Xunit;
using PhiEng.Dynamics;

namespace PhiEng.Tests;

/// <summary>
/// Managed SPH fluid tests: the C# solver (<see cref="FluidIntegration"/>)
/// must be deterministic, contain fluid in a tank, settle still columns,
/// conserve mass under adaptive resolution, and sink heavy phases.
/// </summary>
public class FluidTests
{
    private const float Dt = 1.0f / 60.0f;

    private static World2 Tank(int nx = 18, int ny = 12, float dx = 0.25f)
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(2.0f, -1.0f), new BoxShape2(new Vector2(3.0f, 0.5f)));
        world.AddStaticBody(new Vector2(-1.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        world.AddStaticBody(new Vector2(5.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        var pts = new List<Vector2>();
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                pts.Add(new Vector2(ix * dx, iy * dx));
            }
        }
        world.Fluid.AddParticles(pts);
        return world;
    }

    private static List<Vector2> Positions(World2 world)
    {
        var result = new List<Vector2>(world.Fluid.Particles.Count);
        foreach (var p in world.Fluid.Particles)
        {
            result.Add(p.Position);
        }
        return result;
    }

    [Fact]
    public void Fluid_Managed_Determinism()
    {
        List<Vector2> Run()
        {
            var world = Tank();
            for (int i = 0; i < 120; i++)
            {
                world.Step(Dt);
            }
            return Positions(world);
        }

        var a = Run();
        var b = Run();
        Assert.Equal(a.Count, b.Count);
        for (int i = 0; i < a.Count; i++)
        {
            Assert.Equal(a[i], b[i]);
        }
    }

    [Fact]
    public void Fluid_Managed_DamBreak_ContainedAndSettles()
    {
        var world = Tank();
        for (int i = 0; i < 300; i++)
        {
            world.Step(Dt);
        }
        Assert.Equal(18 * 12, world.Fluid.Particles.Count);
        float maxSpeed = 0.0f;
        foreach (var p in world.Fluid.Particles)
        {
            Assert.True(float.IsFinite(p.Position.X) && float.IsFinite(p.Position.Y), "particle position must stay finite");
            Assert.True(p.Position.X > -1.5f && p.Position.X < 5.5f, $"particle escaped tank sideways: {p.Position}");
            Assert.True(p.Position.Y > -0.6f, $"particle tunneled through floor: {p.Position}");
            maxSpeed = MathF.Max(maxSpeed, p.Velocity.Length());
        }
        Assert.True(maxSpeed < 30.0f, $"flow should stay bounded, maxSpeed={maxSpeed}");
    }

    [Fact]
    public void Fluid_Managed_TwoPhase_HeavySinks()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(2.0f, -1.0f), new BoxShape2(new Vector2(3.0f, 0.5f)));
        world.AddStaticBody(new Vector2(-1.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        world.AddStaticBody(new Vector2(5.0f, 2.0f), new BoxShape2(new Vector2(0.5f, 3.5f)));
        world.Fluid.Config.PhaseCount = 2;
        world.Fluid.Config.PhaseRestDensity = new float[] { 2.0f, 1.0f, 1.0f, 1.0f };
        world.Fluid.Config.PhaseMass = new float[] { 0.125f, 0.0625f, 0.0625f, 0.0625f };
        var light = new List<Vector2>();
        var lightPhases = new List<byte>();
        for (int iy = 0; iy < 8; iy++)
        {
            for (int ix = 0; ix < 18; ix++)
            {
                light.Add(new Vector2(ix * 0.25f, iy * 0.25f));
                lightPhases.Add(1);
            }
        }
        world.Fluid.AddParticlesPhase(light, lightPhases);
        var heavy = new List<Vector2>();
        var heavyPhases = new List<byte>();
        for (int iy = 8; iy < 12; iy++)
        {
            for (int ix = 6; ix < 12; ix++)
            {
                heavy.Add(new Vector2(ix * 0.25f, iy * 0.25f));
                heavyPhases.Add(0);
            }
        }
        world.Fluid.AddParticlesPhase(heavy, heavyPhases);

        float Center(byte phase)
        {
            float s = 0.0f;
            int n = 0;
            foreach (var p in world.Fluid.Particles)
            {
                if (p.Phase == phase)
                {
                    s += p.Position.Y;
                    n++;
                }
            }
            return n > 0 ? s / n : 0.0f;
        }

        float y0 = Center(0);
        for (int i = 0; i < 600; i++)
        {
            world.Step(Dt);
        }
        foreach (var p in world.Fluid.Particles)
        {
            Assert.True(float.IsFinite(p.Position.X) && float.IsFinite(p.Position.Y), "two-phase positions must stay finite");
        }
        float y1 = Center(0);
        Assert.True(y1 < y0, $"heavy droplet did not sink: {y0} -> {y1}");
    }

    [Fact]
    public void Fluid_Managed_AdaptiveResolution_ConservesMass()
    {
        var world = Tank(10, 6);
        float Mass()
        {
            float s = 0.0f;
            foreach (var p in world.Fluid.Particles)
            {
                s += p.Mass;
            }
            return s;
        }

        float m0 = Mass();
        world.Fluid.Config.AdaptiveResolution = true;
        for (int i = 0; i < 120; i++)
        {
            world.Step(Dt);
        }
        float m1 = Mass();
        Assert.True(MathF.Abs(m1 - m0) < 1e-3f * MathF.Max(1.0f, m0), $"adaptive resampling must conserve mass: {m0} -> {m1}");
        foreach (var p in world.Fluid.Particles)
        {
            Assert.True(float.IsFinite(p.Position.X) && float.IsFinite(p.Position.Y), "positions must stay finite");
        }
    }

    [Fact]
    public void Fluid_Managed_Turbulence_Deterministic()
    {
        List<Vector2> Run()
        {
            var world = Tank(10, 6);
            var ambient = world.Ambient;
            ambient.Turbulence.Enabled = true;
            ambient.Turbulence.Amplitude = 4.0f;
            ambient.Turbulence.Frequency = 0.5f;
            ambient.Turbulence.TimeScale = 1.0f;
            ambient.Turbulence.Seed = 7u;
            ambient.Turbulence.Octaves = 3;
            ambient.Turbulence.DivergenceFree = false;
            world.Ambient = ambient;
            for (int i = 0; i < 60; i++)
            {
                world.Step(Dt);
            }
            return Positions(world);
        }

        var a = Run();
        var b = Run();
        Assert.Equal(a.Count, b.Count);
        for (int i = 0; i < a.Count; i++)
        {
            Assert.Equal(a[i], b[i]);
        }
    }

    [Fact]
    public void Fluid_Managed_BoundaryParticles_Persisted()
    {
        var world = Tank(6, 8);
        world.Fluid.Config.BoundaryParticles = true;
        for (int i = 0; i < 10; i++)
        {
            world.Step(Dt);
        }
        Assert.Equal(1ul, world.Fluid.BoundaryRegenCount);
        for (int i = 0; i < 10; i++)
        {
            world.Step(Dt);
        }
        Assert.Equal(1ul, world.Fluid.BoundaryRegenCount);
        // Moving a static body invalidates the ghosts.
        var b0 = world.Bodies[0];
        var t = b0.Transform;
        t.Position += new Vector2(1.0f, 0.0f);
        b0.Transform = t;
        world.Step(Dt);
        Assert.Equal(2ul, world.Fluid.BoundaryRegenCount);
    }

    [Fact]
    public void Fluid3_Managed_DeterminismAndContainment()
    {
        List<Vector3> Run()
        {
            var world = new World3();
            world.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
            world.AddStaticBody(new Vector3(1.0f, -1.0f, 1.0f), new BoxShape3(new Vector3(2.0f, 0.5f, 2.0f)));
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
            world.Fluid.AddParticles(pts);
            for (int i = 0; i < 60; i++)
            {
                world.Step(Dt);
            }
            var result = new List<Vector3>(world.Fluid.Particles.Count);
            foreach (var p in world.Fluid.Particles)
            {
                Assert.True(float.IsFinite(p.Position.X) && float.IsFinite(p.Position.Y) && float.IsFinite(p.Position.Z), "3D positions must stay finite");
                result.Add(p.Position);
            }
            return result;
        }

        var a = Run();
        var b = Run();
        Assert.Equal(a.Count, b.Count);
        for (int i = 0; i < a.Count; i++)
        {
            Assert.Equal(a[i], b[i]);
        }
    }
}
