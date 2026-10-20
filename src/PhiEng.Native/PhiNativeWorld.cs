using System.Numerics;
using PhiEng.Broadphase;

namespace PhiEng.Native;

/// <summary>
/// Managed wrapper over a native Rust <c>World2</c>. Mimics the local
/// <c>PhiEng.Dynamics.World2</c> API (add/step/query) so user scripts
/// never touch raw pointers. Owns the native handle; call <see cref="Dispose"/>
/// (or use <c>using</c>) to free the Rust world.
/// </summary>
public sealed class PhiNativeWorld : IDisposable
{
    /// <summary>Circle shape tag. Must match SHAPE_CIRCLE in ffi.rs.</summary>
    public const uint ShapeCircle = 0;
    /// <summary>Box shape tag. Must match SHAPE_BOX in ffi.rs.</summary>
    public const uint ShapeBox = 1;

    private IntPtr _handle;
    private bool _disposed;
    // Serializes all native calls (Rust step takes &mut — concurrent steps
    // would be native UB) and closes the Dispose/Step check-then-use race:
    // every public method holds the gate across its check + native call.
    private readonly object _gate = new();

    /// <summary>Gravity applied by the native world (defaults to (0, -9.81)).</summary>
    public Vector2 Gravity { get; private set; } = new Vector2(0.0f, -9.81f);

    public PhiNativeWorld()
    {
        _handle = NativeMethods.phieng_world_new();
        if (_handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("phieng_world_new returned null.");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed || _handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(PhiNativeWorld));
        }
    }

    public void SetGravity(Vector2 gravity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Gravity = gravity;
            NativeMethods.phieng_world_set_gravity(_handle, gravity.X, gravity.Y);
        }
    }

    /// <summary>
    /// Set the gravity force model + integration scheme (uniform /
    /// central / N-body / Barnes-Hut, Euler / Verlet). Validation
    /// mirrors Rust: unknown tags, non-finite G/attractor/mu, or
    /// negative softening/theta are rejected without mutating the world.
    /// </summary>
    public void SetGravityConfig(Dynamics.GravityConfig2 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_gravity_config(
                _handle, (uint)config.Mode, (uint)config.Integrator,
                config.BigG, config.Softening,
                config.Attractor.X, config.Attractor.Y,
                config.AttractorMu, config.Theta);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected gravity config (check tags and ranges).");
            }
        }
    }

    /// <summary>Read the gravity config from the native world.</summary>
    public unsafe Dynamics.GravityConfig2 GetGravityConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint mode, integ;
            float* values = stackalloc float[6];
            int ok = NativeMethods.phieng_world_get_gravity_config(
                _handle, &mode, &integ, values);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native gravity config read failed.");
            }
            return new Dynamics.GravityConfig2
            {
                Mode = (Dynamics.GravityMode)mode,
                Integrator = (Dynamics.Integrator)integ,
                BigG = values[0],
                Softening = values[1],
                Attractor = new Vector2(values[2], values[3]),
                AttractorMu = values[4],
                Theta = values[5],
            };
        }
    }

    /// <summary>
    /// Read the gravity config without throwing: returns <c>false</c>
    /// on invalid input.
    /// </summary>
    public unsafe bool TryGetGravityConfig(out Dynamics.GravityConfig2 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint mode, integ;
            float* values = stackalloc float[6];
            int ok = NativeMethods.phieng_world_get_gravity_config(
                _handle, &mode, &integ, values);
            config = new Dynamics.GravityConfig2
            {
                Mode = (Dynamics.GravityMode)mode,
                Integrator = (Dynamics.Integrator)integ,
                BigG = values[0],
                Softening = values[1],
                Attractor = new Vector2(values[2], values[3]),
                AttractorMu = values[4],
                Theta = values[5],
            };
            return ok != 0;
        }
    }

    /// <summary>Set the 2D ambient-force config (spring/SHM, drag, turbulence).</summary>
    public unsafe void SetAmbientConfig(Dynamics.AmbientConfig2 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = ToFfi(config);
            int ok = NativeMethods.phieng_world_set_ambient_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected ambient config (check tags and ranges).");
            }
        }
    }

    /// <summary>Read the ambient-force config from the native world.</summary>
    public unsafe Dynamics.AmbientConfig2 GetAmbientConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.AmbientConfig2Ffi ffi;
            int ok = NativeMethods.phieng_world_get_ambient_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native ambient config read failed.");
            }
            return FromFfi(ffi);
        }
    }

    /// <summary>Set the 2D CCD config (adaptive substepping).</summary>
    public unsafe void SetCcdConfig(Dynamics.CcdConfig config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.CcdConfigFfi
            {
                Enabled = config.Enabled ? 1u : 0u,
                Tier = (uint)config.Tier,
                MaxMovement = config.MaxMovement,
                MaxSubsteps = (uint)config.MaxSubsteps,
            };
            int ok = NativeMethods.phieng_world_set_ccd_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected CCD config (check ranges).");
            }
        }
    }

    /// <summary>Read the CCD config from the native world.</summary>
    public unsafe Dynamics.CcdConfig GetCcdConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.CcdConfigFfi ffi;
            int ok = NativeMethods.phieng_world_get_ccd_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native CCD config read failed.");
            }
            return new Dynamics.CcdConfig
            {
                Enabled = ffi.Enabled != 0,
                Tier = (Dynamics.CcdTier)ffi.Tier,
                MaxMovement = ffi.MaxMovement,
                MaxSubsteps = (int)ffi.MaxSubsteps,
            };
        }
    }

    /// <summary>Set the simulation clock (2D).</summary>
    public void SetTime(float t)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (NativeMethods.phieng_world_set_time(_handle, t) == 0)
            {
                throw new ArgumentException("Native world rejected time (non-finite).");
            }
        }
    }

    /// <summary>Read the simulation clock (2D).</summary>
    public unsafe float GetTime()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float t;
            if (NativeMethods.phieng_world_get_time(_handle, &t) == 0)
            {
                throw new InvalidOperationException("Native time read failed.");
            }
            return t;
        }
    }

    /// <summary>Set a body's ambient-force coupling multiplier (2D).</summary>
    public void SetBodyAmbientScale(int bodyId, float scale)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (NativeMethods.phieng_world_set_body_ambient_scale(_handle, (uint)bodyId, scale) == 0)
            {
                throw new ArgumentException("Native world rejected body ambient scale (unknown body or non-finite).");
            }
        }
    }

    private static NativeMethods.AmbientConfig2Ffi ToFfi(Dynamics.AmbientConfig2 c) => new()
    {
        SpringEnabled = c.Spring.Enabled ? 1u : 0u,
        SpringAx = c.Spring.Anchor.X,
        SpringAy = c.Spring.Anchor.Y,
        SpringStiffness = c.Spring.Stiffness,
        SpringDamping = c.Spring.Damping,
        SpringMaxDisplacement = c.Spring.MaxDisplacement,
        DragEnabled = c.Drag.Enabled ? 1u : 0u,
        DragLinear = c.Drag.Linear,
        DragQuadratic = c.Drag.Quadratic,
        AirDensity = c.Drag.AirDensity,
        DragAngular = c.Drag.Angular,
        WindX = c.Drag.Wind.X,
        WindY = c.Drag.Wind.Y,
        TurbulenceEnabled = c.Turbulence.Enabled ? 1u : 0u,
        TurbulenceAmplitude = c.Turbulence.Amplitude,
        TurbulenceFrequency = c.Turbulence.Frequency,
        TurbulenceTimeScale = c.Turbulence.TimeScale,
        TurbulenceSeed = c.Turbulence.Seed,
        TurbulenceOctaves = (uint)c.Turbulence.Octaves,
        TurbulenceDivergenceFree = c.Turbulence.DivergenceFree ? 1u : 0u,
    };

    private static Dynamics.AmbientConfig2 FromFfi(NativeMethods.AmbientConfig2Ffi f) => new()
    {
        Spring = new Dynamics.SpringConfig2
        {
            Enabled = f.SpringEnabled != 0,
            Anchor = new Vector2(f.SpringAx, f.SpringAy),
            Stiffness = f.SpringStiffness,
            Damping = f.SpringDamping,
            MaxDisplacement = f.SpringMaxDisplacement,
        },
        Drag = new Dynamics.DragConfig2
        {
            Enabled = f.DragEnabled != 0,
            Linear = f.DragLinear,
            Quadratic = f.DragQuadratic,
            AirDensity = f.AirDensity,
            Angular = f.DragAngular,
            Wind = new Vector2(f.WindX, f.WindY),
        },
        Turbulence = new Dynamics.TurbulenceConfig2
        {
            Enabled = f.TurbulenceEnabled != 0,
            Amplitude = f.TurbulenceAmplitude,
            Frequency = f.TurbulenceFrequency,
            TimeScale = f.TurbulenceTimeScale,
            Seed = f.TurbulenceSeed,
            Octaves = (int)f.TurbulenceOctaves,
            DivergenceFree = f.TurbulenceDivergenceFree != 0,
        },
    };

    /// <summary>Set the 2D SPH fluid config (EOS, viscosity, CFL, phases, ...).</summary>
    public unsafe void SetFluidConfig(Dynamics.FluidConfig2 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = ToFfi(config);
            int ok = NativeMethods.phieng_world_set_fluid_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected fluid config (check ranges).");
            }
        }
    }

    /// <summary>Read the 2D SPH fluid config from the native world.</summary>
    public unsafe Dynamics.FluidConfig2 GetFluidConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.FluidConfig2Ffi ffi;
            int ok = NativeMethods.phieng_world_get_fluid_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native fluid config read failed.");
            }
            return FromFfi(ffi);
        }
    }

    private static unsafe NativeMethods.FluidConfig2Ffi ToFfi(Dynamics.FluidConfig2 c)
    {
        var f = new NativeMethods.FluidConfig2Ffi
        {
            ParticleMass = c.ParticleMass,
            RestDensity = c.RestDensity,
            SmoothingLength = c.SmoothingLength,
            Stiffness = c.Stiffness,
            Gamma = c.Gamma,
            ArtificialViscosity = c.ArtificialViscosity,
            SurfaceTension = c.SurfaceTension,
            Cfl = c.Cfl,
            MaxSubsteps = (uint)c.MaxSubsteps,
            AdaptiveResolution = c.AdaptiveResolution ? (byte)1 : (byte)0,
            SplitDensityRatio = c.SplitDensityRatio,
            MergeDensityRatio = c.MergeDensityRatio,
            BoundaryParticles = c.BoundaryParticles ? (byte)1 : (byte)0,
            Viscosity = c.Viscosity,
            Drag = c.Drag,
            WallFriction = c.WallFriction,
            PhaseCount = c.PhaseCount,
            MaxVelocity = c.MaxVelocity,
            ParticleRadius = c.ParticleRadius,
            Enabled = c.Enabled ? (byte)1 : (byte)0,
        };
        for (int i = 0; i < 4; i++)
        {
            f.PhaseRestDensity[i] = c.PhaseRestDensity[i];
            f.PhaseMass[i] = c.PhaseMass[i];
        }
        return f;
    }

    private static unsafe Dynamics.FluidConfig2 FromFfi(NativeMethods.FluidConfig2Ffi f)
    {
        var rho = new float[4];
        var mass = new float[4];
        for (int i = 0; i < 4; i++)
        {
            rho[i] = f.PhaseRestDensity[i];
            mass[i] = f.PhaseMass[i];
        }
        return new Dynamics.FluidConfig2
        {
            ParticleMass = f.ParticleMass,
            RestDensity = f.RestDensity,
            SmoothingLength = f.SmoothingLength,
            Stiffness = f.Stiffness,
            Gamma = f.Gamma,
            ArtificialViscosity = f.ArtificialViscosity,
            SurfaceTension = f.SurfaceTension,
            Cfl = f.Cfl,
            MaxSubsteps = (int)f.MaxSubsteps,
            AdaptiveResolution = f.AdaptiveResolution != 0,
            SplitDensityRatio = f.SplitDensityRatio,
            MergeDensityRatio = f.MergeDensityRatio,
            BoundaryParticles = f.BoundaryParticles != 0,
            Viscosity = f.Viscosity,
            Drag = f.Drag,
            WallFriction = f.WallFriction,
            PhaseCount = f.PhaseCount,
            PhaseRestDensity = rho,
            PhaseMass = mass,
            MaxVelocity = f.MaxVelocity,
            ParticleRadius = f.ParticleRadius,
            Enabled = f.Enabled != 0,
        };
    }

    /// <summary>Set the 2D thermal config (conduction, radiation, convection).</summary>
    public unsafe void SetThermalConfig(Dynamics.ThermalConfig2 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.ThermalConfig2Ffi
            {
                Enabled = config.Enabled ? (byte)1 : (byte)0,
                AmbientTemperature = config.AmbientTemperature,
                StefanBoltzmann = config.StefanBoltzmann,
                ConvectionCoefficient = config.ConvectionCoefficient,
                RadiationEnabled = config.RadiationEnabled ? (byte)1 : (byte)0,
                ConductionEnabled = config.ConductionEnabled ? (byte)1 : (byte)0,
                ConvectionEnabled = config.ConvectionEnabled ? (byte)1 : (byte)0,
                MaxTemperature = config.MaxTemperature,
                MinTemperature = config.MinTemperature,
                RadiationRadiusFactor = config.RadiationRadiusFactor,
                MaxTemperatureRate = config.MaxTemperatureRate,
                RadiationOcclusion = config.RadiationOcclusion ? (byte)1 : (byte)0,
                ForcedConvectionCoefficient = config.ForcedConvectionCoefficient,
                ExpansionCoefficient = config.ExpansionCoefficient,
                ExpansionReferenceTemperature = config.ExpansionReferenceTemperature,
                ViewFactorModel = (uint)config.ViewFactorModel,
                SkyTemperature = config.SkyTemperature,
                GroundTemperature = config.GroundTemperature,
                OccluderTransmittance = config.OccluderTransmittance,
            };
            int ok = NativeMethods.phieng_world_set_thermal_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected thermal config (check ranges).");
            }
        }
    }

    /// <summary>Read the 2D thermal config from the native world.</summary>
    public unsafe Dynamics.ThermalConfig2 GetThermalConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.ThermalConfig2Ffi ffi;
            int ok = NativeMethods.phieng_world_get_thermal_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native thermal config read failed.");
            }
            return new Dynamics.ThermalConfig2
            {
                Enabled = ffi.Enabled != 0,
                AmbientTemperature = ffi.AmbientTemperature,
                StefanBoltzmann = ffi.StefanBoltzmann,
                ConvectionCoefficient = ffi.ConvectionCoefficient,
                RadiationEnabled = ffi.RadiationEnabled != 0,
                ConductionEnabled = ffi.ConductionEnabled != 0,
                ConvectionEnabled = ffi.ConvectionEnabled != 0,
                MaxTemperature = ffi.MaxTemperature,
                MinTemperature = ffi.MinTemperature,
                RadiationRadiusFactor = ffi.RadiationRadiusFactor,
                MaxTemperatureRate = ffi.MaxTemperatureRate,
                RadiationOcclusion = ffi.RadiationOcclusion != 0,
                ForcedConvectionCoefficient = ffi.ForcedConvectionCoefficient,
                ExpansionCoefficient = ffi.ExpansionCoefficient,
                ExpansionReferenceTemperature = ffi.ExpansionReferenceTemperature,
            };
        }
    }

    /// <summary>Set a body's temperature in Kelvin.</summary>
    public void SetBodyTemperature(int bodyId, float temperature)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_body_temperature(_handle, (uint)bodyId, temperature);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId} or non-finite temperature.");
            }
        }
    }

    /// <summary>Read a body's temperature in Kelvin.</summary>
    public unsafe float GetBodyTemperature(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float temp;
            int ok = NativeMethods.phieng_world_get_body_temperature(_handle, (uint)bodyId, &temp);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return temp;
        }
    }

    /// <summary>Set a body's thermal material properties.</summary>
    public void SetBodyThermalMaterial(
        int bodyId,
        float specificHeat,
        float thermalConductivity,
        float emissivity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_body_thermal_material(
                _handle, (uint)bodyId, specificHeat, thermalConductivity, emissivity);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId),
                    $"Unknown native body id {bodyId} or invalid thermal material values.");
            }
        }
    }

    /// <summary>Read a body's thermal material properties.</summary>
    public unsafe void GetBodyThermalMaterial(
        int bodyId,
        out float specificHeat,
        out float thermalConductivity,
        out float emissivity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float sh, k, e;
            int ok = NativeMethods.phieng_world_get_body_thermal_material(
                _handle, (uint)bodyId, &sh, &k, &e);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            specificHeat = sh;
            thermalConductivity = k;
            emissivity = e;
        }
    }

    /// <summary>Add SPH fluid particles. Returns the number added.</summary>
    public unsafe uint AddFluidParticles(System.Collections.Generic.IEnumerable<Vector2> positions)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var list = positions as System.Collections.Generic.IList<Vector2> ?? new System.Collections.Generic.List<Vector2>(positions);
            var flat = new float[list.Count * 2];
            for (int i = 0; i < list.Count; i++)
            {
                flat[i * 2] = list[i].X;
                flat[i * 2 + 1] = list[i].Y;
            }
            fixed (float* p = flat)
            {
                return NativeMethods.phieng_world_fluid_add(_handle, p, (uint)list.Count);
            }
        }
    }

    /// <summary>Add SPH fluid particles with per-particle phases. Returns the number added.</summary>
    public unsafe uint AddFluidParticlesPhase(System.Collections.Generic.IEnumerable<Vector2> positions, System.Collections.Generic.IEnumerable<byte> phases)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var posList = positions as System.Collections.Generic.IList<Vector2> ?? new System.Collections.Generic.List<Vector2>(positions);
            var phaseList = phases as System.Collections.Generic.IList<byte> ?? new System.Collections.Generic.List<byte>(phases);
            if (posList.Count != phaseList.Count)
            {
                throw new ArgumentException("positions and phases must have the same length.");
            }
            var flat = new float[posList.Count * 2];
            var ph = new byte[phaseList.Count];
            for (int i = 0; i < posList.Count; i++)
            {
                flat[i * 2] = posList[i].X;
                flat[i * 2 + 1] = posList[i].Y;
                ph[i] = phaseList[i];
            }
            fixed (float* p = flat)
            fixed (byte* q = ph)
            {
                return NativeMethods.phieng_world_fluid_add_phase(_handle, p, q, (uint)posList.Count);
            }
        }
    }

    /// <summary>Number of fluid particles in the native world.</summary>
    public uint FluidCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return NativeMethods.phieng_world_fluid_count(_handle);
            }
        }
    }

    /// <summary>Remove all fluid particles (keeps allocations).</summary>
    public void FluidClear()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.phieng_world_fluid_clear(_handle);
        }
    }

    /// <summary>Read up to <paramref name="max"/> fluid particles (position, velocity).</summary>
    public unsafe List<(Vector2 Position, Vector2 Velocity)> GetFluidParticles(int max)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var flat = new float[System.Math.Max(0, max) * 4];
            uint n;
            fixed (float* p = flat)
            {
                n = NativeMethods.phieng_world_fluid_get(_handle, p, (uint)System.Math.Max(0, max));
            }
            var result = new List<(Vector2, Vector2)>((int)n);
            for (int i = 0; i < n; i++)
            {
                result.Add((new Vector2(flat[i * 4], flat[i * 4 + 1]), new Vector2(flat[i * 4 + 2], flat[i * 4 + 3])));
            }
            return result;
        }
    }

    public int BodyCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();                return NativeMethods.phieng_world_body_count(_handle);
            }
        }
    }

    /// <summary>Add a circle body. Returns the body id.</summary>
    public int AddCircleBody(Vector2 position, float radius, float density, bool isStatic, float friction = Dynamics.PhysicsMaterial.DefaultFriction, float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int id = NativeMethods.phieng_world_add_body(
                _handle, position.X, position.Y,
                ShapeCircle, radius, 0.0f, density,
                (byte)(isStatic ? 1 : 0), friction, restitution);
            if (id < 0)
            {
                throw new ArgumentException("Native world rejected circle body (check radius/density).");
            }
            return id;
        }
    }

    /// <summary>Add a box body. Returns the body id.</summary>
    public int AddBoxBody(Vector2 position, Vector2 halfExtents, float density, bool isStatic, float friction = Dynamics.PhysicsMaterial.DefaultFriction, float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int id = NativeMethods.phieng_world_add_body(
                _handle, position.X, position.Y,
                ShapeBox, halfExtents.X, halfExtents.Y, density,
                (byte)(isStatic ? 1 : 0), friction, restitution);
            if (id < 0)
            {
                throw new ArgumentException("Native world rejected box body (check half extents/density).");
            }
            return id;
        }
    }

    /// <summary>Add a static (immovable) box, mirroring World2.AddStaticBody.</summary>
    public int AddStaticBody(Vector2 position, Vector2 halfExtents)
        => AddBoxBody(position, halfExtents, 1.0f, isStatic: true);

    /// <summary>Add a dynamic circle, mirroring the demo's ball spawning.</summary>
    public int AddDynamicBall(Vector2 position, float radius, float density)
        => AddCircleBody(position, radius, density, isStatic: false);

    /// <summary>Add a dynamic box.</summary>
    public int AddDynamicBox(Vector2 position, Vector2 halfExtents, float density)
        => AddBoxBody(position, halfExtents, density, isStatic: false);

    /// <summary>
    /// Bulk body creation: <paramref name="descs"/> are validated exactly
    /// like the scalar adders, in a single native transition. Returns one
    /// id per descriptor (<c>-1</c> for a rejected descriptor; valid
    /// siblings are still added — no rollback).
    /// </summary>
    public unsafe int[] AddBodies(BodyDesc2D[] descs)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(descs);
            var ids = new int[descs.Length];
            if (descs.Length == 0)
            {
                return ids;
            }
            fixed (BodyDesc2D* p = descs)
            fixed (int* q = ids)
            {
                NativeMethods.phieng_world_add_bodies(_handle, p, (uint)descs.Length, q);
            }
            return ids;
        }
    }

    /// <summary>Advance the native simulation by <paramref name="dt"/> seconds.</summary>
    public void Step(float dt)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.phieng_world_step(_handle, dt);
        }
    }

    /// <summary>Read a body's (position, rotationRadians).</summary>
    public unsafe (Vector2 Position, float Rotation) GetBodyTransform(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector2 pos;
            float rot;
            int ok = NativeMethods.phieng_world_get_body_transform(_handle, (uint)bodyId, &pos, &rot);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return (pos, rot);
        }
    }

    /// <summary>Read a body's linear velocity.</summary>
    /// <summary>Sets a body's linear velocity.</summary>
    public void SetBodyVelocity(int bodyId, Vector2 velocity)
    {
        if (bodyId < 0)
            throw new ArgumentOutOfRangeException(nameof(bodyId));
        NativeMethods.phieng_world_set_body_velocity(
            _handle,
            (uint)bodyId,
            velocity.X,
            velocity.Y);
    }

    /// <summary>
    /// Reads a body's shape bounding radius. Returns 0 for an unknown id
    /// or a shapeless body.
    /// </summary>
    public float GetBodyShapeRadius(int bodyId)
    {
        if (bodyId < 0)
            throw new ArgumentOutOfRangeException(nameof(bodyId));
        return NativeMethods.phieng_world_get_body_shape_radius(_handle, (uint)bodyId);
    }

    public unsafe Vector2 GetBodyVelocity(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector2 vel;
            int ok = NativeMethods.phieng_world_get_body_velocity(_handle, (uint)bodyId, &vel);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return vel;
        }
    }

    /// <summary>
    /// Read a body's (position, rotationRadians) without throwing: returns
    /// <c>false</c> for unknown/stale ids. Hot polling loops should prefer
    /// this (or <see cref="GetTransformsBulk"/>) over catching exceptions.
    /// </summary>
    public unsafe bool TryGetBodyTransform(int bodyId, out Vector2 position, out float rotation)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector2 pos;
            float rot;
            int ok = NativeMethods.phieng_world_get_body_transform(_handle, (uint)bodyId, &pos, &rot);
            position = pos;
            rotation = rot;
            return ok != 0;
        }
    }

    /// <summary>
    /// Read a body's linear velocity without throwing: returns <c>false</c>
    /// for unknown/stale ids.
    /// </summary>
    public unsafe bool TryGetBodyVelocity(int bodyId, out Vector2 velocity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector2 vel;
            int ok = NativeMethods.phieng_world_get_body_velocity(_handle, (uint)bodyId, &vel);
            velocity = vel;
            return ok != 0;
        }
    }

    /// <summary>
    /// Bulk transform read: <paramref name="ids"/> resolve in a single
    /// native transition instead of N getter calls. Slot <c>i</c> of every
    /// output belongs to <c>ids[i]</c> (index alignment is preserved, never
    /// compacted). Stale/unknown ids write <see cref="float.NaN"/> into
    /// their slot instead of aborting the batch. Pass <c>null</c> for
    /// <paramref name="velocities"/> to skip velocity readback.
    /// </summary>
    /// <returns>Number of ids that resolved to a live body.</returns>
    public unsafe int GetTransformsBulk(uint[] ids, Vector2[] positions, float[] rotations, Vector2[]? velocities = null)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(ids);
            ArgumentNullException.ThrowIfNull(positions);
            ArgumentNullException.ThrowIfNull(rotations);
            if (positions.Length < ids.Length)
            {
                throw new ArgumentException("positions is shorter than ids.", nameof(positions));
            }
            if (rotations.Length < ids.Length)
            {
                throw new ArgumentException("rotations is shorter than ids.", nameof(rotations));
            }
            if (velocities != null && velocities.Length < ids.Length)
            {
                throw new ArgumentException("velocities is shorter than ids.", nameof(velocities));
            }
            if (ids.Length == 0)
            {
                return 0;
            }
            fixed (uint* pIds = ids)
            fixed (Vector2* pPos = positions)
            fixed (float* pRot = rotations)
            {
                if (velocities == null)
                {
                    return (int)NativeMethods.phieng_world_get_transforms_bulk(
                        _handle, pIds, (uint)ids.Length, (float*)pPos, pRot, null);
                }
                fixed (Vector2* pVel = velocities)
                {
                    return (int)NativeMethods.phieng_world_get_transforms_bulk(
                        _handle, pIds, (uint)ids.Length, (float*)pPos, pRot, (float*)pVel);
                }
            }
        }
    }

    /// <summary>
    /// Replace a body's material (friction, density, restitution) in the
    /// native world; mass/inertia are recomputed from its shape.
    /// </summary>
    public void SetBodyMaterial(int bodyId, Dynamics.PhysicsMaterial material)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_body_material(
                _handle, (uint)bodyId, material.Friction, material.Density, material.Restitution);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Read a body's material from the native world.</summary>
    public unsafe Dynamics.PhysicsMaterial GetBodyMaterial(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector3 packed;
            int ok = NativeMethods.phieng_world_get_body_material(_handle, (uint)bodyId, &packed);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            // Packed as (friction, density, restitution); the constructor
            // sanitizes, mirroring the Rust side.
            return new Dynamics.PhysicsMaterial(packed.X, packed.Z, packed.Y);
        }
    }

    /// <summary>
    /// Set a body's gravitational coupling multiplier (1.0 full
    /// coupling, 0.0 immune, negative repels; finite only).
    /// </summary>
    public void SetBodyGravityScale(int bodyId, float scale)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_body_gravity_scale(_handle, (uint)bodyId, scale);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId} or non-finite scale.");
            }
        }
    }

    /// <summary>Read a body's gravitational coupling multiplier.</summary>
    public unsafe float GetBodyGravityScale(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float scale;
            int ok = NativeMethods.phieng_world_get_body_gravity_scale(_handle, (uint)bodyId, &scale);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return scale;
        }
    }

    /// <summary>
    /// Read a body's gravitational coupling multiplier without throwing:
    /// returns <c>false</c> for unknown/stale ids.
    /// </summary>
    public unsafe bool TryGetBodyGravityScale(int bodyId, out float scale)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float s;
            int ok = NativeMethods.phieng_world_get_body_gravity_scale(_handle, (uint)bodyId, &s);
            scale = s;
            return ok != 0;
        }
    }

    /// <summary>
    /// Select the broad-phase backend. <see cref="BroadphaseMode.Hybrid"/>
    /// re-resolves to a concrete backend on every step from the live body
    /// count; the other modes are fixed. Takes effect on the next step.
    /// </summary>
    public void SetBroadphaseMode(BroadphaseMode mode)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_broadphase_mode(_handle, (uint)mode);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(mode), $"Unknown broadphase mode tag {(uint)mode}.");
            }
        }
    }

    /// <summary>
    /// Select the narrowphase collision-tier ceiling. Pairs whose required
    /// tier exceeds <paramref name="config"/>'s ceiling are skipped (no
    /// collision generated). Disabled (the default) means full fidelity.
    /// </summary>
    public unsafe void SetCollisionTier(CollisionTierConfig config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.CollisionTierConfigFfi
            {
                Enabled = config.Enabled ? 1u : 0u,
                MaxTier = (uint)config.MaxTier,
            };
            int ok = NativeMethods.phieng_world_set_collision_tier(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(config), $"Unknown collision tier tag {(uint)config.MaxTier}.");
            }
        }
    }

    /// <summary>Read the narrowphase collision-tier ceiling.</summary>
    public unsafe CollisionTierConfig GetCollisionTier()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.CollisionTierConfigFfi ffi;
            int ok = NativeMethods.phieng_world_get_collision_tier(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native collision tier read failed.");
            }
            return new CollisionTierConfig
            {
                Enabled = ffi.Enabled != 0,
                MaxTier = (CollisionTier)ffi.MaxTier,
            };
        }
    }

    /// <summary>Read the configured broad-phase backend.</summary>
    public unsafe BroadphaseMode GetBroadphaseMode()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint mode;
            int ok = NativeMethods.phieng_world_get_broadphase_mode(_handle, &mode);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native broadphase mode read failed.");
            }
            return (BroadphaseMode)mode;
        }
    }

    /// <summary>
    /// Set the sleep velocity thresholds (0.0 disables sleeping).
    /// </summary>
    public void SetSleepThresholds(float linear, float angular)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_set_sleep_thresholds(_handle, linear, angular);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected sleep thresholds (must be finite, >= 0).");
            }
        }
    }

    /// <summary>Read the sleep velocity thresholds (linear, angular).</summary>
    public unsafe (float Linear, float Angular) GetSleepThresholds()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float* values = stackalloc float[2];
            int ok = NativeMethods.phieng_world_get_sleep_thresholds(_handle, values);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native sleep threshold read failed.");
            }
            return (values[0], values[1]);
        }
    }

    /// <summary>
    /// Remove a body by id. Remaining bodies are reindexed by the native
    /// core, mirroring <c>World2.RemoveBody</c>.
    /// </summary>
    public void RemoveBody(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_remove_body(_handle, (uint)bodyId);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Joint count in the native world.</summary>
    public int JointCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return NativeMethods.phieng_world_joint_count(_handle);
            }
        }
    }

    /// <summary>
    /// Add a joint via the generic tag + JSON bridge
    /// (tags: 0 distance, 1 hinge, 2 prismatic, 3 fixed; see
    /// <c>JOINT_*</c> in ffi.rs). <paramref name="json"/> carries the
    /// per-type params; returns the joint id.
    /// </summary>
    public unsafe int AddJoint(uint tag, int bodyA, int bodyB, string json)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(json);
            fixed (byte* p = utf8)
            {
                int id = NativeMethods.phieng_world_add_joint(
                    _handle, tag, (uint)bodyA, (uint)bodyB, p, (UIntPtr)utf8.Length);
                if (id < 0)
                {
                    throw new ArgumentException($"Native world rejected joint (tag={tag}, bodies=({bodyA},{bodyB})).");
                }
                return id;
            }
        }
    }

    /// <summary>Add a distance joint (rod). Returns the joint id.</summary>
    public int AddDistanceJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float restLength, float compliance = 0.0f)
        => AddJoint(0, bodyA, bodyB,
            System.Text.Json.JsonSerializer.Serialize(new
            {
                anchorA = new[] { anchorA.X, anchorA.Y },
                anchorB = new[] { anchorB.X, anchorB.Y },
                restLength,
                compliance,
            }));

    /// <summary>Add a hinge joint. <c>limitMin &gt; limitMax</c> means free.</summary>
    public int AddHingeJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float limitMin, float limitMax, float compliance = 0.0f)
        => AddJoint(1, bodyA, bodyB,
            System.Text.Json.JsonSerializer.Serialize(new
            {
                anchorA = new[] { anchorA.X, anchorA.Y },
                anchorB = new[] { anchorB.X, anchorB.Y },
                limitMin,
                limitMax,
                compliance,
            }));

    /// <summary>Add a prismatic (slider) joint.</summary>
    public int AddPrismaticJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        Vector2 axis, float limitMin, float limitMax, float compliance = 0.0f)
        => AddJoint(2, bodyA, bodyB,
            System.Text.Json.JsonSerializer.Serialize(new
            {
                anchorA = new[] { anchorA.X, anchorA.Y },
                anchorB = new[] { anchorB.X, anchorB.Y },
                axis = new[] { axis.X, axis.Y },
                limitMin,
                limitMax,
                compliance,
            }));

    /// <summary>Add a fixed (weld) joint.</summary>
    public int AddFixedJoint(int bodyA, int bodyB, Vector2 anchorA, Vector2 anchorB,
        float compliance = 0.0f)
        => AddJoint(3, bodyA, bodyB,
            System.Text.Json.JsonSerializer.Serialize(new
            {
                anchorA = new[] { anchorA.X, anchorA.Y },
                anchorB = new[] { anchorB.X, anchorB.Y },
                compliance,
            }));

    /// <summary>Remove a joint by id.</summary>
    public void RemoveJoint(int jointId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world_remove_joint(_handle, (uint)jointId);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(jointId), $"Unknown native joint id {jointId}.");
            }
        }
    }

    /// <summary>
    /// Save the native world to a file. <paramref name="format"/>: 0 JSON,
    /// 1 binary. <paramref name="mode"/>: 0 exact resume, 1 fresh start.
    /// </summary>
    public unsafe void Save(string path, uint format, uint mode)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(path);
            fixed (byte* p = utf8)
            {
                int rc = NativeMethods.phieng_world_save(
                    _handle, p, (UIntPtr)utf8.Length, format, mode);
                if (rc != 1)
                {
                    throw new IOException($"Native world save failed (code {rc}): {path}.");
                }
            }
        }
    }

    /// <summary>
    /// Load a native world from a file (format auto-detected).
    /// <paramref name="mode"/>: 0 exact resume, 1 fresh start.
    /// </summary>
    public static unsafe PhiNativeWorld Load(string path, uint mode)
    {
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(path);
        fixed (byte* p = utf8)
        {
            IntPtr handle = NativeMethods.phieng_world_load(p, (UIntPtr)utf8.Length, mode);
            if (handle == IntPtr.Zero)
            {
                throw new IOException($"Native world load failed: {path}.");
            }
            return new PhiNativeWorld(handle);
        }
    }

    private PhiNativeWorld(IntPtr handle)
    {
        _handle = handle;
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.phieng_world_free(_handle);
                _handle = IntPtr.Zero;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~PhiNativeWorld()
    {
        // Finalizer path: user code is gone, but a racing Step may still
        // hold the gate briefly; the lock keeps the free unambiguous.
        // (Dispose suppresses this, so it only runs for undisposed worlds.)
        lock (_gate)
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.phieng_world_free(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Set the 2D nuclear configuration and its nuclide table.
    /// </summary>
    /// <param name="config">The nuclear configuration.</param>
    public unsafe void SetNuclearConfig(Dynamics.NuclearConfig config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearConfigFfi
            {
                Enabled = config.Enabled ? (byte)1 : (byte)0,
                TimeScale = config.TimeScale,
                FissionEnergy = config.FissionEnergy,
                AmbientFlux = config.AmbientFlux,
                FissionRate = config.FissionRate,
                FuelPerFission = config.FuelPerFission,
                NeutronRange = config.NeutronRange,
                DoseRange = config.DoseRange,
                DelayedNeutronFraction = config.DelayedNeutronFraction,
                DelayedNeutronDelay = config.DelayedNeutronDelay,
                MaxActivity = config.MaxActivity,
                DosePerJoule = config.DosePerJoule,
                MaxDose = config.MaxDose,
                IsotopeCount = (uint)config.Isotopes.Count,
            };
            int ok = NativeMethods.phieng_world_set_nuclear_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclear config (check ranges).");
            }
            for (int i = 0; i < config.Isotopes.Count; i++)
            {
                SetIsotope((uint)i, config.Isotopes[i]);
            }
        }
    }

    /// <summary>Get the 2D nuclear configuration.</summary>
    public unsafe Dynamics.NuclearConfig GetNuclearConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearConfigFfi();
            int ok = NativeMethods.phieng_world_get_nuclear_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclear config request.");
            }
            return new Dynamics.NuclearConfig
            {
                Enabled = ffi.Enabled != 0,
                TimeScale = ffi.TimeScale,
                FissionEnergy = ffi.FissionEnergy,
                AmbientFlux = ffi.AmbientFlux,
                FissionRate = ffi.FissionRate,
                FuelPerFission = ffi.FuelPerFission,
                NeutronRange = ffi.NeutronRange,
                DoseRange = ffi.DoseRange,
                DelayedNeutronFraction = ffi.DelayedNeutronFraction,
                DelayedNeutronDelay = ffi.DelayedNeutronDelay,
                MaxActivity = ffi.MaxActivity,
                DosePerJoule = ffi.DosePerJoule,
                MaxDose = ffi.MaxDose,
            };
        }
    }

    /// <summary>Set one nuclide in the 2D table.</summary>
    /// <param name="index">The nuclide index (0 is the placeholder).</param>
    /// <param name="isotope">The nuclide.</param>
    public unsafe void SetIsotope(uint index, Dynamics.Isotope isotope)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.IsotopeFfi
            {
                HalfLife = isotope.HalfLife,
                DecayEnergy = isotope.DecayEnergy,
                IonizationFraction = isotope.IonizationFraction,
                Daughter = isotope.Daughter,
                AbsorptionCrossSection = isotope.AbsorptionCrossSection,
                FissionCrossSection = isotope.FissionCrossSection,
                NeutronYield = isotope.NeutronYield,
                IsFissile = isotope.IsFissile ? (byte)1 : (byte)0,
            };
            int ok = NativeMethods.phieng_world_set_isotope(_handle, index, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclide (check ranges).");
            }
        }
    }

    /// <summary>Set the 2D per-body nuclear state.</summary>
    /// <param name="bodyId">The body id.</param>
    /// <param name="state">The nuclear state.</param>
    public unsafe void SetBodyNuclearState(uint bodyId, Dynamics.NuclearState state)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearBodyFfi
            {
                Isotope = state.Isotope,
                Activity = state.Activity,
                Fuel = state.Fuel,
                Neutrons = state.Neutrons,
                DelayedNeutrons = state.DelayedNeutrons,
                Dose = state.Dose,
            };
            int ok = NativeMethods.phieng_world_set_body_nuclear_state(_handle, bodyId, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclear state (check ranges).");
            }
        }
    }

    /// <summary>Get the 2D per-body nuclear state.</summary>
    /// <param name="bodyId">The body id.</param>
    public unsafe Dynamics.NuclearState GetBodyNuclearState(uint bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearBodyFfi();
            int ok = NativeMethods.phieng_world_get_body_nuclear_state(_handle, bodyId, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclear state request.");
            }
            return new Dynamics.NuclearState
            {
                Isotope = (ushort)ffi.Isotope,
                Activity = ffi.Activity,
                Fuel = ffi.Fuel,
                Neutrons = ffi.Neutrons,
                DelayedNeutrons = ffi.DelayedNeutrons,
                Dose = ffi.Dose,
            };
        }
    }

}
