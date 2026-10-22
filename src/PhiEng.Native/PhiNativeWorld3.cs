using System.Numerics;
using PhiEng.Broadphase;
using PhiEng.Quantum;

namespace PhiEng.Native;

/// <summary>
/// Managed wrapper over a native Rust <c>World3</c>. Mirrors
/// <c>PhiNativeWorld</c> for the 3D pipeline (GJK/EPA + 6-DOF solver).
/// </summary>
public sealed class PhiNativeWorld3 : IDisposable
{
    /// <summary>Sphere shape tag. Must match SHAPE3_SPHERE in ffi.rs.</summary>
    public const uint ShapeSphere = 0;
    /// <summary>Box shape tag. Must match SHAPE3_BOX in ffi.rs.</summary>
    public const uint ShapeBox = 1;
    /// <summary>Capsule shape tag. Must match SHAPE3_CAPSULE in ffi.rs.</summary>
    public const uint ShapeCapsule = 2;

    private IntPtr _handle;
    private bool _disposed;
    // Gate lock: same contract as PhiNativeWorld (serialize native
    // calls, close the Dispose/Step check-then-use race).
    private readonly object _gate = new();

    public Vector3 Gravity { get; private set; } = new Vector3(0.0f, -9.81f, 0.0f);

    public PhiNativeWorld3()
    {
        _handle = NativeMethods.phieng_world3_new();
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("phieng_world3_new returned null.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed || _handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(PhiNativeWorld3));
    }

    public void SetGravity(Vector3 gravity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Gravity = gravity;
            NativeMethods.phieng_world3_set_gravity(_handle, gravity.X, gravity.Y, gravity.Z);
        }
    }

    /// <summary>
    /// Set the 3D gravity force model + integration scheme (same
    /// contract as <c>PhiNativeWorld.SetGravityConfig</c>).
    /// </summary>
    public void SetGravityConfig(Dynamics.GravityConfig3 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_gravity_config(
                _handle, (uint)config.Mode, (uint)config.Integrator,
                config.BigG, config.Softening,
                config.Attractor.X, config.Attractor.Y, config.Attractor.Z,
                config.AttractorMu, config.Theta);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected gravity config (check tags and ranges).");
            }
        }
    }

    /// <summary>Read the 3D gravity config from the native world.</summary>
    public unsafe Dynamics.GravityConfig3 GetGravityConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint mode, integ;
            float* values = stackalloc float[7];
            int ok = NativeMethods.phieng_world3_get_gravity_config(
                _handle, &mode, &integ, values);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D gravity config read failed.");
            }
            return new Dynamics.GravityConfig3
            {
                Mode = (Dynamics.GravityMode)mode,
                Integrator = (Dynamics.Integrator)integ,
                BigG = values[0],
                Softening = values[1],
                Attractor = new Vector3(values[2], values[3], values[4]),
                AttractorMu = values[5],
                Theta = values[6],
            };
        }
    }

    /// <summary>
    /// Read the 3D gravity config without throwing: returns <c>false</c>
    /// on invalid input.
    /// </summary>
    public unsafe bool TryGetGravityConfig(out Dynamics.GravityConfig3 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint mode, integ;
            float* values = stackalloc float[7];
            int ok = NativeMethods.phieng_world3_get_gravity_config(
                _handle, &mode, &integ, values);
            config = new Dynamics.GravityConfig3
            {
                Mode = (Dynamics.GravityMode)mode,
                Integrator = (Dynamics.Integrator)integ,
                BigG = values[0],
                Softening = values[1],
                Attractor = new Vector3(values[2], values[3], values[4]),
                AttractorMu = values[5],
                Theta = values[6],
            };
            return ok != 0;
        }
    }

    /// <summary>Set the 3D ambient-force config (spring/SHM, drag, turbulence).</summary>
    public unsafe void SetAmbientConfig(Dynamics.AmbientConfig3 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = ToFfi(config);
            int ok = NativeMethods.phieng_world3_set_ambient_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected ambient config (check tags and ranges).");
            }
        }
    }

    /// <summary>Read the ambient-force config from the native 3D world.</summary>
    public unsafe Dynamics.AmbientConfig3 GetAmbientConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.AmbientConfig3Ffi ffi;
            int ok = NativeMethods.phieng_world3_get_ambient_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D ambient config read failed.");
            }
            return FromFfi(ffi);
        }
    }

    /// <summary>Set the 3D CCD config (adaptive substepping).</summary>
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
            int ok = NativeMethods.phieng_world3_set_ccd_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected CCD config (check ranges).");
            }
        }
    }

    /// <summary>Read the CCD config from the native 3D world.</summary>
    public unsafe Dynamics.CcdConfig GetCcdConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.CcdConfigFfi ffi;
            int ok = NativeMethods.phieng_world3_get_ccd_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D CCD config read failed.");
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

    /// <summary>Set the simulation clock (3D).</summary>
    public void SetTime(float t)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (NativeMethods.phieng_world3_set_time(_handle, t) == 0)
            {
                throw new ArgumentException("Native 3D world rejected time (non-finite).");
            }
        }
    }

    /// <summary>Read the simulation clock (3D).</summary>
    public unsafe float GetTime()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float t;
            if (NativeMethods.phieng_world3_get_time(_handle, &t) == 0)
            {
                throw new InvalidOperationException("Native 3D time read failed.");
            }
            return t;
        }
    }

    private static NativeMethods.AmbientConfig3Ffi ToFfi(Dynamics.AmbientConfig3 c) => new()
    {
        SpringEnabled = c.Spring.Enabled ? 1u : 0u,
        SpringAx = c.Spring.Anchor.X,
        SpringAy = c.Spring.Anchor.Y,
        SpringAz = c.Spring.Anchor.Z,
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
        WindZ = c.Drag.Wind.Z,
        TurbulenceEnabled = c.Turbulence.Enabled ? 1u : 0u,
        TurbulenceAmplitude = c.Turbulence.Amplitude,
        TurbulenceFrequency = c.Turbulence.Frequency,
        TurbulenceTimeScale = c.Turbulence.TimeScale,
        TurbulenceSeed = c.Turbulence.Seed,
        TurbulenceOctaves = (uint)c.Turbulence.Octaves,
        TurbulenceDivergenceFree = c.Turbulence.DivergenceFree ? 1u : 0u,
    };

    private static Dynamics.AmbientConfig3 FromFfi(NativeMethods.AmbientConfig3Ffi f) => new()
    {
        Spring = new Dynamics.SpringConfig3
        {
            Enabled = f.SpringEnabled != 0,
            Anchor = new Vector3(f.SpringAx, f.SpringAy, f.SpringAz),
            Stiffness = f.SpringStiffness,
            Damping = f.SpringDamping,
            MaxDisplacement = f.SpringMaxDisplacement,
        },
        Drag = new Dynamics.DragConfig3
        {
            Enabled = f.DragEnabled != 0,
            Linear = f.DragLinear,
            Quadratic = f.DragQuadratic,
            AirDensity = f.AirDensity,
            Angular = f.DragAngular,
            Wind = new Vector3(f.WindX, f.WindY, f.WindZ),
        },
        Turbulence = new Dynamics.TurbulenceConfig3
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

    /// <summary>Set the 3D SPH fluid config (EOS, viscosity, CFL, phases, ...).</summary>
    public unsafe void SetFluidConfig(Dynamics.FluidConfig3 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = ToFfi(config);
            int ok = NativeMethods.phieng_world3_set_fluid_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected fluid config (check ranges).");
            }
        }
    }

    /// <summary>Read the 3D SPH fluid config from the native world.</summary>
    public unsafe Dynamics.FluidConfig3 GetFluidConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.FluidConfig3Ffi ffi;
            int ok = NativeMethods.phieng_world3_get_fluid_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D fluid config read failed.");
            }
            return FromFfi(ffi);
        }
    }

    private static unsafe NativeMethods.FluidConfig3Ffi ToFfi(Dynamics.FluidConfig3 c)
    {
        var f = new NativeMethods.FluidConfig3Ffi
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

    private static unsafe Dynamics.FluidConfig3 FromFfi(NativeMethods.FluidConfig3Ffi f)
    {
        var rho = new float[4];
        var mass = new float[4];
        for (int i = 0; i < 4; i++)
        {
            rho[i] = f.PhaseRestDensity[i];
            mass[i] = f.PhaseMass[i];
        }
        return new Dynamics.FluidConfig3
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

    /// <summary>Set the 3D thermal config (conduction, radiation, convection).</summary>
    public unsafe void SetThermalConfig(Dynamics.ThermalConfig3 config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.ThermalConfig3Ffi
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
            int ok = NativeMethods.phieng_world3_set_thermal_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected thermal config (check ranges).");
            }
        }
    }

    /// <summary>Read the 3D thermal config from the native world.</summary>
    public unsafe Dynamics.ThermalConfig3 GetThermalConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.ThermalConfig3Ffi ffi;
            int ok = NativeMethods.phieng_world3_get_thermal_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D thermal config read failed.");
            }
            return new Dynamics.ThermalConfig3
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
            int ok = NativeMethods.phieng_world3_set_body_temperature(_handle, (uint)bodyId, temperature);
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
            int ok = NativeMethods.phieng_world3_get_body_temperature(_handle, (uint)bodyId, &temp);
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
            int ok = NativeMethods.phieng_world3_set_body_thermal_material(
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
            int ok = NativeMethods.phieng_world3_get_body_thermal_material(
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

    /// <summary>Add 3D SPH fluid particles. Returns the number added.</summary>
    public unsafe uint AddFluidParticles(System.Collections.Generic.IEnumerable<Vector3> positions)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var list = positions as System.Collections.Generic.IList<Vector3> ?? new System.Collections.Generic.List<Vector3>(positions);
            var flat = new float[list.Count * 3];
            for (int i = 0; i < list.Count; i++)
            {
                flat[i * 3] = list[i].X;
                flat[i * 3 + 1] = list[i].Y;
                flat[i * 3 + 2] = list[i].Z;
            }
            fixed (float* p = flat)
            {
                return NativeMethods.phieng_world3_fluid_add(_handle, p, (uint)list.Count);
            }
        }
    }

    /// <summary>Add 3D SPH fluid particles with per-particle phases. Returns the number added.</summary>
    public unsafe uint AddFluidParticlesPhase(System.Collections.Generic.IEnumerable<Vector3> positions, System.Collections.Generic.IEnumerable<byte> phases)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var posList = positions as System.Collections.Generic.IList<Vector3> ?? new System.Collections.Generic.List<Vector3>(positions);
            var phaseList = phases as System.Collections.Generic.IList<byte> ?? new System.Collections.Generic.List<byte>(phases);
            if (posList.Count != phaseList.Count)
            {
                throw new ArgumentException("positions and phases must have the same length.");
            }
            var flat = new float[posList.Count * 3];
            var ph = new byte[phaseList.Count];
            for (int i = 0; i < posList.Count; i++)
            {
                flat[i * 3] = posList[i].X;
                flat[i * 3 + 1] = posList[i].Y;
                flat[i * 3 + 2] = posList[i].Z;
                ph[i] = phaseList[i];
            }
            fixed (float* p = flat)
            fixed (byte* q = ph)
            {
                return NativeMethods.phieng_world3_fluid_add_phase(_handle, p, q, (uint)posList.Count);
            }
        }
    }

    /// <summary>Number of fluid particles in the native 3D world.</summary>
    public uint FluidCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return NativeMethods.phieng_world3_fluid_count(_handle);
            }
        }
    }

    /// <summary>Remove all 3D fluid particles (keeps allocations).</summary>
    public void FluidClear()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.phieng_world3_fluid_clear(_handle);
        }
    }

    /// <summary>Read up to <paramref name="max"/> 3D fluid particles (position, velocity).</summary>
    public unsafe List<(Vector3 Position, Vector3 Velocity)> GetFluidParticles(int max)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var flat = new float[System.Math.Max(0, max) * 6];
            uint n;
            fixed (float* p = flat)
            {
                n = NativeMethods.phieng_world3_fluid_get(_handle, p, (uint)System.Math.Max(0, max));
            }
            var result = new List<(Vector3, Vector3)>((int)n);
            for (int i = 0; i < n; i++)
            {
                result.Add((
                    new Vector3(flat[i * 6], flat[i * 6 + 1], flat[i * 6 + 2]),
                    new Vector3(flat[i * 6 + 3], flat[i * 6 + 4], flat[i * 6 + 5])));
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
                ThrowIfDisposed();
                return NativeMethods.phieng_world3_body_count(_handle);
            }
        }
    }

    public int AddSphereBody(Vector3 position, float radius, float density, bool isStatic,
        float friction = Dynamics.PhysicsMaterial.DefaultFriction,
        float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
    {
        int id;
        lock (_gate)
        {
            ThrowIfDisposed();
            id = NativeMethods.phieng_world3_add_body(
                _handle, position.X, position.Y, position.Z,
                ShapeSphere, radius, 0.0f, 0.0f, density,
                (byte)(isStatic ? 1 : 0), friction, restitution);
        }
        if (id < 0) throw new ArgumentException("Native 3D world rejected sphere body.");
        return id;
    }

    public int AddBoxBody(Vector3 position, Vector3 halfExtents, float density, bool isStatic,
        float friction = Dynamics.PhysicsMaterial.DefaultFriction,
        float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
    {
        int id;
        lock (_gate)
        {
            ThrowIfDisposed();
            id = NativeMethods.phieng_world3_add_body(
                _handle, position.X, position.Y, position.Z,
                ShapeBox, halfExtents.X, halfExtents.Y, halfExtents.Z, density,
                (byte)(isStatic ? 1 : 0), friction, restitution);
        }
        if (id < 0) throw new ArgumentException("Native 3D world rejected box body.");
        return id;
    }

    public int AddStaticBody(Vector3 position, Vector3 halfExtents)
        => AddBoxBody(position, halfExtents, 1.0f, isStatic: true);

    public int AddDynamicBall(Vector3 position, float radius, float density)
        => AddSphereBody(position, radius, density, isStatic: false);

    /// <summary>Add a capsule body (segment along local Y). Returns the body id.</summary>
    public int AddCapsuleBody(Vector3 position, float radius, float halfLength, float density, bool isStatic,
        float friction = Dynamics.PhysicsMaterial.DefaultFriction,
        float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
    {
        int id;
        lock (_gate)
        {
            ThrowIfDisposed();
            id = NativeMethods.phieng_world3_add_body(
                _handle, position.X, position.Y, position.Z,
                ShapeCapsule, radius, halfLength, 0.0f, density,
                (byte)(isStatic ? 1 : 0), friction, restitution);
        }
        if (id < 0) throw new ArgumentException("Native 3D world rejected capsule body.");
        return id;
    }

    /// <summary>
    /// Bulk 3D body creation: <paramref name="descs"/> are validated exactly
    /// like the scalar adders, in a single native transition. Returns one
    /// id per descriptor (<c>-1</c> for a rejected descriptor; valid
    /// siblings are still added — no rollback).
    /// </summary>
    public unsafe int[] AddBodies(BodyDesc3D[] descs)
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
            fixed (BodyDesc3D* p = descs)
            fixed (int* q = ids)
            {
                NativeMethods.phieng_world3_add_bodies(_handle, p, (uint)descs.Length, q);
            }
            return ids;
        }
    }

    public void Step(float dt)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.phieng_world3_step(_handle, dt);
        }
    }

    public unsafe (Vector3 Position, Quaternion Rotation) GetBodyTransform(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector3 pos;
            Quaternion rot;
            int ok = NativeMethods.phieng_world3_get_body_transform(_handle, (uint)bodyId, &pos, &rot);
            if (ok == 0)
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
            return (pos, rot);
        }
    }

    /// <summary>
    /// Reads a body's shape bounding radius. Returns 0 for an unknown id
    /// or a shapeless body.
    /// </summary>
    public float GetBodyShapeRadius(int bodyId)
    {
        if (bodyId < 0)
            throw new ArgumentOutOfRangeException(nameof(bodyId));
        return NativeMethods.phieng_world3_get_body_shape_radius(_handle, (uint)bodyId);
    }

    public unsafe Vector3 GetBodyVelocity(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector3 vel;
            int ok = NativeMethods.phieng_world3_get_body_velocity(_handle, (uint)bodyId, &vel);
            if (ok == 0)
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
            return vel;
        }
    }

    /// <summary>
    /// Read a 3D body's (position, rotation) without throwing: returns
    /// <c>false</c> for unknown/stale ids.
    /// </summary>
    public unsafe bool TryGetBodyTransform(int bodyId, out Vector3 position, out Quaternion rotation)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector3 pos;
            Quaternion rot;
            int ok = NativeMethods.phieng_world3_get_body_transform(_handle, (uint)bodyId, &pos, &rot);
            position = pos;
            rotation = rot;
            return ok != 0;
        }
    }

    /// <summary>
    /// Read a 3D body's linear velocity without throwing: returns
    /// <c>false</c> for unknown/stale ids.
    /// </summary>
    public unsafe bool TryGetBodyVelocity(int bodyId, out Vector3 velocity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            Vector3 vel;
            int ok = NativeMethods.phieng_world3_get_body_velocity(_handle, (uint)bodyId, &vel);
            velocity = vel;
            return ok != 0;
        }
    }

    /// <summary>
    /// Bulk 3D transform read: <paramref name="ids"/> resolve in a single
    /// native transition. Slot <c>i</c> of every output belongs to
    /// <c>ids[i]</c> (index alignment preserved). Stale ids write
    /// <see cref="float.NaN"/> into their slot. Pass <c>null</c> for
    /// <paramref name="velocities"/> to skip velocity readback.
    /// </summary>
    /// <returns>Number of ids that resolved to a live body.</returns>
    public unsafe int GetTransformsBulk(uint[] ids, Vector3[] positions, Quaternion[] rotations, Vector3[]? velocities = null)
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
            fixed (Vector3* pPos = positions)
            fixed (Quaternion* pRot = rotations)
            {
                if (velocities == null)
                {
                    return (int)NativeMethods.phieng_world3_get_transforms_bulk(
                        _handle, pIds, (uint)ids.Length, (float*)pPos, (float*)pRot, null);
                }
                fixed (Vector3* pVel = velocities)
                {
                    return (int)NativeMethods.phieng_world3_get_transforms_bulk(
                        _handle, pIds, (uint)ids.Length, (float*)pPos, (float*)pRot, (float*)pVel);
                }
            }
        }
    }

    /// <summary>
    /// Remove a body by id. Remaining bodies are reindexed by the native
    /// core, mirroring <c>World3.RemoveBody</c>.
    /// </summary>
    public void RemoveBody(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_remove_body(_handle, (uint)bodyId);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
            }
        }
    }

    /// <summary>
    /// Add a 3D joint from a JSON param blob. Generic tag + JSON params:
    /// 0 distance, 1 hinge, 2 prismatic, 3 fixed. Mirrors
    /// <c>World3.AddJoint</c> + the native joint bridge.
    /// </summary>
    /// <returns>Joint id, or -1 on any invalid input.</returns>
    public unsafe int AddJoint(int tag, int bodyA, int bodyB, string json)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var bytes = System.Text.Encoding.UTF8.GetBytes(json);
            fixed (byte* ptr = bytes)
            {
                return NativeMethods.phieng_world3_add_joint(
                    _handle, (uint)tag, (uint)bodyA, (uint)bodyB, ptr, (UIntPtr)bytes.Length);
            }
        }
    }

    /// <summary>Remove a 3D joint by id. Mirrors <c>World3.RemoveJoint</c>.</summary>
    public void RemoveJoint(int jointId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_remove_joint(_handle, (uint)jointId);
            if (ok == 0)
                throw new ArgumentOutOfRangeException(nameof(jointId), $"Unknown 3D joint id {jointId}.");
        }
    }

    /// <summary>Number of 3D joints in the world.</summary>
    public int JointCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                int n = NativeMethods.phieng_world3_joint_count(_handle);
                if (n < 0)
                    throw new InvalidOperationException("Native 3D joint count returned null world.");
                return n;
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
            int ok = NativeMethods.phieng_world3_set_body_material(
                _handle, (uint)bodyId, material.Friction, material.Density, material.Restitution);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
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
            int ok = NativeMethods.phieng_world3_get_body_material(_handle, (uint)bodyId, &packed);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
            }
            // Packed as (friction, density, restitution); the constructor
            // sanitizes, mirroring the Rust side.
            return new Dynamics.PhysicsMaterial(packed.X, packed.Z, packed.Y);
        }
    }

    /// <summary>
    /// Set a 3D body's gravitational coupling multiplier (1.0 full
    /// coupling, 0.0 immune, negative repels; finite only).
    /// </summary>
    public void SetBodyGravityScale(int bodyId, float scale)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_body_gravity_scale(_handle, (uint)bodyId, scale);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId} or non-finite scale.");
            }
        }
    }

    /// <summary>
    /// Set a body's collision filter: the layer it belongs to and the
    /// layers it accepts. Two bodies may collide only when each
    /// accepts the other's layer. The default filter accepts
    /// everything, so an unconfigured body collides with everything.
    /// </summary>
    public void SetBodyCollisionFilter(int bodyId, uint group, uint mask)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_body_collision_filter(_handle, (uint)bodyId, group, mask);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>
    /// Cast a ray into the world and return the first thing it hits.
    /// </summary>
    /// <returns>
    /// The body, the distance along the ray, the contact point and the
    /// contact normal; null when the ray hits nothing.
    /// </returns>
    public unsafe (uint Body, float T, Vector3 Point, Vector3 Normal)? Raycast(
        Vector3 origin, Vector3 direction, float maxT, uint group, uint mask)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var buffer = new float[8];
            int ok;
            fixed (float* p = buffer)
            {
                ok = NativeMethods.phieng_world3_raycast(_handle, origin.X, origin.Y, origin.Z,
                    direction.X, direction.Y, direction.Z, maxT, group, mask, p);
            }
            if (ok < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(origin), "Invalid ray.");
            }
            if (ok == 0)
            {
                return null;
            }
            return ((uint)buffer[0], buffer[1],
                new Vector3(buffer[2], buffer[3], buffer[4]),
                new Vector3(buffer[5], buffer[6], buffer[7]));
        }
    }

    /// <summary>
    /// Makes a body a sensor: it reports its contacts as contact
    /// events and never pushes, so it falls through whatever it
    /// touches. That is what a trigger volume is.
    /// </summary>
    public void SetBodySensor(int bodyId, bool isSensor)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_body_sensor(
                _handle, (uint)bodyId, isSensor ? (byte)1 : (byte)0);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Turns the contact-event collection on or off.</summary>
    public void SetCollectContactEvents(bool collect)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.phieng_world3_set_collect_contact_events(
                _handle, collect ? (byte)1 : (byte)0);
        }
    }

    /// <summary>The contact events of the last step.</summary>
    public unsafe (uint BodyA, uint BodyB, int Kind, Vector3 Point, Vector3 Normal)[] ContactEvents()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int count = NativeMethods.phieng_world3_contact_event_count(_handle);
            if (count <= 0) return Array.Empty<(uint, uint, int, Vector3, Vector3)>();
            var result = new (uint, uint, int, Vector3, Vector3)[count];
            var buffer = new float[9];
            for (int i = 0; i < count; i++)
            {
                fixed (float* p = buffer)
                {
                    if (NativeMethods.phieng_world3_get_contact_event(_handle, (uint)i, p) == 0)
                    {
                        throw new InvalidOperationException("Contact event read failed.");
                    }
                }
                result[i] = ((uint)buffer[0], (uint)buffer[1], (int)buffer[2],
                    new Vector3(buffer[3], buffer[4], buffer[5]),
                    new Vector3(buffer[6], buffer[7], buffer[8]));
            }
            return result;
        }
    }

    /// <summary>
    /// Place a body: set its position and rotation. The body keeps its
    /// velocity; the broad phase picks the new place up at the start of
    /// the next step.
    /// </summary>
    public void SetBodyTransform(int bodyId, Vector3 position, Quaternion rotation)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_body_transform(
                _handle, (uint)bodyId, position.X, position.Y, position.Z,
                rotation.X, rotation.Y, rotation.Z, rotation.W);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Set a body's angular velocity, in radians per second.</summary>
    public void SetBodyAngularVelocity(int bodyId, Vector3 angularVelocity)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_set_body_angular_velocity(
                _handle, (uint)bodyId, angularVelocity.X, angularVelocity.Y, angularVelocity.Z);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Read a body's angular velocity, in radians per second.</summary>
    public unsafe Vector3 GetBodyAngularVelocity(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var angularVelocity = new float[3];
            int ok;
            fixed (float* p = angularVelocity)
            {
                ok = NativeMethods.phieng_world3_get_body_angular_velocity(
                    _handle, (uint)bodyId, p);
            }
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return new Vector3(angularVelocity[0], angularVelocity[1], angularVelocity[2]);
        }
    }

    /// <summary>Apply an impulse to a body, in newton-seconds.</summary>
    public void ApplyImpulse(int bodyId, Vector3 impulse)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_apply_impulse(
                _handle, (uint)bodyId, impulse.X, impulse.Y, impulse.Z);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Apply a force to a body, in newtons, for <paramref name="dt"/> seconds.</summary>
    public void ApplyForce(int bodyId, Vector3 force, float dt)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_apply_force(
                _handle, (uint)bodyId, force.X, force.Y, force.Z, dt);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>Wake a body and everything it touches, so a sleeper responds again.</summary>
    public void WakeBody(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int ok = NativeMethods.phieng_world3_wake_body(_handle, (uint)bodyId);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
        }
    }

    /// <summary>The id of the body at an index, or -1 past the end.</summary>
    public int GetBodyIdAt(int index)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return NativeMethods.phieng_world3_get_body_id_at(_handle, (uint)index);
        }
    }

    /// <summary>
    /// The bodies whose AABB overlaps the query box, in ascending id
    /// order. A body is reported only when the filter accepts it.
    /// </summary>
    public unsafe uint[] OverlapAabb(
        float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
        uint group, uint mask)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            int count = (int)NativeMethods.phieng_world3_overlap_aabb(_handle, minX, minY, minZ, maxX, maxY, maxZ, group, mask, null, 0);
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minX), "Invalid query box.");
            }
            if (count == 0)
            {
                return Array.Empty<uint>();
            }
            var ids = new uint[count];
            fixed (uint* p = ids)
            {
                NativeMethods.phieng_world3_overlap_aabb(_handle, minX, minY, minZ, maxX, maxY, maxZ, group, mask, p, (uint)count);
            }
            return ids;
        }
    }

    /// <summary>
    /// Read a body's collision filter: the layer it belongs to and the
    /// layers it accepts.
    /// </summary>
    public unsafe (uint Group, uint Mask) GetBodyCollisionFilter(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint group, mask;
            int ok = NativeMethods.phieng_world3_get_body_collision_filter(_handle, (uint)bodyId, &group, &mask);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native body id {bodyId}.");
            }
            return (group, mask);
        }
    }

    /// <summary>Read a 3D body's gravitational coupling multiplier.</summary>
    public unsafe float GetBodyGravityScale(int bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float scale;
            int ok = NativeMethods.phieng_world3_get_body_gravity_scale(_handle, (uint)bodyId, &scale);
            if (ok == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(bodyId), $"Unknown native 3D body id {bodyId}.");
            }
            return scale;
        }
    }

    /// <summary>
    /// Read a 3D body's gravitational coupling multiplier without
    /// throwing: returns <c>false</c> for unknown/stale ids.
    /// </summary>
    public unsafe bool TryGetBodyGravityScale(int bodyId, out float scale)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float s;
            int ok = NativeMethods.phieng_world3_get_body_gravity_scale(_handle, (uint)bodyId, &s);
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
            int ok = NativeMethods.phieng_world3_set_broadphase_mode(_handle, (uint)mode);
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
            int ok = NativeMethods.phieng_world3_set_collision_tier(_handle, &ffi);
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
            int ok = NativeMethods.phieng_world3_get_collision_tier(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D collision tier read failed.");
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
            int ok = NativeMethods.phieng_world3_get_broadphase_mode(_handle, &mode);
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
            int ok = NativeMethods.phieng_world3_set_sleep_thresholds(_handle, linear, angular);
            if (ok == 0)
            {
                throw new ArgumentException("Native 3D world rejected sleep thresholds (must be finite, >= 0).");
            }
        }
    }

    /// <summary>Read the 3D sleep velocity thresholds (linear, angular).</summary>
    public unsafe (float Linear, float Angular) GetSleepThresholds()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            float* values = stackalloc float[2];
            int ok = NativeMethods.phieng_world3_get_sleep_thresholds(_handle, values);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native 3D sleep threshold read failed.");
            }
            return (values[0], values[1]);
        }
    }

    /// <summary>
    /// Save the native 3D world to a file. <paramref name="format"/>: 0 JSON,
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
                int rc = NativeMethods.phieng_world3_save(
                    _handle, p, (UIntPtr)utf8.Length, format, mode);
                if (rc != 1)
                {
                    throw new IOException($"Native 3D world save failed (code {rc}): {path}.");
                }
            }
        }
    }

    /// <summary>
    /// Load a native 3D world from a file (format auto-detected).
    /// <paramref name="mode"/>: 0 exact resume, 1 fresh start.
    /// </summary>
    public static unsafe PhiNativeWorld3 Load(string path, uint mode)
    {
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(path);
        fixed (byte* p = utf8)
        {
            IntPtr handle = NativeMethods.phieng_world3_load(p, (UIntPtr)utf8.Length, mode);
            if (handle == IntPtr.Zero)
            {
                throw new IOException($"Native 3D world load failed: {path}.");
            }
            return new PhiNativeWorld3(handle);
        }
    }

    private PhiNativeWorld3(IntPtr handle)
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
                NativeMethods.phieng_world3_free(_handle);
                _handle = IntPtr.Zero;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~PhiNativeWorld3()
    {
        // Finalizer path: see PhiNativeWorld for the locking rationale.
        lock (_gate)
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.phieng_world3_free(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Set the 3D nuclear configuration and its nuclide table.
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
                ActivationRate = config.ActivationRate,
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
            int ok = NativeMethods.phieng_world3_set_nuclear_config(_handle, &ffi);
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

    /// <summary>Get the 3D nuclear configuration.</summary>
    public unsafe Dynamics.NuclearConfig GetNuclearConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearConfigFfi();
            int ok = NativeMethods.phieng_world3_get_nuclear_config(_handle, &ffi);
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

    /// <summary>Set one nuclide in the 3D table.</summary>
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
            int ok = NativeMethods.phieng_world3_set_isotope(_handle, index, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclide (check ranges).");
            }
        }
    }

    /// <summary>Get one nuclide from the 3D table.</summary>
    /// <param name="index">The nuclide index (0 is the placeholder).</param>
    public unsafe Dynamics.Isotope GetIsotope(uint index)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.IsotopeFfi();
            int ok = NativeMethods.phieng_world3_get_isotope(_handle, index, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclide request.");
            }
            return new Dynamics.Isotope
            {
                HalfLife = ffi.HalfLife,
                DecayEnergy = ffi.DecayEnergy,
                IonizationFraction = ffi.IonizationFraction,
                Daughter = (ushort)ffi.Daughter,
                AbsorptionCrossSection = ffi.AbsorptionCrossSection,
                FissionCrossSection = ffi.FissionCrossSection,
                NeutronYield = ffi.NeutronYield,
                IsFissile = ffi.IsFissile != 0,
            };
        }
    }

    /// <summary>Set the 3D per-body nuclear state.</summary>
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
                CaptureProgress = state.CaptureProgress,
            };
            int ok = NativeMethods.phieng_world3_set_body_nuclear_state(_handle, bodyId, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected the nuclear state (check ranges).");
            }
        }
    }

    /// <summary>Get the 3D per-body nuclear state.</summary>
    /// <param name="bodyId">The body id.</param>
    public unsafe Dynamics.NuclearState GetBodyNuclearState(uint bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.NuclearBodyFfi();
            int ok = NativeMethods.phieng_world3_get_body_nuclear_state(_handle, bodyId, &ffi);
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
                CaptureProgress = ffi.CaptureProgress,
            };
        }
    }

    /// <summary>Configure the native world's quantum field (rebuilds its wavefunction in the ground state).</summary>
    public unsafe void SetQuantumConfig(QuantumConfig config)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            var ffi = new NativeMethods.QuantumConfigFfi
            {
                Enabled = config.Enabled ? (byte)1 : (byte)0,
                Method = (uint)config.Method,
                Boundary = (uint)config.Boundary,
                GridSize = (uint)config.GridSize,
                DomainMin = config.DomainMin,
                DomainMax = config.DomainMax,
                Mass = config.Mass,
                Hbar = config.Hbar,
                Omega = config.Omega,
                Seed = config.Seed,
            };
            int ok = NativeMethods.phieng_world3_quantum_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new ArgumentException("Native world rejected quantum config (check ranges).");
            }
        }
    }

    /// <summary>Read the native world's quantum field configuration.</summary>
    public unsafe QuantumConfig GetQuantumConfig()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            NativeMethods.QuantumConfigFfi ffi;
            int ok = NativeMethods.phieng_world3_quantum_get_config(_handle, &ffi);
            if (ok == 0)
            {
                throw new InvalidOperationException("Native quantum config read failed.");
            }
            return new QuantumConfig
            {
                Enabled = ffi.Enabled != 0,
                Method = (EvolutionMethod)ffi.Method,
                Boundary = (Boundary1D)ffi.Boundary,
                GridSize = (int)ffi.GridSize,
                DomainMin = ffi.DomainMin,
                DomainMax = ffi.DomainMax,
                Mass = ffi.Mass,
                Hbar = ffi.Hbar,
                Omega = ffi.Omega,
                Seed = ffi.Seed,
            };
        }
    }

    /// <summary>The native wavefunction's amplitudes, real and imaginary parts.</summary>
    public unsafe (float[] Re, float[] Im) QuantumState()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint count = NativeMethods.phieng_world3_quantum_state(_handle, null, null, 0);
            if (count == 0) return (Array.Empty<float>(), Array.Empty<float>());
            var re = new float[count];
            var im = new float[count];
            fixed (float* pr = re)
            fixed (float* pi = im)
            {
                uint written = NativeMethods.phieng_world3_quantum_state(_handle, pr, pi, count);
                if (written != count)
                {
                    throw new InvalidOperationException("Native quantum state read failed.");
                }
            }
            return (re, im);
        }
    }

    /// <summary>The native measurement probabilities (squared moduli).</summary>
    public unsafe float[] QuantumProbabilities()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint count = NativeMethods.phieng_world3_quantum_probability(_handle, null, 0);
            if (count == 0) return Array.Empty<float>();
            var probs = new float[count];
            fixed (float* p = probs)
            {
                uint written = NativeMethods.phieng_world3_quantum_probability(_handle, p, count);
                if (written != count)
                {
                    throw new InvalidOperationException("Native quantum probability read failed.");
                }
            }
            return probs;
        }
    }

    /// <summary>Draws measurement outcomes from the native wavefunction (Born rule).</summary>
    public unsafe float[] QuantumSample()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint count = NativeMethods.phieng_world3_quantum_sample(_handle, null, 0);
            if (count == 0) return Array.Empty<float>();
            var draws = new float[count];
            fixed (float* p = draws)
            {
                uint written = NativeMethods.phieng_world3_quantum_sample(_handle, p, count);
                if (written != count)
                {
                    throw new InvalidOperationException("Native quantum sample failed.");
                }
            }
            return draws;
        }
    }

    /// <summary>
    /// Drives a body by the native world's quantum field: the body's x
    /// position becomes a Born-rule draw, its x velocity the Bohmian
    /// flow. Returns false when the field is off or the body is unknown.
    /// </summary>
    public bool QuantumDriveBody(uint bodyId)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return NativeMethods.phieng_world3_quantum_drive_body(_handle, bodyId) != 0;
        }
    }

    /// <summary>
    /// The Bohmian force at <paramref name="x"/> in the native world, or
    /// 0.0 when the field is off.
    /// </summary>
    public float QuantumForceAt(float x)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            return NativeMethods.phieng_world3_quantum_force_at(_handle, x);
        }
    }

    /// <summary>The native wavefunction's density on the grid.</summary>
    public unsafe float[] QuantumDensity()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint count = NativeMethods.phieng_world3_quantum_density(_handle, null, 0);
            if (count == 0) return Array.Empty<float>();
            var density = new float[count];
            fixed (float* p = density)
            {
                uint written = NativeMethods.phieng_world3_quantum_density(_handle, p, count);
                if (written != count)
                {
                    throw new InvalidOperationException("Native quantum density read failed.");
                }
            }
            return density;
        }
    }

    /// <summary>The native wavefunction's phase on the grid.</summary>
    public unsafe float[] QuantumPhase()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            uint count = NativeMethods.phieng_world3_quantum_phase(_handle, null, 0);
            if (count == 0) return Array.Empty<float>();
            var phase = new float[count];
            fixed (float* p = phase)
            {
                uint written = NativeMethods.phieng_world3_quantum_phase(_handle, p, count);
                if (written != count)
                {
                    throw new InvalidOperationException("Native quantum phase read failed.");
                }
            }
            return phase;
        }
    }

}
