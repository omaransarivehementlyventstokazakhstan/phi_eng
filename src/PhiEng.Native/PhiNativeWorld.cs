using System.Numerics;

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
}
