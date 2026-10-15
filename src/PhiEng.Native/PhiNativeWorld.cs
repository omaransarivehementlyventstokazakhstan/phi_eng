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

    public int BodyCount
    {
        get
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                return NativeMethods.phieng_world_body_count(_handle);
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
