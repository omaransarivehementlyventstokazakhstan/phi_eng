using System.Numerics;

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
}
