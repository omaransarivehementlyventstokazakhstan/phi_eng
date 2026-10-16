using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace PhiEng.Native;

/// <summary>
/// Raw P/Invoke bindings to the Rust physics core (<c>phieng.dll</c> /
/// <c>libphieng.so</c> / <c>libphieng.dylib</c>, built from
/// <c>crates/engine</c> with <c>crate-type = ["cdylib", "rlib"]</c>).
/// Mirrors <c>crates/engine/src/ffi.rs</c> — keep signatures in sync.
/// Uses source-generated <c>LibraryImport</c> with a <c>DllImportResolver</c>
/// that resolves the native library from the NuGet package's
/// <c>runtimes/{rid}/native/</c> folder based on the current RID.
/// </summary>
internal static partial class NativeMethods
{
    private const string LibraryName = "phieng";

    // 2D joint tags (mirror ffi.rs JOINT_* constants)
    public const uint JOINT_DISTANCE = 0;
    public const uint JOINT_HINGE = 1;
    public const uint JOINT_PRISMATIC = 2;
    public const uint JOINT_FIXED = 3;

    // 3D joint tags (mirror ffi.rs JOINT3_* constants)
    public const uint JOINT3_DISTANCE = 0;
    public const uint JOINT3_HINGE = 1;
    public const uint JOINT3_PRISMATIC = 2;
    public const uint JOINT3_FIXED = 3;

    static NativeMethods()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, ResolveLibrary);
    }

    private static IntPtr ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!libraryName.Equals("phieng", StringComparison.OrdinalIgnoreCase))
        {
            return IntPtr.Zero;
        }

        // RID mapping: .NET RID -> native library filename
        var rid = RuntimeInformation.RuntimeIdentifier;
        string libraryFile = rid switch
        {
            "win-x64" or "win-x86" or "win-arm64" => "phieng.dll",
            "linux-x64" or "linux-arm64" or "linux-arm" => "libphieng.so",
            "osx-x64" or "osx-arm64" => "libphieng.dylib",
            _ => throw new DllNotFoundException($"Unsupported RID: {rid}. No native library available.")
        };

        // The native libs are packed under runtimes/{rid}/native/ in the NuGet package.
        // At runtime they get extracted to the app's output directory under
        // runtimes/{rid}/native/. We probe the standard locations.
        var assemblyDir = Path.GetDirectoryName(assembly.Location)!;
        var candidatePaths = new[]
        {
            // 1. Next to the assembly (self-contained deploy)
            Path.Combine(assemblyDir, libraryFile),
            // 2. In runtimes/{rid}/native/ relative to assembly (NuGet package layout)
            Path.Combine(assemblyDir, "runtimes", rid, "native", libraryFile),
            // 3. In the base directory (global tool, etc.)
            Path.Combine(AppContext.BaseDirectory, libraryFile),
            Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", libraryFile),
        };

        foreach (var path in candidatePaths)
        {
            if (File.Exists(path))
            {
                return NativeLibrary.Load(path, assembly, searchPath);
            }
        }

        return IntPtr.Zero; // Let the default resolver try as fallback
    }

    [LibraryImport("phieng")]
    internal static partial IntPtr phieng_world_new();

    [LibraryImport("phieng")]
    internal static partial void phieng_world_free(IntPtr world);

    [LibraryImport("phieng")]
    internal static partial void phieng_world_step(IntPtr world, float dt);

    [LibraryImport("phieng")]
    internal static partial void phieng_world_set_gravity(IntPtr world, float gx, float gy);

    /// <summary>
    /// Sets the gravity force model + integration scheme (mirrors
    /// <c>phieng_dynamics::gravity</c>). mode: 0 uniform, 1 central,
    /// 2 N-body, 3 Barnes-Hut. integrator: 0 Euler, 1 Verlet.
    /// </summary>
    /// <returns>1 on success, 0 for null world or out-of-range input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_gravity_config(
        IntPtr world,
        uint mode,
        uint integrator,
        float bigG,
        float softening,
        float ax,
        float ay,
        float mu,
        float theta);

    /// <summary>
    /// Reads the gravity config: mode/integrator tags plus 6 floats
    /// (bigG, softening, ax, ay, mu, theta).
    /// </summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_gravity_config(
        IntPtr world,
        uint* outMode,
        uint* outIntegrator,
        float* outValues);

    /// <summary>Sets a body's gravitational coupling multiplier.</summary>
    /// <returns>1 on success, 0 for null world, unknown body, or non-finite scale.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_body_gravity_scale(
        IntPtr world,
        uint bodyId,
        float scale);

    /// <summary>Reads a body's gravitational coupling multiplier.</summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_body_gravity_scale(
        IntPtr world,
        uint bodyId,
        float* outScale);

    /// <summary>
    /// Ambient-force config FFI struct (2D). Field order mirrors Rust
    /// <c>AmbientConfig2Ffi</c> — never reorder.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AmbientConfig2Ffi
    {
        public uint SpringEnabled;
        public float SpringAx;
        public float SpringAy;
        public float SpringStiffness;
        public float SpringDamping;
        public float SpringMaxDisplacement;
        public uint DragEnabled;
        public float DragLinear;
        public float DragQuadratic;
        public float AirDensity;
        public float DragAngular;
        public float WindX;
        public float WindY;
        public uint TurbulenceEnabled;
        public float TurbulenceAmplitude;
        public float TurbulenceFrequency;
        public float TurbulenceTimeScale;
        public uint TurbulenceSeed;
        public uint TurbulenceOctaves;
        public uint TurbulenceDivergenceFree;
    }

    /// <summary>Ambient-force config FFI struct (3D). Mirrors Rust <c>AmbientConfig3Ffi</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AmbientConfig3Ffi
    {
        public uint SpringEnabled;
        public float SpringAx;
        public float SpringAy;
        public float SpringAz;
        public float SpringStiffness;
        public float SpringDamping;
        public float SpringMaxDisplacement;
        public uint DragEnabled;
        public float DragLinear;
        public float DragQuadratic;
        public float AirDensity;
        public float DragAngular;
        public float WindX;
        public float WindY;
        public float WindZ;
        public uint TurbulenceEnabled;
        public float TurbulenceAmplitude;
        public float TurbulenceFrequency;
        public float TurbulenceTimeScale;
        public uint TurbulenceSeed;
        public uint TurbulenceOctaves;
        public uint TurbulenceDivergenceFree;
    }

    /// <summary>Sets the 2D ambient-force config.</summary>
    /// <returns>1 on success, 0 for null world/struct or out-of-range input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_set_ambient_config(
        IntPtr world,
        AmbientConfig2Ffi* cfg);

    /// <summary>Reads the 2D ambient-force config.</summary>
    /// <returns>1 on success, 0 for null world/out.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_ambient_config(
        IntPtr world,
        AmbientConfig2Ffi* outCfg);

    /// <summary>Sets the 3D ambient-force config.</summary>
    /// <returns>1 on success, 0 for null world/struct or out-of-range input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_set_ambient_config(
        IntPtr world,
        AmbientConfig3Ffi* cfg);

    /// <summary>Reads the 3D ambient-force config.</summary>
    /// <returns>1 on success, 0 for null world/out.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_ambient_config(
        IntPtr world,
        AmbientConfig3Ffi* outCfg);

    /// <summary>Sets the 2D simulation clock.</summary>
    /// <returns>1 on success, 0 for null world or non-finite input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_time(IntPtr world, float t);

    /// <summary>Reads the 2D simulation clock.</summary>
    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_time(IntPtr world, float* outT);

    /// <summary>Sets the 3D simulation clock.</summary>
    /// <returns>1 on success, 0 for null world or non-finite input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_time(IntPtr world, float t);

    /// <summary>Reads the 3D simulation clock.</summary>
    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_time(IntPtr world, float* outT);

    /// <summary>Sets a body's ambient-force coupling multiplier (2D).</summary>
    /// <returns>1 on success, 0 for null world, unknown body, or non-finite scale.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_body_ambient_scale(
        IntPtr world,
        uint bodyId,
        float scale);

    /// <summary>Reads a body's ambient-force coupling multiplier (2D).</summary>
    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_body_ambient_scale(
        IntPtr world,
        uint bodyId,
        float* outScale);

    /// <summary>Sets a body's ambient-force coupling multiplier (3D).</summary>
    /// <returns>1 on success, 0 for null world, unknown body, or non-finite scale.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_body_ambient_scale(
        IntPtr world,
        uint bodyId,
        float scale);

    /// <summary>Reads a body's ambient-force coupling multiplier (3D).</summary>
    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_body_ambient_scale(
        IntPtr world,
        uint bodyId,
        float* outScale);

    /// <summary>
    /// Sets the sleep velocity thresholds (0.0 disables sleeping).
    /// </summary>
    /// <returns>1 on success, 0 for null world or negative/non-finite input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_sleep_thresholds(
        IntPtr world,
        float linear,
        float angular);

    /// <summary>Copies the sleep velocity thresholds out (linear, angular).</summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_sleep_thresholds(
        IntPtr world,
        float* @out);

    [LibraryImport("phieng"), SuppressGCTransition]
    internal static partial int phieng_world_body_count(IntPtr world);

    /// <returns>Body id, or -1 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_add_body(
        IntPtr world,
        float posX,
        float posY,
        uint shapeTag,
        float a,
        float b,
        float density,
        byte isStatic,
        float friction,
        float restitution);

    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world_get_body_transform(
        IntPtr world,
        uint bodyId,
        Vector2* outPos,
        float* outRot);

    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world_get_body_velocity(
        IntPtr world,
        uint bodyId,
        Vector2* outVel);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_set_body_material(
        IntPtr world,
        uint bodyId,
        float friction,
        float density,
        float restitution);

    /// <summary>
    /// Bulk body creation: <paramref name="n"/> descriptors at
    /// <paramref name="descs"/> (see <c>BodyDesc2D</c> layout) are validated
    /// exactly like <c>phieng_world_add_body</c>. <paramref name="outIds"/>[i]
    /// gets the new body id, or -1 for a rejected descriptor (no rollback).
    /// </summary>
    /// <returns>Number of bodies added, 0 on null world/descs/ids.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world_add_bodies(
        IntPtr world,
        BodyDesc2D* descs,
        uint n,
        int* outIds);

    /// <summary>
    /// Bulk transform read: <paramref name="n"/> ids at <paramref name="ids"/>
    /// resolve in one transition. Slot i of every output belongs to ids[i]:
    /// <paramref name="outPos"/> holds 2*n (x, y), <paramref name="outRot"/>
    /// holds n angles, <paramref name="outVel"/> holds 2*n or may be null to
    /// skip velocities. Stale ids write NaN into their slot.
    /// </summary>
    /// <returns>Number of ids that resolved to a live body.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world_get_transforms_bulk(
        IntPtr world,
        uint* ids,
        uint n,
        float* outPos,
        float* outRot,
        float* outVel);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world_get_body_material(
        IntPtr world,
        uint bodyId,
        Vector3* outMat);

    // ---- Fluid particles (mirrors the phieng_world_fluid_* functions in ffi.rs) ----
    // Positions are interleaved pairs [x0, y0, ...]; reads add zero-velocity
    // particles. Queries copy interleaved [x, y, vx, vy] per particle.

    /// <returns>Number of particles added, 0 on null world/positions or bad input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world_fluid_add(
        IntPtr world,
        float* positions,
        uint count);

    /// <returns>Particle count, 0 on null world.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static partial uint phieng_world_fluid_count(IntPtr world);

    /// <returns>1 on success, 0 on null world.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_fluid_clear(IntPtr world);

    /// <returns>Number of particles written, 0 on null world/out.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world_fluid_get(
        IntPtr world,
        float* @out,
        uint max);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_remove_body(IntPtr world, uint bodyId);

    // ---- Joints (mirrors the phieng_world_*_joint functions in ffi.rs) ----
    // Generic tag + JSON params: 0 distance, 1 hinge, 2 prismatic, 3 fixed.
    // jsonUtf8 points at jsonLen UTF-8 bytes (no NUL terminator needed).

    /// <returns>Joint id, or -1 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_add_joint(
        IntPtr world,
        uint tag,
        uint bodyA,
        uint bodyB,
        byte* jsonUtf8,
        UIntPtr jsonLen);

    /// <returns>1 on success, 0 for null world or unknown joint.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world_remove_joint(IntPtr world, uint jointId);

    /// <returns>Joint count, or -1 for null world.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static partial int phieng_world_joint_count(IntPtr world);

    // ---- Snapshots (mirrors the phieng_world_save/load functions in ffi.rs) ----
    // format: 0 JSON, 1 binary. mode: 0 exact, 1 fresh.
    // Returns 1 on success, 0 on invalid input, -1 on IO failure, -2 on encode failure.

    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_save(
        IntPtr world,
        byte* pathUtf8,
        UIntPtr pathLen,
        uint format,
        uint mode);

    /// <returns>New world handle, or null on any failure (caller owns it).</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial IntPtr phieng_world_load(
        byte* pathUtf8,
        UIntPtr pathLen,
        uint mode);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_save(
        IntPtr world,
        byte* pathUtf8,
        UIntPtr pathLen,
        uint format,
        uint mode);

    /// <returns>New 3D world handle, or null on any failure (caller owns it).</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial IntPtr phieng_world3_load(
        byte* pathUtf8,
        UIntPtr pathLen,
        uint mode);

    // ---- 3D world (mirrors the phieng_world3_* functions in ffi.rs) ----

    // ---- 3D Joints (mirrors the phieng_world3_*_joint functions in ffi.rs) ----
    // Generic tag + JSON params: 0 distance, 1 hinge, 2 prismatic, 3 fixed.
    // jsonUtf8 points at jsonLen UTF-8 bytes (no NUL terminator needed).

    /// <returns>Joint id, or -1 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_add_joint(
        IntPtr world,
        uint tag,
        uint bodyA,
        uint bodyB,
        byte* jsonUtf8,
        UIntPtr jsonLen);

    /// <returns>1 on success, 0 for null world or unknown joint.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_remove_joint(IntPtr world, uint jointId);

    /// <returns>Joint count, or -1 for null world.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static partial int phieng_world3_joint_count(IntPtr world);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr phieng_world3_new();

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_free(IntPtr world);

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_step(IntPtr world, float dt);

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_set_gravity(IntPtr world, float gx, float gy, float gz);

    /// <summary>
    /// Sets the 3D gravity force model + integration scheme (same
    /// contract as <c>phieng_world_set_gravity_config</c>, plus az).
    /// </summary>
    /// <returns>1 on success, 0 for null world or out-of-range input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_gravity_config(
        IntPtr world,
        uint mode,
        uint integrator,
        float bigG,
        float softening,
        float ax,
        float ay,
        float az,
        float mu,
        float theta);

    /// <summary>
    /// Reads the 3D gravity config: mode/integrator tags plus 7 floats
    /// (bigG, softening, ax, ay, az, mu, theta).
    /// </summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_gravity_config(
        IntPtr world,
        uint* outMode,
        uint* outIntegrator,
        float* outValues);

    /// <summary>Sets a 3D body's gravitational coupling multiplier.</summary>
    /// <returns>1 on success, 0 for null world, unknown body, or non-finite scale.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_body_gravity_scale(
        IntPtr world,
        uint bodyId,
        float scale);

    /// <summary>Reads a 3D body's gravitational coupling multiplier.</summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_body_gravity_scale(
        IntPtr world,
        uint bodyId,
        float* outScale);

    /// <summary>
    /// Sets the 3D sleep velocity thresholds (0.0 disables sleeping).
    /// </summary>
    /// <returns>1 on success, 0 for null world or negative/non-finite input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_sleep_thresholds(
        IntPtr world,
        float linear,
        float angular);

    /// <summary>Copies the 3D sleep velocity thresholds out (linear, angular).</summary>
    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_sleep_thresholds(
        IntPtr world,
        float* @out);

    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static partial int phieng_world3_body_count(IntPtr world);

    /// <returns>Body id, or -1 on invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_add_body(
        IntPtr world,
        float posX, float posY, float posZ,
        uint shapeTag,
        float a, float b, float c,
        float density,
        byte isStatic,
        float friction,
        float restitution);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world3_get_body_transform(
        IntPtr world,
        uint bodyId,
        Vector3* outPos,
        Quaternion* outQuat);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world3_get_body_velocity(
        IntPtr world,
        uint bodyId,
        Vector3* outVel);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_remove_body(IntPtr world, uint bodyId);

    /// <summary>
    /// Bulk 3D body creation: same contract as
    /// <c>phieng_world_add_bodies</c> with <c>BodyDesc3D</c> descriptors
    /// (sphere tag 0, box tag 1, capsule tag 2).
    /// </summary>
    /// <returns>Number of bodies added, 0 on null world/descs/ids.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world3_add_bodies(
        IntPtr world,
        BodyDesc3D* descs,
        uint n,
        int* outIds);

    /// <summary>
    /// Bulk 3D transform read: same contract as
    /// <c>phieng_world_get_transforms_bulk</c> with 3D strides —
    /// <paramref name="outPos"/> holds 3*n, <paramref name="outQuat"/> holds
    /// 4*n (x, y, z, w), <paramref name="outVel"/> holds 3*n or may be null.
    /// </summary>
    /// <returns>Number of ids that resolved to a live body.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world3_get_transforms_bulk(
        IntPtr world,
        uint* ids,
        uint n,
        float* outPos,
        float* outQuat,
        float* outVel);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_body_material(
        IntPtr world,
        uint bodyId,
        float friction,
        float density,
        float restitution);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static unsafe partial int phieng_world3_get_body_material(
        IntPtr world,
        uint bodyId,
        Vector3* outMat);

    // ---- 3D fluid particles (mirrors the phieng_world3_fluid_* functions in ffi.rs) ----
    // Positions are interleaved triples; queries copy [x, y, z, vx, vy, vz] per particle.

    /// <returns>Number of particles added, 0 on null world/positions or bad input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world3_fluid_add(
        IntPtr world,
        float* positions,
        uint count);

    /// <returns>Particle count, 0 on null world.</returns>
    [LibraryImport(LibraryName), SuppressGCTransition]
    internal static partial uint phieng_world3_fluid_count(IntPtr world);

    /// <returns>1 on success, 0 on null world.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_fluid_clear(IntPtr world);

    /// <returns>Number of particles written, 0 on null world/out.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial uint phieng_world3_fluid_get(
        IntPtr world,
        float* @out,
        uint max);
}
