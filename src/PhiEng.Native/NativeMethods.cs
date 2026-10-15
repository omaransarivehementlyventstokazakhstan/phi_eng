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

    [LibraryImport("phieng")]
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
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world_get_body_transform(
        IntPtr world,
        uint bodyId,
        Vector2* outPos,
        float* outRot);

    /// <returns>1 on success, 0 on invalid input.</returns>
    [LibraryImport(LibraryName)]
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

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
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
    [LibraryImport(LibraryName)]
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
    [LibraryImport(LibraryName)]
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
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_joint_count(IntPtr world);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr phieng_world3_new();

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_free(IntPtr world);

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_step(IntPtr world, float dt);

    [LibraryImport(LibraryName)]
    internal static partial void phieng_world3_set_gravity(IntPtr world, float gx, float gy, float gz);

    [LibraryImport(LibraryName)]
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
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_body_transform(
        IntPtr world,
        uint bodyId,
        Vector3* outPos,
        Quaternion* outQuat);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
    internal static unsafe partial int phieng_world3_get_body_velocity(
        IntPtr world,
        uint bodyId,
        Vector3* outVel);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_remove_body(IntPtr world, uint bodyId);

    /// <returns>1 on success, 0 for null world or unknown body.</returns>
    [LibraryImport(LibraryName)]
    internal static partial int phieng_world3_set_body_material(
        IntPtr world,
        uint bodyId,
        float friction,
        float density,
        float restitution);

    /// <returns>1 on success, 0 on any invalid input.</returns>
    [LibraryImport(LibraryName)]
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
    [LibraryImport(LibraryName)]
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
