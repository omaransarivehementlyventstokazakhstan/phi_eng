using System.Runtime.InteropServices;

namespace PhiEng.Native;

/// <summary>
/// Blittable 2D body descriptor for bulk creation
/// (<see cref="PhiNativeWorld.AddBodies"/>). Layout must match the Rust
/// <c>BodyDesc2D</c> struct field-for-field (<c>crates/engine/src/ffi.rs</c>):
/// pos(2×f32), tag(u32), a/b(2×f32), density(f32), isStatic(byte),
/// 3 pad bytes, friction/restitution(2×f32). Total: 36 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct BodyDesc2D
{
    public float PosX;
    public float PosY;
    /// <summary>0 = circle (<c>A</c> = radius), 1 = box (<c>A</c>/<c>B</c> = half extents).</summary>
    public uint ShapeTag;
    public float A;
    public float B;
    public float Density;
    /// <summary>0 = dynamic, nonzero = static.</summary>
    public byte IsStatic;
    private byte _pad0;
    private byte _pad1;
    private byte _pad2;
    public float Friction;
    public float Restitution;

    public BodyDesc2D(float posX, float posY, uint shapeTag, float a, float b,
        float density, bool isStatic, float friction, float restitution)
    {
        PosX = posX;
        PosY = posY;
        ShapeTag = shapeTag;
        A = a;
        B = b;
        Density = density;
        IsStatic = (byte)(isStatic ? 1 : 0);
        _pad0 = _pad1 = _pad2 = 0;
        Friction = friction;
        Restitution = restitution;
    }

    /// <summary>Circle descriptor (tag 0).</summary>
    public static BodyDesc2D Circle(float posX, float posY, float radius, float density,
        bool isStatic = false,
        float friction = Dynamics.PhysicsMaterial.DefaultFriction,
        float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
        => new(posX, posY, PhiNativeWorld.ShapeCircle, radius, 0.0f,
            density, isStatic, friction, restitution);

    /// <summary>Box descriptor (tag 1).</summary>
    public static BodyDesc2D Box(float posX, float posY, float halfX, float halfY, float density,
        bool isStatic = false,
        float friction = Dynamics.PhysicsMaterial.DefaultFriction,
        float restitution = Dynamics.PhysicsMaterial.DefaultRestitution)
        => new(posX, posY, PhiNativeWorld.ShapeBox, halfX, halfY,
            density, isStatic, friction, restitution);
}

/// <summary>
/// Blittable 3D body descriptor for bulk creation
/// (<see cref="PhiNativeWorld3.AddBodies"/>). Layout must match the Rust
/// <c>BodyDesc3D</c> struct field-for-field (<c>crates/engine/src/ffi.rs</c>).
/// Total: 44 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct BodyDesc3D
{
    public float PosX;
    public float PosY;
    public float PosZ;
    /// <summary>0 = sphere (<c>A</c> = radius), 1 = box (<c>A</c>/<c>B</c>/<c>C</c> = half extents), 2 = capsule (<c>A</c> = radius, <c>B</c> = half length).</summary>
    public uint ShapeTag;
    public float A;
    public float B;
    public float C;
    public float Density;
    /// <summary>0 = dynamic, nonzero = static.</summary>
    public byte IsStatic;
    private byte _pad0;
    private byte _pad1;
    private byte _pad2;
    public float Friction;
    public float Restitution;

    public BodyDesc3D(float posX, float posY, float posZ, uint shapeTag,
        float a, float b, float c, float density, bool isStatic,
        float friction, float restitution)
    {
        PosX = posX;
        PosY = posY;
        PosZ = posZ;
        ShapeTag = shapeTag;
        A = a;
        B = b;
        C = c;
        Density = density;
        IsStatic = (byte)(isStatic ? 1 : 0);
        _pad0 = _pad1 = _pad2 = 0;
        Friction = friction;
        Restitution = restitution;
    }
}
