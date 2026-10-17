using System.Numerics;
using PhiEng.Broadphase;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PhiEng.Dynamics;

/// <summary>
/// Save/load for 2D and 3D worlds (JSON + binary), mirroring Rust
/// <c>snapshot.rs</c> in crates/dynamics/src/snapshot.rs — same schema
/// version, same field semantics, same load modes.
/// </summary>
public enum SnapshotLoadMode
{
    /// <summary>Restore everything; stepping on is bit-identical to never saving.</summary>
    Exact,
    /// <summary>Restore structure + config; solver state starts cold.</summary>
    Fresh,
}

/// <summary>Snapshot format tags (mirror Rust FORMAT_JSON/FORMAT_BINARY).</summary>
public enum SnapshotFormat
{
    /// <summary>Human-readable JSON snapshots.</summary>
    Json = 0,
    /// <summary>Compact CRC-checked binary snapshots.</summary>
    Binary = 1,
}

/// <summary>Save/load for 2D and 3D worlds in both formats.</summary>
public static class Snapshot
{
    /// <summary>
    /// Snapshot schema version. Must match Rust SNAPSHOT_SCHEMA_VERSION.
    /// v4 adds the gravity config block and per-body gravity scales; v5
    /// adds the ambient-force config block (spring/drag/turbulence), the
    /// simulation clock, and per-body ambient scales. Writers emit v3 for
    /// default worlds and v4 for gravity-only worlds (byte-compatible
    /// with older files); anything using ambient features uses v5.
    /// </summary>
    public const uint SchemaVersion = 5;

    /// <summary>Schema that introduced the gravity block (v4).</summary>
    public const uint GravitySchemaVersion = 4;

    /// <summary>
    /// Oldest schema this build can read. v3 predates gravity fields;
    /// they load as defaults. Anything outside [3, 5] is rejected.
    /// </summary>
    public const uint MinReadableSchemaVersion = 3;

    private const byte KindWorld2 = 2;
    private const byte KindWorld3 = 3;
    private static readonly byte[] Magic = [(byte)'P', (byte)'H', (byte)'E', (byte)'1'];

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    // ============ public World2 API ============

    /// <summary>Serializes a 2D world to JSON.</summary>
    public static string World2ToJson(World2 world, SnapshotLoadMode mode)
        => JsonSerializer.Serialize(World2Dto.From(world, mode), JsonOptions);

    /// <summary>Deserializes a 2D world from JSON (rejects unknown versions).</summary>
    public static World2 World2FromJson(string text, SnapshotLoadMode mode)
    {
        World2Dto? dto;
        try
        {
            dto = JsonSerializer.Deserialize<World2Dto>(text);
        }
        catch (JsonException ex)
        {
            throw new ArgumentException($"Malformed world JSON: {ex.Message}.");
        }
        if (dto == null) throw new ArgumentException("Malformed world JSON.");
        return World2FromDto(dto, mode);
    }

    /// <summary>Serializes a 2D world to the versioned binary envelope.</summary>
    public static byte[] World2ToBinary(World2 world, SnapshotLoadMode mode)
    {
        var dto = World2Dto.From(world, mode);
        return EncodeEnvelope(KindWorld2, dto.Schema, w => WriteWorld2Payload(w, dto));
    }

    /// <summary>Deserializes a 2D world from a binary envelope (checks magic, version, CRC).</summary>
    public static World2 World2FromBinary(byte[] bytes, SnapshotLoadMode mode)
    {
        var (version, reader) = DecodeEnvelope(bytes, KindWorld2);
        var dto = ReadWorld2Payload(reader, version);
        return World2FromDto(dto, mode);
    }

    /// <summary>Saves a 2D world to a file (format auto-selected by <paramref name="format"/>).</summary>
    public static void World2SaveFile(World2 world, string path, SnapshotFormat format, SnapshotLoadMode mode)
    {
        if (format == SnapshotFormat.Json)
            File.WriteAllText(path, World2ToJson(world, mode));
        else if (format == SnapshotFormat.Binary)
            File.WriteAllBytes(path, World2ToBinary(world, mode));
        else
            throw new ArgumentException($"Unknown snapshot format: {format}.");
    }

    /// <summary>Loads a 2D world from a file (JSON vs binary sniffed by content).</summary>
    public static World2 World2LoadFile(string path, SnapshotLoadMode mode)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int i = 0;
        while (i < bytes.Length && char.IsWhiteSpace((char)bytes[i])) i++;
        if (i < bytes.Length && bytes[i] == (byte)'{')
            return World2FromJson(System.Text.Encoding.UTF8.GetString(bytes), mode);
        return World2FromBinary(bytes, mode);
    }

    // ============ public World3 API ============

    /// <summary>Serializes a 3D world to JSON.</summary>
    public static string World3ToJson(World3 world, SnapshotLoadMode mode)
        => JsonSerializer.Serialize(World3Dto.From(world, mode), JsonOptions);

    /// <summary>Deserializes a 3D world from JSON (rejects unknown versions).</summary>
    public static World3 World3FromJson(string text, SnapshotLoadMode mode)
    {
        World3Dto? dto;
        try
        {
            dto = JsonSerializer.Deserialize<World3Dto>(text);
        }
        catch (JsonException ex)
        {
            throw new ArgumentException($"Malformed world JSON: {ex.Message}.");
        }
        if (dto == null) throw new ArgumentException("Malformed world JSON.");
        return World3FromDto(dto, mode);
    }

    /// <summary>Serializes a 3D world to the versioned binary envelope.</summary>
    public static byte[] World3ToBinary(World3 world, SnapshotLoadMode mode)
    {
        var dto = World3Dto.From(world, mode);
        return EncodeEnvelope(KindWorld3, dto.Schema, w => WriteWorld3Payload(w, dto));
    }

    /// <summary>Deserializes a 3D world from a binary envelope (checks magic, version, CRC).</summary>
    public static World3 World3FromBinary(byte[] bytes, SnapshotLoadMode mode)
    {
        var (version, reader) = DecodeEnvelope(bytes, KindWorld3);
        var dto = ReadWorld3Payload(reader, version);
        return World3FromDto(dto, mode);
    }

    /// <summary>Saves a 3D world to a file (format auto-selected by <paramref name="format"/>).</summary>
    public static void World3SaveFile(World3 world, string path, SnapshotFormat format, SnapshotLoadMode mode)
    {
        if (format == SnapshotFormat.Json)
            File.WriteAllText(path, World3ToJson(world, mode));
        else if (format == SnapshotFormat.Binary)
            File.WriteAllBytes(path, World3ToBinary(world, mode));
        else
            throw new ArgumentException($"Unknown snapshot format: {format}.");
    }

    /// <summary>Loads a 3D world from a file (JSON vs binary sniffed by content).</summary>
    public static World3 World3LoadFile(string path, SnapshotLoadMode mode)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int i = 0;
        while (i < bytes.Length && char.IsWhiteSpace((char)bytes[i])) i++;
        if (i < bytes.Length && bytes[i] == (byte)'{')
            return World3FromJson(System.Text.Encoding.UTF8.GetString(bytes), mode);
        return World3FromBinary(bytes, mode);
    }

    // ============ envelope ============

    private static (uint Version, BinaryReader Reader) DecodeEnvelope(byte[] bytes, byte wantKind)
    {
        if (bytes.Length < 13) throw new ArgumentException("Snapshot too short.");
        if (bytes[0] != (byte)'P' || bytes[1] != (byte)'H' || bytes[2] != (byte)'E' || bytes[3] != (byte)'1')
            throw new ArgumentException("Bad snapshot magic.");
        uint version = BitConverter.ToUInt32(bytes, 4);
        if (version < MinReadableSchemaVersion || version > SchemaVersion)
            throw new ArgumentException($"Unsupported snapshot schema version: {version}.");
        if (bytes[8] != wantKind) throw new ArgumentException("Snapshot kind mismatch.");
        uint len = BitConverter.ToUInt32(bytes, 9);
        if (bytes.Length != 13 + len + 4) throw new ArgumentException("Snapshot length mismatch.");
        var payload = bytes.Skip(13).Take((int)len).ToArray();
        uint want = BitConverter.ToUInt32(bytes, 13 + (int)len);
        if (Crc32(payload, 0, payload.Length) != want)
            throw new ArgumentException("Snapshot CRC mismatch.");
        return (version, new BinaryReader(new MemoryStream(payload)));
    }

    // CRC32 (IEEE, table-driven) to match Rust crc32fast on the same bytes.
    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }

    private static uint Crc32(byte[] data, int offset, int count)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < count; i++)
            crc = CrcTable[(crc ^ data[offset + i]) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }

    // ============ DTOs (field names match the Rust JSON schema) ============

    private sealed class Vec2Dto
    {
        [JsonPropertyName("x")] public float X { get; set; }
        [JsonPropertyName("y")] public float Y { get; set; }
    }

    private sealed class Vec3Dto
    {
        [JsonPropertyName("x")] public float X { get; set; }
        [JsonPropertyName("y")] public float Y { get; set; }
        [JsonPropertyName("z")] public float Z { get; set; }
    }

    private sealed class MaterialDto
    {
        [JsonPropertyName("friction")] public float Friction { get; set; }
        [JsonPropertyName("restitution")] public float Restitution { get; set; }
        [JsonPropertyName("density")] public float Density { get; set; }
    }

    private sealed class Shape2Dto
    {
        [JsonPropertyName("type")] public string Type { get; set; } = "";
        [JsonPropertyName("radius")] public float Radius { get; set; }
        [JsonPropertyName("half_extents")] public Vec2Dto? HalfExtents { get; set; }
        [JsonPropertyName("vertices")] public List<Vec2Dto>? Vertices { get; set; }
    }

    private sealed class Shape3Dto
    {
        [JsonPropertyName("type")] public string Type { get; set; } = "";
        [JsonPropertyName("radius")] public float Radius { get; set; }
        [JsonPropertyName("half_extents")] public Vec3Dto? HalfExtents { get; set; }
        [JsonPropertyName("half_length")] public float HalfLength { get; set; }
        [JsonPropertyName("vertices")] public List<Vec3Dto>? Vertices { get; set; }
    }

    private sealed class Body2Dto
    {
        [JsonPropertyName("id")] public int Id { get; set; }
        [JsonPropertyName("is_static")] public bool IsStatic { get; set; }
        [JsonPropertyName("is_sleeping")] public bool IsSleeping { get; set; }
        [JsonPropertyName("linear_damping")] public float LinearDamping { get; set; }
        [JsonPropertyName("angular_damping")] public float AngularDamping { get; set; }
        [JsonPropertyName("material")] public MaterialDto Material { get; set; } = new();
        [JsonPropertyName("shape")] public Shape2Dto? Shape { get; set; }
        [JsonPropertyName("position")] public Vec2Dto Position { get; set; } = new();
        [JsonPropertyName("rotation")] public float[] Rotation { get; set; } = new float[4];
        [JsonPropertyName("velocity")] public Vec2Dto Velocity { get; set; } = new();
        [JsonPropertyName("angular_velocity")] public float AngularVelocity { get; set; }
        /// <summary>
        /// Gravitational coupling multiplier. Initializer 1.0 so v3
        /// files (no key) decode to full coupling, mirroring Rust.
        /// </summary>
        [JsonPropertyName("gravity_scale")] public float GravityScale { get; set; } = 1.0f;
        /// <summary>Ambient-force coupling multiplier (schema v5; default 1.0).</summary>
        [JsonPropertyName("ambient_scale")] public float AmbientScale { get; set; } = 1.0f;
    }

    private sealed class Body3Dto
    {
        [JsonPropertyName("id")] public int Id { get; set; }
        [JsonPropertyName("is_static")] public bool IsStatic { get; set; }
        [JsonPropertyName("is_sleeping")] public bool IsSleeping { get; set; }
        [JsonPropertyName("linear_damping")] public float LinearDamping { get; set; }
        [JsonPropertyName("angular_damping")] public float AngularDamping { get; set; }
        [JsonPropertyName("material")] public MaterialDto Material { get; set; } = new();
        [JsonPropertyName("shape")] public Shape3Dto? Shape { get; set; }
        [JsonPropertyName("position")] public Vec3Dto Position { get; set; } = new();
        [JsonPropertyName("rotation")] public float[] Rotation { get; set; } = new float[4];
        [JsonPropertyName("velocity")] public Vec3Dto Velocity { get; set; } = new();
        [JsonPropertyName("angular_velocity")] public Vec3Dto AngularVelocity { get; set; } = new();
        /// <summary>See <c>Body2Dto.GravityScale</c>.</summary>
        [JsonPropertyName("gravity_scale")] public float GravityScale { get; set; } = 1.0f;
        /// <summary>See <c>Body2Dto.AmbientScale</c>.</summary>
        [JsonPropertyName("ambient_scale")] public float AmbientScale { get; set; } = 1.0f;
    }

    private sealed class Joint2Dto
    {
        [JsonPropertyName("type")] public string Type { get; set; } = "";
        [JsonPropertyName("id")] public int Id { get; set; }
        [JsonPropertyName("body_a")] public int BodyA { get; set; }
        [JsonPropertyName("body_b")] public int BodyB { get; set; }
        [JsonPropertyName("anchor_a")] public Vec2Dto AnchorA { get; set; } = new();
        [JsonPropertyName("anchor_b")] public Vec2Dto AnchorB { get; set; } = new();
        [JsonPropertyName("rest_length")] public float RestLength { get; set; }
        [JsonPropertyName("axis_a")] public Vec2Dto? AxisA { get; set; }
        [JsonPropertyName("limit_min")] public float LimitMin { get; set; }
        [JsonPropertyName("limit_max")] public float LimitMax { get; set; }
        [JsonPropertyName("ref_angle")] public float? RefAngle { get; set; }
        [JsonPropertyName("impulse")] public Vec2Dto Impulse { get; set; } = new();
        [JsonPropertyName("impulse_scalar")] public float ImpulseScalar { get; set; }
        [JsonPropertyName("angular_impulse")] public float AngularImpulse { get; set; }
        [JsonPropertyName("perp_impulse")] public float PerpImpulse { get; set; }
        [JsonPropertyName("limit_impulse")] public float LimitImpulse { get; set; }
    }

    private sealed class Contact2Dto
    {
        [JsonPropertyName("body_a")] public int BodyA { get; set; }
        [JsonPropertyName("body_b")] public int BodyB { get; set; }
        [JsonPropertyName("entries")] public List<ContactEntry2Dto> Entries { get; set; } = new();
    }

    private sealed class ContactEntry2Dto
    {
        [JsonPropertyName("point")] public Vec2Dto Point { get; set; } = new();
        [JsonPropertyName("normal")] public Vec2Dto Normal { get; set; } = new();
        [JsonPropertyName("normal_impulse")] public float NormalImpulse { get; set; }
        [JsonPropertyName("tangent_impulse")] public float TangentImpulse { get; set; }
    }

    private sealed class Contact3Dto
    {
        [JsonPropertyName("body_a")] public int BodyA { get; set; }
        [JsonPropertyName("body_b")] public int BodyB { get; set; }
        [JsonPropertyName("entries")] public List<ContactEntry3Dto> Entries { get; set; } = new();
    }

    private sealed class ContactEntry3Dto
    {
        [JsonPropertyName("point")] public Vec3Dto Point { get; set; } = new();
        [JsonPropertyName("normal")] public Vec3Dto Normal { get; set; } = new();
        [JsonPropertyName("normal_impulse")] public float NormalImpulse { get; set; }
        [JsonPropertyName("tangent1_impulse")] public float Tangent1Impulse { get; set; }
        [JsonPropertyName("tangent2_impulse")] public float Tangent2Impulse { get; set; }
    }

    private sealed class SolverDto
    {
        [JsonPropertyName("iterations")] public int Iterations { get; set; }
        [JsonPropertyName("slop")] public float Slop { get; set; }
        [JsonPropertyName("baumgarte")] public float Baumgarte { get; set; }
        [JsonPropertyName("early_exit_tolerance")] public float EarlyExitTolerance { get; set; }
    }

    private sealed class ValidationDto
    {
        [JsonPropertyName("check_finite")] public bool CheckFinite { get; set; }
        [JsonPropertyName("max_penetration")] public float MaxPenetration { get; set; }
        [JsonPropertyName("max_mass_ratio")] public float MaxMassRatio { get; set; }
        [JsonPropertyName("max_coordinate")] public float MaxCoordinate { get; set; }
        [JsonPropertyName("max_velocity")] public float MaxVelocity { get; set; }
        [JsonPropertyName("max_angular_velocity")] public float MaxAngularVelocity { get; set; }
        [JsonPropertyName("log_warnings")] public bool LogWarnings { get; set; }
        [JsonPropertyName("auto_correct")] public bool AutoCorrect { get; set; }
    }

    /// <summary>
    /// Gravity config DTO (schema v4). Property initializers equal the
    /// engine defaults, so files predating the gravity block (v3 and
    /// hand-written JSON) decode to defaults — the C# equivalent of
    /// Rust's #[serde(default)].
    /// </summary>
    private sealed class GravityConfig2Dto
    {
        [JsonPropertyName("mode")] public uint Mode { get; set; } = 0;
        [JsonPropertyName("integrator")] public uint Integrator { get; set; } = 0;
        [JsonPropertyName("big_g")] public float BigG { get; set; } = 1.0f;
        [JsonPropertyName("softening")] public float Softening { get; set; } = 0.0f;
        [JsonPropertyName("attractor")] public Vec2Dto Attractor { get; set; } = new();
        [JsonPropertyName("attractor_mu")] public float AttractorMu { get; set; } = 0.0f;
        [JsonPropertyName("theta")] public float Theta { get; set; } = 0.5f;

        public bool IsDefault() =>
            Mode == 0 && Integrator == 0 && BigG == 1.0f && Softening == 0.0f &&
            Attractor.X == 0.0f && Attractor.Y == 0.0f && AttractorMu == 0.0f &&
            Theta == 0.5f;
    }

    /// <summary>3D counterpart of <see cref="GravityConfig2Dto"/>.</summary>
    private sealed class GravityConfig3Dto
    {
        [JsonPropertyName("mode")] public uint Mode { get; set; } = 0;
        [JsonPropertyName("integrator")] public uint Integrator { get; set; } = 0;
        [JsonPropertyName("big_g")] public float BigG { get; set; } = 1.0f;
        [JsonPropertyName("softening")] public float Softening { get; set; } = 0.0f;
        [JsonPropertyName("attractor")] public Vec3Dto Attractor { get; set; } = new();
        [JsonPropertyName("attractor_mu")] public float AttractorMu { get; set; } = 0.0f;
        [JsonPropertyName("theta")] public float Theta { get; set; } = 0.5f;

        public bool IsDefault() =>
            Mode == 0 && Integrator == 0 && BigG == 1.0f && Softening == 0.0f &&
            Attractor.X == 0.0f && Attractor.Y == 0.0f && Attractor.Z == 0.0f &&
            AttractorMu == 0.0f && Theta == 0.5f;
    }

    /// <summary>Spring / SHM config DTO (schema v5).</summary>
    private sealed class Spring2Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("anchor")] public Vec2Dto Anchor { get; set; } = new();
        [JsonPropertyName("stiffness")] public float Stiffness { get; set; }
        [JsonPropertyName("damping")] public float Damping { get; set; }
        [JsonPropertyName("max_displacement")] public float MaxDisplacement { get; set; }

        public bool IsDefault() =>
            !Enabled && Anchor.X == 0.0f && Anchor.Y == 0.0f &&
            Stiffness == 0.0f && Damping == 0.0f && MaxDisplacement == 0.0f;
    }

    /// <summary>Drag config DTO (schema v5).</summary>
    private sealed class Drag2Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("linear")] public float Linear { get; set; }
        [JsonPropertyName("quadratic")] public float Quadratic { get; set; }
        [JsonPropertyName("air_density")] public float AirDensity { get; set; } = 1.0f;
        [JsonPropertyName("angular")] public float Angular { get; set; }
        [JsonPropertyName("wind")] public Vec2Dto Wind { get; set; } = new();

        public bool IsDefault() =>
            !Enabled && Linear == 0.0f && Quadratic == 0.0f && AirDensity == 1.0f &&
            Angular == 0.0f && Wind.X == 0.0f && Wind.Y == 0.0f;
    }

    /// <summary>Turbulence config DTO (schema v5).</summary>
    private sealed class Turbulence2Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("amplitude")] public float Amplitude { get; set; }
        [JsonPropertyName("frequency")] public float Frequency { get; set; } = 1.0f;
        [JsonPropertyName("time_scale")] public float TimeScale { get; set; } = 1.0f;
        [JsonPropertyName("seed")] public uint Seed { get; set; }
        [JsonPropertyName("octaves")] public int Octaves { get; set; } = 3;
        [JsonPropertyName("divergence_free")] public bool DivergenceFree { get; set; }

        public bool IsDefault() =>
            !Enabled && Amplitude == 0.0f && Frequency == 1.0f && TimeScale == 1.0f &&
            Seed == 0 && Octaves == 3 && !DivergenceFree;
    }

    /// <summary>Ambient-force config DTO (schema v5, 2D).</summary>
    private sealed class AmbientConfig2Dto
    {
        [JsonPropertyName("spring")] public Spring2Dto Spring { get; set; } = new();
        [JsonPropertyName("drag")] public Drag2Dto Drag { get; set; } = new();
        [JsonPropertyName("turbulence")] public Turbulence2Dto Turbulence { get; set; } = new();

        public bool IsDefault() => Spring.IsDefault() && Drag.IsDefault() && Turbulence.IsDefault();
    }

    /// <summary>Spring / SHM config DTO (schema v5, 3D).</summary>
    private sealed class Spring3Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("anchor")] public Vec3Dto Anchor { get; set; } = new();
        [JsonPropertyName("stiffness")] public float Stiffness { get; set; }
        [JsonPropertyName("damping")] public float Damping { get; set; }
        [JsonPropertyName("max_displacement")] public float MaxDisplacement { get; set; }

        public bool IsDefault() =>
            !Enabled && Anchor.X == 0.0f && Anchor.Y == 0.0f && Anchor.Z == 0.0f &&
            Stiffness == 0.0f && Damping == 0.0f && MaxDisplacement == 0.0f;
    }

    /// <summary>Drag config DTO (schema v5, 3D).</summary>
    private sealed class Drag3Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("linear")] public float Linear { get; set; }
        [JsonPropertyName("quadratic")] public float Quadratic { get; set; }
        [JsonPropertyName("air_density")] public float AirDensity { get; set; } = 1.0f;
        [JsonPropertyName("angular")] public float Angular { get; set; }
        [JsonPropertyName("wind")] public Vec3Dto Wind { get; set; } = new();

        public bool IsDefault() =>
            !Enabled && Linear == 0.0f && Quadratic == 0.0f && AirDensity == 1.0f &&
            Angular == 0.0f && Wind.X == 0.0f && Wind.Y == 0.0f && Wind.Z == 0.0f;
    }

    /// <summary>Turbulence config DTO (schema v5, 3D).</summary>
    private sealed class Turbulence3Dto
    {
        [JsonPropertyName("enabled")] public bool Enabled { get; set; }
        [JsonPropertyName("amplitude")] public float Amplitude { get; set; }
        [JsonPropertyName("frequency")] public float Frequency { get; set; } = 1.0f;
        [JsonPropertyName("time_scale")] public float TimeScale { get; set; } = 1.0f;
        [JsonPropertyName("seed")] public uint Seed { get; set; }
        [JsonPropertyName("octaves")] public int Octaves { get; set; } = 3;
        [JsonPropertyName("divergence_free")] public bool DivergenceFree { get; set; }

        public bool IsDefault() =>
            !Enabled && Amplitude == 0.0f && Frequency == 1.0f && TimeScale == 1.0f &&
            Seed == 0 && Octaves == 3 && !DivergenceFree;
    }

    /// <summary>Ambient-force config DTO (schema v5, 3D).</summary>
    private sealed class AmbientConfig3Dto
    {
        [JsonPropertyName("spring")] public Spring3Dto Spring { get; set; } = new();
        [JsonPropertyName("drag")] public Drag3Dto Drag { get; set; } = new();
        [JsonPropertyName("turbulence")] public Turbulence3Dto Turbulence { get; set; } = new();

        public bool IsDefault() => Spring.IsDefault() && Drag.IsDefault() && Turbulence.IsDefault();
    }

    private sealed class FluidParticle2Dto
    {
        [JsonPropertyName("position")] public Vec2Dto Position { get; set; } = new();
        [JsonPropertyName("velocity")] public Vec2Dto Velocity { get; set; } = new();
    }

    private sealed class FluidConfig2Dto
    {
        [JsonPropertyName("particle_mass")] public float ParticleMass { get; set; } = 0.0625f;
        [JsonPropertyName("rest_density")] public float RestDensity { get; set; } = 1.0f;
        [JsonPropertyName("smoothing_length")] public float SmoothingLength { get; set; } = 0.5f;
        [JsonPropertyName("stiffness")] public float Stiffness { get; set; } = 100.0f;
        [JsonPropertyName("gamma")] public float Gamma { get; set; } = 1.0f;
        [JsonPropertyName("artificial_viscosity")] public float ArtificialViscosity { get; set; }
        [JsonPropertyName("surface_tension")] public float SurfaceTension { get; set; }
        [JsonPropertyName("cfl")] public float Cfl { get; set; }
        [JsonPropertyName("max_substeps")] public int MaxSubsteps { get; set; } = 1;
        [JsonPropertyName("viscosity")] public float Viscosity { get; set; } = 0.3f;
        [JsonPropertyName("drag")] public float Drag { get; set; } = 4.0f;
        [JsonPropertyName("max_velocity")] public float MaxVelocity { get; set; } = 30.0f;
        [JsonPropertyName("particle_radius")] public float ParticleRadius { get; set; } = 0.125f;
        [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
    }

    private sealed class FluidState2Dto
    {
        [JsonPropertyName("config")] public FluidConfig2Dto Config { get; set; } = new();
        [JsonPropertyName("particles")] public List<FluidParticle2Dto> Particles { get; set; } = new();
    }

    private sealed class FluidParticle3Dto
    {
        [JsonPropertyName("position")] public Vec3Dto Position { get; set; } = new();
        [JsonPropertyName("velocity")] public Vec3Dto Velocity { get; set; } = new();
    }

    private sealed class FluidConfig3Dto
    {
        [JsonPropertyName("particle_mass")] public float ParticleMass { get; set; } = 0.0625f;
        [JsonPropertyName("rest_density")] public float RestDensity { get; set; } = 1.0f;
        [JsonPropertyName("smoothing_length")] public float SmoothingLength { get; set; } = 0.5f;
        [JsonPropertyName("stiffness")] public float Stiffness { get; set; } = 100.0f;
        [JsonPropertyName("gamma")] public float Gamma { get; set; } = 1.0f;
        [JsonPropertyName("artificial_viscosity")] public float ArtificialViscosity { get; set; }
        [JsonPropertyName("surface_tension")] public float SurfaceTension { get; set; }
        [JsonPropertyName("cfl")] public float Cfl { get; set; }
        [JsonPropertyName("max_substeps")] public int MaxSubsteps { get; set; } = 1;
        [JsonPropertyName("viscosity")] public float Viscosity { get; set; } = 0.3f;
        [JsonPropertyName("drag")] public float Drag { get; set; } = 4.0f;
        [JsonPropertyName("max_velocity")] public float MaxVelocity { get; set; } = 30.0f;
        [JsonPropertyName("particle_radius")] public float ParticleRadius { get; set; } = 0.125f;
        [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
    }

    private sealed class FluidState3Dto
    {
        [JsonPropertyName("config")] public FluidConfig3Dto Config { get; set; } = new();
        [JsonPropertyName("particles")] public List<FluidParticle3Dto> Particles { get; set; } = new();
    }

    private sealed class World2Dto
    {
        [JsonPropertyName("schema")] public uint Schema { get; set; }
        [JsonPropertyName("gravity")] public Vec2Dto Gravity { get; set; } = new();
        /// <summary>
        /// Gravity config (schema v4). Initializer default so v3 files
        /// decode to the legacy uniform-field behavior.
        /// </summary>
        [JsonPropertyName("gravity_config")] public GravityConfig2Dto GravityConfig { get; set; } = new();
        /// <summary>Ambient-force config (schema v5); defaults so pre-v5 files decode to no-op.</summary>
        [JsonPropertyName("ambient")] public AmbientConfig2Dto Ambient { get; set; } = new();
        /// <summary>Simulation clock (schema v5); 0 for pre-v5 files.</summary>
        [JsonPropertyName("time")] public float Time { get; set; }
        [JsonPropertyName("dt")] public float Dt { get; set; }
        [JsonPropertyName("linear_sleep_threshold")] public float LinearSleepThreshold { get; set; }
        [JsonPropertyName("angular_sleep_threshold")] public float AngularSleepThreshold { get; set; }
        [JsonPropertyName("solver")] public SolverDto Solver { get; set; } = new();
        [JsonPropertyName("validation")] public ValidationDto Validation { get; set; } = new();
        [JsonPropertyName("next_body_id")] public int NextBodyId { get; set; }
        [JsonPropertyName("next_joint_id")] public int NextJointId { get; set; }
        [JsonPropertyName("cell_size")] public float CellSize { get; set; }
        [JsonPropertyName("bodies")] public List<Body2Dto> Bodies { get; set; } = new();
        [JsonPropertyName("joints")] public List<Joint2Dto> Joints { get; set; } = new();
        [JsonPropertyName("contacts")] public List<Contact2Dto> Contacts { get; set; } = new();
        [JsonPropertyName("fluid")] public FluidState2Dto Fluid { get; set; } = new();

        public static World2Dto From(World2 w, SnapshotLoadMode mode)
        {
            // Conditional schema: default worlds stay v3 (byte-compatible
            // with older files); gravity-only worlds v4; ambient worlds
            // v5. Mirrors the Rust writer rule.
            bool hasAmbient = w.Ambient.IsActive() || w.Bodies.Any(b => b.AmbientScale != 1.0f)
                || w.Fluid.Config.Gamma != 1.0f || w.Fluid.Config.ArtificialViscosity != 0.0f
                || w.Fluid.Config.SurfaceTension != 0.0f
                || w.Fluid.Config.Cfl != 0.0f || w.Fluid.Config.MaxSubsteps != 1;
            bool hasGravity = !w.GravityConfig.IsDefault() || w.Bodies.Any(b => b.GravityScale != 1.0f);
            uint schema = hasAmbient ? SchemaVersion : hasGravity ? GravitySchemaVersion : MinReadableSchemaVersion;
            var dto = new World2Dto
            {
                Schema = schema,
                Gravity = V2(w.Gravity),
                GravityConfig = new GravityConfig2Dto
                {
                    Mode = (uint)w.GravityConfig.Mode,
                    Integrator = (uint)w.GravityConfig.Integrator,
                    BigG = w.GravityConfig.BigG,
                    Softening = w.GravityConfig.Softening,
                    Attractor = V2(w.GravityConfig.Attractor),
                    AttractorMu = w.GravityConfig.AttractorMu,
                    Theta = w.GravityConfig.Theta,
                },
                Ambient = new AmbientConfig2Dto
                {
                    Spring = new Spring2Dto
                    {
                        Enabled = w.Ambient.Spring.Enabled,
                        Anchor = V2(w.Ambient.Spring.Anchor),
                        Stiffness = w.Ambient.Spring.Stiffness,
                        Damping = w.Ambient.Spring.Damping,
                        MaxDisplacement = w.Ambient.Spring.MaxDisplacement,
                    },
                    Drag = new Drag2Dto
                    {
                        Enabled = w.Ambient.Drag.Enabled,
                        Linear = w.Ambient.Drag.Linear,
                        Quadratic = w.Ambient.Drag.Quadratic,
                        AirDensity = w.Ambient.Drag.AirDensity,
                        Angular = w.Ambient.Drag.Angular,
                        Wind = V2(w.Ambient.Drag.Wind),
                    },
                    Turbulence = new Turbulence2Dto
                    {
                        Enabled = w.Ambient.Turbulence.Enabled,
                        Amplitude = w.Ambient.Turbulence.Amplitude,
                        Frequency = w.Ambient.Turbulence.Frequency,
                        TimeScale = w.Ambient.Turbulence.TimeScale,
                        Seed = w.Ambient.Turbulence.Seed,
                        Octaves = w.Ambient.Turbulence.Octaves,
                        DivergenceFree = w.Ambient.Turbulence.DivergenceFree,
                    },
                },
                Time = hasAmbient ? w.Time : 0.0f,
                Dt = w.Dt,
                LinearSleepThreshold = w.LinearSleepThreshold,
                AngularSleepThreshold = w.AngularSleepThreshold,
                Solver = new SolverDto
                {
                    Iterations = w.Solver.Iterations,
                    Slop = w.Solver.Slop,
                    Baumgarte = w.Solver.Baumgarte,
                    EarlyExitTolerance = w.Solver.EarlyExitTolerance,
                },
                Validation = new ValidationDto
                {
                    CheckFinite = w.Validation.CheckFinite,
                    MaxPenetration = w.Validation.MaxPenetration,
                    MaxMassRatio = w.Validation.MaxMassRatio,
                    MaxCoordinate = w.Validation.MaxCoordinate,
                    MaxVelocity = w.Validation.MaxVelocity,
                    MaxAngularVelocity = w.Validation.MaxAngularVelocity,
                    LogWarnings = w.Validation.LogWarnings,
                    AutoCorrect = w.Validation.AutoCorrect,
                },
                NextBodyId = w.NextBodyId,
                NextJointId = w.NextJointId,
                CellSize = w.Broadphase.CellSize,
            };
            foreach (var b in w.Bodies)
            {
                var r = b.Transform.Rotation;
                var shape = b.Shape == null ? null : b.Shape switch
                {
                    CircleShape c => new Shape2Dto { Type = "circle", Radius = c.Radius },
                    BoxShape2 x => new Shape2Dto
                    {
                        Type = "box",
                        HalfExtents = new Vec2Dto { X = x.HalfExtents.X, Y = x.HalfExtents.Y },
                    },
                    PolygonShape p => new Shape2Dto
                    {
                        Type = "polygon",
                        Vertices = p.Vertices.ConvertAll(v => new Vec2Dto { X = v.X, Y = v.Y }),
                    },
                    _ => throw new ArgumentException("Unknown 2D shape."),
                };
                dto.Bodies.Add(new Body2Dto
                {
                    Id = b.Id,
                    IsStatic = b.IsStatic,
                    IsSleeping = b.IsSleeping,
                    LinearDamping = b.LinearDamping,
                    AngularDamping = b.AngularDamping,
                    Material = new MaterialDto
                    {
                        Friction = b.Material.Friction,
                        Restitution = b.Material.Restitution,
                        Density = b.Material.Density,
                    },
                    Shape = shape,
                    Position = V2(b.Transform.Position),
                    Rotation = new[] { r.M11, r.M12, r.M21, r.M22 },
                    Velocity = V2(b.Velocity),
                    AngularVelocity = b.AngularVelocity,
                    GravityScale = b.GravityScale,
                    AmbientScale = b.AmbientScale,
                });
            }
            bool fresh = mode == SnapshotLoadMode.Fresh;
            foreach (var j in w.Joints)
            {
                var d = new Joint2Dto { Id = j.Id, BodyA = j.BodyA, BodyB = j.BodyB };
                switch (j)
                {
                    case DistanceJoint x:
                        d.Type = "distance";
                        d.AnchorA = V2(x.AnchorA); d.AnchorB = V2(x.AnchorB);
                        d.RestLength = x.RestLength;
                        d.ImpulseScalar = fresh ? 0.0f : x.Impulse;
                        break;
                    case HingeJoint x:
                        d.Type = "hinge";
                        d.AnchorA = V2(x.AnchorA); d.AnchorB = V2(x.AnchorB);
                        d.LimitMin = x.LimitMin; d.LimitMax = x.LimitMax;
                        d.Impulse = V2(fresh ? Vector2.Zero : x.Impulse);
                        d.AngularImpulse = fresh ? 0.0f : x.AngularImpulse;
                        break;
                    case PrismaticJoint x:
                        d.Type = "prismatic";
                        d.AnchorA = V2(x.AnchorA); d.AnchorB = V2(x.AnchorB);
                        d.AxisA = V2(x.AxisA);
                        d.LimitMin = x.LimitMin; d.LimitMax = x.LimitMax;
                        d.RefAngle = fresh ? null : x.RefAngle;
                        d.PerpImpulse = fresh ? 0.0f : x.PerpImpulse;
                        d.AngularImpulse = fresh ? 0.0f : x.AngularImpulse;
                        d.LimitImpulse = fresh ? 0.0f : x.LimitImpulse;
                        break;
                    case FixedJoint x:
                        d.Type = "fixed";
                        d.AnchorA = V2(x.AnchorA); d.AnchorB = V2(x.AnchorB);
                        d.RefAngle = fresh ? null : x.RefAngle;
                        d.Impulse = V2(fresh ? Vector2.Zero : x.Impulse);
                        d.AngularImpulse = fresh ? 0.0f : x.AngularImpulse;
                        break;
                    default:
                        throw new ArgumentException("Unknown 2D joint.");
                }
                dto.Joints.Add(d);
            }
            if (!fresh)
            {
                foreach (var kv in w.ContactCache.OrderBy(kv => kv.Key.Item1).ThenBy(kv => kv.Key.Item2))
                {
                    var c = new Contact2Dto { BodyA = kv.Key.Item1, BodyB = kv.Key.Item2 };
                    foreach (var e in kv.Value)
                    {
                        c.Entries.Add(new ContactEntry2Dto
                        {
                            Point = V2(e.Point),
                            Normal = V2(e.Normal),
                            NormalImpulse = e.NormalImpulse,
                            TangentImpulse = e.TangentImpulse,
                        });
                    }
                    dto.Contacts.Add(c);
                }
            }
            dto.Fluid = new FluidState2Dto
            {
                Config = new FluidConfig2Dto
                {
                    ParticleMass = w.Fluid.Config.ParticleMass,
                    RestDensity = w.Fluid.Config.RestDensity,
                    SmoothingLength = w.Fluid.Config.SmoothingLength,
                    Stiffness = w.Fluid.Config.Stiffness,
                    Gamma = w.Fluid.Config.Gamma,
                    ArtificialViscosity = w.Fluid.Config.ArtificialViscosity,
                    SurfaceTension = w.Fluid.Config.SurfaceTension,
                    Cfl = w.Fluid.Config.Cfl,
                    MaxSubsteps = w.Fluid.Config.MaxSubsteps,
                    Viscosity = w.Fluid.Config.Viscosity,
                    Drag = w.Fluid.Config.Drag,
                    MaxVelocity = w.Fluid.Config.MaxVelocity,
                    ParticleRadius = w.Fluid.Config.ParticleRadius,
                    Enabled = w.Fluid.Config.Enabled,
                },
            };
            foreach (var p in w.Fluid.Particles)
            {
                dto.Fluid.Particles.Add(new FluidParticle2Dto
                {
                    Position = V2(p.Position),
                    Velocity = V2(p.Velocity),
                });
            }
            return dto;
        }
    }

    private sealed class Joint3Dto
    {
        [JsonPropertyName("type")] public string Type { get; set; } = "";
        [JsonPropertyName("id")] public int Id { get; set; }
        [JsonPropertyName("body_a")] public int BodyA { get; set; }
        [JsonPropertyName("body_b")] public int BodyB { get; set; }
        [JsonPropertyName("anchor_a")] public Vec3Dto? AnchorA { get; set; }
        [JsonPropertyName("anchor_b")] public Vec3Dto? AnchorB { get; set; }
        [JsonPropertyName("rest_length")] public float? RestLength { get; set; }
        [JsonPropertyName("impulse")] public float? Impulse { get; set; }
        [JsonPropertyName("axis_a")] public Vec3Dto? AxisA { get; set; }
        [JsonPropertyName("axis_b")] public Vec3Dto? AxisB { get; set; }
        [JsonPropertyName("limit_min")] public float? LimitMin { get; set; }
        [JsonPropertyName("limit_max")] public float? LimitMax { get; set; }
        [JsonPropertyName("impulse_vec")] public Vec3Dto? ImpulseVec { get; set; }
        [JsonPropertyName("align_u1")] public float? AlignU1 { get; set; }
        [JsonPropertyName("align_u2")] public float? AlignU2 { get; set; }
        [JsonPropertyName("twist_impulse")] public float? TwistImpulse { get; set; }
        [JsonPropertyName("ref_orientation")] public float[]? RefOrientation { get; set; }
        [JsonPropertyName("perp_u1")] public float? PerpU1 { get; set; }
        [JsonPropertyName("perp_u2")] public float? PerpU2 { get; set; }
        [JsonPropertyName("angular_impulse")] public Vec3Dto? AngularImpulse { get; set; }
        [JsonPropertyName("limit_impulse")] public float? LimitImpulse { get; set; }
    }

    private sealed class World3Dto
    {
        [JsonPropertyName("schema")] public uint Schema { get; set; }
        [JsonPropertyName("gravity")] public Vec3Dto Gravity { get; set; } = new();
        /// <summary>See <c>World2Dto.GravityConfig</c>.</summary>
        [JsonPropertyName("gravity_config")] public GravityConfig3Dto GravityConfig { get; set; } = new();
        /// <summary>See <c>World2Dto.Ambient</c>.</summary>
        [JsonPropertyName("ambient")] public AmbientConfig3Dto Ambient { get; set; } = new();
        /// <summary>See <c>World2Dto.Time</c>.</summary>
        [JsonPropertyName("time")] public float Time { get; set; }
        [JsonPropertyName("dt")] public float Dt { get; set; }
        [JsonPropertyName("linear_sleep_threshold")] public float LinearSleepThreshold { get; set; }
        [JsonPropertyName("angular_sleep_threshold")] public float AngularSleepThreshold { get; set; }
        [JsonPropertyName("solver")] public SolverDto Solver { get; set; } = new();
        [JsonPropertyName("validation")] public ValidationDto Validation { get; set; } = new();
        [JsonPropertyName("next_body_id")] public int NextBodyId { get; set; }
        [JsonPropertyName("next_joint_id")] public int NextJointId { get; set; }
        [JsonPropertyName("bodies")] public List<Body3Dto> Bodies { get; set; } = new();
        [JsonPropertyName("joints")] public List<Joint3Dto> Joints { get; set; } = new();
        [JsonPropertyName("contacts")] public List<Contact3Dto> Contacts { get; set; } = new();
        [JsonPropertyName("fluid")] public FluidState3Dto Fluid { get; set; } = new();

        public static World3Dto From(World3 w, SnapshotLoadMode mode)
        {
            // Conditional schema: see World2Dto.From.
            bool hasAmbient = w.Ambient.IsActive() || w.Bodies.Any(b => b.AmbientScale != 1.0f)
                || w.Fluid.Config.Gamma != 1.0f || w.Fluid.Config.ArtificialViscosity != 0.0f
                || w.Fluid.Config.SurfaceTension != 0.0f
                || w.Fluid.Config.Cfl != 0.0f || w.Fluid.Config.MaxSubsteps != 1;
            bool hasGravity = !w.GravityConfig.IsDefault() || w.Bodies.Any(b => b.GravityScale != 1.0f);
            uint schema = hasAmbient ? SchemaVersion : hasGravity ? GravitySchemaVersion : MinReadableSchemaVersion;
            var dto = new World3Dto
            {
                Schema = schema,
                Gravity = V3(w.Gravity),
                GravityConfig = new GravityConfig3Dto
                {
                    Mode = (uint)w.GravityConfig.Mode,
                    Integrator = (uint)w.GravityConfig.Integrator,
                    BigG = w.GravityConfig.BigG,
                    Softening = w.GravityConfig.Softening,
                    Attractor = V3(w.GravityConfig.Attractor),
                    AttractorMu = w.GravityConfig.AttractorMu,
                    Theta = w.GravityConfig.Theta,
                },
                Ambient = new AmbientConfig3Dto
                {
                    Spring = new Spring3Dto
                    {
                        Enabled = w.Ambient.Spring.Enabled,
                        Anchor = V3(w.Ambient.Spring.Anchor),
                        Stiffness = w.Ambient.Spring.Stiffness,
                        Damping = w.Ambient.Spring.Damping,
                        MaxDisplacement = w.Ambient.Spring.MaxDisplacement,
                    },
                    Drag = new Drag3Dto
                    {
                        Enabled = w.Ambient.Drag.Enabled,
                        Linear = w.Ambient.Drag.Linear,
                        Quadratic = w.Ambient.Drag.Quadratic,
                        AirDensity = w.Ambient.Drag.AirDensity,
                        Angular = w.Ambient.Drag.Angular,
                        Wind = V3(w.Ambient.Drag.Wind),
                    },
                    Turbulence = new Turbulence3Dto
                    {
                        Enabled = w.Ambient.Turbulence.Enabled,
                        Amplitude = w.Ambient.Turbulence.Amplitude,
                        Frequency = w.Ambient.Turbulence.Frequency,
                        TimeScale = w.Ambient.Turbulence.TimeScale,
                        Seed = w.Ambient.Turbulence.Seed,
                        Octaves = w.Ambient.Turbulence.Octaves,
                        DivergenceFree = w.Ambient.Turbulence.DivergenceFree,
                    },
                },
                Time = hasAmbient ? w.Time : 0.0f,
                Dt = w.Dt,
                LinearSleepThreshold = w.LinearSleepThreshold,
                AngularSleepThreshold = w.AngularSleepThreshold,
                Solver = new SolverDto
                {
                    Iterations = w.Solver.Iterations,
                    Slop = w.Solver.Slop,
                    Baumgarte = w.Solver.Baumgarte,
                    EarlyExitTolerance = w.Solver.EarlyExitTolerance,
                },
                Validation = new ValidationDto
                {
                    CheckFinite = w.Validation.CheckFinite,
                    MaxPenetration = w.Validation.MaxPenetration,
                    MaxMassRatio = w.Validation.MaxMassRatio,
                    MaxCoordinate = w.Validation.MaxCoordinate,
                    MaxVelocity = w.Validation.MaxVelocity,
                    MaxAngularVelocity = w.Validation.MaxAngularVelocity,
                    LogWarnings = w.Validation.LogWarnings,
                    AutoCorrect = w.Validation.AutoCorrect,
                },
                NextBodyId = w.NextBodyId,
                NextJointId = w.NextJointId,
            };
            foreach (var j in w.Joints)
            {
                var jd = new Joint3Dto
                {
                    Id = j.Id,
                    BodyA = j.Bodies().Item1,
                    BodyB = j.Bodies().Item2,
                };
                switch (j.Type)
                {
                    case Joint3Type.Distance3:
                        var d = j.Distance!.Value;
                        jd.Type = "distance3";
                        jd.AnchorA = V3(d.AnchorA);
                        jd.AnchorB = V3(d.AnchorB);
                        jd.RestLength = d.RestLength;
                        jd.Impulse = d.Impulse;
                        break;
                    case Joint3Type.Hinge3:
                        var h = j.Hinge!.Value;
                        jd.Type = "hinge3";
                        jd.AnchorA = V3(h.AnchorA);
                        jd.AnchorB = V3(h.AnchorB);
                        jd.AxisA = V3(h.AxisA);
                        jd.AxisB = V3(h.AxisB);
                        jd.LimitMin = h.LimitMin;
                        jd.LimitMax = h.LimitMax;
                        jd.ImpulseVec = V3(h.Impulse);
                        jd.AlignU1 = h.AlignU1;
                        jd.AlignU2 = h.AlignU2;
                        jd.TwistImpulse = h.TwistImpulse;
                        break;
                    case Joint3Type.Prismatic3:
                        var p = j.Prismatic!.Value;
                        jd.Type = "prismatic3";
                        jd.AnchorA = V3(p.AnchorA);
                        jd.AnchorB = V3(p.AnchorB);
                        jd.AxisA = V3(p.AxisA);
                        jd.LimitMin = p.LimitMin;
                        jd.LimitMax = p.LimitMax;
                        jd.RefOrientation = p.RefOrientation.HasValue
                            ? new[] { p.RefOrientation.Value.X, p.RefOrientation.Value.Y, p.RefOrientation.Value.Z, p.RefOrientation.Value.W }
                            : null;
                        jd.PerpU1 = p.PerpU1;
                        jd.PerpU2 = p.PerpU2;
                        jd.AngularImpulse = V3(p.AngularImpulse);
                        jd.LimitImpulse = p.LimitImpulse;
                        break;
                    case Joint3Type.Fixed3:
                        var f = j.Fixed!.Value;
                        jd.Type = "fixed3";
                        jd.AnchorA = V3(f.AnchorA);
                        jd.AnchorB = V3(f.AnchorB);
                        jd.RefOrientation = f.RefOrientation.HasValue
                            ? new[] { f.RefOrientation.Value.X, f.RefOrientation.Value.Y, f.RefOrientation.Value.Z, f.RefOrientation.Value.W }
                            : null;
                        jd.ImpulseVec = V3(f.Impulse);
                        jd.AngularImpulse = V3(f.AngularImpulse);
                        break;
                }
                dto.Joints.Add(jd);
            }
            if (mode == SnapshotLoadMode.Fresh)
            {
                foreach (var jd in dto.Joints)
                {
                    jd.Impulse = 0f;
                    jd.ImpulseVec = new Vec3Dto();
                    jd.AlignU1 = 0f;
                    jd.AlignU2 = 0f;
                    jd.TwistImpulse = 0f;
                    jd.PerpU1 = 0f;
                    jd.PerpU2 = 0f;
                    jd.AngularImpulse = new Vec3Dto();
                    jd.LimitImpulse = 0f;
                    jd.RefOrientation = null;
                }
                dto.Contacts.Clear();
            }
            foreach (var b in w.Bodies)
            {
                Shape3Dto? shape = b.Shape switch
                {
                    SphereShape3 s => new Shape3Dto { Type = "sphere", Radius = s.Radius },
                    BoxShape3 x => new Shape3Dto
                    {
                        Type = "box",
                        HalfExtents = new Vec3Dto
                        {
                            X = x.HalfExtents.X,
                            Y = x.HalfExtents.Y,
                            Z = x.HalfExtents.Z,
                        },
                    },
                    CapsuleShape3 c => new Shape3Dto
                    {
                        Type = "capsule",
                        Radius = c.Radius,
                        HalfLength = c.HalfLength,
                    },
                    ConvexHullShape3 h => new Shape3Dto
                    {
                        Type = "convexhull",
                        Vertices = h.Vertices.ConvertAll(v => new Vec3Dto
                        {
                            X = v.X,
                            Y = v.Y,
                            Z = v.Z,
                        }),
                    },
                    null => null,
                    _ => throw new ArgumentException("Unknown 3D shape."),
                };
                var q = b.Transform.Rotation;
                dto.Bodies.Add(new Body3Dto
                {
                    Id = b.Id,
                    IsStatic = b.IsStatic,
                    IsSleeping = b.IsSleeping,
                    LinearDamping = b.LinearDamping,
                    AngularDamping = b.AngularDamping,
                    Material = new MaterialDto
                    {
                        Friction = b.Material.Friction,
                        Restitution = b.Material.Restitution,
                        Density = b.Material.Density,
                    },
                    Shape = shape,
                    Position = V3(b.Transform.Position),
                    Rotation = new[] { q.X, q.Y, q.Z, q.W },
                    Velocity = V3(b.Velocity),
                    AngularVelocity = V3(b.AngularVelocity),
                    GravityScale = b.GravityScale,
                    AmbientScale = b.AmbientScale,
                });
            }
            if (mode == SnapshotLoadMode.Exact)
            {
                foreach (var kv in w.ContactCache.OrderBy(kv => kv.Key.Item1).ThenBy(kv => kv.Key.Item2))
                {
                    var c = new Contact3Dto { BodyA = kv.Key.Item1, BodyB = kv.Key.Item2 };
                    foreach (var e in kv.Value)
                    {
                        c.Entries.Add(new ContactEntry3Dto
                        {
                            Point = V3(e.Point),
                            Normal = V3(e.Normal),
                            NormalImpulse = e.NormalImpulse,
                            Tangent1Impulse = e.Tangent1Impulse,
                            Tangent2Impulse = e.Tangent2Impulse,
                        });
                    }
                    dto.Contacts.Add(c);
                }
            }
            dto.Fluid = new FluidState3Dto
            {
                Config = new FluidConfig3Dto
                {
                    ParticleMass = w.Fluid.Config.ParticleMass,
                    RestDensity = w.Fluid.Config.RestDensity,
                    SmoothingLength = w.Fluid.Config.SmoothingLength,
                    Stiffness = w.Fluid.Config.Stiffness,
                    Gamma = w.Fluid.Config.Gamma,
                    ArtificialViscosity = w.Fluid.Config.ArtificialViscosity,
                    SurfaceTension = w.Fluid.Config.SurfaceTension,
                    Cfl = w.Fluid.Config.Cfl,
                    MaxSubsteps = w.Fluid.Config.MaxSubsteps,
                    Viscosity = w.Fluid.Config.Viscosity,
                    Drag = w.Fluid.Config.Drag,
                    MaxVelocity = w.Fluid.Config.MaxVelocity,
                    ParticleRadius = w.Fluid.Config.ParticleRadius,
                    Enabled = w.Fluid.Config.Enabled,
                },
            };
            foreach (var p in w.Fluid.Particles)
            {
                dto.Fluid.Particles.Add(new FluidParticle3Dto
                {
                    Position = V3(p.Position),
                    Velocity = V3(p.Velocity),
                });
            }
            return dto;
        }
    }

    private static Vec2Dto V2(Vector2 v) => new() { X = v.X, Y = v.Y };
    private static Vector2 UV2(Vec2Dto v) => new(v.X, v.Y);
    private static Vec3Dto V3(Vector3 v) => new() { X = v.X, Y = v.Y, Z = v.Z };
    private static Vector3 UV3(Vec3Dto v) => new(v.X, v.Y, v.Z);

    private static void CheckSchema(uint schema)
    {
        if (schema < MinReadableSchemaVersion || schema > SchemaVersion)
            throw new ArgumentException($"Unsupported snapshot schema version: {schema}.");
    }

    private static float Fin(float v, string what) =>
        float.IsFinite(v) ? v : throw new ArgumentException($"Non-finite snapshot {what}.");

    private static GravityConfig2 GravityConfigFrom(GravityConfig2Dto c)
    {
        if (c.Mode > 3) throw new ArgumentException($"Unknown gravity mode: {c.Mode}.");
        if (c.Integrator > 1) throw new ArgumentException($"Unknown integrator: {c.Integrator}.");
        if (c.Softening < 0.0f) throw new ArgumentException("Gravity softening must be >= 0.");
        if (c.Theta < 0.0f) throw new ArgumentException("Gravity theta must be >= 0.");
        return new GravityConfig2
        {
            Mode = (GravityMode)c.Mode,
            Integrator = (Integrator)c.Integrator,
            BigG = Fin(c.BigG, "gravity.big_g"),
            Softening = Fin(c.Softening, "gravity.softening"),
            Attractor = new Vector2(Fin(c.Attractor.X, "gravity.ax"), Fin(c.Attractor.Y, "gravity.ay")),
            AttractorMu = Fin(c.AttractorMu, "gravity.mu"),
            Theta = Fin(c.Theta, "gravity.theta"),
        };
    }

    private static GravityConfig3 GravityConfigFrom(GravityConfig3Dto c)
    {
        if (c.Mode > 3) throw new ArgumentException($"Unknown gravity mode: {c.Mode}.");
        if (c.Integrator > 1) throw new ArgumentException($"Unknown integrator: {c.Integrator}.");
        if (c.Softening < 0.0f) throw new ArgumentException("Gravity softening must be >= 0.");
        if (c.Theta < 0.0f) throw new ArgumentException("Gravity theta must be >= 0.");
        return new GravityConfig3
        {
            Mode = (GravityMode)c.Mode,
            Integrator = (Integrator)c.Integrator,
            BigG = Fin(c.BigG, "gravity.big_g"),
            Softening = Fin(c.Softening, "gravity.softening"),
            Attractor = new Vector3(
                Fin(c.Attractor.X, "gravity.ax"),
                Fin(c.Attractor.Y, "gravity.ay"),
                Fin(c.Attractor.Z, "gravity.az")),
            AttractorMu = Fin(c.AttractorMu, "gravity.mu"),
            Theta = Fin(c.Theta, "gravity.theta"),
        };
    }

    private static AmbientConfig2 AmbientConfigFrom(AmbientConfig2Dto c)
    {
        if (c.Spring.Stiffness < 0.0f || c.Spring.Damping < 0.0f || c.Spring.MaxDisplacement < 0.0f)
            throw new ArgumentException("Ambient spring coefficients must be >= 0.");
        if (c.Drag.Linear < 0.0f || c.Drag.Quadratic < 0.0f || c.Drag.AirDensity < 0.0f || c.Drag.Angular < 0.0f)
            throw new ArgumentException("Ambient drag coefficients must be >= 0.");
        if (c.Turbulence.Amplitude < 0.0f)
            throw new ArgumentException("Ambient turbulence amplitude must be >= 0.");
        return new AmbientConfig2
        {
            Spring = new SpringConfig2
            {
                Enabled = c.Spring.Enabled,
                Anchor = new Vector2(
                    Fin(c.Spring.Anchor.X, "ambient.spring.ax"),
                    Fin(c.Spring.Anchor.Y, "ambient.spring.ay")),
                Stiffness = Fin(c.Spring.Stiffness, "ambient.spring.stiffness"),
                Damping = Fin(c.Spring.Damping, "ambient.spring.damping"),
                MaxDisplacement = Fin(c.Spring.MaxDisplacement, "ambient.spring.max_displacement"),
            },
            Drag = new DragConfig2
            {
                Enabled = c.Drag.Enabled,
                Linear = Fin(c.Drag.Linear, "ambient.drag.linear"),
                Quadratic = Fin(c.Drag.Quadratic, "ambient.drag.quadratic"),
                AirDensity = Fin(c.Drag.AirDensity, "ambient.drag.air_density"),
                Angular = Fin(c.Drag.Angular, "ambient.drag.angular"),
                Wind = new Vector2(
                    Fin(c.Drag.Wind.X, "ambient.drag.wind.x"),
                    Fin(c.Drag.Wind.Y, "ambient.drag.wind.y")),
            },
            Turbulence = new TurbulenceConfig2
            {
                Enabled = c.Turbulence.Enabled,
                Amplitude = Fin(c.Turbulence.Amplitude, "ambient.turbulence.amplitude"),
                Frequency = Fin(c.Turbulence.Frequency, "ambient.turbulence.frequency"),
                TimeScale = Fin(c.Turbulence.TimeScale, "ambient.turbulence.time_scale"),
                Seed = c.Turbulence.Seed,
                Octaves = c.Turbulence.Octaves,
                DivergenceFree = c.Turbulence.DivergenceFree,
            },
        };
    }

    private static AmbientConfig3 AmbientConfigFrom(AmbientConfig3Dto c)
    {
        if (c.Spring.Stiffness < 0.0f || c.Spring.Damping < 0.0f || c.Spring.MaxDisplacement < 0.0f)
            throw new ArgumentException("Ambient spring coefficients must be >= 0.");
        if (c.Drag.Linear < 0.0f || c.Drag.Quadratic < 0.0f || c.Drag.AirDensity < 0.0f || c.Drag.Angular < 0.0f)
            throw new ArgumentException("Ambient drag coefficients must be >= 0.");
        if (c.Turbulence.Amplitude < 0.0f)
            throw new ArgumentException("Ambient turbulence amplitude must be >= 0.");
        return new AmbientConfig3
        {
            Spring = new SpringConfig3
            {
                Enabled = c.Spring.Enabled,
                Anchor = new Vector3(
                    Fin(c.Spring.Anchor.X, "ambient.spring.ax"),
                    Fin(c.Spring.Anchor.Y, "ambient.spring.ay"),
                    Fin(c.Spring.Anchor.Z, "ambient.spring.az")),
                Stiffness = Fin(c.Spring.Stiffness, "ambient.spring.stiffness"),
                Damping = Fin(c.Spring.Damping, "ambient.spring.damping"),
                MaxDisplacement = Fin(c.Spring.MaxDisplacement, "ambient.spring.max_displacement"),
            },
            Drag = new DragConfig3
            {
                Enabled = c.Drag.Enabled,
                Linear = Fin(c.Drag.Linear, "ambient.drag.linear"),
                Quadratic = Fin(c.Drag.Quadratic, "ambient.drag.quadratic"),
                AirDensity = Fin(c.Drag.AirDensity, "ambient.drag.air_density"),
                Angular = Fin(c.Drag.Angular, "ambient.drag.angular"),
                Wind = new Vector3(
                    Fin(c.Drag.Wind.X, "ambient.drag.wind.x"),
                    Fin(c.Drag.Wind.Y, "ambient.drag.wind.y"),
                    Fin(c.Drag.Wind.Z, "ambient.drag.wind.z")),
            },
            Turbulence = new TurbulenceConfig3
            {
                Enabled = c.Turbulence.Enabled,
                Amplitude = Fin(c.Turbulence.Amplitude, "ambient.turbulence.amplitude"),
                Frequency = Fin(c.Turbulence.Frequency, "ambient.turbulence.frequency"),
                TimeScale = Fin(c.Turbulence.TimeScale, "ambient.turbulence.time_scale"),
                Seed = c.Turbulence.Seed,
                Octaves = c.Turbulence.Octaves,
                DivergenceFree = c.Turbulence.DivergenceFree,
            },
        };
    }

    private static Shape2 ShapeFrom(Shape2Dto? s)
    {
        if (s == null) throw new ArgumentException("Missing shape.");
        return s.Type switch
        {
            "circle" => new CircleShape(Fin(s.Radius, "circle.radius")),
            "box" when s.HalfExtents != null => new BoxShape2(
                new Vector2(Fin(s.HalfExtents.X, "box.x"), Fin(s.HalfExtents.Y, "box.y"))),
            "polygon" when s.Vertices != null => new PolygonShape(
                s.Vertices.ConvertAll(v => new Vector2(Fin(v.X, "poly.x"), Fin(v.Y, "poly.y")))),
            _ => throw new ArgumentException("Unknown 2D shape."),
        };
    }

    private static Shape3 ShapeFrom(Shape3Dto? s)
    {
        if (s == null) throw new ArgumentException("Missing shape.");
        return s.Type switch
        {
            "sphere" => new SphereShape3(Fin(s.Radius, "sphere.radius")),
            "box" when s.HalfExtents != null => new BoxShape3(
                new Vector3(
                    Fin(s.HalfExtents.X, "box.x"),
                    Fin(s.HalfExtents.Y, "box.y"),
                    Fin(s.HalfExtents.Z, "box.z"))),
            "capsule" => new CapsuleShape3(
                Fin(s.Radius, "capsule.radius"), Fin(s.HalfLength, "capsule.half_length")),
            "convexhull" when s.Vertices != null => new ConvexHullShape3(
                s.Vertices.ConvertAll(v => new Vector3(
                    Fin(v.X, "hull.x"), Fin(v.Y, "hull.y"), Fin(v.Z, "hull.z")))),
            _ => throw new ArgumentException("Unknown 3D shape."),
        };
    }

    private static World2 World2FromDto(World2Dto dto, SnapshotLoadMode mode)
    {
        CheckSchema(dto.Schema);
        if (dto.Bodies.Count > 1_000_000 || dto.Joints.Count > 1_000_000)
            throw new ArgumentException("Snapshot body/joint count exceeds 1,000,000.");
        var w = new World2
        {
            Gravity = UV2(dto.Gravity),
            GravityConfig = GravityConfigFrom(dto.GravityConfig),
            Ambient = AmbientConfigFrom(dto.Ambient),
            Time = Fin(dto.Time, "time"),
            Dt = Fin(dto.Dt, "dt"),
            LinearSleepThreshold = Fin(dto.LinearSleepThreshold, "sleep"),
            AngularSleepThreshold = Fin(dto.AngularSleepThreshold, "sleep"),
            Broadphase = new SpatialHash2(Fin(dto.CellSize, "cell_size")),
            Validation = new ValidationConfig
            {
                CheckFinite = dto.Validation.CheckFinite,
                MaxPenetration = Fin(dto.Validation.MaxPenetration, "validation"),
                MaxMassRatio = Fin(dto.Validation.MaxMassRatio, "validation"),
                MaxCoordinate = Fin(dto.Validation.MaxCoordinate, "validation"),
                MaxVelocity = Fin(dto.Validation.MaxVelocity, "validation"),
                MaxAngularVelocity = Fin(dto.Validation.MaxAngularVelocity, "validation"),
                LogWarnings = dto.Validation.LogWarnings,
                AutoCorrect = dto.Validation.AutoCorrect,
            },
        };
        w.Solver.Iterations = dto.Solver.Iterations > 1024
            ? throw new ArgumentException("Solver iterations exceed 1024.")
            : dto.Solver.Iterations;
        w.Solver.Slop = Fin(dto.Solver.Slop, "solver");
        w.Solver.Baumgarte = Fin(dto.Solver.Baumgarte, "solver");
        w.Solver.EarlyExitTolerance = Fin(dto.Solver.EarlyExitTolerance, "solver");

        foreach (var b in dto.Bodies)
        {
            if (b.Id != w.Bodies.Count)
                throw new ArgumentException($"Body id out of order (got {b.Id}).");
            RigidBody2 body = b.IsStatic ? RigidBody2.StaticBody(w.NextBodyId) : new RigidBody2(w.NextBodyId);
            var t = body.Transform;
            t.Position = UV2(b.Position);
            body.Transform = t;
            if (b.Shape != null)
            {
                var shape = ShapeFrom(b.Shape);
                body.SetShape(shape, Fin(b.Material.Density, "density"));
                body.SetMaterial(new PhysicsMaterial(
                    Fin(b.Material.Friction, "friction"),
                    Fin(b.Material.Restitution, "restitution"),
                    Fin(b.Material.Density, "density")));
            }
            else
            {
                body.Material = new PhysicsMaterial(
                    Fin(b.Material.Friction, "friction"),
                    Fin(b.Material.Restitution, "restitution"),
                    Fin(b.Material.Density, "density"));
            }
            if (b.Rotation.Length != 4)
                throw new ArgumentException("Body rotation must have 4 components.");
            t = body.Transform;
            t.Rotation = new Matrix3x2(
                Fin(b.Rotation[0], "rot"), Fin(b.Rotation[1], "rot"),
                Fin(b.Rotation[2], "rot"), Fin(b.Rotation[3], "rot"), 0.0f, 0.0f);
            body.Transform = t;
            body.Velocity = UV2(b.Velocity);
            body.AngularVelocity = Fin(b.AngularVelocity, "spin");
            body.LinearDamping = Fin(b.LinearDamping, "damping");
            body.AngularDamping = Fin(b.AngularDamping, "damping");
            body.GravityScale = Fin(b.GravityScale, "gravity_scale");
            body.AmbientScale = Fin(b.AmbientScale, "ambient_scale");
            body.IsSleeping = b.IsSleeping;
            w.AddBody(body);
        }
        if (dto.NextBodyId != w.Bodies.Count)
            throw new ArgumentException("next_body_id disagrees with body count.");

        foreach (var j in dto.Joints)
        {
            if (j.Id != w.Joints.Count)
                throw new ArgumentException($"Joint id out of order (got {j.Id}).");
            Joint2 joint = j.Type switch
            {
                "distance" => new DistanceJoint(j.BodyA, j.BodyB, UV2(j.AnchorA), UV2(j.AnchorB),
                    Fin(j.RestLength, "rest"), 0.0f, w.Bodies.Count),
                "hinge" => new HingeJoint(j.BodyA, j.BodyB, UV2(j.AnchorA), UV2(j.AnchorB),
                    Fin(j.LimitMin, "lo"), Fin(j.LimitMax, "hi"), 0.0f, w.Bodies.Count),
                "prismatic" => new PrismaticJoint(j.BodyA, j.BodyB, UV2(j.AnchorA), UV2(j.AnchorB),
                    UV2(j.AxisA ?? throw new ArgumentException("Missing prismatic axis.")),
                    Fin(j.LimitMin, "lo"), Fin(j.LimitMax, "hi"), 0.0f, w.Bodies.Count),
                "fixed" => new FixedJoint(j.BodyA, j.BodyB, UV2(j.AnchorA), UV2(j.AnchorB),
                    0.0f, w.Bodies.Count),
                _ => throw new ArgumentException($"Unknown joint type: {j.Type}."),
            };
            if (mode == SnapshotLoadMode.Exact)
            {
                switch (joint)
                {
                    case DistanceJoint x:
                        x.Impulse = Fin(j.ImpulseScalar, "impulse");
                        break;
                    case HingeJoint x:
                        x.Impulse = UV2(j.Impulse);
                        x.AngularImpulse = Fin(j.AngularImpulse, "angular");
                        break;
                    case PrismaticJoint x:
                        x.RefAngle = j.RefAngle is float a ? Fin(a, "ref") : null;
                        x.PerpImpulse = Fin(j.PerpImpulse, "perp");
                        x.AngularImpulse = Fin(j.AngularImpulse, "angular");
                        x.LimitImpulse = Fin(j.LimitImpulse, "limit");
                        break;
                    case FixedJoint x:
                        x.RefAngle = j.RefAngle is float f ? Fin(f, "ref") : null;
                        x.Impulse = UV2(j.Impulse);
                        x.AngularImpulse = Fin(j.AngularImpulse, "angular");
                        break;
                }
            }
            w.AddJoint(joint);
        }
        if (dto.NextJointId != w.Joints.Count)
            throw new ArgumentException("next_joint_id disagrees with joint count.");

        if (mode == SnapshotLoadMode.Exact)
        {
            foreach (var c in dto.Contacts)
            {
                if (c.BodyA < 0 || c.BodyB < 0 || c.BodyA >= w.Bodies.Count || c.BodyB >= w.Bodies.Count)
                    throw new ArgumentException("Contact pair references unknown body.");
                if (c.BodyA == c.BodyB)
                    throw new ArgumentException("Contact pair must reference two bodies.");
                if (c.Entries.Count > 16)
                    throw new ArgumentException("Contact entry count exceeds 16.");
                var list = new List<CachedContact>(c.Entries.Count);
                foreach (var e in c.Entries)
                {
                    list.Add(new CachedContact
                    {
                        Point = UV2(e.Point),
                        Normal = UV2(e.Normal),
                        NormalImpulse = Fin(e.NormalImpulse, "jn"),
                        TangentImpulse = Fin(e.TangentImpulse, "jt"),
                    });
                }
                w.ContactCache[(c.BodyA, c.BodyB)] = list;
            }
        }
        ApplyFluid2(w, dto.Fluid);
        return w;
    }

    private static void ApplyFluid2(World2 w, FluidState2Dto f)
    {
        if (f.Particles.Count > FluidState2.MaxParticles)
            throw new ArgumentException("Fluid particle count exceeds maximum.");
        var cfg = f.Config;
        w.Fluid.Config = new FluidConfig2
        {
            ParticleMass = Fin(cfg.ParticleMass, "fluid"),
            RestDensity = Fin(cfg.RestDensity, "fluid"),
            SmoothingLength = Fin(cfg.SmoothingLength, "fluid"),
            Stiffness = Fin(cfg.Stiffness, "fluid"),
            Gamma = Fin(cfg.Gamma, "fluid"),
            ArtificialViscosity = Fin(cfg.ArtificialViscosity, "fluid"),
            SurfaceTension = Fin(cfg.SurfaceTension, "fluid"),
            Cfl = Fin(cfg.Cfl, "fluid"),
            MaxSubsteps = cfg.MaxSubsteps,
            Viscosity = Fin(cfg.Viscosity, "fluid"),
            Drag = Fin(cfg.Drag, "fluid"),
            MaxVelocity = Fin(cfg.MaxVelocity, "fluid"),
            ParticleRadius = Fin(cfg.ParticleRadius, "fluid"),
            Enabled = cfg.Enabled,
        };
        w.Fluid.Clear();
        foreach (var p in f.Particles)
        {
            w.Fluid.Particles.Add(new FluidParticle2
            {
                Position = UV2(p.Position),
                Velocity = UV2(p.Velocity),
            });
        }
    }

    private static World3 World3FromDto(World3Dto dto, SnapshotLoadMode mode)
    {
        CheckSchema(dto.Schema);
        if (dto.Bodies.Count > 1_000_000)
            throw new ArgumentException("Snapshot body count exceeds 1,000,000.");
        var w = new World3
        {
            Gravity = UV3(dto.Gravity),
            GravityConfig = GravityConfigFrom(dto.GravityConfig),
            Ambient = AmbientConfigFrom(dto.Ambient),
            Time = Fin(dto.Time, "time"),
            Dt = Fin(dto.Dt, "dt"),
            LinearSleepThreshold = Fin(dto.LinearSleepThreshold, "sleep"),
            AngularSleepThreshold = Fin(dto.AngularSleepThreshold, "sleep"),
            Validation = new ValidationConfig
            {
                CheckFinite = dto.Validation.CheckFinite,
                MaxPenetration = Fin(dto.Validation.MaxPenetration, "validation"),
                MaxMassRatio = Fin(dto.Validation.MaxMassRatio, "validation"),
                MaxCoordinate = Fin(dto.Validation.MaxCoordinate, "validation"),
                MaxVelocity = Fin(dto.Validation.MaxVelocity, "validation"),
                MaxAngularVelocity = Fin(dto.Validation.MaxAngularVelocity, "validation"),
                LogWarnings = dto.Validation.LogWarnings,
                AutoCorrect = dto.Validation.AutoCorrect,
            },
        };
        w.Solver.Iterations = dto.Solver.Iterations > 1024
            ? throw new ArgumentException("Solver iterations exceed 1024.")
            : dto.Solver.Iterations;
        w.Solver.Slop = Fin(dto.Solver.Slop, "solver");
        w.Solver.Baumgarte = Fin(dto.Solver.Baumgarte, "solver");
        w.Solver.EarlyExitTolerance = Fin(dto.Solver.EarlyExitTolerance, "solver");

        foreach (var b in dto.Bodies)
        {
            if (b.Id != w.Bodies.Count)
                throw new ArgumentException($"Body id out of order (got {b.Id}).");
            RigidBody3 body = b.IsStatic ? RigidBody3.StaticBody(w.NextBodyId) : new RigidBody3(w.NextBodyId);
            var t = body.Transform;
            t.Position = UV3(b.Position);
            body.Transform = t;
            if (b.Shape != null)
            {
                var shape = ShapeFrom(b.Shape);
                body.SetShape(shape, Fin(b.Material.Density, "density"));
                body.SetMaterial(new PhysicsMaterial(
                    Fin(b.Material.Friction, "friction"),
                    Fin(b.Material.Restitution, "restitution"),
                    Fin(b.Material.Density, "density")));
            }
            else
            {
                body.Material = new PhysicsMaterial(
                    Fin(b.Material.Friction, "friction"),
                    Fin(b.Material.Restitution, "restitution"),
                    Fin(b.Material.Density, "density"));
            }
            if (b.Rotation.Length != 4)
                throw new ArgumentException("Body rotation must have 4 components.");
            t = body.Transform;
            var q = new Quaternion(
                Fin(b.Rotation[0], "q"), Fin(b.Rotation[1], "q"),
                Fin(b.Rotation[2], "q"), Fin(b.Rotation[3], "q"));
            t.Rotation = Quaternion.Normalize(q);
            body.Transform = t;
            body.Velocity = UV3(b.Velocity);
            body.AngularVelocity = UV3(b.AngularVelocity);
            body.LinearDamping = Fin(b.LinearDamping, "damping");
            body.AngularDamping = Fin(b.AngularDamping, "damping");
            body.GravityScale = Fin(b.GravityScale, "gravity_scale");
            body.AmbientScale = Fin(b.AmbientScale, "ambient_scale");
            body.IsSleeping = b.IsSleeping;
            w.AddBody(body);
        }
        if (dto.NextBodyId != w.Bodies.Count)
            throw new ArgumentException("next_body_id disagrees with body count.");

        foreach (var jd in dto.Joints)
        {
            if (jd.Id != w.Joints.Count)
                throw new ArgumentException($"Joint id out of order (got {jd.Id}).");
            Joint3 joint = jd.Type switch
            {
                "distance3" => Joint3.MakeDistance(new DistanceJoint3
                {
                    Id = jd.Id,
                    BodyA = jd.BodyA,
                    BodyB = jd.BodyB,
                    AnchorA = UV3(jd.AnchorA ?? throw new ArgumentException("Missing anchor_a.")),
                    AnchorB = UV3(jd.AnchorB ?? throw new ArgumentException("Missing anchor_b.")),
                    RestLength = Fin(jd.RestLength ?? throw new ArgumentException("Missing rest_length."), "rest_length"),
                    Impulse = mode == SnapshotLoadMode.Exact
                        ? Fin(jd.Impulse ?? 0f, "impulse")
                        : 0f,
                }),
                "hinge3" => Joint3.MakeHinge(new HingeJoint3
                {
                    Id = jd.Id,
                    BodyA = jd.BodyA,
                    BodyB = jd.BodyB,
                    AnchorA = UV3(jd.AnchorA ?? throw new ArgumentException("Missing anchor_a.")),
                    AnchorB = UV3(jd.AnchorB ?? throw new ArgumentException("Missing anchor_b.")),
                    AxisA = Joint3Math.ValidateAxis(UV3(jd.AxisA ?? throw new ArgumentException("Missing axis_a."))),
                    AxisB = Joint3Math.ValidateAxis(UV3(jd.AxisB ?? throw new ArgumentException("Missing axis_b."))),
                    LimitMin = Fin(jd.LimitMin ?? throw new ArgumentException("Missing limit_min."), "limit_min"),
                    LimitMax = Fin(jd.LimitMax ?? throw new ArgumentException("Missing limit_max."), "limit_max"),
                    Impulse = mode == SnapshotLoadMode.Exact
                        ? UV3(jd.ImpulseVec ?? new Vec3Dto())
                        : Vector3.Zero,
                    AlignU1 = mode == SnapshotLoadMode.Exact ? Fin(jd.AlignU1 ?? 0f, "align_u1") : 0f,
                    AlignU2 = mode == SnapshotLoadMode.Exact ? Fin(jd.AlignU2 ?? 0f, "align_u2") : 0f,
                    TwistImpulse = mode == SnapshotLoadMode.Exact ? Fin(jd.TwistImpulse ?? 0f, "twist_impulse") : 0f,
                }),
                "prismatic3" => Joint3.MakePrismatic(new PrismaticJoint3
                {
                    Id = jd.Id,
                    BodyA = jd.BodyA,
                    BodyB = jd.BodyB,
                    AnchorA = UV3(jd.AnchorA ?? throw new ArgumentException("Missing anchor_a.")),
                    AnchorB = UV3(jd.AnchorB ?? throw new ArgumentException("Missing anchor_b.")),
                    AxisA = Joint3Math.ValidateAxis(UV3(jd.AxisA ?? throw new ArgumentException("Missing axis_a."))),
                    LimitMin = Fin(jd.LimitMin ?? throw new ArgumentException("Missing limit_min."), "limit_min"),
                    LimitMax = Fin(jd.LimitMax ?? throw new ArgumentException("Missing limit_max."), "limit_max"),
                    RefOrientation = mode == SnapshotLoadMode.Exact && jd.RefOrientation != null
                        ? new Quaternion(
                            Fin(jd.RefOrientation[0], "ref_q"),
                            Fin(jd.RefOrientation[1], "ref_q"),
                            Fin(jd.RefOrientation[2], "ref_q"),
                            Fin(jd.RefOrientation[3], "ref_q"))
                        : null,
                    PerpU1 = mode == SnapshotLoadMode.Exact ? Fin(jd.PerpU1 ?? 0f, "perp_u1") : 0f,
                    PerpU2 = mode == SnapshotLoadMode.Exact ? Fin(jd.PerpU2 ?? 0f, "perp_u2") : 0f,
                    AngularImpulse = mode == SnapshotLoadMode.Exact
                        ? UV3(jd.AngularImpulse ?? new Vec3Dto())
                        : Vector3.Zero,
                    LimitImpulse = mode == SnapshotLoadMode.Exact ? Fin(jd.LimitImpulse ?? 0f, "limit_impulse") : 0f,
                }),
                "fixed3" => Joint3.MakeFixed(new FixedJoint3
                {
                    Id = jd.Id,
                    BodyA = jd.BodyA,
                    BodyB = jd.BodyB,
                    AnchorA = UV3(jd.AnchorA ?? throw new ArgumentException("Missing anchor_a.")),
                    AnchorB = UV3(jd.AnchorB ?? throw new ArgumentException("Missing anchor_b.")),
                    RefOrientation = mode == SnapshotLoadMode.Exact && jd.RefOrientation != null
                        ? new Quaternion(
                            Fin(jd.RefOrientation[0], "ref_q"),
                            Fin(jd.RefOrientation[1], "ref_q"),
                            Fin(jd.RefOrientation[2], "ref_q"),
                            Fin(jd.RefOrientation[3], "ref_q"))
                        : null,
                    Impulse = mode == SnapshotLoadMode.Exact
                        ? UV3(jd.ImpulseVec ?? new Vec3Dto())
                        : Vector3.Zero,
                    AngularImpulse = mode == SnapshotLoadMode.Exact
                        ? UV3(jd.AngularImpulse ?? new Vec3Dto())
                        : Vector3.Zero,
                }),
                _ => throw new ArgumentException($"Unknown 3D joint type: {jd.Type}"),
            };
            w.AddJoint(joint);
        }
        if (dto.NextJointId != w.Joints.Count)
            throw new ArgumentException("next_joint_id disagrees with joint count.");

        if (mode == SnapshotLoadMode.Exact)
        {
            foreach (var c in dto.Contacts)
            {
                if (c.BodyA < 0 || c.BodyB < 0 || c.BodyA >= w.Bodies.Count || c.BodyB >= w.Bodies.Count)
                    throw new ArgumentException("Contact pair references unknown body.");
                if (c.BodyA == c.BodyB)
                    throw new ArgumentException("Contact pair must reference two bodies.");
                if (c.Entries.Count > 16)
                    throw new ArgumentException("Contact entry count exceeds 16.");
                var list = new List<CachedContact3D>(c.Entries.Count);
                foreach (var e in c.Entries)
                {
                    list.Add(new CachedContact3D
                    {
                        Point = UV3(e.Point),
                        Normal = UV3(e.Normal),
                        NormalImpulse = Fin(e.NormalImpulse, "jn"),
                        Tangent1Impulse = Fin(e.Tangent1Impulse, "jt1"),
                        Tangent2Impulse = Fin(e.Tangent2Impulse, "jt2"),
                    });
                }
                w.ContactCache[(c.BodyA, c.BodyB)] = list;
            }
        }
        ApplyFluid3(w, dto.Fluid);
        return w;
    }

    private static void ApplyFluid3(World3 w, FluidState3Dto f)
    {
        if (f.Particles.Count > FluidState3.MaxParticles)
            throw new ArgumentException("Fluid particle count exceeds maximum.");
        var cfg = f.Config;
        w.Fluid.Config = new FluidConfig3
        {
            ParticleMass = Fin(cfg.ParticleMass, "fluid"),
            RestDensity = Fin(cfg.RestDensity, "fluid"),
            SmoothingLength = Fin(cfg.SmoothingLength, "fluid"),
            Stiffness = Fin(cfg.Stiffness, "fluid"),
            Gamma = Fin(cfg.Gamma, "fluid"),
            ArtificialViscosity = Fin(cfg.ArtificialViscosity, "fluid"),
            SurfaceTension = Fin(cfg.SurfaceTension, "fluid"),
            Cfl = Fin(cfg.Cfl, "fluid"),
            MaxSubsteps = cfg.MaxSubsteps,
            Viscosity = Fin(cfg.Viscosity, "fluid"),
            Drag = Fin(cfg.Drag, "fluid"),
            MaxVelocity = Fin(cfg.MaxVelocity, "fluid"),
            ParticleRadius = Fin(cfg.ParticleRadius, "fluid"),
            Enabled = cfg.Enabled,
        };
        w.Fluid.Clear();
        foreach (var p in f.Particles)
        {
            w.Fluid.Particles.Add(new FluidParticle3
            {
                Position = UV3(p.Position),
                Velocity = UV3(p.Velocity),
            });
        }
    }

    // ============ binary payloads (byte-identical layout to Rust) ============

    private static void WriteSolver(BinaryWriter w, SolverDto s)
    {
        w.Write(s.Iterations);
        w.Write(s.Slop);
        w.Write(s.Baumgarte);
        w.Write(s.EarlyExitTolerance);
    }

    private static void WriteValidation(BinaryWriter w, ValidationDto v)
    {
        w.Write(v.CheckFinite);
        w.Write(v.MaxPenetration);
        w.Write(v.MaxMassRatio);
        w.Write(v.MaxCoordinate);
        w.Write(v.MaxVelocity);
        w.Write(v.MaxAngularVelocity);
        w.Write(v.LogWarnings);
        w.Write(v.AutoCorrect);
    }

    private static void WriteShape2(BinaryWriter w, Shape2Dto? s)
    {
        if (s == null) { w.Write((byte)255); return; }
        switch (s.Type)
        {
            case "circle":
                w.Write((byte)0);
                w.Write(s.Radius);
                break;
            case "box":
                w.Write((byte)1);
                w.Write(s.HalfExtents!.X);
                w.Write(s.HalfExtents!.Y);
                break;
            case "polygon":
                w.Write((byte)2);
                w.Write(s.Vertices!.Count);
                foreach (var v in s.Vertices!) { w.Write(v.X); w.Write(v.Y); }
                break;
            default:
                throw new ArgumentException($"Unknown 2D shape: {s.Type}.");
        }
    }

    private static Shape2Dto? ReadShape2(BinaryReader r)
    {
        byte tag = r.ReadByte();
        return tag switch
        {
            255 => null,
            0 => new Shape2Dto { Type = "circle", Radius = r.ReadSingle() },
            1 => new Shape2Dto
            {
                Type = "box",
                HalfExtents = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
            },
            2 => ReadPolygon2(r),
            _ => throw new ArgumentException("Unknown 2D shape tag."),
        };
    }

    private static Shape2Dto ReadPolygon2(BinaryReader r)
    {
        int n = r.ReadInt32();
        if (n < 0 || n > 1024) throw new ArgumentException("Polygon vertex count out of range.");
        var d = new Shape2Dto { Type = "polygon", Vertices = new List<Vec2Dto>(n) };
        for (int i = 0; i < n; i++)
            d.Vertices!.Add(new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() });
        return d;
    }

    private static void WriteShape3(BinaryWriter w, Shape3Dto? s)
    {
        if (s == null) { w.Write((byte)255); return; }
        switch (s.Type)
        {
            case "sphere":
                w.Write((byte)0);
                w.Write(s.Radius);
                break;
            case "box":
                w.Write((byte)1);
                w.Write(s.HalfExtents!.X);
                w.Write(s.HalfExtents!.Y);
                w.Write(s.HalfExtents!.Z);
                break;
            case "capsule":
                w.Write((byte)2);
                w.Write(s.Radius);
                w.Write(s.HalfLength);
                break;
            case "convexhull":
                w.Write((byte)3);
                w.Write(s.Vertices!.Count);
                foreach (var v in s.Vertices!) { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); }
                break;
            default:
                throw new ArgumentException($"Unknown 3D shape: {s.Type}.");
        }
    }

    private static Shape3Dto? ReadShape3(BinaryReader r)
    {
        byte tag = r.ReadByte();
        return tag switch
        {
            255 => null,
            0 => new Shape3Dto { Type = "sphere", Radius = r.ReadSingle() },
            1 => new Shape3Dto
            {
                Type = "box",
                HalfExtents = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
            },
            2 => new Shape3Dto { Type = "capsule", Radius = r.ReadSingle(), HalfLength = r.ReadSingle() },
            3 => ReadHull3(r),
            _ => throw new ArgumentException("Unknown 3D shape tag."),
        };
    }

    private static Shape3Dto ReadHull3(BinaryReader r)
    {
        int n = r.ReadInt32();
        if (n < 0 || n > 4096) throw new ArgumentException("Hull vertex count out of range.");
        var d = new Shape3Dto { Type = "convexhull", Vertices = new List<Vec3Dto>(n) };
        for (int i = 0; i < n; i++)
            d.Vertices!.Add(new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() });
        return d;
    }

    private static void WriteJoint2(BinaryWriter w, Joint2Dto j)
    {
        switch (j.Type)
        {
            case "distance":
                w.Write((byte)0);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA.X); w.Write(j.AnchorA.Y);
                w.Write(j.AnchorB.X); w.Write(j.AnchorB.Y);
                w.Write(j.RestLength);
                w.Write(j.ImpulseScalar);
                break;
            case "hinge":
                w.Write((byte)1);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA.X); w.Write(j.AnchorA.Y);
                w.Write(j.AnchorB.X); w.Write(j.AnchorB.Y);
                w.Write(j.LimitMin); w.Write(j.LimitMax);
                w.Write(j.Impulse.X); w.Write(j.Impulse.Y);
                w.Write(j.AngularImpulse);
                break;
            case "prismatic":
                w.Write((byte)2);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA.X); w.Write(j.AnchorA.Y);
                w.Write(j.AnchorB.X); w.Write(j.AnchorB.Y);
                w.Write(j.AxisA!.X); w.Write(j.AxisA!.Y);
                w.Write(j.LimitMin); w.Write(j.LimitMax);
                w.Write(j.RefAngle.HasValue);
                if (j.RefAngle.HasValue) w.Write(j.RefAngle.Value);
                w.Write(j.PerpImpulse);
                w.Write(j.AngularImpulse);
                w.Write(j.LimitImpulse);
                break;
            case "fixed":
                w.Write((byte)3);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA.X); w.Write(j.AnchorA.Y);
                w.Write(j.AnchorB.X); w.Write(j.AnchorB.Y);
                w.Write(j.RefAngle.HasValue);
                if (j.RefAngle.HasValue) w.Write(j.RefAngle.Value);
                w.Write(j.Impulse.X); w.Write(j.Impulse.Y);
                w.Write(j.AngularImpulse);
                break;
            default:
                throw new ArgumentException($"Unknown joint type: {j.Type}.");
        }
    }

    private static Joint2Dto ReadJoint2(BinaryReader r)
    {
        byte tag = r.ReadByte();
        var j = new Joint2Dto
        {
            Id = r.ReadInt32(),
            BodyA = r.ReadInt32(),
            BodyB = r.ReadInt32(),
            AnchorA = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
            AnchorB = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
        };
        switch (tag)
        {
            case 0:
                j.Type = "distance";
                j.RestLength = r.ReadSingle();
                j.ImpulseScalar = r.ReadSingle();
                break;
            case 1:
                j.Type = "hinge";
                j.LimitMin = r.ReadSingle();
                j.LimitMax = r.ReadSingle();
                j.Impulse = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() };
                j.AngularImpulse = r.ReadSingle();
                break;
            case 2:
                j.Type = "prismatic";
                j.AxisA = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() };
                j.LimitMin = r.ReadSingle();
                j.LimitMax = r.ReadSingle();
                j.RefAngle = r.ReadBoolean() ? r.ReadSingle() : null;
                j.PerpImpulse = r.ReadSingle();
                j.AngularImpulse = r.ReadSingle();
                j.LimitImpulse = r.ReadSingle();
                break;
            case 3:
                j.Type = "fixed";
                j.RefAngle = r.ReadBoolean() ? r.ReadSingle() : null;
                j.Impulse = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() };
                j.AngularImpulse = r.ReadSingle();
                break;
            default:
                throw new ArgumentException("Unknown joint tag.");
        }
        return j;
    }

    private static void WriteWorld2Payload(BinaryWriter w, World2Dto dto)
    {
        w.Write(dto.Gravity.X); w.Write(dto.Gravity.Y);
        w.Write(dto.Dt);
        w.Write(dto.LinearSleepThreshold);
        w.Write(dto.AngularSleepThreshold);
        WriteSolver(w, dto.Solver);
        WriteValidation(w, dto.Validation);
        // Schema v4: gravity config block. v3 payloads omit it (strict:
        // a v4-tagged payload without it fails truncation, never misparses).
        bool gravityV4 = dto.Schema >= GravitySchemaVersion;
        if (gravityV4) WriteGravityConfig2(w, dto.GravityConfig);
        // Schema v5: ambient config block + clock.
        bool ambientV5 = dto.Schema >= SchemaVersion;
        if (ambientV5)
        {
            WriteAmbientConfig2(w, dto.Ambient);
            w.Write(dto.Time);
        }
        w.Write(dto.NextBodyId);
        w.Write(dto.NextJointId);
        w.Write(dto.CellSize);
        w.Write(dto.Bodies.Count);
        foreach (var b in dto.Bodies)
        {
            w.Write(b.Id);
            w.Write((byte)((b.IsStatic ? 1 : 0) | (b.IsSleeping ? 2 : 0)));
            w.Write(b.LinearDamping);
            w.Write(b.AngularDamping);
            w.Write(b.Material.Friction);
            w.Write(b.Material.Restitution);
            w.Write(b.Material.Density);
            WriteShape2(w, b.Shape);
            w.Write(b.Position.X); w.Write(b.Position.Y);
            foreach (var f in b.Rotation) w.Write(f);
            w.Write(b.Velocity.X); w.Write(b.Velocity.Y);
            w.Write(b.AngularVelocity);
            if (gravityV4) w.Write(b.GravityScale);
            if (ambientV5) w.Write(b.AmbientScale);
        }
        w.Write(dto.Joints.Count);
        foreach (var j in dto.Joints) WriteJoint2(w, j);
        w.Write(dto.Contacts.Count);
        foreach (var c in dto.Contacts)
        {
            w.Write(c.BodyA);
            w.Write(c.BodyB);
            w.Write(c.Entries.Count);
            foreach (var e in c.Entries)
            {
                w.Write(e.Point.X); w.Write(e.Point.Y);
                w.Write(e.Normal.X); w.Write(e.Normal.Y);
                w.Write(e.NormalImpulse);
                w.Write(e.TangentImpulse);
            }
        }
        WriteFluid2(w, dto.Fluid, dto.Schema);
    }

    private static void WriteFluidConfig2(BinaryWriter w, FluidConfig2Dto c, uint version)
    {
        w.Write(c.ParticleMass);
        w.Write(c.RestDensity);
        w.Write(c.SmoothingLength);
        w.Write(c.Stiffness);
        if (version >= SchemaVersion)
        {
            w.Write(c.Gamma);
            w.Write(c.ArtificialViscosity);
            w.Write(c.SurfaceTension);
            w.Write(c.Cfl);
            w.Write(c.MaxSubsteps);
        }
        w.Write(c.Viscosity);
        w.Write(c.Drag);
        w.Write(c.MaxVelocity);
        w.Write(c.ParticleRadius);
        w.Write(c.Enabled);
    }

    private static FluidConfig2Dto ReadFluidConfig2(BinaryReader r, uint version)
    {
        var c = new FluidConfig2Dto
        {
            ParticleMass = r.ReadSingle(),
            RestDensity = r.ReadSingle(),
            SmoothingLength = r.ReadSingle(),
            Stiffness = r.ReadSingle(),
        };
        if (version >= SchemaVersion)
        {
            c.Gamma = r.ReadSingle();
            c.ArtificialViscosity = r.ReadSingle();
            c.SurfaceTension = r.ReadSingle();
            c.Cfl = r.ReadSingle();
            c.MaxSubsteps = r.ReadInt32();
        }
        c.Viscosity = r.ReadSingle();
        c.Drag = r.ReadSingle();
        c.MaxVelocity = r.ReadSingle();
        c.ParticleRadius = r.ReadSingle();
        c.Enabled = r.ReadBoolean();
        return c;
    }

    private static void WriteFluid2(BinaryWriter w, FluidState2Dto f, uint version)
    {
        WriteFluidConfig2(w, f.Config, version);
        w.Write(f.Particles.Count);
        foreach (var p in f.Particles)
        {
            w.Write(p.Position.X); w.Write(p.Position.Y);
            w.Write(p.Velocity.X); w.Write(p.Velocity.Y);
        }
    }

    private static FluidState2Dto ReadFluid2(BinaryReader r, uint version)
    {
        var f = new FluidState2Dto { Config = ReadFluidConfig2(r, version) };
        int n = r.ReadInt32();
        if (n < 0 || n > FluidState2.MaxParticles)
            throw new ArgumentException("Fluid particle count out of range.");
        for (int i = 0; i < n; i++)
        {
            f.Particles.Add(new FluidParticle2Dto
            {
                Position = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
                Velocity = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
            });
        }
        return f;
    }

    private static void WriteFluidConfig3(BinaryWriter w, FluidConfig3Dto c, uint version)
    {
        w.Write(c.ParticleMass);
        w.Write(c.RestDensity);
        w.Write(c.SmoothingLength);
        w.Write(c.Stiffness);
        if (version >= SchemaVersion)
        {
            w.Write(c.Gamma);
            w.Write(c.ArtificialViscosity);
            w.Write(c.SurfaceTension);
            w.Write(c.Cfl);
            w.Write(c.MaxSubsteps);
        }
        w.Write(c.Viscosity);
        w.Write(c.Drag);
        w.Write(c.MaxVelocity);
        w.Write(c.ParticleRadius);
        w.Write(c.Enabled);
    }

    private static FluidConfig3Dto ReadFluidConfig3(BinaryReader r, uint version)
    {
        var c = new FluidConfig3Dto
        {
            ParticleMass = r.ReadSingle(),
            RestDensity = r.ReadSingle(),
            SmoothingLength = r.ReadSingle(),
            Stiffness = r.ReadSingle(),
        };
        if (version >= SchemaVersion)
        {
            c.Gamma = r.ReadSingle();
            c.ArtificialViscosity = r.ReadSingle();
            c.SurfaceTension = r.ReadSingle();
            c.Cfl = r.ReadSingle();
            c.MaxSubsteps = r.ReadInt32();
        }
        c.Viscosity = r.ReadSingle();
        c.Drag = r.ReadSingle();
        c.MaxVelocity = r.ReadSingle();
        c.ParticleRadius = r.ReadSingle();
        c.Enabled = r.ReadBoolean();
        return c;
    }

    private static void WriteFluid3(BinaryWriter w, FluidState3Dto f, uint version)
    {
        WriteFluidConfig3(w, f.Config, version);
        w.Write(f.Particles.Count);
        foreach (var p in f.Particles)
        {
            w.Write(p.Position.X); w.Write(p.Position.Y); w.Write(p.Position.Z);
            w.Write(p.Velocity.X); w.Write(p.Velocity.Y); w.Write(p.Velocity.Z);
        }
    }

    private static FluidState3Dto ReadFluid3(BinaryReader r, uint version)
    {
        var f = new FluidState3Dto { Config = ReadFluidConfig3(r, version) };
        int n = r.ReadInt32();
        if (n < 0 || n > FluidState3.MaxParticles)
            throw new ArgumentException("Fluid particle count out of range.");
        for (int i = 0; i < n; i++)
        {
            f.Particles.Add(new FluidParticle3Dto
            {
                Position = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                Velocity = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
            });
        }
        return f;
    }

    private static World2Dto ReadWorld2Payload(BinaryReader r, uint version)
    {
        var dto = new World2Dto
        {
            Schema = version,
            Gravity = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
            Dt = r.ReadSingle(),
            LinearSleepThreshold = r.ReadSingle(),
            AngularSleepThreshold = r.ReadSingle(),
            Solver = ReadSolver(r),
            Validation = ReadValidation(r),
            // v4+ payloads carry the gravity block; older ones resolve
            // to defaults (same values a v3 file implies).
            GravityConfig = version >= GravitySchemaVersion ? ReadGravityConfig2(r) : new GravityConfig2Dto(),
            // v5 payloads carry the ambient block + clock.
            Ambient = version >= SchemaVersion ? ReadAmbientConfig2(r) : new AmbientConfig2Dto(),
            Time = version >= SchemaVersion ? r.ReadSingle() : 0.0f,
            NextBodyId = r.ReadInt32(),
            NextJointId = r.ReadInt32(),
            CellSize = r.ReadSingle(),
        };
        bool gravityV4 = version >= GravitySchemaVersion;
        bool ambientV5 = version >= SchemaVersion;
        int nb = r.ReadInt32();
        if (nb < 0 || nb > 1_000_000) throw new ArgumentException("Body count out of range.");
        for (int i = 0; i < nb; i++)
        {
            int id = r.ReadInt32();
            byte flags = r.ReadByte();
            if ((flags & ~3) != 0) throw new ArgumentException("Bad body flags.");
            var b = new Body2Dto
            {
                Id = id,
                IsStatic = (flags & 1) != 0,
                IsSleeping = (flags & 2) != 0,
                LinearDamping = r.ReadSingle(),
                AngularDamping = r.ReadSingle(),
                Material = new MaterialDto
                {
                    Friction = r.ReadSingle(),
                    Restitution = r.ReadSingle(),
                    Density = r.ReadSingle(),
                },
                Shape = ReadShape2(r),
                Position = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
                Rotation = new[] { r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle() },
                Velocity = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
                AngularVelocity = r.ReadSingle(),
                GravityScale = gravityV4 ? r.ReadSingle() : 1.0f,
                AmbientScale = ambientV5 ? r.ReadSingle() : 1.0f,
            };
            dto.Bodies.Add(b);
        }
        int nj = r.ReadInt32();
        if (nj < 0 || nj > 1_000_000) throw new ArgumentException("Joint count out of range.");
        for (int i = 0; i < nj; i++) dto.Joints.Add(ReadJoint2(r));
        int nc = r.ReadInt32();
        if (nc < 0 || nc > 1_000_000) throw new ArgumentException("Contact pair count out of range.");
        for (int i = 0; i < nc; i++)
        {
            var c = new Contact2Dto { BodyA = r.ReadInt32(), BodyB = r.ReadInt32() };
            int m = r.ReadInt32();
            if (m < 0 || m > 16) throw new ArgumentException("Contact entry count out of range.");
            for (int k = 0; k < m; k++)
            {
                c.Entries.Add(new ContactEntry2Dto
                {
                    Point = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
                    Normal = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
                    NormalImpulse = r.ReadSingle(),
                    TangentImpulse = r.ReadSingle(),
                });
            }
            dto.Contacts.Add(c);
        }
        dto.Fluid = ReadFluid2(r, version);
        if (r.BaseStream.Position != r.BaseStream.Length)
            throw new ArgumentException("Trailing bytes in snapshot payload.");
        return dto;
    }

    private static SolverDto ReadSolver(BinaryReader r) => new()
    {
        Iterations = r.ReadInt32(),
        Slop = r.ReadSingle(),
        Baumgarte = r.ReadSingle(),
        EarlyExitTolerance = r.ReadSingle(),
    };

    private static void WriteGravityConfig2(BinaryWriter w, GravityConfig2Dto c)
    {
        w.Write(c.Mode);
        w.Write(c.Integrator);
        w.Write(c.BigG);
        w.Write(c.Softening);
        w.Write(c.Attractor.X); w.Write(c.Attractor.Y);
        w.Write(c.AttractorMu);
        w.Write(c.Theta);
    }

    private static GravityConfig2Dto ReadGravityConfig2(BinaryReader r) => new()
    {
        Mode = r.ReadUInt32(),
        Integrator = r.ReadUInt32(),
        BigG = r.ReadSingle(),
        Softening = r.ReadSingle(),
        Attractor = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
        AttractorMu = r.ReadSingle(),
        Theta = r.ReadSingle(),
    };

    private static void WriteGravityConfig3(BinaryWriter w, GravityConfig3Dto c)
    {
        w.Write(c.Mode);
        w.Write(c.Integrator);
        w.Write(c.BigG);
        w.Write(c.Softening);
        w.Write(c.Attractor.X); w.Write(c.Attractor.Y); w.Write(c.Attractor.Z);
        w.Write(c.AttractorMu);
        w.Write(c.Theta);
    }

    private static GravityConfig3Dto ReadGravityConfig3(BinaryReader r) => new()
    {
        Mode = r.ReadUInt32(),
        Integrator = r.ReadUInt32(),
        BigG = r.ReadSingle(),
        Softening = r.ReadSingle(),
        Attractor = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
        AttractorMu = r.ReadSingle(),
        Theta = r.ReadSingle(),
    };

    private static void WriteAmbientConfig2(BinaryWriter w, AmbientConfig2Dto c)
    {
        w.Write((byte)(c.Spring.Enabled ? 1 : 0));
        w.Write(c.Spring.Anchor.X); w.Write(c.Spring.Anchor.Y);
        w.Write(c.Spring.Stiffness);
        w.Write(c.Spring.Damping);
        w.Write(c.Spring.MaxDisplacement);
        w.Write((byte)(c.Drag.Enabled ? 1 : 0));
        w.Write(c.Drag.Linear);
        w.Write(c.Drag.Quadratic);
        w.Write(c.Drag.AirDensity);
        w.Write(c.Drag.Angular);
        w.Write(c.Drag.Wind.X); w.Write(c.Drag.Wind.Y);
        w.Write((byte)(c.Turbulence.Enabled ? 1 : 0));
        w.Write(c.Turbulence.Amplitude);
        w.Write(c.Turbulence.Frequency);
        w.Write(c.Turbulence.TimeScale);
        w.Write(c.Turbulence.Seed);
        w.Write((byte)c.Turbulence.Octaves);
        w.Write((byte)(c.Turbulence.DivergenceFree ? 1 : 0));
    }

    private static AmbientConfig2Dto ReadAmbientConfig2(BinaryReader r) => new()
    {
        Spring = new Spring2Dto
        {
            Enabled = r.ReadByte() != 0,
            Anchor = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
            Stiffness = r.ReadSingle(),
            Damping = r.ReadSingle(),
            MaxDisplacement = r.ReadSingle(),
        },
        Drag = new Drag2Dto
        {
            Enabled = r.ReadByte() != 0,
            Linear = r.ReadSingle(),
            Quadratic = r.ReadSingle(),
            AirDensity = r.ReadSingle(),
            Angular = r.ReadSingle(),
            Wind = new Vec2Dto { X = r.ReadSingle(), Y = r.ReadSingle() },
        },
        Turbulence = new Turbulence2Dto
        {
            Enabled = r.ReadByte() != 0,
            Amplitude = r.ReadSingle(),
            Frequency = r.ReadSingle(),
            TimeScale = r.ReadSingle(),
            Seed = r.ReadUInt32(),
            Octaves = r.ReadByte(),
            DivergenceFree = r.ReadByte() != 0,
        },
    };

    private static void WriteAmbientConfig3(BinaryWriter w, AmbientConfig3Dto c)
    {
        w.Write((byte)(c.Spring.Enabled ? 1 : 0));
        w.Write(c.Spring.Anchor.X); w.Write(c.Spring.Anchor.Y); w.Write(c.Spring.Anchor.Z);
        w.Write(c.Spring.Stiffness);
        w.Write(c.Spring.Damping);
        w.Write(c.Spring.MaxDisplacement);
        w.Write((byte)(c.Drag.Enabled ? 1 : 0));
        w.Write(c.Drag.Linear);
        w.Write(c.Drag.Quadratic);
        w.Write(c.Drag.AirDensity);
        w.Write(c.Drag.Angular);
        w.Write(c.Drag.Wind.X); w.Write(c.Drag.Wind.Y); w.Write(c.Drag.Wind.Z);
        w.Write((byte)(c.Turbulence.Enabled ? 1 : 0));
        w.Write(c.Turbulence.Amplitude);
        w.Write(c.Turbulence.Frequency);
        w.Write(c.Turbulence.TimeScale);
        w.Write(c.Turbulence.Seed);
        w.Write((byte)c.Turbulence.Octaves);
        w.Write((byte)(c.Turbulence.DivergenceFree ? 1 : 0));
    }

    private static AmbientConfig3Dto ReadAmbientConfig3(BinaryReader r) => new()
    {
        Spring = new Spring3Dto
        {
            Enabled = r.ReadByte() != 0,
            Anchor = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
            Stiffness = r.ReadSingle(),
            Damping = r.ReadSingle(),
            MaxDisplacement = r.ReadSingle(),
        },
        Drag = new Drag3Dto
        {
            Enabled = r.ReadByte() != 0,
            Linear = r.ReadSingle(),
            Quadratic = r.ReadSingle(),
            AirDensity = r.ReadSingle(),
            Angular = r.ReadSingle(),
            Wind = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
        },
        Turbulence = new Turbulence3Dto
        {
            Enabled = r.ReadByte() != 0,
            Amplitude = r.ReadSingle(),
            Frequency = r.ReadSingle(),
            TimeScale = r.ReadSingle(),
            Seed = r.ReadUInt32(),
            Octaves = r.ReadByte(),
            DivergenceFree = r.ReadByte() != 0,
        },
    };


    private static ValidationDto ReadValidation(BinaryReader r) => new()
    {
        CheckFinite = r.ReadBoolean(),
        MaxPenetration = r.ReadSingle(),
        MaxMassRatio = r.ReadSingle(),
        MaxCoordinate = r.ReadSingle(),
        MaxVelocity = r.ReadSingle(),
        MaxAngularVelocity = r.ReadSingle(),
        LogWarnings = r.ReadBoolean(),
        AutoCorrect = r.ReadBoolean(),
    };

    private static void WriteWorld3Payload(BinaryWriter w, World3Dto dto)
    {
        w.Write(dto.Gravity.X); w.Write(dto.Gravity.Y); w.Write(dto.Gravity.Z);
        w.Write(dto.Dt);
        w.Write(dto.LinearSleepThreshold);
        w.Write(dto.AngularSleepThreshold);
        WriteSolver(w, dto.Solver);
        WriteValidation(w, dto.Validation);
        // Schema v4: gravity config block (see WriteWorld2Payload).
        bool gravityV4 = dto.Schema >= GravitySchemaVersion;
        if (gravityV4) WriteGravityConfig3(w, dto.GravityConfig);
        bool ambientV5 = dto.Schema >= SchemaVersion;
        if (ambientV5)
        {
            WriteAmbientConfig3(w, dto.Ambient);
            w.Write(dto.Time);
        }
        w.Write(dto.NextBodyId);
        w.Write(dto.NextJointId);
        w.Write(dto.Bodies.Count);
        foreach (var b in dto.Bodies)
        {
            w.Write(b.Id);
            w.Write((byte)((b.IsStatic ? 1 : 0) | (b.IsSleeping ? 2 : 0)));
            w.Write(b.LinearDamping);
            w.Write(b.AngularDamping);
            w.Write(b.Material.Friction);
            w.Write(b.Material.Restitution);
            w.Write(b.Material.Density);
            WriteShape3(w, b.Shape);
            w.Write(b.Position.X); w.Write(b.Position.Y); w.Write(b.Position.Z);
            foreach (var f in b.Rotation) w.Write(f);
            w.Write(b.Velocity.X); w.Write(b.Velocity.Y); w.Write(b.Velocity.Z);
            w.Write(b.AngularVelocity.X);
            w.Write(b.AngularVelocity.Y);
            w.Write(b.AngularVelocity.Z);
            if (gravityV4) w.Write(b.GravityScale);
            if (ambientV5) w.Write(b.AmbientScale);
        }
        w.Write(dto.Joints.Count);
        foreach (var j in dto.Joints) WriteJoint3(w, j);
        w.Write(dto.Contacts.Count);
        foreach (var c in dto.Contacts)
        {
            w.Write(c.BodyA);
            w.Write(c.BodyB);
            w.Write(c.Entries.Count);
            foreach (var e in c.Entries)
            {
                w.Write(e.Point.X); w.Write(e.Point.Y); w.Write(e.Point.Z);
                w.Write(e.Normal.X); w.Write(e.Normal.Y); w.Write(e.Normal.Z);
                w.Write(e.NormalImpulse);
                w.Write(e.Tangent1Impulse);
                w.Write(e.Tangent2Impulse);
            }
        }
        WriteFluid3(w, dto.Fluid, dto.Schema);
    }

    private static World3Dto ReadWorld3Payload(BinaryReader r, uint version)
    {
        var dto = new World3Dto
        {
            Schema = version,
            Gravity = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
            Dt = r.ReadSingle(),
            LinearSleepThreshold = r.ReadSingle(),
            AngularSleepThreshold = r.ReadSingle(),
            Solver = ReadSolver(r),
            Validation = ReadValidation(r),
            GravityConfig = version >= GravitySchemaVersion ? ReadGravityConfig3(r) : new GravityConfig3Dto(),
            Ambient = version >= SchemaVersion ? ReadAmbientConfig3(r) : new AmbientConfig3Dto(),
            Time = version >= SchemaVersion ? r.ReadSingle() : 0.0f,
            NextBodyId = r.ReadInt32(),
            NextJointId = r.ReadInt32(),
        };
        int nb = r.ReadInt32();
        if (nb < 0 || nb > 1_000_000) throw new ArgumentException("Body count out of range.");
        for (int i = 0; i < nb; i++)
        {
            int id = r.ReadInt32();
            byte flags = r.ReadByte();
            if ((flags & ~3) != 0) throw new ArgumentException("Bad body flags.");
            dto.Bodies.Add(new Body3Dto
            {
                Id = id,
                IsStatic = (flags & 1) != 0,
                IsSleeping = (flags & 2) != 0,
                LinearDamping = r.ReadSingle(),
                AngularDamping = r.ReadSingle(),
                Material = new MaterialDto
                {
                    Friction = r.ReadSingle(),
                    Restitution = r.ReadSingle(),
                    Density = r.ReadSingle(),
                },
                Shape = ReadShape3(r),
                Position = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                Rotation = new[] { r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle() },
                Velocity = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                AngularVelocity = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                GravityScale = version >= GravitySchemaVersion ? r.ReadSingle() : 1.0f,
                AmbientScale = version >= SchemaVersion ? r.ReadSingle() : 1.0f,
            });
        }
        int nj = r.ReadInt32();
        if (nj < 0 || nj > 1_000_000) throw new ArgumentException("Joint count out of range.");
        for (int i = 0; i < nj; i++)
            dto.Joints.Add(ReadJoint3(r));
        int nc = r.ReadInt32();
        if (nc < 0 || nc > 1_000_000) throw new ArgumentException("Contact pair count out of range.");
        for (int i = 0; i < nc; i++)
        {
            var c = new Contact3Dto { BodyA = r.ReadInt32(), BodyB = r.ReadInt32() };
            int m = r.ReadInt32();
            if (m < 0 || m > 16) throw new ArgumentException("Contact entry count out of range.");
            for (int k = 0; k < m; k++)
            {
                c.Entries.Add(new ContactEntry3Dto
                {
                    Point = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                    Normal = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                    NormalImpulse = r.ReadSingle(),
                    Tangent1Impulse = r.ReadSingle(),
                    Tangent2Impulse = r.ReadSingle(),
                });
            }
            dto.Contacts.Add(c);
        }
        dto.Fluid = ReadFluid3(r, version);
        if (r.BaseStream.Position != r.BaseStream.Length)
            throw new ArgumentException("Trailing bytes in snapshot payload.");
        return dto;
    }

    private static void WriteJoint3(BinaryWriter w, Joint3Dto j)
    {
        switch (j.Type)
        {
            case "distance3":
                w.Write((byte)0);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA!.X); w.Write(j.AnchorA.Y); w.Write(j.AnchorA.Z);
                w.Write(j.AnchorB!.X); w.Write(j.AnchorB.Y); w.Write(j.AnchorB.Z);
                w.Write(j.RestLength ?? 0f);
                w.Write(j.Impulse ?? 0f);
                break;
            case "hinge3":
                w.Write((byte)1);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA!.X); w.Write(j.AnchorA.Y); w.Write(j.AnchorA.Z);
                w.Write(j.AnchorB!.X); w.Write(j.AnchorB.Y); w.Write(j.AnchorB.Z);
                w.Write(j.AxisA!.X); w.Write(j.AxisA.Y); w.Write(j.AxisA.Z);
                w.Write(j.AxisB!.X); w.Write(j.AxisB.Y); w.Write(j.AxisB.Z);
                w.Write(j.LimitMin ?? 0f); w.Write(j.LimitMax ?? 0f);
                w.Write(j.ImpulseVec?.X ?? 0f); w.Write(j.ImpulseVec?.Y ?? 0f); w.Write(j.ImpulseVec?.Z ?? 0f);
                w.Write(j.AlignU1 ?? 0f); w.Write(j.AlignU2 ?? 0f);
                w.Write(j.TwistImpulse ?? 0f);
                break;
            case "prismatic3":
                w.Write((byte)2);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA!.X); w.Write(j.AnchorA.Y); w.Write(j.AnchorA.Z);
                w.Write(j.AnchorB!.X); w.Write(j.AnchorB.Y); w.Write(j.AnchorB.Z);
                w.Write(j.AxisA!.X); w.Write(j.AxisA.Y); w.Write(j.AxisA.Z);
                w.Write(j.LimitMin ?? 0f); w.Write(j.LimitMax ?? 0f);
                if (j.RefOrientation != null)
                {
                    w.Write((byte)1);
                    foreach (var f in j.RefOrientation) w.Write(f);
                }
                else w.Write((byte)0);
                w.Write(j.PerpU1 ?? 0f); w.Write(j.PerpU2 ?? 0f);
                w.Write(j.AngularImpulse?.X ?? 0f); w.Write(j.AngularImpulse?.Y ?? 0f); w.Write(j.AngularImpulse?.Z ?? 0f);
                w.Write(j.LimitImpulse ?? 0f);
                break;
            case "fixed3":
                w.Write((byte)3);
                w.Write(j.Id); w.Write(j.BodyA); w.Write(j.BodyB);
                w.Write(j.AnchorA!.X); w.Write(j.AnchorA.Y); w.Write(j.AnchorA.Z);
                w.Write(j.AnchorB!.X); w.Write(j.AnchorB.Y); w.Write(j.AnchorB.Z);
                if (j.RefOrientation != null)
                {
                    w.Write((byte)1);
                    foreach (var f in j.RefOrientation) w.Write(f);
                }
                else w.Write((byte)0);
                w.Write(j.ImpulseVec?.X ?? 0f); w.Write(j.ImpulseVec?.Y ?? 0f); w.Write(j.ImpulseVec?.Z ?? 0f);
                w.Write(j.AngularImpulse?.X ?? 0f); w.Write(j.AngularImpulse?.Y ?? 0f); w.Write(j.AngularImpulse?.Z ?? 0f);
                break;
            default:
                throw new ArgumentException($"Unknown 3D joint type: {j.Type}");
        }
    }

    private static Joint3Dto ReadJoint3(BinaryReader r)
    {
        byte tag = r.ReadByte();
        int id = r.ReadInt32();
        int bodyA = r.ReadInt32();
        int bodyB = r.ReadInt32();
        var anchorA = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() };
        var anchorB = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() };
        switch (tag)
        {
            case 0:
                return new Joint3Dto
                {
                    Type = "distance3", Id = id, BodyA = bodyA, BodyB = bodyB,
                    AnchorA = anchorA, AnchorB = anchorB,
                    RestLength = r.ReadSingle(), Impulse = r.ReadSingle(),
                };
            case 1:
                var axisA = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() };
                var axisB = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() };
                return new Joint3Dto
                {
                    Type = "hinge3", Id = id, BodyA = bodyA, BodyB = bodyB,
                    AnchorA = anchorA, AnchorB = anchorB,
                    AxisA = axisA, AxisB = axisB,
                    LimitMin = r.ReadSingle(), LimitMax = r.ReadSingle(),
                    ImpulseVec = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                    AlignU1 = r.ReadSingle(), AlignU2 = r.ReadSingle(),
                    TwistImpulse = r.ReadSingle(),
                };
            case 2:
                var pAxis = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() };
                float lo = r.ReadSingle(); float hi = r.ReadSingle();
                float[]? ro = null;
                if (r.ReadByte() == 1)
                    ro = new[] { r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle() };
                return new Joint3Dto
                {
                    Type = "prismatic3", Id = id, BodyA = bodyA, BodyB = bodyB,
                    AnchorA = anchorA, AnchorB = anchorB,
                    AxisA = pAxis, LimitMin = lo, LimitMax = hi,
                    RefOrientation = ro,
                    PerpU1 = r.ReadSingle(), PerpU2 = r.ReadSingle(),
                    AngularImpulse = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                    LimitImpulse = r.ReadSingle(),
                };
            case 3:
                float[]? fro = null;
                if (r.ReadByte() == 1)
                    fro = new[] { r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle() };
                return new Joint3Dto
                {
                    Type = "fixed3", Id = id, BodyA = bodyA, BodyB = bodyB,
                    AnchorA = anchorA, AnchorB = anchorB,
                    RefOrientation = fro,
                    ImpulseVec = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                    AngularImpulse = new Vec3Dto { X = r.ReadSingle(), Y = r.ReadSingle(), Z = r.ReadSingle() },
                };
            default:
                throw new ArgumentException("Unknown 3D joint tag.");
        }
    }

    private static byte[] EncodeEnvelope(byte kind, uint version, Action<BinaryWriter> writePayload)
    {
        using var payload = new MemoryStream();
        using (var w = new BinaryWriter(payload, System.Text.Encoding.UTF8, leaveOpen: true))
            writePayload(w);
        byte[] body = payload.ToArray();
        using var ms = new MemoryStream();
        using var w2 = new BinaryWriter(ms);
        w2.Write(Magic);
        w2.Write(version);
        w2.Write(kind);
        w2.Write(body.Length);
        w2.Write(body);
        w2.Write(Crc32(body, 0, body.Length));
        return ms.ToArray();
    }
}
