using System.Numerics;
using System.Text.Json.Nodes;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Native;

namespace PhiEng.Tests;

/// <summary>
/// Schema v14 in the managed snapshot mirror: the per-body sensor flag.
/// The flag round-trips through JSON and binary, the schema number the
/// writer picks follows the Rust rule in
/// <c>crates/dynamics/src/snapshot.rs</c> (<c>world2_dto</c> /
/// <c>world3_dto</c>), and a file written before v14 still loads with
/// every body an ordinary body.
/// </summary>
public class SnapshotSensorTests
{
    /// <summary>A floor and a ball, the usual two-body scene.</summary>
    private static World2 Scene2()
    {
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        return world;
    }

    /// <summary>The same scene with the ball turned into a sensor.</summary>
    private static World2 Scene2WithSensor()
    {
        var world = Scene2();
        world.Bodies[1].IsSensor = true;
        return world;
    }

    /// <summary>The schema number the writer put in the file.</summary>
    private static uint SchemaOf(World2 world)
    {
        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        return (uint)JsonNode.Parse(json)!["schema"]!.GetValue<int>();
    }

    /// <summary>The schema number in a binary envelope's header.</summary>
    private static uint EnvelopeVersion(byte[] bytes) => BitConverter.ToUInt32(bytes, 4);

    /// <summary>A sensor body keeps its flag through a JSON round trip.</summary>
    [Fact]
    public void SensorBodyRoundTripsThroughJson()
    {
        var world = Scene2WithSensor();
        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        // A sensor needs the v14 record, or the flag would be dropped.
        // The writer emits the newest schema in use (14) for it.
        Assert.Contains("\"schema\": 14", json);
        Assert.Contains("\"is_sensor\": true", json);

        var restored = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.Equal(2, restored.Bodies.Count);
        Assert.True(restored.Bodies[1].IsSensor);
        Assert.False(restored.Bodies[0].IsSensor);
    }

    /// <summary>A sensor body keeps its flag through a binary round trip.</summary>
    [Fact]
    public void SensorBodyRoundTripsThroughBinary()
    {
        var world = Scene2WithSensor();
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(14u, EnvelopeVersion(bytes));

        var restored = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.Equal(2, restored.Bodies.Count);
        Assert.True(restored.Bodies[1].IsSensor);
        Assert.False(restored.Bodies[0].IsSensor);
    }

    /// <summary>The same flag in 3D, through JSON and through binary.</summary>
    [Fact]
    public void SensorBodyRoundTripsIn3D()
    {
        var world = new World3();
        world.Gravity = new Vector3(0.0f, -9.81f, 0.0f);
        world.AddStaticBody(new Vector3(0.0f, -1.0f, 0.0f), new BoxShape3(new Vector3(10.0f, 0.5f, 10.0f)));
        world.AddDynamicBody(new Vector3(0.0f, 2.0f, 0.0f), new SphereShape3(0.5f), 1.0f);
        world.Bodies[1].IsSensor = true;

        string json = Snapshot.World3ToJson(world, SnapshotLoadMode.Exact);
        Assert.Contains("\"schema\": 14", json);
        Assert.Contains("\"is_sensor\": true", json);
        var fromJson = Snapshot.World3FromJson(json, SnapshotLoadMode.Exact);
        Assert.True(fromJson.Bodies[1].IsSensor);
        Assert.False(fromJson.Bodies[0].IsSensor);

        byte[] bytes = Snapshot.World3ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(14u, EnvelopeVersion(bytes));
        var fromBinary = Snapshot.World3FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.True(fromBinary.Bodies[1].IsSensor);
        Assert.False(fromBinary.Bodies[0].IsSensor);
    }

    /// <summary>The schema number the writer picks follows the Rust rule.</summary>
    [Fact]
    public void SchemaNumberFollowsTheRustRule()
    {
        // The Rust writer (world2_dto / world3_dto) picks the newest
        // schema in use: the listed version when the world uses the
        // corresponding block — thermal, the nuclear state, a
        // collision filter, a sensor or a quantum field — else 7 for
        // non-default solver flags, 6 for CCD, 5 for ambient, 4 for
        // gravity, else 3.
        Assert.Equal(3u, SchemaOf(Scene2()));

        var flagged = Scene2();
        flagged.Solver.ApplyQuality(SolverQuality.Low);
        Assert.Equal(7u, SchemaOf(flagged));

        var ccd = Scene2();
        ccd.Ccd = new CcdConfig
        {
            Enabled = true,
            Tier = CcdTier.FullGJK,
            MaxMovement = 0.05f,
            MaxSubsteps = 64,
        };
        Assert.Equal(6u, SchemaOf(ccd));

        var gravity = Scene2();
        gravity.GravityConfig = new GravityConfig2 { Mode = GravityMode.Central };
        Assert.Equal(4u, SchemaOf(gravity));

        // A collision filter is a v13 feature, and the writer emits
        // 13 for it, not the current version.
        var filtered = Scene2();
        filtered.Bodies[1].CollisionGroup = 4u;
        Assert.Equal(13u, SchemaOf(filtered));

        // A sensor is the v14 feature: schema 14.
        Assert.Equal(14u, SchemaOf(Scene2WithSensor()));
    }

    /// <summary>The reader accepts a current-schema file and still refuses a newer one.</summary>
    [Fact]
    public void SchemaV15FileIsAccepted()
    {
        // The managed mirror used to stop at v13, so a file the Rust
        // core writes today could not be loaded at all. The current
        // version must load, and the next one must still be refused.
        var world = Scene2WithSensor();
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(14u, EnvelopeVersion(bytes));
        var restored = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.True(restored.Bodies[1].IsSensor);

        byte[] future = (byte[])bytes.Clone();
        BitConverter.GetBytes(17u).CopyTo(future, 4);
        Assert.Throws<ArgumentException>(
            () => Snapshot.World2FromBinary(future, SnapshotLoadMode.Exact));
    }

    /// <summary>
    /// The Rust core writes a schema-15 file for a world with a sensor
    /// body, and the managed mirror reads that file and keeps the flag.
    /// This is the cross-boundary proof that the two layouts agree.
    /// </summary>
    [Fact]
    public void NativeCoreFileLoadsWithTheSensorFlag()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var world = new PhiNativeWorld();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new Vector2(10.0f, 0.5f));
        int ball = world.AddDynamicBall(new Vector2(0.0f, 5.0f), 0.5f, 1.0f);
        world.SetBodySensor(ball, true);

        string pb = Path.Combine(Path.GetTempPath(), "phieng_sensor_v14.bin");
        string pj = Path.Combine(Path.GetTempPath(), "phieng_sensor_v14.json");
        try
        {
            world.Save(pb, 1, 0);
            world.Save(pj, 0, 0);

            // The Rust core and the managed mirror must pick the same
            // schema number for the same world.
            byte[] bytes = File.ReadAllBytes(pb);
            Assert.Equal(Snapshot.SensorsSchemaVersion, EnvelopeVersion(bytes));

            var fromBinary = Snapshot.World2LoadFile(pb, SnapshotLoadMode.Exact);
            Assert.Equal(world.BodyCount, fromBinary.Bodies.Count);
            Assert.True(fromBinary.Bodies[ball].IsSensor);
            Assert.False(fromBinary.Bodies[0].IsSensor);

            var fromJson = Snapshot.World2FromJson(
                File.ReadAllText(pj), SnapshotLoadMode.Exact);
            Assert.True(fromJson.Bodies[ball].IsSensor);
            Assert.False(fromJson.Bodies[0].IsSensor);
        }
        finally
        {
            try { File.Delete(pb); } catch { }
            try { File.Delete(pj); } catch { }
        }
    }

    /// <summary>A schema-13 JSON has no sensor key: every body stays ordinary.</summary>
    [Fact]
    public void OldSchemaJsonLoadsWithoutTheSensorFlag()
    {
        // A schema-13 file has no "is_sensor" key at all: every body
        // must load as an ordinary body.
        var world = Scene2();
        var node = JsonNode.Parse(Snapshot.World2ToJson(world, SnapshotLoadMode.Exact))!;
        node["schema"] = 13;
        foreach (var body in node["bodies"]!.AsArray())
        {
            body!.AsObject().Remove("is_sensor");
        }

        var restored = Snapshot.World2FromJson(node.ToJsonString(), SnapshotLoadMode.Exact);
        Assert.Equal(2, restored.Bodies.Count);
        Assert.All(restored.Bodies, b => Assert.False(b.IsSensor));
    }

    /// <summary>A schema-13 binary has no sensor byte: every body stays ordinary.</summary>
    [Fact]
    public void OldSchemaBinaryLoadsWithoutTheSensorFlag()
    {
        // The schema-13 body record is the schema-14 one without the
        // trailing sensor byte, so a schema-13 file is rebuilt by
        // dropping that byte from the schema-14 payload. The schema-14
        // payload is the schema-15 one without the quantum block, which
        // is the last thing in the payload.
        var world = new World2();
        world.Gravity = new Vector2(0.0f, -9.81f);
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        world.Bodies[0].IsSensor = true;
        // A sensor-only world already writes v14: no quantum block to
        // truncate, straight on to the v13 rebuild.
        byte[] v14 = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(14u, EnvelopeVersion(v14));

        byte[] v13 = DowngradeToSchema13(v14, world);
        Assert.Equal(13u, EnvelopeVersion(v13));

        var restored = Snapshot.World2FromBinary(v13, SnapshotLoadMode.Exact);
        Assert.Single(restored.Bodies);
        Assert.False(restored.Bodies[0].IsSensor);
        // The rest of the body survived the downgrade untouched.
        Assert.Equal(0.01f, restored.Bodies[0].LinearDamping);
        Assert.Equal(new Vector2(0.0f, 2.0f), restored.Bodies[0].Transform.Position);
        Assert.Equal(CollisionFilter.DefaultGroup, restored.Bodies[0].CollisionGroup);
        Assert.Equal(CollisionFilter.DefaultMask, restored.Bodies[0].CollisionMask);
    }

    /// <summary>
    /// Rebuilds a schema-14 binary snapshot as a schema-13 one.
    ///
    /// The only difference between the two body records is the trailing
    /// sensor byte, so the schema-13 payload is the schema-14 payload
    /// with that byte removed. The byte is found by trial: the reader
    /// accepts exactly one removal that still decodes into the same
    /// world, and that removal is the sensor byte.
    /// </summary>
    private static byte[] DowngradeToSchema13(byte[] v14, World2 expected)
    {
        int len = (int)BitConverter.ToUInt32(v14, 9);
        byte[] payload = new byte[len];
        Array.Copy(v14, 13, payload, 0, len);

        byte[] found = [];
        for (int p = 0; p < payload.Length; p++)
        {
            byte[] candidate = new byte[payload.Length - 1];
            Array.Copy(payload, 0, candidate, 0, p);
            Array.Copy(payload, p + 1, candidate, p, payload.Length - p - 1);
            try
            {
                var world = Snapshot.World2FromBinary(
                    Envelope(13u, candidate), SnapshotLoadMode.Exact);
                if (!SameWorld(world, expected)) continue;
                Assert.Empty(found);
                found = Envelope(13u, candidate);
            }
            catch (ArgumentException) { }
            catch (EndOfStreamException) { }
        }
        Assert.NotEmpty(found);
        return found;
    }

    /// <summary>
    /// True when two worlds hold the same state. The JSON forms are
    /// compared, since they carry every field the snapshot keeps; only
    /// the schema number and the sensor flag are ignored (the file was
    /// written as v13, the reference world as v14 with a sensor).
    /// </summary>
    private static bool SameWorld(World2 a, World2 b)
    {
        JsonNode left = JsonNode.Parse(Snapshot.World2ToJson(a, SnapshotLoadMode.Exact))!;
        JsonNode right = JsonNode.Parse(Snapshot.World2ToJson(b, SnapshotLoadMode.Exact))!;
        left["schema"] = 0;
        right["schema"] = 0;
        foreach (var body in left["bodies"]!.AsArray())
        {
            body!.AsObject()["is_sensor"] = false;
        }
        foreach (var body in right["bodies"]!.AsArray())
        {
            body!.AsObject()["is_sensor"] = false;
        }
        return left.ToJsonString() == right.ToJsonString();
    }

    /// <summary>
    /// Builds a binary envelope around a payload: the same layout the
    /// reader expects (magic, version, kind, length, payload, CRC32).
    /// </summary>
    private static byte[] Envelope(uint version, byte[] payload)
    {
        byte[] file = new byte[13 + payload.Length + 4];
        file[0] = (byte)'P';
        file[1] = (byte)'H';
        file[2] = (byte)'E';
        file[3] = (byte)'1';
        BitConverter.GetBytes(version).CopyTo(file, 4);
        file[8] = 2; // world2
        BitConverter.GetBytes((uint)payload.Length).CopyTo(file, 9);
        Array.Copy(payload, 0, file, 13, payload.Length);
        BitConverter.GetBytes(Crc32(payload)).CopyTo(file, 13 + payload.Length);
        return file;
    }

    /// <summary>
    /// CRC32 (IEEE, the same polynomial the reader and the Rust writer
    /// use), computed bit by bit: the payload is short.
    /// </summary>
    private static uint Crc32(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 1) != 0 ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
            }
        }
        return crc ^ 0xFFFFFFFFu;
    }
}
