using System.Numerics;
using System.Text.Json.Nodes;
using Xunit;
using PhiEng.Dynamics;
using PhiEng.Math;
using PhiEng.Quantum;

namespace PhiEng.Tests;

/// <summary>
/// Schema v15 in the managed snapshot mirror: the quantum field block.
/// The field round-trips through JSON and binary, the schema number the
/// writer picks follows the Rust rule in
/// <c>crates/dynamics/src/snapshot.rs</c> (<c>world2_dto</c> /
/// <c>world3_dto</c>), a world without a wavefunction keeps the schema
/// it had before the field existed, a file written before v15 still
/// loads with the field off, and the bytes the Rust writer produces are
/// the bytes this mirror produces.
/// </summary>
public class SnapshotQuantumTests
{
    /// <summary>The schema the quantum block was introduced in.</summary>
    private const uint SchemaQuantum = 15;

    /// <summary>The schema the quantum block was appended to.</summary>
    private const uint SchemaSensors = 14;

    /// <summary>A floor and a ball: the usual two-body scene.</summary>
    private static World2 Scene()
    {
        var world = new World2();
        world.AddStaticBody(new Vector2(0.0f, -1.0f), new BoxShape2(new Vector2(10.0f, 0.5f)));
        world.AddDynamicBody(new Vector2(0.0f, 2.0f), new CircleShape(0.5f), 1.0f);
        return world;
    }

    /// <summary>The same scene with the quantum field switched on.</summary>
    private static World2 QuantumScene()
    {
        var world = Scene();
        world.Quantum.Configure(QuantumConfig());
        return world;
    }

    /// <summary>The configuration both engines pin: a small trap.</summary>
    private static QuantumConfig QuantumConfig() => new()
    {
        Enabled = true,
        GridSize = 32,
        DomainMin = -4.0f,
        DomainMax = 4.0f,
        Mass = 1.0f,
        Hbar = 1.0f,
        Omega = 1.0f,
        Method = EvolutionMethod.CrankNicolson,
        Boundary = Boundary1D.Dirichlet,
        Seed = 7,
    };

    /// <summary>
    /// The world whose bytes both engines pin: no bodies, so the payload
    /// carries the field alone, and a wavefunction of exact values, so
    /// the two engines agree without leaning on the eigen-solver's last
    /// bit.
    /// </summary>
    private static World2 PinnedWorld()
    {
        var world = new World2();
        // The two engines disagree on two defaults (the sleep thresholds
        // and the validation switch), so the pinned world sets them
        // explicitly: the comparison is about the layout, not about those
        // defaults.
        world.LinearSleepThreshold = 0.05f;
        world.AngularSleepThreshold = 0.05f;
        world.Validation.CheckFinite = false;
        world.Quantum.Configure(QuantumConfig());
        Complex32[] amplitudes = world.Quantum.StateMut.Amplitudes;
        for (int i = 0; i < amplitudes.Length; i++)
        {
            amplitudes[i].Re = 0.5f - i * 0.015625f;
            amplitudes[i].Im = i * 0.03125f;
        }
        return world;
    }

    /// <summary>The schema number the writer put in the JSON.</summary>
    private static uint JsonSchema(World2 world)
    {
        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        return (uint)JsonNode.Parse(json)!["schema"]!.GetValue<int>();
    }

    /// <summary>The schema number in a binary envelope's header.</summary>
    private static uint EnvelopeVersion(byte[] bytes) => BitConverter.ToUInt32(bytes, 4);

    /// <summary>Two fields carry the same configuration and wavefunction.</summary>
    private static void AssertQuantumEqual(World2 a, World2 b)
    {
        QuantumConfig ca = a.Quantum.Config;
        QuantumConfig cb = b.Quantum.Config;
        Assert.Equal(ca.Enabled, cb.Enabled);
        Assert.Equal(ca.Method, cb.Method);
        Assert.Equal(ca.Boundary, cb.Boundary);
        Assert.Equal(ca.GridSize, cb.GridSize);
        Assert.Equal(Bits(ca.DomainMin), Bits(cb.DomainMin));
        Assert.Equal(Bits(ca.DomainMax), Bits(cb.DomainMax));
        Assert.Equal(Bits(ca.Mass), Bits(cb.Mass));
        Assert.Equal(Bits(ca.Hbar), Bits(cb.Hbar));
        Assert.Equal(Bits(ca.Omega), Bits(cb.Omega));
        Assert.Equal(ca.Seed, cb.Seed);
        Assert.Equal(Bits(a.Quantum.Time), Bits(b.Quantum.Time));
        Complex32[] aa = a.Quantum.State.Amplitudes;
        Complex32[] bb = b.Quantum.State.Amplitudes;
        Assert.Equal(aa.Length, bb.Length);
        Assert.NotEmpty(aa);
        for (int i = 0; i < aa.Length; i++)
        {
            Assert.Equal(Bits(aa[i].Re), Bits(bb[i].Re));
            Assert.Equal(Bits(aa[i].Im), Bits(bb[i].Im));
        }
    }

    /// <summary>The bit pattern of a float.</summary>
    private static int Bits(float v) => BitConverter.SingleToInt32Bits(v);

    /// <summary>
    /// M35: a world with a wavefunction round-trips through the binary
    /// and the JSON, configuration and amplitudes intact.
    /// </summary>
    [Fact]
    public void M35_AWorldWithAWavefunctionRoundTrips()
    {
        var world = QuantumScene();
        // Three world steps evolve the wavefunction, so the round trip
        // also proves the clock restores: a loader that reset the clock
        // would fail the comparison below.
        for (int i = 0; i < 3; i++)
        {
            world.Step(1.0f / 60.0f);
        }
        Assert.True(world.Quantum.Time > 0.0f, "the field must have advanced");
        Assert.True(world.Quantum.IsEnabled);

        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(SchemaQuantum, EnvelopeVersion(bytes));
        var fromBinary = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.True(fromBinary.Quantum.IsEnabled);
        AssertQuantumEqual(world, fromBinary);

        string json = Snapshot.World2ToJson(world, SnapshotLoadMode.Exact);
        Assert.Equal(SchemaQuantum, JsonSchema(world));
        Assert.Contains("\"quantum\"", json);
        var fromJson = Snapshot.World2FromJson(json, SnapshotLoadMode.Exact);
        Assert.True(fromJson.Quantum.IsEnabled);
        AssertQuantumEqual(world, fromJson);

        // The 3D world carries the same field.
        var world3 = new World3();
        world3.Quantum.Configure(QuantumConfig());
        byte[] bytes3 = Snapshot.World3ToBinary(world3, SnapshotLoadMode.Exact);
        Assert.Equal(SchemaQuantum, EnvelopeVersion(bytes3));
        var fromBinary3 = Snapshot.World3FromBinary(bytes3, SnapshotLoadMode.Exact);
        Assert.Equal(32, fromBinary3.Quantum.Config.GridSize);
        Assert.Equal(
            world3.Quantum.State.Amplitudes.Length,
            fromBinary3.Quantum.State.Amplitudes.Length);
    }

    /// <summary>
    /// M36: a world without a wavefunction keeps the schema it had
    /// before the field existed, and still loads.
    /// </summary>
    [Fact]
    public void M36_AWorldWithoutAWavefunctionKeepsTheOldSchema()
    {
        var world = Scene();
        Assert.False(world.Quantum.IsEnabled);

        // The default world stays on the oldest schema: the quantum
        // block does not drag every world up to v15.
        Assert.Equal(3u, JsonSchema(world));
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(3u, EnvelopeVersion(bytes));

        var fromBinary = Snapshot.World2FromBinary(bytes, SnapshotLoadMode.Exact);
        Assert.False(fromBinary.Quantum.IsEnabled);
        Assert.Equal(world.Bodies.Count, fromBinary.Bodies.Count);
        var fromJson = Snapshot.World2FromJson(
            Snapshot.World2ToJson(world, SnapshotLoadMode.Exact), SnapshotLoadMode.Exact);
        Assert.False(fromJson.Quantum.IsEnabled);
        Assert.Equal(world.Bodies.Count, fromJson.Bodies.Count);
    }

    /// <summary>
    /// M37: a file written before v15 still loads, with the field off.
    /// </summary>
    [Fact]
    public void M37_AnOldFileLoadsWithTheFieldOff()
    {
        var world = QuantumScene();
        byte[] bytes = Snapshot.World2ToBinary(world, SnapshotLoadMode.Exact);
        Assert.Equal(SchemaQuantum, EnvelopeVersion(bytes));

        // The quantum block is the last thing in the payload, so a v14
        // file is the same payload without it. The reader accepts
        // exactly one truncation - the block's own length.
        byte[] v14 = DowngradeToSchema14(bytes);
        Assert.Equal(SchemaSensors, EnvelopeVersion(v14));

        var restored = Snapshot.World2FromBinary(v14, SnapshotLoadMode.Exact);
        Assert.False(restored.Quantum.IsEnabled);
        Assert.Equal(world.Bodies.Count, restored.Bodies.Count);
        Assert.Equal(
            Bits(world.Bodies[1].Transform.Position.Y),
            Bits(restored.Bodies[1].Transform.Position.Y));
    }

    /// <summary>
    /// M38: the bytes the Rust writer produces load in this mirror with
    /// the amplitudes intact - and this mirror writes the same bytes.
    /// </summary>
    [Fact]
    public void M38_TheRustWrittenFileLoadsWithTheAmplitudesIntact()
    {
        byte[] rustBytes = Convert.FromHexString(PinnedSnapshot);

        // The Rust-written file loads here, amplitudes and all.
        var restored = Snapshot.World2FromBinary(rustBytes, SnapshotLoadMode.Exact);
        Assert.True(restored.Quantum.IsEnabled);
        Assert.Equal(32, restored.Quantum.Config.GridSize);
        Assert.Equal(7ul, restored.Quantum.Config.Seed);
        Assert.Equal(30, restored.Quantum.State.Amplitudes.Length);
        for (int i = 0; i < restored.Quantum.State.Amplitudes.Length; i++)
        {
            Assert.Equal(Bits(0.5f - i * 0.015625f), Bits(restored.Quantum.State.Amplitudes[i].Re));
            Assert.Equal(Bits(i * 0.03125f), Bits(restored.Quantum.State.Amplitudes[i].Im));
        }

        // And this mirror writes the very same bytes: the two layouts
        // agree, byte for byte.
        byte[] managed = Snapshot.World2ToBinary(PinnedWorld(), SnapshotLoadMode.Exact);
        Assert.Equal(rustBytes, managed);
    }

    /// <summary>
    /// Rebuilds a schema-15 binary snapshot as a schema-14 one: the
    /// quantum block is the last thing in the payload, so the schema-14
    /// payload is the schema-15 payload without it.
    /// </summary>
    private static byte[] DowngradeToSchema14(byte[] v15)
    {
        int len = (int)BitConverter.ToUInt32(v15, 9);
        byte[] payload = new byte[len];
        Array.Copy(v15, 13, payload, 0, len);

        byte[]? found = null;
        for (int cut = 1; cut <= payload.Length; cut++)
        {
            byte[] candidate = new byte[payload.Length - cut];
            Array.Copy(payload, 0, candidate, 0, payload.Length - cut);
            try
            {
                Snapshot.World2FromBinary(Envelope(SchemaSensors, candidate), SnapshotLoadMode.Exact);
            }
            catch (ArgumentException) { continue; }
            catch (EndOfStreamException) { continue; }
            Assert.Null(found);
            found = Envelope(SchemaSensors, candidate);
        }
        Assert.NotNull(found);
        return found!;
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

    /// <summary>
    /// The bytes the Rust writer produces for the pinned world, in hex.
    /// The Rust writer pins the same bytes (see the M38 test in
    /// crates/engine/tests/quantum_snapshot.rs).
    /// </summary>
    private const string PinnedSnapshot =
        "504845310f00000002e902000000000000c3f51cc18988883ccdcc4c3dcdcc4c" +
        "3d080000000ad7a33bcdcc4c3ebd3786350101000000803f00007a4400401c46" +
        "00007a4400007a44000100000000000000000000803f00000000000000000000" +
        "0000000000000000003f00000000000000000000000000000000000000000000" +
        "00000000000000000000803f0000000000000000000000000000000000000080" +
        "3f0000803f000000000300000000000000cdcccc3d0800000000339392434f86" +
        "73330000204101010100409c45000000000000a0400000c84201000000000400" +
        "0000000000000000000000339392430000000000000000000000000000000000" +
        "0000803fd5e90c2e000000000000803f0000803fbd3786350000803f0000803f" +
        "a69bc43b0000003fcaf249710000803fcaf24971010000000000000000000000" +
        "0000000000000000000000000000000000000000000000000000000000000080" +
        "3f0000000000000000000000000000803d0000803f0000003f0000c842000080" +
        "3f00000000000000000000000001000000000000003fcdcccc3f000000000001" +
        "0000803f0000803f0000803f0000803f0000803d0000803d0000803d0000803d" +
        "9a99993e000080400000f0410000003e01000000000100000000000000002000" +
        "0000000080c0000080400000803f0000803f0000803f07000000000000000000" +
        "00001e0000000000003f0000f83e0000f03e0000e83e0000e03e0000d83e0000" +
        "d03e0000c83e0000c03e0000b83e0000b03e0000a83e0000a03e0000983e0000" +
        "903e0000883e0000803e0000703e0000603e0000503e0000403e0000303e0000" +
        "203e0000103e0000003e0000e03d0000c03d0000a03d0000803d0000403d0000" +
        "00000000003d0000803d0000c03d0000003e0000203e0000403e0000603e0000" +
        "803e0000903e0000a03e0000b03e0000c03e0000d03e0000e03e0000f03e0000" +
        "003f0000083f0000103f0000183f0000203f0000283f0000303f0000383f0000" +
        "403f0000483f0000503f0000583f0000603f0000683f4e845ad5";
}
