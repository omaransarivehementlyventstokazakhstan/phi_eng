//! Schema v15 in the snapshot system: the quantum field block (M35–M38,
//! mirrored Rust→C#).
//!
//! The wavefunction is part of the world, so it is part of the
//! snapshot: the configuration, the field's clock and the amplitudes
//! travel in their own version-gated block, appended at the end of the
//! payload. These tests pin:
//!
//! * a world that carries a wavefunction round-trips through the binary
//!   and the JSON with its configuration and its amplitudes intact;
//! * a world that does not keeps the schema — and the bytes — it had
//!   before the field existed;
//! * a file written before v15 still loads, with the field off;
//! * the bytes the Rust writer produces are the bytes the managed
//!   mirror produces: the two layouts agree, byte for byte.
//!
//! The plan for the whole ladder is in `QUANTUM_PLAN.md`.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::{
    world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, LoadMode, World2,
};
use phieng_quantum::evolution::EvolutionMethod;
use phieng_quantum::field::QuantumConfig;
use phieng_quantum::hamiltonian::Boundary1D;

/// The schema the quantum block was introduced in.
const SCHEMA_QUANTUM: u32 = 15;

/// The schema the quantum block was appended to.
const SCHEMA_SENSORS: u32 = 14;

/// A floor and a ball: the usual two-body scene, with no wavefunction.
fn scene() -> World2 {
    let mut w = World2::new();
    w.add_static_body(
        Vec2::new(0.0, -1.0),
        Shape2::Box {
            half_extents: Vec2::new(10.0, 0.5),
        },
    )
    .expect("valid shape");
    w.add_dynamic_body(Vec2::new(0.0, 2.0), Shape2::Circle { radius: 0.5 }, 1.0)
        .expect("valid shape");
    w
}

/// The same scene with the quantum field switched on.
fn quantum_scene() -> World2 {
    let mut w = scene();
    w.quantum.configure(QuantumConfig {
        enabled: true,
        grid_size: 32,
        domain_min: -4.0,
        domain_max: 4.0,
        mass: 1.0,
        hbar: 1.0,
        omega: 1.0,
        method: EvolutionMethod::CrankNicolson,
        boundary: Boundary1D::Dirichlet,
        seed: 7,
    });
    w
}

/// The world whose bytes both engines pin: no bodies, so the payload
/// carries the field alone, and a wavefunction of exact values, so the
/// two engines agree without leaning on the eigen-solver's last bit.
fn pinned_world() -> World2 {
    let mut w = World2::new();
    // The two engines disagree on two defaults (the sleep thresholds and
    // the validation switch), so the pinned world sets them explicitly:
    // the comparison is about the layout, not about those defaults.
    w.linear_sleep_threshold = 0.05;
    w.angular_sleep_threshold = 0.05;
    w.validation.check_finite = false;
    w.quantum.configure(QuantumConfig {
        enabled: true,
        grid_size: 32,
        domain_min: -4.0,
        domain_max: 4.0,
        mass: 1.0,
        hbar: 1.0,
        omega: 1.0,
        method: EvolutionMethod::CrankNicolson,
        boundary: Boundary1D::Dirichlet,
        seed: 7,
    });
    for (i, a) in w
        .quantum
        .state_mut()
        .amplitudes_mut()
        .iter_mut()
        .enumerate()
    {
        a.re = 0.5 - i as f32 * 0.015625;
        a.im = i as f32 * 0.03125;
    }
    w
}

/// The schema version in a binary envelope's header.
fn envelope_version(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

/// The schema number the writer put in the JSON.
fn json_schema(text: &str) -> u32 {
    let marker = "\"schema\": ";
    let at = text.find(marker).expect("the JSON carries a schema");
    let rest = &text[at + marker.len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().expect("the schema is a number")
}

/// Two fields carry the same configuration and the same wavefunction.
fn assert_quantum_eq(a: &World2, b: &World2) {
    let (ca, cb) = (a.quantum.config(), b.quantum.config());
    assert_eq!(ca.enabled, cb.enabled, "the field's switch drifted");
    assert_eq!(ca.method, cb.method, "the evolution method drifted");
    assert_eq!(ca.boundary, cb.boundary, "the boundary condition drifted");
    assert_eq!(ca.grid_size, cb.grid_size, "the grid size drifted");
    assert_eq!(ca.domain_min.to_bits(), cb.domain_min.to_bits());
    assert_eq!(ca.domain_max.to_bits(), cb.domain_max.to_bits());
    assert_eq!(ca.mass.to_bits(), cb.mass.to_bits());
    assert_eq!(ca.hbar.to_bits(), cb.hbar.to_bits());
    assert_eq!(ca.omega.to_bits(), cb.omega.to_bits());
    assert_eq!(ca.seed, cb.seed, "the measurement seed drifted");
    assert_eq!(
        a.quantum.time().to_bits(),
        b.quantum.time().to_bits(),
        "the field's clock drifted"
    );
    let (aa, bb) = (
        a.quantum.state().amplitudes(),
        b.quantum.state().amplitudes(),
    );
    assert_eq!(aa.len(), bb.len(), "the wavefunction's length drifted");
    assert!(!aa.is_empty(), "the wavefunction must not be empty");
    for (x, y) in aa.iter().zip(bb.iter()) {
        assert_eq!(x.re.to_bits(), y.re.to_bits(), "a real part drifted");
        assert_eq!(x.im.to_bits(), y.im.to_bits(), "an imaginary part drifted");
    }
}

/// M35: a world with a wavefunction round-trips through the binary and
/// the JSON, configuration and amplitudes intact.
#[test]
fn m35_a_world_with_a_wavefunction_round_trips() {
    let mut w = quantum_scene();
    // Three world steps evolve the wavefunction, so the round trip also
    // proves the clock restores: a loader that reset the clock would
    // fail the comparison below.
    for _ in 0..3 {
        w.step(1.0 / 60.0);
    }
    assert!(w.quantum.time() > 0.0, "the field must have advanced");
    assert!(w.quantum.is_enabled(), "the field must be on");

    let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
    assert_eq!(
        envelope_version(&bytes),
        SCHEMA_QUANTUM,
        "a world with a wavefunction is written at the quantum schema"
    );
    let r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
    assert_quantum_eq(&w, &r);
    assert!(r.quantum.is_enabled(), "the restored field must be on");

    let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
    assert_eq!(json_schema(&text), SCHEMA_QUANTUM);
    assert!(
        text.contains("\"quantum\""),
        "the JSON must carry the quantum block"
    );
    let rj = world2_from_json(&text, LoadMode::Exact).expect("deserializes");
    assert_quantum_eq(&w, &rj);
    assert!(rj.quantum.is_enabled(), "the restored field must be on");

    // The 3D world carries the same field.
    let mut w3 = phieng_dynamics::World3::new();
    w3.quantum.configure(QuantumConfig {
        enabled: true,
        grid_size: 32,
        domain_min: -4.0,
        domain_max: 4.0,
        mass: 1.0,
        hbar: 1.0,
        omega: 1.0,
        method: EvolutionMethod::CrankNicolson,
        boundary: Boundary1D::Dirichlet,
        seed: 7,
    });
    let bytes3 = phieng_dynamics::world3_to_binary(&w3, LoadMode::Exact).expect("serializes");
    assert_eq!(envelope_version(&bytes3), SCHEMA_QUANTUM);
    let r3 = phieng_dynamics::world3_from_binary(&bytes3, LoadMode::Exact).expect("deserializes");
    assert_eq!(r3.quantum.config().grid_size, 32);
    assert_eq!(
        r3.quantum.state().amplitudes().len(),
        w3.quantum.state().amplitudes().len()
    );
}

/// M36: a world without a wavefunction keeps the schema — and the bytes
/// — it had before the field existed, and still loads.
#[test]
fn m36_a_world_without_a_wavefunction_keeps_the_old_schema() {
    let w = scene();
    assert!(!w.quantum.is_enabled(), "the field must be off");

    let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
    assert_eq!(
        json_schema(&text),
        3,
        "a default world stays on the oldest schema"
    );
    assert!(
        !text.contains("quantum"),
        "a default world must not carry a quantum key"
    );

    let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
    assert_eq!(
        envelope_version(&bytes),
        3,
        "a default world stays on the oldest schema"
    );

    // It still loads, and the field stays off.
    let r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
    assert!(!r.quantum.is_enabled(), "the restored field must be off");
    assert_eq!(r.bodies.len(), w.bodies.len());
    let rj = world2_from_json(&text, LoadMode::Exact).expect("deserializes");
    assert!(!rj.quantum.is_enabled(), "the restored field must be off");
    assert_eq!(rj.bodies.len(), w.bodies.len());
}

/// M37: a file written before v15 still loads, with the field off.
#[test]
fn m37_an_old_file_loads_with_the_field_off() {
    let w = quantum_scene();
    let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
    assert_eq!(envelope_version(&bytes), SCHEMA_QUANTUM);

    // The quantum block is the last thing in the payload, so a v14 file
    // is the same payload without it. The reader accepts exactly one
    // truncation — the block's own length — and refuses every other
    // (too many bytes: trailing garbage; too few: a truncated block).
    let payload_len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
    let payload = &bytes[13..13 + payload_len];
    let mut downgraded: Option<Vec<u8>> = None;
    for cut in 1..=payload.len() {
        let candidate = &payload[..payload.len() - cut];
        let file = envelope(SCHEMA_SENSORS, candidate);
        if let Ok(world) = world2_from_binary(&file, LoadMode::Exact) {
            assert!(
                downgraded.is_none(),
                "only one truncation may decode: the quantum block's length"
            );
            downgraded = Some(file);
            assert!(
                !world.quantum.is_enabled(),
                "an old file must load with the field off"
            );
            assert_eq!(world.bodies.len(), w.bodies.len());
            assert_eq!(
                world.bodies[1].transform.position.y.to_bits(),
                w.bodies[1].transform.position.y.to_bits(),
                "the rest of the world must survive the downgrade"
            );
        }
    }
    let file = downgraded.expect("the v14 payload must decode");
    assert_eq!(envelope_version(&file), SCHEMA_SENSORS);
}

/// M38: the bytes the Rust writer produces are the bytes the managed
/// mirror produces: the two layouts agree, byte for byte.
#[test]
fn m38_the_writers_bytes_are_pinned() {
    let bytes = world2_to_binary(&pinned_world(), LoadMode::Exact).expect("serializes");
    assert_eq!(
        bytes,
        from_hex(PINNED_SNAPSHOT),
        "the Rust writer's bytes moved: the managed mirror writes the same ones"
    );
}

/// The bytes the Rust writer produces for [`pinned_world`], in hex. The
/// managed mirror writes the same bytes (see the M38 test in
/// tests/PhiEng.Tests/SnapshotQuantumTests.cs).
const PINNED_SNAPSHOT: &str = r#"504845310f00000002e902000000000000c3f51cc18988883ccdcc4c3dcdcc4c
3d080000000ad7a33bcdcc4c3ebd3786350101000000803f00007a4400401c46
00007a4400007a44000100000000000000000000803f00000000000000000000
0000000000000000003f00000000000000000000000000000000000000000000
00000000000000000000803f0000000000000000000000000000000000000080
3f0000803f000000000300000000000000cdcccc3d0800000000339392434f86
73330000204101010100409c45000000000000a0400000c84201000000000400
0000000000000000000000339392430000000000000000000000000000000000
0000803fd5e90c2e000000000000803f0000803fbd3786350000803f0000803f
a69bc43b0000003fcaf249710000803fcaf24971010000000000000000000000
0000000000000000000000000000000000000000000000000000000000000080
3f0000000000000000000000000000803d0000803f0000003f0000c842000080
3f00000000000000000000000001000000000000003fcdcccc3f000000000001
0000803f0000803f0000803f0000803f0000803d0000803d0000803d0000803d
9a99993e000080400000f0410000003e01000000000100000000000000002000
0000000080c0000080400000803f0000803f0000803f07000000000000000000
00001e0000000000003f0000f83e0000f03e0000e83e0000e03e0000d83e0000
d03e0000c83e0000c03e0000b83e0000b03e0000a83e0000a03e0000983e0000
903e0000883e0000803e0000703e0000603e0000503e0000403e0000303e0000
203e0000103e0000003e0000e03d0000c03d0000a03d0000803d0000403d0000
00000000003d0000803d0000c03d0000003e0000203e0000403e0000603e0000
803e0000903e0000a03e0000b03e0000c03e0000d03e0000e03e0000f03e0000
003f0000083f0000103f0000183f0000203f0000283f0000303f0000383f0000
403f0000483f0000503f0000583f0000603f0000683f4e845ad5"#;

/// Decodes a hex string into the bytes it stands for.
fn from_hex(hex: &str) -> Vec<u8> {
    let digits: Vec<u8> = hex
        .as_bytes()
        .iter()
        .copied()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    assert_eq!(digits.len() % 2, 0, "a hex string has an even length");
    digits
        .chunks(2)
        .map(|pair| {
            let s = std::str::from_utf8(pair).expect("hex is ASCII");
            u8::from_str_radix(s, 16).expect("hex is hexadecimal")
        })
        .collect()
}

/// Builds a binary envelope around a payload: the same layout the reader
/// expects (magic, version, kind, length, payload, CRC32).
fn envelope(version: u32, payload: &[u8]) -> Vec<u8> {
    let mut file = Vec::with_capacity(13 + payload.len() + 4);
    file.extend_from_slice(b"PHE1");
    file.extend_from_slice(&version.to_le_bytes());
    file.push(2); // world2
    file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    file.extend_from_slice(payload);
    file.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    file
}
