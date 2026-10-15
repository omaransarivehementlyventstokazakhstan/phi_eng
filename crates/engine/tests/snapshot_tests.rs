//! Snapshot parity tests (E block): save/load round-trips through the
//! public snapshot API, mirrored by E01+ tests in C# (see
//! tests/PhiEng.Tests/SnapshotTests.cs). Parity enforced by
//! tools/check_test_parity.py.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::{
    world2_from_binary, world2_from_json, world2_to_binary, world2_to_json, LoadMode, World2,
};

const DT: f32 = 1.0 / 60.0;

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
    w.add_dynamic_body(
        Vec2::new(1.0, 3.0),
        Shape2::Box {
            half_extents: Vec2::new(0.4, 0.4),
        },
        1.5,
    )
    .expect("valid shape");
    w
}

fn hash_world(w: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &w.bodies {
        for v in [
            body.transform.position.x,
            body.transform.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angular_velocity,
            body.rotation_angle(),
        ] {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// E01: JSON exact round-trip preserves state hash.
#[test]
fn e01_json_exact_roundtrip() {
    let mut w = scene();
    for _ in 0..120 {
        w.step(DT);
    }
    let text = world2_to_json(&w, LoadMode::Exact).expect("serializes");
    let r = world2_from_json(&text, LoadMode::Exact).expect("deserializes");
    assert_eq!(hash_world(&w), hash_world(&r));
}

/// E02: binary exact round-trip preserves state hash.
#[test]
fn e02_binary_exact_roundtrip() {
    let mut w = scene();
    for _ in 0..120 {
        w.step(DT);
    }
    let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
    let r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
    assert_eq!(hash_world(&w), hash_world(&r));
}

/// E03: save at frame N, step M more both ways — identical futures.
#[test]
fn e03_continuation_matches() {
    let mut w = scene();
    for _ in 0..60 {
        w.step(DT);
    }
    let bytes = world2_to_binary(&w, LoadMode::Exact).expect("serializes");
    let mut r = world2_from_binary(&bytes, LoadMode::Exact).expect("deserializes");
    for _ in 0..60 {
        w.step(DT);
        r.step(DT);
    }
    assert_eq!(hash_world(&w), hash_world(&r));
}

/// E04: fresh mode drops transient state but stays finite and converges.
#[test]
fn e04_fresh_mode_converges() {
    let mut w = scene();
    for _ in 0..120 {
        w.step(DT);
    }
    let text = world2_to_json(&w, LoadMode::Fresh).expect("serializes");
    let mut r = world2_from_json(&text, LoadMode::Fresh).expect("deserializes");
    assert!(r.contact_cache.is_empty());
    for _ in 0..120 {
        r.step(DT);
    }
    for body in &r.bodies {
        assert!(body.transform.position.x.is_finite());
        assert!(body.velocity.x.is_finite());
    }
}

/// E05: hostile inputs are rejected, never panic.
#[test]
fn e05_rejects_hostile_input() {
    use phieng_dynamics::SnapshotError;
    assert!(matches!(
        world2_from_json("{nope", LoadMode::Exact),
        Err(SnapshotError::Parse(_))
    ));
    assert!(matches!(
        world2_from_binary(b"junk", LoadMode::Exact),
        Err(SnapshotError::Parse(_))
    ));
    let bytes = world2_to_binary(&scene(), LoadMode::Exact).expect("serializes");
    // Wrong version.
    let mut v = bytes.clone();
    v[4] = 99;
    let len = u32::from_le_bytes([v[9], v[10], v[11], v[12]]) as usize;
    let crc = crc32fast::hash(&v[13..13 + len]).to_le_bytes();
    v[13 + len..13 + len + 4].copy_from_slice(&crc);
    assert!(matches!(
        world2_from_binary(&v, LoadMode::Exact),
        Err(SnapshotError::Version(99))
    ));
    // Corrupt a payload byte (CRC catches it).
    let mut c = bytes.clone();
    c[20] ^= 0xFF;
    assert!(matches!(
        world2_from_binary(&c, LoadMode::Exact),
        Err(SnapshotError::Corrupt(_))
    ));
    // Truncate.
    assert!(matches!(
        world2_from_binary(&bytes[..bytes.len() / 2], LoadMode::Exact),
        Err(SnapshotError::Parse(_))
    ));
}

/// E06: JSON and binary decodes agree with each other.
#[test]
fn e06_json_binary_equivalent() {
    let mut w = scene();
    for _ in 0..60 {
        w.step(DT);
    }
    let a = world2_from_json(
        &world2_to_json(&w, LoadMode::Exact).expect("json"),
        LoadMode::Exact,
    )
    .expect("json load");
    let b = world2_from_binary(
        &world2_to_binary(&w, LoadMode::Exact).expect("bin"),
        LoadMode::Exact,
    )
    .expect("bin load");
    assert_eq!(hash_world(&a), hash_world(&b));
}

/// E07: file round-trip through a temp file (both formats).
#[test]
fn e07_file_roundtrip() {
    use phieng_dynamics::{world2_load_file, world2_save_file, FORMAT_BINARY, FORMAT_JSON};
    let mut w = scene();
    for _ in 0..60 {
        w.step(DT);
    }
    let dir = std::env::temp_dir();
    let pj = dir.join("phieng_e07.json");
    let pb = dir.join("phieng_e07.bin");
    world2_save_file(&w, &pj, FORMAT_JSON, LoadMode::Exact).expect("json save");
    world2_save_file(&w, &pb, FORMAT_BINARY, LoadMode::Exact).expect("bin save");
    let rj = world2_load_file(&pj, LoadMode::Exact).expect("json load");
    let rb = world2_load_file(&pb, LoadMode::Exact).expect("bin load");
    assert_eq!(hash_world(&w), hash_world(&rj));
    assert_eq!(hash_world(&w), hash_world(&rb));
    let _ = std::fs::remove_file(pj);
    let _ = std::fs::remove_file(pb);
}
