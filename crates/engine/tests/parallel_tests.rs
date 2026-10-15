//! Parallel island solver tests (P block): the parallel path must be
//! bit-identical to the serial path, and multi-island scenes must
//! actually partition. Mirrored by P31+ tests in C# (see
//! tests/PhiEng.Tests/ParallelTests.cs). Parity enforced by
//! tools/check_test_parity.py.

use glam::Vec2;
use phieng_dynamics::body::Shape2;
use phieng_dynamics::joint::make_distance_joint;
use phieng_dynamics::World2;

const DT: f32 = 1.0 / 60.0;

fn hash_world2(world: &World2) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for body in &world.bodies {
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
    // Joint impulses are solver state too: same trajectory must carry
    // the same accumulators.
    for joint in &world.joints {
        let (a, b) = joint.bodies();
        h ^= a as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Two separated 3-box stacks: two islands (plus resting contacts).
fn two_stacks(parallel: bool) -> World2 {
    let mut world = World2::new();
    world.solver.parallel = parallel;
    world
        .add_static_body(
            Vec2::new(0.0, -1.0),
            Shape2::Box {
                half_extents: Vec2::new(10.0, 0.5),
            },
        )
        .expect("valid shape");
    for stack_x in [-3.0f32, 3.0] {
        for i in 0..3 {
            world
                .add_dynamic_body(
                    Vec2::new(stack_x, 0.05 + i as f32 * 1.02),
                    Shape2::Box {
                        half_extents: Vec2::new(0.5, 0.5),
                    },
                    1.0,
                )
                .expect("valid shape");
        }
    }
    world
}

/// P31: parallel and serial runs of a multi-island scene agree bit-exactly.
#[test]
fn p31_parallel_matches_serial() {
    let mut serial = two_stacks(false);
    let mut parallel = two_stacks(true);
    for _ in 0..300 {
        serial.step(DT);
        parallel.step(DT);
    }
    assert_eq!(
        hash_world2(&serial),
        hash_world2(&parallel),
        "parallel islands must match serial bit-exactly"
    );
}

/// P32: a joint bridging two piles merges them into one island, and the
/// parallel run still matches serial bit-exactly.
#[test]
fn p32_joint_connected_matches_serial() {
    fn build(parallel: bool) -> World2 {
        let mut world = two_stacks(parallel);
        // Rod from the top of the left stack to the top of the right one.
        let top_left = 3;
        let top_right = 6;
        let joint = make_distance_joint(
            0,
            top_left,
            top_right,
            Vec2::ZERO,
            Vec2::ZERO,
            6.0,
            0.0,
            world.bodies.len(),
        )
        .expect("valid joint");
        world.add_joint(joint).expect("valid joint");
        world
    }
    let mut serial = build(false);
    let mut parallel = build(true);
    for _ in 0..300 {
        serial.step(DT);
        parallel.step(DT);
    }
    assert_eq!(
        hash_world2(&serial),
        hash_world2(&parallel),
        "joint-connected parallel run must match serial bit-exactly"
    );
    // The rod spans the gap: both tops must still be up (no tunneling).
    for id in [3, 6] {
        assert!(
            parallel.bodies[id].transform.position.y > 1.0,
            "stack top tunneled: y={}",
            parallel.bodies[id].transform.position.y
        );
    }
}

/// P33: single-island scenes take the exact serial path even with the
/// parallel flag on (no thread overhead, identical results).
#[test]
fn p33_single_island_fallback() {
    fn build(parallel: bool) -> World2 {
        let mut world = World2::new();
        world.solver.parallel = parallel;
        world
            .add_static_body(
                Vec2::new(0.0, -1.0),
                Shape2::Box {
                    half_extents: Vec2::new(10.0, 0.5),
                },
            )
            .expect("valid shape");
        world
            .add_dynamic_body(
                Vec2::new(0.0, 3.0),
                Shape2::Box {
                    half_extents: Vec2::new(0.5, 0.5),
                },
                1.0,
            )
            .expect("valid shape");
        world
    }
    let mut serial = build(false);
    let mut parallel = build(true);
    for _ in 0..300 {
        serial.step(DT);
        parallel.step(DT);
    }
    assert_eq!(
        hash_world2(&serial),
        hash_world2(&parallel),
        "single-island parallel run must match serial bit-exactly"
    );
}

/// P34: island builder partitions deterministically (unit-level pin).
#[test]
fn p34_island_partitioning() {
    use phieng_dynamics::island::{assign_constraints, build_islands, island_index_map};
    // Two disjoint pairs + one joint-bridged pair.
    let islands = build_islands(6, &[(0, 1), (2, 3)], &[(4, 5)]);
    assert_eq!(islands.len(), 3);
    assert_eq!(islands[0].bodies, vec![0, 1]);
    assert_eq!(islands[1].bodies, vec![2, 3]);
    assert_eq!(islands[2].bodies, vec![4, 5]);
    // Shuffled input, same set → same islands.
    let islands2 = build_islands(6, &[(2, 3), (0, 1)], &[(5, 4)]);
    assert_eq!(islands[0].bodies, islands2[0].bodies);
    assert_eq!(islands[1].bodies, islands2[1].bodies);
    assert_eq!(islands[2].bodies, islands2[2].bodies);
    // Assignment keeps input order per island (all pairs live fully
    // inside one island — split pairs are caller error, debug-asserted).
    let map = island_index_map(&islands, 6);
    let assigned = assign_constraints(&islands, &[(0, 1), (2, 3), (1, 0)], &map);
    assert_eq!(assigned[0], vec![0, 2]);
    assert_eq!(assigned[1], vec![1]);
    assert_eq!(assigned[2], Vec::<usize>::new());
}
