//! Deterministic Barnes-Hut approximation for N-body gravity.
//!
//! Collapses distant groups to their center of mass when a cell's
//! size-to-distance ratio falls below `theta`
//! ([`crate::gravity::GravityConfig2::theta`]), turning the exact
//! O(n^2) pairwise loop into O(n log n). Wires into the pre-solve
//! gravity kick via [`apply_bh_forces_2`] / [`apply_bh_forces_3`]
//! (called from the
//! `BarnesHut` arm of `gravity_kick_2/3`, after the shared uniform
//! pass — this module applies the tree forces only).
//!
//! Determinism contract (bit-identical runs by construction):
//! - Insertion in body-index order; subdivision at cell midpoints with
//!   a fixed low-side rule, so ties and NaNs route deterministically.
//! - Children visited in fixed index order; mass/COM accumulation in
//!   child order; forces accumulate in walk order. No hash maps, no
//!   RNG, no wall clock, no thread-dependent scheduling.
//! - A cell containing the query body is NEVER approximated (its
//!   contribution would include self-force): the walk always descends
//!   into the routed child. Single-body cells approximate exactly
//!   (COM == body), so two-body scenes are bit-identical to exact
//!   N-body regardless of `theta`.
//!
//! Termination guards (no hangs, no panics):
//! - `MAX_TREE_DEPTH` caps subdivision; a full leaf at max depth
//!   accumulates bodies instead of splitting (direct pairwise inside
//!   the leaf, self skipped).
//! - Zero-extent cells never subdivide. NaN positions route
//!   deterministically (all comparisons false -> high side) and
//!   produce NaN forces like the exact path — caught downstream by
//!   body validation, never a hang.
//!
//! Approximation status: same-mode runs are bit-identical, but exact
//! and BH trajectories never match each other — tests must assert
//! per-mode self-consistency, never cross-mode equality. `theta <= 0`
//! degenerates to exact descent (correct, slow).
//!
//! Cost model: the tree (arena + per-leaf index lists) is rebuilt per
//! kick — BH trades per-step allocation for asymptotic force cost.
//! The exact path stays allocation-free; BH is opt-in via
//! `GravityMode::BarnesHut` and stays off by default.
//!
//! Measured crossover (`bench_gravity`, release, portable profile):
//! exact O(n^2) quadruples per doubling as theory predicts, while BH
//! scales near-linearly. At `theta = 0.5` BH breaks even around
//! ~700 bodies (n=500: 0.6 vs 0.6 ms/step; n=1000: 2.2 vs 1.4;
//! n=2000: 8.7 vs 3.5–3.8) and pulls away past that; `theta = 1.0`
//! is ~2.5x faster still at some accuracy cost. Below ~500 bodies
//! the exact path wins outright (tree overhead dominates).

use super::body::{RigidBody2, RigidBody3};
use super::gravity::{pair_accelerations_2, pair_accelerations_3};
use glam::{Vec2, Vec3};
use phieng_math::EPS_SQ;

/// Maximum subdivision depth: bounds coincident-point splitting.
/// A full leaf at max depth accumulates instead of subdividing, so
/// insertion always terminates.
const MAX_TREE_DEPTH: u32 = 32;

// ---------------------------------------------------------------------------
// 2D quadtree.
// ---------------------------------------------------------------------------

struct QuadNode {
    min: Vec2,
    max: Vec2,
    /// Body indices when this is a leaf; empty once subdivided.
    bodies: Vec<usize>,
    /// Child arena indices in fixed order:
    /// 0 = (-x,-y), 1 = (+x,-y), 2 = (-x,+y), 3 = (+x,+y).
    children: [Option<usize>; 4],
    mass: f32,
    com: Vec2,
}

impl QuadNode {
    fn is_leaf(&self) -> bool {
        self.children == [None, None, None, None]
    }

    /// Cell size for the opening criterion: longest side.
    fn extent(&self) -> f32 {
        (self.max.x - self.min.x).max(self.max.y - self.min.y)
    }
}

struct QuadTree {
    nodes: Vec<QuadNode>,
}

/// Route a point to a child quadrant with the same rule insertion
/// uses (low side strict, ties and NaNs go high — deterministic).
fn route_quad(node: &QuadNode, p: Vec2) -> usize {
    let mid = (node.min + node.max) * 0.5;
    let ix = if p.x < mid.x { 0 } else { 1 };
    let iy = if p.y < mid.y { 0 } else { 1 };
    iy * 2 + ix
}

fn subdivide_quad(nodes: &mut Vec<QuadNode>, node_idx: usize) {
    let (min, max) = {
        let n = &nodes[node_idx];
        (n.min, n.max)
    };
    let mid = (min + max) * 0.5;
    let quads = [
        (min, mid),
        (Vec2::new(mid.x, min.y), Vec2::new(max.x, mid.y)),
        (Vec2::new(min.x, mid.y), Vec2::new(mid.x, max.y)),
        (mid, max),
    ];
    for (lo, hi) in quads {
        nodes.push(QuadNode {
            min: lo,
            max: hi,
            bodies: Vec::new(),
            children: [None, None, None, None],
            mass: 0.0,
            com: Vec2::ZERO,
        });
    }
    // Pushed in order 0..4, so arena indices are contiguous.
    let base = nodes.len() - 4;
    nodes[node_idx].children = [Some(base), Some(base + 1), Some(base + 2), Some(base + 3)];
}

fn insert_quad(
    nodes: &mut Vec<QuadNode>,
    node_idx: usize,
    body_idx: usize,
    positions: &[Vec2],
    depth: u32,
) {
    if !nodes[node_idx].is_leaf() {
        let child = route_quad(&nodes[node_idx], positions[body_idx]);
        let child_idx = nodes[node_idx].children[child].expect("internal child present");
        insert_quad(nodes, child_idx, body_idx, positions, depth + 1);
        return;
    }
    if nodes[node_idx].bodies.is_empty()
        || depth >= MAX_TREE_DEPTH
        || nodes[node_idx].extent() == 0.0
    {
        nodes[node_idx].bodies.push(body_idx);
        return;
    }
    subdivide_quad(nodes, node_idx);
    // Reinsert the displaced residents in order, then the newcomer.
    let existing = std::mem::take(&mut nodes[node_idx].bodies);
    for b in existing {
        let child = route_quad(&nodes[node_idx], positions[b]);
        let child_idx = nodes[node_idx].children[child].expect("internal child present");
        insert_quad(nodes, child_idx, b, positions, depth + 1);
    }
    let child = route_quad(&nodes[node_idx], positions[body_idx]);
    let child_idx = nodes[node_idx].children[child].expect("internal child present");
    insert_quad(nodes, child_idx, body_idx, positions, depth + 1);
}

/// Post-order mass/COM accumulation in fixed child order.
/// Zero-mass cells fall back to the geometric center (unreachable
/// through validated construction, which floors mass — pure defense).
fn accumulate_quad(nodes: &mut Vec<QuadNode>, node_idx: usize, positions: &[Vec2], masses: &[f32]) {
    if nodes[node_idx].is_leaf() {
        let mut mass = 0.0f32;
        let mut moment = Vec2::ZERO;
        let count = nodes[node_idx].bodies.len();
        for k in 0..count {
            let b = nodes[node_idx].bodies[k];
            mass += masses[b];
            moment += positions[b] * masses[b];
        }
        let node = &mut nodes[node_idx];
        node.mass = mass;
        node.com = if mass > 0.0 {
            moment / mass
        } else {
            (node.min + node.max) * 0.5
        };
        return;
    }
    let mut mass = 0.0f32;
    let mut moment = Vec2::ZERO;
    for c in 0..4 {
        if let Some(child) = nodes[node_idx].children[c] {
            accumulate_quad(nodes, child, positions, masses);
            mass += nodes[child].mass;
            moment += nodes[child].com * nodes[child].mass;
        }
    }
    let node = &mut nodes[node_idx];
    node.mass = mass;
    node.com = if mass > 0.0 {
        moment / mass
    } else {
        (node.min + node.max) * 0.5
    };
}

fn build_quad(positions: &[Vec2], masses: &[f32]) -> QuadTree {
    let mut tree = QuadTree { nodes: Vec::new() };
    if positions.is_empty() {
        return tree;
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    tree.nodes.push(QuadNode {
        min,
        max,
        bodies: Vec::new(),
        children: [None, None, None, None],
        mass: 0.0,
        com: Vec2::ZERO,
    });
    for i in 0..positions.len() {
        insert_quad(&mut tree.nodes, 0, i, positions, 0);
    }
    accumulate_quad(&mut tree.nodes, 0, positions, masses);
    tree
}

/// Force walk for one query body: fixed child order, index-ordered
/// leaf lists, self skipped. Cells containing the query (found via
/// the same routing insertion used) are always descended, never
/// approximated — zero self-force, exactly.
#[allow(clippy::too_many_arguments)]
fn walk_quad(
    nodes: &[QuadNode],
    node_idx: usize,
    qpos: Vec2,
    qidx: usize,
    positions: &[Vec2],
    masses: &[f32],
    big_g: f32,
    softening: f32,
    theta: f32,
    acc: &mut Vec2,
) {
    let node = &nodes[node_idx];
    if node.is_leaf() {
        for &k in &node.bodies {
            if k == qidx {
                continue;
            }
            let (a, _) = pair_accelerations_2(qpos, 1.0, positions[k], masses[k], big_g, softening);
            *acc += a;
        }
        return;
    }
    let routed = route_quad(node, qpos);
    for c in 0..4 {
        let Some(child) = node.children[c] else {
            continue;
        };
        if c == routed {
            walk_quad(
                nodes, child, qpos, qidx, positions, masses, big_g, softening, theta, acc,
            );
            continue;
        }
        let cn = &nodes[child];
        let d = cn.com - qpos;
        let dist2 = d.length_squared() + softening * softening;
        if dist2 < EPS_SQ {
            continue;
        }
        if cn.extent() / dist2.sqrt() < theta {
            let (a, _) = pair_accelerations_2(qpos, 1.0, cn.com, cn.mass, big_g, softening);
            *acc += a;
        } else {
            walk_quad(
                nodes, child, qpos, qidx, positions, masses, big_g, softening, theta, acc,
            );
        }
    }
}

/// Barnes-Hut force contribution over 2D bodies (tree part only —
/// the caller runs the uniform background pass first, mirroring
/// `gravity_kick_2`). Applies per-body `gravity_scale` and optional
/// symmetric drag exactly like the exact path. Statics are skipped on
/// the felt side but exert via stored mass.
pub fn apply_bh_forces_2(
    bodies: &mut [RigidBody2],
    big_g: f32,
    softening: f32,
    theta: f32,
    h: f32,
    drag: bool,
) {
    let n = bodies.len();
    if n == 0 {
        return;
    }
    let positions: Vec<Vec2> = bodies.iter().map(|b| b.transform.position).collect();
    let masses: Vec<f32> = bodies.iter().map(|b| b.mass).collect();
    let tree = build_quad(&positions, &masses);
    for i in 0..n {
        if bodies[i].is_static {
            continue;
        }
        let mut acc = Vec2::ZERO;
        walk_quad(
            &tree.nodes,
            0,
            positions[i],
            i,
            &positions,
            &masses,
            big_g,
            softening,
            theta,
            &mut acc,
        );
        let b = &mut bodies[i];
        let mut dv = acc * (b.gravity_scale * h);
        if drag {
            dv -= b.velocity * (b.linear_damping * h);
        }
        b.velocity += dv;
    }
}

// ---------------------------------------------------------------------------
// 3D octree (same contract as the quadtree above).
// ---------------------------------------------------------------------------

struct OctNode {
    min: Vec3,
    max: Vec3,
    bodies: Vec<usize>,
    /// Child arena indices in fixed order: bit 0 = +x, bit 1 = +y,
    /// bit 2 = +z (index = iz*4 + iy*2 + ix).
    children: [Option<usize>; 8],
    mass: f32,
    com: Vec3,
}

impl OctNode {
    fn is_leaf(&self) -> bool {
        self.children == [None, None, None, None, None, None, None, None]
    }

    fn extent(&self) -> f32 {
        (self.max.x - self.min.x)
            .max(self.max.y - self.min.y)
            .max(self.max.z - self.min.z)
    }
}

struct OctTree {
    nodes: Vec<OctNode>,
}

fn route_oct(node: &OctNode, p: Vec3) -> usize {
    let mid = (node.min + node.max) * 0.5;
    let ix = if p.x < mid.x { 0 } else { 1 };
    let iy = if p.y < mid.y { 0 } else { 1 };
    let iz = if p.z < mid.z { 0 } else { 1 };
    iz * 4 + iy * 2 + ix
}

fn subdivide_oct(nodes: &mut Vec<OctNode>, node_idx: usize) {
    let (min, max) = {
        let n = &nodes[node_idx];
        (n.min, n.max)
    };
    let mid = (min + max) * 0.5;
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                let lo = Vec3::new(
                    if ix == 0 { min.x } else { mid.x },
                    if iy == 0 { min.y } else { mid.y },
                    if iz == 0 { min.z } else { mid.z },
                );
                let hi = Vec3::new(
                    if ix == 0 { mid.x } else { max.x },
                    if iy == 0 { mid.y } else { max.y },
                    if iz == 0 { mid.z } else { max.z },
                );
                nodes.push(OctNode {
                    min: lo,
                    max: hi,
                    bodies: Vec::new(),
                    children: [None, None, None, None, None, None, None, None],
                    mass: 0.0,
                    com: Vec3::ZERO,
                });
            }
        }
    }
    // Pushed iz-major in 0..8 order, so arena indices are contiguous.
    let base = nodes.len() - 8;
    nodes[node_idx].children = [
        Some(base),
        Some(base + 1),
        Some(base + 2),
        Some(base + 3),
        Some(base + 4),
        Some(base + 5),
        Some(base + 6),
        Some(base + 7),
    ];
}

fn insert_oct(
    nodes: &mut Vec<OctNode>,
    node_idx: usize,
    body_idx: usize,
    positions: &[Vec3],
    depth: u32,
) {
    if !nodes[node_idx].is_leaf() {
        let child = route_oct(&nodes[node_idx], positions[body_idx]);
        let child_idx = nodes[node_idx].children[child].expect("internal child present");
        insert_oct(nodes, child_idx, body_idx, positions, depth + 1);
        return;
    }
    if nodes[node_idx].bodies.is_empty()
        || depth >= MAX_TREE_DEPTH
        || nodes[node_idx].extent() == 0.0
    {
        nodes[node_idx].bodies.push(body_idx);
        return;
    }
    subdivide_oct(nodes, node_idx);
    let existing = std::mem::take(&mut nodes[node_idx].bodies);
    for b in existing {
        let child = route_oct(&nodes[node_idx], positions[b]);
        let child_idx = nodes[node_idx].children[child].expect("internal child present");
        insert_oct(nodes, child_idx, b, positions, depth + 1);
    }
    let child = route_oct(&nodes[node_idx], positions[body_idx]);
    let child_idx = nodes[node_idx].children[child].expect("internal child present");
    insert_oct(nodes, child_idx, body_idx, positions, depth + 1);
}

fn accumulate_oct(nodes: &mut Vec<OctNode>, node_idx: usize, positions: &[Vec3], masses: &[f32]) {
    if nodes[node_idx].is_leaf() {
        let mut mass = 0.0f32;
        let mut moment = Vec3::ZERO;
        let count = nodes[node_idx].bodies.len();
        for k in 0..count {
            let b = nodes[node_idx].bodies[k];
            mass += masses[b];
            moment += positions[b] * masses[b];
        }
        let node = &mut nodes[node_idx];
        node.mass = mass;
        node.com = if mass > 0.0 {
            moment / mass
        } else {
            (node.min + node.max) * 0.5
        };
        return;
    }
    let mut mass = 0.0f32;
    let mut moment = Vec3::ZERO;
    for c in 0..8 {
        if let Some(child) = nodes[node_idx].children[c] {
            accumulate_oct(nodes, child, positions, masses);
            mass += nodes[child].mass;
            moment += nodes[child].com * nodes[child].mass;
        }
    }
    let node = &mut nodes[node_idx];
    node.mass = mass;
    node.com = if mass > 0.0 {
        moment / mass
    } else {
        (node.min + node.max) * 0.5
    };
}

fn build_oct(positions: &[Vec3], masses: &[f32]) -> OctTree {
    let mut tree = OctTree { nodes: Vec::new() };
    if positions.is_empty() {
        return tree;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    tree.nodes.push(OctNode {
        min,
        max,
        bodies: Vec::new(),
        children: [None, None, None, None, None, None, None, None],
        mass: 0.0,
        com: Vec3::ZERO,
    });
    for i in 0..positions.len() {
        insert_oct(&mut tree.nodes, 0, i, positions, 0);
    }
    accumulate_oct(&mut tree.nodes, 0, positions, masses);
    tree
}

#[allow(clippy::too_many_arguments)]
fn walk_oct(
    nodes: &[OctNode],
    node_idx: usize,
    qpos: Vec3,
    qidx: usize,
    positions: &[Vec3],
    masses: &[f32],
    big_g: f32,
    softening: f32,
    theta: f32,
    acc: &mut Vec3,
) {
    let node = &nodes[node_idx];
    if node.is_leaf() {
        for &k in &node.bodies {
            if k == qidx {
                continue;
            }
            let (a, _) = pair_accelerations_3(qpos, 1.0, positions[k], masses[k], big_g, softening);
            *acc += a;
        }
        return;
    }
    let routed = route_oct(node, qpos);
    for c in 0..8 {
        let Some(child) = node.children[c] else {
            continue;
        };
        if c == routed {
            walk_oct(
                nodes, child, qpos, qidx, positions, masses, big_g, softening, theta, acc,
            );
            continue;
        }
        let cn = &nodes[child];
        let d = cn.com - qpos;
        let dist2 = d.length_squared() + softening * softening;
        if dist2 < EPS_SQ {
            continue;
        }
        if cn.extent() / dist2.sqrt() < theta {
            let (a, _) = pair_accelerations_3(qpos, 1.0, cn.com, cn.mass, big_g, softening);
            *acc += a;
        } else {
            walk_oct(
                nodes, child, qpos, qidx, positions, masses, big_g, softening, theta, acc,
            );
        }
    }
}

/// 3D counterpart of [`apply_bh_forces_2`].
pub fn apply_bh_forces_3(
    bodies: &mut [RigidBody3],
    big_g: f32,
    softening: f32,
    theta: f32,
    h: f32,
    drag: bool,
) {
    let n = bodies.len();
    if n == 0 {
        return;
    }
    let positions: Vec<Vec3> = bodies.iter().map(|b| b.transform.position).collect();
    let masses: Vec<f32> = bodies.iter().map(|b| b.mass).collect();
    let tree = build_oct(&positions, &masses);
    for i in 0..n {
        if bodies[i].is_static {
            continue;
        }
        let mut acc = Vec3::ZERO;
        walk_oct(
            &tree.nodes,
            0,
            positions[i],
            i,
            &positions,
            &masses,
            big_g,
            softening,
            theta,
            &mut acc,
        );
        let b = &mut bodies[i];
        let mut dv = acc * (b.gravity_scale * h);
        if drag {
            dv -= b.velocity * (b.linear_damping * h);
        }
        b.velocity += dv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::{
        gravity_kick_2, gravity_kick_3, GravityConfig2, GravityConfig3, GravityMode,
    };

    fn two_bodies_2d() -> Vec<RigidBody2> {
        let mut a = RigidBody2::new(0);
        a.transform.position = Vec2::new(0.0, 0.0);
        a.mass = 3.0;
        let mut b = RigidBody2::new(1);
        b.transform.position = Vec2::new(4.0, 1.0);
        b.mass = 7.0;
        vec![a, b]
    }

    /// Two bodies never approximate (single-body cells are exact), so
    /// BH must agree with exact N-body bit-for-bit regardless of theta.
    #[test]
    fn test_bh_two_body_matches_exact() {
        for theta in [0.0, 0.3, 0.5, 1.0, 5.0] {
            let mut exact = two_bodies_2d();
            let mut approx = two_bodies_2d();
            let cfg = GravityConfig2 {
                mode: GravityMode::NBody,
                ..Default::default()
            };
            gravity_kick_2(&mut exact, &cfg, Vec2::ZERO, 1.0 / 60.0, false);
            apply_bh_forces_2(&mut approx, 1.0, 0.0, theta, 1.0 / 60.0, false);
            assert_eq!(exact[0].velocity, approx[0].velocity);
            assert_eq!(exact[1].velocity, approx[1].velocity);
        }
    }

    /// Same scene kicked twice (fresh trees) must agree bit-exactly.
    #[test]
    fn test_bh_deterministic() {
        fn scene() -> Vec<RigidBody2> {
            let mut v = Vec::new();
            for i in 0..20 {
                let mut b = RigidBody2::new(i);
                b.transform.position = Vec2::new(
                    (i as f32 * 1.7) % 40.0 - 20.0,
                    (i as f32 * 2.3) % 40.0 - 20.0,
                );
                b.mass = 1.0 + (i % 5) as f32;
                v.push(b);
            }
            v
        }
        let mut a = scene();
        let mut b = scene();
        apply_bh_forces_2(&mut a, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        apply_bh_forces_2(&mut b, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.velocity, y.velocity);
        }
    }

    /// A tight distant group collapses to its COM: approximation error
    /// against exact must stay small.
    #[test]
    fn test_bh_distant_group_accuracy() {
        // Query at the origin; cluster of 4 unit masses around (20, 0)
        // within a 0.2 box: s/d ~ 0.01, deep in the approx regime.
        let mut bodies = Vec::new();
        let mut q = RigidBody2::new(0);
        q.transform.position = Vec2::ZERO;
        q.mass = 1.0;
        bodies.push(q);
        for (i, off) in [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.2, 0.0),
            Vec2::new(0.0, 0.2),
            Vec2::new(0.2, 0.2),
        ]
        .iter()
        .enumerate()
        {
            let mut b = RigidBody2::new(i + 1);
            b.transform.position = Vec2::new(20.0, 0.0) + *off;
            b.mass = 1.0;
            bodies.push(b);
        }
        let mut exact = bodies.clone();
        let mut approx = bodies.clone();
        let cfg = GravityConfig2 {
            mode: GravityMode::NBody,
            ..Default::default()
        };
        gravity_kick_2(&mut exact, &cfg, Vec2::ZERO, 1.0, false);
        apply_bh_forces_2(&mut approx, 1.0, 0.0, 1.0, 1.0, false);
        // Exact pull on the query: ~4 * 1/400 = 0.01. Relative error
        // must stay well below 1% for this textbook configuration.
        let e = exact[0].velocity;
        let a = approx[0].velocity;
        let rel = (e - a).length() / e.length().max(1e-9);
        assert!(rel < 1e-2, "rel err {rel}: exact {e:?} approx {a:?}");
    }

    #[test]
    fn test_bh_empty_and_singleton_noop() {
        let mut empty: Vec<RigidBody2> = Vec::new();
        apply_bh_forces_2(&mut empty, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        let mut one = vec![RigidBody2::new(0)];
        one[0].transform.position = Vec2::new(3.0, 4.0);
        apply_bh_forces_2(&mut one, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        assert_eq!(one[0].velocity, Vec2::ZERO);
    }

    #[test]
    fn test_bh_coincident_terminates_finite() {
        // Five bodies on one point: depth cap stops subdivision;
        // skip guards keep every force finite and deterministic.
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..5 {
            let mut x = RigidBody2::new(i);
            x.transform.position = Vec2::ZERO;
            x.mass = 1.0;
            a.push(x.clone());
            b.push(x);
        }
        apply_bh_forces_2(&mut a, 1.0, 0.0, 0.5, 1.0 / 60.0, false);
        apply_bh_forces_2(&mut b, 1.0, 0.0, 0.5, 1.0 / 60.0, false);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(x.velocity.x.is_finite() && x.velocity.y.is_finite());
            assert_eq!(x.velocity, y.velocity);
        }
    }

    #[test]
    fn test_bh3_two_body_matches_exact() {
        let mk = |id: usize, p: Vec3, m: f32| {
            let mut b = RigidBody3::new(id);
            b.transform.position = p;
            b.mass = m;
            b
        };
        let mut exact = vec![mk(0, Vec3::ZERO, 2.0), mk(1, Vec3::new(1.0, 2.0, 3.0), 5.0)];
        let mut approx = exact.clone();
        let cfg = GravityConfig3 {
            mode: GravityMode::NBody,
            ..Default::default()
        };
        gravity_kick_3(&mut exact, &cfg, Vec3::ZERO, 1.0 / 60.0, false);
        apply_bh_forces_3(&mut approx, 1.0, 0.0, 0.7, 1.0 / 60.0, false);
        assert_eq!(exact[0].velocity, approx[0].velocity);
        assert_eq!(exact[1].velocity, approx[1].velocity);
    }

    #[test]
    fn test_bh3_deterministic() {
        fn scene() -> Vec<RigidBody3> {
            let mut v = Vec::new();
            for i in 0..12 {
                let mut b = RigidBody3::new(i);
                b.transform.position = Vec3::new(
                    (i as f32 * 1.7) % 20.0 - 10.0,
                    (i as f32 * 2.3) % 20.0 - 10.0,
                    (i as f32 * 3.1) % 20.0 - 10.0,
                );
                b.mass = 1.0 + (i % 3) as f32;
                v.push(b);
            }
            v
        }
        let mut a = scene();
        let mut b = scene();
        apply_bh_forces_3(&mut a, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        apply_bh_forces_3(&mut b, 1.0, 0.1, 0.5, 1.0 / 60.0, false);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.velocity, y.velocity);
        }
    }
}
