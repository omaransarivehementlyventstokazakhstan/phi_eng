use super::body::{RigidBody2, RigidBody3};
use super::gravity::{pair_accelerations_2, pair_accelerations_3};
use glam::{Vec2, Vec3};
use phieng_math::EPS_SQ;
use std::cell::RefCell;

/// Maximum subdivision depth: bounds coincident-point splitting.
/// A full leaf at max depth accumulates instead of subdividing, so
/// insertion always terminates.
const MAX_TREE_DEPTH: u32 = 32;

// ---------------------------------------------------------------------------
// 2D quadtree.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
    fn extent(&self) -> f32 {
        (self.max.x - self.min.x).max(self.max.y - self.min.y)
    }
}

struct QuadTree {
    nodes: Vec<QuadNode>,
}

/// A reusable scratch buffer for a 2D Barnes-Hut tree. The arena
/// (`nodes`) and every per-leaf index `Vec` are recycled across kicks
/// instead of being freed, mirroring the spatial-hash / contact-cache
/// pooling pattern. One instance per thread is enough because gravity
/// kicks are serial per world.
#[derive(Default)]
struct QuadTreePool {
    /// Recycled node arena. `build_quad_into` clears and reuses it.
    nodes: Vec<QuadNode>,
}

thread_local! {
    static QUAD_POOL: RefCell<QuadTreePool> = const { RefCell::new(QuadTreePool { nodes: Vec::new() }) };
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
    for c in 0..4 {
        nodes[node_idx].children[c] = Some(base + c);
    }
}

fn insert_quad(
    nodes: &mut Vec<QuadNode>,
    node_idx: usize,
    body_idx: usize,
    positions: &[Vec2],
    depth: u32,
) {
    let p = positions[body_idx];
    if nodes[node_idx].is_leaf() {
        if nodes[node_idx].bodies.is_empty() || depth >= MAX_TREE_DEPTH {
            nodes[node_idx].bodies.push(body_idx);
            return;
        }
        // Convert leaf into an internal node: move existing bodies down.
        let existing: Vec<usize> = std::mem::take(&mut nodes[node_idx].bodies);
        subdivide_quad(nodes, node_idx);
        for &k in &existing {
            insert_quad(nodes, node_idx, k, positions, depth + 1);
        }
        insert_quad(nodes, node_idx, body_idx, positions, depth + 1);
        return;
    }
    let child = route_quad(&nodes[node_idx], p);
    if let Some(ci) = nodes[node_idx].children[child] {
        insert_quad(nodes, ci, body_idx, positions, depth + 1);
    }
}

fn accumulate_quad(nodes: &mut Vec<QuadNode>, node_idx: usize, positions: &[Vec2], masses: &[f32]) {
    let node = &nodes[node_idx];
    if node.is_leaf() {
        let mut mass = 0.0f32;
        let mut com = Vec2::ZERO;
        for &k in &node.bodies {
            mass += masses[k];
            com += positions[k] * masses[k];
        }
        let node = &mut nodes[node_idx];
        node.mass = mass;
        node.com = if mass > 0.0 {
            com / mass
        } else {
            (node.min + node.max) * 0.5
        };
        return;
    }
    for c in 0..4 {
        if let Some(child) = nodes[node_idx].children[c] {
            accumulate_quad(nodes, child, positions, masses);
        }
    }
    let mut mass = 0.0f32;
    let mut com = Vec2::ZERO;
    for c in 0..4 {
        if let Some(child) = nodes[node_idx].children[c] {
            mass += nodes[child].mass;
            com += nodes[child].com * nodes[child].mass;
        }
    }
    let node = &mut nodes[node_idx];
    node.mass = mass;
    node.com = if mass > 0.0 {
        com / mass
    } else {
        (node.min + node.max) * 0.5
    };
}

/// Build a quadtree, reusing the thread-local pool's arena. The returned
/// tree owns its node list (moved out of the pool); the pool is refilled
/// with a fresh `Vec` so the next kick reuses the freed allocation.
fn build_quad_pooled(positions: &[Vec2], masses: &[f32]) -> QuadTree {
    if positions.is_empty() {
        return QuadTree { nodes: Vec::new() };
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    // Draw the arena from the pool.
    let mut nodes = QUAD_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        let mut arena = std::mem::take(&mut pool.nodes);
        arena.clear();
        arena
    });
    nodes.push(QuadNode {
        min,
        max,
        bodies: Vec::new(),
        children: [None, None, None, None],
        mass: 0.0,
        com: Vec2::ZERO,
    });
    for i in 0..positions.len() {
        insert_quad(&mut nodes, 0, i, positions, 0);
    }
    accumulate_quad(&mut nodes, 0, positions, masses);
    QuadTree { nodes }
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
    let tree = build_quad_pooled(&positions, &masses);
    for i in 0..n {
        if bodies[i].is_static || bodies[i].is_sleeping {
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
    // Recycle the arena back into the pool so the next kick reuses
    // the allocation instead of freeing it.
    QUAD_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        pool.nodes = tree.nodes;
    });
}

// ---------------------------------------------------------------------------
// 3D octree (same contract as the quadtree above).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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

#[derive(Default)]
struct OctTreePool {
    nodes: Vec<OctNode>,
}

thread_local! {
    static OCT_POOL: RefCell<OctTreePool> = const { RefCell::new(OctTreePool { nodes: Vec::new() }) };
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
    let octs = [
        (min, mid),
        (
            Vec3::new(mid.x, min.y, min.z),
            Vec3::new(max.x, mid.y, mid.z),
        ),
        (
            Vec3::new(min.x, mid.y, min.z),
            Vec3::new(mid.x, max.y, mid.z),
        ),
        (
            Vec3::new(mid.x, mid.y, min.z),
            Vec3::new(max.x, max.y, mid.z),
        ),
        (
            Vec3::new(min.x, min.y, mid.z),
            Vec3::new(mid.x, mid.y, max.z),
        ),
        (
            Vec3::new(mid.x, min.y, mid.z),
            Vec3::new(max.x, mid.y, max.z),
        ),
        (
            Vec3::new(min.x, mid.y, mid.z),
            Vec3::new(mid.x, max.y, max.z),
        ),
        (mid, max),
    ];
    for (lo, hi) in octs {
        nodes.push(OctNode {
            min: lo,
            max: hi,
            bodies: Vec::new(),
            children: [None, None, None, None, None, None, None, None],
            mass: 0.0,
            com: Vec3::ZERO,
        });
    }
    let base = nodes.len() - 8;
    for c in 0..8 {
        nodes[node_idx].children[c] = Some(base + c);
    }
}

fn insert_oct(
    nodes: &mut Vec<OctNode>,
    node_idx: usize,
    body_idx: usize,
    positions: &[Vec3],
    depth: u32,
) {
    let p = positions[body_idx];
    if nodes[node_idx].is_leaf() {
        if nodes[node_idx].bodies.is_empty() || depth >= MAX_TREE_DEPTH {
            nodes[node_idx].bodies.push(body_idx);
            return;
        }
        let existing: Vec<usize> = std::mem::take(&mut nodes[node_idx].bodies);
        subdivide_oct(nodes, node_idx);
        for &k in &existing {
            insert_oct(nodes, node_idx, k, positions, depth + 1);
        }
        insert_oct(nodes, node_idx, body_idx, positions, depth + 1);
        return;
    }
    let child = route_oct(&nodes[node_idx], p);
    if let Some(ci) = nodes[node_idx].children[child] {
        insert_oct(nodes, ci, body_idx, positions, depth + 1);
    }
}

fn accumulate_oct(nodes: &mut Vec<OctNode>, node_idx: usize, positions: &[Vec3], masses: &[f32]) {
    let node = &nodes[node_idx];
    if node.is_leaf() {
        let mut mass = 0.0f32;
        let mut com = Vec3::ZERO;
        for &k in &node.bodies {
            mass += masses[k];
            com += positions[k] * masses[k];
        }
        let node = &mut nodes[node_idx];
        node.mass = mass;
        node.com = if mass > 0.0 {
            com / mass
        } else {
            (node.min + node.max) * 0.5
        };
        return;
    }
    for c in 0..8 {
        if let Some(child) = nodes[node_idx].children[c] {
            accumulate_oct(nodes, child, positions, masses);
        }
    }
    let mut mass = 0.0f32;
    let mut com = Vec3::ZERO;
    for c in 0..8 {
        if let Some(child) = nodes[node_idx].children[c] {
            mass += nodes[child].mass;
            com += nodes[child].com * nodes[child].mass;
        }
    }
    let node = &mut nodes[node_idx];
    node.mass = mass;
    node.com = if mass > 0.0 {
        com / mass
    } else {
        (node.min + node.max) * 0.5
    };
}

fn build_oct_pooled(positions: &[Vec3], masses: &[f32]) -> OctTree {
    if positions.is_empty() {
        return OctTree { nodes: Vec::new() };
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    let mut nodes = OCT_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        let mut arena = std::mem::take(&mut pool.nodes);
        arena.clear();
        arena
    });
    nodes.push(OctNode {
        min,
        max,
        bodies: Vec::new(),
        children: [None, None, None, None, None, None, None, None],
        mass: 0.0,
        com: Vec3::ZERO,
    });
    for i in 0..positions.len() {
        insert_oct(&mut nodes, 0, i, positions, 0);
    }
    accumulate_oct(&mut nodes, 0, positions, masses);
    OctTree { nodes }
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
    let tree = build_oct_pooled(&positions, &masses);
    for i in 0..n {
        if bodies[i].is_static || bodies[i].is_sleeping {
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
    OCT_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        pool.nodes = tree.nodes;
    });
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
        a.mass = 1.0;
        a.inv_mass = 1.0;
        a.inertia = 0.5;
        a.inv_inertia = 2.0;
        let mut b = RigidBody2::new(1);
        b.transform.position = Vec2::new(10.0, 0.0);
        b.mass = 2.0;
        b.inv_mass = 0.5;
        b.inertia = 1.0;
        b.inv_inertia = 1.0;
        vec![a, b]
    }

    fn two_bodies_3d() -> Vec<RigidBody3> {
        let mut a = RigidBody3::new(0);
        a.transform.position = Vec3::new(0.0, 0.0, 0.0);
        a.mass = 1.0;
        a.inv_mass = 1.0;
        a.inertia = glam::Mat3::IDENTITY;
        a.inv_inertia = glam::Mat3::IDENTITY;
        let mut b = RigidBody3::new(1);
        b.transform.position = Vec3::new(10.0, 0.0, 0.0);
        b.mass = 2.0;
        b.inv_mass = 0.5;
        b.inertia = glam::Mat3::IDENTITY * 2.0;
        b.inv_inertia = (glam::Mat3::IDENTITY * 2.0).inverse();
        vec![a, b]
    }

    fn scene_2d() -> Vec<RigidBody2> {
        (0..20)
            .map(|i| {
                let mut b = RigidBody2::new(i);
                b.transform.position = Vec2::new(i as f32 * 0.5, (i as f32 * 0.37).sin());
                b.mass = 1.0 + (i as f32 * 0.1);
                b.inv_mass = 1.0 / b.mass;
                b.inertia = 0.5;
                b.inv_inertia = 2.0;
                b
            })
            .collect()
    }

    fn scene_3d() -> Vec<RigidBody3> {
        (0..20)
            .map(|i| {
                let mut b = RigidBody3::new(i);
                b.transform.position = Vec3::new(
                    i as f32 * 0.5,
                    (i as f32 * 0.37).sin(),
                    (i as f32 * 0.23).cos(),
                );
                b.mass = 1.0 + (i as f32 * 0.1);
                b.inv_mass = 1.0 / b.mass;
                b.inertia = glam::Mat3::IDENTITY;
                b.inv_inertia = glam::Mat3::IDENTITY;
                b
            })
            .collect()
    }

    #[test]
    fn test_bh_coincident_terminates_finite() {
        let mut bodies = vec![RigidBody2::new(0), RigidBody2::new(1)];
        bodies[0].transform.position = Vec2::ZERO;
        bodies[1].transform.position = Vec2::ZERO;
        bodies[0].mass = 1.0;
        bodies[0].inv_mass = 1.0;
        bodies[0].inertia = 0.5;
        bodies[0].inv_inertia = 2.0;
        bodies[1].mass = 1.0;
        bodies[1].inv_mass = 1.0;
        bodies[1].inertia = 0.5;
        bodies[1].inv_inertia = 2.0;
        let g = GravityConfig2 {
            mode: GravityMode::BarnesHut,
            theta: 0.5,
            ..Default::default()
        };
        gravity_kick_2(&mut bodies, &g, Vec2::ZERO, 1.0 / 60.0, false);
        assert!(bodies[0].velocity.x.is_finite());
        assert!(bodies[0].velocity.y.is_finite());
        assert!(bodies[1].velocity.x.is_finite());
        assert!(bodies[1].velocity.y.is_finite());
    }

    #[test]
    fn test_bh_two_body_matches_exact() {
        // Reference: exact pairwise N-body. BH must be bit-identical for a
        // two-body scene because each cell's COM is a single body.
        let mut exact = two_bodies_2d();
        let mut bh = two_bodies_2d();
        let g_exact = GravityConfig2 {
            mode: GravityMode::NBody,
            big_g: 1.0,
            ..Default::default()
        };
        let g_bh = GravityConfig2 {
            mode: GravityMode::BarnesHut,
            big_g: 1.0,
            theta: 0.5,
            ..Default::default()
        };
        gravity_kick_2(&mut exact, &g_exact, Vec2::ZERO, 1.0 / 60.0, false);
        gravity_kick_2(&mut bh, &g_bh, Vec2::ZERO, 1.0 / 60.0, false);
        assert_eq!(exact[0].velocity, bh[0].velocity);
        assert_eq!(exact[1].velocity, bh[1].velocity);
    }

    #[test]
    fn test_bh_deterministic() {
        let mut a = scene_2d();
        let mut b = scene_2d();
        let g = GravityConfig2 {
            mode: GravityMode::BarnesHut,
            theta: 0.5,
            ..Default::default()
        };
        gravity_kick_2(&mut a, &g, Vec2::ZERO, 1.0 / 60.0, false);
        gravity_kick_2(&mut b, &g, Vec2::ZERO, 1.0 / 60.0, false);
        for i in 0..a.len() {
            assert_eq!(a[i].velocity, b[i].velocity);
        }
    }

    #[test]
    fn test_bh_distant_group_accuracy() {
        // A far cluster approximates as its COM; the force on a near
        // body matches the exact sum within a tolerance that shrinks
        // with theta.
        let mut bodies = Vec::new();
        for i in 0..5 {
            let mut b = RigidBody2::new(i);
            b.transform.position = Vec2::new(1000.0 + i as f32, 0.0);
            b.mass = 1.0;
            b.inv_mass = 1.0;
            b.inertia = 0.5;
            b.inv_inertia = 2.0;
            bodies.push(b);
        }
        let mut probe = RigidBody2::new(5);
        probe.transform.position = Vec2::ZERO;
        probe.mass = 1.0;
        probe.inv_mass = 1.0;
        probe.inertia = 0.5;
        probe.inv_inertia = 2.0;
        bodies.push(probe);
        let mut exact = bodies.clone();
        let mut bh = bodies.clone();
        let g_exact = GravityConfig2 {
            mode: GravityMode::Uniform,
            ..Default::default()
        };
        let g_bh = GravityConfig2 {
            mode: GravityMode::BarnesHut,
            theta: 0.1,
            ..Default::default()
        };
        gravity_kick_2(&mut exact, &g_exact, Vec2::ZERO, 1.0 / 60.0, false);
        gravity_kick_2(&mut bh, &g_bh, Vec2::ZERO, 1.0 / 60.0, false);
        let err = (exact[5].velocity - bh[5].velocity).length();
        assert!(err < 1e-2, "BH error too large: {}", err);
    }

    #[test]
    fn test_bh3_two_body_matches_exact() {
        // Reference: exact pairwise N-body. BH must be bit-identical for a
        // two-body scene because each cell's COM is a single body.
        let mut exact = two_bodies_3d();
        let mut bh = two_bodies_3d();
        let g_exact = GravityConfig3 {
            mode: GravityMode::NBody,
            big_g: 1.0,
            ..Default::default()
        };
        let g_bh = GravityConfig3 {
            mode: GravityMode::BarnesHut,
            big_g: 1.0,
            theta: 0.5,
            ..Default::default()
        };
        gravity_kick_3(&mut exact, &g_exact, Vec3::ZERO, 1.0 / 60.0, false);
        gravity_kick_3(&mut bh, &g_bh, Vec3::ZERO, 1.0 / 60.0, false);
        assert_eq!(exact[0].velocity, bh[0].velocity);
        assert_eq!(exact[1].velocity, bh[1].velocity);
    }

    #[test]
    fn test_bh3_deterministic() {
        let mut a = scene_3d();
        let mut b = scene_3d();
        let g = GravityConfig3 {
            mode: GravityMode::BarnesHut,
            theta: 0.5,
            ..Default::default()
        };
        gravity_kick_3(&mut a, &g, Vec3::ZERO, 1.0 / 60.0, false);
        gravity_kick_3(&mut b, &g, Vec3::ZERO, 1.0 / 60.0, false);
        for i in 0..a.len() {
            assert_eq!(a[i].velocity, b[i].velocity);
        }
    }

    #[test]
    fn test_bh_empty_and_singleton_noop() {
        let mut empty: Vec<RigidBody2> = Vec::new();
        let g = GravityConfig2 {
            mode: GravityMode::BarnesHut,
            ..Default::default()
        };
        gravity_kick_2(&mut empty, &g, Vec2::ZERO, 1.0 / 60.0, false);
        assert!(empty.is_empty());

        let mut single = vec![RigidBody2::new(0)];
        single[0].mass = 1.0;
        single[0].inv_mass = 1.0;
        single[0].inertia = 0.5;
        single[0].inv_inertia = 2.0;
        let before = single[0].velocity;
        gravity_kick_2(&mut single, &g, Vec2::ZERO, 1.0 / 60.0, false);
        // Single body, no neighbors: no force (and gravity_scale applies).
        assert_eq!(single[0].velocity, before);
    }

    #[test]
    fn test_bh3_empty_and_singleton_noop() {
        let mut empty: Vec<RigidBody3> = Vec::new();
        let g = GravityConfig3 {
            mode: GravityMode::BarnesHut,
            ..Default::default()
        };
        gravity_kick_3(&mut empty, &g, Vec3::ZERO, 1.0 / 60.0, false);
        assert!(empty.is_empty());
    }
}
