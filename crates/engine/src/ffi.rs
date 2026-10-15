//! C-compatible FFI for driving the Rust 2D physics core from C#.
//!
//! Design notes (mirrored by `src/PhiEng.Native/NativeMethods.cs`):
//! - The world is an opaque `*mut World2`; null is never returned by
//!   [`phieng_world_new`] and every entry point tolerates null (no-op / error code).
//! - Body ids are the engine's `usize` ids, truncated to `u32` on add and
//!   widened back on query. All indices are bounds-checked.
//! - `is_static` is a `u8` (0 = dynamic) to avoid C/C# `bool` size ambiguity.
//! - Transforms are copied out through caller-owned pointers (`out_pos` points
//!   at 2 `f32`s, `out_rot`/`out_vel` follow the same pattern), so no
//!   allocation happens on the FFI boundary.

use glam::{Vec2, Vec3};
use phieng_dynamics::{PhysicsMaterial, Shape2, World2};
use phieng_math::Transform2;

/// Shape tag for a 2D circle (`a` = radius, `b` unused).
pub const SHAPE_CIRCLE: u32 = 0;
/// Shape tag for a 2D box (`a` = half_x, `b` = half_y).
pub const SHAPE_BOX: u32 = 1;

fn valid(v: f32) -> bool {
    v.is_finite()
}

fn valid_positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

/// Look up a body by its public id. Returns None for stale/foreign ids.
fn body_by_id(world: &World2, body_id: u32) -> Option<usize> {
    let index = body_id as usize;
    world
        .bodies
        .get(index)
        .filter(|b| b.id == index)
        .map(|_| index)
}

/// Create a new 2D physics world. The caller owns the pointer and must
/// release it with [`phieng_world_free`]. Never returns null.
#[no_mangle]
pub extern "C" fn phieng_world_new() -> *mut World2 {
    Box::into_raw(Box::new(World2::new()))
}

/// Destroy a world previously created with [`phieng_world_new`].
/// Null is a no-op.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_free(world: *mut World2) {
    if !world.is_null() {
        drop(Box::from_raw(world));
    }
}

/// Advance the simulation by `dt` seconds. Null world, or non-positive /
/// non-finite `dt`, is a no-op. Very large steps are clamped to 1/15s to
/// avoid explosions from caller error.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_step(world: *mut World2, dt: f32) {
    if world.is_null() || !valid(dt) || dt <= 0.0 {
        return;
    }
    (*world).step(dt.min(1.0 / 15.0));
}

/// Set world gravity. Null is a no-op; non-finite components are ignored.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_gravity(world: *mut World2, gx: f32, gy: f32) {
    if world.is_null() || !valid(gx) || !valid(gy) {
        return;
    }
    (*world).gravity = Vec2::new(gx, gy);
}

/// Number of bodies in the world, or -1 for null.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_body_count(world: *mut World2) -> i32 {
    if world.is_null() {
        return -1;
    }
    (*world).bodies.len() as i32
}

/// Remove a body by id. Returns 1 on success, 0 for null world or
/// unknown body. Remaining bodies are reindexed (see
/// [`World2::remove_body`]).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_remove_body(world: *mut World2, body_id: u32) -> i32 {
    if world.is_null() {
        return 0;
    }
    if body_by_id(&*world, body_id).is_none() {
        return 0;
    }
    (*world).remove_body(body_id as usize);
    1
}

/// Add a rigid body. `shape_tag` is [`SHAPE_CIRCLE`] (`a` = radius) or
/// [`SHAPE_BOX`] (`a` = half_x, `b` = half_y). `density` derives mass;
/// `friction`/`restitution` form the body material. `is_static` != 0 makes
/// an immovable body. Returns the body id, or -1 on any invalid input.
#[no_mangle]
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn phieng_world_add_body(
    world: *mut World2,
    pos_x: f32,
    pos_y: f32,
    shape_tag: u32,
    a: f32,
    b: f32,
    density: f32,
    is_static: u8,
    friction: f32,
    restitution: f32,
) -> i32 {
    if world.is_null() || !valid(pos_x) || !valid(pos_y) {
        return -1;
    }
    let shape = match shape_tag {
        SHAPE_CIRCLE if valid_positive(a) => Shape2::Circle { radius: a },
        SHAPE_BOX if valid_positive(a) && valid_positive(b) => Shape2::Box {
            half_extents: Vec2::new(a, b),
        },
        _ => return -1,
    };
    if !valid_positive(density) {
        return -1;
    }

    let world = &mut *world;
    let position = Vec2::new(pos_x, pos_y);
    let material = PhysicsMaterial::new(friction, restitution, density);
    if is_static != 0 {
        let mut body = phieng_dynamics::RigidBody2::static_body(world.next_body_id);
        body.transform = Transform2::from_position_angle(position, 0.0);
        if body.set_shape(shape, density).is_err() || body.set_material(material).is_err() {
            return -1;
        }
        world.add_body(body).map(|id| id as i32).unwrap_or(-1)
    } else {
        let mut body = phieng_dynamics::RigidBody2::new(world.next_body_id);
        body.transform = Transform2::from_position_angle(position, 0.0);
        if body.set_shape(shape, density).is_err() || body.set_material(material).is_err() {
            return -1;
        }
        world.add_body(body).map(|id| id as i32).unwrap_or(-1)
    }
}

/// Replace a body's material (friction, restitution, density), recomputing
/// mass/inertia from its current shape. Returns 1 on success, 0 for null
/// world or unknown body. Invalid components fall back to safe defaults
/// (see [`PhysicsMaterial::new`]).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_body_material(
    world: *mut World2,
    body_id: u32,
    friction: f32,
    density: f32,
    restitution: f32,
) -> i32 {
    if world.is_null() {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    // Input is pre-sanitized by PhysicsMaterial::new, so this is infallible
    // in practice; the error is unreachable but mapped explicitly.
    let _ =
        world_ref.bodies[index].set_material(PhysicsMaterial::new(friction, restitution, density));
    1
}

/// Copy a body's material out: `out_mat[0..3]` = (friction, density,
/// restitution). Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_mat` must be either null (clean `0`) or valid for writes of
/// 3 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_material(
    world: *mut World2,
    body_id: u32,
    out_mat: *mut f32,
) -> i32 {
    if world.is_null() || out_mat.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    let material = world_ref.bodies[index].material;
    *out_mat.add(0) = material.friction;
    *out_mat.add(1) = material.density;
    *out_mat.add(2) = material.restitution;
    1
}

/// Copy a body's transform out: `out_pos[0..2]` = (x, y),
/// `*out_rot` = rotation angle in radians. Returns 1 on success, 0 on any
/// invalid input (null world/pointers, unknown body).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_pos` must be either null (clean `0`) or valid for writes of
/// 2 `f32`s; `out_rot` either null or valid for 1 `f32`. Both must be
/// properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_transform(
    world: *mut World2,
    body_id: u32,
    out_pos: *mut f32,
    out_rot: *mut f32,
) -> i32 {
    if world.is_null() || out_pos.is_null() || out_rot.is_null() {
        return 0;
    }
    // Explicit reference: indexing directly through a raw pointer is rejected
    // by the compiler (dangerous_implicit_autorefs).
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    let body = &world_ref.bodies[index];
    *out_pos.add(0) = body.transform.position.x;
    *out_pos.add(1) = body.transform.position.y;
    *out_rot = body.rotation_angle();
    1
}

/// Copy a body's linear velocity out: `out_vel[0..2]` = (vx, vy).
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_vel` must be either null (clean `0`) or valid for writes of
/// 2 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_velocity(
    world: *mut World2,
    body_id: u32,
    out_vel: *mut f32,
) -> i32 {
    if world.is_null() || out_vel.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    let body = &world_ref.bodies[index];
    *out_vel.add(0) = body.velocity.x;
    *out_vel.add(1) = body.velocity.y;
    1
}

/// Add SPH fluid particles from interleaved `positions` (`count` pairs
/// `[x0, y0, x1, y1, ...]`), starting at zero velocity. Returns the
/// number added (capped by the fluid particle limit), or 0 on null
/// world/positions or any non-finite input (atomic: nothing is added).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which yields a clean `0`). `positions` must be null (clean `0`) or
/// valid for reads of `2 * count` `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_fluid_add(
    world: *mut World2,
    positions: *const f32,
    count: u32,
) -> u32 {
    if world.is_null() || positions.is_null() {
        return 0;
    }
    let n = (count as usize).min(1_000_000);
    let src = unsafe { std::slice::from_raw_parts(positions, n.saturating_mul(2)) };
    if src.iter().any(|v| !v.is_finite()) {
        return 0;
    }
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks
    // (`src` is exactly `2 * n` long by construction above).
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let pts: Vec<Vec2> = src.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect();
    unsafe { &mut *world }.fluid.add_particles(&pts) as u32
}

/// Fluid particle count. Returns 0 on null world (which is also the
/// empty-fluid answer).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which yields a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_fluid_count(world: *const World2) -> u32 {
    if world.is_null() {
        return 0;
    }
    unsafe { &*world }.fluid.particles.len() as u32
}

/// Remove all fluid particles (keeps allocations). Returns 1 on
/// success, 0 on null world.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which yields a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_fluid_clear(world: *mut World2) -> i32 {
    if world.is_null() {
        return 0;
    }
    unsafe { &mut *world }.fluid.clear();
    1
}

/// Copy particle state out interleaved (`max` quads
/// `[x, y, vx, vy]`). Returns the number of particles written, 0 on
/// null world/out.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which yields a clean `0`). `out` must be null (clean `0`) or valid
/// for writes of `4 * max` `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_fluid_get(
    world: *const World2,
    out: *mut f32,
    max: u32,
) -> u32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let n = (max as usize).min(world_ref.fluid.particles.len());
    for (i, p) in world_ref.fluid.particles.iter().take(n).enumerate() {
        unsafe {
            *out.add(i * 4) = p.position.x;
            *out.add(i * 4 + 1) = p.position.y;
            *out.add(i * 4 + 2) = p.velocity.x;
            *out.add(i * 4 + 3) = p.velocity.y;
        }
    }
    n as u32
}

// ============================================================================
// Joint FFI (generic tag + JSON params)
// ============================================================================

/// Distance joint tag: `{"anchorA":[x,y],"anchorB":[x,y],
/// "restLength":f,"compliance":f}`.
pub const JOINT_DISTANCE: u32 = 0;
/// Hinge joint tag: `{"anchorA":[x,y],"anchorB":[x,y],
/// "limitMin":f,"limitMax":f,"compliance":f}` (`min > max` = free).
pub const JOINT_HINGE: u32 = 1;
/// Prismatic joint tag: `{"anchorA":[x,y],"anchorB":[x,y],"axis":[x,y],
/// "limitMin":f,"limitMax":f,"compliance":f}` (`min > max` = free travel).
pub const JOINT_PRISMATIC: u32 = 2;
/// Fixed joint tag: `{"anchorA":[x,y],"anchorB":[x,y],"compliance":f}`.
pub const JOINT_FIXED: u32 = 3;

/// Maximum accepted JSON blob length (64 KiB). Larger inputs are rejected
/// instead of parsed — a joint description is tens of bytes; anything
/// bigger is caller error, not a joint.
const MAX_JOINT_JSON_LEN: usize = 64 * 1024;

/// Read a `[x, y]` array field from a JSON object.
fn json_vec2(v: &serde_json::Value, field: &str) -> Option<Vec2> {
    let arr = v.get(field)?.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let x = arr[0].as_f64()? as f32;
    let y = arr[1].as_f64()? as f32;
    if x.is_finite() && y.is_finite() {
        Some(Vec2::new(x, y))
    } else {
        None
    }
}

/// Read a required finite `f32` field.
fn json_f32(v: &serde_json::Value, field: &str) -> Option<f32> {
    let x = v.get(field)?.as_f64()? as f32;
    x.is_finite().then_some(x)
}

/// Read the optional `compliance` field (defaults to rigid 0.0 when
/// absent; must be finite when present — validated again downstream).
fn json_compliance(v: &serde_json::Value) -> Option<f32> {
    match v.get("compliance") {
        None => Some(0.0),
        Some(c) => {
            let x = c.as_f64()? as f32;
            x.is_finite().then_some(x)
        }
    }
}

/// Add a joint from a JSON param blob.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which is a clean `-1`). `json_ptr` must point at `json_len` readable
/// bytes of UTF-8 when `json_len > 0`; null with nonzero length is `-1`.
/// Never panics on hostile input: malformed JSON, unknown tags, stale
/// body ids, and validation failures all map to `-1`.
///
/// Returns the joint id, or -1 on any invalid input.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_add_joint(
    world: *mut World2,
    tag: u32,
    body_a: u32,
    body_b: u32,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    use phieng_dynamics::{
        make_distance_joint, make_fixed_joint, make_hinge_joint, make_prismatic_joint,
    };

    if world.is_null() || json_ptr.is_null() || json_len == 0 || json_len > MAX_JOINT_JSON_LEN {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let Ok(text) = std::str::from_utf8(bytes) else {
        return -1;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return -1;
    };

    let world_ref: &mut World2 = unsafe { &mut *world };
    let n = world_ref.bodies.len();
    let next_id = world_ref.next_joint_id;

    let joint = match tag {
        JOINT_DISTANCE => {
            let (Some(a), Some(b), Some(rest), Some(comp)) = (
                json_vec2(&v, "anchorA"),
                json_vec2(&v, "anchorB"),
                json_f32(&v, "restLength"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_distance_joint(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                rest,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT_HINGE => {
            let (Some(a), Some(b), Some(lo), Some(hi), Some(comp)) = (
                json_vec2(&v, "anchorA"),
                json_vec2(&v, "anchorB"),
                json_f32(&v, "limitMin"),
                json_f32(&v, "limitMax"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_hinge_joint(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                lo,
                hi,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT_PRISMATIC => {
            let (Some(a), Some(b), Some(ax), Some(lo), Some(hi), Some(comp)) = (
                json_vec2(&v, "anchorA"),
                json_vec2(&v, "anchorB"),
                json_vec2(&v, "axis"),
                json_f32(&v, "limitMin"),
                json_f32(&v, "limitMax"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_prismatic_joint(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                ax,
                lo,
                hi,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT_FIXED => {
            let (Some(a), Some(b), Some(comp)) = (
                json_vec2(&v, "anchorA"),
                json_vec2(&v, "anchorB"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_fixed_joint(next_id, body_a as usize, body_b as usize, a, b, comp, n) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        _ => return -1,
    };

    match world_ref.add_joint(joint) {
        Ok(id) => id as i32,
        Err(_) => -1,
    }
}

/// Remove a joint by id. Returns 1 on success, 0 for null world or
/// unknown joint. Never panics.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which is a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_remove_joint(world: *mut World2, joint_id: u32) -> i32 {
    if world.is_null() {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    if (joint_id as usize) >= world_ref.joints.len() {
        return 0;
    }
    world_ref.remove_joint(joint_id as usize);
    1
}

/// Number of joints in the world, or -1 for null.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// which yields `-1`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_joint_count(world: *mut World2) -> i32 {
    if world.is_null() {
        return -1;
    }
    unsafe { (*world).joints.len() as i32 }
}

// ============================================================================
// 3D joint FFI (generic tag + JSON params, mirrors the 2D joint bridge)
// ============================================================================

/// 3D distance joint tag: `{"anchorA":[x,y,z],"anchorB":[x,y,z],
/// "restLength":f,"compliance":f}`.
pub const JOINT3_DISTANCE: u32 = 0;
/// 3D hinge joint tag: `{"anchorA":[x,y,z],"anchorB":[x,y,z],
/// "axisA":[x,y,z],"axisB":[x,y,z],"limitMin":f,"limitMax":f,
/// "compliance":f}` (`min > max` = free twist).
pub const JOINT3_HINGE: u32 = 1;
/// 3D prismatic joint tag: `{"anchorA":[x,y,z],"anchorB":[x,y,z],
/// "axis":[x,y,z],"limitMin":f,"limitMax":f,"compliance":f}`
/// (`min > max` = free travel).
pub const JOINT3_PRISMATIC: u32 = 2;
/// 3D fixed joint tag: `{"anchorA":[x,y,z],"anchorB":[x,y,z],
/// "compliance":f}`.
pub const JOINT3_FIXED: u32 = 3;

/// Read a `[x, y, z]` array field from a JSON object.
fn json_vec3(v: &serde_json::Value, field: &str) -> Option<glam::Vec3> {
    let arr = v.get(field)?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    let x = arr[0].as_f64()? as f32;
    let y = arr[1].as_f64()? as f32;
    let z = arr[2].as_f64()? as f32;
    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some(glam::Vec3::new(x, y, z))
    } else {
        None
    }
}

/// Add a 3D joint from a JSON param blob.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which is a clean `-1`). `json_ptr` must point at `json_len` readable
/// bytes of UTF-8 when `json_len > 0`; null with nonzero length is `-1`.
/// Never panics on hostile input: malformed JSON, unknown tags, stale
/// body ids, and validation failures all map to `-1`.
///
/// Returns the joint id, or -1 on any invalid input.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_add_joint(
    world: *mut phieng_dynamics::World3,
    tag: u32,
    body_a: u32,
    body_b: u32,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    use phieng_dynamics::{
        make_distance_joint_3, make_fixed_joint_3, make_hinge_joint_3, make_prismatic_joint_3,
    };

    if world.is_null() || json_ptr.is_null() || json_len == 0 || json_len > MAX_JOINT_JSON_LEN {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let Ok(text) = std::str::from_utf8(bytes) else {
        return -1;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return -1;
    };

    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let n = world_ref.bodies.len();
    let next_id = world_ref.next_joint_id;

    let joint = match tag {
        JOINT3_DISTANCE => {
            let (Some(a), Some(b), Some(rest), Some(comp)) = (
                json_vec3(&v, "anchorA"),
                json_vec3(&v, "anchorB"),
                json_f32(&v, "restLength"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_distance_joint_3(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                rest,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT3_HINGE => {
            let (Some(a), Some(b), Some(ax_a), Some(ax_b), Some(lo), Some(hi), Some(comp)) = (
                json_vec3(&v, "anchorA"),
                json_vec3(&v, "anchorB"),
                json_vec3(&v, "axisA"),
                json_vec3(&v, "axisB"),
                json_f32(&v, "limitMin"),
                json_f32(&v, "limitMax"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_hinge_joint_3(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                ax_a,
                ax_b,
                lo,
                hi,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT3_PRISMATIC => {
            let (Some(a), Some(b), Some(ax), Some(lo), Some(hi), Some(comp)) = (
                json_vec3(&v, "anchorA"),
                json_vec3(&v, "anchorB"),
                json_vec3(&v, "axis"),
                json_f32(&v, "limitMin"),
                json_f32(&v, "limitMax"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_prismatic_joint_3(
                next_id,
                body_a as usize,
                body_b as usize,
                a,
                b,
                ax,
                lo,
                hi,
                comp,
                n,
            ) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        JOINT3_FIXED => {
            let (Some(a), Some(b), Some(comp)) = (
                json_vec3(&v, "anchorA"),
                json_vec3(&v, "anchorB"),
                json_compliance(&v),
            ) else {
                return -1;
            };
            match make_fixed_joint_3(next_id, body_a as usize, body_b as usize, a, b, comp, n) {
                Ok(j) => j,
                Err(_) => return -1,
            }
        }
        _ => return -1,
    };

    match world_ref.add_joint(joint) {
        Ok(id) => id as i32,
        Err(_) => -1,
    }
}

/// Remove a 3D joint by id. Returns 1 on success, 0 for null world or
/// unknown joint. Never panics.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which is a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_remove_joint(
    world: *mut phieng_dynamics::World3,
    joint_id: u32,
) -> i32 {
    if world.is_null() {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    if (joint_id as usize) >= world_ref.joints.len() {
        return 0;
    }
    world_ref.remove_joint(joint_id as usize);
    1
}

/// Number of 3D joints in the world, or -1 for null.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which yields `-1`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_joint_count(world: *mut phieng_dynamics::World3) -> i32 {
    if world.is_null() {
        return -1;
    }
    unsafe { (*world).joints.len() as i32 }
}

// ============================================================================
// Snapshot FFI (save/load, JSON + binary, exact + fresh)
// ============================================================================

/// Save format tags for the snapshot FFI (mirrored in C#).
pub const SNAPSHOT_JSON: u32 = 0;
/// Save format tags for the snapshot FFI (mirrored in C#).
pub const SNAPSHOT_BINARY: u32 = 1;
/// Exact resume: restores warm-start caches and joint impulses.
/// (Value matches [`phieng_dynamics::LoadMode::Exact`] positionally.)
pub const SNAPSHOT_EXACT: u32 = 0;
/// Fresh start: structure + config only, solver state starts cold.
pub const SNAPSHOT_FRESH: u32 = 1;

/// Save a 2D world to a file. Returns 1 on success, 0 for invalid input
/// (null world/path), -1 on IO failure, -2 on encode failure. Never panics.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `path_ptr` must be null (clean `0`) or point at
/// `path_len` readable UTF-8 bytes of a file path.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_save(
    world: *const World2,
    path_ptr: *const u8,
    path_len: usize,
    format: u32,
    mode: u32,
) -> i32 {
    use phieng_dynamics::{LoadMode, FORMAT_BINARY, FORMAT_JSON};
    use std::path::Path;

    if world.is_null() || path_ptr.is_null() || path_len == 0 || path_len > 4096 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(path_ptr, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return 0;
    };
    let mode = match mode {
        SNAPSHOT_EXACT => LoadMode::Exact,
        SNAPSHOT_FRESH => LoadMode::Fresh,
        _ => return 0,
    };
    let world_ref: &World2 = unsafe { &*world };
    let res = match format {
        FORMAT_JSON => {
            phieng_dynamics::world2_save_file(world_ref, Path::new(path), FORMAT_JSON, mode)
        }
        FORMAT_BINARY => {
            phieng_dynamics::world2_save_file(world_ref, Path::new(path), FORMAT_BINARY, mode)
        }
        _ => return 0,
    };
    match res {
        Ok(()) => 1,
        Err(phieng_dynamics::SnapshotError::Io(_)) => -1,
        Err(_) => -2,
    }
}

/// Load a 2D world from a file (format auto-detected). Returns a new
/// world pointer on success, null on any failure. The caller owns the
/// pointer ([`phieng_world_free`]). Never panics.
///
/// # Safety
/// `path_ptr` must be null (clean null return) or point at `path_len`
/// readable UTF-8 bytes of a file path.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_load(
    path_ptr: *const u8,
    path_len: usize,
    mode: u32,
) -> *mut World2 {
    use phieng_dynamics::LoadMode;
    use std::path::Path;

    if path_ptr.is_null() || path_len == 0 || path_len > 4096 {
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(path_ptr, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return std::ptr::null_mut();
    };
    let mode = match mode {
        SNAPSHOT_EXACT => LoadMode::Exact,
        SNAPSHOT_FRESH => LoadMode::Fresh,
        _ => return std::ptr::null_mut(),
    };
    match phieng_dynamics::world2_load_file(Path::new(path), mode) {
        Ok(w) => Box::into_raw(Box::new(w)),
        Err(_) => std::ptr::null_mut(),
    }
}

// ============================================================================
// 3D FFI
// ============================================================================

/// 3D Shape tags
pub const SHAPE3_SPHERE: u32 = 0;
pub const SHAPE3_BOX: u32 = 1;
pub const SHAPE3_CAPSULE: u32 = 2;

/// Create a new 3D physics world
#[no_mangle]
pub extern "C" fn phieng_world3_new() -> *mut phieng_dynamics::World3 {
    Box::into_raw(Box::new(phieng_dynamics::World3::new()))
}

/// Destroy a 3D world
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_free(world: *mut phieng_dynamics::World3) {
    if !world.is_null() {
        drop(Box::from_raw(world));
    }
}

/// Step 3D world
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_step(world: *mut phieng_dynamics::World3, dt: f32) {
    if world.is_null() || !dt.is_finite() || dt <= 0.0 {
        return;
    }
    (*world).step(dt.min(1.0 / 15.0));
}

/// Set 3D gravity
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_gravity(
    world: *mut phieng_dynamics::World3,
    gx: f32,
    gy: f32,
    gz: f32,
) {
    if world.is_null() || !gx.is_finite() || !gy.is_finite() || !gz.is_finite() {
        return;
    }
    (*world).gravity = glam::Vec3::new(gx, gy, gz);
}

/// Get 3D body count
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_body_count(world: *mut phieng_dynamics::World3) -> i32 {
    if world.is_null() {
        return -1;
    }
    (*world).bodies.len() as i32
}

/// Look up a 3D body by its public id. Returns None for stale/foreign ids.
fn body3_by_id(world: &phieng_dynamics::World3, body_id: u32) -> Option<usize> {
    let index = body_id as usize;
    world
        .bodies
        .get(index)
        .filter(|b| b.id == index)
        .map(|_| index)
}

/// Remove a 3D body by id. Returns 1 on success, 0 for null world or
/// unknown body. Remaining bodies are reindexed.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_remove_body(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
) -> i32 {
    if world.is_null() {
        return 0;
    }
    if body3_by_id(&*world, body_id).is_none() {
        return 0;
    }
    (*world).remove_body(body_id as usize);
    1
}

/// Replace a 3D body's material (friction, density, restitution),
/// recomputing mass/inertia from its current shape. Returns 1 on
/// success, 0 for null world or unknown body.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_body_material(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    friction: f32,
    density: f32,
    restitution: f32,
) -> i32 {
    if world.is_null() {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    // Input is pre-sanitized by PhysicsMaterial::new, so this is infallible
    // in practice; the error is unreachable but mapped explicitly.
    let _ = world_ref.bodies[index].set_material(phieng_dynamics::PhysicsMaterial::new(
        friction,
        restitution,
        density,
    ));
    1
}

/// Copy a 3D body's material out: `out_mat[0..3]` = (friction, density,
/// restitution). Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_mat` must be either null (clean `0`) or valid for writes of
/// 3 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_material(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    out_mat: *mut f32,
) -> i32 {
    if world.is_null() || out_mat.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    let material = world_ref.bodies[index].material;
    *out_mat.add(0) = material.friction;
    *out_mat.add(1) = material.density;
    *out_mat.add(2) = material.restitution;
    1
}

/// Add 3D SPH fluid particles from interleaved `positions` (`count`
/// triples `[x0, y0, z0, ...]`), starting at zero velocity. Returns the
/// number added, or 0 on null world/positions or any non-finite input
/// (atomic: nothing is added).
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which yields a clean `0`). `positions` must be null (clean `0`) or
/// valid for reads of `3 * count` `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_fluid_add(
    world: *mut phieng_dynamics::World3,
    positions: *const f32,
    count: u32,
) -> u32 {
    if world.is_null() || positions.is_null() {
        return 0;
    }
    let n = (count as usize).min(1_000_000);
    let src = unsafe { std::slice::from_raw_parts(positions, n.saturating_mul(3)) };
    if src.iter().any(|v| !v.is_finite()) {
        return 0;
    }
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks
    // (`src` is exactly `3 * n` long by construction above).
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let pts: Vec<Vec3> = src
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();
    unsafe { &mut *world }.fluid.add_particles(&pts) as u32
}

/// 3D fluid particle count. Returns 0 on null world (which is also the
/// empty-fluid answer).
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which yields a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_fluid_count(world: *const phieng_dynamics::World3) -> u32 {
    if world.is_null() {
        return 0;
    }
    unsafe { &*world }.fluid.particles.len() as u32
}

/// Remove all 3D fluid particles (keeps allocations). Returns 1 on
/// success, 0 on null world.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which yields a clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_fluid_clear(world: *mut phieng_dynamics::World3) -> i32 {
    if world.is_null() {
        return 0;
    }
    unsafe { &mut *world }.fluid.clear();
    1
}

/// Copy 3D particle state out interleaved (`max` sextuples
/// `[x, y, z, vx, vy, vz]`). Returns the number of particles written,
/// 0 on null world/out.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which yields a clean `0`). `out` must be null (clean `0`) or valid
/// for writes of `6 * max` `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_fluid_get(
    world: *const phieng_dynamics::World3,
    out: *mut f32,
    max: u32,
) -> u32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let n = (max as usize).min(world_ref.fluid.particles.len());
    for (i, p) in world_ref.fluid.particles.iter().take(n).enumerate() {
        unsafe {
            *out.add(i * 6) = p.position.x;
            *out.add(i * 6 + 1) = p.position.y;
            *out.add(i * 6 + 2) = p.position.z;
            *out.add(i * 6 + 3) = p.velocity.x;
            *out.add(i * 6 + 4) = p.velocity.y;
            *out.add(i * 6 + 5) = p.velocity.z;
        }
    }
    n as u32
}

/// Add a 3D body
#[no_mangle]
#[allow(clippy::too_many_arguments)]
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
pub unsafe extern "C" fn phieng_world3_add_body(
    world: *mut phieng_dynamics::World3,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    shape_tag: u32,
    a: f32,
    b: f32,
    c: f32,
    density: f32,
    is_static: u8,
    friction: f32,
    restitution: f32,
) -> i32 {
    if world.is_null() || !pos_x.is_finite() || !pos_y.is_finite() || !pos_z.is_finite() {
        return -1;
    }

    let shape = match shape_tag {
        SHAPE3_SPHERE if a > 0.0 && a.is_finite() => phieng_dynamics::Shape3::Sphere { radius: a },
        SHAPE3_BOX
            if a > 0.0 && b > 0.0 && c > 0.0 && a.is_finite() && b.is_finite() && c.is_finite() =>
        {
            phieng_dynamics::Shape3::Box {
                half_extents: glam::Vec3::new(a, b, c),
            }
        }
        SHAPE3_CAPSULE if a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() => {
            phieng_dynamics::Shape3::Capsule {
                radius: a,
                half_length: b,
            }
        }
        _ => return -1,
    };

    if !density.is_finite() || density <= 0.0 {
        return -1;
    }

    let world = &mut *world;
    let position = glam::Vec3::new(pos_x, pos_y, pos_z);
    let material = phieng_dynamics::PhysicsMaterial::new(friction, restitution, density);

    if is_static != 0 {
        let mut body = phieng_dynamics::RigidBody3::static_body(world.next_body_id);
        body.transform.position = position;
        if body.set_shape(shape, density).is_err() || body.set_material(material).is_err() {
            return -1;
        }
        world.add_body(body).map(|id| id as i32).unwrap_or(-1)
    } else {
        let mut body = phieng_dynamics::RigidBody3::new(world.next_body_id);
        body.transform.position = position;
        if body.set_shape(shape, density).is_err() || body.set_material(material).is_err() {
            return -1;
        }
        world.add_body(body).map(|id| id as i32).unwrap_or(-1)
    }
}

/// Get 3D body transform
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_pos` must be either null (clean `0`) or valid for writes of
/// 3 `f32`s; `out_quat` either null or valid for 4 `f32`s. Both must be
/// properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_transform(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    out_pos: *mut f32,
    out_quat: *mut f32,
) -> i32 {
    if world.is_null() || out_pos.is_null() || out_quat.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = &*world;
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    let body = &world_ref.bodies[index];
    *out_pos.add(0) = body.transform.position.x;
    *out_pos.add(1) = body.transform.position.y;
    *out_pos.add(2) = body.transform.position.z;
    *out_quat.add(0) = body.transform.rotation.x;
    *out_quat.add(1) = body.transform.rotation.y;
    *out_quat.add(2) = body.transform.rotation.z;
    *out_quat.add(3) = body.transform.rotation.w;
    1
}

/// Get 3D body velocity
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// which the function documents as a clean no-op / error code).
/// Use-after-free and forged pointers are undefined behavior.
/// `out_vel` must be either null (clean `0`) or valid for writes of
/// 3 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_velocity(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    out_vel: *mut f32,
) -> i32 {
    if world.is_null() || out_vel.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = &*world;
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    let body = &world_ref.bodies[index];
    *out_vel.add(0) = body.velocity.x;
    *out_vel.add(1) = body.velocity.y;
    *out_vel.add(2) = body.velocity.z;
    1
}

// ============================================================================
// 3D snapshot FFI (save/load, JSON + binary, exact + fresh)
// ============================================================================

/// Save a 3D world to a file. Returns 1 on success, 0 for invalid input,
/// -1 on IO failure, -2 on encode failure. Never panics.
///
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `path_ptr` must be null (clean `0`) or point at
/// `path_len` readable UTF-8 bytes of a file path.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_save(
    world: *const phieng_dynamics::World3,
    path_ptr: *const u8,
    path_len: usize,
    format: u32,
    mode: u32,
) -> i32 {
    use phieng_dynamics::{LoadMode, FORMAT_BINARY, FORMAT_JSON};
    use std::path::Path;

    if world.is_null() || path_ptr.is_null() || path_len == 0 || path_len > 4096 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(path_ptr, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return 0;
    };
    let mode = match mode {
        SNAPSHOT_EXACT => LoadMode::Exact,
        SNAPSHOT_FRESH => LoadMode::Fresh,
        _ => return 0,
    };
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let res = match format {
        FORMAT_JSON => {
            phieng_dynamics::world3_save_file(world_ref, Path::new(path), FORMAT_JSON, mode)
        }
        FORMAT_BINARY => {
            phieng_dynamics::world3_save_file(world_ref, Path::new(path), FORMAT_BINARY, mode)
        }
        _ => return 0,
    };
    match res {
        Ok(()) => 1,
        Err(phieng_dynamics::SnapshotError::Io(_)) => -1,
        Err(_) => -2,
    }
}

/// Load a 3D world from a file (format auto-detected). Returns a new
/// world pointer on success, null on any failure. The caller owns the
/// pointer ([`phieng_world3_free`]). Never panics.
///
/// # Safety
/// `path_ptr` must be null (clean null return) or point at `path_len`
/// readable UTF-8 bytes of a file path.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_load(
    path_ptr: *const u8,
    path_len: usize,
    mode: u32,
) -> *mut phieng_dynamics::World3 {
    use phieng_dynamics::LoadMode;
    use std::path::Path;

    if path_ptr.is_null() || path_len == 0 || path_len > 4096 {
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(path_ptr, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return std::ptr::null_mut();
    };
    let mode = match mode {
        SNAPSHOT_EXACT => LoadMode::Exact,
        SNAPSHOT_FRESH => LoadMode::Fresh,
        _ => return std::ptr::null_mut(),
    };
    match phieng_dynamics::world3_load_file(Path::new(path), mode) {
        Ok(w) => Box::into_raw(Box::new(w)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball_drop() -> (*mut World2, u32) {
        let world = phieng_world_new();
        unsafe {
            assert_eq!(phieng_world_body_count(world), 0);
            let floor =
                phieng_world_add_body(world, 0.0, -1.0, SHAPE_BOX, 10.0, 0.5, 1.0, 1, 0.5, 0.0);
            assert_eq!(floor, 0);
            let ball =
                phieng_world_add_body(world, 0.0, 5.0, SHAPE_CIRCLE, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
            assert_eq!(ball, 1);
            (world, ball as u32)
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        unsafe {
            phieng_world_free(std::ptr::null_mut()); // no-op, must not crash
            phieng_world_step(std::ptr::null_mut(), 1.0 / 60.0);
            phieng_world_set_gravity(std::ptr::null_mut(), 0.0, -9.81);
            assert_eq!(phieng_world_body_count(std::ptr::null_mut()), -1);
            assert_eq!(
                phieng_world_add_body(
                    std::ptr::null_mut(),
                    0.0,
                    0.0,
                    SHAPE_CIRCLE,
                    1.0,
                    0.0,
                    1.0,
                    0,
                    0.5,
                    0.0
                ),
                -1
            );
            let mut pos = [0.0f32; 2];
            let mut rot = 0.0f32;
            assert_eq!(
                phieng_world_get_body_transform(
                    std::ptr::null_mut(),
                    0,
                    pos.as_mut_ptr(),
                    &mut rot
                ),
                0
            );
            // Null out-pointers on a live world must also fail cleanly.
            let (world, _) = ball_drop();
            assert_eq!(
                phieng_world_get_body_transform(world, 0, std::ptr::null_mut(), &mut rot),
                0
            );
            assert_eq!(
                phieng_world_get_body_transform(world, 0, pos.as_mut_ptr(), std::ptr::null_mut()),
                0
            );
            // Unknown body id.
            assert_eq!(
                phieng_world_get_body_transform(world, 999, pos.as_mut_ptr(), &mut rot),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn test_ffi_rejects_bad_shapes() {
        unsafe {
            let world = phieng_world_new();
            // Unknown tag, negative radius, zero extents, bad density.
            assert_eq!(
                phieng_world_add_body(world, 0.0, 0.0, 99, 1.0, 1.0, 1.0, 0, 0.5, 0.0),
                -1
            );
            assert_eq!(
                phieng_world_add_body(world, 0.0, 0.0, SHAPE_CIRCLE, -1.0, 0.0, 1.0, 0, 0.5, 0.0),
                -1
            );
            assert_eq!(
                phieng_world_add_body(world, 0.0, 0.0, SHAPE_BOX, 1.0, 0.0, 1.0, 0, 0.5, 0.0),
                -1
            );
            assert_eq!(
                phieng_world_add_body(world, 0.0, 0.0, SHAPE_CIRCLE, 1.0, 0.0, 0.0, 0, 0.5, 0.0),
                -1
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn test_ffi_ball_drop_roundtrip() {
        unsafe {
            let (world, ball) = ball_drop();
            let dt = 1.0 / 60.0;
            for _ in 0..120 {
                phieng_world_step(world, dt);
            }
            let mut pos = [0.0f32; 2];
            let mut rot = 0.0f32;
            let mut vel = [0.0f32; 2];
            assert_eq!(
                phieng_world_get_body_transform(world, ball, pos.as_mut_ptr(), &mut rot),
                1
            );
            assert_eq!(
                phieng_world_get_body_velocity(world, ball, vel.as_mut_ptr()),
                1
            );
            assert!(
                pos[1] < 5.0,
                "ball should have fallen via FFI stepping, y={}",
                pos[1]
            );
            assert_eq!(phieng_world_body_count(world), 2);
            phieng_world_free(world);
        }
    }

    #[test]
    fn test_ffi_remove_body() {
        unsafe {
            let (world, ball) = ball_drop();
            assert_eq!(phieng_world_body_count(world), 2);
            // Unknown id and null world are rejected.
            assert_eq!(phieng_world_remove_body(world, 999), 0);
            assert_eq!(phieng_world_remove_body(std::ptr::null_mut(), ball), 0);
            // Removing the floor reindexes the ball from 1 to 0.
            assert_eq!(phieng_world_remove_body(world, 0), 1);
            assert_eq!(phieng_world_body_count(world), 1);
            let mut pos = [0.0f32; 2];
            let mut rot = 0.0f32;
            assert_eq!(
                phieng_world_get_body_transform(world, 0, pos.as_mut_ptr(), &mut rot),
                1
            );
            assert_eq!(
                phieng_world_get_body_transform(world, 1, pos.as_mut_ptr(), &mut rot),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn test_ffi3_null_safety() {
        unsafe {
            use phieng_dynamics::World3;
            phieng_world3_free(std::ptr::null_mut());
            phieng_world3_step(std::ptr::null_mut(), 1.0 / 60.0);
            phieng_world3_set_gravity(std::ptr::null_mut(), 0.0, -9.81, 0.0);
            assert_eq!(phieng_world3_body_count(std::ptr::null_mut()), -1);
            assert_eq!(
                phieng_world3_add_body(
                    std::ptr::null_mut(),
                    0.0,
                    0.0,
                    0.0,
                    SHAPE3_SPHERE,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0,
                    0.5,
                    0.1,
                ),
                -1
            );
            assert_eq!(phieng_world3_remove_body(std::ptr::null_mut(), 0), 0);
            assert_eq!(
                phieng_world3_set_body_material(std::ptr::null_mut(), 0, 0.5, 1.0, 0.1),
                0
            );
            let mut mat = [0.0f32; 3];
            assert_eq!(
                phieng_world3_get_body_material(std::ptr::null_mut(), 0, mat.as_mut_ptr()),
                0
            );
            let mut pos = [0.0f32; 3];
            let mut quat = [0.0f32; 4];
            assert_eq!(
                phieng_world3_get_body_transform(
                    std::ptr::null_mut(),
                    0,
                    pos.as_mut_ptr(),
                    quat.as_mut_ptr()
                ),
                0
            );
            let world: *mut World3 = phieng_world3_new();
            assert!(!world.is_null());
            // Bad shapes and unknown ids fail cleanly on a live world.
            assert_eq!(
                phieng_world3_add_body(world, 0.0, 0.0, 0.0, 99, 1.0, 1.0, 1.0, 1.0, 0, 0.5, 0.1,),
                -1
            );
            assert_eq!(
                phieng_world3_add_body(
                    world,
                    0.0,
                    0.0,
                    0.0,
                    SHAPE3_SPHERE,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    0,
                    0.5,
                    0.1,
                ),
                -1
            );
            // Capsule tag validates radius/half-length the same way.
            assert_eq!(
                phieng_world3_add_body(
                    world,
                    0.0,
                    0.0,
                    0.0,
                    SHAPE3_CAPSULE,
                    0.5,
                    1.0,
                    0.0,
                    1.0,
                    0,
                    0.5,
                    0.1,
                ),
                0
            );
            assert_eq!(
                phieng_world3_add_body(
                    world,
                    0.0,
                    0.0,
                    0.0,
                    SHAPE3_CAPSULE,
                    0.5,
                    -1.0,
                    0.0,
                    1.0,
                    0,
                    0.5,
                    0.1,
                ),
                -1
            );
            assert_eq!(
                phieng_world3_get_body_transform(world, 999, pos.as_mut_ptr(), quat.as_mut_ptr()),
                0
            );
            assert_eq!(phieng_world3_remove_body(world, 999), 0);
            assert_eq!(
                phieng_world3_get_body_material(world, 999, mat.as_mut_ptr()),
                0
            );
            phieng_world3_free(world);
        }
    }

    #[test]
    fn test_ffi3_ball_drop_and_material_roundtrip() {
        unsafe {
            use phieng_dynamics::World3;
            let world: *mut World3 = phieng_world3_new();
            let floor = phieng_world3_add_body(
                world, 0.0, -1.0, 0.0, SHAPE3_BOX, 10.0, 0.5, 10.0, 1.0, 1, 0.5, 0.0,
            );
            assert_eq!(floor, 0);
            let ball = phieng_world3_add_body(
                world,
                0.0,
                5.0,
                0.0,
                SHAPE3_SPHERE,
                0.5,
                0.0,
                0.0,
                1.0,
                0,
                0.5,
                0.0,
            );
            assert_eq!(ball, 1);
            assert_eq!(phieng_world3_body_count(world), 2);

            let mut mat = [0.0f32; 3];
            assert_eq!(
                phieng_world3_get_body_material(world, ball as u32, mat.as_mut_ptr()),
                1
            );
            assert!((mat[0] - 0.5).abs() < 1e-6, "friction={}", mat[0]);
            assert!((mat[1] - 1.0).abs() < 1e-6, "density={}", mat[1]);
            assert!((mat[2] - 0.0).abs() < 1e-6, "restitution={}", mat[2]);
            assert_eq!(
                phieng_world3_set_body_material(world, ball as u32, 0.9, 2.0, 0.8),
                1
            );
            assert_eq!(
                phieng_world3_get_body_material(world, ball as u32, mat.as_mut_ptr()),
                1
            );
            assert!((mat[0] - 0.9).abs() < 1e-6);
            assert!((mat[1] - 2.0).abs() < 1e-6);
            assert!((mat[2] - 0.8).abs() < 1e-6);

            let dt = 1.0 / 60.0;
            for _ in 0..120 {
                phieng_world3_step(world, dt);
            }
            let mut pos = [0.0f32; 3];
            let mut quat = [0.0f32; 4];
            let mut vel = [0.0f32; 3];
            assert_eq!(
                phieng_world3_get_body_transform(
                    world,
                    ball as u32,
                    pos.as_mut_ptr(),
                    quat.as_mut_ptr()
                ),
                1
            );
            assert_eq!(
                phieng_world3_get_body_velocity(world, ball as u32, vel.as_mut_ptr()),
                1
            );
            assert!(pos[1] < 5.0, "ball should have fallen, y={}", pos[1]);
            // Unit quaternion comes back intact.
            let qlen =
                (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
                    .sqrt();
            assert!((qlen - 1.0).abs() < 1e-4, "quat len={qlen}");

            // Removal reindexes the survivors.
            assert_eq!(phieng_world3_remove_body(world, 0), 1);
            assert_eq!(phieng_world3_body_count(world), 1);
            assert_eq!(
                phieng_world3_get_body_transform(world, 0, pos.as_mut_ptr(), quat.as_mut_ptr()),
                1
            );
            assert_eq!(
                phieng_world3_get_body_transform(world, 1, pos.as_mut_ptr(), quat.as_mut_ptr()),
                0
            );
            phieng_world3_free(world);
        }
    }

    #[test]
    fn test_ffi_body_material_roundtrip() {
        unsafe {
            let (world, ball) = ball_drop();

            // Creation-time material comes back intact.
            let mut mat = [0.0f32; 3];
            assert_eq!(
                phieng_world_get_body_material(world, ball, mat.as_mut_ptr()),
                1
            );
            assert!((mat[0] - 0.5).abs() < 1e-6, "friction={}", mat[0]);
            assert!((mat[1] - 1.0).abs() < 1e-6, "density={}", mat[1]);
            assert!((mat[2] - 0.0).abs() < 1e-6, "restitution={}", mat[2]);

            // Modification round-trips and rejects unknown ids.
            assert_eq!(
                phieng_world_set_body_material(world, ball, 0.9, 2.0, 0.8),
                1
            );
            assert_eq!(
                phieng_world_get_body_material(world, ball, mat.as_mut_ptr()),
                1
            );
            assert!((mat[0] - 0.9).abs() < 1e-6);
            assert!((mat[1] - 2.0).abs() < 1e-6);
            assert!((mat[2] - 0.8).abs() < 1e-6);
            assert_eq!(phieng_world_set_body_material(world, 999, 0.9, 2.0, 0.8), 0);
            assert_eq!(
                phieng_world_get_body_material(world, 999, mat.as_mut_ptr()),
                0
            );
            assert_eq!(
                phieng_world_set_body_material(std::ptr::null_mut(), ball, 0.9, 2.0, 0.8),
                0
            );
            phieng_world_free(world);
        }
    }

    fn add_joint_ffi(world: *mut World2, tag: u32, a: u32, b: u32, json: &str) -> i32 {
        unsafe { phieng_world_add_joint(world, tag, a, b, json.as_ptr(), json.len()) }
    }

    #[test]
    fn f31_joint_null_safety() {
        unsafe {
            // Null world and null/empty/oversize blobs are clean -1.
            assert_eq!(
                phieng_world_add_joint(
                    std::ptr::null_mut(),
                    JOINT_DISTANCE,
                    0,
                    1,
                    b"{}".as_ptr(),
                    2,
                ),
                -1
            );
            let world = phieng_world_new();
            assert_eq!(phieng_world_joint_count(std::ptr::null_mut()), -1);
            assert_eq!(phieng_world_joint_count(world), 0);
            assert_eq!(
                phieng_world_add_joint(world, JOINT_DISTANCE, 0, 1, std::ptr::null(), 10),
                -1
            );
            assert_eq!(
                phieng_world_add_joint(world, JOINT_DISTANCE, 0, 1, b"{}".as_ptr(), 0),
                -1
            );
            assert_eq!(
                phieng_world_add_joint(
                    world,
                    JOINT_DISTANCE,
                    0,
                    1,
                    b"{}".as_ptr(),
                    MAX_JOINT_JSON_LEN + 1,
                ),
                -1
            );
            assert_eq!(phieng_world_remove_joint(std::ptr::null_mut(), 0), 0);
            assert_eq!(phieng_world_remove_joint(world, 999), 0);
            phieng_world_free(world);
        }
    }

    #[test]
    fn f32_joint_rejects_bad_input() {
        unsafe {
            let (world, ball) = ball_drop();
            // Unknown tag.
            assert_eq!(
                add_joint_ffi(
                    world,
                    99,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Malformed JSON.
            assert_eq!(add_joint_ffi(world, JOINT_DISTANCE, 0, ball, "{nope"), -1);
            // Non-UTF8 bytes.
            assert_eq!(
                phieng_world_add_joint(world, JOINT_DISTANCE, 0, ball, b"\xff\xfe".as_ptr(), 2),
                -1
            );
            // Missing required field.
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0]}"#
                ),
                -1
            );
            // Wrong array width.
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Stale body id.
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    0,
                    999,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Same body.
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    ball,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Negative rest length rejected by validation.
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"restLength":-1.0}"#
                ),
                -1
            );
            // Non-zero compliance rejected (rigid-only v1).
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"restLength":1.0,"compliance":0.5}"#
                ),
                -1
            );
            // Hinge NaN limit rejected (serde_json rejects NaN outright).
            assert_eq!(
                add_joint_ffi(
                    world,
                    JOINT_HINGE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0],"limitMin":0.0,"limitMax":NaN}"#
                ),
                -1
            );
            assert_eq!(phieng_world_joint_count(world), 0);
            phieng_world_free(world);
        }
    }

    fn save_joint_scene() -> (*mut World2, u32) {
        let (world, ball) = ball_drop();
        let json =
            r#"{"anchorA":[0.0,-1.0],"anchorB":[0.0,0.0],"restLength":6.0,"compliance":0.0}"#;
        let j = unsafe {
            phieng_world_add_joint(world, JOINT_DISTANCE, 0, ball, json.as_ptr(), json.len())
        };
        assert_eq!(j, 0);
        unsafe {
            let dt = 1.0 / 60.0;
            for _ in 0..60 {
                phieng_world_step(world, dt);
            }
        }
        (world, ball)
    }

    fn ball_transform(world: *mut World2, id: u32) -> (f32, f32) {
        let mut pos = [0.0f32; 2];
        let mut rot = 0.0f32;
        unsafe {
            assert_eq!(
                phieng_world_get_body_transform(world, id, pos.as_mut_ptr(), &mut rot),
                1
            );
        }
        (pos[0], pos[1])
    }

    #[test]
    fn f34_save_load_roundtrip() {
        let (world, ball) = save_joint_scene();
        let dir = std::env::temp_dir();
        let pb = dir.join("phieng_f34.bin");
        let pj = dir.join("phieng_f34.json");
        let ps = |p: &std::path::Path| {
            let s = p.to_str().expect("temp path is utf-8");
            (s.as_ptr(), s.len())
        };
        unsafe {
            let (ptr, len) = ps(&pb);
            assert_eq!(
                phieng_world_save(world, ptr, len, SNAPSHOT_BINARY, SNAPSHOT_EXACT),
                1
            );
            let (ptr, len) = ps(&pj);
            assert_eq!(
                phieng_world_save(world, ptr, len, SNAPSHOT_JSON, SNAPSHOT_EXACT),
                1
            );

            let (ptr, len) = ps(&pb);
            let rb = phieng_world_load(ptr, len, SNAPSHOT_EXACT);
            assert!(!rb.is_null());
            let (ptr, len) = ps(&pj);
            let rj = phieng_world_load(ptr, len, SNAPSHOT_EXACT);
            assert!(!rj.is_null());

            assert_eq!(phieng_world_body_count(rb), phieng_world_body_count(world));
            assert_eq!(
                phieng_world_joint_count(rb),
                phieng_world_joint_count(world)
            );
            let dt = 1.0 / 60.0;
            for _ in 0..60 {
                phieng_world_step(world, dt);
                phieng_world_step(rb, dt);
                phieng_world_step(rj, dt);
            }
            assert_eq!(ball_transform(world, ball), ball_transform(rb, ball));
            assert_eq!(ball_transform(world, ball), ball_transform(rj, ball));

            phieng_world_free(world);
            phieng_world_free(rb);
            phieng_world_free(rj);
        }
        let _ = std::fs::remove_file(pb);
        let _ = std::fs::remove_file(pj);
    }

    #[test]
    fn f35_save_load_rejects() {
        unsafe {
            // Null world / bad path / bad codes.
            assert_eq!(
                phieng_world_save(
                    std::ptr::null(),
                    b"/x".as_ptr(),
                    2,
                    SNAPSHOT_BINARY,
                    SNAPSHOT_EXACT
                ),
                0
            );
            let (world, _) = ball_drop();
            assert_eq!(
                phieng_world_save(world, std::ptr::null(), 10, SNAPSHOT_BINARY, SNAPSHOT_EXACT),
                0
            );
            assert_eq!(
                phieng_world_save(world, b"/x".as_ptr(), 2, 99, SNAPSHOT_EXACT),
                0
            );
            // Missing file loads null.
            let missing = std::env::temp_dir().join("phieng_nope_missing.bin");
            let s = missing.to_str().expect("temp path is utf-8");
            assert!(phieng_world_load(s.as_ptr(), s.len(), SNAPSHOT_EXACT).is_null());
            // Junk file loads null.
            let junk = std::env::temp_dir().join("phieng_f35.bin");
            std::fs::write(&junk, [1u8, 2, 3, 4]).expect("write temp");
            let s = junk.to_str().expect("temp path is utf-8");
            assert!(phieng_world_load(s.as_ptr(), s.len(), SNAPSHOT_EXACT).is_null());
            let _ = std::fs::remove_file(junk);
            phieng_world_free(world);
        }
    }

    #[test]
    fn f33_joint_roundtrip_removal() {
        unsafe {
            let (world, ball) = ball_drop();
            // Distance rod floor(0) -> ball(1), rest length 6 (ball starts at y=5).
            let j = add_joint_ffi(
                world,
                JOINT_DISTANCE,
                0,
                ball,
                r#"{"anchorA":[0.0,-1.0],"anchorB":[0.0,0.0],"restLength":6.0,"compliance":0.0}"#,
            );
            assert_eq!(j, 0);
            assert_eq!(phieng_world_joint_count(world), 1);
            // Hinge with free limits (min > max).
            let h = add_joint_ffi(
                world,
                JOINT_HINGE,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0],"anchorB":[0.0,0.0],"limitMin":1.0,"limitMax":-1.0}"#,
            );
            assert_eq!(h, 1);
            // Prismatic and fixed.
            let p = add_joint_ffi(
                world,
                JOINT_PRISMATIC,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0],"anchorB":[0.0,0.0],"axis":[1.0,0.0],"limitMin":-1.0,"limitMax":1.0}"#,
            );
            assert_eq!(p, 2);
            let f = add_joint_ffi(
                world,
                JOINT_FIXED,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0],"anchorB":[0.0,0.0]}"#,
            );
            assert_eq!(f, 3);
            assert_eq!(phieng_world_joint_count(world), 4);

            // Step with joints active: ball must stay finite and held.
            let dt = 1.0 / 60.0;
            for _ in 0..120 {
                phieng_world_step(world, dt);
            }
            let mut pos = [0.0f32; 2];
            let mut rot = 0.0f32;
            assert_eq!(
                phieng_world_get_body_transform(world, ball, pos.as_mut_ptr(), &mut rot),
                1
            );
            assert!(pos[0].is_finite() && pos[1].is_finite());

            // Removal drops the joint; unknown ids are clean 0s.
            assert_eq!(phieng_world_remove_joint(world, 1), 1);
            assert_eq!(phieng_world_joint_count(world), 3);
            assert_eq!(phieng_world_remove_joint(world, 1), 1);
            assert_eq!(phieng_world_joint_count(world), 2);
            assert_eq!(phieng_world_remove_joint(world, 999), 0);
            // Removing the ball drops its remaining joints (reindexed).
            assert_eq!(phieng_world_remove_body(world, ball), 1);
            assert_eq!(phieng_world_joint_count(world), 0);
            phieng_world_free(world);
        }
    }

    type World3Ptr = *mut phieng_dynamics::World3;

    fn ball_drop_3() -> (World3Ptr, u32) {
        let world = phieng_world3_new();
        unsafe {
            assert_eq!(phieng_world3_body_count(world), 0);
            let floor = phieng_world3_add_body(
                world, 0.0, -1.0, 0.0, SHAPE3_BOX, 10.0, 0.5, 10.0, 1.0, 1, 0.5, 0.0,
            );
            assert_eq!(floor, 0);
            let ball = phieng_world3_add_body(
                world,
                0.0,
                5.0,
                0.0,
                SHAPE3_SPHERE,
                0.5,
                0.0,
                0.0,
                1.0,
                0,
                0.5,
                0.0,
            );
            assert_eq!(ball, 1);
            (world, ball as u32)
        }
    }

    fn add_joint3_ffi(world: World3Ptr, tag: u32, a: u32, b: u32, json: &str) -> i32 {
        unsafe { phieng_world3_add_joint(world, tag, a, b, json.as_ptr(), json.len()) }
    }

    #[test]
    fn f37_joint3_null_safety() {
        unsafe {
            // Null world and null/empty/oversize blobs are clean -1.
            assert_eq!(
                phieng_world3_add_joint(
                    std::ptr::null_mut(),
                    JOINT3_DISTANCE,
                    0,
                    1,
                    b"{}".as_ptr(),
                    2,
                ),
                -1
            );
            let world = phieng_world3_new();
            assert_eq!(phieng_world3_joint_count(std::ptr::null_mut()), -1);
            assert_eq!(phieng_world3_joint_count(world), 0);
            assert_eq!(
                phieng_world3_add_joint(world, JOINT3_DISTANCE, 0, 1, std::ptr::null(), 10),
                -1
            );
            assert_eq!(
                phieng_world3_add_joint(world, JOINT3_DISTANCE, 0, 1, b"{}".as_ptr(), 0),
                -1
            );
            assert_eq!(
                phieng_world3_add_joint(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    1,
                    b"{}".as_ptr(),
                    MAX_JOINT_JSON_LEN + 1,
                ),
                -1
            );
            assert_eq!(phieng_world3_remove_joint(std::ptr::null_mut(), 0), 0);
            assert_eq!(phieng_world3_remove_joint(world, 999), 0);
            phieng_world3_free(world);
        }
    }

    #[test]
    fn f38_joint3_rejects_bad_input() {
        unsafe {
            let (world, ball) = ball_drop_3();
            // Unknown tag.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    99,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Malformed JSON.
            assert_eq!(add_joint3_ffi(world, JOINT3_DISTANCE, 0, ball, "{nope"), -1);
            // Non-UTF8 bytes.
            assert_eq!(
                phieng_world3_add_joint(world, JOINT3_DISTANCE, 0, ball, b"\xff\xfe".as_ptr(), 2),
                -1
            );
            // Missing required field.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0]}"#
                ),
                -1
            );
            // Wrong array width (2D-sized anchors rejected).
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0],"anchorB":[0,0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Stale body id.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    999,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Same body.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    ball,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"restLength":1.0}"#
                ),
                -1
            );
            // Negative rest length rejected by validation.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"restLength":-1.0}"#
                ),
                -1
            );
            // Non-zero compliance rejected (rigid-only v1).
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_DISTANCE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"restLength":1.0,"compliance":0.5}"#
                ),
                -1
            );
            // Zero hinge axis rejected.
            assert_eq!(
                add_joint3_ffi(
                    world,
                    JOINT3_HINGE,
                    0,
                    ball,
                    r#"{"anchorA":[0,0,0],"anchorB":[0,0,0],"axisA":[0,0,0],"axisB":[0,0,1],"limitMin":1.0,"limitMax":-1.0}"#
                ),
                -1
            );
            assert_eq!(phieng_world3_joint_count(world), 0);
            phieng_world3_free(world);
        }
    }

    #[test]
    fn f39_joint3_roundtrip_removal() {
        unsafe {
            let (world, ball) = ball_drop_3();
            // Distance rod floor(0) -> ball(1), rest length 6 (ball at y=5).
            let j = add_joint3_ffi(
                world,
                JOINT3_DISTANCE,
                0,
                ball,
                r#"{"anchorA":[0.0,-1.0,0.0],"anchorB":[0.0,0.0,0.0],"restLength":6.0,"compliance":0.0}"#,
            );
            assert_eq!(j, 0);
            assert_eq!(phieng_world3_joint_count(world), 1);
            // Hinge with free twist (min > max).
            let h = add_joint3_ffi(
                world,
                JOINT3_HINGE,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0,0.0],"anchorB":[0.0,0.0,0.0],"axisA":[0.0,0.0,1.0],"axisB":[0.0,0.0,1.0],"limitMin":1.0,"limitMax":-1.0}"#,
            );
            assert_eq!(h, 1);
            // Prismatic and fixed.
            let p = add_joint3_ffi(
                world,
                JOINT3_PRISMATIC,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0,0.0],"anchorB":[0.0,0.0,0.0],"axis":[1.0,0.0,0.0],"limitMin":-1.0,"limitMax":1.0}"#,
            );
            assert_eq!(p, 2);
            let f = add_joint3_ffi(
                world,
                JOINT3_FIXED,
                0,
                ball,
                r#"{"anchorA":[0.0,0.0,0.0],"anchorB":[0.0,0.0,0.0]}"#,
            );
            assert_eq!(f, 3);
            assert_eq!(phieng_world3_joint_count(world), 4);

            // Step with joints active: ball must stay finite and held.
            let dt = 1.0 / 60.0;
            for _ in 0..120 {
                phieng_world3_step(world, dt);
            }
            let mut pos = [0.0f32; 3];
            let mut quat = [0.0f32; 4];
            assert_eq!(
                phieng_world3_get_body_transform(world, ball, pos.as_mut_ptr(), quat.as_mut_ptr()),
                1
            );
            assert!(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite());

            // Removal drops the joint; unknown ids are clean 0s.
            assert_eq!(phieng_world3_remove_joint(world, 1), 1);
            assert_eq!(phieng_world3_joint_count(world), 3);
            assert_eq!(phieng_world3_remove_joint(world, 1), 1);
            assert_eq!(phieng_world3_joint_count(world), 2);
            assert_eq!(phieng_world3_remove_joint(world, 999), 0);
            // Removing the ball drops its remaining joints.
            assert_eq!(phieng_world3_remove_body(world, ball), 1);
            assert_eq!(phieng_world3_joint_count(world), 0);
            phieng_world3_free(world);
        }
    }

    #[test]
    fn test_ffi_fluid_roundtrip() {
        unsafe {
            // Null inputs are clean zeros.
            assert_eq!(
                phieng_world_fluid_add(std::ptr::null_mut(), std::ptr::null(), 0),
                0
            );
            assert_eq!(phieng_world_fluid_count(std::ptr::null()), 0);
            assert_eq!(phieng_world_fluid_clear(std::ptr::null_mut()), 0);
            assert_eq!(
                phieng_world_fluid_get(std::ptr::null(), std::ptr::null_mut(), 0),
                0
            );
            // 2D add/count/get/clear.
            let world = phieng_world_new();
            let pts = [0.0f32, 0.0, 0.25, 0.0, 0.0, 0.25];
            assert_eq!(phieng_world_fluid_add(world, pts.as_ptr(), 3), 3);
            assert_eq!(phieng_world_fluid_count(world), 3);
            // Non-finite input is atomic: nothing added.
            let bad = [0.0f32, f32::NAN];
            assert_eq!(phieng_world_fluid_add(world, bad.as_ptr(), 1), 0);
            assert_eq!(phieng_world_fluid_count(world), 3);
            let mut out = [0.0f32; 8];
            assert_eq!(phieng_world_fluid_get(world, out.as_mut_ptr(), 2), 2);
            assert_eq!((out[0], out[1], out[4], out[5]), (0.0, 0.0, 0.25, 0.0));
            assert_eq!(phieng_world_fluid_clear(world), 1);
            assert_eq!(phieng_world_fluid_count(world), 0);
            phieng_world_free(world);
            // 3D add/count/get/clear.
            let world3 = phieng_world3_new();
            let pts3 = [0.0f32, 0.0, 0.0, 0.25, 0.0, 0.0];
            assert_eq!(phieng_world3_fluid_add(world3, pts3.as_ptr(), 2), 2);
            assert_eq!(phieng_world3_fluid_count(world3), 2);
            assert_eq!(
                phieng_world3_fluid_add(std::ptr::null_mut(), pts3.as_ptr(), 2),
                0
            );
            let mut out3 = [0.0f32; 12];
            assert_eq!(phieng_world3_fluid_get(world3, out3.as_mut_ptr(), 2), 2);
            assert_eq!(
                (out3[0], out3[1], out3[2], out3[6], out3[7], out3[8]),
                (0.0, 0.0, 0.0, 0.25, 0.0, 0.0)
            );
            assert_eq!(phieng_world3_fluid_clear(world3), 1);
            assert_eq!(phieng_world3_fluid_count(world3), 0);
            phieng_world3_free(world3);
        }
    }
}
