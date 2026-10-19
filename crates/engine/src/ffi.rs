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
//! - Bulk transfers (`phieng_world_add_bodies`,
//!   `phieng_world_get_transforms_bulk` and their `world3` mirrors) move
//!   whole arrays in one transition; per-slot failures are reported
//!   in-band (id `-1`, NaN sentinel) without aborting the batch.
//! - Gravity configuration (`phieng_world_set/get_gravity_config`,
//!   per-body `phieng_world_set/get_body_gravity_scale` and `world3`
//!   mirrors) exposes `phieng_dynamics::gravity` without changing the
//!   stepping contract.

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

/// Set the gravity force model + integration scheme (see
/// `phieng_dynamics::gravity`). `mode`: 0 uniform, 1 central,
/// 2 N-body, 3 Barnes-Hut. `integrator`: 0 semi-implicit Euler,
/// 1 velocity-Verlet. `big_g` must be finite; `softening` and
/// `theta` must be finite and >= 0; attractor components and `mu`
/// must be finite. Returns 1 on success, 0 for null world or any
/// out-of-range input (world left unchanged).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_gravity_config(
    world: *mut World2,
    mode: u32,
    integrator: u32,
    big_g: f32,
    softening: f32,
    ax: f32,
    ay: f32,
    mu: f32,
    theta: f32,
) -> i32 {
    use phieng_dynamics::{GravityConfig2, GravityMode, Integrator};
    if world.is_null() {
        return 0;
    }
    let mode = match mode {
        0 => GravityMode::Uniform,
        1 => GravityMode::Central,
        2 => GravityMode::NBody,
        3 => GravityMode::BarnesHut,
        _ => return 0,
    };
    let integrator = match integrator {
        0 => Integrator::SemiImplicitEuler,
        1 => Integrator::VelocityVerlet,
        _ => return 0,
    };
    if !valid(big_g)
        || !valid(softening)
        || softening < 0.0
        || !valid(ax)
        || !valid(ay)
        || !valid(mu)
        || !valid(theta)
        || theta < 0.0
    {
        return 0;
    }
    (*world).gravity_config = GravityConfig2 {
        mode,
        integrator,
        big_g,
        softening,
        attractor: Vec2::new(ax, ay),
        attractor_mu: mu,
        theta,
    };
    1
}

/// Set the sleep velocity thresholds (0.0 disables sleeping).
/// Negative or non-finite values are rejected without mutating the
/// world. Returns 1 on success, 0 for null world or bad input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_sleep_thresholds(
    world: *mut World2,
    linear: f32,
    angular: f32,
) -> i32 {
    if world.is_null() || !valid(linear) || linear < 0.0 || !valid(angular) || angular < 0.0 {
        return 0;
    }
    (*world).linear_sleep_threshold = linear;
    (*world).angular_sleep_threshold = angular;
    1
}

/// Copy the sleep velocity thresholds out.
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `out` must be either null (clean `0`) or valid for
/// writes of 2 `f32`s (linear, angular) and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_sleep_thresholds(
    world: *mut World2,
    out: *mut f32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    unsafe {
        *out.add(0) = (*world).linear_sleep_threshold;
        *out.add(1) = (*world).angular_sleep_threshold;
    }
    1
}

/// Copy the gravity config out: `out_mode`/`out_integrator` get the
/// tags, `out_values[0..6]` = (big_g, softening, ax, ay, mu, theta).
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). Out-pointers must be either null (clean `0`) or valid
/// for writes of 1 `u32` / 1 `u32` / 6 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_gravity_config(
    world: *mut World2,
    out_mode: *mut u32,
    out_integrator: *mut u32,
    out_values: *mut f32,
) -> i32 {
    if world.is_null() || out_mode.is_null() || out_integrator.is_null() || out_values.is_null() {
        return 0;
    }
    let cfg = (*world).gravity_config;
    *out_mode = cfg.mode as u32;
    *out_integrator = cfg.integrator as u32;
    *out_values.add(0) = cfg.big_g;
    *out_values.add(1) = cfg.softening;
    *out_values.add(2) = cfg.attractor.x;
    *out_values.add(3) = cfg.attractor.y;
    *out_values.add(4) = cfg.attractor_mu;
    *out_values.add(5) = cfg.theta;
    1
}

// ============================================================================
// Ambient force config FFI (spring / SHM, drag, turbulence)
// ============================================================================

/// C-ABI mirror of the 2D ambient-force configuration. Flags are `u32`
/// (0 = false). Field order is the ABI contract — never reorder.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AmbientConfig2Ffi {
    /// Spring/SHM master switch (0/1).
    pub spring_enabled: u32,
    /// Spring world anchor x.
    pub spring_ax: f32,
    /// Spring world anchor y.
    pub spring_ay: f32,
    /// Spring stiffness `k`.
    pub spring_stiffness: f32,
    /// Spring viscous damping `c`.
    pub spring_damping: f32,
    /// Spring rest-length leash (0 = unlimited).
    pub spring_max_displacement: f32,
    /// Drag master switch (0/1).
    pub drag_enabled: u32,
    /// Linear (Stokes) drag coefficient.
    pub drag_linear: f32,
    /// Quadratic drag coefficient.
    pub drag_quadratic: f32,
    /// Air density multiplier.
    pub air_density: f32,
    /// Angular drag coefficient.
    pub drag_angular: f32,
    /// Ambient wind x.
    pub wind_x: f32,
    /// Ambient wind y.
    pub wind_y: f32,
    /// Turbulence master switch (0/1).
    pub turbulence_enabled: u32,
    /// Peak turbulence acceleration.
    pub turbulence_amplitude: f32,
    /// Turbulence spatial frequency.
    pub turbulence_frequency: f32,
    /// Turbulence temporal rate.
    pub turbulence_time_scale: f32,
    /// Turbulence noise seed.
    pub turbulence_seed: u32,
    /// Turbulence fBm octaves (clamped to 1..=8).
    pub turbulence_octaves: u32,
    /// Turbulence curl-noise (divergence-free) flag (0/1).
    pub turbulence_divergence_free: u32,
}

/// 3D counterpart of [`AmbientConfig2Ffi`] (wind is a `Vec3`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AmbientConfig3Ffi {
    /// Spring/SHM master switch (0/1).
    pub spring_enabled: u32,
    /// Spring world anchor x.
    pub spring_ax: f32,
    /// Spring world anchor y.
    pub spring_ay: f32,
    /// Spring world anchor z.
    pub spring_az: f32,
    /// Spring stiffness `k`.
    pub spring_stiffness: f32,
    /// Spring viscous damping `c`.
    pub spring_damping: f32,
    /// Spring rest-length leash (0 = unlimited).
    pub spring_max_displacement: f32,
    /// Drag master switch (0/1).
    pub drag_enabled: u32,
    /// Linear (Stokes) drag coefficient.
    pub drag_linear: f32,
    /// Quadratic drag coefficient.
    pub drag_quadratic: f32,
    /// Air density multiplier.
    pub air_density: f32,
    /// Angular drag coefficient.
    pub drag_angular: f32,
    /// Ambient wind x.
    pub wind_x: f32,
    /// Ambient wind y.
    pub wind_y: f32,
    /// Ambient wind z.
    pub wind_z: f32,
    /// Turbulence master switch (0/1).
    pub turbulence_enabled: u32,
    /// Peak turbulence acceleration.
    pub turbulence_amplitude: f32,
    /// Turbulence spatial frequency.
    pub turbulence_frequency: f32,
    /// Turbulence temporal rate.
    pub turbulence_time_scale: f32,
    /// Turbulence noise seed.
    pub turbulence_seed: u32,
    /// Turbulence fBm octaves (clamped to 1..=8).
    pub turbulence_octaves: u32,
    /// Turbulence curl-noise (divergence-free) flag (0/1).
    pub turbulence_divergence_free: u32,
}

fn ambient2_from_ffi(c: &AmbientConfig2Ffi) -> Option<phieng_dynamics::AmbientConfig2> {
    use phieng_dynamics::{AmbientConfig2, DragConfig2, SpringConfig2, TurbulenceConfig2};
    for v in [
        c.spring_ax,
        c.spring_ay,
        c.spring_stiffness,
        c.spring_damping,
        c.spring_max_displacement,
        c.drag_linear,
        c.drag_quadratic,
        c.air_density,
        c.drag_angular,
        c.wind_x,
        c.wind_y,
        c.turbulence_amplitude,
        c.turbulence_frequency,
        c.turbulence_time_scale,
    ] {
        if !valid(v) {
            return None;
        }
    }
    if c.spring_stiffness < 0.0
        || c.spring_damping < 0.0
        || c.spring_max_displacement < 0.0
        || c.drag_linear < 0.0
        || c.drag_quadratic < 0.0
        || c.air_density < 0.0
        || c.drag_angular < 0.0
        || c.turbulence_amplitude < 0.0
    {
        return None;
    }
    Some(AmbientConfig2 {
        spring: SpringConfig2 {
            enabled: c.spring_enabled != 0,
            anchor: Vec2::new(c.spring_ax, c.spring_ay),
            stiffness: c.spring_stiffness,
            damping: c.spring_damping,
            max_displacement: c.spring_max_displacement,
        },
        drag: DragConfig2 {
            enabled: c.drag_enabled != 0,
            linear: c.drag_linear,
            quadratic: c.drag_quadratic,
            air_density: c.air_density,
            angular: c.drag_angular,
            wind: Vec2::new(c.wind_x, c.wind_y),
        },
        turbulence: TurbulenceConfig2 {
            enabled: c.turbulence_enabled != 0,
            amplitude: c.turbulence_amplitude,
            frequency: c.turbulence_frequency,
            time_scale: c.turbulence_time_scale,
            seed: c.turbulence_seed,
            octaves: c.turbulence_octaves.min(255) as u8,
            divergence_free: c.turbulence_divergence_free != 0,
        },
    })
}

fn ambient2_to_ffi(a: &phieng_dynamics::AmbientConfig2) -> AmbientConfig2Ffi {
    AmbientConfig2Ffi {
        spring_enabled: a.spring.enabled as u32,
        spring_ax: a.spring.anchor.x,
        spring_ay: a.spring.anchor.y,
        spring_stiffness: a.spring.stiffness,
        spring_damping: a.spring.damping,
        spring_max_displacement: a.spring.max_displacement,
        drag_enabled: a.drag.enabled as u32,
        drag_linear: a.drag.linear,
        drag_quadratic: a.drag.quadratic,
        air_density: a.drag.air_density,
        drag_angular: a.drag.angular,
        wind_x: a.drag.wind.x,
        wind_y: a.drag.wind.y,
        turbulence_enabled: a.turbulence.enabled as u32,
        turbulence_amplitude: a.turbulence.amplitude,
        turbulence_frequency: a.turbulence.frequency,
        turbulence_time_scale: a.turbulence.time_scale,
        turbulence_seed: a.turbulence.seed,
        turbulence_octaves: a.turbulence.octaves as u32,
        turbulence_divergence_free: a.turbulence.divergence_free as u32,
    }
}

fn ambient3_from_ffi(c: &AmbientConfig3Ffi) -> Option<phieng_dynamics::AmbientConfig3> {
    use phieng_dynamics::{AmbientConfig3, DragConfig3, SpringConfig3, TurbulenceConfig3};
    for v in [
        c.spring_ax,
        c.spring_ay,
        c.spring_az,
        c.spring_stiffness,
        c.spring_damping,
        c.spring_max_displacement,
        c.drag_linear,
        c.drag_quadratic,
        c.air_density,
        c.drag_angular,
        c.wind_x,
        c.wind_y,
        c.wind_z,
        c.turbulence_amplitude,
        c.turbulence_frequency,
        c.turbulence_time_scale,
    ] {
        if !valid(v) {
            return None;
        }
    }
    if c.spring_stiffness < 0.0
        || c.spring_damping < 0.0
        || c.spring_max_displacement < 0.0
        || c.drag_linear < 0.0
        || c.drag_quadratic < 0.0
        || c.air_density < 0.0
        || c.drag_angular < 0.0
        || c.turbulence_amplitude < 0.0
    {
        return None;
    }
    Some(AmbientConfig3 {
        spring: SpringConfig3 {
            enabled: c.spring_enabled != 0,
            anchor: Vec3::new(c.spring_ax, c.spring_ay, c.spring_az),
            stiffness: c.spring_stiffness,
            damping: c.spring_damping,
            max_displacement: c.spring_max_displacement,
        },
        drag: DragConfig3 {
            enabled: c.drag_enabled != 0,
            linear: c.drag_linear,
            quadratic: c.drag_quadratic,
            air_density: c.air_density,
            angular: c.drag_angular,
            wind: Vec3::new(c.wind_x, c.wind_y, c.wind_z),
        },
        turbulence: TurbulenceConfig3 {
            enabled: c.turbulence_enabled != 0,
            amplitude: c.turbulence_amplitude,
            frequency: c.turbulence_frequency,
            time_scale: c.turbulence_time_scale,
            seed: c.turbulence_seed,
            octaves: c.turbulence_octaves.min(255) as u8,
            divergence_free: c.turbulence_divergence_free != 0,
        },
    })
}

fn ambient3_to_ffi(a: &phieng_dynamics::AmbientConfig3) -> AmbientConfig3Ffi {
    AmbientConfig3Ffi {
        spring_enabled: a.spring.enabled as u32,
        spring_ax: a.spring.anchor.x,
        spring_ay: a.spring.anchor.y,
        spring_az: a.spring.anchor.z,
        spring_stiffness: a.spring.stiffness,
        spring_damping: a.spring.damping,
        spring_max_displacement: a.spring.max_displacement,
        drag_enabled: a.drag.enabled as u32,
        drag_linear: a.drag.linear,
        drag_quadratic: a.drag.quadratic,
        air_density: a.drag.air_density,
        drag_angular: a.drag.angular,
        wind_x: a.drag.wind.x,
        wind_y: a.drag.wind.y,
        wind_z: a.drag.wind.z,
        turbulence_enabled: a.turbulence.enabled as u32,
        turbulence_amplitude: a.turbulence.amplitude,
        turbulence_frequency: a.turbulence.frequency,
        turbulence_time_scale: a.turbulence.time_scale,
        turbulence_seed: a.turbulence.seed,
        turbulence_octaves: a.turbulence.octaves as u32,
        turbulence_divergence_free: a.turbulence.divergence_free as u32,
    }
}

/// Set the 2D ambient-force config from a caller-owned struct. Returns 1
/// on success, 0 for null world/struct or any out-of-range input (world
/// left unchanged).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `cfg` must be null (clean `0`) or point at an initialized
/// [`AmbientConfig2Ffi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_ambient_config(
    world: *mut World2,
    cfg: *const AmbientConfig2Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = ambient2_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).ambient = parsed;
    1
}

/// Copy the 2D ambient-force config out. Returns 1 on success, 0 for
/// null world/out.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `out` must be null (clean `0`) or valid for a write of one
/// [`AmbientConfig2Ffi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_ambient_config(
    world: *mut World2,
    out: *mut AmbientConfig2Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = ambient2_to_ffi(&(*world).ambient);
    1
}

/// Set the 3D ambient-force config. See
/// [`phieng_world_set_ambient_config`].
/// # Safety
/// See [`phieng_world_set_ambient_config`] (3D world/struct).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_ambient_config(
    world: *mut phieng_dynamics::World3,
    cfg: *const AmbientConfig3Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = ambient3_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).ambient = parsed;
    1
}

/// Copy the 3D ambient-force config out. See
/// [`phieng_world_get_ambient_config`].
/// # Safety
/// See [`phieng_world_get_ambient_config`] (3D world/struct).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_ambient_config(
    world: *mut phieng_dynamics::World3,
    out: *mut AmbientConfig3Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = ambient3_to_ffi(&(*world).ambient);
    1
}

/// CCD config layout mirrored to C# (shared by 2D and 3D worlds).
/// `#[repr(C)]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CcdConfigFfi {
    /// Master switch (0/1).
    pub enabled: u32,
    /// CCD tier tag (0 = Off, 1 = SweptAABB, 2 = FullGJK).
    pub tier: u32,
    /// Maximum travel per substep.
    pub max_movement: f32,
    /// Hard cap on substeps per world step.
    pub max_substeps: u32,
}

fn ccd_tier_from_tag(tag: u32) -> Option<phieng_dynamics::ccd::CcdTier> {
    use phieng_dynamics::ccd::CcdTier;
    match tag {
        0 => Some(CcdTier::Off),
        1 => Some(CcdTier::SweptAABB),
        2 => Some(CcdTier::FullGJK),
        _ => None,
    }
}

fn ccd_tier_to_tag(t: phieng_dynamics::ccd::CcdTier) -> u32 {
    use phieng_dynamics::ccd::CcdTier;
    match t {
        CcdTier::Off => 0,
        CcdTier::SweptAABB => 1,
        CcdTier::FullGJK => 2,
    }
}

fn ccd_from_ffi(c: &CcdConfigFfi) -> Option<phieng_dynamics::ccd::CcdConfig> {
    if !valid(c.max_movement) || c.max_movement <= 0.0 || c.max_substeps < 1 {
        return None;
    }
    let tier = ccd_tier_from_tag(c.tier)?;
    Some(phieng_dynamics::ccd::CcdConfig {
        enabled: c.enabled != 0,
        tier,
        max_movement: c.max_movement,
        max_substeps: c.max_substeps,
    })
}

fn ccd_to_ffi(c: &phieng_dynamics::ccd::CcdConfig) -> CcdConfigFfi {
    CcdConfigFfi {
        enabled: c.enabled as u32,
        tier: ccd_tier_to_tag(c.tier),
        max_movement: c.max_movement,
        max_substeps: c.max_substeps,
    }
}

/// Set the 2D CCD config from a caller-owned struct. Returns 1 on
/// success, 0 for null world/struct or any out-of-range input (world
/// left unchanged).
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `cfg` must be null (clean `0`) or point at an initialized
/// [`CcdConfigFfi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_ccd_config(
    world: *mut World2,
    cfg: *const CcdConfigFfi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = ccd_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).ccd = parsed;
    1
}

/// Copy the 2D CCD config out. Returns 1 on success, 0 for null
/// world/out.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `out` must be null (clean `0`) or valid for a write of one
/// [`CcdConfigFfi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_ccd_config(
    world: *mut World2,
    out: *mut CcdConfigFfi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = ccd_to_ffi(&(*world).ccd);
    1
}

/// Set the 3D CCD config. See [`phieng_world_set_ccd_config`].
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `cfg` must be null (clean `0`) or point at an initialized
/// [`CcdConfigFfi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_ccd_config(
    world: *mut phieng_dynamics::World3,
    cfg: *const CcdConfigFfi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = ccd_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).ccd = parsed;
    1
}

/// Copy the 3D CCD config out. See [`phieng_world_get_ccd_config`].
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `out` must be null (clean `0`) or valid for a write of one
/// [`CcdConfigFfi`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_ccd_config(
    world: *mut phieng_dynamics::World3,
    out: *mut CcdConfigFfi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = ccd_to_ffi(&(*world).ccd);
    1
}

/// SPH fluid config layout (2D) mirrored to C#. `#[repr(C)]`; the
/// `phase_rest_density` / `phase_mass` arrays are 4 floats each.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FluidConfig2Ffi {
    pub particle_mass: f32,
    pub rest_density: f32,
    pub smoothing_length: f32,
    pub stiffness: f32,
    pub gamma: f32,
    pub artificial_viscosity: f32,
    pub surface_tension: f32,
    pub cfl: f32,
    pub max_substeps: u32,
    pub adaptive_resolution: bool,
    pub split_density_ratio: f32,
    pub merge_density_ratio: f32,
    pub boundary_particles: bool,
    pub viscosity: f32,
    pub drag: f32,
    pub wall_friction: f32,
    pub phase_count: u8,
    pub phase_rest_density: [f32; 4],
    pub phase_mass: [f32; 4],
    pub max_velocity: f32,
    pub particle_radius: f32,
    pub enabled: bool,
}

/// 3D counterpart (identical field set to the 2D config).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FluidConfig3Ffi {
    pub particle_mass: f32,
    pub rest_density: f32,
    pub smoothing_length: f32,
    pub stiffness: f32,
    pub gamma: f32,
    pub artificial_viscosity: f32,
    pub surface_tension: f32,
    pub cfl: f32,
    pub max_substeps: u32,
    pub adaptive_resolution: bool,
    pub split_density_ratio: f32,
    pub merge_density_ratio: f32,
    pub boundary_particles: bool,
    pub viscosity: f32,
    pub drag: f32,
    pub wall_friction: f32,
    pub phase_count: u8,
    pub phase_rest_density: [f32; 4],
    pub phase_mass: [f32; 4],
    pub max_velocity: f32,
    pub particle_radius: f32,
    pub enabled: bool,
}

fn valid_fluid_config_f32(v: f32) -> bool {
    v.is_finite()
}

fn fluid2_from_ffi(c: &FluidConfig2Ffi) -> Option<phieng_dynamics::FluidConfig2> {
    use phieng_dynamics::FluidConfig2;
    for v in [
        c.particle_mass,
        c.rest_density,
        c.smoothing_length,
        c.stiffness,
        c.gamma,
        c.artificial_viscosity,
        c.surface_tension,
        c.cfl,
        c.split_density_ratio,
        c.merge_density_ratio,
        c.viscosity,
        c.drag,
        c.wall_friction,
        c.max_velocity,
        c.particle_radius,
        c.phase_rest_density[0],
        c.phase_rest_density[1],
        c.phase_rest_density[2],
        c.phase_rest_density[3],
        c.phase_mass[0],
        c.phase_mass[1],
        c.phase_mass[2],
        c.phase_mass[3],
    ] {
        if !valid_fluid_config_f32(v) {
            return None;
        }
    }
    if c.particle_mass <= 0.0
        || c.rest_density <= 0.0
        || c.smoothing_length <= 0.0
        || c.stiffness < 0.0
        || c.gamma < 1.0
        || c.artificial_viscosity < 0.0
        || c.artificial_viscosity > 2.0
        || c.surface_tension < 0.0
        || c.cfl < 0.0
        || c.cfl > 1.0
        || c.max_substeps < 1
        || c.split_density_ratio <= 0.0
        || c.merge_density_ratio <= 1.0
        || c.viscosity < 0.0
        || c.viscosity > 1.0
        || c.drag < 0.0
        || c.wall_friction < 0.0
        || c.wall_friction > 1.0
        || c.phase_count < 1
        || c.phase_count > 4
        || c.phase_rest_density.iter().any(|&v| v <= 0.0)
        || c.phase_mass.iter().any(|&v| v <= 0.0)
        || c.max_velocity < 0.0
        || c.particle_radius < 0.0
    {
        return None;
    }
    Some(FluidConfig2 {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        phase_count: c.phase_count,
        phase_rest_density: c.phase_rest_density,
        phase_mass: c.phase_mass,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
    })
}

fn fluid3_from_ffi(c: &FluidConfig3Ffi) -> Option<phieng_dynamics::FluidConfig3> {
    use phieng_dynamics::FluidConfig3;
    for v in [
        c.particle_mass,
        c.rest_density,
        c.smoothing_length,
        c.stiffness,
        c.gamma,
        c.artificial_viscosity,
        c.surface_tension,
        c.cfl,
        c.split_density_ratio,
        c.merge_density_ratio,
        c.viscosity,
        c.drag,
        c.wall_friction,
        c.max_velocity,
        c.particle_radius,
        c.phase_rest_density[0],
        c.phase_rest_density[1],
        c.phase_rest_density[2],
        c.phase_rest_density[3],
        c.phase_mass[0],
        c.phase_mass[1],
        c.phase_mass[2],
        c.phase_mass[3],
    ] {
        if !valid_fluid_config_f32(v) {
            return None;
        }
    }
    if c.particle_mass <= 0.0
        || c.rest_density <= 0.0
        || c.smoothing_length <= 0.0
        || c.stiffness < 0.0
        || c.gamma < 1.0
        || c.artificial_viscosity < 0.0
        || c.artificial_viscosity > 2.0
        || c.surface_tension < 0.0
        || c.cfl < 0.0
        || c.cfl > 1.0
        || c.max_substeps < 1
        || c.split_density_ratio <= 0.0
        || c.merge_density_ratio <= 1.0
        || c.viscosity < 0.0
        || c.viscosity > 1.0
        || c.drag < 0.0
        || c.wall_friction < 0.0
        || c.wall_friction > 1.0
        || c.phase_count < 1
        || c.phase_count > 4
        || c.phase_rest_density.iter().any(|&v| v <= 0.0)
        || c.phase_mass.iter().any(|&v| v <= 0.0)
        || c.max_velocity < 0.0
        || c.particle_radius < 0.0
    {
        return None;
    }
    Some(FluidConfig3 {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        phase_count: c.phase_count,
        phase_rest_density: c.phase_rest_density,
        phase_mass: c.phase_mass,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
    })
}

fn fluid2_to_ffi(c: &phieng_dynamics::FluidConfig2) -> FluidConfig2Ffi {
    FluidConfig2Ffi {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        phase_count: c.phase_count,
        phase_rest_density: c.phase_rest_density,
        phase_mass: c.phase_mass,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
    }
}

fn fluid3_to_ffi(c: &phieng_dynamics::FluidConfig3) -> FluidConfig3Ffi {
    FluidConfig3Ffi {
        particle_mass: c.particle_mass,
        rest_density: c.rest_density,
        smoothing_length: c.smoothing_length,
        stiffness: c.stiffness,
        gamma: c.gamma,
        artificial_viscosity: c.artificial_viscosity,
        surface_tension: c.surface_tension,
        cfl: c.cfl,
        max_substeps: c.max_substeps,
        adaptive_resolution: c.adaptive_resolution,
        split_density_ratio: c.split_density_ratio,
        merge_density_ratio: c.merge_density_ratio,
        boundary_particles: c.boundary_particles,
        viscosity: c.viscosity,
        drag: c.drag,
        wall_friction: c.wall_friction,
        phase_count: c.phase_count,
        phase_rest_density: c.phase_rest_density,
        phase_mass: c.phase_mass,
        max_velocity: c.max_velocity,
        particle_radius: c.particle_radius,
        enabled: c.enabled,
    }
}

/// Set the 2D SPH fluid config. Returns 1 on success, 0 for null or bad input.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_fluid_config(
    world: *mut World2,
    cfg: *const FluidConfig2Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = fluid2_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).fluid.config = parsed;
    1
}

/// Get the 2D SPH fluid config. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `FluidConfig2Ffi`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_fluid_config(
    world: *const World2,
    out: *mut FluidConfig2Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = fluid2_to_ffi(&(*world).fluid.config);
    1
}

/// Set the 3D SPH fluid config. Returns 1 on success, 0 for null or bad input.
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_fluid_config(
    world: *mut phieng_dynamics::World3,
    cfg: *const FluidConfig3Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = fluid3_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).fluid.config = parsed;
    1
}

/// Get the 3D SPH fluid config. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `FluidConfig3Ffi`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_fluid_config(
    world: *const phieng_dynamics::World3,
    out: *mut FluidConfig3Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = fluid3_to_ffi(&(*world).fluid.config);
    1
}

/// Set the simulation clock (2D). Returns 1 on success, 0 for null world
/// or non-finite input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_time(world: *mut World2, t: f32) -> i32 {
    if world.is_null() || !valid(t) {
        return 0;
    }
    (*world).time = t;
    1
}

/// Read the simulation clock (2D). Returns 1 on success, 0 on bad input.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one f32.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_time(world: *mut World2, out: *mut f32) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = (*world).time;
    1
}

/// Set the simulation clock (3D). See [`phieng_world_set_time`].
/// # Safety
/// See [`phieng_world_set_time`] (3D world).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_time(
    world: *mut phieng_dynamics::World3,
    t: f32,
) -> i32 {
    if world.is_null() || !valid(t) {
        return 0;
    }
    (*world).time = t;
    1
}

/// Read the simulation clock (3D). See [`phieng_world_get_time`].
/// # Safety
/// See [`phieng_world_get_time`] (3D world).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_time(
    world: *mut phieng_dynamics::World3,
    out: *mut f32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = (*world).time;
    1
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

/// Blittable 2D body descriptor for [`phieng_world_add_bodies`].
/// `#[repr(C)]` layout (36 bytes) must match the C# `BodyDesc2D`
/// struct field-for-field (`src/PhiEng.Native/BodyDescriptors.cs`):
/// pos(2×f32), tag(u32), a/b(2×f32), density(f32), is_static(u8),
/// 3 pad bytes, friction/restitution(2×f32).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BodyDesc2D {
    pub pos_x: f32,
    pub pos_y: f32,
    pub shape_tag: u32,
    pub a: f32,
    pub b: f32,
    pub density: f32,
    pub is_static: u8,
    pub _pad: [u8; 3],
    pub friction: f32,
    pub restitution: f32,
}

// Guard the cross-language layout contract: any field change that
// alters the size breaks the C# mirror silently. Update both sides.
const _: () = assert!(std::mem::size_of::<BodyDesc2D>() == 36);

/// Bulk body creation: `n` descriptors at `descs` are validated and
/// added exactly like [`phieng_world_add_body`] (same tags, same
/// rejection rules), one transition instead of N. `out_ids[i]` gets
/// the new body id, or -1 when descriptor `i` is rejected; other
/// bodies are unaffected (non-atomic by design — no partial rollback).
/// Returns the number of bodies added, or 0 for null world/descs/ids.
/// `n` is capped at 1,000,000 like [`phieng_world_fluid_add`].
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `descs` must be null (clean `0`) or valid for reads of
/// `n` [`BodyDesc2D`]s; `out_ids` null (clean `0`) or valid for writes
/// of `n` `i32`s. Both must be properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_add_bodies(
    world: *mut World2,
    descs: *const BodyDesc2D,
    n: u32,
    out_ids: *mut i32,
) -> u32 {
    if world.is_null() || descs.is_null() || out_ids.is_null() {
        return 0;
    }
    let n = (n as usize).min(1_000_000);
    if n == 0 {
        return 0;
    }
    let src = unsafe { std::slice::from_raw_parts(descs, n) };
    let world = unsafe { &mut *world };
    let mut added: u32 = 0;
    for (i, d) in src.iter().enumerate() {
        if !valid(d.pos_x) || !valid(d.pos_y) {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        let shape = match d.shape_tag {
            SHAPE_CIRCLE if valid_positive(d.a) => Shape2::Circle { radius: d.a },
            SHAPE_BOX if valid_positive(d.a) && valid_positive(d.b) => Shape2::Box {
                half_extents: Vec2::new(d.a, d.b),
            },
            _ => {
                unsafe { *out_ids.add(i) = -1 };
                continue;
            }
        };
        if !valid_positive(d.density) {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        let position = Vec2::new(d.pos_x, d.pos_y);
        let material = PhysicsMaterial::new(d.friction, d.restitution, d.density);
        let mut body = if d.is_static != 0 {
            phieng_dynamics::RigidBody2::static_body(world.next_body_id)
        } else {
            phieng_dynamics::RigidBody2::new(world.next_body_id)
        };
        body.transform = Transform2::from_position_angle(position, 0.0);
        if body.set_shape(shape, d.density).is_err() || body.set_material(material).is_err() {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        match world.add_body(body) {
            Ok(id) => {
                unsafe { *out_ids.add(i) = id as i32 };
                added += 1;
            }
            Err(_) => unsafe { *out_ids.add(i) = -1 },
        }
    }
    added
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

/// Set a body's gravitational coupling multiplier (`gravity_scale`:
/// 1.0 full coupling, 0.0 immune, negative repels; finite only).
/// Returns 1 on success, 0 for null world, unknown body, or
/// non-finite scale.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_body_gravity_scale(
    world: *mut World2,
    body_id: u32,
    scale: f32,
) -> i32 {
    if world.is_null() || !valid(scale) {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].gravity_scale = scale;
    1
}

/// Copy a body's gravitational coupling multiplier out.
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `out_scale` must be either null (clean `0`) or valid
/// for writes of 1 `f32` and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_gravity_scale(
    world: *mut World2,
    body_id: u32,
    out_scale: *mut f32,
) -> i32 {
    if world.is_null() || out_scale.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_scale = world_ref.bodies[index].gravity_scale;
    1
}

/// Set a body's ambient-force coupling multiplier (2D). See
/// [`phieng_world_set_body_gravity_scale`].
/// # Safety
/// See [`phieng_world_set_body_gravity_scale`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_body_ambient_scale(
    world: *mut World2,
    body_id: u32,
    scale: f32,
) -> i32 {
    if world.is_null() || !valid(scale) {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].ambient_scale = scale;
    1
}

/// Copy a body's ambient-force coupling multiplier out (2D).
/// # Safety
/// See [`phieng_world_get_body_gravity_scale`].
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_ambient_scale(
    world: *mut World2,
    body_id: u32,
    out_scale: *mut f32,
) -> i32 {
    if world.is_null() || out_scale.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_scale = world_ref.bodies[index].ambient_scale;
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

/// Bulk transform + velocity read: `n` body ids at `ids` are resolved in
/// one transition instead of N getter calls. Slot `i` of every output
/// array corresponds to `ids[i]` (index alignment is preserved, never
/// compacted): `out_pos[2*i, 2*i+1]` = (x, y), `out_rot[i]` = rotation
/// angle in radians, and — when `out_vel` is non-null —
/// `out_vel[2*i, 2*i+1]` = (vx, vy). Stale/unknown ids write NaN into
/// their slot (all of pos/rot/vel) instead of aborting the batch.
/// `out_vel` may be null to skip velocity readback; `out_pos` and
/// `out_rot` must be valid. Returns the number of ids that resolved
/// to a live body, or 0 for null world/ids/pos/rot.
/// # Safety
/// `world` must be a live pointer from [`phieng_world_new`] (or null,
/// clean `0`). `ids` must be null (clean `0`) or valid for reads of
/// `n` `u32`s; `out_pos`/`out_rot` null (clean `0`) or valid for writes
/// of `2*n` / `n` `f32`s; `out_vel` null (skip) or valid for `2*n`
/// `f32`s. All must be properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_transforms_bulk(
    world: *mut World2,
    ids: *const u32,
    n: u32,
    out_pos: *mut f32,
    out_rot: *mut f32,
    out_vel: *mut f32,
) -> u32 {
    if world.is_null() || ids.is_null() || out_pos.is_null() || out_rot.is_null() {
        return 0;
    }
    let n = n as usize;
    if n == 0 {
        return 0;
    }
    let id_slice = unsafe { std::slice::from_raw_parts(ids, n) };
    let world_ref: &World2 = unsafe { &*world };
    let write_vel = !out_vel.is_null();
    let mut valid: u32 = 0;
    for (i, &body_id) in id_slice.iter().enumerate() {
        match body_by_id(world_ref, body_id) {
            Some(index) => {
                let body = &world_ref.bodies[index];
                unsafe {
                    *out_pos.add(2 * i) = body.transform.position.x;
                    *out_pos.add(2 * i + 1) = body.transform.position.y;
                    *out_rot.add(i) = body.rotation_angle();
                    if write_vel {
                        *out_vel.add(2 * i) = body.velocity.x;
                        *out_vel.add(2 * i + 1) = body.velocity.y;
                    }
                }
                valid += 1;
            }
            None => unsafe {
                *out_pos.add(2 * i) = f32::NAN;
                *out_pos.add(2 * i + 1) = f32::NAN;
                *out_rot.add(i) = f32::NAN;
                if write_vel {
                    *out_vel.add(2 * i) = f32::NAN;
                    *out_vel.add(2 * i + 1) = f32::NAN;
                }
            },
        }
    }
    valid
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

/// Add 2D SPH fluid particles with per-particle phase indices. `positions`
/// is interleaved `[x0, y0, x1, y1, ...]`, `phases[i]` is the phase of
/// particle `i` (clamped to 0..=3 internally). Returns the number added,
/// or 0 on null world/positions/phases or any non-finite position.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
/// `positions` must be null or valid for reads of `2 * count` `f32`s.
/// `phases` must be null or valid for reads of `count` `u8`s.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_fluid_add_phase(
    world: *mut World2,
    positions: *const f32,
    phases: *const u8,
    count: u32,
) -> u32 {
    if world.is_null() || positions.is_null() || phases.is_null() {
        return 0;
    }
    let n = (count as usize).min(1_000_000);
    let src = unsafe { std::slice::from_raw_parts(positions, n.saturating_mul(2)) };
    if src.iter().any(|v| !v.is_finite()) {
        return 0;
    }
    let ph = unsafe { std::slice::from_raw_parts(phases, n) };
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks
    // (`src` is exactly `2 * n` long by construction above).
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let pts: Vec<Vec2> = src.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect();
    let phases: Vec<u8> = ph.to_vec();
    unsafe { &mut *world }
        .fluid
        .add_particles_phase(&pts, &phases) as u32
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

/// Set the 3D gravity force model + integration scheme (same contract
/// as [`phieng_world_set_gravity_config`], with `az` for the
/// attractor's z component).
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn phieng_world3_set_gravity_config(
    world: *mut phieng_dynamics::World3,
    mode: u32,
    integrator: u32,
    big_g: f32,
    softening: f32,
    ax: f32,
    ay: f32,
    az: f32,
    mu: f32,
    theta: f32,
) -> i32 {
    use phieng_dynamics::{GravityConfig3, GravityMode, Integrator};
    if world.is_null() {
        return 0;
    }
    let mode = match mode {
        0 => GravityMode::Uniform,
        1 => GravityMode::Central,
        2 => GravityMode::NBody,
        3 => GravityMode::BarnesHut,
        _ => return 0,
    };
    let integrator = match integrator {
        0 => Integrator::SemiImplicitEuler,
        1 => Integrator::VelocityVerlet,
        _ => return 0,
    };
    if !big_g.is_finite()
        || !softening.is_finite()
        || softening < 0.0
        || !ax.is_finite()
        || !ay.is_finite()
        || !az.is_finite()
        || !mu.is_finite()
        || !theta.is_finite()
        || theta < 0.0
    {
        return 0;
    }
    (*world).gravity_config = GravityConfig3 {
        mode,
        integrator,
        big_g,
        softening,
        attractor: glam::Vec3::new(ax, ay, az),
        attractor_mu: mu,
        theta,
    };
    1
}

/// Set the 3D sleep velocity thresholds (same contract as
/// [`phieng_world_set_sleep_thresholds`]).
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_sleep_thresholds(
    world: *mut phieng_dynamics::World3,
    linear: f32,
    angular: f32,
) -> i32 {
    if world.is_null()
        || !linear.is_finite()
        || linear < 0.0
        || !angular.is_finite()
        || angular < 0.0
    {
        return 0;
    }
    (*world).linear_sleep_threshold = linear;
    (*world).angular_sleep_threshold = angular;
    1
}

/// Copy the 3D sleep velocity thresholds out.
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `out` must be either null (clean `0`) or valid for
/// writes of 2 `f32`s (linear, angular) and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_sleep_thresholds(
    world: *mut phieng_dynamics::World3,
    out: *mut f32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    unsafe {
        *out.add(0) = (*world).linear_sleep_threshold;
        *out.add(1) = (*world).angular_sleep_threshold;
    }
    1
}

/// Copy the 3D gravity config out: `out_mode`/`out_integrator` get the
/// tags, `out_values[0..7]` = (big_g, softening, ax, ay, az, mu, theta).
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). Out-pointers must be either null (clean `0`) or valid
/// for writes of 1 `u32` / 1 `u32` / 7 `f32`s and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_gravity_config(
    world: *mut phieng_dynamics::World3,
    out_mode: *mut u32,
    out_integrator: *mut u32,
    out_values: *mut f32,
) -> i32 {
    if world.is_null() || out_mode.is_null() || out_integrator.is_null() || out_values.is_null() {
        return 0;
    }
    let cfg = (*world).gravity_config;
    *out_mode = cfg.mode as u32;
    *out_integrator = cfg.integrator as u32;
    *out_values.add(0) = cfg.big_g;
    *out_values.add(1) = cfg.softening;
    *out_values.add(2) = cfg.attractor.x;
    *out_values.add(3) = cfg.attractor.y;
    *out_values.add(4) = cfg.attractor.z;
    *out_values.add(5) = cfg.attractor_mu;
    *out_values.add(6) = cfg.theta;
    1
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

/// Set a 3D body's gravitational coupling multiplier (same contract
/// as [`phieng_world_set_body_gravity_scale`]).
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). Use-after-free and forged pointers are undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_body_gravity_scale(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    scale: f32,
) -> i32 {
    if world.is_null() || !scale.is_finite() {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].gravity_scale = scale;
    1
}

/// Copy a 3D body's gravitational coupling multiplier out.
/// Returns 1 on success, 0 on any invalid input.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `out_scale` must be either null (clean `0`) or valid
/// for writes of 1 `f32` and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_gravity_scale(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    out_scale: *mut f32,
) -> i32 {
    if world.is_null() || out_scale.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_scale = world_ref.bodies[index].gravity_scale;
    1
}

/// Set a body's ambient-force coupling multiplier (3D).
/// # Safety
/// See [`phieng_world_set_body_gravity_scale`] (3D world).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_body_ambient_scale(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    scale: f32,
) -> i32 {
    if world.is_null() || !valid(scale) {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].ambient_scale = scale;
    1
}

/// Copy a body's ambient-force coupling multiplier out (3D).
/// # Safety
/// See [`phieng_world_get_body_gravity_scale`] (3D world).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_ambient_scale(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    out_scale: *mut f32,
) -> i32 {
    if world.is_null() || out_scale.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_scale = world_ref.bodies[index].ambient_scale;
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

/// Add 3D SPH fluid particles with per-particle phase indices. See
/// [`phieng_world_fluid_add_phase`] (3D world/struct).
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
/// `positions` must be null or valid for reads of `3 * count` `f32`s.
/// `phases` must be null or valid for reads of `count` `u8`s.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_fluid_add_phase(
    world: *mut phieng_dynamics::World3,
    positions: *const f32,
    phases: *const u8,
    count: u32,
) -> u32 {
    if world.is_null() || positions.is_null() || phases.is_null() {
        return 0;
    }
    let n = (count as usize).min(1_000_000);
    let src = unsafe { std::slice::from_raw_parts(positions, n.saturating_mul(3)) };
    if src.iter().any(|v| !v.is_finite()) {
        return 0;
    }
    let ph = unsafe { std::slice::from_raw_parts(phases, n) };
    // MSRV 1.78: chunks_exact is used instead of the newer as_chunks
    // (`src` is exactly `3 * n` long by construction above).
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let pts: Vec<Vec3> = src
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();
    let phases: Vec<u8> = ph.to_vec();
    unsafe { &mut *world }
        .fluid
        .add_particles_phase(&pts, &phases) as u32
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

/// Blittable 3D body descriptor for [`phieng_world3_add_bodies`].
/// `#[repr(C)]` layout (44 bytes) must match the C# `BodyDesc3D`
/// struct field-for-field (`src/PhiEng.Native/BodyDescriptors.cs`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BodyDesc3D {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub shape_tag: u32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub density: f32,
    pub is_static: u8,
    pub _pad: [u8; 3],
    pub friction: f32,
    pub restitution: f32,
}

// Guard the cross-language layout contract: any field change that
// alters the size breaks the C# mirror silently. Update both sides.
const _: () = assert!(std::mem::size_of::<BodyDesc3D>() == 44);

/// Bulk 3D body creation: same contract as [`phieng_world_add_bodies`]
/// (per-descriptor validation identical to [`phieng_world3_add_body`]:
/// `SHAPE3_SPHERE` = 0 (`a` = radius), `SHAPE3_BOX` = 1 (`a,b,c` =
/// half extents), `SHAPE3_CAPSULE` = 2 (`a` = radius, `b` =
/// half length); rejected descriptors write -1 into `out_ids` and do
/// not roll back the rest). Returns the number added, or 0 for null
/// world/descs/ids. `n` capped at 1,000,000.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `descs` must be null (clean `0`) or valid for reads of
/// `n` [`BodyDesc3D`]s; `out_ids` null (clean `0`) or valid for writes
/// of `n` `i32`s. Both must be properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_add_bodies(
    world: *mut phieng_dynamics::World3,
    descs: *const BodyDesc3D,
    n: u32,
    out_ids: *mut i32,
) -> u32 {
    if world.is_null() || descs.is_null() || out_ids.is_null() {
        return 0;
    }
    let n = (n as usize).min(1_000_000);
    if n == 0 {
        return 0;
    }
    let src = unsafe { std::slice::from_raw_parts(descs, n) };
    let world = unsafe { &mut *world };
    let mut added: u32 = 0;
    for (i, d) in src.iter().enumerate() {
        if !d.pos_x.is_finite() || !d.pos_y.is_finite() || !d.pos_z.is_finite() {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        let shape = match d.shape_tag {
            SHAPE3_SPHERE if d.a > 0.0 && d.a.is_finite() => {
                phieng_dynamics::Shape3::Sphere { radius: d.a }
            }
            SHAPE3_BOX
                if d.a > 0.0
                    && d.b > 0.0
                    && d.c > 0.0
                    && d.a.is_finite()
                    && d.b.is_finite()
                    && d.c.is_finite() =>
            {
                phieng_dynamics::Shape3::Box {
                    half_extents: glam::Vec3::new(d.a, d.b, d.c),
                }
            }
            SHAPE3_CAPSULE if d.a > 0.0 && d.b > 0.0 && d.a.is_finite() && d.b.is_finite() => {
                phieng_dynamics::Shape3::Capsule {
                    radius: d.a,
                    half_length: d.b,
                }
            }
            _ => {
                unsafe { *out_ids.add(i) = -1 };
                continue;
            }
        };
        if !d.density.is_finite() || d.density <= 0.0 {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        let position = glam::Vec3::new(d.pos_x, d.pos_y, d.pos_z);
        let material = phieng_dynamics::PhysicsMaterial::new(d.friction, d.restitution, d.density);
        let mut body = if d.is_static != 0 {
            phieng_dynamics::RigidBody3::static_body(world.next_body_id)
        } else {
            phieng_dynamics::RigidBody3::new(world.next_body_id)
        };
        body.transform.position = position;
        if body.set_shape(shape, d.density).is_err() || body.set_material(material).is_err() {
            unsafe { *out_ids.add(i) = -1 };
            continue;
        }
        match world.add_body(body) {
            Ok(id) => {
                unsafe { *out_ids.add(i) = id as i32 };
                added += 1;
            }
            Err(_) => unsafe { *out_ids.add(i) = -1 },
        }
    }
    added
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

/// Bulk 3D transform + velocity read: same contract as
/// [`phieng_world_get_transforms_bulk`] with 3D strides — `out_pos`
/// holds `3*n` floats (x, y, z), `out_quat` holds `4*n` (x, y, z, w),
/// `out_vel` holds `3*n` or is null to skip velocities. Stale ids
/// write NaN into their slot; returns the number of live ids, or 0
/// for null world/ids/pos/quat.
/// # Safety
/// `world` must be a live pointer from [`phieng_world3_new`] (or null,
/// clean `0`). `ids` must be null (clean `0`) or valid for reads of
/// `n` `u32`s; `out_pos`/`out_quat` null (clean `0`) or valid for
/// writes of `3*n` / `4*n` `f32`s; `out_vel` null (skip) or valid for
/// `3*n` `f32`s. All must be properly aligned.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_transforms_bulk(
    world: *mut phieng_dynamics::World3,
    ids: *const u32,
    n: u32,
    out_pos: *mut f32,
    out_quat: *mut f32,
    out_vel: *mut f32,
) -> u32 {
    if world.is_null() || ids.is_null() || out_pos.is_null() || out_quat.is_null() {
        return 0;
    }
    let n = n as usize;
    if n == 0 {
        return 0;
    }
    let id_slice = unsafe { std::slice::from_raw_parts(ids, n) };
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let write_vel = !out_vel.is_null();
    let mut valid: u32 = 0;
    for (i, &body_id) in id_slice.iter().enumerate() {
        match body3_by_id(world_ref, body_id) {
            Some(index) => {
                let body = &world_ref.bodies[index];
                unsafe {
                    *out_pos.add(3 * i) = body.transform.position.x;
                    *out_pos.add(3 * i + 1) = body.transform.position.y;
                    *out_pos.add(3 * i + 2) = body.transform.position.z;
                    *out_quat.add(4 * i) = body.transform.rotation.x;
                    *out_quat.add(4 * i + 1) = body.transform.rotation.y;
                    *out_quat.add(4 * i + 2) = body.transform.rotation.z;
                    *out_quat.add(4 * i + 3) = body.transform.rotation.w;
                    if write_vel {
                        *out_vel.add(3 * i) = body.velocity.x;
                        *out_vel.add(3 * i + 1) = body.velocity.y;
                        *out_vel.add(3 * i + 2) = body.velocity.z;
                    }
                }
                valid += 1;
            }
            None => unsafe {
                *out_pos.add(3 * i) = f32::NAN;
                *out_pos.add(3 * i + 1) = f32::NAN;
                *out_pos.add(3 * i + 2) = f32::NAN;
                *out_quat.add(4 * i) = f32::NAN;
                *out_quat.add(4 * i + 1) = f32::NAN;
                *out_quat.add(4 * i + 2) = f32::NAN;
                *out_quat.add(4 * i + 3) = f32::NAN;
                if write_vel {
                    *out_vel.add(3 * i) = f32::NAN;
                    *out_vel.add(3 * i + 1) = f32::NAN;
                    *out_vel.add(3 * i + 2) = f32::NAN;
                }
            },
        }
    }
    valid
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

/// Broadphase mode tags (mirror the `BroadphaseMode` discriminants in
/// `phieng_broadphase::broadphase_mode`).
pub const BROADPHASE_BRUTE_FORCE: u32 = 0;
pub const BROADPHASE_SPATIAL_HASH: u32 = 1;
pub const BROADPHASE_DYNAMIC_AABB_TREE: u32 = 2;
pub const BROADPHASE_HYBRID: u32 = 3;

fn broadphase_mode_from_u32(v: u32) -> Option<phieng_broadphase::BroadphaseMode> {
    use phieng_broadphase::BroadphaseMode;
    match v {
        BROADPHASE_BRUTE_FORCE => Some(BroadphaseMode::BruteForce),
        BROADPHASE_SPATIAL_HASH => Some(BroadphaseMode::SpatialHash),
        BROADPHASE_DYNAMIC_AABB_TREE => Some(BroadphaseMode::DynamicAABBTree),
        BROADPHASE_HYBRID => Some(BroadphaseMode::Hybrid),
        _ => None,
    }
}

fn broadphase_mode_to_u32(m: phieng_broadphase::BroadphaseMode) -> u32 {
    use phieng_broadphase::BroadphaseMode;
    match m {
        BroadphaseMode::BruteForce => BROADPHASE_BRUTE_FORCE,
        BroadphaseMode::SpatialHash => BROADPHASE_SPATIAL_HASH,
        BroadphaseMode::DynamicAABBTree => BROADPHASE_DYNAMIC_AABB_TREE,
        BroadphaseMode::Hybrid => BROADPHASE_HYBRID,
    }
}

/// Set the 2D broadphase mode. Returns 1 on success, 0 on null/unknown tag.
/// The backend is rebuilt immediately (and again on each step while the
/// mode is `Hybrid`), so the change takes effect on the next step.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_broadphase_mode(world: *mut World2, mode: u32) -> i32 {
    if world.is_null() {
        return 0;
    }
    let Some(m) = broadphase_mode_from_u32(mode) else {
        return 0;
    };
    let w = &mut *world;
    let cell = w.broadphase.cell_size();
    w.broadphase_mode = m;
    w.broadphase = phieng_broadphase::Broadphase2::new(m, cell);
    1
}

/// Get the 2D broadphase mode tag. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `u32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_broadphase_mode(
    world: *const World2,
    out: *mut u32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = broadphase_mode_to_u32((*world).broadphase_mode);
    1
}

/// Set the 3D broadphase mode. Returns 1 on success, 0 on null/unknown tag.
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_broadphase_mode(
    world: *mut phieng_dynamics::World3,
    mode: u32,
) -> i32 {
    if world.is_null() {
        return 0;
    }
    let Some(m) = broadphase_mode_from_u32(mode) else {
        return 0;
    };
    let w = &mut *world;
    w.broadphase_mode = m;
    w.broadphase = phieng_broadphase::Broadphase3::new(m);
    1
}

/// Get the 3D broadphase mode tag. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `u32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_broadphase_mode(
    world: *const phieng_dynamics::World3,
    out: *mut u32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = broadphase_mode_to_u32((*world).broadphase_mode);
    1
}

/// Thermal config FFI mirror (2D).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ThermalConfig2Ffi {
    pub enabled: bool,
    pub ambient_temperature: f32,
    pub stefan_boltzmann: f32,
    pub convection_coefficient: f32,
    pub radiation_enabled: bool,
    pub conduction_enabled: bool,
    pub convection_enabled: bool,
    pub max_temperature: f32,
    pub min_temperature: f32,
    pub radiation_radius_factor: f32,
    pub max_temperature_rate: f32,
}

pub type ThermalConfig3Ffi = ThermalConfig2Ffi;

fn thermal2_from_ffi(c: &ThermalConfig2Ffi) -> Option<phieng_dynamics::ThermalConfig2D> {
    use phieng_dynamics::ThermalConfig2D;
    for v in [
        c.ambient_temperature,
        c.stefan_boltzmann,
        c.convection_coefficient,
        c.max_temperature,
        c.min_temperature,
        c.radiation_radius_factor,
        c.max_temperature_rate,
    ] {
        if !v.is_finite() {
            return None;
        }
    }
    if c.ambient_temperature <= 0.0
        || c.stefan_boltzmann < 0.0
        || c.convection_coefficient < 0.0
        || c.max_temperature <= c.min_temperature
        || c.radiation_radius_factor <= 0.0
        || c.max_temperature_rate <= 0.0
    {
        return None;
    }
    Some(ThermalConfig2D {
        enabled: c.enabled,
        ambient_temperature: c.ambient_temperature,
        stefan_boltzmann: c.stefan_boltzmann,
        convection_coefficient: c.convection_coefficient,
        radiation_enabled: c.radiation_enabled,
        conduction_enabled: c.conduction_enabled,
        convection_enabled: c.convection_enabled,
        max_temperature: c.max_temperature,
        min_temperature: c.min_temperature,
        radiation_radius_factor: c.radiation_radius_factor,
        max_temperature_rate: c.max_temperature_rate,
    })
}

fn thermal2_to_ffi(c: &phieng_dynamics::ThermalConfig2D) -> ThermalConfig2Ffi {
    ThermalConfig2Ffi {
        enabled: c.enabled,
        ambient_temperature: c.ambient_temperature,
        stefan_boltzmann: c.stefan_boltzmann,
        convection_coefficient: c.convection_coefficient,
        radiation_enabled: c.radiation_enabled,
        conduction_enabled: c.conduction_enabled,
        convection_enabled: c.convection_enabled,
        max_temperature: c.max_temperature,
        min_temperature: c.min_temperature,
        radiation_radius_factor: c.radiation_radius_factor,
        max_temperature_rate: c.max_temperature_rate,
    }
}

/// Set the 2D thermal config. Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_thermal_config(
    world: *mut World2,
    cfg: *const ThermalConfig2Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = thermal2_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).thermal.config = parsed;
    1
}

/// Get the 2D thermal config. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `ThermalConfig2Ffi`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_thermal_config(
    world: *const World2,
    out: *mut ThermalConfig2Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = thermal2_to_ffi(&(*world).thermal.config);
    1
}

/// Set the 3D thermal config. Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_thermal_config(
    world: *mut phieng_dynamics::World3,
    cfg: *const ThermalConfig3Ffi,
) -> i32 {
    if world.is_null() || cfg.is_null() {
        return 0;
    }
    let Some(parsed) = thermal2_from_ffi(unsafe { &*cfg }) else {
        return 0;
    };
    (*world).thermal.config = parsed;
    1
}

/// Get the 3D thermal config. Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `ThermalConfig3Ffi`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_thermal_config(
    world: *const phieng_dynamics::World3,
    out: *mut ThermalConfig3Ffi,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    *out = thermal2_to_ffi(&(*world).thermal.config);
    1
}

/// Set a body's temperature (2D). Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_body_temperature(
    world: *mut World2,
    body_id: u32,
    temperature: f32,
) -> i32 {
    if world.is_null() || !valid(temperature) || temperature < 0.0 {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].temperature = temperature;
    1
}

/// Get a body's temperature (2D). Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `f32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_temperature(
    world: *const World2,
    body_id: u32,
    out: *mut f32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    *out = world_ref.bodies[index].temperature;
    1
}

/// Set a body's temperature (3D). Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_body_temperature(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    temperature: f32,
) -> i32 {
    if world.is_null() || !valid(temperature) || temperature < 0.0 {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].temperature = temperature;
    1
}

/// Get a body's temperature (3D). Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out` must be null or valid for one `f32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_temperature(
    world: *const phieng_dynamics::World3,
    body_id: u32,
    out: *mut f32,
) -> i32 {
    if world.is_null() || out.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    *out = world_ref.bodies[index].temperature;
    1
}

/// Set a body's thermal material properties (2D).
/// Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world_set_body_thermal_material(
    world: *mut World2,
    body_id: u32,
    specific_heat: f32,
    thermal_conductivity: f32,
    emissivity: f32,
) -> i32 {
    if world.is_null()
        || !valid(specific_heat)
        || specific_heat <= 0.0
        || !valid(thermal_conductivity)
        || thermal_conductivity < 0.0
        || !valid(emissivity)
        || emissivity < 0.0
        || emissivity > 1.0
    {
        return 0;
    }
    let world_ref: &mut World2 = unsafe { &mut *world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].material.specific_heat = specific_heat;
    world_ref.bodies[index].material.thermal_conductivity = thermal_conductivity;
    world_ref.bodies[index].material.emissivity = emissivity;
    1
}

/// Get a body's thermal material properties (2D).
/// Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out_sh`/`out_k`/`out_e` must be null or valid for one `f32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world_get_body_thermal_material(
    world: *const World2,
    body_id: u32,
    out_sh: *mut f32,
    out_k: *mut f32,
    out_e: *mut f32,
) -> i32 {
    if world.is_null() || out_sh.is_null() || out_k.is_null() || out_e.is_null() {
        return 0;
    }
    let world_ref: &World2 = unsafe { &*world };
    let Some(index) = body_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_sh = world_ref.bodies[index].material.specific_heat;
    *out_k = world_ref.bodies[index].material.thermal_conductivity;
    *out_e = world_ref.bodies[index].material.emissivity;
    1
}

/// Set a body's thermal material properties (3D).
/// Returns 1 on success, 0 on null/invalid.
/// # Safety
/// `world` must be a live pointer from `phieng_world3_new` (or null).
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_set_body_thermal_material(
    world: *mut phieng_dynamics::World3,
    body_id: u32,
    specific_heat: f32,
    thermal_conductivity: f32,
    emissivity: f32,
) -> i32 {
    if world.is_null()
        || !valid(specific_heat)
        || specific_heat <= 0.0
        || !valid(thermal_conductivity)
        || thermal_conductivity < 0.0
        || !valid(emissivity)
        || emissivity < 0.0
        || emissivity > 1.0
    {
        return 0;
    }
    let world_ref: &mut phieng_dynamics::World3 = unsafe { &mut *world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    world_ref.bodies[index].material.specific_heat = specific_heat;
    world_ref.bodies[index].material.thermal_conductivity = thermal_conductivity;
    world_ref.bodies[index].material.emissivity = emissivity;
    1
}

/// Get a body's thermal material properties (3D).
/// Returns 1 on success, 0 on null/out.
/// # Safety
/// `world` must be live or null; `out_sh`/`out_k`/`out_e` must be null or valid for one `f32`.
#[no_mangle]
pub unsafe extern "C" fn phieng_world3_get_body_thermal_material(
    world: *const phieng_dynamics::World3,
    body_id: u32,
    out_sh: *mut f32,
    out_k: *mut f32,
    out_e: *mut f32,
) -> i32 {
    if world.is_null() || out_sh.is_null() || out_k.is_null() || out_e.is_null() {
        return 0;
    }
    let world_ref: &phieng_dynamics::World3 = unsafe { &*world };
    let Some(index) = body3_by_id(world_ref, body_id) else {
        return 0;
    };
    *out_sh = world_ref.bodies[index].material.specific_heat;
    *out_k = world_ref.bodies[index].material.thermal_conductivity;
    *out_e = world_ref.bodies[index].material.emissivity;
    1
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
    fn bulk_add_matches_scalar_add() {
        // Same descriptors through both paths must yield identical ids
        // and identical world state (bulk validation == scalar validation).
        unsafe {
            let mk = |x: f32| BodyDesc2D {
                pos_x: x,
                pos_y: 2.0,
                shape_tag: SHAPE_CIRCLE,
                a: 0.5,
                b: 0.0,
                density: 1.0,
                is_static: 0,
                _pad: [0; 3],
                friction: 0.5,
                restitution: 0.0,
            };
            let w1 = phieng_world_new();
            let i0 = phieng_world_add_body(w1, 0.0, 2.0, SHAPE_CIRCLE, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
            let i1 = phieng_world_add_body(w1, 1.0, 2.0, SHAPE_CIRCLE, 0.5, 0.0, 1.0, 0, 0.5, 0.0);
            let w2 = phieng_world_new();
            let descs = [mk(0.0), mk(1.0)];
            let mut ids = [-9i32; 2];
            let added = phieng_world_add_bodies(w2, descs.as_ptr(), 2, ids.as_mut_ptr());
            assert_eq!(added, 2);
            assert_eq!(ids, [i0, i1]);
            assert_eq!(phieng_world_body_count(w2), 2);
            // Null-safety and empty batch.
            assert_eq!(
                phieng_world_add_bodies(std::ptr::null_mut(), descs.as_ptr(), 2, ids.as_mut_ptr()),
                0
            );
            assert_eq!(
                phieng_world_add_bodies(w2, std::ptr::null(), 2, ids.as_mut_ptr()),
                0
            );
            assert_eq!(
                phieng_world_add_bodies(w2, descs.as_ptr(), 2, std::ptr::null_mut()),
                0
            );
            assert_eq!(
                phieng_world_add_bodies(w2, descs.as_ptr(), 0, ids.as_mut_ptr()),
                0
            );
            phieng_world_free(w1);
            phieng_world_free(w2);
        }
    }

    #[test]
    fn bulk_add_rejects_per_slot_without_rollback() {
        // Bad descriptors write -1 into their own slot; valid siblings
        // are still added (non-atomic by design).
        unsafe {
            let good = BodyDesc2D {
                pos_x: 0.0,
                pos_y: 0.0,
                shape_tag: SHAPE_BOX,
                a: 0.5,
                b: 0.5,
                density: 1.0,
                is_static: 0,
                _pad: [0; 3],
                friction: 0.5,
                restitution: 0.0,
            };
            let bad_tag = BodyDesc2D {
                shape_tag: 99,
                ..good
            };
            let bad_density = BodyDesc2D {
                pos_x: 5.0,
                density: -2.0,
                ..good
            };
            let descs = [good, bad_tag, bad_density, good];
            let world = phieng_world_new();
            let mut ids = [-9i32; 4];
            let added = phieng_world_add_bodies(world, descs.as_ptr(), 4, ids.as_mut_ptr());
            assert_eq!(added, 2);
            assert!(ids[0] >= 0 && ids[3] >= 0 && ids[0] != ids[3]);
            assert_eq!(ids[1], -1);
            assert_eq!(ids[2], -1);
            assert_eq!(phieng_world_body_count(world), 2);
            phieng_world_free(world);
        }
    }

    #[test]
    fn bulk_read_roundtrip_with_stale_sentinel() {
        // Index alignment is preserved: output[i] always belongs to
        // ids[i]; stale ids read NaN and do not abort the batch.
        unsafe {
            let (world, ball) = ball_drop();
            let ids = [0u32, ball, 999];
            let mut pos = [0.0f32; 6];
            let mut rot = [0.0f32; 3];
            let mut vel = [0.0f32; 6];
            let valid = phieng_world_get_transforms_bulk(
                world,
                ids.as_ptr(),
                3,
                pos.as_mut_ptr(),
                rot.as_mut_ptr(),
                vel.as_mut_ptr(),
            );
            assert_eq!(valid, 2);
            // Floor at (0,-1): slot 0 must match the scalar getter.
            let mut p0 = [0.0f32; 2];
            let mut r0 = 0.0f32;
            assert_eq!(
                phieng_world_get_body_transform(world, 0, p0.as_mut_ptr(), &mut r0),
                1
            );
            assert_eq!([pos[0], pos[1]], p0);
            assert_eq!(rot[0], r0);
            // Stale slot: all NaN.
            assert!(pos[4].is_nan() && pos[5].is_nan() && rot[2].is_nan());
            assert!(vel[4].is_nan() && vel[5].is_nan());
            // Null velocity pointer skips velocity readback.
            let valid2 = phieng_world_get_transforms_bulk(
                world,
                ids.as_ptr(),
                2,
                pos.as_mut_ptr(),
                rot.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(valid2, 2);
            // Null-safety.
            assert_eq!(
                phieng_world_get_transforms_bulk(
                    std::ptr::null_mut(),
                    ids.as_ptr(),
                    2,
                    pos.as_mut_ptr(),
                    rot.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                0
            );
            assert_eq!(
                phieng_world_get_transforms_bulk(
                    world,
                    std::ptr::null(),
                    2,
                    pos.as_mut_ptr(),
                    rot.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                0
            );
            assert_eq!(
                phieng_world_get_transforms_bulk(
                    world,
                    ids.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    rot.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn ambient_config_roundtrip() {
        unsafe {
            let world = phieng_world_new();
            let cfg = AmbientConfig2Ffi {
                spring_enabled: 1,
                spring_ax: 1.5,
                spring_ay: -2.5,
                spring_stiffness: 30.0,
                spring_damping: 0.5,
                spring_max_displacement: 4.0,
                drag_enabled: 1,
                drag_linear: 0.3,
                drag_quadratic: 0.7,
                air_density: 1.2,
                drag_angular: 0.05,
                wind_x: 1.0,
                wind_y: 0.0,
                turbulence_enabled: 1,
                turbulence_amplitude: 2.0,
                turbulence_frequency: 0.6,
                turbulence_time_scale: 1.3,
                turbulence_seed: 42,
                turbulence_octaves: 4,
                turbulence_divergence_free: 1,
            };
            assert_eq!(phieng_world_set_ambient_config(world, &cfg), 1);
            let mut out = std::mem::zeroed::<AmbientConfig2Ffi>();
            assert_eq!(phieng_world_get_ambient_config(world, &mut out), 1);
            assert_eq!(out.spring_enabled, 1);
            assert_eq!(out.spring_stiffness, 30.0);
            assert_eq!(out.spring_ax, 1.5);
            assert_eq!(out.wind_x, 1.0);
            assert_eq!(out.turbulence_seed, 42);
            assert_eq!(out.turbulence_octaves, 4);
            assert_eq!(out.turbulence_divergence_free, 1);
            // Null tolerance.
            assert_eq!(
                phieng_world_set_ambient_config(std::ptr::null_mut(), &cfg),
                0
            );
            assert_eq!(
                phieng_world_get_ambient_config(world, std::ptr::null_mut()),
                0
            );
            // Negative coefficient rejected without mutating the world.
            let bad = AmbientConfig2Ffi {
                spring_stiffness: -1.0,
                ..cfg
            };
            assert_eq!(phieng_world_set_ambient_config(world, &bad), 0);
            let mut after = std::mem::zeroed::<AmbientConfig2Ffi>();
            assert_eq!(phieng_world_get_ambient_config(world, &mut after), 1);
            assert_eq!(after.spring_stiffness, 30.0);
            // Clock round-trip.
            assert_eq!(phieng_world_set_time(world, 3.5), 1);
            let mut t = 0.0f32;
            assert_eq!(phieng_world_get_time(world, &mut t), 1);
            assert_eq!(t, 3.5);
            assert_eq!(phieng_world_set_time(world, f32::NAN), 0);
            phieng_world_free(world);
        }
    }

    #[test]
    fn ccd_config_roundtrip() {
        unsafe {
            let world = phieng_world_new();
            let cfg = CcdConfigFfi {
                enabled: 1,
                tier: 2, // FullGJK
                max_movement: 0.05,
                max_substeps: 64,
            };
            assert_eq!(phieng_world_set_ccd_config(world, &cfg), 1);
            let mut out = std::mem::zeroed::<CcdConfigFfi>();
            assert_eq!(phieng_world_get_ccd_config(world, &mut out), 1);
            assert_eq!(out.enabled, 1);
            assert_eq!(out.tier, 2);
            assert_eq!(out.max_movement, 0.05);
            assert_eq!(out.max_substeps, 64);
            // Null tolerance.
            assert_eq!(phieng_world_set_ccd_config(std::ptr::null_mut(), &cfg), 0);
            assert_eq!(phieng_world_get_ccd_config(world, std::ptr::null_mut()), 0);
            // Bad inputs rejected without mutating the world.
            for bad in [
                CcdConfigFfi {
                    max_movement: 0.0,
                    ..cfg
                },
                CcdConfigFfi {
                    max_movement: -1.0,
                    ..cfg
                },
                CcdConfigFfi {
                    max_movement: f32::NAN,
                    ..cfg
                },
                CcdConfigFfi {
                    max_substeps: 0,
                    ..cfg
                },
                CcdConfigFfi { tier: 99, ..cfg },
            ] {
                assert_eq!(phieng_world_set_ccd_config(world, &bad), 0);
            }
            let mut after = std::mem::zeroed::<CcdConfigFfi>();
            assert_eq!(phieng_world_get_ccd_config(world, &mut after), 1);
            assert_eq!(after.max_movement, 0.05);
            assert_eq!(after.max_substeps, 64);
            phieng_world_free(world);

            // 3D mirror.
            let world3 = phieng_world3_new();
            assert_eq!(phieng_world3_set_ccd_config(world3, &cfg), 1);
            let mut out3 = std::mem::zeroed::<CcdConfigFfi>();
            assert_eq!(phieng_world3_get_ccd_config(world3, &mut out3), 1);
            assert_eq!(out3.enabled, 1);
            assert_eq!(out3.max_movement, 0.05);
            assert_eq!(phieng_world3_set_ccd_config(std::ptr::null_mut(), &cfg), 0);
            assert_eq!(
                phieng_world3_get_ccd_config(world3, std::ptr::null_mut()),
                0
            );
            phieng_world3_free(world3);
        }
    }

    #[test]
    fn fluid_config_roundtrip() {
        unsafe {
            let world = phieng_world_new();
            let cfg = FluidConfig2Ffi {
                particle_mass: 0.125,
                rest_density: 2.0,
                smoothing_length: 0.75,
                stiffness: 200.0,
                gamma: 7.0,
                artificial_viscosity: 0.1,
                surface_tension: 0.05,
                cfl: 0.4,
                max_substeps: 4,
                adaptive_resolution: true,
                split_density_ratio: 0.4,
                merge_density_ratio: 1.8,
                boundary_particles: true,
                viscosity: 0.15,
                drag: 2.0,
                wall_friction: 0.3,
                phase_count: 2,
                phase_rest_density: [1.0, 2.0, 1.0, 1.0],
                phase_mass: [0.125, 0.25, 0.125, 0.125],
                max_velocity: 50.0,
                particle_radius: 0.25,
                enabled: true,
            };
            assert_eq!(phieng_world_set_fluid_config(world, &cfg), 1);
            let mut out = std::mem::zeroed::<FluidConfig2Ffi>();
            assert_eq!(phieng_world_get_fluid_config(world, &mut out), 1);
            assert_eq!(out.particle_mass, 0.125);
            assert_eq!(out.rest_density, 2.0);
            assert_eq!(out.smoothing_length, 0.75);
            assert_eq!(out.gamma, 7.0);
            assert_eq!(out.cfl, 0.4);
            assert_eq!(out.max_substeps, 4);
            assert!(out.adaptive_resolution);
            assert!(out.boundary_particles);
            assert_eq!(out.phase_count, 2);
            assert_eq!(out.phase_rest_density[1], 2.0);
            assert_eq!(out.phase_mass[1], 0.25);
            assert_eq!(out.max_velocity, 50.0);
            // Null tolerance.
            assert_eq!(phieng_world_set_fluid_config(std::ptr::null_mut(), &cfg), 0);
            assert_eq!(
                phieng_world_get_fluid_config(world, std::ptr::null_mut()),
                0
            );
            // Bad inputs rejected without mutating the world.
            for bad in [
                FluidConfig2Ffi {
                    particle_mass: -1.0,
                    ..cfg
                },
                FluidConfig2Ffi {
                    rest_density: 0.0,
                    ..cfg
                },
                FluidConfig2Ffi { gamma: 0.5, ..cfg },
                FluidConfig2Ffi {
                    phase_count: 0,
                    ..cfg
                },
                FluidConfig2Ffi {
                    phase_count: 5,
                    ..cfg
                },
                FluidConfig2Ffi {
                    max_substeps: 0,
                    ..cfg
                },
                FluidConfig2Ffi { cfl: 1.5, ..cfg },
                FluidConfig2Ffi {
                    artificial_viscosity: 3.0,
                    ..cfg
                },
                FluidConfig2Ffi {
                    wall_friction: 2.0,
                    ..cfg
                },
                FluidConfig2Ffi {
                    phase_rest_density: [0.0, 1.0, 1.0, 1.0],
                    ..cfg
                },
            ] {
                assert_eq!(phieng_world_set_fluid_config(world, &bad), 0);
            }
            let mut after = std::mem::zeroed::<FluidConfig2Ffi>();
            assert_eq!(phieng_world_get_fluid_config(world, &mut after), 1);
            assert_eq!(after.particle_mass, 0.125);
            assert_eq!(after.phase_count, 2);
            // add_phase.
            let pts: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
            let phases: [u8; 3] = [0, 1, 0];
            assert_eq!(
                phieng_world_fluid_add_phase(world, pts.as_ptr(), phases.as_ptr(), 3),
                3
            );
            assert_eq!(phieng_world_fluid_count(world), 3);
            assert_eq!(
                phieng_world_fluid_add_phase(
                    std::ptr::null_mut(),
                    pts.as_ptr(),
                    phases.as_ptr(),
                    3
                ),
                0
            );
            assert_eq!(
                phieng_world_fluid_add_phase(world, std::ptr::null(), phases.as_ptr(), 3),
                0
            );
            assert_eq!(
                phieng_world_fluid_add_phase(world, pts.as_ptr(), std::ptr::null(), 3),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn fluid_config_roundtrip_3d() {
        unsafe {
            let world = phieng_world3_new();
            let cfg = FluidConfig3Ffi {
                particle_mass: 0.125,
                rest_density: 2.0,
                smoothing_length: 0.75,
                stiffness: 200.0,
                gamma: 7.0,
                artificial_viscosity: 0.1,
                surface_tension: 0.05,
                cfl: 0.4,
                max_substeps: 4,
                adaptive_resolution: true,
                split_density_ratio: 0.4,
                merge_density_ratio: 1.8,
                boundary_particles: true,
                viscosity: 0.15,
                drag: 2.0,
                wall_friction: 0.3,
                phase_count: 2,
                phase_rest_density: [1.0, 2.0, 1.0, 1.0],
                phase_mass: [0.125, 0.25, 0.125, 0.125],
                max_velocity: 50.0,
                particle_radius: 0.25,
                enabled: true,
            };
            assert_eq!(phieng_world3_set_fluid_config(world, &cfg), 1);
            let mut out = std::mem::zeroed::<FluidConfig3Ffi>();
            assert_eq!(phieng_world3_get_fluid_config(world, &mut out), 1);
            assert_eq!(out.phase_count, 2);
            assert_eq!(out.gamma, 7.0);
            assert_eq!(
                phieng_world3_set_fluid_config(std::ptr::null_mut(), &cfg),
                0
            );
            assert_eq!(
                phieng_world3_get_fluid_config(world, std::ptr::null_mut()),
                0
            );
            let bad = FluidConfig3Ffi {
                particle_mass: 0.0,
                ..cfg
            };
            assert_eq!(phieng_world3_set_fluid_config(world, &bad), 0);
            let pts: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
            let phases: [u8; 3] = [1, 0, 1];
            assert_eq!(
                phieng_world3_fluid_add_phase(world, pts.as_ptr(), phases.as_ptr(), 3),
                3
            );
            assert_eq!(phieng_world3_fluid_count(world), 3);
            assert_eq!(
                phieng_world3_fluid_add_phase(
                    std::ptr::null_mut(),
                    pts.as_ptr(),
                    phases.as_ptr(),
                    3
                ),
                0
            );
            phieng_world3_free(world);
        }
    }

    #[test]
    fn gravity_config_roundtrip() {
        // Set/get round-trips every mode + integrator; bad inputs are
        // rejected without mutating the world.
        unsafe {
            let world = phieng_world_new();
            for mode in 0..=3u32 {
                for integ in 0..=1u32 {
                    assert_eq!(
                        phieng_world_set_gravity_config(
                            world, mode, integ, 2.0, 0.1, 1.0, 2.0, 3.0, 0.5
                        ),
                        1
                    );
                    let mut m = 99u32;
                    let mut it = 99u32;
                    let mut v = [0.0f32; 6];
                    assert_eq!(
                        phieng_world_get_gravity_config(world, &mut m, &mut it, v.as_mut_ptr()),
                        1
                    );
                    assert_eq!((m, it), (mode, integ));
                    assert_eq!(v, [2.0, 0.1, 1.0, 2.0, 3.0, 0.5]);
                }
            }
            // Unknown tags, non-finite G, negative softening/theta: reject.
            assert_eq!(
                phieng_world_set_gravity_config(world, 99, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                0
            );
            assert_eq!(
                phieng_world_set_gravity_config(world, 0, 7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                0
            );
            assert_eq!(
                phieng_world_set_gravity_config(world, 2, 0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.5),
                0
            );
            assert_eq!(
                phieng_world_set_gravity_config(world, 2, 0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5),
                0
            );
            assert_eq!(
                phieng_world_set_gravity_config(world, 3, 0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5),
                0
            );
            // Rejected sets leave the last good config in place.
            let mut m = 99u32;
            let mut it = 99u32;
            let mut v = [0.0f32; 6];
            assert_eq!(
                phieng_world_get_gravity_config(world, &mut m, &mut it, v.as_mut_ptr()),
                1
            );
            assert_eq!((m, it), (3, 1));
            // Null-safety.
            assert_eq!(
                phieng_world_set_gravity_config(
                    std::ptr::null_mut(),
                    2,
                    0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5
                ),
                0
            );
            assert_eq!(
                phieng_world_get_gravity_config(
                    std::ptr::null_mut(),
                    &mut m,
                    &mut it,
                    v.as_mut_ptr()
                ),
                0
            );
            assert_eq!(
                phieng_world_get_gravity_config(
                    world,
                    std::ptr::null_mut(),
                    &mut it,
                    v.as_mut_ptr()
                ),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn gravity_scale_roundtrip() {
        unsafe {
            let (world, ball) = ball_drop();
            // Default scale is full coupling.
            let mut s = 0.0f32;
            assert_eq!(phieng_world_get_body_gravity_scale(world, ball, &mut s), 1);
            assert_eq!(s, 1.0);
            // Set/get incl. immune (0) and repulsive (negative).
            for scale in [0.0f32, 0.5, 2.0, -1.0] {
                assert_eq!(phieng_world_set_body_gravity_scale(world, ball, scale), 1);
                assert_eq!(phieng_world_get_body_gravity_scale(world, ball, &mut s), 1);
                assert_eq!(s, scale);
            }
            // Unknown id, non-finite scale, nulls: reject.
            assert_eq!(phieng_world_set_body_gravity_scale(world, 999, 1.0), 0);
            assert_eq!(
                phieng_world_set_body_gravity_scale(world, ball, f32::INFINITY),
                0
            );
            assert_eq!(phieng_world_get_body_gravity_scale(world, 999, &mut s), 0);
            assert_eq!(
                phieng_world_get_body_gravity_scale(world, ball, std::ptr::null_mut()),
                0
            );
            assert_eq!(
                phieng_world_set_body_gravity_scale(std::ptr::null_mut(), ball, 1.0),
                0
            );
            phieng_world_free(world);
        }
    }

    #[test]
    fn sleep_thresholds_roundtrip() {
        unsafe {
            let world = phieng_world_new();
            // Defaults are the 0.3.1 sleep thresholds.
            let mut t = [0.0f32; 2];
            assert_eq!(phieng_world_get_sleep_thresholds(world, t.as_mut_ptr()), 1);
            assert_eq!(t, [0.05, 0.05]);
            // Zero disables sleeping; negatives and NaN reject.
            assert_eq!(phieng_world_set_sleep_thresholds(world, 0.0, 0.0), 1);
            assert_eq!(phieng_world_get_sleep_thresholds(world, t.as_mut_ptr()), 1);
            assert_eq!(t, [0.0, 0.0]);
            assert_eq!(phieng_world_set_sleep_thresholds(world, -1.0, 0.0), 0);
            assert_eq!(phieng_world_set_sleep_thresholds(world, 0.0, f32::NAN), 0);
            assert_eq!(phieng_world_get_sleep_thresholds(world, t.as_mut_ptr()), 1);
            assert_eq!(t, [0.0, 0.0]);
            // Null-safety.
            assert_eq!(
                phieng_world_set_sleep_thresholds(std::ptr::null_mut(), 0.0, 0.0),
                0
            );
            assert_eq!(
                phieng_world_get_sleep_thresholds(std::ptr::null_mut(), t.as_mut_ptr()),
                0
            );
            assert_eq!(
                phieng_world_get_sleep_thresholds(world, std::ptr::null_mut()),
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
    fn bulk3_add_and_read_roundtrip() {
        // 3D bulk creation matches scalar ids; bulk read matches the
        // scalar transform getter slot-for-slot, with NaN for stale ids.
        unsafe {
            use phieng_dynamics::World3;
            let mk = |x: f32| BodyDesc3D {
                pos_x: x,
                pos_y: 2.0,
                pos_z: 0.0,
                shape_tag: SHAPE3_SPHERE,
                a: 0.5,
                b: 0.0,
                c: 0.0,
                density: 1.0,
                is_static: 0,
                _pad: [0; 3],
                friction: 0.5,
                restitution: 0.0,
            };
            let w1: *mut World3 = phieng_world3_new();
            let i0 = phieng_world3_add_body(
                w1,
                0.0,
                2.0,
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
            let w2: *mut World3 = phieng_world3_new();
            let descs = [mk(0.0), mk(1.0)];
            let mut ids = [-9i32; 2];
            let added = phieng_world3_add_bodies(w2, descs.as_ptr(), 2, ids.as_mut_ptr());
            assert_eq!(added, 2);
            assert_eq!(ids, [i0, i0 + 1]);
            // Bulk read vs scalar read on the same body.
            let q = [ids[1] as u32, 999];
            let mut pos = [0.0f32; 6];
            let mut quat = [0.0f32; 8];
            let valid = phieng_world3_get_transforms_bulk(
                w2,
                q.as_ptr(),
                2,
                pos.as_mut_ptr(),
                quat.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(valid, 1);
            let mut sp = [0.0f32; 3];
            let mut sq = [0.0f32; 4];
            assert_eq!(
                phieng_world3_get_body_transform(
                    w2,
                    ids[1] as u32,
                    sp.as_mut_ptr(),
                    sq.as_mut_ptr()
                ),
                1
            );
            assert_eq!([pos[0], pos[1], pos[2]], sp);
            assert_eq!([quat[0], quat[1], quat[2], quat[3]], sq);
            // Slot 0 (q[0] = live id) is finite; slot 1 (q[1] = 999)
            // is the NaN stale sentinel.
            assert!(pos[0].is_finite() && quat[0].is_finite());
            assert!(pos[3].is_nan() && quat[4].is_nan());
            // Null-safety.
            assert_eq!(
                phieng_world3_get_transforms_bulk(
                    std::ptr::null_mut(),
                    q.as_ptr(),
                    2,
                    pos.as_mut_ptr(),
                    quat.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                0
            );
            phieng_world3_free(w1);
            phieng_world3_free(w2);
        }
    }

    #[test]
    fn gravity3_config_roundtrip() {
        unsafe {
            use phieng_dynamics::World3;
            let world: *mut World3 = phieng_world3_new();
            assert_eq!(
                phieng_world3_set_gravity_config(world, 3, 1, 1.5, 0.2, 1.0, 2.0, 3.0, 0.0, 0.7),
                1
            );
            let mut m = 99u32;
            let mut it = 99u32;
            let mut v = [0.0f32; 7];
            assert_eq!(
                phieng_world3_get_gravity_config(world, &mut m, &mut it, v.as_mut_ptr()),
                1
            );
            assert_eq!((m, it), (3, 1));
            assert_eq!(v, [1.5, 0.2, 1.0, 2.0, 3.0, 0.0, 0.7]);
            // Rejections leave state unchanged; unknown ids fail clean.
            assert_eq!(
                phieng_world3_set_gravity_config(world, 9, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                0
            );
            assert_eq!(
                phieng_world3_get_gravity_config(
                    std::ptr::null_mut(),
                    &mut m,
                    &mut it,
                    v.as_mut_ptr()
                ),
                0
            );
            // Scale round-trip incl. immune, negative, rejections.
            let ball = phieng_world3_add_body(
                world,
                0.0,
                2.0,
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
            assert!(ball >= 0);
            let mut s = 0.0f32;
            assert_eq!(
                phieng_world3_get_body_gravity_scale(world, ball as u32, &mut s),
                1
            );
            assert_eq!(s, 1.0);
            assert_eq!(
                phieng_world3_set_body_gravity_scale(world, ball as u32, 0.0),
                1
            );
            assert_eq!(
                phieng_world3_get_body_gravity_scale(world, ball as u32, &mut s),
                1
            );
            assert_eq!(s, 0.0);
            assert_eq!(phieng_world3_set_body_gravity_scale(world, 999, 1.0), 0);
            assert_eq!(
                phieng_world3_set_body_gravity_scale(world, ball as u32, f32::NAN),
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
