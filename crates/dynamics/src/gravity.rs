//! Gravitational physics: uniform fields, central attractors, and full
//! N-body pairwise Newtonian gravity, integrated with semi-implicit
//! Euler or velocity-Verlet leapfrog.
//!
//! Design notes (see [`crate::World2::step`] for the pipeline position):
//! - The uniform `world.gravity` vector is always applied as a background
//!   field (superposition): Central and N-body modes add their forces on
//!   top of it. The SPH fluid keeps using only the uniform field.
//! - Per-body `gravity_scale` (default 1.0, stored on each body)
//!   multiplies every gravitational acceleration the body feels.
//!   `0.0` makes a body immune (it still collides and still *exerts*
//!   force by its mass); negative values repel. Only finiteness is
//!   enforced — the scale is otherwise the caller's physics.
//! - Exertion always uses the body's stored `mass` (statics included,
//!   so a massive static body is a free central planet in N-body mode).
//!   Shapeless bodies carry their default unit mass.
//! - Massless probes need no validation exemption: any positive density
//!   works (the `MIN_DENSITY` floor makes the exerted force negligible
//!   while the felt acceleration `G*M/r^2` is mass-independent, exactly
//!   as the equivalence principle demands). Combine tiny density with
//!   `gravity_scale` for test particles.
//! - Determinism rules for the pairwise path: index-ordered `i<j` loops,
//!   symmetric `+-F` application (momentum conserved to float rounding),
//!   no hash-map iteration, no RNG, no wall clock. Bit-identical across
//!   runs by construction; cross-language equality is *not* claimed.
//! - Orbital scenes must zero damping and disable sleep (thresholds to
//!   0.0); in Verlet mode linear damping rides the kicks as symmetric
//!   drag instead of the Euler multiplicative decay (documented
//!   semantic difference, same `linear_damping` field).
//! - Softening `ε` regularizes `1/r^2` at close range
//!   (`F = G*m1*m2*d/(r^2+ε^2)^{3/2}`). Coincident pairs
//!   (`r^2+ε^2 < EPS_SQ`) are skipped — no invented force; the contact
//!   pipeline owns that regime.

use super::body::{RigidBody2, RigidBody3};
use glam::{Vec2, Vec3};
use phieng_math::EPS_SQ;

/// How gravitational acceleration is computed each step.
/// Discriminants are part of the snapshot v4 file format — never reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GravityMode {
    /// Classic uniform field (`world.gravity`), scaled per body.
    /// Bit-identical to pre-gravity behavior when every `gravity_scale`
    /// is 1.0. O(1) per body.
    Uniform = 0,
    /// Single fixed attractor: `a = mu*d/(|d|^2+ε^2)^{3/2}` toward
    /// `attractor`, times per-body scale. O(n). Kepler orbits.
    Central = 1,
    /// Full pairwise Newtonian gravity over all body pairs. O(n^2).
    NBody = 2,
    /// Barnes-Hut approximation of [`GravityMode::NBody`]: distant
    /// groups collapse to their center of mass when the cell's
    /// size-to-distance ratio is below `theta` (see
    /// [`crate::barnes_hut`]). O(n log n). Approximate by design —
    /// same-mode runs are bit-identical, but exact and BH trajectories
    /// never match each other. Discriminant appended (never reorder:
    /// discriminants are file format).
    BarnesHut = 3,
}

/// Time-integration scheme for the gravity/free-flight part of the step.
/// The contact/joint solver runs unchanged inside either scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Integrator {
    /// Semi-implicit Euler (status quo): one full-step gravity kick
    /// pre-solve, damping inside `integrate`. Default.
    SemiImplicitEuler = 0,
    /// Kick-drift-kick leapfrog: half-step gravity kick pre-solve,
    /// drift (positions only — linear damping rides the kicks as
    /// symmetric drag), forces recomputed at the new positions,
    /// second half-step kick. Better energy behavior for orbits.
    VelocityVerlet = 1,
}

/// Gravity configuration for a 2D world. All fields participate in
/// snapshot schema v4 (see [`crate::snapshot`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityConfig2 {
    /// Force model (default [`GravityMode::Uniform`]).
    pub mode: GravityMode,
    /// Integration scheme (default [`Integrator::SemiImplicitEuler`]).
    pub integrator: Integrator,
    /// Gravitational constant in scene units (game units: 1.0 is the
    /// sane default, *not* SI 6.674e-11). Finite only.
    pub big_g: f32,
    /// Softening length; must be finite and >= 0. Regularizes
    /// close encounters; coincident pairs are skipped regardless.
    pub softening: f32,
    /// Fixed attractor point (Central mode only).
    pub attractor: Vec2,
    /// `G*M` of the fixed attractor (Central mode only).
    pub attractor_mu: f32,
    /// Barnes-Hut opening criterion (BarnesHut mode only): a cell
    /// approximates as one mass when `size / distance < theta`.
    /// Finite and >= 0; 0 degenerates to exact descent (slow but
    /// correct). Classic default 0.5.
    pub theta: f32,
}

impl Default for GravityConfig2 {
    fn default() -> Self {
        Self {
            mode: GravityMode::Uniform,
            integrator: Integrator::SemiImplicitEuler,
            big_g: 1.0,
            softening: 0.0,
            attractor: Vec2::ZERO,
            attractor_mu: 0.0,
            theta: 0.5,
        }
    }
}

impl GravityConfig2 {
    /// True for a freshly constructed config: uniform field, Euler,
    /// unit G, no softening, no attractor. Snapshot writers emit
    /// schema v3 for worlds satisfying this (plus unit body scales),
    /// so pre-gravity files stay byte-identical.
    pub fn is_default(&self) -> bool {
        self.mode == GravityMode::Uniform
            && self.integrator == Integrator::SemiImplicitEuler
            && self.big_g == 1.0
            && self.softening == 0.0
            && self.attractor == Vec2::ZERO
            && self.attractor_mu == 0.0
            && self.theta == 0.5
    }
}

/// Gravity configuration for a 3D world. Same semantics as
/// [`GravityConfig2`] with a 3D attractor point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityConfig3 {
    /// Force model (default [`GravityMode::Uniform`]).
    pub mode: GravityMode,
    /// Integration scheme (default [`Integrator::SemiImplicitEuler`]).
    pub integrator: Integrator,
    /// Gravitational constant in scene units. Finite only.
    pub big_g: f32,
    /// Softening length; finite and >= 0.
    pub softening: f32,
    /// Fixed attractor point (Central mode only).
    pub attractor: Vec3,
    /// `G*M` of the fixed attractor (Central mode only).
    pub attractor_mu: f32,
    /// Barnes-Hut opening criterion (BarnesHut mode only).
    /// See [`GravityConfig2::theta`].
    pub theta: f32,
}

impl Default for GravityConfig3 {
    fn default() -> Self {
        Self {
            mode: GravityMode::Uniform,
            integrator: Integrator::SemiImplicitEuler,
            big_g: 1.0,
            softening: 0.0,
            attractor: Vec3::ZERO,
            attractor_mu: 0.0,
            theta: 0.5,
        }
    }
}

impl GravityConfig3 {
    /// See [`GravityConfig2::is_default`].
    pub fn is_default(&self) -> bool {
        self.mode == GravityMode::Uniform
            && self.integrator == Integrator::SemiImplicitEuler
            && self.big_g == 1.0
            && self.softening == 0.0
            && self.attractor == Vec3::ZERO
            && self.attractor_mu == 0.0
            && self.theta == 0.5
    }
}

/// Acceleration of a point at `pos` toward a central attractor of
/// strength `mu`, softened by `ε`. Returns zero for coincident
/// points (contact pipeline owns that regime).
pub fn central_accel_2(pos: Vec2, attractor: Vec2, mu: f32, softening: f32) -> Vec2 {
    let d = attractor - pos;
    let d2 = d.length_squared() + softening * softening;
    if d2 < EPS_SQ {
        return Vec2::ZERO;
    }
    let inv = 1.0 / d2.sqrt();
    d * (mu * inv / d2)
}

/// 3D counterpart of [`central_accel_2`].
pub fn central_accel_3(pos: Vec3, attractor: Vec3, mu: f32, softening: f32) -> Vec3 {
    let d = attractor - pos;
    let d2 = d.length_squared() + softening * softening;
    if d2 < EPS_SQ {
        return Vec3::ZERO;
    }
    let inv = 1.0 / d2.sqrt();
    d * (mu * inv / d2)
}

/// Symmetric pairwise accelerations for bodies `i` and `j`
/// (Newton's third law): returns `(a_i, a_j)`, unscaled — the caller
/// applies per-body `gravity_scale`. Zero for coincident pairs.
pub fn pair_accelerations_2(
    pi: Vec2,
    mi: f32,
    pj: Vec2,
    mj: f32,
    big_g: f32,
    softening: f32,
) -> (Vec2, Vec2) {
    let delta = pj - pi;
    let d2 = delta.length_squared() + softening * softening;
    if d2 < EPS_SQ {
        return (Vec2::ZERO, Vec2::ZERO);
    }
    let inv = 1.0 / d2.sqrt();
    let s = big_g * inv / d2;
    (delta * (s * mj), delta * (-(s * mi)))
}

/// 3D counterpart of [`pair_accelerations_2`].
pub fn pair_accelerations_3(
    pi: Vec3,
    mi: f32,
    pj: Vec3,
    mj: f32,
    big_g: f32,
    softening: f32,
) -> (Vec3, Vec3) {
    let delta = pj - pi;
    let d2 = delta.length_squared() + softening * softening;
    if d2 < EPS_SQ {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let inv = 1.0 / d2.sqrt();
    let s = big_g * inv / d2;
    (delta * (s * mj), delta * (-(s * mi)))
}

/// Gravity velocity kick over all 2D bodies: uniform background field
/// plus Central/N-body terms per `cfg`, times per-body scale, over
/// interval `h`. When `drag` is set (Verlet mode), linear damping rides
/// the kick as symmetric `-c*v*h`; otherwise damping stays in
/// `integrate` (Euler path — bit-identical to the legacy loop when the
/// config is default). Index order, no allocation, deterministic.
/// Statics never move but still exert by stored mass.
pub fn gravity_kick_2(
    bodies: &mut [RigidBody2],
    cfg: &GravityConfig2,
    uniform: Vec2,
    h: f32,
    drag: bool,
) {
    // Uniform background field first (single pass; for default configs
    // this is exactly the legacy `velocity += gravity * dt` loop:
    // `scale * h` with scale == 1.0 is bit-identical to `* h`).
    for b in bodies.iter_mut() {
        if b.is_static || b.is_sleeping {
            continue;
        }
        let mut dv = uniform * (b.gravity_scale * h);
        if drag {
            dv -= b.velocity * (b.linear_damping * h);
        }
        b.velocity += dv;
    }
    match cfg.mode {
        GravityMode::Uniform => {}
        GravityMode::Central => {
            for b in bodies.iter_mut() {
                if b.is_static || b.is_sleeping {
                    continue;
                }
                let a = central_accel_2(
                    b.transform.position,
                    cfg.attractor,
                    cfg.attractor_mu,
                    cfg.softening,
                );
                let mut dv = a * (b.gravity_scale * h);
                if drag {
                    dv -= b.velocity * (b.linear_damping * h);
                }
                b.velocity += dv;
            }
        }
        GravityMode::NBody => {
            if cfg.big_g != 0.0 {
                let n = bodies.len();
                for i in 0..n {
                    for j in (i + 1)..n {
                        // Disjoint mutable borrows, MSRV-safe
                        // (`get_many_mut` needs newer rustc).
                        let (lo, hi) = bodies.split_at_mut(j);
                        let (bi, bj) = (&mut lo[i], &mut hi[0]);
                        let (ai, aj) = pair_accelerations_2(
                            bi.transform.position,
                            bi.mass,
                            bj.transform.position,
                            bj.mass,
                            cfg.big_g,
                            cfg.softening,
                        );
                        if !bi.is_static && !bi.is_sleeping {
                            let mut dv = ai * (bi.gravity_scale * h);
                            if drag {
                                dv -= bi.velocity * (bi.linear_damping * h);
                            }
                            bi.velocity += dv;
                        }
                        if !bj.is_static && !bj.is_sleeping {
                            let mut dv = aj * (bj.gravity_scale * h);
                            if drag {
                                dv -= bj.velocity * (bj.linear_damping * h);
                            }
                            bj.velocity += dv;
                        }
                    }
                }
            }
        }
        GravityMode::BarnesHut => {
            // Approximate N-body via the deterministic Barnes-Hut tree
            // (see `crate::barnes_hut`): the uniform pass above already
            // ran; the tree replaces only the pairwise loop. Approximate
            // by design — same-mode runs are bit-identical, but exact
            // and BH trajectories never match each other.
            if cfg.big_g != 0.0 {
                super::barnes_hut::apply_bh_forces_2(
                    bodies,
                    cfg.big_g,
                    cfg.softening,
                    cfg.theta,
                    h,
                    drag,
                );
            }
        }
    }
}

/// 3D counterpart of [`gravity_kick_2`].
pub fn gravity_kick_3(
    bodies: &mut [RigidBody3],
    cfg: &GravityConfig3,
    uniform: Vec3,
    h: f32,
    drag: bool,
) {
    for b in bodies.iter_mut() {
        if b.is_static || b.is_sleeping {
            continue;
        }
        let mut dv = uniform * (b.gravity_scale * h);
        if drag {
            dv -= b.velocity * (b.linear_damping * h);
        }
        b.velocity += dv;
    }
    match cfg.mode {
        GravityMode::Uniform => {}
        GravityMode::Central => {
            for b in bodies.iter_mut() {
                if b.is_static || b.is_sleeping {
                    continue;
                }
                let a = central_accel_3(
                    b.transform.position,
                    cfg.attractor,
                    cfg.attractor_mu,
                    cfg.softening,
                );
                let mut dv = a * (b.gravity_scale * h);
                if drag {
                    dv -= b.velocity * (b.linear_damping * h);
                }
                b.velocity += dv;
            }
        }
        GravityMode::NBody => {
            if cfg.big_g != 0.0 {
                let n = bodies.len();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let (lo, hi) = bodies.split_at_mut(j);
                        let (bi, bj) = (&mut lo[i], &mut hi[0]);
                        let (ai, aj) = pair_accelerations_3(
                            bi.transform.position,
                            bi.mass,
                            bj.transform.position,
                            bj.mass,
                            cfg.big_g,
                            cfg.softening,
                        );
                        if !bi.is_static && !bi.is_sleeping {
                            let mut dv = ai * (bi.gravity_scale * h);
                            if drag {
                                dv -= bi.velocity * (bi.linear_damping * h);
                            }
                            bi.velocity += dv;
                        }
                        if !bj.is_static && !bj.is_sleeping {
                            let mut dv = aj * (bj.gravity_scale * h);
                            if drag {
                                dv -= bj.velocity * (bj.linear_damping * h);
                            }
                            bj.velocity += dv;
                        }
                    }
                }
            }
        }
        GravityMode::BarnesHut => {
            // Same contract as the 2D arm (see above).
            if cfg.big_g != 0.0 {
                super::barnes_hut::apply_bh_forces_3(
                    bodies,
                    cfg.big_g,
                    cfg.softening,
                    cfg.theta,
                    h,
                    drag,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_known_answer() {
        // mu = 100 at the origin, point at (10, 0):
        // a = 100 * (-10, 0) / 1000 = (-1, 0).
        let a = central_accel_2(Vec2::new(10.0, 0.0), Vec2::ZERO, 100.0, 0.0);
        assert!((a - Vec2::new(-1.0, 0.0)).length() < 1e-6);
        let a3 = central_accel_3(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, 100.0, 0.0);
        assert!((a3 - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_central_coincident_is_zero() {
        assert_eq!(
            central_accel_2(Vec2::ZERO, Vec2::ZERO, 100.0, 0.0),
            Vec2::ZERO
        );
    }

    #[test]
    fn test_pair_momentum_symmetric() {
        // m_i * a_i + m_j * a_j must vanish to float rounding.
        let (ai, aj) =
            pair_accelerations_2(Vec2::new(0.0, 0.0), 3.0, Vec2::new(4.0, 1.0), 7.0, 1.0, 0.0);
        let total = ai * 3.0 + aj * 7.0;
        assert!(total.length() < 1e-5, "momentum leak: {total:?}");
    }

    #[test]
    fn test_pair_coincident_is_zero() {
        let (ai, aj) = pair_accelerations_2(Vec2::ONE, 1.0, Vec2::ONE, 1.0, 1.0, 0.0);
        assert_eq!(ai, Vec2::ZERO);
        assert_eq!(aj, Vec2::ZERO);
    }

    #[test]
    fn test_config_defaults() {
        assert!(GravityConfig2::default().is_default());
        assert!(GravityConfig3::default().is_default());
        let c = GravityConfig2 {
            mode: GravityMode::NBody,
            ..Default::default()
        };
        assert!(!c.is_default());
        assert_eq!(GravityMode::NBody as u32, 2);
        assert_eq!(GravityMode::BarnesHut as u32, 3);
        assert_eq!(Integrator::VelocityVerlet as u32, 1);
    }
}
