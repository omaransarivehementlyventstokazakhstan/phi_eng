//! Electrostatics: Coulomb pair forces, the electric field and the
//! potential.
//!
//! Every routine is `O(n)` per query or `O(n^2)` for the full pair sum.
//! The pair force is computed once and applied with opposite signs, so
//! momentum is conserved to float rounding even when the two bodies have
//! different masses.

use glam::{Vec2, Vec3};
use phieng_math::EPS_SQ;

/// The force on a point charge at `pi` from a point charge at `pj`.
///
/// A positive result points from `pi` toward `pj` for like charges; the
/// sign flips when the charges have opposite signs. Coincident pairs
/// (after softening) report zero: the contact pipeline owns that regime.
#[inline]
pub fn pair_force_2(pi: Vec2, qi: f32, pj: Vec2, qj: f32, k: f32, softening: f32) -> Vec2 {
    let d = pi - pj;
    let r2 = d.length_squared() + softening * softening;
    if r2 < EPS_SQ || !r2.is_finite() {
        return Vec2::ZERO;
    }
    let s = k * qi * qj;
    if !s.is_finite() {
        return Vec2::ZERO;
    }
    let inv = 1.0 / r2.sqrt();
    d * (s * inv / r2)
}

/// 3D counterpart of [`pair_force_2`].
#[inline]
pub fn pair_force_3(pi: Vec3, qi: f32, pj: Vec3, qj: f32, k: f32, softening: f32) -> Vec3 {
    let d = pi - pj;
    let r2 = d.length_squared() + softening * softening;
    if r2 < EPS_SQ || !r2.is_finite() {
        return Vec3::ZERO;
    }
    let s = k * qi * qj;
    if !s.is_finite() {
        return Vec3::ZERO;
    }
    let inv = 1.0 / r2.sqrt();
    d * (s * inv / r2)
}

/// Fill `out` with the net Coulomb force on each of `pos`.
///
/// `out` and `pos` must have the same length; `out` is overwritten (not
/// accumulated) so the caller can reuse it every step. The loop is the
/// index-ordered `i < j` order the engine pins everywhere.
pub fn pairwise_forces_2(pos: &[Vec2], q: &[f32], k: f32, softening: f32, out: &mut [Vec2]) {
    let n = pos.len().min(q.len()).min(out.len());
    for v in out.iter_mut().take(n) {
        *v = Vec2::ZERO;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let f = pair_force_2(pos[i], q[i], pos[j], q[j], k, softening);
            out[i] += f;
            out[j] -= f;
        }
    }
}

/// 3D counterpart of [`pairwise_forces_2`].
pub fn pairwise_forces_3(pos: &[Vec3], q: &[f32], k: f32, softening: f32, out: &mut [Vec3]) {
    let n = pos.len().min(q.len()).min(out.len());
    for v in out.iter_mut().take(n) {
        *v = Vec3::ZERO;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let f = pair_force_3(pos[i], q[i], pos[j], q[j], k, softening);
            out[i] += f;
            out[j] -= f;
        }
    }
}

/// The electric field at `p` from the point charges in `pos`/`q`.
///
/// `E = k sum_j q_j (p - p_j) / (|p - p_j|^2 + eps^2)^{3/2}`.
pub fn field_at_2(p: Vec2, pos: &[Vec2], q: &[f32], k: f32, softening: f32) -> Vec2 {
    let n = pos.len().min(q.len());
    let mut e = Vec2::ZERO;
    for j in 0..n {
        let d = p - pos[j];
        let r2 = d.length_squared() + softening * softening;
        if r2 < EPS_SQ || !r2.is_finite() {
            continue;
        }
        let s = k * q[j];
        if !s.is_finite() {
            continue;
        }
        let inv = 1.0 / r2.sqrt();
        e += d * (s * inv / r2);
    }
    e
}

/// 3D counterpart of [`field_at_2`].
pub fn field_at_3(p: Vec3, pos: &[Vec3], q: &[f32], k: f32, softening: f32) -> Vec3 {
    let n = pos.len().min(q.len());
    let mut e = Vec3::ZERO;
    for j in 0..n {
        let d = p - pos[j];
        let r2 = d.length_squared() + softening * softening;
        if r2 < EPS_SQ || !r2.is_finite() {
            continue;
        }
        let s = k * q[j];
        if !s.is_finite() {
            continue;
        }
        let inv = 1.0 / r2.sqrt();
        e += d * (s * inv / r2);
    }
    e
}

/// The electric potential at `p` from the point charges in `pos`/`q`.
///
/// `V = k sum_j q_j / sqrt(|p_j - p|^2 + eps^2)`. Coincident charges
/// contribute zero rather than an infinity.
pub fn potential_at_2(p: Vec2, pos: &[Vec2], q: &[f32], k: f32, softening: f32) -> f32 {
    let n = pos.len().min(q.len());
    let mut v = 0.0f32;
    for j in 0..n {
        let d = pos[j] - p;
        let r2 = d.length_squared() + softening * softening;
        if r2 < EPS_SQ || !r2.is_finite() {
            continue;
        }
        let s = k * q[j];
        if !s.is_finite() {
            continue;
        }
        v += s / r2.sqrt();
    }
    v
}

/// 3D counterpart of [`potential_at_2`].
pub fn potential_at_3(p: Vec3, pos: &[Vec3], q: &[f32], k: f32, softening: f32) -> f32 {
    let n = pos.len().min(q.len());
    let mut v = 0.0f32;
    for j in 0..n {
        let d = pos[j] - p;
        let r2 = d.length_squared() + softening * softening;
        if r2 < EPS_SQ || !r2.is_finite() {
            continue;
        }
        let s = k * q[j];
        if !s.is_finite() {
            continue;
        }
        v += s / r2.sqrt();
    }
    v
}

/// The field of a single fixed point charge at `source`.
pub fn central_field_2(p: Vec2, source: Vec2, q: f32, k: f32, softening: f32) -> Vec2 {
    field_at_2(p, &[source], &[q], k, softening)
}

/// 3D counterpart of [`central_field_2`].
pub fn central_field_3(p: Vec3, source: Vec3, q: f32, k: f32, softening: f32) -> Vec3 {
    field_at_3(p, &[source], &[q], k, softening)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_charges_repel() {
        let f = pair_force_2(Vec2::ZERO, 1.0, Vec2::new(2.0, 0.0), 1.0, 1.0, 0.0);
        // d = pi - pj = (-2,0), r2 = 4, s = 1, f = (-2,0) * (1 * 0.5 / 4).
        assert!((f - Vec2::new(-0.25, 0.0)).length() < 1e-6);
        assert!(
            f.x < 0.0,
            "like charges must be pushed away from the source"
        );
    }

    #[test]
    fn opposite_charges_attract() {
        let f = pair_force_2(Vec2::ZERO, 1.0, Vec2::new(2.0, 0.0), -1.0, 1.0, 0.0);
        assert!((f - Vec2::new(0.25, 0.0)).length() < 1e-6);
    }

    #[test]
    fn force_follows_inverse_square() {
        let near = pair_force_2(Vec2::ZERO, 1.0, Vec2::new(1.0, 0.0), 1.0, 1.0, 0.0);
        let far = pair_force_2(Vec2::ZERO, 1.0, Vec2::new(2.0, 0.0), 1.0, 1.0, 0.0);
        assert!((near.x / far.x - 4.0).abs() < 1e-5);
    }

    #[test]
    fn coincident_pair_is_zero() {
        let f = pair_force_2(Vec2::ONE, 1.0, Vec2::ONE, 1.0, 1.0, 0.0);
        assert_eq!(f, Vec2::ZERO);
    }

    #[test]
    fn pairwise_momentum_is_symmetric() {
        let pos = [Vec2::ZERO, Vec2::new(3.0, 4.0), Vec2::new(-2.0, 1.0)];
        let q = [2.0, -3.0, 5.0];
        let mut out = [Vec2::ZERO; 3];
        pairwise_forces_2(&pos, &q, 1.0, 0.1, &mut out);
        let total = out[0] + out[1] + out[2];
        assert!(total.length() < 1e-5, "momentum leak: {total:?}");
    }

    #[test]
    fn field_is_the_superposition() {
        let pos = [Vec2::new(1.0, 0.0), Vec2::new(-1.0, 0.0)];
        let q = [1.0, 2.0];
        let p = Vec2::new(0.0, 1.0);
        let total = field_at_2(p, &pos, &q, 1.0, 0.0);
        let a = field_at_2(p, &pos[0..1], &q[0..1], 1.0, 0.0);
        let b = field_at_2(p, &pos[1..2], &q[1..2], 1.0, 0.0);
        assert!((total - (a + b)).length() < 1e-6);
    }

    #[test]
    fn potential_of_a_point_charge() {
        let v = potential_at_2(Vec2::new(3.0, 0.0), &[Vec2::ZERO], &[2.0], 1.0, 0.0);
        assert!((v - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn three_d_force_and_field_mirror() {
        let f = pair_force_3(Vec3::ZERO, 1.0, Vec3::new(0.0, 0.0, 2.0), 1.0, 1.0, 0.0);
        assert!((f - Vec3::new(0.0, 0.0, -0.25)).length() < 1e-6);
        let e = field_at_3(Vec3::new(2.0, 0.0, 0.0), &[Vec3::ZERO], &[1.0], 1.0, 0.0);
        assert!((e - Vec3::new(0.25, 0.0, 0.0)).length() < 1e-6);
    }
}
