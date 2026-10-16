//! Deterministic value noise for ambient force fields (turbulence).
//!
//! Every entry point is a pure function of `(position, time, seed)` — no
//! RNG, no wall clock, no global state. Identical inputs always produce
//! identical outputs, so scenes that use turbulence remain bit-identical
//! run-to-run (the same determinism contract as [`crate::gravity`]).
//!
//! The primitive is gradient-free value noise: integer lattice points are
//! hashed to `[0, 1)`, then blended with a quintic (C2) fade. fBm sums
//! octaves of halving amplitude / doubling frequency. Time evolution is a
//! smooth lerp between two integer time slices of the same field, which
//! keeps the field continuous (no "boiling") without needing 4D noise.
//!
//! Cross-language note: like gravity, the numeric output is **not**
//! promised to match the C# mirror bit-for-bit (different float
//! operations). Parity tests bind to statistical/energy bounds, never to
//! exact trajectories.

use glam::{Vec2, Vec3};

/// Multiplier applied to the integer time slice when deriving a per-slice
/// seed (golden-ratio odd constant).
const TIME_PRIME: u32 = 0x9e37_79b9;

/// Integer avalanche hash (finalizer from the `murmur3`/`lowbias32`
/// family). Deterministic across platforms; NOT cryptographic.
#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Hash a 2D integer lattice point + seed to `[0, 1)`.
#[inline]
fn lattice_2(ix: i32, iy: i32, seed: u32) -> f32 {
    let h = hash_u32(
        (ix as u32)
            .wrapping_mul(0x1f1f_1f1f)
            .wrapping_add((iy as u32).wrapping_mul(0x9e37_79b9))
            .wrapping_add(seed),
    );
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Hash a 3D integer lattice point + seed to `[0, 1)`.
#[inline]
fn lattice_3(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let h = hash_u32(
        (ix as u32)
            .wrapping_mul(0x1f1f_1f1f)
            .wrapping_add((iy as u32).wrapping_mul(0x9e37_79b9))
            .wrapping_add((iz as u32).wrapping_mul(0x85eb_ca6b))
            .wrapping_add(seed),
    );
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Quintic fade `6t^5 - 15t^4 + 10t^3` (zero 1st/2nd derivatives at 0/1).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 2D value noise in `[0, 1)`.
pub fn value_noise_2(x: f32, y: f32, seed: u32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let (xi, yi) = (x0 as i32, y0 as i32);
    let tx = fade(x - x0);
    let ty = fade(y - y0);
    let v00 = lattice_2(xi, yi, seed);
    let v10 = lattice_2(xi.wrapping_add(1), yi, seed);
    let v01 = lattice_2(xi, yi.wrapping_add(1), seed);
    let v11 = lattice_2(xi.wrapping_add(1), yi.wrapping_add(1), seed);
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty)
}

/// 3D value noise in `[0, 1)`.
pub fn value_noise_3(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let z0 = z.floor();
    let (xi, yi, zi) = (x0 as i32, y0 as i32, z0 as i32);
    let tx = fade(x - x0);
    let ty = fade(y - y0);
    let tz = fade(z - z0);
    let v000 = lattice_3(xi, yi, zi, seed);
    let v100 = lattice_3(xi.wrapping_add(1), yi, zi, seed);
    let v010 = lattice_3(xi, yi.wrapping_add(1), zi, seed);
    let v110 = lattice_3(xi.wrapping_add(1), yi.wrapping_add(1), zi, seed);
    let v001 = lattice_3(xi, yi, zi.wrapping_add(1), seed);
    let v101 = lattice_3(xi.wrapping_add(1), yi, zi.wrapping_add(1), seed);
    let v011 = lattice_3(xi, yi.wrapping_add(1), zi.wrapping_add(1), seed);
    let v111 = lattice_3(
        xi.wrapping_add(1),
        yi.wrapping_add(1),
        zi.wrapping_add(1),
        seed,
    );
    let a = lerp(lerp(v000, v100, tx), lerp(v010, v110, tx), ty);
    let b = lerp(lerp(v001, v101, tx), lerp(v011, v111, tx), ty);
    lerp(a, b, tz)
}

/// Static (time-independent) 2D fBm in `[-1, 1]`.
fn fbm_2_static(x: f32, y: f32, seed: u32, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves {
        sum += amp * (value_noise_2(x * freq, y * freq, seed.wrapping_add(o)) * 2.0 - 1.0);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Static (time-independent) 3D fBm in `[-1, 1]`.
fn fbm_3_static(x: f32, y: f32, z: f32, seed: u32, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves {
        sum +=
            amp * (value_noise_3(x * freq, y * freq, z * freq, seed.wrapping_add(o)) * 2.0 - 1.0);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Per-slice seed for time `slice` (integer time).
#[inline]
fn slice_seed(seed: u32, slice: i64) -> u32 {
    seed.wrapping_add((slice as u32).wrapping_mul(TIME_PRIME))
}

/// Time-evolving 2D fBm in `[-1, 1]`. `t` is the (already scaled) time.
pub fn fbm_2(x: f32, y: f32, t: f32, seed: u32, octaves: u8) -> f32 {
    let oct = (octaves as u32).clamp(1, 8);
    let t0 = t.floor();
    let ft = fade(t - t0);
    let s0 = slice_seed(seed, t0 as i64);
    let s1 = slice_seed(seed, (t0 as i64).wrapping_add(1));
    let a = fbm_2_static(x, y, s0, oct);
    let b = fbm_2_static(x, y, s1, oct);
    lerp(a, b, ft)
}

/// Time-evolving 3D fBm in `[-1, 1]`.
pub fn fbm_3(x: f32, y: f32, z: f32, t: f32, seed: u32, octaves: u8) -> f32 {
    let oct = (octaves as u32).clamp(1, 8);
    let t0 = t.floor();
    let ft = fade(t - t0);
    let s0 = slice_seed(seed, t0 as i64);
    let s1 = slice_seed(seed, (t0 as i64).wrapping_add(1));
    let a = fbm_3_static(x, y, z, s0, oct);
    let b = fbm_3_static(x, y, z, s1, oct);
    lerp(a, b, ft)
}

/// Divergence-free 2D flow via the curl of a scalar potential:
/// `v = (dψ/dy, -dψ/dx)`. Finite-difference derivatives of the fBm
/// potential. Magnitude is of order 1 (bounded by octave gradients).
pub fn curl_2(x: f32, y: f32, t: f32, seed: u32, octaves: u8) -> Vec2 {
    // Step small enough to resolve the highest octave, large enough to
    // stay above float noise.
    const E: f32 = 1e-3;
    let dpdy = (fbm_2(x, y + E, t, seed, octaves) - fbm_2(x, y - E, t, seed, octaves)) / (2.0 * E);
    let dpdx = (fbm_2(x + E, y, t, seed, octaves) - fbm_2(x - E, y, t, seed, octaves)) / (2.0 * E);
    Vec2::new(dpdy, -dpdx)
}

/// Divergence-free 3D flow via the curl of a vector potential. The three
/// potential components use distinct seed offsets so the field is not
/// degenerate.
pub fn curl_3(x: f32, y: f32, z: f32, t: f32, seed: u32, octaves: u8) -> Vec3 {
    const E: f32 = 1e-3;
    const OA: u32 = 0x1111_1111;
    const OB: u32 = 0x2222_2222;
    const OC: u32 = 0x3333_3333;
    // Potential components (ψx, ψy, ψz) as independent fBm fields.
    let psi_x = |x: f32, y: f32, z: f32| fbm_3(x, y, z, t, seed ^ OA, octaves);
    let psi_y = |x: f32, y: f32, z: f32| fbm_3(x, y, z, t, seed ^ OB, octaves);
    let psi_z = |x: f32, y: f32, z: f32| fbm_3(x, y, z, t, seed ^ OC, octaves);
    // curl = (dψz/dy - dψy/dz, dψx/dz - dψz/dx, dψy/dx - dψx/dy)
    let ddy = |f: &dyn Fn(f32, f32, f32) -> f32, x: f32, y: f32, z: f32| {
        (f(x, y + E, z) - f(x, y - E, z)) / (2.0 * E)
    };
    let ddz = |f: &dyn Fn(f32, f32, f32) -> f32, x: f32, y: f32, z: f32| {
        (f(x, y, z + E) - f(x, y, z - E)) / (2.0 * E)
    };
    let ddx = |f: &dyn Fn(f32, f32, f32) -> f32, x: f32, y: f32, z: f32| {
        (f(x + E, y, z) - f(x - E, y, z)) / (2.0 * E)
    };
    Vec3::new(
        ddy(&psi_z, x, y, z) - ddz(&psi_y, x, y, z),
        ddx(&psi_x, x, y, z) - ddz(&psi_z, x, y, z),
        ddx(&psi_y, x, y, z) - ddy(&psi_x, x, y, z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_noise_range() {
        for i in 0..50 {
            let x = i as f32 * 0.37;
            let y = i as f32 * -0.53;
            let n = value_noise_2(x, y, 0x1234);
            assert!((0.0..1.0).contains(&n), "noise out of range: {n}");
            let n3 = value_noise_3(x, y, x + y, 0x1234);
            assert!((0.0..1.0).contains(&n3), "3D noise out of range: {n3}");
        }
    }

    #[test]
    fn test_noise_deterministic() {
        // Same inputs -> identical bits (pure function contract).
        let a = fbm_2(1.25, -2.5, 3.0, 99, 4);
        let b = fbm_2(1.25, -2.5, 3.0, 99, 4);
        assert_eq!(a.to_bits(), b.to_bits());
        let c = fbm_3(1.0, 2.0, 3.0, 0.5, 7, 3);
        let d = fbm_3(1.0, 2.0, 3.0, 0.5, 7, 3);
        assert_eq!(c.to_bits(), d.to_bits());
    }

    #[test]
    fn test_noise_lipschitz_and_continuity() {
        // Small position changes must produce small output changes
        // (continuity; no lattice seams).
        let base = fbm_2(1.0, 1.0, 0.0, 5, 3);
        for k in 1..=8 {
            let eps = k as f32 * 1e-3;
            let n = fbm_2(1.0 + eps, 1.0, 0.0, 5, 3);
            assert!(
                (n - base).abs() < 0.5,
                "discontinuity at eps={eps}: {n} vs {base}"
            );
        }
    }

    #[test]
    fn test_noise_time_continuity() {
        let a = fbm_2(0.5, 0.5, 10.0, 3, 3);
        let b = fbm_2(0.5, 0.5, 10.0001, 3, 3);
        assert!((a - b).abs() < 0.01, "time discontinuity: {a} vs {b}");
    }

    #[test]
    fn test_noise_bounded() {
        for i in 0..200 {
            let x = i as f32 * 0.11 - 5.0;
            let y = i as f32 * -0.07 + 3.0;
            let n = fbm_2(x, y, i as f32 * 0.05, i as u32, 5);
            assert!(n.abs() <= 1.0 + 1e-5, "fbm exceeded unit range: {n}");
        }
    }

    #[test]
    fn test_curl_divergence_free() {
        // Numerical divergence of the 2D curl field should be ~0.
        let e = 1e-3;
        for i in 0..20 {
            let x = i as f32 * 0.3;
            let y = i as f32 * 0.21;
            let dvx = (curl_2(x + e, y, 0.0, 11, 3).x - curl_2(x - e, y, 0.0, 11, 3).x) / (2.0 * e);
            let dvy = (curl_2(x, y + e, 0.0, 11, 3).y - curl_2(x, y - e, 0.0, 11, 3).y) / (2.0 * e);
            assert!(
                (dvx + dvy).abs() < 1.0,
                "curl field not divergence-free at ({x},{y}): {}",
                dvx + dvy
            );
        }
    }
}
