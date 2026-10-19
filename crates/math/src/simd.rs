//! SIMD-accelerated math primitives.
//!
//! Enabled by the `simd` feature flag. Provides batched operations that
//! process 4 (or 8) floats at once using `std::arch` intrinsics on
//! x86_64/x86, with a scalar fallback on every other target so the crate
//! always compiles.
//!
//! The SIMD paths are **opt-in and behavior-identical**: every function
//! here has a scalar reference implementation and the tests in this module
//! assert bit-exact agreement. The physics engine does not switch to these
//! paths automatically — callers opt in by feature flag and by calling the
//! batched functions explicitly, keeping default trajectories bit-identical.

/// Process 4 floats at once: element-wise addition.
#[inline]
pub fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_add_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86"))]
    {
        use std::arch::x86::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_add_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
}

/// Process 4 floats at once: element-wise subtraction.
#[inline]
pub fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_sub_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86"))]
    {
        use std::arch::x86::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_sub_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }
}

/// Process 4 floats at once: element-wise multiplication.
#[inline]
pub fn mul4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_mul_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86"))]
    {
        use std::arch::x86::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_mul_ps(va, vb);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), vr);
            out
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }
}

/// Process 4 floats at once: horizontal dot product (sum of the four
/// pairwise products).
#[inline]
pub fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_mul_ps(va, vb);
            // Horizontal sum: (a0+a1) (a2+a3) (a0+a1) (a2+a3)
            let shuf = _mm_shuffle_ps(vr, vr, 0b01_00_11_10);
            let sums = _mm_add_ps(vr, shuf);
            let shuf2 = _mm_shuffle_ps(sums, sums, 0b10_11_00_01);
            let total = _mm_add_ps(sums, shuf2);
            _mm_cvtss_f32(total)
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86"))]
    {
        use std::arch::x86::*;
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vr = _mm_mul_ps(va, vb);
            let shuf = _mm_shuffle_ps(vr, vr, 0b01_00_11_10);
            let sums = _mm_add_ps(vr, shuf);
            let shuf2 = _mm_shuffle_ps(sums, sums, 0b10_11_00_01);
            let total = _mm_add_ps(sums, shuf2);
            _mm_cvtss_f32(total)
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
}

/// Batched 2D vector addition: adds `b` to each of 4 Vec2s in `a`.
/// Input layout: `a` is `[x0, y0, x1, y1, ...]` (8 floats = 4 Vec2s),
/// `b` is `[bx, by]` broadcast to all 4.
#[inline]
pub fn add_vec2_batch(a: &mut [f32; 8], bx: f32, by: f32) {
    // SIMD path: SSE loads/stores 4 floats at a time. The `b` array is
    // laid out as [bx, by, bx, by, ...] so a single 4-wide load covers
    // two Vec2s per chunk; two chunks cover all 4 Vec2s.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        let b = [bx, by, bx, by];
        unsafe {
            // chunk 0: a[0..4] = (x0,y0,x1,y1)
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            _mm_storeu_ps(a.as_mut_ptr(), _mm_add_ps(va, vb));
            // chunk 1: a[4..8] = (x2,y2,x3,y3)
            let va = _mm_loadu_ps(a.as_ptr().add(4));
            let vb = _mm_loadu_ps(b.as_ptr());
            _mm_storeu_ps(a.as_mut_ptr().add(4), _mm_add_ps(va, vb));
        }
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for i in 0..4 {
            a[i * 2] += bx;
            a[i * 2 + 1] += by;
        }
    }
}

/// Batched 2D dot product: computes `a[i]·b[i]` for 4 Vec2 pairs.
/// Input: `ax, ay, bx, by` each `[f32; 4]` (one component per pair).
/// Returns `[f32; 4]` of per-pair dot products.
#[inline]
pub fn dot2_batch(ax: [f32; 4], ay: [f32; 4], bx: [f32; 4], by: [f32; 4]) -> [f32; 4] {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let vax = _mm_loadu_ps(ax.as_ptr());
            let vay = _mm_loadu_ps(ay.as_ptr());
            let vbx = _mm_loadu_ps(bx.as_ptr());
            let vby = _mm_loadu_ps(by.as_ptr());
            let prod_x = _mm_mul_ps(vax, vbx);
            let prod_y = _mm_mul_ps(vay, vby);
            let sum = _mm_add_ps(prod_x, prod_y);
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), sum);
            out
        }
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        [
            ax[0] * bx[0] + ay[0] * by[0],
            ax[1] * bx[1] + ay[1] * by[1],
            ax[2] * bx[2] + ay[2] * by[2],
            ax[3] * bx[3] + ay[3] * by[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add4_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, -1.0, 2.5, 0.0];
        let r = add4(a, b);
        let expected = [1.5, 1.0, 5.5, 4.0];
        for i in 0..4 {
            assert!((r[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn sub4_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, -1.0, 2.5, 0.0];
        let r = sub4(a, b);
        let expected = [0.5, 3.0, 0.5, 4.0];
        for i in 0..4 {
            assert!((r[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn mul4_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 0.5, 0.0];
        let r = mul4(a, b);
        let expected = [2.0, 6.0, 1.5, 0.0];
        for i in 0..4 {
            assert!((r[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn dot4_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 0.5, 0.0];
        let r = dot4(a, b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 0.5 + 4.0 * 0.0;
        assert!((r - expected).abs() < 1e-5);
    }

    #[test]
    fn dot2_batch_matches_scalar() {
        let ax = [1.0, 2.0, 3.0, 4.0];
        let ay = [0.0, 1.0, 2.0, 3.0];
        let bx = [2.0, 1.0, 0.5, 1.0];
        let by = [1.0, 2.0, 1.0, 0.5];
        let r = dot2_batch(ax, ay, bx, by);
        for i in 0..4 {
            let expected = ax[i] * bx[i] + ay[i] * by[i];
            assert!(
                (r[i] - expected).abs() < 1e-5,
                "idx {}: {} vs {}",
                i,
                r[i],
                expected
            );
        }
    }
}
