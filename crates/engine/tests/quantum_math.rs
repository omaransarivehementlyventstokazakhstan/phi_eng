//! The quantum math layer (M01–M08, mirrored Rust↔C#).
//!
//! Quantum mechanics is arithmetic over the complex numbers, and a
//! measurement is a random draw. These tests pin both:
//!
//! * the complex arithmetic and its transcendental functions, bit for
//!   bit, so the Rust core and the managed mirror cannot drift apart;
//! * the seeded random number generator, bit for bit, so a sampled
//!   measurement is reproducible on every platform.
//!
//! The pins are the contract. If one of them moves, the arithmetic
//! changed and every quantum result built on it changes with it — the
//! same rule the deterministic trigonometry follows.

use phieng_math::complex::Complex32;
use phieng_math::rng::DeterministicRng;

/// The bit pattern of a float, as the pins are written.
fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// The two bit patterns of a complex number.
fn cbits(c: Complex32) -> (u32, u32) {
    (bits(c.re), bits(c.im))
}

/// M01: the complex arithmetic's pinned bit patterns.
///
/// The operands are `a = 1.5 - 2.25i` and `b = -0.75 + 3.5i`.
#[test]
fn m01_complex_arithmetic_is_portable() {
    let a = Complex32::new(1.5, -2.25);
    let b = Complex32::new(-0.75, 3.5);
    assert_eq!(cbits(a + b), (0x3f40_0000, 0x3fa0_0000));
    assert_eq!(cbits(a - b), (0x4010_0000, 0xc0b8_0000));
    assert_eq!(cbits(a * b), (0x40d8_0000, 0x40de_0000));
    assert_eq!(cbits(a / b), (0x3f0a_9d59, 0xbf06_de48));
    assert_eq!(cbits(a * 2.5), (0x4070_0000, 0xc0b4_0000));
    assert_eq!(cbits(-a), (0xbfc0_0000, 0x4010_0000));
    assert_eq!(cbits(a.conj()), (0x3fc0_0000, 0x4010_0000));
    assert_eq!(cbits(a.inv()), (0xbe9d_89d9, 0xbe52_0d21));
    assert_eq!(bits(a.abs()), 0x402d_1104);
    assert_eq!(bits(a.norm_sqr()), 0x40ea_0000);
    let (r, theta) = a.to_polar();
    assert_eq!(bits(r), 0x402d_1104);
    assert_eq!(bits(theta), 0xbf7b_985f);
}

/// M02: the complex transcendentals' pinned bit patterns.
#[test]
fn m02_complex_transcendentals_are_portable() {
    // (z, exp, ln, sqrt, sin, cos) as (re, im) bit pairs.
    let cases: [(Complex32, [(u32, u32); 4]); 8] = [
        (
            Complex32::new(0.0, 0.0),
            [
                (0x3f80_0000, 0x0000_0000),
                (0xff80_0000, 0x0000_0000),
                (0x0000_0000, 0x0000_0000),
                (0x0000_0000, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(1.0, 0.0),
            [
                (0x402d_f854, 0x0000_0000),
                (0x0000_0000, 0x0000_0000),
                (0x3f80_0000, 0x0000_0000),
                (0x3f57_6aa4, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(0.0, 1.0),
            [
                (0x3f0a_5140, 0x3f57_6aa4),
                (0x0000_0000, 0x3fc9_0fdb),
                (0x3f35_04f3, 0x3f35_04f3),
                (0x0000_0000, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(0.5, 0.25),
            [
                (0x3fcc_79c8, 0x3ed0_d84a),
                (0xbf14_e220, 0x3eed_6338),
                (0x3f3a_48cd, 0x3e2f_e733),
                (0x3ebf_2b45, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(-1.25, 2.0),
            [
                (0xbdf4_2dd9, 0x3e85_62a0),
                (0x3f5b_a773, 0x4008_4805),
                (0x3f3e_961e, 0x3fab_eeba),
                (0xbe03_8371, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(2.0, -3.5),
            [
                (0xc0dd_6ccc, 0x4025_e2a0),
                (0x3fb2_701d, 0xbf86_9c79),
                (0x3fde_46e1, 0xbf80_fe06),
                (0x41f0_e4f4, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(-0.75, -0.125),
            [
                (0x3eef_f6f4, 0xbd71_38eb),
                (0xbe8c_4774, 0xc03e_7e0f),
                (0x3d93_4b6b, 0xbf5e_7739),
                (0xbf45_bbdf, 0x0000_0000),
            ],
        ),
        (
            Complex32::new(3.0, std::f32::consts::FRAC_PI_4),
            [
                (0x4163_3dee, 0x4163_3dee),
                (0x3f90_dd76, 0x3e83_1959),
                (0x3fdf_9009, 0x3e66_3c13),
                (0x3d83_c5bc, 0x0000_0000),
            ],
        ),
    ];

    for (z, expected) in cases {
        let e = expected[0];
        let l = expected[1];
        let s = expected[2];
        let si = expected[3];
        assert_eq!(cbits(z.exp()), e, "exp({:?})", z);
        assert_eq!(cbits(z.ln()), l, "ln({:?})", z);
        assert_eq!(cbits(z.sqrt()), s, "sqrt({:?})", z);
        assert_eq!(cbits(z.sin()), si, "sin({:?})", z);
    }

    // The phase factor: the workhorse of every gate and Fourier twiddle.
    let phases: [(f32, (u32, u32)); 6] = [
        (0.0, (0x3f80_0000, 0x0000_0000)),
        (0.5, (0x3f60_a940, 0x3ef5_7744)),
        (1.0, (0x3f0a_5140, 0x3f57_6aa4)),
        (-0.75, (0x3f3b_4ff6, 0xbf2e_7fe1)),
        (std::f32::consts::PI, (0xbf80_0000, 0xb3bb_bd2e)),
        (2.5, (0xbf4d_17bf, 0x3f19_3578)),
    ];
    for (theta, expected) in phases {
        assert_eq!(cbits(Complex32::exp_i(theta)), expected, "exp_i({theta})");
    }
}

/// M03: the edge cases, pinned.
///
/// The two places a naive formula breaks are pinned here: the modulus
/// (which overflows if the components are squared first) and the
/// division (which overflows if the divisor's square is formed).
#[test]
fn m03_complex_edge_cases_are_portable() {
    let zero = Complex32::zero();
    // A NaN's bit pattern is not pinned: the payload of a NaN is not a
    // contract, and the two languages spell "not a number" differently.
    // What is pinned is that the answer is not a number.
    assert!((zero / zero).re.is_nan());
    assert!((zero / zero).im.is_nan());
    assert!((Complex32::one() / zero).re.is_nan());
    assert!((Complex32::one() / zero).im.is_nan());

    // A huge and a tiny magnitude: the scaled modulus survives both.
    assert_eq!(
        bits(Complex32::new(1e30, 1e30).abs()),
        0x718e_cc90,
        "a huge modulus must not overflow"
    );
    assert_eq!(
        bits(Complex32::new(1e-30, 1e-30).abs()),
        0x0de5_7822,
        "a tiny modulus must not underflow"
    );

    // The principal square root, including the branch cases.
    assert_eq!(
        cbits(Complex32::new(-4.0, 0.0).sqrt()),
        (0x0000_0000, 0x4000_0000)
    );
    assert_eq!(
        cbits(Complex32::new(3.0, 4.0).sqrt()),
        (0x4000_0000, 0x3f80_0000)
    );
    assert_eq!(
        cbits(Complex32::new(-3.0, 4.0).sqrt()),
        (0x3f80_0000, 0x4000_0000)
    );
    assert_eq!(
        cbits(Complex32::new(3.0, -4.0).sqrt()),
        (0x4000_0000, 0xbf80_0000)
    );

    // The squared modulus does overflow — that is documented, and the
    // division is the one that must not.
    assert_eq!(
        bits(Complex32::new(1e20, 1e20).norm_sqr()),
        0x7f80_0000,
        "the squared modulus overflows, by design"
    );
    assert_eq!(
        cbits(Complex32::new(1e20, 1e20) / Complex32::new(1e20, 1e20)),
        (0x3f80_0000, 0x0000_0000),
        "the division must not overflow"
    );
}

/// M04: the random number generator's pinned stream.
#[test]
fn m04_rng_stream_is_portable() {
    let mut rng = DeterministicRng::new(1);
    let expected: [u32; 16] = [
        0x7c64_370d,
        0x62ff_bc74,
        0x058b_01af,
        0x248e_bb73,
        0x9686_3623,
        0xfc41_7b82,
        0x591b_1abf,
        0x7359_a0d8,
        0x2d73_df81,
        0x7a18_ef6f,
        0x40ce_785d,
        0x8ac1_66f9,
        0xa54e_c268,
        0x7397_865a,
        0x47eb_e3ee,
        0x1ce0_1314,
    ];
    for want in expected {
        assert_eq!(rng.next_u32(), want);
    }
    assert_eq!(
        rng.state(),
        [3659518936, 1389279675, 1908592355, 2094870070],
        "the state after sixteen draws is pinned too"
    );
}

/// M05: the generator's seeding.
///
/// The same seed must give the same stream, and two different seeds
/// must not.
#[test]
fn m05_rng_seeding_is_reproducible() {
    let mut a = DeterministicRng::new(42);
    let mut b = DeterministicRng::new(42);
    for _ in 0..32 {
        assert_eq!(a.next_u32(), b.next_u32());
    }

    let mut c = DeterministicRng::new(43);
    let mut d = DeterministicRng::new(42);
    let differs = (0..32).any(|_| c.next_u32() != d.next_u32());
    assert!(differs, "two different seeds must give different streams");
}

/// M06: the generator's float mapping, pinned.
#[test]
fn m06_rng_float_mapping_is_portable() {
    let mut rng = DeterministicRng::new(12345);
    let expected: [u32; 8] = [
        0x3f62_3f08,
        0x3b78_5100,
        0x3f4e_d462,
        0x3f6c_6baa,
        0x3eaa_0ad8,
        0x3e39_a1ac,
        0x3f27_aa48,
        0x3eed_9958,
    ];
    for want in expected {
        let v = rng.next_f32();
        assert_eq!(bits(v), want);
        assert!((0.0..1.0).contains(&v), "the draw must be in [0, 1)");
    }
}

/// M07: the generator's normal deviate, pinned.
#[test]
fn m07_rng_normal_is_portable() {
    let mut rng = DeterministicRng::new(7);
    let expected: [u32; 6] = [
        0xbf59_ff96,
        0x3e7a_cd55,
        0x3e4c_e768,
        0x4018_5864,
        0xbf36_4ea7,
        0x3e51_d25d,
    ];
    for want in expected {
        assert_eq!(bits(rng.next_standard_normal()), want);
    }
}

/// M08: a stream can be saved and resumed exactly.
///
/// A saved game has to continue with the same numbers it would have
/// drawn without the save.
#[test]
fn m08_rng_state_resumes_the_stream() {
    let mut rng = DeterministicRng::new(99);
    for _ in 0..5 {
        rng.next_u32();
    }
    let saved = rng.state();
    let expected: Vec<u32> = (0..8).map(|_| rng.next_u32()).collect();

    let mut resumed = DeterministicRng::from_state(saved);
    let got: Vec<u32> = (0..8).map(|_| resumed.next_u32()).collect();
    assert_eq!(got, expected, "the resumed stream must match");
}
