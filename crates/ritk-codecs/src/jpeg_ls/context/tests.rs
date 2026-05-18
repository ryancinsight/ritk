use super::*;

#[test]
fn quant_boundary_mapping() {
    // T1=3, T2=7, T3=21
    let (t1, t2, t3) = (3, 7, 21);
    assert_eq!(quant(-30, t1, t2, t3), -4); // d <= -T3
    assert_eq!(quant(-21, t1, t2, t3), -4); // d == -T3 → ≤ -T3
    assert_eq!(quant(-10, t1, t2, t3), -3); // -T3 < d ≤ -T2
    assert_eq!(quant(-7, t1, t2, t3), -3); // d == -T2 → ≤ -T2
    assert_eq!(quant(-5, t1, t2, t3), -2);
    assert_eq!(quant(-3, t1, t2, t3), -2); // d == -T1 → ≤ -T1
    assert_eq!(quant(-1, t1, t2, t3), -1);
    assert_eq!(quant(0, t1, t2, t3), 0);
    assert_eq!(quant(1, t1, t2, t3), 1);
    assert_eq!(quant(2, t1, t2, t3), 1); // d < T1
    assert_eq!(quant(3, t1, t2, t3), 2); // d == T1 → ≥ T1
    assert_eq!(quant(6, t1, t2, t3), 2);
    assert_eq!(quant(7, t1, t2, t3), 3);
    assert_eq!(quant(20, t1, t2, t3), 3);
    assert_eq!(quant(21, t1, t2, t3), 4); // d ≥ T3
    assert_eq!(quant(100, t1, t2, t3), 4);
}

#[test]
fn sign_normalize_positive_q1() {
    let (q1, q2, q3, s) = sign_normalize(2, -3, 1);
    assert_eq!((q1, q2, q3), (2, -3, 1));
    assert_eq!(s, 1);
}

#[test]
fn sign_normalize_negative_q1() {
    let (q1, q2, q3, s) = sign_normalize(-2, 3, -1);
    assert_eq!((q1, q2, q3), (2, -3, 1));
    assert_eq!(s, -1);
}

#[test]
fn sign_normalize_q1_zero_negative_q2() {
    let (q1, q2, q3, s) = sign_normalize(0, -3, 1);
    assert_eq!((q1, q2, q3), (0, 3, -1));
    assert_eq!(s, -1);
}

#[test]
fn sign_normalize_all_zero() {
    let (q1, q2, q3, s) = sign_normalize(0, 0, 0);
    assert_eq!((q1, q2, q3, s), (0, 0, 0, 1));
}

#[test]
fn context_index_q1q2q3_all_zero() {
    // Q3=0 → index 0
    assert_eq!(context_index(0, 0, 0), 0);
}

#[test]
fn context_index_q1q2_zero_q3_4() {
    // Q3=4 → index 4
    assert_eq!(context_index(0, 0, 4), 4);
}

#[test]
fn context_index_q1_zero_q2_1_q3_neg4() {
    // Q2=1, Q3=-4 → 5 + 0*9 + 0 = 5
    assert_eq!(context_index(0, 1, -4), 5);
}

#[test]
fn context_index_q1_zero_q2_4_q3_4() {
    // Q2=4, Q3=4 → 5 + 3*9 + 8 = 5 + 27 + 8 = 40
    assert_eq!(context_index(0, 4, 4), 40);
}

#[test]
fn context_index_q1_1_min() {
    // Q1=1, Q2=-4, Q3=-4 → 41 + 0 + 0 + 0 = 41
    assert_eq!(context_index(1, -4, -4), 41);
}

#[test]
fn context_index_q1_4_max() {
    // Q1=4, Q2=4, Q3=4 → 41 + 3*81 + 8*9 + 8 = 41+243+72+8 = 364
    assert_eq!(context_index(4, 4, 4), 364);
}

#[test]
fn context_index_max_value_is_364() {
    assert_eq!(context_index(4, 4, 4), 364);
}

#[test]
fn default_thresholds_8bit() {
    // maxval=255: factor=(255+128)/256=1
    // t1 = max(3, 2).min(32) = 3
    // t2 = max(7, 3).min(64) = 7
    // t3 = max(21, 4).min(255) = 21
    let (t1, t2, t3) = default_thresholds(255);
    assert_eq!(t1, 3);
    assert_eq!(t2, 7);
    assert_eq!(t3, 21);
}

#[test]
fn default_thresholds_16bit() {
    // maxval=65535: factor=(65535+128)/256=256
    // t1=max(768,2).min(8192)=768
    // t2=max(1792,3).min(16384)=1792
    // t3=max(5376,4).min(65535)=5376
    let (t1, t2, t3) = default_thresholds(65535);
    assert_eq!(t1, 768);
    assert_eq!(t2, 1792);
    assert_eq!(t3, 5376);
}

#[test]
fn inverse_map_even_zero() {
    assert_eq!(inverse_map(0), 0);
}

#[test]
fn inverse_map_even_positive() {
    // MErrval=4 (even) → errval = 2
    assert_eq!(inverse_map(4), 2);
}

#[test]
fn inverse_map_odd_negative() {
    // MErrval=1 (odd) → errval = -(1+1)/2 = -1
    assert_eq!(inverse_map(1), -1);
}

#[test]
fn inverse_map_odd_large() {
    // MErrval=7 (odd) → errval = -(7+1)/2 = -4
    assert_eq!(inverse_map(7), -4);
}

#[test]
fn inverse_map_forward_inverse_bijection() {
    // Forward: errval ≥ 0 → 2*errval; errval < 0 → -2*errval - 1
    for errval in -50i32..=50i32 {
        let me = if errval >= 0 {
            (errval * 2) as u32
        } else {
            (-2 * errval - 1) as u32
        };
        assert_eq!(
            inverse_map(me),
            errval,
            "round-trip failed for errval={}",
            errval
        );
    }
}

#[test]
fn compute_k_zero_when_a_small() {
    // A=1, N=1: N<<0 = 1 >= 1 = A → k=0
    assert_eq!(compute_k(1, 1, 8), 0);
}

#[test]
fn compute_k_increases_with_large_a() {
    // A=8, N=1: N<<0=1 < 8, N<<1=2 < 8, N<<2=4 < 8, N<<3=8 >= 8 → k=3
    assert_eq!(compute_k(8, 1, 8), 3);
}

#[test]
fn compute_k_clamped_to_qbpp() {
    // A=1000000, N=1, qbpp=8: k stops at 8
    assert_eq!(compute_k(1_000_000, 1, 8), 8);
}

#[test]
fn update_context_accumulates_errval() {
    let mut ctx = ContextState::default();
    update_context(&mut ctx, 3, 0); // NEAR=0
    assert_eq!(ctx.a, 3); // |errval| = 3
    assert_eq!(ctx.n, 2); // n increments to 2
}

#[test]
fn update_context_bias_positive_decrements_b() {
    let mut ctx = ContextState {
        a: 0,
        b: 5,
        c: 0,
        n: 4,
    };
    // b > 0 → b -= n → b = 5 - 4 = 1 → clamp to min(1, 0) = 0 in the code?
    // Actually: b = (b - n).min(0) = (1).min(0) = 0. Wait let me re-check the code.
    // After errval=0: ctx.b += 0*(2*0+1) = 0, so b stays 5.
    // Then: b > 0 → b -= n (=4) → b = 1 → b.min(0) = 0? No: `b -= n` then `b = b.min(0)`.
    // Hmm, actually: b = b - n = 5 - 5 (n=5 after += 1) → b = 0.
    update_context(&mut ctx, 0, 0);
    assert_eq!(ctx.n, 5);
    // b=5, n becomes 5: b -= n → b = 0 → min(0,0) = 0
    assert_eq!(ctx.b, 0);
}
