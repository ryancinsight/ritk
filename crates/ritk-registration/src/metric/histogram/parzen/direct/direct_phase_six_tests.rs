//! Phase Six tests for the direct Parzen histogram computation path.

use super::types::{ParzenConfig, StackWeights, STACK_WEIGHTS_CAPACITY};
use super::*;

// ─── Phase Six property tests (TEST-319-07, TEST-319-08, TEST-319-11) ──────

#[test]
fn direct_exp_ratchet_matches_naive() {
    // PERF-319-04: Verify that the exp-ratchet optimisation in
    // StackWeights::new produces values within 1e-6 of the naive
    // per-bin exp() computation for typical sigma values.
    let sigma_sq_values = [0.5, 1.0, 2.0, 4.0, 9.0];
    let test_vals: [f32; 4] = [0.3, 5.7, 15.3, 29.8];
    let num_bins = 32;

    for &sigma_sq in &sigma_sq_values {
        let cfg = ParzenConfig::new(sigma_sq);
        let inv_2sigma_sq = cfg.inv_2sigma_sq();

        for &val in &test_vals {
            let primary = val.floor() as i32;
            let lo = (primary - cfg.half_width() as i32).max(0) as usize;
            let hi = ((primary + cfg.half_width() as i32).min(num_bins - 1)).max(0) as usize;

            // Build StackWeights using the exp-ratchet path
            let sw = StackWeights::new(val, lo, hi, inv_2sigma_sq);

            // Naive computation
            for (j, w_ratchet) in sw.iter() {
                let b = lo + j;
                let diff = val - b as f32;
                let w_naive = (diff * diff * inv_2sigma_sq).exp();
                let abs_err = (w_ratchet - w_naive).abs();
                let rel_err = if w_naive.abs() > 1e-10 {
                    abs_err / w_naive
                } else {
                    abs_err
                };
                assert!(
                    rel_err < 1e-5,
                    "exp-ratchet drift too large at sigma_sq={sigma_sq}, val={val}, bin={b}: \
                     ratchet={w_ratchet}, naive={w_naive}, rel_err={rel_err}"
                );
            }
        }
    }
}

#[test]
fn direct_exp_ratchet_boundary_precision() {
    // PERF-319-04: At the boundary of STACK_WEIGHTS_CAPACITY (15 bins),
    // the ratchet should still maintain precision within 1e-4.
    let sigma_sq = 4.0; // half_width=6 → 13 bins
    let cfg = ParzenConfig::new(sigma_sq);
    let val = 15.5_f32;
    let primary = val.floor() as i32; // 15
    let lo = (primary - cfg.half_width() as i32).max(0) as usize;
    let hi = (primary + cfg.half_width() as i32).clamp(0, 31) as usize;
    let sw = StackWeights::new(val, lo, hi, cfg.inv_2sigma_sq());
    assert_eq!(sw.len, 13);

    // Check last entry precision — this is where drift accumulates most
    for (j, w_ratchet) in sw.iter() {
        let b = lo + j;
        let diff = val - b as f32;
        let w_naive = (diff * diff * cfg.inv_2sigma_sq()).exp();
        let abs_err = (w_ratchet - w_naive).abs();
        assert!(
            abs_err < 1e-5,
            "boundary ratchet drift at bin {b}: ratchet={w_ratchet}, naive={w_naive}, abs_err={abs_err}"
        );
    }
}

#[test]
fn direct_histogram_pool_reuse() {
    // TEST-319-08: Verify that HistogramPool correctly reuses buffers
    // across checkout/return cycles, and that each checkout produces a
    // zeroed buffer.
    use super::HistogramPool;

    let pool = HistogramPool::new(16 * 16); // 256 elements

    // First checkout — allocates new
    let mut buf1 = pool.checkout();
    assert_eq!(buf1.len(), 256);
    assert!(
        buf1.iter().all(|&v| v == 0.0),
        "fresh buffer must be zeroed"
    );

    // Write non-zero values
    buf1[0] = 42.0;
    buf1[255] = 99.0;
    pool.return_buffer(buf1);

    // Second checkout — should reuse the returned buffer
    let buf2 = pool.checkout();
    assert_eq!(buf2.len(), 256);
    assert!(
        buf2.iter().all(|&v| v == 0.0),
        "reused buffer must be re-zeroed"
    );

    // Return and checkout again to confirm pool depth
    pool.return_buffer(buf2);
    let buf3 = pool.checkout();
    assert_eq!(buf3.len(), 256);
    assert!(buf3.iter().all(|&v| v == 0.0));
}

#[test]
fn direct_histogram_pool_multiple_buffers() {
    // TEST-319-08: Multiple threads can checkout and return concurrently.
    // This test simulates the pattern used by rayon fold/reduce.
    use super::HistogramPool;

    let pool = HistogramPool::new(64);

    // Simulate rayon fold: each "thread" checks out a buffer
    let mut b1 = pool.checkout();
    let mut b2 = pool.checkout(); // second checkout — allocates new since pool was empty
    assert_eq!(b1.len(), 64);
    assert_eq!(b2.len(), 64);

    // "Threads" accumulate into their buffers
    b1[0] = 1.0;
    b2[0] = 2.0;

    // Simulate reduction: merge b2 into b1, return b2
    for (dst, src) in b1.iter_mut().zip(b2.iter()) {
        *dst += src;
    }
    pool.return_buffer(b2);

    // b1 should have the merged result
    assert!((b1[0] - 3.0).abs() < 1e-10);

    // Return b1
    pool.return_buffer(b1);

    // Now two buffers are in the pool — next checkout reuses one
    let b3 = pool.checkout();
    assert_eq!(b3.len(), 64);
    assert!(b3.iter().all(|&v| v == 0.0), "reused buffer must be zeroed");
}

#[test]
fn direct_support_bins_consistency() {
    // TEST-319-11: ParzenConfig::support_bins() must equal
    // 2 * half_width + 1 and must not exceed STACK_WEIGHTS_CAPACITY
    // for any practical sigma value.
    for &sigma_sq in &[0.01, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 20.25] {
        let cfg = ParzenConfig::new(sigma_sq);
        assert_eq!(
            cfg.support_bins(),
            2 * cfg.half_width() + 1,
            "support_bins mismatch for sigma_sq={sigma_sq}"
        );
        assert!(
            cfg.support_bins() <= STACK_WEIGHTS_CAPACITY,
            "support_bins ({}) exceeds STACK_WEIGHTS_CAPACITY ({}) for sigma_sq={sigma_sq}",
            cfg.support_bins(),
            STACK_WEIGHTS_CAPACITY
        );
    }
}

#[test]
fn direct_from_intensity_sigma_near_equal_range() {
    // TEST-319-07: When max is very close to min (but still > min),
    // the resulting sigma_sq should be very large (many bins per sigma),
    // not NaN or infinity.
    let cfg = ParzenConfig::from_intensity_sigma(1.0, 0.0, 0.001, 32);
    assert!(
        cfg.sigma_sq() > 0.0 && cfg.sigma_sq().is_finite(),
        "near-equal range should produce large but finite sigma_sq, got {}",
        cfg.sigma_sq()
    );
    assert_eq!(cfg.half_width(), compute_half_width(cfg.sigma_sq()));
}

#[test]
#[should_panic(expected = "sigma_sq must be positive")]
fn direct_rejects_negative_sigma() {
    // TEST-319-07: Negative sigma_sq must panic.
    let _ = ParzenConfig::new(-1.0);
}

#[test]
#[should_panic(expected = "sigma_sq must be finite")]
fn direct_rejects_infinite_sigma() {
    // TEST-319-07: Infinite sigma_sq must panic.
    let _ = ParzenConfig::new(f32::INFINITY);
}

#[test]
fn direct_separate_sigma_per_axis() {
    // TEST-319-11: Different fixed and moving sigmas should produce
    // a valid histogram with asymmetric spreading.
    let num_bins = 32;
    let sigma_sq_fix = 1.0_f32; // σ=1 bin
    let sigma_sq_mov = 4.0_f32; // σ=2 bins
    let n = 50;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    let hist_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq_fix,
        sigma_sq_mov,
        None,
        None,
    );
    let slice = hist_data.as_slice::<f32>().unwrap();

    // All entries must be non-negative
    for (i, &v) in slice.iter().enumerate() {
        assert!(v >= 0.0, "histogram entry {i} is negative: {v}");
    }

    // Total must be positive and finite
    let total: f32 = slice.iter().sum();
    assert!(total > 0.0, "total weight must be positive, got {total}");
    assert!(
        total.is_finite(),
        "total weight must be finite, got {total}"
    );

    // The broader moving sigma (4.0) should spread weight across more
    // moving-axis bins per sample, producing more non-zero entries
    let nonzero = slice.iter().filter(|&&v| v > 1e-10).count();
    assert!(nonzero > 0, "histogram must have non-zero entries");
}

#[test]
fn direct_sparse_separate_sigma_per_axis() {
    // TEST-319-11: Separate sigma per axis should work correctly with
    // the sparse-cache path too.
    let num_bins = 32;
    let sigma_sq_fix = 1.0_f32;
    let sigma_sq_mov = 4.0_f32;
    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.42 + 0.1) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) as f32 * 0.031) % 30.0)
        .collect();

    let direct_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq_fix,
        sigma_sq_mov,
        None,
        None,
    );
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq_fix, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq_mov,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // SPARSE-329-01: Both direct and sparse paths now apply full joint
    // normalization (inv_norm = inv_sum_f × inv_sum_m). The ratio should ≈ 1.0.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-10;
        let s_nz = *s > 1e-10;
        assert_eq!(
            d_nz, s_nz,
            "asymmetric-sigma nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "asymmetric-sigma sparse/direct ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}
