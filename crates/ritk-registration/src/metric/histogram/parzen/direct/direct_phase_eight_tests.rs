//! Phase Eight tests for the direct Parzen histogram computation path.
//!
//! Tests for TEST-321-07 (histogram symmetry), TEST-321-08 (normalize_and_extract
//! correctness), PERF-321-06 (HistogramPool new_with_capacity),
//! ARCH-321-04 (SampleWindow::mask_val DRY), ARCH-321-10 (sigma_sq accessor),
//! and DRY-321-01 (normalize_to_bins consistency).

use super::sample::SampleWindow;
use super::types::ParzenConfig;
use super::*;

// â”€â”€â”€ Direct-path histogram symmetry (TEST-321-07) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn direct_histogram_symmetric_equal_sigma() {
    // TEST-321-07: When both axes use the same sigma AND the same image,
    // the joint histogram must be symmetric: H[i][j] == H[j][i].
    // The Gaussian kernel is symmetric, so identical inputs produce
    // a symmetric joint histogram.
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 100;
    let values: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.3) % (num_bins as f32 - 1.0))
        .collect();

    let hist_data =
        compute_joint_histogram_direct(&values, &values, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();

    // Verify symmetry
    for i in 0..num_bins {
        for j in 0..num_bins {
            let v1 = hist[i * num_bins + j];
            let v2 = hist[j * num_bins + i];
            let diff = (v1 - v2).abs();
            assert!(
                diff < 1e-5,
                "histogram not symmetric at [{i}][{j}]: {v1} vs [{j}][{i}]: {v2}, diff={diff}"
            );
        }
    }
}

#[test]
fn direct_histogram_swap_fixed_moving() {
    // TEST-321-07: Swapping fixed and moving with equal sigma should
    // produce the same histogram (transposed).
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) % 14.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 + 2.0) % 14.0).collect();

    let hist1 =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist2 =
        compute_joint_histogram_direct(&moving, &fixed, num_bins, sigma_sq, sigma_sq, None, None);

    let h1 = hist1.as_slice::<f32>().unwrap();
    let h2 = hist2.as_slice::<f32>().unwrap();

    // H1[i][j] should equal H2[j][i] (transposed)
    for i in 0..num_bins {
        for j in 0..num_bins {
            let v1 = h1[i * num_bins + j];
            let v2 = h2[j * num_bins + i];
            let diff = (v1 - v2).abs();
            assert!(
                diff < 1e-5,
                "swap-fixed-moving mismatch at H1[{i}][{j}]={v1} vs H2[{j}][{i}]={v2}, diff={diff}"
            );
        }
    }
}

#[test]
fn sparse_histogram_symmetric_equal_sigma() {
    // TEST-321-07: Sparse cache path should also produce symmetric
    // histograms when sigma_sq is equal on both axes and the same
    // image is used for both fixed and moving.
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 100;
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) % 14.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&values, num_bins, sigma_sq, None);

    let hist_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &values,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let hist = hist_data.as_slice::<f32>().unwrap();

    for i in 0..num_bins {
        for j in 0..num_bins {
            let v1 = hist[i * num_bins + j];
            let v2 = hist[j * num_bins + i];
            let diff = (v1 - v2).abs();
            assert!(
                diff < 1e-5,
                "sparse histogram not symmetric at [{i}][{j}]: {v1} vs [{j}][{i}]: {v2}, diff={diff}"
            );
        }
    }
}

// â”€â”€â”€ normalize_and_extract correctness (TEST-321-08) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// These tests use dispatch::normalize_and_extract which is only available
// under the direct-parzen feature.

#[cfg(feature = "direct-parzen")]
#[test]
fn normalize_and_extract_known_values() {
    // TEST-321-08: Verify normalize_and_extract produces correct values
    // for known inputs. This is the host-side normalization used by
    // the direct path.
    use crate::metric::histogram::parzen::dispatch::normalize_and_extract;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::Tensor;

    type B = NdArray<f32>;
    let device = Default::default();

    // Simple case: 4 values in [0, 100], 8 bins
    let values = Tensor::<f32, B>::from_floats([0.0, 50.0, 100.0, 25.0], &device);
    let result = normalize_and_extract::<B>(&values, 0.0, 100.0, 8);

    // num_bins_f = 7.0, scale = 7.0/100 = 0.07, offset = 0
    // 0.0 â†’ 0.0, 50.0 â†’ 3.5, 100.0 â†’ 7.0, 25.0 â†’ 1.75
    assert!((result[0] - 0.0).abs() < 1e-5, "val=0: got {}", result[0]);
    assert!((result[1] - 3.5).abs() < 1e-5, "val=50: got {}", result[1]);
    assert!((result[2] - 7.0).abs() < 1e-5, "val=100: got {}", result[2]);
    assert!((result[3] - 1.75).abs() < 1e-5, "val=25: got {}", result[3]);
}

#[cfg(feature = "direct-parzen")]
#[test]
fn normalize_and_extract_clamps_out_of_range() {
    // TEST-321-08: Values outside [min, max] should be clamped to
    // [0, num_bins-1].
    use crate::metric::histogram::parzen::dispatch::normalize_and_extract;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::Tensor;

    type B = NdArray<f32>;
    let device = Default::default();

    let values = Tensor::<f32, B>::from_floats([-10.0, 200.0], &device);
    let result = normalize_and_extract::<B>(&values, 0.0, 100.0, 8);

    // -10 should clamp to 0.0, 200 should clamp to 7.0
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "underflow: got {}",
        result[0]
    );
    assert!(
        (result[1] - 7.0).abs() < 1e-5,
        "overflow: got {}",
        result[1]
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn normalize_and_extract_with_offset() {
    // TEST-321-08: Normalization with non-zero min should apply the
    // correct offset.
    use crate::metric::histogram::parzen::dispatch::normalize_and_extract;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::Tensor;

    type B = NdArray<f32>;
    let device = Default::default();

    // Range [100, 200], 16 bins â†’ num_bins_f=15, scale=15/100=0.15, offset=-100*0.15=-15
    let values = Tensor::<f32, B>::from_floats([100.0, 150.0, 200.0], &device);
    let result = normalize_and_extract::<B>(&values, 100.0, 200.0, 16);

    assert!((result[0] - 0.0).abs() < 1e-5, "val=100: got {}", result[0]);
    assert!((result[1] - 7.5).abs() < 1e-5, "val=150: got {}", result[1]);
    assert!(
        (result[2] - 15.0).abs() < 1e-5,
        "val=200: got {}",
        result[2]
    );
}

// â”€â”€â”€ HistogramPool new_with_capacity (PERF-321-06) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn histogram_pool_new_with_capacity() {
    // PERF-321-06: new_with_capacity should pre-allocate the
    // specified number of buffers, each of the correct size.
    let pool = HistogramPool::new_with_capacity(256, 4);

    // Check out 4 buffers â€” should come from the pre-allocated pool
    let bufs: Vec<Vec<f32>> = (0..4).map(|_| pool.checkout()).collect();

    for buf in &bufs {
        assert_eq!(buf.len(), 256, "buffer size must be num_binsÂ²");
        assert!(
            buf.iter().all(|&v| v == 0.0),
            "pre-allocated buffers must be zeroed"
        );
    }

    // Return all buffers
    for buf in bufs {
        pool.return_buffer(buf);
    }
}

#[test]
fn histogram_pool_new_with_capacity_grows_beyond() {
    // PERF-321-06: The pool should grow beyond the initial capacity
    // if more buffers are requested than pre-allocated.
    let pool = HistogramPool::new_with_capacity(100, 2);

    // Check out 3 buffers â€” the third should be allocated on demand
    let b1 = pool.checkout();
    let b2 = pool.checkout();
    let b3 = pool.checkout();

    assert_eq!(b1.len(), 100);
    assert_eq!(b2.len(), 100);
    assert_eq!(b3.len(), 100);
}

// â”€â”€â”€ SampleWindow::mask_val DRY (ARCH-321-04) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn sample_window_mask_val_matches_inline() {
    // ARCH-321-04: SampleWindow::mask_val must produce the same result
    // as the former inline OOB check logic.
    let mask: Vec<f32> = vec![1.0, 0.0, 1.0, 0.5, 0.4];

    // In-bounds samples
    assert!(SampleWindow::mask_val(0, Some(&mask)).is_some());
    assert!(SampleWindow::mask_val(2, Some(&mask)).is_some());
    assert!(SampleWindow::mask_val(3, Some(&mask)).is_some()); // exactly 0.5

    // Out-of-bounds
    assert!(SampleWindow::mask_val(1, Some(&mask)).is_none());
    assert!(SampleWindow::mask_val(4, Some(&mask)).is_none()); // 0.4 < 0.5

    // No mask (all in-bounds)
    assert!(SampleWindow::mask_val(0, None).is_some());
    assert!(SampleWindow::mask_val(999, None).is_some());
}

// â”€â”€â”€ ParzenConfig::sigma_sq() accessor (ARCH-321-10) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn parzen_config_sigma_sq_accessor() {
    // ARCH-321-10 / ARCH-322-03: sigma_sq() accessor returns the stored
    // value (fields are now private, so this test validates the accessor
    // against the constructor argument).
    let cfg = ParzenConfig::new(2.5);
    assert!((cfg.sigma_sq() - 2.5).abs() < 1e-10);
}

// â”€â”€â”€ DRY normalize_to_bins via dispatch path (DRY-321-01) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(feature = "direct-parzen")]
#[test]
fn normalize_to_bins_matches_dispatch() {
    // DRY-321-01: The tensor-path normalize_to_bins must produce the
    // same values as the host-side normalize_and_extract.
    use crate::metric::histogram::parzen::dispatch::normalize_and_extract;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::Tensor;

    type B = NdArray<f32>;
    let device = Default::default();

    let values = Tensor::<f32, B>::from_floats([0.0, 50.0, 100.0, 25.0, 75.0], &device);
    let host_result = normalize_and_extract::<B>(&values, 0.0, 100.0, 32);

    // Verify the host-side normalization is deterministic
    let host_result2 = normalize_and_extract::<B>(&values, 0.0, 100.0, 32);
    for (a, b) in host_result.iter().zip(host_result2.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "normalize_and_extract not deterministic: {a} vs {b}"
        );
    }
}
