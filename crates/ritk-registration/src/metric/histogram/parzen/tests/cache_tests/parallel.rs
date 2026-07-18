//! Parallel dispatch path verification.

use super::super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn direct_parallel_matches_sparse() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);

    let n = 1000;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.18 + 10.0) % 255.0).collect();

    let fixed_tensor = Tensor::<f32, B>::from_floats(fixed.as_slice(), &dev);
    let moving_tensor = Tensor::<f32, B>::from_floats(moving.as_slice(), &dev);

    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed_tensor, &moving_tensor, None);
    let dispatch_data = dispatch_hist.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

    let num_bins = hist.num_bins;
    let fix_min = hist.min_intensity;
    let fix_max = hist.max_intensity;
    let fix_sigma = hist.parzen_sigma;
    let mov_sigma = hist.moving_parzen_sigma.unwrap_or(fix_sigma);
    let mov_min = hist.moving_min_intensity.unwrap_or(fix_min);
    let mov_max = hist.moving_max_intensity.unwrap_or(fix_max);

    let sigma_sq_fix =
        direct::ParzenConfig::from_intensity_sigma(fix_sigma, fix_min, fix_max, num_bins)
            .sigma_sq();
    let sigma_sq_mov =
        direct::ParzenConfig::from_intensity_sigma(mov_sigma, mov_min, mov_max, num_bins)
            .sigma_sq();

    let fixed_norm = dispatch::normalize_and_extract(&fixed_tensor, fix_min, fix_max, num_bins);
    let moving_norm = dispatch::normalize_and_extract(&moving_tensor, mov_min, mov_max, num_bins);

    let sparse = direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);
    let sparse_data = direct::compute_joint_histogram_from_cache_sparse(
        &sparse,
        &moving_norm,
        num_bins,
        sigma_sq_mov,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in dispatch_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "parallel direct vs sparse nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }
    let dispatch_total: f32 = dispatch_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    assert!(
        dispatch_total > 0.0,
        "dispatch total {dispatch_total} must be positive"
    );
    assert!(
        sparse_total > 0.0,
        "sparse total {sparse_total} must be positive"
    );
    let ratio = dispatch_total / sparse_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "dispatch/sparse total ratio {ratio} should be â‰ˆ 1.0 (SPARSE-329-01)"
    );
}
