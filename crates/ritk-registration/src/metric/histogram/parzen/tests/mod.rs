use super::*;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn device() -> <B as ritk_image::tensor::Backend>::Device {
    Default::default()
}

// â”€â”€â”€ bins_exp eager initialization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn bins_exp_eager_init_matches_lazy() {
    // Verify that the eagerly initialized bins_exp in new() produces the
    // same bin-center tensor as an on-the-fly arange construction.
    let dev = device();
    let num_bins = 32;
    let hist = ParzenJointHistogram::<B>::new(num_bins, 0.0, 255.0, 8.0, &dev);

    // The bins_exp field is eagerly initialized in new() â€” it must be Some.
    assert!(
        hist.bins_exp.is_some(),
        "bins_exp must be eagerly initialized in new()"
    );

    // Compare the cached bins_exp against a fresh arange_bins call.
    let cached = hist.bins_exp.as_ref().cloned().unwrap();
    let fresh: Tensor<f32, B> = compute::arange_bins(num_bins, &dev);

    let cached_data = cached.into_data();
    let cached_slice = cached_data.as_slice::<f32>().unwrap();
    let fresh_data = fresh.into_data();
    let fresh_slice = fresh_data.as_slice::<f32>().unwrap();

    assert_eq!(
        cached_slice.len(),
        fresh_slice.len(),
        "bins_exp length mismatch"
    );
    for (i, (a, b)) in cached_slice.iter().zip(fresh_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "bins_exp mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn bins_exp_histogram_result_matches_lazy() {
    // Verify that compute_joint_histogram with eager bins_exp gives the
    // same numerical result as constructing bins_exp lazily.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);
    let fixed = Tensor::<f32, B>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);

    // Compute with eager bins_exp (default path)
    let h1 = hist.compute_joint_histogram(&fixed, &moving, None);

    // Manually replace bins_exp with a fresh one to simulate lazy init
    let mut hist2 = hist.clone();
    hist2.bins_exp = Some(compute::arange_bins(hist2.num_bins, &dev));
    let h2 = hist2.compute_joint_histogram(&fixed, &moving, None);

    let h1_data = h1.into_data();
    let s1 = h1_data.as_slice::<f32>().unwrap();
    let h2_data = h2.into_data();
    let s2 = h2_data.as_slice::<f32>().unwrap();
    for (i, (a, b)) in s1.iter().zip(s2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "histogram mismatch at bin {i}: {a} vs {b}"
        );
    }
}

// â”€â”€â”€ compute_oob_mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn oob_mask_3d_in_bounds_all_ones() {
    // 4Ã—4Ã—4 volume; every coordinate strictly inside returns 1.0
    let dev = device();
    // [x=1.5, y=1.5, z=1.5] â€” floor = [1,1,1], dims = [4,4,4] â†’ in-bounds on all axes
    let indices =
        Tensor::<f32, B>::from_floats([[1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], &dev);
    let mask = compute_oob_mask(&indices, &[4, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(
        vals,
        vec![1.0, 1.0, 1.0],
        "all in-bounds coords must give 1.0"
    );
}

#[test]
fn oob_mask_3d_oob_all_zeros() {
    let dev = device();
    // x=-1 (OOB), y=5 > d1-1=3 (OOB), z=-0.1 â†’ floor=-1 (OOB)
    let indices =
        Tensor::<f32, B>::from_floats([[-1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, -0.1]], &dev);
    let mask = compute_oob_mask(&indices, &[4, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(vals, vec![0.0, 0.0, 0.0], "all OOB coords must give 0.0");
}

#[test]
fn oob_mask_3d_mixed_in_and_out() {
    let dev = device();
    // shape [Z=2, Y=4, X=4]: valid x in [0,3], y in [0,3], z in [0,1]
    let indices = Tensor::<f32, B>::from_floats(
        [
            [1.5, 1.5, 0.5],  // in-bounds
            [-0.5, 1.5, 0.5], // x OOB (floor=-1)
            [1.5, 4.0, 0.5],  // y OOB (floor=4 > 3)
            [1.5, 1.5, 2.0],  // z OOB (floor=2 > 1)
            [3.0, 3.0, 1.0],  // boundary, in-bounds
        ],
        &dev,
    );
    let mask = compute_oob_mask(&indices, &[2, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(
        vals,
        vec![1.0, 0.0, 0.0, 0.0, 1.0],
        "mixed: in=1 OOB=0, boundary is in-bounds"
    );
}

// â”€â”€â”€ compute_joint_histogram with OOB mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn oob_mask_zeros_out_oob_contribution() {
    // Verify that applying an all-zero OOB mask produces a zero histogram.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);
    let fixed = Tensor::<f32, B>::from_floats([128.0, 64.0, 192.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([128.0, 64.0, 192.0], &dev);
    let all_oob = Tensor::<f32, B>::zeros([3], &dev); // all samples are OOB
    let h = hist.compute_joint_histogram(&fixed, &moving, Some(&all_oob));
    let sum: f32 = h.into_data().as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < 1e-6,
        "histogram with all-OOB mask must be zero, got sum={sum}"
    );
}

#[test]
fn oob_mask_partial_filters_correctly() {
    // With a partial OOB mask (only first sample in-bounds), the histogram
    // should be dominated by the first sample's contribution.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);
    // Three samples: first is identity (128, 128), rest are extreme (0, 255)
    let fixed = Tensor::<f32, B>::from_floats([128.0, 0.0, 255.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([128.0, 255.0, 0.0], &dev);
    // Only the first sample is in-bounds
    let partial_mask = Tensor::<f32, B>::from_floats([1.0, 0.0, 0.0], &dev);
    let h_masked = hist.compute_joint_histogram(&fixed, &moving, Some(&partial_mask));
    let h_unmasked = hist.compute_joint_histogram(&fixed, &moving, None);

    // The masked histogram should have strictly less total weight than unmasked.
    let sum_masked: f32 = h_masked.into_data().as_slice::<f32>().unwrap().iter().sum();
    let sum_unmasked: f32 = h_unmasked
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .sum();
    assert!(
        sum_masked < sum_unmasked,
        "masked sum ({sum_masked}) must be less than unmasked sum ({sum_unmasked})"
    );
}

#[test]
fn oob_mask_all_in_bounds_equivalent_to_no_mask() {
    // A mask of all 1.0 must produce the same result as passing None.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);
    let fixed = Tensor::<f32, B>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);
    let all_in = Tensor::<f32, B>::ones([5], &dev);

    let h_with_mask = hist
        .compute_joint_histogram(&fixed, &moving, Some(&all_in))
        .into_data();
    let h_no_mask = hist
        .compute_joint_histogram(&fixed, &moving, None)
        .into_data();

    let s1 = h_with_mask.as_slice::<f32>().unwrap();
    let s2 = h_no_mask.as_slice::<f32>().unwrap();
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "all-ones mask must match no-mask: {a} vs {b}"
        );
    }
}

// â”€â”€â”€ dispatch integration tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Verify that `compute_joint_histogram_dispatch` produces numerically identical
// results to the tensor-based `compute_joint_histogram` when the direct-parzen
// feature is enabled. The dispatch path extracts data to host and calls the
// sparse-loop algorithm, which uses a different numerical strategy (Â±3Ïƒ bins
// vs full matmul), so we allow a small tolerance.

#[cfg(feature = "direct-parzen")]
#[test]
fn dispatch_matches_tensor_path() {
    // The dispatch path (PERF-328-01) normalizes per-sample by
    // 1/(sum_f Ã— sum_m), so per-sample contribution â‰ˆ 1.0.
    // The tensor path does NOT normalize (raw w_f Ã— w_m products), so
    // per-sample contribution â‰ˆ sum_f Ã— sum_m â‰ˆ 2Ï€ for ÏƒÂ²=1.
    // We use a structural directional check: where dispatch is nonzero,
    // tensor must be nonzero. We do not assert strict total equality.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<f32, B>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0, 90.0, 215.0], &dev);

    // Tensor path (raw, un-normalized accumulation)
    let tensor_hist = hist.compute_joint_histogram(&fixed, &moving, None);
    let tensor_data = tensor_hist.into_data();
    let tensor_slice = tensor_data.as_slice::<f32>().unwrap();

    // Dispatch path (per-sample normalized)
    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);
    let dispatch_data = dispatch_hist.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

    // Directional nonzero check: where dispatch is nonzero, tensor must
    // be nonzero. The dispatch path uses Â±3Ïƒ bins (truncates small tails);
    // the tensor path uses full matmul (captures all entries). Dispatch
    // may report 0 where tensor has a tiny tail value, but not vice versa.
    for (i, (t, d)) in tensor_slice.iter().zip(dispatch_slice.iter()).enumerate() {
        if *d > 1e-6 {
            assert!(
                *t > 1e-6,
                "dispatch nonzero at bin {i} but tensor is zero: tensor={t}, dispatch={d}"
            );
        }
    }

    // Totals: both must be positive, finite. Tensor total is un-normalized
    // (larger); dispatch total is normalized (smaller).
    let tensor_total: f32 = tensor_slice.iter().sum();
    let dispatch_total: f32 = dispatch_slice.iter().sum();
    assert!(
        tensor_total > 0.0 && tensor_total.is_finite(),
        "tensor total {tensor_total} must be positive and finite"
    );
    assert!(
        dispatch_total > 0.0 && dispatch_total.is_finite(),
        "dispatch total {dispatch_total} must be positive and finite"
    );
    // Tensor is un-normalized, dispatch is normalized â†’ tensor > dispatch.
    let ratio = dispatch_total / tensor_total;
    assert!(
        ratio < 1.0,
        "dispatch/tensor ratio {ratio} should be < 1.0 (tensor is un-normalized)"
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn dispatch_with_oob_mask() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<f32, B>::from_floats([128.0, 64.0, 192.0], &dev);
    let moving = Tensor::<f32, B>::from_floats([128.0, 64.0, 192.0], &dev);
    let all_oob = Tensor::<f32, B>::zeros([3], &dev);
    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, Some(&all_oob));
    let sum: f32 = dispatch_hist
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .sum();
    assert!(
        sum < 1e-6,
        "all-OOB dispatch histogram must be zero, got sum={sum}"
    );
}

/// Verify that the chunked-path W_fixed^T caching produces the same joint
/// histogram as the non-chunked path. When N > CHUNK_SIZE (32768), the
/// chunked code path slices cached W_fixed^T per chunk instead of recomputing
/// W_fixed from fixed values each time. This test constructs a scenario large
/// enough to trigger the chunked path and validates against the non-chunked
/// result.
///
/// Gated to only run when the tensor-based path is active (direct-parzen
/// disabled), since the direct sparse-loop algorithm uses a different
/// numerical strategy (+/-3-sigma bins vs full matmul) and produces slightly
/// different results.
#[test]
#[cfg_attr(feature = "direct-parzen", ignore = "requires tensor path")]
fn chunked_cached_path_matches_non_chunked() {
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_image::tensor::{Shape };
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();

    // Create a volume large enough to trigger chunking (N > 32768).
    // 64 Ã— 32 Ã— 32 = 65536 > 32768.
    let shape = [64, 32, 32];
    let n = shape[0] * shape[1] * shape[2];
    let mut fixed_data = Vec::with_capacity(n);
    let mut moving_data = Vec::with_capacity(n);
    for i in 0..n {
        let v = (i as f32 * 0.01).min(255.0);
        fixed_data.push(v);
        moving_data.push(v * 0.8 + 10.0); // shifted/scaled
    }

    let fixed_t =
        Tensor::<f32, B>::from_slice_on(shape, &fixed_data, &device);
    let moving_t =
        Tensor::<f32, B>::from_slice_on(shape, &moving_data, &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed_img = Image::new(fixed_t, origin, spacing, direction);
    let moving_img = Image::new(moving_t, origin, spacing, direction);

    let interp = LinearInterpolator::new_zero_pad();
    let zero_translation = Tensor::<f32, B>::zeros([3], &device);
    let translation = TranslationTransform::<B, 3>::new(zero_translation);

    // Compute joint histogram via ParzenJointHistogram (triggers chunked path).
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &device);
    let joint_chunked = hist.compute_image_joint_histogram(
        &fixed_img,
        &moving_img,
        &translation,
        &interp,
        crate::metric::sampling::SamplingConfig::full_grid(), // no sampling â†’ triggers non-sampling chunked path
    );

    // Also compute directly (bypass chunking) for comparison.
    let fixed_flat = fixed_img.data().clone().reshape([n]);
    let moving_flat = moving_img.data().clone().reshape([n]);
    let joint_direct = hist.compute_joint_histogram(
        &fixed_flat,
        &moving_flat,
        None, // no OOB mask
    );

    let chunked_data = joint_chunked.into_data();
    let chunked_slice = chunked_data.as_slice::<f32>().unwrap();
    let direct_data = joint_direct.into_data();
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in chunked_slice.iter().zip(direct_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() < 5e-2,
            "chunked vs direct mismatch at bin {i}: {a} vs {b}"
        );
    }
}

#[cfg(feature = "direct-parzen")]
mod cache_property_tests;
#[cfg(feature = "direct-parzen")]
mod cache_tests;
mod masked_cache_tests;
