use super::*;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn device() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

// ─── compute_oob_mask_3d ─────────────────────────────────────────────────

#[test]
fn oob_mask_3d_in_bounds_all_ones() {
    // 4×4×4 volume; every coordinate strictly inside returns 1.0
    let dev = device();
    // [x=1.5, y=1.5, z=1.5] — floor = [1,1,1], dims = [4,4,4] → in-bounds on all axes
    let indices =
        Tensor::<B, 2>::from_floats([[1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], &dev);
    let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
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
    // x=-1 (OOB), y=5 > d1-1=3 (OOB), z=-0.1 → floor=-1 (OOB)
    let indices =
        Tensor::<B, 2>::from_floats([[-1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, -0.1]], &dev);
    let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(vals, vec![0.0, 0.0, 0.0], "all OOB coords must give 0.0");
}

#[test]
fn oob_mask_3d_mixed_in_and_out() {
    let dev = device();
    // shape [Z=2, Y=4, X=4]: valid x in [0,3], y in [0,3], z in [0,1]
    let indices = Tensor::<B, 2>::from_floats(
        [
            [1.5, 1.5, 0.5],  // in-bounds
            [-0.5, 1.5, 0.5], // x OOB (floor=-1)
            [1.5, 4.0, 0.5],  // y OOB (floor=4 > 3)
            [1.5, 1.5, 2.0],  // z OOB (floor=2 > 1)
            [3.0, 3.0, 1.0],  // boundary, in-bounds
        ],
        &dev,
    );
    let mask = compute_oob_mask_3d(&indices, &[2, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(
        vals,
        vec![1.0, 0.0, 0.0, 0.0, 1.0],
        "mixed: in=1 OOB=0, boundary is in-bounds"
    );
}

// ─── compute_joint_histogram with OOB mask ───────────────────────────────

#[test]
fn oob_mask_zeros_out_oob_contribution() {
    // Verify that applying an all-zero OOB mask produces a zero histogram.
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
    let dev = device();

    let fixed = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let all_oob = Tensor::<B, 1>::zeros([3], &dev); // all samples are OOB

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
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
    let dev = device();

    // Three samples: first is identity (128, 128), rest are extreme (0, 255)
    let fixed = Tensor::<B, 1>::from_floats([128.0, 0.0, 255.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([128.0, 255.0, 0.0], &dev);
    // Only the first sample is in-bounds
    let partial_mask = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0], &dev);

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
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
    let dev = device();

    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);
    let all_in = Tensor::<B, 1>::ones([5], &dev);

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

/// Verify that the chunked-path W_fixed^T caching produces the same joint
/// histogram as the non-chunked path. When N > CHUNK_SIZE (32768), the
/// chunked code path slices cached W_fixed^T per chunk instead of recomputing
/// W_fixed from fixed values each time. This test constructs a scenario large
/// enough to trigger the chunked path and validates against the non-chunked
/// result.
#[test]
fn chunked_cached_path_matches_non_chunked() {
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Create a volume large enough to trigger chunking (N > 32768).
    // 64 × 32 × 32 = 65536 > 32768.
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
        Tensor::<B, 3>::from_data(TensorData::new(fixed_data, Shape::new(shape)), &device);
    let moving_t =
        Tensor::<B, 3>::from_data(TensorData::new(moving_data, Shape::new(shape)), &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed_img = Image::new(fixed_t, origin, spacing, direction);
    let moving_img = Image::new(moving_t, origin, spacing, direction);

    let interp = LinearInterpolator::new_zero_pad();
    let zero_translation = Tensor::<B, 1>::zeros([3], &device);
    let translation = TranslationTransform::<B, 3>::new(zero_translation);

    // Compute joint histogram via ParzenJointHistogram (triggers chunked path).
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0);
    let joint_chunked = hist.compute_image_joint_histogram(
        &fixed_img,
        &moving_img,
        &translation,
        &interp,
        None, // no sampling → triggers non-sampling chunked path
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
            (a - b).abs() < 1e-3,
            "chunked vs direct mismatch at bin {i}: {a} vs {b}"
        );
    }
}
