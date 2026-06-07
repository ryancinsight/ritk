//! Property-based histogram tests: symmetry, normalization, boundary.

#![allow(clippy::needless_range_loop)]

use super::super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_symmetry_identical_images() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed =
        Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0, 40.0], &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed, &fixed, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    let num_bins = 16;
    for a in 0..num_bins {
        for b in 0..num_bins {
            let ab = slice[a * num_bins + b];
            let ba = slice[b * num_bins + a];
            let diff = (ab - ba).abs();
            assert!(
                diff < 1e-4,
                "symmetry violation at ({a},{b}): H[a,b]={ab}, H[b,a]={ba}, diff={diff}"
            );
        }
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_normalization_total_weight() {
    let dev = device();
    let n = 100;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 2.55) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 1.87 + 5.0) % 255.0).collect();
    let fixed_tensor = Tensor::<B, 1>::from_floats(fixed.as_slice(), &dev);
    let moving_tensor = Tensor::<B, 1>::from_floats(moving.as_slice(), &dev);
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed_tensor, &moving_tensor, None);
    let sum: f32 = h.into_data().as_slice::<f32>().unwrap().iter().sum();
    let expected_min = n as f32 * 0.5;
    let expected_max = n as f32 * 1.5;
    assert!(
        sum > expected_min,
        "normalized histogram total {sum} should be > {expected_min} (n × 0.5)"
    );
    assert!(
        sum < expected_max,
        "normalized histogram total {sum} should be < {expected_max} (n × 1.5)"
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_boundary_bins_populated() {
    let dev = device();
    let fixed = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0, 0.0, 0.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0, 0.0, 0.0], &dev);
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    let num_bins = 32;
    assert!(
        slice[0] > 0.1,
        "bin (0,0) must be populated for zero-valued samples, got {}",
        slice[0]
    );
    let sum_first_4_rows: f32 = slice[0..4].iter().sum();
    assert!(
        sum_first_4_rows > 0.5,
        "first 4 bins in row 0 should have weight, got {sum_first_4_rows}"
    );
    for b in 5..num_bins {
        assert!(
            slice[b] < 1e-6,
            "bin (0, {b}) should be ~0 beyond support, got {}",
            slice[b]
        );
    }
}
