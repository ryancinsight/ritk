use crate::interpolation::kernel::sinc::{compute_lanczos_weights, lanczos_kernel};
use crate::interpolation::{LanczosInterpolator, SincInterpolator};
use coeus_core::SequentialBackend;
use ritk_core::interpolation::Interpolator;
use coeus_tensor::Tensor;
use ritk_image::tensor::{Shape, TensorData};

type TestBackend = SequentialBackend;

#[test]
fn test_lanczos_kernel_at_origin() {
    // At x=0, kernel should be exactly 1
    let k = lanczos_kernel::<3>(0.0);
    assert!((k - 1.0).abs() < 1e-6, "Expected 1.0 at origin, got {}", k);
}

#[test]
fn test_lanczos_kernel_outside_support() {
    // Outside window size, kernel should be 0
    let k3 = lanczos_kernel::<3>(3.5);
    assert!(k3.abs() < 1e-6, "Expected 0 outside support, got {}", k3);

    let k4 = lanczos_kernel::<4>(5.0);
    assert!(k4.abs() < 1e-6, "Expected 0 outside support, got {}", k4);
}

#[test]
fn test_lanczos_kernel_symmetry() {
    // Kernel should be symmetric around origin
    for x in &[0.1, 0.5, 1.0, 1.5, 2.0] {
        let k_pos = lanczos_kernel::<3>(*x);
        let k_neg = lanczos_kernel::<3>(-*x);
        assert!(
            (k_pos - k_neg).abs() < 1e-6,
            "Kernel not symmetric at x={}: {} vs {}",
            x,
            k_pos,
            k_neg
        );
    }
}

#[test]
fn test_lanczos_kernel_zeros() {
    // Lanczos kernel should have zeros at integer positions (except origin)
    for n in 1..=2 {
        let k = lanczos_kernel::<3>(n as f32);
        assert!(k.abs() < 1e-6, "Expected zero at x={}, got {}", n, k);
    }
}

#[test]
fn test_lanczos_weights_bounds() {
    let weights = compute_lanczos_weights::<3>(5.5, 10);
    for &(idx, _w) in &weights.taps[..weights.len] {
        assert!(idx >= 0, "Negative index in weights");
        assert!((idx as usize) < 10, "Index out of bounds in weights");
    }
}

#[test]
fn test_sinc_interpolator_2d_at_grid_points() {
    let device = Default::default();

    // Create a simple 4x4 image with known values
    let data_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec.clone(), ([4, 4])),
        &device,
    );

    let interpolator = SincInterpolator::new();

    // At integer coordinates, should return exact values
    let indices = Tensor::<f32, TestBackend>::from_floats(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        &device,
    );

    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    // At (0,0): value should be 0
    assert!(
        (slice[0] - 0.0).abs() < 0.1,
        "Expected ~0.0, got {}",
        slice[0]
    );
    // At (1,0): value should be 1
    assert!(
        (slice[1] - 1.0).abs() < 0.1,
        "Expected ~1.0, got {}",
        slice[1]
    );
    // At (0,1): value should be 4
    assert!(
        (slice[2] - 4.0).abs() < 0.1,
        "Expected ~4.0, got {}",
        slice[2]
    );
    // At (1,1): value should be 5
    assert!(
        (slice[3] - 5.0).abs() < 0.1,
        "Expected ~5.0, got {}",
        slice[3]
    );
}

#[test]
fn test_sinc_interpolator_3d_at_grid_points() {
    let device = Default::default();

    // Create a 2x2x2 volume
    let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec.clone(), ([2, 2, 2])),
        &device,
    );

    let interpolator = SincInterpolator::new();

    // Test at corner points
    let indices =
        Tensor::<f32, TestBackend>::from_floats([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], &device);

    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    // At (0,0,0): should be ~0
    assert!(
        (slice[0] - 0.0).abs() < 0.1,
        "Expected ~0.0, got {}",
        slice[0]
    );
    // At (1,1,1): should be ~111
    assert!(
        (slice[1] - 111.0).abs() < 0.1,
        "Expected ~111.0, got {}",
        slice[1]
    );
}

#[test]
fn test_sinc_interpolator_constant_image() {
    let device = Default::default();

    // Constant image: all values are 42.0
    let data_vec: Vec<f32> = vec![42.0; 64];
    let data =
        Tensor::<f32, TestBackend>::from_data((data_vec, ([8, 8])), &device);

    let interpolator = SincInterpolator::new();

    // Interpolate at various positions
    let indices = Tensor::<f32, TestBackend>::from_floats(
        [
            [0.5, 0.5], // Center of first quadrant
            [3.7, 2.3], // Arbitrary position
            [7.0, 7.0], // Near edge
        ],
        &device,
    );

    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    // The windowed-sinc kernel is not renormalized (ITK semantics — see
    // `compute_lanczos_weights`), so a constant `C` reconstructs as
    // `C · (Σ wx)(Σ wy)`, where the per-axis weight sum deviates from 1 by the
    // Lanczos-`A` partition-of-unity defect `δ = maxₓ |1 − Σ_k L_A(k − x)|`.
    // For `A = 3`, `δ ≈ 5.70e-3`, so the 2-D reconstruction error is bounded by
    // `C·(1 − (1 − δ)²) ≈ 42 · 0.01137 ≈ 0.478` (attained at the half-integer
    // offset (0.5, 0.5)). Renormalizing would null this defect but break parity
    // with `sitk.Resample(..., sitkLanczosWindowedSinc)`.
    const A3_PU_DEFECT: f32 = 5.70e-3;
    let tol = 42.0 * (1.0 - (1.0 - A3_PU_DEFECT).powi(2)) + 1e-3;
    for (i, &val) in slice.iter().enumerate() {
        assert!(
            (val - 42.0).abs() <= tol,
            "Expected 42.0 within windowed-sinc defect {tol:.4} at index {i}, got {val}",
        );
    }
}

#[test]
fn test_sinc_interpolator_bandlimited_signal() {
    // Test reconstruction of a bandlimited signal (cosine)
    // Sinc interpolation should perfectly reconstruct signals below Nyquist

    let device = Default::default();
    let n = 32;

    // Generate samples of cos(2πx/8) - frequency well below Nyquist (Nyquist = 0.5 cycles/pixel)
    let period = 8.0;
    let data_vec: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * (i as f32) / period).cos())
        .collect();

    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec.clone(), ([n])),
        &device,
    );

    let interpolator = SincInterpolator::new();

    // Reshape to 2D for interpolator (1D case not directly supported, use [1, N])
    let data_2d = data.clone().reshape([1, n]);
    let x_test = 7.5f32; // Half-pixel offset

    let indices = Tensor::<f32, TestBackend>::from_floats([[x_test, 0.0]], &device);
    let result = interpolator.interpolate(&data_2d, indices);
    let interpolated = result.into_data().as_slice::<f32>().unwrap()[0];

    // Expected value
    let expected = (2.0 * std::f32::consts::PI * x_test / period).cos();

    // Sinc interpolation should closely approximate the true value
    assert!(
        (interpolated - expected).abs() < 0.2,
        "Expected {:.4}, got {:.4}",
        expected,
        interpolated
    );
}

#[test]
fn test_lanczos_interpolator_various_window_sizes() {
    let device = Default::default();

    let data_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let data =
        Tensor::<f32, TestBackend>::from_data((data_vec, ([4, 4])), &device);

    // Test with different window sizes
    let interp3 = LanczosInterpolator::<3>::new();
    let interp4 = LanczosInterpolator::<4>::new();
    let interp5 = LanczosInterpolator::<5>::new();

    let indices = Tensor::<f32, TestBackend>::from_floats([[1.5, 1.5]], &device);

    let r3 = interp3
        .interpolate(&data, indices.clone())
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    let r4 = interp4
        .interpolate(&data, indices.clone())
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    let r5 = interp5
        .interpolate(&data, indices)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];

    // All should give reasonable results (not NaN, not wildly different)
    assert!(r3.is_finite(), "Lanczos-3 produced non-finite result");
    assert!(r4.is_finite(), "Lanczos-4 produced non-finite result");
    assert!(r5.is_finite(), "Lanczos-5 produced non-finite result");

    // Results should be in the valid range
    let min_val = 0.0f32;
    let max_val = 15.0f32;
    assert!(
        r3 >= min_val && r3 <= max_val,
        "Lanczos-3 result {} out of range",
        r3
    );
    assert!(
        r4 >= min_val && r4 <= max_val,
        "Lanczos-4 result {} out of range",
        r4
    );
    assert!(
        r5 >= min_val && r5 <= max_val,
        "Lanczos-5 result {} out of range",
        r5
    );
}

#[test]
#[should_panic(expected = "Lanczos window size must be >= 2")]
fn test_lanczos_interpolator_invalid_window_size() {
    let _ = LanczosInterpolator::<1>::new();
}
