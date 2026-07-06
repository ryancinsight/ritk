use super::*;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// [1, 1, 2, 2, 2] volume; voxel value = z*4 + y*2 + x (row-major, values 0..7).
fn unit_cube() -> Tensor<B, 5> {
    let vals: Vec<f32> = (0..8).map(|i| i as f32).collect();
    Tensor::<B, 5>::from_data(
        TensorData::new(vals, Shape::new([1, 1, 2, 2, 2])),
        &Default::default(),
    )
}

/// Constant grid [1, 3, 2, 2, 2] — every output voxel samples the same (z,y,x).
fn constant_grid(z: f32, y: f32, x: f32) -> Tensor<B, 5> {
    let n = 8usize; // 2×2×2 spatial
    let mut vals = Vec::with_capacity(3 * n);
    vals.extend(std::iter::repeat_n(z, n));
    vals.extend(std::iter::repeat_n(y, n));
    vals.extend(std::iter::repeat_n(x, n));
    Tensor::<B, 5>::from_data(
        TensorData::new(vals, Shape::new([1, 3, 2, 2, 2])),
        &Default::default(),
    )
}

fn assert_all_close(out: Tensor<B, 5>, expected: f32, eps: f32) {
    let data = out.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    for (i, &v) in slice.iter().enumerate() {
        assert!(
            (v - expected).abs() < eps,
            "voxel[{i}]: expected {expected}, got {v}"
        );
    }
}

#[test]
fn sampling_at_corner_000_returns_value_0() {
    // wz0=1, wy0=1, wx0=1 → full weight on v000=0.
    let out = trilinear_interpolation::<B>(unit_cube(), constant_grid(0.0, 0.0, 0.0));
    assert_all_close(out, 0.0, 1e-5);
}

#[test]
fn sampling_at_corner_111_returns_value_7() {
    // wz1=1, wy1=1, wx1=1 → full weight on v111=7.
    let out = trilinear_interpolation::<B>(unit_cube(), constant_grid(1.0, 1.0, 1.0));
    assert_all_close(out, 7.0, 1e-5);
}

#[test]
fn center_sample_returns_arithmetic_mean_3_5() {
    // At (0.5,0.5,0.5): all 8 trilinear weights = 0.5^3 = 0.125.
    // Result = 0.125 × Σ(0..7) = 0.125 × 28 = 3.5.
    let out = trilinear_interpolation::<B>(unit_cube(), constant_grid(0.5, 0.5, 0.5));
    assert_all_close(out, 3.5, 1e-5);
}

#[test]
fn out_of_bounds_low_clamps_to_corner_000() {
    // z=-1.0: floor=-1.0, z0_idx=clamp(-1,0,1)=0, z1_idx=clamp(0,0,1)=0.
    // wz1 = -1.0 - (-1.0) = 0.0, wz0 = 1.0 → full weight on edge index 0. Same y, x.
    let out = trilinear_interpolation::<B>(unit_cube(), constant_grid(-1.0, -1.0, -1.0));
    assert_all_close(out, 0.0, 1e-5);
}

#[test]
fn out_of_bounds_high_clamps_to_corner_111() {
    // z=2.5: floor=2.0, z0_idx=clamp(2,0,1)=1, z1_idx=clamp(3,0,1)=1.
    // wz1=0.5, wz0=0.5; both indices=1 → full weight on index 1. Same y, x → v111=7.
    let out = trilinear_interpolation::<B>(unit_cube(), constant_grid(2.5, 2.5, 2.5));
    assert_all_close(out, 7.0, 1e-5);
}

#[test]
fn multichannel_channels_interpolated_independently() {
    // Two-channel volume: ch0 constant 1.0, ch1 constant 2.0.
    // Any sampling point must preserve per-channel constant values.
    let ch0 = [1.0f32; 8];
    let ch1 = [2.0f32; 8];
    let vals: Vec<f32> = ch0.into_iter().chain(ch1).collect();
    let image = Tensor::<B, 5>::from_data(
        TensorData::new(vals, Shape::new([1, 2, 2, 2, 2])),
        &Default::default(),
    );
    let out = trilinear_interpolation::<B>(image, constant_grid(0.5, 0.5, 0.5));
    let data = out.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    assert_eq!(slice.len(), 16);
    for (i, &v) in slice[0..8].iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "ch0[{i}]: expected 1.0, got {v}");
    }
    for (i, &v) in slice[8..16].iter().enumerate() {
        assert!((v - 2.0).abs() < 1e-5, "ch1[{i}]: expected 2.0, got {v}");
    }
}
