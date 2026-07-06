//! Differential coverage: `trilinear_interpolation` must be
//! value-identical to the Burn-generic `trilinear_interpolation` it mirrors.

use crate::interpolation::tensor_trilinear;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

/// Run both paths on the same [B, C, D, H, W] image and [B, 3, D', H', W']
/// grid and assert bitwise-identical output (both paths perform the same
/// sequence of scalar floating-point operations in the same order, so no
/// reordering-induced epsilon is expected — see numerical_discipline).
#[allow(clippy::too_many_arguments)]
fn assert_matches_burn(
    image: Vec<f32>,
    b: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    grid: Vec<f32>,
    out_d: usize,
    out_h: usize,
    out_w: usize,
) {
    let device = Default::default();
    let image_tensor = Tensor::<TestBackend, 5>::from_data(
        TensorData::new(image.clone(), Shape::new([b, c, d, h, w])),
        &device,
    );
    let grid_tensor = Tensor::<TestBackend, 5>::from_data(
        TensorData::new(grid.clone(), Shape::new([b, 3, out_d, out_h, out_w])),
        &device,
    );

    let burn_result = tensor_trilinear::trilinear_interpolation(image_tensor, grid_tensor);
    let burn_data = burn_result.into_data();
    let burn_slice: &[f32] = burn_data.as_slice().expect("burn result slice");

    let coeus_result =
        super::trilinear_interpolation::<f32>(&image, b, c, d, h, w, &grid, out_d, out_h, out_w);

    assert_eq!(
        coeus_result.len(),
        burn_slice.len(),
        "coeus and burn trilinear output length must match"
    );
    for (i, (&c_val, &b_val)) in coeus_result.iter().zip(burn_slice.iter()).enumerate() {
        assert_eq!(
            c_val, b_val,
            "coeus/burn trilinear divergence at flat index {i}: coeus={c_val}, burn={b_val}"
        );
    }
}

#[test]
fn matches_burn_center_sample() {
    // z=0: [1,2;3,4]  z=1: [5,6;7,8], sample at (0.5, 0.5, 0.5) -> mean of all 8 corners.
    assert_matches_burn(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        1,
        1,
        2,
        2,
        2,
        vec![0.5, 0.5, 0.5],
        1,
        1,
        1,
    );
}

#[test]
fn matches_burn_multi_channel_exact_corner() {
    let mut data = vec![10.0_f32; 8];
    data.extend(vec![20.0_f32; 8]);
    assert_matches_burn(data, 1, 2, 2, 2, 2, vec![0.0, 0.0, 0.0], 1, 1, 1);
}

#[test]
fn matches_burn_negative_coordinate_extrapolation() {
    // Sampling below the grid (negative z/y/x) exercises the independent
    // per-neighbor clamp path (both z0 and z1 clamp to index 0).
    assert_matches_burn(
        (0..27).map(|i| i as f32).collect(),
        1,
        1,
        3,
        3,
        3,
        vec![-1.5, -1.5, -1.5],
        1,
        1,
        1,
    );
}

#[test]
fn matches_burn_beyond_extent_extrapolation() {
    // Sampling above the grid exercises the top-edge clamp path.
    assert_matches_burn(
        (0..27).map(|i| i as f32).collect(),
        1,
        1,
        3,
        3,
        3,
        vec![5.0, 5.0, 5.0],
        1,
        1,
        1,
    );
}

#[test]
fn matches_burn_multi_batch_multi_point_grid() {
    let image: Vec<f32> = (0..2 * 4 * 4 * 4).map(|i| i as f32 * 0.5).collect();
    // 2 batches, out grid of 2x2x2 sample points per batch, non-integer coords.
    let grid: Vec<f32> = vec![
        0.25, 1.75, 2.5, 0.5, // z coords for 4 sample points, batch 0
        0.75, 1.25, 3.0, 0.1, // y coords, batch 0
        1.5, 0.0, 2.9, 0.4, // x coords, batch 0
        3.0, 0.0, 1.1, 2.6, // z coords, batch 1
        0.0, 3.0, 0.6, 1.9, // y coords, batch 1
        2.2, 1.0, 0.0, 3.0, // x coords, batch 1
    ];
    assert_matches_burn(image, 2, 1, 4, 4, 4, grid, 2, 2, 1);
}
