//! Out-of-bounds mask computation for multi-dimensional voxel indices.

/// Compute a `{0.0, 1.0}` in-bounds mask for moving-image voxel indices.
///
/// Returns an `[N]` float tensor: `1.0` = in-bounds, `0.0` = out-of-bounds.
/// Mirrors the zero-pad criterion in `LinearInterpolator`: a sample is
/// in-bounds when `floor(coord_d) ∈ [0, dim_d − 1]` for every axis.
///
/// Column convention matches the interpolation kernels:
/// column `c` maps to `shape[D - 1 - c]`.
pub fn compute_oob_mask<B: ritk_image::tensor::Backend>(
    indices: &ritk_image::tensor::Tensor<B, 2>,
    shape: &[usize],
) -> ritk_image::tensor::Tensor<B, 1> {
    assert!(!shape.is_empty(), "image dimensionality must be non-zero");

    let [n, _] = indices.dims();
    let device = indices.device();
    let mut mask = ritk_image::tensor::Tensor::<B, 1>::ones([n], &device);
    let dims = shape.len();

    for column in 0..dims {
        let axis = dims - 1 - column;
        let coord = indices.clone().narrow(1, column, 1).squeeze_dims(&[1]);
        let lower = coord.floor();
        let in_axis = lower
            .clone()
            .equal(lower.clamp(0.0, (shape[axis] - 1) as f64))
            .float();
        mask = mask * in_axis;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::compute_oob_mask;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::Tensor;

    type B = NdArray<f32>;

    #[test]
    fn oob_mask_respects_inner_first_columns_for_2d() {
        let device = Default::default();
        let indices = Tensor::<B, 2>::from_floats(
            [[0.0, 0.0], [2.0, 1.0], [3.0, 1.0], [2.0, 2.0], [-1.0, 0.0]],
            &device,
        );

        let mask = compute_oob_mask(&indices, &[2, 3]);
        let values = mask.into_data().into_vec::<f32>().unwrap();

        assert_eq!(values, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
