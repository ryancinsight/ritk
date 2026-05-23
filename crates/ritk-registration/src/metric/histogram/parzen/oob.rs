use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Compute a `{0.0, 1.0}` in-bounds mask for 3-D moving-image voxel indices.
///
/// Returns an `[N]` float tensor: `1.0` = in-bounds, `0.0` = out-of-bounds.
/// Mirrors the zero-pad criterion in `LinearInterpolator`: a sample is
/// in-bounds when `floor(coord_d) ∈ [0, dim_d − 1]` for every axis.
///
/// Column convention (matches `interpolation::linear::dim3`):
/// - column 0 → x (→ `shape[2]`, the X / last dimension)
/// - column 1 → y (→ `shape[1]`, the Y / middle dimension)
/// - column 2 → z (→ `shape[0]`, the Z / first dimension)
pub(crate) fn compute_oob_mask_3d<B: Backend>(
    indices: &Tensor<B, 2>, // [N, 3]
    shape: &[usize],        // at least 3 elements: [d0=Z, d1=Y, d2=X]
) -> Tensor<B, 1> {
    let d0 = shape[0]; // Z
    let d1 = shape[1]; // Y
    let d2 = shape[2]; // X

    let x = indices.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
    let z = indices.clone().narrow(1, 2, 1).squeeze_dims(&[1]);

    let x0 = x.clone().floor();
    let y0 = y.clone().floor();
    let z0 = z.clone().floor();

    let x_in = x0.clone().equal(x0.clamp(0.0, (d2 - 1) as f64)).float();
    let y_in = y0.clone().equal(y0.clamp(0.0, (d1 - 1) as f64)).float();
    let z_in = z0.clone().equal(z0.clamp(0.0, (d0 - 1) as f64)).float();

    x_in * y_in * z_in
}
