use burn::tensor::{backend::Backend, Int, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

#[inline]
pub(crate) fn gather_2d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: &Tensor<B, 1, Int>,
    yi: &Tensor<B, 1, Int>,
    stride_y: i32,
) -> Tensor<B, 1> {
    let idx = yi.clone() * stride_y + xi.clone();
    flat_data.clone().gather(0, idx)
}

pub(crate) fn interpolate_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Y
    let d1 = shape.dims[1]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // Extract coordinates: narrow consumes self, so clone indices once and narrow each column.
    let indices_local = indices;
    let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices_local.narrow(1, 1, 1).squeeze_dims(&[1]);

    // Compute floor coordinates and weights.
    // Original: x0.clone().floor(), x0.clone() for weight, x0.clone() for x1,
    //   x0.clone() for clamp = 4 clones.
    // Now: clone x0 once for weight derivation, then consume x0 for x1,
    //   then consume for clamp = 1 clone.
    let x0 = x.clone().floor();
    let wx = x - x0.clone();
    let y0 = y.clone().floor();
    let wy = y - y0.clone();

    // Compute x1, y1 — x0/y0 still owned after weight derivation
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;

    // Clamp indices — x0/y0 and x1/y1 consumed by clamp+int.
    let x0_i = x0.clone().clamp(0.0, (d1 - 1) as f64).int();
    let y0_i = y0.clone().clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d1 - 1) as f64).int();
    let y1_i = y1.clamp(0.0, (d0 - 1) as f64).int();

    // Stride for [Y, X] layout (d0, d1)
    let stride_y = d1 as i32;

    // Pre-flatten data — reshape consumes self, but data is &Tensor so clone once.
    let flat_data = data.clone().reshape([d0 * d1]);

    // Gather all 4 voxel values — gather consumes self on NdArray, so flat_data must be cloned
    // inside gather_2d (handled there).
    let v00 = gather_2d(&flat_data, &x0_i, &y0_i, stride_y);
    let v01 = gather_2d(&flat_data, &x0_i, &y1_i, stride_y);
    let v10 = gather_2d(&flat_data, &x1_i, &y0_i, stride_y);
    let v11 = gather_2d(&flat_data, &x1_i, &y1_i, stride_y);

    // Pre-compute (1 - weight)
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one - wy.clone();

    // Bilinear interpolation
    let c0 = v00 * one_minus_wx.clone() + v10 * wx.clone();
    let c1 = v01 * one_minus_wx + v11 * wx;

    let result = c0 * one_minus_wy + c1 * wy;

    let x_mask = in_bounds_mask(x0, (d1 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y0, (d0 - 1) as f64, mode);

    match (x_mask, y_mask) {
        (Some(xm), Some(ym)) => result * xm * ym,
        _ => result,
    }
}
