use burn::tensor::{backend::Backend, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

pub(crate) fn interpolate_1d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // Extract coordinate: [N, 1] -> [N]
    // narrow consumes self, but indices is owned so final narrow consumes it.
    let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

    // Compute floor coordinate — floor consumes self, so clone x0 for weight
    // derivation before floor moves x0. Original pattern used 3 clones
    // (floor, x0.clone() for weight, x0.clone() for x1, x0.clone() for clamp);
    // now only 1 clone remains (for the weight).
    let x0 = x.clone().floor();
    let wx = x - x0.clone();

    // Compute x1 — x0 still owned after weight derivation
    let x1 = x0.clone() + 1.0;

    // Clamp indices — x0 and x1 are consumed by clamp+int.
    let x0_i = x0.clone().clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d0 - 1) as f64).int();

    // Pre-flatten data — reshape consumes self, but data is &Tensor so clone once.
    let flat_data = data.clone().reshape([d0]);

    // Gather 2 values — flat_data must be cloned since gather consumes self on NdArray.
    let v0 = flat_data.clone().gather(0, x0_i);
    let v1 = flat_data.gather(0, x1_i);

    // Linear interpolation
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one - wx.clone();

    let result = v0 * one_minus_wx + v1 * wx;

    if let Some(mask) = in_bounds_mask(x0, (d0 - 1) as f64, mode) {
        result * mask
    } else {
        result
    }
}
