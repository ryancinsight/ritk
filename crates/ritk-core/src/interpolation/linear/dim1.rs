use burn::tensor::{backend::Backend, Tensor};

pub(crate) fn interpolate_1d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // Extract coordinate: [N, 1] -> [N]
    let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

    // Compute floor coordinate
    let x0 = x.clone().floor();

    // Compute interpolation weight
    let wx = x - x0.clone();

    // Compute x1
    let x1 = x0.clone() + 1.0;

    // Clamp indices
    let x0_i = x0.clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d0 - 1) as f64).int();

    // Pre-flatten data (identity for 1D but keeps types consistent)
    let flat_data = data.clone().reshape([d0]);

    // Gather 2 values
    let v0 = flat_data.clone().gather(0, x0_i);
    let v1 = flat_data.clone().gather(0, x1_i);

    // Linear interpolation
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one - wx.clone();

    v0 * one_minus_wx + v1 * wx
}
