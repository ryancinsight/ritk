use burn::tensor::{backend::Backend, Int, Tensor};

#[inline]
pub(crate) fn gather_3d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: &Tensor<B, 1, Int>,
    yi: &Tensor<B, 1, Int>,
    zi: &Tensor<B, 1, Int>,
    stride_y: i32,
    stride_z: i32,
) -> Tensor<B, 1> {
    let idx = zi.clone() * stride_z + yi.clone() * stride_y + xi.clone();
    flat_data.clone().gather(0, idx)
}

pub(crate) fn interpolate_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Z
    let d1 = shape.dims[1]; // Y
    let d2 = shape.dims[2]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // Extract coordinates using narrow to avoid unnecessary clones
    // Extract coordinates: narrow gives [Batch, 1], squeeze_dims removes dim 1 to get [Batch]
    // indices: [Batch, 3] -> (x, y, z)
    let x = indices.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
    let z = indices.narrow(1, 2, 1).squeeze_dims(&[1]);

    // Compute floor coordinates
    let x0 = x.clone().floor();
    let y0 = y.clone().floor();
    let z0 = z.clone().floor();

    // Compute interpolation weights
    let wx = x - x0.clone();
    let wy = y - y0.clone();
    let wz = z - z0.clone();

    // Compute x1, y1, z1
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;
    let z1 = z0.clone() + 1.0;

    // Clamp indices to valid range
    let x0_i = x0.clamp(0.0, (d2 - 1) as f64).int();
    let y0_i = y0.clamp(0.0, (d1 - 1) as f64).int();
    let z0_i = z0.clamp(0.0, (d0 - 1) as f64).int();

    let x1_i = x1.clamp(0.0, (d2 - 1) as f64).int();
    let y1_i = y1.clamp(0.0, (d1 - 1) as f64).int();
    let z1_i = z1.clamp(0.0, (d0 - 1) as f64).int();

    // Stride for [Z, Y, X] layout (d0, d1, d2)
    let stride_z = (d1 * d2) as i32;
    let stride_y = d2 as i32;

    // Pre-flatten data once to avoid repeated reshaping
    let flat_data = data.clone().reshape([d0 * d1 * d2]);

    // Gather all 8 voxel values
    let v000 = gather_3d(&flat_data, &x0_i, &y0_i, &z0_i, stride_y, stride_z);
    let v001 = gather_3d(&flat_data, &x0_i, &y0_i, &z1_i, stride_y, stride_z);
    let v010 = gather_3d(&flat_data, &x0_i, &y1_i, &z0_i, stride_y, stride_z);
    let v011 = gather_3d(&flat_data, &x0_i, &y1_i, &z1_i, stride_y, stride_z);
    let v100 = gather_3d(&flat_data, &x1_i, &y0_i, &z0_i, stride_y, stride_z);
    let v101 = gather_3d(&flat_data, &x1_i, &y0_i, &z1_i, stride_y, stride_z);
    let v110 = gather_3d(&flat_data, &x1_i, &y1_i, &z0_i, stride_y, stride_z);
    let v111 = gather_3d(&flat_data, &x1_i, &y1_i, &z1_i, stride_y, stride_z);

    // Pre-compute (1 - weight) values to reduce operations
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one.clone() - wy.clone();
    let one_minus_wz = one - wz.clone();

    // Trilinear interpolation
    // Interpolate along X
    let c00 = v000 * one_minus_wx.clone() + v100 * wx.clone();
    let c01 = v001 * one_minus_wx.clone() + v101 * wx.clone();
    let c10 = v010 * one_minus_wx.clone() + v110 * wx.clone();
    let c11 = v011 * one_minus_wx + v111 * wx;

    // Interpolate along Y
    let c0 = c00 * one_minus_wy.clone() + c10 * wy.clone();
    let c1 = c01 * one_minus_wy.clone() + c11 * wy.clone();

    // Interpolate along Z
    c0 * one_minus_wz + c1 * wz
}
