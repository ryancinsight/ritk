use burn::tensor::{backend::Backend, Int, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

#[inline]
pub(crate) fn gather_4d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: &Tensor<B, 1, Int>,
    yi: &Tensor<B, 1, Int>,
    zi: &Tensor<B, 1, Int>,
    wi: &Tensor<B, 1, Int>,
    strides: [i32; 3],
) -> Tensor<B, 1> {
    let [stride_y, stride_z, stride_w] = strides;
    let idx = wi.clone() * stride_w + zi.clone() * stride_z + yi.clone() * stride_y + xi.clone();
    flat_data.clone().gather(0, idx)
}

pub(crate) fn interpolate_4d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // W (time/4th dim)
    let d1 = shape.dims[1]; // Z
    let d2 = shape.dims[2]; // Y
    let d3 = shape.dims[3]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // Extract coordinates: narrow gives [Batch, 1], squeeze_dims removes dim 1 to get [Batch]
    // indices: [Batch, 4] -> (x, y, z, w)
    // narrow consumes self, so clone indices once and narrow each column from that clone.
    let indices_local = indices;
    let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
    let z = indices_local.clone().narrow(1, 2, 1).squeeze_dims(&[1]);
    let w = indices_local.narrow(1, 3, 1).squeeze_dims(&[1]);

    // Compute floor coordinates and weights.
    // Original: 4 clones per axis (floor, weight, x1, clamp) = 16 clones.
    // Now: 1 clone per axis (weight) + 1 clone per axis (x1) = 8 clones.
    let x0 = x.clone().floor();
    let wx = x - x0.clone();
    let y0 = y.clone().floor();
    let wy = y - y0.clone();
    let z0 = z.clone().floor();
    let wz = z - z0.clone();
    let w0 = w.clone().floor();
    let ww = w - w0.clone();

    // Compute x1, y1, z1, w1 — x0/y0/z0/w0 still owned after weight derivation
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;
    let z1 = z0.clone() + 1.0;
    let w1 = w0.clone() + 1.0;

    // Clamp indices — x0/y0/z0/w0 and x1/y1/z1/w1 consumed by clamp+int.
    let x0_i = x0.clone().clamp(0.0, (d3 - 1) as f64).int();
    let y0_i = y0.clone().clamp(0.0, (d2 - 1) as f64).int();
    let z0_i = z0.clone().clamp(0.0, (d1 - 1) as f64).int();
    let w0_i = w0.clone().clamp(0.0, (d0 - 1) as f64).int();

    let x1_i = x1.clamp(0.0, (d3 - 1) as f64).int();
    let y1_i = y1.clamp(0.0, (d2 - 1) as f64).int();
    let z1_i = z1.clamp(0.0, (d1 - 1) as f64).int();
    let w1_i = w1.clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [W, Z, Y, X] layout (d0, d1, d2, d3)
    let stride_w = (d1 * d2 * d3) as i32;
    let stride_z = (d2 * d3) as i32;
    let stride_y = d3 as i32;

    // Pre-flatten data — reshape consumes self, but data is &Tensor so clone once.
    let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

    // Gather all 16 values — gather consumes self on NdArray, so flat_data must be cloned
    // inside gather_4d (handled there).
    let v0000 = gather_4d(
        &flat_data,
        &x0_i,
        &y0_i,
        &z0_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v0001 = gather_4d(
        &flat_data,
        &x0_i,
        &y0_i,
        &z0_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v0010 = gather_4d(
        &flat_data,
        &x0_i,
        &y0_i,
        &z1_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v0011 = gather_4d(
        &flat_data,
        &x0_i,
        &y0_i,
        &z1_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v0100 = gather_4d(
        &flat_data,
        &x0_i,
        &y1_i,
        &z0_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v0101 = gather_4d(
        &flat_data,
        &x0_i,
        &y1_i,
        &z0_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v0110 = gather_4d(
        &flat_data,
        &x0_i,
        &y1_i,
        &z1_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v0111 = gather_4d(
        &flat_data,
        &x0_i,
        &y1_i,
        &z1_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v1000 = gather_4d(
        &flat_data,
        &x1_i,
        &y0_i,
        &z0_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v1001 = gather_4d(
        &flat_data,
        &x1_i,
        &y0_i,
        &z0_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v1010 = gather_4d(
        &flat_data,
        &x1_i,
        &y0_i,
        &z1_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v1011 = gather_4d(
        &flat_data,
        &x1_i,
        &y0_i,
        &z1_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v1100 = gather_4d(
        &flat_data,
        &x1_i,
        &y1_i,
        &z0_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v1101 = gather_4d(
        &flat_data,
        &x1_i,
        &y1_i,
        &z0_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );
    let v1110 = gather_4d(
        &flat_data,
        &x1_i,
        &y1_i,
        &z1_i,
        &w0_i,
        [stride_y, stride_z, stride_w],
    );
    let v1111 = gather_4d(
        &flat_data,
        &x1_i,
        &y1_i,
        &z1_i,
        &w1_i,
        [stride_y, stride_z, stride_w],
    );

    // Pre-compute (1 - weight)
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one.clone() - wy.clone();
    let one_minus_wz = one.clone() - wz.clone();
    let one_minus_ww = one - ww.clone();

    // Quadrilinear interpolation
    // Interpolate along X
    let c000 = v0000 * one_minus_wx.clone() + v1000 * wx.clone();
    let c001 = v0001 * one_minus_wx.clone() + v1001 * wx.clone();
    let c010 = v0010 * one_minus_wx.clone() + v1010 * wx.clone();
    let c011 = v0011 * one_minus_wx.clone() + v1011 * wx.clone();
    let c100 = v0100 * one_minus_wx.clone() + v1100 * wx.clone();
    let c101 = v0101 * one_minus_wx.clone() + v1101 * wx.clone();
    let c110 = v0110 * one_minus_wx.clone() + v1110 * wx.clone();
    let c111 = v0111 * one_minus_wx + v1111 * wx;

    // Interpolate along Y
    let c00 = c000 * one_minus_wy.clone() + c100 * wy.clone();
    let c01 = c001 * one_minus_wy.clone() + c101 * wy.clone();
    let c10 = c010 * one_minus_wy.clone() + c110 * wy.clone();
    let c11 = c011 * one_minus_wy.clone() + c111 * wy.clone();

    // Interpolate along Z
    let c0 = c00 * one_minus_wz.clone() + c10 * wz.clone();
    let c1 = c01 * one_minus_wz.clone() + c11 * wz.clone();

    // Interpolate along W
    let result = c0 * one_minus_ww + c1 * ww;

    let x_mask = in_bounds_mask(x0, (d3 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y0, (d2 - 1) as f64, mode);
    let z_mask = in_bounds_mask(z0, (d1 - 1) as f64, mode);
    let w_mask = in_bounds_mask(w0, (d0 - 1) as f64, mode);

    match (x_mask, y_mask, z_mask, w_mask) {
        (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
        _ => result,
    }
}
