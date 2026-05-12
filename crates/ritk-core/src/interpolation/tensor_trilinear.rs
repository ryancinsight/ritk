use burn::tensor::{backend::Backend, Int, Tensor};

/// Trilinear interpolation for 3D tensors.
///
/// # Arguments
/// * `image` - Input image [B, C, D, H, W]
/// * `grid` - Sampling grid [B, 3, D, H, W] in voxel coordinates (z, y, x)
///
/// # Returns
/// * Interpolated image [B, C, D, H, W]
pub fn trilinear_interpolation<B: Backend>(
    image: Tensor<B, 5>,
    grid: Tensor<B, 5>,
) -> Tensor<B, 5> {
    let [b, c, d, h, w] = image.dims();

    // Split grid into z, y, x
    let z = grid.clone().slice([0..b, 0..1, 0..d, 0..h, 0..w]);
    let y = grid.clone().slice([0..b, 1..2, 0..d, 0..h, 0..w]);
    let x = grid.slice([0..b, 2..3, 0..d, 0..h, 0..w]);

    // Floor and Ceil
    let z0 = z.clone().floor();
    let z1 = z0.clone().add_scalar(1.0);
    let y0 = y.clone().floor();
    let y1 = y0.clone().add_scalar(1.0);
    let x0 = x.clone().floor();
    let x1 = x0.clone().add_scalar(1.0);

    // Weights
    let wz1 = z.clone().sub(z0.clone());
    let wz0 = wz1.clone().neg().add_scalar(1.0);
    let wy1 = y.clone().sub(y0.clone());
    let wy0 = wy1.clone().neg().add_scalar(1.0);
    let wx1 = x.clone().sub(x0.clone());
    let wx0 = wx1.clone().neg().add_scalar(1.0);

    // Clip coordinates
    let z0_idx = z0.clamp(0.0, (d - 1) as f32).int();
    let z1_idx = z1.clamp(0.0, (d - 1) as f32).int();
    let y0_idx = y0.clamp(0.0, (h - 1) as f32).int();
    let y1_idx = y1.clamp(0.0, (h - 1) as f32).int();
    let x0_idx = x0.clamp(0.0, (w - 1) as f32).int();
    let x1_idx = x1.clamp(0.0, (w - 1) as f32).int();

    // Flatten image once: [B, C, D*H*W]
    let flat_img = image.reshape([b, c, d * h * w]);

    // Pre-calculate strides
    let stride_d = (h * w) as i32;
    let stride_h = w as i32;

    // Pre-calculate offsets
    let z0_off = z0_idx.mul_scalar(stride_d);
    let z1_off = z1_idx.mul_scalar(stride_d);
    let y0_off = y0_idx.mul_scalar(stride_h);
    let y1_off = y1_idx.mul_scalar(stride_h);

    // Calculate combined indices once
    let idx_00 = z0_off.clone() + y0_off.clone();
    let idx_01 = z0_off.clone() + y1_off.clone();
    let idx_10 = z1_off.clone() + y0_off.clone();
    let idx_11 = z1_off.clone() + y1_off.clone();

    let idx_000 = (idx_00.clone() + x0_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_001 = (idx_00 + x1_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_010 = (idx_01.clone() + x0_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_011 = (idx_01 + x1_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_100 = (idx_10.clone() + x0_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_101 = (idx_10 + x1_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_110 = (idx_11.clone() + x0_idx.clone()).reshape([b, 1, d * h * w]);
    let idx_111 = (idx_11 + x1_idx.clone()).reshape([b, 1, d * h * w]);

    let mut out_channels = Vec::with_capacity(c);

    for c_idx in 0..c {
        let channel_img = flat_img.clone().slice([0..b, c_idx..c_idx + 1, 0..d * h * w]);

        let gather_val = |idx: &Tensor<B, 3, Int>| -> Tensor<B, 5> {
            channel_img.clone().gather(2, idx.clone()).reshape([b, 1, d, h, w])
        };

        let v000 = gather_val(&idx_000);
        let v001 = gather_val(&idx_001);
        let v010 = gather_val(&idx_010);
        let v011 = gather_val(&idx_011);
        let v100 = gather_val(&idx_100);
        let v101 = gather_val(&idx_101);
        let v110 = gather_val(&idx_110);
        let v111 = gather_val(&idx_111);

        // Interpolate X first
        let w00 = v000 * wx0.clone() + v001 * wx1.clone();
        let w01 = v010 * wx0.clone() + v011 * wx1.clone();
        let w10 = v100 * wx0.clone() + v101 * wx1.clone();
        let w11 = v110 * wx0.clone() + v111 * wx1.clone();

        // Interpolate Y
        let w0 = w00 * wy0.clone() + w01 * wy1.clone();
        let w1 = w10 * wy0.clone() + w11 * wy1.clone();

        // Interpolate Z
        let c_out = w0 * wz0.clone() + w1 * wz1.clone();
        out_channels.push(c_out);
    }

    Tensor::cat(out_channels, 1)
}
