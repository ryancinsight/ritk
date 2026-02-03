use burn::{
    tensor::{Tensor, Int, backend::Backend},
};

/// Trilinear interpolation for 3D tensors.
///
/// # Arguments
/// * `image` - Input image [B, C, D, H, W]
/// * `grid` - Sampling grid [B, 3, D, H, W] in voxel coordinates (z, y, x)
///
/// # Returns
/// * Interpolated image [B, C, D, H, W]
pub fn trilinear_interpolation<B: Backend>(image: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
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
    
    // Helper to gather values: I(idx)
    let gather_val = |idx: Tensor<B, 5, Int>| -> Tensor<B, 5> {
            // Flatten indices: [B, 1, D*H*W]
            let flat_idx_view = idx.reshape([b, 1, d * h * w]);
            
            // Expand indices to match channels: [B, C, D*H*W]
            // Note: gather requires index dims to match input dims or be broadcastable.
            // Explicit repeat is safer for now.
            let flat_idx_expanded = flat_idx_view.repeat(&[1, c, 1]);
            
            let gathered = flat_img.clone().gather(2, flat_idx_expanded);
            
            // Reshape back
            gathered.reshape([b, c, d, h, w])
    };
    
    // Calculate combined indices
    let idx_00 = z0_off.clone() + y0_off.clone();
    let idx_01 = z0_off.clone() + y1_off.clone();
    let idx_10 = z1_off.clone() + y0_off.clone();
    let idx_11 = z1_off.clone() + y1_off.clone();

    let v000 = gather_val(idx_00.clone() + x0_idx.clone());
    let v001 = gather_val(idx_00.clone() + x1_idx.clone());
    let v010 = gather_val(idx_01.clone() + x0_idx.clone());
    let v011 = gather_val(idx_01.clone() + x1_idx.clone());
    let v100 = gather_val(idx_10.clone() + x0_idx.clone());
    let v101 = gather_val(idx_10.clone() + x1_idx.clone());
    let v110 = gather_val(idx_11.clone() + x0_idx.clone());
    let v111 = gather_val(idx_11.clone() + x1_idx.clone());
    
    // Interpolate X first
    let w00 = v000 * wx0.clone() + v001 * wx1.clone();
    let w01 = v010 * wx0.clone() + v011 * wx1.clone();
    let w10 = v100 * wx0.clone() + v101 * wx1.clone();
    let w11 = v110 * wx0.clone() + v111 * wx1.clone();
    
    // Interpolate Y
    let w0 = w00 * wy0.clone() + w01 * wy1.clone();
    let w1 = w10 * wy0.clone() + w11 * wy1.clone();
    
    // Interpolate Z
    w0 * wz0 + w1 * wz1
}
