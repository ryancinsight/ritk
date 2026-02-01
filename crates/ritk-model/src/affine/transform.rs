use burn::{
    module::Module,
    tensor::{Tensor, Int, backend::Backend},
};
use std::marker::PhantomData;

#[derive(Module, Debug)]
pub struct AffineTransform<B: Backend> {
    phantom: PhantomData<B>,
}

impl<B: Backend> AffineTransform<B> {
    pub fn new() -> Self {
        Self { phantom: PhantomData }
    }

    /// Forward pass for Affine Transform
    /// 
    /// # Arguments
    /// 
    /// * `image` - Input image [B, C, D, H, W]
    /// * `theta` - Affine matrix [B, 3, 4] or [B, 12] (flattened)
    /// 
    /// # Returns
    /// 
    /// * Transformed image [B, C, D, H, W]
    pub fn forward(&self, image: Tensor<B, 5>, theta: Tensor<B, 2>) -> Tensor<B, 5> {
        let [b, _c, d, h, w] = image.dims();
        let device = image.device();

        // Ensure theta is [B, 3, 4]
        let theta = if theta.shape().dims[1] == 12 {
            theta.reshape([b, 3, 4])
        } else {
            theta.reshape([b, 3, 4]) // Assume it fits or let it panic
        };

        // 1. Create normalized meshgrid [-1, 1]
        // [B, 4, D*H*W]
        let grid = self.normalized_meshgrid(b, d, h, w, &device); 

        // 2. Apply Affine Transform
        // theta: [B, 3, 4]
        // grid: [B, 4, N]
        // result: [B, 3, N]
        let warped_grid = theta.matmul(grid);

        // 3. Reshape to [B, 3, D, H, W]
        let warped_grid = warped_grid.reshape([b, 3, d, h, w]);

        // 4. Denormalize to pixel coordinates for interpolation
        let pixel_grid = self.denormalize_grid(warped_grid, d, h, w);

        // 5. Interpolate
        self.trilinear_interpolation(image, pixel_grid)
    }

    fn normalized_meshgrid(&self, b: usize, d: usize, h: usize, w: usize, device: &B::Device) -> Tensor<B, 3> {
        // Generate grid in range [-1, 1]
        // D coords
        let d_range = Tensor::arange(0..d as i64, device).float() * 2.0 / ((d - 1) as f32) - 1.0;
        let d_grid = d_range.reshape([1, 1, d, 1, 1]).repeat(&[b, 1, 1, h, w]);
        
        // H coords
        let h_range = Tensor::arange(0..h as i64, device).float() * 2.0 / ((h - 1) as f32) - 1.0;
        let h_grid = h_range.reshape([1, 1, 1, h, 1]).repeat(&[b, 1, d, 1, w]);
        
        // W coords
        let w_range = Tensor::arange(0..w as i64, device).float() * 2.0 / ((w - 1) as f32) - 1.0;
        let w_grid = w_range.reshape([1, 1, 1, 1, w]).repeat(&[b, 1, d, h, 1]);

        // Ones for homogeneous coords
        let ones = Tensor::ones([b, 1, d, h, w], device);

        // Stack: [B, 4, D, H, W] -> (d, h, w, 1) order matching theta logic
        // But we flatten spatial dims to N = D*H*W
        let grid = Tensor::cat(vec![d_grid, h_grid, w_grid, ones], 1);
        
        grid.reshape([b, 4, d * h * w])
    }

    fn denormalize_grid(&self, grid: Tensor<B, 5>, d: usize, h: usize, w: usize) -> Tensor<B, 5> {
        // grid is [B, 3, D, H, W] in [-1, 1]
        // convert to [0, D-1], [0, H-1], [0, W-1]
        
        let [b, _, _, _, _] = grid.dims();
        
        let z = grid.clone().slice([0..b, 0..1, 0..d, 0..h, 0..w]);
        let y = grid.clone().slice([0..b, 1..2, 0..d, 0..h, 0..w]);
        let x = grid.slice([0..b, 2..3, 0..d, 0..h, 0..w]);

        let z = (z + 1.0) * ((d - 1) as f32 / 2.0);
        let y = (y + 1.0) * ((h - 1) as f32 / 2.0);
        let x = (x + 1.0) * ((w - 1) as f32 / 2.0);

        Tensor::cat(vec![z, y, x], 1)
    }

    fn trilinear_interpolation(&self, image: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
        // Same as SpatialTransformer
        let [b, c, d, h, w] = image.dims();
        
        let z = grid.clone().slice([0..b, 0..1, 0..d, 0..h, 0..w]);
        let y = grid.clone().slice([0..b, 1..2, 0..d, 0..h, 0..w]);
        let x = grid.slice([0..b, 2..3, 0..d, 0..h, 0..w]);
        
        let z0 = z.clone().floor();
        let z1 = z0.clone().add_scalar(1.0);
        let y0 = y.clone().floor();
        let y1 = y0.clone().add_scalar(1.0);
        let x0 = x.clone().floor();
        let x1 = x0.clone().add_scalar(1.0);
        
        let wz1 = z.clone().sub(z0.clone());
        let wz0 = wz1.clone().neg().add_scalar(1.0);
        let wy1 = y.clone().sub(y0.clone());
        let wy0 = wy1.clone().neg().add_scalar(1.0);
        let wx1 = x.clone().sub(x0.clone());
        let wx0 = wx1.clone().neg().add_scalar(1.0);
        
        let z0_idx = z0.clamp(0.0, (d - 1) as f32).int();
        let z1_idx = z1.clamp(0.0, (d - 1) as f32).int();
        let y0_idx = y0.clamp(0.0, (h - 1) as f32).int();
        let y1_idx = y1.clamp(0.0, (h - 1) as f32).int();
        let x0_idx = x0.clamp(0.0, (w - 1) as f32).int();
        let x1_idx = x1.clamp(0.0, (w - 1) as f32).int();
        
        let get_val = |z_idx: Tensor<B, 5, Int>, y_idx: Tensor<B, 5, Int>, x_idx: Tensor<B, 5, Int>| -> Tensor<B, 5> {
             let flat_idx = z_idx.mul_scalar((h * w) as i32) + y_idx.mul_scalar(w as i32) + x_idx;
             let flat_img = image.clone().reshape([b, c, d * h * w]);
             let flat_idx_view = flat_idx.reshape([b, 1, d * h * w]);
             let flat_idx_expanded = flat_idx_view.repeat(&[1, c, 1]);
             flat_img.gather(2, flat_idx_expanded).reshape([b, c, d, h, w])
        };

        let v000 = get_val(z0_idx.clone(), y0_idx.clone(), x0_idx.clone());
        let v001 = get_val(z0_idx.clone(), y0_idx.clone(), x1_idx.clone());
        let v010 = get_val(z0_idx.clone(), y1_idx.clone(), x0_idx.clone());
        let v011 = get_val(z0_idx.clone(), y1_idx.clone(), x1_idx.clone());
        let v100 = get_val(z1_idx.clone(), y0_idx.clone(), x0_idx.clone());
        let v101 = get_val(z1_idx.clone(), y0_idx.clone(), x1_idx.clone());
        let v110 = get_val(z1_idx.clone(), y1_idx.clone(), x0_idx.clone());
        let v111 = get_val(z1_idx.clone(), y1_idx.clone(), x1_idx.clone());

        let w000 = wz0.clone() * wy0.clone() * wx0.clone();
        let w001 = wz0.clone() * wy0.clone() * wx1.clone();
        let w010 = wz0.clone() * wy1.clone() * wx0.clone();
        let w011 = wz0.clone() * wy1.clone() * wx1.clone();
        let w100 = wz1.clone() * wy0.clone() * wx0.clone();
        let w101 = wz1.clone() * wy0.clone() * wx1.clone();
        let w110 = wz1.clone() * wy1.clone() * wx0.clone();
        let w111 = wz1.clone() * wy1.clone() * wx1.clone();

        v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 +
        v100 * w100 + v101 * w101 + v110 * w110 + v111 * w111
    }
}
