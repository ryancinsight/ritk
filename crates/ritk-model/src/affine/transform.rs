use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use std::marker::PhantomData;
use crate::interpolation::trilinear_interpolation;

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
        trilinear_interpolation(image, pixel_grid)
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
}
