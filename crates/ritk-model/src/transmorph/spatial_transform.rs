use burn::{
    module::Module,
    tensor::{Tensor, Int, backend::Backend},
};
use std::marker::PhantomData;

#[derive(Module, Debug)]
pub struct SpatialTransformer<B: Backend> {
    phantom: PhantomData<B>,
}

impl<B: Backend> SpatialTransformer<B> {
    pub fn new() -> Self {
        Self { phantom: PhantomData }
    }

    pub fn forward(&self, image: Tensor<B, 5>, flow: Tensor<B, 5>) -> Tensor<B, 5> {
        // image: [B, C, D, H, W]
        // flow: [B, 3, D, H, W]
        
        let [b, _c, d, h, w] = image.dims();
        let device = image.device();

        // 1. Create meshgrid
        let grid = self.meshgrid(b, d, h, w, &device); // [B, 3, D, H, W]

        // 2. Add flow to grid
        let sampling_grid = grid + flow;

        // 3. Interpolate
        self.trilinear_interpolation(image, sampling_grid)
    }

    fn meshgrid(&self, b: usize, d: usize, h: usize, w: usize, device: &B::Device) -> Tensor<B, 5> {
        // Create coordinate grids
        // Range 0..D, 0..H, 0..W
        
        // D coords
        let d_range = Tensor::arange(0..d as i64, device).float();
        let d_grid = d_range.reshape([1, 1, d, 1, 1]).repeat(&[b, 1, 1, h, w]);
        
        // H coords
        let h_range = Tensor::arange(0..h as i64, device).float();
        let h_grid = h_range.reshape([1, 1, 1, h, 1]).repeat(&[b, 1, d, 1, w]);
        
        // W coords
        let w_range = Tensor::arange(0..w as i64, device).float();
        let w_grid = w_range.reshape([1, 1, 1, 1, w]).repeat(&[b, 1, d, h, 1]);
        
        // Stack: [B, 3, D, H, W] -> (d, h, w) order matching flow
        Tensor::cat(vec![d_grid, h_grid, w_grid], 1)
    }

    fn trilinear_interpolation(&self, image: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
        // grid: [B, 3, D, H, W] contains (z, y, x) coordinates
        // image: [B, C, D, H, W]
        
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
        
        // Gather values: I(z0, y0, x0), etc.
        let get_val = |z_idx: Tensor<B, 5, Int>, y_idx: Tensor<B, 5, Int>, x_idx: Tensor<B, 5, Int>| -> Tensor<B, 5> {
             // Calculate flat index for spatial dims: [B, 1, D, H, W]
             let flat_idx = z_idx.mul_scalar((h * w) as i32) + y_idx.mul_scalar(w as i32) + x_idx;
             
             // Flatten image spatial: [B, C, D*H*W]
             let flat_img = image.clone().reshape([b, c, d * h * w]);
             
             // Flatten indices: [B, 1, D*H*W] -> repeat for C -> [B, C, D*H*W]
             let flat_idx_view = flat_idx.reshape([b, 1, d * h * w]);
             let flat_idx_expanded = flat_idx_view.repeat(&[1, c, 1]);
             
             let gathered = flat_img.gather(2, flat_idx_expanded);
             
             // Reshape back
             gathered.reshape([b, c, d, h, w])
        };
        
        let v000 = get_val(z0_idx.clone(), y0_idx.clone(), x0_idx.clone());
        let v001 = get_val(z0_idx.clone(), y0_idx.clone(), x1_idx.clone());
        let v010 = get_val(z0_idx.clone(), y1_idx.clone(), x0_idx.clone());
        let v011 = get_val(z0_idx.clone(), y1_idx.clone(), x1_idx.clone());
        let v100 = get_val(z1_idx.clone(), y0_idx.clone(), x0_idx.clone());
        let v101 = get_val(z1_idx.clone(), y0_idx.clone(), x1_idx.clone());
        let v110 = get_val(z1_idx.clone(), y1_idx.clone(), x0_idx.clone());
        let v111 = get_val(z1_idx.clone(), y1_idx.clone(), x1_idx.clone());
        
        // Interpolate X first
        let w00 = v000 * wx0.clone() + v001 * wx1.clone();
        let w01 = v010 * wx0.clone() + v011 * wx1.clone();
        let w10 = v100 * wx0.clone() + v101 * wx1.clone();
        let w11 = v110 * wx0.clone() + v111 * wx1.clone();
        
        // Interpolate Y
        let w0 = w00 * wy0.clone() + w01 * wy1.clone();
        let w1 = w10 * wy0.clone() + w11 * wy1.clone();
        
        // Interpolate Z
        let result = w0 * wz0 + w1 * wz1;
        
        result
    }
}
