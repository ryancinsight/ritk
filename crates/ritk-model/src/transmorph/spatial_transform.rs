use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use std::marker::PhantomData;
use crate::interpolation::trilinear_interpolation;

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

        // 1. Split flow into components
        // flow is [B, 3, D, H, W]
        let flow_d = flow.clone().slice([0..b, 0..1, 0..d, 0..h, 0..w]);
        let flow_h = flow.clone().slice([0..b, 1..2, 0..d, 0..h, 0..w]);
        let flow_w = flow.slice([0..b, 2..3, 0..d, 0..h, 0..w]);

        // 2. Create coordinate grids (small, broadcastable)
        // D coords: [1, 1, D, 1, 1]
        let d_range = Tensor::arange(0..d as i64, &device)
            .float()
            .reshape([1, 1, d, 1, 1]);
            
        // H coords: [1, 1, 1, H, 1]
        let h_range = Tensor::arange(0..h as i64, &device)
            .float()
            .reshape([1, 1, 1, h, 1]);
            
        // W coords: [1, 1, 1, 1, W]
        let w_range = Tensor::arange(0..w as i64, &device)
            .float()
            .reshape([1, 1, 1, 1, w]);

        // 3. Add flow to grid (Broadcasting avoids allocating full identity grid)
        let sample_d = flow_d + d_range;
        let sample_h = flow_h + h_range;
        let sample_w = flow_w + w_range;

        // 4. Stack back to [B, 3, D, H, W] for interpolation
        let sampling_grid = Tensor::cat(vec![sample_d, sample_h, sample_w], 1);

        // 5. Interpolate
        trilinear_interpolation(image, sampling_grid)
    }
}
