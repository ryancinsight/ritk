//! Spatial Sampling - Grid Sampling and Flow Composition
//!
//! Provides 3D grid sampling with trilinear interpolation and flow field
//! composition for deformable image registration.

use burn::prelude::*;

/// Padding mode for grid sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPaddingMode {
    /// Use zeros for out-of-bounds samples
    Zero,
    /// Use border values for out-of-bounds samples
    Border,
    /// Reflect coordinates at the border
    Reflection,
}

impl Default for GridPaddingMode {
    fn default() -> Self {
        GridPaddingMode::Border
    }
}

/// Interpolation mode for grid sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Nearest neighbor sampling
    Nearest,
    /// Bilinear/trilinear interpolation
    Linear,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        InterpolationMode::Linear
    }
}

/// Configuration for grid sampling operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSamplerConfig {
    /// Padding mode for out-of-bounds coordinates
    pub padding_mode: GridPaddingMode,
    /// Interpolation mode
    pub interpolation: InterpolationMode,
    /// Align corners (true: -1/1 align to corners, false: to pixel centers)
    pub align_corners: bool,
}

impl Default for GridSamplerConfig {
    fn default() -> Self {
        Self {
            padding_mode: GridPaddingMode::Border,
            interpolation: InterpolationMode::Linear,
            align_corners: true,
        }
    }
}

/// 3D Grid Sampler with trilinear interpolation
///
/// Samples input tensors at specified grid coordinates.
pub struct GridSampler<B: Backend> {
    config: GridSamplerConfig,
    _device: B::Device,
}

impl<B: Backend> GridSampler<B> {
    /// Create new grid sampler with default configuration
    pub fn new(device: B::Device) -> Self {
        Self {
            config: GridSamplerConfig::default(),
            _device: device,
        }
    }

    /// Create grid sampler with custom configuration
    pub fn with_config(config: GridSamplerConfig, device: B::Device) -> Self {
        Self { config, _device: device }
    }

    /// Sample input tensor at grid coordinates
    pub fn sample(&self, input: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
        match self.config.interpolation {
            InterpolationMode::Linear => self.sample_trilinear(input, grid),
            InterpolationMode::Nearest => self.sample_nearest(input, grid),
        }
    }

    /// Trilinear interpolation sampling
    fn sample_trilinear(&self, input: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, _channels, d_in, h_in, w_in] = input.dims();
        let [_, d_out, h_out, w_out, _] = grid.dims();

        // Denormalize grid coordinates to [0, size-1] range
        let (ix, iy, iz) = self.denormalize_coordinates(grid, w_in, h_in, d_in);

        // Compute valid mask for Zero padding
        let mask: Option<Tensor<B, 4>> = if self.config.padding_mode == GridPaddingMode::Zero {
            let x_valid = ix.clone().greater_equal_elem(0.0).int() * ix.clone().lower_equal_elem((w_in - 1) as f32).int();
            let y_valid = iy.clone().greater_equal_elem(0.0).int() * iy.clone().lower_equal_elem((h_in - 1) as f32).int();
            let z_valid = iz.clone().greater_equal_elem(0.0).int() * iz.clone().lower_equal_elem((d_in - 1) as f32).int();
            Some((x_valid * y_valid * z_valid).float())
        } else {
            None
        };

        // Clamp coordinates for safe gathering (handles Border padding and ensures Zero padding doesn't panic)
        let ix = ix.clamp(0.0, (w_in - 1) as f32);
        let iy = iy.clamp(0.0, (h_in - 1) as f32);
        let iz = iz.clamp(0.0, (d_in - 1) as f32);

        // Get corner indices for interpolation
        let ix0 = ix.clone().floor();
        let iy0 = iy.clone().floor();
        let iz0 = iz.clone().floor();

        let ix1 = (ix0.clone() + 1.0).clamp(0.0, (w_in - 1) as f32);
        let iy1 = (iy0.clone() + 1.0).clamp(0.0, (h_in - 1) as f32);
        let iz1 = (iz0.clone() + 1.0).clamp(0.0, (d_in - 1) as f32);

        // Calculate weights (clone ix, iy, iz to avoid move)
        let wx1 = ix.clone() - ix0.clone();
        let wy1 = iy.clone() - iy0.clone();
        let wz1 = iz.clone() - iz0.clone();
        let wx0 = wx1.clone().neg().add_scalar(1.0);
        let wy0 = wy1.clone().neg().add_scalar(1.0);
        let wz0 = wz1.clone().neg().add_scalar(1.0);

        // Cast to int for indexing
        let ix0_i = ix0.int();
        let iy0_i = iy0.int();
        let iz0_i = iz0.int();
        let ix1_i = ix1.int();
        let iy1_i = iy1.int();
        let iz1_i = iz1.int();

        // Gather values at 8 corners
        let v000 = self.gather(&input, &iz0_i, &iy0_i, &ix0_i, d_out, h_out, w_out);
        let v100 = self.gather(&input, &iz0_i, &iy0_i, &ix1_i, d_out, h_out, w_out);
        let v010 = self.gather(&input, &iz0_i, &iy1_i, &ix0_i, d_out, h_out, w_out);
        let v110 = self.gather(&input, &iz0_i, &iy1_i, &ix1_i, d_out, h_out, w_out);
        let v001 = self.gather(&input, &iz1_i, &iy0_i, &ix0_i, d_out, h_out, w_out);
        let v101 = self.gather(&input, &iz1_i, &iy0_i, &ix1_i, d_out, h_out, w_out);
        let v011 = self.gather(&input, &iz1_i, &iy1_i, &ix0_i, d_out, h_out, w_out);
        let v111 = self.gather(&input, &iz1_i, &iy1_i, &ix1_i, d_out, h_out, w_out);

        // Reshape weights for broadcasting
        let wx0 = wx0.reshape([batch, 1, d_out, h_out, w_out]);
        let wx1 = wx1.reshape([batch, 1, d_out, h_out, w_out]);
        let wy0 = wy0.reshape([batch, 1, d_out, h_out, w_out]);
        let wy1 = wy1.reshape([batch, 1, d_out, h_out, w_out]);
        let wz0 = wz0.reshape([batch, 1, d_out, h_out, w_out]);
        let wz1 = wz1.reshape([batch, 1, d_out, h_out, w_out]);

        // Trilinear interpolation
        let c00 = v000 * wx0.clone() + v100 * wx1.clone();
        let c10 = v010 * wx0.clone() + v110 * wx1.clone();
        let c01 = v001 * wx0.clone() + v101 * wx1.clone();
        let c11 = v011 * wx0.clone() + v111 * wx1;

        let c0 = c00 * wy0.clone() + c10 * wy1.clone();
        let c1 = c01 * wy0.clone() + c11 * wy1;

        let result = c0 * wz0 + c1 * wz1;

        // Apply zero padding mask if needed
        if let Some(mask) = mask {
            result * mask.unsqueeze::<5>()
        } else {
            result
        }
    }

    /// Nearest neighbor sampling
    fn sample_nearest(&self, input: Tensor<B, 5>, grid: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch, _channels, d_in, h_in, w_in] = input.dims();
        let [_, d_out, h_out, w_out, _] = grid.dims();

        // Denormalize coordinates
        let (ix, iy, iz) = self.denormalize_coordinates(grid, w_in, h_in, d_in);

        // Round to nearest integer
        let ix_n = ix.round().clamp(0.0, (w_in - 1) as f32).int();
        let iy_n = iy.round().clamp(0.0, (h_in - 1) as f32).int();
        let iz_n = iz.round().clamp(0.0, (d_in - 1) as f32).int();

        // Gather values
        self.gather(&input, &iz_n, &iy_n, &ix_n, d_out, h_out, w_out)
    }

    /// Denormalize coordinates from [-1, 1] to [0, size-1]
    fn denormalize_coordinates(
        &self,
        grid: Tensor<B, 5>,
        w_in: usize,
        h_in: usize,
        d_in: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, d_out, h_out, w_out, _] = grid.dims();

        let x = grid.clone()
            .slice([0..batch, 0..d_out, 0..h_out, 0..w_out, 0..1])
            .squeeze(4);
        let y = grid.clone()
            .slice([0..batch, 0..d_out, 0..h_out, 0..w_out, 1..2])
            .squeeze(4);
        let z = grid.clone()
            .slice([0..batch, 0..d_out, 0..h_out, 0..w_out, 2..3])
            .squeeze(4);

        if self.config.align_corners {
            let ix = (x + 1.0) * ((w_in - 1) as f32) / 2.0;
            let iy = (y + 1.0) * ((h_in - 1) as f32) / 2.0;
            let iz = (z + 1.0) * ((d_in - 1) as f32) / 2.0;
            (ix, iy, iz)
        } else {
            let ix = ((x + 1.0) * (w_in as f32) - 1.0) / 2.0;
            let iy = ((y + 1.0) * (h_in as f32) - 1.0) / 2.0;
            let iz = ((z + 1.0) * (d_in as f32) - 1.0) / 2.0;
            (ix, iy, iz)
        }
    }

    /// Gather values from input at specified coordinates
    fn gather(
        &self,
        input: &Tensor<B, 5>,
        iz: &Tensor<B, 4, Int>,
        iy: &Tensor<B, 4, Int>,
        ix: &Tensor<B, 4, Int>,
        d_out: usize,
        h_out: usize,
        w_out: usize,
    ) -> Tensor<B, 5> {
        let [batch, channels, d_in, h_in, w_in] = input.dims();

        // Flatten input
        let input_flat = input.clone().reshape([batch, channels, d_in * h_in * w_in]);

        // Compute flat indices
        let idx = iz.clone().mul_scalar((h_in * w_in) as i32)
            + iy.clone().mul_scalar(w_in as i32)
            + ix.clone();

        // Reshape for gathering
        let idx_flat = idx.reshape([batch, 1, d_out * h_out * w_out]);
        let idx_rep = idx_flat.repeat(&[1, channels, 1]);

        // Gather and reshape back
        let gathered = input_flat.gather(2, idx_rep);
        gathered.reshape([batch, channels, d_out, h_out, w_out])
    }
}

/// Flow field composer for displacement field operations
pub struct FlowComposer<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> FlowComposer<B> {
    /// Create new flow composer
    pub fn new(_device: B::Device) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compose two displacement fields
    pub fn compose(&self, flow1: &Tensor<B, 5>, flow2: &Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, _, d, h, w] = flow1.dims();
        let device = flow1.device();

        // Create base grid
        let base_grid = self.create_grid([d, h, w], &device);
        let base_grid = base_grid.repeat(&[b, 1, 1, 1, 1]);

        // Convert flow2 to sampling grid
        let sample_grid = self.flow_to_grid(flow2, &base_grid, w, h, d);

        // Sample flow1
        let sampler = GridSampler::new(device);
        let sampled_flow1 = sampler.sample(flow1.clone(), sample_grid);

        // Result
        sampled_flow1 + flow2.clone()
    }

    /// Warp image using displacement field
    pub fn warp(&self, image: &Tensor<B, 5>, displacement: &Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, _, d, h, w] = displacement.dims();
        let device = displacement.device();

        let base_grid = self.create_grid([d, h, w], &device);
        let base_grid = base_grid.repeat(&[b, 1, 1, 1, 1]);

        let sample_grid = self.flow_to_grid(displacement, &base_grid, w, h, d);

        let sampler = GridSampler::new(device);
        sampler.sample(image.clone(), sample_grid)
    }

    /// Create normalized grid
    fn create_grid(&self, shape: [usize; 3], device: &B::Device) -> Tensor<B, 5> {
        let [d, h, w] = shape;

        let x = Tensor::arange(0..w as i64, device).float();
        let y = Tensor::arange(0..h as i64, device).float();
        let z = Tensor::arange(0..d as i64, device).float();

        let x_norm = x.mul_scalar(2.0 / (w as f32 - 1.0)).sub_scalar(1.0);
        let y_norm = y.mul_scalar(2.0 / (h as f32 - 1.0)).sub_scalar(1.0);
        let z_norm = z.mul_scalar(2.0 / (d as f32 - 1.0)).sub_scalar(1.0);

        let x_grid = x_norm.reshape([1, 1, w, 1]).repeat(&[d, h, 1, 1]);
        let y_grid = y_norm.reshape([1, h, 1, 1]).repeat(&[d, 1, w, 1]);
        let z_grid = z_norm.reshape([d, 1, 1, 1]).repeat(&[1, h, w, 1]);

        let grid = Tensor::cat(vec![x_grid, y_grid, z_grid], 3);
        grid.unsqueeze::<5>()
    }

    /// Convert flow to sampling grid
    fn flow_to_grid(
        &self,
        flow: &Tensor<B, 5>,
        base_grid: &Tensor<B, 5>,
        w: usize,
        h: usize,
        d: usize,
    ) -> Tensor<B, 5> {
        let [b, _, d_out, h_out, w_out] = flow.dims();

        let flow_perm = flow.clone().permute([0, 2, 3, 4, 1]);

        let flow_x = flow_perm.clone().slice([0..b, 0..d_out, 0..h_out, 0..w_out, 0..1]);
        let flow_y = flow_perm.clone().slice([0..b, 0..d_out, 0..h_out, 0..w_out, 1..2]);
        let flow_z = flow_perm.clone().slice([0..b, 0..d_out, 0..h_out, 0..w_out, 2..3]);

        let flow_x_norm = flow_x.mul_scalar(2.0 / (w as f32 - 1.0));
        let flow_y_norm = flow_y.mul_scalar(2.0 / (h as f32 - 1.0));
        let flow_z_norm = flow_z.mul_scalar(2.0 / (d as f32 - 1.0));

        let flow_norm = Tensor::cat(vec![flow_x_norm, flow_y_norm, flow_z_norm], 4);

        base_grid.clone() + flow_norm
    }
}
