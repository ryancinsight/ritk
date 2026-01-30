use burn::tensor::{Tensor, Shape};
use burn::tensor::backend::Backend;
use burn::tensor::ops::ConvOptions;
use crate::image::Image;
use crate::spatial::Spacing;

/// Gaussian smoothing filter.
///
/// Applies a Gaussian smoothing filter to an image using separable 1D convolutions.
/// This implementation respects the physical spacing of the image.
pub struct GaussianFilter<B: Backend> {
    sigmas: Vec<f64>,
    max_kernel_width: usize,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> GaussianFilter<B> {
    /// Create a new Gaussian filter with the given standard deviation (in physical units).
    ///
    /// # Arguments
    /// * `sigmas` - Standard deviation for each dimension in physical units (mm).
    pub fn new(sigmas: Vec<f64>) -> Self {
        Self {
            sigmas,
            max_kernel_width: 32, // Default max kernel width to prevent excessive computation
            _b: std::marker::PhantomData,
        }
    }

    /// Set the maximum kernel width (radius * 2 + 1).
    pub fn with_max_kernel_width(mut self, width: usize) -> Self {
        self.max_kernel_width = width;
        self
    }

    /// Apply the filter to an image.
    pub fn apply<const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let data = self.apply_tensor(image.data().clone(), image.spacing());
        
        Image::new(
            data,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        )
    }

    /// Apply the filter to a tensor directly.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `spacing` - Physical spacing of the data (used to determine kernel size)
    pub fn apply_tensor<const D: usize>(&self, input: Tensor<B, D>, spacing: &Spacing<D>) -> Tensor<B, D> {
        let mut data = input;
        let device = data.device();

        // Apply 1D convolution along each dimension
        for d in 0..D {
            let sigma = if d < self.sigmas.len() { self.sigmas[d] } else { self.sigmas[0] };
            let spacing_val = spacing[d];
            
            // Skip if sigma is close to zero
            if sigma <= 1e-6 {
                continue;
            }

            // Calculate kernel
            let pixel_sigma = sigma / spacing_val;
            let radius = (3.0 * pixel_sigma).ceil() as usize;
            let width = (2 * radius + 1).min(self.max_kernel_width);
            let actual_radius = (width - 1) / 2;
            
            let kernel = self.generate_kernel(pixel_sigma, actual_radius);
            let kernel_tensor = Tensor::<B, 1>::from_floats(kernel.as_slice(), &device);
            
            // Convolve along dimension d
            data = self.convolve_1d::<D>(data, kernel_tensor, d);
        }
        data
    }

    fn generate_kernel(&self, sigma: f64, radius: usize) -> Vec<f32> {
        let mut kernel = Vec::with_capacity(2 * radius + 1);
        let mut sum = 0.0;
        let sigma2 = sigma * sigma;
        let two_sigma2 = 2.0 * sigma2;
        // let _factor = 1.0 / (2.0 * std::f64::consts::PI * sigma2).sqrt(); // Normalization handles this

        for i in 0..=(2 * radius) {
            let x = (i as f64) - (radius as f64);
            let val = (-x * x / two_sigma2).exp(); // Unnormalized Gaussian
            kernel.push(val as f32);
            sum += val;
        }

        // Normalize
        for val in &mut kernel {
            *val /= sum as f32;
        }

        kernel
    }

    fn convolve_1d<const D: usize>(
        &self, 
        input: Tensor<B, D>, 
        kernel: Tensor<B, 1>, 
        dim: usize
    ) -> Tensor<B, D> {
        let shape = input.shape();
        let dims: [usize; D] = shape.dims();
        let _device = input.device();
        
        // 1. Permute target dimension to the last
        let mut permute_indices = [0isize; D];
        let mut idx = 0;
        for i in 0..D {
            if i != dim {
                permute_indices[idx] = i as isize;
                idx += 1;
            }
        }
        permute_indices[D-1] = dim as isize;
        
        let input_permuted = input.clone().permute(permute_indices);
        
        // 2. Flatten other dimensions into batch
        let last_dim_size = dims[dim];
        let mut batch_size = 1;
        for i in 0..D {
            if i != dim {
                batch_size *= dims[i];
            }
        }
        
        // Reshape to [Batch, 1, Length] for conv1d
        // Input: [Batch, Channels=1, Length]
        let input_reshaped = input_permuted.reshape([batch_size, 1, last_dim_size]);
        
        // Kernel: [OutChannels=1, InChannels=1, KernelSize]
        let kernel_size = kernel.dims()[0];
        let kernel_reshaped = kernel.reshape([1, 1, kernel_size]);
        
        // Padding to maintain size
        let padding = kernel_size / 2;
        
        // Perform convolution
        let options = ConvOptions::new([1], [padding], [1], 1);
        let output_reshaped = burn::tensor::module::conv1d(
            input_reshaped,
            kernel_reshaped,
            None, // bias
            options,
        );
        
        // 3. Reshape back and inverse permute
        // Output shape matches input_permuted shape since we used padding
        // But we need to be careful if padding logic in conv1d changes size slightly (e.g. even kernel)
        // With stride 1 and padding = floor(k/2) and odd kernel, size should be preserved.
        
        // Reshape back to permuted shape
        let mut permuted_shape = [0; D];
        let mut p_idx = 0;
        for i in 0..D {
            if i != dim {
                permuted_shape[p_idx] = dims[i];
                p_idx += 1;
            }
        }
        permuted_shape[D-1] = last_dim_size; // Assuming size preserved
        
        let output_permuted = output_reshaped.reshape(Shape::new(permuted_shape));
        
        // Inverse permutation
        // We need to map back.
        // Current dims are [d_others..., d_target]
        // We want [d_0, d_1, ... d_D-1]
        
        // Construct inverse indices
        let mut inv_permute_indices = [0isize; D];
        for (new_pos, &old_pos) in permute_indices.iter().enumerate() {
            inv_permute_indices[old_pos as usize] = new_pos as isize;
        }
        
        output_permuted.permute(inv_permute_indices)
    }
}
