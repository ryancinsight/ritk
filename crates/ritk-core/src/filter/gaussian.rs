use crate::image::Image;
use crate::spatial::Spacing;
use burn::tensor::backend::Backend;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Shape, Tensor};

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

        Image::new(data, *image.origin(), *image.spacing(), *image.direction())
    }

    /// Apply the filter to a tensor directly.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `spacing` - Physical spacing of the data (used to determine kernel size)
    pub fn apply_tensor<const D: usize>(
        &self,
        input: Tensor<B, D>,
        spacing: &Spacing<D>,
    ) -> Tensor<B, D> {
        let mut data = input;
        let device = data.device();

        // Apply 1D convolution along each dimension
        for d in 0..D {
            let sigma = if d < self.sigmas.len() {
                self.sigmas[d]
            } else {
                self.sigmas[0]
            };
            let spacing_val = spacing[d];

            // Skip if sigma is close to zero
            if sigma <= 1e-6 {
                continue;
            }

            // Calculate kernel
            let pixel_sigma = sigma / spacing_val;
            let radius = (3.0 * pixel_sigma).ceil() as usize;
            let mut width = (2 * radius + 1).min(self.max_kernel_width);
            if width.is_multiple_of(2) {
                width -= 1;
            }
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
        dim: usize,
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
        permute_indices[D - 1] = dim as isize;

        let input_permuted = input.clone().permute(permute_indices);

        // 2. Flatten other dimensions into batch
        let last_dim_size = dims[dim];
        let mut batch_size = 1;
        for (i, &d) in dims.iter().enumerate() {
            if i != dim {
                batch_size *= d;
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

        // Chunking for large batches to avoid WGPU dispatch limits
        const CHUNK_SIZE: usize = 32768;
        let output_reshaped = if batch_size <= CHUNK_SIZE {
            burn::tensor::module::conv1d(
                input_reshaped,
                kernel_reshaped,
                None, // bias
                options,
            )
        } else {
            let num_chunks = batch_size.div_ceil(CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, batch_size);

                let chunk_range = start..end;
                let chunk_input = input_reshaped.clone().slice([chunk_range]);
                let chunk_output = burn::tensor::module::conv1d(
                    chunk_input,
                    kernel_reshaped.clone(),
                    None,
                    options.clone(),
                );
                chunks.push(chunk_output);
            }
            Tensor::cat(chunks, 0)
        };

        // 3. Reshape back and inverse permute
        // Output shape matches input_permuted shape since we used padding
        // But we need to be careful if padding logic in conv1d changes size slightly (e.g. even kernel)
        // With stride 1 and padding = floor(k/2) and odd kernel, size should be preserved.

        // Reshape back to permuted shape
        let mut permuted_shape = [0; D];
        let mut p_idx = 0;
        for (i, &d) in dims.iter().enumerate() {
            if i != dim {
                permuted_shape[p_idx] = d;
                p_idx += 1;
            }
        }
        permuted_shape[D - 1] = last_dim_size; // Assuming size preserved

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ops::extract_vec_infallible;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        let (v, _) = extract_vec_infallible(img);
        v
    }

    // ── generate_kernel ───────────────────────────────────────────────────────

    /// The Gaussian kernel normalizes correctly: interior voxels of a constant image
    /// are preserved under convolution.
    ///
    /// # Derivation
    /// The kernel weights sum to 1.0 by construction. For interior voxels where the
    /// full kernel fits within the image, the convolution output equals the constant:
    ///   out(x) = Σ w_k * C = C * Σ w_k = C * 1.0 = C.
    ///
    /// Boundary voxels receive partial kernel support under zero-padding and may
    /// deviate. We test only the center voxel of a large image (size=15) where the
    /// radius-3 kernel (sigma=1, radius=ceil(3*1)=3) fits fully.
    #[test]
    fn gaussian_kernel_sums_to_one() {
        let size = 15usize;
        let filter = GaussianFilter::<B>::new(vec![1.0]);
        let img = make_image(vec![3.0_f32; size * size * size], [size, size, size]);
        let out = filter.apply(&img);
        let vals = voxels(&out);
        // Center voxel index: (size/2) * size * size + (size/2) * size + (size/2)
        let cx = size / 2;
        let center_idx = cx * size * size + cx * size + cx;
        let v = vals[center_idx];
        assert!(
            (v - 3.0).abs() < 5e-3,
            "center voxel of constant image under Gaussian must stay ≈ 3.0; got {v}"
        );
    }

    /// Zero sigma must skip smoothing (output identical to input).
    ///
    /// # Derivation
    /// The implementation has `if sigma <= 1e-6 { continue; }` which bypasses the
    /// convolution entirely. The output tensor must be identical to the input.
    #[test]
    fn zero_sigma_skips_smoothing() {
        let filter = GaussianFilter::<B>::new(vec![0.0]);
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = make_image(data.clone(), [2, 2, 2]);
        let out = filter.apply(&img);
        let got = voxels(&out);
        for (i, (&a, &b)) in got.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "zero sigma must not change voxel {i}: expected {b}, got {a}"
            );
        }
    }

    /// Spatial metadata (origin, spacing, direction) is preserved.
    #[test]
    fn gaussian_preserves_metadata() {
        let filter = GaussianFilter::<B>::new(vec![0.5]);
        let sp = Spacing::new([2.0, 3.0, 4.0]);
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 2 * 2 * 2], Shape::new([2usize, 2, 2])),
            &device,
        );
        let img = Image::new(
            t,
            Point::new([10.0, 20.0, 30.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = filter.apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
        assert_eq!(out.origin(), img.origin(), "origin must be preserved");
    }

    /// Output shape must equal input shape after smoothing (padding=kernel_size/2).
    #[test]
    fn gaussian_preserves_shape() {
        let filter = GaussianFilter::<B>::new(vec![1.5]);
        let img = make_image(vec![1.0_f32; 5 * 6 * 7], [5, 6, 7]);
        let out = filter.apply(&img);
        assert_eq!(
            out.shape(),
            img.shape(),
            "shape must be preserved after Gaussian"
        );
    }
}
