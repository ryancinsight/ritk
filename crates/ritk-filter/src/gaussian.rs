use crate::edge::GaussianSigma;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Gaussian smoothing filter.
///
/// Applies a Gaussian smoothing filter to an image using separable 1D convolutions.
/// This implementation respects the physical spacing of the image.
pub struct GaussianFilter<B: Backend> {
    sigmas: Vec<GaussianSigma>,
    max_kernel_width: usize,
    _b: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> Clone for GaussianFilter<B> {
    fn clone(&self) -> Self {
        Self {
            sigmas: self.sigmas.clone(),
            max_kernel_width: self.max_kernel_width,
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> GaussianFilter<B> {
    /// Create a new Gaussian filter with the given standard deviation (in physical units).
    ///
    /// # Arguments
    /// * `sigmas` - Standard deviation for each dimension in physical units (mm).
    pub fn new(sigmas: Vec<GaussianSigma>) -> Self {
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
        let (values, dims) = extract_vec_infallible(image);
        rebuild(
            self.apply_values(values, dims, image.spacing().to_array()),
            dims,
            image,
        )
    }

    /// Apply zero-padded separable Gaussian smoothing to a Coeus-native volume.
    ///
    /// The native path uses the same kernel-width, degenerate-axis, and
    /// zero-padding contract as the legacy convolution path.
    pub fn apply_native<C>(
        &self,
        image: &ritk_image::native::Image<f32, C, 3>,
        backend: &C,
    ) -> anyhow::Result<ritk_image::native::Image<f32, C, 3>>
    where
        C: ComputeBackend,
        C::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            self.apply_values(
                image.data_slice()?.to_vec(),
                image.shape(),
                image.spacing().to_array(),
            ),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn apply_values<const D: usize>(
        &self,
        mut values: Vec<f32>,
        dims: [usize; D],
        spacing: [f64; D],
    ) -> Vec<f32> {
        for (axis, (&axis_len, &axis_spacing)) in dims.iter().zip(spacing.iter()).enumerate() {
            let sigma = self
                .sigmas
                .get(axis)
                .or_else(|| self.sigmas.first())
                .expect("invariant: GaussianFilter requires at least one sigma")
                .get();
            if sigma <= 1e-6 || axis_len <= 1 {
                continue;
            }
            let pixel_sigma = sigma / axis_spacing;
            let radius = (3.0 * pixel_sigma).ceil() as usize;
            let mut width = (2 * radius + 1).min(self.max_kernel_width);
            if width.is_multiple_of(2) {
                width -= 1;
            }
            let kernel = self.generate_kernel(pixel_sigma, (width - 1) / 2);
            values = convolve_axis_zero_padded(&values, &dims, axis, &kernel);
        }
        values
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
        let device = input.device();
        let dims: [usize; D] = input.shape().dims();
        let values = input
            .into_data()
            .into_vec::<f32>()
            .expect("invariant: GaussianFilter operates on f32 tensors");
        Tensor::from_data(
            TensorData::new(
                self.apply_values(values, dims, spacing.to_array()),
                Shape::new(dims),
            ),
            &device,
        )
    }

    fn generate_kernel(&self, sigma: f64, radius: usize) -> Vec<f32> {
        crate::gaussian_kernel(sigma as f32, Some(radius))
    }
}

fn convolve_axis_zero_padded<const D: usize>(
    input: &[f32],
    dims: &[usize; D],
    axis: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let radius = (kernel.len() / 2) as isize;
    let stride = dims[axis + 1..].iter().product::<usize>();
    let axis_len = dims[axis] as isize;
    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(input.len(), |index| {
        let coordinate = (index / stride % dims[axis]) as isize;
        kernel
            .iter()
            .enumerate()
            .filter_map(|(tap, &weight)| {
                let source_coordinate = coordinate + tap as isize - radius;
                (source_coordinate >= 0 && source_coordinate < axis_len).then(|| {
                    input[(index as isize + (source_coordinate - coordinate) * stride as isize)
                        as usize]
                        * weight
                })
            })
            .sum()
    })
}

#[cfg(test)]
#[path = "tests_gaussian.rs"]
mod tests;
