use crate::edge::GaussianSigma;
use ritk_core::image::Image;
use ritk_image::tensor::ops::ConvOptions;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor};
use ritk_spatial::Spacing;
use ritk_wgpu_compat::apply_row_chunks;

/// Default Gaussian kernel half-extent cap (`radius·2 + 1`), bounding the
/// per-axis convolution cost. Shared by [`GaussianFilter::new`] and the
/// burn-free [`gaussian_smooth_flat_3d`] entry so both smooth identically.
pub const DEFAULT_MAX_KERNEL_WIDTH: usize = 32;

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
            max_kernel_width: DEFAULT_MAX_KERNEL_WIDTH,
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
        let (tensor, origin, spacing, direction) = image.clone().into_parts();
        let data = self.apply_tensor(tensor, &spacing);
        Image::new(data, origin, spacing, direction)
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
        let dims: [usize; D] = data.shape().dims();

        // Apply 1D convolution along each dimension
        for d in 0..D {
            let sigma = if d < self.sigmas.len() {
                self.sigmas[d].get()
            } else {
                self.sigmas[0].get()
            };
            let spacing_val = spacing[d];

            // Skip if sigma is close to zero
            if sigma <= 1e-6 {
                continue;
            }

            // Skip a degenerate (size-1) axis: a length-1 signal cannot be
            // smoothed, and convolving it with a wide kernel under zero padding
            // would multiply the slice by only the kernel's centre weight,
            // darkening the whole image (e.g. a z=1 2-D-promoted volume → ≈0.2×).
            if dims[d] <= 1 {
                continue;
            }

            let kernel = axis_kernel(sigma, spacing_val, self.max_kernel_width);
            let kernel_tensor = Tensor::<B, 1>::from_floats(kernel.as_slice(), &device);

            // Convolve along dimension d
            data = self.convolve_1d::<D>(data, kernel_tensor, d);
        }
        data
    }

    /// Coeus-native sister of [`GaussianFilter::apply`] for 3-D images.
    ///
    /// Runs the identical separable zero-padded Gaussian smoothing as the Burn
    /// [`apply`](Self::apply) path — the same per-axis [`axis_kernel`] builder
    /// (shared SSOT for the kernel) and the same zero (constant-0) boundary
    /// convolution that Burn's `conv1d(padding = k/2)` performs — via the pure
    /// host core [`convolve_zero_pad_3d`]. No Burn tensor is constructed;
    /// spatial metadata is preserved.
    ///
    /// The result matches the Burn path to a derived floating-point tolerance
    /// (not bitwise): both evaluate the same separable zero-pad convolution with
    /// the same kernels, but Burn's `conv1d` and this host core sum the taps in
    /// different orders, so results differ only by accumulation rounding
    /// (`O(width · ε · ‖I‖∞)` per axis; see the differential tests).
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<C>(
        &self,
        image: &ritk_image::native::Image<f32, C, 3>,
        backend: &C,
    ) -> anyhow::Result<ritk_image::native::Image<f32, C, 3>>
    where
        C: coeus_core::ComputeBackend,
        C::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let sigmas: [f64; 3] = std::array::from_fn(|d| {
            if d < self.sigmas.len() {
                self.sigmas[d].get()
            } else {
                self.sigmas[0].get()
            }
        });
        let max_w = self.max_kernel_width;
        crate::native_support::map_flat_image(image, backend, move |vals, dims| {
            gaussian_smooth_native_flat(vals, dims, sigmas, spacing, max_w)
        })
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

        let mut permute_indices = [0isize; D];
        let mut index = 0;
        for axis in 0..D {
            if axis != dim {
                permute_indices[index] = axis as isize;
                index += 1;
            }
        }
        permute_indices[D - 1] = dim as isize;

        let input_permuted = input.permute(permute_indices);
        let last_dim_size = dims[dim];
        let batch_size = dims
            .iter()
            .enumerate()
            .filter_map(|(axis, &size)| (axis != dim).then_some(size))
            .product();
        let input_reshaped = input_permuted.reshape([batch_size, 1, last_dim_size]);
        let kernel_size = kernel.dims()[0];
        let kernel_reshaped = kernel.reshape([1, 1, kernel_size]);
        let padding = kernel_size / 2;
        let options = ConvOptions::new([1], [padding], [1], 1);
        let output_reshaped =
            apply_row_chunks(input_reshaped, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk| {
                ritk_image::tensor::module::conv1d(
                    chunk,
                    kernel_reshaped.clone(),
                    None,
                    options.clone(),
                )
            });

        let mut permuted_shape = [0; D];
        let mut permuted_index = 0;
        for (axis, &size) in dims.iter().enumerate() {
            if axis != dim {
                permuted_shape[permuted_index] = size;
                permuted_index += 1;
            }
        }
        permuted_shape[D - 1] = last_dim_size;
        let output_permuted = output_reshaped.reshape(Shape::new(permuted_shape));

        let mut inverse_permutation = [0isize; D];
        for (new_position, &old_position) in permute_indices.iter().enumerate() {
            inverse_permutation[old_position as usize] = new_position as isize;
        }
        output_permuted.permute(inverse_permutation)
    }
}

/// Build the 1-D Gaussian smoothing kernel for one axis.
///
/// Shared SSOT for the per-axis kernel used by both the Burn
/// [`GaussianFilter::apply`] path and the Coeus-native
/// [`GaussianFilter::apply_native`] path, so both convolve with identical
/// coefficients. The pixel-space sigma is `sigma_phys / spacing`; the kernel
/// half-width is `⌈3·pixel_sigma⌉` capped at `max_kernel_width` and forced odd,
/// then the coefficients are the normalised sampled Gaussian
/// ([`crate::gaussian_kernel`]).
fn axis_kernel(sigma_phys: f64, spacing: f64, max_kernel_width: usize) -> Vec<f32> {
    let pixel_sigma = sigma_phys / spacing;
    let radius = (3.0 * pixel_sigma).ceil() as usize;
    let mut width = (2 * radius + 1).min(max_kernel_width);
    if width.is_multiple_of(2) {
        width -= 1;
    }
    let actual_radius = (width - 1) / 2;
    crate::gaussian_kernel(pixel_sigma as f32, Some(actual_radius))
}

/// Burn-free host core for [`GaussianFilter::apply_native`]: separable
/// zero-padded Gaussian smoothing on a flat z-major buffer, matching the Burn
/// `conv1d(padding = k/2)` contract (per-axis [`axis_kernel`] +
/// [`convolve_zero_pad_3d`]). Shared by the native Gaussian path and the native
/// [`CannyEdgeDetector`](crate::edge::CannyEdgeDetector) smoothing stage, so no
/// Burn backend is needed to smooth.
///
/// The result matches the Burn path to a derived floating-point tolerance (not
/// bitwise): both evaluate the same kernels but sum the taps in different orders
/// (`conv1d` vs this correlation), differing only by accumulation rounding
/// (`O(width · ε · ‖I‖∞)` per axis).
/// Burn-free separable Gaussian smoothing of a flat z-major 3-D volume — the
/// public entry to the [`GaussianFilter::apply_native`] host core for callers
/// operating directly on flat host buffers (no Burn backend, no native `Image`
/// construction). Physical `sigmas`/`spacing` per axis; uses the shared
/// [`DEFAULT_MAX_KERNEL_WIDTH`], so the result matches [`GaussianFilter::apply`]
/// to the same derived accumulation-rounding tolerance as `apply_native`.
#[must_use]
pub fn gaussian_smooth_flat_3d(
    vals: &[f32],
    dims: [usize; 3],
    sigmas: [GaussianSigma; 3],
    spacing: [f64; 3],
) -> Vec<f32> {
    let sig = [sigmas[0].get(), sigmas[1].get(), sigmas[2].get()];
    gaussian_smooth_native_flat(vals, dims, sig, spacing, DEFAULT_MAX_KERNEL_WIDTH)
}

pub(crate) fn gaussian_smooth_native_flat(
    vals: &[f32],
    dims: [usize; 3],
    sigmas: [f64; 3],
    spacing: [f64; 3],
    max_kernel_width: usize,
) -> Vec<f32> {
    let mut data = vals.to_vec();
    for d in 0..3 {
        if sigmas[d] <= 1e-6 || dims[d] <= 1 {
            continue;
        }
        let kernel = axis_kernel(sigmas[d], spacing[d], max_kernel_width);
        data = convolve_zero_pad_3d(&data, dims, d, &kernel);
    }
    data
}

/// Convolve a flat C-order `[nz, ny, nx]` volume along `dim` with `kernel`,
/// using zero (constant-0) boundary padding.
///
/// This reproduces Burn `conv1d` with `padding = kernel.len() / 2`: for output
/// position `p`, `out[p] = Σ_k in[p − r + k] · kernel[k]` with `r = k/2` and any
/// out-of-range tap treated as `0`. The kernel is applied as correlation;
/// Gaussian kernels are symmetric, so this equals convolution. Zero padding
/// darkens boundary voxels (a constant field is *not* preserved at the edge),
/// matching the Burn Gaussian filter exactly in contract.
pub(crate) fn convolve_zero_pad_3d(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let r = (kernel.len() / 2) as isize;
    let (stride, len) = match dim {
        0 => (ny * nx, nz),
        1 => (nx, ny),
        2 => (1, nx),
        _ => unreachable!("3-D volume has axes 0..=2"),
    };
    let stride_i = stride as isize;
    let len_i = len as isize;
    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |idx| {
        // Position of this voxel along `dim`, and the flat offset of its line start.
        let p = match dim {
            0 => idx / (ny * nx),
            1 => (idx % (ny * nx)) / nx,
            _ => idx % nx,
        } as isize;
        let base = idx as isize - p * stride_i;
        let mut acc = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            let sp = p + k as isize - r;
            if sp >= 0 && sp < len_i {
                acc += data[(base + sp * stride_i) as usize] * w;
            }
        }
        acc
    })
}

#[cfg(test)]
#[path = "tests_gaussian.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_gaussian_native.rs"]
mod tests_native;
