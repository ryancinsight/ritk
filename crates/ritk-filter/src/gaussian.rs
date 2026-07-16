use crate::edge::GaussianSigma;
use ritk_image::native::Image;
use ritk_spatial::Spacing;

/// Default Gaussian kernel half-extent cap (`radius·2 + 1`), bounding the
/// per-axis convolution cost. Shared by [`GaussianFilter::apply`] and the
/// burn-free [`gaussian_smooth_flat_3d`] entry so both smooth identically.
pub const DEFAULT_MAX_KERNEL_WIDTH: usize = 32;

/// Gaussian smoothing filter.
///
/// Applies a Gaussian smoothing filter to an image using separable 1D convolutions.
/// This implementation respects the physical spacing of the image.
#[derive(Clone)]
pub struct GaussianFilter {
    sigmas: Vec<GaussianSigma>,
    max_kernel_width: usize,
}

impl GaussianFilter {
    /// Create a new Gaussian filter with the given standard deviation (in physical units).
    ///
    /// # Arguments
    /// * `sigmas` - Standard deviation for each dimension in physical units (mm).
    pub fn new(sigmas: Vec<GaussianSigma>) -> Self {
        Self {
            sigmas,
            max_kernel_width: DEFAULT_MAX_KERNEL_WIDTH,
        }
    }

    /// Set the maximum kernel width (radius * 2 + 1).
    pub fn with_max_kernel_width(mut self, width: usize) -> Self {
        self.max_kernel_width = width;
        self
    }

    /// Apply the filter to a 3-D Coeus-native image.
    ///
    /// Runs the identical separable zero-padded Gaussian smoothing as the legacy
    /// Burn path — the same per-axis `axis_kernel` builder (shared SSOT for the
    /// kernel) and the same zero (constant-0) boundary convolution that Burn's
    /// `conv1d(padding = k/2)` performs — via the pure host core
    /// `convolve_zero_pad_3d`. No Burn tensor is constructed; spatial metadata
    /// is preserved.
    ///
    /// The result matches the legacy Burn path to a derived floating-point
    /// tolerance (not bitwise): both evaluate the same separable zero-pad
    /// convolution with the same kernels, but Burn's `conv1d` and this host
    /// core sum the taps in different orders, so results differ only by
    /// accumulation rounding (`O(width · ε · ‖I‖∞)` per axis; see the
    /// differential tests).
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
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
}

/// Build the 1-D Gaussian smoothing kernel for one axis.
///
/// Shared SSOT for the per-axis kernel used by both the legacy Burn
/// `GaussianFilter::apply` path and the Coeus-native
/// `GaussianFilter::apply` path, so both convolve with identical
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

/// Burn-free host core for [`GaussianFilter::apply`]: separable
/// zero-padded Gaussian smoothing on a flat z-major 3-D volume — the
/// public entry to the host core for callers
/// operating directly on flat host buffers (no Burn backend, no native `Image`
/// construction). Physical `sigmas`/`spacing` per axis; uses the shared
/// [`DEFAULT_MAX_KERNEL_WIDTH`], so the result matches the legacy
/// [`GaussianFilter::apply`] to the same derived accumulation-rounding
/// tolerance.
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
