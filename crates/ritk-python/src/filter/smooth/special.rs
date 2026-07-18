//! Special-purpose filters: median, bilateral, N4 bias correction, and bin-shrink downsampling.
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, image_to_vec, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::bias::N4Config;
use ritk_filter::{
    BilateralFilter, BinShrinkImageFilter, BinomialBlurImageFilter, BoxMeanImageFilter,
    BoxSigmaImageFilter, MeanImageFilter, MedianFilter, N4BiasFieldCorrectionFilter,
    NoiseImageFilter, RankImageFilter, SpatialConvolutionFilter };

/// Apply a mean (box) filter: each voxel becomes the average of the
/// axis-aligned cube of half-width `radius` (replicate padding).
/// ITK Parity: MeanImageFilter.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn mean_filter(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        MeanImageFilter::new(radius)
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Apply a box mean filter with per-axis radii: each voxel becomes the average
/// over the `(2r+1)` window clipped to the image bounds (dividing by the
/// in-bounds count â€” shrink boundary, unlike `mean_filter`'s clamped full
/// window). ITK Parity: BoxMeanImageFilter (`sitk.BoxMean`, radius `[rx,ry,rz]`).
#[pyfunction]
#[pyo3(signature = (image, radius_z=1, radius_y=1, radius_x=1))]
pub fn box_mean(
    py: Python<'_>,
    image: &PyImage,
    radius_z: usize,
    radius_y: usize,
    radius_x: usize,
) -> PyImage {
    let image = py_image_to_burn(image);
    let out =
        py.allow_threads(|| BoxMeanImageFilter::new([radius_z, radius_y, radius_x]).apply(&image));
    burn_into_py_image(out)
}

/// Apply a box sigma filter: the per-axis sample standard deviation over the
/// `(2r+1)` window clipped to the image bounds (Bessel-corrected, divisor
/// `nâˆ’1`; shrink boundary). ITK Parity: BoxSigmaImageFilter (`sitk.BoxSigma`,
/// radius `[rx,ry,rz]`).
#[pyfunction]
#[pyo3(signature = (image, radius_z=1, radius_y=1, radius_x=1))]
pub fn box_sigma(
    py: Python<'_>,
    image: &PyImage,
    radius_z: usize,
    radius_y: usize,
    radius_x: usize,
) -> PyImage {
    let image = py_image_to_burn(image);
    let out =
        py.allow_threads(|| BoxSigmaImageFilter::new([radius_z, radius_y, radius_x]).apply(&image));
    burn_into_py_image(out)
}

/// Estimate local image noise: the per-axis sample standard deviation over the
/// full `(2r+1)` window under a ZeroFluxNeumann boundary (out-of-bounds
/// neighbours clamp to the edge, so the window count is constant). Differs from
/// `box_sigma` only at the border (full clamped window vs clipped window). ITK
/// Parity: NoiseImageFilter (`sitk.Noise`, radius `[rx,ry,rz]`, default 1).
#[pyfunction]
#[pyo3(signature = (image, radius_z=1, radius_y=1, radius_x=1))]
pub fn local_noise(
    py: Python<'_>,
    image: &PyImage,
    radius_z: usize,
    radius_y: usize,
    radius_x: usize,
) -> PyImage {
    let image = py_image_to_burn(image);
    let out =
        py.allow_threads(|| NoiseImageFilter::new([radius_z, radius_y, radius_x]).apply(&image));
    burn_into_py_image(out)
}

/// Apply a box rank filter: the order statistic at `floor(rankÂ·(nâˆ’1))` of the
/// sorted `(2r+1)` window clipped to the image bounds (`rank=0.5` is the median;
/// shrink boundary). ITK Parity: RankImageFilter (`sitk.Rank`, radius `[rx,ry,rz]`).
#[pyfunction]
#[pyo3(signature = (image, rank=0.5, radius_z=1, radius_y=1, radius_x=1))]
pub fn rank(
    py: Python<'_>,
    image: &PyImage,
    rank: f64,
    radius_z: usize,
    radius_y: usize,
    radius_x: usize,
) -> PyImage {
    let image = py_image_to_burn(image);
    let out = py
        .allow_threads(|| RankImageFilter::new([radius_z, radius_y, radius_x], rank).apply(&image));
    burn_into_py_image(out)
}

/// Apply a binomial blur: the separable `[Â¼,Â½,Â¼]` kernel along each axis,
/// repeated `repetitions` times (reflect boundary). ITK Parity:
/// BinomialBlurImageFilter (`sitk.BinomialBlur`).
#[pyfunction]
#[pyo3(signature = (image, repetitions=1))]
pub fn binomial_blur(py: Python<'_>, image: &PyImage, repetitions: usize) -> PyImage {
    let image = py_image_to_burn(image);
    let out = py.allow_threads(|| BinomialBlurImageFilter::new(repetitions).apply(&image));
    burn_into_py_image(out)
}

/// Apply a median (rank) filter for impulse-noise removal.
///
/// For each voxel the output is the median of all voxels within the
/// axis-aligned cube of half-width `radius` voxels. Out-of-bounds positions
/// are clamped to the nearest valid voxel (replicate padding).
///
/// Args:
///     image: Input PyImage.
///     radius: Neighbourhood half-width in voxels (default 1 â†’ 3Ã—3Ã—3 cube).
///         The kernel contains `(2*radius + 1)^3` samples.
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn median_filter(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let filter = MedianFilter::new(radius);
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Apply a bilateral filter (edge-preserving smoothing).
///
/// Each output voxel is a weighted average of its neighbourhood, where the
/// weight combines:
/// - A **spatial** Gaussian: `exp(âˆ’||p âˆ’ q||Â² / (2 Ïƒ_sÂ²))`
/// - A **range** Gaussian: `exp(âˆ’(I(p) âˆ’ I(q))Â² / (2 Ïƒ_rÂ²))`
///
/// Args:
///     image: Input PyImage.
///     spatial_sigma: Spatial Gaussian sigma in voxels.
///     range_sigma: Intensity range sigma (same units as voxel values).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
pub fn bilateral_filter(
    py: Python<'_>,
    image: &PyImage,
    spatial_sigma: f64,
    range_sigma: f64,
) -> RitkResult<PyImage> {
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let filter = BilateralFilter::new(spatial_sigma, range_sigma);
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Apply N4 bias field correction to an MRI image.
///
/// Corrects low-frequency multiplicative intensity inhomogeneity caused by
/// RF coil non-uniformity. Based on Tustison et al. (2010),
/// *IEEE Trans. Med. Imaging* 29(6):1310â€“1320.
///
/// This is a from-scratch N4 (not a wrapper around ITK) that follows the ITK/ANTs
/// algorithm: ITK `SharpenImage` histogram sharpening (Wiener deconvolution + the
/// E[v|u] expectation map) and the Leeâ€“Wolbergâ€“Shin multilevel B-spline
/// (scattered-data) fit, with the bias estimated on the input downsampled by a
/// shrink factor and the log-bias control lattice evaluated at full resolution.
/// It reduces within-tissue coefficient of variation comparably to ANTsPy's
/// `n4_bias_field_correction`. Because N4 is ill-posed the *estimated bias field*
/// still differs in detail from ANTs/SimpleITK (which themselves differ); ANTsPy
/// is the preferred reference here.
///
/// Args:
///     image: Input PyImage (must be f32, values > 0).
///     num_fitting_levels: Number of B-spline refinement levels (default 4).
///     num_iterations: Maximum iterations per level (default 50).
///     noise_estimate: Histogram-sharpening / Wiener noise term (default 0.01).
///     shrink_factor: The bias field is estimated on the input downsampled by
///         this isotropic factor (ITK/ANTs `shrinkFactor`, default 4), then
///         evaluated at full resolution. The factor is adapted down so the
///         smallest shrunk dimension stays â‰¥ 4; for small volumes (â‰² 32 voxels
///         per side) pass ``shrink_factor=1`` â€” the default 4 is tuned for
///         clinical-resolution images and, like ANTs at shrink 4, under-corrects
///         small phantoms.
///
/// Returns:
///     Bias-corrected PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, num_fitting_levels=4, num_iterations=50, noise_estimate=0.01, shrink_factor=4))]
pub fn n4_bias_correction(
    py: Python<'_>,
    image: &PyImage,
    num_fitting_levels: usize,
    num_iterations: usize,
    noise_estimate: f64,
    shrink_factor: usize,
) -> RitkResult<PyImage> {
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let config = N4Config {
            num_fitting_levels,
            num_iterations,
            noise_estimate,
            shrink_factor,
            ..Default::default()
        };
        let filter = N4BiasFieldCorrectionFilter::new(config);
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Apply bin-shrink downsampling (integer sub-sampling by bin averaging).
///
/// Reduces image dimensions by integer factors by averaging all voxels
/// within each non-overlapping bin. This provides anti-aliasing compared
/// to naive sub-sampling (which just takes every Nth voxel).
///
/// Output shape\[d\] = floor(input_shape\[d\] / factor\[d\]).
/// Spacing is multiplied by the shrink factor.
///
/// Args:
///     image: Input PyImage.
///     factor_z: Shrink factor along Z axis (default 2).
///     factor_y: Shrink factor along Y axis (default 2).
///     factor_x: Shrink factor along X axis (default 2).
///
/// Returns:
///     Downsampled PyImage with reduced shape and scaled spacing.
#[pyfunction]
#[pyo3(signature = (image, factor_z=2, factor_y=2, factor_x=2))]
pub fn bin_shrink(
    py: Python<'_>,
    image: &PyImage,
    factor_z: usize,
    factor_y: usize,
    factor_x: usize,
) -> PyImage {
    let image = py_image_to_burn(image);
    let result = py.allow_threads(|| {
        let filter = BinShrinkImageFilter::new(vec![factor_z, factor_y, factor_x]);
        filter.apply(&image)
    });
    burn_into_py_image(result)
}

/// Convolve a 3-D image with a kernel image using zero-flux Neumann boundary
/// conditions.
///
/// Matches `sitk.Convolve` (`ConvolutionImageFilter`). The kernel is centred
/// on each output voxel; boundary voxels clamp to the nearest edge value
/// (zero-flux Neumann). The caller is responsible for kernel normalisation.
///
/// # Mathematical contract
///
/// For kernel shape `[Kz, Ky, Kx]` and half-extents `hz = Kz/2` etc.:
///
/// ```text
/// output[z, y, x] =
///     Î£_{kz,ky,kx}  kernel[kz, ky, kx]
///     Â· image[clamp(z + kz âˆ’ hz, 0, Dzâˆ’1),
///              clamp(y + ky âˆ’ hy, 0, Dyâˆ’1),
///              clamp(x + kx âˆ’ hx, 0, Dxâˆ’1)]
/// ```
///
/// Args:
///     image:  Input 3-D PyImage.
///     kernel: Kernel 3-D PyImage (odd dimensions recommended for exact
///             centring). Must be non-empty.
///
/// Returns:
///     Convolved PyImage of the same shape as `image`.
///
/// Raises:
///     ValueError:    when the kernel shape is incompatible with its buffer.
///     RuntimeError:  on internal tensor extraction failure.
#[pyfunction]
pub fn spatial_convolve(py: Python<'_>, image: &PyImage, kernel: &PyImage) -> RitkResult<PyImage> {
    let image_inner = py_image_to_burn(image);
    let (kernel_vals, kernel_dims) = image_to_vec(kernel.inner.as_ref());
    py.allow_threads(|| {
        let filter = SpatialConvolutionFilter::new(kernel_vals, kernel_dims)
            .map_err(|e| RitkPyError::value(e.to_string()))?;
        filter
            .apply(&image_inner)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}
