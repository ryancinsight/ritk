//! Python-exposed image filtering functions.
//!
//! All filters delegate to `ritk-core::filter` implementations (SSOT).
//!
//! # Submodules
//! - `arithmetic`:  Pixelwise binary image operations (add, subtract, multiply, divide, min, max).
//! - `fft`:         Frequency-domain transforms (forward FFT, inverse FFT, FFT shift).
//! - `smooth`:      Gaussian, discrete Gaussian, median, bilateral, N4, anisotropic/curvature diffusion, recursive Gaussian.
//! - `edge`:        Gradient magnitude, Laplacian, Canny, LoG, Sobel.
//! - `vessel`:      Frangi vesselness, Sato line filter.
//! - `intensity`:   Rescale, windowing, threshold variants, sigmoid, binary threshold, blend.
//! - `morphology`:  Grayscale erosion/dilation, label morphology, top-hat, hit-or-miss, reconstruction.
//! - `projection`:  MaxIP, MinIP, MeanIP, SumIP, StdDevIP along arbitrary axis.
//! - `spatial`:     Resample image, distance transform.

mod arithmetic;
mod deconvolution;
mod edge;
mod fft;
mod intensity;
mod morphology;
mod noise;
mod projection;
mod smooth;
mod spatial;
mod vessel;

use pyo3::prelude::*;

pub use arithmetic::*;
pub use deconvolution::*;
pub use edge::*;
pub use fft::*;
pub use intensity::*;
pub use morphology::*;
pub use noise::*;
pub use projection::*;
pub use smooth::*;
pub use spatial::*;
pub use vessel::*;

/// Register the `filter` submodule and all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "filter")?;
    // Smoothing & diffusion
    m.add_class::<PySpacingMode>()?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(discrete_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(n4_bias_correction, &m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(curvature_anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian_directional, &m)?)?;
    // Edge detection
    m.add_function(wrap_pyfunction!(gradient_magnitude, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian, &m)?)?;
    m.add_function(wrap_pyfunction!(canny_edge_detect, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_of_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel_gradient, &m)?)?;
    // Vesselness
    m.add_function(wrap_pyfunction!(frangi_vesselness, &m)?)?;
    m.add_function(wrap_pyfunction!(sato_line_filter, &m)?)?;
    // Intensity transforms
    m.add_function(wrap_pyfunction!(rescale_intensity, &m)?)?;
    m.add_function(wrap_pyfunction!(intensity_windowing, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_below, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_above, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_outside, &m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(blend_images, &m)?)?;
    // Additional intensity transforms
    m.add_function(wrap_pyfunction!(normalize_image, &m)?)?;
    m.add_function(wrap_pyfunction!(unsharp_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(zero_crossing_image, &m)?)?;
    // Arithmetic
    m.add_function(wrap_pyfunction!(add_images, &m)?)?;
    m.add_function(wrap_pyfunction!(subtract_images, &m)?)?;
    m.add_function(wrap_pyfunction!(multiply_images, &m)?)?;
    m.add_function(wrap_pyfunction!(divide_images, &m)?)?;
    m.add_function(wrap_pyfunction!(minimum_images, &m)?)?;
    m.add_function(wrap_pyfunction!(maximum_images, &m)?)?;
    // Unary math (ITK Abs/Sqrt/Square/Exp/Log/Sin/Cos/Tan/Asin/Acos/Atan/BoundedReciprocal)
    m.add_function(wrap_pyfunction!(abs_image, &m)?)?;
    m.add_function(wrap_pyfunction!(sqrt_image, &m)?)?;
    m.add_function(wrap_pyfunction!(square_image, &m)?)?;
    m.add_function(wrap_pyfunction!(exp_image, &m)?)?;
    m.add_function(wrap_pyfunction!(log_image, &m)?)?;
    m.add_function(wrap_pyfunction!(sin_image, &m)?)?;
    m.add_function(wrap_pyfunction!(cos_image, &m)?)?;
    m.add_function(wrap_pyfunction!(tan_image, &m)?)?;
    m.add_function(wrap_pyfunction!(asin_image, &m)?)?;
    m.add_function(wrap_pyfunction!(acos_image, &m)?)?;
    m.add_function(wrap_pyfunction!(atan_image, &m)?)?;
    m.add_function(wrap_pyfunction!(bounded_reciprocal_image, &m)?)?;
    m.add_function(wrap_pyfunction!(clamp_image, &m)?)?;
    m.add_function(wrap_pyfunction!(invert_intensity, &m)?)?;
    m.add_function(wrap_pyfunction!(mask_image, &m)?)?;
    m.add_function(wrap_pyfunction!(mask_negated_image, &m)?)?;
    // Per-component (RGB/vector) filters
    m.add_function(wrap_pyfunction!(crate::color::color_median, &m)?)?;
    // Morphology
    m.add_function(wrap_pyfunction!(grayscale_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(label_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(label_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(label_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(label_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(white_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(black_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(hit_or_miss, &m)?)?;
    m.add_function(wrap_pyfunction!(morphological_reconstruction, &m)?)?;
    // Spatial transforms
    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;
    m.add_function(wrap_pyfunction!(rotate_image, &m)?)?;
    m.add_function(wrap_pyfunction!(shift_image, &m)?)?;
    m.add_function(wrap_pyfunction!(zoom_image, &m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, &m)?)?;
    // FFT / frequency domain
    m.add_function(wrap_pyfunction!(forward_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_shift, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_convolve, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_convolve_3d, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_normalized_correlate, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_normalized_correlate_3d, &m)?)?;
    // Frequency-domain filters
    m.add_function(wrap_pyfunction!(fft_ideal_low_pass, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_ideal_high_pass, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_butterworth_low_pass, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_butterworth_high_pass, &m)?)?;
    // Intensity projection
    m.add_function(wrap_pyfunction!(max_intensity_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(min_intensity_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_intensity_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(sum_intensity_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(stddev_intensity_projection, &m)?)?;
    // Noise simulation
    m.add_function(wrap_pyfunction!(additive_gaussian_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(salt_and_pepper_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(shot_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(speckle_noise, &m)?)?;
    // Deconvolution
    m.add_function(wrap_pyfunction!(wiener_deconvolution, &m)?)?;
    m.add_function(wrap_pyfunction!(tikhonov_deconvolution, &m)?)?;
    m.add_function(wrap_pyfunction!(richardson_lucy_deconvolution, &m)?)?;
    m.add_function(wrap_pyfunction!(landweber_deconvolution, &m)?)?;
    // Coherence-enhancing diffusion
    m.add_function(wrap_pyfunction!(coherence_enhancing_diffusion, &m)?)?;
    // Bin-shrink downsampling
    m.add_function(wrap_pyfunction!(bin_shrink, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
