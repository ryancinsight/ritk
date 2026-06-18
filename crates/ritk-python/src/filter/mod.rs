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
mod transform;
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
pub use transform::*;
pub use vessel::*;

/// Register the `filter` submodule and all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "filter")?;
    // Smoothing & diffusion
    m.add_class::<PySpacingMode>()?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(discrete_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(discrete_gaussian_derivative, &m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(binomial_blur, &m)?)?;
    m.add_function(wrap_pyfunction!(box_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(box_sigma, &m)?)?;
    m.add_function(wrap_pyfunction!(local_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(rank, &m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(n4_bias_correction, &m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(curvature_anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(curvature_flow, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian_directional, &m)?)?;
    // Edge detection
    m.add_function(wrap_pyfunction!(gradient_magnitude, &m)?)?;
    m.add_function(wrap_pyfunction!(derivative, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_sharpening, &m)?)?;
    m.add_function(wrap_pyfunction!(zero_crossing_based_edge_detection, &m)?)?;
    m.add_function(wrap_pyfunction!(iso_contour_distance, &m)?)?;
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
    m.add_function(wrap_pyfunction!(double_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(blend_images, &m)?)?;
    // Additional intensity transforms
    m.add_function(wrap_pyfunction!(normalize_image, &m)?)?;
    m.add_function(wrap_pyfunction!(normalize_to_constant, &m)?)?;
    m.add_function(wrap_pyfunction!(unsharp_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(zero_crossing_image, &m)?)?;
    // Arithmetic
    m.add_function(wrap_pyfunction!(add_images, &m)?)?;
    m.add_function(wrap_pyfunction!(subtract_images, &m)?)?;
    m.add_function(wrap_pyfunction!(multiply_images, &m)?)?;
    m.add_function(wrap_pyfunction!(divide_images, &m)?)?;
    m.add_function(wrap_pyfunction!(minimum_images, &m)?)?;
    m.add_function(wrap_pyfunction!(maximum_images, &m)?)?;
    m.add_function(wrap_pyfunction!(squared_difference_images, &m)?)?;
    m.add_function(wrap_pyfunction!(absolute_value_difference_images, &m)?)?;
    m.add_function(wrap_pyfunction!(atan2_images, &m)?)?;
    m.add_function(wrap_pyfunction!(pow_images, &m)?)?;
    // Unary math (ITK Abs/Sqrt/Square/Exp/Log/Sin/Cos/Tan/Asin/Acos/Atan/BoundedReciprocal)
    m.add_function(wrap_pyfunction!(abs_image, &m)?)?;
    m.add_function(wrap_pyfunction!(sqrt_image, &m)?)?;
    m.add_function(wrap_pyfunction!(square_image, &m)?)?;
    m.add_function(wrap_pyfunction!(exp_image, &m)?)?;
    m.add_function(wrap_pyfunction!(log_image, &m)?)?;
    m.add_function(wrap_pyfunction!(log10_image, &m)?)?;
    m.add_function(wrap_pyfunction!(exp_negative_image, &m)?)?;
    m.add_function(wrap_pyfunction!(unary_minus_image, &m)?)?;
    m.add_function(wrap_pyfunction!(round_image, &m)?)?;
    m.add_function(wrap_pyfunction!(not_image, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_not, &m)?)?;
    m.add_function(wrap_pyfunction!(modulus, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_magnitude_images, &m)?)?;
    m.add_function(wrap_pyfunction!(equal_images, &m)?)?;
    m.add_function(wrap_pyfunction!(not_equal_images, &m)?)?;
    m.add_function(wrap_pyfunction!(greater_images, &m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal_images, &m)?)?;
    m.add_function(wrap_pyfunction!(less_images, &m)?)?;
    m.add_function(wrap_pyfunction!(less_equal_images, &m)?)?;
    m.add_function(wrap_pyfunction!(and_images, &m)?)?;
    m.add_function(wrap_pyfunction!(or_images, &m)?)?;
    m.add_function(wrap_pyfunction!(xor_images, &m)?)?;
    m.add_function(wrap_pyfunction!(divide_real_images, &m)?)?;
    m.add_function(wrap_pyfunction!(divide_floor_images, &m)?)?;
    m.add_function(wrap_pyfunction!(nary_add, &m)?)?;
    m.add_function(wrap_pyfunction!(nary_maximum, &m)?)?;
    m.add_function(wrap_pyfunction!(ternary_add_images, &m)?)?;
    m.add_function(wrap_pyfunction!(ternary_magnitude_images, &m)?)?;
    m.add_function(wrap_pyfunction!(ternary_magnitude_squared_images, &m)?)?;
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
    m.add_function(wrap_pyfunction!(masked_assign, &m)?)?;
    // Per-component (RGB/vector) filters
    m.add_function(wrap_pyfunction!(crate::color::color_median, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::color_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::color::color_smoothing_recursive_gaussian,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::color::compose, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::gradient, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::gradient_recursive_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::scalar_to_rgb_colormap, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::label_to_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::label_overlay, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::physical_point_image_source, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::vector_index_selection_cast, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::vector_magnitude, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::color::edge_potential, &m)?)?;
    // Morphology
    m.add_function(wrap_pyfunction!(grayscale_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(erode_object_morphology, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_thinning, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_pruning, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(label_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(label_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(label_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(label_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(white_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(black_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(hit_or_miss, &m)?)?;
    m.add_function(wrap_pyfunction!(morphological_reconstruction, &m)?)?;
    m.add_function(wrap_pyfunction!(h_maxima, &m)?)?;
    m.add_function(wrap_pyfunction!(h_minima, &m)?)?;
    m.add_function(wrap_pyfunction!(h_convex, &m)?)?;
    m.add_function(wrap_pyfunction!(h_concave, &m)?)?;
    m.add_function(wrap_pyfunction!(regional_maxima, &m)?)?;
    m.add_function(wrap_pyfunction!(regional_minima, &m)?)?;
    m.add_function(wrap_pyfunction!(valued_regional_maxima, &m)?)?;
    m.add_function(wrap_pyfunction!(valued_regional_minima, &m)?)?;
    m.add_function(wrap_pyfunction!(opening_by_reconstruction, &m)?)?;
    m.add_function(wrap_pyfunction!(closing_by_reconstruction, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_fillhole, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_grind_peak, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_contour, &m)?)?;
    m.add_function(wrap_pyfunction!(label_contour, &m)?)?;
    m.add_function(wrap_pyfunction!(voting_binary, &m)?)?;
    m.add_function(wrap_pyfunction!(voting_binary_hole_filling, &m)?)?;
    m.add_function(wrap_pyfunction!(voting_binary_iterative_hole_filling, &m)?)?;

    m.add_function(wrap_pyfunction!(flip, &m)?)?;
    m.add_function(wrap_pyfunction!(constant_pad, &m)?)?;
    m.add_function(wrap_pyfunction!(mirror_pad, &m)?)?;
    m.add_function(wrap_pyfunction!(wrap_pad, &m)?)?;
    m.add_function(wrap_pyfunction!(region_of_interest, &m)?)?;
    m.add_function(wrap_pyfunction!(crop, &m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_shift, &m)?)?;
    m.add_function(wrap_pyfunction!(join_series, &m)?)?;
    m.add_function(wrap_pyfunction!(tile, &m)?)?;
    m.add_function(wrap_pyfunction!(checker_board, &m)?)?;
    m.add_function(wrap_pyfunction!(slice_image, &m)?)?;
    m.add_function(wrap_pyfunction!(expand, &m)?)?;
    m.add_function(wrap_pyfunction!(shrink, &m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_image_source, &m)?)?;
    m.add_function(wrap_pyfunction!(grid_image_source, &m)?)?;
    m.add_function(wrap_pyfunction!(gabor_image_source, &m)?)?;
    m.add_function(wrap_pyfunction!(zero_flux_neumann_pad, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_pad, &m)?)?;
    m.add_function(wrap_pyfunction!(permute_axes, &m)?)?;
    m.add_function(wrap_pyfunction!(paste, &m)?)?;
    // Spatial transforms
    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;
    m.add_function(wrap_pyfunction!(warp, &m)?)?;
    m.add_function(wrap_pyfunction!(stochastic_fractal_dimension, &m)?)?;
    m.add_function(wrap_pyfunction!(transform_to_displacement_field, &m)?)?;
    m.add_function(wrap_pyfunction!(bspline_decomposition, &m)?)?;
    m.add_function(wrap_pyfunction!(rotate_image, &m)?)?;
    m.add_function(wrap_pyfunction!(shift_image, &m)?)?;
    m.add_function(wrap_pyfunction!(zoom_image, &m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, &m)?)?;
    m.add_function(wrap_pyfunction!(signed_distance_map, &m)?)?;
    // FFT / frequency domain
    m.add_function(wrap_pyfunction!(forward_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(real_to_half_hermitian_forward_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(half_hermitian_to_real_inverse_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse_fft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft_shift, &m)?)?;
    m.add_function(wrap_pyfunction!(complex_to_real, &m)?)?;
    m.add_function(wrap_pyfunction!(complex_to_imaginary, &m)?)?;
    m.add_function(wrap_pyfunction!(complex_to_modulus, &m)?)?;
    m.add_function(wrap_pyfunction!(complex_to_phase, &m)?)?;
    m.add_function(wrap_pyfunction!(real_and_imaginary_to_complex, &m)?)?;
    m.add_function(wrap_pyfunction!(magnitude_and_phase_to_complex, &m)?)?;
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
    m.add_function(wrap_pyfunction!(median_intensity_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_projection, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold_projection, &m)?)?;
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
    m.add_function(wrap_pyfunction!(projected_landweber_deconvolution, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse_deconvolution, &m)?)?;
    // Coherence-enhancing diffusion
    m.add_function(wrap_pyfunction!(coherence_enhancing_diffusion, &m)?)?;
    // Bin-shrink downsampling
    m.add_function(wrap_pyfunction!(bin_shrink, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
