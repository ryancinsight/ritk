//! Python-exposed image filtering functions.
//!
//! All filters delegate to `ritk-core::filter` implementations (SSOT).
//!
//! # Submodules
//! - `arithmetic`: Pixelwise binary image operations (add, subtract, multiply, divide, min, max).
//! - `smooth`:    Gaussian, discrete Gaussian, median, bilateral, N4, anisotropic/curvature diffusion, recursive Gaussian.
//! - `edge`:      Gradient magnitude, Laplacian, Canny, LoG, Sobel.
//! - `vessel`:    Frangi vesselness, Sato line filter.
//! - `intensity`: Rescale, windowing, threshold variants, sigmoid, binary threshold, blend.
//! - `morphology`: Grayscale erosion/dilation, label morphology, top-hat, hit-or-miss, reconstruction.
//! - `spatial`:   Resample image, distance transform.

mod arithmetic;
mod edge;
mod intensity;
mod morphology;
mod smooth;
mod spatial;
mod vessel;

use pyo3::prelude::*;

pub use arithmetic::*;
pub use edge::*;
pub use intensity::*;
pub use morphology::*;
pub use smooth::*;
pub use spatial::*;
pub use vessel::*;

/// Register the `filter` submodule and all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "filter")?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(discrete_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(n4_bias_correction, &m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(curvature_anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(gradient_magnitude, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian, &m)?)?;
    m.add_function(wrap_pyfunction!(canny_edge_detect, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_of_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel_gradient, &m)?)?;
    m.add_function(wrap_pyfunction!(frangi_vesselness, &m)?)?;
    m.add_function(wrap_pyfunction!(sato_line_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(rescale_intensity, &m)?)?;
    m.add_function(wrap_pyfunction!(intensity_windowing, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_below, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_above, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_outside, &m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(blend_images, &m)?)?;
    m.add_function(wrap_pyfunction!(add_images, &m)?)?;
    m.add_function(wrap_pyfunction!(subtract_images, &m)?)?;
    m.add_function(wrap_pyfunction!(multiply_images, &m)?)?;
    m.add_function(wrap_pyfunction!(divide_images, &m)?)?;
    m.add_function(wrap_pyfunction!(minimum_images, &m)?)?;
    m.add_function(wrap_pyfunction!(maximum_images, &m)?)?;
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
    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
