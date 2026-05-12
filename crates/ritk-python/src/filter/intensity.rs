//! Intensity transform filters: rescale, windowing, thresholds, sigmoid, binary threshold.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::{
    BinaryThresholdImageFilter, IntensityWindowingFilter, RescaleIntensityFilter,
    SigmoidImageFilter, ThresholdImageFilter,
};

/// Linearly rescale image intensity to [out_min, out_max].
///
/// output(x) = (I(x) - I_min) / (I_max - I_min) * (out_max - out_min) + out_min
///
/// Args:
///     image:   Input PyImage.
///     out_min: Minimum output value (default 0.0).
///     out_max: Maximum output value (default 1.0).
///
/// Returns:
///     Rescaled PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, out_min=0.0_f32, out_max=1.0_f32))]
pub fn rescale_intensity(
    py: Python<'_>,
    image: &PyImage,
    out_min: f32,
    out_max: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = RescaleIntensityFilter::new(out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Clamp to [window_min, window_max] then rescale to [out_min, out_max].
///
/// Pixels below window_min map to out_min; pixels above window_max map to out_max.
///
/// Args:
///     image:      Input PyImage.
///     window_min: Lower intensity window bound.
///     window_max: Upper intensity window bound.
///     out_min:    Minimum output value (default 0.0).
///     out_max:    Maximum output value (default 1.0).
#[pyfunction]
#[pyo3(signature = (image, window_min, window_max, out_min=0.0_f32, out_max=1.0_f32))]
pub fn intensity_windowing(
    py: Python<'_>,
    image: &PyImage,
    window_min: f32,
    window_max: f32,
    out_min: f32,
    out_max: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = IntensityWindowingFilter::new(window_min, window_max, out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels strictly below threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_below(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::below(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels strictly above threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_above(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::above(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels outside [lower, upper] to outside_value; keep interior pixels unchanged.
#[pyfunction]
#[pyo3(signature = (image, lower, upper, outside_value=0.0_f32))]
pub fn threshold_outside(
    py: Python<'_>,
    image: &PyImage,
    lower: f32,
    upper: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::outside(lower, upper, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Sigmoid intensity transform.
///
/// f(I; alpha, beta) = (max - min) / (1 + exp(-(I - beta) / alpha)) + min
///
/// Parameter convention matches SimpleITK `SigmoidImageFilter`:
/// - alpha controls the slope (transition width): positive → increasing, negative → decreasing.
/// - beta is the shift (inflection point): output = (max + min) / 2 when I = beta.
///
/// At I = beta: exp(0) = 1, so output = (max - min) / 2 + min = (max + min) / 2.
///
/// Args:
///     image:      Input PyImage.
///     alpha:      Transition width / slope. Larger |alpha| = more gradual transition.
///     beta:       Inflection point (input value where output = (max+min)/2).
///     min_output: Minimum output value (default 0.0).
///     max_output: Maximum output value (default 1.0).
#[pyfunction]
#[pyo3(signature = (image, alpha, beta, min_output=0.0_f32, max_output=1.0_f32))]
pub fn sigmoid_filter(
    py: Python<'_>,
    image: &PyImage,
    alpha: f32,
    beta: f32,
    min_output: f32,
    max_output: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        // Rust SigmoidImageFilter uses (inflection=alpha_rust, width=beta_rust).
        // Python/SimpleITK convention: alpha=width, beta=inflection.
        // Map: inflection=beta (Python), width=alpha (Python).
        let filter = SigmoidImageFilter::new(beta, alpha, min_output, max_output);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Binary threshold: foreground if I in [lower_threshold, upper_threshold], else background.
///
/// Args:
///     image:            Input PyImage.
///     lower_threshold:  Inclusive lower bound.
///     upper_threshold:  Inclusive upper bound.
///     foreground:       Value for pixels inside the interval (default 1.0).
///     background:       Value for pixels outside the interval (default 0.0).
#[pyfunction]
#[pyo3(signature = (image, lower_threshold, upper_threshold, foreground=1.0_f32, background=0.0_f32))]
pub fn binary_threshold(
    py: Python<'_>,
    image: &PyImage,
    lower_threshold: f32,
    upper_threshold: f32,
    foreground: f32,
    background: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = BinaryThresholdImageFilter::new(
            lower_threshold,
            upper_threshold,
            foreground,
            background,
        );
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}
