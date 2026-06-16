//! Intensity transform filters: rescale, windowing, thresholds, sigmoid, binary threshold, blend,
//! normalize, unsharp mask, and zero crossing.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_filter::{
    BinaryThresholdImageFilter, BlendImageFilter, ClampPolicy, IntensityWindowingFilter,
    NormalizeImageFilter, RescaleIntensityFilter, SigmoidImageFilter, ThresholdImageFilter,
    UnsharpMaskFilter, ZeroCrossingImageFilter,
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = RescaleIntensityFilter::new(out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = IntensityWindowingFilter::new(window_min, window_max, out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Set pixels strictly below threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_below(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::below(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Set pixels strictly above threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_above(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::above(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::outside(lower, upper, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        // Rust SigmoidImageFilter uses (inflection=alpha_rust, width=beta_rust).
        // Python/SimpleITK convention: alpha=width, beta=inflection.
        // Map: inflection=beta (Python), width=alpha (Python).
        let filter = SigmoidImageFilter::new(beta, alpha, min_output, max_output);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = BinaryThresholdImageFilter::new(
            lower_threshold,
            upper_threshold,
            foreground,
            background,
        );
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Linearly blend two co-registered images.
///
/// out(x) = (1 - alpha) * a(x) + alpha * b(x)
///
/// alpha=0 returns a; alpha=1 returns b. Spatial metadata is taken from a.
/// Both images must have identical shapes.
///
/// ITK Parity: BlendImageFilter
#[pyfunction]
#[pyo3(signature = (a, b, alpha=0.5_f32))]
pub fn blend_images(py: Python<'_>, a: &PyImage, b: &PyImage, alpha: f32) -> RitkResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    py.allow_threads(|| {
        BlendImageFilter::new(alpha)
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Zero-mean, unit-variance intensity normalization.
///
/// Computes global mean μ and standard deviation σ, then outputs
/// `(I(x) − μ) / σ` for each voxel. Constant images (σ = 0) produce
/// all-zero output. Equivalent to ITK `NormalizeImageFilter`.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     Normalized PyImage with identical shape and spatial metadata.
#[pyfunction]
pub fn normalize_image(py: Python<'_>, image: &PyImage) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = NormalizeImageFilter::new();
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Unsharp mask sharpening filter.
///
/// Sharpens an image by amplifying the high-frequency residual between the
/// input and its Gaussian-blurred version.
///
/// Given input `I` and blurred version `B = Gaussian(I, sigma)`:
/// ```text
/// mask(p)   = I(p) - B(p)
/// output(p) = I(p) + amount * max(|mask(p)| - threshold, 0) * sign(mask(p))
/// ```
/// Output is clamped to `[min(I), max(I)]` when `clamp=True`.
///
/// Equivalent to ITK `UnsharpMaskingImageFilter` with identical parameter
/// semantics. ImageJ "Unsharp Mask" is the special case `threshold=0.0`.
///
/// Args:
///     image:     Input PyImage.
///     sigma:     Gaussian blur sigma in physical units (mm). Applied
///                isotropically to all three axes (default 1.0).
///     amount:    Sharpening strength in [0, ∞). Default 0.5.
///     threshold: Minimum absolute mask value to trigger sharpening.
///                Voxels with |mask| < threshold are left unchanged (default 0.0).
///     clamp:     If True, clamp output to original intensity range (default True).
///
/// Returns:
///     Sharpened PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0_f64, amount=0.5_f64, threshold=0.0_f64, clamp=true))]
pub fn unsharp_mask(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    amount: f64,
    threshold: f64,
    clamp: bool,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let clamp_policy = if clamp {
            ClampPolicy::ClampToInputRange
        } else {
            ClampPolicy::NoClamp
        };
        let filter = UnsharpMaskFilter::new(
            vec![GaussianSigma::new_unchecked(sigma); 3],
            amount,
            threshold,
            clamp_policy,
        );
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Detect zero crossings in a 3-D image.
///
/// Marks voxels where the image crosses zero (sign change in 6-connected
/// neighbourhood or exact zero). Typical use: detect edges from a
/// Laplacian-of-Gaussian image.
///
/// Equivalent to ITK `ZeroCrossingImageFilter`.
///
/// Args:
///     image:            Input PyImage (typically LoG output).
///     foreground_value: Output value at zero-crossing voxels (default 1.0).
///     background_value: Output value at non-crossing voxels (default 0.0).
///
/// Returns:
///     Binary-valued PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, foreground_value=1.0_f32, background_value=0.0_f32))]
pub fn zero_crossing_image(
    py: Python<'_>,
    image: &PyImage,
    foreground_value: f32,
    background_value: f32,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = ZeroCrossingImageFilter::new()
            .with_foreground(foreground_value)
            .with_background(background_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
