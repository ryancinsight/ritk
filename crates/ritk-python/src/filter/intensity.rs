//! Intensity transform filters: rescale, windowing, thresholds, sigmoid, binary threshold, blend,
//! normalize, unsharp mask, and zero crossing.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, into_py_image, py_image_to_burn, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_filter::{
    AdaptiveHistogramEqualizationFilter, BinaryThresholdImageFilter, BitwiseNotImageFilter,
    BlendImageFilter, ClampPolicy, DoubleThresholdImageFilter, IntensityWindowingFilter,
    NormalizeImageFilter, NormalizeToConstantImageFilter, RescaleIntensityFilter,
    ShiftScaleImageFilter, SigmoidImageFilter, ThresholdImageFilter, UnsharpMaskFilter,
    ZeroCrossingImageFilter,
};
use std::sync::Arc;

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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = RescaleIntensityFilter::new(out_min, out_max);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = IntensityWindowingFilter::new(window_min, window_max, out_min, out_max);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::below(threshold, outside_value);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::above(threshold, outside_value);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = ThresholdImageFilter::outside(lower, upper, outside_value);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        // Rust SigmoidImageFilter uses (inflection=alpha_rust, width=beta_rust).
        // Python/SimpleITK convention: alpha=width, beta=inflection.
        // Map: inflection=beta (Python), width=alpha (Python).
        let filter = SigmoidImageFilter::new(beta, alpha, min_output, max_output);
        filter
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        BinaryThresholdImageFilter::new(lower_threshold, upper_threshold, foreground, background)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Double-threshold (hysteresis): a voxel is `inside_value` if it is in the inner
/// band `[t2,t3]`, or in the outer band `[t1,t4]` and connected through it to an
/// inner-band voxel. ITK Parity: DoubleThresholdImageFilter (`sitk.DoubleThreshold`).
#[pyfunction]
#[pyo3(signature = (image, threshold1=0.0, threshold2=1.0, threshold3=254.0, threshold4=255.0, inside_value=1.0, outside_value=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn double_threshold(
    py: Python<'_>,
    image: &PyImage,
    threshold1: f32,
    threshold2: f32,
    threshold3: f32,
    threshold4: f32,
    inside_value: f32,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        DoubleThresholdImageFilter::new(
            threshold1,
            threshold2,
            threshold3,
            threshold4,
            inside_value,
            outside_value,
        )
        .apply_native(native.as_ref(), &backend)
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
    let a_native = Arc::clone(&a.inner);
    let b_native = Arc::clone(&b.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        BlendImageFilter::new(alpha)
            .apply_native(a_native.as_ref(), b_native.as_ref(), &backend)
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
pub fn normalize_image(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = NormalizeImageFilter::new();
        filter
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Scale the image so the sum of all voxels equals `constant`
/// (`out = in · constant / Σin`). ITK Parity: NormalizeToConstantImageFilter
/// (`sitk.NormalizeToConstant`).
#[pyfunction]
#[pyo3(signature = (image, constant=1.0))]
pub fn normalize_to_constant(
    py: Python<'_>,
    image: &PyImage,
    constant: f64,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        NormalizeToConstantImageFilter::new(constant)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
/// With `clamp=False` (the default) the result matches SimpleITK's
/// `UnsharpMask` (which only clamps to the *output pixel type's* range — a no-op
/// for ritk's f32). `clamp=True` additionally clamps the output to the input
/// value range `[min(I), max(I)]`, the ImageJ "Unsharp Mask" behaviour.
///
/// Args:
///     image:     Input PyImage.
///     sigma:     Gaussian blur sigma in physical units (mm). Applied
///                isotropically to all three axes (default 1.0).
///     amount:    Sharpening strength in [0, ∞). Default 0.5.
///     threshold: Minimum absolute mask value to trigger sharpening.
///                Voxels with |mask| < threshold are left unchanged (default 0.0).
///     clamp:     If True, clamp output to the input value range (ImageJ style).
///                Default False — matches SimpleITK's `UnsharpMask`.
///
/// Returns:
///     Sharpened PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0_f64, amount=0.5_f64, threshold=0.0_f64, clamp=false))]
pub fn unsharp_mask(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    amount: f64,
    threshold: f64,
    clamp: bool,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
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
            .apply_native(native.as_ref(), &backend)
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
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        let filter = ZeroCrossingImageFilter::new()
            .with_foreground(foreground_value)
            .with_background(background_value);
        filter
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Stark adaptive (local) histogram equalization, matching
/// `SimpleITK.AdaptiveHistogramEqualization`.
///
/// `alpha`/`beta` interpolate between classic adaptive equalization (`alpha=0`)
/// and unsharp masking; `radius` is the per-axis box-window radius `[z, y, x]`.
///
/// Args:
///     image:  Input scalar PyImage.
///     radius: Box-window radius `(rz, ry, rx)` (default (5, 5, 5)).
///     alpha:  Equalization exponent (default 0.3).
///     beta:   Unsharp/identity blend (default 0.3).
///
/// Returns:
///     Equalized PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, radius=(5, 5, 5), alpha=0.3_f64, beta=0.3_f64))]
pub fn adaptive_histogram_equalization(
    py: Python<'_>,
    image: &PyImage,
    radius: (usize, usize, usize),
    alpha: f64,
    beta: f64,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        AdaptiveHistogramEqualizationFilter {
            radius: [radius.0, radius.1, radius.2],
            alpha,
            beta,
        }
        .apply_native(native.as_ref(), &backend)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Bitwise complement of an integer-valued image, matching
/// `SimpleITK.BitwiseNot` for the corresponding pixel type.
///
/// `~x = (2**bits - 1) - x` for unsigned, `~x = -x - 1` for signed (two's
/// complement). ritk's f32 backend carries no integer type, so pass the width.
///
/// Args:
///     image:  Integer-valued PyImage (values rounded to nearest integer).
///     bits:   Bit width for the unsigned complement (default 8).
///     signed: Two's-complement signed interpretation (default False).
///
/// Returns:
///     Complemented PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, bits=8, signed=false))]
pub fn bitwise_not(
    py: Python<'_>,
    image: &PyImage,
    bits: u32,
    signed: bool,
) -> RitkResult<PyImage> {
    // TODO: BitwiseNotImageFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    let filter = if signed {
        BitwiseNotImageFilter::signed()
    } else {
        BitwiseNotImageFilter::unsigned(bits)
    };
    let out = py.allow_threads(|| filter.apply(&arc));
    Ok(burn_into_py_image(out))
}

/// Apply a linear shift-then-scale to every voxel.
///
/// `out(x) = (in(x) + shift) * scale`
///
/// Matches `SimpleITK.ShiftScale` (`sitk.ShiftScale`). The only divergence
/// from ITK is f32 rounding of the multiply-add; max absolute error < 1.0
/// on typical medical images (single-precision rounding, no accumulation).
///
/// Args:
///     image: Input PyImage.
///     shift: Added to each voxel before multiplication (default 0.0).
///     scale: Multiplied after shift (default 1.0).
///
/// Returns:
///     Transformed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, shift = 0.0_f32, scale = 1.0_f32))]
pub fn shift_scale(py: Python<'_>, image: &PyImage, shift: f32, scale: f32) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        ShiftScaleImageFilter::new(shift, scale)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
