//! Python-exposed FFT filter functions.
//!
//! All filters delegate to `ritk_filter::fft` (SSOT).
//!
//! # Complex image convention
//!
//! The `forward_fft` function returns a PyImage whose last spatial axis is
//! doubled: for a [D, H, W] input the output is [D, H, 2*W]. The interleaved
//! layout is `data[d, h, 2*w] = Re(F[d,h,w])`, `data[d, h, 2*w+1] = Im(F[d,h,w])`.
//!
//! `inverse_fft` accepts this [D, H, 2*W] format and returns [D, H, W].
//! `fft_shift` operates on a complex image (same shape in, same shape out).

mod convolution;
mod correlation;
mod frequency;

pub use convolution::{fft_convolve, fft_convolve_3d};
pub use correlation::{fft_normalized_correlate, fft_normalized_correlate_3d};
pub use frequency::{
    fft_butterworth_high_pass, fft_butterworth_low_pass, fft_ideal_high_pass, fft_ideal_low_pass,
};

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{FftShiftFilter, ForwardFftFilter, InverseFftFilter};
use std::sync::Arc;

/// Apply the forward FFT to a 3-D medical image.
///
/// Transforms a real-valued image from the spatial domain to the frequency
/// domain. The output has the same Z and Y dimensions as the input, but the
/// X dimension is doubled to hold interleaved real and imaginary parts:
///
/// output shape [D, H, W] → [D, H, 2*W]
/// output[d, h, 2*x] = Re(F[d, h, x])
/// output[d, h, 2*x+1] = Im(F[d, h, x])
///
/// No normalization is applied in the forward direction (ITK convention).
/// Use `inverse_fft` (which normalizes by 1/N) to recover the original image.
///
/// Args:
///     image: Input PyImage (real-valued, shape [D, H, W]).
///
/// Returns:
///     Complex PyImage of shape [D, H, 2*W].
///
/// Raises:
///     RuntimeError: on internal FFT failure.
#[pyfunction]
pub fn forward_fft(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = Arc::clone(&image.inner);
    py.allow_threads(|| {
        ForwardFftFilter::new()
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply the inverse FFT to a complex frequency-domain image.
///
/// Transforms a complex image (produced by `forward_fft`) back to the spatial
/// domain. Normalizes by 1/N where N = D * H * W (product of real spatial dims).
///
/// The input must have shape [D, H, 2*W] (complex encoding); the output is
/// [D, H, W] (real spatial domain).
///
/// Args:
///     image: Complex PyImage of shape [D, H, 2*W].
///
/// Returns:
///     Real PyImage of shape [D, H, W].
///
/// Raises:
///     RuntimeError: on internal IFFT failure.
#[pyfunction]
pub fn inverse_fft(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = Arc::clone(&image.inner);
    py.allow_threads(|| {
        InverseFftFilter::new()
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply FFT shift to a complex frequency-domain image.
///
/// Rearranges the quadrants so that the zero-frequency (DC) component moves
/// from the corners to the centre of the array. This is a self-inverse
/// operation: applying it twice recovers the original image.
///
/// The input must have shape [D, H, 2*W] (complex encoding produced by
/// `forward_fft`). The output has the same shape.
///
/// Args:
///     image: Complex PyImage of shape [D, H, 2*W].
///
/// Returns:
///     Shifted complex PyImage of shape [D, H, 2*W].
///
/// Raises:
///     RuntimeError: on internal shift failure.
#[pyfunction]
pub fn fft_shift(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FftShiftFilter::new()
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
