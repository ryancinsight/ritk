//! Frequency-domain filter functions (ideal and Butterworth pass/reject).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{FftFilterKind, FrequencyDomainFilter};
use std::sync::Arc;

/// Apply ideal low-pass filter in the frequency domain.
///
/// Performs forward FFT → shift → apply ideal low-pass mask → unshift → inverse FFT.
/// The mask is 1.0 for frequencies with normalised radius ≤ `cutoff` and 0.0 otherwise.
///
/// Args:
///     image: Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3))]
pub fn fft_ideal_low_pass(py: Python<'_>, image: &PyImage, cutoff: f64) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply(img.as_ref(), FftFilterKind::IdealLowPass, cutoff, 2)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply ideal high-pass filter in the frequency domain.
///
/// Performs forward FFT → shift → apply ideal high-pass mask → unshift → inverse FFT.
/// The mask is 1.0 for frequencies with normalised radius ≥ `cutoff` and 0.0 otherwise.
///
/// Args:
///     image: Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3))]
pub fn fft_ideal_high_pass(py: Python<'_>, image: &PyImage, cutoff: f64) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply(img.as_ref(), FftFilterKind::IdealHighPass, cutoff, 2)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply Butterworth low-pass filter in the frequency domain.
///
/// Performs forward FFT → shift → apply Butterworth low-pass mask → unshift → inverse FFT.
/// The transfer function is `H(r) = 1 / (1 + (r/cutoff)^(2*order))`.
///
/// Args:
///     image: Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///     order: Butterworth filter order (higher = sharper transition).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3, order = 2))]
pub fn fft_butterworth_low_pass(
    py: Python<'_>,
    image: &PyImage,
    cutoff: f64,
    order: usize,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply(
                img.as_ref(),
                FftFilterKind::ButterworthLowPass,
                cutoff,
                order,
            )
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply Butterworth high-pass filter in the frequency domain.
///
/// Performs forward FFT → shift → apply Butterworth high-pass mask → unshift → inverse FFT.
/// The transfer function is `H(r) = 1 - 1 / (1 + (r/cutoff)^(2*order))`.
///
/// Args:
///     image: Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///     order: Butterworth filter order (higher = sharper transition).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3, order = 2))]
pub fn fft_butterworth_high_pass(
    py: Python<'_>,
    image: &PyImage,
    cutoff: f64,
    order: usize,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply(
                img.as_ref(),
                FftFilterKind::ButterworthHighPass,
                cutoff,
                order,
            )
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
