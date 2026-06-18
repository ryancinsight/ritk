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
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_filter::{
    FftShiftFilter, ForwardFftFilter, InverseFftFilter, RealToHalfHermitianForwardFftFilter,
};
use ritk_image::Image;
use std::sync::Arc;

/// Deinterleave a complex image `[D, H, 2W]` (real,imag pairs along X) into a
/// real `[D, H, W]` image by mapping each `(re, im)` pair through `f`.
fn complex_map(image: &PyImage, f: impl Fn(f32, f32) -> f32) -> RitkResult<PyImage> {
    let [d, h, w2] = image.inner.shape();
    if w2 % 2 != 0 {
        return Err(RitkPyError::value(format!(
            "complex op: last axis {w2} is odd; expected an interleaved [D,H,2W] complex image"
        )));
    }
    let w = w2 / 2;
    let data = image.inner.data_slice();
    let mut out = vec![0.0_f32; d * h * w];
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let re = data[z * h * w2 + y * w2 + 2 * x];
                let im = data[z * h * w2 + y * w2 + 2 * x + 1];
                out[z * h * w + y * w + x] = f(re, im);
            }
        }
    }
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(out, Shape::new([d, h, w])),
        &NdArrayDevice::default(),
    );
    Ok(into_py_image(Image::new(
        tensor,
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    )))
}

/// Build an interleaved complex image `[D, H, 2W]` from two real `[D, H, W]`
/// inputs `a`, `b`, mapping each pair through `f -> (re, im)`.
fn build_complex(
    a: &PyImage,
    b: &PyImage,
    f: impl Fn(f32, f32) -> (f32, f32),
) -> RitkResult<PyImage> {
    let [d, h, w] = a.inner.shape();
    if b.inner.shape() != [d, h, w] {
        return Err(RitkPyError::value(format!(
            "complex build: shapes differ ({:?} vs {:?})",
            [d, h, w],
            b.inner.shape()
        )));
    }
    let da = a.inner.data_slice();
    let db = b.inner.data_slice();
    let w2 = w * 2;
    let mut out = vec![0.0_f32; d * h * w2];
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let (re, im) = f(da[z * h * w + y * w + x], db[z * h * w + y * w + x]);
                out[z * h * w2 + y * w2 + 2 * x] = re;
                out[z * h * w2 + y * w2 + 2 * x + 1] = im;
            }
        }
    }
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(out, Shape::new([d, h, w2])),
        &NdArrayDevice::default(),
    );
    Ok(into_py_image(Image::new(
        tensor,
        *a.inner.origin(),
        *a.inner.spacing(),
        *a.inner.direction(),
    )))
}

/// Build a complex image from real and imaginary parts (interleaved `[D,H,2W]`).
/// ITK Parity: RealAndImaginaryToComplexImageFilter (`sitk.RealAndImaginaryToComplex`).
#[pyfunction]
pub fn real_and_imaginary_to_complex(real: &PyImage, imaginary: &PyImage) -> RitkResult<PyImage> {
    build_complex(real, imaginary, |re, im| (re, im))
}

/// Build a complex image from magnitude and phase: `re = m·cos(p)`, `im = m·sin(p)`.
/// ITK Parity: MagnitudeAndPhaseToComplexImageFilter (`sitk.MagnitudeAndPhaseToComplex`).
#[pyfunction]
pub fn magnitude_and_phase_to_complex(magnitude: &PyImage, phase: &PyImage) -> RitkResult<PyImage> {
    build_complex(magnitude, phase, |m, p| (m * p.cos(), m * p.sin()))
}

/// Real part of a complex image. ITK Parity: ComplexToRealImageFilter
/// (`sitk.ComplexToReal`).
#[pyfunction]
pub fn complex_to_real(image: &PyImage) -> RitkResult<PyImage> {
    complex_map(image, |re, _| re)
}

/// Imaginary part of a complex image. ITK Parity: ComplexToImaginaryImageFilter
/// (`sitk.ComplexToImaginary`).
#[pyfunction]
pub fn complex_to_imaginary(image: &PyImage) -> RitkResult<PyImage> {
    complex_map(image, |_, im| im)
}

/// Modulus (magnitude) `√(re²+im²)` of a complex image. ITK Parity:
/// ComplexToModulusImageFilter (`sitk.ComplexToModulus`).
#[pyfunction]
pub fn complex_to_modulus(image: &PyImage) -> RitkResult<PyImage> {
    complex_map(image, |re, im| (re * re + im * im).sqrt())
}

/// Phase `atan2(im, re)` of a complex image. ITK Parity: ComplexToPhaseImageFilter
/// (`sitk.ComplexToPhase`).
#[pyfunction]
pub fn complex_to_phase(image: &PyImage) -> RitkResult<PyImage> {
    complex_map(image, |re, im| im.atan2(re))
}

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

/// Real-to-half-Hermitian forward FFT, matching
/// `SimpleITK.RealToHalfHermitianForwardFFT`.
///
/// The DFT of a real image is Hermitian-symmetric, so only the first `W/2+1`
/// last-axis columns are independent. This returns that non-redundant half:
/// input `[D, H, W]` → output `[D, H, 2*(W/2+1)]` interleaved `(Re, Im)`. The
/// retained values equal `forward_fft`'s leading columns bitwise.
///
/// Args:
///     image: Input PyImage (real-valued, shape [D, H, W]).
///
/// Returns:
///     Half-Hermitian complex PyImage of shape [D, H, 2*(W/2+1)].
#[pyfunction]
pub fn real_to_half_hermitian_forward_fft(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = Arc::clone(&image.inner);
    py.allow_threads(|| {
        RealToHalfHermitianForwardFftFilter::new()
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
