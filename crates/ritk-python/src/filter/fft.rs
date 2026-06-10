//! Python-exposed FFT filter functions.
//!
//! All filters delegate to `ritk_core::filter::fft` (SSOT).
//!
//! # Complex image convention
//!
//! The `forward_fft` function returns a PyImage whose last spatial axis is
//! doubled: for a [D, H, W] input the output is [D, H, 2*W]. The interleaved
//! layout is `data[d, h, 2*w] = Re(F[d,h,w])`, `data[d, h, 2*w+1] = Im(F[d,h,w])`.
//!
//! `inverse_fft` accepts this [D, H, 2*W] format and returns [D, H, W].
//! `fft_shift` operates on a complex image (same shape in, same shape out).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, vec_to_image_like, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_core::filter::{
    FftConvolution3DFilter, FftConvolutionFilter, FftFilterKind, FftNormalizedCorrelation3DFilter,
    FftNormalizedCorrelationFilter, FftShiftFilter, ForwardFftFilter, FrequencyDomainFilter,
    InverseFftFilter,
};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::sync::Arc;

/// Apply the forward FFT to a 3-D medical image.
///
/// Transforms a real-valued image from the spatial domain to the frequency
/// domain. The output has the same Z and Y dimensions as the input, but the
/// X dimension is doubled to hold interleaved real and imaginary parts:
///
///   output shape [D, H, W] → [D, H, 2*W]
///   output[d, h, 2*x]   = Re(F[d, h, x])
///   output[d, h, 2*x+1] = Im(F[d, h, x])
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
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ForwardFftFilter::new()
            .apply_3d(image.as_ref())
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
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        InverseFftFilter::new()
            .apply_3d(image.as_ref())
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
            .apply_3d(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

// ── Slice-based 2-D FFT convolution and NCC ───────────────────────────────────

/// Build a 2-D `Image<Backend, 2>` from a flat slice of f32 values.
fn build_image_2d(vals: Vec<f32>, rows: usize, cols: usize) -> Image<crate::image::Backend, 2> {
    let device = NdArrayDevice::default();
    let td = TensorData::new(vals, Shape::new([rows, cols]));
    let tensor = Tensor::<crate::image::Backend, 2>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
}

/// Apply FFT-based 2-D convolution to each Z-slice of a 3-D volume.
///
/// Convolves each Z-slice of the input volume with the provided kernel using
/// the FFT convolution theorem (O(N log N) per slice). The kernel must be a
/// single-slice image (shape [1, KH, KW]). The output has the "same" spatial
/// shape as the input (matching the convention of ITK's
/// `itk::FFTConvolutionImageFilter`).
///
/// Args:
///     image:  Input volume PyImage of shape [D, H, W].
///     kernel: Kernel PyImage of shape [1, KH, KW] (single Z-slice).
///
/// Returns:
///     Convolved PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:   if kernel has more than 1 Z-slice.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
pub fn fft_convolve(py: Python<'_>, image: &PyImage, kernel: &PyImage) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let kern = Arc::clone(&kernel.inner);

    py.allow_threads(|| {
        let kernel_shape = kern.shape();
        if kernel_shape[0] != 1 {
            return Err(RitkPyError::value(format!(
                "fft_convolve: kernel must have exactly 1 Z-slice, got {}",
                kernel_shape[0]
            )));
        }

        let [d, h, w] = img.shape();
        let kh = kernel_shape[1];
        let kw = kernel_shape[2];

        // Extract raw voxel data.
        let img_vals: Vec<f32> = img.data_slice().into_owned();
        let k_vals: Vec<f32> = kern.data_slice().into_owned();

        // Build 2-D kernel image from the first (only) Z-slice of the kernel.
        let k_slice: Vec<f32> = k_vals[..kh * kw].to_vec();
        let kernel_2d = build_image_2d(k_slice, kh, kw);
        let filter = FftConvolutionFilter::<crate::image::Backend>::new(&kernel_2d)
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;

        // Process each Z-slice independently.
        let mut out_vals = Vec::with_capacity(d * h * w);
        for z in 0..d {
            let start = z * h * w;
            let slice_vals: Vec<f32> = img_vals[start..start + h * w].to_vec();
            let slice_img = build_image_2d(slice_vals, h, w);
            let result = filter
                .apply(&slice_img)
                .map_err(|e| RitkPyError::runtime(e.to_string()))?;
            let result_vals: Vec<f32> = result.data_slice().into_owned();
            out_vals.extend(result_vals);
        }

        Ok(into_py_image(vec_to_image_like(out_vals, [d, h, w], &img)))
    })
}

/// Apply full 3-D FFT convolution of a volume with a 3-D kernel.
///
/// Convolves the entire 3-D volume with the 3-D kernel using a full separable
/// 3-D FFT (not slice-by-slice). This correctly models 3-D blur and volumetric
/// filtering where the kernel varies along all three axes.
///
/// The kernel is a 3-D PyImage of shape [KD, KH, KW]. The volume is convolved
/// independently along all three spatial dimensions via the convolution theorem.
/// Output has the same spatial shape as the input ("same" convention).
///
/// Args:
///     volume: Input volume PyImage of shape [D, H, W].
///     kernel: Kernel PyImage of shape [KD, KH, KW].
///
/// Returns:
///     Convolved PyImage of shape [D, H, W].
///
/// Raises:
///     RuntimeError: on internal FFT failure.
#[pyfunction]
pub fn fft_convolve_3d(py: Python<'_>, volume: &PyImage, kernel: &PyImage) -> RitkResult<PyImage> {
    let vol = Arc::clone(&volume.inner);
    let kern = Arc::clone(&kernel.inner);

    py.allow_threads(|| {
        let filter = FftConvolution3DFilter::<crate::image::Backend>::new(kern.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        filter
            .apply(vol.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply FFT-based normalized cross-correlation to each Z-slice.
///
/// For each Z-slice of the input volume, computes the normalized cross-
/// correlation (NCC) against the provided template. The template must be a
/// single-slice image (shape [1, TH, TW]).
///
/// The template is mean-subtracted: `T̂ = T − mean(T)`. The output is
/// partially normalized (divided by the L₂ norm of `T̂`). Full normalization
/// by local image patch energy is not performed.
///
/// Args:
///     image:    Input volume PyImage of shape [D, H, W].
///     template: Template PyImage of shape [1, TH, TW] (single Z-slice).
///
/// Returns:
///     Cross-correlation map PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:   if template has more than 1 Z-slice.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
pub fn fft_normalized_correlate(
    py: Python<'_>,
    image: &PyImage,
    template: &PyImage,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let tmpl = Arc::clone(&template.inner);

    py.allow_threads(|| {
        let tmpl_shape = tmpl.shape();
        if tmpl_shape[0] != 1 {
            return Err(RitkPyError::value(format!(
                "fft_normalized_correlate: template must have exactly 1 Z-slice, got {}",
                tmpl_shape[0]
            )));
        }

        let [d, h, w] = img.shape();
        let th = tmpl_shape[1];
        let tw = tmpl_shape[2];

        let img_vals: Vec<f32> = img.data_slice().into_owned();
        let t_vals: Vec<f32> = tmpl.data_slice().into_owned();

        // Build 2-D template image from the first (only) Z-slice.
        let t_slice: Vec<f32> = t_vals[..th * tw].to_vec();
        let template_2d = build_image_2d(t_slice, th, tw);
        let filter = FftNormalizedCorrelationFilter::<crate::image::Backend>::new(&template_2d)
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;

        // Process each Z-slice independently.
        let mut out_vals = Vec::with_capacity(d * h * w);
        for z in 0..d {
            let start = z * h * w;
            let slice_vals: Vec<f32> = img_vals[start..start + h * w].to_vec();
            let slice_img = build_image_2d(slice_vals, h, w);
            let result = filter
                .apply(&slice_img)
                .map_err(|e| RitkPyError::runtime(e.to_string()))?;
            let result_vals: Vec<f32> = result.data_slice().into_owned();
            out_vals.extend(result_vals);
        }

        Ok(into_py_image(vec_to_image_like(out_vals, [d, h, w], &img)))
    })
}

/// Apply ideal low-pass filter in the frequency domain.
///
/// Performs forward FFT → shift → apply ideal low-pass mask → unshift → inverse FFT.
/// The mask is 1.0 for frequencies with normalised radius ≤ `cutoff` and 0.0 otherwise.
///
/// Args:
///     image:  Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:  if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3))]
pub fn fft_ideal_low_pass(py: Python<'_>, image: &PyImage, cutoff: f64) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply_3d(img.as_ref(), FftFilterKind::IdealLowPass, cutoff, 2)
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
///     image:  Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:  if cutoff ≤ 0 or cutoff > 0.5.
///     RuntimeError: on internal FFT failure.
#[pyfunction]
#[pyo3(signature = (image, cutoff = 0.3))]
pub fn fft_ideal_high_pass(py: Python<'_>, image: &PyImage, cutoff: f64) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    py.allow_threads(|| {
        FrequencyDomainFilter::new()
            .apply_3d(img.as_ref(), FftFilterKind::IdealHighPass, cutoff, 2)
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
///     image:  Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///     order:  Butterworth filter order (higher = sharper transition).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:  if cutoff ≤ 0 or cutoff > 0.5.
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
            .apply_3d(
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
///     image:  Input PyImage of shape [D, H, W].
///     cutoff: Normalised cutoff frequency in (0, 0.5] (0.5 = Nyquist).
///     order:  Butterworth filter order (higher = sharper transition).
///
/// Returns:
///     Filtered PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError:  if cutoff ≤ 0 or cutoff > 0.5.
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
            .apply_3d(
                img.as_ref(),
                FftFilterKind::ButterworthHighPass,
                cutoff,
                order,
            )
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply full 3-D FFT normalized cross-correlation of a volume with a 3-D template.
///
/// Computes the normalized cross-correlation (NCC) between the entire 3-D volume
/// and the 3-D template using a full separable 3-D FFT (not slice-by-slice).
/// This correctly models 3-D template matching where the template varies along
/// all three spatial dimensions.
///
/// The template is mean-subtracted: `T̂ = T − mean(T)`. The output is partially
/// normalized (divided by the L₂ norm of `T̂`). Full normalization by local
/// volume patch energy is not performed.
///
/// The output has the same spatial shape as the input, with zero-offset (no
/// centring — unlike convolution, cross-correlation is not centred).
///
/// Args:
///     volume:   Input volume PyImage of shape [D, H, W].
///     template: Template PyImage of shape [TD, TH, TW].
///
/// Returns:
///     Cross-correlation map PyImage of shape [D, H, W].
///
/// Raises:
///     RuntimeError: on internal FFT failure.
#[pyfunction]
pub fn fft_normalized_correlate_3d(
    py: Python<'_>,
    volume: &PyImage,
    template: &PyImage,
) -> RitkResult<PyImage> {
    let vol = Arc::clone(&volume.inner);
    let tmpl = Arc::clone(&template.inner);

    py.allow_threads(|| {
        let filter = FftNormalizedCorrelation3DFilter::<crate::image::Backend>::new(tmpl.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        filter
            .apply(vol.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
