//! FFT-based convolution filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, vec_to_image_like, PyImage};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_core::image::Image;
use ritk_filter::{FftConvolution3DFilter, FftConvolutionFilter};
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_spatial::{Direction, Point, Spacing};
use std::sync::Arc;

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
///     image: Input volume PyImage of shape [D, H, W].
///     kernel: Kernel PyImage of shape [1, KH, KW] (single Z-slice).
///
/// Returns:
///     Convolved PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if kernel has more than 1 Z-slice.
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
