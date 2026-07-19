//! FFT-based convolution filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, image_to_vec, into_py_image, vec_to_image_like, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::{FftConvolution3DFilter, FftConvolutionFilter};
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::sync::Arc;

/// Build a 2-D `NativeImage<f32, MoiraiBackend, 2>` from a flat slice of f32 values.
fn build_image_2d(vals: Vec<f32>, rows: usize, cols: usize) -> NativeImage<f32, MoiraiBackend, 2> {
    NativeImage::from_flat_on(
        vals,
        [rows, cols],
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
        &MoiraiBackend,
    )
    .expect("build_image_2d: valid shape and data")
}

/// Apply FFT-based 2-D convolution to each Z-slice of a 3-D volume.
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

        let img_vals = image_to_vec(img.as_ref()).0;
        let k_vals = image_to_vec(kern.as_ref()).0;

        let k_slice: Vec<f32> = k_vals[..kh * kw].to_vec();
        let kernel_2d = build_image_2d(k_slice, kh, kw);
        let filter = FftConvolutionFilter::<MoiraiBackend>::new(&kernel_2d)
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;

        let mut out_vals = Vec::with_capacity(d * h * w);
        for z in 0..d {
            let start = z * h * w;
            let slice_vals: Vec<f32> = img_vals[start..start + h * w].to_vec();
            let slice_img = build_image_2d(slice_vals, h, w);
            let result = filter
                .apply(&slice_img)
                .map_err(|e| RitkPyError::runtime(e.to_string()))?;
            let result_vals: Vec<f32> = result.data_vec_on(&MoiraiBackend);
            out_vals.extend(result_vals);
        }

        Ok(into_py_image(vec_to_image_like(out_vals, [d, h, w], &img)))
    })
}

/// Apply full 3-D FFT convolution of a volume with a 3-D kernel.
#[pyfunction]
pub fn fft_convolve_3d(py: Python<'_>, volume: &PyImage, kernel: &PyImage) -> RitkResult<PyImage> {
    let vol = image_from_py(volume);
    let kern = image_from_py(kernel);

    py.allow_threads(|| {
        let filter = FftConvolution3DFilter::<MoiraiBackend>::new(&kern)
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        filter
            .apply(&vol)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
