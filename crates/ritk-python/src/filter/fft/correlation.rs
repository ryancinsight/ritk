//! FFT-based normalized cross-correlation filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, image_to_vec, into_py_image, vec_to_image_like, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::{
    normalized_correlation as core_normalized_correlation, FftNormalizedCorrelation3DFilter,
    FftNormalizedCorrelationFilter,
};
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

/// Apply FFT-based normalized cross-correlation to each Z-slice.
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

        let img_vals = image_to_vec(img.as_ref()).0;
        let t_vals = image_to_vec(tmpl.as_ref()).0;

        let t_slice: Vec<f32> = t_vals[..th * tw].to_vec();
        let template_2d = build_image_2d(t_slice, th, tw);
        let filter = FftNormalizedCorrelationFilter::<MoiraiBackend>::new(&template_2d)
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

/// Apply full 3-D FFT normalized cross-correlation of a volume with a 3-D template.
#[pyfunction]
pub fn fft_normalized_correlate_3d(
    py: Python<'_>,
    volume: &PyImage,
    template: &PyImage,
) -> RitkResult<PyImage> {
    let vol = image_from_py(volume);
    let tmpl = image_from_py(template);

    py.allow_threads(|| {
        let filter = FftNormalizedCorrelation3DFilter::<MoiraiBackend>::new(&tmpl)
            .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        filter
            .apply(&vol)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Spatial-domain normalized correlation of an image with a template, gated by a mask.
#[pyfunction]
pub fn normalized_correlation(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    template: &PyImage,
) -> RitkResult<PyImage> {
    let img = image_from_py(image);
    let msk = image_from_py(mask);
    let tpl = image_from_py(template);
    py.allow_threads(|| {
        core_normalized_correlation(&img, &msk, &tpl)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Masked FFT normalized cross-correlation (Padfield 2012).
#[pyfunction]
#[pyo3(signature = (fixed, moving, fixed_mask, moving_mask,
                    required_number_of_overlapping_pixels=0, required_fraction_of_overlapping_pixels=0.0_f32))]
pub fn masked_fft_normalized_correlation(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    fixed_mask: &PyImage,
    moving_mask: &PyImage,
    required_number_of_overlapping_pixels: u64,
    required_fraction_of_overlapping_pixels: f32,
) -> RitkResult<PyImage> {
    let f = image_from_py(fixed);
    let m = image_from_py(moving);
    let fm = image_from_py(fixed_mask);
    let mm = image_from_py(moving_mask);
    py.allow_threads(|| {
        ritk_filter::MaskedFftNormalizedCorrelationFilter {
            required_number_of_overlapping_pixels,
            required_fraction_of_overlapping_pixels,
        }
        .apply(&f, &m, &fm, &mm)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
