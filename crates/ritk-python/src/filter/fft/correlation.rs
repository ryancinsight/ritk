//! FFT-based normalized cross-correlation filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, vec_to_image_like, PyImage};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_core::image::Image;
use ritk_filter::{
    normalized_correlation as core_normalized_correlation, FftNormalizedCorrelation3DFilter,
    FftNormalizedCorrelationFilter,
};
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
///     image: Input volume PyImage of shape [D, H, W].
///     template: Template PyImage of shape [1, TH, TW] (single Z-slice).
///
/// Returns:
///     Cross-correlation map PyImage of shape [D, H, W].
///
/// Raises:
///     ValueError: if template has more than 1 Z-slice.
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
///     volume: Input volume PyImage of shape [D, H, W].
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

/// Spatial-domain normalized correlation of an image with a template, gated by a
/// mask. Matches `SimpleITK.NormalizedCorrelation`.
///
/// The template is normalized to mean zero and unit norm; each masked voxel gets
/// the correlation of its locally-centered neighbourhood with the unit template
/// (ZeroFluxNeumann boundary). Masked-out voxels are 0.
///
/// Args:
///     image:    Input PyImage.
///     mask:     Mask PyImage (same shape; non-zero selects the voxel).
///     template: Odd-sized template PyImage (the neighbourhood operator).
///
/// Returns:
///     Correlation PyImage, same shape and metadata as the input.
///
/// Raises:
///     RuntimeError: on a shape mismatch or an even template extent.
#[pyfunction]
pub fn normalized_correlation(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    template: &PyImage,
) -> RitkResult<PyImage> {
    let img = Arc::clone(&image.inner);
    let msk = Arc::clone(&mask.inner);
    let tpl = Arc::clone(&template.inner);
    py.allow_threads(|| {
        core_normalized_correlation(img.as_ref(), msk.as_ref(), tpl.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Masked FFT normalized cross-correlation (Padfield 2012), matching
/// `SimpleITK.MaskedFFTNormalizedCorrelation`.
///
/// Correlates `fixed`/`moving` over every translation using their masks, via a
/// handful of FFTs. Output extent is `fixed + moving − 1` per axis. Voxels with
/// overlap below `required_number`/`required_fraction·maxOverlap` (or zero
/// denominator) are 0.
///
/// Args:
///     fixed, moving:           input images.
///     fixed_mask, moving_mask: their masks (non-zero = valid).
///     required_number_of_overlapping_pixels: min overlap (default 0).
///     required_fraction_of_overlapping_pixels: min overlap fraction (default 0.0).
///
/// Returns:
///     NCC PyImage of extent fixed+moving-1.
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
    let f = Arc::clone(&fixed.inner);
    let m = Arc::clone(&moving.inner);
    let fm = Arc::clone(&fixed_mask.inner);
    let mm = Arc::clone(&moving_mask.inner);
    py.allow_threads(|| {
        ritk_filter::MaskedFftNormalizedCorrelationFilter {
            required_number_of_overlapping_pixels,
            required_fraction_of_overlapping_pixels,
        }
        .apply(f.as_ref(), m.as_ref(), fm.as_ref(), mm.as_ref())
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
