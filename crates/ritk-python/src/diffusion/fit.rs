//! Diffusion tensor fitting and input conversion for the Python boundary.

use pyo3::prelude::*;

use ritk_diffusion::maps::{fit_diffusion_maps, DiffusionMapsConfig};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};

use crate::diffusion::maps::PyDiffusionMaps;
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, PyImage};

/// Fit one diffusion tensor per voxel and derive its scalar maps.
///
/// Delegates to `ritk_diffusion::maps::fit_diffusion_maps`, so the result is
/// identical to `ritk dwi tensor` on the same inputs.
///
/// The GIL is released for the duration of the fit, which is the expensive
/// part — a whole-brain volume is hundreds of thousands of least-squares
/// solves.
///
/// Args:
///     volumes: Diffusion-weighted volumes in acquisition order, one Image per
///         b-value. Every volume must share one grid.
///     bvals: One b-value per volume, in s/mm².
///     bvecs: One gradient direction per volume, each a 3-sequence in
///         image-axis order. A b = 0 volume takes any direction, conventionally
///         (0, 0, 0).
///     background_fraction: Voxels whose b = 0 signal falls below this fraction
///         of the reference volume's upper percentile are not fitted. Outside
///         the head the signal is noise and a tensor fitted to noise is
///         strongly anisotropic, so an unmasked map is dominated by a bright rim
///         tracing the skull. Pass 0.0 to fit every voxel.
///
/// Returns:
///     DiffusionMaps: the fitted field and its derived maps.
///
/// Raises:
///     ValueError: if the volume, b-value and direction counts disagree, the
///         volumes do not share a grid, the scheme has no b = 0 reference, or
///         `background_fraction` is not a usable number.
#[pyfunction]
#[pyo3(signature = (volumes, bvals, bvecs, background_fraction=None))]
pub fn fit_tensor_maps(
    py: Python<'_>,
    volumes: Vec<PyRef<'_, PyImage>>,
    bvals: Vec<f64>,
    bvecs: Vec<[f64; 3]>,
    background_fraction: Option<f64>,
) -> RitkResult<PyDiffusionMaps> {
    if volumes.len() != bvals.len() || volumes.len() != bvecs.len() {
        return Err(RitkPyError::value(format!(
            "the series has {} volumes but {} b-values and {} directions were given",
            volumes.len(),
            bvals.len(),
            bvecs.len()
        )));
    }

    let scheme = scheme_from(&bvals, &bvecs)?;

    let mut shape = None;
    let mut data = Vec::with_capacity(volumes.len());
    for (index, volume) in volumes.iter().enumerate() {
        let (values, dims) = image_to_vec(volume.inner.as_ref());
        match shape {
            None => shape = Some(dims),
            Some(expected) if expected != dims => {
                return Err(RitkPyError::value(format!(
                    "volume {index} has shape {dims:?} but volume 0 has {expected:?}; \
                     a series shares one grid"
                )));
            }
            Some(_) => {}
        }
        data.push(values);
    }
    let shape = shape.ok_or_else(|| RitkPyError::value("a series needs at least one volume"))?;

    let config = DiffusionMapsConfig {
        background_fraction: background_fraction
            .unwrap_or_else(|| DiffusionMapsConfig::default().background_fraction),
        ..DiffusionMapsConfig::default()
    };

    // The fit touches no Python objects, so the interpreter is free to run
    // other threads for what is by far the longest part of the call.
    let maps = py
        .allow_threads(|| {
            let borrowed: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
            fit_diffusion_maps(&scheme, &borrowed, &config)
        })
        .map_err(|error| RitkPyError::value(format!("fitting the tensor field: {error}")))?;

    Ok(PyDiffusionMaps::new(maps, shape))
}

/// Build a gradient scheme from paired b-values and directions.
fn scheme_from(bvals: &[f64], bvecs: &[[f64; 3]]) -> RitkResult<GradientScheme> {
    let mut entries = Vec::with_capacity(bvals.len());
    for (index, (bval, bvec)) in bvals.iter().zip(bvecs).enumerate() {
        let weighting =
            DiffusionWeighting::from_seconds_per_square_millimeter(*bval).map_err(|error| {
                RitkPyError::value(format!("b-value {index} ({bval}) is not usable: {error}"))
            })?;
        entries.push(
            GradientDirection::new(weighting, ritk_spatial::Vector::new(*bvec)).map_err(
                |error| {
                    RitkPyError::value(format!(
                        "direction {index} ({bvec:?}) is not usable: {error}"
                    ))
                },
            )?,
        );
    }
    GradientScheme::new(entries, GradientFrame::ImageAxis)
        .map_err(|error| RitkPyError::value(format!("building the gradient scheme: {error}")))
}
