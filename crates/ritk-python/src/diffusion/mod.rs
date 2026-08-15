//! Python-exposed diffusion tensor fitting.
//!
//! A thin layer over [`ritk_diffusion::maps`]: it converts arguments, releases
//! the GIL around the fit, and maps errors to Python exceptions. The estimator,
//! the background mask and the physical rejection bounds all live in the Rust
//! crate, so this module and the `ritk dwi` CLI compute identically.

use numpy::{PyArray1, PyArray3, PyArray4, PyArrayMethods};
use pyo3::prelude::*;

use ritk_diffusion::maps::{fit_diffusion_maps, DiffusionMaps, DiffusionMapsConfig};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, PyImage};

/// Fitted tensor field, with the scalar maps derived from it.
///
/// Returned by [`fit_tensor_maps`]. Every accessor produces a fresh NumPy array
/// shaped `[Z, Y, X]`, matching the volumes the fit was given; the eigenvector
/// field is `[Z, Y, X, 3]`.
///
/// Voxels that were not fitted read zero in every map. `mask` distinguishes
/// those from a voxel genuinely measured as isotropic — a distinction the maps
/// alone cannot express, since both are zero.
#[pyclass(name = "DiffusionMaps", module = "ritk.diffusion")]
pub struct PyDiffusionMaps {
    maps: DiffusionMaps,
    shape: [usize; 3],
}

impl PyDiffusionMaps {
    /// Reshape a per-voxel scalar map into a `[Z, Y, X]` NumPy array.
    fn scalar<'py>(
        &self,
        py: Python<'py>,
        values: &[f64],
    ) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "maps are returned at image precision, matching every other RITK array"
        )]
        let narrowed: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        PyArray1::<f32>::from_vec_bound(py, narrowed)
            .reshape(self.shape)
            .map_err(|error| RitkPyError::runtime(format!("reshaping a map: {error}")))
    }
}

#[pymethods]
impl PyDiffusionMaps {
    /// Number of voxels in the volume.
    fn __len__(&self) -> usize {
        self.maps.len()
    }

    /// Count of voxels that yielded a physically admissible tensor.
    #[getter]
    fn fitted_count(&self) -> usize {
        self.maps.fitted_count()
    }

    /// Which voxels were fitted, as a `[Z, Y, X]` boolean array.
    ///
    /// Returns:
    ///     numpy.ndarray: bool array, True where a tensor was fitted.
    fn mask<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<bool>>> {
        PyArray1::<bool>::from_vec_bound(py, self.maps.mask().to_vec())
            .reshape(self.shape)
            .map_err(|error| RitkPyError::runtime(format!("reshaping the mask: {error}")))
    }

    /// Fractional anisotropy, in `[0, 1]`.
    ///
    /// Returns:
    ///     numpy.ndarray: float32 array shaped [Z, Y, X].
    fn fractional_anisotropy<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        self.scalar(py, &self.maps.fractional_anisotropy())
    }

    /// Mean diffusivity, in mm²/s.
    ///
    /// Returns:
    ///     numpy.ndarray: float32 array shaped [Z, Y, X].
    fn mean_diffusivity<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        self.scalar(py, &self.maps.mean_diffusivity())
    }

    /// Axial diffusivity `λ₁`, in mm²/s.
    ///
    /// Returns:
    ///     numpy.ndarray: float32 array shaped [Z, Y, X].
    fn axial_diffusivity<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        self.scalar(py, &self.maps.axial_diffusivity())
    }

    /// Radial diffusivity `(λ₂ + λ₃) / 2`, in mm²/s.
    ///
    /// Returns:
    ///     numpy.ndarray: float32 array shaped [Z, Y, X].
    fn radial_diffusivity<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        self.scalar(py, &self.maps.radial_diffusivity())
    }

    /// Principal eigenvector per voxel — the local fibre orientation.
    ///
    /// Unit length wherever a tensor was fitted, exactly zero elsewhere. The
    /// vector carries no sign: `v` and `−v` describe the same fibre.
    ///
    /// Returns:
    ///     numpy.ndarray: float32 array shaped [Z, Y, X, 3].
    fn principal_eigenvector<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray4<f32>>> {
        let [depth, rows, columns] = self.shape;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "unit-vector components are returned at image precision"
        )]
        let flat: Vec<f32> = self
            .maps
            .principal_eigenvector()
            .iter()
            .flat_map(|vector| vector.iter().map(|value| *value as f32))
            .collect();

        PyArray1::<f32>::from_vec_bound(py, flat)
            .reshape([depth, rows, columns, 3])
            .map_err(|error| {
                RitkPyError::runtime(format!("reshaping the eigenvector field: {error}"))
            })
    }
}

/// Fit one diffusion tensor per voxel and derive its scalar maps.
///
/// Delegates to `ritk_diffusion::maps::fit_diffusion_maps`, so the result is
/// identical to `ritk dwi tensor` on the same inputs.
///
/// The GIL is released for the duration of the fit, which is the expensive part
/// — a whole-brain volume is hundreds of thousands of least-squares solves.
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

    // The fit touches no Python objects, so the interpreter is free to run other
    // threads for what is by far the longest part of the call.
    let maps = py
        .allow_threads(|| {
            let borrowed: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
            fit_diffusion_maps(&scheme, &borrowed, &config)
        })
        .map_err(|error| RitkPyError::value(format!("fitting the tensor field: {error}")))?;

    Ok(PyDiffusionMaps { maps, shape })
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

/// Register the `diffusion` submodule into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "diffusion")?;
    m.add_class::<PyDiffusionMaps>()?;
    m.add_function(wrap_pyfunction!(fit_tensor_maps, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
