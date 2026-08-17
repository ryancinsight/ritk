//! Python representation of fitted diffusion maps.

use numpy::{PyArray1, PyArray3, PyArray4, PyArrayMethods};
use pyo3::prelude::*;

use ritk_diffusion::maps::DiffusionMaps;

use crate::errors::{RitkPyError, RitkResult};

/// Fitted tensor field, with the scalar maps derived from it.
///
/// Returned by [`crate::diffusion::fit_tensor_maps`]. Every accessor produces
/// a fresh NumPy array shaped `[Z, Y, X]`, matching the volumes the fit was
/// given; the eigenvector field is `[Z, Y, X, 3]`.
///
/// Voxels that were not fitted read zero in every map. `mask` distinguishes
/// those from a voxel genuinely measured as isotropic — a distinction the
/// maps alone cannot express, since both are zero.
#[pyclass(name = "DiffusionMaps", module = "ritk.diffusion")]
pub struct PyDiffusionMaps {
    maps: DiffusionMaps,
    shape: [usize; 3],
}

impl PyDiffusionMaps {
    pub(crate) fn new(maps: DiffusionMaps, shape: [usize; 3]) -> Self {
        Self { maps, shape }
    }

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
