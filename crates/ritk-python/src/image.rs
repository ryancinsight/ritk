//! Python-exposed Image class wrapping `ritk_core::Image<NdArray<f32>, 3>`.
//!
//! # Invariants
//! - Shape convention: [Z, Y, X] (ritk-core canonical order).
//! - Spacing and origin are in physical units (millimetres by default).
//! - Direction is a 3×3 rotation matrix (identity by default).
//! - The inner `Arc<Image<Backend, 3>>` allows cheap clone across Python objects.

use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use numpy::{ndarray::Array3, IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::{
    image::Image,
    spatial::{Direction, Point, Spacing},
};
use std::sync::Arc;

/// Concrete backend used throughout ritk-python.
pub type Backend = NdArray<f32>;

// ── PyImage ───────────────────────────────────────────────────────────────────

/// Medical image with physical-space metadata.
///
/// Wraps `ritk_core::Image<NdArray<f32>, 3>` (always CPU, f32 precision).
/// Shape convention: [Z, Y, X].
///
/// # Example (Python)
/// ```python
/// import numpy as np
/// import ritk
///
/// arr = np.zeros((64, 64, 64), dtype=np.float32)
/// img = ritk.Image(arr, spacing=(0.5, 0.5, 1.0), origin=(0.0, 0.0, 0.0))
/// print(img.shape)    # (64, 64, 64)
/// print(img.spacing)  # (0.5, 0.5, 1.0)
/// out = img.to_numpy()
/// assert out.shape == (64, 64, 64)
/// ```
#[pyclass(name = "Image")]
pub struct PyImage {
    pub inner: Arc<Image<Backend, 3>>,
}

#[pymethods]
impl PyImage {
    /// Construct a PyImage from a NumPy f32 array with shape [Z, Y, X].
    ///
    /// Args:
    ///     array:   f32 ndarray with shape [Z, Y, X].
    ///     spacing: Physical voxel size (sz, sy, sx) in mm.
    ///              Defaults to (1.0, 1.0, 1.0).
    ///     origin:  Physical coordinate of first voxel (oz, oy, ox) in mm.
    ///              Defaults to (0.0, 0.0, 0.0).
    #[new]
    #[pyo3(signature = (array, spacing=None, origin=None))]
    fn new_from_numpy<'py>(
        _py: Python<'py>,
        array: PyReadonlyArray3<'py, f32>,
        spacing: Option<[f64; 3]>,
        origin: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let shape = array.shape();
        let (z, y, x) = (shape[0], shape[1], shape[2]);

        // Iterate in array order (handles non-contiguous layouts correctly).
        let flat: Vec<f32> = array.as_array().iter().copied().collect();

        let device = NdArrayDevice::default();
        let td = TensorData::new(flat, Shape::new([z, y, x]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);

        let sp = spacing.unwrap_or([1.0, 1.0, 1.0]);
        let orig = origin.unwrap_or([0.0, 0.0, 0.0]);

        let image = Image::new(
            tensor,
            Point::new(orig),
            Spacing::new(sp),
            Direction::identity(),
        );
        Ok(Self {
            inner: Arc::new(image),
        })
    }

    /// Convert image data to a NumPy f32 array with shape [Z, Y, X].
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let values: Vec<f32> = self
            .inner
            .data()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

        let shape = self.inner.shape();
        let arr = Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(arr.into_pyarray_bound(py))
    }

    /// Image shape as (Z, Y, X).
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.inner.shape();
        (s[0], s[1], s[2])
    }

    /// Physical voxel size as (sz, sy, sx) in mm.
    #[getter]
    fn spacing(&self) -> (f64, f64, f64) {
        let sp = self.inner.spacing();
        (sp[0], sp[1], sp[2])
    }

    /// Physical coordinate of first voxel as (oz, oy, ox) in mm.
    #[getter]
    fn origin(&self) -> (f64, f64, f64) {
        let o = self.inner.origin();
        (o[0], o[1], o[2])
    }

    fn __repr__(&self) -> String {
        let s = self.inner.shape();
        let sp = self.inner.spacing();
        let o = self.inner.origin();
        format!(
            "Image(shape=({},{},{}), spacing=({:.3},{:.3},{:.3}), origin=({:.3},{:.3},{:.3}))",
            s[0], s[1], s[2], sp[0], sp[1], sp[2], o[0], o[1], o[2],
        )
    }
}

// ── Conversion helpers (used by io, filter, segmentation, registration) ────────

/// Wrap a `ritk_core::Image<Backend, 3>` in a `PyImage`.
pub fn into_py_image(image: Image<Backend, 3>) -> PyImage {
    PyImage {
        inner: Arc::new(image),
    }
}

/// Extract tensor data as `Vec<f32>` plus shape `[Z, Y, X]`.
///
/// # Errors
/// Returns `PyRuntimeError` if the tensor dtype is not f32.
pub fn image_to_vec(image: &Image<Backend, 3>) -> PyResult<(Vec<f32>, [usize; 3])> {
    let shape = image.shape();
    let values: Vec<f32> = image
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    Ok((values, shape))
}

/// Construct an `Image<Backend, 3>` from a flat `Vec<f32>`, shape `[Z, Y, X]`,
/// and spatial metadata cloned from a reference image.
pub fn vec_to_image_like(
    values: Vec<f32>,
    shape: [usize; 3],
    reference: &Image<Backend, 3>,
) -> Image<Backend, 3> {
    let device = NdArrayDevice::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        reference.origin().clone(),
        reference.spacing().clone(),
        reference.direction().clone(),
    )
}

/// Construct an `Image<Backend, 3>` from a flat `Vec<f32>` with explicit metadata.
pub fn vec_to_image(
    values: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<Backend, 3> {
    let device = NdArrayDevice::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(tensor, origin, spacing, direction)
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `image` submodule and its classes/functions into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "image")?;
    m.add_class::<PyImage>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
