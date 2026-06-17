//! Python-exposed multi-component (RGB/vector) color image and per-component
//! filters.
//!
//! `ColorImage` wraps `ritk_image::ColorVolume<Backend, 3>` (channel-interleaved
//! `[Z, Y, X, 3]`). Per-component filters reuse the scalar filter library via
//! `ritk_filter::map_color_components`, matching ITK's vector-image semantics
//! (each component filtered independently).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use numpy::{ndarray::Array4, IntoPyArray, PyArray4, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{map_color_components, MedianFilter};
use ritk_image::ColorVolume;
use std::sync::Arc;

type Rgb = ColorVolume<Backend, 3>;

/// A 3-component (RGB) color image, channel-interleaved as `[Z, Y, X, 3]`.
#[pyclass(name = "ColorImage")]
pub struct PyColorImage {
    pub inner: Arc<Rgb>,
}

#[pymethods]
impl PyColorImage {
    /// Construct from an f32 NumPy array of shape `[Z, Y, X, 3]`.
    #[new]
    #[pyo3(signature = (array, spacing=None, origin=None))]
    fn new_from_numpy(
        array: PyReadonlyArray4<'_, f32>,
        spacing: Option<[f64; 3]>,
        origin: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let shape = array.shape();
        let (z, y, x, ch) = (shape[0], shape[1], shape[2], shape[3]);
        if ch != 3 {
            return Err(RitkPyError::value(format!(
                "ColorImage requires a [Z, Y, X, 3] array; got channel axis = {ch}"
            ))
            .into());
        }
        let flat: Vec<f32> = array.as_array().iter().copied().collect();
        let device = NdArrayDevice::default();
        let tensor =
            Tensor::<Backend, 4>::from_data(TensorData::new(flat, Shape::new([z, y, x, ch])), &device);
        let sp = spacing.unwrap_or([1.0, 1.0, 1.0]);
        let orig = origin.unwrap_or([0.0, 0.0, 0.0]);
        let vol = ColorVolume::try_new(
            tensor,
            Point::new([orig[2], orig[1], orig[0]]),
            Spacing::new(sp),
            Direction::identity(),
        )
        .map_err(|e| RitkPyError::value(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(vol),
        })
    }

    /// Convert to an f32 NumPy array of shape `[Z, Y, X, 3]`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray4<f32>>> {
        let [z, y, x, ch] = self.inner.shape();
        let vals = self.inner.data_vec();
        Array4::from_shape_vec((z, y, x, ch), vals)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
            .map(|a| a.into_pyarray_bound(py))
    }

    /// Shape as `(Z, Y, X, C)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let s = self.inner.shape();
        (s[0], s[1], s[2], s[3])
    }
}

/// Per-component median filter on a color image (box neighbourhood, radius
/// `radius`). ITK Parity: MedianImageFilter on a vector image.
#[pyfunction]
#[pyo3(signature = (image, radius = 1))]
pub fn color_median(
    py: Python<'_>,
    image: &PyColorImage,
    radius: usize,
) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| {
            map_color_components(arc.as_ref(), |img| {
                MedianFilter::new(radius)
                    .apply(img)
                    .expect("median filter is infallible on a valid scalar image")
            })
        })
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}
