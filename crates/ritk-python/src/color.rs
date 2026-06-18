//! Python-exposed multi-component (RGB/vector) color image and per-component
//! filters.
//!
//! `ColorImage` wraps `ritk_image::ColorVolume<Backend, 3>` (channel-interleaved
//! `[Z, Y, X, 3]`). Per-component filters reuse the scalar filter library via
//! `ritk_filter::map_color_components`, matching ITK's vector-image semantics
//! (each component filtered independently).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use numpy::{ndarray::Array4, IntoPyArray, PyArray4, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{
    map_color_components, GradientImageFilter, GradientRecursiveGaussianImageFilter,
    MeanImageFilter, MedianFilter, RecursiveGaussianFilter,
};
use ritk_image::{ColorVolume, Image};
use std::sync::Arc;

/// Build a scalar `Image` from a row-major `[z,y,x]` buffer and spatial metadata.
fn scalar_image(
    vals: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<Backend, 3> {
    let device = NdArrayDevice::default();
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
    Image::new(tensor, origin, spacing, direction)
}

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
        let tensor = Tensor::<Backend, 4>::from_data(
            TensorData::new(flat, Shape::new([z, y, x, ch])),
            &device,
        );
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

/// Per-component mean (box) filter on a color image.
/// ITK Parity: MeanImageFilter on a vector image.
#[pyfunction]
#[pyo3(signature = (image, radius = 1))]
pub fn color_mean(py: Python<'_>, image: &PyColorImage, radius: usize) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| {
            map_color_components(arc.as_ref(), |img| {
                MeanImageFilter::new(radius)
                    .apply(img)
                    .expect("mean filter is infallible on a valid scalar image")
            })
        })
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Compose three scalar images into a 3-component (RGB) color image.
///
/// ITK Parity: ComposeImageFilter (`sitk.Compose`).
#[pyfunction]
pub fn compose(c0: &PyImage, c1: &PyImage, c2: &PyImage) -> RitkResult<PyColorImage> {
    let dims = c0.inner.shape();
    let (d1, d2) = (c1.inner.shape(), c2.inner.shape());
    if dims != d1 || dims != d2 {
        return Err(RitkPyError::value(format!(
            "compose: component shapes differ ({dims:?} / {d1:?} / {d2:?})"
        )));
    }
    let b0 = c0.inner.data_slice().into_owned();
    let b1 = c1.inner.data_slice().into_owned();
    let b2 = c2.inner.data_slice().into_owned();
    let vol = ColorVolume::<Backend, 3>::from_component_buffers(
        &[b0, b1, b2],
        dims,
        *c0.inner.origin(),
        *c0.inner.spacing(),
        *c0.inner.direction(),
        &NdArrayDevice::default(),
    )
    .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(vol),
    })
}

/// Central-difference image gradient → 3-component vector image with components
/// in sitk axis order `(∂/∂x, ∂/∂y, ∂/∂z)`.
///
/// ITK Parity: GradientImageFilter (`sitk.Gradient`).
#[pyfunction]
#[pyo3(signature = (image, use_image_spacing=true))]
pub fn gradient(
    py: Python<'_>,
    image: &PyImage,
    use_image_spacing: bool,
) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| GradientImageFilter::new(use_image_spacing).apply(arc.as_ref()))
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Gaussian-smoothed image gradient → 3-component vector image with components
/// in sitk axis order `(∂/∂x, ∂/∂y, ∂/∂z)`.
///
/// ITK Parity: GradientRecursiveGaussianImageFilter (`sitk.GradientRecursiveGaussian`).
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0))]
pub fn gradient_recursive_gaussian(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| GradientRecursiveGaussianImageFilter::new(sigma).apply(arc.as_ref()))
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Extract one component (`index` ∈ [0, 2]) of a color image as a scalar image.
///
/// ITK Parity: VectorIndexSelectionCastImageFilter (`sitk.VectorIndexSelectionCast`).
#[pyfunction]
pub fn vector_index_selection_cast(image: &PyColorImage, index: usize) -> RitkResult<PyImage> {
    let [d, r, c, ch] = image.inner.shape();
    if index >= ch {
        return Err(RitkPyError::value(format!(
            "vector_index_selection_cast: index {index} out of range for {ch} components"
        )));
    }
    let comp = image.inner.into_component_buffers().swap_remove(index);
    Ok(into_py_image(scalar_image(
        comp,
        [d, r, c],
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    )))
}

/// Per-voxel Euclidean magnitude of a color image: `√(Σ_k component_k²)`.
///
/// ITK Parity: VectorMagnitudeImageFilter (`sitk.VectorMagnitude`).
#[pyfunction]
pub fn vector_magnitude(image: &PyColorImage) -> PyImage {
    let [d, r, c, _ch] = image.inner.shape();
    let comps = image.inner.into_component_buffers();
    let n = d * r * c;
    let mut mag = vec![0.0_f32; n];
    for buf in &comps {
        for (i, &v) in buf.iter().enumerate() {
            mag[i] += v * v;
        }
    }
    for m in &mut mag {
        *m = m.sqrt();
    }
    into_py_image(scalar_image(
        mag,
        [d, r, c],
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    ))
}

/// Per-component smoothing recursive (Deriche) Gaussian on a color image.
/// ITK Parity: SmoothingRecursiveGaussianImageFilter on a vector image.
#[pyfunction]
pub fn color_smoothing_recursive_gaussian(
    py: Python<'_>,
    image: &PyColorImage,
    sigma: f64,
) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| {
            map_color_components(arc.as_ref(), |img| {
                RecursiveGaussianFilter::new(sigma)
                    .apply(img)
                    .expect("recursive Gaussian is infallible on a valid scalar image")
            })
        })
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}
