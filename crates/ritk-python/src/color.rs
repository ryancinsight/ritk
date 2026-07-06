//! Python-exposed multi-component (RGB/vector) color image and per-component
//! filters.
//!
//! `ColorImage` wraps `ritk_image::ColorVolume<Backend, 3>` (channel-interleaved
//! `[Z, Y, X, 3]`). Per-component filters reuse the scalar filter library via
//! `ritk_filter::map_color_components`, matching ITK's vector-image semantics
//! (each component filtered independently).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use numpy::{ndarray::Array4, IntoPyArray, PyArray4, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{
    map_color_components, physical_point_image_source as core_physical_point_image_source,
    Colormap, GradientImageFilter, GradientRecursiveGaussianImageFilter,
    LabelMapContourOverlayFilter, LabelOverlayFilter, LabelToRGBFilter, MeanImageFilter,
    MedianFilter, RecursiveGaussianFilter, ScalarToRGBColormapFilter,
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

/// Generate a physical-point vector image: each voxel holds its own physical
/// coordinate `(origin_d + index_d·spacing_d)` as a 3-component vector (sitk
/// `(x, y, z)` component order). ITK Parity: PhysicalPointImageSource
/// (`sitk.PhysicalPointSource`).
#[pyfunction]
#[pyo3(signature = (size, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)))]
pub fn physical_point_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    origin: (f64, f64, f64),
    spacing: (f64, f64, f64),
) -> RitkResult<PyColorImage> {
    let ([cx, cy, cz], dims) = py.allow_threads(|| {
        core_physical_point_image_source(
            [size.0, size.1, size.2],
            [origin.0, origin.1, origin.2],
            [spacing.0, spacing.1, spacing.2],
        )
    });
    let vol = ColorVolume::<Backend, 3>::from_component_buffers(
        &[cx, cy, cz],
        dims,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
        &NdArrayDevice::default(),
    )
    .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(vol),
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

/// Map a scalar image to a 3-component RGB image via a colormap (channel values
/// in `[0, 255]`). Only the linear LUTs `grey`/`red`/`green`/`blue` are
/// supported; perceptual maps (hot/jet/…) raise an error.
///
/// ITK Parity: ScalarToRGBColormapImageFilter (`sitk.ScalarToRGBColormap`).
#[pyfunction]
#[pyo3(signature = (image, colormap="grey"))]
pub fn scalar_to_rgb_colormap(
    py: Python<'_>,
    image: &PyImage,
    colormap: &str,
) -> RitkResult<PyColorImage> {
    let cmap = Colormap::from_name(colormap).map_err(|e| RitkPyError::value(e.to_string()))?;
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| ScalarToRGBColormapFilter::new(cmap).apply(arc.as_ref()))
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Map a label image to RGB using ITK's default 30-colour label table
/// (background voxels → black). ITK Parity: LabelToRGBImageFilter
/// (`sitk.LabelToRGB`).
#[pyfunction]
#[pyo3(signature = (image, background=0))]
pub fn label_to_rgb(py: Python<'_>, image: &PyImage, background: i64) -> RitkResult<PyColorImage> {
    let arc = Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| LabelToRGBFilter::new(background).apply(arc.as_ref()))
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Overlay a `label` image on a grayscale `image` as RGB, alpha-blending each
/// label with ITK's colour table at `opacity` (background passes grayscale
/// through). ITK Parity: LabelOverlayImageFilter (`sitk.LabelOverlay`).
#[pyfunction]
#[pyo3(signature = (image, label, opacity=0.5, background=0))]
pub fn label_overlay(
    py: Python<'_>,
    image: &PyImage,
    label: &PyImage,
    opacity: f64,
    background: i64,
) -> RitkResult<PyColorImage> {
    let img = Arc::clone(&image.inner);
    let lab = Arc::clone(&label.inner);
    let out = py
        .allow_threads(|| {
            LabelOverlayFilter::new(opacity, background).apply(img.as_ref(), lab.as_ref())
        })
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
    Ok(PyColorImage {
        inner: Arc::new(out),
    })
}

/// Overlay the contours of a `label` image on a grayscale `image` as RGB,
/// alpha-blending each label's contour band with ITK's colour table at
/// `opacity`. ITK Parity: LabelMapContourOverlayImageFilter
/// (`sitk.LabelMapContourOverlay` with default geometry: dilation radius 1,
/// contour thickness 1, CONTOUR, HIGH_LABEL_ON_TOP).
#[pyfunction]
#[pyo3(signature = (image, label, opacity=0.5, background=0))]
pub fn label_map_contour_overlay(
    py: Python<'_>,
    image: &PyImage,
    label: &PyImage,
    opacity: f64,
    background: i64,
) -> RitkResult<PyColorImage> {
    let img = Arc::clone(&image.inner);
    let lab = Arc::clone(&label.inner);
    let out = py
        .allow_threads(|| {
            LabelMapContourOverlayFilter::new(opacity, background).apply(img.as_ref(), lab.as_ref())
        })
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
    let mag = vector_magnitude_buffer(&image.inner.into_component_buffers(), d * r * c);
    into_py_image(scalar_image(
        mag,
        [d, r, c],
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    ))
}

/// Per-voxel Euclidean magnitude `sqrt(Σ_k c_k²)` of the channel buffers.
fn vector_magnitude_buffer(comps: &[Vec<f32>], n: usize) -> Vec<f32> {
    let mut mag = vec![0.0_f32; n];
    for buf in comps {
        for (i, &v) in buf.iter().enumerate() {
            mag[i] += v * v;
        }
    }
    for m in &mut mag {
        *m = m.sqrt();
    }
    mag
}

/// Edge potential `exp(−|vector|)` of a vector (gradient) image: small where the
/// gradient is large (edges), near 1 in flat regions.
///
/// ITK Parity: EdgePotentialImageFilter (`sitk.EdgePotential`).
#[pyfunction]
pub fn edge_potential(image: &PyColorImage) -> PyImage {
    let [d, r, c, _ch] = image.inner.shape();
    let mut mag = vector_magnitude_buffer(&image.inner.into_component_buffers(), d * r * c);
    for m in &mut mag {
        *m = (-*m).exp();
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
