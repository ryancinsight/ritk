//! Python-exposed Image class wrapping native `ritk_image::native::Image`.

use crate::errors::{RitkPyError, RitkResult};
use coeus_core::{MoiraiBackend, SequentialBackend};
use numpy::{ndarray::Array3, IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use std::sync::Arc;

/// Native backend used throughout ritk-python scalar image bindings.
pub type Backend = MoiraiBackend;

/// Legacy Burn backend retained for burn-only filters and transforms.
pub type BurnBackend = burn_ndarray::NdArray<f32>;

/// Native 3-D scalar image carrier used by `PyImage`.
pub type ScalarImage = NativeImage<f32, MoiraiBackend, 3>;

/// Legacy Burn-backed scalar image carrier for adapters into burn-only crates.
pub type BurnImage<const D: usize> = ritk_core::image::Image<BurnBackend, D>;

/// Medical image with physical-space metadata.
#[pyclass(name = "Image")]
pub struct PyImage {
    pub inner: Arc<ScalarImage>,
}

#[pymethods]
impl PyImage {
    /// Construct a PyImage from a NumPy f32 array with shape [Z, Y, X].
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
        let flat: Vec<f32> = array.as_array().iter().copied().collect();
        let sp = spacing.unwrap_or([1.0, 1.0, 1.0]);
        let orig = origin.unwrap_or([0.0, 0.0, 0.0]);
        let image = NativeImage::from_flat_on(
            flat,
            [z, y, x],
            Point::new([orig[2], orig[1], orig[0]]),
            Spacing::new(sp),
            Direction::identity(),
            &MoiraiBackend,
        )
        .map_err(|e| RitkPyError::runtime(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(image),
        })
    }

    /// Convert image data to a NumPy f32 array with shape [Z, Y, X].
    fn to_numpy<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        let shape = self.inner.shape();
        let backend = MoiraiBackend;
        let cow = self.inner.data_cow_on(&backend);
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), cow.into_owned())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
            .map(|arr| arr.into_pyarray_bound(py))
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
        (o[2], o[1], o[0])
    }

    /// Direction cosine matrix as a row-major 9-tuple in SimpleITK order.
    #[getter]
    fn direction(&self) -> [f64; 9] {
        let d = self.inner.direction();
        let mut out = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[i * 3 + j] = d[(i, 2 - j)];
            }
        }
        out
    }

    fn __repr__(&self) -> String {
        let s = self.inner.shape();
        let sp = self.inner.spacing();
        let o = self.inner.origin();
        format!(
            "Image(shape=({},{},{}), spacing=({:.3},{:.3},{:.3}), origin=({:.3},{:.3},{:.3}))",
            s[0], s[1], s[2], sp[0], sp[1], sp[2], o[2], o[1], o[0],
        )
    }
}

pub trait IntoPyImage {
    fn into_py_image(self) -> PyImage;
}

impl IntoPyImage for ScalarImage {
    fn into_py_image(self) -> PyImage {
        PyImage {
            inner: Arc::new(self),
        }
    }
}

impl IntoPyImage for BurnImage<3> {
    fn into_py_image(self) -> PyImage {
        let (values, shape) = burn_image_to_vec(&self);
        vec_to_image(
            values,
            shape,
            *self.origin(),
            *self.spacing(),
            *self.direction(),
        )
        .into_py_image()
    }
}

/// Wrap either a native or Burn-backed image in `PyImage`.
pub fn into_py_image<I: IntoPyImage>(image: I) -> PyImage {
    image.into_py_image()
}

/// Convert a native image from `ritk-io` onto `MoiraiBackend`.
pub fn native_into_py_image(image: ritk_io::NativeImage) -> PyImage {
    let backend = SequentialBackend;
    let values = image.data_vec_on(&backend);
    let shape = image.shape();
    into_py_image(
        NativeImage::from_flat_on(
            values,
            shape,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            &MoiraiBackend,
        )
        .expect("native_into_py_image: known-valid shape"),
    )
}

/// Convert the current Python image container into the native image used by `ritk-io`.
pub fn py_image_to_native(image: &PyImage) -> RitkResult<ritk_io::NativeImage> {
    let (values, shape) = image_to_vec(image.inner.as_ref());
    ritk_io::NativeImage::from_flat(
        values,
        shape,
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    )
    .map_err(|e| RitkPyError::runtime(e.to_string()))
}

/// Convert a native Python image to a legacy Burn-backed image for burn-only algorithms.
pub fn py_image_to_burn(image: &PyImage) -> BurnImage<3> {
    let (values, shape) = image_to_vec(image.inner.as_ref());
    let device = <BurnBackend as ritk_image::tensor::backend::Backend>::Device::default();
    let tensor = ritk_image::tensor::Tensor::<BurnBackend, 3>::from_data(
        ritk_image::tensor::TensorData::new(values, ritk_image::tensor::Shape::new(shape)),
        &device,
    );
    BurnImage::new(
        tensor,
        *image.inner.origin(),
        *image.inner.spacing(),
        *image.inner.direction(),
    )
}

/// Extract logical row-major image data as `Vec<f32>` plus shape `[Z, Y, X]`.
pub fn image_to_vec(image: &ScalarImage) -> (Vec<f32>, [usize; 3]) {
    let shape = image.shape();
    let values = image.data_cow_on(&MoiraiBackend).into_owned();
    (values, shape)
}

/// Call `f` with a logical row-major slice view of a native image.
pub(crate) fn with_image_slice<R, F: FnOnce(&[f32]) -> R>(image: &ScalarImage, f: F) -> R {
    let cow = image.data_cow_on(&MoiraiBackend);
    f(cow.as_ref())
}

/// Wrap a legacy Burn-backed image result back into a `PyImage` by converting
/// via a flat `Vec<f32>` round-trip. Used as the return adapter for burn-only
/// filter algorithms that cannot yet operate on native Coeus images.
pub fn burn_into_py_image(image: BurnImage<3>) -> PyImage {
    let (values, shape) = burn_image_to_vec(&image);
    into_py_image(vec_to_image(
        values,
        shape,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    ))
}

pub(crate) fn burn_image_to_vec<const D: usize>(image: &BurnImage<D>) -> (Vec<f32>, [usize; D]) {
    let dims = image.shape();
    let values = image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("burn_image_to_vec: contiguous f32 image")
        .to_vec();
    (values, dims)
}

/// Construct a native image from a flat `Vec<f32>`, shape `[Z, Y, X]`,
/// and spatial metadata cloned from a reference image.
pub fn vec_to_image_like(
    values: Vec<f32>,
    shape: [usize; 3],
    reference: &ScalarImage,
) -> ScalarImage {
    vec_to_image(
        values,
        shape,
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
    )
}

/// Construct a native image from a flat `Vec<f32>` with explicit metadata.
pub fn vec_to_image(
    values: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> ScalarImage {
    NativeImage::from_flat_on(values, shape, origin, spacing, direction, &MoiraiBackend)
        .expect("vec_to_image: valid inputs")
}

/// Register the `image` submodule and its classes/functions into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "image")?;
    m.add_class::<PyImage>()?;
    m.add_class::<crate::color::PyColorImage>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
