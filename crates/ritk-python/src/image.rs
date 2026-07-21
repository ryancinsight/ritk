//! Python-exposed Image class wrapping native `ritk_image::Image`.

use crate::array_utils::copy_array3_to_vec;
use crate::errors::{RitkPyError, RitkResult};
use coeus_core::{MoiraiBackend, SequentialBackend};
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::Image as NativeImage;
use std::sync::Arc;

/// Native backend used throughout ritk-python scalar image bindings.
pub type Backend = MoiraiBackend;

/// Native 3-D scalar image carrier used by `PyImage`.
pub type ScalarImage = NativeImage<f32, MoiraiBackend, 3>;

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
        let flat: Vec<f32> = copy_array3_to_vec(&array)
            .map_err(|e| RitkPyError::value(format!("failed to read input array: {e}")))?;
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
        PyArray1::<f32>::from_vec_bound(py, cow.into_owned())
            .reshape([shape[0], shape[1], shape[2]])
            .map_err(|e| RitkPyError::runtime(e.to_string()))
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

/// Wrap a native image in `PyImage`.
pub fn into_py_image<I: IntoPyImage>(image: I) -> PyImage {
    image.into_py_image()
}

/// Convert a native image from `ritk-io` onto `MoiraiBackend`.
pub fn native_into_py_image(image: ritk_io::NativeImage) -> PyImage {
    let values = image.data_vec_on(&SequentialBackend);
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

/// Clone the native image owned by the Python carrier.
pub fn image_from_py(image: &PyImage) -> ScalarImage {
    image.inner.as_ref().clone()
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

/// Call `f` with logical row-major views of two native images.
pub(crate) fn with_image_pair_slices<R, F: FnOnce(&[f32], &[f32]) -> R>(
    first: &ScalarImage,
    second: &ScalarImage,
    f: F,
) -> R {
    with_image_slice(first, |first_values| {
        with_image_slice(second, |second_values| f(first_values, second_values))
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_image_pair_preserves_borrowed_storage() {
        let image = vec_to_image(
            vec![1.0, 2.0, 3.0, 4.0],
            [1, 2, 2],
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let storage = image
            .data_slice()
            .expect("contiguous image must expose its storage");

        with_image_pair_slices(&image, &image, |first, second| {
            assert_eq!(first, storage);
            assert_eq!(second, storage);
            assert_eq!(first.as_ptr(), storage.as_ptr());
            assert_eq!(second.as_ptr(), storage.as_ptr());
        });
    }
}
