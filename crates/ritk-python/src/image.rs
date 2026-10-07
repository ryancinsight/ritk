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

/// Return the native direction for NumPy's `[Z, Y, X]` storage convention.
///
/// Native image metadata stores direction columns in tensor-axis order, while
/// Python and SimpleITK expose physical coordinates in `(X, Y, Z)` order. The
/// permutation makes tensor axis 0 advance physical Z and tensor axis 2
/// advance physical X; [`PyImage::direction`] maps it back to the public
/// SimpleITK layout.
pub(crate) const fn numpy_array_direction() -> Direction<3> {
    Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
}

/// RITK geometry for a procedural source whose parameters are sitk `(x, y, z)`.
///
/// The core generators evaluate `p_d = origin_d + index_d·spacing_d` on the
/// sitk axes and return a `[z, y, x]` buffer, so the metadata must describe
/// *that* mapping:
///
/// - [`Point`] is a physical `(x, y, z)` position, so the origin transfers
///   unchanged. Reversing it claims the first voxel sits at
///   `(origin_z, origin_y, origin_x)`, contradicting the generated data.
/// - [`Spacing`] is axis-aligned `[Δdepth, Δrow, Δcol]`, so it is the reverse
///   of the sitk triple.
/// - The direction is NumPy's `[Z, Y, X]` permutation, making tensor axis 0
///   advance physical Z and axis 2 advance physical X — the same metadata
///   [`PyImage::new_from_numpy`] builds for a `[Z, Y, X]` array. An identity
///   direction instead makes physical X advance with the *depth* index, which
///   no choice of spacing or origin can repair.
pub(crate) fn source_geometry(
    origin_xyz: (f64, f64, f64),
    spacing_xyz: (f64, f64, f64),
) -> (Point<3>, Spacing<3>, Direction<3>) {
    (
        Point::new([origin_xyz.0, origin_xyz.1, origin_xyz.2]),
        Spacing::new([spacing_xyz.2, spacing_xyz.1, spacing_xyz.0]),
        numpy_array_direction(),
    )
}

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
            numpy_array_direction(),
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
        PyArray1::<f32>::from_vec(py, cow.into_owned())
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
    let values = image.data_cow_on(&SequentialBackend).into_owned();
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
    let m = PyModule::new(parent.py(), "image")?;
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

    #[test]
    fn numpy_array_direction_maps_tensor_axes_to_physical_axes() {
        let image = vec_to_image(
            vec![0.0; 8],
            [2, 2, 2],
            Point::origin(),
            Spacing::uniform(1.0),
            numpy_array_direction(),
        );

        assert_eq!(
            image.continuous_index_to_physical_point(&Point::new([1.0, 0.0, 0.0])),
            Point::new([0.0, 0.0, 1.0])
        );
        assert_eq!(
            image.continuous_index_to_physical_point(&Point::new([0.0, 0.0, 1.0])),
            Point::new([1.0, 0.0, 0.0])
        );
    }

    /// A source's metadata must reproduce the map its core generator used.
    ///
    /// The cores evaluate `p_d = origin_d + index_d·spacing_d` on the sitk axes
    /// and return a `[z, y, x]` buffer, so with sitk `origin = (1, 2, 3)` and
    /// `spacing = (0.5, 0.7, 0.9)` voxel `(kz, ky, kx)` sits at physical
    /// `(1 + 0.5·kx, 2 + 0.7·ky, 3 + 0.9·kz)`. Three defects break this and none
    /// is visible from the voxel values alone: an identity direction advances
    /// physical X with the *depth* index, a reversed origin puts the first
    /// voxel at `(3, 2, 1)`, and an unreversed spacing gives X the depth pitch.
    #[test]
    fn source_geometry_reproduces_the_core_index_to_physical_map() {
        let (origin, spacing, direction) = source_geometry((1.0, 2.0, 3.0), (0.5, 0.7, 0.9));
        let image = vec_to_image(vec![0.0; 2 * 3 * 4], [2, 3, 4], origin, spacing, direction);

        for kz in 0..2_usize {
            for ky in 0..3_usize {
                for kx in 0..4_usize {
                    let physical = image.continuous_index_to_physical_point(&Point::new([
                        kz as f64, ky as f64, kx as f64,
                    ]));
                    let label = format!("(kz={kz}, ky={ky}, kx={kx})");
                    assert!(
                        (physical[0] - (1.0 + 0.5 * kx as f64)).abs() < 1e-12,
                        "physical X at {label} is {}, expected {}",
                        physical[0],
                        1.0 + 0.5 * kx as f64
                    );
                    assert!(
                        (physical[1] - (2.0 + 0.7 * ky as f64)).abs() < 1e-12,
                        "physical Y at {label} is {}, expected {}",
                        physical[1],
                        2.0 + 0.7 * ky as f64
                    );
                    assert!(
                        (physical[2] - (3.0 + 0.9 * kz as f64)).abs() < 1e-12,
                        "physical Z at {label} is {}, expected {}",
                        physical[2],
                        3.0 + 0.9 * kz as f64
                    );
                }
            }
        }
    }
}
