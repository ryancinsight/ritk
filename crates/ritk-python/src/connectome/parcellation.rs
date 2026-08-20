//! Python representation of a label volume.

use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ritk_parcellation::{Parcellation, ParcellationGrid, RegionStatistics};
use ritk_spatial::Point;

use crate::errors::{RitkPyError, RitkResult};

/// A labelled anatomical volume.
///
/// Every voxel carries the identifier of the region it belongs to, with `0`
/// meaning background. Construct one from a `[Z, Y, X]` integer array and the
/// geometry the volume was acquired on.
#[pyclass(name = "Parcellation", module = "ritk.connectome")]
pub struct PyParcellation {
    inner: Parcellation,
    /// Array shape as Python sees it, `[Z, Y, X]`.
    shape: [usize; 3],
}

impl PyParcellation {
    pub(crate) const fn inner(&self) -> &Parcellation {
        &self.inner
    }
}

#[pymethods]
impl PyParcellation {
    /// Build a parcellation from a label volume and its geometry.
    ///
    /// Args:
    ///     labels: `[Z, Y, X]` array of region identifiers. Zero is background.
    ///     spacing: Voxel size in mm, outermost axis first — the same order an
    ///         Image reports, so `spacing[0]` is the slice thickness.
    ///     origin: Physical position of voxel (0, 0, 0)'s centre, in mm.
    ///     direction: Row-major 3x3 direction cosines, in the same axis order as
    ///         `spacing`. Defaults to the identity. Supplying the volume's real
    ///         direction matters: an oblique volume read as axis-aligned returns
    ///         the label of a different region without failing.
    ///     region_names: Optional `(label, name)` pairs.
    ///
    /// Returns:
    ///     Parcellation
    ///
    /// Raises:
    ///     ValueError: if the geometry cannot describe a volume, or every voxel
    ///         is background — a parcellation with no regions cannot answer any
    ///         question asked of it.
    #[new]
    #[pyo3(signature = (labels, spacing, origin, direction=None, region_names=None))]
    fn new(
        labels: PyReadonlyArray3<'_, u32>,
        spacing: [f64; 3],
        origin: [f64; 3],
        direction: Option<[f64; 9]>,
        region_names: Option<Vec<(u32, String)>>,
    ) -> RitkResult<Self> {
        let view = labels.as_array();
        let shape = [view.shape()[0], view.shape()[1], view.shape()[2]];
        let flat: Vec<u32> = view.iter().copied().collect();

        const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let grid = ParcellationGrid::from_image_order(
            shape,
            spacing,
            origin,
            direction.unwrap_or(IDENTITY),
        )
        .map_err(RitkPyError::value)?;

        let inner = Parcellation::new(
            flat.into_boxed_slice(),
            grid,
            region_names.unwrap_or_default(),
        )
        .map_err(RitkPyError::value)?;

        Ok(Self { inner, shape })
    }

    /// Number of voxels in the volume.
    fn __len__(&self) -> usize {
        self.inner.labels().len()
    }

    /// Number of distinct non-background regions.
    #[getter]
    fn region_count(&self) -> usize {
        self.inner.region_count()
    }

    /// Sorted non-background region labels.
    #[getter]
    fn region_labels(&self) -> Vec<u32> {
        self.inner.region_labels()
    }

    /// Array shape, `[Z, Y, X]`.
    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// The label volume, as a `[Z, Y, X]` array.
    fn labels<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<u32>>> {
        PyArray1::<u32>::from_slice_bound(py, self.inner.labels())
            .reshape(self.shape)
            .map_err(|error| RitkPyError::runtime(format!("reshaping the labels: {error}")))
    }

    /// Human-readable name of a region, or None when none was supplied.
    fn name_of(&self, label: u32) -> Option<String> {
        self.inner.name_of(label).map(ToOwned::to_owned)
    }

    /// Region label at a physical point.
    ///
    /// Args:
    ///     point: `(x, y, z)` in mm.
    ///
    /// Returns:
    ///     int | None: the label, or None when the point falls outside the
    ///     volume. A point inside the volume but on an unlabelled voxel returns
    ///     0 — outside the field of view and unassigned within it are different
    ///     answers, so they are not collapsed.
    fn label_at(&self, point: [f64; 3]) -> Option<u32> {
        self.inner.label_at(&Point::new(point))
    }

    /// Per-region size, position, and extent.
    ///
    /// Returns:
    ///     list[dict]: one entry per region, ordered by label, each with
    ///     `label`, `voxel_count`, `volume_mm3`, `centroid` (physical `(x, y, z)`),
    ///     and `extent` (index-space `[nx, ny, nz]` of its bounding box).
    ///
    ///     Region volume is what normalises a connectome: a larger region
    ///     attracts more streamline endpoints for reasons of geometry rather
    ///     than anatomy.
    fn region_statistics<'py>(&self, py: Python<'py>) -> RitkResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .region_statistics()
            .into_iter()
            .map(|statistics| statistics_to_dict(py, &statistics))
            .collect()
    }

    fn __repr__(&self) -> String {
        let [z, y, x] = self.shape;
        format!(
            "Parcellation(shape=[{z}, {y}, {x}], regions={})",
            self.inner.region_count()
        )
    }
}

fn statistics_to_dict<'py>(
    py: Python<'py>,
    statistics: &RegionStatistics,
) -> RitkResult<Bound<'py, PyDict>> {
    let entry = PyDict::new_bound(py);
    let map = |error: PyErr| RitkPyError::runtime(format!("building a statistics entry: {error}"));
    entry.set_item("label", statistics.label()).map_err(map)?;
    entry
        .set_item("voxel_count", statistics.voxel_count())
        .map_err(map)?;
    entry
        .set_item("volume_mm3", statistics.volume())
        .map_err(map)?;
    entry
        .set_item("centroid", statistics.centroid().to_array())
        .map_err(map)?;
    entry.set_item("extent", statistics.extent()).map_err(map)?;
    Ok(entry)
}
