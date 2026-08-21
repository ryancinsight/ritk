//! Atlas-driven parcellation: register labelled atlases onto a subject and fuse
//! their labels into one parcellation of that subject.
//!
//! `ritk.registration` already exposed the two fusion rules, but only over
//! atlases a caller had somehow already warped into subject space. This module
//! exposes the pipeline that does the warping, so a parcellation can be
//! produced from Python rather than only consumed there.
//!
//! Registration, warping, and fusion all stay in
//! [`ritk_registration::parcellation`]; this module converts between images and
//! that function's flat volumes, and hands the result back as the
//! `ritk.connectome.Parcellation` the connectome builder already accepts.

use numpy::{PyArray1, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use ritk_parcellation::storage::label_from_stored;
use ritk_registration::atlas::label_fusion::LabelFusionConfig;
use ritk_registration::{
    parcellate_with_atlas_set, AtlasParcellationConfig, LabelFusion, LabelledAtlas,
};

use crate::connectome::PyParcellation;
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, PyImage};

/// Accepted values of the `fusion` argument, for the error message.
const FUSION_CHOICES: &str = "\"majority\" or \"joint\"";

/// A subject parcellation and the evidence behind it.
///
/// Returned rather than a bare label volume because a parcellation without its
/// agreement is not interpretable: the labels are equally confident-looking
/// wherever the atlases split, which is exactly at the parcel boundaries where
/// streamline endpoints land.
#[pyclass(name = "AtlasParcellationResult", module = "ritk.registration")]
pub struct PyAtlasParcellationResult {
    parcellation: Py<PyParcellation>,
    agreement: Vec<f32>,
    shape: [usize; 3],
    registration_quality: Vec<f64>,
}

#[pymethods]
impl PyAtlasParcellationResult {
    /// The subject's parcellation, on the subject's own grid.
    #[getter]
    fn parcellation(&self, py: Python<'_>) -> Py<PyParcellation> {
        self.parcellation.clone_ref(py)
    }

    /// Per-voxel agreement in `[0, 1]`, as a `[Z, Y, X]` array.
    ///
    /// For majority voting this is the fraction of atlases that voted for the
    /// winning label; for joint label fusion it is the summed weight behind it.
    /// A single atlas has nothing to disagree with, so every voxel reads 1,
    /// which is a statement about the method and not about the anatomy.
    #[getter]
    fn agreement<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray3<f32>>> {
        PyArray1::<f32>::from_slice_bound(py, &self.agreement)
            .reshape(self.shape)
            .map_err(|error| RitkPyError::runtime(format!("reshaping the agreement: {error}")))
    }

    /// Final cross-correlation of each atlas's registration, in input order.
    ///
    /// A value far below the others marks an atlas that did not register, whose
    /// labels are then noise in the vote rather than a second opinion.
    #[getter]
    fn registration_quality(&self) -> Vec<f64> {
        self.registration_quality.clone()
    }
}

/// Parcellate a subject by deforming labelled atlases onto it.
///
/// Each atlas is registered to the subject independently, its labels warped
/// through the recovered deformation, and the results fused.
///
/// Args:
///     subject: Intensity image to parcellate. Supplies both the intensity the
///         registration matches and the grid the result lands on.
///     atlas_intensities: Atlas intensity images, one per atlas.
///     atlas_labels: Atlas label volumes, in the same order. Labels are read
///         from the stored floats by rounding, with anything at or below zero
///         treated as background.
///     fusion: "majority" for the label most atlases agree on, ties going to
///         the smaller label; "joint" to weight each atlas by how well its
///         warped intensities match the subject locally. Defaults to
///         "majority".
///     iterations: Registration iterations per resolution level, coarsest
///         first. The list length sets the number of levels. Defaults to the
///         three-level ANTs-style schedule.
///     patch_radius: Patch radius for joint label fusion, ignored otherwise.
///     beta: Regularisation of the joint-fusion similarity matrix, ignored
///         otherwise.
///     region_names: Optional (label, name) pairs. Atlases fused together
///         necessarily share one label scheme, so the names describe all of
///         them and are carried onto the result.
///
/// Returns:
///     AtlasParcellationResult
///
/// Raises:
///     ValueError: if the atlas lists differ in length, are empty, if `fusion`
///         is not a recognised rule, or if an atlas does not lie on the
///         subject's grid. Every atlas must already be on that grid; a
///         registration recovers a deformation and never a resampling, so a
///         mismatched atlas is rejected rather than resampled silently.
///     RuntimeError: if a registration or the fusion fails.
#[pyfunction]
#[pyo3(signature = (
    subject,
    atlas_intensities,
    atlas_labels,
    fusion = "majority",
    iterations = None,
    patch_radius = 2,
    beta = 0.1,
    region_names = None,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is an independent axis of the pipeline, and \
              collapsing them into an options class would hide the two that \
              only apply to one fusion rule"
)]
pub fn parcellate_with_atlases(
    py: Python<'_>,
    subject: &PyImage,
    atlas_intensities: Vec<Py<PyImage>>,
    atlas_labels: Vec<Py<PyImage>>,
    fusion: &str,
    iterations: Option<Vec<usize>>,
    patch_radius: usize,
    beta: f64,
    region_names: Option<Vec<(u32, String)>>,
) -> RitkResult<PyAtlasParcellationResult> {
    if atlas_intensities.len() != atlas_labels.len() {
        return Err(RitkPyError::value(format!(
            "each atlas needs both an intensity and a label volume: got {} intensities \
             and {} label volumes",
            atlas_intensities.len(),
            atlas_labels.len()
        )));
    }
    if atlas_intensities.is_empty() {
        return Err(RitkPyError::value(
            "parcellation needs at least one atlas".to_owned(),
        ));
    }

    let (_, shape) = image_to_vec(subject.inner.as_ref());
    let voxels = shape[0] * shape[1] * shape[2];
    let names = region_names.unwrap_or_default();

    let atlases = atlas_intensities
        .iter()
        .zip(&atlas_labels)
        .enumerate()
        .map(|(index, (intensity, labels))| {
            read_atlas(py, index, intensity, labels, shape, voxels, &names)
        })
        .collect::<RitkResult<Vec<_>>>()?;

    let config = AtlasParcellationConfig {
        registration: {
            let mut registration = AtlasParcellationConfig::default().registration;
            if let Some(schedule) = iterations {
                if schedule.is_empty() {
                    return Err(RitkPyError::value(
                        "iterations needs at least one level".to_owned(),
                    ));
                }
                registration.num_levels = schedule.len();
                registration.iterations_per_level = schedule;
            }
            registration
        },
        fusion: match fusion {
            "majority" => LabelFusion::MajorityVote,
            "joint" => LabelFusion::JointLabelFusion(LabelFusionConfig { patch_radius, beta }),
            other => {
                return Err(RitkPyError::value(format!(
                    "unknown fusion rule {other:?}; expected {FUSION_CHOICES}"
                )));
            }
        },
    };

    // Bind the borrow before releasing the GIL: the closure must capture a
    // plain reference, not a guard that is not `Ungil`.
    let image = subject.inner.as_ref();
    let result = py
        .allow_threads(|| parcellate_with_atlas_set(image, &atlases, &config))
        .map_err(|error| RitkPyError::runtime(error.to_string()))?;

    Ok(PyAtlasParcellationResult {
        parcellation: Py::new(py, PyParcellation::from_parts(result.parcellation, shape))?,
        agreement: result.agreement,
        shape,
        registration_quality: result.registration_quality,
    })
}

/// Read one atlas pair into the flat form the pipeline takes, rejecting any
/// volume that does not cover the subject's grid.
fn read_atlas(
    py: Python<'_>,
    index: usize,
    intensity: &Py<PyImage>,
    labels: &Py<PyImage>,
    shape: [usize; 3],
    voxels: usize,
    region_names: &[(u32, String)],
) -> RitkResult<LabelledAtlas> {
    let (intensity_values, intensity_shape) = image_to_vec(intensity.borrow(py).inner.as_ref());
    let (label_values, label_shape) = image_to_vec(labels.borrow(py).inner.as_ref());

    if intensity_shape != shape || label_shape != shape {
        return Err(RitkPyError::value(format!(
            "atlas {index} must lie on the subject's grid {shape:?}, but its intensity is \
             {intensity_shape:?} and its labels {label_shape:?}. Resample it first — a \
             registration recovers a deformation, never a resampling."
        )));
    }

    debug_assert_eq!(intensity_values.len(), voxels);
    Ok(LabelledAtlas {
        intensity: intensity_values,
        labels: label_values
            .iter()
            .copied()
            .map(label_from_stored)
            .collect(),
        region_names: region_names.to_vec(),
    })
}
