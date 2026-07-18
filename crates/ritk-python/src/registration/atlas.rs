//! Atlas building and label fusion: population atlas, majority vote, and Joint Label Fusion.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_registration::atlas::label_fusion::{
    joint_label_fusion, majority_vote, LabelFusionConfig };
use ritk_registration::atlas::{AtlasConfig, AtlasRegistration};
use ritk_registration::diffeomorphic::multires_syn::{InverseConsistency, MultiResSyNConfig};

/// Configuration options for [`build_atlas`].
///
/// All fields mirror the positional parameters of the previous `build_atlas`
/// pyfunction signature and carry the same defaults.
#[pyclass(name = "AtlasBuildOptions")]
#[derive(Clone)]
pub struct PyAtlasBuildOptions {
    /// Maximum outer template-building iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Per-voxel RMS change threshold for early stopping.
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
    /// Per-level SyN iteration counts; `None` â†’ [100, 70, 20].
    #[pyo3(get, set)]
    pub syn_iterations: Option<Vec<usize>>,
    /// Gaussian smoothing sigma for velocity fields (voxels).
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    /// Cross-correlation window radius (voxels).
    #[pyo3(get, set)]
    pub cc_radius: usize,
    /// Maximum per-iteration displacement in voxels.
    #[pyo3(get, set)]
    pub gradient_step: f64 }

#[pymethods]
impl PyAtlasBuildOptions {
    #[new]
    #[pyo3(signature = (
        max_iterations = 5,
        convergence_threshold = 0.01,
        syn_iterations = None,
        sigma_smooth = 3.0,
        cc_radius = 2,
        gradient_step = 0.25,
    ))]
    pub fn new(
        max_iterations: usize,
        convergence_threshold: f64,
        syn_iterations: Option<Vec<usize>>,
        sigma_smooth: f64,
        cc_radius: usize,
        gradient_step: f64,
    ) -> Self {
        Self {
            max_iterations,
            convergence_threshold,
            syn_iterations,
            sigma_smooth,
            cc_radius,
            gradient_step }
    }
}

/// Build a population-specific atlas template from multiple subject images.
///
/// Iteratively registers all subjects to a mean template using Multi-Resolution
/// SyN, refines the template, and applies mean-drift sharpening until
/// convergence.
///
/// Args:
///     subjects: List of subject images (all must share the same shape).
///     opts:     `AtlasBuildOptions` controlling SyN configuration and stopping criteria.
///
/// Returns:
///     (template, convergence_history):
///     - `template`: the final atlas template image with spatial metadata from
///       the first subject.
///     - `convergence_history`: per-iteration RMS change values.
///
/// Raises:
///     RuntimeError: if subjects is empty, shapes mismatch, or registration fails.
#[pyfunction]
#[pyo3(signature = (subjects, opts = None))]
pub fn build_atlas(
    py: Python<'_>,
    subjects: Vec<Py<PyImage>>,
    opts: Option<PyAtlasBuildOptions>,
) -> RitkResult<(PyImage, Vec<f64>)> {
    let opts = opts.unwrap_or_else(|| PyAtlasBuildOptions::new(5, 0.01, None, 3.0, 2, 0.25));
    let max_iterations = opts.max_iterations;
    let convergence_threshold = opts.convergence_threshold;
    let syn_iterations = opts.syn_iterations;
    let sigma_smooth = opts.sigma_smooth;
    let cc_radius = opts.cc_radius;
    let gradient_step = opts.gradient_step;
    if subjects.is_empty() {
        return Err(RitkPyError::runtime(
            "subjects list is empty; at least one subject is required",
        ));
    }

    let mut subject_vecs: Vec<Vec<f32>> = Vec::with_capacity(subjects.len());
    let (first_vals, first_shape) = image_to_vec(subjects[0].borrow(py).inner.as_ref());
    subject_vecs.push(first_vals);

    for (i, subj) in subjects.iter().enumerate().skip(1) {
        let (vals, shape) = image_to_vec(subj.borrow(py).inner.as_ref());
        if shape != first_shape {
            return Err(RitkPyError::runtime(format!(
                "subject[{}] shape {:?} != subject[0] shape {:?}",
                i, shape, first_shape
            )));
        }
        subject_vecs.push(vals);
    }

    py.allow_threads(|| {
        let subject_slices: Vec<&[f32]> = subject_vecs.iter().map(|v| v.as_slice()).collect();
        let syn_config = MultiResSyNConfig {
            num_levels: 3,
            iterations_per_level: syn_iterations.unwrap_or_else(|| vec![100, 70, 20]),
            sigma_smooth,
            convergence_threshold: 1e-6,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: cc_radius,
            enforce_inverse_consistency: InverseConsistency::Enforced,
            gradient_step };
        let config = AtlasConfig {
            max_iterations,
            convergence_threshold,
            syn_config };
        let reg = AtlasRegistration::new(config);
        reg.build_atlas(&subject_slices, first_shape, [1.0, 1.0, 1.0])
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(|result| {
        let template_image = vec_to_image(
            result.template,
            first_shape,
            *subjects[0].borrow(py).inner.origin(),
            *subjects[0].borrow(py).inner.spacing(),
            *subjects[0].borrow(py).inner.direction(),
        );
        (into_py_image(template_image), result.convergence_history)
    })
}

/// Fuse multiple atlas label maps via majority voting.
///
/// For each voxel the output label is the mode across all atlas label maps.
/// Ties are broken by selecting the smallest label value.
///
/// Atlas label images are expected to contain integer labels stored as f32.
/// Values are rounded to the nearest u32 before voting.
///
/// Args:
///     atlas_labels: List of label map images (all must share the same shape).
///
/// Returns:
///     (labels, confidence):
///     - `labels`: fused label map (u32 labels stored as f32).
///     - `confidence`: per-voxel fraction of atlases voting for the winning label.
///
/// Raises:
///     RuntimeError: if atlas_labels is empty or shapes mismatch.
#[pyfunction]
pub fn majority_vote_fusion(
    py: Python<'_>,
    atlas_labels: Vec<Py<PyImage>>,
) -> RitkResult<(PyImage, PyImage)> {
    if atlas_labels.is_empty() {
        return Err(RitkPyError::runtime(
            "atlas_labels list is empty; at least one atlas is required",
        ));
    }

    let mut label_vecs: Vec<Vec<u32>> = Vec::with_capacity(atlas_labels.len());
    let (first_vals, first_shape) = image_to_vec(atlas_labels[0].borrow(py).inner.as_ref());
    label_vecs.push(first_vals.iter().map(|&v| v.round() as u32).collect());

    for (i, lbl) in atlas_labels.iter().enumerate().skip(1) {
        let (vals, shape) = image_to_vec(lbl.borrow(py).inner.as_ref());
        if shape != first_shape {
            return Err(RitkPyError::runtime(format!(
                "atlas_labels[{}] shape {:?} != atlas_labels[0] shape {:?}",
                i, shape, first_shape
            )));
        }
        label_vecs.push(vals.iter().map(|&v| v.round() as u32).collect());
    }

    py.allow_threads(|| {
        let label_slices: Vec<&[u32]> = label_vecs.iter().map(|v| v.as_slice()).collect();
        majority_vote(&label_slices, first_shape).map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(|result| {
        let labels_f32: Vec<f32> = result.labels.iter().map(|&l| l as f32).collect();
        let labels_image = vec_to_image(
            labels_f32,
            first_shape,
            *atlas_labels[0].borrow(py).inner.origin(),
            *atlas_labels[0].borrow(py).inner.spacing(),
            *atlas_labels[0].borrow(py).inner.direction(),
        );
        let confidence_image = vec_to_image(
            result.confidence,
            first_shape,
            *atlas_labels[0].borrow(py).inner.origin(),
            *atlas_labels[0].borrow(py).inner.spacing(),
            *atlas_labels[0].borrow(py).inner.direction(),
        );
        (into_py_image(labels_image), into_py_image(confidence_image))
    })
}

/// Fuse atlas label maps using the Joint Label Fusion (JLF) algorithm.
///
/// Weighted voting where per-voxel weights are derived from local patch
/// intensity similarity between warped atlas images and the target image.
///
/// Args:
///     target:       Target intensity image (the image being segmented).
///     atlas_images: List of warped atlas intensity images registered to target space.
///     atlas_labels: List of atlas label maps registered to target space.
///     patch_radius: Local patch radius for similarity computation (default 2).
///     beta:         Regularization parameter for the pairwise similarity matrix (default 0.1).
///
/// Returns:
///     (labels, confidence):
///     - `labels`: fused label map (u32 labels stored as f32).
///     - `confidence`: per-voxel sum of JLF weights assigned to the winning label.
///
/// Raises:
///     RuntimeError: if atlas counts mismatch, shapes mismatch, or fusion fails.
#[pyfunction]
#[pyo3(signature = (target, atlas_images, atlas_labels, patch_radius=2, beta=0.1))]
pub fn joint_label_fusion_py(
    py: Python<'_>,
    target: &PyImage,
    atlas_images: Vec<Py<PyImage>>,
    atlas_labels: Vec<Py<PyImage>>,
    patch_radius: usize,
    beta: f64,
) -> RitkResult<(PyImage, PyImage)> {
    let (target_vals, target_shape) = image_to_vec(target.inner.as_ref());

    if atlas_images.len() != atlas_labels.len() {
        return Err(RitkPyError::runtime(format!(
            "atlas_images length {} != atlas_labels length {}",
            atlas_images.len(),
            atlas_labels.len()
        )));
    }
    if atlas_images.is_empty() {
        return Err(RitkPyError::runtime(
            "atlas_images list is empty; at least one atlas is required",
        ));
    }

    let mut img_vecs: Vec<Vec<f32>> = Vec::with_capacity(atlas_images.len());
    let mut lbl_vecs: Vec<Vec<u32>> = Vec::with_capacity(atlas_labels.len());

    for (i, (img, lbl)) in atlas_images.iter().zip(atlas_labels.iter()).enumerate() {
        let (img_vals, img_shape) = image_to_vec(img.borrow(py).inner.as_ref());
        if img_shape != target_shape {
            return Err(RitkPyError::runtime(format!(
                "atlas_images[{}] shape {:?} != target shape {:?}",
                i, img_shape, target_shape
            )));
        }
        let (lbl_vals, lbl_shape) = image_to_vec(lbl.borrow(py).inner.as_ref());
        if lbl_shape != target_shape {
            return Err(RitkPyError::runtime(format!(
                "atlas_labels[{}] shape {:?} != target shape {:?}",
                i, lbl_shape, target_shape
            )));
        }
        img_vecs.push(img_vals);
        lbl_vecs.push(lbl_vals.iter().map(|&v| v.round() as u32).collect());
    }

    py.allow_threads(|| {
        let img_slices: Vec<&[f32]> = img_vecs.iter().map(|v| v.as_slice()).collect();
        let lbl_slices: Vec<&[u32]> = lbl_vecs.iter().map(|v| v.as_slice()).collect();
        let config = LabelFusionConfig { patch_radius, beta };
        joint_label_fusion(
            &target_vals,
            &img_slices,
            &lbl_slices,
            target_shape,
            &config,
        )
        .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(|result| {
        let labels_f32: Vec<f32> = result.labels.iter().map(|&l| l as f32).collect();
        let labels_image = vec_to_image(
            labels_f32,
            target_shape,
            *target.inner.origin(),
            *target.inner.spacing(),
            *target.inner.direction(),
        );
        let confidence_image = vec_to_image(
            result.confidence,
            target_shape,
            *target.inner.origin(),
            *target.inner.spacing(),
            *target.inner.direction(),
        );
        (into_py_image(labels_image), into_py_image(confidence_image))
    })
}
