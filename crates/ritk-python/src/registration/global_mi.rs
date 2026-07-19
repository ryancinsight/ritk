//! Python boundary for classical Leto mutual-information registration.

use pyo3::prelude::*;
use ritk_registration::classical::{
    engine::ClassicalConfig, engine::MutualInformationMetric, image_to_leto_volume,
    ImageRegistration,
};
use ritk_registration::AffineTransform;

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{with_image_pair_slices, PyImage};

/// Options for deterministic native mutual-information registration.
#[pyclass(name = "GlobalMiOptions")]
#[derive(Clone, Debug)]
pub struct PyGlobalMiOptions {
    /// Transformation family: `translation`, `rigid`, or `affine`.
    #[pyo3(get, set)]
    pub transform_type: String,
    /// Histogram bins used by the normalized mutual-information metric.
    #[pyo3(get, set)]
    pub num_mi_bins: usize,
    /// Maximum accepted hill-climbing iterations.
    #[pyo3(get, set)]
    pub maximum_iterations: usize,
    /// Minimum mutual-information improvement required for another iteration.
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Index-space perturbation size in voxels.
    #[pyo3(get, set)]
    pub step_multiplier: f64,
}

impl Default for PyGlobalMiOptions {
    fn default() -> Self {
        Self {
            transform_type: "rigid".to_owned(),
            num_mi_bins: 32,
            maximum_iterations: 100,
            tolerance: 1e-6,
            step_multiplier: 1.0,
        }
    }
}

#[pymethods]
impl PyGlobalMiOptions {
    /// Construct native mutual-information registration options.
    #[new]
    #[pyo3(signature = (
        transform_type = "rigid".to_owned(),
        num_mi_bins = 32,
        maximum_iterations = 100,
        tolerance = 1e-6,
        step_multiplier = 1.0,
    ))]
    pub fn new(
        transform_type: String,
        num_mi_bins: usize,
        maximum_iterations: usize,
        tolerance: f64,
        step_multiplier: f64,
    ) -> Self {
        Self {
            transform_type,
            num_mi_bins,
            maximum_iterations,
            tolerance,
            step_multiplier,
        }
    }
}

/// Register `moving` to `fixed` with deterministic native mutual information.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn global_mi_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyGlobalMiOptions>,
) -> RitkResult<(Vec<f32>, f64, PyObject)> {
    let opts = opts.unwrap_or_default();
    validate_options(&opts)?;
    if fixed.inner.shape() != moving.inner.shape() {
        return Err(RitkPyError::value(format!(
            "fixed and moving shapes must match, got {:?} and {:?}",
            fixed.inner.shape(),
            moving.inner.shape()
        )));
    }

    let fixed_volume = image_to_leto_volume(fixed.inner.as_ref())
        .map_err(|error| RitkPyError::runtime(error.to_string()))?;
    let moving_volume = image_to_leto_volume(moving.inner.as_ref())
        .map_err(|error| RitkPyError::runtime(error.to_string()))?;
    let (minimum, maximum) = intensity_range(fixed, moving)?;
    let engine = ImageRegistration::with_config(
        ClassicalConfig {
            max_iterations: opts.maximum_iterations,
            tolerance: opts.tolerance,
            step_multiplier: opts.step_multiplier,
        },
        MutualInformationMetric::new(opts.num_mi_bins, minimum, maximum),
    );
    let transform_type = opts.transform_type.clone();
    let result = py
        .allow_threads(|| match transform_type.as_str() {
            "translation" => engine.translation_registration_mutual_info(
                &moving_volume,
                &fixed_volume,
                &AffineTransform::IDENTITY,
            ),
            "rigid" => engine.rigid_registration_mutual_info(
                &moving_volume,
                &fixed_volume,
                &AffineTransform::IDENTITY,
            ),
            "affine" => engine.affine_registration_mutual_info(
                &moving_volume,
                &fixed_volume,
                &AffineTransform::IDENTITY,
            ),
            _ => unreachable!("transform_type was validated"),
        })
        .map_err(|error| RitkPyError::runtime(error.to_string()))?;

    let matrix = physical_matrix(&result.transform, fixed, moving)?;
    let info = pyo3::types::PyDict::new_bound(py);
    info.set_item(
        "convergence_history",
        vec![format!("{:?}", result.quality.convergence)],
    )?;
    info.set_item("iterations_per_level", vec![result.quality.iterations])?;
    info.set_item(
        "converged",
        result.quality.convergence == ritk_registration::validation::ConvergenceStatus::Converged,
    )?;
    Ok((
        matrix,
        result.quality.mutual_information,
        info.unbind().into(),
    ))
}

fn validate_options(opts: &PyGlobalMiOptions) -> RitkResult<()> {
    if !matches!(
        opts.transform_type.as_str(),
        "translation" | "rigid" | "affine"
    ) {
        return Err(RitkPyError::value(format!(
            "transform_type must be translation, rigid, or affine, got {}",
            opts.transform_type
        )));
    }
    if opts.num_mi_bins < 4 {
        return Err(RitkPyError::value(format!(
            "num_mi_bins must be at least 4, got {}",
            opts.num_mi_bins
        )));
    }
    if opts.maximum_iterations == 0 {
        return Err(RitkPyError::value("maximum_iterations must be positive"));
    }
    if !opts.tolerance.is_finite() || opts.tolerance < 0.0 {
        return Err(RitkPyError::value(format!(
            "tolerance must be finite and non-negative, got {}",
            opts.tolerance
        )));
    }
    if !opts.step_multiplier.is_finite() || opts.step_multiplier <= 0.0 {
        return Err(RitkPyError::value(format!(
            "step_multiplier must be finite and positive, got {}",
            opts.step_multiplier
        )));
    }
    Ok(())
}

fn intensity_range(fixed: &PyImage, moving: &PyImage) -> RitkResult<(f64, f64)> {
    let (minimum, maximum) = with_image_pair_slices(
        fixed.inner.as_ref(),
        moving.inner.as_ref(),
        |fixed_values, moving_values| {
            let mut values = fixed_values.iter().chain(moving_values.iter()).copied();
            let first = values.next()?;
            Some(values.fold((first, first), |(minimum, maximum), value| {
                (minimum.min(value), maximum.max(value))
            }))
        },
    )
    .ok_or_else(|| RitkPyError::value("registration images must not be empty"))?;
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(RitkPyError::value(
            "registration images must contain only finite values",
        ));
    }
    if minimum == maximum {
        return Err(RitkPyError::value(
            "mutual-information registration requires nonconstant images",
        ));
    }
    Ok((f64::from(minimum), f64::from(maximum)))
}

fn physical_matrix(
    index_transform: &AffineTransform,
    fixed: &PyImage,
    moving: &PyImage,
) -> RitkResult<Vec<f32>> {
    let transform = ritk_registration::classical::index_affine_to_physical(
        index_transform,
        fixed.inner.as_ref(),
        moving.inner.as_ref(),
    )
    .map_err(|error| RitkPyError::runtime(error.to_string()))?;
    let matrix = transform.matrix();
    let translation = transform.translation();
    Ok(vec![
        matrix[0],
        matrix[1],
        matrix[2],
        translation[0],
        matrix[3],
        matrix[4],
        matrix[5],
        translation[1],
        matrix[6],
        matrix[7],
        matrix[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}
