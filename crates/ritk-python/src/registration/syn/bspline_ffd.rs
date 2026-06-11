use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration, VolumeDims};

use super::shared::{load_matching_inputs, to_py_moving};

/// Configuration options for [`bspline_ffd_register`].
#[pyclass(name = "BSplineFfdConfig")]
#[derive(Clone)]
pub struct PyBSplineFfdConfig {
    #[pyo3(get, set)]
    pub initial_control_spacing: usize,
    #[pyo3(get, set)]
    pub num_levels: usize,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub regularization_weight: f64,
}

impl Default for PyBSplineFfdConfig {
    fn default() -> Self {
        Self {
            initial_control_spacing: 8,
            num_levels: 3,
            max_iterations: 100,
            learning_rate: 0.01,
            regularization_weight: 0.001,
        }
    }
}

#[pymethods]
impl PyBSplineFfdConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using BSpline Free-Form Deformation.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn bspline_ffd_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyBSplineFfdConfig>,
) -> RitkResult<PyImage> {
    let opts = opts.unwrap_or_default();
    let inputs = load_matching_inputs(fixed, moving)?;

    py.allow_threads(|| {
        let config = BSplineFFDConfig {
            initial_control_spacing: [
                opts.initial_control_spacing,
                opts.initial_control_spacing,
                opts.initial_control_spacing,
            ],
            num_levels: opts.num_levels,
            max_iterations_per_level: opts.max_iterations,
            learning_rate: opts.learning_rate,
            regularization_weight: opts.regularization_weight,
            ..Default::default()
        };
        BSplineFFDRegistration::register(
            &inputs.fixed_vals,
            &inputs.moving_vals,
            VolumeDims(inputs.fixed_shape),
            [1.0, 1.0, 1.0],
            &config,
        )
        .map_err(|e| e.to_string())
    })
    .map_err(crate::errors::RitkPyError::runtime)
    .map(|result| to_py_moving(result.warped_moving, &inputs))
}
