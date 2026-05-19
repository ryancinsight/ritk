use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

use super::shared::{load_matching_inputs, to_py_warped_and_displacement};

/// Configuration options for [`lddmm_register`].
#[pyclass(name = "LddmmConfig")]
#[derive(Clone)]
pub struct PyLddmmConfig {
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub num_time_steps: usize,
    #[pyo3(get, set)]
    pub kernel_sigma: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub regularization_weight: f64,
}

impl Default for PyLddmmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            num_time_steps: 10,
            kernel_sigma: 2.0,
            learning_rate: 0.1,
            regularization_weight: 1.0,
        }
    }
}

#[pymethods]
impl PyLddmmConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using LDDMM.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn lddmm_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyLddmmConfig>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let inputs = load_matching_inputs(fixed, moving)?;

    py.allow_threads(|| {
        let config = LddmmConfig {
            max_iterations: opts.max_iterations,
            num_time_steps: opts.num_time_steps,
            kernel_sigma: opts.kernel_sigma,
            learning_rate: opts.learning_rate,
            regularization_weight: opts.regularization_weight,
            ..Default::default()
        };
        let reg = LddmmRegistration::new(config);
        reg.register(&inputs.fixed_vals, &inputs.moving_vals, inputs.fixed_shape, [1.0, 1.0, 1.0])
            .map_err(|e| e.to_string())
    })
    .map_err(crate::errors::RitkPyError::runtime)
    .map(|result| {
        to_py_warped_and_displacement(
            result.warped_moving,
            result.displacement_field,
            &inputs,
        )
    })
}
