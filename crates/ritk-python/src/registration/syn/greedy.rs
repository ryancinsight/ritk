use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};

use super::shared::{load_matching_inputs, to_py_pair};

/// Configuration options for [`syn_register`].
#[pyclass(name = "SynConfig")]
#[derive(Clone)]
pub struct PySynConfig {
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    #[pyo3(get, set)]
    pub cc_radius: usize,
    #[pyo3(get, set)]
    pub gradient_step: f64,
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
}

impl Default for PySynConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            sigma_smooth: 3.0,
            cc_radius: 2,
            gradient_step: 0.25,
            convergence_threshold: 1e-8,
        }
    }
}

#[pymethods]
impl PySynConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using greedy SyN.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PySynConfig>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let inputs = load_matching_inputs(fixed, moving)?;

    py.allow_threads(|| {
        let config = SyNConfig {
            max_iterations: opts.max_iterations,
            sigma_smooth: opts.sigma_smooth,
            cc_window_radius: opts.cc_radius,
            gradient_step: opts.gradient_step,
            convergence_threshold: opts.convergence_threshold,
            ..Default::default()
        };
        let reg = SyNRegistration::new(config);
        reg.register(&inputs.fixed_vals, &inputs.moving_vals, inputs.fixed_shape, [1.0, 1.0, 1.0])
            .map_err(|e| e.to_string())
    })
    .map_err(crate::errors::RitkPyError::runtime)
    .map(|result| to_py_pair(result.warped_fixed, result.warped_moving, &inputs))
}
