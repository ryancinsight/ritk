use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};

use super::shared::{load_matching_inputs, to_py_pair};

/// Configuration options for [`bspline_syn_register`].
#[pyclass(name = "BSplineSynOptions")]
#[derive(Clone)]
pub struct PyBSplineSynOptions {
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub control_spacing_z: usize,
    #[pyo3(get, set)]
    pub control_spacing_y: usize,
    #[pyo3(get, set)]
    pub control_spacing_x: usize,
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    #[pyo3(get, set)]
    pub cc_radius: usize,
    #[pyo3(get, set)]
    pub regularization_weight: f64,
    #[pyo3(get, set)]
    pub gradient_step: f64 }

impl Default for PyBSplineSynOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            control_spacing_z: 8,
            control_spacing_y: 8,
            control_spacing_x: 8,
            sigma_smooth: 1.0,
            cc_radius: 2,
            regularization_weight: 0.001,
            gradient_step: 0.25 }
    }
}

#[pymethods]
impl PyBSplineSynOptions {
    #[new]
    #[pyo3(signature = (
        max_iterations = 100,
        control_spacing_z = 8,
        control_spacing_y = 8,
        control_spacing_x = 8,
        sigma_smooth = 1.0,
        cc_radius = 2,
        regularization_weight = 0.001,
        gradient_step = 0.25,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_iterations: usize,
        control_spacing_z: usize,
        control_spacing_y: usize,
        control_spacing_x: usize,
        sigma_smooth: f64,
        cc_radius: usize,
        regularization_weight: f64,
        gradient_step: f64,
    ) -> Self {
        Self {
            max_iterations,
            control_spacing_z,
            control_spacing_y,
            control_spacing_x,
            sigma_smooth,
            cc_radius,
            regularization_weight,
            gradient_step }
    }
}

/// Register a moving image to a fixed image using BSpline SyN.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn bspline_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyBSplineSynOptions>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let inputs = load_matching_inputs(fixed, moving)?;

    py.allow_threads(|| {
        let config = BSplineSyNConfig {
            max_iterations: opts.max_iterations,
            control_spacing: [
                opts.control_spacing_z,
                opts.control_spacing_y,
                opts.control_spacing_x,
            ],
            sigma_smooth: opts.sigma_smooth,
            convergence_threshold: 1e-6,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: opts.cc_radius,
            regularization_weight: opts.regularization_weight,
            gradient_step: opts.gradient_step };
        let reg = BSplineSyNRegistration::new(config);
        reg.register(
            &inputs.fixed_vals,
            &inputs.moving_vals,
            inputs.fixed_shape,
            [1.0, 1.0, 1.0],
        )
        .map_err(|e| e.to_string())
    })
    .map_err(crate::errors::RitkPyError::runtime)
    .map(|result| to_py_pair(result.warped_fixed, result.warped_moving, &inputs))
}
