use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use ritk_registration::diffeomorphic::multires_syn::{
    InverseConsistency, MultiResSyNConfig, MultiResSyNRegistration,
};

use super::shared::{load_matching_inputs, to_py_pair};

/// Inverse-consistency policy for Multi-Resolution SyN, replacing `inverse_consistency: bool`.
///
/// Eliminates boolean blindness: `inverse_consistency="enforced"` vs
/// `inverse_consistency="relaxed"` is self-documenting compared to `True/False`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyInverseConsistency {
    /// No inverse-consistency enforcement (relaxed update, default).
    #[default]
    Relaxed,
    /// Enforce inverse consistency via `v ← (v − compose(v₁,v₂)) / 2`.
    Enforced,
}

impl<'py> FromPyObject<'py> for PyInverseConsistency {
    fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "relaxed" => Ok(Self::Relaxed),
            "enforced" => Ok(Self::Enforced),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown inverse consistency policy '{}'. Choices: relaxed, enforced",
                other
            ))),
        }
    }
}

impl IntoPy<PyObject> for PyInverseConsistency {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::Relaxed => "relaxed".into_py(py),
            Self::Enforced => "enforced".into_py(py),
        }
    }
}

impl From<PyInverseConsistency> for InverseConsistency {
    fn from(val: PyInverseConsistency) -> Self {
        match val {
            PyInverseConsistency::Relaxed => InverseConsistency::Relaxed,
            PyInverseConsistency::Enforced => InverseConsistency::Enforced,
        }
    }
}

/// Configuration options for [`multires_syn_register`].
#[pyclass(name = "MultiResSynOptions")]
#[derive(Clone)]
pub struct PyMultiresSynOptions {
    #[pyo3(get, set)]
    pub num_levels: usize,
    #[pyo3(get, set)]
    pub iterations: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    #[pyo3(get, set)]
    pub cc_radius: usize,
    #[pyo3(get, set)]
    pub inverse_consistency: PyInverseConsistency,
    #[pyo3(get, set)]
    pub gradient_step: f64,
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
}

#[pymethods]
impl PyMultiresSynOptions {
    #[new]
    #[pyo3(signature = (
        num_levels = 3,
        iterations = None,
        sigma_smooth = 3.0,
        cc_radius = 2,
        inverse_consistency = PyInverseConsistency::Enforced,
        gradient_step = 0.25,
        convergence_threshold = 1e-8,
    ))]
    pub fn new(
        num_levels: usize,
        iterations: Option<Vec<usize>>,
        sigma_smooth: f64,
        cc_radius: usize,
        inverse_consistency: PyInverseConsistency,
        gradient_step: f64,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            num_levels,
            iterations,
            sigma_smooth,
            cc_radius,
            inverse_consistency,
            gradient_step,
            convergence_threshold,
        }
    }
}

/// Register a moving image to a fixed image using Multi-Resolution SyN.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn multires_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyMultiresSynOptions>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_else(|| {
        PyMultiresSynOptions::new(3, None, 3.0, 2, PyInverseConsistency::Enforced, 0.25, 1e-8)
    });
    let inputs = load_matching_inputs(fixed, moving)?;

    py.allow_threads(|| {
        let config = MultiResSyNConfig {
            num_levels: opts.num_levels,
            iterations_per_level: opts.iterations.unwrap_or_else(|| vec![100, 70, 20]),
            sigma_smooth: opts.sigma_smooth,
            convergence_threshold: opts.convergence_threshold,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: opts.cc_radius,
            enforce_inverse_consistency: InverseConsistency::from(opts.inverse_consistency),
            gradient_step: opts.gradient_step,
        };
        let reg = MultiResSyNRegistration::new(config);
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
