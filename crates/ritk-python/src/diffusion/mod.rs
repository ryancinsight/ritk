//! Python bindings for diffusion tensor fitting.
//!
//! The binding leaves estimation and physical validation in
//! [`ritk_diffusion::maps`]. This module owns only submodule registration and
//! the stable Python-facing exports.

mod fit;
mod maps;

use pyo3::prelude::*;

pub use fit::fit_tensor_maps;
pub use maps::PyDiffusionMaps;

/// Register the diffusion submodule into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(parent.py(), "diffusion")?;
    module.add_class::<PyDiffusionMaps>()?;
    module.add_function(wrap_pyfunction!(fit_tensor_maps, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
