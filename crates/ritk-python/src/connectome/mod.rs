//! Python bindings for parcellations and the connectomes built from them.
//!
//! Construction, every graph measure, and every physical validation stay in
//! `ritk-parcellation` and `ritk-connectome`. This module owns only the
//! conversion between NumPy arrays and those crates' types, plus submodule
//! registration.
//!
//! # Array conventions
//!
//! Label volumes arrive `[Z, Y, X]`, matching every other RITK array, and their
//! `spacing`, `origin`, and `direction` are given in the same outermost-first
//! order an `Image` reports. `ParcellationGrid::from_image_order` performs the
//! reversal into the grid's own innermost-first order, so the binding never
//! reimplements it.
//!
//! Streamlines arrive as a sequence of `[N, 3]` float arrays in the
//! parcellation's physical frame — the same frame `ritk.diffusion` tracking
//! produces once its points are mapped to millimetres.

mod matrix;
mod parcellation;

use pyo3::prelude::*;

pub use matrix::{build_connectivity_matrix, PyConnectivityMatrix, PyGraphMeasures};
pub use parcellation::PyParcellation;

/// Register the connectome submodule into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(parent.py(), "connectome")?;
    module.add_class::<PyParcellation>()?;
    module.add_class::<PyConnectivityMatrix>()?;
    module.add_class::<PyGraphMeasures>()?;
    module.add_function(wrap_pyfunction!(build_connectivity_matrix, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
