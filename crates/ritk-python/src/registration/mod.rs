//! Python-exposed deformable image registration algorithms.
//!
//! All registration functions delegate to `ritk-registration` crate
//! implementations.  This module handles PyO3 boundary conversion
//! (PyImage ↔ flat `Vec<f32>`) and result packing only.
//!
//! # Submodules
//! - `demons`: Thirion, Diffeomorphic, and Symmetric Demons.
//! - `multires`: Multi-resolution and Inverse-consistent Demons.
//! - `syn`:     Greedy SyN, BSpline FFD, Multi-resolution SyN, BSpline SyN, LDDMM.
//! - `atlas`:   Population atlas building, majority vote fusion, Joint Label Fusion.

mod atlas;
mod demons;
mod global_mi;
mod multires;
mod syn;

pub use atlas::*;
pub use demons::*;
pub use global_mi::*;
pub use multires::*;
pub use syn::*;

use pyo3::prelude::*;

/// Register the `registration` submodule and all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "registration")?;
    m.add_function(wrap_pyfunction!(demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(diffeomorphic_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(level_set_motion_register, &m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse_consistent_demons_register, &m)?)?;
    m.add_class::<PyMultiresDemonsOptions>()?;
    m.add_function(wrap_pyfunction!(multires_demons_register, &m)?)?;
    m.add_class::<PySynConfig>()?;
    m.add_function(wrap_pyfunction!(syn_register, &m)?)?;
    m.add_class::<PyBSplineFfdConfig>()?;
    m.add_function(wrap_pyfunction!(bspline_ffd_register, &m)?)?;
    m.add_class::<PyMultiresSynOptions>()?;
    m.add_function(wrap_pyfunction!(multires_syn_register, &m)?)?;
    m.add_class::<PyBSplineSynOptions>()?;
    m.add_function(wrap_pyfunction!(bspline_syn_register, &m)?)?;
    m.add_class::<PyLddmmConfig>()?;
    m.add_function(wrap_pyfunction!(lddmm_register, &m)?)?;
    m.add_class::<PyAtlasBuildOptions>()?;
    m.add_function(wrap_pyfunction!(build_atlas, &m)?)?;
    m.add_function(wrap_pyfunction!(majority_vote_fusion, &m)?)?;
    m.add_function(wrap_pyfunction!(joint_label_fusion_py, &m)?)?;
    m.add_class::<PyGlobalMiOptions>()?;
    m.add_function(wrap_pyfunction!(global_mi_register, &m)?)?;
    m.add_class::<PyCmaMiOptions>()?;
    m.add_function(wrap_pyfunction!(cma_mi_register, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
