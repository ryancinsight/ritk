//! Python extension module for ritk — medical image processing toolkit.
//!
//! This crate exposes RITK functionality to Python via PyO3. The module
//! is compiled as a native extension (`cdylib`) named `_ritk` and placed
//! inside the `ritk` Python package.  The `ritk/__init__.py` re-exports
//! the submodules so they are importable as:
//!   `import ritk; ritk.image, ritk.io, ritk.filter, ...`

pub mod filter;
pub mod image;
pub mod io;
pub mod registration;
pub mod segmentation;
pub mod statistics;

use pyo3::prelude::*;

/// Top-level native extension module `_ritk`.
///
/// Registered submodules are re-exported by `ritk/__init__.py` so they
/// are importable as `ritk.image`, `ritk.io`, `ritk.filter`, etc.
#[pymodule]
fn _ritk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    image::register(m)?;
    io::register(m)?;
    filter::register(m)?;
    registration::register(m)?;
    segmentation::register(m)?;
    statistics::register(m)?;
    Ok(())
}
