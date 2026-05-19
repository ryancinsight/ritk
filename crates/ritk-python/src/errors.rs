//! `RitkPyError`: a non-`PyErr` error type for `#[pyfunction]` / `#[pymethods]`.
//!
//! PyO3 0.22 generates `OkWrap::wrap(ret).map_err(Into::<PyErr>::into)` in every
//! `#[pyfunction]` wrapper. When `ret: Result<T, PyErr>`, the `Into::<PyErr>::into`
//! call is identity (`From<PyErr> for PyErr`), which `clippy::useless_conversion`
//! flags at the function's return-type span.
//!
//! Returning `Result<T, RitkPyError>` instead of `PyResult<T>` breaks the identity:
//! the generated `Into::<PyErr>::into` now calls `From<RitkPyError> for PyErr` — a
//! distinct, non-identity conversion — and the lint no longer fires.

use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Opaque wrapper around `PyErr` that makes `From<RitkPyError> for PyErr`
/// a non-identity conversion, silencing `clippy::useless_conversion` in
/// PyO3-generated `#[pyfunction]` / `#[pymethods]` wrappers.
pub struct RitkPyError(PyErr);

impl From<RitkPyError> for PyErr {
    #[inline]
    fn from(e: RitkPyError) -> Self {
        e.0
    }
}

impl RitkPyError {
    /// `PyRuntimeError` — computation failure.
    #[inline]
    pub fn runtime(msg: impl ToString) -> Self {
        Self(PyRuntimeError::new_err(msg.to_string()))
    }

    /// `PyValueError` — invalid argument.
    #[inline]
    pub fn value(msg: impl ToString) -> Self {
        Self(PyValueError::new_err(msg.to_string()))
    }

    /// `PyIOError` — file-system or I/O failure.
    #[inline]
    pub fn io(msg: impl ToString) -> Self {
        Self(PyIOError::new_err(msg.to_string()))
    }

    /// `PyKeyError` — missing key in a dict or mapping.
    #[inline]
    pub fn key(msg: impl ToString) -> Self {
        Self(PyKeyError::new_err(msg.to_string()))
    }

    /// `PyTypeError` — unexpected type.
    #[inline]
    pub fn type_err(msg: impl ToString) -> Self {
        Self(PyTypeError::new_err(msg.to_string()))
    }

    /// Wrap an existing `PyErr` directly.
    #[inline]
    pub fn from_py(e: PyErr) -> Self {
        Self(e)
    }
}

/// Allow `?` on `PyResult<T>` operations inside `RitkResult<T>` functions.
///
/// Used by functions that call PyO3 dict/list/extract operations (which return
/// `PyResult<T>`) and propagate those errors via `?`.  The conversion is
/// non-identity (`From<PyErr> for RitkPyError ≠ From<PyErr> for PyErr`).
impl From<PyErr> for RitkPyError {
    #[inline]
    fn from(e: PyErr) -> Self {
        Self(e)
    }
}

/// Convenience type alias for functions returning `RitkPyError`.
pub type RitkResult<T> = Result<T, RitkPyError>;
