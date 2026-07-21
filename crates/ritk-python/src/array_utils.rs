//! Array utilities for copying NumPy arrays into Rust `Vec`s.
//!
//! These helpers avoid any source-level dependency on `ndarray` while still
//! handling non-contiguous NumPy inputs correctly.

use numpy::{Element, Ix3, Ix4, PyArrayMethods, PyReadonlyArray};
use pyo3::PyResult;

/// Copy the contents of a 3-D NumPy array into a Rust `Vec` in C-contiguous order.
///
/// Contiguous arrays take the fast zero-copy path. Non-contiguous arrays are
/// transparently copied into a fresh C-contiguous NumPy array via NumPy's
/// `PyArray_CastToType` and then read back as a `Vec`.
///
/// # Errors
///
/// Returns an error if NumPy cannot copy the source array (e.g. dtype or
/// dimension mismatch).
pub fn copy_array3_to_vec<'py, T>(array: &PyReadonlyArray<'py, T, Ix3>) -> PyResult<Vec<T>>
where
    T: Element + Copy,
{
    // Fast path: the backing memory is already contiguous.
    if let Ok(slice) = array.as_slice() {
        return Ok(slice.to_vec());
    }

    // Fallback: allocate a C-contiguous copy and read it back element-by-element.
    // `cast` with `is_fortran=false` produces a C-order copy of the same dtype.
    let copy = array.cast::<T>(false)?;
    copy.to_vec()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("failed to read contiguous copy: {e}")))
}

/// Copy the contents of a 4-D NumPy array into a Rust `Vec` in C-contiguous order.
///
/// See [`copy_array3_to_vec`] for details.
pub fn copy_array4_to_vec<'py, T>(array: &PyReadonlyArray<'py, T, Ix4>) -> PyResult<Vec<T>>
where
    T: Element + Copy,
{
    if let Ok(slice) = array.as_slice() {
        return Ok(slice.to_vec());
    }

    let copy = array.cast::<T>(false)?;
    copy.to_vec()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("failed to read contiguous copy: {e}")))
}
