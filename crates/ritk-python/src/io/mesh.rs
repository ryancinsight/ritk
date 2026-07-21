//! Python-exposed mesh I/O (OBJ, STL, PLY, glTF, VTK polydata, VTP).
//!
//! # Supported formats
//! | Extension(s)       | Read | Write | Notes                      |
//! |--------------------|------|-------|----------------------------|
//! | `.obj`             | ✓    | ✓     | Wavefront OBJ ASCII        |
//! | `.stl`             | ✓    | ✓     | STL ASCII + binary         |
//! | `.ply`             | ✓    | ✓     | PLY ASCII + binary LE      |
//! | `.gltf`            | ✗    | ✓     | glTF 2.0 JSON (base64 buf) |
//! | `.vtk`             | ✓    | ✓     | VTK legacy polydata        |
//! | `.vtp`             | ✓    | ✓     | VTK XML polydata           |
//!
//! All functions raise `IOError` on read/write failure.

use crate::errors::{RitkPyError, RitkResult};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use ritk_io::{
    read_obj_mesh, read_ply_mesh, read_stl_mesh, read_vtk_polydata, read_vtp_polydata, write_gltf,
    write_obj_mesh, write_ply_ascii, write_ply_binary_le, write_stl_ascii, write_stl_binary,
    write_vtk_polydata, write_vtp_polydata, VtkPolyData,
};
use std::path::Path;

// ── PyMesh ────────────────────────────────────────────────────────────────────

/// Polygonal mesh with points, polygon connectivity, and optional normals.
///
/// Wraps `VtkPolyData` for Python access.  Points are stored as float32.
///
/// # Example (Python)
/// ```python
/// import ritk
/// mesh = ritk.io.read_mesh("brain.obj")
/// pts = mesh.points          # numpy f32 array [N, 3]
/// print(mesh.n_points, mesh.n_cells)
/// ritk.io.write_mesh("brain_copy.ply", mesh)
/// ```
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub inner: VtkPolyData,
}

#[pymethods]
impl PyMesh {
    /// Point coordinates as a float32 numpy array of shape [N, 3].
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let mut data = Vec::with_capacity(self.inner.points.len() * 3);
        for p in &self.inner.points {
            data.push(p[0]);
            data.push(p[1]);
            data.push(p[2]);
        }
        Ok(PyArray1::<f32>::from_vec_bound(py, data)
            .reshape([self.inner.points.len(), 3])
            .map_err(|e| RitkPyError::runtime(e.to_string()))?)
    }

    /// Polygon connectivity as a Python list of lists of int indices.
    #[getter]
    fn polygons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let outer = PyList::empty_bound(py);
        for poly in &self.inner.polygons {
            let inner = PyList::empty_bound(py);
            for &idx in poly {
                inner.append(idx as i64)?;
            }
            outer.append(inner)?;
        }
        Ok(outer)
    }

    /// Per-point normals as a float32 numpy array [N, 3], or None if absent.
    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
        use ritk_io::AttributeArray;
        if let Some(AttributeArray::Normals { values }) = self.inner.point_data.get("Normals") {
            let n = values.len();
            let mut data = Vec::with_capacity(n * 3);
            for v in values {
                data.push(v[0]);
                data.push(v[1]);
                data.push(v[2]);
            }
            let arr = PyArray1::<f32>::from_vec_bound(py, data)
                .reshape([n, 3])
                .map_err(|e| RitkPyError::runtime(e.to_string()))?;
            Ok(Some(arr))
        } else {
            Ok(None)
        }
    }

    /// Number of points.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.points.len()
    }

    /// Total number of cells (all cell types combined).
    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.num_cells()
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh(n_points={}, n_cells={})",
            self.inner.points.len(),
            self.inner.num_cells()
        )
    }
}

// ── read_mesh ─────────────────────────────────────────────────────────────────

/// Read a polygonal mesh from file.
///
/// The format is inferred from the file extension.
///
/// Supported read formats: `.obj`, `.stl`, `.ply`, `.vtk` (polydata), `.vtp`.
///
/// Args:
///     path: File path (str).
///
/// Returns:
///     Mesh object with `points` (float32 ndarray \[N,3\]), `polygons` (list of lists),
///     and optionally `normals` (float32 ndarray \[N,3\]).
///
/// Raises:
///     IOError: on read failure or unsupported extension.
#[pyfunction]
pub fn read_mesh(py: Python<'_>, path: &str) -> RitkResult<PyMesh> {
    let path_owned = path.to_string();
    py.allow_threads(move || {
        let p = Path::new(&path_owned);
        let lower = path_owned.to_lowercase();
        let poly = if lower.ends_with(".obj") {
            read_obj_mesh(p).map_err(|e| RitkPyError::io(format!("OBJ read error: {e}")))?
        } else if lower.ends_with(".stl") {
            read_stl_mesh(p).map_err(|e| RitkPyError::io(format!("STL read error: {e}")))?
        } else if lower.ends_with(".ply") {
            read_ply_mesh(p).map_err(|e| RitkPyError::io(format!("PLY read error: {e}")))?
        } else if lower.ends_with(".vtk") {
            read_vtk_polydata(p).map_err(|e| RitkPyError::io(format!("VTK read error: {e}")))?
        } else if lower.ends_with(".vtp") {
            read_vtp_polydata(p).map_err(|e| RitkPyError::io(format!("VTP read error: {e}")))?
        } else {
            return Err(RitkPyError::io(format!(
                "Unsupported mesh extension in '{}'. Supported: \
                 .obj, .stl, .ply, .vtk (polydata), .vtp",
                path_owned
            )));
        };
        Ok(PyMesh { inner: poly })
    })
}

// ── write_mesh ────────────────────────────────────────────────────────────────

/// Write a polygonal mesh to file.
///
/// The format is inferred from the file extension.
/// - `.obj`  → OBJ ASCII
/// - `.stl`  → STL binary
/// - `.ply`  → PLY binary little-endian
/// - `.gltf` → glTF 2.0 JSON (geometry as base64 data URI)
/// - `.vtk`  → VTK legacy polydata
/// - `.vtp`  → VTK XML polydata
///
/// Args:
///     path: Destination file path (str).
///     mesh: Mesh object to write.
///
/// Raises:
///     IOError: on write failure or unsupported extension.
#[pyfunction]
pub fn write_mesh(py: Python<'_>, path: &str, mesh: &PyMesh) -> RitkResult<()> {
    let path_owned = path.to_string();
    let poly = mesh.inner.clone();
    py.allow_threads(move || {
        let p = Path::new(&path_owned);
        let lower = path_owned.to_lowercase();
        if lower.ends_with(".obj") {
            write_obj_mesh(p, &poly)
                .map_err(|e| RitkPyError::io(format!("OBJ write error: {e}")))?;
        } else if lower.ends_with(".stl") {
            write_stl_binary(p, &poly)
                .map_err(|e| RitkPyError::io(format!("STL write error: {e}")))?;
        } else if lower.ends_with(".stl.ascii") {
            write_stl_ascii(p, &poly)
                .map_err(|e| RitkPyError::io(format!("STL ASCII write error: {e}")))?;
        } else if lower.ends_with(".ply") {
            write_ply_binary_le(p, &poly)
                .map_err(|e| RitkPyError::io(format!("PLY write error: {e}")))?;
        } else if lower.ends_with(".ply.ascii") {
            write_ply_ascii(p, &poly)
                .map_err(|e| RitkPyError::io(format!("PLY ASCII write error: {e}")))?;
        } else if lower.ends_with(".gltf") {
            write_gltf(p, &poly).map_err(|e| RitkPyError::io(format!("glTF write error: {e}")))?;
        } else if lower.ends_with(".vtk") {
            write_vtk_polydata(p, &poly)
                .map_err(|e| RitkPyError::io(format!("VTK write error: {e}")))?;
        } else if lower.ends_with(".vtp") {
            write_vtp_polydata(p, &poly)
                .map_err(|e| RitkPyError::io(format!("VTP write error: {e}")))?;
        } else {
            return Err(RitkPyError::io(format!(
                "Unsupported mesh write extension in '{}'. Supported: \
                 .obj, .stl, .ply, .gltf, .vtk, .vtp",
                path_owned
            )));
        }
        Ok(())
    })
}
