//! Python-exposed composite transform I/O (read/write JSON).

use crate::errors::{RitkPyError, RitkResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_core::transform::composite_io::{CompositeTransform, TransformDescription};

// ── read_transform ────────────────────────────────────────────────────────────

/// Read a composite transform from a JSON file.
///
/// Returns a dict with keys `"dimensionality"`, `"description"`, `"transforms"`.
#[pyfunction]
pub fn read_transform(py: Python<'_>, path: String) -> RitkResult<PyObject> {
    let composite = CompositeTransform::load_json(&path)
        .map_err(|e| RitkPyError::io(format!("Transform read error: {e}")))?;

    let transforms_list = PyList::empty_bound(py);
    for t in &composite.transforms {
        let d = PyDict::new_bound(py);
        match t {
            TransformDescription::Translation { offset } => {
                d.set_item("type", "translation")?;
                d.set_item("offset", offset.clone())?;
            }
            TransformDescription::Rigid { rotation, translation } => {
                d.set_item("type", "rigid")?;
                d.set_item("rotation", rotation.clone())?;
                d.set_item("translation", translation.clone())?;
            }
            TransformDescription::Affine { matrix } => {
                d.set_item("type", "affine")?;
                d.set_item("matrix", matrix.clone())?;
            }
            TransformDescription::DisplacementField { dims, origin, spacing, components } => {
                d.set_item("type", "displacement_field")?;
                d.set_item("dims", dims.clone())?;
                d.set_item("origin", origin.clone())?;
                d.set_item("spacing", spacing.clone())?;
                d.set_item("components", components.clone())?;
            }
            TransformDescription::BSpline {
                grid_dims, grid_origin, grid_spacing, components,
            } => {
                d.set_item("type", "bspline")?;
                d.set_item("grid_dims", grid_dims.clone())?;
                d.set_item("grid_origin", grid_origin.clone())?;
                d.set_item("grid_spacing", grid_spacing.clone())?;
                d.set_item("components", components.clone())?;
            }
        }
        transforms_list.append(d)?;
    }

    let result = PyDict::new_bound(py);
    result.set_item("dimensionality", composite.dimensionality)?;
    result.set_item("description", &composite.description)?;
    result.set_item("transforms", transforms_list)?;
    Ok(result.into())
}

// ── write_transform ───────────────────────────────────────────────────────────

/// Write a composite transform to a JSON file.
#[pyfunction]
#[pyo3(signature = (path, dimensionality, transforms, description=""))]
pub fn write_transform(
    py: Python<'_>,
    path: String,
    dimensionality: usize,
    transforms: Vec<PyObject>,
    description: &str,
) -> RitkResult<()> {
    let mut composite = CompositeTransform::new(dimensionality);
    composite.description = description.to_string();

    for obj in &transforms {
        let dict = obj
            .downcast_bound::<PyDict>(py)
            .map_err(|_| RitkPyError::io("each transform must be a dict"))?;

        let type_str: String = dict
            .get_item("type")?
            .ok_or_else(|| RitkPyError::io("transform dict missing 'type' key"))?
            .extract()?;

        let td = match type_str.as_str() {
            "translation" => {
                let offset: Vec<f64> = dict
                    .get_item("offset")?
                    .ok_or_else(|| RitkPyError::io("translation missing 'offset'"))?
                    .extract()?;
                TransformDescription::Translation { offset }
            }
            "rigid" => {
                let rotation: Vec<f64> = dict
                    .get_item("rotation")?
                    .ok_or_else(|| RitkPyError::io("rigid missing 'rotation'"))?
                    .extract()?;
                let translation: Vec<f64> = dict
                    .get_item("translation")?
                    .ok_or_else(|| RitkPyError::io("rigid missing 'translation'"))?
                    .extract()?;
                TransformDescription::Rigid { rotation, translation }
            }
            "affine" => {
                let matrix: Vec<f64> = dict
                    .get_item("matrix")?
                    .ok_or_else(|| RitkPyError::io("affine missing 'matrix'"))?
                    .extract()?;
                TransformDescription::Affine { matrix }
            }
            "displacement_field" => {
                let dims: Vec<usize> = dict
                    .get_item("dims")?
                    .ok_or_else(|| RitkPyError::io("displacement_field missing 'dims'"))?
                    .extract()?;
                let origin: Vec<f64> = dict
                    .get_item("origin")?
                    .ok_or_else(|| RitkPyError::io("displacement_field missing 'origin'"))?
                    .extract()?;
                let spacing: Vec<f64> = dict
                    .get_item("spacing")?
                    .ok_or_else(|| RitkPyError::io("displacement_field missing 'spacing'"))?
                    .extract()?;
                let components: Vec<Vec<f64>> = dict
                    .get_item("components")?
                    .ok_or_else(|| RitkPyError::io("displacement_field missing 'components'"))?
                    .extract()?;
                TransformDescription::DisplacementField { dims, origin, spacing, components }
            }
            "bspline" => {
                let grid_dims: Vec<usize> = dict
                    .get_item("grid_dims")?
                    .ok_or_else(|| RitkPyError::io("bspline missing 'grid_dims'"))?
                    .extract()?;
                let grid_origin: Vec<f64> = dict
                    .get_item("grid_origin")?
                    .ok_or_else(|| RitkPyError::io("bspline missing 'grid_origin'"))?
                    .extract()?;
                let grid_spacing: Vec<f64> = dict
                    .get_item("grid_spacing")?
                    .ok_or_else(|| RitkPyError::io("bspline missing 'grid_spacing'"))?
                    .extract()?;
                let components: Vec<Vec<f64>> = dict
                    .get_item("components")?
                    .ok_or_else(|| RitkPyError::io("bspline missing 'components'"))?
                    .extract()?;
                TransformDescription::BSpline {
                    grid_dims, grid_origin, grid_spacing, components,
                }
            }
            other => {
                return Err(RitkPyError::io(format!(
                    "unknown transform type '{other}'; expected: \
                     translation, rigid, affine, displacement_field, bspline"
                )));
            }
        };

        composite.push(td);
    }

    composite
        .save_json(&path)
        .map_err(|e| RitkPyError::io(format!("Transform write error: {e}")))
}
