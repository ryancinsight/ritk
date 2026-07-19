//! Threshold filter: extract dataset elements within a scalar range.
//!
//! # Mathematical Specification
//!
//! For `VtkImageData` with point-data scalar field S (â€–Sâ€– = n_points):
//!   Output âŠ† {(x, y, z, s) | S(x,y,z) âˆˆ [lower, upper]}
//! Each passing point becomes a `VtkCellType::Vertex` cell in a
//! `VtkUnstructuredGrid`, preserving its 3-D coordinate and scalar value.
//!
//! For `VtkUnstructuredGrid` with cell-data scalar field S:
//!   Output âŠ† {cell_i | S(cell_i) âˆˆ [lower, upper]}
//! All points are copied; only cells whose associated scalar passes are retained.
//!
//! Bounds are inclusive: `lower â‰¤ S â‰¤ upper`.

use crate::domain::mtime::{Modifiable, ModifiedTime};
use crate::domain::vtk_data_object::{
    AttributeArray, VtkCellType, VtkDataObject, VtkUnstructuredGrid,
};
use crate::domain::vtk_pipeline::VtkFilter;
use anyhow::{bail, Result};
use std::any::Any;

/// Threshold filter: retain only elements where a named scalar field is within
/// `[lower, upper]`.
///
/// Supported input variants: `VtkImageData` (point threshold) and
/// `VtkUnstructuredGrid` (cell threshold).
#[derive(Debug, Clone)]
pub struct ThresholdFilter {
    /// Name of the scalar field to threshold on.
    scalar_name: String,
    /// Inclusive lower bound.
    lower: f64,
    /// Inclusive upper bound.
    upper: f64,
    /// Modification timestamp; bumped on any parameter change.
    mtime: ModifiedTime,
}

impl ThresholdFilter {
    /// Construct a threshold filter for the given field and range.
    pub fn new(scalar_name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self {
            scalar_name: scalar_name.into(),
            lower,
            upper,
            mtime: ModifiedTime::tick(),
        }
    }

    /// Set the threshold range (inclusive lower and upper bounds).
    ///
    /// Bumps the modification time so that downstream pipeline stages
    /// detect the parameter change.
    pub fn set_range(&mut self, lower: f64, upper: f64) {
        self.lower = lower;
        self.upper = upper;
        self.modified();
    }

    /// Set the scalar field name.
    ///
    /// Bumps the modification time so that downstream pipeline stages
    /// detect the parameter change.
    pub fn set_scalar_name(&mut self, name: impl Into<String>) {
        self.scalar_name = name.into();
        self.modified();
    }

    /// Returns the scalar field name.
    pub fn scalar_name(&self) -> &str {
        &self.scalar_name
    }

    /// Returns the inclusive lower bound.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Returns the inclusive upper bound.
    pub fn upper(&self) -> f64 {
        self.upper
    }
}

impl Modifiable for ThresholdFilter {
    fn get_mtime(&self) -> ModifiedTime {
        self.mtime
    }

    fn modified(&mut self) {
        self.mtime = ModifiedTime::tick();
    }
}

impl VtkFilter for ThresholdFilter {
    fn mtime(&self) -> ModifiedTime {
        self.get_mtime()
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }

    fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
        match input {
            VtkDataObject::ImageData(img) => {
                let scalars = img.point_data.get(&self.scalar_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "ThresholdFilter: scalar field '{}' not found in ImageData point_data",
                        self.scalar_name
                    )
                })?;
                let values = match scalars {
                    AttributeArray::Scalars {
                        values,
                        num_components: 1,
                    } => values,
                    _ => bail!(
                        "ThresholdFilter: scalar field '{}' must have num_components=1",
                        self.scalar_name
                    ),
                };
                let e = &img.whole_extent;
                let nx = (e[1] - e[0] + 1) as usize;
                let ny = (e[3] - e[2] + 1) as usize;
                let nz = (e[5] - e[4] + 1) as usize;

                let mut out = VtkUnstructuredGrid::new();
                let mut out_scalars: Vec<f32> = Vec::new();

                // Narrow thresholds to f32 to match the precision of the stored scalars.
                // Comparing f32 cast to f64 against an f64 threshold introduces
                // representation error: 0.8_f32 as f64 = 0.800000011â€¦, which would
                // falsely exclude the upper boundary.
                let lo = self.lower as f32;
                let hi = self.upper as f32;

                for iz in 0..nz {
                    for iy in 0..ny {
                        for ix in 0..nx {
                            let flat = iz * ny * nx + iy * nx + ix;
                            let v = values[flat];
                            if v >= lo && v <= hi {
                                let x = img.origin[0] + (e[0] as f64 + ix as f64) * img.spacing[0];
                                let y = img.origin[1] + (e[2] as f64 + iy as f64) * img.spacing[1];
                                let z = img.origin[2] + (e[4] as f64 + iz as f64) * img.spacing[2];
                                let pt_idx = out.points.len() as u32;
                                out.points.push([x as f32, y as f32, z as f32]);
                                out.cells.push(vec![pt_idx]);
                                out.cell_types.push(VtkCellType::Vertex);
                                out_scalars.push(values[flat]);
                            }
                        }
                    }
                }

                out.cell_data.insert(
                    self.scalar_name.clone(),
                    AttributeArray::Scalars {
                        values: out_scalars,
                        num_components: 1,
                    },
                );
                Ok(VtkDataObject::UnstructuredGrid(out))
            }

            VtkDataObject::UnstructuredGrid(ug) => {
                let scalars = ug.cell_data.get(&self.scalar_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "ThresholdFilter: scalar field '{}' not found in UnstructuredGrid cell_data",
                        self.scalar_name
                    )
                })?;
                let values = match scalars {
                    AttributeArray::Scalars {
                        values,
                        num_components: 1,
                    } => values.clone(),
                    _ => bail!(
                        "ThresholdFilter: scalar field '{}' must have num_components=1",
                        self.scalar_name
                    ),
                };

                let mut out = VtkUnstructuredGrid::new();
                out.points = ug.points.clone();
                let mut out_scalars: Vec<f32> = Vec::new();

                let lo = self.lower as f32;
                let hi = self.upper as f32;

                for (i, (cell, ctype)) in ug.cells.iter().zip(ug.cell_types.iter()).enumerate() {
                    let v = values[i];
                    if v >= lo && v <= hi {
                        out.cells.push(cell.clone());
                        out.cell_types.push(*ctype);
                        out_scalars.push(values[i]);
                    }
                }

                out.cell_data.insert(
                    self.scalar_name.clone(),
                    AttributeArray::Scalars {
                        values: out_scalars,
                        num_components: 1,
                    },
                );
                Ok(VtkDataObject::UnstructuredGrid(out))
            }

            other => Err(anyhow::anyhow!(
                "ThresholdFilter requires ImageData or UnstructuredGrid input; received {}",
                crate::domain::filters::normals::data_object_type_name(&other)
            )),
        }
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_threshold.rs"]
mod tests;
