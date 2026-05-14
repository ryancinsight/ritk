use std::collections::HashMap;
use super::{AttributeArray, VtkCellType};

/// VTK unstructured grid dataset (DATASET UNSTRUCTURED_GRID).
///
/// Cells are arbitrary polyhedra; each cell is a list of point indices.
/// cell_types.len() must equal cells.len(); every index in every cell
/// must be in [0, n_points).
#[derive(Debug, Clone, Default)]
pub struct VtkUnstructuredGrid {
    pub points: Vec<[f32; 3]>,
    pub cells: Vec<Vec<u32>>,
    pub cell_types: Vec<VtkCellType>,
    pub point_data: HashMap<String, AttributeArray>,
    pub cell_data: HashMap<String, AttributeArray>,
}

impl VtkUnstructuredGrid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.cells.len() != self.cell_types.len() {
            return Err(format!(
                "VtkUnstructuredGrid: cells.len()={} != cell_types.len()={}",
                self.cells.len(),
                self.cell_types.len()
            ));
        }
        let np = self.n_points();
        for (ci, cell) in self.cells.iter().enumerate() {
            for &idx in cell {
                if idx as usize >= np {
                    return Err(format!(
                        "VtkUnstructuredGrid: cell {} index {} out of range (n_points={})",
                        ci, idx, np
                    ));
                }
            }
        }
        Ok(())
    }
}
