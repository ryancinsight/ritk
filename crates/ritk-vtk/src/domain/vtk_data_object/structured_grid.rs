use super::AttributeArray;
use std::collections::HashMap;

/// VTK structured grid dataset (DATASET STRUCTURED_GRID).
///
/// Points indexed by (i,j,k): i in [0,nx), j in [0,ny), k in [0,nz).
/// n_points = nx*ny*nz; n_cells = max(nx-1,1)*max(ny-1,1)*max(nz-1,1).
#[derive(Debug, Clone, Default)]
pub struct VtkStructuredGrid {
    pub dimensions: [usize; 3],
    pub points: Vec<[f32; 3]>,
    pub point_data: HashMap<String, AttributeArray>,
    pub cell_data: HashMap<String, AttributeArray> }

impl VtkStructuredGrid {
    pub fn new(dimensions: [usize; 3]) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    pub fn n_points(&self) -> usize {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    pub fn n_cells(&self) -> usize {
        let [nx, ny, nz] = self.dimensions;
        nx.saturating_sub(1).max(1) * ny.saturating_sub(1).max(1) * nz.saturating_sub(1).max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = self.n_points();
        if self.points.len() != expected {
            return Err(format!(
                "VtkStructuredGrid: points.len()={} but dimensions {:?} require n_points={}",
                self.points.len(),
                self.dimensions,
                expected
            ));
        }
        Ok(())
    }
}
