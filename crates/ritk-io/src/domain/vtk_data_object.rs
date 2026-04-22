//! VTK data-object type hierarchy.
//!
//! # Mathematical Specification
//!
//! ## VtkPolyData
//! A polygonal mesh M = (V, C, A_P, A_C) where:
//! - V ⊂ ℝ³: ordered point set, |V| = n_points
//! - C: connectivity table partitioned into {vertices, lines, polygons, strips}
//!   Each cell c_i is a sequence of point indices from [0, n_points).
//! - A_P: named per-point attribute arrays, each of length n_points × ncomp.
//! - A_C: named per-cell attribute arrays, each of length n_cells × ncomp.
//!
//! ## Invariants
//! - All point indices in C are in [0, n_points).
//! - Attribute array lengths equal n_points × ncomp (point data) or
//!   n_cells × ncomp (cell data), where n_cells = |vertices|+|lines|+|polygons|+|strips|.
//!
//! # Reference
//! VTK File Formats (legacy), sections 4.1-4.6, Kitware Inc.

use std::collections::HashMap;

/// Attribute array attached to points or cells in a VTK dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeArray {
    /// Scalar field. `values.len() == n_elements * num_components`.
    Scalars {
        values: Vec<f32>,
        num_components: usize,
    },
    /// 3-component vector field. `values.len() == n_elements`.
    Vectors { values: Vec<[f32; 3]> },
    /// Unit-normal field. `values.len() == n_elements`.
    Normals { values: Vec<[f32; 3]> },
    /// Texture coordinate field. `values.len() == n_elements * dim`, `dim` in {1, 2, 3}.
    TextureCoords { values: Vec<f32>, dim: usize },
}

/// VTK polygonal mesh dataset (DATASET POLYDATA).
///
/// Invariant: every index stored in `vertices`, `lines`, `polygons`, and
/// `triangle_strips` is strictly less than `points.len()`.
#[derive(Debug, Clone, Default)]
pub struct VtkPolyData {
    /// Point coordinates [x, y, z] in mesh space.
    pub points: Vec<[f32; 3]>,
    /// VERTICES cells: each inner `Vec` is one vertex cell (list of point indices).
    pub vertices: Vec<Vec<u32>>,
    /// LINES cells: each inner `Vec` is an ordered sequence of point indices forming a polyline.
    pub lines: Vec<Vec<u32>>,
    /// POLYGONS cells: each inner `Vec` is a closed polygon (point indices).
    pub polygons: Vec<Vec<u32>>,
    /// TRIANGLE_STRIPS cells: each inner `Vec` is a triangle strip.
    pub triangle_strips: Vec<Vec<u32>>,
    /// Per-point attribute arrays keyed by name.
    pub point_data: HashMap<String, AttributeArray>,
    /// Per-cell attribute arrays keyed by name.
    pub cell_data: HashMap<String, AttributeArray>,
}

impl VtkPolyData {
    /// Total number of cells across all cell types.
    pub fn num_cells(&self) -> usize {
        self.vertices.len()
            + self.lines.len()
            + self.polygons.len()
            + self.triangle_strips.len()
    }

    /// Validate all index-bound and attribute-length invariants.
    ///
    /// Returns `Ok(())` if all invariants hold, otherwise a `String` describing
    /// the first violated invariant.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.points.len();

        let check_cells = |cells: &[Vec<u32>], kind: &str| -> Result<(), String> {
            for (i, cell) in cells.iter().enumerate() {
                for &idx in cell {
                    if idx as usize >= n {
                        return Err(format!(
                            "{} cell {}: index {} out of range (n_points = {})",
                            kind, i, idx, n
                        ));
                    }
                }
            }
            Ok(())
        };
        check_cells(&self.vertices, "VERTICES")?;
        check_cells(&self.lines, "LINES")?;
        check_cells(&self.polygons, "POLYGONS")?;
        check_cells(&self.triangle_strips, "TRIANGLE_STRIPS")?;

        for (name, attr) in &self.point_data {
            match attr {
                AttributeArray::Scalars { values, num_components } => {
                    let expected = n * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "point_data '{}': Scalars length {} != n_points*ncomp = {}",
                            name, values.len(), expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != n {
                        return Err(format!(
                            "point_data '{}': length {} != n_points = {}",
                            name, values.len(), n
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != n * dim {
                        return Err(format!(
                            "point_data '{}': TextureCoords length {} != n_points*dim = {}",
                            name, values.len(), n * dim
                        ));
                    }
                }
            }
        }

        let nc = self.num_cells();
        for (name, attr) in &self.cell_data {
            match attr {
                AttributeArray::Scalars { values, num_components } => {
                    let expected = nc * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "cell_data '{}': Scalars length {} != n_cells*ncomp = {}",
                            name, values.len(), expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != nc {
                        return Err(format!(
                            "cell_data '{}': length {} != n_cells = {}",
                            name, values.len(), nc
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != nc * dim {
                        return Err(format!(
                            "cell_data '{}': TextureCoords length {} != n_cells*dim = {}",
                            name, values.len(), nc * dim
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Top-level VTK data object discriminating between supported dataset types.
#[derive(Debug, Clone)]
pub enum VtkDataObject {
    /// Polygonal mesh (DATASET POLYDATA).
    PolyData(VtkPolyData),
    /// Structured grid dataset (DATASET STRUCTURED_GRID).
    StructuredGrid(VtkStructuredGrid),
    /// Unstructured grid dataset (DATASET UNSTRUCTURED_GRID).
    UnstructuredGrid(VtkUnstructuredGrid),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// VTK structured grid dataset (DATASET STRUCTURED_GRID).
///
/// Points indexed by (i,j,k): i in [0,nx), j in [0,ny), k in [0,nz).
/// n_points = nx*ny*nz; n_cells = max(nx-1,1)*max(ny-1,1)*max(nz-1,1).
#[derive(Debug, Clone, Default)]
pub struct VtkStructuredGrid {
    pub dimensions: [usize; 3],
    pub points: Vec<[f32; 3]>,
    pub point_data: HashMap<String, AttributeArray>,
    pub cell_data: HashMap<String, AttributeArray>,
}
impl VtkStructuredGrid {
    pub fn new(dimensions: [usize; 3]) -> Self {
        Self { dimensions, ..Default::default() }
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
                self.points.len(), self.dimensions, expected));
        }
        Ok(())
    }
}

/// VTK unstructured grid dataset (DATASET UNSTRUCTURED_GRID).
///
/// Cells are arbitrary polyhedra; each cell is a list of point indices.
/// cell_types.len() must equal cells.len(); every index in every cell
/// must be in [0, n_points).
#[derive(Debug, Clone, Default)]
pub struct VtkUnstructuredGrid {
    pub points: Vec<[f32; 3]>,
    pub cells: Vec<Vec<u32>>,
    pub cell_types: Vec<u8>,
    pub point_data: HashMap<String, AttributeArray>,
    pub cell_data: HashMap<String, AttributeArray>,
}
impl VtkUnstructuredGrid {
    pub fn new() -> Self { Self::default() }
    pub fn n_points(&self) -> usize { self.points.len() }
    pub fn n_cells(&self) -> usize { self.cells.len() }
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.len() != self.cell_types.len() {
            return Err(format!(
                "VtkUnstructuredGrid: cells.len()={} != cell_types.len()={}",
                self.cells.len(), self.cell_types.len()));
        }
        let np = self.n_points();
        for (ci, cell) in self.cells.iter().enumerate() {
            for &idx in cell {
                if idx as usize >= np {
                    return Err(format!(
                        "VtkUnstructuredGrid: cell {} index {} out of range (n_points={})",
                        ci, idx, np));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> VtkPolyData {
        VtkPolyData {
            points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            polygons: vec![vec![0, 1, 2]],
            ..Default::default()
        }
    }

    #[test]
    fn test_vtk_poly_data_default_empty() {
        let p = VtkPolyData::default();
        assert!(p.points.is_empty());
        assert!(p.polygons.is_empty());
        assert_eq!(p.num_cells(), 0);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_vtk_poly_data_validate_ok() {
        assert_eq!(triangle().validate(), Ok(()));
    }

    #[test]
    fn test_vtk_poly_data_validate_out_of_range() {
        let mut p = triangle();
        p.polygons[0].push(99); // index 99 does not exist
        assert!(p.validate().is_err());
        let msg = p.validate().unwrap_err();
        assert!(msg.contains("POLYGONS"), "error must name the cell type");
    }

    #[test]
    fn test_vtk_poly_data_validate_scalar_length() {
        let mut p = triangle();
        // n_points = 3, ncomp = 1, expected length = 3; supply 2 -> error.
        p.point_data.insert(
            "intensity".to_string(),
            AttributeArray::Scalars { values: vec![1.0, 2.0], num_components: 1 },
        );
        let result = p.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("intensity"));
    }

    #[test]
    fn test_vtk_data_object_polydata_wraps() {
        let obj = VtkDataObject::PolyData(triangle());
        match obj {
            VtkDataObject::PolyData(p) => {
                assert_eq!(p.points.len(), 3);
                assert_eq!(p.polygons.len(), 1);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_vtk_poly_data_num_cells_multi_type() {
        let p = VtkPolyData {
            points: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            vertices: vec![vec![0], vec![1]],
            lines: vec![vec![0, 1]],
            polygons: vec![vec![0, 1, 2]],
            triangle_strips: vec![vec![0, 1, 2]],
            ..Default::default()
        };
        assert_eq!(p.num_cells(), 5);
    }

    #[test]
    fn test_vtk_poly_data_validate_cell_data_scalar_ok() {
        let mut p = VtkPolyData {
            points: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            polygons: vec![vec![0, 1, 2]], // 1 cell
            ..Default::default()
        };
        p.cell_data.insert(
            "pressure".to_string(),
            AttributeArray::Scalars { values: vec![42.0], num_components: 1 },
        );
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_attribute_array_equality() {
        let a = AttributeArray::Scalars { values: vec![1.0, 2.0], num_components: 1 };
        let b = AttributeArray::Scalars { values: vec![1.0, 2.0], num_components: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_vtk_structured_grid_n_points() {
        assert_eq!(VtkStructuredGrid::new([2, 3, 4]).n_points(), 24);
    }
    #[test]
    fn test_vtk_structured_grid_n_cells() {
        assert_eq!(VtkStructuredGrid::new([2, 3, 4]).n_cells(), 6);
    }
    #[test]
    fn test_vtk_structured_grid_validate_ok() {
        let mut g = VtkStructuredGrid::new([2, 2, 2]);
        g.points = vec![[0.0; 3]; 8];
        assert!(g.validate().is_ok());
    }
    #[test]
    fn test_vtk_structured_grid_validate_wrong_len() {
        let mut g = VtkStructuredGrid::new([2, 2, 2]);
        g.points = vec![[0.0; 3]; 5];
        assert!(g.validate().unwrap_err().contains("n_points"));
    }
    #[test]
    fn test_vtk_unstructured_grid_validate_ok() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0; 3]; 4];
        g.cells = vec![vec![0, 1, 2, 3]];
        g.cell_types = vec![10];
        assert!(g.validate().is_ok());
    }


    #[test]
    fn test_vtk_unstructured_grid_cell_type_mismatch() {
        let mut g = VtkUnstructuredGrid::new();
        g.cells = vec![vec![0], vec![1]];
        g.cell_types = vec![5];
        assert!(g.validate().unwrap_err().contains("cell_types"));
    }
    #[test]
    fn test_vtk_unstructured_grid_index_out_of_range() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0; 3]; 3];
        g.cells = vec![vec![0, 1, 99]];
        g.cell_types = vec![5];
        assert!(g.validate().unwrap_err().contains("99"));
    }
    #[test]
    fn test_vtk_data_object_structured_variant() {
        match VtkDataObject::StructuredGrid(VtkStructuredGrid::new([2, 2, 2])) {
            VtkDataObject::StructuredGrid(g) => assert_eq!(g.n_points(), 8),
            _ => panic!("wrong variant"),
        }
    }
    #[test]
    fn test_vtk_data_object_unstructured_variant() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[1.0, 0.0, 0.0]];
        match VtkDataObject::UnstructuredGrid(g) {
            VtkDataObject::UnstructuredGrid(g) => assert_eq!(g.n_points(), 1),
            _ => panic!("wrong variant"),
        }
    }
    #[test]
    fn test_vtk_unstructured_grid_default_is_empty() {
        let g = VtkUnstructuredGrid::default();
        assert_eq!(g.n_points(), 0);
        assert_eq!(g.n_cells(), 0);
        assert!(g.validate().is_ok());
    }

}