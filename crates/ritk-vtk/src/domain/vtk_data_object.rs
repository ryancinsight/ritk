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
        self.vertices.len() + self.lines.len() + self.polygons.len() + self.triangle_strips.len()
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
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = n * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "point_data '{}': Scalars length {} != n_points*ncomp = {}",
                            name,
                            values.len(),
                            expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != n {
                        return Err(format!(
                            "point_data '{}': length {} != n_points = {}",
                            name,
                            values.len(),
                            n
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != n * dim {
                        return Err(format!(
                            "point_data '{}': TextureCoords length {} != n_points*dim = {}",
                            name,
                            values.len(),
                            n * dim
                        ));
                    }
                }
            }
        }

        let nc = self.num_cells();
        for (name, attr) in &self.cell_data {
            match attr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = nc * num_components;
                    if values.len() != expected {
                        return Err(format!(
                            "cell_data '{}': Scalars length {} != n_cells*ncomp = {}",
                            name,
                            values.len(),
                            expected
                        ));
                    }
                }
                AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
                    if values.len() != nc {
                        return Err(format!(
                            "cell_data '{}': length {} != n_cells = {}",
                            name,
                            values.len(),
                            nc
                        ));
                    }
                }
                AttributeArray::TextureCoords { values, dim } => {
                    if values.len() != nc * dim {
                        return Err(format!(
                            "cell_data '{}': TextureCoords length {} != n_cells*dim = {}",
                            name,
                            values.len(),
                            nc * dim
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
    /// Regular Cartesian image data (.vti).
    ImageData(VtkImageData),
}

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

/// VTK cell type codes per VTK File Formats specification (Table 2, Kitware Inc.).
///
/// # Invariants
/// - `VtkCellType::to_u8()` returns the canonical VTK integer cell type code.
/// - `VtkCellType::from_u8(v)` is the left inverse of `to_u8` for all known codes.
/// - Unknown codes return `None` from `from_u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VtkCellType {
    Vertex = 1,
    PolyVertex = 2,
    Line = 3,
    PolyLine = 4,
    Triangle = 5,
    TriangleStrip = 6,
    Polygon = 7,
    Pixel = 8,
    Quad = 9,
    Tetra = 10,
    Voxel = 11,
    Hexahedron = 12,
    Wedge = 13,
    Pyramid = 14,
    PentagonalPrism = 15,
    HexagonalPrism = 16,
    QuadraticEdge = 21,
    QuadraticTriangle = 22,
    QuadraticQuad = 23,
    QuadraticTetra = 24,
    QuadraticHexahedron = 25,
    QuadraticWedge = 26,
    QuadraticPyramid = 27,
    BiquadraticQuad = 28,
    TriquadraticHexahedron = 29,
    QuadraticLinearQuad = 30,
    QuadraticLinearWedge = 31,
    BiquadraticQuadraticWedge = 32,
    BiquadraticQuadraticHexahedron = 33,
    BilinearQuadraticWedge = 34,
}

impl VtkCellType {
    /// Return the canonical VTK integer cell type code.
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Parse a VTK integer cell type code.
    ///
    /// Returns `None` when the code does not correspond to a known VTK cell type.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Vertex),
            2 => Some(Self::PolyVertex),
            3 => Some(Self::Line),
            4 => Some(Self::PolyLine),
            5 => Some(Self::Triangle),
            6 => Some(Self::TriangleStrip),
            7 => Some(Self::Polygon),
            8 => Some(Self::Pixel),
            9 => Some(Self::Quad),
            10 => Some(Self::Tetra),
            11 => Some(Self::Voxel),
            12 => Some(Self::Hexahedron),
            13 => Some(Self::Wedge),
            14 => Some(Self::Pyramid),
            15 => Some(Self::PentagonalPrism),
            16 => Some(Self::HexagonalPrism),
            21 => Some(Self::QuadraticEdge),
            22 => Some(Self::QuadraticTriangle),
            23 => Some(Self::QuadraticQuad),
            24 => Some(Self::QuadraticTetra),
            25 => Some(Self::QuadraticHexahedron),
            26 => Some(Self::QuadraticWedge),
            27 => Some(Self::QuadraticPyramid),
            28 => Some(Self::BiquadraticQuad),
            29 => Some(Self::TriquadraticHexahedron),
            30 => Some(Self::QuadraticLinearQuad),
            31 => Some(Self::QuadraticLinearWedge),
            32 => Some(Self::BiquadraticQuadraticWedge),
            33 => Some(Self::BiquadraticQuadraticHexahedron),
            34 => Some(Self::BilinearQuadraticWedge),
            _ => None,
        }
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

/// VTK ImageData domain object — a regular Cartesian grid of scalar/vector/tensor fields.
///
/// # Invariants (mathematical)
/// - `whole_extent[2i+1] >= whole_extent[2i]` for i in {0,1,2}.
/// - `n_points() = product over i of (whole_extent[2i+1] - whole_extent[2i] + 1)`.
/// - `n_cells() = product over i of max(1, whole_extent[2i+1] - whole_extent[2i])`.
/// - Each `AttributeArray::Scalars` in `point_data` has `values.len() == n_points() * num_components`.
/// - Each `AttributeArray::Scalars` in `cell_data` has `values.len() == n_cells() * num_components`.
/// - Each `AttributeArray::Vectors` in `point_data` has `values.len() == n_points()`.
#[derive(Debug, Clone, Default)]
pub struct VtkImageData {
    /// `[x0, x1, y0, y1, z0, z1]` — inclusive extent indices (VTK convention).
    pub whole_extent: [i64; 6],
    /// Physical origin of the dataset (coordinate of the first point).
    pub origin: [f64; 3],
    /// Voxel spacing in each axis direction.
    pub spacing: [f64; 3],
    /// Named point-centered attribute arrays, keyed by name.
    pub point_data: HashMap<String, AttributeArray>,
    /// Named cell-centered attribute arrays, keyed by name.
    pub cell_data: HashMap<String, AttributeArray>,
}

impl VtkImageData {
    /// Number of points = product of (extent_max - extent_min + 1) over all 3 axes.
    pub fn n_points(&self) -> usize {
        let e = &self.whole_extent;
        ((e[1] - e[0] + 1) as usize) * ((e[3] - e[2] + 1) as usize) * ((e[5] - e[4] + 1) as usize)
    }

    /// Number of cells = product of max(1, extent_max - extent_min) over all 3 axes.
    pub fn n_cells(&self) -> usize {
        let e = &self.whole_extent;
        (1_usize.max((e[1] - e[0]) as usize))
            * (1_usize.max((e[3] - e[2]) as usize))
            * (1_usize.max((e[5] - e[4]) as usize))
    }

    /// Validate all invariants.
    ///
    /// Returns `Err` with a description of the first violation found.
    pub fn validate(&self) -> Result<(), String> {
        let e = &self.whole_extent;
        for i in 0..3 {
            if e[2 * i + 1] < e[2 * i] {
                return Err(format!(
                    "whole_extent[{}] < whole_extent[{}]: {} < {}",
                    2 * i + 1,
                    2 * i,
                    e[2 * i + 1],
                    e[2 * i]
                ));
            }
        }
        let np = self.n_points();
        for (name, arr) in &self.point_data {
            match arr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = np * (*num_components as usize);
                    if values.len() != expected {
                        return Err(format!(
                            "point_data '{}': expected {} values, got {}",
                            name,
                            expected,
                            values.len()
                        ));
                    }
                }
                AttributeArray::Vectors { values } => {
                    // values: Vec<[f32; 3]> — one 3-vector per point.
                    if values.len() != np {
                        return Err(format!(
                            "point_data vectors '{}': expected {} vectors, got {}",
                            name,
                            np,
                            values.len()
                        ));
                    }
                }
                _ => {}
            }
        }
        let nc = self.n_cells();
        for (name, arr) in &self.cell_data {
            match arr {
                AttributeArray::Scalars {
                    values,
                    num_components,
                } => {
                    let expected = nc * (*num_components as usize);
                    if values.len() != expected {
                        return Err(format!(
                            "cell_data '{}': expected {} values, got {}",
                            name,
                            expected,
                            values.len()
                        ));
                    }
                }
                _ => {}
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
            AttributeArray::Scalars {
                values: vec![1.0, 2.0],
                num_components: 1,
            },
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
            AttributeArray::Scalars {
                values: vec![42.0],
                num_components: 1,
            },
        );
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_attribute_array_equality() {
        let a = AttributeArray::Scalars {
            values: vec![1.0, 2.0],
            num_components: 1,
        };
        let b = AttributeArray::Scalars {
            values: vec![1.0, 2.0],
            num_components: 1,
        };
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
        g.cell_types = vec![VtkCellType::Tetra];
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_vtk_unstructured_grid_cell_type_mismatch() {
        let mut g = VtkUnstructuredGrid::new();
        g.cells = vec![vec![0], vec![1]];
        g.cell_types = vec![VtkCellType::Triangle];
        assert!(g.validate().unwrap_err().contains("cell_types"));
    }
    #[test]
    fn test_vtk_unstructured_grid_index_out_of_range() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0; 3]; 3];
        g.cells = vec![vec![0, 1, 99]];
        g.cell_types = vec![VtkCellType::Triangle];
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

    #[test]
    fn test_vtk_cell_type_round_trip_all_known() {
        let known: &[(u8, VtkCellType)] = &[
            (1, VtkCellType::Vertex),
            (5, VtkCellType::Triangle),
            (10, VtkCellType::Tetra),
            (12, VtkCellType::Hexahedron),
            (21, VtkCellType::QuadraticEdge),
            (34, VtkCellType::BilinearQuadraticWedge),
        ];
        for &(code, variant) in known {
            assert_eq!(
                VtkCellType::from_u8(code),
                Some(variant),
                "from_u8({code}) must return Some({variant:?})"
            );
            assert_eq!(
                variant.to_u8(),
                code,
                "{variant:?}.to_u8() must return {code}"
            );
        }
    }

    #[test]
    fn test_vtk_cell_type_unknown_returns_none() {
        for v in [0u8, 17, 18, 19, 20, 35, 200, 255] {
            assert_eq!(
                VtkCellType::from_u8(v),
                None,
                "from_u8({v}) must return None for unknown code"
            );
        }
    }

    #[test]
    fn test_vtk_image_data_n_points_and_cells() {
        // WholeExtent "0 2 0 3 0 4":
        // n_points = (2-0+1) * (3-0+1) * (4-0+1) = 3 * 4 * 5 = 60
        // n_cells  = max(1,2) * max(1,3) * max(1,4) = 2 * 3 * 4 = 24
        let img = VtkImageData {
            whole_extent: [0, 2, 0, 3, 0, 4],
            ..Default::default()
        };
        assert_eq!(img.n_points(), 60);
        assert_eq!(img.n_cells(), 24);
    }

    #[test]
    fn test_vtk_image_data_validate_ok() {
        // extent [0,1,0,1,0,1] → n_points = 2*2*2 = 8
        let mut img = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            ..Default::default()
        };
        img.point_data.insert(
            "s".to_string(),
            AttributeArray::Scalars {
                values: vec![0.0f32; 8],
                num_components: 1,
            },
        );
        assert_eq!(img.validate(), Ok(()));
    }

    #[test]
    fn test_vtk_image_data_validate_wrong_scalar_len() {
        // extent [0,1,0,1,0,1] → n_points = 8; supplying 5 → Err
        let mut img = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            ..Default::default()
        };
        img.point_data.insert(
            "s".to_string(),
            AttributeArray::Scalars {
                values: vec![0.0f32; 5],
                num_components: 1,
            },
        );
        let r = img.validate();
        assert!(r.is_err(), "mismatched scalar length must return Err");
        assert!(
            r.unwrap_err().contains("s"),
            "error message must name the field"
        );
    }

    #[test]
    fn test_vtk_image_data_data_object_variant() {
        // extent [0,2,0,3,0,4] → n_points = 60
        let img = VtkImageData {
            whole_extent: [0, 2, 0, 3, 0, 4],
            ..Default::default()
        };
        match VtkDataObject::ImageData(img) {
            VtkDataObject::ImageData(g) => assert_eq!(g.n_points(), 60),
            _ => panic!("wrong VtkDataObject variant"),
        }
    }
}
