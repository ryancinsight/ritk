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
