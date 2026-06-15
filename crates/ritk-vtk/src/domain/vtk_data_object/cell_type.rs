/// VTK cell type codes per VTK File Formats specification (Table 2, Kitware Inc.).
///
/// # Invariants
/// - `u8::from(cell_type)` returns the canonical VTK integer cell type code.
/// - `VtkCellType::try_from(v)` is the left inverse of `u8::from` for all known codes.
/// - Unknown codes return `Err` from `try_from`.
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

/// Error returned when a `u8` value does not correspond to a known VTK cell type code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownVtkCellType(pub u8);

impl std::fmt::Display for UnknownVtkCellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown VTK cell type code: {}", self.0)
    }
}

impl std::error::Error for UnknownVtkCellType {}

impl From<VtkCellType> for u8 {
    /// Return the canonical VTK integer cell type code.
    #[inline]
    fn from(ct: VtkCellType) -> u8 {
        ct as u8
    }
}

impl TryFrom<u8> for VtkCellType {
    type Error = UnknownVtkCellType;

    /// Parse a VTK integer cell type code.
    ///
    /// Returns `Err(UnknownVtkCellType)` when the code does not correspond to
    /// a known VTK cell type.
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Self::Vertex),
            2 => Ok(Self::PolyVertex),
            3 => Ok(Self::Line),
            4 => Ok(Self::PolyLine),
            5 => Ok(Self::Triangle),
            6 => Ok(Self::TriangleStrip),
            7 => Ok(Self::Polygon),
            8 => Ok(Self::Pixel),
            9 => Ok(Self::Quad),
            10 => Ok(Self::Tetra),
            11 => Ok(Self::Voxel),
            12 => Ok(Self::Hexahedron),
            13 => Ok(Self::Wedge),
            14 => Ok(Self::Pyramid),
            15 => Ok(Self::PentagonalPrism),
            16 => Ok(Self::HexagonalPrism),
            21 => Ok(Self::QuadraticEdge),
            22 => Ok(Self::QuadraticTriangle),
            23 => Ok(Self::QuadraticQuad),
            24 => Ok(Self::QuadraticTetra),
            25 => Ok(Self::QuadraticHexahedron),
            26 => Ok(Self::QuadraticWedge),
            27 => Ok(Self::QuadraticPyramid),
            28 => Ok(Self::BiquadraticQuad),
            29 => Ok(Self::TriquadraticHexahedron),
            30 => Ok(Self::QuadraticLinearQuad),
            31 => Ok(Self::QuadraticLinearWedge),
            32 => Ok(Self::BiquadraticQuadraticWedge),
            33 => Ok(Self::BiquadraticQuadraticHexahedron),
            34 => Ok(Self::BilinearQuadraticWedge),
            _ => Err(UnknownVtkCellType(v)),
        }
    }
}
