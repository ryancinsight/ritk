use std::collections::HashMap;

use gaia::Polyline;

/// The parsed text header of a `.tck` file.
///
/// All key-value pairs from the header section are stored in the `fields`
/// map.  Well-known keys are additionally parsed into typed fields.
#[derive(Debug, Clone, PartialEq)]
pub struct TckHeader {
    /// All raw key-value pairs from the header, in original order.
    pub fields: HashMap<String, String>,

    /// Streamline count declared in the header (`count` key).
    pub count: Option<i64>,

    /// Total streamline count (`total_count` key), if present.
    pub total_count: Option<i64>,

    /// Binary data type, e.g. `"Float32LE"`.
    pub datatype: TckDatatype,

    /// `4×4` row-major transform matrix from the `transform` key, if
    /// present.  Maps voxel→scanner space.
    pub transform: Option<[[f64; 4]; 4]>,

    /// MRtrix version string (`mrtrix_version` key).
    pub mrtrix_version: Option<String>,

    /// File path stored in the header (`file` key).
    pub file_path: Option<String>,

    /// Free-form comments (`comments` key).
    pub comments: Option<String>,
}

impl Default for TckHeader {
    fn default() -> Self {
        Self {
            fields: HashMap::new(),
            count: None,
            total_count: None,
            datatype: TckDatatype::Float32LE,
            transform: None,
            mrtrix_version: None,
            file_path: None,
            comments: None,
        }
    }
}

/// Supported binary datatypes in a `.tck` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TckDatatype {
    /// 32-bit little-endian float.
    Float32LE,
    /// 32-bit big-endian float.
    Float32BE,
    /// 64-bit little-endian float.
    Float64LE,
    /// 64-bit big-endian float.
    Float64BE,
}

impl TckDatatype {
    /// Number of bytes per point (3 scalars × byte width).
    pub(crate) fn bytes_per_point(self) -> usize {
        match self {
            TckDatatype::Float32LE | TckDatatype::Float32BE => 12,
            TckDatatype::Float64LE | TckDatatype::Float64BE => 24,
        }
    }

    /// Parse a `datatype` header value.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "Float32LE" => Some(TckDatatype::Float32LE),
            "Float32BE" => Some(TckDatatype::Float32BE),
            "Float64LE" => Some(TckDatatype::Float64LE),
            "Float64BE" => Some(TckDatatype::Float64BE),
            _ => None,
        }
    }

    /// Canonical string for the header.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            TckDatatype::Float32LE => "Float32LE",
            TckDatatype::Float32BE => "Float32BE",
            TckDatatype::Float64LE => "Float64LE",
            TckDatatype::Float64BE => "Float64BE",
        }
    }
}

/// The full contents of a `.tck` file.
///
/// Streamlines are stored in scanner-space millimetre coordinates.
#[derive(Debug, Clone)]
pub struct TckTractogram {
    /// Parsed header metadata.
    pub header: TckHeader,

    /// Streamlines in file order, each a valid Gaia polyline.
    pub streamlines: Vec<Polyline<f64>>,
}
