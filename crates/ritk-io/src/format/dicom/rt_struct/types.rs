//! Domain types for RT Structure Set Storage.

/// SOP Class UID for RT Structure Set Storage.
pub const RT_STRUCT_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.3";

/// Geometric type of an RT contour (DICOM tag 3006,0042).
///
/// The type determines how `points` are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourGeometricType {
    /// Single control point.
    Point,
    /// Ordered polyline lying on a plane; the last point does NOT connect to the first.
    OpenPlanar,
    /// Closed polygon lying on a plane; the last point implicitly connects to the first.
    ClosedPlanar,
}

impl ContourGeometricType {
    /// Parse from the DICOM string representation.
    pub fn from_dicom_str(s: &str) -> Option<Self> {
        match s.trim() {
            "POINT" => Some(Self::Point),
            "OPEN_PLANAR" => Some(Self::OpenPlanar),
            "CLOSED_PLANAR" => Some(Self::ClosedPlanar),
            _ => None,
        }
    }

    /// Return the canonical DICOM string for this type.
    pub fn as_dicom_str(self) -> &'static str {
        match self {
            Self::Point => "POINT",
            Self::OpenPlanar => "OPEN_PLANAR",
            Self::ClosedPlanar => "CLOSED_PLANAR",
        }
    }
}

/// A single contour slice within an RT ROI.
///
/// # Mathematical specification
///
/// A contour is a sequence of N ≥ 1 points in patient coordinate space (mm):
/// - `CLOSED_PLANAR`: N points lie on one plane; the polygon is implicitly closed.
/// - `OPEN_PLANAR`:   N ≥ 2 points on one plane; the polyline is open.
/// - `POINT`:         N = 1 control point.
///
/// Encoding: `X₀\Y₀\Z₀\X₁\Y₁\Z₁\…\X_{N-1}\Y_{N-1}\Z_{N-1}`
#[derive(Debug, Clone)]
pub struct RtContour {
    /// Geometric type of this contour.
    pub geometric_type: ContourGeometricType,
    /// Patient-coordinate points `[X, Y, Z]` in mm.
    pub points: Vec<[f64; 3]>,
}

/// ROI metadata and contours extracted from an RT Structure Set.
#[derive(Debug, Clone)]
pub struct RtRoiInfo {
    /// Unique ROI identifier within the structure set.
    pub roi_number: u32,
    /// Human-readable name.
    pub roi_name: String,
    /// Optional free-text description.
    pub roi_description: Option<String>,
    /// RT ROI Interpreted Type from `(3006,00A4)`, e.g. `"GTV"`, `"CTV"`, `"PTV"`.
    pub roi_interpreted_type: Option<String>,
    /// Display color `[R, G, B]` from `(3006,002A)`.
    pub display_color: Option<[u8; 3]>,
    /// Contour slices.
    pub contours: Vec<RtContour>,
}

/// Parsed representation of a DICOM RT Structure Set file.
#[derive(Debug, Clone)]
pub struct RtStructureSet {
    /// Structure Set Label `(3006,0002)`.
    pub structure_set_label: String,
    /// Structure Set Name `(3006,0004)` — optional.
    pub structure_set_name: Option<String>,
    /// ROIs sorted ascending by `roi_number`.
    pub rois: Vec<RtRoiInfo>,
}
