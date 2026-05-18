use ritk_io::DicomReadMetadata;
use serde::{Deserialize, Serialize};

/// Geometry summary for display and validation.
///
/// The summary is intentionally modality-aware:
/// - CT may use DICOM metadata or image geometry for display.
/// - MRI preserves the same affine contract without CT-specific assumptions.
/// - Ultrasound requires acquisition-specific orientation handling and should
///   not be normalized through CT bed/table heuristics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometrySummary {
    /// Image dimensions.
    pub dimensions: [usize; 3],
    /// Voxel spacing derived from the loaded image geometry.
    pub spacing: [f64; 3],
    /// Image origin derived from the loaded image geometry.
    pub origin: [f64; 3],
    /// Direction matrix flattened in row-major display order derived from the loaded image geometry.
    pub direction: [f64; 9],
}

impl GeometrySummary {
    /// Build a geometry summary from DICOM metadata.
    pub fn from_dicom(metadata: &DicomReadMetadata) -> Self {
        Self {
            dimensions: metadata.dimensions,
            spacing: metadata.spacing,
            origin: metadata.origin,
            direction: metadata.direction,
        }
    }
}

/// Simple human-readable viewer status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewerStatus {
    /// Status message.
    pub message: String,
}

impl ViewerStatus {
    /// Create a new status message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Viewer operation result type.
pub type ViewerResult<T> = Result<T, anyhow::Error>;

/// Intensity display defaults derived from DICOM modality.
///
/// Window centre and width follow standard clinical display conventions:
///
/// | Modality | Centre | Width | Rationale |
/// |----------|---------|-------|--------------------------------------------------|
/// | CT | -400 HU | 1500 | Standard lung window; HU range [-1150, 350] |
/// | MR/MRI | 600 | 1200 | Relative intensity; typical brain soft-tissue |
/// | US | 128 | 256 | 8-bit acoustic impedance range [0, 255] |
/// | Default | 128 | 256 | Conservative unsigned 8-bit equivalent |
///
/// # Mathematical basis
/// For a window (c, w), the display range is [c − w/2, c + w/2].
/// CT lung: [-400 − 750, -400 + 750] = [-1150, 350] HU (standard lung protocol).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityDisplay {
    /// Window centre for intensity display.
    pub window_center: f64,
    /// Window width for intensity display.
    pub window_width: f64,
    /// Modality string used to select the defaults.
    pub modality: String,
}

impl ModalityDisplay {
    /// Return display defaults for the given DICOM modality string.
    ///
    /// Matching is exact and case-sensitive to preserve DICOM tag semantics.
    /// Unknown or absent modalities fall back to the 8-bit unsigned default.
    pub fn for_modality(modality: Option<&str>) -> Self {
        match modality {
            Some("CT") => Self {
                window_center: -400.0,
                window_width: 1500.0,
                modality: "CT".to_string(),
            },
            Some("MR") => Self {
                window_center: 600.0,
                window_width: 1200.0,
                modality: "MR".to_string(),
            },
            Some("MRI") => Self {
                window_center: 600.0,
                window_width: 1200.0,
                modality: "MRI".to_string(),
            },
            Some("US") => Self {
                window_center: 128.0,
                window_width: 256.0,
                modality: "US".to_string(),
            },
            other => Self {
                window_center: 128.0,
                window_width: 256.0,
                modality: other.unwrap_or("").to_string(),
            },
        }
    }
}
