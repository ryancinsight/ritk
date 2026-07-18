//! Domain types for DICOM Segmentation Storage.

/// SOP Class UID for Segmentation Storage.
pub const SEG_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.66.4";

/// Algorithm type for a segment (DICOM PS3.3 C.8.20.3, tag 0062,0008).
///
/// DICOM defines three exhaustive values for BINARY and FRACTIONAL segmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentAlgorithmType {
    /// Fully automatic algorithm.
    Automatic,
    /// Semi-automatic algorithm (some user interaction).
    SemiAutomatic,
    /// Manually drawn.
    Manual,
}

impl SegmentAlgorithmType {
    /// Parse from the DICOM CS string representation.
    pub fn from_dicom_str(s: &str) -> Self {
        match s.trim() {
            "AUTOMATIC" => Self::Automatic,
            "SEMIAUTOMATIC" => Self::SemiAutomatic,
            _ => Self::Manual,
        }
    }

    /// Return the canonical DICOM string for this type.
    pub fn as_dicom_str(&self) -> &str {
        match self {
            Self::Automatic => "AUTOMATIC",
            Self::SemiAutomatic => "SEMIAUTOMATIC",
            Self::Manual => "MANUAL",
        }
    }
}

/// Segmentation type (DICOM PS3.3 C.8.20.2, tag 0062,0001).
///
/// DICOM defines two exhaustive values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentationType {
    /// Each voxel is classified as belonging to a segment (1) or not (0).
    Binary,
    /// Each voxel stores a probability or partial-volume fraction in [0, 255].
    Fractional,
}

impl SegmentationType {
    /// Parse from the DICOM CS string representation, defaulting to Binary.
    pub fn from_dicom_str(s: &str) -> Self {
        match s.trim() {
            "FRACTIONAL" => Self::Fractional,
            _ => Self::Binary,
        }
    }

    /// Return the canonical DICOM string for this type.
    pub fn as_dicom_str(&self) -> &str {
        match self {
            Self::Binary => "BINARY",
            Self::Fractional => "FRACTIONAL",
        }
    }
}

/// Metadata for one segment label defined in the Segment Sequence (0062,0002).
#[derive(Debug, Clone)]
pub struct DicomSegmentInfo {
    /// SegmentNumber (0062,0004) US â€” 1-based segment index.
    pub segment_number: u16,
    /// SegmentLabel (0062,0005) LO.
    pub segment_label: String,
    /// SegmentDescription (0062,0006) ST, optional.
    pub segment_description: Option<String>,
    /// AlgorithmType (0062,0008) CS, optional.
    pub algorithm_type: Option<SegmentAlgorithmType>,
}

/// Complete in-memory representation of a DICOM-SEG object.
///
/// # Invariants
/// - `pixel_data.len() == n_frames`
/// - `pixel_data[f].len() == rows * cols` for all f
/// - For BINARY: pixel_data\[f\]\[i\] âˆˆ {0, 1}
/// - `frame_segment_numbers.len() == n_frames`
/// - `image_position_per_frame.len() == n_frames`
#[derive(Debug, Clone)]
pub struct DicomSegmentation {
    /// Pixel rows per frame (0028,0010).
    pub rows: usize,
    /// Pixel columns per frame (0028,0011).
    pub cols: usize,
    /// Number of frames (0028,0008); defaults to 1 when absent.
    pub n_frames: usize,
    /// BitsAllocated (0028,0100): 1 for BINARY, 8 for FRACTIONAL.
    pub bits_allocated: u16,
    /// SegmentationType (0062,0001): BINARY or FRACTIONAL.
    pub segmentation_type: SegmentationType,
    /// One entry per segment defined in SegmentSequence (0062,0002).
    pub segments: Vec<DicomSegmentInfo>,
    /// ReferencedSegmentNumber per frame; length == n_frames.
    pub frame_segment_numbers: Vec<u16>,
    /// Decoded pixel values per frame; each inner vec length == rows*cols.
    /// BINARY: 0 or 1. FRACTIONAL: raw byte values \[0, 255\].
    pub pixel_data: Vec<Vec<u8>>,
    /// ImagePositionPatient per frame from (5200,9230) â†’ (0020,9113) â†’ (0020,0032).
    pub image_position_per_frame: Vec<Option<[f64; 3]>>,
    /// ImageOrientationPatient from shared FG (5200,9229) â†’ (0020,9116) â†’ (0020,0037).
    pub image_orientation: Option<[f64; 6]>,
    /// PixelSpacing from shared FG (5200,9229) â†’ (0028,9110) â†’ (0028,0030).
    pub pixel_spacing: Option<[f64; 2]>,
    /// SliceThickness from shared FG (5200,9229) â†’ (0028,9110) â†’ (0018,0050).
    pub slice_thickness: Option<f64>,
}
