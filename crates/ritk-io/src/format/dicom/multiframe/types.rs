//! Public data types for multi-frame DICOM reader and writer.

use crate::format::dicom::reader::types::literal_arraystring;
use arrayvec::ArrayString;
use ritk_dicom::PixelSignedness;
use std::path::PathBuf;

/// SOP Class UID for Multi-Frame Grayscale Word Secondary Capture Image Storage.
pub(crate) const MF_GRAYSCALE_WORD_SC_UID: &str = "1.2.840.10008.5.1.4.1.1.7.3";

/// Per-frame spatial and photometric metadata extracted from DICOM Enhanced Multiframe
/// per-frame functional group sequence (5200,9230) and shared functional group
/// sequence (5200,9229).
///
/// # Mathematical specification
///
/// For an Enhanced CT/MR/PET image with N frames, each frame index k âˆˆ \[0, N\):
/// - `image_position\[k\]`: physical position P_k âˆˆ â„Â³ (mm) of frame k's origin.
///   Source: (5200,9230)\[k\] â†’ (0020,9113)\[0\] â†’ (0020,0032).
///   Falls back to shared (5200,9229) â†’ (0020,9113)\[0\] â†’ (0020,0032).
/// - `image_orientation\[k\]`: IOP cosines F_k âˆˆ â„^6.
///   Source: (5200,9229 or 9230) â†’ (0020,9116)\[0\] â†’ (0020,0037).
/// - `rescale_slope\[k\]` / `rescale_intercept\[k\]`: linear intensity transform.
///   Source: (5200,9230)\[k\] â†’ (0028,9145)\[0\] â†’ (0028,1053/1052).
///   Falls back to shared groups, then to `None`.
/// - `pixel_spacing\[k\]` / `slice_thickness\[k\]`: from (0028,9110) â†’ (0028,0030/0018,0050).
#[derive(Debug, Clone, Default)]
pub struct PerFrameInfo {
    /// Physical position P_k of this frame's origin in mm.
    pub image_position: Option<[f64; 3]>,
    /// Image orientation cosines for this frame.
    pub image_orientation: Option<[f64; 6]>,
    /// Pixel spacing (row, column) in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Slice thickness in mm.
    pub slice_thickness: Option<f64>,
    /// Rescale slope for this frame (0028,1053).
    pub rescale_slope: Option<f64>,
    /// Rescale intercept for this frame (0028,1052).
    pub rescale_intercept: Option<f64>,
}

/// Summary information about a multi-frame DICOM file.
#[derive(Debug, Clone)]
pub struct MultiFrameInfo {
    /// Source file path.
    pub path: PathBuf,
    /// Number of frames.
    pub n_frames: usize,
    /// Pixel rows per frame.
    pub rows: usize,
    /// Pixel columns per frame.
    pub cols: usize,
    /// SamplesPerPixel (0028,0002). Scalar volume loading supports only 1.
    pub samples_per_pixel: usize,
    /// Bits allocated per sample (8 or 16).
    pub bits_allocated: u16,
    /// PixelRepresentation (0028,0103): unsigned or signed two's complement.
    /// Defaults to unsigned per DICOM PS3.3 C.7.6.3.1.
    pub pixel_representation: PixelSignedness,
    /// Pixel spacing [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Frame thickness (SliceThickness) in mm.
    pub frame_thickness: Option<f64>,
    /// Modality string.
    pub modality: Option<ArrayString<16>>,
    /// SOP Class UID.
    pub sop_class_uid: Option<ArrayString<64>>,
    /// ImagePositionPatient for frame 0: [x, y, z] in mm.
    pub image_position: Option<[f64; 3]>,
    /// ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z].
    pub image_orientation: Option<[f64; 6]>,
    /// RescaleSlope (0028,1053). Defaults to 1.0 when absent.
    pub rescale_slope: f64,
    /// RescaleIntercept (0028,1052). Defaults to 0.0 when absent.
    pub rescale_intercept: f64,
    /// Per-frame functional group data. Empty for non-enhanced multiframe objects.
    /// Length == n_frames when functional group sequences are present, 0 otherwise.
    pub per_frame: Vec<PerFrameInfo>,
}

/// Optional spatial metadata for multi-frame DICOM output.
///
/// When provided to [`write_dicom_multiframe_with_options`](super::write_dicom_multiframe_with_options),
/// spatial tags are emitted: ImagePositionPatient (0020,0032), ImageOrientationPatient (0020,0037),
/// PixelSpacing (0028,0030), SliceThickness (0018,0050), and Modality (0008,0060).
///
/// When absent, the writer behaves identically to [`write_dicom_multiframe`](super::write_dicom_multiframe).
#[derive(Debug, Clone)]
pub struct MultiFrameSpatialMetadata {
    /// ImagePositionPatient for frame 0: [x, y, z] in mm.
    pub origin: [f64; 3],
    /// Pixel spacing [row_spacing, col_spacing] in mm.
    pub pixel_spacing: [f64; 2],
    /// Slice thickness in mm.
    pub slice_thickness: f64,
    /// ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z].
    pub image_orientation: [f64; 6],
    /// Modality string (e.g., "CT", "MR", "OT").
    pub modality: ArrayString<16>,
}

/// Builder for multi-frame DICOM write options.
///
/// # Invariants
/// - `sop_class_uid` defaults to `MF_GRAYSCALE_WORD_SC_UID`.
/// - `instance_number` defaults to 1.
/// - When `spatial` is `None`, no spatial tags are emitted.
///
/// Use [`write_dicom_multiframe_with_config`](super::write_dicom_multiframe_with_config)
/// to supply an explicit config.
#[derive(Debug, Clone)]
pub struct MultiFrameWriterConfig {
    /// SOP Class UID (0008,0016). Defaults to Multi-Frame Grayscale Word SC UID.
    pub sop_class_uid: ArrayString<64>,
    /// Optional spatial metadata emitted as IPP/IOP/PixelSpacing/SliceThickness/Modality.
    pub spatial: Option<MultiFrameSpatialMetadata>,
    /// InstanceNumber (0020,0013). Defaults to 1.
    pub instance_number: u32,
}

impl Default for MultiFrameWriterConfig {
    fn default() -> Self {
        Self {
            sop_class_uid: literal_arraystring(MF_GRAYSCALE_WORD_SC_UID),
            spatial: None,
            instance_number: 1,
        }
    }
}
