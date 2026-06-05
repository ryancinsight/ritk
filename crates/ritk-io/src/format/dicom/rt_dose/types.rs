//! Domain types for RT Dose Storage.

use arrayvec::ArrayString;

/// SOP Class UID for RT Dose Storage.
pub const RT_DOSE_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.2";

/// In-memory representation of an RT Dose grid.
///
/// # Invariants
/// - `dose_gy.len() == n_frames * rows * cols`
/// - `frame_offsets.len() == n_frames`
/// - `dose_gy[frame * rows * cols + row * cols + col]`
///   = `raw_pixel_u32 * dose_grid_scaling`
#[derive(Debug, Clone)]
pub struct RtDoseGrid {
    /// Grid rows (0028,0010).
    pub rows: usize,
    /// Grid columns (0028,0011).
    pub cols: usize,
    /// Number of dose planes (0028,0008); defaults to 1 when absent.
    pub n_frames: usize,
    /// DoseType (3004,0004): PHYSICAL, EFFECTIVE, or ERROR.
    pub dose_type: ArrayString<16>,
    /// DoseSummationType (3004,0002): PLAN, BEAM, FRACTION, CONTROL_PT, etc.
    pub dose_summation_type: ArrayString<16>,
    /// DoseGridScaling (3004,000E): factor converting raw pixel values to Gy.
    pub dose_grid_scaling: f64,
    /// Z-offset per frame (mm) from GridFrameOffsetVector (3004,000C).
    /// Length == n_frames. When the tag is absent, offsets are 0.0, 1.0, 2.0, …
    pub frame_offsets: Vec<f64>,
    /// Dose values in Gy per voxel. Length = n_frames * rows * cols.
    /// Flat index: frame * rows * cols + row * cols + col.
    pub dose_gy: Vec<f64>,
    /// ImagePositionPatient (0020,0032) — origin of the dose grid in mm.
    pub image_position: Option<[f64; 3]>,
    /// ImageOrientationPatient (0020,0037) — 6-component direction cosines.
    pub image_orientation: Option<[f64; 6]>,
    /// PixelSpacing (0028,0030) — [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Referenced RT Plan SOPInstanceUID from ReferencedRTPlanSequence (300C,0002).
    pub referenced_rt_plan_sop_instance_uid: Option<ArrayString<64>>,
}
