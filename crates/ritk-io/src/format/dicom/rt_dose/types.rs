//! Domain types for RT Dose Storage.

use arrayvec::ArrayString;

/// SOP Class UID for RT Dose Storage.
pub const RT_DOSE_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.481.2";

/// DoseType (3004,0004) â€” DICOM PS3.3 C.7.6.8 Enumerated Values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtDoseType {
    /// Physical dose.
    Physical,
    /// Effective dose.
    Effective,
    /// Error (difference between calculated and delivered dose).
    Error,
    /// Non-standard value.
    Other(ArrayString<16>),
}

impl RtDoseType {
    /// Parse from the DICOM CS string representation.
    pub fn from_dicom_str(s: &str) -> Self {
        match s.trim() {
            "PHYSICAL" => Self::Physical,
            "EFFECTIVE" => Self::Effective,
            "ERROR" => Self::Error,
            other => Self::Other(ArrayString::<16>::try_from(other).unwrap_or_default()),
        }
    }

    /// Return the canonical DICOM CS string for this type.
    pub fn as_dicom_str(&self) -> &str {
        match self {
            Self::Physical => "PHYSICAL",
            Self::Effective => "EFFECTIVE",
            Self::Error => "ERROR",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// DoseSummationType (3004,0002) â€” DICOM PS3.3 C.7.6.8 Enumerated Values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtDoseSummationType {
    /// Dose summed over entire planned treatment.
    Plan,
    /// Dose summed over multiple plans.
    MultiplePlan,
    /// Dose summed over all fractions.
    Fraction,
    /// Dose for a beam.
    Beam,
    /// Dose for a brachytherapy channel.
    Brachy,
    /// Dose for a fraction of a session.
    FractionSession,
    /// Dose for a beam session.
    BeamSession,
    /// Dose for a brachytherapy session.
    BrachySession,
    /// Dose for a control point.
    ControlPoint,
    /// Dose record.
    Record,
    /// Non-standard value.
    Other(ArrayString<16>),
}

impl RtDoseSummationType {
    /// Parse from the DICOM CS string representation.
    pub fn from_dicom_str(s: &str) -> Self {
        match s.trim() {
            "PLAN" => Self::Plan,
            "MULTI_PLAN" => Self::MultiplePlan,
            "FRACTION" => Self::Fraction,
            "BEAM" => Self::Beam,
            "BRACHY" => Self::Brachy,
            "FRACTION_SESSION" => Self::FractionSession,
            "BEAM_SESSION" => Self::BeamSession,
            "BRACHY_SESSION" => Self::BrachySession,
            "CONTROL_POINT" => Self::ControlPoint,
            "RECORD" => Self::Record,
            other => Self::Other(ArrayString::<16>::try_from(other).unwrap_or_default()),
        }
    }

    /// Return the canonical DICOM CS string for this type.
    pub fn as_dicom_str(&self) -> &str {
        match self {
            Self::Plan => "PLAN",
            Self::MultiplePlan => "MULTI_PLAN",
            Self::Fraction => "FRACTION",
            Self::Beam => "BEAM",
            Self::Brachy => "BRACHY",
            Self::FractionSession => "FRACTION_SESSION",
            Self::BeamSession => "BEAM_SESSION",
            Self::BrachySession => "BRACHY_SESSION",
            Self::ControlPoint => "CONTROL_POINT",
            Self::Record => "RECORD",
            Self::Other(s) => s.as_str(),
        }
    }
}

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
    pub dose_type: RtDoseType,
    /// DoseSummationType (3004,0002): PLAN, BEAM, FRACTION, CONTROL_POINT, etc.
    pub dose_summation_type: RtDoseSummationType,
    /// DoseGridScaling (3004,000E): factor converting raw pixel values to Gy.
    pub dose_grid_scaling: f64,
    /// Z-offset per frame (mm) from GridFrameOffsetVector (3004,000C).
    /// Length == n_frames. When the tag is absent, offsets are 0.0, 1.0, 2.0, â€¦
    pub frame_offsets: Vec<f64>,
    /// Dose values in Gy per voxel. Length = n_frames * rows * cols.
    /// Flat index: frame * rows * cols + row * cols + col.
    pub dose_gy: Vec<f64>,
    /// ImagePositionPatient (0020,0032) â€” origin of the dose grid in mm.
    pub image_position: Option<[f64; 3]>,
    /// ImageOrientationPatient (0020,0037) â€” 6-component direction cosines.
    pub image_orientation: Option<[f64; 6]>,
    /// PixelSpacing (0028,0030) â€” [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Referenced RT Plan SOPInstanceUID from ReferencedRTPlanSequence (300C,0002).
    pub referenced_rt_plan_sop_instance_uid: Option<ArrayString<64>>,
}
