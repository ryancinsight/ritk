//! DICOM metadata types extracted from the series reader.
//!
//! This module contains the per-slice and per-series metadata structs, the
//! patient position enum, and associated helpers. These types are the typed
//! output of the DICOM read path and carry no I/O logic.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use super::object_model::DicomPreservationSet;

/// Per-slice DICOM metadata extracted during series loading.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSliceMetadata {
    /// Source file path for the slice.
    pub path: PathBuf,
    /// Recursive preservation data extracted from the slice.
    pub preservation: DicomPreservationSet,
    /// SOP Instance UID if available.
    pub sop_instance_uid: Option<String>,
    /// Instance number if available.
    pub instance_number: Option<i32>,
    /// Slice location if available.
    pub slice_location: Option<f64>,
    /// Image position patient (x, y, z) in mm.
    pub image_position_patient: Option<[f64; 3]>,
    /// Image orientation patient as two direction cosines.
    pub image_orientation_patient: Option<[f64; 6]>,
    /// Pixel spacing (row, column) in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Slice thickness in mm.
    pub slice_thickness: Option<f64>,
    /// Rescale slope.
    pub rescale_slope: f32,
    /// Rescale intercept.
    pub rescale_intercept: f32,
    /// SOP Class UID if available.
    pub sop_class_uid: Option<String>,
    /// Transfer syntax UID if available.
    pub transfer_syntax_uid: Option<String>,
    /// Custom per-slice tags preserved as text.
    pub private_tags: HashMap<String, String>,
    /// PixelRepresentation (0028,0103): 0 = unsigned, 1 = signed two's complement.
    /// Defaults to 0 when absent per DICOM PS3.3 C.7.6.3.1.
    pub pixel_representation: u16,
    /// BitsAllocated (0028,0100) for this slice. Defaults to 16 when absent.
    pub bits_allocated: u16,
    /// WindowCenter (0028,1050) for display, first value when multi-valued DS.
    pub window_center: Option<f64>,
    /// WindowWidth (0028,1051) for display, first value when multi-valued DS.
    pub window_width: Option<f64>,
    /// GantryDetectorTilt (0018,1120) in degrees. Positive = toward patient's feet.
    /// Used to synthesize oblique IOP when IOP is absent or effectively axial.
    pub gantry_tilt: Option<f64>,
    /// Patient position (0018,5100) as a setup code.
    ///
    /// This is preserved as a typed semantic label, not as a geometric transform.
    /// Physical voxel orientation still comes from IOP/IPP and spacing.
    pub patient_position: Option<PatientPosition>,
}

impl Default for DicomSliceMetadata {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            preservation: DicomPreservationSet::new(),
            sop_instance_uid: None,
            instance_number: None,
            slice_location: None,
            image_position_patient: None,
            image_orientation_patient: None,
            pixel_spacing: None,
            slice_thickness: None,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            sop_class_uid: None,
            transfer_syntax_uid: None,
            private_tags: HashMap::new(),
            pixel_representation: 0,
            bits_allocated: 16,
            window_center: None,
            window_width: None,
            gantry_tilt: None,
            patient_position: None,
        }
    }
}

/// Patient setup code from DICOM tag (0018,5100).
///
/// The code classifies acquisition setup only. It does not change image-space
/// geometry, which is still derived from IOP/IPP and spacing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatientPosition {
    HeadFirstSupine,
    HeadFirstProne,
    FeetFirstSupine,
    FeetFirstProne,
    HeadFirstDecubitusRight,
    HeadFirstDecubitusLeft,
    FeetFirstDecubitusRight,
    FeetFirstDecubitusLeft,
    Unknown(String),
}

impl PatientPosition {
    /// Parse a DICOM CS code into a typed patient position.
    pub fn from_code(code: &str) -> Self {
        let normalized = code.trim().to_uppercase();
        match normalized.as_str() {
            "HFS" => Self::HeadFirstSupine,
            "HFP" => Self::HeadFirstProne,
            "FFS" => Self::FeetFirstSupine,
            "FFP" => Self::FeetFirstProne,
            "HFDR" => Self::HeadFirstDecubitusRight,
            "HFDL" => Self::HeadFirstDecubitusLeft,
            "FFDR" => Self::FeetFirstDecubitusRight,
            "FFDL" => Self::FeetFirstDecubitusLeft,
            _ if normalized.is_empty() => Self::Unknown(String::new()),
            _ => Self::Unknown(normalized),
        }
    }

    /// Return the canonical DICOM code for display and serialization.
    pub fn code(&self) -> &str {
        match self {
            Self::HeadFirstSupine => "HFS",
            Self::HeadFirstProne => "HFP",
            Self::FeetFirstSupine => "FFS",
            Self::FeetFirstProne => "FFP",
            Self::HeadFirstDecubitusRight => "HFDR",
            Self::HeadFirstDecubitusLeft => "HFDL",
            Self::FeetFirstDecubitusRight => "FFDR",
            Self::FeetFirstDecubitusLeft => "FFDL",
            Self::Unknown(code) => code.as_str(),
        }
    }

    /// Human-readable label for metadata tables and diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::HeadFirstSupine => "Head First Supine",
            Self::HeadFirstProne => "Head First Prone",
            Self::FeetFirstSupine => "Feet First Supine",
            Self::FeetFirstProne => "Feet First Prone",
            Self::HeadFirstDecubitusRight => "Head First Decubitus Right",
            Self::HeadFirstDecubitusLeft => "Head First Decubitus Left",
            Self::FeetFirstDecubitusRight => "Feet First Decubitus Right",
            Self::FeetFirstDecubitusLeft => "Feet First Decubitus Left",
            Self::Unknown(_) => "Unknown",
        }
    }
}

impl fmt::Display for PatientPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Parse a patient position code string, returning `None` for empty/whitespace input.
pub fn parse_patient_position(code: &str) -> Option<PatientPosition> {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PatientPosition::from_code(trimmed))
    }
}

/// Series-level DICOM metadata.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DicomReadMetadata {
    /// Series instance UID if available.
    pub series_instance_uid: Option<String>,
    /// Study instance UID if available.
    pub study_instance_uid: Option<String>,
    /// Frame of reference UID if available.
    pub frame_of_reference_uid: Option<String>,
    /// Series description if available.
    pub series_description: Option<String>,
    /// Modality if available.
    pub modality: Option<String>,
    /// Patient ID if available.
    pub patient_id: Option<String>,
    /// Patient name if available.
    pub patient_name: Option<String>,
    /// Study date if available.
    pub study_date: Option<String>,
    /// Series date if available.
    pub series_date: Option<String>,
    /// Series time if available.
    pub series_time: Option<String>,
    /// Image dimensions in `[rows, cols, slices]`.
    pub dimensions: [usize; 3],
    /// Physical spacing in `[x, y, z]` order.
    pub spacing: [f64; 3],
    /// Physical origin in mm.
    pub origin: [f64; 3],
    /// Direction cosines in row-major 3x3 order.
    pub direction: [f64; 9],
    /// Bits allocated if available.
    pub bits_allocated: Option<u16>,
    /// Bits stored if available.
    pub bits_stored: Option<u16>,
    /// High bit if available.
    pub high_bit: Option<u16>,
    /// Photometric interpretation if available.
    pub photometric_interpretation: Option<String>,
    /// Slice metadata in load order.
    pub slices: Vec<DicomSliceMetadata>,
    /// Custom series-level tags preserved as text.
    pub private_tags: HashMap<String, String>,
    /// Recursive preservation data for the series.
    pub preservation: DicomPreservationSet,
    /// (0010,1030) Patient Weight [kg]. Required for SUVbw normalisation.
    pub patient_weight_kg: Option<f64>,
    /// (0054,1102) Decay Correction: "NONE", "START", or "ADMIN".
    pub decay_correction: Option<String>,
    /// (0054,0016)[0]/(0018,1074) Radionuclide Total Dose [Bq].
    pub radionuclide_total_dose_bq: Option<f64>,
    /// (0054,0016)[0]/(0018,1072) Radiopharmaceutical Start Time.
    pub radiopharmaceutical_start_time: Option<String>,
    /// (0054,0016)[0]/(0018,1076) Radionuclide Half Life [s].
    pub radionuclide_half_life_s: Option<f64>,
}

/// A simplified DICOM series descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSeriesInfo {
    /// Series path.
    pub path: PathBuf,
    /// Number of slices discovered in the directory.
    pub num_slices: usize,
    /// Series metadata.
    pub metadata: DicomReadMetadata,
}
