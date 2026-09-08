//! Typed metadata records produced by the DICOM series reader.
//!
//! This module defines the per-slice and per-series metadata structs, the
//! patient position enum, and associated helpers. These types carry no I/O
//! logic — they are the typed output of the read path.

use anyhow::{bail, Result};
use arrayvec::ArrayString;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use crate::format::dicom::object_model::{DicomObjectModel, DicomPreservationSet};
use ritk_dicom::{ParseBudget, PixelSignedness};

const DEFAULT_MAX_RETAINED_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_MAX_DECODED_BYTES: usize = 1024 * 1024 * 1024;

/// Resource ceilings for one DICOM read workflow.
///
/// The parser budget limits each encoded input and its structural traversal.
/// The retained and decoded ceilings cover the two longer-lived allocations
/// owned by the reader and loader. Keeping these ceilings separate lets a
/// caller admit a large study made of small instances without allowing one
/// malformed instance to request the study's whole storage allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DicomReadBudget {
    parser: ParseBudget,
    max_retained_bytes: usize,
    max_decoded_bytes: usize,
}

impl Default for DicomReadBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl DicomReadBudget {
    /// Default ceilings for a desktop or browser-backed viewer workflow.
    ///
    /// The one-gibibyte retained and decoded ceilings are explicit finite
    /// bounds; callers handling constrained devices should construct a
    /// smaller budget with [`Self::try_new`].
    pub const DEFAULT: Self = Self {
        parser: ParseBudget::DEFAULT,
        max_retained_bytes: DEFAULT_MAX_RETAINED_BYTES,
        max_decoded_bytes: DEFAULT_MAX_DECODED_BYTES,
    };

    /// Construct a workflow budget with explicit storage ceilings.
    ///
    /// # Errors
    ///
    /// Returns an error when either workflow ceiling is zero.
    pub fn try_new(
        parser: ParseBudget,
        max_retained_bytes: usize,
        max_decoded_bytes: usize,
    ) -> Result<Self> {
        if max_retained_bytes == 0 {
            bail!("DICOM retained-byte budget must be nonzero");
        }
        if max_decoded_bytes == 0 {
            bail!("DICOM decoded-byte budget must be nonzero");
        }
        Ok(Self {
            parser,
            max_retained_bytes,
            max_decoded_bytes,
        })
    }

    /// Return the structural parser budget used for each instance.
    #[must_use]
    pub const fn parser(&self) -> ParseBudget {
        self.parser
    }

    /// Return the maximum retained encoded bytes for one study.
    #[must_use]
    pub const fn max_retained_bytes(&self) -> usize {
        self.max_retained_bytes
    }

    /// Return the maximum decoded workspace bytes for one load.
    #[must_use]
    pub const fn max_decoded_bytes(&self) -> usize {
        self.max_decoded_bytes
    }

    /// Check a cumulative retained-byte total before storing another member.
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` exceeds the study ceiling.
    pub fn checked_retained_bytes(&self, bytes: usize) -> Result<usize> {
        if bytes > self.max_retained_bytes {
            bail!(
                "DICOM retained study bytes {bytes} exceed budget {}",
                self.max_retained_bytes
            );
        }
        Ok(bytes)
    }

    /// Check decoded workspace bytes before allocating a volume or frame set.
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` exceeds the decoded workspace ceiling.
    pub fn checked_decoded_bytes(&self, bytes: usize) -> Result<usize> {
        if bytes > self.max_decoded_bytes {
            bail!(
                "DICOM decoded workspace bytes {bytes} exceed budget {}",
                self.max_decoded_bytes
            );
        }
        Ok(bytes)
    }
}

/// Per-slice DICOM metadata extracted during series loading.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSliceMetadata {
    pub path: PathBuf,
    pub preservation: DicomPreservationSet,
    pub sop_instance_uid: Option<ArrayString<64>>,
    pub instance_number: Option<i32>,
    pub slice_location: Option<f64>,
    /// Image position patient (x, y, z) in mm.
    pub image_position_patient: Option<[f64; 3]>,
    /// Image orientation patient as two direction cosines.
    pub image_orientation_patient: Option<[f64; 6]>,
    /// Pixel spacing (row, column) in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    pub slice_thickness: Option<f64>,
    pub rescale_slope: f32,
    pub rescale_intercept: f32,
    pub sop_class_uid: Option<ArrayString<64>>,
    pub transfer_syntax_uid: Option<ArrayString<64>>,
    pub private_tags: HashMap<String, String>,
    /// PixelRepresentation (0028,0103): unsigned or signed two's complement.
    pub pixel_representation: PixelSignedness,
    /// BitsAllocated (0028,0100). Defaults to 16 when absent.
    pub bits_allocated: u16,
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
    /// GantryDetectorTilt (0018,1120) in degrees.
    pub gantry_tilt: Option<f64>,
    pub patient_position: Option<PatientPosition>,
    /// Exact Part 10 bytes validated by the scanner, retained for pixel decode.
    /// Filesystem, dropped-byte, and SCP scans populate this field so decoding
    /// never substitutes a file changed after metadata validation. Manually
    /// constructed descriptors may omit it to request path-based decoding.
    pub part10_bytes: Option<Vec<u8>>,
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
            pixel_representation: PixelSignedness::Unsigned,
            bits_allocated: 16,
            window_center: None,
            window_width: None,
            gantry_tilt: None,
            patient_position: None,
            part10_bytes: None,
        }
    }
}

/// Patient setup code from DICOM tag (0018,5100).
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
    Unknown(ArrayString<4>),
}

impl PatientPosition {
    /// Alias for [`from_code`](Self::from_code) matching the DICOM naming convention.
    pub fn from_dicom_code(code: &str) -> Self {
        Self::from_code(code)
    }

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
            _ if normalized.is_empty() => Self::Unknown(ArrayString::new()),
            _ => {
                let mut arr = ArrayString::<4>::new();
                for ch in normalized.chars() {
                    if arr.try_push(ch).is_err() {
                        break;
                    }
                }
                Self::Unknown(arr)
            }
        }
    }

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

pub(super) fn parse_patient_position(code: &str) -> Option<PatientPosition> {
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
    pub series_instance_uid: Option<ArrayString<64>>,
    pub study_instance_uid: Option<ArrayString<64>>,
    pub frame_of_reference_uid: Option<ArrayString<64>>,
    pub series_description: Option<String>,
    pub modality: Option<ArrayString<16>>,
    pub patient_id: Option<String>,
    pub patient_name: Option<String>,
    pub study_date: Option<ArrayString<8>>,
    pub series_date: Option<ArrayString<8>>,
    pub series_time: Option<ArrayString<16>>,
    /// Image dimensions `\[rows, cols, slices\]`.
    pub dimensions: [usize; 3],
    /// Physical spacing `\[Δz, ΔRow, ΔCol\]`.
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
    /// Direction cosines in row-major 3×3 order.
    pub direction: [f64; 9],
    pub bits_allocated: Option<u16>,
    pub bits_stored: Option<u16>,
    pub high_bit: Option<u16>,
    pub photometric_interpretation: Option<ArrayString<16>>,
    pub slices: Vec<DicomSliceMetadata>,
    pub private_tags: HashMap<String, String>,
    pub preservation: DicomPreservationSet,
    /// (0010,1030) Patient Weight \[kg\].
    pub patient_weight_kg: Option<f64>,
    /// (0054,1102) Decay Correction: "NONE", "START", or "ADMIN".
    pub decay_correction: Option<ArrayString<16>>,
    /// (0054,0016)\[0\]/(0018,1074) Radionuclide Total Dose \[Bq\].
    pub radionuclide_total_dose_bq: Option<f64>,
    /// (0054,0016)\[0\]/(0018,1072) Radiopharmaceutical Start Time.
    pub radiopharmaceutical_start_time: Option<ArrayString<16>>,
    /// (0054,0016)\[0\]/(0018,1076) Radionuclide Half Life \[s\].
    pub radionuclide_half_life_s: Option<f64>,
}

/// Simplified DICOM series descriptor returned from directory scan.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSeriesInfo {
    pub path: std::path::PathBuf,
    pub num_slices: usize,
    pub metadata: DicomReadMetadata,
}

/// Accumulator for first-seen series-level fields during multi-file scanning.
#[derive(Default)]
pub(super) struct SeriesFirstSeen {
    pub rows: Option<u32>,
    pub cols: Option<u32>,
    pub pixel_spacing: Option<[f64; 2]>,
    pub slice_thickness: Option<f64>,
    pub series_instance_uid: Option<ArrayString<64>>,
    pub study_instance_uid: Option<ArrayString<64>>,
    pub series_description: Option<String>,
    pub modality: Option<ArrayString<16>>,
    pub patient_id: Option<String>,
    pub patient_name: Option<String>,
    pub study_date: Option<ArrayString<8>>,
    pub series_date: Option<ArrayString<8>>,
    pub series_time: Option<ArrayString<16>>,
    pub frame_of_reference_uid: Option<ArrayString<64>>,
    pub bits_allocated: Option<u16>,
    pub bits_stored: Option<u16>,
    pub high_bit: Option<u16>,
    pub photometric_interpretation: Option<ArrayString<16>>,
    pub transfer_syntax_uid: Option<ArrayString<64>>,
    pub patient_weight_kg: Option<f64>,
    pub decay_correction: Option<ArrayString<16>>,
    pub radionuclide_total_dose_bq: Option<f64>,
    pub radiopharmaceutical_start_time: Option<ArrayString<16>>,
    pub radionuclide_half_life_s: Option<f64>,
}

/// Spatial geometry of a reconstructed DICOM series.
#[derive(Debug, Clone, Copy)]
pub(super) struct SeriesGeometry {
    /// Number of pixel rows per slice.
    pub rows: usize,
    /// Number of pixel columns per slice.
    pub cols: usize,
    /// Voxel spacing \[mm\]: (row, col, slice).
    pub spacing: [f64; 3],
    /// Image position of the first slice in patient coordinates \[mm\].
    pub origin: [f64; 3],
    /// 3×3 row-major direction cosine matrix (col 0 = normal, col 1 = F_c, col 2 = F_r).
    pub direction: [f64; 9],
}

/// Construct an `ArrayString<N>` from a string literal, panicking with a
/// descriptive message if the literal exceeds capacity.
///
/// Replaces the `ArrayString::from(LITERAL).expect("infallible: validated precondition")` pattern for string
/// literals that are known by construction to fit.
///
/// # Example
/// ```ignore
/// let val: ArrayString<64> = literal_arraystring("1.2.840.10008.1.1");
/// ```
#[inline]
#[track_caller]
pub fn literal_arraystring<const N: usize>(s: &'static str) -> ArrayString<N> {
    ArrayString::from(s).unwrap_or_else(|_| {
        panic!(
            "literal \"{}\" ({} bytes) exceeds ArrayString<{}> capacity",
            s,
            s.len(),
            N
        )
    })
}

/// Truncate a string to fit `ArrayString<N>`, emitting a warning if truncation occurs.
///
/// This is the DRY helper for the `ArrayString::from(s).unwrap_or_else(|_| ...)` pattern
/// used when converting DICOM VR fields that may exceed their maximum length.
pub(crate) fn truncate_arraystring<const N: usize>(s: &str) -> ArrayString<N> {
    let mut end = N.min(s.len());
    while !s.is_char_boundary(end) {
        // Zero is always a character boundary, so this cannot underflow.
        end -= 1;
    }
    if end < s.len() {
        tracing::warn!(
            capacity_bytes = N,
            input_bytes = s.len(),
            "DICOM text exceeds fixed capacity; truncating at a UTF-8 boundary"
        );
    }
    let truncated = &s[..end];
    ArrayString::from(truncated).expect("truncated string fits ArrayString by construction")
}

/// Convert a DICOM UID string to `Option<ArrayString<64>>`.
///
/// DICOM UIDs are formally limited to 64 characters per the standard.
/// If a UID exceeds this length (non-conformant), a warning is emitted
/// and the value is truncated to 64 chars.
pub(crate) fn uid_to_arraystring(s: &str) -> Option<ArrayString<64>> {
    Some(truncate_arraystring::<64>(s))
}

/// Convert a CS (Code String) value to `ArrayString<16>`.
///
/// DICOM CS values are limited to 16 characters per the standard.
/// If a value exceeds this length (non-conformant), a warning is emitted
/// and the value is truncated to 16 chars.
pub(crate) fn cs_to_arraystring(s: &str) -> ArrayString<16> {
    truncate_arraystring::<16>(s)
}

/// Convert a DA (Date) value to `ArrayString<8>`.
///
/// DICOM DA values are exactly 8 characters (YYYYMMDD) per the standard.
/// If a value exceeds this length (non-conformant), a warning is emitted
/// and the value is truncated to 8 chars.
pub(crate) fn da_to_arraystring(s: &str) -> ArrayString<8> {
    truncate_arraystring::<8>(s)
}

/// Convert a TM (Time) value to `ArrayString<16>`.
///
/// DICOM TM values are limited to 16 characters per the standard.
/// If a value exceeds this length (non-conformant), a warning is emitted
/// and the value is truncated to 16 chars.
pub(crate) fn tm_to_arraystring(s: &str) -> ArrayString<16> {
    truncate_arraystring::<16>(s)
}

/// Assemble a [`DicomReadMetadata`] from a [`SeriesFirstSeen`] accumulator, sorted slices,
/// and computed geometry fields.
pub(super) fn assemble_metadata(
    first: SeriesFirstSeen,
    slices: Vec<DicomSliceMetadata>,
    geometry: SeriesGeometry,
    series_object: DicomObjectModel,
) -> DicomReadMetadata {
    let SeriesGeometry {
        rows,
        cols,
        spacing,
        origin,
        direction,
    } = geometry;
    DicomReadMetadata {
        series_instance_uid: first.series_instance_uid,
        study_instance_uid: first.study_instance_uid,
        frame_of_reference_uid: first.frame_of_reference_uid,
        series_description: first.series_description,
        modality: first.modality,
        patient_id: first.patient_id,
        patient_name: first.patient_name,
        study_date: first.study_date,
        series_date: first.series_date,
        series_time: first.series_time,
        dimensions: [rows, cols, slices.len()],
        spacing,
        origin,
        direction,
        bits_allocated: first.bits_allocated,
        bits_stored: first.bits_stored,
        high_bit: first.high_bit,
        photometric_interpretation: first.photometric_interpretation,
        slices,
        private_tags: HashMap::new(),
        preservation: DicomPreservationSet {
            object: series_object,
            preserved: Vec::new(),
        },
        patient_weight_kg: first.patient_weight_kg,
        decay_correction: first.decay_correction,
        radionuclide_total_dose_bq: first.radionuclide_total_dose_bq,
        radiopharmaceutical_start_time: first.radiopharmaceutical_start_time,
        radionuclide_half_life_s: first.radionuclide_half_life_s,
    }
}
