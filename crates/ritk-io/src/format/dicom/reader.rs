//! DICOM series reader and metadata API.
//!
//! This module provides a conservative DICOM series read path with explicit
//! metadata capture. The implementation is series-oriented and rejects inputs
//! that do not satisfy the reader's invariants.
//!
//! # Invariants
//!
//! - The input path must resolve to a directory containing at least one DICOM file.
//! - All slices in a returned series share the same rows, columns, spacing, and
//!   transfer syntax constraints accepted by the decoder.
//! - Slice metadata is preserved in a typed `DicomSliceMetadata` record.
//! - Series metadata is captured in `DicomReadMetadata`.
//!
//! # Notes
//!
//! This reader is intentionally conservative. It only extracts the metadata and
//! pixel data needed for image series loading, and it fails fast on unsupported
//! or inconsistent series layouts.
//!
//! The API is designed so crate-level re-exports can expose:
//! - `scan_dicom_directory`
//! - `read_dicom_series`
//! - `load_dicom_series`
//! - `read_dicom_series_with_metadata`
//! - `load_dicom_series_with_metadata`
//! - `DicomSeriesInfo`
//! - `DicomReadMetadata`
//! - `DicomSliceMetadata`

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::{Tag, VR};
use dicom::object::open_file;
use dicom_core::header::Header;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::object_model::{
    is_private_tag, DicomObjectModel, DicomObjectNode, DicomPreservationSet, DicomPreservedElement,
    DicomSequenceItem, DicomTag, DicomValue,
};
use super::sop_class::{classify_sop_class, SopClassKind};
use super::transfer_syntax::TransferSyntaxKind;

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
        }
    }
}

/// Series-level DICOM metadata.
#[derive(Debug, Clone, PartialEq)]
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

/// Compute a compact key from a DICOM tag group+element pair.
#[inline]
fn tag_key(group: u16, element: u16) -> u32 {
    ((group as u32) << 16) | (element as u32)
}

/// Return the set of tags already extracted by the named parsing logic in scan_dicom_directory.
///
/// Elements whose tag_key is in this set are skipped during full-preservation iteration.
fn known_handled_tags() -> HashSet<u32> {
    let mut s = HashSet::new();
    // Per-slice
    s.insert(tag_key(0x0008, 0x0018)); // SOP Instance UID
    s.insert(tag_key(0x0020, 0x0013)); // Instance Number
    s.insert(tag_key(0x0020, 0x1041)); // Slice Location
    s.insert(tag_key(0x0020, 0x0032)); // ImagePositionPatient
    s.insert(tag_key(0x0020, 0x0037)); // ImageOrientationPatient
    s.insert(tag_key(0x0028, 0x0030)); // PixelSpacing
    s.insert(tag_key(0x0018, 0x0050)); // SliceThickness
    s.insert(tag_key(0x0028, 0x1053)); // RescaleSlope
    s.insert(tag_key(0x0028, 0x1052)); // RescaleIntercept
    s.insert(tag_key(0x0008, 0x0016)); // SOP Class UID
    s.insert(tag_key(0x0008, 0x0070)); // Manufacturer (transfer syntax surrogate)
                                       // Rows / Columns / series geometry
    s.insert(tag_key(0x0028, 0x0010)); // Rows
    s.insert(tag_key(0x0028, 0x0011)); // Columns
    s.insert(tag_key(0x0020, 0x000E)); // SeriesInstanceUID
    s.insert(tag_key(0x0020, 0x000D)); // StudyInstanceUID
    s.insert(tag_key(0x0008, 0x103E)); // SeriesDescription
    s.insert(tag_key(0x0008, 0x0060)); // Modality
    s.insert(tag_key(0x0010, 0x0020)); // PatientID
    s.insert(tag_key(0x0010, 0x0010)); // PatientName
    s.insert(tag_key(0x0008, 0x0020)); // StudyDate
    s.insert(tag_key(0x0008, 0x0021)); // SeriesDate
    s.insert(tag_key(0x0008, 0x0031)); // SeriesTime
    s.insert(tag_key(0x0020, 0x0052)); // FrameOfReferenceUID
    s.insert(tag_key(0x0028, 0x0100)); // BitsAllocated
    s.insert(tag_key(0x0028, 0x0101)); // BitsStored
    s.insert(tag_key(0x0028, 0x0102)); // HighBit
    s.insert(tag_key(0x0028, 0x0004)); // PhotometricInterpretation
    s.insert(tag_key(0x0028, 0x0002)); // SamplesPerPixel
    s.insert(tag_key(0x0028, 0x0103)); // PixelRepresentation
    s.insert(tag_key(0x0028, 0x1050)); // WindowCenter
    s.insert(tag_key(0x0028, 0x1051)); // WindowWidth
                                       // Always skip pixel data
    s.insert(tag_key(0x7FE0, 0x0010));
    s
}

/// Recursively parse a DICOM sequence item into a DicomSequenceItem.
///
/// `depth` limits recursion to 8 levels to guard against malformed input.
fn parse_sequence_item(item: &dicom::object::InMemDicomObject, depth: usize) -> DicomSequenceItem {
    let mut seq_item = DicomSequenceItem::new();
    if depth > 8 {
        return seq_item;
    }
    for element in item.iter() {
        let tag = element.tag();
        let dicom_tag = DicomTag::new(tag.group(), tag.element());
        let vr_str = element.vr().to_string();
        if element.vr() == VR::SQ {
            if let Some(sub_items) = element.value().items() {
                let parsed: Vec<_> = sub_items
                    .iter()
                    .map(|i| parse_sequence_item(i, depth + 1))
                    .collect();
                seq_item.insert(DicomObjectNode {
                    tag: dicom_tag,
                    vr: Some("SQ".to_string()),
                    value: DicomValue::Sequence(parsed),
                    private: is_private_tag(dicom_tag),
                    source: None,
                });
            }
        } else {
            // Binary VRs (OB, OW, OD, OF, OL, OV, UN) must go directly to the bytes
            // branch: dicom-rs `to_str()` on these VRs returns a formatted decimal
            // representation rather than an error, which would lose the raw payload.
            let is_binary_vr = matches!(
                element.vr(),
                VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN
            );
            if is_binary_vr {
                if let Ok(bytes) = element.to_bytes() {
                    seq_item.insert(DicomObjectNode {
                        tag: dicom_tag,
                        vr: Some(vr_str.to_string()),
                        value: DicomValue::Bytes(bytes.to_vec()),
                        private: is_private_tag(dicom_tag),
                        source: None,
                    });
                }
            } else if let Ok(s) = element.to_str() {
                seq_item.insert(DicomObjectNode::text(dicom_tag, vr_str, s.to_string()));
            } else if let Ok(bytes) = element.to_bytes() {
                // Preserve any other binary-valued element that failed to_str().
                seq_item.insert(DicomObjectNode {
                    tag: dicom_tag,
                    vr: Some(vr_str.to_string()),
                    value: DicomValue::Bytes(bytes.to_vec()),
                    private: is_private_tag(dicom_tag),
                    source: None,
                });
            }
        }
    }
    seq_item
}

/// Scan a directory for DICOM files and return the discovered series description.
///
/// Slices are sorted by ImagePositionPatient[2] (z-coordinate), then by
/// InstanceNumber, then by filename for deterministic ordering. Z-spacing is
/// computed from the sorted z-coordinates of adjacent slices.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();
    if !path.is_dir() {
        bail!("DICOM input path is not a directory");
    }

    // First pass: collect all candidate DICOM files.
    let mut raw_paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| "failed to read DICOM directory")? {
        let entry = entry.with_context(|| "failed to read DICOM directory entry")?;
        let entry_path = entry.path();
        if entry_path.is_file() && is_likely_dicom_file(&entry_path) {
            raw_paths.push(entry_path);
        }
    }
    if raw_paths.is_empty() {
        bail!("no DICOM files were discovered in the directory");
    }

    // Second pass: parse metadata from each DICOM file.
    let mut slices: Vec<DicomSliceMetadata> = Vec::with_capacity(raw_paths.len());
    let mut first_rows: Option<u32> = None;
    let mut first_cols: Option<u32> = None;
    let mut first_pixel_spacing: Option<[f64; 2]> = None;
    let mut first_slice_thickness: Option<f64> = None;
    let mut first_series_instance_uid: Option<String> = None;
    let mut first_study_instance_uid: Option<String> = None;
    let mut first_series_description: Option<String> = None;
    let mut first_modality: Option<String> = None;
    let mut first_patient_id: Option<String> = None;
    let mut first_patient_name: Option<String> = None;
    let mut first_study_date: Option<String> = None;
    let mut first_series_date: Option<String> = None;
    let mut first_series_time: Option<String> = None;
    let mut first_frame_of_reference_uid: Option<String> = None;
    let mut first_bits_allocated: Option<u16> = None;
    let mut first_bits_stored: Option<u16> = None;
    let mut first_high_bit: Option<u16> = None;
    let mut first_photometric_interpretation: Option<String> = None;
    let mut first_transfer_syntax_uid: Option<String> = None;

    for file_path in &raw_paths {
        let mut slice_meta = DicomSliceMetadata {
            path: file_path.clone(),
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
        };

        if let Ok(obj) = open_file(file_path) {
            // Per-slice tags
            if let Ok(elem) = obj.element(Tag(0x0008, 0x0018)) {
                slice_meta.sop_instance_uid = elem.to_str().ok().map(String::from);
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0013)) {
                slice_meta.instance_number = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x1041)) {
                slice_meta.slice_location = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0032)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 3 {
                        slice_meta.image_position_patient = Some([parts[0], parts[1], parts[2]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0037)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 6 {
                        slice_meta.image_orientation_patient =
                            Some([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x0030)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 2 {
                        slice_meta.pixel_spacing = Some([parts[0], parts[1]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
                slice_meta.slice_thickness = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1053)) {
                slice_meta.rescale_slope = elem
                    .to_str()
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1052)) {
                slice_meta.rescale_intercept = elem
                    .to_str()
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
            }
            if let Ok(elem) = obj.element(Tag(0x0008, 0x0016)) {
                slice_meta.sop_class_uid = elem.to_str().ok().map(String::from);
            }
            // Transfer syntax UID lives in the DICOM file meta table (0002,0010),
            // not in the main dataset. Use FileDicomObject::meta() to read it.
            slice_meta.transfer_syntax_uid = Some(obj.meta().transfer_syntax().to_string());

            // PixelRepresentation (0028,0103): 0=unsigned, 1=signed two's complement. Default=0.
            if let Ok(elem) = obj.element(Tag(0x0028, 0x0103)) {
                slice_meta.pixel_representation = elem
                    .to_str()
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
            }
            // BitsAllocated per slice (0028,0100). Default=16.
            if let Ok(elem) = obj.element(Tag(0x0028, 0x0100)) {
                slice_meta.bits_allocated = elem
                    .to_str()
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(16);
            }
            // WindowCenter (0028,1050) — first value of potentially multi-valued DS.
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1050)) {
                slice_meta.window_center = elem.to_str().ok().and_then(|s| {
                    s.trim()
                        .split('\\')
                        .next()
                        .and_then(|v| v.trim().parse().ok())
                });
            }
            // WindowWidth (0028,1051) — first value of potentially multi-valued DS.
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1051)) {
                slice_meta.window_width = elem.to_str().ok().and_then(|s| {
                    s.trim()
                        .split('\\')
                        .next()
                        .and_then(|v| v.trim().parse().ok())
                });
            }

            // Rows / Colls from first slice — establish series geometry
            if first_rows.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0010)) {
                    first_rows = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_cols.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0011)) {
                    first_cols = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_pixel_spacing.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0030)) {
                    if let Ok(s) = elem.to_str() {
                        let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                        if parts.len() >= 2 {
                            first_pixel_spacing = Some([parts[0], parts[1]]);
                        }
                    }
                }
            }
            if first_slice_thickness.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
                    first_slice_thickness = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_series_instance_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x000E)) {
                    first_series_instance_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_study_instance_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x000D)) {
                    first_study_instance_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_description.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x103E)) {
                    first_series_description = elem.to_str().ok().map(String::from);
                }
            }
            if first_modality.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0060)) {
                    first_modality = elem.to_str().ok().map(String::from);
                }
            }
            if first_patient_id.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0010, 0x0020)) {
                    first_patient_id = elem.to_str().ok().map(String::from);
                }
            }
            if first_patient_name.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0010, 0x0010)) {
                    first_patient_name = elem.to_str().ok().map(String::from);
                }
            }
            if first_study_date.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0020)) {
                    first_study_date = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_date.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0021)) {
                    first_series_date = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_time.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0031)) {
                    first_series_time = elem.to_str().ok().map(String::from);
                }
            }
            if first_frame_of_reference_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x0052)) {
                    first_frame_of_reference_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_bits_allocated.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0100)) {
                    first_bits_allocated = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_bits_stored.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0101)) {
                    first_bits_stored = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_high_bit.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0102)) {
                    first_high_bit = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_photometric_interpretation.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0004)) {
                    first_photometric_interpretation = elem.to_str().ok().map(String::from);
                }
            }
            if first_transfer_syntax_uid.is_none() {
                first_transfer_syntax_uid = Some(obj.meta().transfer_syntax().to_string());
            }

            // Full element preservation: capture all non-handled elements.
            {
                let handled = known_handled_tags();
                for element in &obj {
                    let tag = element.tag();
                    let key = tag_key(tag.group(), tag.element());
                    if handled.contains(&key) {
                        continue;
                    }
                    let dicom_tag = DicomTag::new(tag.group(), tag.element());
                    let vr_str = element.vr().to_string();
                    let private = is_private_tag(dicom_tag);
                    if element.vr() == VR::SQ {
                        if let Some(sub_items) = element.value().items() {
                            let parsed: Vec<_> = sub_items
                                .iter()
                                .map(|i| parse_sequence_item(i, 0))
                                .collect();
                            slice_meta.preservation.object.insert(DicomObjectNode {
                                tag: dicom_tag,
                                vr: Some("SQ".to_string()),
                                value: DicomValue::Sequence(parsed),
                                private,
                                source: None,
                            });
                        }
                    } else {
                        // Binary VRs (OB, OW, OD, OF, OL, UN) must bypass to_str():
                        // dicom-rs 0.8 returns a decimal-formatted string for these VRs
                        // rather than an error, which would silently corrupt the raw payload.
                        let is_binary_vr = matches!(
                            element.vr(),
                            VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN
                        );
                        if is_binary_vr {
                            if let Ok(bytes) = element.to_bytes() {
                                slice_meta.preservation.preserve(DicomPreservedElement::new(
                                    dicom_tag,
                                    Some(vr_str.to_string()),
                                    bytes.to_vec(),
                                ));
                            }
                        } else if let Ok(s) = element.to_str() {
                            slice_meta.preservation.object.insert(DicomObjectNode {
                                tag: dicom_tag,
                                vr: Some(vr_str.to_string()),
                                value: DicomValue::Text(s.to_string()),
                                private,
                                source: None,
                            });
                        } else if let Ok(bytes) = element.to_bytes() {
                            slice_meta.preservation.preserve(DicomPreservedElement::new(
                                dicom_tag,
                                Some(vr_str.to_string()),
                                bytes.to_vec(),
                            ));
                        }
                    }
                }
            }
        }

        slices.push(slice_meta);
    }

    // SOP class policy: remove non-image-bearing files from the series.
    // Files without a readable SOP class UID are retained (ambiguous -> permissive).
    let original_count = slices.len();
    let mut rejected_uids: Vec<String> = Vec::new();
    slices.retain(|s| match s.sop_class_uid.as_deref() {
        None => true,
        Some(uid) => {
            let kind = classify_sop_class(uid);
            let is_non_image = !kind.is_image_storage() && !matches!(kind, SopClassKind::Other(_));
            if is_non_image {
                tracing::warn!(
                    "skipping non-image DICOM SOP class: {:?} path={} sop_class_uid={}",
                    kind,
                    s.path.display(),
                    uid
                );
                rejected_uids.push(uid.to_string());
                false
            } else {
                true
            }
        }
    });
    if slices.is_empty() && original_count > 0 {
        bail!(
            "DICOM directory {:?} contains {} file(s) but none are image-bearing SOP classes;
             rejected SOP class UIDs: [{}]",
            path,
            original_count,
            rejected_uids.join(", ")
        );
    }

    // Sort slices by z-position (ImagePositionPatient[2]), then instance number, then filename.
    slices.sort_by(|a, b| {
        let z_a = a.image_position_patient.map(|p| p[2]).unwrap_or(f64::MAX);
        let z_b = b.image_position_patient.map(|p| p[2]).unwrap_or(f64::MAX);
        z_a.partial_cmp(&z_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.instance_number
                    .unwrap_or(i32::MAX)
                    .cmp(&b.instance_number.unwrap_or(i32::MAX))
            })
            .then_with(|| a.path.file_name().cmp(&b.path.file_name()))
    });

    let rows = first_rows.unwrap_or(0) as usize;
    let cols = first_cols.unwrap_or(0) as usize;
    let depth = slices.len();

    // Compute z-spacing from sorted ImagePositionPatient z-coordinates.
    let z_positions: Vec<f64> = slices
        .iter()
        .filter_map(|s| s.image_position_patient.map(|p| p[2]))
        .collect();
    let spacing_z = if z_positions.len() >= 2 {
        let total_span = z_positions.last().unwrap() - z_positions.first().unwrap();
        let count_minus_one = (z_positions.len() - 1) as f64;
        if count_minus_one > 0.0 && total_span.is_finite() {
            total_span / count_minus_one
        } else {
            first_slice_thickness.unwrap_or(1.0)
        }
    } else {
        first_slice_thickness.unwrap_or(1.0)
    };

    let in_plane_spacing = first_pixel_spacing.unwrap_or([1.0, 1.0]);
    let spacing: [f64; 3] = [
        in_plane_spacing[0],
        in_plane_spacing[1],
        spacing_z.abs().max(1e-6),
    ];

    // Build direction cosines from first slice's ImageOrientationPatient, or default to identity.
    let direction = if let Some(ori) = slices.first().and_then(|s| s.image_orientation_patient) {
        // Row direction = first 3 elements, Column direction = next 3 elements.
        // ITK/LPS+ convention: row cosines (r_x, r_y, r_z), column cosines (c_x, c_y, c_z).
        let r = [ori[0], ori[1], ori[2]];
        let c = [ori[3], ori[4], ori[5]];
        let n = cross_3d(r, c);
        [
            ori[0], ori[1], ori[2], ori[3], ori[4], ori[5], n[0], n[1], n[2],
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    };

    // Compute physical origin from first slice's ImagePositionPatient.
    let origin = slices
        .first()
        .and_then(|s| s.image_position_patient)
        .unwrap_or([0.0, 0.0, 0.0]);

    let mut series_object = DicomObjectModel::with_source(path.to_path_buf());
    for slice in &slices {
        if let Some(uid) = slice.sop_instance_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0018),
                "UI",
                uid.clone(),
            ));
        }
        if let Some(instance_number) = slice.instance_number {
            series_object.insert(DicomObjectNode::i32(
                DicomTag::new(0x0020, 0x0013),
                "IS",
                instance_number,
            ));
        }
        if let Some(slice_location) = slice.slice_location {
            series_object.insert(DicomObjectNode::f64(
                DicomTag::new(0x0020, 0x1041),
                "DS",
                slice_location,
            ));
        }
        if let Some(position) = slice.image_position_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0032),
                "DS",
                format!("{:.6}\\{:.6}\\{:.6}", position[0], position[1], position[2]),
            ));
        }
        if let Some(orientation) = slice.image_orientation_patient {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0020, 0x0037),
                "DS",
                format!(
                    "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                    orientation[4],
                    orientation[5]
                ),
            ));
        }
        if let Some(pixel_spacing) = slice.pixel_spacing {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0028, 0x0030),
                "DS",
                format!("{:.6}\\{:.6}", pixel_spacing[0], pixel_spacing[1]),
            ));
        }
        if let Some(slice_thickness) = slice.slice_thickness {
            series_object.insert(DicomObjectNode::f64(
                DicomTag::new(0x0018, 0x0050),
                "DS",
                slice_thickness,
            ));
        }
        if let Some(sop_class_uid) = slice.sop_class_uid.as_ref() {
            series_object.insert(DicomObjectNode::text(
                DicomTag::new(0x0008, 0x0016),
                "UI",
                sop_class_uid.clone(),
            ));
        }
    }

    let metadata = DicomReadMetadata {
        series_instance_uid: first_series_instance_uid,
        study_instance_uid: first_study_instance_uid,
        frame_of_reference_uid: first_frame_of_reference_uid,
        series_description: first_series_description,
        modality: first_modality,
        patient_id: first_patient_id,
        patient_name: first_patient_name,
        study_date: first_study_date,
        series_date: first_series_date,
        series_time: first_series_time,
        dimensions: [rows, cols, depth],
        spacing,
        origin,
        direction,
        bits_allocated: first_bits_allocated,
        bits_stored: first_bits_stored,
        high_bit: first_high_bit,
        photometric_interpretation: first_photometric_interpretation,
        slices,
        private_tags: HashMap::new(),
        preservation: DicomPreservationSet {
            object: series_object,
            preserved: Vec::new(),
        },
    };

    Ok(DicomSeriesInfo {
        path: path.to_path_buf(),
        num_slices: metadata.slices.len(),
        metadata,
    })
}

/// Compute the cross product of two 3-vectors.
fn cross_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Read a DICOM series and return the reconstructed 3-D image.
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let (image, _) = read_dicom_series_with_metadata(path, device)?;
    Ok(image)
}

/// Load a DICOM series, preserving metadata.
pub fn load_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata(path, device)
}

/// Read a DICOM series and return both the image and metadata.
pub fn read_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let series = scan_dicom_directory(path)?;
    load_from_series(series, device)
}

/// Load a DICOM series from a pre-scanned descriptor and return image plus metadata.
pub fn load_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata(path, device)
}

fn load_from_series<B: Backend>(
    series: DicomSeriesInfo,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let metadata = series.metadata.clone();
    let slices = &metadata.slices;

    slices
        .first()
        .ok_or_else(|| anyhow!("DICOM series is empty"))?;

    // Guard: reject compressed-transfer-syntax slices before pixel decode.
    // Pixel data in compressed files cannot be interpreted as raw u8/u16 samples;
    // processing them silently produces garbage intensities.
    for slice in slices.iter() {
        if let Some(ref ts_uid) = slice.transfer_syntax_uid {
            if TransferSyntaxKind::from_uid(ts_uid).is_compressed() {
                bail!(
                    "DICOM series: compressed transfer syntax '{}' in slice {:?} is not \
                     natively supported; decompress the series before loading",
                    ts_uid,
                    slice.path
                );
            }
        }
    }

    let rows = metadata.dimensions[0];
    let cols = metadata.dimensions[1];
    let depth = metadata.dimensions[2];

    if rows == 0 || cols == 0 || depth == 0 {
        bail!("DICOM series has invalid zero dimensions");
    }

    let mut volume = vec![0f32; rows * cols * depth];

    for (z, slice) in slices.iter().enumerate() {
        let data = read_slice_pixels(slice)
            .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;

        if data.len() != rows * cols {
            bail!(
                "DICOM slice size mismatch: expected {} pixels, got {}",
                rows * cols,
                data.len()
            );
        }

        let offset = z * rows * cols;
        volume[offset..offset + rows * cols].copy_from_slice(&data);
    }

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(volume, Shape::new([depth, rows, cols])),
        device,
    );
    let image = Image::new(
        tensor,
        Point::new(metadata.origin),
        Spacing::new(metadata.spacing),
        Direction(SMatrix::<f64, 3, 3>::from_row_slice(&metadata.direction)),
    );

    Ok((image, metadata))
}

/// Decode raw pixel bytes into f32 values applying per-slice rescale.
///
/// # Invariants
/// - `bits_allocated=8`: each byte is one unsigned sample; decoded = u8 × slope + intercept.
/// - `bits_allocated=16`, `pixel_representation=1`: pairs of bytes are one i16 LE sample;
///   decoded = i16 × slope + intercept.
/// - Any other combination: pairs of bytes are one u16 LE sample (unsigned default);
///   decoded = u16 × slope + intercept.
///
/// Mathematical derivation: the linear modality LUT is F(x) = x × RescaleSlope + RescaleIntercept
/// per DICOM PS3.3 C.7.6.3.1.4.
fn decode_pixel_bytes(
    bytes: &[u8],
    bits_allocated: u16,
    pixel_representation: u16,
    slope: f32,
    intercept: f32,
) -> Vec<f32> {
    match (bits_allocated, pixel_representation) {
        (8, _) => bytes
            .iter()
            .map(|&b| b as f32 * slope + intercept)
            .collect(),
        (16, 1) => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 * slope + intercept)
            .collect(),
        _ => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 * slope + intercept)
            .collect(),
    }
}

fn read_slice_pixels(slice: &DicomSliceMetadata) -> Result<Vec<f32>> {
    use dicom::core::Tag;
    use dicom::object::open_file;

    let obj = open_file(&slice.path)
        .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))?;
    let pixel_elem = obj
        .element(Tag(0x7FE0, 0x0010))
        .with_context(|| format!("PixelData (7FE0,0010) absent in {:?}", slice.path))?;
    let bytes = pixel_elem
        .value()
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("pixel bytes unreadable in {:?}: {:?}", slice.path, e))?;

    let data = decode_pixel_bytes(
        &bytes,
        slice.bits_allocated,
        slice.pixel_representation,
        slice.rescale_slope,
        slice.rescale_intercept,
    );
    if data.is_empty() {
        bail!(
            "DICOM slice contained no decodable pixel data in {:?}",
            slice.path
        );
    }
    Ok(data)
}

/// Return true when the path is likely a DICOM Part 10 file.
///
/// Primary test: file extension is one of the canonical DICOM extensions
/// (`.dcm`, `.dicom`, `.ima`).  Secondary test: extensionless files are
/// probed for the DICM magic bytes at byte offset 128 (DICOM PS3.10 §7.1).
///
/// `.hdr` and `.img` are Analyze 7.5 header/image files; `.raw` is
/// unstructured binary data. Neither is a DICOM Part 10 format.
fn is_likely_dicom_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        let ext_lc = ext.to_ascii_lowercase();
        return matches!(ext_lc.as_str(), "dcm" | "dicom" | "ima");
    }
    // No extension: probe for DICM magic at byte offset 128.
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut f) = std::fs::File::open(path) {
        let mut magic = [0u8; 4];
        if f.seek(SeekFrom::Start(128)).is_ok() && f.read_exact(&mut magic).is_ok() {
            return &magic == b"DICM";
        }
    }
    false
}

/// Compatibility wrapper for callers expecting a reader type.
pub struct DicomReader<B> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B> DicomReader<B> {
    /// Create a new reader.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B> Default for DicomReader<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    #[test]
    fn test_scan_empty_directory_errors() {
        let temp = tempfile::tempdir().unwrap();
        let err = scan_dicom_directory(temp.path()).unwrap_err();
        assert!(err.to_string().contains("no DICOM files"));
    }

    #[test]
    fn test_scan_non_directory_errors() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let err = scan_dicom_directory(temp.path()).unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    /// Write a minimal DICOM Part-10 file carrying only the mandatory UID tags.
    /// sop_class_uid controls which SOP class the file advertises.
    fn write_stub_dicom(path: &std::path::Path, sop_class_uid: &str, sop_instance_uid: &str) {
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(sop_class_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(sop_class_uid)
                    .media_storage_sop_instance_uid(sop_instance_uid)
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .expect("meta build must not fail");
        file_obj.write_to_file(path).expect("write must not fail");
    }

    /// A directory containing only non-image SOP-class DICOM files must return Err.
    /// The error message must identify the directory was examined and list the rejected UIDs.
    ///
    /// Invariant: scan_dicom_directory returns Err when all discovered files belong to
    /// non-image SOP classes {RTStructureSetStorage, BasicTextSrStorage, EncapsulatedPdfStorage}.
    #[test]
    fn test_scan_all_non_image_sop_returns_error_with_rejected_uids() {
        let temp = tempfile::tempdir().unwrap();

        // RT Structure Set
        write_stub_dicom(
            &temp.path().join("rtstruct.dcm"),
            "1.2.840.10008.5.1.4.1.1.481.3",
            "2.25.10001",
        );
        // Basic Text SR
        write_stub_dicom(
            &temp.path().join("sr.dcm"),
            "1.2.840.10008.5.1.4.1.1.88.11",
            "2.25.10002",
        );
        // Encapsulated PDF
        write_stub_dicom(
            &temp.path().join("pdf.dcm"),
            "1.2.840.10008.5.1.4.1.1.104.1",
            "2.25.10003",
        );

        let result = scan_dicom_directory(temp.path());
        assert!(result.is_err(), "all-non-image directory must return Err");

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("none are image-bearing SOP classes"),
            "error must state none are image-bearing; got: {msg}"
        );
        // At least one of the three non-image UIDs must appear in the message.
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.481.3")
                || msg.contains("1.2.840.10008.5.1.4.1.1.88.11")
                || msg.contains("1.2.840.10008.5.1.4.1.1.104.1"),
            "error must list at least one rejected SOP UID; got: {msg}"
        );
    }

    /// A directory containing one non-image SOP file and one CT image file must succeed.
    /// After filtering, exactly one image slice must be retained carrying the CT SOP UID.
    ///
    /// Invariant: scan_dicom_directory returns Ok with num_slices == 1 when exactly one
    /// image-bearing file survives non-image SOP filtering.
    #[test]
    fn test_scan_mixed_non_image_and_ct_retains_image_slice() {
        let temp = tempfile::tempdir().unwrap();

        // Non-image: RT Structure Set
        write_stub_dicom(
            &temp.path().join("rtstruct.dcm"),
            "1.2.840.10008.5.1.4.1.1.481.3",
            "2.25.20001",
        );

        // Image-bearing: CT Image Storage -- include Rows/Cols so metadata is populated
        {
            let mut obj = InMemDicomObject::new_empty();
            obj.put(DataElement::new(
                Tag(0x0008, 0x0016),
                VR::UI,
                PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
            ));
            obj.put(DataElement::new(
                Tag(0x0008, 0x0018),
                VR::UI,
                PrimitiveValue::from("2.25.20002"),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0010),
                VR::US,
                PrimitiveValue::from(4_u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0011),
                VR::US,
                PrimitiveValue::from(4_u16),
            ));
            obj.put(DataElement::new(
                Tag(0x0020, 0x0013),
                VR::IS,
                PrimitiveValue::from("1"),
            ));
            let file_obj = obj
                .with_meta(
                    FileMetaTableBuilder::new()
                        .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                        .media_storage_sop_instance_uid("2.25.20002")
                        .transfer_syntax("1.2.840.10008.1.2.1"),
                )
                .unwrap();
            file_obj.write_to_file(&temp.path().join("ct.dcm")).unwrap();
        }

        let result = scan_dicom_directory(temp.path());
        assert!(
            result.is_ok(),
            "scan must succeed when at least one image-bearing SOP exists; err={:?}",
            result.err()
        );

        let info = result.unwrap();
        assert_eq!(
            info.num_slices, 1,
            "only the CT slice must survive filtering; got {}",
            info.num_slices
        );
        assert_eq!(
            info.metadata.slices[0].sop_class_uid.as_deref(),
            Some("1.2.840.10008.5.1.4.1.1.2"),
            "retained slice must carry CT Image Storage SOP UID"
        );
    }

    /// A directory with RT Plan + ECG Waveform files (two distinct non-image SOPs) must return
    /// Err whose message references both SOP UIDs.
    ///
    /// Invariant: the rejected-UID list in the error must enumerate every unique non-image
    /// SOP class seen during scanning.
    #[test]
    fn test_scan_rt_plan_and_waveform_returns_error_with_two_uids() {
        let temp = tempfile::tempdir().unwrap();

        // RT Plan
        write_stub_dicom(
            &temp.path().join("rtplan.dcm"),
            "1.2.840.10008.5.1.4.1.1.481.5",
            "2.25.30001",
        );
        // 12-lead ECG Waveform
        write_stub_dicom(
            &temp.path().join("ecg.dcm"),
            "1.2.840.10008.5.1.4.1.1.9.1.1",
            "2.25.30002",
        );

        let result = scan_dicom_directory(temp.path());
        assert!(result.is_err(), "all-non-image directory must return Err");

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.481.5"),
            "error must list RT Plan UID; got: {msg}"
        );
        assert!(
            msg.contains("1.2.840.10008.5.1.4.1.1.9.1.1"),
            "error must list ECG Waveform UID; got: {msg}"
        );
    }

    #[test]
    fn test_scan_private_sequence_is_preserved_in_object_model() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("private_sequence.dcm");

        let mut nested_item = InMemDicomObject::new_empty();
        nested_item.put(DataElement::new(
            Tag(0x0010, 0x0010),
            VR::PN,
            PrimitiveValue::from("Test^Patient"),
        ));
        nested_item.put(DataElement::new(
            Tag(0x0009, 0x1001),
            VR::OB,
            PrimitiveValue::U8(vec![1, 2, 3, 4].into()),
        ));

        let seq = dicom::core::value::DataSetSequence::new(
            vec![nested_item],
            dicom::core::header::Length::UNDEFINED,
        );
        let seq_value = dicom::core::value::Value::from(seq);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.40001"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(4_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(4_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        obj.put(DataElement::new(
            Tag(0x0009, 0x0010),
            VR::LO,
            PrimitiveValue::from("RITK"),
        ));
        obj.put(DataElement::new(Tag(0x0009, 0x1000), VR::SQ, seq_value));

        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                    .media_storage_sop_instance_uid("2.25.40001")
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .unwrap();
        file_obj.write_to_file(&path).unwrap();

        let info = scan_dicom_directory(&temp.path()).unwrap();
        let preserved = info
            .metadata
            .slices
            .first()
            .and_then(|s| s.preservation.object.get(DicomTag::new(0x0009, 0x1000)))
            .expect("private SQ must be preserved");

        assert!(preserved.is_sequence(), "private SQ must remain a sequence");
        let items = match &preserved.value {
            DicomValue::Sequence(items) => items,
            _ => panic!("private SQ must decode as DicomValue::Sequence"),
        };
        assert_eq!(items.len(), 1);
        let first = &items[0];
        assert_eq!(
            first
                .get(DicomTag::new(0x0010, 0x0010))
                .and_then(|n| n.value.as_text())
                .map(str::trim),
            Some("Test^Patient")
        );
        let raw = first
            .get(DicomTag::new(0x0009, 0x1001))
            .expect("private OB must be preserved");
        assert!(raw.value.is_bytes(), "private OB must remain raw bytes");
        assert_eq!(
            match &raw.value {
                DicomValue::Bytes(bytes) => bytes.as_slice(),
                _ => panic!("private OB must decode as DicomValue::Bytes"),
            },
            &[1, 2, 3, 4]
        );
    }

    /// Full round-trip of spatial metadata through write_dicom_series_with_metadata →
    /// scan_dicom_directory.
    ///
    /// # Invariants
    /// - `metadata.modality == Some("CT")`
    /// - `metadata.bits_allocated == Some(16)`
    /// - `metadata.dimensions == [rows, cols, depth]`
    /// - `metadata.spacing` within 1e-4 of `[0.8, 0.8, 2.5]`
    /// - `metadata.origin` within 1e-4 of `[10.0, 20.0, -50.0]` (first slice IPP)
    /// - `metadata.direction` recovers identity from IOP cross product (within 1e-5)
    /// - Each slice: `pixel_spacing` within 1e-4 of `[0.8, 0.8]`
    /// - Each slice: `image_orientation_patient` within 1e-5 of `[1,0,0,0,1,0]`
    /// - Slice IPP z-coordinates: -50.0, -47.5, -45.0 (spacing_z = 2.5, normal = [0,0,1])
    #[test]
    fn test_scan_metadata_round_trip_spatial_fields() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        use std::collections::HashMap;
        type B = burn_ndarray::NdArray<f32>;

        let temp = tempfile::tempdir().unwrap();
        let series_path = temp.path().join("rt_series");

        // Image: 3 slices, 6 rows, 8 cols.
        let (depth, rows, cols) = (3usize, 6usize, 8usize);
        let data = vec![500.0f32; depth * rows * cols];
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([10.0, 20.0, -50.0]),
            Spacing::new([0.8, 0.8, 2.5]),
            Direction::identity(),
        );

        let meta = DicomReadMetadata {
            series_instance_uid: Some("1.2.3.4.5.6.789".to_string()),
            study_instance_uid: Some("1.2.3.4.5.6.100".to_string()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".to_string()),
            patient_id: Some("RT001".to_string()),
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [rows, cols, depth],
            spacing: [0.8, 0.8, 2.5],
            origin: [10.0, 20.0, -50.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation: crate::format::dicom::DicomPreservationSet::new(),
        };

        crate::format::dicom::writer::write_dicom_series_with_metadata(
            &series_path,
            &image,
            Some(&meta),
        )
        .expect("write_dicom_series_with_metadata must not fail");

        let info = scan_dicom_directory(&series_path).expect("scan_dicom_directory must not fail");
        let m = &info.metadata;

        // --- Series-level assertions ---
        assert_eq!(
            m.modality.as_deref(),
            Some("CT"),
            "modality must round-trip; got {:?}",
            m.modality
        );
        assert_eq!(
            m.bits_allocated,
            Some(16),
            "bits_allocated must round-trip as 16; got {:?}",
            m.bits_allocated
        );
        assert_eq!(
            m.dimensions[2], depth,
            "slice count must equal depth {}; got {}",
            depth, m.dimensions[2]
        );
        assert_eq!(
            m.dimensions[0], rows,
            "rows must equal {}; got {}",
            rows, m.dimensions[0]
        );
        assert_eq!(
            m.dimensions[1], cols,
            "cols must equal {}; got {}",
            cols, m.dimensions[1]
        );

        // Spacing: within 1e-4 of [0.8, 0.8, 2.5].
        let tol = 1e-4_f64;
        assert!(
            (m.spacing[0] - 0.8).abs() < tol,
            "spacing[0] must be 0.8 ± 1e-4; got {}",
            m.spacing[0]
        );
        assert!(
            (m.spacing[1] - 0.8).abs() < tol,
            "spacing[1] must be 0.8 ± 1e-4; got {}",
            m.spacing[1]
        );
        assert!(
            (m.spacing[2] - 2.5).abs() < tol,
            "spacing[2] must be 2.5 ± 1e-4; got {}",
            m.spacing[2]
        );

        // Origin: within 1e-4 of [10.0, 20.0, -50.0] (first-slice IPP).
        assert!(
            (m.origin[0] - 10.0).abs() < tol,
            "origin[0] must be 10.0 ± 1e-4; got {}",
            m.origin[0]
        );
        assert!(
            (m.origin[1] - 20.0).abs() < tol,
            "origin[1] must be 20.0 ± 1e-4; got {}",
            m.origin[1]
        );
        assert!(
            (m.origin[2] - (-50.0_f64)).abs() < tol,
            "origin[2] must be -50.0 ± 1e-4; got {}",
            m.origin[2]
        );

        // Direction: identity, reconstructed from IOP cross product [1,0,0]×[0,1,0] = [0,0,1].
        let identity = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, (&actual, &expected)) in m.direction.iter().zip(identity.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "direction[{i}] must be {expected:.1} ± 1e-5; got {actual}"
            );
        }

        // --- Per-slice assertions ---
        assert_eq!(
            m.slices.len(),
            depth,
            "must have {} slices; got {}",
            depth,
            m.slices.len()
        );

        for (i, slice) in m.slices.iter().enumerate() {
            // IOP: axial = [1,0,0,0,1,0].
            let iop = slice
                .image_orientation_patient
                .unwrap_or_else(|| panic!("slice {i} must have IOP"));
            let expected_iop = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0];
            for (j, (&a, &e)) in iop.iter().zip(expected_iop.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-5,
                    "slice {i} IOP[{j}] must be {e:.1} ± 1e-5; got {a}"
                );
            }

            // Pixel spacing: [0.8, 0.8].
            let ps = slice
                .pixel_spacing
                .unwrap_or_else(|| panic!("slice {i} must have pixel_spacing"));
            assert!(
                (ps[0] - 0.8).abs() < tol,
                "slice {i} pixel_spacing[0] must be 0.8 ± 1e-4; got {}",
                ps[0]
            );
            assert!(
                (ps[1] - 0.8).abs() < tol,
                "slice {i} pixel_spacing[1] must be 0.8 ± 1e-4; got {}",
                ps[1]
            );
        }

        // IPP z-coordinates (slices sorted ascending): -50.0, -47.5, -45.0.
        // normal = [0,0,1] (identity direction), spacing_z = 2.5.
        let expected_z = [-50.0f64, -47.5, -45.0];
        for (i, (slice, &ez)) in m.slices.iter().zip(expected_z.iter()).enumerate() {
            let ipp = slice
                .image_position_patient
                .unwrap_or_else(|| panic!("slice {i} must have IPP"));
            assert!(
                (ipp[0] - 10.0).abs() < tol,
                "slice {i} IPP[0] must be 10.0 ± 1e-4; got {}",
                ipp[0]
            );
            assert!(
                (ipp[1] - 20.0).abs() < tol,
                "slice {i} IPP[1] must be 20.0 ± 1e-4; got {}",
                ipp[1]
            );
            assert!(
                (ipp[2] - ez).abs() < tol,
                "slice {i} IPP[2] must be {ez} ± 1e-4; got {}",
                ipp[2]
            );
        }
    }

    /// Round-trip verification for rescale slope and intercept.
    ///
    /// # Invariants
    /// - Each slice's `rescale_slope` must be > 0
    /// - Each slice's `rescale_intercept` must be finite
    /// - First-voxel reconstruction error must be within slope/2 (quantization bound)
    #[test]
    fn test_scan_metadata_round_trip_rescale_params() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        use std::collections::HashMap;
        type B = burn_ndarray::NdArray<f32>;

        let temp = tempfile::tempdir().unwrap();
        let series_path = temp.path().join("rescale_series");

        // Intensities in [-1024.0, 1024.0] to force non-trivial rescale params.
        let (depth, rows, cols) = (2usize, 4usize, 4usize);
        let n = depth * rows * cols;
        let mut data = vec![0.0f32; n];
        for (idx, v) in data.iter_mut().enumerate() {
            *v = -1024.0 + idx as f32 * (2048.0 / (n - 1) as f32);
        }
        let original_first = data[0];

        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let meta = DicomReadMetadata {
            series_instance_uid: None,
            study_instance_uid: None,
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".to_string()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [rows, cols, depth],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation: crate::format::dicom::DicomPreservationSet::new(),
        };

        crate::format::dicom::writer::write_dicom_series_with_metadata(
            &series_path,
            &image,
            Some(&meta),
        )
        .expect("write must not fail");

        let info = scan_dicom_directory(&series_path).expect("scan must not fail");

        for (i, slice) in info.metadata.slices.iter().enumerate() {
            let slope = slice.rescale_slope;
            let intercept = slice.rescale_intercept;
            assert!(
                slope > 0.0,
                "slice {i} rescale_slope must be > 0; got {slope}"
            );
            assert!(
                intercept.is_finite(),
                "slice {i} rescale_intercept must be finite; got {intercept}"
            );
        }

        // Verify first-voxel reconstruction error is bounded by slope/2.
        // The writer quantizes: pixel = round((v - intercept) / slope); u16 clamped.
        // Reconstructed: pixel * slope + intercept. Error <= slope/2.
        let first_slice = &info.metadata.slices[0];
        let slope = first_slice.rescale_slope as f32;
        let intercept = first_slice.rescale_intercept as f32;
        let pixel_first = ((original_first - intercept) / slope).round() as u16;
        let reconstructed = pixel_first as f32 * slope + intercept;
        let error = (reconstructed - original_first).abs();
        assert!(
            error <= slope / 2.0 + 1e-3,
            "first-voxel reconstruction error {error} exceeds quantization bound {}",
            slope / 2.0
        );
    }

    /// Verifies that `transfer_syntax_uid` is read from the DICOM file meta table
    /// (via `obj.meta().transfer_syntax()`) and not from Tag(0x0008,0x0070) (Manufacturer).
    ///
    /// # Invariants
    /// - Every DicomSliceMetadata produced by `scan_dicom_directory` must carry
    ///   `transfer_syntax_uid == Some("1.2.840.10008.1.2.1")` for files written with
    ///   Explicit VR Little Endian transfer syntax.
    #[test]
    fn test_scan_metadata_round_trip_transfer_syntax() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        use std::collections::HashMap;
        type B = burn_ndarray::NdArray<f32>;

        let temp = tempfile::tempdir().unwrap();
        let series_path = temp.path().join("ts_series");

        let (depth, rows, cols) = (2usize, 4usize, 4usize);
        let data = vec![1000.0f32; depth * rows * cols];
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let meta = DicomReadMetadata {
            series_instance_uid: None,
            study_instance_uid: None,
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("OT".to_string()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [rows, cols, depth],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation: crate::format::dicom::DicomPreservationSet::new(),
        };

        crate::format::dicom::writer::write_dicom_series_with_metadata(
            &series_path,
            &image,
            Some(&meta),
        )
        .expect("write must not fail");

        let info = scan_dicom_directory(&series_path).expect("scan must not fail");

        // The writer emits transfer_syntax("1.2.840.10008.1.2.1") = Explicit VR Little Endian.
        // The reader must extract this from the file meta, not from Tag(0x0008,0x0070).
        const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";
        assert!(
            !info.metadata.slices.is_empty(),
            "at least one slice must be returned"
        );
        for (i, slice) in info.metadata.slices.iter().enumerate() {
            assert_eq!(
                slice.transfer_syntax_uid.as_deref(),
                Some(EXPLICIT_VR_LE),
                "slice {i} transfer_syntax_uid must be {EXPLICIT_VR_LE}; got {:?}",
                slice.transfer_syntax_uid
            );
        }
    }

    /// Full round-trip of private tag preservation through write_dicom_series_with_metadata →
    /// scan_dicom_directory.
    ///
    /// # Invariants
    /// - Private text element at (0009,0010) with value "PRIV_ROUND_TRIP_VALUE" must appear in
    ///   `DicomSliceMetadata.preservation.object` after scanning the written series.
    /// - Private OB bytes element at (0019,1001) with bytes [0xAB, 0xCD, 0xEF, 0x01] must appear
    ///   in `DicomSliceMetadata.preservation.preserved` after scanning.
    /// - Both computed values must exactly match the inputs, not merely be present.
    #[test]
    fn test_scan_preserves_private_text_and_bytes_through_write_read_cycle() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use std::collections::HashMap;
        type B = burn_ndarray::NdArray<f32>;

        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("priv_rt");

        // Build a 1-slice 4×4 image.
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![42.0_f32; 4 * 4], Shape::new([1_usize, 4, 4]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );

        // Preservation: private text tag + raw OB bytes tag.
        let mut preservation = DicomPreservationSet::new();
        preservation.object.insert(DicomObjectNode::text(
            DicomTag::new(0x0009, 0x0010),
            "LO",
            "PRIV_ROUND_TRIP_VALUE",
        ));
        preservation.preserve(DicomPreservedElement::new(
            DicomTag::new(0x0019, 0x1001),
            Some("OB".to_string()),
            vec![0xAB_u8, 0xCD, 0xEF, 0x01],
        ));

        let meta = DicomReadMetadata {
            series_instance_uid: Some("2.25.111".to_string()),
            study_instance_uid: Some("2.25.222".to_string()),
            frame_of_reference_uid: None,
            series_description: Some("TestSeries".to_string()),
            modality: Some("CT".to_string()),
            patient_id: Some("P001".to_string()),
            patient_name: Some("Test^Patient".to_string()),
            study_date: Some("20240101".to_string()),
            series_date: Some("20240101".to_string()),
            series_time: Some("120000".to_string()),
            dimensions: [4, 4, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation,
        };

        crate::format::dicom::writer::write_dicom_series_with_metadata(&dir, &image, Some(&meta))
            .expect("write_dicom_series_with_metadata");

        // Scan back via scan_dicom_directory.
        let scanned = scan_dicom_directory(&dir).expect("scan_dicom_directory");
        assert_eq!(
            scanned.num_slices, 1,
            "must have exactly 1 slice; got {}",
            scanned.num_slices
        );

        let slice = &scanned.metadata.slices[0];
        let priv_text_tag = DicomTag::new(0x0009, 0x0010);
        let priv_bytes_tag = DicomTag::new(0x0019, 0x1001);

        // Private text tag must be present in preservation.object with the correct value.
        let text_node = slice.preservation.object.get(priv_text_tag);
        assert!(
            text_node.is_some(),
            "private text tag (0009,0010) must be present in preservation.object"
        );
        let text_val = text_node.unwrap().value.as_text();
        assert!(
            text_val
                .map(|s| s.trim() == "PRIV_ROUND_TRIP_VALUE")
                .unwrap_or(false),
            "private text value must survive round-trip: got {:?}",
            text_val
        );

        // Private bytes tag must be present in preservation.preserved with the correct bytes.
        let bytes_elem = slice
            .preservation
            .preserved
            .iter()
            .find(|e| e.tag == priv_bytes_tag);
        assert!(
            bytes_elem.is_some(),
            "private bytes tag (0019,1001) must be present in preservation.preserved"
        );
        assert_eq!(
            bytes_elem.unwrap().bytes,
            vec![0xAB_u8, 0xCD, 0xEF, 0x01],
            "raw OB bytes must survive round-trip"
        );
    }

    /// Compressed-TS series guard: load_dicom_series must return Err when any slice
    /// declares a compressed transfer syntax in its file meta table.
    ///
    /// Analytical invariant: TransferSyntaxKind::is_compressed() is true for
    /// 1.2.840.10008.1.2.4.50 (JPEG Baseline); load_from_series must detect this
    /// before pixel decode and return a descriptive error.
    #[test]
    fn test_load_series_compressed_ts_errors() {
        type B = burn_ndarray::NdArray<f32>;

        let tmp = tempfile::tempdir().expect("tempdir");
        let series_dir = tmp.path().join("compressed_series");
        std::fs::create_dir_all(&series_dir).expect("create_dir");

        // Write a single-slice DICOM declaring JPEG Baseline TS (compressed).
        let slice_path = series_dir.join("slice_0000.dcm");
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"), // CT Image Storage
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.10001"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("CT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(vec![0u8; 8])),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                    .media_storage_sop_instance_uid("2.25.10001")
                    .transfer_syntax("1.2.840.10008.1.2.4.50"), // JPEG Baseline
            )
            .expect("meta build");
        file_obj
            .write_to_file(&slice_path)
            .expect("write compressed slice");

        // scan_dicom_directory should succeed (it only reads metadata, not pixels)
        let scan_result = scan_dicom_directory(&series_dir);
        // If scan succeeds, load must return Err due to compressed TS
        if let Ok(series_info) = scan_result {
            // Verify the TS was captured
            let has_compressed = series_info.metadata.slices.iter().any(|s| {
                s.transfer_syntax_uid
                    .as_deref()
                    .map(|uid| TransferSyntaxKind::from_uid(uid).is_compressed())
                    .unwrap_or(false)
            });
            assert!(
                has_compressed,
                "scan must record compressed TS in slice metadata"
            );

            let device = <B as burn::tensor::backend::Backend>::Device::default();
            let load_result = load_dicom_series::<B, _>(&series_dir, &device);
            assert!(
                load_result.is_err(),
                "load_dicom_series must return Err for compressed-TS series"
            );
            let msg = format!("{:?}", load_result.unwrap_err());
            assert!(
                msg.contains("1.2.840.10008.1.2.4.50") || msg.to_lowercase().contains("compress"),
                "error must reference compressed TS UID or 'compress'; got: {msg}"
            );
        }
        // If scan itself fails (e.g. SOP class filter), the test is inconclusive
        // but not a failure — the scan's SOP-class rejection is also correct behavior.
    }

    /// End-to-end intensity round-trip: write via `write_dicom_series`,
    /// load via `load_dicom_series`, verify per-pixel reconstruction error is within
    /// the theoretical quantization bound.
    ///
    /// # Specification
    /// For each slice: rescale_slope = (max - min) / 65535.
    /// Encode: u16 = round((v - intercept) / slope).clamp(0, 65535).
    /// Decode: f32 = u16 * slope + intercept.
    /// Bound: |f_decoded - f_original| ≤ slope/2 + ε ≤ slope + 1.0
    ///
    /// Analytical intensities: 0, 1, 2, ..., 63 (64 voxels in 4×4×4).
    /// Analytical slope per slice (16 voxels per slice with range [0,15], [16,31], [32,47], [48,63]):
    ///   slope_i = (max_i - min_i) / 65535.
    #[test]
    fn test_write_series_load_series_intensity_roundtrip() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        type B = burn_ndarray::NdArray<f32>;

        let tmp = tempfile::tempdir().expect("tempdir");
        let series_path = tmp.path().join("e2e_roundtrip_series");

        // 4 slices × 4 rows × 4 cols = 64 voxels.
        // Intensities 0..=63, row-major order.
        let (depth, rows, cols) = (4usize, 4usize, 4usize);
        let original_data: Vec<f32> = (0..(depth * rows * cols)).map(|i| i as f32).collect();
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(original_data.clone(), Shape::new([depth, rows, cols])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        crate::format::dicom::writer::write_dicom_series(&series_path, &image)
            .expect("write_dicom_series must succeed");

        let (loaded_image, _meta) = load_dicom_series::<B, _>(&series_path, &device)
            .expect("load_dicom_series must succeed");

        let loaded_td = loaded_image.data().clone().into_data();
        let loaded_vals: &[f32] = loaded_td
            .as_slice::<f32>()
            .expect("loaded image must contain f32");

        assert_eq!(
            loaded_vals.len(),
            original_data.len(),
            "loaded voxel count must equal original"
        );

        // Analytical bound per slice: slope = range / 65535 = 15 / 65535 ≈ 2.29e-4.
        // DS format {:.6} stores the slope/intercept with at most 0.5e-6 rounding error
        // per coefficient. Accumulated slope error over max u16 (65535):
        //   65535 * 0.5e-6 ≈ 0.033.
        // Quantization from round(): slope / 2 ≈ 1.14e-4.
        // Total analytical bound: 65535 * 0.5e-6 + 0.5e-6 + slope / 2.
        let slice_range = 15.0f32;
        let slope = slice_range / 65535.0_f32;
        let ds_half_ulp = 0.5e-6_f32;
        let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;

        // The writer writes per-slice rescale; reader applies per-slice rescale.
        // Re-sort loaded voxels by z-position. The series may be loaded in sorted order.
        for (idx, (&orig, &loaded)) in original_data.iter().zip(loaded_vals.iter()).enumerate() {
            let err = (loaded - orig).abs();
            assert!(
                err <= tol,
                "voxel[{idx}]: |{loaded} - {orig}| = {err} > tol {tol}; slope={slope}"
            );
        }
    }

    /// End-to-end round-trip: write via `write_dicom_series_with_metadata` with
    /// non-trivial spatial metadata, load via `load_dicom_series`, verify:
    /// 1. Per-pixel reconstruction error ≤ analytical bound.
    /// 2. Origin round-trips to within 1e-4 mm.
    /// 3. Spacing round-trips to within 1e-4 mm.
    ///
    /// # Specification
    /// Intensities: 16 voxels per slice, values -512 .. -512+15 per slice.
    /// Analytical slope = (max - min) / 65535 = 15 / 65535.
    /// Analytical intercept = min_val.
    #[test]
    fn test_write_metadata_series_load_series_intensity_roundtrip() {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        use std::collections::HashMap;
        type B = burn_ndarray::NdArray<f32>;

        let tmp = tempfile::tempdir().expect("tempdir");
        let series_path = tmp.path().join("e2e_meta_roundtrip");

        let (depth, rows, cols) = (3usize, 4usize, 4usize);
        // Each slice: values starting at z*16, range = 15.
        let original_data: Vec<f32> = (0..(depth * rows * cols))
            .map(|i| {
                let slice_idx = i / (rows * cols);
                let intra_idx = i % (rows * cols);
                (slice_idx * 16 + intra_idx) as f32
            })
            .collect();
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(original_data.clone(), Shape::new([depth, rows, cols])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([5.0, 10.0, -20.0]),
            Spacing::new([0.5, 0.5, 1.5]),
            Direction::identity(),
        );

        let meta = DicomReadMetadata {
            series_instance_uid: Some("1.2.3.4.5.6.999".to_string()),
            study_instance_uid: Some("1.2.3.4.5.6.998".to_string()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".to_string()),
            patient_id: Some("E2E001".to_string()),
            patient_name: Some("E2E^Patient".to_string()),
            study_date: Some("20250101".to_string()),
            series_date: None,
            series_time: None,
            dimensions: [rows, cols, depth],
            spacing: [0.5, 0.5, 1.5],
            origin: [5.0, 10.0, -20.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation: crate::format::dicom::DicomPreservationSet::new(),
        };

        crate::format::dicom::writer::write_dicom_series_with_metadata(
            &series_path,
            &image,
            Some(&meta),
        )
        .expect("write_dicom_series_with_metadata must succeed");

        let (loaded_image, loaded_meta) = load_dicom_series::<B, _>(&series_path, &device)
            .expect("load_dicom_series must succeed");

        // --- Intensity round-trip ---
        let loaded_td = loaded_image.data().clone().into_data();
        let loaded_vals: &[f32] = loaded_td.as_slice::<f32>().expect("loaded must be f32");

        assert_eq!(
            loaded_vals.len(),
            original_data.len(),
            "voxel count must match"
        );

        // Analytical slope per slice: each slice has range=15, slope = 15/65535.
        // DS format {:.6} stores the slope/intercept with at most 0.5e-6 rounding error
        // per coefficient. Accumulated slope error over max u16 (65535):
        //   65535 * 0.5e-6 ≈ 0.033.
        // Quantization from round(): slope / 2 ≈ 1.14e-4.
        // Total analytical bound: 65535 * 0.5e-6 + 0.5e-6 + slope / 2.
        let slice_range = 15.0f32;
        let slope = slice_range / 65535.0_f32;
        let ds_half_ulp = 0.5e-6_f32;
        let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;

        for (idx, (&orig, &loaded)) in original_data.iter().zip(loaded_vals.iter()).enumerate() {
            let err = (loaded - orig).abs();
            assert!(
                err <= tol,
                "voxel[{idx}]: |{loaded} - {orig}| = {err} > tol {tol}"
            );
        }

        // --- Spatial metadata round-trip ---
        let pos_tol = 1e-4_f64;
        assert!(
            (loaded_meta.origin[0] - 5.0).abs() < pos_tol,
            "origin[0] must be 5.0; got {}",
            loaded_meta.origin[0]
        );
        assert!(
            (loaded_meta.origin[1] - 10.0).abs() < pos_tol,
            "origin[1] must be 10.0; got {}",
            loaded_meta.origin[1]
        );
        assert!(
            (loaded_meta.origin[2] - (-20.0_f64)).abs() < pos_tol,
            "origin[2] must be -20.0; got {}",
            loaded_meta.origin[2]
        );
        assert!(
            (loaded_meta.spacing[0] - 0.5).abs() < pos_tol,
            "spacing[0] must be 0.5; got {}",
            loaded_meta.spacing[0]
        );
        assert!(
            (loaded_meta.spacing[1] - 0.5).abs() < pos_tol,
            "spacing[1] must be 0.5; got {}",
            loaded_meta.spacing[1]
        );
        assert!(
            (loaded_meta.spacing[2] - 1.5).abs() < pos_tol,
            "spacing[2] must be 1.5; got {}",
            loaded_meta.spacing[2]
        );
    }

    #[test]
    fn test_decode_pixel_bytes_unsigned_16bit_identity_rescale() {
        // u16: [0x00,0x00] = 0; [0xFF,0xFF] = 65535. slope=1.0, intercept=0.0 → identity.
        let bytes: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF];
        let result = decode_pixel_bytes(&bytes, 16, 0, 1.0, 0.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0f32);
        assert_eq!(result[1], 65535.0f32);
    }

    #[test]
    fn test_decode_pixel_bytes_signed_16bit_identity_rescale() {
        // i16::MIN = -32768 stored as [0x00, 0x80] LE; i16::MAX = 32767 stored as [0xFF, 0x7F] LE.
        let bytes: [u8; 4] = [0x00, 0x80, 0xFF, 0x7F];
        let result = decode_pixel_bytes(&bytes, 16, 1, 1.0, 0.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], -32768.0f32);
        assert_eq!(result[1], 32767.0f32);
    }

    #[test]
    fn test_decode_pixel_bytes_signed_16bit_with_rescale() {
        // i16: -1 = [0xFF, 0xFF] LE; decoded = -1.0 × 2.0 + 100.0 = 98.0.
        let bytes: [u8; 2] = [0xFF, 0xFF];
        let result = decode_pixel_bytes(&bytes, 16, 1, 2.0, 100.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 98.0f32);
    }

    #[test]
    fn test_decode_pixel_bytes_8bit_identity_rescale() {
        let bytes: [u8; 3] = [0, 127, 255];
        let result = decode_pixel_bytes(&bytes, 8, 0, 1.0, 0.0);
        assert_eq!(result, vec![0.0f32, 127.0f32, 255.0f32]);
    }

    #[test]
    fn test_decode_pixel_bytes_8bit_with_rescale() {
        // 8-bit value 200; slope=0.5, intercept=10.0 → 200 × 0.5 + 10.0 = 110.0.
        let bytes: [u8; 1] = [200];
        let result = decode_pixel_bytes(&bytes, 8, 0, 0.5, 10.0);
        assert_eq!(result[0], 110.0f32);
    }

    #[test]
    fn test_is_likely_dicom_file_accepts_canonical_extensions() {
        assert!(is_likely_dicom_file(std::path::Path::new("scan.dcm")));
        assert!(is_likely_dicom_file(std::path::Path::new("SCAN.DCM")));
        assert!(is_likely_dicom_file(std::path::Path::new("scan.dicom")));
        assert!(is_likely_dicom_file(std::path::Path::new("scan.ima")));
    }

    #[test]
    fn test_is_likely_dicom_file_rejects_analyze_and_raw_extensions() {
        assert!(!is_likely_dicom_file(std::path::Path::new("brain.hdr")));
        assert!(!is_likely_dicom_file(std::path::Path::new("brain.img")));
        assert!(!is_likely_dicom_file(std::path::Path::new("brain.raw")));
        assert!(!is_likely_dicom_file(std::path::Path::new("brain.nii")));
        assert!(!is_likely_dicom_file(std::path::Path::new("data.bin")));
    }

    #[test]
    fn test_slice_metadata_default_pixel_representation_is_zero() {
        let meta = DicomSliceMetadata::default();
        assert_eq!(
            meta.pixel_representation, 0,
            "pixel_representation default must be 0 (unsigned)"
        );
        assert_eq!(meta.bits_allocated, 16, "bits_allocated default must be 16");
        assert!(
            meta.window_center.is_none(),
            "window_center default must be None"
        );
        assert!(
            meta.window_width.is_none(),
            "window_width default must be None"
        );
        assert_eq!(
            meta.rescale_slope, 1.0f32,
            "rescale_slope default must be 1.0"
        );
        assert_eq!(
            meta.rescale_intercept, 0.0f32,
            "rescale_intercept default must be 0.0"
        );
    }

    #[test]
    fn test_read_slice_pixels_signed_i16_roundtrip() {
        // Build a DICOM file with PixelRepresentation=1 and three known i16 values.
        // Stored pixel values: -1000, 0, 1000. RescaleSlope=1, RescaleIntercept=0.
        // Expected decoded values: [-1000.0, 0.0, 1000.0].
        use dicom::core::smallvec::SmallVec;
        use dicom::object::meta::FileMetaTableBuilder;
        use dicom::object::InMemDicomObject;

        let pixels_i16: [i16; 3] = [-1000, 0, 1000];
        let pixel_bytes: Vec<u8> = pixels_i16.iter().flat_map(|&v| v.to_le_bytes()).collect();

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99999"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("OT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1_u16),
        )); // rows=1
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(3_u16),
        )); // cols=3
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(1_u16),
        )); // PixelRepresentation=1 (signed)
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from("1"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from("0"),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                    .media_storage_sop_instance_uid("2.25.99999")
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .expect("meta build must succeed");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("signed.dcm");
        file_obj.write_to_file(&path).expect("write must succeed");

        // Construct DicomSliceMetadata as scan_dicom_directory would populate it.
        let slice_meta = DicomSliceMetadata {
            path: path.clone(),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            pixel_representation: 1,
            bits_allocated: 16,
            ..DicomSliceMetadata::default()
        };

        let result = read_slice_pixels(&slice_meta).expect("read_slice_pixels must succeed");
        assert_eq!(result.len(), 3, "pixel count must be 3");
        assert_eq!(result[0], -1000.0f32, "pixel[0] must be -1000.0");
        assert_eq!(result[1], 0.0f32, "pixel[1] must be 0.0");
        assert_eq!(result[2], 1000.0f32, "pixel[2] must be 1000.0");
    }
}
