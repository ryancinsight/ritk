//! Per-file and per-bytes DICOM metadata parsing for the series scan path.
//!
//! `parse_dicom_file` extracts metadata from a DICOM Part-10 file on disk.
//! `parse_dicom_bytes` extracts the same metadata from in-memory Part-10 bytes
//! constructed from SCP-received instances.
//! `parse_dicom_file_bytes` extracts metadata from already-formed Part-10 byte
//! payloads (e.g. from drag-and-drop DICOM files).
//! All three delegate to `extract_dicom_metadata` for the shared tag-extraction
//! logic.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dicom::core::{Tag, VR};
use dicom::object::DefaultDicomObject;
use dicom_core::header::Header;
use ritk_dicom::{parse_bytes_with, parse_file_with, DicomRsBackend};

use super::preservation::{known_handled_tags, parse_sequence_item, tag_key};
use super::types::{
    cs_to_arraystring, da_to_arraystring, parse_patient_position, tm_to_arraystring,
    uid_to_arraystring, DicomSliceMetadata, SeriesFirstSeen,
};
use crate::format::dicom::object_model::{
    is_private_tag, DicomElementClass, DicomObjectNode, DicomPreservationSet,
    DicomPreservedElement, DicomTag, DicomValue,
};
use arrayvec::ArrayString;

/// Parse a single DICOM Part-10 file, populating `first` with series-level
/// first-seen fields and returning the per-slice metadata, per-file image
/// dimensions, and per-file SeriesInstanceUID.
///
/// Returns `None` when the file cannot be opened by the DICOM backend.
pub(crate) fn parse_dicom_file(
    file_path: &Path,
    first: &mut SeriesFirstSeen,
) -> Option<(DicomSliceMetadata, (u32, u32), Option<String>)> {
    let obj = parse_file_with::<DicomRsBackend, _>(file_path).ok()?;
    extract_dicom_metadata(&obj, file_path.to_path_buf(), first)
}

/// Parse in-memory DICOM Part-10 bytes, populating `first` with series-level
/// first-seen fields and returning the per-slice metadata, image dimensions,
/// and SeriesInstanceUID.
///
/// `sop_instance_uid` is used to construct a synthetic `scp://` path for
/// diagnostics and to populate the `sop_instance_uid` field.
///
/// Returns `None` when the bytes cannot be parsed by the DICOM backend.
pub(crate) fn parse_dicom_bytes(
    bytes: &[u8],
    sop_instance_uid: &str,
    first: &mut SeriesFirstSeen,
) -> Option<(DicomSliceMetadata, (u32, u32), Option<String>)> {
    let obj = parse_bytes_with::<DicomRsBackend>(bytes).ok()?;
    let path_for_meta = PathBuf::from(format!("scp://{}", sop_instance_uid));
    let mut result = extract_dicom_metadata(&obj, path_for_meta, first);
    if let Some((ref mut slice_meta, _, _)) = result {
        slice_meta.sop_instance_uid = uid_to_arraystring(sop_instance_uid);
    }
    result
}

/// Parse already-formed DICOM Part-10 byte payload (e.g. from a dropped file),
/// populating `first` with series-level first-seen fields and returning the
/// per-slice metadata, image dimensions, and SeriesInstanceUID.
///
/// Unlike [`parse_dicom_bytes`], this function does not wrap the bytes in an
/// SCP Part-10 header — the input must already be a valid DICOM Part 10 file
/// (128-byte preamble + DICM magic + File Meta Information + dataset).
///
/// `synthetic_path` is used for diagnostics; the SOP Instance UID is extracted
/// from the DICOM metadata itself.
///
/// Returns `None` when the bytes cannot be parsed by the DICOM backend.
pub(crate) fn parse_dicom_file_bytes(
    bytes: &[u8],
    synthetic_path: &Path,
    first: &mut SeriesFirstSeen,
) -> Option<(DicomSliceMetadata, (u32, u32), Option<String>)> {
    let obj = parse_bytes_with::<DicomRsBackend>(bytes).ok()?;
    extract_dicom_metadata(&obj, synthetic_path.to_path_buf(), first)
}

/// Shared tag-extraction logic for both file-based and in-memory DICOM parsing.
///
/// Populates `first` with series-level first-seen fields and returns the
/// per-slice metadata, per-file image dimensions, and per-file
/// SeriesInstanceUID.
fn extract_dicom_metadata(
    obj: &DefaultDicomObject,
    path_for_meta: PathBuf,
    first: &mut SeriesFirstSeen,
) -> Option<(DicomSliceMetadata, (u32, u32), Option<String>)> {
    let mut slice_meta = DicomSliceMetadata {
        path: path_for_meta,
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
        part10_bytes: None,
    };

    // --- Per-slice fields ---
    if slice_meta.sop_instance_uid.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x0018)) {
            slice_meta.sop_instance_uid = elem
                .to_str()
                .ok()
                .as_ref()
                .and_then(|s| uid_to_arraystring(s));
        }
    }
    let file_series_uid = obj
        .element(Tag(0x0020, 0x000E))
        .ok()
        .and_then(|e| e.to_str().ok().map(String::from));
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
        slice_meta.sop_class_uid = elem
            .to_str()
            .ok()
            .as_ref()
            .and_then(|s| uid_to_arraystring(s));
    }
    // Transfer syntax from file meta (0002,0010), not main dataset.
    slice_meta.transfer_syntax_uid = uid_to_arraystring(obj.meta().transfer_syntax());

    if let Ok(elem) = obj.element(Tag(0x0028, 0x0103)) {
        slice_meta.pixel_representation = elem
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
    }
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
    if let Ok(elem) = obj.element(Tag(0x0018, 0x1120)) {
        slice_meta.gantry_tilt = elem.to_str().ok().and_then(|s| s.trim().parse().ok());
    }
    if let Ok(elem) = obj.element(Tag(0x0018, 0x5100)) {
        slice_meta.patient_position = elem
            .to_str()
            .ok()
            .as_deref()
            .and_then(parse_patient_position);
    }

    // Per-file dimension tracking (for canonical-dimension plurality selection).
    let this_rows: Option<u32> = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse().ok());
    let this_cols: Option<u32> = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.parse().ok());
    let file_dim = (this_rows.unwrap_or(0), this_cols.unwrap_or(0));

    // --- Series-level first-seen accumulation ---
    if first.rows.is_none() {
        first.rows = this_rows;
    }
    if first.cols.is_none() {
        first.cols = this_cols;
    }
    if first.pixel_spacing.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0028, 0x0030)) {
            if let Ok(s) = elem.to_str() {
                let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                if parts.len() >= 2 {
                    first.pixel_spacing = Some([parts[0], parts[1]]);
                }
            }
        }
    }
    if first.slice_thickness.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
            first.slice_thickness = elem.to_str().ok().and_then(|s| s.parse().ok());
        }
    }
    if first.series_instance_uid.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0020, 0x000E)) {
            first.series_instance_uid = elem
                .to_str()
                .ok()
                .as_ref()
                .and_then(|s| uid_to_arraystring(s));
        }
    }
    if first.study_instance_uid.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0020, 0x000D)) {
            first.study_instance_uid = elem
                .to_str()
                .ok()
                .as_ref()
                .and_then(|s| uid_to_arraystring(s));
        }
    }
    if first.series_description.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x103E)) {
            first.series_description = elem.to_str().ok().map(String::from);
        }
    }
    if first.modality.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x0060)) {
            first.modality = elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
        }
    }
    if first.patient_id.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0010, 0x0020)) {
            first.patient_id = elem.to_str().ok().map(String::from);
        }
    }
    if first.patient_name.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0010, 0x0010)) {
            first.patient_name = elem.to_str().ok().map(String::from);
        }
    }
    if first.study_date.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x0020)) {
            first.study_date = elem.to_str().ok().map(|s| da_to_arraystring(s.trim()));
        }
    }
    if first.series_date.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x0021)) {
            first.series_date = elem.to_str().ok().map(|s| da_to_arraystring(s.trim()));
        }
    }
    if first.series_time.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0008, 0x0031)) {
            first.series_time = elem.to_str().ok().map(|s| tm_to_arraystring(s.trim()));
        }
    }
    if first.frame_of_reference_uid.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0020, 0x0052)) {
            first.frame_of_reference_uid = elem
                .to_str()
                .ok()
                .as_ref()
                .and_then(|s| uid_to_arraystring(s));
        }
    }
    if first.bits_allocated.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0028, 0x0100)) {
            first.bits_allocated = elem.to_str().ok().and_then(|s| s.parse().ok());
        }
    }
    if first.bits_stored.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0028, 0x0101)) {
            first.bits_stored = elem.to_str().ok().and_then(|s| s.parse().ok());
        }
    }
    if first.high_bit.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0028, 0x0102)) {
            first.high_bit = elem.to_str().ok().and_then(|s| s.parse().ok());
        }
    }
    if first.photometric_interpretation.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0028, 0x0004)) {
            first.photometric_interpretation =
                elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
        }
    }
    if first.transfer_syntax_uid.is_none() {
        first.transfer_syntax_uid = uid_to_arraystring(obj.meta().transfer_syntax());
    }
    if first.patient_weight_kg.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0010, 0x1030)) {
            first.patient_weight_kg = elem.to_str().ok().and_then(|s| s.trim().parse().ok());
        }
    }
    if first.decay_correction.is_none() {
        if let Ok(elem) = obj.element(Tag(0x0054, 0x1102)) {
            first.decay_correction = elem.to_str().ok().map(|s| cs_to_arraystring(s.trim()));
        }
    }
    // RadiopharmaceuticalInformationSequence (0054,0016) → first item sub-fields.
    if first.radionuclide_total_dose_bq.is_none()
        || first.radionuclide_half_life_s.is_none()
        || first.radiopharmaceutical_start_time.is_none()
    {
        if let Ok(seq_elem) = obj.element(Tag(0x0054, 0x0016)) {
            if let Some(items) = seq_elem.value().items() {
                if let Some(first_item) = items.first() {
                    if first.radionuclide_total_dose_bq.is_none() {
                        if let Ok(e) = first_item.element(Tag(0x0018, 0x1074)) {
                            first.radionuclide_total_dose_bq =
                                e.to_str().ok().and_then(|s| s.trim().parse().ok());
                        }
                    }
                    if first.radionuclide_half_life_s.is_none() {
                        if let Ok(e) = first_item.element(Tag(0x0018, 0x1075)) {
                            first.radionuclide_half_life_s =
                                e.to_str().ok().and_then(|s| s.trim().parse().ok());
                        }
                    }
                    if first.radiopharmaceutical_start_time.is_none() {
                        if let Ok(e) = first_item.element(Tag(0x0018, 0x1072)) {
                            first.radiopharmaceutical_start_time =
                                e.to_str().ok().map(|s| tm_to_arraystring(s.trim()));
                        }
                    }
                }
            }
        }
    }

    // --- Full element preservation ---
    // Capture all non-handled elements into the slice preservation model.
    {
        let handled = known_handled_tags();
        for element in obj {
            let tag = element.tag();
            let key = tag_key(tag.group(), tag.element());
            if handled.contains(&key) {
                continue;
            }
            let dicom_tag = DicomTag::new(tag.group(), tag.element());
            let vr_str = element.vr().to_string();
            let element_class = if is_private_tag(dicom_tag) {
                DicomElementClass::Private
            } else {
                DicomElementClass::Standard
            };
            if element.vr() == VR::SQ {
                if let Some(sub_items) = element.value().items() {
                    let parsed: Vec<_> = sub_items
                        .iter()
                        .map(|i| parse_sequence_item(i, 0))
                        .collect();
                    slice_meta.preservation.object.insert(DicomObjectNode {
                        tag: dicom_tag,
                        vr: Some(ArrayString::<2>::try_from("SQ").unwrap_or_default()),
                        value: DicomValue::Sequence(parsed),
                        element_class,
                        source: None,
                    });
                }
            } else {
                // Binary VRs bypass to_str(): dicom-rs 0.8 decimal-formats them
                // silently instead of erroring, which corrupts raw payloads.
                let is_binary_vr = matches!(
                    element.vr(),
                    VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN
                );
                if is_binary_vr {
                    if let Ok(bytes) = element.to_bytes() {
                        slice_meta.preservation.preserve(DicomPreservedElement::new(
                            dicom_tag,
                            Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                            bytes.to_vec(),
                        ));
                    }
                } else if let Ok(s) = element.to_str() {
                    slice_meta.preservation.object.insert(DicomObjectNode {
                        tag: dicom_tag,
                        vr: Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                        value: DicomValue::Text(s.to_string()),
                        element_class,
                        source: None,
                    });
                } else if let Ok(bytes) = element.to_bytes() {
                    slice_meta.preservation.preserve(DicomPreservedElement::new(
                        dicom_tag,
                        Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                        bytes.to_vec(),
                    ));
                }
            }
        }
    }

    Some((slice_meta, file_dim, file_series_uid))
}
