//! RT Structure Set reader — parse a DICOM RT Structure Set file into [`RtStructureSet`].

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use dicom::core::value::Value;
use dicom::core::Tag;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::collections::HashMap;
use std::path::Path;

use super::types::{RtContour, RtRoiInfo, RtStructureSet, RT_STRUCT_SOP_CLASS_UID};
use super::utils::{parse_color, parse_contour_data};
use crate::format::dicom::reader::types::truncate_arraystring;

/// Read and parse a DICOM RT Structure Set file.
///
/// # Errors
/// - Returns `Err` when the path does not exist or is unreadable.
/// - Returns `Err` when the SOP Class UID ≠ `1.2.840.10008.5.1.4.1.1.481.3`.
///
/// # Invariants
/// 1. ROIs are sorted ascending by `roi_number`.
/// 2. All contour data is fully parsed; partial triples are discarded.
/// 3. `structure_set_name` is `None` when the tag is absent or empty.
pub fn read_rt_struct<P: AsRef<Path>>(path: P) -> Result<RtStructureSet> {
    let path = path.as_ref();
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("open DICOM file: {}", path.display()))?;

    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != RT_STRUCT_SOP_CLASS_UID {
        bail!(
            "SOP Class UID '{}' is not RT Structure Set Storage ({})",
            sop,
            RT_STRUCT_SOP_CLASS_UID
        );
    }

    let structure_set_label = obj
        .element(Tag(0x3006, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let structure_set_name = obj
        .element(Tag(0x3006, 0x0004))
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // Step 1: build ROI map from StructureSetROISequence (3006,0020).
    let mut roi_map: HashMap<u32, RtRoiInfo> = HashMap::new();
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0020)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let roi_number: u32 = item
                    .element(Tag(0x3006, 0x0022))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let roi_name = item
                    .element(Tag(0x3006, 0x0026))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();

                let roi_description = item
                    .element(Tag(0x3006, 0x0028))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                roi_map.insert(
                    roi_number,
                    RtRoiInfo {
                        roi_number,
                        roi_name,
                        roi_description,
                        roi_interpreted_type: None,
                        display_color: None,
                        contours: Vec::new(),
                    },
                );
            }
        }
    }

    // Step 2: update roi_interpreted_type from RTROIObservationsSequence (3006,0080).
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0080)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let ref_roi: u32 = item
                    .element(Tag(0x3006, 0x0084))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let interpreted_type = item
                    .element(Tag(0x3006, 0x00A4))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                if let Some(roi) = roi_map.get_mut(&ref_roi) {
                    roi.roi_interpreted_type = interpreted_type;
                }
            }
        }
    }

    // Step 3: update display_color + contours from ROIContourSequence (3006,0039).
    if let Ok(seq_elem) = obj.element(Tag(0x3006, 0x0039)) {
        if let Value::Sequence(seq) = seq_elem.value() {
            for item in seq.items() {
                let ref_roi: u32 = item
                    .element(Tag(0x3006, 0x0084))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let display_color = item
                    .element(Tag(0x3006, 0x002A))
                    .ok()
                    .and_then(|e| e.to_str().ok())
                    .and_then(|s| parse_color(s.trim()));

                let mut contours: Vec<RtContour> = Vec::new();
                if let Ok(cs_elem) = item.element(Tag(0x3006, 0x0040)) {
                    if let Value::Sequence(cs) = cs_elem.value() {
                        for ci in cs.items() {
                            let geometric_type = ci
                                .element(Tag(0x3006, 0x0042))
                                .ok()
                                .and_then(|e| e.to_str().ok())
                                .map(|s| {
                                    let trimmed = s.trim();
                                    match ArrayString::<16>::from(trimmed) {
                                        Ok(v) => v,
                                        Err(_) => {
                                            tracing::warn!("ContourGeometricType exceeds 16 chars, truncating: {}", &trimmed[..16]);
                                                                truncate_arraystring::<16>(trimmed)
                                        }
                                    }
                                })
                                .unwrap_or_else(ArrayString::new);

                            let points = ci
                                .element(Tag(0x3006, 0x0050))
                                .ok()
                                .and_then(|e| e.to_str().ok())
                                .map(|s| parse_contour_data(s.trim()))
                                .unwrap_or_default();

                            contours.push(RtContour {
                                geometric_type,
                                points,
                            });
                        }
                    }
                }

                if let Some(roi) = roi_map.get_mut(&ref_roi) {
                    if display_color.is_some() {
                        roi.display_color = display_color;
                    }
                    roi.contours = contours;
                }
            }
        }
    }

    let mut rois: Vec<RtRoiInfo> = roi_map.into_values().collect();
    rois.sort_by_key(|r| r.roi_number);

    Ok(RtStructureSet {
        structure_set_label,
        structure_set_name,
        rois,
    })
}
