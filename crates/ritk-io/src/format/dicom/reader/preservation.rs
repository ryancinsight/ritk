//! Tag preservation helpers for the DICOM series reader.
//!
//! These utilities identify and preserve DICOM elements that are not
//! explicitly extracted by the named parsing logic.

use std::collections::HashSet;

use dicom::core::VR;
use dicom_core::header::Header;

use crate::format::dicom::object_model::{
    is_private_tag, DicomObjectNode, DicomSequenceItem, DicomTag, DicomValue,
};

/// Compute a compact key from a DICOM tag group+element pair.
#[inline]
pub(super) fn tag_key(group: u16, element: u16) -> u32 {
    ((group as u32) << 16) | (element as u32)
}

/// Return the set of DICOM tags already extracted by the named parsing logic.
///
/// Elements whose `tag_key` is in this set are skipped during full-preservation
/// iteration to avoid double-capturing named fields.
pub(super) fn known_handled_tags() -> HashSet<u32> {
    let mut s = HashSet::new();
    // Per-slice
    s.insert(tag_key(0x0008, 0x0018)); // SOP Instance UID
    s.insert(tag_key(0x0020, 0x0013)); // Instance Number
    s.insert(tag_key(0x0020, 0x1041)); // Slice Location
    s.insert(tag_key(0x0020, 0x0032)); // ImagePositionPatient
    s.insert(tag_key(0x0020, 0x0037)); // ImageOrientationPatient
    s.insert(tag_key(0x0028, 0x0030)); // PixelSpacing
    s.insert(tag_key(0x0018, 0x0050)); // SliceThickness
    s.insert(tag_key(0x0018, 0x5100)); // PatientPosition
    s.insert(tag_key(0x0028, 0x1053)); // RescaleSlope
    s.insert(tag_key(0x0028, 0x1052)); // RescaleIntercept
    s.insert(tag_key(0x0008, 0x0016)); // SOP Class UID
    s.insert(tag_key(0x0008, 0x0070)); // Manufacturer
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
    // PET radiopharmaceutical tags
    s.insert(tag_key(0x0010, 0x1030)); // PatientWeight
    s.insert(tag_key(0x0054, 0x1102)); // DecayCorrection
    s.insert(tag_key(0x0054, 0x0016)); // RadiopharmaceuticalInformationSequence
    s
}

/// Recursively parse a DICOM sequence item into a [`DicomSequenceItem`].
///
/// `depth` limits recursion to 8 levels to guard against malformed input.
pub(super) fn parse_sequence_item(
    item: &dicom::object::InMemDicomObject,
    depth: usize,
) -> DicomSequenceItem {
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
