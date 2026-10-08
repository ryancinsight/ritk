//! Tag preservation helpers for the DICOM series reader.
//!
//! These utilities identify and preserve DICOM elements that are not
//! explicitly extracted by the named parsing logic.

use std::collections::HashSet;
use std::sync::LazyLock;

use dicom::core::VR;
use dicom::object::DefaultDicomObject;
use dicom_core::header::Header;

use crate::format::dicom::object_model::{
    is_private_tag, DicomElementClass, DicomObjectNode, DicomPreservationSet,
    DicomPreservedElement, DicomRetentionLoss, DicomRetentionReason, DicomSequenceItem, DicomTag,
    DicomValue,
};
use arrayvec::ArrayString;

/// Compute a compact key from a DICOM tag group+element pair.
#[inline]
pub(super) fn tag_key(group: u16, element: u16) -> u32 {
    ((group as u32) << 16) | (element as u32)
}

/// Return the set of DICOM tags already extracted by the named parsing logic.
///
/// Elements whose `tag_key` is in this set are skipped during full-preservation
/// iteration to avoid double-capturing named fields.
///
/// # Optimization
///
/// Built once via `LazyLock` and reused across all DICOM file parses, eliminating
/// 30+ `HashSet::insert` calls per file.  For a 1000-file series this avoids
/// 30 000+ per-element insertions.
pub(super) fn known_handled_tags() -> &'static HashSet<u32> {
    static KNOWN: LazyLock<HashSet<u32>> = LazyLock::new(|| {
        let mut s = HashSet::with_capacity(32);
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
    });
    &KNOWN
}

/// Maximum sequence nesting depth that is walked and retained.
///
/// A deeper subtree is not silently dropped: the sequence element whose child
/// would exceed this limit is recorded as a
/// [`DicomRetentionReason::NestingDepthExceeded`] loss so a conversion
/// preflight can reject the source instead of importing a truncated tree.
///
/// This bounds retention, not anonymization traversal
/// (`anonymize::MAX_SEQUENCE_DEPTH`), so the two are deliberately separate.
pub(super) const MAX_RETAINED_SEQUENCE_DEPTH: usize = 8;

/// Recursively parse a DICOM sequence item into a [`DicomSequenceItem`].
///
/// `depth` limits recursion to [`MAX_RETAINED_SEQUENCE_DEPTH`] levels to guard
/// against malformed input. Every element that cannot be interpreted or
/// retained is appended to `losses`; nothing is discarded without a record.
pub(super) fn parse_sequence_item(
    item: &dicom::object::InMemDicomObject,
    depth: usize,
    losses: &mut Vec<DicomRetentionLoss>,
) -> DicomSequenceItem {
    let mut seq_item = DicomSequenceItem::new();
    for element in item.iter() {
        let tag = element.tag();
        let dicom_tag = DicomTag::new(tag.group(), tag.element());
        let vr_str = element.vr().to_string();
        let element_class = if is_private_tag(dicom_tag) {
            DicomElementClass::Private
        } else {
            DicomElementClass::Standard
        };
        if element.vr() == VR::SQ {
            if depth + 1 > MAX_RETAINED_SEQUENCE_DEPTH {
                losses.push(DicomRetentionLoss::new(
                    dicom_tag,
                    DicomRetentionReason::NestingDepthExceeded,
                ));
                continue;
            }
            match element.value().items() {
                Some(sub_items) => {
                    let parsed: Vec<_> = sub_items
                        .iter()
                        .map(|i| parse_sequence_item(i, depth + 1, losses))
                        .collect();
                    seq_item.insert(DicomObjectNode {
                        tag: dicom_tag,
                        vr: Some(ArrayString::<2>::try_from("SQ").unwrap_or_default()),
                        value: DicomValue::Sequence(parsed),
                        element_class,
                        source: None,
                    });
                }
                // An SQ element with no item view is retained opaquely when it
                // can be re-encoded, and recorded as a loss otherwise.
                None => match element.to_bytes() {
                    Ok(bytes) => {
                        seq_item.insert(DicomObjectNode {
                            tag: dicom_tag,
                            vr: Some(ArrayString::<2>::try_from("SQ").unwrap_or_default()),
                            value: DicomValue::Bytes(bytes.to_vec()),
                            element_class,
                            source: None,
                        });
                    }
                    Err(_) => losses.push(DicomRetentionLoss::new(
                        dicom_tag,
                        DicomRetentionReason::SequenceItemsUnavailable,
                    )),
                },
            }
        } else {
            let is_binary_vr = matches!(
                element.vr(),
                VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN
            );
            if is_binary_vr {
                match element.to_bytes() {
                    Ok(bytes) => {
                        seq_item.insert(DicomObjectNode {
                            tag: dicom_tag,
                            vr: Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                            value: DicomValue::Bytes(bytes.to_vec()),
                            element_class,
                            source: None,
                        });
                    }
                    Err(_) => losses.push(DicomRetentionLoss::new(
                        dicom_tag,
                        DicomRetentionReason::ValueBytesUnavailable,
                    )),
                }
            } else {
                match element.to_str() {
                    Ok(s) => {
                        seq_item.insert(DicomObjectNode::text(dicom_tag, vr_str, s.to_string()));
                    }
                    _ => match element.to_bytes() {
                        Ok(bytes) => {
                            seq_item.insert(DicomObjectNode {
                                tag: dicom_tag,
                                vr: Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                                value: DicomValue::Bytes(bytes.to_vec()),
                                element_class,
                                source: None,
                            });
                        }
                        Err(_) => losses.push(DicomRetentionLoss::new(
                            dicom_tag,
                            DicomRetentionReason::ValueBytesUnavailable,
                        )),
                    },
                }
            }
        }
    }
    seq_item
}

/// Capture every non-handled element of `obj` into `preservation`.
///
/// Every element is interpreted, retained opaquely, or recorded as a scoped
/// loss; nothing is discarded without a record. The walker lives beside the
/// sequence walker rather than in `parse` so the retention rules stay in one
/// module.
pub(super) fn preserve_unhandled_elements(
    obj: &DefaultDicomObject,
    preservation: &mut DicomPreservationSet,
) {
    let handled = known_handled_tags();
    let mut losses: Vec<DicomRetentionLoss> = Vec::new();
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
            match element.value().items() {
                Some(sub_items) => {
                    let parsed: Vec<_> = sub_items
                        .iter()
                        .map(|i| parse_sequence_item(i, 0, &mut losses))
                        .collect();
                    preservation.object.insert(DicomObjectNode {
                        tag: dicom_tag,
                        vr: Some(ArrayString::<2>::try_from("SQ").unwrap_or_default()),
                        value: DicomValue::Sequence(parsed),
                        element_class,
                        source: None,
                    });
                }
                // An SQ element with no item view is retained opaquely when
                // it can be re-encoded, and recorded as a loss otherwise.
                None => match element.to_bytes() {
                    Ok(bytes) => {
                        preservation.object.insert(DicomObjectNode {
                            tag: dicom_tag,
                            vr: Some(ArrayString::<2>::try_from("SQ").unwrap_or_default()),
                            value: DicomValue::Bytes(bytes.to_vec()),
                            element_class,
                            source: None,
                        });
                    }
                    Err(_) => losses.push(DicomRetentionLoss::new(
                        dicom_tag,
                        DicomRetentionReason::SequenceItemsUnavailable,
                    )),
                },
            }
        } else {
            // Binary VRs bypass to_str(): dicom-rs 0.8 decimal-formats them
            // silently instead of erroring, which corrupts raw payloads.
            let is_binary_vr = matches!(
                element.vr(),
                VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN
            );
            if is_binary_vr {
                match element.to_bytes() {
                    Ok(bytes) => {
                        preservation.preserve(DicomPreservedElement::new(
                            dicom_tag,
                            Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                            bytes.to_vec(),
                        ));
                    }
                    Err(_) => losses.push(DicomRetentionLoss::new(
                        dicom_tag,
                        DicomRetentionReason::ValueBytesUnavailable,
                    )),
                }
            } else {
                match element.to_str() {
                    Ok(s) => {
                        preservation.object.insert(DicomObjectNode {
                            tag: dicom_tag,
                            vr: Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                            value: DicomValue::Text(s.to_string()),
                            element_class,
                            source: None,
                        });
                    }
                    _ => match element.to_bytes() {
                        Ok(bytes) => {
                            preservation.preserve(DicomPreservedElement::new(
                                dicom_tag,
                                Some(ArrayString::<2>::try_from(vr_str).unwrap_or_default()),
                                bytes.to_vec(),
                            ));
                        }
                        Err(_) => losses.push(DicomRetentionLoss::new(
                            dicom_tag,
                            DicomRetentionReason::ValueBytesUnavailable,
                        )),
                    },
                }
            }
        }
    }
    preservation.losses.extend(losses);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::value::{DataSetSequence, PrimitiveValue, Value};
    use dicom::core::{DataElement, Tag, VR};
    use dicom::object::InMemDicomObject;

    const NESTED_SEQUENCE: Tag = Tag(0x0040, 0xA730);
    const LEAF: Tag = Tag(0x0008, 0x0080);

    /// The retention-side tag for [`NESTED_SEQUENCE`].
    fn nested_tag() -> DicomTag {
        DicomTag::new(NESTED_SEQUENCE.group(), NESTED_SEQUENCE.element())
    }

    /// One sequence element holding exactly one item.
    fn wrap(inner: InMemDicomObject) -> InMemDicomObject {
        let mut parent = InMemDicomObject::new_empty();
        parent.put(DataElement::new(
            NESTED_SEQUENCE,
            VR::SQ,
            Value::Sequence(DataSetSequence::from(vec![inner])),
        ));
        parent
    }

    /// An object with `levels` nested sequence elements above a text leaf.
    fn nested(levels: usize) -> InMemDicomObject {
        let mut current = InMemDicomObject::new_empty();
        current.put(DataElement::new(LEAF, VR::LO, PrimitiveValue::from("leaf")));
        for _ in 0..levels {
            current = wrap(current);
        }
        current
    }

    /// The nesting limit is a retention bound, not a silent truncation: the
    /// element whose child would exceed it is recorded, so the unwalked subtree
    /// is visible to a conversion preflight.
    #[test]
    fn nesting_past_the_retention_bound_records_a_scoped_loss() {
        let object = nested(MAX_RETAINED_SEQUENCE_DEPTH + 1);
        let mut losses = Vec::new();
        let item = parse_sequence_item(&object, 0, &mut losses);

        assert_eq!(
            losses.len(),
            1,
            "exactly one truncation point exists, got {losses:?}"
        );
        assert_eq!(losses[0].tag, nested_tag());
        assert_eq!(losses[0].reason, DicomRetentionReason::NestingDepthExceeded);
        assert!(
            item.get(nested_tag()).is_some(),
            "the walkable part of the tree is still retained"
        );
    }

    /// The deepest fully walkable tree records no loss, so the bound above is
    /// the only thing that produces one.
    #[test]
    fn nesting_at_the_retention_bound_records_no_loss() {
        let object = nested(MAX_RETAINED_SEQUENCE_DEPTH);
        let mut losses = Vec::new();
        let _ = parse_sequence_item(&object, 0, &mut losses);

        assert!(
            losses.is_empty(),
            "a tree within the bound must be fully retained, got {losses:?}"
        );
    }

    /// Binary payloads are retained opaquely rather than dropped, and a private
    /// group is classified as private at depth.
    #[test]
    fn a_nested_binary_private_element_is_retained_opaquely() {
        const PRIVATE_BINARY: Tag = Tag(0x0009, 0x1001);
        let private_tag = DicomTag::new(PRIVATE_BINARY.group(), PRIVATE_BINARY.element());
        let mut leaf = InMemDicomObject::new_empty();
        leaf.put(DataElement::new(
            PRIVATE_BINARY,
            VR::OB,
            PrimitiveValue::from(vec![0xAB_u8, 0xCD, 0xEF, 0x01]),
        ));

        let mut losses = Vec::new();
        let item = parse_sequence_item(&wrap(leaf), 0, &mut losses);
        assert!(losses.is_empty(), "an OB payload is retainable: {losses:?}");

        let sequence = item.get(nested_tag()).expect("the sequence is retained");
        let DicomValue::Sequence(inner) = &sequence.value else {
            panic!("the sequence keeps its typed node");
        };
        let node = inner[0]
            .get(private_tag)
            .expect("the private binary element is retained");
        assert_eq!(node.element_class, DicomElementClass::Private);
        assert_eq!(node.value, DicomValue::Bytes(vec![0xAB, 0xCD, 0xEF, 0x01]));
    }
}
