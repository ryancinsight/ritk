use super::*;
use arrayvec::ArrayString;

#[test]
fn private_tag_detection_uses_odd_groups() {
    assert!(is_private_tag(DicomTag::new(0x0019, 0x10AA)));
    assert!(is_private_tag(DicomTag::new(0x0029, 0x0010)));
    assert!(!is_private_tag(DicomTag::new(0x0028, 0x0010)));
}

#[test]
fn canonical_tag_format_is_stable() {
    assert_eq!(DicomTag::new(0x0019, 0x10AA).canonical(), "0019,10AA");
    assert_eq!(DicomTag::new(0x7FE0, 0x0010).canonical(), "7FE0,0010");
}

#[test]
fn sequence_items_preserve_order_and_replace_by_tag() {
    let mut item = DicomSequenceItem::new();
    item.insert(DicomObjectNode::text(
        DicomTag::new(0x0010, 0x0010),
        "PN",
        "Test^Patient",
    ));
    item.insert(DicomObjectNode::text(
        DicomTag::new(0x0010, 0x0020),
        "LO",
        "PAT001",
    ));
    item.insert(DicomObjectNode::text(
        DicomTag::new(0x0010, 0x0020),
        "LO",
        "PAT002",
    ));

    assert_eq!(item.len(), 2);
    assert_eq!(
        item.get(DicomTag::new(0x0010, 0x0020))
            .and_then(|n| n.value.as_text()),
        Some("PAT002")
    );
    assert_eq!(item.elements[0].tag(), DicomTag::new(0x0010, 0x0010));
}

#[test]
fn object_model_replaces_nodes_by_tag() {
    let mut model = DicomObjectModel::new();
    model.insert(DicomObjectNode::text(
        DicomTag::new(0x0008, 0x0060),
        "CS",
        "OT",
    ));
    model.insert(DicomObjectNode::text(
        DicomTag::new(0x0008, 0x0060),
        "CS",
        "CT",
    ));
    assert_eq!(model.len(), 1);
    assert_eq!(
        model
            .get(DicomTag::new(0x0008, 0x0060))
            .and_then(|n| n.value.as_text()),
        Some("CT")
    );
}

#[test]
fn preservation_set_tracks_object_and_raw_elements() {
    let mut set = DicomPreservationSet::new();
    set.object.insert(DicomObjectNode::text(
        DicomTag::new(0x0008, 0x103E),
        "LO",
        "Series Description",
    ));
    set.preserve(DicomPreservedElement::new(
        DicomTag::new(0x0009, 0x1001),
        Some(ArrayString::<2>::try_from("OB").unwrap_or_default()),
        vec![1, 2, 3, 4],
    ));

    assert!(!set.is_empty());
    assert_eq!(set.object.len(), 1);
    assert_eq!(set.preserved.len(), 1);
    assert_eq!(set.preserved[0].bytes, vec![1, 2, 3, 4]);
}
