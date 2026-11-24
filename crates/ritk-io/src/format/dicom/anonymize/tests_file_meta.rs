//! Group 0002 elements found outside the file meta group.
//!
//! PS3.10 section 7.1 keeps the File Meta Information ahead of the data set.
//! Two inputs put its elements elsewhere: an encoder that writes one inside a
//! sequence item, and a file whose group length (0002,0000) ends before the
//! last file meta element. Both reach `anonymize_object` as data set elements
//! and both describe the source site.
//!
//! # Test Coverage Index (this file)
//! 1. nested_file_meta_elements_are_removed_at_every_depth
//! 2. file_meta_elements_past_a_short_group_length_are_removed

use super::{anonymize_object, AnonymizeOptions};
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, Sequence, Value, VR};
use ritk_dicom::dataset::{Dataset, DicomFile, FileMetaBuilder};

const SOP_CLASS_UID: DicomTag = DicomTag::new(0x0008, 0x0016);
const SOP_INSTANCE_UID: DicomTag = DicomTag::new(0x0008, 0x0018);
/// Two sequences the profile keeps, so recursion must reach their items.
const REFERENCED_IMAGE_SEQUENCE: DicomTag = DicomTag::new(0x0008, 0x1140);
const ANATOMIC_REGION_SEQUENCE: DicomTag = DicomTag::new(0x0008, 0x2218);
const SLICE_THICKNESS: DicomTag = DicomTag::new(0x0018, 0x0050);
const MEDIA_STORAGE_SOP_INSTANCE_UID: DicomTag = DicomTag::new(0x0002, 0x0003);
const IMPLEMENTATION_CLASS_UID: DicomTag = DicomTag::new(0x0002, 0x0012);
const IMPLEMENTATION_VERSION_NAME: DicomTag = DicomTag::new(0x0002, 0x0013);
const SOURCE_APPLICATION_ENTITY_TITLE: DicomTag = DicomTag::new(0x0002, 0x0016);
const SENDING_APPLICATION_ENTITY_TITLE: DicomTag = DicomTag::new(0x0002, 0x0017);

fn text(tag: DicomTag, vr: VR, value: &str) -> DataElement {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

fn sequence(tag: DicomTag, items: Vec<Dataset>) -> DataElement {
    DataElement::new(tag, VR::SQ, Value::Sequence(Sequence::from(items)))
}

/// A data set holding only the two SOP UIDs, under Explicit VR Little Endian.
fn object(extra: Vec<DataElement>) -> DicomFile {
    let mut dataset = Dataset::new();
    dataset.put(text(SOP_CLASS_UID, VR::UI, "1.2.840.10008.5.1.4.1.1.4"));
    dataset.put(text(SOP_INSTANCE_UID, VR::UI, "1.2.3.4.5"));
    for element in extra {
        dataset.put(element);
    }
    dataset
        .with_meta(
            FileMetaBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
                .media_storage_sop_instance_uid("1.2.3.4.5")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("invariant: synthetic meta table is valid")
}

/// Every group 0002 element of `dataset` as `(depth, tag)`, in data set order.
fn group_two(dataset: &Dataset, depth: u32, found: &mut Vec<(u32, DicomTag)>) {
    for element in dataset.iter() {
        if element.tag().group == 0x0002 {
            found.push((depth, element.tag()));
        }
        if let Value::Sequence(sequence) = element.value() {
            for item in sequence.items() {
                group_two(item, depth + 1, found);
            }
        }
    }
}

fn group_two_of(object: &DicomFile) -> Vec<(u32, DicomTag)> {
    let mut found = Vec::new();
    group_two(object, 0, &mut found);
    found
}

/// The first item of the sequence at `tag`.
fn first_item(dataset: &Dataset, tag: DicomTag) -> &Dataset {
    match dataset.element(tag).expect("sequence present").value() {
        Value::Sequence(sequence) => sequence.items().first().expect("one item"),
        _ => panic!("{tag:?} is not a sequence"),
    }
}

fn thickness(dataset: &Dataset) -> String {
    dataset
        .element(SLICE_THICKNESS)
        .expect("SliceThickness present")
        .to_str()
        .expect("SliceThickness is text")
        .trim()
        .to_owned()
}

fn contains(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|window| window == needle)
}

#[test]
fn nested_file_meta_elements_are_removed_at_every_depth() {
    let mut inner = Dataset::new();
    inner.put(text(
        SENDING_APPLICATION_ENTITY_TITLE,
        VR::AE,
        "SITE_DEEPER",
    ));
    inner.put(text(SLICE_THICKNESS, VR::DS, "7.5"));
    let mut outer = Dataset::new();
    outer.put(text(MEDIA_STORAGE_SOP_INSTANCE_UID, VR::UI, "1.2.3.999"));
    outer.put(text(SOURCE_APPLICATION_ENTITY_TITLE, VR::AE, "SITE_NESTED"));
    outer.put(text(SLICE_THICKNESS, VR::DS, "2.5"));
    outer.put(sequence(ANATOMIC_REGION_SEQUENCE, vec![inner]));
    let source = object(vec![sequence(REFERENCED_IMAGE_SEQUENCE, vec![outer])]);

    let directory = tempfile::tempdir().expect("tempdir");
    let source_path = directory.path().join("source.dcm");
    source
        .write_to_file(&source_path)
        .expect("write the source");
    let source = DicomFile::open(&source_path).expect("re-read the source");
    assert_eq!(
        group_two_of(&source),
        [
            (1, MEDIA_STORAGE_SOP_INSTANCE_UID),
            (1, SOURCE_APPLICATION_ENTITY_TITLE),
            (2, SENDING_APPLICATION_ENTITY_TITLE),
        ]
    );

    let (anonymized, _result) =
        anonymize_object(source, &AnonymizeOptions::default()).expect("anonymize the object");
    let path = directory.path().join("anonymized.dcm");
    anonymized.write_to_file(&path).expect("write the object");
    let reread = DicomFile::open(&path).expect("re-read the object");

    assert_eq!(group_two_of(&reread), []);
    let outer = first_item(&reread, REFERENCED_IMAGE_SEQUENCE);
    assert_eq!(thickness(outer), "2.5");
    assert_eq!(
        thickness(first_item(outer, ANATOMIC_REGION_SEQUENCE)),
        "7.5"
    );
    let bytes = std::fs::read(&path).expect("read the written bytes");
    for needle in [b"SITE_".as_slice(), b"1.2.3.999"] {
        assert!(
            !contains(&bytes, needle),
            "{}",
            String::from_utf8_lossy(needle)
        );
    }
}

#[test]
fn file_meta_elements_past_a_short_group_length_are_removed() {
    /// Preamble, `DICM`, then the (0002,0000) tag, VR and length fields.
    const GROUP_LENGTH_VALUE: usize = 128 + 4 + 8;
    const GROUP_LENGTH_END: usize = GROUP_LENGTH_VALUE + 4;

    let mut source = object(Vec::new());
    source.update_meta(|meta| {
        meta.implementation_class_uid = "1.2.3.888".to_owned();
        meta.implementation_version_name = Some("FIXTURE_WRITER".to_owned());
        meta.source_application_entity_title = Some("SITE_SCANNER".to_owned());
    });
    let directory = tempfile::tempdir().expect("tempdir");
    let source_path = directory.path().join("source.dcm");
    source
        .write_to_file(&source_path)
        .expect("write the source");

    // End the group before Implementation Class UID (0002,0012): the elements
    // from there on are then read as the start of the data set.
    let mut bytes = std::fs::read(&source_path).expect("read the source bytes");
    let implementation_class = bytes[GROUP_LENGTH_END..]
        .windows(4)
        .position(|window| window == [0x02, 0x00, 0x12, 0x00])
        .expect("the writer stores Implementation Class UID");
    let shortened = u32::try_from(implementation_class).expect("offset fits the length field");
    bytes[GROUP_LENGTH_VALUE..GROUP_LENGTH_END].copy_from_slice(&shortened.to_le_bytes());
    std::fs::write(&source_path, &bytes).expect("write the shortened source");

    let source = DicomFile::open(&source_path).expect("re-read the shortened source");
    assert_eq!(
        group_two_of(&source),
        [
            (0, IMPLEMENTATION_CLASS_UID),
            (0, IMPLEMENTATION_VERSION_NAME),
            (0, SOURCE_APPLICATION_ENTITY_TITLE),
        ]
    );
    assert_eq!(source.meta().source_application_entity_title, None);

    let (anonymized, _result) =
        anonymize_object(source, &AnonymizeOptions::default()).expect("anonymize the object");
    let path = directory.path().join("anonymized.dcm");
    anonymized.write_to_file(&path).expect("write the object");
    let reread = DicomFile::open(&path).expect("re-read the object");

    assert_eq!(group_two_of(&reread), []);
    assert_eq!(reread.meta().source_application_entity_title, None);
    let bytes = std::fs::read(&path).expect("read the written bytes");
    for needle in [b"SITE_SCANNER".as_slice(), b"FIXTURE_WRITER", b"1.2.3.888"] {
        assert!(
            !contains(&bytes, needle),
            "{}",
            String::from_utf8_lossy(needle)
        );
    }
}
