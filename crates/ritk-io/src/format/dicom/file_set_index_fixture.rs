//! Conformant Part 10 file-set index construction for tests.
//!
//! The DICOMDIR reader validates PS3.3 F.3.2.2 structure — record in-use
//! flags, the root chain, lower-level links, and reachability — before it will
//! admit an IMAGE record. Hand-rolling that structure in a consumer's test is
//! how one gets a fixture the reader correctly rejects, which is what happened
//! to `ritk-snap` when the reader was tightened. One builder lives here, in
//! the crate that owns the format, and every test writes its indices through
//! it.
//!
//! Gated behind `test-util` so nothing enters a default build.

use std::path::Path;

use dicom::core::header::Length;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

/// One referenced Part 10 instance, as the index must describe it.
///
/// The reader compares each field against the referenced file, so these must
/// be the identities that file itself carries.
#[derive(Clone, Copy, Debug)]
pub struct FileSetMember<'a> {
    /// ReferencedFileID (0004,1500): backslash-separated path components
    /// relative to the index, as `IMAGES\SLICE001`.
    pub file_id: &'a str,
    /// ReferencedSOPClassUIDInFile (0004,1510).
    pub sop_class_uid: &'a str,
    /// ReferencedSOPInstanceUIDInFile (0004,1511).
    pub sop_instance_uid: &'a str,
    /// ReferencedTransferSyntaxUIDInFile (0004,1512).
    pub transfer_syntax_uid: &'a str,
}

/// Identity written into the PATIENT, STUDY and SERIES records above the
/// images.
///
/// The reader inspects only IMAGE records, but the hierarchy above them must
/// exist and link correctly for those records to be reachable.
#[derive(Clone, Copy, Debug)]
pub struct FileSetIdentity<'a> {
    /// StudyInstanceUID (0020,000D).
    pub study_instance_uid: &'a str,
    /// SeriesInstanceUID (0020,000E).
    pub series_instance_uid: &'a str,
    /// Modality (0008,0060) of the series record.
    pub modality: &'a str,
}

const RECORD_IN_USE: u16 = u16::MAX;
const ITEM_TAG: [u8; 4] = [0xfe, 0xff, 0x00, 0xe0];

/// Write a DICOMDIR whose records satisfy the reader's structural contract.
///
/// The record tree is PATIENT to STUDY to SERIES to one IMAGE per member,
/// linked by byte offsets patched in after encoding: an offset cannot be known
/// before the stream exists, which is why this writes and then patches rather
/// than serializing once.
///
/// # Panics
///
/// Panics when encoding fails, or when the encoded stream does not carry one
/// item per record — which would mean a record grew a nested sequence this
/// builder does not account for.
pub fn write_file_set_index(
    index: &Path,
    identity: FileSetIdentity<'_>,
    members: &[FileSetMember<'_>],
) {
    let mut records = vec![record("PATIENT"), record("STUDY"), record("SERIES")];
    for (tag, vr, value) in [
        (Tag(0x0010, 0x0010), VR::PN, "Synthetic"),
        (Tag(0x0010, 0x0020), VR::LO, "SYNTHETIC"),
    ] {
        records[0].put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    for (tag, vr, value) in [
        (Tag(0x0008, 0x0020), VR::DA, "20260905"),
        (Tag(0x0008, 0x0030), VR::TM, "120000"),
        (Tag(0x0020, 0x0010), VR::SH, "STUDY"),
        (Tag(0x0020, 0x000D), VR::UI, identity.study_instance_uid),
    ] {
        records[1].put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    for (tag, vr, value) in [
        (Tag(0x0008, 0x0060), VR::CS, identity.modality),
        (Tag(0x0020, 0x0011), VR::IS, "1"),
        (Tag(0x0020, 0x000E), VR::UI, identity.series_instance_uid),
    ] {
        records[2].put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    for (position, member) in members.iter().enumerate() {
        let mut image = record("IMAGE");
        for (tag, vr, value) in [
            (Tag(0x0004, 0x1500), VR::CS, member.file_id),
            (Tag(0x0004, 0x1510), VR::UI, member.sop_class_uid),
            (Tag(0x0004, 0x1511), VR::UI, member.sop_instance_uid),
            (Tag(0x0004, 0x1512), VR::UI, member.transfer_syntax_uid),
        ] {
            image.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
        }
        image.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from((position + 1).to_string()),
        ));
        records.push(image);
    }

    let mut object = InMemDicomObject::new_empty();
    object.put(DataElement::new(
        Tag(0x0004, 0x1130),
        VR::CS,
        PrimitiveValue::from("SYNTHETIC"),
    ));
    for tag in [Tag(0x0004, 0x1200), Tag(0x0004, 0x1202)] {
        object.put(DataElement::new(tag, VR::UL, PrimitiveValue::from(0_u32)));
    }
    object.put(DataElement::new(
        Tag(0x0004, 0x1212),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    object.put(DataElement::new(
        Tag(0x0004, 0x1220),
        VR::SQ,
        Value::from(DataSetSequence::new(records, Length::UNDEFINED)),
    ));

    let file = object
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.1.3.10")
                .media_storage_sop_instance_uid("2.25.76001")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("index metadata");
    let mut bytes = Vec::new();
    file.write_all(&mut bytes).expect("index encoding");

    // Every record is a flat data set, so each Item tag opens exactly one.
    let offsets: Vec<u32> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, tag)| *tag == ITEM_TAG)
        .map(|(offset, _)| u32::try_from(offset).expect("small fixture offset"))
        .collect();
    assert_eq!(
        offsets.len(),
        members.len() + 3,
        "one item per record: three hierarchy records plus one image per member"
    );

    write_offsets(&mut bytes, 0x1200, &[offsets[0]]);
    write_offsets(&mut bytes, 0x1202, &[offsets[0]]);
    // Siblings: the images chain to each other; the hierarchy records do not.
    let mut next = vec![0; offsets.len()];
    for (link, offset) in next.iter_mut().skip(3).zip(offsets.iter().skip(4)) {
        *link = *offset;
    }
    write_offsets(&mut bytes, 0x1400, &next);
    // Children: patient to study, study to series, series to the first image.
    let mut lower = vec![0; offsets.len()];
    lower[0] = offsets[1];
    lower[1] = offsets[2];
    if let Some(first_image) = offsets.get(3) {
        lower[2] = *first_image;
    }
    write_offsets(&mut bytes, 0x1420, &lower);

    std::fs::write(index, bytes).expect("write linked index");
}

fn record(kind: &str) -> InMemDicomObject {
    let mut record = InMemDicomObject::new_empty();
    for tag in [Tag(0x0004, 0x1400), Tag(0x0004, 0x1420)] {
        record.put(DataElement::new(tag, VR::UL, PrimitiveValue::from(0_u32)));
    }
    record.put(DataElement::new(
        Tag(0x0004, 0x1410),
        VR::US,
        PrimitiveValue::from(RECORD_IN_USE),
    ));
    record.put(DataElement::new(
        Tag(0x0004, 0x1430),
        VR::CS,
        PrimitiveValue::from(kind),
    ));
    record
}

fn write_offsets(bytes: &mut [u8], element: u16, offsets: &[u32]) {
    let [lo, hi] = element.to_le_bytes();
    let locations: Vec<_> = bytes
        .windows(8)
        .enumerate()
        .filter(|(_, header)| *header == [4, 0, lo, hi, b'U', b'L', 4, 0])
        .map(|(offset, _)| offset + 8)
        .collect();
    assert_eq!(
        locations.len(),
        offsets.len(),
        "one encoded UL per offset to patch"
    );
    for (location, value) in locations.into_iter().zip(offsets) {
        bytes[location..location + 4].copy_from_slice(&value.to_le_bytes());
    }
}
