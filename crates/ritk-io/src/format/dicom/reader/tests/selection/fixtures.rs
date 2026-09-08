//! Synthetic Part 10 instances and linked directory records.
use super::super::support::*;
use dicom::core::header::Length;
use dicom::core::value::{DataSetSequence, Value};
use std::path::Path;
pub(super) const SERIES: &str = "2.25.73001";
pub(super) const OTHER: &str = "2.25.73002";
const CT: &str = "1.2.840.10008.5.1.4.1.1.2";

pub(super) fn instance(uid: Option<&str>, number: u16, value: u16) -> Vec<u8> {
    let mut object = InMemDicomObject::new_empty();
    for (tag, vr, text) in [
        (Tag(0x0008, 0x0016), VR::UI, CT.to_owned()),
        (Tag(0x0008, 0x0018), VR::UI, format!("2.25.74001.{number}")),
        (Tag(0x0008, 0x0060), VR::CS, "CT".to_owned()),
        (Tag(0x0020, 0x000D), VR::UI, "2.25.75001".to_owned()),
        (Tag(0x0008, 0x103E), VR::LO, format!("Acquisition {value}")),
        (Tag(0x0020, 0x0013), VR::IS, number.to_string()),
        (
            Tag(0x0020, 0x0032),
            VR::DS,
            format!("10\\20\\{}", 2 * number),
        ),
        (Tag(0x0020, 0x0037), VR::DS, "1\\0\\0\\0\\1\\0".to_owned()),
        (Tag(0x0028, 0x0030), VR::DS, "0.5\\0.5".to_owned()),
        (Tag(0x0018, 0x0050), VR::DS, "2".to_owned()),
        (Tag(0x0028, 0x0004), VR::CS, "MONOCHROME2".to_owned()),
    ] {
        object.put(DataElement::new(tag, vr, PrimitiveValue::from(text)));
    }
    if let Some(uid) = uid {
        object.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(uid),
        ));
    }
    for (tag, value) in [
        (Tag(0x0028, 0x0010), 2_u16),
        (Tag(0x0028, 0x0011), 2),
        (Tag(0x0028, 0x0002), 1),
        (Tag(0x0028, 0x0100), 16),
        (Tag(0x0028, 0x0101), 16),
        (Tag(0x0028, 0x0102), 15),
        (Tag(0x0028, 0x0103), 0),
    ] {
        object.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
    }
    object.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(vec![value; 4].into()),
    ));
    let file = object
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(CT)
                .media_storage_sop_instance_uid(format!("2.25.74001.{number}"))
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("fixture metadata");
    let mut bytes = Vec::new();
    file.write_all(&mut bytes).expect("fixture encoding");
    bytes
}

/// Write linked directory records with byte offsets per DICOM PS3.3 F.3.2.2.
/// https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.3.2.2.html
pub(super) fn index(path: &Path, references: &[&str]) {
    let mut records = vec![record("PATIENT"), record("STUDY"), record("SERIES")];
    for (record_index, tag, vr, value) in [
        (0, Tag(0x0010, 0x0010), VR::PN, "Synthetic"),
        (0, Tag(0x0010, 0x0020), VR::LO, "SYNTHETIC"),
        (1, Tag(0x0008, 0x0020), VR::DA, "20260905"),
        (1, Tag(0x0008, 0x0030), VR::TM, "120000"),
        (1, Tag(0x0020, 0x0010), VR::SH, "STUDY"),
        (1, Tag(0x0020, 0x000D), VR::UI, "2.25.75001"),
        (2, Tag(0x0008, 0x0060), VR::CS, "CT"),
        (2, Tag(0x0020, 0x0011), VR::IS, "1"),
        (2, Tag(0x0020, 0x000E), VR::UI, SERIES),
    ] {
        records[record_index].put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    for (position, reference) in references.iter().enumerate() {
        let mut image = record("IMAGE");
        for (tag, vr, value) in [
            (Tag(0x0004, 0x1500), VR::CS, (*reference).to_owned()),
            (Tag(0x0004, 0x1510), VR::UI, CT.to_owned()),
            (
                Tag(0x0004, 0x1511),
                VR::UI,
                format!("2.25.74001.{}", position + 1),
            ),
            (
                Tag(0x0004, 0x1512),
                VR::UI,
                "1.2.840.10008.1.2.1".to_owned(),
            ),
            (Tag(0x0020, 0x0013), VR::IS, (position + 1).to_string()),
        ] {
            image.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
        }
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
    // The fixture has no nested sequences or binary payloads in its records.
    // Each Item tag therefore identifies exactly one directory record.
    let offsets: Vec<u32> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, tag)| *tag == [0xfe, 0xff, 0x00, 0xe0])
        .map(|(offset, _)| u32::try_from(offset).expect("small fixture offset"))
        .collect();
    assert_eq!(offsets.len(), references.len() + 3);
    write_offsets(&mut bytes, 0x1200, &[offsets[0]]);
    write_offsets(&mut bytes, 0x1202, &[offsets[0]]);
    let mut next = vec![0; offsets.len()];
    for (link, offset) in next.iter_mut().skip(3).zip(offsets.iter().skip(4)) {
        *link = *offset;
    }
    write_offsets(&mut bytes, 0x1400, &next);
    let mut lower = vec![0; offsets.len()];
    lower[0] = offsets[1];
    lower[1] = offsets[2];
    if let Some(first_image) = offsets.get(3) {
        lower[2] = *first_image;
    }
    write_offsets(&mut bytes, 0x1420, &lower);
    std::fs::write(path, bytes).expect("write linked index");
}

fn record(kind: &str) -> InMemDicomObject {
    let mut record = InMemDicomObject::new_empty();
    for tag in [Tag(0x0004, 0x1400), Tag(0x0004, 0x1420)] {
        record.put(DataElement::new(tag, VR::UL, PrimitiveValue::from(0_u32)));
    }
    record.put(DataElement::new(
        Tag(0x0004, 0x1410),
        VR::US,
        PrimitiveValue::from(u16::MAX),
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
    assert_eq!(locations.len(), offsets.len());
    for (location, value) in locations.into_iter().zip(offsets) {
        bytes[location..location + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn item_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes
        .windows(4)
        .enumerate()
        .filter(|(_, tag)| *tag == [0xfe, 0xff, 0x00, 0xe0])
        .map(|(offset, _)| offset)
        .collect()
}

pub(super) fn set_record_in_use(path: &Path, record_index: usize, value: u16) {
    let mut bytes = std::fs::read(path).expect("read linked index");
    let offsets = item_offsets(&bytes);
    let start = *offsets.get(record_index).expect("record index");
    let end = offsets
        .get(record_index + 1)
        .copied()
        .unwrap_or(bytes.len());
    write_record_value(&mut bytes[start..end], 0x1410, b"US", &value.to_le_bytes());
    std::fs::write(path, bytes).expect("write linked index");
}

pub(super) fn set_record_next(path: &Path, record_index: usize, value: u32) {
    let mut bytes = std::fs::read(path).expect("read linked index");
    let offsets = item_offsets(&bytes);
    let start = *offsets.get(record_index).expect("record index");
    let end = offsets
        .get(record_index + 1)
        .copied()
        .unwrap_or(bytes.len());
    write_record_value(&mut bytes[start..end], 0x1400, b"UL", &value.to_le_bytes());
    std::fs::write(path, bytes).expect("write linked index");
}

pub(super) fn set_record_sop_instance(path: &Path, record_index: usize, value: &str) {
    let mut bytes = std::fs::read(path).expect("read linked index");
    let offsets = item_offsets(&bytes);
    let start = *offsets.get(record_index).expect("record index");
    let end = offsets
        .get(record_index + 1)
        .copied()
        .unwrap_or(bytes.len());
    write_record_value(&mut bytes[start..end], 0x1511, b"UI", value.as_bytes());
    std::fs::write(path, bytes).expect("write linked index");
}

fn write_record_value(item: &mut [u8], element: u16, vr: &[u8; 2], value: &[u8]) {
    let [lo, hi] = element.to_le_bytes();
    let header = [4, 0, lo, hi, vr[0], vr[1]];
    let header_offset = item
        .windows(8)
        .position(|candidate| {
            candidate[..6] == header
                && candidate[6..8]
                    == [
                        u8::try_from(value.len()).expect("fixture value length fits u8"),
                        0,
                    ]
        })
        .expect("record element header");
    let value_start = header_offset + 8;
    let value_end = value_start + value.len();
    assert_eq!(
        item[value_start..value_end].len(),
        value.len(),
        "fixture replacement must preserve encoded length"
    );
    item[value_start..value_end].copy_from_slice(value);
}

pub(super) fn indexed_study() -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("indexed study");
    let images = directory.path().join("IMAGES");
    std::fs::create_dir(&images).expect("images directory");
    for (number, name, value) in [(1, "ONE", 11), (2, "TWO", 13), (3, "THREE", 17)] {
        std::fs::write(images.join(name), instance(Some(SERIES), number, value))
            .expect("indexed image");
    }
    index(
        &directory.path().join("DICOMDIR"),
        &["IMAGES\\ONE", "IMAGES\\TWO", "IMAGES\\THREE"],
    );
    directory
}
