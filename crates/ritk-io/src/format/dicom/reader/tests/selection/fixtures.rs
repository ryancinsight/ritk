//! Synthetic Part 10 instances and linked directory records.
use super::super::support::*;
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
///
/// The record tree is built by `file_set_index_fixture`, the one builder the
/// crate exposes to consumers' tests as well; this wrapper only supplies the
/// identities these selection fixtures use.
pub(super) fn index(path: &Path, references: &[&str]) {
    use crate::format::dicom::file_set_index_fixture::{
        write_file_set_index, FileSetIdentity, FileSetMember,
    };

    let sop_instance_uids: Vec<String> = (1..=references.len())
        .map(|position| format!("2.25.74001.{position}"))
        .collect();
    let members: Vec<_> = references
        .iter()
        .zip(&sop_instance_uids)
        .map(|(file_id, sop_instance_uid)| FileSetMember {
            file_id,
            sop_class_uid: CT,
            sop_instance_uid,
            transfer_syntax_uid: "1.2.840.10008.1.2.1",
        })
        .collect();
    write_file_set_index(
        path,
        FileSetIdentity {
            study_instance_uid: "2.25.75001",
            series_instance_uid: SERIES,
            modality: "CT",
        },
        &members,
    );
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
