//! Part 10 Enhanced PET fixtures for reader and viewer integration tests.

use std::path::{Path, PathBuf};

use dicom::{
    core::{
        value::{DataSetSequence, Value},
        DataElement, PrimitiveValue, Tag, VR,
    },
    dictionary_std::uids::ENHANCED_PET_IMAGE_STORAGE,
    object::{meta::FileMetaTableBuilder, InMemDicomObject},
};
pub(super) fn item_with_items(
    object: &mut InMemDicomObject,
    tag: Tag,
    items: Vec<InMemDicomObject>,
) {
    object.put(DataElement::new(
        tag,
        VR::SQ,
        Value::from(DataSetSequence::from(items)),
    ));
}

pub(super) fn mapping(unit: &str, samples: Option<Vec<f64>>) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    let mut code = InMemDicomObject::new_empty();
    code.put(DataElement::new(Tag(8, 0x0100), VR::SH, unit));
    code.put(DataElement::new(Tag(8, 0x0102), VR::SH, "UCUM"));
    item_with_items(&mut item, Tag(0x40, 0x08ea), vec![code]);
    item.put(DataElement::new(
        Tag(0x40, 0x9216),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    item.put(DataElement::new(
        Tag(0x40, 0x9211),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    if let Some(values) = samples {
        item.put(DataElement::new(
            Tag(0x40, 0x9212),
            VR::FD,
            PrimitiveValue::F64(values.into()),
        ));
    } else {
        item.put(DataElement::new(
            Tag(0x40, 0x9225),
            VR::FD,
            PrimitiveValue::from(2500.0),
        ));
        item.put(DataElement::new(
            Tag(0x40, 0x9224),
            VR::FD,
            PrimitiveValue::from(0.0),
        ));
    }
    item
}

#[cfg(test)]
pub(super) fn write_dynamic_gated(
    path: &std::path::Path,
    gate_group: Tag,
    gate_attribute: Tag,
    gate_values: [f64; 2],
) {
    write_pet_instance(
        &dynamic_gated(gate_group, gate_attribute, gate_values),
        path,
    );
}

pub(super) fn dynamic_gated(
    gate_group: Tag,
    gate_attribute: Tag,
    gate_values: [f64; 2],
) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for (tag, vr, value) in [
        (Tag(8, 0x0016), VR::UI, ENHANCED_PET_IMAGE_STORAGE),
        (Tag(8, 0x0018), VR::UI, "2.25.324.2"),
        (Tag(8, 0x0060), VR::CS, "PT"),
        (Tag(0x20, 0x000e), VR::UI, "2.25.325"),
        (Tag(0x28, 0x0004), VR::CS, "MONOCHROME2"),
        (Tag(0x28, 8), VR::IS, "8"),
        (Tag(0x28, 0x0030), VR::DS, "2\\2"),
        (Tag(0x20, 0x0037), VR::DS, "1\\0\\0\\0\\1\\0"),
        (Tag(0x10, 0x1030), VR::DS, "74"),
        (Tag(0x18, 0x9758), VR::CS, "NO"),
    ] {
        obj.put(DataElement::new(tag, vr, value));
    }
    for (tag, value) in [
        (2, 1_u16),
        (0x10, 1),
        (0x11, 1),
        (0x0100, 16),
        (0x0101, 16),
        (0x0102, 15),
        (0x0103, 0),
    ] {
        obj.put(DataElement::new(
            Tag(0x28, tag),
            VR::US,
            PrimitiveValue::from(value),
        ));
    }
    let mut isotope = InMemDicomObject::new_empty();
    isotope.put(DataElement::new(Tag(0x18, 0x1074), VR::DS, "370"));
    isotope.put(DataElement::new(Tag(0x18, 0x1075), VR::DS, "60"));
    isotope.put(DataElement::new(
        Tag(0x18, 0x1078),
        VR::DT,
        "20261118235900+0100",
    ));
    item_with_items(&mut obj, Tag(0x54, 0x0016), vec![isotope]);

    let content_tag = Tag(0x20, 0x9111);
    let dimensions = [
        (Tag(0x20, 0x9128), content_tag),
        (Tag(0x20, 0x9056), content_tag),
        (gate_attribute, gate_group),
        (Tag(0x20, 0x9057), content_tag),
    ]
    .map(|(attribute, group)| {
        InMemDicomObject::from_element_iter([
            DataElement::new(Tag(0x20, 0x9164), VR::UI, "2.25.324.4"),
            DataElement::new(Tag(0x20, 0x9165), VR::AT, PrimitiveValue::from(attribute)),
            DataElement::new(Tag(0x20, 0x9167), VR::AT, PrimitiveValue::from(group)),
        ])
    });
    item_with_items(
        &mut obj,
        Tag(0x20, 0x9221),
        vec![InMemDicomObject::from_element_iter([DataElement::new(
            Tag(0x20, 0x9164),
            VR::UI,
            "2.25.324.4",
        )])],
    );
    item_with_items(&mut obj, Tag(0x20, 0x9222), dimensions.into());
    let mut frames = Vec::new();
    let mut pixels = Vec::new();
    for z in [2, 1] {
        for time in [2, 1] {
            for gate in [2, 1] {
                let stored = u16::try_from(time + z - 1).unwrap();
                pixels.push(stored);
                let mut frame = InMemDicomObject::new_empty();
                let mut content = InMemDicomObject::new_empty();
                content.put(DataElement::new(
                    Tag(0x20, 0x9157),
                    VR::UL,
                    PrimitiveValue::U32(vec![time, 1, gate, z].into()),
                ));
                content.put(DataElement::new(
                    Tag(0x20, 0x9128),
                    VR::UL,
                    PrimitiveValue::from(time),
                ));
                content.put(DataElement::new(Tag(0x20, 0x9056), VR::SH, "1"));
                content.put(DataElement::new(
                    Tag(0x20, 0x9057),
                    VR::UL,
                    PrimitiveValue::from(z),
                ));
                content.put(DataElement::new(
                    Tag(0x18, 0x9151),
                    VR::DT,
                    if time == 1 {
                        "20261118235900+0100"
                    } else {
                        "20261119000000+0100"
                    },
                ));
                item_with_items(&mut frame, content_tag, vec![content]);
                let mut cycle = InMemDicomObject::new_empty();
                cycle.put(DataElement::new(
                    gate_attribute,
                    VR::FD,
                    PrimitiveValue::from(gate_values[usize::try_from(gate - 1).unwrap()]),
                ));
                item_with_items(&mut frame, gate_group, vec![cycle]);
                let mut plane = InMemDicomObject::new_empty();
                plane.put(DataElement::new(
                    Tag(0x20, 0x0032),
                    VR::DS,
                    if z == 1 { "0\\0\\4" } else { "0\\0\\7" },
                ));
                item_with_items(&mut frame, Tag(0x20, 0x9113), vec![plane]);
                let law = if gate == 1 {
                    mapping("Bq/ml", None)
                } else {
                    mapping(
                        "{SUVbw}g/ml",
                        Some((1..=4).map(|raw| f64::from(raw + time + 2)).collect()),
                    )
                };
                item_with_items(&mut frame, Tag(0x40, 0x9096), vec![law]);
                frames.push(frame);
            }
        }
    }
    item_with_items(&mut obj, Tag(0x5200, 0x9230), frames);
    obj.put(DataElement::new(
        Tag(0x7fe0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(pixels.into()),
    ));
    obj
}

pub(super) fn write_pet_instance(obj: &InMemDicomObject, path: &std::path::Path) {
    let uid = obj.element(Tag(8, 0x0018)).unwrap().to_str().unwrap();
    obj.clone()
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(ENHANCED_PET_IMAGE_STORAGE)
                .media_storage_sop_instance_uid(uid.as_ref())
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .unwrap()
        .write_to_file(path)
        .unwrap();
}

/// Write a two-member Enhanced PET concatenation with two times, two gates,
/// and two slices per volume. Every reconstructed volume spans both members.
pub fn write_dynamic_gated_pet_concatenation(directory: &Path) -> [PathBuf; 2] {
    let first = directory.join("z-first.dcm");
    let second = directory.join("a-second.dcm");
    let obj = dynamic_gated(Tag(0x18, 0x9118), Tag(0x20, 0x9153), [100.0, 350.0]);
    for (i, path) in [&first, &second].into_iter().enumerate() {
        let mut part = obj.clone();
        let uid = format!("2.25.324.2.{}", i + 1);
        part.put(DataElement::new(Tag(8, 0x0018), VR::UI, uid));
        part.put(DataElement::new(Tag(0x28, 8), VR::IS, "4"));
        for (tag, value) in [(0x9161, "2.25.324.3"), (0x0242, "2.25.324.2")] {
            part.put(DataElement::new(Tag(0x20, tag), VR::UI, value));
        }
        for (tag, value) in [(0x9162, u16::try_from(i + 1).unwrap()), (0x9163, 2)] {
            part.put(DataElement::new(
                Tag(0x20, tag),
                VR::US,
                PrimitiveValue::from(value),
            ));
        }
        part.put(DataElement::new(
            Tag(0x20, 0x9228),
            VR::UL,
            PrimitiveValue::from(u32::try_from(4 * i).unwrap()),
        ));
        item_with_items(
            &mut part,
            Tag(0x5200, 0x9229),
            vec![InMemDicomObject::new_empty()],
        );
        let frames = obj.element(Tag(0x5200, 0x9230)).unwrap().items().unwrap();
        item_with_items(
            &mut part,
            Tag(0x5200, 0x9230),
            frames[4 * i..4 * i + 4].to_vec(),
        );
        part.put(DataElement::new(
            Tag(0x7fe0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(
                if i == 0 {
                    vec![3, 3, 2, 2]
                } else {
                    vec![2, 2, 1, 1]
                }
                .into(),
            ),
        ));
        write_pet_instance(&part, path);
    }
    [first, second]
}
