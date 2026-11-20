//! Part 10 Enhanced PET fixtures for reader and viewer integration tests.

use std::path::{Path, PathBuf};

use ritk_dicom::{
    dataset::{DataElement, Dataset, FileMetaBuilder, PrimitiveValue, Sequence, Value, VR},
    DicomTag as Tag,
};

const ENHANCED_PET_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.130";
pub(super) fn item_with_items(object: &mut Dataset, tag: Tag, items: Vec<Dataset>) {
    object.put(DataElement::new(
        tag,
        VR::SQ,
        Value::from(Sequence::from(items)),
    ));
}

pub(super) fn mapping(unit: &str, samples: Option<Vec<f64>>) -> Dataset {
    let mut item = Dataset::new();
    let mut code = Dataset::new();
    code.put(DataElement::new(
        Tag::new(8, 0x0100),
        VR::SH,
        PrimitiveValue::from(unit),
    ));
    code.put(DataElement::new(
        Tag::new(8, 0x0102),
        VR::SH,
        PrimitiveValue::from("UCUM"),
    ));
    item_with_items(&mut item, Tag::new(0x40, 0x08ea), vec![code]);
    item.put(DataElement::new(
        Tag::new(0x40, 0x9216),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    item.put(DataElement::new(
        Tag::new(0x40, 0x9211),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    if let Some(values) = samples {
        item.put(DataElement::new(
            Tag::new(0x40, 0x9212),
            VR::FD,
            PrimitiveValue::F64(values.into()),
        ));
    } else {
        item.put(DataElement::new(
            Tag::new(0x40, 0x9225),
            VR::FD,
            PrimitiveValue::from(2500.0),
        ));
        item.put(DataElement::new(
            Tag::new(0x40, 0x9224),
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
) -> Dataset {
    let mut obj = Dataset::new();
    for (tag, vr, value) in [
        (Tag::new(8, 0x0016), VR::UI, ENHANCED_PET_IMAGE_STORAGE),
        (Tag::new(8, 0x0018), VR::UI, "2.25.324.2"),
        (Tag::new(8, 0x0060), VR::CS, "PT"),
        (Tag::new(0x20, 0x000e), VR::UI, "2.25.325"),
        (Tag::new(0x28, 0x0004), VR::CS, "MONOCHROME2"),
        (Tag::new(0x28, 8), VR::IS, "8"),
        (Tag::new(0x28, 0x0030), VR::DS, "2\\2"),
        (Tag::new(0x20, 0x0037), VR::DS, "1\\0\\0\\0\\1\\0"),
        (Tag::new(0x10, 0x1030), VR::DS, "74"),
        (Tag::new(0x18, 0x9758), VR::CS, "NO"),
    ] {
        obj.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
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
            Tag::new(0x28, tag),
            VR::US,
            PrimitiveValue::from(value),
        ));
    }
    let mut isotope = Dataset::new();
    isotope.put(DataElement::new(
        Tag::new(0x18, 0x1074),
        VR::DS,
        PrimitiveValue::from("370"),
    ));
    isotope.put(DataElement::new(
        Tag::new(0x18, 0x1075),
        VR::DS,
        PrimitiveValue::from("60"),
    ));
    isotope.put(DataElement::new(
        Tag::new(0x18, 0x1078),
        VR::DT,
        PrimitiveValue::from("20261118235900+0100"),
    ));
    item_with_items(&mut obj, Tag::new(0x54, 0x0016), vec![isotope]);

    let content_tag = Tag::new(0x20, 0x9111);
    let dimensions = [
        (Tag::new(0x20, 0x9128), content_tag),
        (Tag::new(0x20, 0x9056), content_tag),
        (gate_attribute, gate_group),
        (Tag::new(0x20, 0x9057), content_tag),
    ]
    .map(|(attribute, group)| {
        Dataset::from_elements([
            DataElement::new(
                Tag::new(0x20, 0x9164),
                VR::UI,
                PrimitiveValue::from("2.25.324.4"),
            ),
            DataElement::new(
                Tag::new(0x20, 0x9165),
                VR::AT,
                PrimitiveValue::from(attribute),
            ),
            DataElement::new(Tag::new(0x20, 0x9167), VR::AT, PrimitiveValue::from(group)),
        ])
    });
    item_with_items(
        &mut obj,
        Tag::new(0x20, 0x9221),
        vec![Dataset::from_elements([DataElement::new(
            Tag::new(0x20, 0x9164),
            VR::UI,
            PrimitiveValue::from("2.25.324.4"),
        )])],
    );
    item_with_items(&mut obj, Tag::new(0x20, 0x9222), dimensions.into());
    let mut frames = Vec::new();
    let mut pixels = Vec::new();
    for z in [2, 1] {
        for time in [2, 1] {
            for gate in [2, 1] {
                let stored = u16::try_from(time + z - 1).unwrap();
                pixels.push(stored);
                let mut frame = Dataset::new();
                let mut content = Dataset::new();
                content.put(DataElement::new(
                    Tag::new(0x20, 0x9157),
                    VR::UL,
                    PrimitiveValue::U32(vec![time, 1, gate, z].into()),
                ));
                content.put(DataElement::new(
                    Tag::new(0x20, 0x9128),
                    VR::UL,
                    PrimitiveValue::from(time),
                ));
                content.put(DataElement::new(
                    Tag::new(0x20, 0x9056),
                    VR::SH,
                    PrimitiveValue::from("1"),
                ));
                content.put(DataElement::new(
                    Tag::new(0x20, 0x9057),
                    VR::UL,
                    PrimitiveValue::from(z),
                ));
                content.put(DataElement::new(
                    Tag::new(0x18, 0x9151),
                    VR::DT,
                    PrimitiveValue::from(if time == 1 {
                        "20261118235900+0100"
                    } else {
                        "20261119000000+0100"
                    }),
                ));
                item_with_items(&mut frame, content_tag, vec![content]);
                let mut cycle = Dataset::new();
                cycle.put(DataElement::new(
                    gate_attribute,
                    VR::FD,
                    PrimitiveValue::from(gate_values[usize::try_from(gate - 1).unwrap()]),
                ));
                item_with_items(&mut frame, gate_group, vec![cycle]);
                let mut plane = Dataset::new();
                plane.put(DataElement::new(
                    Tag::new(0x20, 0x0032),
                    VR::DS,
                    PrimitiveValue::from(if z == 1 { "0\\0\\4" } else { "0\\0\\7" }),
                ));
                item_with_items(&mut frame, Tag::new(0x20, 0x9113), vec![plane]);
                let law = if gate == 1 {
                    mapping("Bq/ml", None)
                } else {
                    mapping(
                        "{SUVbw}g/ml",
                        Some((1..=4).map(|raw| f64::from(raw + time + 2)).collect()),
                    )
                };
                item_with_items(&mut frame, Tag::new(0x40, 0x9096), vec![law]);
                frames.push(frame);
            }
        }
    }
    item_with_items(&mut obj, Tag::new(0x5200, 0x9230), frames);
    obj.put(DataElement::new(
        Tag::new(0x7fe0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(pixels.into()),
    ));
    obj
}

pub(super) fn write_pet_instance(obj: &Dataset, path: &std::path::Path) {
    let uid = obj.element(Tag::new(8, 0x0018)).unwrap().to_str().unwrap();
    obj.clone()
        .with_meta(
            FileMetaBuilder::new()
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
    let obj = dynamic_gated(
        Tag::new(0x18, 0x9118),
        Tag::new(0x20, 0x9153),
        [100.0, 350.0],
    );
    for (i, path) in [&first, &second].into_iter().enumerate() {
        let mut part = obj.clone();
        let uid = format!("2.25.324.2.{}", i + 1);
        part.put(DataElement::new(
            Tag::new(8, 0x0018),
            VR::UI,
            PrimitiveValue::from(uid),
        ));
        part.put(DataElement::new(
            Tag::new(0x28, 8),
            VR::IS,
            PrimitiveValue::from("4"),
        ));
        for (tag, value) in [(0x9161, "2.25.324.3"), (0x0242, "2.25.324.2")] {
            part.put(DataElement::new(
                Tag::new(0x20, tag),
                VR::UI,
                PrimitiveValue::from(value),
            ));
        }
        for (tag, value) in [(0x9162, u16::try_from(i + 1).unwrap()), (0x9163, 2)] {
            part.put(DataElement::new(
                Tag::new(0x20, tag),
                VR::US,
                PrimitiveValue::from(value),
            ));
        }
        part.put(DataElement::new(
            Tag::new(0x20, 0x9228),
            VR::UL,
            PrimitiveValue::from(u32::try_from(4 * i).unwrap()),
        ));
        item_with_items(&mut part, Tag::new(0x5200, 0x9229), vec![Dataset::new()]);
        let frames = obj
            .element(Tag::new(0x5200, 0x9230))
            .unwrap()
            .items()
            .unwrap();
        item_with_items(
            &mut part,
            Tag::new(0x5200, 0x9230),
            frames[4 * i..4 * i + 4].to_vec(),
        );
        part.put(DataElement::new(
            Tag::new(0x7fe0, 0x0010),
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
