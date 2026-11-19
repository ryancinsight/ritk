//! Part-10 Enhanced PET quantification with independent diagnostic rescale.

use aequitas::systems::si::units::GramPerCubicCentimeter;
use dicom::{
    core::{
        value::{DataSetSequence, Value},
        DataElement, PrimitiveValue, Tag, VR,
    },
    dictionary_std::uids::ENHANCED_PET_IMAGE_STORAGE,
    object::{meta::FileMetaTableBuilder, InMemDicomObject},
};
use tracing_subscriber::fmt::format::FmtSpan;

fn item_with_items(object: &mut InMemDicomObject, tag: Tag, items: Vec<InMemDicomObject>) {
    object.put(DataElement::new(
        tag,
        VR::SQ,
        Value::from(DataSetSequence::from(items)),
    ));
}

fn mapping(unit: &str, samples: Option<Vec<f64>>) -> InMemDicomObject {
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

#[test]
fn enhanced_pet_quantifies_lut_and_activity_while_preserving_display_transform() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().expect("create fixture directory");
    let path = directory.path().join("enhanced-pet.dcm");
    tracing::info_span!("arrange").in_scope(|| {
        let mut obj = InMemDicomObject::new_empty();
        for (tag, vr, value) in [
            (Tag(8, 0x0016), VR::UI, ENHANCED_PET_IMAGE_STORAGE),
            (Tag(8, 0x0018), VR::UI, "2.25.324.1"),
            (Tag(8, 0x0060), VR::CS, "PT"),
            (Tag(0x20, 0x000e), VR::UI, "2.25.324"),
            (Tag(0x28, 0x0004), VR::CS, "MONOCHROME2"),
            (Tag(0x28, 8), VR::IS, "2"),
            (Tag(0x28, 0x1052), VR::DS, "11"),
            (Tag(0x28, 0x1053), VR::DS, "7"),
            (Tag(0x10, 0x1030), VR::DS, "74"),
            (Tag(0x18, 0x9758), VR::CS, "YES"),
            (Tag(0x18, 0x9701), VR::DT, "20261119000000.0+0100"),
        ] {
            obj.put(DataElement::new(tag, vr, value));
        }
        for (tag, value) in [
            (2, 1_u16),
            (0x10, 2),
            (0x11, 2),
            (0x0100, 16),
            (0x0101, 16),
            (0x0102, 15),
            (0x0103, 0),
            (0x0120, 4),
        ] {
            obj.put(DataElement::new(
                Tag(0x28, tag),
                VR::US,
                PrimitiveValue::from(value),
            ));
        }
        obj.put(DataElement::new(
            Tag(0x7fe0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(vec![0, 2, 3, 4, 0, 2, 3, 4].into()),
        ));
        let mut isotope = InMemDicomObject::new_empty();
        isotope.put(DataElement::new(Tag(0x18, 0x1074), VR::DS, "370"));
        isotope.put(DataElement::new(Tag(0x18, 0x1075), VR::DS, "60"));
        isotope.put(DataElement::new(
            Tag(0x18, 0x1078),
            VR::DT,
            "20261118235900.0+0100",
        ));
        item_with_items(&mut obj, Tag(0x54, 0x0016), vec![isotope]);

        let mut first = InMemDicomObject::new_empty();
        item_with_items(
            &mut first,
            Tag(0x40, 0x9096),
            vec![mapping("{SUVbw}g/ml", Some(vec![0.5, 1.5, 2.5, 3.5]))],
        );
        let mut next = InMemDicomObject::new_empty();
        item_with_items(&mut next, Tag(0x40, 0x9096), vec![mapping("Bq/ml", None)]);
        item_with_items(&mut obj, Tag(0x5200, 0x9230), vec![first, next]);
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(ENHANCED_PET_IMAGE_STORAGE)
                .media_storage_sop_instance_uid("2.25.324.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("Part 10 file meta")
        .write_to_file(&path)
        .expect("write Enhanced PET");
    });
    let volume = tracing::info_span!("act")
        .in_scope(|| super::super::load_dicom_multiframe_flat(&path).expect("decode Enhanced PET"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(volume.shape, [2, 2, 2]);
        assert_eq!(
            volume.data,
            [11.0, 25.0, 32.0, 39.0, 11.0, 25.0, 32.0, 39.0]
        );
        let suv = volume.suv_bw.expect("quantitative samples");
        let value = |index| suv.at(index).map(|v| v.in_unit::<GramPerCubicCentimeter>());
        assert_eq!(
            [value(0), value(1), value(2), value(3)],
            [None, Some(1.5), Some(2.5), None]
        );
        assert_eq!(
            [value(4), value(5), value(6), value(7)],
            [None, Some(2.0), Some(3.0), None]
        );
    });
}
