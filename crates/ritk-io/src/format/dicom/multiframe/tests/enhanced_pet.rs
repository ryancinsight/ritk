//! Part-10 Enhanced PET quantification with independent diagnostic rescale.

use super::super::test_fixture::{
    dynamic_gated, item_with_items, mapping, write_dynamic_gated,
    write_dynamic_gated_pet_concatenation, write_pet_instance,
};
use aequitas::systems::si::units::GramPerCubicCentimeter;
use dicom::{
    core::{DataElement, PrimitiveValue, Tag, VR},
    dictionary_std::uids::ENHANCED_PET_IMAGE_STORAGE,
    object::{meta::FileMetaTableBuilder, InMemDicomObject},
};
use tracing_subscriber::fmt::format::FmtSpan;

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

#[test]
fn enhanced_pet_time_and_cardiac_gates_keep_spatial_geometry_and_suv() {
    use crate::{DicomReadBudget, EnhancedPetCoordinate as Coordinate};
    use aequitas::systems::si::units::Millisecond;

    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dynamic-gated.dcm");
    tracing::info_span!("arrange").in_scope(|| {
        write_dynamic_gated(&path, Tag(0x18, 0x9118), Tag(0x20, 0x9153), [100.0, 350.0])
    });
    let (from_file, from_bytes) = tracing::info_span!("act").in_scope(|| {
        let bytes = std::fs::read(&path).unwrap();
        (
            crate::load_enhanced_pet_volumes(&path).unwrap(),
            crate::load_enhanced_pet_volumes_from_bytes(&path, &bytes, &DicomReadBudget::DEFAULT)
                .unwrap(),
        )
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(from_file, from_bytes);
    assert_eq!(from_file.len(), 4);
    for (index, volume) in from_file.iter().enumerate() {
        let time = index / 2;
        let gate = index % 2;
        assert_eq!(volume.shape, [2, 1, 1]);
        assert_eq!(volume.origin.to_array(), [0.0, 0.0, 4.0]);
        assert_eq!(volume.spacing.to_array(), [3.0, 2.0, 2.0]);
        let expected = (time + 1) as f32;
        assert_eq!(volume.data.as_slice(), [expected, expected + 1.0]);
        assert_eq!(
            volume.dimensions[0].coordinate,
            Some(Coordinate::TemporalPosition((time + 1) as u32))
        );
        let Some(Coordinate::TriggerDelay(delay)) = volume.dimensions[2].coordinate else {
            panic!("expected a typed cardiac trigger delay");
        };
        assert_eq!(delay.in_unit::<Millisecond>(), [100.0, 350.0][gate]);
        let suv = volume.suv_bw.as_ref().unwrap();
        let measured =
            [0, 1].map(|voxel| suv.at(voxel).unwrap().in_unit::<GramPerCubicCentimeter>());
        assert_eq!(
            measured,
            match (time, gate) {
                (0, 0) => [0.5, 1.0],
                (1, 0) => [2.0, 3.0],
                (0, _) => [4.0, 5.0],
                _ => [6.0, 7.0],
            }
        );
    }
    let bytes = std::fs::read(&path).unwrap();
    let limit =
        DicomReadBudget::try_new(DicomReadBudget::DEFAULT.parser(), bytes.len(), 63).unwrap();
    assert!(crate::load_enhanced_pet_volumes_from_bytes(&path, &bytes, &limit).is_err());
    assert!(super::super::load_dicom_multiframe_flat(&path).is_err());
}

#[test]
fn enhanced_pet_respiratory_phase_uses_ratio_and_an_independent_volume() {
    use crate::EnhancedPetCoordinate;
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("respiratory.dcm");
    tracing::info_span!("arrange").in_scope(|| {
        write_dynamic_gated(&path, Tag(0x20, 0x9253), Tag(0x20, 0x9245), [20.0, 70.0]);
    });
    let volumes =
        tracing::info_span!("act").in_scope(|| crate::load_enhanced_pet_volumes(&path).unwrap());
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(volumes.len(), 4);
    for (volume, expected) in volumes.iter().zip([0.2, 0.7, 0.2, 0.7]) {
        let Some(EnhancedPetCoordinate::CyclePhase(phase)) = volume.dimensions[2].coordinate else {
            panic!("expected a typed respiratory ratio");
        };
        assert_eq!(phase.into_base(), expected);
        assert_eq!(volume.shape, [2, 1, 1]);
    }
}

#[test]
fn enhanced_pet_concatenation_resolves_shuffled_members_and_rejects_missing_frames() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().unwrap();
    let standalone = directory.path().join("s-standalone.dcm");
    let [first, second] = tracing::info_span!("arrange").in_scope(|| {
        let obj = dynamic_gated(Tag(0x18, 0x9118), Tag(0x20, 0x9153), [100.0, 350.0]);
        write_pet_instance(&obj, &standalone);
        write_dynamic_gated_pet_concatenation(directory.path())
    });
    let (expected, reconstructed) = tracing::info_span!("act").in_scope(|| {
        (
            crate::load_enhanced_pet_volumes(&standalone).unwrap(),
            crate::load_enhanced_pet_volumes(&second).unwrap(),
        )
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(reconstructed, expected);
    let budget = crate::DicomReadBudget::DEFAULT;
    let members = crate::scan_dicom_path(&first).unwrap().metadata.slices;
    assert_eq!(members.len(), 2);
    let original = members
        .iter()
        .map(|member| member.part10_bytes.as_ref().unwrap().len())
        .sum();
    let insufficient_decode =
        crate::DicomReadBudget::try_new(budget.parser(), original, 63).unwrap();
    let insufficient_input =
        crate::DicomReadBudget::try_new(budget.parser(), original - 1, 64).unwrap();
    for (budget, message) in [
        (insufficient_decode, "decoded"),
        (insufficient_input, "retained-byte"),
    ] {
        let error = crate::load_enhanced_pet_volumes_from_instances(&members, &budget).unwrap_err();
        assert!(format!("{error:#}").contains(message));
    }

    let mut part = dicom::object::open_file(&second).unwrap().into_inner();
    part.put(DataElement::new(
        Tag(0x20, 0x9228),
        VR::UL,
        PrimitiveValue::from(3_u32),
    ));
    write_pet_instance(&part, &second);
    let error = crate::load_enhanced_pet_volumes(&first).unwrap_err();
    assert!(format!("{error:#}").contains("overlap"));
    part.put(DataElement::new(
        Tag(0x20, 0x9228),
        VR::UL,
        PrimitiveValue::from(4_u32),
    ));
    write_pet_instance(&part, &second);
    std::fs::remove_file(&first).unwrap();
    // The retained scanner snapshot remains decodable after source deletion.
    assert_eq!(
        crate::load_enhanced_pet_volumes_from_instances(&members, &budget).unwrap(),
        expected
    );
    let error = crate::load_enhanced_pet_volumes(&second).unwrap_err();
    assert!(format!("{error:#}").contains("offset"));
}

#[test]
fn enhanced_pet_interpolates_a_spatial_gap_between_complete_members() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().unwrap();
    let selected = tracing::info_span!("arrange").in_scope(|| {
        let [first, second] = write_dynamic_gated_pet_concatenation(directory.path());
        let template = dicom::object::open_file(&second).unwrap().into_inner();
        let frame = template
            .element(Tag(0x5200, 0x9230))
            .unwrap()
            .items()
            .unwrap()
            .last()
            .unwrap();
        for (index, path) in [&first, &second].into_iter().enumerate() {
            let mut member = dicom::object::open_file(path).unwrap().into_inner();
            member.put(DataElement::new(Tag(0x28, 8), VR::IS, "2"));
            member.put(DataElement::new(
                Tag(0x20, 0x9228),
                VR::UL,
                PrimitiveValue::from(u32::try_from(2 * index).unwrap()),
            ));
            let positions = if index == 0 { [4, 7] } else { [13, 16] };
            let frames = positions
                .into_iter()
                .enumerate()
                .map(|(local, position)| {
                    let mut frame = frame.clone();
                    let mut content =
                        frame.element(Tag(0x20, 0x9111)).unwrap().items().unwrap()[0].clone();
                    let ordinal = u32::try_from(index * 2 + local + 1).unwrap();
                    content.put(DataElement::new(
                        Tag(0x20, 0x9157),
                        VR::UL,
                        PrimitiveValue::U32(vec![1, 1, 1, ordinal].into()),
                    ));
                    content.put(DataElement::new(
                        Tag(0x20, 0x9057),
                        VR::UL,
                        PrimitiveValue::from(ordinal),
                    ));
                    item_with_items(&mut frame, Tag(0x20, 0x9111), vec![content]);
                    let plane = InMemDicomObject::from_element_iter([DataElement::new(
                        Tag(0x20, 0x0032),
                        VR::DS,
                        format!("0\\0\\{position}"),
                    )]);
                    item_with_items(&mut frame, Tag(0x20, 0x9113), vec![plane]);
                    frame
                })
                .collect();
            item_with_items(&mut member, Tag(0x5200, 0x9230), frames);
            let start = u16::try_from(2 * index + 1).unwrap();
            member.put(DataElement::new(
                Tag(0x7fe0, 0x0010),
                VR::OW,
                PrimitiveValue::U16(vec![start, start + 1].into()),
            ));
            write_pet_instance(&member, path);
        }
        second
    });
    let reconstructions =
        tracing::info_span!("act").in_scope(|| crate::load_enhanced_pet_volumes(selected).unwrap());
    let _assert = tracing::info_span!("assert").entered();
    let [volume] = reconstructions.as_slice() else {
        panic!("expected a single temporal gate");
    };
    assert_eq!(volume.shape, [5, 1, 1]);
    assert_eq!(volume.origin.to_array(), [0.0, 0.0, 4.0]);
    assert_eq!(volume.spacing.to_array(), [3.0, 2.0, 2.0]);
    assert_eq!(volume.data.as_slice(), &[1.0, 2.0, 2.5, 3.0, 4.0]);
    let suv = volume.suv_bw.as_ref().unwrap();
    assert_eq!(
        [0, 1, 2, 3, 4].map(|index| suv.at(index).unwrap().in_unit::<GramPerCubicCentimeter>()),
        [0.5, 1.0, 1.25, 1.5, 2.0]
    );
}
