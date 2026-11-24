//! Part-10 Enhanced PET quantification with independent diagnostic rescale.

use super::super::test_fixture::{
    dynamic_gated, item_with_items, mapping, write_dynamic_gated,
    write_dynamic_gated_pet_concatenation, write_pet_instance,
};
use aequitas::systems::si::units::GramPerCubicCentimeter;
use ritk_dicom::{
    dataset::{DataElement, Dataset, DicomFile, FileMetaBuilder, PrimitiveValue, VR},
    DicomTag as Tag,
};

const ENHANCED_PET_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.130";
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
        let mut obj = Dataset::new();
        for (tag, vr, value) in [
            (Tag::new(8, 0x0016), VR::UI, ENHANCED_PET_IMAGE_STORAGE),
            (Tag::new(8, 0x0018), VR::UI, "2.25.324.1"),
            (Tag::new(8, 0x0060), VR::CS, "PT"),
            (Tag::new(0x20, 0x000e), VR::UI, "2.25.324"),
            (Tag::new(0x28, 0x0004), VR::CS, "MONOCHROME2"),
            (Tag::new(0x28, 8), VR::IS, "2"),
            (Tag::new(0x28, 0x1052), VR::DS, "11"),
            (Tag::new(0x28, 0x1053), VR::DS, "7"),
            (Tag::new(0x10, 0x1030), VR::DS, "74"),
            (Tag::new(0x18, 0x9758), VR::CS, "YES"),
            (Tag::new(0x18, 0x9701), VR::DT, "20261119000000.0+0100"),
        ] {
            obj.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
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
                Tag::new(0x28, tag),
                VR::US,
                PrimitiveValue::from(value),
            ));
        }
        obj.put(DataElement::new(
            Tag::new(0x7fe0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(vec![0, 2, 3, 4, 0, 2, 3, 4].into()),
        ));
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
            PrimitiveValue::from("20261118235900.0+0100"),
        ));
        item_with_items(&mut obj, Tag::new(0x54, 0x0016), vec![isotope]);

        let mut first = Dataset::new();
        item_with_items(
            &mut first,
            Tag::new(0x40, 0x9096),
            vec![mapping("{SUVbw}g/ml", Some(vec![0.5, 1.5, 2.5, 3.5]))],
        );
        let mut next = Dataset::new();
        item_with_items(
            &mut next,
            Tag::new(0x40, 0x9096),
            vec![mapping("Bq/ml", None)],
        );
        item_with_items(&mut obj, Tag::new(0x5200, 0x9230), vec![first, next]);
        obj.with_meta(
            FileMetaBuilder::new()
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
    let directory = tempfile::tempdir().expect("create the PET study directory");
    let path = directory.path().join("dynamic-gated.dcm");
    tracing::info_span!("arrange").in_scope(|| {
        write_dynamic_gated(
            &path,
            Tag::new(0x18, 0x9118),
            Tag::new(0x20, 0x9153),
            [100.0, 350.0],
        )
    });
    let (from_file, from_bytes) = tracing::info_span!("act").in_scope(|| {
        let bytes = std::fs::read(&path).expect("read the Part 10 fixture");
        (
            crate::load_enhanced_pet_volumes(&path).expect("load the concatenated volume set"),
            crate::load_enhanced_pet_volumes_from_bytes(&path, &bytes, &DicomReadBudget::DEFAULT)
                .expect("load the same volume set from bytes"),
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
        let suv = volume.suv_bw.as_ref().expect("retain body-weight SUV");
        let measured = [0, 1].map(|voxel| {
            suv.at(voxel)
                .expect("read the calibrated SUV voxel")
                .in_unit::<GramPerCubicCentimeter>()
        });
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
    let bytes = std::fs::read(&path).expect("read the budget-test fixture");
    let limit = DicomReadBudget::try_new(DicomReadBudget::DEFAULT.parser(), bytes.len(), 63)
        .expect("build the decoded-byte budget");
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
    let directory = tempfile::tempdir().expect("create the gated study directory");
    let path = directory.path().join("respiratory.dcm");
    tracing::info_span!("arrange").in_scope(|| {
        write_dynamic_gated(
            &path,
            Tag::new(0x20, 0x9253),
            Tag::new(0x20, 0x9245),
            [20.0, 70.0],
        );
    });
    let volumes = tracing::info_span!("act")
        .in_scope(|| crate::load_enhanced_pet_volumes(&path).expect("load the gated PET volumes"));
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
    let directory = tempfile::tempdir().expect("create the concatenation directory");
    let standalone = directory.path().join("s-standalone.dcm");
    let [first, second] = tracing::info_span!("arrange").in_scope(|| {
        let obj = dynamic_gated(
            Tag::new(0x18, 0x9118),
            Tag::new(0x20, 0x9153),
            [100.0, 350.0],
        );
        write_pet_instance(&obj, &standalone);
        write_dynamic_gated_pet_concatenation(directory.path())
    });
    let (expected, reconstructed) = tracing::info_span!("act").in_scope(|| {
        (
            crate::load_enhanced_pet_volumes(&standalone)
                .expect("load the standalone reference volume"),
            crate::load_enhanced_pet_volumes(&second)
                .expect("load the volume set from the second member"),
        )
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(reconstructed, expected);
    let budget = crate::DicomReadBudget::DEFAULT;
    let members = crate::scan_dicom_path(&first)
        .expect("scan members of the concatenation")
        .metadata
        .slices;
    assert_eq!(members.len(), 2);
    let original = members
        .iter()
        .map(|member| {
            member
                .part10_bytes
                .as_ref()
                .expect("retain the scanned native Part 10 bytes")
                .len()
        })
        .sum();
    let insufficient_decode = crate::DicomReadBudget::try_new(budget.parser(), original, 63)
        .expect("construct the decoded output budget");
    let insufficient_input = crate::DicomReadBudget::try_new(budget.parser(), original - 1, 64)
        .expect("construct the cumulative input budget");
    for (budget, message) in [
        (insufficient_decode, "decoded"),
        (insufficient_input, "retained-byte"),
    ] {
        let error = crate::load_enhanced_pet_volumes_from_instances(&members, &budget)
            .expect_err("reject the over-budget concatenation");
        assert!(format!("{error:#}").contains(message));
    }

    let mut part =
        (*DicomFile::open(&second).expect("open the second native DICOM member")).clone();
    part.put(DataElement::new(
        Tag::new(0x20, 0x9228),
        VR::UL,
        PrimitiveValue::from(3_u32),
    ));
    write_pet_instance(&part, &second);
    let error =
        crate::load_enhanced_pet_volumes(&first).expect_err("reject the mixed concatenation");
    assert!(format!("{error:#}").contains("overlap"));
    part.put(DataElement::new(
        Tag::new(0x20, 0x9228),
        VR::UL,
        PrimitiveValue::from(4_u32),
    ));
    write_pet_instance(&part, &second);
    std::fs::remove_file(&first).expect("remove the member after scanning");
    // The retained scanner snapshot remains decodable after source deletion.
    assert_eq!(
        crate::load_enhanced_pet_volumes_from_instances(&members, &budget)
            .expect("decode the retained input bytes"),
        expected
    );
    let error = crate::load_enhanced_pet_volumes(&second)
        .expect_err("reject a concatenation with a missing member");
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
    let directory = tempfile::tempdir().expect("create the spatial-gap study directory");
    let selected = tracing::info_span!("arrange").in_scope(|| {
        let [first, second] = write_dynamic_gated_pet_concatenation(directory.path());
        let template =
            (*DicomFile::open(&second).expect("open the second native DICOM member")).clone();
        let frame = template
            .element(Tag::new(0x5200, 0x9230))
            .expect("find the original frame sequence")
            .items()
            .expect("read the frame sequence items")
            .last()
            .expect("inspect the original final frame");
        for (index, path) in [&first, &second].into_iter().enumerate() {
            let mut member =
                (*DicomFile::open(path).expect("open the native DICOM member")).clone();
            member.put(DataElement::new(
                Tag::new(0x28, 8),
                VR::IS,
                PrimitiveValue::from("2"),
            ));
            member.put(DataElement::new(
                Tag::new(0x20, 0x9228),
                VR::UL,
                PrimitiveValue::from(
                    u32::try_from(2 * index).expect("the frame offset fits in u32"),
                ),
            ));
            let positions = if index == 0 { [4, 7] } else { [13, 16] };
            let frames = positions
                .into_iter()
                .enumerate()
                .map(|(local, position)| {
                    let mut frame = frame.clone();
                    let mut content = frame
                        .element(Tag::new(0x20, 0x9111))
                        .expect("find the frame content sequence")
                        .items()
                        .expect("read the frame content items")[0]
                        .clone();
                    let ordinal = u32::try_from(index * 2 + local + 1)
                        .expect("the spatial ordinal fits in u32");
                    content.put(DataElement::new(
                        Tag::new(0x20, 0x9157),
                        VR::UL,
                        PrimitiveValue::U32(vec![1, 1, 1, ordinal].into()),
                    ));
                    content.put(DataElement::new(
                        Tag::new(0x20, 0x9057),
                        VR::UL,
                        PrimitiveValue::from(ordinal),
                    ));
                    item_with_items(&mut frame, Tag::new(0x20, 0x9111), vec![content]);
                    let plane = Dataset::from_elements([DataElement::new(
                        Tag::new(0x20, 0x0032),
                        VR::DS,
                        PrimitiveValue::from(format!("0\\0\\{position}")),
                    )]);
                    item_with_items(&mut frame, Tag::new(0x20, 0x9113), vec![plane]);
                    frame
                })
                .collect();
            item_with_items(&mut member, Tag::new(0x5200, 0x9230), frames);
            let start =
                u16::try_from(2 * index + 1).expect("the synthetic input sample fits in u16");
            member.put(DataElement::new(
                Tag::new(0x7fe0, 0x0010),
                VR::OW,
                PrimitiveValue::U16(vec![start, start + 1].into()),
            ));
            write_pet_instance(&member, path);
        }
        second
    });
    let reconstructions = tracing::info_span!("act").in_scope(|| {
        crate::load_enhanced_pet_volumes(selected).expect("load the reconstructed spatial volume")
    });
    let _assert = tracing::info_span!("assert").entered();
    let [volume] = reconstructions.as_slice() else {
        panic!("expected a single temporal gate");
    };
    assert_eq!(volume.shape, [5, 1, 1]);
    assert_eq!(volume.origin.to_array(), [0.0, 0.0, 4.0]);
    assert_eq!(volume.spacing.to_array(), [3.0, 2.0, 2.0]);
    assert_eq!(volume.data.as_slice(), &[1.0, 2.0, 2.5, 3.0, 4.0]);
    use crate::SliceProvenance::{Acquired, Interpolated};
    assert_eq!(
        volume.slice_provenance.as_deref(),
        Some([Acquired, Acquired, Interpolated, Acquired, Acquired].as_slice())
    );
    let suv = volume
        .suv_bw
        .as_ref()
        .expect("retain SUV after spatial reconstruction");
    assert_eq!(
        [0, 1, 2, 3, 4].map(|index| suv
            .at(index)
            .expect("read the reconstructed SUV voxel")
            .in_unit::<GramPerCubicCentimeter>()),
        [0.5, 1.0, 1.25, 1.5, 2.0]
    );
}
