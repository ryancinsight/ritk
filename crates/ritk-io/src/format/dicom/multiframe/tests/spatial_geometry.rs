use super::*;
use crate::format::dicom::reader::geometry::SliceProvenance;
use ritk_dicom::dataset::{Length, Sequence};

fn sequence(dataset: &mut Dataset, tag: DicomTag, items: Vec<Dataset>) {
    dataset.put(DataElement::new(
        tag,
        VR::SQ,
        Sequence::new(items, Length::Undefined),
    ));
}

fn write_oblique_multiframe(path: &std::path::Path, units: Option<[&str; 4]>) {
    let mut dataset = Dataset::new();
    dataset.put_value(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        MF_GRAYSCALE_WORD_SC_UID,
    );
    dataset.put_value(DicomTag::new(0x0008, 0x0018), VR::UI, "2.25.5730");
    dataset.put_value(DicomTag::new(0x0028, 0x0008), VR::IS, "4");
    dataset.put_value(DicomTag::new(0x0028, 0x0010), VR::US, 1_u16);
    dataset.put_value(DicomTag::new(0x0028, 0x0011), VR::US, 1_u16);
    dataset.put_value(DicomTag::new(0x0028, 0x0100), VR::US, 16_u16);
    dataset.put_value(DicomTag::new(0x0028, 0x0101), VR::US, 16_u16);
    dataset.put_value(DicomTag::new(0x0028, 0x0102), VR::US, 15_u16);
    dataset.put_value(DicomTag::new(0x0028, 0x0103), VR::US, 0_u16);
    dataset.put_value(DicomTag::new(0x0020, 0x0032), VR::DS, r"12\invalid\4\7");

    let mut pixel_measures = Dataset::new();
    pixel_measures.put_value(DicomTag::new(0x0028, 0x0030), VR::DS, r"3\5");
    pixel_measures.put_value(DicomTag::new(0x0018, 0x0050), VR::DS, "9");
    let mut orientation = Dataset::new();
    orientation.put_value(DicomTag::new(0x0020, 0x0037), VR::DS, r"0\1\0\0\0\1");
    let mut shared = Dataset::new();
    sequence(
        &mut shared,
        DicomTag::new(0x0028, 0x9110),
        vec![pixel_measures],
    );
    sequence(
        &mut shared,
        DicomTag::new(0x0020, 0x9116),
        vec![orientation],
    );
    sequence(&mut dataset, DicomTag::new(0x5200, 0x9229), vec![shared]);

    let per_frame = [17, 21, 29, 33]
        .into_iter()
        .enumerate()
        .map(|(index, position)| {
            let mut plane = Dataset::new();
            plane.put_value(
                DicomTag::new(0x0020, 0x0032),
                VR::DS,
                format!(r"{position}\-23\41"),
            );
            let mut frame = Dataset::new();
            sequence(&mut frame, DicomTag::new(0x0020, 0x9113), vec![plane]);
            if let Some(units) = units {
                let mut transform = Dataset::new();
                transform.put_value(DicomTag::new(0x0028, 0x1054), VR::LO, units[index]);
                transform.put_value(DicomTag::new(0x0028, 0x1052), VR::DS, "0");
                transform.put_value(
                    DicomTag::new(0x0028, 0x1053),
                    VR::DS,
                    if index == 1 { "2" } else { "1" },
                );
                sequence(&mut frame, DicomTag::new(0x0028, 0x9145), vec![transform]);
            }
            frame
        })
        .collect();
    sequence(&mut dataset, DicomTag::new(0x5200, 0x9230), per_frame);
    let samples = [100_u16, 200, 400, 500]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    dataset.put_value(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::from(samples),
    );
    dataset
        .with_meta(
            FileMetaBuilder::new()
                .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                .media_storage_sop_instance_uid("2.25.5730")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("build multiframe Part 10")
        .write_to_file(path)
        .expect("write multiframe Part 10");
}

#[test]
fn oblique_multiframe_preserves_patient_origin_and_interpolated_plane() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let fixture = tracing::info_span!("arrange").in_scope(|| {
        let fixture = tempfile::tempdir().expect("temporary fixture directory");
        write_oblique_multiframe(&fixture.path().join("oblique-multiframe.dcm"), None);
        fixture
    });
    let (volume, info) = tracing::info_span!("act").in_scope(|| {
        let path = fixture.path().join("oblique-multiframe.dcm");
        let volume = load_dicom_multiframe_flat(&path).expect("decode partial stack");
        let info = read_multiframe_info(&path).expect("read typed DICOM lengths");
        (volume, info)
    });
    tracing::info_span!("assert").in_scope(|| {
        // The image-plane cross product is +X and the median gap is 4 mm.
        // The point at x = 25 mm is halfway between the 200 and 400 samples.
        assert_eq!(volume.shape, [5, 1, 1]);
        assert_eq!(volume.origin.to_array(), [17.0, -23.0, 41.0]);
        assert_eq!(volume.spacing.to_array(), [4.0, 3.0, 5.0]);
        assert_eq!(volume.data, [100.0, 200.0, 300.0, 400.0, 500.0]);
        assert_eq!(
            volume
                .slice_provenance
                .as_deref()
                .expect("resampled plane sources")[2],
            SliceProvenance::Interpolated
        );
        assert!(
            info.image_position.is_none(),
            "malformed tuple cannot shift patient axes"
        );
    });
}

#[test]
fn oblique_multiframe_requires_compatible_calibrated_units_for_interpolation() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    // DICOM Rescale Type permits vendor-defined LO labels.
    let unit_a = "VENDOR_ACTIVITY_CONCENTRATION_SCALE";
    let unit_b = "VENDOR_ATTENUATION_REFERENCE_SCALE";
    let (fixture, matching, mixed, missing, matching_labels, mixed_labels) =
        tracing::info_span!("arrange").in_scope(|| {
            let fixture = tempfile::tempdir().expect("temporary fixture directory");
            let matching = fixture.path().join("matching.dcm");
            let mixed = fixture.path().join("mixed.dcm");
            let missing = fixture.path().join("missing.dcm");
            write_oblique_multiframe(&matching, Some(["HU"; 4]));
            write_oblique_multiframe(&mixed, Some(["HU", "HU", "MGML", "HU"]));
            write_oblique_multiframe(&missing, Some(["HU", "HU", "", "HU"]));
            let matching_labels = fixture.path().join("matching-labels.dcm");
            let mixed_labels = fixture.path().join("mixed-labels.dcm");
            write_oblique_multiframe(&matching_labels, Some([unit_a; 4]));
            write_oblique_multiframe(&mixed_labels, Some([unit_a, unit_a, unit_b, unit_a]));
            (
                fixture,
                matching,
                mixed,
                missing,
                matching_labels,
                mixed_labels,
            )
        });
    let (matching, mixed, missing, matching_labels, mixed_labels) = tracing::info_span!("act")
        .in_scope(|| {
            (
                load_dicom_multiframe_flat(matching),
                load_dicom_multiframe_flat(mixed),
                load_dicom_multiframe_flat(missing),
                load_dicom_multiframe_flat(matching_labels),
                load_dicom_multiframe_flat(mixed_labels),
            )
        });
    tracing::info_span!("assert").in_scope(|| {
        // Both adjacent samples rescale to 400 HU, so the midpoint is 400 HU.
        assert_eq!(
            matching.expect("common calibrated scale").data,
            [100.0, 400.0, 400.0, 400.0, 500.0]
        );
        assert_eq!(
            matching_labels.expect("common vendor scale").data,
            [100.0, 400.0, 400.0, 400.0, 500.0]
        );
        let error = mixed_labels
            .err()
            .expect("different vendor labels must stop interpolation");
        let message = format!("{error:#}");
        assert!(
            message.contains("interpolation")
                && message.contains(unit_a)
                && message.contains(unit_b),
            "{message}"
        );
        for (result, expected_unit) in [(mixed, "MassConcentration"), (missing, "Unspecified")] {
            let error = result
                .err()
                .expect("incompatible value units must stop interpolation");
            let message = format!("{error:#}");
            assert!(
                message.contains("interpolation")
                    && message.contains("Hounsfield")
                    && message.contains(expected_unit),
                "{message}"
            );
        }
    });
    drop(fixture);
}
