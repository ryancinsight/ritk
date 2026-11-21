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

fn write_oblique_multiframe(path: &std::path::Path) {
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
        .map(|position| {
            let mut plane = Dataset::new();
            plane.put_value(
                DicomTag::new(0x0020, 0x0032),
                VR::DS,
                format!(r"{position}\-23\41"),
            );
            let mut frame = Dataset::new();
            sequence(&mut frame, DicomTag::new(0x0020, 0x9113), vec![plane]);
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
        write_oblique_multiframe(&fixture.path().join("oblique-multiframe.dcm"));
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
