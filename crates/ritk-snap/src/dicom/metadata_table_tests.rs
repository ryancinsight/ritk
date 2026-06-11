use super::*;
use arrayvec::ArrayString;
use ritk_io::PixelSignedness;
use ritk_io::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomReadMetadata,
    DicomSliceMetadata, DicomTag, PatientPosition,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn metadata_fixture() -> DicomReadMetadata {
    let mut private_tags = HashMap::new();
    private_tags.insert("0019,10AA".to_string(), "PRIVATE_SERIES".to_string());

    let mut preservation = DicomPreservationSet::new();
    preservation.object.insert(DicomObjectNode::text(
        DicomTag::new(0x0009, 0x1000),
        "LO",
        "SERIES_NODE",
    ));
    preservation.preserve(DicomPreservedElement::new(
        DicomTag::new(0x0019, 0x1001),
        Some(ArrayString::<2>::try_from("OB").unwrap_or_default()),
        vec![1, 2, 3, 4],
    ));

    DicomReadMetadata {
        series_instance_uid: Some("1.2.840.series".try_into().unwrap()),
        study_instance_uid: Some("1.2.840.study".try_into().unwrap()),
        frame_of_reference_uid: Some("1.2.840.frame".try_into().unwrap()),
        series_description: Some("Axial CT".to_string()),
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: Some("P001".to_string()),
        patient_name: Some("DOE^JANE".to_string()),
        study_date: Some(ArrayString::from("20260101").unwrap()),
        series_date: Some(ArrayString::from("20260102").unwrap()),
        series_time: Some(ArrayString::from("120000").unwrap()),
        dimensions: [3, 4, 5],
        spacing: [0.7, 0.8, 1.5],
        origin: [10.0, 20.0, -30.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(12),
        high_bit: Some(11),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
        slices: vec![DicomSliceMetadata {
            path: PathBuf::from("slice001.dcm"),
            sop_instance_uid: Some("1.2.840.slice".try_into().unwrap()),
            instance_number: Some(7),
            slice_location: Some(42.25),
            image_position_patient: Some([1.0, 2.0, 3.0]),
            image_orientation_patient: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            patient_position: Some(PatientPosition::HeadFirstSupine),
            pixel_spacing: Some([0.7, 0.8]),
            slice_thickness: Some(1.5),
            rescale_slope: 2.0,
            rescale_intercept: -1024.0,
            sop_class_uid: Some("1.2.840.sop".try_into().unwrap()),
            transfer_syntax_uid: Some("1.2.840.10008.1.2.1".try_into().unwrap()),
            pixel_representation: PixelSignedness::Signed,
            bits_allocated: 16,
            window_center: Some(40.0),
            window_width: Some(400.0),
            gantry_tilt: Some(0.5),
            ..Default::default()
        }],
        private_tags,
        preservation,
        patient_weight_kg: None,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    }
}

#[test]
fn metadata_rows_include_series_slice_private_and_preserved_values() {
    let rows = build_metadata_rows(&metadata_fixture());

    assert!(
        rows.iter().any(|row| {
            row.scope == MetadataScope::Series
                && row.tag == "0020,000E"
                && row.keyword == "SeriesInstanceUID"
                && row.value == "1.2.840.series"
        }),
        "series UID row must be present"
    );
    assert!(
        rows.iter().any(|row| {
            row.scope == MetadataScope::FirstSlice
                && row.tag == "0020,0032"
                && row.keyword == "ImagePositionPatient"
                && row.value == "1.000000 x 2.000000 x 3.000000"
        }),
        "first-slice image position row must be present"
    );
    assert!(
        rows.iter().any(|row| {
            row.scope == MetadataScope::FirstSlice
                && row.tag == "0018,5100"
                && row.keyword == "PatientPosition"
                && row.value == "HFS (Head First Supine)"
        }),
        "first-slice patient-position row must be present"
    );
    assert!(
        rows.iter().any(|row| {
            row.scope == MetadataScope::PrivateTag
                && row.tag == "0019,10AA"
                && row.value == "PRIVATE_SERIES"
        }),
        "private tag row must be present"
    );
    assert!(
        rows.iter().any(|row| {
            row.scope == MetadataScope::PreservedRaw
                && row.tag == "0019,1001"
                && row.vr == "OB"
                && row.value == "4 bytes"
        }),
        "raw preserved element row must expose byte length"
    );
}

#[test]
fn metadata_scope_labels_are_stable() {
    assert_eq!(MetadataScope::Series.label(), "Series");
    assert_eq!(MetadataScope::FirstSlice.label(), "Slice[0]");
    assert_eq!(MetadataScope::PreservedNode.label(), "Preserved");
    assert_eq!(MetadataScope::PreservedRaw.label(), "Raw");
    assert_eq!(MetadataScope::PrivateTag.label(), "Private");
}
