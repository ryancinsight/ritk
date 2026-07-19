#![allow(unused_imports)]

use arrayvec::ArrayString;

use super::super::geometry::{
    analyze_slice_spacing, dot, normalize, resample_frames_linear, slice_normal_from_iop,
};
use super::super::loader::{
    load_dicom_series_with_metadata, load_from_series, read_dicom_series_with_metadata,
};
use super::super::pixel::{decode_pixel_bytes, read_slice_pixels};
use super::super::scan::scan_dicom_directory;
use super::super::types::{
    DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata, PatientPosition,
};
use super::super::utils::is_likely_dicom_file;
use super::support::*;
use crate::format::dicom::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomTag, DicomValue,
};
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_spatial::{Direction, Point, Spacing};
#[test]
fn test_scan_preserves_private_text_and_bytes_through_write_read_cycle() {
    use ritk_image::tensor::Tensor;
    use std::collections::HashMap;
    type B = coeus_core::SequentialBackend;

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("priv_rt");

    // Build a 1-slice 4×4 image.
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([1_usize, 4, 4], &[42.0_f32; 4 * 4], &device);
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");

    // Preservation: private text tag + raw OB bytes tag.
    let mut preservation = DicomPreservationSet::new();
    preservation.object.insert(DicomObjectNode::text(
        DicomTag::new(0x0009, 0x0010),
        "LO",
        "PRIV_ROUND_TRIP_VALUE",
    ));
    preservation.preserve(DicomPreservedElement::new(
        DicomTag::new(0x0019, 0x1001),
        Some(ArrayString::<2>::try_from("OB").unwrap_or_default()),
        vec![0xAB_u8, 0xCD, 0xEF, 0x01],
    ));

    let meta = DicomReadMetadata {
        series_instance_uid: Some("2.25.111".try_into().unwrap()),
        study_instance_uid: Some("2.25.222".try_into().unwrap()),
        frame_of_reference_uid: None,
        series_description: Some("TestSeries".to_string()),
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: Some("P001".to_string()),
        patient_name: Some("Test^Patient".to_string()),
        study_date: Some(ArrayString::from("20240101").unwrap()),
        series_date: Some(ArrayString::from("20240101").unwrap()),
        series_time: Some(ArrayString::from("120000").unwrap()),
        dimensions: [4, 4, 1],
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
        slices: Vec::new(),
        private_tags: HashMap::new(),
        preservation,
        patient_weight_kg: None,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    };

    crate::format::dicom::writer::write_dicom_series_with_metadata(&dir, &image, Some(&meta))
        .expect("write_dicom_series_with_metadata");

    // Scan back via scan_dicom_directory.
    let scanned = scan_dicom_directory(&dir).expect("scan_dicom_directory");
    assert_eq!(
        scanned.num_slices, 1,
        "must have exactly 1 slice; got {}",
        scanned.num_slices
    );

    let slice = &scanned.metadata.slices[0];
    let priv_text_tag = DicomTag::new(0x0009, 0x0010);
    let priv_bytes_tag = DicomTag::new(0x0019, 0x1001);

    // Private text tag must be present in preservation.object with the correct value.
    let text_node = slice.preservation.object.get(priv_text_tag);
    assert!(
        text_node.is_some(),
        "private text tag (0009,0010) must be present in preservation.object"
    );
    let text_val = text_node.unwrap().value.as_text();
    assert!(
        text_val
            .map(|s| s.trim() == "PRIV_ROUND_TRIP_VALUE")
            .unwrap_or(false),
        "private text value must survive round-trip: got {:?}",
        text_val
    );

    // Private bytes tag must be present in preservation.preserved with the correct bytes.
    let bytes_elem = slice
        .preservation
        .preserved
        .iter()
        .find(|e| e.tag == priv_bytes_tag);
    assert!(
        bytes_elem.is_some(),
        "private bytes tag (0019,1001) must be present in preservation.preserved"
    );
    assert_eq!(
        bytes_elem.unwrap().bytes,
        vec![0xAB_u8, 0xCD, 0xEF, 0x01],
        "raw OB bytes must survive round-trip"
    );
}
