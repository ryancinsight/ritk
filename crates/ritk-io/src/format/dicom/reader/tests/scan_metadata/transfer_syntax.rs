//! Verify that Part 10 transfer syntax survives series write and scan.

use super::super::super::scan::scan_dicom_directory;
use super::super::super::types::DicomReadMetadata;
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use arrayvec::ArrayString;

#[test]
fn test_scan_metadata_round_trip_transfer_syntax() {
    use ritk_core::image::Image;
    use ritk_image::tensor::{Shape, Tensor};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = coeus_core::SequentialBackend;

    let temp = tempfile::tempdir().expect("temporary DICOM series");
    let series_path = temp.path().join("ts_series");

    let (depth, rows, cols) = (2usize, 4usize, 4usize);
    let data = vec![1000.0f32; depth * rows * cols];
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([depth, rows, cols], &(data), &device)
        .expect("test tensor is valid");
    let image = Image::<f32, B, 3>::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");

    let meta = DicomReadMetadata {
        series_instance_uid: None,
        study_instance_uid: None,
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some(ArrayString::from("OT").expect("modality fits its VR")),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.0, 1.0, 1.0].map(
            aequitas::systems::si::quantities::Length::from_unit::<
                aequitas::systems::si::units::Millimeter,
            >,
        ),
        origin: [0.0, 0.0, 0.0].map(
            aequitas::systems::si::quantities::Length::from_unit::<
                aequitas::systems::si::units::Millimeter,
            >,
        ),
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(
            ArrayString::from("MONOCHROME2").expect("photometric interpretation fits its VR"),
        ),
        slices: Vec::new(),
        private_tags: HashMap::new(),
        preservation: crate::format::dicom::DicomPreservationSet::new(),
        patient_weight_kg: None,
        enhanced_pet_suv: None,
        enhanced_pet_selection: None,
        classic_pet_calibration: None,
        pet_pixels_in_bq_per_ml: false,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    };

    crate::format::dicom::writer::write_dicom_series_with_metadata(
        &series_path,
        &image,
        Some(&meta),
    )
    .expect("write must not fail");

    let info = scan_dicom_directory(&series_path).expect("scan must not fail");

    // The writer emits transfer_syntax EXPLICIT_VR_LE = Explicit VR Little Endian.
    // The reader must extract this from the file meta, not from DicomTag::new(0x0008,0x0070).
    assert!(
        !info.metadata.slices.is_empty(),
        "at least one slice must be returned"
    );
    for (i, slice) in info.metadata.slices.iter().enumerate() {
        assert_eq!(
            slice.transfer_syntax_uid.as_deref(),
            Some(EXPLICIT_VR_LE),
            "slice {i} transfer_syntax_uid must be {EXPLICIT_VR_LE}; got {:?}",
            slice.transfer_syntax_uid
        );
    }
}
