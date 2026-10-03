#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::{load_dicom_series, load_native_dicom_series};
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};
use std::collections::HashMap;

#[test]
fn native_series_loader_matches_legacy_loader() {
    type B = coeus_core::SequentialBackend;

    let dir = tempfile::tempdir().expect("tempdir");
    let series_path = dir.path().join("series_native_parity");

    let (depth, rows, cols) = (3usize, 3usize, 4usize);
    let values: Vec<f32> = (0..(depth * rows * cols))
        .map(|i| i as f32 * 0.25 + 2.0)
        .collect();
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([depth, rows, cols], &(values), &device);
    let image = Image::<f32, B, 3>::new(
        tensor,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([1.5, 0.75, 0.5]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");

    let meta = crate::format::dicom::DicomReadMetadata {
        series_instance_uid: Some("2.25.71001".try_into().unwrap()),
        study_instance_uid: Some("2.25.71002".try_into().unwrap()),
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some("CT".try_into().unwrap()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.5, 0.75, 0.5],
        origin: [1.0, 2.0, 3.0],
        direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some("MONOCHROME2".try_into().unwrap()),
        slices: Vec::new(),
        private_tags: HashMap::new(),
        preservation: crate::format::dicom::DicomPreservationSet::new(),
        patient_weight_kg: None,
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
    .expect("write_dicom_series_with_metadata");
    let series = crate::format::dicom::scan_dicom_directory(&series_path)
        .expect("scan series")
        .pop()
        .expect("one series");

    let legacy = load_dicom_series::<B>(&series, &device).expect("legacy load");
    let native = load_native_dicom_series(&series, &SequentialBackend).expect("native series load");

    assert_eq!(native.shape(), legacy.shape());
    let legacy_values = legacy
        .data_slice()
        .expect("legacy series data must be contiguous");
    assert_eq!(
        native.data_slice().expect("native contiguous data"),
        legacy_values,
        "native series facade must use the same decoded voxels"
    );
    assert_eq!(native.origin().to_array(), legacy.origin().to_array());
    assert_eq!(native.spacing().to_array(), legacy.spacing().to_array());
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(
                native.direction()[(row, col)],
                legacy.direction()[(row, col)],
                "direction[{row},{col}]"
            );
        }
    }
}
