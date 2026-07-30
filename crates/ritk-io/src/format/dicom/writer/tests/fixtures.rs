use crate::format::dicom::reader::DicomReadMetadata;
use arrayvec::ArrayString;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};
use std::collections::HashMap;

pub(super) type Backend = coeus_core::SequentialBackend;

pub(super) fn make_image(
    depth: usize,
    rows: usize,
    cols: usize,
    fill: f32,
) -> Image<f32, Backend, 3> {
    let device = Default::default();
    let data = vec![fill; depth * rows * cols];
    let tensor = Tensor::<f32, Backend>::from_slice_on([depth, rows, cols], &(data), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

pub(super) fn make_image_with_spatial(
    depth: usize,
    rows: usize,
    cols: usize,
    fill: f32,
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, Backend, 3> {
    let device = Default::default();
    let data = vec![fill; depth * rows * cols];
    let tensor = Tensor::<f32, Backend>::from_slice_on([depth, rows, cols], &(data), &device);
    Image::new(
        tensor,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

pub(super) fn make_test_metadata() -> DicomReadMetadata {
    let mut private_tags = HashMap::new();
    private_tags.insert("0019,10AA".to_string(), "PRIVATE_SERIES_VALUE".to_string());
    private_tags.insert(
        "0029,10BB".to_string(),
        "PRIVATE_SERIES_VALUE_2".to_string(),
    );

    DicomReadMetadata {
        series_instance_uid: Some("1.2.3.4.5.6.789".try_into().unwrap()),
        study_instance_uid: Some("1.2.3.4.5.6.100".try_into().unwrap()),
        frame_of_reference_uid: Some("1.2.3.4.5.6.200".try_into().unwrap()),
        series_description: Some("Test Series".to_string()),
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: Some("PAT001".to_string()),
        patient_name: Some("Test^Patient".to_string()),
        study_date: Some(ArrayString::from("20240101").unwrap()),
        series_date: Some(ArrayString::from("20240102").unwrap()),
        series_time: Some(ArrayString::from("123456").unwrap()),
        dimensions: [4, 4, 3],
        // Axial convention: spacing=[Δz,ΔRow,ΔCol], direction cols=[NÌ‚, F_c, F_r]
        // NÌ‚=[0,0,1], F_c=[0,1,0], F_r=[1,0,0]
        spacing: [2.5, 0.5, 0.5],
        origin: [10.0, 20.0, 30.0],
        direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
        slices: Vec::new(),
        private_tags,
        preservation: crate::format::dicom::DicomPreservationSet::new(),
        patient_weight_kg: None,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    }
}
