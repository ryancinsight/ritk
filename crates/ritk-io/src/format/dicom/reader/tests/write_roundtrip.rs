#![allow(unused_imports)]

use super::super::geometry::{
    analyze_slice_spacing, dot_3d, normalize_3d, resample_frames_linear, slice_normal_from_iop,
};
use super::super::loader::{
    load_dicom_series, load_dicom_series_with_metadata, load_from_series, read_dicom_series,
    read_dicom_series_with_metadata,
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
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::TransferSyntaxKind;
#[test]
fn test_write_series_load_series_intensity_roundtrip() {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    type B = burn_ndarray::NdArray<f32>;

    let tmp = tempfile::tempdir().expect("tempdir");
    let series_path = tmp.path().join("e2e_roundtrip_series");

    // 4 slices × 4 rows × 4 cols = 64 voxels.
    // Intensities 0..=63, row-major order.
    let (depth, rows, cols) = (4usize, 4usize, 4usize);
    let original_data: Vec<f32> = (0..(depth * rows * cols)).map(|i| i as f32).collect();
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(original_data.clone(), Shape::new([depth, rows, cols])),
        &device,
    );
    let image = Image::<B, 3>::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::format::dicom::writer::write_dicom_series(&series_path, &image)
        .expect("write_dicom_series must succeed");

    let loaded_image =
        load_dicom_series::<B, _>(&series_path, &device).expect("load_dicom_series must succeed");

    let loaded_td = loaded_image.data().clone().into_data();
    let loaded_vals: &[f32] = loaded_td
        .as_slice::<f32>()
        .expect("loaded image must contain f32");

    assert_eq!(
        loaded_vals.len(),
        original_data.len(),
        "loaded voxel count must equal original"
    );

    // Analytical bound per slice: slope = range / 65535 = 15 / 65535 ≈ 2.29e-4.
    // DS format {:.6} stores the slope/intercept with at most 0.5e-6 rounding error
    // per coefficient. Accumulated slope error over max u16 (65535):
    //   65535 * 0.5e-6 ≈ 0.033.
    // Quantization from round(): slope / 2 ≈ 1.14e-4.
    // Total analytical bound: 65535 * 0.5e-6 + 0.5e-6 + slope / 2.
    let slice_range = 15.0f32;
    let slope = slice_range / 65535.0_f32;
    let ds_half_ulp = 0.5e-6_f32;
    let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;

    // The writer writes per-slice rescale; reader applies per-slice rescale.
    // Re-sort loaded voxels by z-position. The series may be loaded in sorted order.
    for (idx, (&orig, &loaded)) in original_data.iter().zip(loaded_vals.iter()).enumerate() {
        let err = (loaded - orig).abs();
        assert!(
            err <= tol,
            "voxel[{idx}]: |{loaded} - {orig}| = {err} > tol {tol}; slope={slope}"
        );
    }
}

#[test]
fn test_write_metadata_series_load_series_intensity_roundtrip() {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = burn_ndarray::NdArray<f32>;

    let tmp = tempfile::tempdir().expect("tempdir");
    let series_path = tmp.path().join("e2e_meta_roundtrip");

    let (depth, rows, cols) = (3usize, 4usize, 4usize);
    // Each slice: values starting at z*16, range = 15.
    let original_data: Vec<f32> = (0..(depth * rows * cols))
        .map(|i| {
            let slice_idx = i / (rows * cols);
            let intra_idx = i % (rows * cols);
            (slice_idx * 16 + intra_idx) as f32
        })
        .collect();
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(original_data.clone(), Shape::new([depth, rows, cols])),
        &device,
    );
    let image = Image::<B, 3>::new(
        tensor,
        Point::new([5.0, 10.0, -20.0]),
        Spacing::new([1.5, 0.5, 0.5]),
        Direction::identity(),
    );

    let meta = DicomReadMetadata {
        series_instance_uid: Some("1.2.3.4.5.6.999".to_string()),
        study_instance_uid: Some("1.2.3.4.5.6.998".to_string()),
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some("CT".to_string()),
        patient_id: Some("E2E001".to_string()),
        patient_name: Some("E2E^Patient".to_string()),
        study_date: Some("20250101".to_string()),
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.5, 0.5, 0.5],
        origin: [5.0, 10.0, -20.0],
        // RITK axial: N̂=[0,0,1], F_c=[0,1,0], F_r=[1,0,0]
        direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some("MONOCHROME2".to_string()),
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
    .expect("write_dicom_series_with_metadata must succeed");

    let (loaded_image, loaded_meta) =
        load_dicom_series_with_metadata::<B, _>(&series_path, &device)
            .expect("load_dicom_series_with_metadata must succeed");

    // --- Intensity round-trip ---
    let loaded_td = loaded_image.data().clone().into_data();
    let loaded_vals: &[f32] = loaded_td.as_slice::<f32>().expect("loaded must be f32");

    assert_eq!(
        loaded_vals.len(),
        original_data.len(),
        "voxel count must match"
    );

    // Analytical slope per slice: each slice has range=15, slope = 15/65535.
    // DS format {:.6} stores the slope/intercept with at most 0.5e-6 rounding error
    // per coefficient. Accumulated slope error over max u16 (65535):
    //   65535 * 0.5e-6 ≈ 0.033.
    // Quantization from round(): slope / 2 ≈ 1.14e-4.
    // Total analytical bound: 65535 * 0.5e-6 + 0.5e-6 + slope / 2.
    let slice_range = 15.0f32;
    let slope = slice_range / 65535.0_f32;
    let ds_half_ulp = 0.5e-6_f32;
    let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;

    for (idx, (&orig, &loaded)) in original_data.iter().zip(loaded_vals.iter()).enumerate() {
        let err = (loaded - orig).abs();
        assert!(
            err <= tol,
            "voxel[{idx}]: |{loaded} - {orig}| = {err} > tol {tol}"
        );
    }

    // --- Spatial metadata round-trip ---
    let pos_tol = 1e-4_f64;
    assert!(
        (loaded_meta.origin[0] - 5.0).abs() < pos_tol,
        "origin[0] must be 5.0; got {}",
        loaded_meta.origin[0]
    );
    assert!(
        (loaded_meta.origin[1] - 10.0).abs() < pos_tol,
        "origin[1] must be 10.0; got {}",
        loaded_meta.origin[1]
    );
    assert!(
        (loaded_meta.origin[2] - (-20.0_f64)).abs() < pos_tol,
        "origin[2] must be -20.0; got {}",
        loaded_meta.origin[2]
    );
    assert!(
        (loaded_meta.spacing[0] - 1.5).abs() < pos_tol,
        "spacing[0] (Δz) must be 1.5; got {}",
        loaded_meta.spacing[0]
    );
    assert!(
        (loaded_meta.spacing[1] - 0.5).abs() < pos_tol,
        "spacing[1] (ΔRow) must be 0.5; got {}",
        loaded_meta.spacing[1]
    );
    assert!(
        (loaded_meta.spacing[2] - 0.5).abs() < pos_tol,
        "spacing[2] (ΔCol) must be 0.5; got {}",
        loaded_meta.spacing[2]
    );
}
