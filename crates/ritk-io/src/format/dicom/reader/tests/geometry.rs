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
fn test_analyze_slice_spacing_uniform() {
    // 5 slices at 0, 1, 2, 3, 4 mm
    let positions = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let report = analyze_slice_spacing(&positions);
    assert!(
        (report.nominal_spacing - 1.0).abs() < 1e-10,
        "nominal_spacing={}",
        report.nominal_spacing
    );
    assert!(
        report.max_relative_deviation < 1e-10,
        "max_relative_deviation={}",
        report.max_relative_deviation
    );
    assert!(!report.is_nonuniform);
    assert!(!report.has_missing_slices);
    assert!(report.missing_between.is_empty());
}

#[test]
fn test_analyze_slice_spacing_nonuniform() {
    // Positions: 0, 1, 2.2, 3.2, 4.2 — gap[1] = 1.2, others = 1.0, median = 1.0
    let positions = vec![0.0_f64, 1.0, 2.2, 3.2, 4.2];
    let report = analyze_slice_spacing(&positions);
    // nominal = median([1.0, 1.2, 1.0, 1.0]) = 1.0
    assert!(
        (report.nominal_spacing - 1.0).abs() < 1e-10,
        "nominal_spacing={}",
        report.nominal_spacing
    );
    // max relative deviation = |1.2 - 1.0| / 1.0 = 0.2 (20%)
    assert!(
        (report.max_relative_deviation - 0.2).abs() < 1e-10,
        "max_relative_deviation={}",
        report.max_relative_deviation
    );
    assert!(report.is_nonuniform);
    // gap[1]=1.2 < 1.5 × 1.0: no missing slices
    assert!(!report.has_missing_slices);
}

#[test]
fn test_analyze_slice_spacing_missing_slice() {
    // 4 slices at 0, 1, 3, 4 mm — gap[1] = 2.0, nominal = 1.0
    let positions = vec![0.0_f64, 1.0, 3.0, 4.0];
    let report = analyze_slice_spacing(&positions);
    // gaps: [1.0, 2.0, 1.0]; median = 1.0
    assert!(
        (report.nominal_spacing - 1.0).abs() < 1e-10,
        "nominal_spacing={}",
        report.nominal_spacing
    );
    assert!(report.has_missing_slices);
    // gap[1] = 2.0 > 1.5 × 1.0
    assert_eq!(report.missing_between, vec![1_usize]);
    // max relative deviation = 1.0 (100%) — also nonuniform
    assert!(report.is_nonuniform);
}

#[test]
fn test_resample_frames_linear_identity_on_uniform() {
    // 4 frames, 2×2 pixels each, uniform spacing 1.0 mm
    let f0 = vec![1.0_f32, 2.0, 3.0, 4.0];
    let f1 = vec![5.0_f32, 6.0, 7.0, 8.0];
    let f2 = vec![9.0_f32, 10.0, 11.0, 12.0];
    let f3 = vec![13.0_f32, 14.0, 15.0, 16.0];
    let frames = vec![f0.clone(), f1.clone(), f2.clone(), f3.clone()];
    let positions = vec![0.0_f64, 1.0, 2.0, 3.0];
    let resampled = resample_frames_linear(&frames, &positions, 1.0);
    assert_eq!(resampled.len(), 4, "frame count");
    for (i, (orig, got)) in frames.iter().zip(resampled.iter()).enumerate() {
        for (j, (&o, &g)) in orig.iter().zip(got.iter()).enumerate() {
            assert!(
                (o - g).abs() < 1e-5,
                "frame[{}] pixel[{}]: expected {}, got {}",
                i,
                j,
                o,
                g
            );
        }
    }
}

#[test]
fn test_resample_frames_linear_missing_slice() {
    // All-constant frames: src[0]=10, src[1]=20, src[2]=40, src[3]=50 (per-pixel)
    let mk = |v: f32| vec![v; 4]; // 2×2 pixels
    let frames = vec![mk(10.0), mk(20.0), mk(40.0), mk(50.0)];
    let positions = vec![0.0_f64, 1.0, 3.0, 4.0];
    let resampled = resample_frames_linear(&frames, &positions, 1.0);
    // N_target = round((4.0 - 0.0) / 1.0) + 1 = 5
    assert_eq!(resampled.len(), 5, "expected 5 output frames");
    // Frame 0 (pos=0.0) → src[0] = 10.0
    for &v in &resampled[0] {
        assert!((v - 10.0).abs() < 1e-5, "frame[0] pixel={}", v);
    }
    // Frame 1 (pos=1.0) → exactly src[1] = 20.0
    for &v in &resampled[1] {
        assert!((v - 20.0).abs() < 1e-5, "frame[1] pixel={}", v);
    }
    // Frame 2 (pos=2.0) → midpoint of src[1](pos=1.0) and src[2](pos=3.0)
    // t = (2.0 - 1.0) / (3.0 - 1.0) = 0.5 → 0.5×20 + 0.5×40 = 30.0
    for &v in &resampled[2] {
        assert!((v - 30.0).abs() < 1e-4, "frame[2] pixel={}", v);
    }
    // Frame 3 (pos=3.0) → exactly src[2] = 40.0
    for &v in &resampled[3] {
        assert!((v - 40.0).abs() < 1e-5, "frame[3] pixel={}", v);
    }
    // Frame 4 (pos=4.0) → src[3] = 50.0
    for &v in &resampled[4] {
        assert!((v - 50.0).abs() < 1e-5, "frame[4] pixel={}", v);
    }
}

#[test]
fn test_resample_frames_linear_nonuniform_interpolation() {
    let mk = |v: f32| vec![v; 1];
    // src values: 0, 10, 20, 30, 40
    let frames = vec![mk(0.0), mk(10.0), mk(20.0), mk(30.0), mk(40.0)];
    let positions = vec![0.0_f64, 1.0, 2.1, 3.1, 4.1];
    let resampled = resample_frames_linear(&frames, &positions, 1.0);
    assert_eq!(resampled.len(), 5, "5 target frames");
    // Frame 0 → exact src[0] = 0.0 (t=0, clamp)
    assert!(
        (resampled[0][0] - 0.0).abs() < 1e-5,
        "frame[0]={}",
        resampled[0][0]
    );
    // Frame 1 → exact src[1] = 10.0 (exact match at pos=1.0)
    assert!(
        (resampled[1][0] - 10.0).abs() < 1e-5,
        "frame[1]={}",
        resampled[1][0]
    );
    // Frame 2 → interpolated between src[1](10.0) and src[2](20.0)
    let t = (2.0_f64 - 1.0) / (2.1 - 1.0);
    let expected = (1.0 - t) as f32 * 10.0 + t as f32 * 20.0;
    assert!(
        (resampled[2][0] - expected).abs() < 1e-4,
        "frame[2]: expected {:.5}, got {:.5}",
        expected,
        resampled[2][0]
    );
}

#[test]
fn test_normalize_3d() {
    let v = normalize_3d([3.0, 0.0, 0.0]).expect("non-zero");
    assert!((v[0] - 1.0).abs() < 1e-10 && v[1].abs() < 1e-10 && v[2].abs() < 1e-10);
    // Diagonal unit vector
    let d = normalize_3d([1.0, 1.0, 1.0]).expect("non-zero");
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-10, "len={}", len);
    // Zero vector → None
    assert!(normalize_3d([0.0, 0.0, 0.0]).is_none());
}

#[test]
fn test_slice_normal_from_iop_axial() {
    let iop = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
    let n = slice_normal_from_iop(iop).expect("valid iop");
    assert!(
        (n[0]).abs() < 1e-10 && (n[1]).abs() < 1e-10 && (n[2] - 1.0).abs() < 1e-10,
        "normal={:?}",
        n
    );
}

#[test]
fn test_dot_3d() {
    assert!((dot_3d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    assert!((dot_3d([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < 1e-10);
}

#[test]
fn test_load_from_series_oblique_direction_uses_column_slice_convention() {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = burn_ndarray::NdArray<f32>;

    let temp = tempfile::tempdir().unwrap();
    let series_path = temp.path().join("coronal_series");

    let (depth, rows, cols) = (3usize, 2usize, 2usize);
    let data = vec![500.0f32; depth * rows * cols];
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new([depth, rows, cols])),
        &device,
    );
    let image = Image::<B, 3>::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.5, 1.0, 1.0]),
        Direction::identity(),
    );

    // Coronal IOP: F_r=[1,0,0], F_c=[0,0,-1], N̂=F_r×F_c=[0,1,0].
    // RITK direction = from_column_slice([N̂, F_c, F_r]) = [0,1,0, 0,0,-1, 1,0,0].
    let meta = DicomReadMetadata {
        series_instance_uid: Some("2.25.61001".to_string()),
        study_instance_uid: Some("2.25.61002".to_string()),
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some("CT".to_string()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.5, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
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
    .expect("write_dicom_series_with_metadata must not fail");

    let (loaded_image, _) = load_dicom_series_with_metadata::<B, _>(&series_path, &device)
        .expect("load_dicom_series_with_metadata must not fail");

    // RITK convention: from_column_slice([N̂, F_c, F_r]) = from_column_slice([0,1,0, 0,0,-1, 1,0,0]):
    //   col0=[0,1,0]: direction[(0,0)]=0, direction[(1,0)]=1, direction[(2,0)]=0
    //   col1=[0,0,-1]: direction[(0,1)]=0, direction[(1,1)]=0, direction[(2,1)]=-1
    //   col2=[1,0,0]:  direction[(0,2)]=1, direction[(1,2)]=0, direction[(2,2)]=0
    let dir = loaded_image.direction().0;
    const TOL: f64 = 1e-5;

    // Column 0 = slice normal N̂ = [0, 1, 0]
    assert!(
        dir[(0, 0)].abs() < TOL,
        "dir[(0,0)] must be 0.0; got {}",
        dir[(0, 0)]
    );
    assert!(
        (dir[(1, 0)] - 1.0).abs() < TOL,
        "dir[(1,0)] must be 1.0; got {}",
        dir[(1, 0)]
    );
    assert!(
        dir[(2, 0)].abs() < TOL,
        "dir[(2,0)] must be 0.0; got {}",
        dir[(2, 0)]
    );

    // Column 1 = col cosines F_c = [0, 0, -1]
    assert!(
        dir[(0, 1)].abs() < TOL,
        "dir[(0,1)] must be 0.0; got {}",
        dir[(0, 1)]
    );
    assert!(
        dir[(1, 1)].abs() < TOL,
        "dir[(1,1)] must be 0.0; got {}",
        dir[(1, 1)]
    );
    // Discriminating: from_column_slice → -1.0; from_row_slice (wrong) → +1.0
    assert!(
        (dir[(2, 1)] + 1.0).abs() < TOL,
        "dir[(2,1)] must be -1.0 (column-slice convention); \
         from_row_slice would give +1.0; got {}",
        dir[(2, 1)]
    );

    // Column 2 = row cosines F_r = [1, 0, 0]
    assert!(
        (dir[(0, 2)] - 1.0).abs() < TOL,
        "dir[(0,2)] must be 1.0; got {}",
        dir[(0, 2)]
    );
    // Discriminating: from_column_slice → 0.0 here; old convention had 1.0 at (1,2)
    assert!(
        dir[(1, 2)].abs() < TOL,
        "dir[(1,2)] must be 0.0 (RITK column-slice convention); got {}",
        dir[(1, 2)]
    );
    assert!(
        dir[(2, 2)].abs() < TOL,
        "dir[(2,2)] must be 0.0; got {}",
        dir[(2, 2)]
    );
}
