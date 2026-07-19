#![allow(unused_imports)]

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
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomTag, DicomValue,
};
use arrayvec::ArrayString;
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_spatial::{Direction, Point, Spacing};

#[test]
fn test_scan_metadata_round_trip_spatial_fields() {
    use ritk_core::image::Image;
    use ritk_image::tensor::{Shape, Tensor};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = coeus_core::SequentialBackend;

    let temp = tempfile::tempdir().unwrap();
    let series_path = temp.path().join("rt_series");

    // Image: 3 slices, 6 rows, 8 cols.
    let (depth, rows, cols) = (3usize, 6usize, 8usize);
    let data = vec![500.0f32; depth * rows * cols];
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([depth, rows, cols], &(data), &device);
    let image = Image::<f32, B, 3>::new(
        tensor,
        Point::new([10.0, 20.0, -50.0]),
        Spacing::new([2.5, 0.8, 0.8]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");

    let meta = DicomReadMetadata {
        series_instance_uid: Some("1.2.3.4.5.6.789".try_into().unwrap()),
        study_instance_uid: Some("1.2.3.4.5.6.100".try_into().unwrap()),
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: Some("RT001".to_string()),
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [2.5, 0.8, 0.8],
        origin: [10.0, 20.0, -50.0],
        direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
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

    let info = scan_dicom_directory(&series_path).expect("scan_dicom_directory must not fail");
    let m = &info.metadata;

    // --- Series-level assertions ---
    assert_eq!(
        m.modality.as_deref(),
        Some("CT"),
        "modality must round-trip; got {:?}",
        m.modality
    );
    assert_eq!(
        m.bits_allocated,
        Some(16),
        "bits_allocated must round-trip as 16; got {:?}",
        m.bits_allocated
    );
    assert_eq!(
        m.dimensions[2], depth,
        "slice count must equal depth {}; got {}",
        depth, m.dimensions[2]
    );
    assert_eq!(
        m.dimensions[0], rows,
        "rows must equal {}; got {}",
        rows, m.dimensions[0]
    );
    assert_eq!(
        m.dimensions[1], cols,
        "cols must equal {}; got {}",
        cols, m.dimensions[1]
    );

    // Spacing: RITK convention [Î”z, Î”Row, Î”Col] = [2.5, 0.8, 0.8].
    let tol = 1e-4_f64;
    assert!(
        (m.spacing[0] - 2.5).abs() < tol,
        "spacing[0] (Î”z) must be 2.5 Â± 1e-4; got {}",
        m.spacing[0]
    );
    assert!(
        (m.spacing[1] - 0.8).abs() < tol,
        "spacing[1] (Î”Row) must be 0.8 Â± 1e-4; got {}",
        m.spacing[1]
    );
    assert!(
        (m.spacing[2] - 0.8).abs() < tol,
        "spacing[2] (Î”Col) must be 0.8 Â± 1e-4; got {}",
        m.spacing[2]
    );

    // Origin: within 1e-4 of [10.0, 20.0, -50.0] (first-slice IPP).
    assert!(
        (m.origin[0] - 10.0).abs() < tol,
        "origin[0] must be 10.0 Â± 1e-4; got {}",
        m.origin[0]
    );
    assert!(
        (m.origin[1] - 20.0).abs() < tol,
        "origin[1] must be 20.0 Â± 1e-4; got {}",
        m.origin[1]
    );
    assert!(
        (m.origin[2] - (-50.0_f64)).abs() < tol,
        "origin[2] must be -50.0 Â± 1e-4; got {}",
        m.origin[2]
    );

    // Direction: RITK axial convention â€” NÌ‚=[0,0,1], F_c=[0,1,0], F_r=[1,0,0].
    // from_column_slice([0,0,1, 0,1,0, 1,0,0])
    let axial_dir = [0.0f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    for (i, (&actual, &expected)) in m.direction.iter().zip(axial_dir.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "direction[{i}] must be {expected:.1} Â± 1e-5 (axial: NÌ‚=[0,0,1], F_c=[0,1,0], F_r=[1,0,0]); got {actual}"
        );
    }

    // --- Per-slice assertions ---
    assert_eq!(
        m.slices.len(),
        depth,
        "must have {} slices; got {}",
        depth,
        m.slices.len()
    );

    for (i, slice) in m.slices.iter().enumerate() {
        // IOP: axial = [1,0,0,0,1,0].
        let iop = slice
            .image_orientation_patient
            .unwrap_or_else(|| panic!("slice {i} must have IOP"));
        let expected_iop = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (j, (&a, &e)) in iop.iter().zip(expected_iop.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "slice {i} IOP[{j}] must be {e:.1} Â± 1e-5; got {a}"
            );
        }

        // Pixel spacing: [0.8, 0.8].
        let ps = slice
            .pixel_spacing
            .unwrap_or_else(|| panic!("slice {i} must have pixel_spacing"));
        assert!(
            (ps[0] - 0.8).abs() < tol,
            "slice {i} pixel_spacing[0] must be 0.8 Â± 1e-4; got {}",
            ps[0]
        );
        assert!(
            (ps[1] - 0.8).abs() < tol,
            "slice {i} pixel_spacing[1] must be 0.8 Â± 1e-4; got {}",
            ps[1]
        );
    }

    // IPP z-coordinates (slices sorted ascending): -50.0, -47.5, -45.0.
    // normal = [0,0,1] (identity direction), spacing_z = 2.5.
    let expected_z = [-50.0f64, -47.5, -45.0];
    for (i, (slice, &ez)) in m.slices.iter().zip(expected_z.iter()).enumerate() {
        let ipp = slice
            .image_position_patient
            .unwrap_or_else(|| panic!("slice {i} must have IPP"));
        assert!(
            (ipp[0] - 10.0).abs() < tol,
            "slice {i} IPP[0] must be 10.0 Â± 1e-4; got {}",
            ipp[0]
        );
        assert!(
            (ipp[1] - 20.0).abs() < tol,
            "slice {i} IPP[1] must be 20.0 Â± 1e-4; got {}",
            ipp[1]
        );
        assert!(
            (ipp[2] - ez).abs() < tol,
            "slice {i} IPP[2] must be {ez} Â± 1e-4; got {}",
            ipp[2]
        );
    }
}

#[test]
fn test_scan_metadata_round_trip_rescale_params() {
    use ritk_core::image::Image;
    use ritk_image::tensor::{Shape, Tensor};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = coeus_core::SequentialBackend;

    let temp = tempfile::tempdir().unwrap();
    let series_path = temp.path().join("rescale_series");

    // Intensities in [-1024.0, 1024.0] to force non-trivial rescale params.
    let (depth, rows, cols) = (2usize, 4usize, 4usize);
    let n = depth * rows * cols;
    let mut data = vec![0.0f32; n];
    for (idx, v) in data.iter_mut().enumerate() {
        *v = -1024.0 + idx as f32 * (2048.0 / (n - 1) as f32);
    }
    let original_first = data[0];

    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([depth, rows, cols], &(data), &device);
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
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        // RITK axial: NÌ‚=[0,0,1], F_c=[0,1,0], F_r=[1,0,0]
        direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
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
    .expect("write must not fail");

    let info = scan_dicom_directory(&series_path).expect("scan must not fail");

    for (i, slice) in info.metadata.slices.iter().enumerate() {
        let slope = slice.rescale_slope;
        let intercept = slice.rescale_intercept;
        assert!(
            slope > 0.0,
            "slice {i} rescale_slope must be > 0; got {slope}"
        );
        assert!(
            intercept.is_finite(),
            "slice {i} rescale_intercept must be finite; got {intercept}"
        );
    }

    // Verify first-voxel reconstruction error is bounded by slope/2.
    // The writer quantizes: pixel = round((v - intercept) / slope); u16 clamped.
    // Reconstructed: pixel * slope + intercept. Error <= slope/2.
    let first_slice = &info.metadata.slices[0];
    let slope = first_slice.rescale_slope as f32;
    let intercept = first_slice.rescale_intercept as f32;
    let pixel_first = ((original_first - intercept) / slope).round() as u16;
    let reconstructed = pixel_first as f32 * slope + intercept;
    let error = (reconstructed - original_first).abs();
    assert!(
        error <= slope / 2.0 + 1e-3,
        "first-voxel reconstruction error {error} exceeds quantization bound {}",
        slope / 2.0
    );
}

#[test]
fn test_scan_metadata_round_trip_transfer_syntax() {
    use ritk_core::image::Image;
    use ritk_image::tensor::{Shape, Tensor};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = coeus_core::SequentialBackend;

    let temp = tempfile::tempdir().unwrap();
    let series_path = temp.path().join("ts_series");

    let (depth, rows, cols) = (2usize, 4usize, 4usize);
    let data = vec![1000.0f32; depth * rows * cols];
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on([depth, rows, cols], &(data), &device);
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
        modality: Some(ArrayString::from("OT").unwrap()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [rows, cols, depth],
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
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
    .expect("write must not fail");

    let info = scan_dicom_directory(&series_path).expect("scan must not fail");

    // The writer emits transfer_syntax EXPLICIT_VR_LE = Explicit VR Little Endian.
    // The reader must extract this from the file meta, not from Tag(0x0008,0x0070).
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
