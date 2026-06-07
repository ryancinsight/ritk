#![allow(unused_imports)]

use arrayvec::ArrayString;

use super::super::geometry::{
    analyze_slice_spacing, dot_3d, normalize_3d, resample_frames_linear, slice_normal_from_iop,
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
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::TransferSyntaxKind;
#[test]
fn test_scan_directory_warns_on_inconsistent_iop() {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = burn_ndarray::NdArray<f32>;

    let temp = tempfile::tempdir().unwrap();
    let dir_a = temp.path().join("iop_axial");
    let dir_b = temp.path().join("iop_coronal");
    let mixed = temp.path().join("iop_mixed");
    std::fs::create_dir_all(&mixed).unwrap();

    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let data = vec![500.0f32; 4]; // 1×2×2

    // Axial series: IOP=[1,0,0,0,1,0], normal=[0,0,1], origin=[0,0,0].
    // IPP for slice 0 = [0,0,0].
    {
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([1usize, 2, 2])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let meta = DicomReadMetadata {
            series_instance_uid: Some("2.25.62001".try_into().unwrap()),
            study_instance_uid: Some("2.25.62002".try_into().unwrap()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some(ArrayString::from("CT").unwrap()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [2, 2, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
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
        crate::format::dicom::writer::write_dicom_series_with_metadata(&dir_a, &image, Some(&meta))
            .expect("write axial series");
    }

    // Coronal series: IOP=[1,0,0,0,0,-1], normal=[0,1,0], origin=[0,1,0].
    // IPP for slice 0 = origin + 0×spacing×normal = [0,1,0].
    // Projected onto any normal: axial IPP=0 ≤ coronal IPP≥0 → axial sorts first.
    {
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([1usize, 2, 2])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 1.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let meta = DicomReadMetadata {
            series_instance_uid: Some("2.25.62003".try_into().unwrap()),
            study_instance_uid: Some("2.25.62004".try_into().unwrap()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some(ArrayString::from("CT").unwrap()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [2, 2, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 1.0, 0.0],
            direction: [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
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
        crate::format::dicom::writer::write_dicom_series_with_metadata(&dir_b, &image, Some(&meta))
            .expect("write coronal series");
    }

    std::fs::copy(dir_a.join("slice_0000.dcm"), mixed.join("slice_0000.dcm"))
        .expect("copy axial slice");
    std::fs::copy(dir_b.join("slice_0000.dcm"), mixed.join("slice_0001.dcm"))
        .expect("copy coronal slice");

    let result = scan_dicom_directory(&mixed);
    assert!(
        result.is_ok(),
        "scan must return Ok for mixed-IOP series; err={:?}",
        result.err()
    );

    let info = result.unwrap();

    // Both slices must be retained regardless of IOP inconsistency.
    assert_eq!(
        info.metadata.dimensions[2], 2,
        "both slices must be loaded; got {}",
        info.metadata.dimensions[2]
    );

    // Canonical IOP is the first (lowest-position) slice after sort = axial.
    // Axial IPP=[0,0,0] projects to 0 along any normal; coronal IPP=[0,1,0]
    // projects to ≥0; when equal, filename tiebreak puts slice_0000 (axial) first.
    // RITK direction[0..3] = N̂ for axial = [0,0,1]; direction[3..6] = F_c = [0,1,0].
    let expected_dir_prefix = [0.0f64, 0.0, 1.0, 0.0, 1.0, 0.0];
    let tol = 1e-5_f64;
    for (i, (&actual, &expected)) in info.metadata.direction[0..6]
        .iter()
        .zip(expected_dir_prefix.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < tol,
            "direction[{i}] must be {expected:.1} ± 1e-5 (axial: N̂=[0,0,1], F_c=[0,1,0]); got {actual}"
        );
    }
}

#[test]
fn test_scan_directory_warns_on_inconsistent_pixel_spacing() {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;
    type B = burn_ndarray::NdArray<f32>;

    let temp = tempfile::tempdir().unwrap();
    let dir_a = temp.path().join("ps_a");
    let dir_b = temp.path().join("ps_b");
    let mixed = temp.path().join("ps_mixed");
    std::fs::create_dir_all(&mixed).unwrap();

    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let data = vec![500.0f32; 4]; // 1×2×2

    // Series A: pixel_spacing=[0.8,0.8], origin=[0,0,0], IPP=[0,0,0].
    {
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([1usize, 2, 2])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 0.8, 0.8]),
            Direction::identity(),
        );
        let meta = DicomReadMetadata {
            series_instance_uid: Some("2.25.63001".try_into().unwrap()),
            study_instance_uid: Some("2.25.63002".try_into().unwrap()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some(ArrayString::from("CT").unwrap()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [2, 2, 1],
            spacing: [1.0, 0.8, 0.8],
            origin: [0.0, 0.0, 0.0],
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
        crate::format::dicom::writer::write_dicom_series_with_metadata(&dir_a, &image, Some(&meta))
            .expect("write series_a");
    }

    // Series B: pixel_spacing=[1.0,1.0], origin=[0,0,1.0], IPP=[0,0,1.0].
    // Different z-position ensures both slices are retained after sort.
    {
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([1usize, 2, 2])),
            &device,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([0.0, 0.0, 1.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let meta = DicomReadMetadata {
            series_instance_uid: Some("2.25.63003".try_into().unwrap()),
            study_instance_uid: Some("2.25.63004".try_into().unwrap()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some(ArrayString::from("CT").unwrap()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [2, 2, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 1.0],
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
        crate::format::dicom::writer::write_dicom_series_with_metadata(&dir_b, &image, Some(&meta))
            .expect("write series_b");
    }

    // Copy slice_0000.dcm (0.8 spacing) first; on NTFS temp-dirs are returned
    // alphabetically so slice_0000 precedes slice_0001 in read_dir iteration,
    // making 0.8-spacing the first_pixel_spacing captured during parsing.
    std::fs::copy(dir_a.join("slice_0000.dcm"), mixed.join("slice_0000.dcm"))
        .expect("copy series_a slice");
    std::fs::copy(dir_b.join("slice_0000.dcm"), mixed.join("slice_0001.dcm"))
        .expect("copy series_b slice");

    let result = scan_dicom_directory(&mixed);
    assert!(
        result.is_ok(),
        "scan must return Ok for mixed-PixelSpacing series; err={:?}",
        result.err()
    );

    let info = result.unwrap();

    // Both slices must be retained regardless of PixelSpacing inconsistency.
    assert_eq!(
        info.metadata.dimensions[2], 2,
        "both slices must be loaded; got {}",
        info.metadata.dimensions[2]
    );

    // spacing[1] and spacing[2] reflect the first slice's pixel spacing (0.8 mm).
    // RITK convention: spacing = [Δz, ΔRow, ΔCol].
    let tol = 1e-5_f64;
    assert!(
        (info.metadata.spacing[1] - 0.8).abs() < tol,
        "spacing[1] (ΔRow) must be 0.8 ± 1e-5 (first slice pixel spacing row); got {}",
        info.metadata.spacing[1]
    );
    assert!(
        (info.metadata.spacing[2] - 0.8).abs() < tol,
        "spacing[2] (ΔCol) must be 0.8 ± 1e-5 (first slice pixel spacing col); got {}",
        info.metadata.spacing[2]
    );
}

#[test]
fn test_physical_transform_depth_index_advances_along_slice_normal() {
    // Invariant: advancing the depth index by 1 must move the physical point by exactly
    // Δz along the slice normal N̂. With spacing=[Δz, ΔRow, ΔCol] and direction
    // cols=[N̂, F_c, F_r]: point(1,0,0) = origin + 1*Δz*N̂.
    use burn::tensor::{Shape, Tensor, TensorData};
    use nalgebra::SMatrix;
    use ritk_core::spatial::{Direction, Point, Spacing};
    type B = burn_ndarray::NdArray<f32>;
    const TOL: f64 = 1e-10;

    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0f32; 2 * 4 * 4], Shape::new([2, 4, 4])),
        &device,
    );
    // Axial: N̂=[0,0,1], F_c=[0,1,0], F_r=[1,0,0], Δz=2.5, ΔRow=0.8, ΔCol=0.8
    let origin = Point::new([10.0, 20.0, -50.0]);
    let spacing = Spacing::new([2.5, 0.8, 0.8]);
    // direction from_column_slice([0,0,1, 0,1,0, 1,0,0]) — axial RITK convention
    let dir = Direction(SMatrix::<f64, 3, 3>::from_column_slice(&[
        0.0, 0.0, 1.0, // col 0 = N̂
        0.0, 1.0, 0.0, // col 1 = F_c
        1.0, 0.0, 0.0, // col 2 = F_r
    ]));
    let image = ritk_core::image::Image::new(tensor, origin, spacing, dir);

    // Voxel (0,0,0): must be at origin
    let p0 = image.transform_continuous_index_to_physical_point(&Point::new([0.0, 0.0, 0.0]));
    assert!((p0[0] - 10.0).abs() < TOL, "origin x; got {}", p0[0]);
    assert!((p0[1] - 20.0).abs() < TOL, "origin y; got {}", p0[1]);
    assert!((p0[2] + 50.0).abs() < TOL, "origin z; got {}", p0[2]);

    // Voxel (1,0,0): depth=1 → origin + 2.5*[0,0,1] = [10,20,-47.5]
    let p1 = image.transform_continuous_index_to_physical_point(&Point::new([1.0, 0.0, 0.0]));
    assert!(
        (p1[0] - 10.0).abs() < TOL,
        "depth=1: x must stay; got {}",
        p1[0]
    );
    assert!(
        (p1[1] - 20.0).abs() < TOL,
        "depth=1: y must stay; got {}",
        p1[1]
    );
    assert!(
        (p1[2] - (-47.5)).abs() < TOL,
        "depth=1: z must advance 2.5mm; got {}",
        p1[2]
    );

    // Voxel (0,1,0): row=1 → origin + 0.8*F_c = origin + 0.8*[0,1,0]
    let p2 = image.transform_continuous_index_to_physical_point(&Point::new([0.0, 1.0, 0.0]));
    assert!((p2[0] - 10.0).abs() < TOL, "row=1: x stays; got {}", p2[0]);
    assert!(
        (p2[1] - 20.8).abs() < TOL,
        "row=1: y advances 0.8mm; got {}",
        p2[1]
    );
    assert!((p2[2] + 50.0).abs() < TOL, "row=1: z stays; got {}", p2[2]);

    // Voxel (0,0,1): col=1 → origin + 0.8*F_r = origin + 0.8*[1,0,0]
    let p3 = image.transform_continuous_index_to_physical_point(&Point::new([0.0, 0.0, 1.0]));
    assert!(
        (p3[0] - 10.8).abs() < TOL,
        "col=1: x advances 0.8mm; got {}",
        p3[0]
    );
    assert!((p3[1] - 20.0).abs() < TOL, "col=1: y stays; got {}", p3[1]);
    assert!((p3[2] + 50.0).abs() < TOL, "col=1: z stays; got {}", p3[2]);
}
