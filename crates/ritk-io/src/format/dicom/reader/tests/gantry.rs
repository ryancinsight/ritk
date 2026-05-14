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
fn test_gantry_tilt_synthesizes_oblique_orientation() {
    // Invariant: axial IOP [1,0,0,0,1,0] + GantryDetectorTilt=15° must produce
    // synthesized F_c=[0,cos(15°),-sin(15°)] and N̂=[0,sin(15°),cos(15°)].
    use dicom::core::{Tag, VR};
    use dicom::object::{FileMetaTableBuilder, InMemDicomObject};
    use dicom_core::smallvec::SmallVec;
    use dicom_core::PrimitiveValue;
    use std::f64::consts::PI;

    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("tilted_series");
    std::fs::create_dir_all(&dir).unwrap();

    let tilt_deg = 15.0_f64;
    let theta = tilt_deg * PI / 180.0;
    let expected_cos = theta.cos();
    let expected_sin = theta.sin();

    let slice_path = dir.join("slice_0000.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.999001"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("CT"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[2u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[2u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[16u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[16u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[15u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[0u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[1u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    // Axial IOP [1,0,0,0,1,0]
    obj.put(dicom::core::DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1.000000\\0.000000\\0.000000\\0.000000\\1.000000\\0.000000"),
    ));
    // IPP at origin
    obj.put(dicom::core::DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from("0.000000\\0.000000\\0.000000"),
    ));
    // PixelSpacing
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("1.000000\\1.000000"),
    ));
    // SliceThickness
    obj.put(dicom::core::DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    // GantryDetectorTilt = 15°
    obj.put(dicom::core::DataElement::new(
        Tag(0x0018, 0x1120),
        VR::DS,
        PrimitiveValue::from(format!("{:.6}", tilt_deg).as_str()),
    ));
    // Pixel data: 2×2 u16 = 8 bytes
    let pixel_u16: Vec<u16> = vec![100, 200, 300, 400];
    obj.put(dicom::core::DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
    ));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.999001")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build failed");
    file_obj
        .write_to_file(&slice_path)
        .expect("write slice failed");

    let info = scan_dicom_directory(&dir).expect("scan must succeed");
    assert_eq!(
        info.num_slices, 1,
        "must find 1 slice; got {}",
        info.num_slices
    );

    let slice = &info.metadata.slices[0];

    // gantry_tilt field must be populated
    let tilt_read = slice
        .gantry_tilt
        .expect("gantry_tilt must be Some after reading tag");
    assert!(
        (tilt_read - tilt_deg).abs() < 1e-5,
        "gantry_tilt must be 15.0; got {}",
        tilt_read
    );

    // IOP must be synthesized from tilt
    let iop = slice
        .image_orientation_patient
        .expect("IOP must be set after tilt synthesis");
    const TOL: f64 = 1e-10;

    // F_r stays [1,0,0]
    assert!(
        (iop[0] - 1.0).abs() < TOL,
        "iop[0]=F_r[0] must be 1; got {}",
        iop[0]
    );
    assert!(
        iop[1].abs() < TOL,
        "iop[1]=F_r[1] must be 0; got {}",
        iop[1]
    );
    assert!(
        iop[2].abs() < TOL,
        "iop[2]=F_r[2] must be 0; got {}",
        iop[2]
    );
    // F_c = [0, cos(θ), -sin(θ)]
    assert!(
        iop[3].abs() < TOL,
        "iop[3]=F_c[0] must be 0; got {}",
        iop[3]
    );
    assert!(
        (iop[4] - expected_cos).abs() < TOL,
        "iop[4]=F_c[1] must be cos(15°)={:.10}; got {}",
        expected_cos,
        iop[4]
    );
    assert!(
        (iop[5] + expected_sin).abs() < TOL,
        "iop[5]=F_c[2] must be -sin(15°)={:.10}; got {}",
        -expected_sin,
        iop[5]
    );

    // direction[0..3] = N̂ = F_r × F_c = [0, sin(15°), cos(15°)]
    let dir = &info.metadata.direction;
    assert!(dir[0].abs() < TOL, "N̂[0] must be 0; got {}", dir[0]);
    assert!(
        (dir[1] - expected_sin).abs() < TOL,
        "N̂[1] must be sin(15°)={:.10}; got {}",
        expected_sin,
        dir[1]
    );
    assert!(
        (dir[2] - expected_cos).abs() < TOL,
        "N̂[2] must be cos(15°)={:.10}; got {}",
        expected_cos,
        dir[2]
    );
}
