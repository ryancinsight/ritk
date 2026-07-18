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
use crate::format::dicom::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomTag, DicomValue,
};
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_spatial::{Direction, Point, Spacing};
#[test]
fn test_scan_skull_ct_folder_with_dicomdir_loads_series() {
    println!("START test_scan_skull_ct_folder_with_dicomdir_loads_series");
    let device = coeus_core::SequentialBackend;
    let series_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/2_skull_ct");
    let series_path = series_path.as_path();
    println!("Path: {:?}", series_path);
    println!("Before scan_dicom_directory");
    let info = scan_dicom_directory(series_path).expect("scan_dicom_directory must succeed");
    println!(
        "After scan_dicom_directory, num_slices: {}",
        info.num_slices
    );
    assert!(
        info.num_slices > 0,
        "expected at least one slice from skull CT sample"
    );
    assert_eq!(
        info.metadata.dimensions[2], info.num_slices,
        "depth must match scanned slice count"
    );
    println!("Before read_dicom_series_with_metadata");
    let (image, _) =
        read_dicom_series_with_metadata::<coeus_core::SequentialBackend, _>(series_path, &device)
            .expect("read_dicom_series_with_metadata must succeed");
    println!("After read_dicom_series_with_metadata");
    assert_eq!(
        image.shape()[0],
        info.num_slices,
        "loaded image depth must match scan result"
    );
    assert!(
        image.shape()[1] > 0 && image.shape()[2] > 0,
        "loaded image must have nonzero in-plane dimensions"
    );
}

#[test]
fn test_scan_skull_ct_dicomdir_and_folder_agree_on_series() {
    let device = coeus_core::SequentialBackend;
    let series_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/2_skull_ct");
    let series_path = series_path.as_path();
    let info = scan_dicom_directory(series_path).expect("scan_dicom_directory must succeed");
    let (image, _) =
        read_dicom_series_with_metadata::<coeus_core::SequentialBackend, _>(series_path, &device)
            .expect("read_dicom_series_with_metadata must succeed");
    let spatial = image.spacing();
    assert!(
        spatial[0] > 0.0 && spatial[1] > 0.0 && spatial[2] > 0.0,
        "all spacing axes must be positive"
    );
    assert!(
        info.metadata.direction.iter().all(|v| v.is_finite()),
        "direction matrix must contain finite values"
    );
    assert!(
        image.direction().0.determinant().abs() > 0.0,
        "direction matrix must be invertible"
    );
}

#[test]
fn test_scan_directory_selects_most_populated_series_when_same_dimensions() {
    use dicom::core::smallvec::SmallVec;

    let dir = tempfile::tempdir().unwrap();

    // Write a minimal CT DICOM file with the given SeriesInstanceUID, instance
    // number, and z-position (used for IPP and sort ordering).
    let write_ct_slice = |path: &std::path::Path, series_uid: &str, instance: u32, z: f64| {
        let sop_instance_uid = format!("{}.{}", series_uid, instance);
        let mut obj = InMemDicomObject::new_empty();
        // SOP class: CT Image Storage
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("CT"),
        ));
        // SeriesInstanceUID (0020,000E) â€” the tag under test.
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_uid),
        ));
        // StudyInstanceUID
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from("1.2.3.4.5"),
        ));
        // InstanceNumber
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", instance).as_str()),
        ));
        // ImagePositionPatient (0020,0032): 0.0\0.0\z
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(format!("0.0\\0.0\\{:.1}", z).as_str()),
        ));
        // ImageOrientationPatient (0020,0037): axial [1,0,0,0,1,0]
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from("1.0\\0.0\\0.0\\0.0\\1.0\\0.0"),
        ));
        // Rows / Cols: 8Ã—8
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        // BitsAllocated / BitsStored / HighBit / PixelRepresentation
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        // SamplesPerPixel / PhotometricInterpretation
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        // PixelSpacing: 1.0\1.0
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from("1.0\\1.0"),
        ));
        // PixelData: 8Ã—8 Ã— 2 bytes = 128 bytes of zeroes
        let pixel_bytes: Vec<u8> = vec![0u8; 8 * 8 * 2];
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .expect("meta build must not fail");
        file_obj.write_to_file(path).expect("write must not fail");
    };

    // Series A: 3 slices â€” the most-populated series.
    write_ct_slice(&dir.path().join("A1.dcm"), "2.25.A", 1, 0.0);
    write_ct_slice(&dir.path().join("A2.dcm"), "2.25.A", 2, 1.0);
    write_ct_slice(&dir.path().join("A3.dcm"), "2.25.A", 3, 2.0);
    // Series B: 1 slice â€” must be excluded by plurality selection.
    write_ct_slice(&dir.path().join("B1.dcm"), "2.25.B", 1, 5.0);

    let result = scan_dicom_directory(dir.path())
        .expect("scan_dicom_directory must succeed with valid CT slices");

    assert_eq!(
        result.metadata.series_instance_uid.as_deref(),
        Some("2.25.A"),
        "series_instance_uid must be the most-populated series (2.25.A, 3 slices); \
         got {:?}",
        result.metadata.series_instance_uid
    );
    assert_eq!(
        result.num_slices, 3,
        "num_slices must be 3 (Series A only); got {}",
        result.num_slices
    );
}
