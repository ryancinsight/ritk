#![allow(unused_imports)]

use super::super::geometry::{
    analyze_slice_spacing, dot_3d, normalize_3d, resample_frames_linear, slice_normal_from_iop,
};
use super::super::loader::{
    load_dicom_series_with_metadata, load_from_series,
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
fn test_decode_pixel_bytes_unsigned_16bit_identity_rescale() {
    // u16: [0x00,0x00] = 0; [0xFF,0xFF] = 65535. slope=1.0, intercept=0.0 → identity.
    let bytes: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF];
    let result = decode_pixel_bytes(&bytes, 16, 0, 1.0, 0.0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 0.0f32);
    assert_eq!(result[1], 65535.0f32);
}

#[test]
fn test_decode_pixel_bytes_signed_16bit_identity_rescale() {
    // i16::MIN = -32768 stored as [0x00, 0x80] LE; i16::MAX = 32767 stored as [0xFF, 0x7F] LE.
    let bytes: [u8; 4] = [0x00, 0x80, 0xFF, 0x7F];
    let result = decode_pixel_bytes(&bytes, 16, 1, 1.0, 0.0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], -32768.0f32);
    assert_eq!(result[1], 32767.0f32);
}

#[test]
fn test_decode_pixel_bytes_signed_16bit_with_rescale() {
    // i16: -1 = [0xFF, 0xFF] LE; decoded = -1.0 × 2.0 + 100.0 = 98.0.
    let bytes: [u8; 2] = [0xFF, 0xFF];
    let result = decode_pixel_bytes(&bytes, 16, 1, 2.0, 100.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 98.0f32);
}

#[test]
fn test_decode_pixel_bytes_8bit_identity_rescale() {
    let bytes: [u8; 3] = [0, 127, 255];
    let result = decode_pixel_bytes(&bytes, 8, 0, 1.0, 0.0);
    assert_eq!(result, vec![0.0f32, 127.0f32, 255.0f32]);
}

#[test]
fn test_decode_pixel_bytes_8bit_with_rescale() {
    // 8-bit value 200; slope=0.5, intercept=10.0 → 200 × 0.5 + 10.0 = 110.0.
    let bytes: [u8; 1] = [200];
    let result = decode_pixel_bytes(&bytes, 8, 0, 0.5, 10.0);
    assert_eq!(result[0], 110.0f32);
}

#[test]
fn test_is_likely_dicom_file_accepts_canonical_extensions() {
    assert!(is_likely_dicom_file(std::path::Path::new("scan.dcm")));
    assert!(is_likely_dicom_file(std::path::Path::new("SCAN.DCM")));
    assert!(is_likely_dicom_file(std::path::Path::new("scan.dicom")));
    assert!(is_likely_dicom_file(std::path::Path::new("scan.ima")));
}

#[test]
fn test_is_likely_dicom_file_rejects_analyze_and_raw_extensions() {
    assert!(!is_likely_dicom_file(std::path::Path::new("brain.hdr")));
    assert!(!is_likely_dicom_file(std::path::Path::new("brain.img")));
    assert!(!is_likely_dicom_file(std::path::Path::new("brain.raw")));
    assert!(!is_likely_dicom_file(std::path::Path::new("brain.nii")));
    assert!(!is_likely_dicom_file(std::path::Path::new("data.bin")));
}

#[test]
fn test_slice_metadata_default_pixel_representation_is_zero() {
    let meta = DicomSliceMetadata::default();
    assert_eq!(
        meta.pixel_representation, 0,
        "pixel_representation default must be 0 (unsigned)"
    );
    assert_eq!(meta.bits_allocated, 16, "bits_allocated default must be 16");
    assert!(
        meta.window_center.is_none(),
        "window_center default must be None"
    );
    assert!(
        meta.window_width.is_none(),
        "window_width default must be None"
    );
    assert_eq!(
        meta.rescale_slope, 1.0f32,
        "rescale_slope default must be 1.0"
    );
    assert_eq!(
        meta.rescale_intercept, 0.0f32,
        "rescale_intercept default must be 0.0"
    );
}

#[test]
fn test_read_slice_pixels_signed_i16_roundtrip() {
    // Build a DICOM file with PixelRepresentation=1 and three known i16 values.
    // Stored pixel values: -1000, 0, 1000. RescaleSlope=1, RescaleIntercept=0.
    // Expected decoded values: [-1000.0, 0.0, 1000.0].
    use dicom::core::smallvec::SmallVec;
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    let pixels_i16: [i16; 3] = [-1000, 0, 1000];
    let pixel_bytes: Vec<u8> = pixels_i16.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99999"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("OT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    )); // rows=1
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(3_u16),
    )); // cols=3
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(1_u16),
    )); // PixelRepresentation=1 (signed)
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x1052),
        VR::DS,
        PrimitiveValue::from("0"),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid("2.25.99999")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build must succeed");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("signed.dcm");
    file_obj.write_to_file(&path).expect("write must succeed");

    // Construct DicomSliceMetadata as scan_dicom_directory would populate it.
    let slice_meta = DicomSliceMetadata {
        path: path.clone(),
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
        pixel_representation: 1,
        bits_allocated: 16,
        ..DicomSliceMetadata::default()
    };

    let result = read_slice_pixels(&slice_meta).expect("read_slice_pixels must succeed");
    assert_eq!(result.len(), 3, "pixel count must be 3");
    assert_eq!(result[0], -1000.0f32, "pixel[0] must be -1000.0");
    assert_eq!(result[1], 0.0f32, "pixel[1] must be 0.0");
    assert_eq!(result[2], 1000.0f32, "pixel[2] must be 1000.0");
}

#[test]
fn test_read_slice_pixels_rejects_rgb_scalar_volume() {
    use dicom::core::smallvec::SmallVec;
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.999991"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("OT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(3_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(8_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("RGB"),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(vec![120, 64, 32])),
    ));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid("2.25.999991")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build must succeed");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rgb.dcm");
    file_obj.write_to_file(&path).expect("write must succeed");

    let slice_meta = DicomSliceMetadata {
        path,
        bits_allocated: 8,
        pixel_representation: 0,
        ..DicomSliceMetadata::default()
    };

    let err = read_slice_pixels(&slice_meta).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SamplesPerPixel=3") && msg.contains("scalar volume loader"),
        "expected scalar loader RGB rejection, got {err:#}"
    );
}
