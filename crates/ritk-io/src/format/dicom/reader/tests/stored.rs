//! Exact stored-sample DICOM import.
#![expect(clippy::unwrap_used, reason = "test fixture setup")]

use super::super::stored::{read_dicom_series_stored, DicomStoredImportError};
use super::super::DicomReadBudget;
use super::support::*;
use dicom::core::smallvec::SmallVec;

/// Writes one CT instance carrying explicit 16-bit pixel values.
///
/// The fixture mirrors the scanner's required acquisition identity so the
/// series assembles, and lets a test vary the encoding, photometry, and
/// rescale fields the stored import inspects.
fn write_ct_slice(
    path: &std::path::Path,
    series_uid: &str,
    instance: u32,
    z: f64,
    transfer_syntax: &str,
    photometric: &str,
    slope: f64,
    intercept: f64,
    values: &[u16],
) {
    let sop_instance_uid = format!("{}.{}", series_uid, instance);
    let mut obj = InMemDicomObject::new_empty();
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
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from(series_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("1.2.3.4.5"),
    ));
    let instance_text = instance.to_string();
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from(instance_text.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from(format!("0.0\\0.0\\{z:.1}").as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1.0\\0.0\\0.0\\0.0\\1.0\\0.0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
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
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from(photometric),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("1.0\\1.0"),
    ));
    let slope_text = slope.to_string();
    obj.put(DataElement::new(
        Tag(0x0028, 0x1053),
        VR::DS,
        PrimitiveValue::from(slope_text.as_str()),
    ));
    let intercept_text = intercept.to_string();
    obj.put(DataElement::new(
        Tag(0x0028, 0x1054),
        VR::DS,
        PrimitiveValue::from(intercept_text.as_str()),
    ));
    let mut pixel_bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        pixel_bytes.extend_from_slice(&value.to_le_bytes());
    }
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
                .transfer_syntax(transfer_syntax),
        )
        .expect("meta build must not fail");
    file_obj.write_to_file(path).expect("write must not fail");
}

/// Writes a two-slice 2×2 series with identity rescale and little-endian data.
fn write_uniform_pair(dir: &std::path::Path, series_uid: &str) -> Vec<u16> {
    write_ct_slice(
        &dir.join("s0.dcm"),
        series_uid,
        1,
        0.0,
        "1.2.840.10008.1.2.1",
        "MONOCHROME2",
        1.0,
        0.0,
        &[1, 2, 3, 4],
    );
    write_ct_slice(
        &dir.join("s1.dcm"),
        series_uid,
        2,
        1.0,
        "1.2.840.10008.1.2.1",
        "MONOCHROME2",
        1.0,
        0.0,
        &[5, 6, 7, 8],
    );
    vec![1, 2, 3, 4, 5, 6, 7, 8]
}

#[test]
fn a_uniform_u16_series_round_trips_exact_stored_samples() {
    let dir = tempfile::tempdir().unwrap();
    let expected = write_uniform_pair(dir.path(), "2.25.73001");

    let imported =
        read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT).expect("stored import");
    let volume = &imported.series().volumes()[0];
    assert_eq!(volume.shape(), [2, 2, 2], "shape is depth, row, column");
    assert_eq!(
        volume.samples().sample_type(),
        ritk_codecs::SampleType::U16,
        "16-bit unsigned pixels stay 16-bit unsigned"
    );

    let mut bytes = Vec::new();
    for value in &expected {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        volume
            .samples()
            .encode(ritk_codecs::ByteOrder::LeastSignificantByteFirst)
            .unwrap(),
        bytes,
        "stored samples keep the source values in row-major, slice-ordered layout"
    );
}

#[test]
fn a_fully_retained_series_reports_no_metadata_loss() {
    let dir = tempfile::tempdir().unwrap();
    write_uniform_pair(dir.path(), "2.25.73002");

    let imported =
        read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT).expect("stored import");
    assert!(
        imported.metadata_losses().is_empty(),
        "a fully retained instance contributes no conversion loss"
    );
}

#[test]
fn a_non_identity_rescale_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_ct_slice(
        &dir.path().join("s0.dcm"),
        "2.25.73003",
        1,
        0.0,
        "1.2.840.10008.1.2.1",
        "MONOCHROME2",
        2.0,
        0.0,
        &[1, 2, 3, 4],
    );

    let error = read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT)
        .expect_err("a modality transform would change the stored values");
    assert!(
        matches!(error, DicomStoredImportError::NonIdentityCalibration { .. }),
        "got {error:?}"
    );
}

#[test]
fn a_compressed_transfer_syntax_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_ct_slice(
        &dir.path().join("s0.dcm"),
        "2.25.73004",
        1,
        0.0,
        "1.2.840.10008.1.2.4.50",
        "MONOCHROME2",
        1.0,
        0.0,
        &[1, 2, 3, 4],
    );

    let error = read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT)
        .expect_err("a compressed syntax has no stored payload before decoding");
    assert!(
        matches!(error, DicomStoredImportError::CompressedSyntax { .. }),
        "got {error:?}"
    );
}

#[test]
fn a_non_monochrome_photometry_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_ct_slice(
        &dir.path().join("s0.dcm"),
        "2.25.73005",
        1,
        0.0,
        "1.2.840.10008.1.2.1",
        "RGB",
        1.0,
        0.0,
        &[1, 2, 3, 4],
    );

    let error = read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT)
        .expect_err("RGB pixels have no scalar stored form");
    assert!(
        matches!(error, DicomStoredImportError::NonMonochrome { .. }),
        "got {error:?}"
    );
}

#[test]
fn a_non_uniform_slice_spacing_is_rejected_rather_than_resampled() {
    let dir = tempfile::tempdir().unwrap();
    for (index, z) in [0.0_f64, 1.0, 3.0].into_iter().enumerate() {
        write_ct_slice(
            &dir.path().join(format!("s{index}.dcm")),
            "2.25.73006",
            index as u32 + 1,
            z,
            "1.2.840.10008.1.2.1",
            "MONOCHROME2",
            1.0,
            0.0,
            &[1, 2, 3, 4],
        );
    }

    let error = read_dicom_series_stored(dir.path(), &DicomReadBudget::DEFAULT)
        .expect_err("interpolating stored samples would fabricate source values");
    assert!(
        matches!(error, DicomStoredImportError::NonUniformGeometry),
        "got {error:?}"
    );
}
