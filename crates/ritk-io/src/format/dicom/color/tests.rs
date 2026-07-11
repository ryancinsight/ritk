//! Native DICOM RGB color-volume tests.

use coeus_core::SequentialBackend;
use ritk_dicom::PixelSignedness;

use super::*;
use arrayvec::ArrayString;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

fn write_rgb_slice(
    path: &Path,
    sop_instance_uid: &str,
    instance_number: u16,
    z_mm: f64,
    samples: &[u8],
    planar_configuration: Option<u16>,
) {
    assert_eq!(samples.len(), 2 * RGB_CHANNELS);
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("OT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("2.25.3000"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.3001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from(instance_number.to_string()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from(format!("0\\0\\{z_mm}")),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1\\0\\0\\0\\1\\0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(3_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("RGB"),
    ));
    if let Some(planar_configuration) = planar_configuration {
        obj.put(DataElement::new(
            Tag(0x0028, 0x0006),
            VR::US,
            PrimitiveValue::from(planar_configuration),
        ));
    }
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("0.5\\0.25"),
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
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(samples.to_vec())),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
            .media_storage_sop_instance_uid(sop_instance_uid)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(path)
    .expect("DICOM RGB slice must be written");
}

#[test]
fn native_color_series_preserves_interleaved_rgb_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_rgb_slice(
        &dir.path().join("slice1.dcm"),
        "2.25.3101",
        1,
        0.0,
        &[255, 0, 0, 0, 255, 0],
        Some(0),
    );
    write_rgb_slice(
        &dir.path().join("slice2.dcm"),
        "2.25.3102",
        2,
        2.0,
        &[0, 0, 255, 255, 255, 255],
        Some(0),
    );

    let backend = SequentialBackend;
    let (volume, metadata) =
        load_dicom_color_series(dir.path(), &backend).expect("RGB load must succeed");

    assert_eq!(volume.shape(), [2, 1, 2, 3]);
    assert_eq!(metadata.dimensions, [1, 2, 2]);
    assert_eq!(metadata.photometric_interpretation.as_deref(), Some("RGB"));

    let samples = volume.data_cow_on(&backend);
    assert_eq!(
        samples.as_ref(),
        &[255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0]
    );
}

#[test]
fn native_color_series_rejects_scalar_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mono.dcm");
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.3201"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.3202"),
    ));
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
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(2_u16),
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
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::U8(SmallVec::from_vec(vec![1, 2])),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
            .media_storage_sop_instance_uid("2.25.3201")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("scalar DICOM must be written");

    let err = load_dicom_color_series(dir.path(), &SequentialBackend).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SamplesPerPixel=1"),
        "expected RGB sample-count rejection, got {msg}"
    );
}

#[test]
fn native_color_series_rejects_planar_rgb_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_rgb_slice(
        &dir.path().join("planar.dcm"),
        "2.25.3151",
        1,
        0.0,
        &[255, 0, 0, 0, 255, 0],
        Some(1),
    );
    let err = load_dicom_color_series(dir.path(), &SequentialBackend).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("PlanarConfiguration=0") && msg.contains("declares 1"),
        "expected planar RGB rejection, got {msg}"
    );
}

#[test]
fn native_color_from_series_preserves_values_and_metadata() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_rgb_slice(
        &dir.path().join("slice1.dcm"),
        "2.25.4101",
        1,
        0.0,
        &[10, 20, 30, 40, 50, 60],
        Some(0),
    );

    let path = dir.path().join("slice1.dcm");
    let bytes = std::fs::read(&path).expect("must read back written DICOM file");

    let slice = DicomSliceMetadata {
        path: path.clone(),
        preservation: Default::default(),
        sop_instance_uid: Some("2.25.4101".try_into().unwrap()),
        instance_number: Some(1),
        slice_location: None,
        image_position_patient: Some([0.0, 0.0, 0.0]),
        image_orientation_patient: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: Some([0.5, 0.25]),
        slice_thickness: None,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
        sop_class_uid: None,
        transfer_syntax_uid: Some("1.2.840.10008.1.2.1".try_into().unwrap()),
        private_tags: Default::default(),
        pixel_representation: PixelSignedness::Unsigned,
        bits_allocated: 8,
        window_center: None,
        window_width: None,
        gantry_tilt: None,
        patient_position: None,
        part10_bytes: Some(bytes),
    };

    let metadata = DicomReadMetadata {
        series_instance_uid: Some("2.25.4001".try_into().unwrap()),
        study_instance_uid: None,
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some(ArrayString::from("OT").unwrap()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [1, 2, 1],
        spacing: [2.0, 0.5, 0.25],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(8),
        bits_stored: Some(8),
        high_bit: Some(7),
        photometric_interpretation: Some(ArrayString::from("RGB").unwrap()),
        slices: vec![slice],
        private_tags: Default::default(),
        preservation: Default::default(),
        patient_weight_kg: None,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    };

    let series = reader::DicomSeriesInfo {
        path: dir.path().to_path_buf(),
        num_slices: 1,
        metadata,
    };

    let backend = SequentialBackend;
    let (volume, meta) =
        load_dicom_color_from_series(series, &backend).expect("native color series must load");
    assert_eq!(volume.shape(), [1, 1, 2, 3]);
    assert_eq!(meta.dimensions, [1, 2, 1]);
    assert_eq!(meta.photometric_interpretation.as_deref(), Some("RGB"));

    assert_eq!(
        volume.data_cow_on(&backend).as_ref(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}
