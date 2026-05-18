use super::*;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

type B = NdArray<f32>;

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
fn read_dicom_color_series_preserves_interleaved_rgb_samples() {
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
    let device = <B as Backend>::Device::default();

    let (volume, metadata) =
        read_dicom_color_series::<B, _>(dir.path(), &device).expect("RGB load must succeed");

    assert_eq!(volume.shape(), [2, 1, 2, 3]);
    assert_eq!(volume.spatial_shape(), [2, 1, 2]);
    assert_eq!(metadata.dimensions, [1, 2, 2]);
    assert_eq!(metadata.photometric_interpretation.as_deref(), Some("RGB"));
    volume.with_data_slice(|samples| {
        assert_eq!(
            samples,
            &[255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0]
        );
    });
}

#[test]
fn read_dicom_color_series_rejects_scalar_samples() {
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
    let device = <B as Backend>::Device::default();

    let err = read_dicom_color_series::<B, _>(dir.path(), &device).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SamplesPerPixel=1"),
        "expected RGB sample-count rejection, got {msg}"
    );
}

#[test]
fn read_dicom_color_series_rejects_planar_rgb_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_rgb_slice(
        &dir.path().join("planar.dcm"),
        "2.25.3151",
        1,
        0.0,
        &[255, 0, 0, 0, 255, 0],
        Some(1),
    );
    let device = <B as Backend>::Device::default();

    let err = read_dicom_color_series::<B, _>(dir.path(), &device).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("PlanarConfiguration=0") && msg.contains("declares 1"),
        "expected planar RGB rejection, got {msg}"
    );
}
