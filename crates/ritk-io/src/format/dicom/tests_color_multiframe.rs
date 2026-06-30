use super::*;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

type B = NdArray<f32>;

fn write_multiframe(
    path: &Path,
    samples_per_pixel: u16,
    photometric: &str,
    planar_configuration: Option<u16>,
    samples: Vec<u8>,
) {
    write_multiframe_with_dims(
        path,
        samples_per_pixel,
        photometric,
        planar_configuration,
        1,
        2,
        samples,
    );
}

#[allow(clippy::too_many_arguments)]
fn write_multiframe_with_dims(
    path: &Path,
    samples_per_pixel: u16,
    photometric: &str,
    planar_configuration: Option<u16>,
    rows: u16,
    cols: u16,
    samples: Vec<u8>,
) {
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.4"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.4101"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("OT"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from("2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(samples_per_pixel),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from(photometric),
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
        PrimitiveValue::from(rows),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(cols),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("0.5\\0.25"),
    ));
    obj.put(DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("2.0"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from("1\\2\\3"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1\\0\\0\\0\\1\\0"),
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
        PrimitiveValue::U8(SmallVec::from_vec(samples)),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.4")
            .media_storage_sop_instance_uid("2.25.4101")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta")
    .write_to_file(path)
    .expect("write RGB multiframe");
}

#[test]
fn read_dicom_color_multiframe_preserves_interleaved_rgb_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rgb_mf.dcm");
    let expected = vec![
        255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0,
    ];
    write_multiframe(
        &path,
        3,
        "RGB",
        Some(0),
        expected.iter().map(|v| *v as u8).collect(),
    );
    let device = <B as Backend>::Device::default();

    let volume = read_dicom_color_multiframe::<B, _>(&path, &device).expect("load RGB MF");

    assert_eq!(volume.shape(), [2, 1, 2, 3]);
    assert_eq!(volume.origin().to_array(), [1.0, 2.0, 3.0]);
    assert_eq!(
        [
            volume.spacing()[0],
            volume.spacing()[1],
            volume.spacing()[2]
        ],
        [2.0, 0.5, 0.25]
    );
    volume.with_data_slice(|samples| {
        assert_eq!(samples, expected.as_slice());
    });
}

#[test]
fn read_dicom_color_multiframe_rejects_scalar_samples() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("scalar_mf.dcm");
    write_multiframe(&path, 1, "MONOCHROME2", None, vec![1, 2, 3, 4]);
    let device = <B as Backend>::Device::default();

    let err = read_dicom_color_multiframe::<B, _>(&path, &device).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SamplesPerPixel=1"),
        "expected scalar rejection, got {msg}"
    );
}

#[test]
fn read_dicom_color_multiframe_rejects_planar_rgb() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("planar_mf.dcm");
    write_multiframe(&path, 3, "RGB", Some(1), vec![0; 12]);
    let device = <B as Backend>::Device::default();

    let err = read_dicom_color_multiframe::<B, _>(&path, &device).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("PlanarConfiguration=0") && msg.contains("declares 1"),
        "expected planar rejection, got {msg}"
    );
}

#[test]
fn read_dicom_color_multiframe_rejects_hostile_dimensions_without_oom() {
    // Rows=60000, Columns=60000, NumberOfFrames=2 (declares ~21.6 billion RGB
    // samples / ~86 GiB as f32) but the PixelData element supplies only 12
    // bytes. Since the eager `vec![0.0; total_samples]` zero-fill was replaced
    // by a capped, incrementally-grown buffer, the native decode must fail with
    // a typed "out of range" error at the first frame rather than attempting
    // a multi-gigabyte allocation.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("hostile_mf.dcm");
    write_multiframe_with_dims(&path, 3, "RGB", Some(0), 60000, 60000, vec![0u8; 12]);
    let device = <B as Backend>::Device::default();

    let err = read_dicom_color_multiframe::<B, _>(&path, &device)
        .expect_err("hostile multiframe dimensions must error, not OOM");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("out of range"),
        "expected a bounds error, got: {msg}"
    );
}
