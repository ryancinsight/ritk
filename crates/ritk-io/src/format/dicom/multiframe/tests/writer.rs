use super::*;
use ritk_dicom::{parse_file_with, DicomRsBackend};

#[test]
fn test_write_multiframe_rejects_zero_dimension() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("zero.dcm");
    let data: Vec<f32> = vec![];
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new([1_usize, 0_usize, 5_usize])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let result = write_dicom_multiframe(&out_path, &image);
    assert!(
        result.is_err(),
        "write_dicom_multiframe must return Err for zero-row image"
    );
}

#[test]
fn test_multiframe_sop_class_is_mf_grayscale_word() {
    // Verifies that write_dicom_multiframe emits the Multi-Frame Grayscale Word SC SOP class
    // (1.2.840.10008.5.1.4.1.1.7.3) rather than Single-frame SC (1.2.840.10008.5.1.4.1.1.7).
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 2 * 3 * 4], Shape::new([2_usize, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert_eq!(
        info.sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.7.3"),
        "SOP class must be Multi-Frame Grayscale Word Secondary Capture"
    );
}

#[test]
fn test_written_multiframe_has_samples_per_pixel_one() {
    use dicom::object::open_file;
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_spp.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 2 * 4 * 5], Shape::new([2_usize, 4, 5])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");

    let obj = open_file(&out_path).expect("open_file");
    let spp: u16 = obj
        .element(dicom::core::Tag(0x0028, 0x0002))
        .expect("SamplesPerPixel (0028,0002) must be present")
        .to_str()
        .expect("SamplesPerPixel must be readable as string")
        .trim()
        .parse()
        .expect("SamplesPerPixel must be numeric");
    assert_eq!(
        spp, 1,
        "SamplesPerPixel must equal 1 for grayscale multi-frame"
    );
}

#[test]
fn test_writer_config_instance_number_propagated() {
    use dicom::object::open_file;
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_inst.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![5.0_f32; 1 * 2 * 3], Shape::new([1_usize, 2, 3])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    let config = MultiFrameWriterConfig {
        instance_number: 42,
        ..MultiFrameWriterConfig::default()
    };
    write_dicom_multiframe_with_config(&out_path, &image, &config).expect("write");

    let obj = open_file(&out_path).expect("open_file");
    let inst_num: u32 = obj
        .element(dicom::core::Tag(0x0020, 0x0013))
        .expect("InstanceNumber (0020,0013) must be present")
        .to_str()
        .expect("InstanceNumber must be readable")
        .trim()
        .parse()
        .expect("InstanceNumber must be numeric");
    assert_eq!(
        inst_num, 42,
        "InstanceNumber must match config.instance_number"
    );
}

#[test]
fn test_multiframe_has_conversion_type_wsd() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("conv_type.dcm");

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 1 * 2 * 2], Shape::new([1_usize, 2, 2])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");

    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    let conv_type = obj
        .element(Tag(0x0008, 0x0064))
        .expect("ConversionType (0008,0064) must be present")
        .to_str()
        .expect("ConversionType must be a string")
        .trim()
        .to_string();
    assert_eq!(
        conv_type, "WSD",
        "ConversionType must be 'WSD' (Workstation)"
    );
}

#[test]
fn test_multiframe_has_study_and_series_uids() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("uids.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 1 * 2 * 2], Shape::new([1_usize, 2, 2])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    let study_uid = obj
        .element(Tag(0x0020, 0x000D))
        .expect("StudyInstanceUID (0020,000D) must be present")
        .to_str()
        .expect("StudyInstanceUID must be a string")
        .trim()
        .to_string();
    let series_uid = obj
        .element(Tag(0x0020, 0x000E))
        .expect("SeriesInstanceUID (0020,000E) must be present")
        .to_str()
        .expect("SeriesInstanceUID must be a string")
        .trim()
        .to_string();
    assert!(!study_uid.is_empty(), "StudyInstanceUID must be non-empty");
    assert!(
        !series_uid.is_empty(),
        "SeriesInstanceUID must be non-empty"
    );
    assert_ne!(
        study_uid, series_uid,
        "StudyInstanceUID and SeriesInstanceUID must be distinct"
    );
}

#[test]
fn test_multiframe_has_type2_patient_study_series_tags() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("type2.dcm");
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![5.0_f32; 1 * 3 * 3], Shape::new([1_usize, 3, 3])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    // Assert presence (value may be empty per Type 2 semantics).
    obj.element(Tag(0x0010, 0x0010))
        .expect("PatientName (0010,0010) must be present");
    obj.element(Tag(0x0010, 0x0020))
        .expect("PatientID (0010,0020) must be present");
    obj.element(Tag(0x0008, 0x0020))
        .expect("StudyDate (0008,0020) must be present");
    obj.element(Tag(0x0008, 0x0090))
        .expect("ReferringPhysicianName (0008,0090) must be present");
    obj.element(Tag(0x0020, 0x0010))
        .expect("StudyID (0020,0010) must be present");
    obj.element(Tag(0x0020, 0x0011))
        .expect("SeriesNumber (0020,0011) must be present");
}
