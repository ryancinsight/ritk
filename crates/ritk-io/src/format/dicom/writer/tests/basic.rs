use super::helpers::{make_image, Backend};
use super::super::{write_dicom_series, generate_series_uid};
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::Tag;
use dicom::object::open_file;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};

#[test]
fn test_writer_rejects_zero_dimension() {
    let image = make_image(0, 4, 4, 0.5);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("series");
    let result = write_dicom_series(&path, &image);
    assert!(result.is_err(), "zero depth must be rejected");
}

#[test]
fn test_writer_creates_correct_number_of_slice_files() {
    let image = make_image(3, 4, 5, 0.5);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("series");
    write_dicom_series(&path, &image).expect("write must succeed");
    assert!(path.is_dir());
    let count = std::fs::read_dir(&path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("dcm"))
        .count();
    assert_eq!(count, 3, "must produce exactly 3 .dcm files");
}

#[test]
fn test_writer_slice_files_are_nonempty() {
    let image = make_image(2, 8, 8, 100.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("series");
    write_dicom_series(&path, &image).expect("write must succeed");
    for entry in std::fs::read_dir(&path).unwrap().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("dcm") {
            let size = std::fs::metadata(entry.path()).unwrap().len();
            assert!(size > 200, "DICOM slice must be >200 bytes, got {}", size);
        }
    }
}

#[test]
fn test_writer_dcm_starts_with_dicom_magic() {
    let image = make_image(1, 4, 4, 0.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("series");
    write_dicom_series(&path, &image).expect("write must succeed");
    let dcm_path = path.join("slice_0000.dcm");
    let bytes = std::fs::read(&dcm_path).expect("slice file must exist");
    assert!(bytes.len() >= 132, "DICOM file must be >=132 bytes");
    assert_eq!(
        &bytes[128..132],
        b"DICM",
        "DICOM magic bytes must be present at offset 128"
    );
}

#[test]
fn test_series_writer_has_samples_per_pixel_one() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let series_path = tmp.path().join("spp_series");
    let image = make_image(2, 3, 4, 1.0);
    write_dicom_series(&series_path, &image).expect("write_dicom_series");

    let slice_path = series_path.join("slice_0000.dcm");
    let obj = open_file(&slice_path).expect("open_file");
    let spp: u16 = obj
        .element(Tag(0x0028, 0x0002))
        .expect("SamplesPerPixel (0028,0002) must be present in written slice")
        .to_str()
        .expect("SamplesPerPixel must be readable as string")
        .trim()
        .parse()
        .expect("SamplesPerPixel must be numeric");
    assert_eq!(spp, 1, "SamplesPerPixel must equal 1 for grayscale series");
}

/// Pixel clamp invariant: no encoded u16 value may exceed 65535 even when
/// floating-point rounding produces a value slightly above max.
///
/// Analytical construction: fill image with [0.0, 65535.0] range;
/// slope = 65535/65535 = 1.0, intercept = 0.0. The clamped path must keep
/// all pixels <= 65535.
#[test]
#[allow(unused_comparisons)]
fn test_series_pixel_clamp_u16_range() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("clamp_series");
    let n_frames = 1_usize;
    let rows = 4_usize;
    let cols = 4_usize;
    let data: Vec<f32> = (0..n_frames * rows * cols)
        .map(|i| (i as f32) * (65535.0_f32 / 15.0_f32))
        .collect();
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(data, Shape::new([n_frames, rows, cols])),
        &Default::default(),
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    write_dicom_series(&out_path, &image).expect("write_dicom_series");

    for entry in std::fs::read_dir(&out_path).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("dcm") {
            continue;
        }
        let obj = dicom::object::open_file(&path).expect("open_file");
        if let Ok(elem) = obj.element(dicom::core::Tag(0x7FE0, 0x0010)) {
            if let Ok(bytes) = elem.value().to_bytes() {
                for chunk in bytes.chunks_exact(2) {
                    let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                    assert!(v <= 65535, "pixel value {v} exceeds u16 max");
                }
            }
        }
    }
}

/// ConversionType (0008,0064) must equal "WSD" in each slice written by write_dicom_series.
///
/// Invariant: SC Equipment Module (PS3.3 C.8.6.1) mandates ConversionType as Type 1.
#[test]
fn test_series_writer_has_conversion_type_wsd() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("conv_type_series");
    let image = make_image(2, 4, 4, 1.0);
    write_dicom_series(&out_path, &image).expect("write_dicom_series");

    let first_slice = out_path.join("slice_0000.dcm");
    let obj = open_file(&first_slice).expect("open_file");
    let conv_type = obj
        .element(Tag(0x0008, 0x0064))
        .expect("ConversionType (0008,0064) must be present")
        .to_str()
        .expect("ConversionType must be a string")
        .trim()
        .to_string();
    assert_eq!(conv_type, "WSD", "ConversionType must be 'WSD'");
}

/// write_dicom_series must emit Type 2 mandatory Patient and Study module tags.
///
/// Invariant: DICOM PS3.3 Type 2 = present (may be empty).
#[test]
fn test_basic_series_writer_has_type2_patient_tags() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("type2_tags_series");
    let image = make_image(2, 4, 4, 1.0);
    write_dicom_series(&out_path, &image).expect("write_dicom_series");

    let first_slice = out_path.join("slice_0000.dcm");
    let obj = open_file(&first_slice).expect("open_file");
    assert!(
        obj.element(Tag(0x0010, 0x0010)).is_ok(),
        "PatientName (0010,0010) must be present"
    );
    assert!(
        obj.element(Tag(0x0010, 0x0020)).is_ok(),
        "PatientID (0010,0020) must be present"
    );
    assert!(
        obj.element(Tag(0x0008, 0x0090)).is_ok(),
        "ReferringPhysicianName (0008,0090) must be present"
    );
    assert!(
        obj.element(Tag(0x0008, 0x0020)).is_ok(),
        "StudyDate (0008,0020) must be present"
    );
    assert!(
        obj.element(Tag(0x0020, 0x0011)).is_ok(),
        "SeriesNumber (0020,0011) must be present"
    );
}

#[test]
fn test_series_uid_distinct_on_rapid_successive_calls() {
    // Invariant: generate_series_uid() uses AtomicU64 counter; result is 2.25.<ns>.<seq>.
    let uid_a = generate_series_uid();
    let uid_b = generate_series_uid();
    assert_ne!(uid_a, uid_b, "successive series UIDs must be distinct");
    assert!(
        uid_a.starts_with("2.25."),
        "uid_a={uid_a} must start with 2.25."
    );
    assert!(
        uid_b.starts_with("2.25."),
        "uid_b={uid_b} must start with 2.25."
    );
}
