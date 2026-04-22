use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use dicom::core::Tag;
use dicom::object::open_file;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_io::{
    read_analyze, read_dicom_series_with_metadata, write_analyze, write_dicom_series_with_metadata,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn make_test_image(depth: usize, rows: usize, cols: usize, fill: f32) -> Image<NdArray<f32>, 3> {
    let device = Default::default();
    let data = vec![fill; depth * rows * cols];
    let tensor = Tensor::<NdArray<f32>, 3>::from_data(
        TensorData::new(data, Shape::new([depth, rows, cols])),
        &device,
    );
    Image::new(
        tensor,
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([0.5, 0.5, 2.5]),
        Direction::identity(),
    )
}

fn make_test_metadata() -> ritk_io::DicomReadMetadata {
    let mut private_tags = HashMap::new();
    private_tags.insert("0019,10AA".to_string(), "PRIVATE_SERIES_VALUE".to_string());
    private_tags.insert(
        "0029,10BB".to_string(),
        "PRIVATE_SERIES_VALUE_2".to_string(),
    );

    ritk_io::DicomReadMetadata {
        series_instance_uid: Some("1.2.3.4.5.6.789".to_string()),
        study_instance_uid: Some("1.2.3.4.5.6.100".to_string()),
        frame_of_reference_uid: Some("1.2.3.4.5.6.200".to_string()),
        series_description: Some("Test Series".to_string()),
        modality: Some("CT".to_string()),
        patient_id: Some("PAT001".to_string()),
        patient_name: Some("Test^Patient".to_string()),
        study_date: Some("20240101".to_string()),
        series_date: Some("20240102".to_string()),
        series_time: Some("123456".to_string()),
        dimensions: [4, 4, 3],
        spacing: [0.5, 0.5, 2.5],
        origin: [10.0, 20.0, 30.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some("MONOCHROME2".to_string()),
        slices: Vec::new(),
        private_tags,
        preservation: ritk_io::DicomPreservationSet::new(),
    }
}

#[test]
fn test_read_analyze_path_leak() {
    type TestBackend = NdArray<f32>;
    let device = Default::default();

    let non_existent_path = PathBuf::from("/non/existent/path/file.hdr");
    let result = read_analyze::<TestBackend, _>(&non_existent_path, &device);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();

    assert!(
        !err_msg.contains(non_existent_path.to_string_lossy().as_ref()),
        "Error message leaks path: {}",
        err_msg
    );
}

#[test]
fn test_dicom_write_preserves_private_tags_and_metadata() {
    let meta = make_test_metadata();
    let image = make_test_image(3, 4, 4, 42.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("dicom_series");

    write_dicom_series_with_metadata(&path, &image, Some(&meta))
        .expect("metadata write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");

    let ipp = obj
        .element(Tag(0x0020, 0x0032))
        .expect("IPP tag must exist");
    let ipp_vals: Vec<f64> = ipp
        .to_str()
        .unwrap()
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ipp_vals, vec![10.0, 20.0, 30.0]);

    let iop = obj
        .element(Tag(0x0020, 0x0037))
        .expect("IOP tag must exist");
    let iop_vals: Vec<f64> = iop
        .to_str()
        .unwrap()
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(iop_vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let ps = obj
        .element(Tag(0x0028, 0x0030))
        .expect("PixelSpacing must exist");
    let ps_vals: Vec<f64> = ps
        .to_str()
        .unwrap()
        .split('\\')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ps_vals, vec![0.5, 0.5]);

    let st = obj
        .element(Tag(0x0018, 0x0050))
        .expect("SliceThickness must exist");
    assert_eq!(st.to_str().unwrap().trim(), "2.500000");

    let tag_a = obj
        .element(Tag(0x0019, 0x10AA))
        .expect("private tag A must exist");
    let tag_b = obj
        .element(Tag(0x0029, 0x10BB))
        .expect("private tag B must exist");
    assert_eq!(tag_a.to_str().unwrap().trim(), "PRIVATE_SERIES_VALUE");
    assert_eq!(tag_b.to_str().unwrap().trim(), "PRIVATE_SERIES_VALUE_2");

    let (_, loaded_meta) =
        read_dicom_series_with_metadata::<NdArray<f32>, _>(&path, &Default::default())
            .expect("DICOM round-trip read must succeed");
    assert_eq!(
        loaded_meta.series_instance_uid.as_deref(),
        Some("1.2.3.4.5.6.789")
    );
    assert_eq!(
        loaded_meta.study_instance_uid.as_deref(),
        Some("1.2.3.4.5.6.100")
    );
    assert_eq!(
        loaded_meta.frame_of_reference_uid.as_deref(),
        Some("1.2.3.4.5.6.200")
    );
    assert_eq!(
        loaded_meta.series_description.as_deref(),
        Some("Test Series")
    );
    assert_eq!(loaded_meta.modality.as_deref(), Some("CT"));
    assert_eq!(loaded_meta.patient_id.as_deref(), Some("PAT001"));
    assert_eq!(loaded_meta.patient_name.as_deref(), Some("Test^Patient"));
    assert_eq!(loaded_meta.study_date.as_deref(), Some("20240101"));
    assert_eq!(loaded_meta.series_date.as_deref(), Some("20240102"));
    assert_eq!(loaded_meta.series_time.as_deref(), Some("123456"));
    assert_eq!(loaded_meta.bits_allocated, Some(16));
    assert_eq!(loaded_meta.bits_stored, Some(16));
    assert_eq!(loaded_meta.high_bit, Some(15));
    assert_eq!(
        loaded_meta.photometric_interpretation.as_deref(),
        Some("MONOCHROME2")
    );
    // Private tag (0019,10AA) is captured in slice preservation (not in the private_tags map,
    // which is no longer populated by the reader -- preservation is the canonical path).
    let priv_node = loaded_meta
        .slices
        .first()
        .and_then(|s| s.preservation.object.get(ritk_io::DicomTag::new(0x0019, 0x10AA)));
    assert!(priv_node.is_some(), "private tag (0019,10AA) must appear in slice preservation set");
    assert_eq!(
        priv_node.and_then(|n| n.value.as_text()).map(str::trim),
        Some("PRIVATE_SERIES_VALUE"),
        "private tag value must survive round-trip through preservation"
    );
}

#[test]
fn test_write_analyze_path_leak() {
    type TestBackend = NdArray<f32>;
    let device = Default::default();

    let non_existent_path = PathBuf::from("/non/existent/path/output.hdr");
    let image = {
        use burn::tensor::{Shape, Tensor, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};

        let data = TensorData::new(vec![0.0f32], Shape::new([1, 1, 1]));
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    };

    let result = write_analyze(&non_existent_path, &image);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();

    assert!(
        !err_msg.contains(non_existent_path.to_string_lossy().as_ref()),
        "Error message leaks path: {}",
        err_msg
    );
}
