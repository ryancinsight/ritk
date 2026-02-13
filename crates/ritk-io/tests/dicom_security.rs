use ritk_io::dicom_io::{DicomSeriesInfo, load_dicom_series};
use burn_ndarray::NdArray;
use std::path::PathBuf;

#[test]
fn test_load_series_path_leak() {
    type TestBackend = NdArray<f32>;
    let device = Default::default();

    let non_existent_path = PathBuf::from("/non/existent/path/file.dcm");
    let series = DicomSeriesInfo {
        series_instance_uid: "uid".to_string(),
        series_description: "desc".to_string(),
        modality: "mod".to_string(),
        patient_id: "pid".to_string(),
        file_paths: vec![non_existent_path.clone()],
    };

    let result = load_dicom_series::<TestBackend>(&series, &device);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();

    // Check if the error message contains the path
    // After fix, we expect this assertion to pass (path is NOT in message)
    assert!(!err_msg.contains(non_existent_path.to_str().unwrap()), "Error message leaks path: {}", err_msg);
}
