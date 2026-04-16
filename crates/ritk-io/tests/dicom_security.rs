use burn_ndarray::NdArray;
use ritk_io::{read_analyze, write_analyze};
use std::path::PathBuf;

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
