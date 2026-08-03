//! Writer tests — round-trip tests live in `reader_tests.rs`.

use coeus_core::SequentialBackend;

#[test]
fn write_mif_series_rejects_empty_volumes() {
    use ritk_image::Image;
    let backend = SequentialBackend;
    let path = std::env::temp_dir().join("empty_series_test.mif");
    let result = crate::write_mif_series::<SequentialBackend, _>(
        &path,
        &[] as &[Image<f32, SequentialBackend, 3>],
        &backend,
    );
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one volume"));
}

#[test]
fn write_mif_series_rejects_heterogeneous_shapes() {
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    let backend = SequentialBackend;

    let img1 = Image::from_flat_on(
        vec![0.0f32; 24],
        [2, 3, 4],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &backend,
    )
    .unwrap();

    let img2 = Image::from_flat_on(
        vec![0.0f32; 60],
        [3, 4, 5],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &backend,
    )
    .unwrap();

    let path = std::env::temp_dir().join("hetero_series_test.mif");
    let result = crate::write_mif_series(&path, &[img1, img2], &backend);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("differs"),
        "error should mention shape difference"
    );
}
