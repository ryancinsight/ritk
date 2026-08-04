//! Writer tests — round-trip tests live in `reader_tests.rs`.

use coeus_core::SequentialBackend;

/// A temp path unique across concurrently running test processes.
///
/// A timestamp alone is not enough: nextest runs test binaries in parallel,
/// clock granularity is coarse on some platforms, and two tests landing in the
/// same tick then read each other's file. The pid separates processes, the
/// counter separates calls within one.
fn unique_temp_path(stem: &str, extension: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{pid}_{nanos:016x}_{seq}.{extension}"))
}

#[test]
fn write_mif_series_rejects_empty_volumes() {
    use ritk_image::Image;
    let backend = SequentialBackend;
    let path = unique_temp_path("empty_series_test", "mif");
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

    let path = unique_temp_path("hetero_series_test", "mif");
    let result = crate::write_mif_series(&path, &[img1, img2], &backend);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("differs"),
        "error should mention shape difference"
    );
}
