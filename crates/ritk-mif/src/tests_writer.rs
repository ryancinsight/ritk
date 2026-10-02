//! Writer tests — round-trip tests live in `tests_reader.rs`.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

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
    let result = crate::write_mif_series::<f32, SequentialBackend, _>(
        &path,
        &[] as &[Image<f32, SequentialBackend, 3>],
        &backend,
    );
    let _ = std::fs::remove_file(&path);
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
    assert!(
        result.unwrap_err().to_string().contains("differs"),
        "error should mention shape difference"
    );
}

/// The inline offset is the smallest multiple of four that holds the whole
/// header, the `file` line's own digits included, for every header length
/// across the decimal-digit boundaries of the offset.
///
/// The file line is `file: . <offset>\nEND\n`, so the header ends at
/// `preceding + 8 + digits(offset) + 5`; the offset is correct when it is a
/// multiple of four, is not before that end, and a multiple of four one step
/// lower would be.
#[test]
fn inline_data_offset_is_the_smallest_aligned_offset_past_the_header() {
    use super::{inline_data_offset, FILE_LINE_FRAME_LEN};
    assert_eq!(FILE_LINE_FRAME_LEN, 13);
    for preceding in 0..2_000_usize {
        let offset = inline_data_offset(preceding);
        let header_end = preceding + FILE_LINE_FRAME_LEN + offset.to_string().len();
        assert_eq!(offset % 4, 0, "preceding {preceding}: offset {offset}");
        assert!(
            offset >= header_end,
            "preceding {preceding}: offset {offset} lies inside a header ending at {header_end}"
        );
        assert!(
            offset - 4 < header_end,
            "preceding {preceding}: offset {offset} wastes a whole alignment unit"
        );
    }
}
