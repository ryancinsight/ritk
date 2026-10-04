//! Tests for the PNG **write** half.
//!
//! The property under test is the module contract: what these write, this
//! crate's own reader must read back.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::*;
use crate::{read_png_series, read_png_to_image};
use ritk_spatial::{Direction, Point, Spacing};

fn backend() -> coeus_core::SequentialBackend {
    coeus_core::SequentialBackend::new()
}

fn ramp(depth: usize, rows: usize, cols: usize) -> Vec<f32> {
    (0..depth * rows * cols).map(|i| i as f32).collect()
}

fn make(
    depth: usize,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
) -> Image<f32, coeus_core::SequentialBackend, 3> {
    Image::from_flat_on(
        data,
        [depth, rows, cols],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &backend(),
    )
    .expect("construct image")
}

#[test]
fn single_slice_round_trips_through_this_crates_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("slice.png");
    let original = ramp(1, 4, 5);
    let image = make(1, 4, 5, original.clone());

    write_png(&image, &path).expect("write single slice");

    let decoded = read_png_to_image(&path, &backend()).expect("read back");
    assert_eq!(decoded.shape(), [1, 4, 5]);

    // PNG stores 8-bit and the writer maps [min, max] onto [0, 255], so the
    // contract is the *stored* value, not the original float. `stored == round((v - min) / range * 255)`.
    let minimum = original.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = original.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = maximum - minimum;
    for (actual, expected) in decoded.data_slice().unwrap().iter().zip(original.iter()) {
        let want = (((expected - minimum) / range) * 255.0).round();
        assert!(
            (actual - want).abs() <= 1.0,
            "stored {actual} differs from expected {want} for source {expected}"
        );
    }
}

#[test]
fn volume_round_trips_through_this_crates_series_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series");
    let original = ramp(3, 4, 5);
    let image = make(3, 4, 5, original.clone());

    write_png_volume(&image, &dir).expect("write volume");

    let decoded = read_png_series(&dir, &backend()).expect("read series back");
    assert_eq!(
        decoded.shape(),
        [3, 4, 5],
        "depth and geometry must survive"
    );

    let minimum = original.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = original.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = maximum - minimum;
    for (actual, expected) in decoded.data_slice().unwrap().iter().zip(original.iter()) {
        let want = (((expected - minimum) / range) * 255.0).round();
        assert!(
            (actual - want).abs() <= 1.0,
            "stored {actual} differs from expected {want}"
        );
    }
}

#[test]
fn slice_filenames_sort_into_written_order_past_nine() {
    // Zero-padded names exist so a natural sort recovers slice order; a
    // twelve-slice volume is what breaks unpadded naming (slice-10 < slice-2).
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("many");
    let image = make(12, 2, 2, ramp(12, 2, 2));
    write_png_volume(&image, &dir).expect("write volume");

    let decoded = read_png_series(&dir, &backend()).expect("read series back");
    assert_eq!(decoded.shape(), [12, 2, 2]);

    let first = crate::write::slice_filename(0);
    let tenth = crate::write::slice_filename(10);
    assert_eq!(first, "slice-0000.png");
    assert_eq!(tenth, "slice-0010.png");
    assert!(first < tenth, "written order must match natural sort");
}

#[test]
fn constant_image_does_not_become_all_black() {
    // The degenerate-window case: every sample identical means the range is
    // zero, and dividing by it yields NaN, which would silently write zeros.
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("flat.png");
    let image = make(1, 4, 4, vec![7.0; 16]);

    write_png(&image, &path).expect("write constant slice");

    let decoded = read_png_to_image(&path, &backend()).expect("read back");
    assert_eq!(decoded.shape(), [1, 4, 4]);
    for (index, value) in decoded.data_slice().unwrap().iter().enumerate() {
        assert!(
            value.is_finite(),
            "constant image produced a non-finite sample at {index}"
        );
    }
}

#[test]
fn single_slice_writer_rejects_a_volume() {
    // Writing a depth-3 volume through the single-slice entry point would
    // silently drop frames if it were allowed.
    let tmp = tempfile::tempdir().expect("tempdir");
    let image = make(3, 2, 2, ramp(3, 2, 2));
    let error = write_png(&image, tmp.path().join("x.png")).unwrap_err();
    assert!(
        format!("{error:#}").contains("single slice"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn zero_dimension_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let image = make(1, 0, 4, Vec::new());
    let error = write_png_volume(&image, tmp.path().join("empty")).unwrap_err();
    assert!(
        format!("{error:#}").contains("must have every dimension > 0"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn encode_png_slice_matches_the_written_series() {
    // The in-memory encoder and the on-disk writer share the window, so a caller
    // composing a montage cannot silently disagree with the series it also wrote.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series");
    let original = ramp(2, 3, 3);
    let image = make(2, 3, 3, original.clone());

    write_png_volume(&image, &dir).expect("write volume");

    for index in 0..2 {
        let encoded = encode_png_slice(&image, index).expect("encode slice");
        let on_disk = image::open(dir.join(crate::write::slice_filename(index)))
            .expect("open written slice")
            .to_luma8();
        assert_eq!(
            encoded.as_raw(),
            on_disk.as_raw(),
            "slice {index}: in-memory encoding differs from the written file"
        );
    }
}

#[test]
fn encode_png_slice_rejects_an_out_of_range_index() {
    let image = make(2, 2, 2, ramp(2, 2, 2));
    let error = encode_png_slice(&image, 2).unwrap_err();
    assert!(
        format!("{error:#}").contains("out of range"),
        "unexpected error: {error:#}"
    );
}
