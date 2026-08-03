//! MINC writer tests migrated to the Atlas-native (Coeus) path — ADR 0002.

use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

#[path = "scaled_fixture.rs"]
mod scaled_fixture;

fn make_test_image(
    nz: usize,
    ny: usize,
    nx: usize,
    start: [f64; 3],
    step: [f64; 3],
) -> Image<f32, B, 3> {
    let total = nz * ny * nx;
    let values: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let origin = Point::new(start);
    let spacing = Spacing::new(step);
    let direction = Direction::identity();
    Image::from_flat_on(
        values,
        [nz, ny, nx],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("valid image dimensions")
}

#[test]
fn write_minc_produces_file() {
    let backend = SequentialBackend;
    let image = make_test_image(4, 4, 4, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("test.mnc");
    let result = crate::write_minc(&image, &path, &backend);
    assert!(result.is_ok(), "write_minc failed: {:?}", result.err());
    assert!(path.exists(), "file was not created");
    let metadata = std::fs::metadata(&path).expect("infallible: validated precondition");
    assert!(
        metadata.len() > 44,
        "file contains more than just a superblock"
    );
}

#[test]
fn write_minc_file_starts_with_hdf5_signature() {
    let backend = SequentialBackend;
    let image = make_test_image(2, 2, 2, [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]);
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("sig.mnc");
    crate::write_minc(&image, &path, &backend).expect("infallible: validated precondition");
    let bytes = std::fs::read(&path).expect("infallible: validated precondition");
    assert_eq!(&bytes[0..8], b"\x89HDF\r\n\x1a\n", "missing HDF5 signature");
}

#[test]
fn write_minc_voxel_data_present_in_file() {
    let backend = SequentialBackend;
    let nz = 2usize;
    let ny = 3usize;
    let nx = 4usize;
    let image = make_test_image(nz, ny, nx, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("voxel.mnc");
    crate::write_minc(&image, &path, &backend).expect("infallible: validated precondition");
    let file_bytes = std::fs::read(&path).expect("infallible: validated precondition");
    let expected_0 = 0.0f32.to_le_bytes();
    let expected_1 = 1.0f32.to_le_bytes();
    let found_0 = file_bytes.windows(4).any(|w| w == expected_0);
    let found_1 = file_bytes.windows(4).any(|w| w == expected_1);
    assert!(found_0, "voxel value 0.0 not found in output");
    assert!(found_1, "voxel value 1.0 not found in output");
}

#[test]
fn write_minc_eof_field_matches_file_size() {
    let backend = SequentialBackend;
    let image = make_test_image(2, 2, 2, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("eof.mnc");
    crate::write_minc(&image, &path, &backend).expect("infallible: validated precondition");
    let bytes = std::fs::read(&path).expect("infallible: validated precondition");
    let eof_bytes: [u8; 8] = bytes[28..36]
        .try_into()
        .expect("infallible: validated precondition");
    let eof_addr = u64::from_le_bytes(eof_bytes);
    assert_eq!(eof_addr, bytes.len() as u64, "EOF address mismatch");
}

#[test]
fn write_minc_then_read_minc_round_trips_voxels() {
    let backend = SequentialBackend;
    let image = make_test_image(2, 2, 2, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("roundtrip.mnc");
    crate::write_minc(&image, &path, &backend).expect("write MINC");

    let read = crate::read_minc(&path, &backend).expect("read MINC");
    assert_eq!(read.shape(), [2, 2, 2]);
    let loaded = read.data_slice().expect("contiguous host data");
    let mut got = loaded.to_vec();
    got.sort_by(|a, b| a.partial_cmp(b).expect("no NaN voxels"));
    let expected: Vec<f32> = (0..8u32).map(|i| i as f32).collect();
    assert_eq!(
        got, expected,
        "all 8 voxel values preserved through round-trip"
    );
}

#[test]
fn read_minc_rejects_shape_exceeding_backed_data() {
    use crate::hdf5_binary::write_minc2_hdf5;

    let backend = SequentialBackend;
    let dir = tempfile::tempdir().expect("infallible: validated precondition");
    let path = dir.path().join("forged.mnc");
    write_minc2_hdf5(
        &path,
        &[0.0_f32; 8],
        [64, 64, 64],
        [0.0; 3],
        [1.0; 3],
        &Direction::identity(),
    )
    .expect("infallible: validated precondition");

    let error = crate::read_minc(&path, &backend)
        .expect_err("shape exceeding backed data must error, not OOM");
    assert!(
        format!("{error:#}").contains("voxel data"),
        "expected voxel-data read error, got {error:#}"
    );
}

#[test]
fn write_minc_rejects_non_finite_geometry_before_file_creation() {
    let backend = SequentialBackend;
    let values = vec![1.0_f32; 8];
    let image = Image::from_flat_on(
        values,
        [2, 2, 2],
        Point::new([f64::NAN, 0.0, 0.0]),
        Spacing::uniform(1.0),
        Direction::identity(),
        &backend,
    )
    .expect("valid storage and shape");
    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("invalid-geometry.mnc");

    let error =
        crate::write_minc(&image, &path, &backend).expect_err("non-finite origin must be rejected");

    assert!(
        error.to_string().contains("origin axis 0"),
        "unexpected error: {error:#}"
    );
    assert!(!path.exists(), "preflight failure must not create a file");
}

#[test]
fn write_minc_round_trips_across_stream_chunks() {
    let backend = SequentialBackend;
    let shape = [5, 64, 64];
    let voxel_count = shape.into_iter().product();
    let values: Vec<f32> = (0..voxel_count)
        .map(|index| {
            let index = u16::try_from(index).expect("test voxel index fits u16");
            f32::from(index).mul_add(0.25, -1_024.0)
        })
        .collect();
    let image = Image::from_flat_on(
        values.clone(),
        shape,
        Point::new([4.5, -8.0, 12.25]),
        Spacing::new([1.5, 0.75, 0.5]),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        &backend,
    )
    .expect("valid streamed image");
    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("streamed.mnc");

    crate::write_minc(&image, &path, &backend).expect("write streamed MINC2 volume");
    let decoded = crate::read_minc(&path, &backend).expect("read streamed MINC2 volume");

    assert_eq!(decoded.shape(), shape);
    assert_eq!(decoded.origin(), image.origin());
    assert_eq!(decoded.spacing(), image.spacing());
    assert_eq!(decoded.direction(), image.direction());
    assert_eq!(decoded.data_slice().expect("host data"), values);
}

#[test]
fn geometry_preflight_rejects_unrepresentable_axis_length() {
    let oversized_axis = usize::try_from(i32::MAX)
        .expect("usize represents i32::MAX")
        .checked_add(1)
        .expect("supported targets represent i32::MAX + 1");
    let error = super::validate_geometry(
        [oversized_axis, 1, 1],
        &Point::origin(),
        &Spacing::uniform(1.0),
        &Direction::identity(),
    )
    .expect_err("MINC2 length attribute is i32");

    assert!(
        error
            .to_string()
            .contains("exceeds the i32 length attribute"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn read_minc_applies_per_slice_integer_scaling() {
    use scaled_fixture::{write_scaled_integer_fixture, ImageRangeFixture};

    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("scaled-int16.mnc");
    let stored = [0_i16, 25, 50, 100, 0, 25, 50, 100];
    write_scaled_integer_fixture(
        &path,
        &stored,
        [2, 2, 2],
        [0, 100],
        ImageRangeFixture::Complete {
            minima: &[-1_000.0, 0.0],
            maxima: &[1_000.0, 200.0],
        },
    )
    .expect("write scaled fixture");

    let image = crate::read_minc(&path, &SequentialBackend).expect("read scaled fixture");
    assert_eq!(image.shape(), [2, 2, 2]);
    assert_eq!(
        image.data_slice().expect("host data"),
        [-1_000.0, -500.0, 0.0, 1_000.0, 0.0, 50.0, 100.0, 200.0]
    );
}

#[test]
fn read_minc_preserves_per_slice_scaling_across_stream_chunks() {
    use scaled_fixture::{write_scaled_integer_fixture, ImageRangeFixture};

    const SHAPE: [usize; 3] = [2, 33, 64];
    const SLICE_LENGTH: usize = SHAPE[1] * SHAPE[2];
    const STORED_PATTERN: [i16; 4] = [0, 25, 50, 100];
    const FIRST_SLICE_PATTERN: [f32; 4] = [-1_000.0, -500.0, 0.0, 1_000.0];
    const SECOND_SLICE_PATTERN: [f32; 4] = [0.0, 50.0, 100.0, 200.0];

    let stored: Vec<i16> = (0..SHAPE.iter().product())
        .map(|index| STORED_PATTERN[index % STORED_PATTERN.len()])
        .collect();
    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("scaled-int16-stream-boundary.mnc");
    write_scaled_integer_fixture(
        &path,
        &stored,
        SHAPE,
        [0, 100],
        ImageRangeFixture::Complete {
            minima: &[-1_000.0, 0.0],
            maxima: &[1_000.0, 200.0],
        },
    )
    .expect("write scaled stream-boundary fixture");

    let image =
        crate::read_minc(&path, &SequentialBackend).expect("read scaled stream-boundary fixture");
    let expected: Vec<f32> = (0..stored.len())
        .map(|index| {
            let pattern = if index < SLICE_LENGTH {
                FIRST_SLICE_PATTERN
            } else {
                SECOND_SLICE_PATTERN
            };
            pattern[index % pattern.len()]
        })
        .collect();
    assert_eq!(image.data_slice().expect("host data"), expected);
}

#[test]
fn read_minc_uses_default_real_range_when_image_ranges_are_absent() {
    use scaled_fixture::{write_scaled_integer_fixture, ImageRangeFixture};

    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("default-range.mnc");
    write_scaled_integer_fixture(
        &path,
        &[0_i16, 25, 50, 100],
        [1, 2, 2],
        [0, 100],
        ImageRangeFixture::Omitted,
    )
    .expect("write default-range fixture");

    let image = crate::read_minc(&path, &SequentialBackend).expect("read default-range fixture");
    assert_eq!(
        image.data_slice().expect("host data"),
        [0.0, 0.25, 0.5, 1.0]
    );
}

#[test]
fn read_minc_rejects_incomplete_image_range_pair() {
    use scaled_fixture::{write_scaled_integer_fixture, ImageRangeFixture};

    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("missing-image-max.mnc");
    write_scaled_integer_fixture(
        &path,
        &[0_i16, 25, 50, 100],
        [1, 2, 2],
        [0, 100],
        ImageRangeFixture::MinimumOnly { minima: &[-100.0] },
    )
    .expect("write malformed range fixture");

    let error = crate::read_minc(&path, &SequentialBackend)
        .expect_err("an incomplete image-range pair must fail");
    assert!(
        error.to_string().contains("image-max is missing"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn read_minc_rejects_stored_integer_outside_valid_range() {
    use scaled_fixture::{write_scaled_integer_fixture, ImageRangeFixture};

    let directory = tempfile::tempdir().expect("create temporary directory");
    let path = directory.path().join("invalid-stored-value.mnc");
    write_scaled_integer_fixture(
        &path,
        &[0_i16, 25, 101, 100],
        [1, 2, 2],
        [0, 100],
        ImageRangeFixture::Complete {
            minima: &[-100.0],
            maxima: &[300.0],
        },
    )
    .expect("write out-of-range fixture");

    let error = crate::read_minc(&path, &SequentialBackend)
        .expect_err("out-of-range stored values must not be silently mapped");
    assert!(
        format!("{error:#}").contains("stored voxel 2 value 101"),
        "unexpected error: {error:#}"
    );
}
