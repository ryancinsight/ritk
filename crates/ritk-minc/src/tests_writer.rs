//! MINC writer tests migrated to the Atlas-native (Coeus) path — ADR 0002.

use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

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
        &[0_u8; 8 * 4],
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
