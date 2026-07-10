use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

fn make_test_image(
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, SequentialBackend, 3> {
    let total = dims.iter().product();
    let values = (0..total).map(|index| index as f32).collect();
    Image::from_flat_on(
        values,
        dims,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("valid native test image")
}

#[test]
fn write_minc_produces_hdf5_file() {
    let image = make_test_image([4, 4, 4], [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mnc");

    write_minc(&image, &path, &SequentialBackend).expect("write MINC");

    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[0..8], b"\x89HDF\r\n\x1a\n");
    assert!(bytes.len() > 44, "file must contain more than a superblock");
}

#[test]
fn write_minc_eof_field_matches_file_size() {
    let image = make_test_image([2, 2, 2], [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("eof.mnc");
    write_minc(&image, &path, &SequentialBackend).unwrap();
    let bytes = std::fs::read(path).unwrap();
    let eof_addr = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
    assert_eq!(eof_addr, bytes.len() as u64);
}

#[test]
fn write_then_read_round_trips_values_and_metadata() {
    let image = make_test_image([2, 3, 4], [-1.0, -2.0, -3.0], [0.5, 1.5, 2.5]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip.mnc");
    write_minc(&image, &path, &SequentialBackend).unwrap();

    let loaded = crate::read_minc(&path, &SequentialBackend).unwrap();

    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
    assert_eq!(loaded.origin(), image.origin());
    assert_eq!(loaded.spacing(), image.spacing());
    assert_eq!(loaded.direction(), image.direction());
}

#[test]
fn read_minc_rejects_shape_exceeding_backed_data() {
    use crate::hdf5_binary::write_minc2_hdf5;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("forged.mnc");
    write_minc2_hdf5(
        &path,
        &[0_u8; 8 * 4],
        [64, 64, 64],
        [0.0; 3],
        [1.0; 3],
        &Direction::identity(),
    )
    .unwrap();

    let error = crate::read_minc(&path, &SequentialBackend)
        .expect_err("shape exceeding backed data must fail");
    assert!(
        format!("{error:#}").contains("voxel data"),
        "expected voxel-data read error, got {error:#}"
    );
}
