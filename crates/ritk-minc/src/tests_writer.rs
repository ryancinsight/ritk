use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_test_image(
    nz: usize,
    ny: usize,
    nx: usize,
    start: [f64; 3],
    step: [f64; 3],
) -> Image<B, 3> {
    let total = nz * ny * nx;
    let values: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let data = TensorData::new(values, Shape::new([nz, ny, nx]));
    let device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(data, &device);
    let origin = Point::new(start);
    let spacing = Spacing::new(step);
    let direction = Direction::identity();
    Image::new(tensor, origin, spacing, direction)
}

#[test]
fn write_minc_produces_file() {
    let image = make_test_image(4, 4, 4, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mnc");
    let result = write_minc::<B, _>(&image, &path);
    assert!(result.is_ok(), "write_minc failed: {:?}", result.err());
    assert!(path.exists(), "file was not created");
    let metadata = std::fs::metadata(&path).unwrap();
    assert!(
        metadata.len() > 44,
        "file contains more than just a superblock"
    );
}

#[test]
fn write_minc_file_starts_with_hdf5_signature() {
    let image = make_test_image(2, 2, 2, [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sig.mnc");
    write_minc::<B, _>(&image, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..8], b"\x89HDF\r\n\x1a\n", "missing HDF5 signature");
}

#[test]
fn write_minc_voxel_data_present_in_file() {
    let nz = 2usize;
    let ny = 3usize;
    let nx = 4usize;
    let image = make_test_image(nz, ny, nx, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("voxel.mnc");
    write_minc::<B, _>(&image, &path).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();
    // Voxel at index 0 should be 0.0f32 and at index 1 should be 1.0f32.
    let expected_0 = 0.0f32.to_le_bytes();
    let expected_1 = 1.0f32.to_le_bytes();
    let found_0 = file_bytes.windows(4).any(|w| w == expected_0);
    let found_1 = file_bytes.windows(4).any(|w| w == expected_1);
    assert!(found_0, "voxel value 0.0 not found in output");
    assert!(found_1, "voxel value 1.0 not found in output");
}

#[test]
fn write_minc_eof_field_matches_file_size() {
    let image = make_test_image(2, 2, 2, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("eof.mnc");
    write_minc::<B, _>(&image, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    // EOF address is stored at bytes 28..36 of superblock v2.
    let eof_bytes: [u8; 8] = bytes[28..36].try_into().unwrap();
    let eof_addr = u64::from_le_bytes(eof_bytes);
    assert_eq!(eof_addr, bytes.len() as u64, "EOF address mismatch");
}

#[test]
fn write_minc_then_read_minc_round_trips_voxels() {
    use crate::read_minc;

    // values 0..8 over a 2×2×2 cube; assert value preservation order-agnostically
    // to stay clear of the MINC dimorder axis-order convention. This is the first
    // end-to-end write→read coverage and guards the HDF5 v1 object-header message
    // alignment and datatype-descriptor encoding the consus reader requires.
    let image = make_test_image(2, 2, 2, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip.mnc");
    write_minc::<B, _>(&image, &path).expect("write MINC");

    let device = Default::default();
    let read = read_minc::<B, _>(&path, &device).expect("read MINC");
    assert_eq!(read.shape(), [2, 2, 2]);
    read.with_data_slice(|loaded| {
        let mut got = loaded.to_vec();
        got.sort_by(|a, b| a.partial_cmp(b).expect("no NaN voxels"));
        let expected: Vec<f32> = (0..8u32).map(|i| i as f32).collect();
        assert_eq!(
            got, expected,
            "all 8 voxel values preserved through round-trip"
        );
    });
}
