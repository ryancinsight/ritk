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

#[test]
fn native_read_minc_matches_burn_round_trip() {
    use crate::read_minc;
    use coeus_core::SequentialBackend;

    // Differential: write a known volume, then read it via both the Burn and
    // Coeus paths (which share decode_minc) and assert identical voxels. Also
    // assert order-agnostic value preservation against the source.
    let image = make_test_image(2, 2, 2, [0.0; 3], [1.0; 3]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coeus.mnc");
    write_minc::<B, _>(&image, &path).expect("write MINC");

    let device = Default::default();
    let burn = read_minc::<B, _>(&path, &device).expect("burn read");
    let coeus = crate::native::read_minc(&path, &SequentialBackend).expect("coeus read");

    assert_eq!(coeus.shape(), burn.shape(), "coeus and burn shapes match");
    let coeus_vals = coeus.data_slice().expect("contiguous host data");
    burn.with_data_slice(|burn_vals| {
        assert_eq!(coeus_vals, burn_vals, "coeus and burn voxels identical");
        let mut sorted = burn_vals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN voxels"));
        let expected: Vec<f32> = (0..8u32).map(|i| i as f32).collect();
        assert_eq!(sorted, expected, "all 8 voxel values preserved");
    });
}

#[test]
fn read_minc_rejects_shape_exceeding_backed_data() {
    use crate::hdf5_binary::write_minc2_hdf5;
    use crate::read_minc;

    // Forge a MINC2 file whose image dataset shape claims 64×64×64 voxels
    // (1 MiB of f32) but whose contiguous data region backs only 8 voxels.
    // The reader derives the read size from the dataspace, so the bounded
    // voxel read (Sprint 447 `read_bounded_with`) must surface a truncation
    // error rather than over-reading or OOM-ing.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("forged.mnc");
    let tiny_data = vec![0u8; 8 * 4]; // 8 f32 samples
    write_minc2_hdf5(
        &path,
        &tiny_data,
        [64, 64, 64],
        [0.0; 3],
        [1.0; 3],
        &Direction::identity(),
    )
    .expect("write forged MINC");

    let device = Default::default();
    let err = read_minc::<B, _>(&path, &device)
        .expect_err("shape exceeding backed data must error, not OOM");
    assert!(
        format!("{err:#}").contains("voxel data"),
        "expected a voxel-data read error, got: {err:#}"
    );
}
