//! Value-semantic coverage for the Atlas-native MGH reader path.

use crate::test_support::{build_mgh_bytes, IDENTITY_DIR};
use crate::MRI_FLOAT;
use coeus_core::SequentialBackend;
use tempfile::tempdir;

#[test]
fn native_read_mgh_preserves_shape_and_voxels() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("coeus.mgh");
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        MRI_FLOAT,
        [1.5, 2.0, 2.5],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh).unwrap();

    let backend = SequentialBackend;
    let image = crate::read_mgh(&path, &backend).expect("coeus MGH read");

    assert_eq!(
        image.shape(),
        [2, 2, 2],
        "coeus image shape is [nz, ny, nx]"
    );
    let loaded = image.data_slice().expect("contiguous host voxel data");
    assert_eq!(loaded.len(), values.len());
    for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "coeus voxel[{i}]: expected {expected}, got {got}"
        );
    }
}
