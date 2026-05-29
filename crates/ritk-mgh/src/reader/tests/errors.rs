use super::*;

#[test]
fn test_read_invalid_version() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_version.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let mgh = build_mgh_bytes(
        2,
        [2, 2, 2],
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2 * 4],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "Reading invalid version must fail");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("version"),
        "Error must mention 'version', got: {msg}"
    );
}

#[test]
fn test_read_unsupported_type_code() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_type.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        99,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "Unsupported type code must fail");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("data type"),
        "Error must mention 'data type', got: {msg}"
    );
}

#[test]
fn test_read_truncated_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("truncated.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let mut buf = vec![0u8; 100];
    buf[0..4].copy_from_slice(&1_i32.to_be_bytes());
    buf[4..8].copy_from_slice(&2_i32.to_be_bytes());
    buf[8..12].copy_from_slice(&2_i32.to_be_bytes());
    buf[12..16].copy_from_slice(&2_i32.to_be_bytes());
    buf[16..20].copy_from_slice(&1_i32.to_be_bytes());
    buf[20..24].copy_from_slice(&MRI_FLOAT.to_be_bytes());
    std::fs::write(&path, &buf).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "Truncated file must fail");
}
