use super::*;

#[test]
fn test_read_invalid_version() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_version.mgh");
    let backend = TestBackend::default();
    let mgh = build_mgh_bytes(
        2,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2 * 4],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &backend);
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
    let backend = TestBackend::default();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        99,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &backend);
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
    let backend = TestBackend::default();
    let mut buf = vec![0u8; 100];
    buf[0..4].copy_from_slice(&1_i32.to_be_bytes());
    buf[4..8].copy_from_slice(&2_i32.to_be_bytes());
    buf[8..12].copy_from_slice(&2_i32.to_be_bytes());
    buf[12..16].copy_from_slice(&2_i32.to_be_bytes());
    buf[16..20].copy_from_slice(&1_i32.to_be_bytes());
    buf[20..24].copy_from_slice(&MRI_FLOAT.to_be_bytes());
    std::fs::write(&path, &buf).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &backend);
    assert!(result.is_err(), "Truncated file must fail");
}

#[test]
fn test_read_multi_frame_fails_rather_than_returning_frame_zero() {
    // The header alone is enough to reject the wrong API. A single-volume read
    // must not allocate or consume any declared frame payload first.
    let dir = tempdir().unwrap();
    let path = dir.path().join("three_frames.mgh");
    let backend = TestBackend::default();

    const FRAMES: i32 = 3;
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        FRAMES,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[],
    );
    std::fs::write(&path, &mgh).unwrap();

    let err = read_mgh::<TestBackend, _>(&path, &backend)
        .expect_err("a 3-frame MGH has no 3-D representation and must not read as one");

    let msg = format!("{err:#}");
    assert!(
        msg.contains("3 frames"),
        "error must name the declared frame count, got: {msg}"
    );
}

#[test]
fn test_read_single_frame_still_reads_at_the_rejection_boundary() {
    // The boundary partner of the test above: nframes == SINGLE_FRAME is the
    // largest accepted value, so the rejection cannot be an off-by-one that
    // also refuses ordinary volumes.
    let dir = tempdir().unwrap();
    let path = dir.path().join("one_frame.mgh");
    let backend = TestBackend::default();

    let voxels: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut payload = Vec::with_capacity(voxels.len() * 4);
    for value in &voxels {
        payload.extend_from_slice(&value.to_be_bytes());
    }

    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &payload,
    );
    std::fs::write(&path, &mgh).unwrap();

    let image = read_mgh::<TestBackend, _>(&path, &backend)
        .expect("a single-frame MGH is the accepted case");
    let loaded = image.data_slice().expect("contiguous host voxel data");
    assert_eq!(
        loaded, voxels,
        "single-frame voxels must survive the frame check unchanged"
    );
}

#[test]
fn test_read_hostile_dims_does_not_oom() {
    // Header claims a 1024^3 float volume (~4.3 GiB) but supplies 16 bytes. The
    // bounded reader must reserve at most one chunk and fail with a read error
    // rather than reserving ~4.3 GiB and aborting on out-of-memory.
    let dir = tempdir().unwrap();
    let path = dir.path().join("hostile_dims.mgh");
    let backend = TestBackend::default();
    let mgh = build_mgh_bytes(
        1,
        [1024, 1024, 1024],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 16],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = read_mgh::<TestBackend, _>(&path, &backend);
    assert!(result.is_err(), "Hostile dimensions must fail, not OOM");
}
