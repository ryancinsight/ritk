use super::*;

#[test]
fn test_all_four_data_types_readable() -> Result<()> {
    let dir = tempdir()?;
    let device: <TestBackend as Backend>::Device = Default::default();

    let vals: Vec<u8> = vec![0, 50, 100, 150, 200, 250, 128, 64];
    let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    let path = dir.path().join("types_u8.mgh");
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        MRI_UCHAR,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &vals,
    );
    std::fs::write(&path, &mgh)?;
    assert_read_values(&path, &device, &expected, "u8")?;

    let vals: Vec<i16> = vec![-32000, -1000, 0, 1000, 5000, 10000, -5000, 32000];
    let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    let data_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
    let path = dir.path().join("types_i16.mgh");
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        MRI_SHORT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;
    assert_read_values(&path, &device, &expected, "i16")?;

    let vals: Vec<i32> = vec![-100_000, -1, 0, 1, 50_000, 100_000, -50_000, 12345];
    let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    let data_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
    let path = dir.path().join("types_i32.mgh");
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        MRI_INT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;
    assert_read_values(&path, &device, &expected, "i32")?;

    let vals = [std::f32::consts::PI,
        -std::f32::consts::E,
        0.0,
        f32::MIN_POSITIVE,
        1.0 / 7.0,
        std::f32::consts::SQRT_2,
        -123_456.79,
        std::f32::consts::LN_2];
    let data_bytes: Vec<u8> = vals.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let path = dir.path().join("types_f32.mgh");
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;
    let image = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    image.with_data_slice(|loaded| {
        for (i, (&got, &expected)) in loaded.iter().zip(vals.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "f32 voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_invalid_version_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_ver.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let mgh = build_mgh_bytes(
        99,
        [2, 2, 2],
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2 * 4],
    );
    std::fs::write(&path, &mgh).unwrap();

    let result = crate::read_mgh::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "Reading invalid version must fail");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("version"),
        "Error must mention 'version', got: {msg}"
    );
}

fn assert_read_values(
    path: &std::path::Path,
    device: &<TestBackend as Backend>::Device,
    expected: &[f32],
    label: &str,
) -> Result<()> {
    let image = crate::read_mgh::<TestBackend, _>(path, device)?;
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), expected.len());
        for (i, (&got, &expected)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "{label} voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}
