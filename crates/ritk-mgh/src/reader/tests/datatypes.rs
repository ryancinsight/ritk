use super::*;

#[test]
fn test_read_f32_data() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("f32.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let values = vec![
        std::f32::consts::PI,
        std::f32::consts::E,
        std::f32::consts::SQRT_2,
        std::f32::consts::LN_2,
        1.0 / 7.0,
        -std::f32::consts::FRAC_PI_2,
        2.0 * std::f32::consts::E,
        1.0 / 3.0,
    ];
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
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

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), values.len());
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
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
fn test_read_u8_data() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("u8.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let u8_vals: Vec<u8> = (0u8..12).map(|i| i * 10).collect();
    let expected: Vec<f32> = u8_vals.iter().map(|&v| v as f32).collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 3, 2],
        MRI_UCHAR,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &u8_vals,
    );
    std::fs::write(&path, &mgh)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 3, 2]);
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), expected.len());
        for (i, (&got, &expected)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "u8 voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_read_i16_data() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("i16.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let i16_vals = vec![
        -1000, -100, 0, 100, 200, 300, 400, 500, -500, -200, 150, 750,
    ];
    let expected: Vec<f32> = i16_vals.iter().map(|&v| v as f32).collect();
    let data_bytes: Vec<u8> = i16_vals
        .iter()
        .flat_map(|v: &i16| v.to_be_bytes())
        .collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 3, 2],
        MRI_SHORT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 3, 2]);
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), expected.len());
        for (i, (&got, &expected)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "i16 voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_read_i32_data() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("i32.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let i32_vals = vec![
        -100_000, -10_000, 0, 10_000, 20_000, 30_000, 40_000, 50_000, -50_000, -20_000, 15_000,
        75_000,
    ];
    let expected: Vec<f32> = i32_vals.iter().map(|&v| v as f32).collect();
    let data_bytes: Vec<u8> = i32_vals
        .iter()
        .flat_map(|v: &i32| v.to_be_bytes())
        .collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 3, 2],
        MRI_INT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 3, 2]);
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), expected.len());
        for (i, (&got, &expected)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "i32 voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}
