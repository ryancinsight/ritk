use super::*;

#[test]
fn test_read_nondefault_spatial() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spatial.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let spacing = [0.5f32, 0.75, 1.25];
    let dir_cols = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let data_bytes: Vec<u8> = (0..(4 * 3 * 2))
        .map(|i| (i as f32) * 0.1)
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let mgh = build_mgh_bytes(
        1,
        [4, 3, 2],
        MRI_FLOAT,
        spacing,
        dir_cols,
        [10.0, 20.0, 30.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 3, 4]);
    let sp = image.spacing();
    assert!((sp[0] - 0.5).abs() < 1e-6, "spacing[0]={}", sp[0]);
    assert!((sp[1] - 0.75).abs() < 1e-6, "spacing[1]={}", sp[1]);
    assert!((sp[2] - 1.25).abs() < 1e-6, "spacing[2]={}", sp[2]);

    let direction = image.direction();
    assert!((direction[(0, 0)] - 0.0).abs() < 1e-6);
    assert!((direction[(1, 0)] - 1.0).abs() < 1e-6);
    assert!((direction[(0, 1)] - (-1.0)).abs() < 1e-6);
    assert!((direction[(1, 1)] - 0.0).abs() < 1e-6);
    assert!((direction[(2, 2)] - 1.0).abs() < 1e-6);

    let origin = image.origin();
    assert!((origin[0] - 10.75).abs() < 1e-6, "origin[0]={}", origin[0]);
    assert!((origin[1] - 19.25).abs() < 1e-6, "origin[1]={}", origin[1]);
    assert!((origin[2] - 29.375).abs() < 1e-6, "origin[2]={}", origin[2]);
    Ok(())
}

#[test]
fn test_read_good_ras_flag_zero() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("no_ras.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let mut buf = Vec::with_capacity(HEADER_SIZE + data_bytes.len());
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&MRI_FLOAT.to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&0_i16.to_be_bytes());
    for _ in 0..15 {
        buf.extend_from_slice(&99.9f32.to_be_bytes());
    }
    buf.resize(HEADER_SIZE, 0u8);
    buf.extend_from_slice(&data_bytes);
    std::fs::write(&path, &buf)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    assert_eq!(image.spacing()[0], 1.0);
    assert_eq!(image.spacing()[1], 1.0);
    assert_eq!(image.spacing()[2], 1.0);
    assert_eq!(image.direction()[(0, 0)], 1.0);
    assert_eq!(image.direction()[(1, 1)], 1.0);
    assert_eq!(image.direction()[(2, 2)], 1.0);
    assert_eq!(image.origin()[0], 0.0);
    assert_eq!(image.origin()[1], 0.0);
    assert_eq!(image.origin()[2], 0.0);
    image.with_data_slice(|loaded| {
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, expected, "voxel[{i}]");
        }
    });
    Ok(())
}
