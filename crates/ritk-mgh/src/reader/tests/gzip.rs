use super::*;

#[test]
fn test_read_mgz() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mgz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let values: Vec<f32> = (0..8).map(|i| (i as f32) * 1.5 + 0.25).collect();
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

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    image.with_data_slice(|loaded| {
        assert_eq!(loaded.len(), values.len());
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "mgz voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_read_mgh_gz_extension() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mgh.gz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
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

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;

    let image = read_mgh::<TestBackend, _>(&path, &device)?;
    image.with_data_slice(|loaded| {
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, expected, "voxel[{i}]");
        }
    });
    Ok(())
}
