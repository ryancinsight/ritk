use super::*;

#[test]
fn test_round_trip_basic() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("roundtrip.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..(3 * 4 * 5) as u32)
        .map(|i| i as f32 * std::f32::consts::PI / 11.0)
        .collect();
    let image = make_image(data_vec.clone(), 3, 4, 5);

    crate::write_mgh(&image, &path)?;
    let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [3, 4, 5]);
    let data = loaded.data().clone().to_data();
    let loaded_vals = data.as_slice::<f32>().unwrap();
    assert_eq!(loaded_vals.len(), data_vec.len());
    for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "voxel[{i}]: expected {expected}, got {got}"
        );
    }
    Ok(())
}

#[test]
fn test_round_trip_mgz() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("roundtrip.mgz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..(2 * 3 * 4) as u32)
        .map(|i| (i as f32).sqrt() + 0.5)
        .collect();
    let image = make_image(data_vec.clone(), 2, 3, 4);

    crate::write_mgh(&image, &path)?;
    let bytes = std::fs::read(&path)?;
    assert_eq!(bytes[0], 0x1f, "First byte must be gzip magic 0x1f");
    assert_eq!(bytes[1], 0x8b, "Second byte must be gzip magic 0x8b");

    let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 3, 4]);
    let data = loaded.data().clone().to_data();
    let loaded_vals = data.as_slice::<f32>().unwrap();
    for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "voxel[{i}]: expected {expected}, got {got}"
        );
    }
    Ok(())
}

#[test]
fn test_round_trip_nondefault_spatial() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spatial_rt.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let mut dir_mat = nalgebra::SMatrix::<f64, 3, 3>::zeros();
    dir_mat[(0, 1)] = -1.0;
    dir_mat[(1, 0)] = 1.0;
    dir_mat[(2, 2)] = 1.0;
    let data_vec: Vec<f32> = (0..(2 * 3 * 4) as u32)
        .map(|i| i as f32 * 0.1 + 1.0)
        .collect();
    let image = crate::test_support::make_image_with_spatial(
        data_vec.clone(),
        2,
        3,
        4,
        Point::new([10.75, 19.25, 29.375]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction(dir_mat),
    );

    crate::write_mgh(&image, &path)?;
    let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert!((loaded.spacing()[0] - 0.5).abs() < 1e-6);
    assert!((loaded.spacing()[1] - 0.75).abs() < 1e-6);
    assert!((loaded.spacing()[2] - 1.25).abs() < 1e-6);
    assert!((loaded.origin()[0] - 10.75).abs() < 1e-5);
    assert!((loaded.origin()[1] - 19.25).abs() < 1e-5);
    assert!((loaded.origin()[2] - 29.375).abs() < 1e-5);

    let data = loaded.data().clone().to_data();
    let loaded_vals = data.as_slice::<f32>().unwrap();
    for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "voxel[{i}]: expected {expected}, got {got}"
        );
    }
    Ok(())
}

#[test]
fn test_mgh_reader_struct_delegates() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("struct.mgh");
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
    std::fs::write(&path, &mgh)?;

    let image = MghReader::read::<TestBackend, _>(&path, &device)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    let data = image.data().clone().to_data();
    let loaded = data.as_slice::<f32>().unwrap();
    for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
        assert_eq!(got, expected, "voxel[{i}]");
    }
    Ok(())
}
