use super::*;

#[test]
fn test_round_trip_f32_values() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rt_f32.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..(2 * 3 * 5) as u32)
        .map(|i| (i as f32) * std::f32::consts::E / 13.0)
        .collect();
    let image = make_image(data_vec.clone(), 2, 3, 5);

    write_mgh(&image, &path)?;
    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 3, 5]);
    loaded.with_data_slice(|loaded_vals| {
        assert_eq!(loaded_vals.len(), data_vec.len());
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_round_trip_mgz() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rt.mgz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..(3 * 4 * 2) as u32)
        .map(|i| (i as f32).sqrt() + 0.5)
        .collect();
    let image = make_image(data_vec.clone(), 3, 4, 2);

    write_mgh(&image, &path)?;
    let bytes = std::fs::read(&path)?;
    assert_eq!(bytes[0], 0x1f);
    assert_eq!(bytes[1], 0x8b);

    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [3, 4, 2]);
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_round_trip_mgh_gz_extension() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mgh.gz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..8u32).map(|i| i as f32 * 2.5).collect();
    let image = make_image(data_vec.clone(), 2, 2, 2);

    write_mgh(&image, &path)?;
    let bytes = std::fs::read(&path)?;
    assert_eq!(bytes[0], 0x1f);
    assert_eq!(bytes[1], 0x8b);

    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(got.to_bits(), expected.to_bits(), "voxel[{i}]");
        }
    });
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
    let image = make_image_with_spatial(
        data_vec.clone(),
        2,
        3,
        4,
        Point::new([10.75, 19.25, 29.375]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction(dir_mat),
    );

    write_mgh(&image, &path)?;
    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert!((loaded.spacing()[0] - 0.5).abs() < 1e-6);
    assert!((loaded.spacing()[1] - 0.75).abs() < 1e-6);
    assert!((loaded.spacing()[2] - 1.25).abs() < 1e-6);
    assert!((loaded.origin()[0] - 10.75).abs() < 1e-5);
    assert!((loaded.origin()[1] - 19.25).abs() < 1e-5);
    assert!((loaded.origin()[2] - 29.375).abs() < 1e-5);
    loaded.with_data_slice(|loaded_vals| {
        assert_eq!(loaded_vals.len(), data_vec.len());
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_writer_struct_delegates() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("struct.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let image = make_image(vec![1.0f32; 8], 2, 2, 2);

    MghWriter::write(&image, &path)?;
    assert!(path.exists(), "Output file must exist");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "Output file must be non-empty"
    );

    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 2, 2]);
    loaded.with_data_slice(|loaded_vals| {
        for (i, &got) in loaded_vals.iter().enumerate() {
            assert_eq!(got, 1.0f32, "voxel[{i}]: expected 1.0, got {got}");
        }
    });
    Ok(())
}

#[test]
fn test_edge_case_values_round_trip() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("edges.mgh");
    let device: <TestBackend as Backend>::Device = Default::default();
    let values = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        std::f32::consts::PI,
    ];
    let image = make_image(values.clone(), 2, 2, 2);

    write_mgh(&image, &path)?;
    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    loaded.with_data_slice(|loaded_vals| {
        assert_eq!(loaded_vals.len(), values.len());
        for (i, (&got, &expected)) in loaded_vals.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{i}]: expected {expected}, got {got}"
            );
        }
    });
    Ok(())
}

#[test]
fn test_round_trip_mgz_with_spatial() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spatial.mgz");
    let device: <TestBackend as Backend>::Device = Default::default();
    let data_vec: Vec<f32> = (0..8u32).map(|i| i as f32 * 10.0).collect();
    let image = make_image_with_spatial(
        data_vec.clone(),
        2,
        2,
        2,
        Point::new([100.0, 200.0, 300.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction::identity(),
    );

    write_mgh(&image, &path)?;
    let bytes = std::fs::read(&path)?;
    assert_eq!(bytes[0], 0x1f);
    assert_eq!(bytes[1], 0x8b);

    let loaded = crate::read_mgh::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [2, 2, 2]);
    assert!((loaded.spacing()[0] - 2.0).abs() < 1e-6);
    assert!((loaded.spacing()[1] - 3.0).abs() < 1e-6);
    assert!((loaded.spacing()[2] - 4.0).abs() < 1e-6);
    assert!((loaded.origin()[0] - 100.0).abs() < 1e-4);
    assert!((loaded.origin()[1] - 200.0).abs() < 1e-4);
    assert!((loaded.origin()[2] - 300.0).abs() < 1e-4);
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(got.to_bits(), expected.to_bits(), "voxel[{i}]");
        }
    });
    Ok(())
}
