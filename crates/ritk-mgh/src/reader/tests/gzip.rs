use super::*;

#[test]
fn test_read_mgz() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mgz");
    let backend = TestBackend::default();
    let values: Vec<f32> = (0..8).map(|i| (i as f32) * 1.5 + 0.25).collect();
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;

    let image = read_mgh::<TestBackend, _>(&path, &backend, ImageReadBudget::DEFAULT)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    image.data_slice().map(|loaded| {
        assert_eq!(loaded.len(), values.len());
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "mgz voxel[{i}]: expected {expected}, got {got}"
            );
        }
    })?;
    Ok(())
}

#[test]
fn test_read_mgh_gz_extension() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mgh.gz");
    let backend = TestBackend::default();
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;

    let image = read_mgh::<TestBackend, _>(&path, &backend, ImageReadBudget::DEFAULT)?;
    image.data_slice().map(|loaded| {
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, expected, "voxel[{i}]");
        }
    })?;
    Ok(())
}

#[test]
fn test_read_external_uppercase_gzip_extensions() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let values: Vec<f32> = (0..8).map(|i| i as f32 + 0.25).collect();
    let data_bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &data_bytes,
    );

    for name in ["EXTERNAL.MGZ", "EXTERNAL.MGH.GZ"] {
        let path = dir.path().join(name);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&mgh)?;
        std::fs::write(&path, encoder.finish()?)?;

        let image = read_mgh::<TestBackend, _>(&path, &backend, ImageReadBudget::DEFAULT)?;
        image.data_slice().map(|loaded| {
            for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
                assert_eq!(got.to_bits(), expected.to_bits(), "{name} voxel[{i}]");
            }
        })?;
    }
    Ok(())
}

/// An MGZ stream is never inflated past what its header declares: the declared
/// volume is rejected by the decoded budget while the compressed file is tiny.
#[test]
fn mgz_declaring_a_volume_above_the_decoded_budget_is_rejected() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("bomb.mgz");
    let backend = TestBackend::default();
    let mgh = build_mgh_bytes(
        1,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &[0u8; 2 * 2 * 2 * 4],
    );
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;

    let budget = ImageReadBudget::new(1 << 20, 31, 1)?;
    let error = read_mgh::<TestBackend, _>(&path, &backend, budget)
        .expect_err("32 declared output bytes exceed a 31-byte budget");
    assert!(
        matches!(
            error,
            MghError::Budget {
                source: ImageReadBudgetError::Exceeded {
                    resource: ImageReadResource::DecodedBytes,
                    actual: 32,
                    maximum: 31,
                }
            }
        ),
        "got: {error}"
    );
    Ok(())
}
