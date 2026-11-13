use super::*;
use crate::reader::read_mgh_with_limit;

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

    let image = read_mgh::<TestBackend, _>(&path, &backend)?;
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

    let image = read_mgh::<TestBackend, _>(&path, &backend)?;
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

        let image = read_mgh::<TestBackend, _>(&path, &backend)?;
        image.data_slice().map(|loaded| {
            for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
                assert_eq!(got.to_bits(), expected.to_bits(), "{name} voxel[{i}]");
            }
        })?;
    }
    Ok(())
}

#[test]
fn mgz_decompression_cap_admits_exactly_the_decompressed_length() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("capped.mgz");
    let backend = TestBackend::default();
    let values = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let mgh = build_mgh_bytes(
        VERSION,
        [2, 2, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &payload,
    );
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&mgh)?;
    std::fs::write(&path, encoder.finish()?)?;
    let length = u64::try_from(mgh.len())?;
    let read = |limit| read_mgh_with_limit::<TestBackend, _>(&path, &backend, limit);

    let error = read(u64::try_from(HEADER_SIZE)? - 1).expect_err("the header is cut off");
    assert!(matches!(error, MghError::Header { .. }), "got: {error}");

    let error = read(length - 1).expect_err("the last payload byte is cut off");
    assert!(
        matches!(error, MghError::TruncatedPayload { voxel, voxel_count }
            if voxel == values.len() - 1 && voxel_count == values.len()),
        "got: {error}"
    );

    read(length)?.data_slice().map(|loaded| {
        let bits: Vec<u32> = loaded.iter().map(|v| v.to_bits()).collect();
        let expected: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, expected);
    })?;
    Ok(())
}
