use super::*;

#[test]
fn test_round_trip_nrrd() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.nrrd");
    let backend = SequentialBackend;

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0..24).map(sample_value).collect();
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.75, 1.5]);
    let direction = Direction::identity();
    let image = ritk_image::Image::from_flat_on(
        data_vec.clone(),
        [2, 3, 4],
        origin,
        spacing,
        direction,
        &backend,
    )
    .expect("valid image");

    crate::write_nrrd(&path, &image, &backend)?;
    let loaded = crate::read_nrrd(&path, &backend)?;

    // Shape
    assert_eq!(loaded.shape(), [2, 3, 4]);

    // Origin (within f64 string-round-trip tolerance)
    assert!((loaded.origin()[0] - 10.0).abs() < 1e-6, "origin[0]");
    assert!((loaded.origin()[1] - 20.0).abs() < 1e-6, "origin[1]");
    assert!((loaded.origin()[2] - 30.0).abs() < 1e-6, "origin[2]");

    // Spacing
    assert!((loaded.spacing()[0] - 0.9).abs() < 1e-6, "spacing[0]");
    assert!((loaded.spacing()[1] - 0.75).abs() < 1e-6, "spacing[1]");
    assert!((loaded.spacing()[2] - 1.5).abs() < 1e-6, "spacing[2]");

    // Voxel values: every element must equal its original value.
    {
        let loaded_vals = loaded.data_slice().expect("contiguous host data");
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got
            );
        }
    }
    Ok(())
}

/// Reading a non-existent file must return an error (not panic).
#[test]
fn test_missing_file_returns_error() {
    let backend = SequentialBackend;
    let result = crate::read_nrrd("/nonexistent/path/file.nrrd", &backend);
    assert_rejects(result, "cannot open NRRD file");
}

/// A file without the NRRD magic line must return an error.
#[test]
fn test_invalid_magic_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("bad_magic.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NOT_NRRD_MAGIC")?;
        writeln!(f, "type: float")?;
    }

    let backend = SequentialBackend;
    let result = crate::read_nrrd(&path, &backend);
    assert_rejects(
        result,
        "Not a supported NRRD file: invalid or unsupported magic line",
    );
    Ok(())
}

/// Gzip-encoded payloads decode to the declared voxel values.
#[test]
fn test_gzip_encoding_decodes_voxel_values() -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("gzip.nrrd");
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&[0, 0, 128, 63])?;
    let compressed = encoder.finish()?;
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 1 1 1")?;
        writeln!(f, "encoding: gzip")?;
        writeln!(f, "endian: little")?;
        writeln!(f)?;
        f.write_all(&compressed)?;
    }

    let backend = SequentialBackend;
    let loaded = crate::read_nrrd(&path, &backend)?;
    assert_eq!(loaded.data_slice().expect("contiguous host data"), &[1.0]);
    Ok(())
}

/// A truncated gzip member returns its typed decompression failure.
#[test]
fn truncated_gzip_payload_returns_typed_error() -> Result<()> {
    use std::io::Write;

    let directory = tempdir()?;
    let path = directory.path().join("truncated-gzip.nrrd");
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "NRRD0004")?;
    writeln!(file, "type: float")?;
    writeln!(file, "dimension: 3")?;
    writeln!(file, "sizes: 1 1 1")?;
    writeln!(file, "encoding: gzip")?;
    writeln!(file, "endian: little")?;
    writeln!(file)?;
    file.write_all(&[0x1f, 0x8b, 0x08, 0x00])?;
    drop(file);

    let error = crate::read_nrrd_stored(&path, ritk_image_io::ImageReadBudget::DEFAULT)
        .expect_err("a truncated gzip member cannot produce the declared sample");
    assert!(
        matches!(error, crate::NrrdStoredReadError::PayloadIo { .. }),
        "truncated compressed bytes must preserve their I/O error, got {error}"
    );
    Ok(())
}

#[test]
fn ascii_encoding_decodes_values_for_compute_images() -> Result<()> {
    use std::io::Write;

    let directory = tempdir()?;
    let path = directory.path().join("ascii.nrrd");
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "NRRD0004")?;
    writeln!(file, "type: float")?;
    writeln!(file, "dimension: 3")?;
    writeln!(file, "sizes: 2 1 1")?;
    writeln!(file, "encoding: ascii")?;
    writeln!(file)?;
    write!(file, "1.25 -2.5")?;

    let image = crate::read_nrrd(&path, &SequentialBackend)?;
    assert_eq!(
        image.data_slice().expect("contiguous host data"),
        &[1.25, -2.5]
    );
    Ok(())
}

#[test]
fn compute_image_reader_rejects_sample_units_it_cannot_retain() -> Result<()> {
    use std::io::Write;

    let directory = tempdir()?;
    let path = directory.path().join("sample-units.nrrd");
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "NRRD0004")?;
    writeln!(file, "type: unsigned char")?;
    writeln!(file, "dimension: 3")?;
    writeln!(file, "sizes: 1 1 1")?;
    writeln!(file, "sample units: HU:=CT")?;
    writeln!(file, "encoding: raw")?;
    writeln!(file)?;
    file.write_all(&[207])?;

    let error = crate::read_nrrd(&path, &SequentialBackend)
        .expect_err("compute-image output cannot retain the sample-unit label");
    assert!(matches!(
        error.downcast_ref::<crate::NrrdStoredReadError>(),
        Some(crate::NrrdStoredReadError::UnsupportedSampleUnits { units })
            if units == "HU:=CT"
    ));
    Ok(())
}

/// Missing `dimension` field must return an error.
#[test]
fn test_missing_dimension_field_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("missing_dim.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        // Intentionally omit 'dimension'.
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f, "endian: little")?;
        writeln!(f)?;
    }

    let backend = SequentialBackend;
    let result = crate::read_nrrd(&path, &backend);
    assert_rejects(
        result,
        "NRRD header is missing required field \"dimension\"",
    );
    Ok(())
}

/// Unsupported NRRD type must return a descriptive error that names the type.
#[test]
fn test_unsupported_type_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("bad_type.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: long double")?; // not supported
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f, "endian: little")?;
        writeln!(f)?;
        f.write_all(&[0u8; 128])?;
    }

    let backend = SequentialBackend;
    let result = crate::read_nrrd(&path, &backend);
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("long double"),
        "Error must name the unsupported type; got: {}",
        msg
    );
    Ok(())
}

/// External data file referenced by `data file:` must be opened and read.
#[test]
fn test_detached_data_file() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let header_path = dir.path().join("volume.nhdr");
    let raw_path = dir.path().join("volume.raw");

    let nx = 2usize;
    let ny = 2usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..8).map(sample_value).collect();

    {
        let mut f = std::fs::File::create(&raw_path)?;
        for &v in &data {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    {
        let mut f = std::fs::File::create(&header_path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: {} {} {}", nx, ny, nz)?;
        writeln!(f, "spacings: 1 1 1")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f, "data file: volume.raw")?;
        writeln!(f)?;
    }

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&header_path, &backend)?;

    assert_eq!(image.shape(), [nz, ny, nx]);
    {
        let vals = image.data_slice().expect("contiguous host data");
        assert_eq!(
            vals,
            data.as_slice(),
            "detached raw file order must preserve RITK ZYX flat values"
        );
        let sum: f32 = vals.iter().sum();
        assert!(
            (sum - 28.0).abs() < 1e-5,
            "Voxel sum mismatch: expected 28, got {}",
            sum
        );
    }
    Ok(())
}
