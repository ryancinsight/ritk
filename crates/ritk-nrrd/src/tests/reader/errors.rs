use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_core::rejection::assert_rejects;
use tempfile::tempdir;

/// Malformed `space directions` with an unterminated vector must fail at the
/// header boundary instead of accepting the already parsed prefix.
#[test]
fn test_unterminated_space_directions_returns_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("unterminated_space_directions.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "space directions: (1,0,0) (0,1,0) (0,0,1")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f)?;
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let err = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)
        .expect_err("unterminated space directions must reject the header");

    assert!(
        err.to_string()
            .contains("Unterminated vector group in '(1,0,0) (0,1,0) (0,0,1'"),
        "error must name the rejected space directions field, got {err}"
    );

    Ok(())
}

/// Malformed `space directions` with trailing non-vector text must fail at the
/// header boundary instead of accepting the valid vector prefix.
#[test]
fn test_trailing_space_directions_tokens_return_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("trailing_space_directions.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "space directions: (1,0,0) (0,1,0) (0,0,1) junk")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f)?;
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let err = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)
        .expect_err("trailing space directions tokens must reject the header");

    assert!(
        err.to_string().contains(
            "Unexpected text outside vector group in '(1,0,0) (0,1,0) (0,0,1) junk': 'junk'"
        ),
        "error must name the rejected space directions suffix, got {err}"
    );

    Ok(())
}

/// `space origin` is a single point, so multiple point vectors must be rejected
/// rather than taking the first and ignoring the rest.
#[test]
fn test_multiple_space_origin_vectors_return_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("multiple_space_origin.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "space directions: (1,0,0) (0,1,0) (0,0,1)")?;
        writeln!(f, "space origin: (0,0,0) (1,1,1)")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f)?;
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let err = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)
        .expect_err("multiple space origin vectors must reject the header");

    assert!(
        err.to_string()
            .contains("'space origin' must contain exactly 1 vector, found 2"),
        "error must name the space origin vector-count contract, got {err}"
    );

    Ok(())
}

/// Reading a non-existent file must return an error (not panic).
#[test]
fn test_missing_file_returns_error() {
    let backend = SequentialBackend;
    let result = crate::read_nrrd::<f32, _, _, _>("/nonexistent/path/file.nrrd", &backend, Exact);
    assert_rejects(result, "Cannot open NRRD file");
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
    let result = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact);
    assert_rejects(
        result,
        "Not a valid NRRD file: magic line does not start with",
    );
    Ok(())
}

/// `encoding: gzip` must return an error with a helpful message.
#[test]
fn test_gzip_encoding_returns_helpful_error() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("gzip.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "encoding: gzip")?;
        writeln!(f, "endian: little")?;
        writeln!(f)?;
    }

    let backend = SequentialBackend;
    let result = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact);
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("gzip") || msg.contains("encoding"),
        "Error message must mention the encoding; got: {}",
        msg
    );
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
    let result = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact);
    assert_rejects(result, "Missing 'dimension' in NRRD header");
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
    let result = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact);
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
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

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
    let image = crate::read_nrrd::<f32, _, _, _>(&header_path, &backend, Exact)?;

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
