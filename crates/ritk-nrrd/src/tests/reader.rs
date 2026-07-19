use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

// ── Helpers ───────────────────────────────────────────────────────────

/// Write a minimal inline NRRD file with `MET_FLOAT`-equivalent (`float`)
/// data and the given spatial metadata.
fn write_inline_nrrd(
    path: &std::path::Path,
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    spacing: [f64; 3],
    origin: [f64; 3],
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "NRRD0004").unwrap();
    writeln!(f, "# written by ritk test helper").unwrap();
    writeln!(f, "type: float").unwrap();
    writeln!(f, "dimension: 3").unwrap();
    writeln!(f, "space: right-anterior-superior").unwrap();
    writeln!(f, "sizes: {} {} {}", nx, ny, nz).unwrap();
    writeln!(
        f,
        "space directions: ({},0,0) (0,{},0) (0,0,{})",
        spacing[0], spacing[1], spacing[2]
    )
    .unwrap();
    writeln!(f, "kinds: domain domain domain").unwrap();
    writeln!(f, "endian: little").unwrap();
    writeln!(f, "encoding: raw").unwrap();
    writeln!(
        f,
        "space origin: ({},{},{})",
        origin[0], origin[1], origin[2]
    )
    .unwrap();
    writeln!(f).unwrap(); // blank line terminates header
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

// ── Shape and metadata ─────────────────────────────────────────────────

/// `sizes: 4 3 2` (nx=4, ny=3, nz=2) must produce RITK shape [2, 3, 4].
#[test]
fn test_shape_permuted_to_zyx() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("shape.nrrd");

    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
    write_inline_nrrd(&path, &data, nx, ny, nz, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;

    assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
    Ok(())
}

/// NRRD raw data is X-fastest. The decoded flat tensor must therefore
/// preserve `index = z * ny * nx + y * nx + x` when shaped as RITK ZYX.
#[test]
fn test_raw_payload_x_fastest_maps_to_zyx_tensor_values() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload_order.nrrd");

    let nx = 3usize;
    let ny = 2usize;
    let nz = 2usize;
    let mut data = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                data.push((100 * x + 10 * y + z) as f32);
            }
        }
    }
    write_inline_nrrd(&path, &data, nx, ny, nz, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;
    assert_eq!(image.shape(), [nz, ny, nx]);
    {
        let values = image.data_slice().expect("contiguous host data");
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let index = z * ny * nx + y * nx + x;
                    let expected = (100 * x + 10 * y + z) as f32;
                    assert_eq!(
                        values[index], expected,
                        "value at internal [z={z}, y={y}, x={x}]"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Spacing extracted from axis-aligned `space directions` must match the
/// magnitudes of NRRD file vectors reordered into RITK [depth,row,col].
#[test]
fn test_spacing_from_space_directions() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spacing_sd.nrrd");
    let data = vec![0.0f32; 2 * 3 * 4];
    write_inline_nrrd(&path, &data, 4, 3, 2, [0.9, 0.75, 1.5], [5.0, 10.0, 15.0]);

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;

    // NRRD file spacings [x,y,z] become RITK metadata [z,y,x].
    assert!((image.spacing()[0] - 1.5).abs() < 1e-9, "spacing[0]");
    assert!((image.spacing()[1] - 0.75).abs() < 1e-9, "spacing[1]");
    assert!((image.spacing()[2] - 0.9).abs() < 1e-9, "spacing[2]");

    // Origin in physical [X, Y, Z] order.
    assert!((image.origin()[0] - 5.0).abs() < 1e-9, "origin[0]");
    assert!((image.origin()[1] - 10.0).abs() < 1e-9, "origin[1]");
    assert!((image.origin()[2] - 15.0).abs() < 1e-9, "origin[2]");

    Ok(())
}

/// Differential oracle: the Atlas-native reader must be value-identical to the
/// Burn reader on the SAME file — both wrap the identical `decode_nrrd` core,
/// so shape, every voxel (bitwise), origin, spacing, and direction must match.
/// Uses anisotropic spacing and a non-zero origin so an axis transposition or
/// metadata reorder in either path would diverge.
#[test]
fn native_reader_matches_coeus_reader() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("differential.nrrd");

    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz))
        .map(|i| (i as f32) * 0.5 - 3.0)
        .collect();
    write_inline_nrrd(
        &path,
        &data,
        nx,
        ny,
        nz,
        [0.9, 0.75, 1.5],
        [5.0, 10.0, 15.0],
    );

    let backend = SequentialBackend;
    let burn = crate::read_nrrd(&path, &backend)?;

    let backend = coeus_core::SequentialBackend;
    let native = crate::read_nrrd(&path, &backend)?;

    assert_eq!(native.shape(), burn.shape(), "shape must match Coeus path");
    assert_eq!(
        native.origin(),
        burn.origin(),
        "origin must match Coeus path"
    );
    assert_eq!(
        native.spacing(),
        burn.spacing(),
        "spacing must match Coeus path"
    );
    assert_eq!(
        native.direction(),
        burn.direction(),
        "direction must match Coeus path"
    );

    let native_vox = native.data_slice().expect("contiguous native voxels");
    {
        let burn_vox = burn.data_slice().expect("contiguous host data");
        assert_eq!(native_vox.len(), burn_vox.len(), "voxel count must match");
        for (i, (&n, &b)) in native_vox.iter().zip(burn_vox.iter()).enumerate() {
            assert_eq!(
                n.to_bits(),
                b.to_bits(),
                "voxel[{i}] must be bitwise-identical to the Burn reader"
            );
        }
    }

    Ok(())
}

/// `spacings` field (no `space directions`) must use NRRD [x,y,z]
/// scalars as RITK [z,y,x] spacing with canonical axis-aligned columns.
#[test]
fn test_spacing_fallback_to_spacings_field() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("spacings_only.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        writeln!(f, "spacings: 0.5 0.5 2.0")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f)?; // blank line
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;

    assert!((image.spacing()[0] - 2.0).abs() < 1e-9, "spacing[0]");
    assert!((image.spacing()[1] - 0.5).abs() < 1e-9, "spacing[1]");
    assert!((image.spacing()[2] - 0.5).abs() < 1e-9, "spacing[2]");

    let d = image.direction().0;
    assert!(d[(0, 0)].abs() < 1e-9, "direction[0,0]");
    assert!(d[(0, 1)].abs() < 1e-9, "direction[0,1]");
    assert!((d[(0, 2)] - 1.0).abs() < 1e-9, "direction[0,2]");
    assert!(d[(1, 0)].abs() < 1e-9, "direction[1,0]");
    assert!((d[(1, 1)] - 1.0).abs() < 1e-9, "direction[1,1]");
    assert!(d[(1, 2)].abs() < 1e-9, "direction[1,2]");
    assert!((d[(2, 0)] - 1.0).abs() < 1e-9, "direction[2,0]");
    assert!(d[(2, 1)].abs() < 1e-9, "direction[2,1]");
    assert!(d[(2, 2)].abs() < 1e-9, "direction[2,2]");

    Ok(())
}

/// Direction matrix columns extracted from non-axis-aligned `space
/// directions` must match the normalised input vectors.
///
/// Test vector: space directions = (2,0,0) (0,3,0) (0,0,4).
/// Expected RITK metadata: spacing = [4, 3, 2], direction columns [Z,Y,X].
#[test]
fn test_direction_from_scaled_space_directions() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let path = dir.path().join("scaled_dirs.nrrd");
    {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "NRRD0004")?;
        writeln!(f, "type: float")?;
        writeln!(f, "dimension: 3")?;
        writeln!(f, "sizes: 2 2 2")?;
        // Non-unit direction vectors: magnitude encodes spacing.
        writeln!(f, "space directions: (2,0,0) (0,3,0) (0,0,4)")?;
        writeln!(f, "endian: little")?;
        writeln!(f, "encoding: raw")?;
        writeln!(f)?;
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;

    assert!((image.spacing()[0] - 4.0).abs() < 1e-9, "spacing[0] = 4");
    assert!((image.spacing()[1] - 3.0).abs() < 1e-9, "spacing[1] = 3");
    assert!((image.spacing()[2] - 2.0).abs() < 1e-9, "spacing[2] = 2");

    let d = image.direction().0;
    assert!(d[(0, 0)].abs() < 1e-9);
    assert!(d[(0, 1)].abs() < 1e-9);
    assert!((d[(0, 2)] - 1.0).abs() < 1e-9);
    assert!(d[(1, 0)].abs() < 1e-9);
    assert!((d[(1, 1)] - 1.0).abs() < 1e-9);
    assert!(d[(1, 2)].abs() < 1e-9);
    assert!((d[(2, 0)] - 1.0).abs() < 1e-9);
    assert!(d[(2, 1)].abs() < 1e-9);
    assert!(d[(2, 2)].abs() < 1e-9);

    Ok(())
}

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
    let err = crate::read_nrrd(&path, &backend)
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
    let err = crate::read_nrrd(&path, &backend)
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
    let err = crate::read_nrrd(&path, &backend)
        .expect_err("multiple space origin vectors must reject the header");

    assert!(
        err.to_string()
            .contains("'space origin' must contain exactly 1 vector, found 2"),
        "error must name the space origin vector-count contract, got {err}"
    );

    Ok(())
}

// ── Round-trip ─────────────────────────────────────────────────────────

/// Write an Image via `write_nrrd` and read it back; verify shape,
/// spatial metadata, and every voxel value.
#[test]
fn test_round_trip_nrrd() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.nrrd");
    let backend = SequentialBackend;

    // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
    let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
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

// ── Error paths ────────────────────────────────────────────────────────

/// Reading a non-existent file must return an error (not panic).
#[test]
fn test_missing_file_returns_error() {
    let backend = SequentialBackend;
    let result = crate::read_nrrd("/nonexistent/path/file.nrrd", &backend);
    assert!(result.is_err(), "Expected Err for missing file");
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
    assert!(result.is_err(), "Expected Err for invalid magic");
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
    let result = crate::read_nrrd(&path, &backend);
    assert!(result.is_err(), "Expected Err for gzip encoding");
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
    let result = crate::read_nrrd(&path, &backend);
    assert!(result.is_err(), "Expected Err for missing dimension");
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
    assert!(result.is_err(), "Expected Err for unsupported type");
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
