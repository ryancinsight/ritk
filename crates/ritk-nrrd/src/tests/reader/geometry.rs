use super::*;

/// `sizes: 4 3 2` (nx=4, ny=3, nz=2) must produce RITK shape [2, 3, 4].
#[test]
fn test_shape_permuted_to_zyx() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("shape.nrrd");

    let nx = 4usize;
    let ny = 3usize;
    let nz = 2usize;
    let data: Vec<f32> = (0..(nx * ny * nz)).map(sample_value).collect();
    write_inline_nrrd(&path, &data, nx, ny, nz, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let backend = SequentialBackend;
    let image = crate::read_nrrd(&path, &backend)?;

    assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
    Ok(())
}

#[test]
fn planar_space_metadata_is_promoted_to_z1() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("planar.nrrd");
    let data = (0..12).map(sample_value).collect::<Vec<_>>();
    write_inline_planar_nrrd(&path, &data, 4, 3);

    let image = crate::read_nrrd(&path, &SequentialBackend)?;
    assert_eq!(image.shape(), [1, 3, 4]);
    assert_eq!(image.spacing(), &Spacing::new([1.0, 2.0, 0.5]));
    assert_eq!(image.origin(), &Point::new([3.0, 4.0, 5.0]));
    assert_eq!(
        image.direction(),
        &Direction::from_rows([[0.0, 0.0, 1.0], [-0.8, 0.6, 0.0], [0.6, 0.8, 0.0]])
    );
    assert_eq!(
        image.data_slice().expect("contiguous planar voxels"),
        data.as_slice()
    );
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
                data.push(sample_value(100 * x + 10 * y + z));
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
                    let expected = sample_value(100 * x + 10 * y + z);
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
/// Coeus reader on the SAME file — both wrap the identical `decode_nrrd` core,
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
        .map(|i| sample_value(i) * 0.5 - 3.0)
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

    let parallel = coeus_core::MoiraiBackend::new();
    let par = crate::read_nrrd(&path, &parallel)?;

    let seq = crate::read_nrrd(&path, &SequentialBackend)?;

    assert_eq!(
        seq.shape(),
        par.shape(),
        "shape must be backend-independent"
    );
    assert_eq!(
        seq.origin(),
        par.origin(),
        "origin must be backend-independent"
    );
    assert_eq!(
        seq.spacing(),
        par.spacing(),
        "spacing must be backend-independent"
    );
    assert_eq!(
        seq.direction(),
        par.direction(),
        "direction must be backend-independent"
    );

    let seq_vox = seq.data_slice().expect("contiguous sequential voxels");
    {
        let par_vox = par.data_slice().expect("contiguous host data");
        assert_eq!(seq_vox.len(), par_vox.len(), "voxel count must match");
        for (i, (&n, &b)) in seq_vox.iter().zip(par_vox.iter()).enumerate() {
            assert_eq!(
                n.to_bits(),
                b.to_bits(),
                "voxel[{i}] must be bitwise-identical across backends"
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
        for i in 0_usize..8 {
            f.write_all(&sample_value(i).to_le_bytes())?;
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
        for i in 0_usize..8 {
            f.write_all(&sample_value(i).to_le_bytes())?;
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
        for i in 0_usize..8 {
            f.write_all(&sample_value(i).to_le_bytes())?;
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
        for i in 0_usize..8 {
            f.write_all(&sample_value(i).to_le_bytes())?;
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
        for i in 0_usize..8 {
            f.write_all(&sample_value(i).to_le_bytes())?;
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
