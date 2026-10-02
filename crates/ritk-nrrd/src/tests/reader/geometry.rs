use super::{write_inline_nrrd, write_inline_planar_nrrd};
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

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
    let image = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

    assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
    Ok(())
}

#[test]
fn planar_space_metadata_is_promoted_to_z1() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("planar.nrrd");
    let data = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    write_inline_planar_nrrd(&path, &data, 4, 3);

    let image = crate::read_nrrd::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(image.shape(), [1, 3, 4]);
    assert_eq!(image.spacing(), &Spacing::new([1.0, 2.0, 0.5]));
    assert_eq!(image.origin(), &Point::new([3.0, 4.0, 0.0]));
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
                data.push((100 * x + 10 * y + z) as f32);
            }
        }
    }
    write_inline_nrrd(&path, &data, nx, ny, nz, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

    let backend = SequentialBackend;
    let image = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;
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
    let image = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

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
fn reader_matches_coeus_reader() -> Result<()> {
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

    let parallel = coeus_core::MoiraiBackend::new();
    let par = crate::read_nrrd::<f32, _, _, _>(&path, &parallel, Exact)?;

    let seq = crate::read_nrrd::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;

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
        for i in 0u32..8 {
            f.write_all(&(i as f32).to_le_bytes())?;
        }
    }

    let backend = SequentialBackend;
    let image = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

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
    let image = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

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
    let loaded = crate::read_nrrd::<f32, _, _, _>(&path, &backend, Exact)?;

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
