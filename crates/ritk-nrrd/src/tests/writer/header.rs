use super::*;

#[test]
fn test_mandatory_header_fields_present() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("mandatory.nrrd");
    let backend = SequentialBackend;

    let image = make_image(
        vec![1.0f32; 2 * 3 * 4],
        [2, 3, 4],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert!(bytes_contain(&bytes, "NRRD0004"), "missing NRRD magic");
    assert!(bytes_contain(&bytes, "type: float"), "missing type");
    assert!(bytes_contain(&bytes, "dimension: 3"), "missing dimension");
    assert!(bytes_contain(&bytes, "encoding: raw"), "missing encoding");
    assert!(bytes_contain(&bytes, "endian: little"), "missing endian");
    assert!(
        bytes_contain(&bytes, "space directions:"),
        "missing space directions"
    );
    assert!(
        bytes_contain(&bytes, "space origin:"),
        "missing space origin"
    );
    assert!(
        bytes_contain(&bytes, "space units: \"mm\" \"mm\" \"mm\""),
        "missing canonical physical units"
    );

    Ok(())
}

/// `sizes` must be written as `nx ny nz` — the NRRD [X,Y,Z] order, which
/// is the reverse of RITK's [Z,Y,X] convention.
/// An Image with RITK shape [nz=2, ny=3, nx=4] must produce `sizes: 4 3 2`.
#[test]
fn test_sizes_written_in_xyz_order() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("sizes.nrrd");
    let backend = SequentialBackend;

    // RITK shape [nz=2, ny=3, nx=4]
    let image = make_image(
        vec![0.0f32; 2 * 3 * 4],
        [2, 3, 4],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "sizes: 4 3 2"),
        "sizes must be nx ny nz = 4 3 2 for RITK shape [2, 3, 4]"
    );

    Ok(())
}

/// For the canonical axial RITK direction, NRRD `space directions` must
/// encode file axes `[x,y,z]` as internal columns `[col,row,depth]`.
#[test]
fn test_space_directions_encodes_spacing_on_diagonal() -> Result<()> {
    let dir_tmp = tempdir()?;
    let path = dir_tmp.path().join("diag_sd.nrrd");
    let backend = SequentialBackend;

    let image = zeros_image(
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([0.9, 0.75, 1.5]),
        axial_direction(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    // For axial internal columns depth=Z, row=Y, col=X:
    //   sd0(file x) = internal col   * spacing[2] = (1.5,0,0)
    //   sd1(file y) = internal row   * spacing[1] = (0,0.75,0)
    //   sd2(file z) = internal depth * spacing[0] = (0,0,0.9)
    assert!(
        bytes_contain(&bytes, "(1.5,0,0)"),
        "sd0 must encode internal column spacing on file X"
    );
    assert!(
        bytes_contain(&bytes, "(0,0.75,0)"),
        "sd1 must encode internal row spacing on file Y"
    );
    assert!(
        bytes_contain(&bytes, "(0,0,0.9)"),
        "sd2 must encode internal depth spacing on file Z"
    );

    Ok(())
}

/// `space origin` must contain the image origin coordinates.
#[test]
fn test_space_origin_written_correctly() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("origin.nrrd");
    let backend = SequentialBackend;

    let image = zeros_image(
        [2, 2, 2],
        Point::new([10.5, 20.25, 30.125]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "10.5"),
        "origin[0] not found in header"
    );
    assert!(
        bytes_contain(&bytes, "20.25"),
        "origin[1] not found in header"
    );
    assert!(
        bytes_contain(&bytes, "30.125"),
        "origin[2] not found in header"
    );

    Ok(())
}

/// The binary payload size must equal `nx * ny * nz * 4` bytes
/// (one 4-byte little-endian f32 per voxel) following the blank terminator.
#[test]
fn test_payload_size_correct() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload.nrrd");
    let backend = SequentialBackend;

    let nz = 3usize;
    let ny = 4usize;
    let nx = 5usize;
    let n_voxels = nz * ny * nx;

    let image = make_image(
        vec![1.0f32; n_voxels],
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let expected_payload = n_voxels * 4;

    let actual_payload = nrrd_payload(&bytes).len();
    assert_eq!(
        actual_payload, expected_payload,
        "payload is {} bytes; expected {} ({} voxels × 4)",
        actual_payload, expected_payload, n_voxels
    );

    Ok(())
}

/// RITK [Z,Y,X] flat tensor values must be written directly because NRRD
/// raw payload order is X-fastest.
#[test]
fn test_payload_written_in_x_fastest_order() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload_order.nrrd");
    let backend = SequentialBackend;

    let nz = 2usize;
    let ny = 2usize;
    let nx = 3usize;
    let mut data_vec = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                data_vec.push(sample_value(100 * x + 10 * y + z));
            }
        }
    }

    let image = make_image(
        data_vec.clone(),
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        axial_direction(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let payload_values = decode_le_f32_payload(nrrd_payload(&bytes));
    assert_eq!(
        payload_values, data_vec,
        "NRRD payload order must match X-fastest RITK ZYX flat storage"
    );

    Ok(())
}

#[test]
fn test_caller_payload_length_must_match_shape() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wrong_payload.nrrd");
    let image = make_image(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    let error =
        write_nrrd_with_data(&path, &image, &[0.0; 7]).expect_err("short payload must be rejected");
    assert!(
        error.to_string().contains("requires 8"),
        "error must report the required voxel count: {error}"
    );
    assert!(!path.exists(), "invalid payload must not create a file");
    Ok(())
}

/// NrrdWriter struct delegates correctly to `write_nrrd`.
#[test]
fn test_writer_struct_creates_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("writer_struct.nrrd");
    let backend = SequentialBackend;

    let image = zeros_image(
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_nrrd(&path, &image, &backend)?;

    assert!(path.exists(), "output file must exist after write");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "output file must be non-empty"
    );

    Ok(())
}
