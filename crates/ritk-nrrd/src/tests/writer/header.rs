use super::{axial_direction, bytes_contain, make_image, zeros_image, TestBackend};
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

/// A written NRRD file must contain the mandatory header fields.
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

/// Non-identity direction matrix must appear in `space directions` as
/// correctly scaled NRRD file-axis vectors.
///
/// Test: internal columns depth=Z, row=-X, col=Y with spacing [2,3,4].
///   file x = internal col   →  sd0 = (0,4,0)
///   file y = internal row   →  sd1 = (-3,0,0)
///   file z = internal depth →  sd2 = (0,0,2)
#[test]
fn test_rotated_direction_in_space_directions() -> Result<()> {
    let dir_tmp = tempdir()?;
    let path = dir_tmp.path().join("rotated.nrrd");
    let backend = SequentialBackend;

    // Build the rotated direction matrix explicitly
    let mut direction = Direction::zeros();
    // Column 0 = internal depth axis = physical Z.
    direction[(0, 0)] = 0.0;
    direction[(1, 0)] = 0.0;
    direction[(2, 0)] = 1.0;
    // Column 1 = internal row axis = physical -X.
    direction[(0, 1)] = -1.0;
    direction[(1, 1)] = 0.0;
    direction[(2, 1)] = 0.0;
    // Column 2 = internal column axis = physical Y.
    direction[(0, 2)] = 0.0;
    direction[(1, 2)] = 1.0;
    direction[(2, 2)] = 0.0;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        direction,
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;

    assert!(
        bytes_contain(&bytes, "(0,4,0)"),
        "sd0 must be (0,4,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(-3,0,0)"),
        "sd1 must be (-3,0,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(0,0,2)"),
        "sd2 must be (0,0,2); file content does not match"
    );

    Ok(())
}

/// A Cartesian image must produce a header with no coordinate-map field, so
/// ordinary volumes stay byte-identical and remain readable by tools that know
/// nothing about the key.
#[test]
fn cartesian_images_write_no_coordinate_map_field() -> Result<()> {
    let image = make_image(
        vec![0.0_f32; 8],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let directory = tempdir()?;
    let path = directory.path().join("plain.nrrd");
    crate::write_nrrd(&path, &image, &SequentialBackend)?;

    let header = std::fs::read_to_string(&path).unwrap_or_default();
    let header = header.split("\n\n").next().unwrap_or_default().to_string();
    assert!(
        !header.contains(crate::coordinate_map::COORDINATE_MAP_KEY),
        "a Cartesian image must not emit the key; header was:\n{header}"
    );

    let loaded: Image<f32, TestBackend, 3> =
        crate::read_nrrd::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert!(loaded.coordinate_map().is_cartesian());
    Ok(())
}
