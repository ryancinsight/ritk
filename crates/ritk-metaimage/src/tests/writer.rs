use crate::write_metaimage_with_data;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use ritk_image::Image;

type TestBackend = SequentialBackend;

fn make_image(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(data, dims, origin, spacing, direction, &SequentialBackend)
        .expect("valid image")
}

// ── Header content ─────────────────────────────────────────────────────

/// Scan `bytes` for the ASCII `needle`; returns true when found.
fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
    let nb = needle.as_bytes();
    haystack.windows(nb.len()).any(|w| w == nb)
}

fn payload_values(bytes: &[u8]) -> Vec<f32> {
    let marker = b"ElementDataFile = LOCAL\n";
    let header_end = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .map(|p| p + marker.len())
        .expect("ElementDataFile = LOCAL not found in file");

    bytes[header_end..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// A written `.mha` file must contain the mandatory header fields.
#[test]
fn test_header_fields_present() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("header_check.mha");
    let backend = SequentialBackend;

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let image = make_image(
        vec![1.0f32; 2 * 3 * 4],
        [2, 3, 4],
        origin,
        spacing,
        direction,
    );

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;

    assert!(bytes_contain(&bytes, "NDims = 3"), "missing NDims");
    assert!(
        bytes_contain(&bytes, "ElementType = MET_FLOAT"),
        "missing ElementType"
    );
    assert!(
        bytes_contain(&bytes, "ElementDataFile = LOCAL"),
        "missing ElementDataFile"
    );
    assert!(
        bytes_contain(&bytes, "BinaryDataByteOrderMSB = False"),
        "missing byte-order field"
    );
    assert!(
        bytes_contain(&bytes, "CompressedData = False"),
        "missing CompressedData"
    );

    Ok(())
}

/// `DimSize` must be written as `nx ny nz` — the MetaImage [X,Y,Z] order,
/// which is the reverse of RITK's [Z,Y,X] convention.
/// An Image with shape [nz=2, ny=3, nx=4] must produce `DimSize = 4 3 2`.
#[test]
fn test_dimsize_written_in_xyz_order() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("dimsize.mha");
    let backend = SequentialBackend;

    // RITK shape [nz=2, ny=3, nx=4]
    let image = make_image(
        vec![0.0f32; 2 * 3 * 4],
        [2, 3, 4],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "DimSize = 4 3 2"),
        "DimSize must be nx ny nz = 4 3 2 for RITK shape [2, 3, 4]"
    );

    Ok(())
}

/// Internal spacing [z,y,x] must be written as MetaImage ElementSpacing [x,y,z].
#[test]
fn test_spatial_metadata_in_header() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spatial.mha");
    let backend = SequentialBackend;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([10.5, 20.25, 30.125]),
        Spacing::new([0.9, 0.75, 1.5]),
        Direction::identity(),
    );

    crate::write_metaimage(&path, &image, &backend)?;

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
        bytes_contain(&bytes, "ElementSpacing = 1.5 0.75 0.9"),
        "ElementSpacing must be written in MetaImage [x,y,z] order"
    );

    Ok(())
}

#[test]
fn test_internal_identity_direction_written_as_file_axis_reorder() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("identity_direction.mha");
    let backend = SequentialBackend;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "TransformMatrix = 0 0 1 0 1 0 1 0 0"),
        "internal [z,y,x] identity must be serialized in file [x,y,z] axis order"
    );

    Ok(())
}

/// The binary payload size must equal `nx * ny * nz * 4` bytes
/// (one 4-byte little-endian f32 per voxel) appended after the header.
#[test]
fn test_payload_size_correct() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload.mha");
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

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let expected_payload_bytes = n_voxels * 4;

    // Locate the end of the header by finding "ElementDataFile = LOCAL\n".
    let marker = b"ElementDataFile = LOCAL\n";
    let header_end = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .map(|p| p + marker.len())
        .expect("ElementDataFile = LOCAL not found in file");

    let actual_payload_bytes = bytes.len() - header_end;
    assert_eq!(
        actual_payload_bytes, expected_payload_bytes,
        "payload is {} bytes; expected {} ({} voxels × 4)",
        actual_payload_bytes, expected_payload_bytes, n_voxels
    );

    Ok(())
}

#[test]
fn test_payload_values_written_without_permutation() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload_values.mha");
    let backend = SequentialBackend;

    let nz = 2usize;
    let ny = 2usize;
    let nx = 3usize;
    let values: Vec<f32> = (0..(nz * ny * nx)).map(|value| value as f32).collect();
    let image = make_image(
        values.clone(),
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    assert_eq!(payload_values(&bytes), values);

    Ok(())
}

#[test]
fn test_caller_payload_length_must_match_shape() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wrong_payload.mha");
    let image = make_image(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    let error = write_metaimage_with_data(&path, &image, &[0.0; 7])
        .expect_err("short payload must be rejected");
    assert!(
        error.to_string().contains("requires 8"),
        "error must report the required voxel count: {error}"
    );
    assert!(!path.exists(), "invalid payload must not create a file");
    Ok(())
}

/// MetaImageWriter struct delegates correctly to `write_metaimage`.
#[test]
fn test_writer_struct_creates_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("writer_struct.mha");
    let backend = SequentialBackend;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    crate::write_metaimage(&path, &image, &backend)?;

    assert!(path.exists(), "output file must exist after write");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "output file must be non-empty"
    );

    Ok(())
}

/// Non-identity internal direction columns must be reordered into file axes.
#[test]
fn test_non_identity_direction_reordered_in_header() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rotated.mha");
    let backend = SequentialBackend;

    // 90-degree rotation around Z: X→Y, Y→−X, Z→Z
    let mut direction = Direction::zeros();
    direction[(0, 0)] = 0.0;
    direction[(0, 1)] = -1.0;
    direction[(0, 2)] = 0.0;
    direction[(1, 0)] = 1.0;
    direction[(1, 1)] = 0.0;
    direction[(1, 2)] = 0.0;
    direction[(2, 0)] = 0.0;
    direction[(2, 1)] = 0.0;
    direction[(2, 2)] = 1.0;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        direction,
    );

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    // ITK MetaImage TransformMatrix is row-major with each row an axis direction
    // cosine (the transpose of the column-major direction matrix). For this image
    // the file-axis directions (after the [x,y,z]↔[z,y,x] reorder) are emitted as
    // rows: 0 0 1 / -1 0 0 / 0 1 0.
    assert!(
        bytes_contain(&bytes, "TransformMatrix = 0 0 1 -1 0 0 0 1 0"),
        "TransformMatrix must be ITK row-major axis directions, got header:\n{}",
        String::from_utf8_lossy(&bytes[..bytes.len().min(300)])
    );

    // Round-trip: reading the written file must recover the exact direction.
    let read_back = crate::read_metaimage(&path, &backend)?;
    let got = read_back.direction().0;
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (got[(i, j)] - direction[(i, j)]).abs() < 1e-9,
                "direction round-trip mismatch at ({i},{j}): got {} want {}",
                got[(i, j)],
                direction[(i, j)]
            );
        }
    }

    Ok(())
}
