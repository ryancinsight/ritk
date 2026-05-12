use crate::{write_metaimage, MetaImageWriter};
use anyhow::Result;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 2 * 3 * 4], Shape::new([2, 3, 4])),
        &device,
    );
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction(SMatrix::identity());
    let image = Image::new(tensor, origin, spacing, direction);

    write_metaimage(&path, &image)?;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // RITK shape [nz=2, ny=3, nx=4]
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0f32; 2 * 3 * 4], Shape::new([2, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_metaimage(&path, &image)?;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([10.5, 20.25, 30.125]),
        Spacing::new([0.9, 0.75, 1.5]),
        Direction(SMatrix::identity()),
    );

    write_metaimage(&path, &image)?;

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
    let device: <TestBackend as Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_metaimage(&path, &image)?;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let nz = 3usize;
    let ny = 4usize;
    let nx = 5usize;
    let n_voxels = nz * ny * nx;

    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; n_voxels], Shape::new([nz, ny, nx])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_metaimage(&path, &image)?;

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
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 2usize;
    let ny = 2usize;
    let nx = 3usize;
    let values: Vec<f32> = (0..(nz * ny * nx)).map(|value| value as f32).collect();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(values.clone(), Shape::new([nz, ny, nx])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    write_metaimage(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    assert_eq!(payload_values(&bytes), values);

    Ok(())
}

/// MetaImageWriter struct delegates correctly to `write_metaimage`.
#[test]
fn test_writer_struct_creates_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("writer_struct.mha");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(SMatrix::identity()),
    );

    let writer = MetaImageWriter;
    writer.write(&path, &image)?;

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
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
    // 90-degree rotation around Z: X→Y, Y→−X, Z→Z
    let mut mat = SMatrix::<f64, 3, 3>::zeros();
    mat[(0, 0)] = 0.0;
    mat[(0, 1)] = -1.0;
    mat[(0, 2)] = 0.0;
    mat[(1, 0)] = 1.0;
    mat[(1, 1)] = 0.0;
    mat[(1, 2)] = 0.0;
    mat[(2, 0)] = 0.0;
    mat[(2, 1)] = 0.0;
    mat[(2, 2)] = 1.0;

    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction(mat),
    );

    write_metaimage(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    assert!(
        bytes_contain(&bytes, "TransformMatrix = 0 -1 0 0 0 1 1 0 0"),
        "internal direction columns must be serialized as MetaImage [x,y,z] columns"
    );

    Ok(())
}
