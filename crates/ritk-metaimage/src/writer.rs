use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a 3-D `Image` to a `.mha` (MetaImage single-file) format.
///
/// # Axis convention
/// RITK stores voxels in `[Z, Y, X]` order.  This function permutes axes
/// `[2, 1, 0]` before writing, producing MetaImage-conventional `[X, Y, Z]`
/// order in the file.  The `DimSize` header field therefore contains
/// `nx ny nz` (i.e. `shape()[2] shape()[1] shape()[0]` of the RITK image).
///
/// # Spatial metadata
/// `origin`, `spacing`, and `direction` are written verbatim from the
/// `Image`'s physical-space metadata (already in `[X, Y, Z]` order).
///
/// # Binary payload
/// Voxel values are written as 32-bit IEEE 754 floats in little-endian byte
/// order immediately after the `ElementDataFile = LOCAL` header line.
pub fn write_metaimage<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    // ── Voxel data ────────────────────────────────────────────────────────
    // Permute from RITK [Z, Y, X] to MetaImage [X, Y, Z].
    let tensor = image.data().clone().permute([2, 1, 0]);
    let tensor_data = tensor.to_data();
    let f32_slice = match tensor_data.as_slice::<f32>() {
        Ok(s) => s,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to extract f32 slice from tensor data: {:?}",
                e
            ))
        }
    };

    // image.shape() is [nz, ny, nx] in RITK convention.
    // After permutation the MetaImage axes are [nx, ny, nz].
    let shape = image.shape();
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Spatial metadata ──────────────────────────────────────────────────
    let spacing = image.spacing();
    let origin = image.origin();
    let dir = image.direction().0;

    // TransformMatrix: row-major serialisation of the direction matrix.
    // Direction[(row, col)] is the (row, col) entry of the 3×3 matrix.
    // MetaImage reads it back as TransformMatrix[row * 3 + col].
    let tm = format!(
        "{} {} {} {} {} {} {} {} {}",
        dir[(0, 0)],
        dir[(0, 1)],
        dir[(0, 2)],
        dir[(1, 0)],
        dir[(1, 1)],
        dir[(1, 2)],
        dir[(2, 0)],
        dir[(2, 1)],
        dir[(2, 2)],
    );

    // ── File I/O ──────────────────────────────────────────────────────────
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MetaImage file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header — field order matches the ITK MetaImageIO convention.
    writeln!(writer, "ObjectType = Image")?;
    writeln!(writer, "NDims = 3")?;
    writeln!(writer, "BinaryData = True")?;
    writeln!(writer, "BinaryDataByteOrderMSB = False")?;
    writeln!(writer, "CompressedData = False")?;
    writeln!(writer, "TransformMatrix = {}", tm)?;
    writeln!(writer, "Offset = {} {} {}", origin[0], origin[1], origin[2])?;
    writeln!(writer, "CenterOfRotation = 0 0 0")?;
    writeln!(
        writer,
        "ElementSpacing = {} {} {}",
        spacing[0], spacing[1], spacing[2]
    )?;
    // DimSize is in MetaImage [X, Y, Z] order.
    writeln!(writer, "DimSize = {} {} {}", nx, ny, nz)?;
    writeln!(writer, "ElementType = MET_FLOAT")?;
    // LOCAL signals that binary data follows immediately.
    writeln!(writer, "ElementDataFile = LOCAL")?;

    // Binary payload — little-endian f32.
    for &v in f32_slice {
        writer.write_all(&v.to_le_bytes())?;
    }

    writer
        .flush()
        .context("Failed to flush MetaImage output file")?;

    Ok(())
}

// ── Public writer struct ──────────────────────────────────────────────────────

/// Thin writer struct for MetaImage files.
///
/// The backend `B` is supplied per-call so a single `MetaImageWriter`
/// instance can write images from different backends.
pub struct MetaImageWriter;

impl MetaImageWriter {
    /// Write `image` to the MetaImage file at `path`.
    pub fn write<B: Backend, P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> Result<()> {
        write_metaimage(path, image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
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

    /// Spacing and origin values written in the header must match the Image's
    /// spatial metadata exactly (within f64 string-roundtrip precision).
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
        // Verify that the exact formatted values appear in the header.
        assert!(
            bytes_contain(&bytes, "10.5"),
            "origin[0] not found in header"
        );
        assert!(
            bytes_contain(&bytes, "20.25"),
            "origin[1] not found in header"
        );
        assert!(
            bytes_contain(&bytes, "0.9"),
            "spacing[0] not found in header"
        );
        assert!(
            bytes_contain(&bytes, "1.5"),
            "spacing[2] not found in header"
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

    /// Non-identity direction matrix must appear verbatim in TransformMatrix.
    #[test]
    fn test_non_identity_direction_in_header() -> Result<()> {
        use nalgebra::SMatrix;

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
        // Row 0 of TransformMatrix = direction row 0 = [0, -1, 0]
        assert!(
            bytes_contain(&bytes, "TransformMatrix = 0 -1 0 1 0 0 0 0 1"),
            "rotation matrix not found in TransformMatrix field"
        );

        Ok(())
    }
}
