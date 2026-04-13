use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a 3-D `Image` to a NRRD (Nearly Raw Raster Data) file.
///
/// # Format
/// Writes NRRD version 4 (`NRRD0004`) with `encoding: raw` and
/// `endian: little`.  The file is self-contained (inline data).
///
/// # Axis convention
/// RITK stores voxels in `[Z, Y, X]` order.  This function permutes axes
/// `[2, 1, 0]` before writing, producing NRRD-conventional `[X, Y, Z]`
/// order.  The `sizes` header field therefore contains `nx ny nz`
/// (i.e. `shape()[2] shape()[1] shape()[0]` of the RITK image).
///
/// # Spatial metadata
/// * `space directions` — each vector is the `i`-th column of the direction
///   matrix scaled by `spacing[i]`:
///   `v_i = D[:, i] * spacing[i]`.
/// * `space origin` — the image origin in physical `[X, Y, Z]` space.
///
/// # Binary payload
/// Voxel values are written as 32-bit IEEE 754 floats in little-endian byte
/// order, immediately after a blank header-terminator line.
pub fn write_nrrd<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    // ── Voxel data ────────────────────────────────────────────────────────
    // Permute from RITK [Z, Y, X] to NRRD [X, Y, Z].
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
    // After permutation the NRRD axes are [nx, ny, nz].
    let shape = image.shape();
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Spatial metadata ──────────────────────────────────────────────────
    let spacing = image.spacing();
    let origin = image.origin();
    let dir = image.direction().0;

    // space directions: i-th vector = D[:, i] * spacing[i]
    // D[:, i] = [D[0,i], D[1,i], D[2,i]]^T
    // Written as "(D[0,i]*sp, D[1,i]*sp, D[2,i]*sp)" for i = 0, 1, 2.
    let sd0 = format!(
        "({},{},{})",
        dir[(0, 0)] * spacing[0],
        dir[(1, 0)] * spacing[0],
        dir[(2, 0)] * spacing[0],
    );
    let sd1 = format!(
        "({},{},{})",
        dir[(0, 1)] * spacing[1],
        dir[(1, 1)] * spacing[1],
        dir[(2, 1)] * spacing[1],
    );
    let sd2 = format!(
        "({},{},{})",
        dir[(0, 2)] * spacing[2],
        dir[(1, 2)] * spacing[2],
        dir[(2, 2)] * spacing[2],
    );

    let space_origin = format!("({},{},{})", origin[0], origin[1], origin[2]);

    // ── File I/O ──────────────────────────────────────────────────────────
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header — field order matches the ITK NrrdIO convention.
    writeln!(writer, "NRRD0004")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: float")?;
    writeln!(writer, "dimension: 3")?;
    writeln!(writer, "space: right-anterior-superior")?;
    // sizes is in NRRD [X, Y, Z] order.
    writeln!(writer, "sizes: {} {} {}", nx, ny, nz)?;
    writeln!(writer, "space directions: {} {} {}", sd0, sd1, sd2)?;
    writeln!(writer, "kinds: domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(writer, "space origin: {}", space_origin)?;
    // Blank line terminates the header; binary data follows immediately.
    writeln!(writer)?;

    // Binary payload — little-endian f32.
    for &v in f32_slice {
        writer.write_all(&v.to_le_bytes())?;
    }

    writer.flush().context("Failed to flush NRRD output file")?;

    Ok(())
}

// ── Public writer struct ──────────────────────────────────────────────────────

/// Thin writer struct for NRRD files.
///
/// The backend `B` is supplied per-call so a single `NrrdWriter` instance can
/// write images from different backends.
pub struct NrrdWriter;

impl NrrdWriter {
    /// Write `image` to the NRRD file at `path`.
    pub fn write<B: Backend, P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> Result<()> {
        write_nrrd(path, image)
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

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Scan `haystack` for the ASCII byte pattern `needle`.
    fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
        let nb = needle.as_bytes();
        haystack.windows(nb.len()).any(|w| w == nb)
    }

    // ── Header content ─────────────────────────────────────────────────────

    /// A written NRRD file must contain the mandatory header fields.
    #[test]
    fn test_mandatory_header_fields_present() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("mandatory.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0f32; 2 * 3 * 4], Shape::new([2, 3, 4])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction(SMatrix::identity()),
        );

        write_nrrd(&path, &image)?;

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

        write_nrrd(&path, &image)?;

        let bytes = std::fs::read(&path)?;
        assert!(
            bytes_contain(&bytes, "sizes: 4 3 2"),
            "sizes must be nx ny nz = 4 3 2 for RITK shape [2, 3, 4]"
        );

        Ok(())
    }

    /// For an identity direction matrix, `space directions` must encode the
    /// spacing values on the diagonal:
    ///   `(sx,0,0) (0,sy,0) (0,0,sz)`.
    #[test]
    fn test_space_directions_encodes_spacing_on_diagonal() -> Result<()> {
        let dir_tmp = tempdir()?;
        let path = dir_tmp.path().join("diag_sd.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([0.9, 0.75, 1.5]),
            Direction(SMatrix::identity()),
        );

        write_nrrd(&path, &image)?;

        let bytes = std::fs::read(&path)?;
        // For identity direction:
        //   sd0 = (sp[0]*1, sp[0]*0, sp[0]*0) = (0.9,0,0)
        //   sd1 = (sp[1]*0, sp[1]*1, sp[1]*0) = (0,0.75,0)
        //   sd2 = (sp[2]*0, sp[2]*0, sp[2]*1) = (0,0,1.5)
        assert!(
            bytes_contain(&bytes, "(0.9,0,0)"),
            "sd0 must encode sx on first component"
        );
        assert!(
            bytes_contain(&bytes, "(0,0.75,0)"),
            "sd1 must encode sy on second component"
        );
        assert!(
            bytes_contain(&bytes, "(0,0,1.5)"),
            "sd2 must encode sz on third component"
        );

        Ok(())
    }

    /// `space origin` must contain the image origin coordinates.
    #[test]
    fn test_space_origin_written_correctly() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("origin.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
        let image = Image::new(
            tensor,
            Point::new([10.5, 20.25, 30.125]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction(SMatrix::identity()),
        );

        write_nrrd(&path, &image)?;

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

        write_nrrd(&path, &image)?;

        let bytes = std::fs::read(&path)?;
        let expected_payload = n_voxels * 4;

        // Locate the blank-line header terminator ("\n\n").
        // The blank line is written by `writeln!(writer)?` after the last field.
        let terminator = b"\n\n";
        let header_end = bytes
            .windows(terminator.len())
            .position(|w| w == terminator)
            .map(|p| p + terminator.len())
            .expect("blank-line terminator not found in NRRD file");

        let actual_payload = bytes.len() - header_end;
        assert_eq!(
            actual_payload, expected_payload,
            "payload is {} bytes; expected {} ({} voxels × 4)",
            actual_payload, expected_payload, n_voxels
        );

        Ok(())
    }

    /// NrrdWriter struct delegates correctly to `write_nrrd`.
    #[test]
    fn test_writer_struct_creates_file() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("writer_struct.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction(SMatrix::identity()),
        );

        let writer = NrrdWriter;
        writer.write(&path, &image)?;

        assert!(path.exists(), "output file must exist after write");
        assert!(
            std::fs::metadata(&path)?.len() > 0,
            "output file must be non-empty"
        );

        Ok(())
    }

    /// Non-identity direction matrix must appear in `space directions` as
    /// correctly scaled column vectors.
    ///
    /// Test: 90-degree rotation around Z (X→Y, Y→−X, Z→Z) with spacing [2,3,4].
    ///   D[:, 0] = [0, 1, 0]^T  →  sd0 = (0*2, 1*2, 0*2) = (0,2,0)
    ///   D[:, 1] = [-1, 0, 0]^T →  sd1 = (-1*3, 0*3, 0*3) = (-3,0,0)
    ///   D[:, 2] = [0, 0, 1]^T  →  sd2 = (0*4, 0*4, 1*4) = (0,0,4)
    #[test]
    fn test_rotated_direction_in_space_directions() -> Result<()> {
        let dir_tmp = tempdir()?;
        let path = dir_tmp.path().join("rotated.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let tensor = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);

        // 90-degree rotation around Z axis:
        //   X image axis → physical Y direction
        //   Y image axis → physical -X direction
        //   Z image axis → physical Z direction
        let mut mat = SMatrix::<f64, 3, 3>::zeros();
        // Column 0 = direction of X axis = (0, 1, 0)
        mat[(0, 0)] = 0.0;
        mat[(1, 0)] = 1.0;
        mat[(2, 0)] = 0.0;
        // Column 1 = direction of Y axis = (-1, 0, 0)
        mat[(0, 1)] = -1.0;
        mat[(1, 1)] = 0.0;
        mat[(2, 1)] = 0.0;
        // Column 2 = direction of Z axis = (0, 0, 1)
        mat[(0, 2)] = 0.0;
        mat[(1, 2)] = 0.0;
        mat[(2, 2)] = 1.0;

        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([2.0, 3.0, 4.0]),
            Direction(mat),
        );

        write_nrrd(&path, &image)?;

        let bytes = std::fs::read(&path)?;

        // sd0 = col0 * sp[0] = (0*2, 1*2, 0*2) = (0,2,0)
        assert!(
            bytes_contain(&bytes, "(0,2,0)"),
            "sd0 must be (0,2,0); file content does not match"
        );
        // sd1 = col1 * sp[1] = (-1*3, 0*3, 0*3) = (-3,0,0)
        assert!(
            bytes_contain(&bytes, "(-3,0,0)"),
            "sd1 must be (-3,0,0); file content does not match"
        );
        // sd2 = col2 * sp[2] = (0*4, 0*4, 1*4) = (0,0,4)
        assert!(
            bytes_contain(&bytes, "(0,0,4)"),
            "sd2 must be (0,0,4); file content does not match"
        );

        Ok(())
    }

    /// Write an Image via `write_nrrd` and read it back; verify shape,
    /// spatial metadata, and every voxel value.
    #[test]
    fn test_round_trip_nrrd() -> Result<()> {
        use crate::format::nrrd::read_nrrd;

        let dir = tempdir()?;
        let path = dir.path().join("round_trip.nrrd");
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
        let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data_vec.clone(), Shape::new([2, 3, 4])),
            &device,
        );
        let origin = Point::new([10.0, 20.0, 30.0]);
        let spacing = Spacing::new([0.9, 0.75, 1.5]);
        let direction = Direction(SMatrix::identity());
        let image = Image::new(tensor, origin, spacing, direction);

        write_nrrd(&path, &image)?;
        let loaded = read_nrrd::<TestBackend, _>(&path, &device)?;

        // Shape
        assert_eq!(loaded.shape(), [2, 3, 4]);

        // Origin
        assert!((loaded.origin()[0] - 10.0).abs() < 1e-6, "origin[0]");
        assert!((loaded.origin()[1] - 20.0).abs() < 1e-6, "origin[1]");
        assert!((loaded.origin()[2] - 30.0).abs() < 1e-6, "origin[2]");

        // Spacing
        assert!((loaded.spacing()[0] - 0.9).abs() < 1e-6, "spacing[0]");
        assert!((loaded.spacing()[1] - 0.75).abs() < 1e-6, "spacing[1]");
        assert!((loaded.spacing()[2] - 1.5).abs() < 1e-6, "spacing[2]");

        // Voxel values: every element must equal its original value.
        let loaded_td = loaded.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got
            );
        }

        Ok(())
    }
}
