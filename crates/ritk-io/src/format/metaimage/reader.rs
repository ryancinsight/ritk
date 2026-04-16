use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Read a MetaImage (.mha or .mhd) file into a 3-D `Image`.
///
/// # Axis convention
/// MetaImage stores voxels in `[X, Y, Z]` order.
/// RITK stores voxels in `[Z, Y, X]` order.
/// This function permutes axes `[2, 1, 0]` after loading, so the returned
/// tensor shape is `[nz, ny, nx]`.
///
/// # Spatial metadata
/// `origin`, `spacing`, and `direction` are kept in physical `[X, Y, Z]`
/// space — exactly as stored in the MetaImage header — consistent with the
/// rest of the RITK spatial API.
///
/// # Supported element types
/// `MET_UCHAR`, `MET_SHORT`, `MET_USHORT`, `MET_INT`, `MET_UINT`,
/// `MET_FLOAT`, `MET_DOUBLE`.  All are converted to `f32` in the tensor.
///
/// # File formats
/// * `.mha` — single file; header followed immediately by binary data
///   (`ElementDataFile = LOCAL`).
/// * `.mhd` / `.raw` — ASCII header references a separate raw file.
pub fn read_metaimage<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open MetaImage file {:?}", path))?;
    let mut reader = BufReader::new(file);

    // ── Header parsing ────────────────────────────────────────────────────
    // Read line-by-line, accumulating byte offset so we can seek to the
    // binary payload after the `ElementDataFile` line.
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut byte_offset: u64 = 0;
    let mut found_edf = false;

    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .context("Error reading MetaImage header line")?;
        if n == 0 {
            break; // unexpected EOF before ElementDataFile
        }
        byte_offset += n as u64;

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // MetaImage uses " = " as the key-value separator.
        if let Some(eq_pos) = trimmed.find(" = ") {
            let key = trimmed[..eq_pos].trim().to_string();
            let value = trimmed[eq_pos + 3..].trim().to_string();
            let is_edf = key == "ElementDataFile";
            headers.insert(key, value);
            if is_edf {
                found_edf = true;
                break; // binary data starts at byte_offset
            }
        }
    }

    if !found_edf {
        return Err(anyhow!(
            "ElementDataFile key not found in MetaImage header of {:?}",
            path
        ));
    }

    // ── Required fields ───────────────────────────────────────────────────
    let ndims: usize = headers
        .get("NDims")
        .ok_or_else(|| anyhow!("Missing 'NDims' in MetaImage header"))?
        .parse()
        .context("'NDims' is not a valid integer")?;

    if ndims != 3 {
        return Err(anyhow!(
            "Expected NDims = 3 for a 3-D image, found {}",
            ndims
        ));
    }

    let dim_sizes = parse_usize_vec(
        headers
            .get("DimSize")
            .ok_or_else(|| anyhow!("Missing 'DimSize' in MetaImage header"))?,
        "DimSize",
        3,
    )?;
    let nx = dim_sizes[0];
    let ny = dim_sizes[1];
    let nz = dim_sizes[2];

    let spacing_vals = parse_f64_vec(
        headers
            .get("ElementSpacing")
            .ok_or_else(|| anyhow!("Missing 'ElementSpacing' in MetaImage header"))?,
        "ElementSpacing",
        3,
    )?;

    let offset_str = headers
        .get("Offset")
        .or_else(|| headers.get("Position"))
        .ok_or_else(|| anyhow!("Missing 'Offset' (or 'Position') in MetaImage header"))?;
    let offset_vals = parse_f64_vec(offset_str, "Offset", 3)?;

    // TransformMatrix defaults to identity when absent.
    let tm_src = headers
        .get("TransformMatrix")
        .map(|s| s.as_str())
        .unwrap_or("1 0 0 0 1 0 0 0 1");
    let tm_vals = parse_f64_vec(tm_src, "TransformMatrix", 9)?;

    let element_type = headers
        .get("ElementType")
        .ok_or_else(|| anyhow!("Missing 'ElementType' in MetaImage header"))?
        .clone();

    // BinaryDataByteOrderMSB = True → big-endian; default is little-endian.
    let msb = headers
        .get("BinaryDataByteOrderMSB")
        .map(|s| s.to_uppercase() == "TRUE")
        .unwrap_or(false);

    let element_data_file = headers
        .get("ElementDataFile")
        .ok_or_else(|| anyhow!("Missing 'ElementDataFile' in MetaImage header"))?
        .clone();

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = nx * ny * nz;

    let f32_data: Vec<f32> = if element_data_file.to_uppercase() == "LOCAL" {
        // Seek the BufReader to the exact position after the last header line.
        // BufReader<File> implements Seek; this discards any internal buffer
        // and repositions the underlying file descriptor.
        reader
            .seek(SeekFrom::Start(byte_offset))
            .context("Failed to seek to binary data in .mha file")?;
        let mut raw_bytes = Vec::new();
        reader
            .read_to_end(&mut raw_bytes)
            .context("Failed to read binary voxel data from .mha file")?;
        decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, msb)?
    } else {
        // External .raw file: resolve relative to the header file's directory.
        let raw_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&element_data_file);
        let raw_bytes = std::fs::read(&raw_path)
            .with_context(|| format!("Cannot read raw data file {:?}", raw_path))?;
        decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, msb)?
    };

    if f32_data.len() != total_voxels {
        return Err(anyhow!(
            "Voxel count mismatch: DimSize implies {} voxels but {} were decoded",
            total_voxels,
            f32_data.len()
        ));
    }

    // ── Tensor construction ───────────────────────────────────────────────
    // Create tensor in MetaImage [X, Y, Z] order, then permute to RITK [Z, Y, X].
    let shape_burn = Shape::new([nx, ny, nz]);
    let tensor_data = TensorData::new(f32_data, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);
    let tensor = tensor.permute([2, 1, 0]); // [nx, ny, nz] → [nz, ny, nx]

    // ── Spatial metadata (physical [X, Y, Z] space) ───────────────────────
    let origin = Point::new([offset_vals[0], offset_vals[1], offset_vals[2]]);
    let spacing = Spacing::new([spacing_vals[0], spacing_vals[1], spacing_vals[2]]);

    // TransformMatrix is stored row-major; Direction[(row, col)] = tm_vals[row*3+col].
    // This is identical to the ITK direction matrix convention (columns = axis directions).
    let mut dir_matrix = SMatrix::<f64, 3, 3>::zeros();
    for row in 0..3 {
        for col in 0..3 {
            dir_matrix[(row, col)] = tm_vals[row * 3 + col];
        }
    }
    let direction = Direction(dir_matrix);

    Ok(Image::new(tensor, origin, spacing, direction))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Decode a raw byte buffer into `Vec<f32>` according to `element_type`.
///
/// Precisely `count` elements are decoded; surplus bytes are ignored.
/// Returns an error when the buffer is too short or the type is unknown.
fn decode_bytes_to_f32(
    bytes: &[u8],
    element_type: &str,
    count: usize,
    msb: bool,
) -> Result<Vec<f32>> {
    match element_type {
        "MET_UCHAR" => {
            require_bytes(bytes.len(), count, 1, "MET_UCHAR")?;
            Ok(bytes[..count].iter().map(|&b| b as f32).collect())
        }
        "MET_SHORT" => {
            require_bytes(bytes.len(), count, 2, "MET_SHORT")?;
            Ok(bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    (if msb {
                        i16::from_be_bytes(b)
                    } else {
                        i16::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "MET_USHORT" => {
            require_bytes(bytes.len(), count, 2, "MET_USHORT")?;
            Ok(bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    (if msb {
                        u16::from_be_bytes(b)
                    } else {
                        u16::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "MET_INT" => {
            require_bytes(bytes.len(), count, 4, "MET_INT")?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    (if msb {
                        i32::from_be_bytes(b)
                    } else {
                        i32::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "MET_UINT" => {
            require_bytes(bytes.len(), count, 4, "MET_UINT")?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    (if msb {
                        u32::from_be_bytes(b)
                    } else {
                        u32::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "MET_FLOAT" => {
            require_bytes(bytes.len(), count, 4, "MET_FLOAT")?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    if msb {
                        f32::from_be_bytes(b)
                    } else {
                        f32::from_le_bytes(b)
                    }
                })
                .collect())
        }
        "MET_DOUBLE" => {
            require_bytes(bytes.len(), count, 8, "MET_DOUBLE")?;
            Ok(bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    (if msb {
                        f64::from_be_bytes(b)
                    } else {
                        f64::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        other => Err(anyhow!("Unsupported MetaImage ElementType: '{}'", other)),
    }
}

/// Return an error when `have` bytes are fewer than `count * elem_size`.
fn require_bytes(have: usize, count: usize, elem_size: usize, type_name: &str) -> Result<()> {
    let need = count * elem_size;
    if have < need {
        Err(anyhow!(
            "Insufficient data for {}: need {} bytes, got {}",
            type_name,
            need,
            have
        ))
    } else {
        Ok(())
    }
}

/// Parse a whitespace-separated list of `expected` `f64` values from `s`.
fn parse_f64_vec(s: &str, field: &str, expected: usize) -> Result<Vec<f64>> {
    let vals: Vec<f64> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<f64>()
                .with_context(|| format!("Invalid float in '{}': '{}'", field, t))
        })
        .collect::<Result<Vec<_>>>()?;

    if vals.len() != expected {
        return Err(anyhow!(
            "'{}' must have {} components, got {}",
            field,
            expected,
            vals.len()
        ));
    }
    Ok(vals)
}

/// Parse a whitespace-separated list of `expected` `usize` values from `s`.
fn parse_usize_vec(s: &str, field: &str, expected: usize) -> Result<Vec<usize>> {
    let vals: Vec<usize> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<usize>()
                .with_context(|| format!("Invalid integer in '{}': '{}'", field, t))
        })
        .collect::<Result<Vec<_>>>()?;

    if vals.len() != expected {
        return Err(anyhow!(
            "'{}' must have {} components, got {}",
            field,
            expected,
            vals.len()
        ));
    }
    Ok(vals)
}

// ── Public reader struct ──────────────────────────────────────────────────────

/// Thin reader struct for MetaImage files.
///
/// The backend `B` and device are supplied per-call so a single
/// `MetaImageReader` instance can serve multiple backends.
pub struct MetaImageReader;

impl MetaImageReader {
    /// Read a MetaImage file at `path` into an [`Image`] on `device`.
    pub fn read<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> Result<Image<B, 3>> {
        read_metaimage(path, device)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use nalgebra::SMatrix;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Write a minimal `.mha` file with MET_FLOAT data and identity metadata.
    fn write_minimal_mha(
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
        writeln!(f, "ObjectType = Image").unwrap();
        writeln!(f, "NDims = 3").unwrap();
        writeln!(f, "BinaryData = True").unwrap();
        writeln!(f, "BinaryDataByteOrderMSB = False").unwrap();
        writeln!(f, "CompressedData = False").unwrap();
        writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1").unwrap();
        writeln!(f, "Offset = {} {} {}", origin[0], origin[1], origin[2]).unwrap();
        writeln!(f, "CenterOfRotation = 0 0 0").unwrap();
        writeln!(
            f,
            "ElementSpacing = {} {} {}",
            spacing[0], spacing[1], spacing[2]
        )
        .unwrap();
        writeln!(f, "DimSize = {} {} {}", nx, ny, nz).unwrap();
        writeln!(f, "ElementType = MET_FLOAT").unwrap();
        writeln!(f, "ElementDataFile = LOCAL").unwrap();
        for &v in data {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // ── Shape and metadata ─────────────────────────────────────────────────

    /// The reader must permute MetaImage [X,Y,Z] to RITK [Z,Y,X].
    /// A header with `DimSize = 4 3 2` (nx=4, ny=3, nz=2) must produce
    /// an Image with shape [2, 3, 4] = [nz, ny, nx].
    #[test]
    fn test_shape_permuted_to_zyx() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("shape.mha");

        let nx = 4usize;
        let ny = 3usize;
        let nz = 2usize;
        let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
        write_minimal_mha(&path, &data, nx, ny, nz, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_metaimage::<TestBackend, _>(&path, &device)?;

        assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
        Ok(())
    }

    /// Spacing is kept in physical [X,Y,Z] order, not permuted.
    #[test]
    fn test_spacing_metadata_preserved() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("spacing.mha");
        let data = vec![0.0f32; 4 * 3 * 2];
        write_minimal_mha(&path, &data, 4, 3, 2, [0.9, 0.8, 1.5], [5.0, 6.0, 7.0]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_metaimage::<TestBackend, _>(&path, &device)?;

        assert!((image.spacing()[0] - 0.9).abs() < 1e-9);
        assert!((image.spacing()[1] - 0.8).abs() < 1e-9);
        assert!((image.spacing()[2] - 1.5).abs() < 1e-9);

        assert!((image.origin()[0] - 5.0).abs() < 1e-9);
        assert!((image.origin()[1] - 6.0).abs() < 1e-9);
        assert!((image.origin()[2] - 7.0).abs() < 1e-9);
        Ok(())
    }

    /// Identity TransformMatrix must produce an identity Direction.
    #[test]
    fn test_identity_direction() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("dir.mha");
        let data = vec![0.0f32; 2 * 2 * 2];
        write_minimal_mha(&path, &data, 2, 2, 2, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_metaimage::<TestBackend, _>(&path, &device)?;

        let d = image.direction().0;
        for row in 0..3usize {
            for col in 0..3usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (d[(row, col)] - expected).abs() < 1e-9,
                    "Direction[{},{}] = {} != {}",
                    row,
                    col,
                    d[(row, col)],
                    expected
                );
            }
        }
        Ok(())
    }

    // ── Round-trip ─────────────────────────────────────────────────────────

    /// Write an Image via `write_metaimage` and read it back; verify shape,
    /// spatial metadata, and every voxel value.
    #[test]
    fn test_round_trip_mha() -> Result<()> {
        use crate::format::metaimage::write_metaimage;

        let dir = tempdir()?;
        let path = dir.path().join("round_trip.mha");
        let device: <TestBackend as Backend>::Device = Default::default();

        // RITK [Z,Y,X] shape [2, 3, 4] with analytically known values 0..23.
        let data_vec: Vec<f32> = (0u32..24).map(|i| i as f32).collect();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data_vec.clone(), Shape::new([2, 3, 4])),
            &device,
        );
        let origin = Point::new([10.0, 20.0, 30.0]);
        let spacing = Spacing::new([0.9, 0.8, 1.5]);
        let direction = Direction(SMatrix::identity());
        let image = Image::new(tensor, origin, spacing, direction);

        write_metaimage(&path, &image)?;
        let loaded = read_metaimage::<TestBackend, _>(&path, &device)?;

        // Shape
        assert_eq!(loaded.shape(), [2, 3, 4]);

        // Origin
        assert!((loaded.origin()[0] - 10.0).abs() < 1e-5);
        assert!((loaded.origin()[1] - 20.0).abs() < 1e-5);
        assert!((loaded.origin()[2] - 30.0).abs() < 1e-5);

        // Spacing
        assert!((loaded.spacing()[0] - 0.9).abs() < 1e-5);
        assert!((loaded.spacing()[1] - 0.8).abs() < 1e-5);
        assert!((loaded.spacing()[2] - 1.5).abs() < 1e-5);

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

    // ── Error paths ────────────────────────────────────────────────────────

    /// Reading a non-existent file must return an error (not panic).
    #[test]
    fn test_missing_file_returns_error() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_metaimage::<TestBackend, _>("/nonexistent/path/file.mha", &device);
        assert!(result.is_err(), "Expected Err for missing file, got Ok");
    }

    /// A file that is missing a required header field must return an error.
    #[test]
    fn test_missing_required_field_returns_error() -> Result<()> {
        use std::io::Write;
        let dir = tempdir()?;
        let path = dir.path().join("bad_header.mha");
        {
            let mut f = std::fs::File::create(&path)?;
            // Intentionally omits DimSize.
            writeln!(f, "ObjectType = Image")?;
            writeln!(f, "NDims = 3")?;
            writeln!(f, "ElementSpacing = 1 1 1")?;
            writeln!(f, "Offset = 0 0 0")?;
            writeln!(f, "ElementType = MET_FLOAT")?;
            writeln!(f, "ElementDataFile = LOCAL")?;
        }
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_metaimage::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Expected Err for missing DimSize");
        Ok(())
    }

    /// Unsupported ElementType must return a descriptive error.
    #[test]
    fn test_unsupported_element_type_returns_error() -> Result<()> {
        use std::io::Write;
        let dir = tempdir()?;
        let path = dir.path().join("bad_type.mha");
        {
            let mut f = std::fs::File::create(&path)?;
            writeln!(f, "ObjectType = Image")?;
            writeln!(f, "NDims = 3")?;
            writeln!(f, "DimSize = 2 2 2")?;
            writeln!(f, "ElementSpacing = 1 1 1")?;
            writeln!(f, "Offset = 0 0 0")?;
            writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
            writeln!(f, "ElementType = MET_LONG")?; // not supported
            writeln!(f, "ElementDataFile = LOCAL")?;
            // Write 8*8 = 64 bytes of dummy data (MET_LONG = 8 bytes each, 8 voxels)
            let dummy = vec![0u8; 64];
            f.write_all(&dummy)?;
        }
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_metaimage::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Expected Err for unsupported ElementType");
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("MET_LONG"),
            "Error message must name the unsupported type; got: {}",
            msg
        );
        Ok(())
    }

    /// External `.raw` file referenced from a `.mhd` header must be read.
    #[test]
    fn test_mhd_external_raw_file() -> Result<()> {
        use std::io::Write;
        let dir = tempdir()?;
        let mhd_path = dir.path().join("volume.mhd");
        let raw_path = dir.path().join("volume.raw");

        let nx = 2usize;
        let ny = 2usize;
        let nz = 2usize;
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

        // Write raw binary
        {
            let mut f = std::fs::File::create(&raw_path)?;
            for &v in &data {
                f.write_all(&v.to_le_bytes())?;
            }
        }

        // Write header referencing the raw file by name
        {
            let mut f = std::fs::File::create(&mhd_path)?;
            writeln!(f, "ObjectType = Image")?;
            writeln!(f, "NDims = 3")?;
            writeln!(f, "BinaryData = True")?;
            writeln!(f, "BinaryDataByteOrderMSB = False")?;
            writeln!(f, "CompressedData = False")?;
            writeln!(f, "TransformMatrix = 1 0 0 0 1 0 0 0 1")?;
            writeln!(f, "Offset = 0 0 0")?;
            writeln!(f, "CenterOfRotation = 0 0 0")?;
            writeln!(f, "ElementSpacing = 1 1 1")?;
            writeln!(f, "DimSize = {} {} {}", nx, ny, nz)?;
            writeln!(f, "ElementType = MET_FLOAT")?;
            writeln!(f, "ElementDataFile = volume.raw")?;
        }

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_metaimage::<TestBackend, _>(&mhd_path, &device)?;

        // Shape must be [nz, ny, nx] = [2, 2, 2]
        assert_eq!(image.shape(), [nz, ny, nx]);

        // Voxels: total = 8; verify round-trip of each value
        let loaded_td = image.data().clone().to_data();
        let loaded_vals = loaded_td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), 8);

        // The sum of all voxels must equal 0+1+…+7 = 28 regardless of permutation.
        let sum: f32 = loaded_vals.iter().sum();
        assert!(
            (sum - 28.0).abs() < 1e-5,
            "Voxel sum mismatch: expected 28, got {}",
            sum
        );

        Ok(())
    }
}
