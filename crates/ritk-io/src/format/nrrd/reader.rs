use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Read a NRRD (Nearly Raw Raster Data) file into a 3-D `Image`.
///
/// # Axis convention
/// NRRD files produced by ITK-compatible tools store voxels in `[X, Y, Z]`
/// order (fastest-varying axis first in the `sizes` field).  This function
/// permutes axes `[2, 1, 0]` after loading, so the returned tensor shape is
/// `[nz, ny, nx]` (RITK `[Z, Y, X]` convention).
///
/// # Spatial metadata
/// `origin`, `spacing`, and `direction` are kept in physical `[X, Y, Z]`
/// space — consistent with the rest of the RITK spatial API.
///
/// Direction and spacing are derived from `space directions` when that field
/// is present, falling back to `spacings` (identity direction) otherwise.
///
/// # Encoding
/// Only `raw` encoding is supported.  Files with `gzip` or any other encoding
/// return an error with an actionable message.
///
/// # Supported types
/// `float`, `double`, `short`, `unsigned short`, `int`, `unsigned int`,
/// `uchar` / `unsigned char`, `char` / `signed char`.
/// All are converted to `f32` in the tensor.
///
/// # Inline vs. detached data
/// * Inline: no `data file` field (or `data file: INTERNAL`) — binary data
///   follows the blank header-terminator line in the same file.
/// * Detached: `data file: <filename>` — binary data is in a separate file
///   resolved relative to the NRRD header file's directory.
pub fn read_nrrd<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();

    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open NRRD file {:?}", path))?;
    let mut reader = BufReader::new(file);

    // ── Magic line ────────────────────────────────────────────────────────
    let mut magic = String::new();
    reader
        .read_line(&mut magic)
        .context("Failed to read NRRD magic line")?;
    if !magic.trim_start().starts_with("NRRD") {
        return Err(anyhow!(
            "Not a valid NRRD file: magic line does not start with 'NRRD' (got '{}')",
            magic.trim()
        ));
    }

    // ── Header parsing ────────────────────────────────────────────────────
    // Keys are lowercased for case-insensitive lookup.
    // The first ':' is the key-value separator (handles keys containing spaces).
    let mut headers: HashMap<String, String> = HashMap::new();

    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .context("Error reading NRRD header line")?;
        if n == 0 {
            break; // EOF without blank-line terminator
        }

        let trimmed = line.trim();

        // Blank line signals end of header; data follows immediately.
        if trimmed.is_empty() {
            break;
        }

        // Skip comment lines.
        if trimmed.starts_with('#') {
            continue;
        }

        // Key-value pairs are separated by ": " (NRRD spec §3).
        // We split on the first ':' and trim whitespace from the value.
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim().to_lowercase();
            let value = trimmed[colon_pos + 1..].trim().to_string();
            headers.insert(key, value);
        }
    }

    // ── Required fields ───────────────────────────────────────────────────
    let element_type = headers
        .get("type")
        .ok_or_else(|| anyhow!("Missing 'type' in NRRD header"))?
        .clone();

    let dimension: usize = headers
        .get("dimension")
        .ok_or_else(|| anyhow!("Missing 'dimension' in NRRD header"))?
        .parse()
        .context("'dimension' is not a valid integer")?;

    if dimension != 3 {
        return Err(anyhow!(
            "Expected dimension = 3 for a 3-D NRRD file, found {}",
            dimension
        ));
    }

    let sizes_str = headers
        .get("sizes")
        .ok_or_else(|| anyhow!("Missing 'sizes' in NRRD header"))?;
    let sizes = parse_usize_vec(sizes_str, "sizes", 3)?;
    let nx = sizes[0];
    let ny = sizes[1];
    let nz = sizes[2];

    // ── Encoding ──────────────────────────────────────────────────────────
    let encoding = headers
        .get("encoding")
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "raw".to_string());

    if encoding != "raw" {
        return Err(anyhow!(
            "Unsupported NRRD encoding '{}'. Only 'raw' is supported. \
             Convert the file to raw encoding first (e.g., with \
             `unu convert -i input.nrrd -o output.nrrd -e raw`).",
            encoding
        ));
    }

    // ── Endianness ────────────────────────────────────────────────────────
    // True ⟹ big-endian.  Default is little-endian for multi-byte types.
    let msb = headers
        .get("endian")
        .map(|s| s.to_lowercase() == "big")
        .unwrap_or(false);

    // ── Spacing and direction ─────────────────────────────────────────────
    let (spacing, direction) = if let Some(sd_str) = headers.get("space directions") {
        parse_space_directions(sd_str)?
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp_vals = parse_f64_vec(sp_str, "spacings", 3)?;
        (
            Spacing::new([sp_vals[0], sp_vals[1], sp_vals[2]]),
            Direction::identity(),
        )
    } else {
        // Neither field present: unit spacing, identity direction.
        (Spacing::new([1.0, 1.0, 1.0]), Direction::identity())
    };

    // ── Origin ────────────────────────────────────────────────────────────
    let origin = if let Some(so_str) = headers.get("space origin") {
        parse_nrrd_point(so_str)?
    } else {
        Point::new([0.0, 0.0, 0.0])
    };

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = nx * ny * nz;
    let data_file_field = headers.get("data file").cloned();

    let f32_data: Vec<f32> = match &data_file_field {
        // Inline: data follows the blank terminator line in the same file.
        None => {
            let mut raw_bytes = Vec::new();
            reader
                .read_to_end(&mut raw_bytes)
                .context("Failed to read inline NRRD binary data")?;
            decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, msb)?
        }
        Some(df) if df.to_uppercase() == "INTERNAL" => {
            let mut raw_bytes = Vec::new();
            reader
                .read_to_end(&mut raw_bytes)
                .context("Failed to read inline NRRD binary data (INTERNAL)")?;
            decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, msb)?
        }
        // Detached: data is in an external file.
        Some(df) => {
            let raw_path = path.parent().unwrap_or_else(|| Path::new(".")).join(df);
            let raw_bytes = std::fs::read(&raw_path)
                .with_context(|| format!("Cannot read NRRD data file {:?}", raw_path))?;
            decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, msb)?
        }
    };

    if f32_data.len() != total_voxels {
        return Err(anyhow!(
            "NRRD voxel count mismatch: sizes implies {} voxels but {} were decoded",
            total_voxels,
            f32_data.len()
        ));
    }

    // ── Tensor construction ───────────────────────────────────────────────
    // Create tensor in NRRD [X, Y, Z] order, then permute to RITK [Z, Y, X].
    let shape_burn = Shape::new([nx, ny, nz]);
    let tensor_data = TensorData::new(f32_data, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);
    let tensor = tensor.permute([2, 1, 0]); // [nx, ny, nz] → [nz, ny, nx]

    Ok(Image::new(tensor, origin, spacing, direction))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Parse a `space directions` field into `(Spacing<3>, Direction<3>)`.
///
/// Each direction vector `v_i` encodes the physical displacement per voxel
/// step along image axis `i`:
/// ```text
/// v_i = Direction[:, i] * spacing[i]
/// spacing[i] = |v_i|
/// Direction[:, i] = v_i / |v_i|
/// ```
fn parse_space_directions(s: &str) -> Result<(Spacing<3>, Direction<3>)> {
    let vecs = parse_parenthesized_vectors(s)?;
    if vecs.len() != 3 {
        return Err(anyhow!(
            "'space directions' must contain 3 vectors, found {}",
            vecs.len()
        ));
    }

    let mut dir_matrix = SMatrix::<f64, 3, 3>::zeros();
    let mut spacing_vals = [0.0_f64; 3];

    for (i, v) in vecs.iter().enumerate() {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        spacing_vals[i] = norm;
        if norm > 1e-12 {
            dir_matrix[(0, i)] = v[0] / norm;
            dir_matrix[(1, i)] = v[1] / norm;
            dir_matrix[(2, i)] = v[2] / norm;
        } else {
            // Degenerate zero vector: fall back to the standard basis vector.
            dir_matrix[(i, i)] = 1.0;
        }
    }

    Ok((Spacing::new(spacing_vals), Direction(dir_matrix)))
}

/// Parse a `space origin` field into a `Point<3>`.
///
/// The field value must contain exactly one `(v0,v1,v2)` group.
fn parse_nrrd_point(s: &str) -> Result<Point<3>> {
    let vecs = parse_parenthesized_vectors(s)?;
    if vecs.is_empty() {
        return Err(anyhow!(
            "Invalid 'space origin' format: no parenthesised vector found in '{}'",
            s
        ));
    }
    Ok(Point::new([vecs[0][0], vecs[0][1], vecs[0][2]]))
}

/// Extract all `(v0,v1,v2)` groups from `s` and return them as `Vec<[f64;3]>`.
///
/// Handles spaces inside or between components; stops at any malformed group.
fn parse_parenthesized_vectors(s: &str) -> Result<Vec<[f64; 3]>> {
    let mut vecs: Vec<[f64; 3]> = Vec::new();
    let mut rest = s.trim();

    while let Some(start) = rest.find('(') {
        rest = &rest[start + 1..];
        if let Some(end) = rest.find(')') {
            let inner = &rest[..end];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() != 3 {
                return Err(anyhow!(
                    "Expected 3 components in vector '({})'; got {}",
                    inner,
                    parts.len()
                ));
            }
            let v = [
                parts[0]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[0].trim()))?,
                parts[1]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[1].trim()))?,
                parts[2]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[2].trim()))?,
            ];
            vecs.push(v);
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }

    Ok(vecs)
}

/// Decode a raw byte buffer into `Vec<f32>` according to the NRRD `type`.
///
/// Precisely `count` elements are decoded; surplus trailing bytes are ignored.
fn decode_bytes_to_f32(
    bytes: &[u8],
    element_type: &str,
    count: usize,
    msb: bool,
) -> Result<Vec<f32>> {
    match element_type.to_lowercase().as_str() {
        "uchar" | "unsigned char" | "uint8" => {
            require_bytes(bytes.len(), count, 1, element_type)?;
            Ok(bytes[..count].iter().map(|&b| b as f32).collect())
        }
        "char" | "signed char" | "int8" => {
            require_bytes(bytes.len(), count, 1, element_type)?;
            Ok(bytes[..count].iter().map(|&b| (b as i8) as f32).collect())
        }
        "short" | "int16" | "signed short" | "int 16" => {
            require_bytes(bytes.len(), count, 2, element_type)?;
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
        "unsigned short" | "uint16" | "ushort" | "unsigned short int" => {
            require_bytes(bytes.len(), count, 2, element_type)?;
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
        "int" | "int32" | "signed int" | "int 32" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
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
        "unsigned int" | "uint32" | "uint" | "unsigned int 32" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
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
        "float" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
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
        "double" => {
            require_bytes(bytes.len(), count, 8, element_type)?;
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
        other => Err(anyhow!("Unsupported NRRD type: '{}'", other)),
    }
}

/// Return an error when `have` bytes are fewer than `count * elem_size`.
fn require_bytes(have: usize, count: usize, elem_size: usize, type_name: &str) -> Result<()> {
    let need = count * elem_size;
    if have < need {
        Err(anyhow!(
            "Insufficient data for type '{}': need {} bytes, got {}",
            type_name,
            need,
            have
        ))
    } else {
        Ok(())
    }
}

/// Parse a whitespace-separated list of exactly `expected` `f64` values.
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

/// Parse a whitespace-separated list of exactly `expected` `usize` values.
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

/// Thin reader struct for NRRD files.
///
/// The backend `B` and device are supplied per-call so a single `NrrdReader`
/// instance can serve multiple backends.
pub struct NrrdReader;

impl NrrdReader {
    /// Read a NRRD file at `path` into an [`Image`] on `device`.
    pub fn read<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> Result<Image<B, 3>> {
        read_nrrd(path, device)
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_nrrd::<TestBackend, _>(&path, &device)?;

        assert_eq!(image.shape(), [nz, ny, nx], "shape must be [nz, ny, nx]");
        Ok(())
    }

    /// Spacing extracted from axis-aligned `space directions` must match the
    /// magnitudes of the direction vectors.
    #[test]
    fn test_spacing_from_space_directions() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("spacing_sd.nrrd");
        let data = vec![0.0f32; 2 * 3 * 4];
        write_inline_nrrd(&path, &data, 4, 3, 2, [0.9, 0.75, 1.5], [5.0, 10.0, 15.0]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_nrrd::<TestBackend, _>(&path, &device)?;

        // Spacing in physical [X, Y, Z] order.
        assert!((image.spacing()[0] - 0.9).abs() < 1e-9, "spacing[0]");
        assert!((image.spacing()[1] - 0.75).abs() < 1e-9, "spacing[1]");
        assert!((image.spacing()[2] - 1.5).abs() < 1e-9, "spacing[2]");

        // Origin in physical [X, Y, Z] order.
        assert!((image.origin()[0] - 5.0).abs() < 1e-9, "origin[0]");
        assert!((image.origin()[1] - 10.0).abs() < 1e-9, "origin[1]");
        assert!((image.origin()[2] - 15.0).abs() < 1e-9, "origin[2]");

        Ok(())
    }

    /// `spacings` field (no `space directions`) must fall back to identity
    /// direction and use the scalar values as spacing.
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_nrrd::<TestBackend, _>(&path, &device)?;

        assert!((image.spacing()[0] - 0.5).abs() < 1e-9, "spacing[0]");
        assert!((image.spacing()[1] - 0.5).abs() < 1e-9, "spacing[1]");
        assert!((image.spacing()[2] - 2.0).abs() < 1e-9, "spacing[2]");

        // Direction must be identity.
        let d = image.direction().0;
        assert!((d[(0, 0)] - 1.0).abs() < 1e-9, "direction[0,0]");
        assert!((d[(1, 1)] - 1.0).abs() < 1e-9, "direction[1,1]");
        assert!((d[(2, 2)] - 1.0).abs() < 1e-9, "direction[2,2]");
        assert!(d[(0, 1)].abs() < 1e-9, "direction[0,1] must be 0");

        Ok(())
    }

    /// Direction matrix columns extracted from non-axis-aligned `space
    /// directions` must match the normalised input vectors.
    ///
    /// Test vector: space directions = (2,0,0) (0,3,0) (0,0,4)
    /// Expected: spacing = [2, 3, 4], direction = identity.
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_nrrd::<TestBackend, _>(&path, &device)?;

        assert!((image.spacing()[0] - 2.0).abs() < 1e-9, "spacing[0] = 2");
        assert!((image.spacing()[1] - 3.0).abs() < 1e-9, "spacing[1] = 3");
        assert!((image.spacing()[2] - 4.0).abs() < 1e-9, "spacing[2] = 4");

        // After normalisation direction must be identity.
        let d = image.direction().0;
        assert!((d[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((d[(1, 1)] - 1.0).abs() < 1e-9);
        assert!((d[(2, 2)] - 1.0).abs() < 1e-9);

        Ok(())
    }

    // ── Round-trip ─────────────────────────────────────────────────────────

    /// Write an Image via `write_nrrd` and read it back; verify shape,
    /// spatial metadata, and every voxel value.
    #[test]
    fn test_round_trip_nrrd() -> Result<()> {
        use crate::format::nrrd::write_nrrd;

        let dir = tempdir()?;
        let path = dir.path().join("round_trip.nrrd");
        let device: <TestBackend as Backend>::Device = Default::default();

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

        // Origin (within f64 string-round-trip tolerance)
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

    // ── Error paths ────────────────────────────────────────────────────────

    /// Reading a non-existent file must return an error (not panic).
    #[test]
    fn test_missing_file_returns_error() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_nrrd::<TestBackend, _>("/nonexistent/path/file.nrrd", &device);
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_nrrd::<TestBackend, _>(&path, &device);
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_nrrd::<TestBackend, _>(&path, &device);
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_nrrd::<TestBackend, _>(&path, &device);
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_nrrd::<TestBackend, _>(&path, &device);
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

        let device: <TestBackend as Backend>::Device = Default::default();
        let image = read_nrrd::<TestBackend, _>(&header_path, &device)?;

        assert_eq!(image.shape(), [nz, ny, nx]);

        // Sum of all voxels = 0+1+…+7 = 28, regardless of permutation.
        let td = image.data().clone().to_data();
        let vals = td.as_slice::<f32>().unwrap();
        let sum: f32 = vals.iter().sum();
        assert!(
            (sum - 28.0).abs() < 1e-5,
            "Voxel sum mismatch: expected 28, got {}",
            sum
        );

        Ok(())
    }
}
