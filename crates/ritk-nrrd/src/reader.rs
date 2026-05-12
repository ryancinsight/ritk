use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::Point;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::spatial::{metadata_from_file_space_directions, metadata_from_file_spacings};

/// Read a NRRD (Nearly Raw Raster Data) file into a 3-D `Image`.
///
/// # Axis convention
/// NRRD files produced by ITK-compatible tools store voxels in `[X, Y, Z]`
/// order with X as the fastest-varying raw axis. That flat raw order is the
/// same byte sequence as a RITK tensor shaped `[Z, Y, X]`, so the returned
/// tensor is constructed directly with shape `[nz, ny, nx]`.
///
/// # Spatial metadata
/// Direction and spacing are derived from `space directions` when that field
/// is present. NRRD file-axis vectors `[x,y,z]` are reordered into RITK
/// metadata columns `[depth,row,col] = [z,y,x]`. If only `spacings` is present,
/// the scalar spacings follow the same axis reorder with axis-aligned
/// directions.
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
    let spatial = if let Some(sd_str) = headers.get("space directions") {
        metadata_from_file_space_directions(parse_space_directions(sd_str)?)
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp_vals = parse_f64_vec(sp_str, "spacings", 3)?;
        metadata_from_file_spacings([sp_vals[0], sp_vals[1], sp_vals[2]])
    } else {
        // Neither field present: unit spacing with canonical file-axis order.
        metadata_from_file_spacings([1.0, 1.0, 1.0])
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
    // NRRD raw order is X-fastest. RITK [Z,Y,X] row-major tensors are also
    // X-fastest in flat memory, so the decoded payload can be shaped directly.
    let shape_burn = Shape::new([nz, ny, nx]);
    let tensor_data = TensorData::new(f32_data, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    Ok(Image::new(
        tensor,
        origin,
        spatial.spacing,
        spatial.direction,
    ))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Parse a `space directions` field into three NRRD file-axis vectors.
///
/// Each direction vector `v_i` encodes the physical displacement per voxel
/// step along image axis `i`:
/// ```text
/// v_i = Direction[:, i] * spacing[i]
/// spacing[i] = |v_i|
/// Direction[:, i] = v_i / |v_i|
/// ```
fn parse_space_directions(s: &str) -> Result<[[f64; 3]; 3]> {
    let vecs = parse_parenthesized_vectors(s)?;
    if vecs.len() != 3 {
        return Err(anyhow!(
            "'space directions' must contain 3 vectors, found {}",
            vecs.len()
        ));
    }

    Ok([vecs[0], vecs[1], vecs[2]])
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
