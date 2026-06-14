use crate::spatial::metadata_from_file_transform;
use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_spatial::Point;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Byte order for multi-byte pixel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    MostSignificantByteFirst,
    LeastSignificantByteFirst,
}

/// Read a MetaImage (.mha or .mhd) file into a 3-D `Image`.
///
/// # Axis convention
/// MetaImage stores voxels in X-fastest `[X, Y, Z]` order. The same flat byte
/// sequence is RITK-contiguous when shaped as `[Z, Y, X]`, so the returned
/// tensor shape is `[nz, ny, nx]` without a data permutation.
///
/// # Spatial metadata
/// `origin` remains in physical coordinate order. `ElementSpacing` and
/// `TransformMatrix` are converted from MetaImage `[X,Y,Z]` file axes into
/// RITK `[Z,Y,X]` image-axis metadata.
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

    // 2-D images are promoted to a degenerate `[1, Y, X]` (z = 1) volume: ritk's
    // Image is `Image<B, 3>`, so a 2-D file becomes a single-slice 3-D image.
    if ndims != 2 && ndims != 3 {
        return Err(anyhow!("Expected NDims = 2 or 3, found {}", ndims));
    }

    let dim_sizes = parse_usize_vec(
        headers
            .get("DimSize")
            .ok_or_else(|| anyhow!("Missing 'DimSize' in MetaImage header"))?,
        "DimSize",
        ndims,
    )?;
    let nx = dim_sizes[0];
    let ny = dim_sizes[1];
    let nz = if ndims == 3 { dim_sizes[2] } else { 1 };

    let spacing_raw = parse_f64_vec(
        headers
            .get("ElementSpacing")
            .ok_or_else(|| anyhow!("Missing 'ElementSpacing' in MetaImage header"))?,
        "ElementSpacing",
        ndims,
    )?;
    // Promote spacing with unit z when 2-D.
    let spacing_vals = if ndims == 3 {
        spacing_raw
    } else {
        vec![spacing_raw[0], spacing_raw[1], 1.0]
    };

    let offset_str = headers
        .get("Offset")
        .or_else(|| headers.get("Position"))
        .ok_or_else(|| anyhow!("Missing 'Offset' (or 'Position') in MetaImage header"))?;
    let offset_raw = parse_f64_vec(offset_str, "Offset", ndims)?;
    let offset_vals = if ndims == 3 {
        offset_raw
    } else {
        vec![offset_raw[0], offset_raw[1], 0.0]
    };

    // TransformMatrix is row-major direction cosines (ndims² entries); defaults to
    // identity when absent.  A 2-D `[a b; c d]` matrix promotes to the 3-D
    // `[a b 0; c d 0; 0 0 1]` (identity through-plane z-axis).
    let tm_default = if ndims == 3 { "1 0 0 0 1 0 0 0 1" } else { "1 0 0 1" };
    let tm_src = headers
        .get("TransformMatrix")
        .map(|s| s.as_str())
        .unwrap_or(tm_default);
    let tm_raw = parse_f64_vec(tm_src, "TransformMatrix", ndims * ndims)?;
    let tm_vals = if ndims == 3 {
        tm_raw
    } else {
        vec![
            tm_raw[0], tm_raw[1], 0.0, tm_raw[2], tm_raw[3], 0.0, 0.0, 0.0, 1.0,
        ]
    };

    let element_type = headers
        .get("ElementType")
        .ok_or_else(|| anyhow!("Missing 'ElementType' in MetaImage header"))?
        .clone();

    // BinaryDataByteOrderMSB = True → big-endian; default is little-endian.
    let byte_order = if headers
        .get("BinaryDataByteOrderMSB")
        .map(|s| s.to_uppercase() == "TRUE")
        .unwrap_or(false)
    {
        ByteOrder::MostSignificantByteFirst
    } else {
        ByteOrder::LeastSignificantByteFirst
    };

    // CompressedData = True → the payload is zlib-deflated; default is raw.
    let compressed = headers
        .get("CompressedData")
        .map(|s| s.to_uppercase() == "TRUE")
        .unwrap_or(false);

    let element_data_file = headers
        .get("ElementDataFile")
        .ok_or_else(|| anyhow!("Missing 'ElementDataFile' in MetaImage header"))?
        .clone();

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = nx * ny * nz;

    // Read the payload bytes (still zlib-deflated when `compressed`) from the
    // inline LOCAL stream or the detached raw file, then inflate if needed.
    let payload: Vec<u8> = if element_data_file.to_uppercase() == "LOCAL" {
        // Seek the BufReader to the exact position after the last header line.
        // BufReader<File> implements Seek; this discards any internal buffer
        // and repositions the underlying file descriptor.
        reader
            .seek(SeekFrom::Start(byte_offset))
            .context("Failed to seek to binary data in .mha file")?;
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .context("Failed to read binary voxel data from .mha file")?;
        bytes
    } else {
        // External .raw file: resolve relative to the header file's directory.
        let raw_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&element_data_file);
        std::fs::read(&raw_path)
            .with_context(|| format!("Cannot read raw data file {:?}", raw_path))?
    };

    let raw_bytes: Vec<u8> = if compressed {
        let mut out = Vec::with_capacity(total_voxels * 4);
        flate2::read::ZlibDecoder::new(&payload[..])
            .read_to_end(&mut out)
            .context("Failed to inflate zlib-compressed MetaImage payload")?;
        out
    } else {
        payload
    };

    let f32_data: Vec<f32> =
        decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, byte_order)?;

    if f32_data.len() != total_voxels {
        return Err(anyhow!(
            "Voxel count mismatch: DimSize implies {} voxels but {} were decoded",
            total_voxels,
            f32_data.len()
        ));
    }

    // ── Tensor construction ───────────────────────────────────────────────
    // MetaImage X-fastest flat order equals Burn row-major order for RITK [Z,Y,X].
    let shape_burn = Shape::new([nz, ny, nx]);
    let tensor_data = TensorData::new(f32_data, shape_burn);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    // ── Spatial metadata ──────────────────────────────────────────────────
    let origin = Point::new([offset_vals[0], offset_vals[1], offset_vals[2]]);
    let spatial = metadata_from_file_transform(
        [spacing_vals[0], spacing_vals[1], spacing_vals[2]],
        [
            tm_vals[0], tm_vals[1], tm_vals[2], tm_vals[3], tm_vals[4], tm_vals[5], tm_vals[6],
            tm_vals[7], tm_vals[8],
        ],
    );

    Ok(Image::new(
        tensor,
        origin,
        spatial.spacing,
        spatial.direction,
    ))
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
    byte_order: ByteOrder,
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
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
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
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
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
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
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
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
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
                    if byte_order == ByteOrder::MostSignificantByteFirst {
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
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
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
