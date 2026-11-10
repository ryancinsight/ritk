use crate::error::MetaImageError;
use crate::spatial::metadata_from_file_transform;
use coeus_core::ComputeBackend;
use ritk_codecs::{decode_bytes_to_f32, parse_f64_vec, parse_usize_vec, ByteOrder};
use ritk_image::Image;
use ritk_spatial::Point;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

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
///
/// # Errors
/// Each [`MetaImageError`] variant names the malformed header field or failed
/// payload operation.
pub fn read_metaimage<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>, MetaImageError> {
    let DecodedMetaImage {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_metaimage(path)?;
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
        .map_err(|source| MetaImageError::Image { source })
}

/// Backend-agnostic decoded MetaImage volume: voxels in `[nz, ny, nx]` order plus
/// the derived physical metadata. Shared by the Coeus and Coeus reader paths.
struct DecodedMetaImage {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

fn decode_metaimage<P: AsRef<Path>>(path: P) -> Result<DecodedMetaImage, MetaImageError> {
    let path = path.as_ref();

    let file = std::fs::File::open(path).map_err(|source| MetaImageError::Open {
        path: path.to_path_buf(),
        source,
    })?;
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
            .map_err(|source| MetaImageError::HeaderRead { source })?;
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
        return Err(MetaImageError::MissingElementDataFile {
            path: path.to_path_buf(),
        });
    }

    // ── Required fields ───────────────────────────────────────────────────
    let ndims: usize = require_field(&headers, "NDims")?
        .parse()
        .map_err(|source| MetaImageError::InvalidNDims { source })?;

    // 2-D images are promoted to a degenerate `[1, Y, X]` (z = 1) volume: ritk's
    // Image is `Image<f32, B, 3>`, so a 2-D file becomes a single-slice 3-D image.
    if ndims != 2 && ndims != 3 {
        return Err(MetaImageError::UnsupportedNDims { found: ndims });
    }

    let dim_sizes = parse_field(parse_usize_vec, &headers, "DimSize", ndims)?;
    let nx = dim_sizes[0];
    let ny = dim_sizes[1];
    let nz = if ndims == 3 { dim_sizes[2] } else { 1 };

    let spacing_raw = parse_field(parse_f64_vec, &headers, "ElementSpacing", ndims)?;
    // Promote spacing with unit z when 2-D.
    let spacing_vals = if ndims == 3 {
        spacing_raw
    } else {
        vec![spacing_raw[0], spacing_raw[1], 1.0]
    };

    let offset_str = headers
        .get("Offset")
        .or_else(|| headers.get("Position"))
        .ok_or(MetaImageError::MissingField { field: "Offset" })?;
    let offset_raw = parse_f64_vec(offset_str, "Offset", ndims).map_err(|source| {
        MetaImageError::InvalidField {
            field: "Offset",
            source,
        }
    })?;
    let offset_vals = if ndims == 3 {
        offset_raw
    } else {
        vec![offset_raw[0], offset_raw[1], 0.0]
    };

    // TransformMatrix is row-major direction cosines (ndims² entries); defaults to
    // identity when absent.  A 2-D `[a b; c d]` matrix promotes to the 3-D
    // `[a b 0; c d 0; 0 0 1]` (identity through-plane z-axis).
    let tm_default = if ndims == 3 {
        "1 0 0 0 1 0 0 0 1"
    } else {
        "1 0 0 1"
    };
    let tm_src = headers
        .get("TransformMatrix")
        .map(|s| s.as_str())
        .unwrap_or(tm_default);
    let tm_raw = parse_f64_vec(tm_src, "TransformMatrix", ndims * ndims).map_err(|source| {
        MetaImageError::InvalidField {
            field: "TransformMatrix",
            source,
        }
    })?;
    let tm_vals = if ndims == 3 {
        tm_raw
    } else {
        vec![
            tm_raw[0], tm_raw[1], 0.0, tm_raw[2], tm_raw[3], 0.0, 0.0, 0.0, 1.0,
        ]
    };

    let element_type = require_field(&headers, "ElementType")?.clone();
    let (elem_size, signed, is_float) = element_type_spec(&element_type)?;

    // BinaryDataByteOrderMSB = True → big-endian; default is little-endian.
    let byte_order = ByteOrder::from_metaimage_msb(
        headers
            .get("BinaryDataByteOrderMSB")
            .map(|s| s.as_str())
            .unwrap_or("FALSE"),
    );

    // CompressedData = True → the payload is zlib-deflated; default is raw.
    let compressed = headers
        .get("CompressedData")
        .map(|s| s.to_uppercase() == "TRUE")
        .unwrap_or(false);

    let element_data_file = require_field(&headers, "ElementDataFile")?.clone();

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = checked_voxel_count(nx, ny, nz)?;
    let expected_payload_bytes =
        total_voxels
            .checked_mul(elem_size)
            .ok_or(MetaImageError::ByteCountOverflow {
                voxels: total_voxels,
                elem_size,
            })?;

    // Read the payload bytes (still zlib-deflated when `compressed`) from the
    // inline LOCAL stream or the detached raw file, then inflate if needed.
    let payload: Vec<u8> = if element_data_file.to_uppercase() == "LOCAL" {
        // Seek the BufReader to the exact position after the last header line.
        // BufReader<File> implements Seek; this discards any internal buffer
        // and repositions the underlying file descriptor.
        let payload_error = |source| MetaImageError::PayloadRead {
            path: path.to_path_buf(),
            source,
        };
        reader
            .seek(SeekFrom::Start(byte_offset))
            .map_err(payload_error)?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(payload_error)?;
        bytes
    } else {
        // External .raw file: resolve relative to the header file's directory.
        let raw_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&element_data_file);
        let payload_error = |source| MetaImageError::PayloadRead {
            path: raw_path.clone(),
            source,
        };
        std::fs::read(&raw_path).map_err(payload_error)?
    };

    let raw_bytes: Vec<u8> = if compressed {
        // `read_to_end` grows the buffer to the true inflated size; `take`
        // stops it one byte past the header-declared size, which the length
        // check below reports as a mismatch.
        let output_limit = u64::try_from(expected_payload_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(&payload[..])
            .take(output_limit)
            .read_to_end(&mut out)
            .map_err(|source| MetaImageError::Inflate { source })?;
        out
    } else {
        payload
    };
    if raw_bytes.len() != expected_payload_bytes {
        return Err(MetaImageError::PayloadLengthMismatch {
            expected: expected_payload_bytes,
            actual: raw_bytes.len(),
            voxels: total_voxels,
            elem_size,
            element_type,
        });
    }

    let f32_data: Vec<f32> = decode_bytes_to_f32(
        &raw_bytes,
        elem_size,
        signed,
        is_float,
        byte_order,
        total_voxels,
        &element_type,
    )
    .map_err(|source| MetaImageError::Decode { source })?;

    // ── Spatial metadata ──────────────────────────────────────────────────
    // MetaImage X-fastest flat order equals row-major order for RITK [Z,Y,X].
    let origin = Point::new([offset_vals[0], offset_vals[1], offset_vals[2]]);
    let spatial = metadata_from_file_transform(
        [spacing_vals[0], spacing_vals[1], spacing_vals[2]],
        [
            tm_vals[0], tm_vals[1], tm_vals[2], tm_vals[3], tm_vals[4], tm_vals[5], tm_vals[6],
            tm_vals[7], tm_vals[8],
        ],
    );

    Ok(DecodedMetaImage {
        data: f32_data,
        dims: [nz, ny, nx],
        origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn checked_voxel_count(nx: usize, ny: usize, nz: usize) -> Result<usize, MetaImageError> {
    nx.checked_mul(ny)
        .and_then(|xy| xy.checked_mul(nz))
        .ok_or(MetaImageError::VoxelCountOverflow { nx, ny, nz })
}

fn require_field<'h>(
    headers: &'h HashMap<String, String>,
    field: &'static str,
) -> Result<&'h String, MetaImageError> {
    headers
        .get(field)
        .ok_or(MetaImageError::MissingField { field })
}

fn parse_field<T>(
    parse: fn(&str, &str, usize) -> anyhow::Result<Vec<T>>,
    headers: &HashMap<String, String>,
    field: &'static str,
    expected: usize,
) -> Result<Vec<T>, MetaImageError> {
    parse(require_field(headers, field)?, field, expected)
        .map_err(|source| MetaImageError::InvalidField { field, source })
}

/// Translate a MetaImage `ElementType` name into the shared byte-decoder spec.
fn element_type_spec(element_type: &str) -> Result<(usize, bool, bool), MetaImageError> {
    let spec = match element_type {
        "MET_UCHAR" => (1_usize, false, false),
        "MET_SHORT" => (2, true, false),
        "MET_USHORT" => (2, false, false),
        "MET_INT" => (4, true, false),
        "MET_UINT" => (4, false, false),
        "MET_FLOAT" => (4, false, true),
        "MET_DOUBLE" => (8, false, true),
        other => {
            return Err(MetaImageError::UnsupportedElementType {
                element_type: other.to_owned(),
            })
        }
    };
    Ok(spec)
}

// ── Public reader struct ──────────────────────────────────────────────────────

/// Thin reader struct for MetaImage files.
///
/// The backend `B` and device are supplied per-call so a single
/// `MetaImageReader` instance can serve multiple backends.
pub struct MetaImageReader;

impl MetaImageReader {
    /// Read a MetaImage file at `path` into an [`Image`] on `device`.
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        &self,
        path: P,
        backend: &B,
    ) -> Result<Image<f32, B, 3>, MetaImageError> {
        read_metaimage(path, backend)
    }
}
