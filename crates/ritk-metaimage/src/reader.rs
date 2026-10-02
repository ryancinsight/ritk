use crate::spatial::metadata_from_file_transform;
use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use ritk_codecs::sample::{SampleBuffer, SampleType};
use ritk_codecs::{parse_f64_vec, parse_usize_vec};
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
/// `MET_FLOAT`, `MET_DOUBLE`. This convenience API returns `f32`; its
/// explicit sample conversion emits a warning when values change.
///
/// # File formats
/// * `.mha` — single file; header followed immediately by binary data
///   (`ElementDataFile = LOCAL`).
/// * `.mhd` / `.raw` — ASCII header references a separate raw file.
pub fn read_metaimage<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let DecodedMetaImage {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_metaimage(path)?;
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
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

fn decode_metaimage<P: AsRef<Path>>(path: P) -> Result<DecodedMetaImage> {
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
    // Image is `Image<f32, B, 3>`, so a 2-D file becomes a single-slice 3-D image.
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
    let tm_default = if ndims == 3 {
        "1 0 0 0 1 0 0 0 1"
    } else {
        "1 0 0 1"
    };
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
    let sample_type = element_sample_type(&element_type)?;
    let elem_size = sample_type.byte_width();

    // BinaryDataByteOrderMSB = True → big-endian; default is little-endian.
    let byte_order = parse_byte_order_msb(
        headers
            .get("BinaryDataByteOrderMSB")
            .map(|s| s.as_str())
            .unwrap_or("FALSE"),
    )?;

    // CompressedData = True → the payload is zlib-deflated; default is raw.
    let compressed = headers
        .get("CompressedData")
        .map(|value| parse_metaimage_bool(value, "CompressedData"))
        .transpose()?
        .unwrap_or(false);

    let element_data_file = headers
        .get("ElementDataFile")
        .ok_or_else(|| anyhow!("Missing 'ElementDataFile' in MetaImage header"))?
        .clone();

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = checked_voxel_count(nx, ny, nz)?;
    let expected_payload_bytes = total_voxels.checked_mul(elem_size).ok_or_else(|| {
        anyhow!(
            "MetaImage byte count overflow: {} voxels × {} bytes",
            total_voxels,
            elem_size
        )
    })?;

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
        // Cap the decompression capacity hint: `expected_payload_bytes` derives
        // from the header DimSize and may be hostile. `read_to_end` still grows
        // the buffer to the true inflated size; the cap only bounds the
        // speculative reservation against an out-of-memory abort.
        let output_limit = u64::try_from(expected_payload_bytes)
            .context("MetaImage payload length exceeds u64")?
            .checked_add(1)
            .ok_or_else(|| anyhow!("MetaImage payload read limit overflows u64"))?;
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(&payload[..])
            .take(output_limit)
            .read_to_end(&mut out)
            .context("Failed to inflate zlib-compressed MetaImage payload")?;
        out
    } else {
        payload
    };
    if raw_bytes.len() != expected_payload_bytes {
        return Err(anyhow!(
            "MetaImage payload length mismatch: expected {} bytes from DimSize ({} voxels × {} bytes for {}), got {} bytes",
            expected_payload_bytes,
            total_voxels,
            elem_size,
            element_type,
            raw_bytes.len()
        ));
    }

    let converted =
        SampleBuffer::decode(&raw_bytes, sample_type, byte_order)?.convert_lossy::<f32>()?;
    let report = converted.report();
    if report.changed_samples > 0 {
        tracing::warn!(
            source_type = %report.source_type,
            target_type = %report.target_type,
            changed_samples = report.changed_samples,
            "MetaImage sample conversion changed stored representations"
        );
    }
    let f32_data = converted.into_parts().0;

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

fn checked_voxel_count(nx: usize, ny: usize, nz: usize) -> Result<usize> {
    nx.checked_mul(ny)
        .and_then(|xy| xy.checked_mul(nz))
        .ok_or_else(|| {
            anyhow!(
                "MetaImage voxel count overflow: DimSize = {} {} {}",
                nx,
                ny,
                nz
            )
        })
}

/// The stored sample type a MetaImage `ElementType` names.
fn element_sample_type(element_type: &str) -> Result<SampleType> {
    let sample_type = match element_type {
        "MET_UCHAR" => SampleType::U8,
        "MET_SHORT" => SampleType::I16,
        "MET_USHORT" => SampleType::U16,
        "MET_INT" => SampleType::I32,
        "MET_UINT" => SampleType::U32,
        "MET_FLOAT" => SampleType::F32,
        "MET_DOUBLE" => SampleType::F64,
        other => return Err(anyhow!("Unsupported MetaImage ElementType: '{}'", other)),
    };
    Ok(sample_type)
}

/// The payload byte order a `BinaryDataByteOrderMSB` value names.
///
/// # Errors
///
/// Returns an error unless `value` is `True` or `False`, case-insensitively.
pub(crate) fn parse_byte_order_msb(value: &str) -> Result<ByteOrder> {
    if parse_metaimage_bool(value, "BinaryDataByteOrderMSB")? {
        Ok(ByteOrder::BigEndian)
    } else {
        Ok(ByteOrder::LittleEndian)
    }
}

pub(crate) fn parse_metaimage_bool(value: &str, field: &str) -> Result<bool> {
    if value.trim().eq_ignore_ascii_case("TRUE") {
        Ok(true)
    } else if value.trim().eq_ignore_ascii_case("FALSE") {
        Ok(false)
    } else {
        Err(anyhow!(
            "Invalid MetaImage {field} value '{value}'; expected 'True' or 'False'"
        ))
    }
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
    ) -> Result<Image<f32, B, 3>> {
        read_metaimage(path, backend)
    }
}
