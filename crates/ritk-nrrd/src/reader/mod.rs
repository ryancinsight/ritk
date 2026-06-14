mod decode;
use decode::*;

use crate::spatial::{metadata_from_file_space_directions, metadata_from_file_spacings};
use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_spatial::Point;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

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
/// `raw` and `gzip` (`gz`) encodings are supported; any other encoding returns
/// an error with an actionable message.
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

    // 2-D NRRD files are promoted to a degenerate `[1, Y, X]` (z = 1) volume,
    // since ritk's `Image` is 3-D.
    if dimension != 2 && dimension != 3 {
        return Err(anyhow!(
            "Expected dimension = 2 or 3 for a NRRD file, found {}",
            dimension
        ));
    }

    let sizes_str = headers
        .get("sizes")
        .ok_or_else(|| anyhow!("Missing 'sizes' in NRRD header"))?;
    let sizes = parse_usize_vec(sizes_str, "sizes", dimension)?;
    let nx = sizes[0];
    let ny = sizes[1];
    let nz = if dimension == 3 { sizes[2] } else { 1 };

    // ── Encoding ──────────────────────────────────────────────────────────
    let encoding = headers
        .get("encoding")
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "raw".to_string());

    let gzipped = match encoding.as_str() {
        "raw" => false,
        "gzip" | "gz" => true,
        other => {
            return Err(anyhow!(
                "Unsupported NRRD encoding '{}'. Supported: 'raw', 'gzip'.",
                other
            ))
        }
    };

    // ── Endianness ────────────────────────────────────────────────────────
    // True ⟹ big-endian.  Default is little-endian for multi-byte types.
    let byte_order = if headers
        .get("endian")
        .map(|s| s.to_lowercase() == "big")
        .unwrap_or(false)
    {
        ByteOrder::MostSignificantByteFirst
    } else {
        ByteOrder::LeastSignificantByteFirst
    };

    // ── Spacing and direction ─────────────────────────────────────────────
    // 2-D files carry 2-component directions/spacings/origin, promoted with an
    // identity through-plane z-axis (unit z-spacing, zero z-origin).
    let spatial = if let Some(sd_str) = headers.get("space directions") {
        let dirs = if dimension == 3 {
            parse_space_directions(sd_str)?
        } else {
            parse_space_directions_2d(sd_str)?
        };
        metadata_from_file_space_directions(dirs)
    } else if let Some(sp_str) = headers.get("spacings") {
        let sp = parse_f64_vec(sp_str, "spacings", dimension)?;
        let sz = if dimension == 3 { sp[2] } else { 1.0 };
        metadata_from_file_spacings([sp[0], sp[1], sz])
    } else {
        // Neither field present: unit spacing with canonical file-axis order.
        metadata_from_file_spacings([1.0, 1.0, 1.0])
    };

    // ── Origin ────────────────────────────────────────────────────────────
    let origin = if let Some(so_str) = headers.get("space origin") {
        if dimension == 3 {
            parse_nrrd_point(so_str)?
        } else {
            parse_nrrd_point_2d(so_str)?
        }
    } else {
        Point::new([0.0, 0.0, 0.0])
    };

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = nx * ny * nz;
    let data_file_field = headers.get("data file").cloned();

    // Read the payload bytes (still compressed for gzip encoding) from the
    // inline stream or the detached data file, then gunzip if needed.
    let payload: Vec<u8> = match &data_file_field {
        None => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .context("Failed to read inline NRRD binary data")?;
            bytes
        }
        Some(df) if df.to_uppercase() == "INTERNAL" => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .context("Failed to read inline NRRD binary data (INTERNAL)")?;
            bytes
        }
        Some(df) => {
            let raw_path = path.parent().unwrap_or_else(|| Path::new(".")).join(df);
            std::fs::read(&raw_path)
                .with_context(|| format!("Cannot read NRRD data file {:?}", raw_path))?
        }
    };

    let raw_bytes: Vec<u8> = if gzipped {
        let mut out = Vec::with_capacity(payload.len() * 2);
        flate2::read::GzDecoder::new(&payload[..])
            .read_to_end(&mut out)
            .context("Failed to inflate gzip-encoded NRRD payload")?;
        out
    } else {
        payload
    };

    let f32_data: Vec<f32> =
        decode_bytes_to_f32(&raw_bytes, &element_type, total_voxels, byte_order)?;

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
