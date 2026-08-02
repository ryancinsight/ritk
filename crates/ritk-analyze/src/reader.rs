//! Analyze 7.5 reader — parses a 348-byte `.hdr` header and raw `.img` voxel data.
//!
//! # Format Overview
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files sharing the
//! same base name:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw voxel values (little-endian, type given by `datatype` field).
//!
//! A paired NIfTI-1 dataset can use the same extensions, but identifies itself
//! with `ni1\0` at bytes 344–347 and is not an Analyze 7.5 file. This reader
//! rejects that variant explicitly instead of interpreting NIfTI spatial fields
//! as Analyze history fields.
//!
//! # Header Layout (key fields)
//!
//! | Offset | Type  | Field             | Meaning                                  |
//! |--------|-------|-------------------|------------------------------------------|
//! |      0 | i32   | `sizeof_hdr`      | Must equal 348                           |
//! |     40 | i16   | `dim[0]`          | Number of dimensions (typically 4)       |
//! |     42 | i16   | `dim[1]`          | X size (nx)                              |
//! |     44 | i16   | `dim[2]`          | Y size (ny)                              |
//! |     46 | i16   | `dim[3]`          | Z size (nz)                              |
//! |     70 | i16   | `datatype`        | 2=u8, 4=i16, 8=i32, 16=f32, 64=f64      |
//! |     72 | i16   | `bitpix`          | Bits per voxel                           |
//! |     80 | f32   | `pixdim[1]`       | X spacing (mm)                           |
//! |     84 | f32   | `pixdim[2]`       | Y spacing (mm)                           |
//! |     88 | f32   | `pixdim[3]`       | Z spacing (mm)                           |
//! |    108 | f32   | `vox_offset`      | Byte offset to data in `.img` (0 = start)|
//! |    112 | f32   | `funused1`        | Intensity scale factor (0 or 1 = no-op)  |
//! |    253 | i16×5 | `originator`      | Voxel-space origin (x, y, z, 0, 0)       |
//!
//! # Axis Convention
//!
//! Analyze stores voxels with X varying fastest (column-major XYZ).
//! RITK stores tensors with shape `[nz, ny, nx]` (Z-major ZYX).
//! Because both produce the same flat byte sequence for identical (nx, ny, nz),
//! no in-memory permutation is required.
//!
//! # Spatial Metadata
//!
//! The file stores spacing in file-axis order `pixdim[1..3] = [sx, sy, sz]`.
//! RITK's core `Spacing` is per tensor axis `[z, y, x]` (matching the `[nz, ny,
//! nx]` tensor shape), so the file components are reversed to `[sz, sy, sx]` on
//! read — the same column reorder the MetaImage/NRRD readers apply. The core
//! `origin` is a world-space point `[x, y, z]` and is **not** reversed.
//!
//! The physical origin is reconstructed from `originator` voxel coordinates:
//!
//! ```text
//!   origin_x = originator[0] × sx
//!   origin_y = originator[1] × sy
//!   origin_z = originator[2] × sz
//! ```
//!
//! Note: the `originator` field is unreliable across writers (Analyze 7.5 is a
//! deprecated format; SimpleITK does not round-trip a physical origin through
//! it), so origin parity with foreign Analyze files is not guaranteed.

use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_spatial::{Direction, Point, Spacing};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::codec::{read_le, HDR_SIZE};
pub use crate::codec::{DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR};

const DECODE_CHUNK_BYTES: usize = 8 * 1024;

trait AnalyzeVoxel: Sized {
    const WIDTH: usize;

    fn decode(bytes: &[u8]) -> f32;
}

impl AnalyzeVoxel for u8 {
    const WIDTH: usize = 1;

    fn decode(bytes: &[u8]) -> f32 {
        f32::from(bytes[0])
    }
}

impl AnalyzeVoxel for i16 {
    const WIDTH: usize = 2;

    fn decode(bytes: &[u8]) -> f32 {
        f32::from(i16::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: i16 Analyze chunks contain two bytes"),
        ))
    }
}

impl AnalyzeVoxel for i32 {
    const WIDTH: usize = 4;

    fn decode(bytes: &[u8]) -> f32 {
        i32::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: i32 Analyze chunks contain four bytes"),
        ) as f32
    }
}

impl AnalyzeVoxel for f32 {
    const WIDTH: usize = 4;

    fn decode(bytes: &[u8]) -> f32 {
        Self::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: f32 Analyze chunks contain four bytes"),
        )
    }
}

impl AnalyzeVoxel for f64 {
    const WIDTH: usize = 8;

    fn decode(bytes: &[u8]) -> f32 {
        Self::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: f64 Analyze chunks contain eight bytes"),
        ) as f32
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a 3-D image from an Analyze 7.5 `.hdr` / `.img` file pair.
///
/// `path` may point to either the `.hdr` or the `.img` file.  The sibling file
/// is located automatically by replacing the extension.
///
/// # Supported datatypes
/// `DT_UNSIGNED_CHAR` (2), `DT_SIGNED_SHORT` (4), `DT_SIGNED_INT` (8),
/// `DT_FLOAT` (16), `DT_DOUBLE` (64).  All are converted to `f32` in the
/// returned native image buffer.
///
/// # Errors
/// Returns an error when:
/// - Either file cannot be opened or read.
/// - The header is not a little-endian, 348-byte Analyze header, including
///   paired NIfTI data using the same extensions.
/// - The header does not describe exactly one 3-D volume with positive,
///   non-overflowing dimensions.
/// - `datatype` is not supported or `bitpix` does not match it.
/// - Spacing, scale, or offset metadata is non-finite, or the offset is not a
///   supported whole-byte position.
/// - The `.img` file length differs from the exact declared payload size.
/// - Output allocation, seeking, decoding, or image construction fails.
pub fn read_analyze<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<ritk_image::Image<f32, B, 3>> {
    let DecodedAnalyze {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_analyze(path)?;

    ritk_image::Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Substrate-agnostic decode of an Analyze `.hdr`/`.img` pair into flat
/// `[Z, Y, X]` voxels plus spatial metadata for the public reader.
struct DecodedAnalyze {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

fn decode_analyze<P: AsRef<Path>>(path: P) -> Result<DecodedAnalyze> {
    let path = path.as_ref();

    // Derive sibling paths regardless of which file the caller passed.
    let hdr_path = path.with_extension("hdr");
    let img_path = path.with_extension("img");

    // ── Read and validate the 348-byte header ─────────────────────────────────
    let mut hdr_file = File::open(&hdr_path).context("Cannot open Analyze header")?;
    let header_len = hdr_file
        .metadata()
        .context("Cannot inspect Analyze header")?
        .len();
    if header_len < HDR_SIZE as u64 {
        return Err(anyhow!(
            "Invalid Analyze header length: expected {HDR_SIZE} bytes, found {header_len}"
        ));
    }
    let mut hdr = [0u8; HDR_SIZE];
    hdr_file
        .read_exact(&mut hdr)
        .with_context(|| "Cannot read 348-byte header".to_string())?;
    if hdr[344..348] == *b"ni1\0" {
        return Err(anyhow!(
            "Unsupported paired NIfTI-1 header (ni1 magic); use the NIfTI reader with a single-file .nii dataset"
        ));
    }
    if header_len != HDR_SIZE as u64 {
        return Err(anyhow!(
            "Invalid Analyze header length: expected {HDR_SIZE} bytes, found {header_len}"
        ));
    }

    // sizeof_hdr must be exactly 348. Identify the unsupported byte order so a
    // big-endian file is not reported as arbitrary header corruption.
    let sizeof_hdr = read_le::<i32>(&hdr, 0);
    if sizeof_hdr != HDR_SIZE as i32 {
        if i32::from_be_bytes(
            hdr[0..4]
                .try_into()
                .expect("invariant: four-byte header field"),
        ) == HDR_SIZE as i32
        {
            return Err(anyhow!(
                "Unsupported big-endian Analyze file; RITK currently accepts little-endian Analyze 7.5 only"
            ));
        }
        return Err(anyhow!(
            "Invalid Analyze file: sizeof_hdr={} (expected 348)",
            sizeof_hdr
        ));
    }

    // ── Parse image dimensions ────────────────────────────────────────────────
    let dimension_count = read_le::<i16>(&hdr, 40);
    if !(3..=4).contains(&dimension_count) {
        return Err(anyhow!(
            "Unsupported Analyze dimension count {dimension_count}; the RITK reader accepts one 3-D volume"
        ));
    }
    let nx = positive_dimension(read_le::<i16>(&hdr, 42), "nx")?;
    let ny = positive_dimension(read_le::<i16>(&hdr, 44), "ny")?;
    let nz = positive_dimension(read_le::<i16>(&hdr, 46), "nz")?;
    if dimension_count == 4 {
        let volume_count = read_le::<i16>(&hdr, 48);
        if volume_count != 1 {
            return Err(anyhow!(
                "Unsupported Analyze volume count {volume_count}; the RITK reader accepts exactly one 3-D volume"
            ));
        }
    }

    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .context("Analyze voxel count overflows usize")?;

    // ── Parse voxel type ──────────────────────────────────────────────────────
    let datatype = read_le::<i16>(&hdr, 70);
    let bytes_per_voxel = datatype_width(datatype)?;
    let bitpix = read_le::<i16>(&hdr, 72);
    let expected_bitpix = i16::try_from(bytes_per_voxel * 8)
        .expect("invariant: supported Analyze voxel widths fit in i16 bits");
    if bitpix != expected_bitpix {
        return Err(anyhow!(
            "Analyze bitpix {bitpix} does not match datatype {datatype}; expected {expected_bitpix}"
        ));
    }

    // ── Parse physical spacing (pixdim[1..3]) ─────────────────────────────────
    let sx_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 80), "pixdim[1]")?);
    let sy_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 84), "pixdim[2]")?);
    let sz_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 88), "pixdim[3]")?);
    // Fall back to unit spacing when stored value is zero or negative.
    let sx = if sx_raw > 0.0 { sx_raw } else { 1.0 };
    let sy = if sy_raw > 0.0 { sy_raw } else { 1.0 };
    let sz = if sz_raw > 0.0 { sz_raw } else { 1.0 };

    // ── Parse scale factor (funused1 at offset 112) ───────────────────────────
    let scale_raw = finite_header_value(read_le::<f32>(&hdr, 112), "funused1 scale")?;
    let scale = if scale_raw == 0.0 { 1.0_f32 } else { scale_raw };

    // ── Parse vox_offset (offset 108) ────────────────────────────────────────
    let vox_offset_raw = f64::from(finite_header_value(
        read_le::<f32>(&hdr, 108),
        "vox_offset",
    )?);
    if vox_offset_raw < 0.0 || vox_offset_raw.fract() != 0.0 || vox_offset_raw > u64::MAX as f64 {
        return Err(anyhow!(
            "Unsupported Analyze vox_offset {vox_offset_raw}; expected a non-negative whole-byte offset"
        ));
    }
    let vox_offset = vox_offset_raw as u64;

    // ── Parse origin from originator[10] (5 × i16 at offset 253) ─────────────
    let ox_vox = read_le::<i16>(&hdr, 253) as f64;
    let oy_vox = read_le::<i16>(&hdr, 255) as f64;
    let oz_vox = read_le::<i16>(&hdr, 257) as f64;
    let ox = ox_vox * sx;
    let oy = oy_vox * sy;
    let oz = oz_vox * sz;

    // ── Validate and stream .img data ────────────────────────────────────────
    let expected_bytes = voxel_count
        .checked_mul(bytes_per_voxel)
        .context("Analyze payload byte count overflows usize")?;
    let expected_bytes_u64 =
        u64::try_from(expected_bytes).context("Analyze payload byte count exceeds u64")?;
    let expected_file_len = vox_offset
        .checked_add(expected_bytes_u64)
        .context("Analyze payload end offset overflows u64")?;
    let mut img_file = File::open(&img_path).context("Cannot open Analyze data file")?;
    let actual_file_len = img_file
        .metadata()
        .context("Cannot inspect Analyze data file")?
        .len();
    if actual_file_len != expected_file_len {
        return Err(anyhow!(
            "Analyze .img length mismatch: expected {expected_file_len} bytes ({vox_offset} offset + {expected_bytes} payload), found {actual_file_len}"
        ));
    }
    img_file
        .seek(SeekFrom::Start(vox_offset))
        .context("Cannot seek to Analyze voxel payload")?;

    let vals = match datatype {
        DT_UNSIGNED_CHAR => decode_payload::<u8>(&mut img_file, voxel_count, scale),
        DT_SIGNED_SHORT => decode_payload::<i16>(&mut img_file, voxel_count, scale),
        DT_SIGNED_INT => decode_payload::<i32>(&mut img_file, voxel_count, scale),
        DT_FLOAT => decode_payload::<f32>(&mut img_file, voxel_count, scale),
        DT_DOUBLE => decode_payload::<f64>(&mut img_file, voxel_count, scale),
        _ => unreachable!("invariant: datatype_width accepted this datatype"),
    }?;

    tracing::debug!(nx, ny, nz, datatype, "decode_analyze: complete");

    // Spacing reverses file `[sx, sy, sz]` into core tensor-axis order
    // `[sz, sy, sx]`; origin stays a world-space `[x, y, z]` point.
    Ok(DecodedAnalyze {
        data: vals,
        dims: [nz, ny, nx],
        origin: Point::new([ox, oy, oz]),
        spacing: Spacing::new([sz, sy, sx]),
        direction: Direction::identity(),
    })
}

fn positive_dimension(raw: i16, name: &str) -> Result<usize> {
    usize::try_from(raw)
        .map_err(|_| anyhow!("Invalid Analyze dimension {name}={raw}; expected a positive value"))
        .and_then(|value| {
            if value == 0 {
                Err(anyhow!(
                    "Invalid Analyze dimension {name}=0; expected a positive value"
                ))
            } else {
                Ok(value)
            }
        })
}

fn finite_header_value(raw: f32, field: &str) -> Result<f32> {
    if raw.is_finite() {
        Ok(raw)
    } else {
        Err(anyhow!(
            "Invalid Analyze {field}: expected a finite value, found {raw}"
        ))
    }
}

fn datatype_width(datatype: i16) -> Result<usize> {
    match datatype {
        DT_UNSIGNED_CHAR => Ok(1),
        DT_SIGNED_SHORT => Ok(2),
        DT_SIGNED_INT | DT_FLOAT => Ok(4),
        DT_DOUBLE => Ok(8),
        other => Err(anyhow!(
            "Unsupported Analyze datatype {other}. Supported codes: 2 (u8), 4 (i16), 8 (i32), 16 (f32), 64 (f64)."
        )),
    }
}

fn decode_payload<T: AnalyzeVoxel>(
    reader: &mut File,
    voxel_count: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let voxels_per_chunk = DECODE_CHUNK_BYTES / T::WIDTH;
    debug_assert!(voxels_per_chunk > 0);
    let mut values = Vec::new();
    values
        .try_reserve_exact(voxel_count)
        .context("Cannot allocate Analyze output volume")?;
    let mut bytes = [0u8; DECODE_CHUNK_BYTES];
    let mut remaining = voxel_count;

    while remaining > 0 {
        let chunk_voxels = remaining.min(voxels_per_chunk);
        let chunk_bytes = chunk_voxels
            .checked_mul(T::WIDTH)
            .expect("invariant: decode chunk byte count fits its fixed buffer");
        let input = &mut bytes[..chunk_bytes];
        reader
            .read_exact(input)
            .context("Cannot read validated Analyze voxel payload")?;
        values.extend(
            input
                .chunks_exact(T::WIDTH)
                .map(|voxel| T::decode(voxel) * scale),
        );
        remaining -= chunk_voxels;
    }

    Ok(values)
}

// ── Reader wrapper type ───────────────────────────────────────────────────────

/// Read-side wrapper type implementing the `ImageReader` domain trait.
pub struct AnalyzeReader<B: ComputeBackend> {
    pub(crate) backend: B,
}

impl<B: ComputeBackend> AnalyzeReader<B> {
    /// Construct a reader bound to `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read an Analyze image through the bound backend.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<ritk_image::Image<f32, B, 3>> {
        read_analyze(path, &self.backend)
    }
}
