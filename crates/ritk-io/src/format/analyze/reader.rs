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
//! Spacing components are read directly from `pixdim[1..3]`.
//! The physical origin is reconstructed from `originator` voxel coordinates:
//!
//! ```text
//!   origin_x = originator[0] × sx
//!   origin_y = originator[1] × sy
//!   origin_z = originator[2] × sz
//! ```

use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::io::Read;
use std::marker::PhantomData;
use std::path::Path;

// ── Datatype constants ────────────────────────────────────────────────────────

const DT_UNSIGNED_CHAR: i16 = 2;
const DT_SIGNED_SHORT: i16 = 4;
const DT_SIGNED_INT: i16 = 8;
const DT_FLOAT: i16 = 16;
const DT_DOUBLE: i16 = 64;

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a 3-D image from an Analyze 7.5 `.hdr` / `.img` file pair.
///
/// `path` may point to either the `.hdr` or the `.img` file.  The sibling file
/// is located automatically by replacing the extension.
///
/// # Supported datatypes
/// `DT_UNSIGNED_CHAR` (2), `DT_SIGNED_SHORT` (4), `DT_SIGNED_INT` (8),
/// `DT_FLOAT` (16), `DT_DOUBLE` (64).  All are converted to `f32` in the
/// returned tensor.
///
/// # Errors
/// Returns an error when:
/// - Either file cannot be opened or read.
/// - `sizeof_hdr` is not 348 (invalid Analyze file).
/// - Any image dimension is zero.
/// - The `.img` file is smaller than the declared data size.
/// - `datatype` is not one of the five supported codes.
pub fn read_analyze<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();

    // Derive sibling paths regardless of which file the caller passed.
    let hdr_path = path.with_extension("hdr");
    let img_path = path.with_extension("img");

    // ── Read and validate the 348-byte header ─────────────────────────────────
    let mut hdr_file = std::fs::File::open(&hdr_path).context("Cannot open Analyze header")?;
    let mut hdr = [0u8; 348];
    hdr_file
        .read_exact(&mut hdr)
        .with_context(|| "Cannot read 348-byte header".to_string())?;

    // sizeof_hdr must be exactly 348.
    let sizeof_hdr = read_i32(&hdr, 0);
    if sizeof_hdr != 348 {
        return Err(anyhow!(
            "Invalid Analyze file: sizeof_hdr={} (expected 348)",
            sizeof_hdr
        ));
    }

    // ── Parse image dimensions ────────────────────────────────────────────────
    let nx = read_i16(&hdr, 42) as usize;
    let ny = read_i16(&hdr, 44) as usize;
    let nz = read_i16(&hdr, 46) as usize;

    if nx == 0 || ny == 0 || nz == 0 {
        return Err(anyhow!(
            "Invalid Analyze dimensions: nx={} ny={} nz={}",
            nx,
            ny,
            nz
        ));
    }

    // ── Parse voxel type ──────────────────────────────────────────────────────
    let datatype = read_i16(&hdr, 70);

    // ── Parse physical spacing (pixdim[1..3]) ─────────────────────────────────
    let sx_raw = read_f32(&hdr, 80) as f64;
    let sy_raw = read_f32(&hdr, 84) as f64;
    let sz_raw = read_f32(&hdr, 88) as f64;
    // Fall back to unit spacing when stored value is zero or negative.
    let sx = if sx_raw > 0.0 { sx_raw } else { 1.0 };
    let sy = if sy_raw > 0.0 { sy_raw } else { 1.0 };
    let sz = if sz_raw > 0.0 { sz_raw } else { 1.0 };

    // ── Parse scale factor (funused1 at offset 112) ───────────────────────────
    let scale_raw = read_f32(&hdr, 112);
    let scale = if scale_raw == 0.0 { 1.0_f32 } else { scale_raw };

    // ── Parse vox_offset (offset 108) ────────────────────────────────────────
    let vox_offset = {
        let v = read_f32(&hdr, 108) as u64;
        v
    };

    // ── Parse origin from originator[10] (5 × i16 at offset 253) ─────────────
    let ox_vox = read_i16(&hdr, 253) as f64;
    let oy_vox = read_i16(&hdr, 255) as f64;
    let oz_vox = read_i16(&hdr, 257) as f64;
    let ox = ox_vox * sx;
    let oy = oy_vox * sy;
    let oz = oz_vox * sz;

    // ── Read raw voxel data from .img ─────────────────────────────────────────
    let img_bytes = std::fs::read(&img_path).context("Cannot read Analyze data file")?;

    // Skip past vox_offset bytes if non-zero (uncommon for standard files).
    let data_start = vox_offset as usize;
    if data_start > img_bytes.len() {
        return Err(anyhow!(
            "Analyze vox_offset ({}) exceeds .img file size ({})",
            data_start,
            img_bytes.len()
        ));
    }
    let raw = &img_bytes[data_start..];

    let n = nx * ny * nz;

    // ── Convert to Vec<f32> ───────────────────────────────────────────────────
    let vals: Vec<f32> = match datatype {
        DT_UNSIGNED_CHAR => {
            if raw.len() < n {
                return Err(anyhow!(
                    "Analyze .img too small for u8 data: need {} bytes, have {}",
                    n,
                    raw.len()
                ));
            }
            raw[..n].iter().map(|&b| b as f32 * scale).collect()
        }

        DT_SIGNED_SHORT => {
            let need = n * 2;
            if raw.len() < need {
                return Err(anyhow!(
                    "Analyze .img too small for i16 data: need {} bytes, have {}",
                    need,
                    raw.len()
                ));
            }
            raw.chunks_exact(2)
                .take(n)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 * scale)
                .collect()
        }

        DT_SIGNED_INT => {
            let need = n * 4;
            if raw.len() < need {
                return Err(anyhow!(
                    "Analyze .img too small for i32 data: need {} bytes, have {}",
                    need,
                    raw.len()
                ));
            }
            raw.chunks_exact(4)
                .take(n)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 * scale)
                .collect()
        }

        DT_FLOAT => {
            let need = n * 4;
            if raw.len() < need {
                return Err(anyhow!(
                    "Analyze .img too small for f32 data: need {} bytes, have {}",
                    need,
                    raw.len()
                ));
            }
            raw.chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * scale)
                .collect()
        }

        DT_DOUBLE => {
            let need = n * 8;
            if raw.len() < need {
                return Err(anyhow!(
                    "Analyze .img too small for f64 data: need {} bytes, have {}",
                    need,
                    raw.len()
                ));
            }
            raw.chunks_exact(8)
                .take(n)
                .map(|c| {
                    let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    (v * scale as f64) as f32
                })
                .collect()
        }

        other => {
            return Err(anyhow!(
                "Unsupported Analyze datatype {}. \
                 Supported codes: 2 (u8), 4 (i16), 8 (i32), 16 (f32), 64 (f64).",
                other
            ));
        }
    };

    // ── Build RITK Image<B, 3> with shape [nz, ny, nx] ────────────────────────
    let td = TensorData::new(vals, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<B, 3>::from_data(td, device);

    let image = Image::new(
        tensor,
        Point::new([ox, oy, oz]),
        Spacing::new([sx, sy, sz]),
        Direction::identity(),
    );

    tracing::debug!(nx, ny, nz, datatype, "read_analyze: complete");

    Ok(image)
}

// ── Reader wrapper type ───────────────────────────────────────────────────────

/// Read-side wrapper type implementing the `ImageReader` domain trait.
pub struct AnalyzeReader<B: Backend> {
    pub(crate) _device: B::Device,
    pub(crate) _phantom: PhantomData<B>,
}

impl<B: Backend> AnalyzeReader<B> {
    /// Construct a reader bound to `device`.
    pub fn new(device: B::Device) -> Self {
        Self {
            _device: device,
            _phantom: PhantomData,
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Read a little-endian `i32` from `buf` at byte offset `off`.
#[inline]
fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Read a little-endian `i16` from `buf` at byte offset `off`.
#[inline]
fn read_i16(buf: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([buf[off], buf[off + 1]])
}

/// Read a little-endian `f32` from `buf` at byte offset `off`.
#[inline]
fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}
