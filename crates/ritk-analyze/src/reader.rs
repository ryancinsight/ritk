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
use std::io::Read;
use std::path::Path;

use crate::codec::{read_le, HDR_SIZE};
pub use crate::codec::{DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR};

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
/// - `sizeof_hdr` is not 348 (invalid Analyze file).
/// - Any image dimension is zero.
/// - The `.img` file is smaller than the declared data size.
/// - `datatype` is not one of the five supported codes.
pub fn read_analyze<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<ritk_image::native::Image<f32, B, 3>> {
    let DecodedAnalyze {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_analyze(path)?;

    ritk_image::native::Image::from_flat_on(data, dims, origin, spacing, direction, backend)
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
    let mut hdr_file = std::fs::File::open(&hdr_path).context("Cannot open Analyze header")?;
    let mut hdr = [0u8; HDR_SIZE];
    hdr_file
        .read_exact(&mut hdr)
        .with_context(|| "Cannot read 348-byte header".to_string())?;

    // sizeof_hdr must be exactly 348.
    let sizeof_hdr = read_le::<i32>(&hdr, 0);
    if sizeof_hdr != HDR_SIZE as i32 {
        return Err(anyhow!(
            "Invalid Analyze file: sizeof_hdr={} (expected 348)",
            sizeof_hdr
        ));
    }

    // ── Parse image dimensions ────────────────────────────────────────────────
    let nx = read_le::<i16>(&hdr, 42) as usize;
    let ny = read_le::<i16>(&hdr, 44) as usize;
    let nz = read_le::<i16>(&hdr, 46) as usize;

    if nx == 0 || ny == 0 || nz == 0 {
        return Err(anyhow!(
            "Invalid Analyze dimensions: nx={} ny={} nz={}",
            nx,
            ny,
            nz
        ));
    }

    // ── Parse voxel type ──────────────────────────────────────────────────────
    let datatype = read_le::<i16>(&hdr, 70);

    // ── Parse physical spacing (pixdim[1..3]) ─────────────────────────────────
    let sx_raw = read_le::<f32>(&hdr, 80) as f64;
    let sy_raw = read_le::<f32>(&hdr, 84) as f64;
    let sz_raw = read_le::<f32>(&hdr, 88) as f64;
    // Fall back to unit spacing when stored value is zero or negative.
    let sx = if sx_raw > 0.0 { sx_raw } else { 1.0 };
    let sy = if sy_raw > 0.0 { sy_raw } else { 1.0 };
    let sz = if sz_raw > 0.0 { sz_raw } else { 1.0 };

    // ── Parse scale factor (funused1 at offset 112) ───────────────────────────
    let scale_raw = read_le::<f32>(&hdr, 112);
    let scale = if scale_raw == 0.0 { 1.0_f32 } else { scale_raw };

    // ── Parse vox_offset (offset 108) ────────────────────────────────────────
    let vox_offset = { read_le::<f32>(&hdr, 108) as u64 };

    // ── Parse origin from originator[10] (5 × i16 at offset 253) ─────────────
    let ox_vox = read_le::<i16>(&hdr, 253) as f64;
    let oy_vox = read_le::<i16>(&hdr, 255) as f64;
    let oz_vox = read_le::<i16>(&hdr, 257) as f64;
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
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<ritk_image::native::Image<f32, B, 3>> {
        read_analyze(path, &self.backend)
    }
}
