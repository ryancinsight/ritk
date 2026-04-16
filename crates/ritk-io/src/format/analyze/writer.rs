//! Analyze 7.5 writer — produces a `.hdr` header file and a `.img` raw-data file.
//!
//! # Format Overview
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files sharing the
//! same base name:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw IEEE-754 single-precision voxel values (little-endian).
//!
//! # Axis Convention
//!
//! The Analyze format stores voxels with X varying fastest and Z varying slowest
//! (column-major for the [X, Y, Z] axis order):
//!
//! ```text
//!   flat_index(ix, iy, iz) = ix + nx·iy + nx·ny·iz
//! ```
//!
//! RITK stores tensors with shape `[nz, ny, nx]` using Z-major order:
//!
//! ```text
//!   flat_index(iz, iy, ix) = iz·ny·nx + iy·nx + ix
//! ```
//!
//! Both layouts produce the **same byte sequence** for equal (nx, ny, nz), so
//! no axis permutation is required for the raw data.  The header fields are
//! set accordingly: `dim[1]=nx`, `dim[2]=ny`, `dim[3]=nz`.
//!
//! # Spatial Metadata
//!
//! RITK's `origin` and `spacing` are stored in physical `[X, Y, Z]` order.
//! This matches the Analyze convention: `pixdim[1]=sx`, `pixdim[2]=sy`,
//! `pixdim[3]=sz`.  The origin is written to the `originator` field as five
//! little-endian `i16` values encoding voxel coordinates
//! `(round(ox/sx), round(oy/sy), round(oz/sz), 0, 0)`.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;

use std::marker::PhantomData;
use std::path::Path;

// ── Public API ────────────────────────────────────────────────────────────────

/// Write a 3-D image to an Analyze 7.5 `.hdr` + `.img` file pair.
///
/// `path` must have a `.hdr` extension (or any other extension); the `.img`
/// sibling file is derived by replacing the extension with `.img`.  An existing
/// `.img` file at the derived path is overwritten.
///
/// # Errors
/// Returns an error if:
/// - `path`'s parent directory does not exist.
/// - Any dimension exceeds `i16::MAX` (32 767).
/// - Writing the header or data file fails.
pub fn write_analyze<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    // Derive sibling paths (<base>.hdr, <base>.img).
    let hdr_path = path.with_extension("hdr");
    let img_path = path.with_extension("img");

    // Extract voxel values from the tensor as f32.
    let td = image.data().clone().into_data();
    let vals: Vec<f32> = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("Analyze writer requires f32 image data: {:?}", e))?
        .to_vec();

    // Spatial metadata.  RITK shape = [nz, ny, nx]; spacing/origin in XYZ order.
    let shape = image.shape(); // [nz, ny, nx]
    let [nz, ny, nx] = shape;
    let sp = image.spacing(); // [sx, sy, sz]
    let orig = image.origin(); // [ox, oy, oz]

    // Validate dimensions fit in i16 (Analyze constraint).
    for (name, &val) in [("nx", &nx), ("ny", &ny), ("nz", &nz)].iter() {
        if val > i16::MAX as usize {
            anyhow::bail!(
                "Analyze: dimension {name}={val} exceeds i16::MAX ({})",
                i16::MAX
            );
        }
    }

    // ── Build 348-byte header ─────────────────────────────────────────────────
    let mut hdr = [0u8; 348];

    write_i32(&mut hdr, 0, 348); // sizeof_hdr
    write_i32(&mut hdr, 32, 16384); // extents
    hdr[38] = b'r'; // regular

    // image_dimension — dim[8] at offset 40
    write_i16(&mut hdr, 40, 4); // dim[0] = num dimensions
    write_i16(&mut hdr, 42, nx as i16); // dim[1] = X
    write_i16(&mut hdr, 44, ny as i16); // dim[2] = Y
    write_i16(&mut hdr, 46, nz as i16); // dim[3] = Z
    write_i16(&mut hdr, 48, 1); // dim[4] = time (1 volume)

    write_i16(&mut hdr, 70, 16); // datatype = DT_FLOAT (16)
    write_i16(&mut hdr, 72, 32); // bitpix   = 32 bits

    // pixdim[8] at offset 76
    write_f32(&mut hdr, 76, 4.0_f32); // pixdim[0] = number of dims
    write_f32(&mut hdr, 80, sp[0] as f32); // pixdim[1] = sx
    write_f32(&mut hdr, 84, sp[1] as f32); // pixdim[2] = sy
    write_f32(&mut hdr, 88, sp[2] as f32); // pixdim[3] = sz
    write_f32(&mut hdr, 92, 1.0_f32); // pixdim[4] = TR (unused)

    write_f32(&mut hdr, 108, 0.0_f32); // vox_offset
    write_f32(&mut hdr, 112, 1.0_f32); // funused1 = scale factor (1 = no scaling)

    // data_history — descrip[80] at offset 148
    let descrip = b"RITK";
    hdr[148..148 + descrip.len()].copy_from_slice(descrip);

    // originator[10] at offset 253 — voxel-space origin (5 × i16)
    let ox_vox = vox_coord(orig[0], sp[0]);
    let oy_vox = vox_coord(orig[1], sp[1]);
    let oz_vox = vox_coord(orig[2], sp[2]);
    write_i16(&mut hdr, 253, ox_vox); // originator[0] = x voxel
    write_i16(&mut hdr, 255, oy_vox); // originator[1] = y voxel
    write_i16(&mut hdr, 257, oz_vox); // originator[2] = z voxel

    // Write .hdr
    std::fs::write(&hdr_path, &hdr).context("Failed to write Analyze header")?;

    // ── Write .img (raw f32 little-endian, same memory order as RITK) ─────────
    // RITK layout: flat[iz*ny*nx + iy*nx + ix] — identical to Analyze X-fastest.
    let mut img_data = Vec::with_capacity(vals.len() * 4);
    for v in &vals {
        img_data.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&img_path, &img_data).context("Failed to write Analyze data")?;

    tracing::debug!(
        shape = ?shape,
        "write_analyze: complete"
    );

    Ok(())
}

// ── Analyze writer wrapper type ───────────────────────────────────────────────

/// Write-side type implementing the `ImageWriter` domain trait.
pub struct AnalyzeWriter<B> {
    pub(crate) _phantom: PhantomData<B>,
}

impl<B: Backend> AnalyzeWriter<B> {
    /// Construct a new writer.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> Default for AnalyzeWriter<B> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Convert physical origin coordinate to voxel index (rounded, clamped to i16).
#[inline]
fn vox_coord(origin_mm: f64, spacing_mm: f64) -> i16 {
    if spacing_mm.abs() < f64::EPSILON {
        return 0;
    }
    let vox = (origin_mm / spacing_mm).round();
    vox.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// Write a little-endian `i32` at byte offset `off` in `buf`.
#[inline]
fn write_i32(buf: &mut [u8], off: usize, val: i32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `i16` at byte offset `off` in `buf`.
#[inline]
fn write_i16(buf: &mut [u8], off: usize, val: i16) {
    buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `f32` at byte offset `off` in `buf`.
#[inline]
fn write_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}
