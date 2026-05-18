use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::spatial::file_space_directions_from_internal;

/// Write a 3-D `Image` to a NRRD (Nearly Raw Raster Data) file.
///
/// # Format
/// Writes NRRD version 4 (`NRRD0004`) with `encoding: raw` and
/// `endian: little`.  The file is self-contained (inline data).
///
/// # Axis convention
/// RITK stores voxels in `[Z, Y, X]` order. NRRD stores raw data with X as
/// the fastest-varying axis. These flat orders are identical, so voxel bytes
/// are written directly while the `sizes` header is emitted as `nx ny nz`
/// (`shape()[2] shape()[1] shape()[0]` of the RITK image).
///
/// # Spatial metadata
/// * `space directions` — NRRD file-axis vectors `[x,y,z]` are emitted from
///   RITK metadata columns `[col,row,depth]`, each scaled by its matching
///   spacing.
/// * `space origin` — the image origin in physical `[X, Y, Z]` space.
///
/// # Binary payload
/// Voxel values are written as 32-bit IEEE 754 floats in little-endian byte
/// order, immediately after a blank header-terminator line.
pub fn write_nrrd<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    // ── Voxel data ────────────────────────────────────────────────────────
    // RITK [Z,Y,X] flat layout is already NRRD X-fastest raw order.
    let f32_vec = image.try_data_vec()?;
    let f32_slice: &[f32] = &f32_vec;

    // image.shape() is [nz, ny, nx] in RITK convention.
    let shape = image.shape();
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Spatial metadata ──────────────────────────────────────────────────
    let spacing = image.spacing();
    let origin = image.origin();
    let dir = image.direction().0;

    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        [
            dir[(0, 0)],
            dir[(0, 1)],
            dir[(0, 2)],
            dir[(1, 0)],
            dir[(1, 1)],
            dir[(1, 2)],
            dir[(2, 0)],
            dir[(2, 1)],
            dir[(2, 2)],
        ],
    );
    let sd0 = format_nrrd_vector(file_directions[0]);
    let sd1 = format_nrrd_vector(file_directions[1]);
    let sd2 = format_nrrd_vector(file_directions[2]);

    let space_origin = format!("({},{},{})", origin[0], origin[1], origin[2]);

    // ── File I/O ──────────────────────────────────────────────────────────
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    // Header — field order matches the ITK NrrdIO convention.
    writeln!(writer, "NRRD0004")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: float")?;
    writeln!(writer, "dimension: 3")?;
    writeln!(writer, "space: right-anterior-superior")?;
    // sizes is in NRRD [X, Y, Z] order.
    writeln!(writer, "sizes: {} {} {}", nx, ny, nz)?;
    writeln!(writer, "space directions: {} {} {}", sd0, sd1, sd2)?;
    writeln!(writer, "kinds: domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(writer, "space origin: {}", space_origin)?;
    // Blank line terminates the header; binary data follows immediately.
    writeln!(writer)?;

    // Binary payload — little-endian f32.
    for &v in f32_slice {
        writer.write_all(&v.to_le_bytes())?;
    }

    writer.flush().context("Failed to flush NRRD output file")?;

    Ok(())
}

fn format_nrrd_vector(vector: [f64; 3]) -> String {
    format!("({},{},{})", vector[0], vector[1], vector[2])
}

// ── Public writer struct ──────────────────────────────────────────────────────

/// Thin writer struct for NRRD files.
///
/// The backend `B` is supplied per-call so a single `NrrdWriter` instance can
/// write images from different backends.
pub struct NrrdWriter;

impl NrrdWriter {
    /// Write `image` to the NRRD file at `path`.
    pub fn write<B: Backend, P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> Result<()> {
        write_nrrd(path, image)
    }
}
