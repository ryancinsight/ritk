//! VTK legacy structured points format writer.
//!
//! Writes `DATASET STRUCTURED_POINTS` files in **BINARY** encoding with
//! scalar data stored as big-endian `float` (IEEE 754 single precision).
//!
//! ## Coordinate Convention
//!
//! RITK tensor shape is **[nz, ny, nx]** (Z varies slowest, X varies fastest).
//! VTK `DIMENSIONS` expects **[nx, ny, nz]** order, so the first and last
//! tensor dimensions are swapped when emitting the header.
//!
//! RITK spatial metadata (`Point`, `Spacing`) uses **[X, Y, Z]** order,
//! matching VTK's `ORIGIN` and `SPACING` fields directly.
//!
//! VTK stores scalar data with X varying fastest, matching RITK's memory
//! layout. No data permutation is required.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write an `Image<B, 3>` to a VTK legacy structured-points file (BINARY).
///
/// The output file conforms to VTK legacy format version 3.0 with:
/// - `DATASET STRUCTURED_POINTS`
/// - `BINARY` encoding
/// - `SCALARS scalars float 1` point data
/// - Big-endian IEEE 754 single-precision scalar values
///
/// # Errors
///
/// Returns an error when:
/// - The file cannot be created or written.
/// - The tensor data cannot be extracted as `f32`.
pub fn write_vtk<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create VTK file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // --- Extract spatial metadata ---

    let shape = image.shape(); // [nz, ny, nx]
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let total_voxels = nx * ny * nz;

    let origin = image.origin(); // [X, Y, Z] order
    let spacing = image.spacing(); // [X, Y, Z] order

    let ox = origin[0];
    let oy = origin[1];
    let oz = origin[2];

    let sx = spacing[0];
    let sy = spacing[1];
    let sz = spacing[2];

    tracing::debug!(
        nx,
        ny,
        nz,
        ox,
        oy,
        oz,
        sx,
        sy,
        sz,
        "VTK writer: emitting {} voxels",
        total_voxels
    );

    // --- Write header ---
    //
    // VTK legacy format requires lines terminated by '\n'. The header is
    // always ASCII regardless of the BINARY/ASCII declaration (which only
    // governs the data section).

    writeln!(writer, "# vtk DataFile Version 3.0")
        .with_context(|| "failed to write VTK version line")?;
    writeln!(writer, "RITK exported image")
        .with_context(|| "failed to write VTK description line")?;
    writeln!(writer, "BINARY").with_context(|| "failed to write VTK encoding line")?;
    writeln!(writer, "DATASET STRUCTURED_POINTS")
        .with_context(|| "failed to write VTK dataset line")?;
    writeln!(writer, "DIMENSIONS {} {} {}", nx, ny, nz)
        .with_context(|| "failed to write VTK DIMENSIONS")?;
    writeln!(writer, "ORIGIN {} {} {}", ox, oy, oz)
        .with_context(|| "failed to write VTK ORIGIN")?;
    writeln!(writer, "SPACING {} {} {}", sx, sy, sz)
        .with_context(|| "failed to write VTK SPACING")?;
    writeln!(writer, "POINT_DATA {}", total_voxels)
        .with_context(|| "failed to write VTK POINT_DATA")?;
    writeln!(writer, "SCALARS scalars float 1").with_context(|| "failed to write VTK SCALARS")?;
    writeln!(writer, "LOOKUP_TABLE default").with_context(|| "failed to write VTK LOOKUP_TABLE")?;

    // --- Write binary scalar data (big-endian f32) ---
    let f32_vec = image.try_data_vec()?;
    let slice: &[f32] = &f32_vec;

    if slice.len() != total_voxels {
        anyhow::bail!(
            "tensor contains {} elements but expected {} ({}×{}×{})",
            slice.len(),
            total_voxels,
            nx,
            ny,
            nz
        );
    }

    // Pre-allocate the full binary buffer to minimise I/O calls.
    let mut binary_buf = Vec::with_capacity(total_voxels * 4);
    for &val in slice {
        binary_buf.extend_from_slice(&val.to_be_bytes());
    }

    writer
        .write_all(&binary_buf)
        .with_context(|| "failed to write VTK binary scalar data")?;

    writer
        .flush()
        .with_context(|| "failed to flush VTK output")?;

    tracing::debug!(
        path = %path.display(),
        "VTK file written: {} voxels, {} bytes payload",
        total_voxels,
        total_voxels * 4
    );

    Ok(())
}
