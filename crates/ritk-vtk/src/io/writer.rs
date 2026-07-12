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
use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Encode flat voxel data plus geometry as a VTK legacy structured-points
/// stream (BINARY) into an arbitrary writer.
///
/// This is the shared, substrate-free core underlying both the burn-backed
/// [`write_vtk`] and the Coeus-backed `ritk_io` native writer: identical byte
/// output given identical inputs, since neither carrier participates in the
/// encode.
///
/// ## Argument convention
///
/// - `slice` is row-major scalar data with X varying fastest, Y next, Z slowest
///   (matching RITK's `[nz, ny, nx]` tensor memory layout); it is emitted
///   verbatim as big-endian IEEE 754 single-precision (`f32`) with no
///   permutation.
/// - `dims` is `[nz, ny, nx]` — RITK tensor order (Z slowest, X fastest); the
///   emitted `DIMENSIONS` header field is permuted to VTK **[X, Y, Z]** order.
/// - `origin` / `spacing` are `[ox, oy, oz]` / `[sx, sy, sz]` in VTK **[X, Y, Z]**
///   order, matching the `ORIGIN` / `SPACING` fields directly.
///
/// The header is always ASCII (VTK's `BINARY` declaration governs only the data
/// section). The writer is flushed before return.
///
/// # Errors
///
/// Returns an error when the writer fails, or when `slice.len()` does not equal
/// the product of `dims`.
pub fn encode_vtk_flat<W: Write>(
    writer: &mut W,
    slice: &[f32],
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Result<()> {
    let [nz, ny, nx] = dims;
    let total_voxels = nx * ny * nz;

    let [ox, oy, oz] = origin;
    let [sx, sy, sz] = spacing;

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
    if slice.len() != total_voxels {
        anyhow::bail!(
            "data contains {} elements but expected {} ({}×{}×{})",
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
        "VTK data written: {} voxels, {} bytes payload",
        total_voxels,
        total_voxels * 4
    );

    Ok(())
}

/// Write an `Image<B, 3>` to a VTK legacy structured-points file (BINARY).
///
/// The output file conforms to VTK legacy format version 3.0 with:
/// - `DATASET STRUCTURED_POINTS`
/// - `BINARY` encoding
/// - `SCALARS scalars float 1` point data
/// - Big-endian IEEE 754 single-precision scalar values
///
/// Extracts flat data and geometry from the burn tensor carrier, then delegates
/// the byte-level encode to [`encode_vtk_flat`].
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

    let dims = image.shape(); // [nz, ny, nx]
    let origin = image.origin(); // [X, Y, Z] order
    let spacing = image.spacing(); // [X, Y, Z] order
    let origin_arr = [origin[0], origin[1], origin[2]];
    let spacing_arr = [spacing[0], spacing[1], spacing[2]];

    let f32_vec = image.try_data_vec()?;

    encode_vtk_flat(&mut writer, &f32_vec, dims, origin_arr, spacing_arr)?;

    tracing::debug!(path = %path.display(), "VTK file written");

    Ok(())
}
