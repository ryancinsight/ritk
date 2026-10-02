//! VTK legacy structured points format writer.
//!
//! Writes `DATASET STRUCTURED_POINTS` files in **BINARY** encoding with
//! scalar data stored as big-endian samples of the image's own type `T`: the
//! `SCALARS` line names the legacy type of `T` (`unsigned_char`, `char`,
//! `unsigned_short`, `short`, `unsigned_int`, `int`, `vtktypeuint64`,
//! `vtktypeint64`, `float`, or `double`). The legacy format has a name for
//! every sample type, so no `T` is refused.
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

use crate::io::scalar_type::name_for;
use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Sample};
use ritk_image::Image;
use ritk_spatial::Direction;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Encode flat voxel data plus geometry as a VTK legacy structured-points
/// stream (BINARY) into an arbitrary writer.
///
/// This is the shared, substrate-free core underlying both the coeus-backed
/// [`write_vtk`]: identical byte output given identical inputs, since no
/// carrier participates in the encode.
///
/// ## Argument convention
///
/// - `slice` is row-major scalar data with X varying fastest, Y next, Z slowest
///   (matching RITK's `[nz, ny, nx]` tensor memory layout); it is emitted
///   verbatim as big-endian samples of `T` with no permutation. Its type `T`
///   is the stored type: the `SCALARS` line names it.
/// - `dims` is `[nz, ny, nx]` — RITK tensor order (Z slowest, X fastest); the
///   emitted `DIMENSIONS` header field is permuted to VTK **[X, Y, Z]** order.
/// - `origin` / `spacing` are `[ox, oy, oz]` / `[sx, sy, sz]` in VTK **[X, Y, Z]**
///   order, matching the `ORIGIN` / `SPACING` fields directly.
///
/// The header is always ASCII (VTK's `BINARY` declaration governs only the data
/// section). The writer is flushed before return.
///
/// This flat-data encoder has no direction or coordinate-map inputs. Callers
/// writing an [`Image`] should use [`write_vtk`], which checks that its
/// geometry is representable before touching the destination.
///
/// # Errors
///
/// Returns an error when the writer fails, or when `slice.len()` does not equal
/// the product of `dims`.
pub fn encode_vtk_flat<T: Sample, W: Write>(
    writer: &mut W,
    slice: &[T],
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Result<()> {
    let [nz, ny, nx] = dims;
    let total_voxels = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .with_context(|| format!("VTK dimension product overflows usize: {nx}×{ny}×{nz}"))?;

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
    writeln!(writer, "SCALARS scalars {} 1", name_for(T::TYPE))
        .with_context(|| "failed to write VTK SCALARS")?;
    writeln!(writer, "LOOKUP_TABLE default").with_context(|| "failed to write VTK LOOKUP_TABLE")?;

    // --- Write binary scalar data (big-endian, in T's width) ---
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

    write_samples(slice, ByteOrder::BigEndian, writer)
        .with_context(|| "failed to write VTK binary scalar data")?;

    writer
        .flush()
        .with_context(|| "failed to flush VTK output")?;

    tracing::debug!(
        "VTK data written: {} voxels, {} bytes payload",
        total_voxels,
        total_voxels * T::TYPE.byte_width()
    );

    Ok(())
}

/// Write a native Coeus image of `T` to a VTK legacy structured-points file
/// (BINARY), storing `T`'s samples.
///
/// The output file conforms to VTK legacy format version 3.0 with:
/// - `DATASET STRUCTURED_POINTS`
/// - `BINARY` encoding
/// - `SCALARS scalars <name> 1` point data, where `<name>` is the legacy type
///   name of the image's sample type `T` (`char` for `i8`, `vtktypeint64` for
///   `i64`, `float` for `f32`, and so on)
/// - big-endian scalar values of `T`, as many bytes per voxel as `<name>` reads
///   back as
///
/// Extracts flat data and geometry from the native tensor carrier, then
/// delegates the byte-level encode to [`encode_vtk_flat`].
///
/// # Errors
///
/// Returns an error when:
/// - The image has a non-identity direction or non-Cartesian coordinate map,
///   which legacy structured points cannot represent. This check occurs before
///   the destination is created or truncated.
/// - The file cannot be created or written.
pub fn write_vtk<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    anyhow::ensure!(
        image.coordinate_map().is_cartesian(),
        "legacy VTK structured points cannot preserve a non-Cartesian coordinate map"
    );
    anyhow::ensure!(
        image.direction() == &Direction::identity(),
        "legacy VTK structured points cannot preserve a non-identity direction matrix"
    );

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create VTK file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    let dims = image.shape(); // [nz, ny, nx]
    let origin = image.origin(); // [X, Y, Z] order
    let spacing = image.spacing(); // [X, Y, Z] order
    let origin_arr = [origin[0], origin[1], origin[2]];
    let spacing_arr = [spacing[0], spacing[1], spacing[2]];

    let voxels = image.data_cow_on(backend);

    encode_vtk_flat(&mut writer, &voxels, dims, origin_arr, spacing_arr)?;

    tracing::debug!(path = %path.display(), "VTK file written");

    Ok(())
}
