use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::{Image, ImageMetadata};
use ritk_image_io::{validate_coordinate_map, validate_physical_geometry};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

use super::formatting::write_float_payload;
use super::{write_nrrd_header, HeaderBuffer};

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
pub fn write_nrrd<B, P>(path: P, image: &Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    // RITK [Z,Y,X] flat layout is already NRRD X-fastest raw order.  Extract via
    // the backend's fast host path to avoid the `into_data()` materialization.
    let voxels = image.data_cow_on(backend);
    write_nrrd_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        &voxels,
        image.coordinate_map(),
    )
}

/// Like [`write_nrrd`] but uses caller-provided voxel data.
///
/// `image` supplies only spatial metadata; the binary payload comes from
/// `f32_slice`.  This lets a caller that already holds a fast (e.g. zero-copy
/// NdArray) slice skip the generic `into_data()` materialization that dominates
/// write time for large volumes.  `f32_slice.len()` must equal the voxel count.
pub fn write_nrrd_with_data<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
    f32_slice: &[f32],
) -> Result<()> {
    write_nrrd_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        f32_slice,
        image.coordinate_map(),
    )
}

/// NRRD serialization core. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header emission and byte layout
/// live in exactly one place. `f32_slice.len()` must equal the voxel count.
pub(crate) fn write_nrrd_flat(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    f32_slice: &[f32],
    coordinate_map: &CoordinateMap,
) -> Result<()> {
    // shape is [nz, ny, nx] in RITK convention.
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    if f32_slice.len() != voxel_count {
        return Err(anyhow!(
            "NRRD payload has {} voxels but shape [{nz}, {ny}, {nx}] requires {voxel_count}",
            f32_slice.len()
        ));
    }
    validate_physical_geometry(&ImageMetadata::new(*origin, *spacing, *direction))?;
    validate_coordinate_map(coordinate_map, shape)?;

    let mut header = HeaderBuffer::new();
    let header_result = write_nrrd_header(
        &mut header,
        shape,
        spacing,
        origin,
        direction,
        "float",
        coordinate_map,
    );
    if header.exceeded_limit() {
        let maximum_bytes = crate::reader::MAX_HEADER_BYTES;
        anyhow::bail!("NRRD output header is larger than {maximum_bytes} bytes");
    }
    header_result?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(header.bytes())?;

    write_float_payload(&mut writer, f32_slice)?;

    writer.flush().context("Failed to flush NRRD output file")?;

    Ok(())
}

// ── Public writer struct ──────────────────────────────────────────────────────

/// Thin writer struct for NRRD files.
///
/// The backend `B` is supplied per-call so a single `NrrdWriter` instance can
/// write images from different backends.
pub struct NrrdWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NrrdWriter<B> {
    /// Creates a writer that extracts image storage through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> NrrdWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Write `image` to the NRRD file at `path`.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> Result<()> {
        write_nrrd(path, image, &self.backend)
    }
}
