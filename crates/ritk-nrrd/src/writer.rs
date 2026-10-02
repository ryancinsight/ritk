use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::spatial::file_space_directions_from_internal;
use crate::types::name_for;

/// Write a 3-D `Image` of `T` to a NRRD (Nearly Raw Raster Data) file.
///
/// # Format
/// Writes NRRD version 4 (`NRRD0004`) with `encoding: raw` and
/// `endian: little`.  The file is self-contained (inline data). The `type`
/// field names the sample type `T`: NRRD stores all ten of them, so no `T` is
/// refused.
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
/// Voxel values are written as packed `T` samples in little-endian byte order,
/// immediately after a blank header-terminator line.
///
/// # Errors
///
/// Returns an error when the file cannot be created or written, or when an
/// extent overflows the voxel count.
pub fn write_nrrd<T, B, P>(path: P, image: &Image<T, B, 3>, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
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
        crate::coordinate_map::encode(image.coordinate_map()),
    )
}

/// Like [`write_nrrd`] but uses caller-provided voxel data.
///
/// `image` supplies only spatial metadata; the binary payload comes from
/// `values`.  This lets a caller that already holds a fast (e.g. zero-copy
/// NdArray) slice skip the generic `into_data()` materialization that dominates
/// write time for large volumes.  `values.len()` must equal the voxel count.
///
/// # Errors
///
/// Returns an error, before creating a file, when `values.len()` differs from
/// the voxel count of `image`, and an error when creating or writing the file
/// fails.
pub fn write_nrrd_with_data<T: Sample, B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    image: &Image<T, B, 3>,
    values: &[T],
) -> Result<()> {
    write_nrrd_flat(
        path.as_ref(),
        image.shape(),
        image.spacing(),
        image.origin(),
        image.direction(),
        values,
        crate::coordinate_map::encode(image.coordinate_map()),
    )
}

/// NRRD serialization core. Takes flat `[Z, Y, X]` voxels plus the
/// (backend-independent) spatial metadata so header emission and byte layout
/// live in exactly one place. `values.len()` must equal the voxel count.
fn write_nrrd_flat<T: Sample>(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    values: &[T],
    coordinate_map: Option<String>,
) -> Result<()> {
    // shape is [nz, ny, nx] in RITK convention.
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    if values.len() != voxel_count {
        return Err(anyhow!(
            "NRRD payload has {} voxels but shape [{nz}, {ny}, {nx}] requires {voxel_count}",
            values.len()
        ));
    }

    // ── Spatial metadata ──────────────────────────────────────────────────
    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major(direction),
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
    writeln!(writer, "type: {}", name_for(T::TYPE))?;
    writeln!(writer, "dimension: 3")?;
    // ITK/SimpleITK and ritk's own reader work in LPS: the reader stores the
    // `space origin` / `space directions` verbatim (no space conversion), and ITK
    // NRRDs are written LPS. Declaring RAS here made SimpleITK reinterpret the
    // LPS-valued origin/directions and negate the x and y (R↔L, A↔P) components on
    // read, corrupting the origin of an anisotropic-origin volume on round-trip.
    writeln!(writer, "space: left-posterior-superior")?;
    // sizes is in NRRD [X, Y, Z] order.
    writeln!(writer, "sizes: {} {} {}", nx, ny, nz)?;
    writeln!(writer, "space directions: {} {} {}", sd0, sd1, sd2)?;
    writeln!(writer, "kinds: domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(writer, "space origin: {}", space_origin)?;
    // Acquisition geometry, as a NRRD key/value field. Cartesian is written by
    // omission, so ordinary volumes are byte-identical to before.
    if let Some(encoded) = coordinate_map {
        writeln!(
            writer,
            "{}:={}",
            crate::coordinate_map::COORDINATE_MAP_KEY,
            encoded
        )?;
    }
    // Blank line terminates the header; binary data follows immediately.
    writeln!(writer)?;

    write_samples(values, ByteOrder::LittleEndian, &mut writer)
        .context("Failed to write NRRD payload")?;

    writer.flush().context("Failed to flush NRRD output file")?;

    Ok(())
}

/// Write an acquisition series of `T` images to a NRRD file.
///
/// # Acquisition axis
///
/// The gradient/time axis is emitted **first**, as `kinds: list domain domain
/// domain` with a leading `none` in `space directions` — the NA-MIC convention
/// Slicer and DTIPrep produce and the one diffusion tooling expects. That axis
/// varies fastest, so volumes are interleaved voxel-by-voxel in the payload.
/// [`read_nrrd_series`](crate::read_nrrd_series) reads either that layout or a
/// trailing acquisition axis.
///
/// A one-volume series writes as an ordinary `dimension: 3` file, identical to
/// [`write_nrrd`], because that is the canonical form for a single volume.
///
/// # Errors
///
/// Returns an error when `volumes` is empty, when any volume's grid differs
/// from the first, or when writing fails.
pub fn write_nrrd_series<T, B, P>(path: P, volumes: &[Image<T, B, 3>], backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let Some((first, rest)) = volumes.split_first() else {
        return Err(anyhow!(
            "write_nrrd_series: a series requires at least one volume"
        ));
    };

    let shape = first.shape();
    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != shape {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} shape {:?} differs from volume 0 \
                 {shape:?}; a NRRD series has one spatial grid",
                volume.shape()
            ));
        }
        if volume.origin() != first.origin() || volume.spacing() != first.spacing() {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} origin or spacing differs from \
                 volume 0; a NRRD series has one spatial grid"
            ));
        }
    }

    let payloads: Vec<_> = volumes
        .iter()
        .map(|volume| volume.data_cow_on(backend))
        .collect();

    write_nrrd_series_flat(
        path.as_ref(),
        shape,
        first.spacing(),
        first.origin(),
        first.direction(),
        &payloads,
    )
}

fn write_nrrd_series_flat<T: Sample>(
    path: &Path,
    shape: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    payloads: &[impl std::ops::Deref<Target = [T]>],
) -> Result<()> {
    // One volume has no acquisition axis to declare, so it takes the ordinary
    // rank-3 path and stays byte-identical to `write_nrrd`.
    if let [single] = payloads {
        return write_nrrd_flat(path, shape, spacing, origin, direction, single, None);
    }

    let [nz, ny, nx] = shape;
    let voxels_per_volume = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("NRRD shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    for (position, payload) in payloads.iter().enumerate() {
        if payload.len() != voxels_per_volume {
            return Err(anyhow!(
                "write_nrrd_series: volume {position} has {} voxels but shape \
                 [{nz}, {ny}, {nx}] requires {voxels_per_volume}",
                payload.len()
            ));
        }
    }

    let file_directions = file_space_directions_from_internal(
        [spacing[0], spacing[1], spacing[2]],
        direction_row_major(direction),
    );

    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create NRRD file {:?}", path))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "NRRD0004")?;
    writeln!(writer, "# Complete NRRD file written by ritk")?;
    writeln!(writer, "type: {}", name_for(T::TYPE))?;
    writeln!(writer, "dimension: 4")?;
    writeln!(writer, "space: left-posterior-superior")?;
    // The acquisition axis leads, so `sizes` and every per-axis field carry it
    // in slot 0 while the spatial axes keep file order [x, y, z].
    writeln!(writer, "sizes: {} {} {} {}", payloads.len(), nx, ny, nz)?;
    writeln!(
        writer,
        "space directions: none {} {} {}",
        format_nrrd_vector(file_directions[0]),
        format_nrrd_vector(file_directions[1]),
        format_nrrd_vector(file_directions[2])
    )?;
    writeln!(writer, "kinds: list domain domain domain")?;
    writeln!(writer, "endian: little")?;
    writeln!(writer, "encoding: raw")?;
    writeln!(
        writer,
        "space origin: ({},{},{})",
        origin[0], origin[1], origin[2]
    )?;
    writeln!(writer)?;

    // The acquisition axis varies fastest, so voxel i of every volume is
    // written before voxel i+1 of any of them. A bulk per-volume write is not
    // available in this layout; the interleave is the format's own ordering.
    let mut interleaved = Vec::with_capacity(payloads.len() * voxels_per_volume);
    for voxel in 0..voxels_per_volume {
        for payload in payloads {
            interleaved.push(payload[voxel]);
        }
    }
    write_samples(&interleaved, ByteOrder::LittleEndian, &mut writer)
        .context("Failed to write NRRD payload")?;

    writer.flush().context("Failed to flush NRRD output file")?;
    Ok(())
}

/// Flatten a 3×3 direction-cosine matrix to the row-major layout the space
/// directions builder consumes.
fn direction_row_major(direction: &Direction<3>) -> [f64; 9] {
    let d = direction.0;
    [
        d[(0, 0)],
        d[(0, 1)],
        d[(0, 2)],
        d[(1, 0)],
        d[(1, 1)],
        d[(1, 2)],
        d[(2, 0)],
        d[(2, 1)],
        d[(2, 2)],
    ]
}

fn format_nrrd_vector(vector: [f64; 3]) -> String {
    format!("({},{},{})", vector[0], vector[1], vector[2])
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

impl<B: ComputeBackend + Default> NrrdWriter<B> {
    /// Write `image` to the NRRD file at `path`, storing its sample type `T`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`write_nrrd`]: the file cannot be created or
    /// written, or an extent overflows the voxel count.
    pub fn write<T, P>(&self, path: P, image: &Image<T, B, 3>) -> Result<()>
    where
        T: Sample,
        B::DeviceBuffer<T>: CpuAddressableStorage<T>,
        P: AsRef<Path>,
    {
        write_nrrd(path, image, &self.backend)
    }
}
