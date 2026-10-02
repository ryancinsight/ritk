//! MGH / MGZ writer for 3-D volumetric images and acquisition series.
//!
//! The writer emits FreeSurfer MGH with the `type` code of the image's sample
//! type `T`: `MRI_UCHAR`, `MRI_SHORT`, `MRI_INT`, or `MRI_FLOAT` for `u8`,
//! `i16`, `i32`, or `f32`; MGH has no code for the other sample types. Paths
//! ending in `.mgz` or `.mgh.gz` are gzip-compressed. The series writer emits
//! one frame per volume with a shared spatial grid.

use crate::spatial::ras_center_from_geometry;
use crate::types::code_for;
use crate::{is_gzip_path, DOF_UNSET, GOOD_RAS_VALID, PADDING_LEN, SINGLE_FRAME, VERSION};
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use consus_core::{write_to, ByteOrder};
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_codecs::sample::{write_samples, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

#[cfg(test)]
mod tests;

/// Write a 3-D `Image` of `T` as an MGH or MGZ file storing `T`'s samples.
///
/// # Errors
///
/// Returns an error when MGH has no `type` code for `T` (it stores `u8`,
/// `i16`, `i32`, and `f32`), when an extent exceeds the header's `i32`, or when
/// writing fails.
pub fn write_mgh<T, B, P>(image: &Image<T, B, 3>, path: P, backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let code = code_for(T::TYPE)?;
    let voxels = image.data_cow_on(backend);
    write_mgh_stream(
        path.as_ref(),
        Grid {
            shape: image.shape(),
            origin: *image.origin(),
            spacing: *image.spacing(),
            direction: *image.direction(),
        },
        code,
        &voxels,
    )
}

/// The spatial grid one MGH header describes, shared by every frame.
#[derive(Clone, Copy)]
struct Grid {
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

/// Substrate-agnostic MGH file entry: creates the file, applies the gzip
/// branch, and writes the header and `voxels` as one frame.
fn write_mgh_stream<T: Sample>(path: &Path, grid: Grid, code: i32, voxels: &[T]) -> Result<()> {
    write_mgh_file(path, |writer| {
        write_header(writer, grid, SINGLE_FRAME, code)?;
        write_frame(writer, grid.shape, voxels)
    })
}

/// Create `path`, gzip-wrapped for `.mgz` and `.mgh.gz`, and hand `write` the
/// stream.
///
/// The sink is `dyn Write` so the gzip and plain branches share one header
/// and payload writer: a per-file boundary the samples cross in blocks of
/// thousands, never per sample.
fn write_mgh_file(path: &Path, write: impl FnOnce(&mut dyn Write) -> Result<()>) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        write(&mut encoder)?;
        encoder.finish().context("Failed to finalize gzip stream")?;
    } else {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush().context("Failed to flush MGH output")?;
    }
    Ok(())
}

/// Serialize the 284-byte MGH header for `nframes` frames of `code` samples on
/// `grid`.
fn write_header(writer: &mut dyn Write, grid: Grid, nframes: i32, code: i32) -> Result<()> {
    let [nz, ny, nx] = grid.shape;

    write_to(writer, VERSION, ByteOrder::BigEndian)?;
    for (axis, extent) in [("x", nx), ("y", ny), ("z", nz)] {
        let extent = i32::try_from(extent)
            .with_context(|| format!("MGH {axis}-axis extent {extent} exceeds i32"))?;
        write_to(writer, extent, ByteOrder::BigEndian)?;
    }
    write_to(writer, nframes, ByteOrder::BigEndian)?;
    write_to(writer, code, ByteOrder::BigEndian)?;
    write_to(writer, DOF_UNSET, ByteOrder::BigEndian)?;
    write_to(writer, GOOD_RAS_VALID, ByteOrder::BigEndian)?;

    for axis in 0..3 {
        write_to(writer, grid.spacing[axis] as f32, ByteOrder::BigEndian)?;
    }

    for col in 0..3 {
        for row in 0..3 {
            write_to(
                writer,
                grid.direction[(row, col)] as f32,
                ByteOrder::BigEndian,
            )?;
        }
    }

    let c_ras = ras_center_from_geometry(grid.origin, grid.spacing, grid.direction, grid.shape);
    for axis in 0..3 {
        write_to(writer, c_ras[axis] as f32, ByteOrder::BigEndian)?;
    }

    writer
        .write_all(&[0u8; PADDING_LEN])
        .context("Failed to write MGH header padding")
}

/// Write one frame of big-endian samples, which must fill `shape`.
fn write_frame<T: Sample>(writer: &mut dyn Write, shape: [usize; 3], voxels: &[T]) -> Result<()> {
    let n_voxels = voxel_count(shape)?;
    if voxels.len() != n_voxels {
        return Err(anyhow!(
            "Tensor data length {} does not match shape {shape:?} = {n_voxels} voxels",
            voxels.len()
        ));
    }
    write_samples(voxels, ByteOrder::BigEndian, writer).context("Failed to write MGH voxel data")
}

/// Voxels in one frame of `shape`.
fn voxel_count([nz, ny, nx]: [usize; 3]) -> Result<usize> {
    nx.checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("MGH shape [{nz}, {ny}, {nx}] voxel count overflows usize"))
}

/// Write an acquisition series to an MGH or MGZ file.
///
/// Each image in the series must share the same spatial grid (shape, origin,
/// spacing, and direction), because MGH represents a series as one header
/// with `nframes` identical-geometry volumes. A one-volume series writes as
/// `nframes = 1`, identical to [`write_mgh`].
///
/// # Errors
///
/// Returns an error when `volumes` is empty, when any volume's grid differs
/// from the first, when MGH has no `type` code for `T`, or when writing fails.
pub fn write_mgh_series<T, B, P>(path: P, volumes: &[Image<T, B, 3>], backend: &B) -> Result<()>
where
    T: Sample,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    P: AsRef<Path>,
{
    let code = code_for(T::TYPE)?;
    let Some((first, rest)) = volumes.split_first() else {
        return Err(anyhow!(
            "write_mgh_series: a series requires at least one volume"
        ));
    };

    let shape = first.shape();
    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != shape {
            return Err(anyhow!(
                "write_mgh_series: volume {position} shape {:?} differs from volume 0 \
                 {shape:?}; an MGH series has one spatial grid",
                volume.shape()
            ));
        }
        if volume.origin() != first.origin()
            || volume.spacing() != first.spacing()
            || volume.direction() != first.direction()
        {
            return Err(anyhow!(
                "write_mgh_series: volume {position} origin, spacing, or direction \
                 differs from volume 0; an MGH series has one spatial grid"
            ));
        }
    }

    let nframes = i32::try_from(volumes.len())
        .context("MGH series frame count exceeds i32 header capacity")?;
    let grid = Grid {
        shape,
        origin: *first.origin(),
        spacing: *first.spacing(),
        direction: *first.direction(),
    };
    write_mgh_file(path.as_ref(), |writer| {
        write_header(writer, grid, nframes, code)?;
        for (position, volume) in volumes.iter().enumerate() {
            let voxels = volume.data_cow_on(backend);
            write_frame(writer, shape, &voxels)
                .with_context(|| format!("write_mgh_series: volume {position}"))?;
        }
        Ok(())
    })
}

/// Stateless writer for MGH / MGZ files.
pub struct MghWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MghWriter<B> {
    /// Creates a writer that extracts image storage through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: ComputeBackend + Default> MghWriter<B> {
    /// Write `image` to the MGH or MGZ file at `path`.
    ///
    /// # Errors
    ///
    /// Returns the error of [`write_mgh`].
    pub fn write<T, P>(&self, image: &Image<T, B, 3>, path: P) -> Result<()>
    where
        T: Sample,
        B::DeviceBuffer<T>: CpuAddressableStorage<T>,
        P: AsRef<Path>,
    {
        write_mgh(image, path, &self.backend)
    }

    /// Write `volumes` as an MGH or MGZ series to `path`.
    ///
    /// # Errors
    ///
    /// Returns the error of [`write_mgh_series`].
    pub fn write_series<T, P>(&self, volumes: &[Image<T, B, 3>], path: P) -> Result<()>
    where
        T: Sample,
        B::DeviceBuffer<T>: CpuAddressableStorage<T>,
        P: AsRef<Path>,
    {
        write_mgh_series(path, volumes, &self.backend)
    }
}
