//! MGH / MGZ writer for 3-D volumetric images and acquisition series.
//!
//! The writer emits FreeSurfer MGH with `MRI_FLOAT` voxel data. Paths ending
//! in `.mgz` or `.mgh.gz` are gzip-compressed. The series writer emits one
//! frame per volume with a shared spatial grid.

use crate::binary::{write_f32_be, write_i16_be, write_i32_be};
use crate::spatial::ras_center_from_geometry;
use crate::{
    is_gzip_path, DOF_UNSET, GOOD_RAS_VALID, MRI_FLOAT, PADDING_LEN, SINGLE_FRAME, VERSION,
};
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

#[cfg(test)]
mod tests;

/// Write a 3-D `Image` as an MGH or MGZ file.
pub fn write_mgh<B, P>(image: &Image<f32, B, 3>, path: P, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_mgh_stream(
        path.as_ref(),
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &voxels,
    )
}

/// Substrate-agnostic MGH file entry: creates the file, applies the gzip
/// branch, and delegates to [`write_mgh_flat`].
fn write_mgh_stream(
    path: &Path,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    f32_slice: &[f32],
) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_mgh_flat(&mut encoder, shape, origin, spacing, direction, f32_slice)?;
        encoder.finish().context("Failed to finalize gzip stream")?;
    } else {
        let mut writer = BufWriter::new(file);
        write_mgh_flat(&mut writer, shape, origin, spacing, direction, f32_slice)?;
        writer.flush().context("Failed to flush MGH output")?;
    }
    Ok(())
}

/// Serialize the MGH header and big-endian `f32` voxel payload to `writer`
/// from flat `[Z, Y, X]` voxels plus (backend-independent) spatial metadata.
fn write_mgh_flat<W: Write>(
    writer: &mut W,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    f32_slice: &[f32],
) -> Result<()> {
    let [nz, ny, nx] = shape;

    write_i32_be(writer, VERSION)?;
    for (axis, extent) in [("x", nx), ("y", ny), ("z", nz)] {
        let extent = i32::try_from(extent)
            .with_context(|| format!("MGH {axis}-axis extent {extent} exceeds i32"))?;
        write_i32_be(writer, extent)?;
    }
    write_i32_be(writer, SINGLE_FRAME)?;
    write_i32_be(writer, MRI_FLOAT)?;
    write_i32_be(writer, DOF_UNSET)?;
    write_i16_be(writer, GOOD_RAS_VALID)?;

    for axis in 0..3 {
        write_f32_be(writer, spacing[axis] as f32)?;
    }

    for col in 0..3 {
        for row in 0..3 {
            write_f32_be(writer, direction[(row, col)] as f32)?;
        }
    }

    let c_ras = ras_center_from_geometry(origin, spacing, direction, [nz, ny, nx]);
    write_f32_be(writer, c_ras[0] as f32)?;
    write_f32_be(writer, c_ras[1] as f32)?;
    write_f32_be(writer, c_ras[2] as f32)?;

    writer
        .write_all(&[0u8; PADDING_LEN])
        .context("Failed to write MGH header padding")?;

    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("MGH shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    if f32_slice.len() != n_voxels {
        return Err(anyhow!(
            "Tensor data length {} does not match shape [{}, {}, {}] = {} voxels",
            f32_slice.len(),
            nz,
            ny,
            nx,
            n_voxels
        ));
    }

    for &value in f32_slice {
        writer
            .write_all(&value.to_be_bytes())
            .context("Failed to write MGH voxel data")?;
    }

    Ok(())
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
/// from the first, or when writing fails.
pub fn write_mgh_series<B, P>(path: P, volumes: &[Image<f32, B, 3>], backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
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

    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_mgh_series_flat(
            &mut encoder,
            shape,
            *first.origin(),
            *first.spacing(),
            *first.direction(),
            volumes,
            backend,
        )?;
        encoder.finish().context("Failed to finalize gzip stream")?;
    } else {
        let mut writer = BufWriter::new(file);
        write_mgh_series_flat(
            &mut writer,
            shape,
            *first.origin(),
            *first.spacing(),
            *first.direction(),
            volumes,
            backend,
        )?;
        writer.flush().context("Failed to flush MGH output")?;
    }
    Ok(())
}

/// Serialize the MGH header and big-endian `f32` voxel payload for a series.
fn write_mgh_series_flat<W: Write, B: ComputeBackend + Default>(
    writer: &mut W,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    volumes: &[Image<f32, B, 3>],
    backend: &B,
) -> Result<()>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let [nz, ny, nx] = shape;
    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .ok_or_else(|| anyhow!("MGH shape [{nz}, {ny}, {nx}] voxel count overflows usize"))?;
    let nframes_i32 = i32::try_from(volumes.len())
        .context("MGH series frame count exceeds i32 header capacity")?;

    write_i32_be(writer, VERSION)?;
    for (axis, extent) in [("x", nx), ("y", ny), ("z", nz)] {
        let extent = i32::try_from(extent)
            .with_context(|| format!("MGH {axis}-axis extent {extent} exceeds i32"))?;
        write_i32_be(writer, extent)?;
    }
    write_i32_be(writer, nframes_i32)?;
    write_i32_be(writer, MRI_FLOAT)?;
    write_i32_be(writer, DOF_UNSET)?;
    write_i16_be(writer, GOOD_RAS_VALID)?;

    for axis in 0..3 {
        write_f32_be(writer, spacing[axis] as f32)?;
    }

    for col in 0..3 {
        for row in 0..3 {
            write_f32_be(writer, direction[(row, col)] as f32)?;
        }
    }

    let c_ras = ras_center_from_geometry(origin, spacing, direction, [nz, ny, nx]);
    write_f32_be(writer, c_ras[0] as f32)?;
    write_f32_be(writer, c_ras[1] as f32)?;
    write_f32_be(writer, c_ras[2] as f32)?;

    writer
        .write_all(&[0u8; PADDING_LEN])
        .context("Failed to write MGH header padding")?;

    for (position, volume) in volumes.iter().enumerate() {
        let voxels = volume.data_cow_on(backend);
        if voxels.len() != n_voxels {
            return Err(anyhow!(
                "write_mgh_series: volume {position} has {} voxels but shape \
                 [{nz}, {ny}, {nx}] requires {n_voxels}",
                voxels.len()
            ));
        }
        for &value in voxels.as_ref() {
            writer
                .write_all(&value.to_be_bytes())
                .context("Failed to write MGH voxel data")?;
        }
    }

    Ok(())
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

impl<B> MghWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Write `image` to the MGH or MGZ file at `path`.
    pub fn write<P: AsRef<Path>>(&self, image: &Image<f32, B, 3>, path: P) -> Result<()> {
        write_mgh(image, path, &self.backend)
    }

    /// Write `volumes` as an MGH or MGZ series to `path`.
    pub fn write_series<P: AsRef<Path>>(
        &self,
        volumes: &[Image<f32, B, 3>],
        path: P,
    ) -> Result<()> {
        write_mgh_series(path, volumes, &self.backend)
    }
}
