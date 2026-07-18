//! MGH / MGZ writer for 3-D volumetric images.
//!
//! The writer emits FreeSurfer MGH with `MRI_FLOAT` voxel data. Paths ending
//! in `.mgz` or `.mgh.gz` are gzip-compressed.

use crate::binary::{write_f32_be, write_i16_be, write_i32_be};
use crate::spatial::ras_center_from_geometry;
use crate::{is_gzip_path, MRI_FLOAT, PADDING_LEN, VERSION};
use anyhow::{anyhow, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_image::native::Image;
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
    write_i32_be(writer, 1)?;
    write_i32_be(writer, MRI_FLOAT)?;
    write_i32_be(writer, 0)?;
    write_i16_be(writer, 1)?;

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

/// Stateless writer for MGH / MGZ files.
pub struct MghWriter<B: ComputeBackend> {
    backend: B }

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
}
