//! MGH / MGZ writer for 3-D volumetric images.
//!
//! The writer emits FreeSurfer MGH with `MRI_FLOAT` voxel data. Paths ending
//! in `.mgz` or `.mgh.gz` are gzip-compressed.

use crate::binary::{write_f32_be, write_i16_be, write_i32_be};
use crate::spatial::ras_center_from_geometry;
use crate::{is_gzip_path, MRI_FLOAT, PADDING_LEN, VERSION};
use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufWriter, Write};
use std::path::Path;

#[cfg(test)]
mod tests;

/// Write a 3-D `Image` as an MGH or MGZ file.
pub fn write_mgh<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
    let f32_vec = image.try_data_vec()?;
    write_mgh_stream(
        path.as_ref(),
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &f32_vec,
    )
}

/// Substrate-agnostic MGH file entry: creates the file, applies the gzip
/// branch, and delegates to [`write_mgh_flat`]. The shared SSOT the Burn and
/// Atlas-native writers both wrap.
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
    write_i32_be(writer, nx as i32)?;
    write_i32_be(writer, ny as i32)?;
    write_i32_be(writer, nz as i32)?;
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

    let n_voxels = nx * ny * nz;
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

    let mut data_buf = vec![0u8; n_voxels * 4];
    for (i, &value) in f32_slice.iter().enumerate() {
        let offset = i * 4;
        data_buf[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    }
    writer
        .write_all(&data_buf)
        .context("Failed to write MGH voxel data")?;

    Ok(())
}

/// Stateless writer for MGH / MGZ files.
pub struct MghWriter;

impl MghWriter {
    /// Write `image` to the MGH or MGZ file at `path`.
    pub fn write<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
        write_mgh(image, path)
    }
}

/// Atlas-native-substrate MGH writers (plain end-state names, disambiguated
/// from the Burn functions by module path only; folds away when the Burn path
/// is deleted — ADR 0002 A1).
pub mod native {
    use super::write_mgh_stream;
    use anyhow::Result;
    use std::path::Path;

    /// Write an Atlas-native 3-D image as an MGH or MGZ file.
    ///
    /// Host data is extracted layout-independently via `data_cow_on`, then
    /// serialized through the same [`write_mgh_stream`](super::write_mgh_stream)
    /// core as the Burn [`write_mgh`](super::write_mgh) — byte-identical output
    /// for the same logical image.
    pub fn write_mgh<B, P>(
        image: &ritk_image::native::Image<f32, B, 3>,
        path: P,
        backend: &B,
    ) -> Result<()>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
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
}
