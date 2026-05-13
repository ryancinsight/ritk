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
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

#[cfg(test)]
mod tests;

/// Write a 3-D `Image` as an MGH or MGZ file.
pub fn write_mgh<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_mgh_to_writer(image, &mut encoder)?;
        encoder.finish().context("Failed to finalize gzip stream")?;
    } else {
        let mut writer = BufWriter::new(file);
        write_mgh_to_writer(image, &mut writer)?;
        writer.flush().context("Failed to flush MGH output")?;
    }
    Ok(())
}

fn write_mgh_to_writer<B: Backend, W: Write>(image: &Image<B, 3>, writer: &mut W) -> Result<()> {
    let [nz, ny, nx] = image.shape();

    write_i32_be(writer, VERSION)?;
    write_i32_be(writer, nx as i32)?;
    write_i32_be(writer, ny as i32)?;
    write_i32_be(writer, nz as i32)?;
    write_i32_be(writer, 1)?;
    write_i32_be(writer, MRI_FLOAT)?;
    write_i32_be(writer, 0)?;
    write_i16_be(writer, 1)?;

    let spacing = *image.spacing();
    for axis in 0..3 {
        write_f32_be(writer, spacing[axis] as f32)?;
    }

    let direction = *image.direction();
    for col in 0..3 {
        for row in 0..3 {
            write_f32_be(writer, direction[(row, col)] as f32)?;
        }
    }

    let c_ras = ras_center_from_geometry(*image.origin(), spacing, direction, [nz, ny, nx]);
    write_f32_be(writer, c_ras[0] as f32)?;
    write_f32_be(writer, c_ras[1] as f32)?;
    write_f32_be(writer, c_ras[2] as f32)?;

    writer
        .write_all(&[0u8; PADDING_LEN])
        .context("Failed to write MGH header padding")?;

    let tensor_data = image.data().clone().to_data();
    let f32_slice = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow!("Failed to extract f32 slice from tensor: {:?}", e))?;
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
