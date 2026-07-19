//! MGH / MGZ reader for 3-D volumetric images.
//!
//! Voxels are stored in MGH Fortran order `x + y*nx + z*nx*ny`, which is
//! identical to RITK row-major `[z, y, x]` order. No axis permutation is
//! required when constructing the tensor.

use crate::binary::{read_f32_be, read_i16_be, read_i32_be};
use crate::spatial::{derive_image_geometry, RasValidity};
use crate::types::bytes_per_voxel;
use crate::{is_gzip_path, MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR, PADDING_LEN, VERSION};
use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use flate2::read::GzDecoder;
use ritk_image::Image;
use std::io::{BufReader, Read};
use std::path::Path;

#[cfg(test)]
mod tests;

/// Read an MGH or MGZ file into a 3-D `Image`.
///
/// Files ending in `.mgz` or `.mgh.gz` are decompressed with gzip before
/// parsing. All other paths are treated as uncompressed MGH.
pub fn read_mgh<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let gz = GzDecoder::new(BufReader::new(file));
        let mut reader = BufReader::new(gz);
        read_mgh_from_reader(&mut reader, backend)
            .with_context(|| format!("Failed to parse MGZ file {:?}", path))
    } else {
        let mut reader = BufReader::new(file);
        read_mgh_from_reader(&mut reader, backend)
            .with_context(|| format!("Failed to parse MGH file {:?}", path))
    }
}

/// Decoded MGH volume: voxels in `[nz, ny, nx]` order plus physical geometry.
struct DecodedMgh {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

fn read_mgh_from_reader<B: ComputeBackend, R: Read>(
    reader: &mut R,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let DecodedMgh {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mgh(reader)?;
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

fn decode_mgh<R: Read>(reader: &mut R) -> Result<DecodedMgh> {
    let version = read_i32_be(reader)?;
    if version != VERSION {
        bail!(
            "Invalid MGH version: expected {}, found {}",
            VERSION,
            version
        );
    }

    let width = read_i32_be(reader)?;
    let height = read_i32_be(reader)?;
    let depth = read_i32_be(reader)?;
    let nframes = read_i32_be(reader)?;
    let mri_type = read_i32_be(reader)?;
    let _dof = read_i32_be(reader)?;

    if width <= 0 || height <= 0 || depth <= 0 {
        bail!(
            "Invalid MGH dimensions: width={}, height={}, depth={}",
            width,
            height,
            depth
        );
    }
    if nframes <= 0 {
        bail!("Invalid MGH nframes: {}", nframes);
    }
    if nframes > 1 {
        tracing::warn!(
            nframes,
            "MGH file contains multiple frames; only frame 0 will be loaded"
        );
    }

    let good_ras_flag = read_i16_be(reader)?;
    let spacing_xyz = [
        read_f32_be(reader)?,
        read_f32_be(reader)?,
        read_f32_be(reader)?,
    ];
    let direction_columns = read_direction_columns(reader)?;
    let c_ras = [
        read_f32_be(reader)?,
        read_f32_be(reader)?,
        read_f32_be(reader)?,
    ];

    let mut padding = [0u8; PADDING_LEN];
    reader
        .read_exact(&mut padding)
        .context("Failed to read MGH header padding")?;

    let nx = width as usize;
    let ny = height as usize;
    let nz = depth as usize;
    let (spacing, direction, origin) = derive_image_geometry(
        if good_ras_flag == 1 {
            RasValidity::Valid
        } else {
            RasValidity::Synthetic
        },
        [nx, ny, nz],
        spacing_xyz,
        direction_columns,
        c_ras,
    );

    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| anyhow::anyhow!("Volume dimensions overflow: {}x{}x{}", nx, ny, nz))?;
    let bpv = bytes_per_voxel(mri_type)?;
    let data_size = n_voxels.checked_mul(bpv).ok_or_else(|| {
        anyhow::anyhow!("Data size overflow: {} voxels Ã— {} bytes", n_voxels, bpv)
    })?;

    // Bound the speculative allocation: `data_size` derives from the header
    // dimensions and may exceed the bytes actually present. `read_exact_bounded`
    // grows the buffer per confirmed chunk and reports truncation rather than
    // reserving the full claimed size (out-of-memory abort on a hostile header).
    let raw = consus_io::read_exact_bounded(reader, data_size)
        .context("Failed to read MGH voxel data")?;
    let f32_data = decode_voxels(mri_type, &raw);

    Ok(DecodedMgh {
        data: f32_data,
        dims: [nz, ny, nx],
        origin,
        spacing,
        direction,
    })
}

fn read_direction_columns<R: Read>(reader: &mut R) -> Result<[[f32; 3]; 3]> {
    let mut columns = [[0.0f32; 3]; 3];
    for column in &mut columns {
        for value in column {
            *value = read_f32_be(reader)?;
        }
    }
    Ok(columns)
}

fn decode_voxels(mri_type: i32, raw: &[u8]) -> Vec<f32> {
    match mri_type {
        MRI_UCHAR => raw.iter().map(|&b| b as f32).collect(),
        MRI_SHORT => raw
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]) as f32)
            .collect(),
        MRI_INT => raw
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        MRI_FLOAT => raw
            .chunks_exact(4)
            .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => unreachable!("bytes_per_voxel validates the type code"),
    }
}

/// Stateless reader for MGH / MGZ files.
pub struct MghReader;

impl MghReader {
    /// Read an MGH or MGZ file into a 3-D `Image`.
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        path: P,
        backend: &B,
    ) -> Result<Image<f32, B, 3>> {
        read_mgh(path, backend)
    }
}
