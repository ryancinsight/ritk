//! MGH / MGZ reader for 3-D volumetric images and acquisition series.
//!
//! Voxels are stored in MGH Fortran order `x + y*nx + z*nx*ny`, which is
//! identical to RITK row-major `[z, y, x]` order. No axis permutation is
//! required when constructing the tensor. Each frame is a contiguous block
//! of `nx * ny * nz` voxels; the `nframes` header field counts consecutive
//! volumes of identical geometry.

use crate::binary::{read_f32_be, read_i16_be, read_i32_be};
use crate::spatial::{derive_image_geometry, RasValidity};
use crate::types::VoxelType;
use crate::{is_gzip_path, GOOD_RAS_VALID, PADDING_LEN, VERSION};
use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use flate2::read::GzDecoder;
use ritk_image::Image;
use std::io::{BufReader, Read};
use std::path::Path;

#[cfg(test)]
mod tests;
mod voxel_decode;

/// Read an MGH or MGZ file into a 3-D `Image`.
///
/// Files ending in `.mgz` or `.mgh.gz` are decompressed with gzip before
/// parsing. All other paths are treated as uncompressed MGH.
///
/// # Errors
///
/// Returns an error when the header version, dimensions, or data type code are
/// invalid, when the payload is shorter than the header declares, or when the
/// file declares more than one frame — a multi-frame volume has no correct
/// single-volume decoding, so it fails rather than silently yielding frame 0.
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

/// Decoded MGH volume(s): one entry per frame, each in `[nz, ny, nx]` order,
/// sharing one physical geometry.
struct DecodedMgh {
    volumes: Vec<Vec<f32>>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

impl DecodedMgh {
    /// Take the sole volume, rejecting a multi-frame file.
    ///
    /// The single-volume reader carries a `[nz, ny, nx]` contract, so a
    /// multi-frame file has no correct representation through it; returning
    /// frame 0 would discard the rest of the acquisition while reporting
    /// success.
    fn into_single_volume(
        mut self,
    ) -> Result<(
        Vec<f32>,
        [usize; 3],
        ritk_spatial::Point<3>,
        ritk_spatial::Spacing<3>,
        ritk_spatial::Direction<3>,
    )> {
        if self.volumes.len() != 1 {
            bail!(
                "MGH file declares {} frames; this reader returns a 3-D Image, which \
                 represents exactly one frame. Multi-frame MGH (diffusion series, \
                 time series) has no correct single-volume decoding — reading frame 0 \
                 alone would silently discard {} frames of the acquisition.",
                self.volumes.len(),
                self.volumes.len() - 1
            );
        }
        let data = self
            .volumes
            .pop()
            .expect("invariant: length checked to be exactly one above");
        Ok((data, self.dims, self.origin, self.spacing, self.direction))
    }
}

fn read_mgh_from_reader<B: ComputeBackend, R: Read>(
    reader: &mut R,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let (data, dims, origin, spacing, direction) = decode_mgh(reader)?.into_single_volume()?;
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Read an MGH or MGZ acquisition series as one image per frame.
///
/// Each returned image shares the file's single spatial grid, in acquisition
/// order. A single-frame file is a one-image series, so this reader accepts
/// an ordinary volume; [`read_mgh`] does not accept the converse, rejecting a
/// multi-frame file rather than returning its first frame.
///
/// # Errors
///
/// Returns an error when the header version, dimensions, or data type code are
/// invalid, when the payload is shorter than the header declares, or when the
/// file contains zero frames.
pub fn read_mgh_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let gz = GzDecoder::new(BufReader::new(file));
        let mut reader = BufReader::new(gz);
        read_mgh_series_from_reader(&mut reader, backend)
            .with_context(|| format!("Failed to parse MGZ series {:?}", path))
    } else {
        let mut reader = BufReader::new(file);
        read_mgh_series_from_reader(&mut reader, backend)
            .with_context(|| format!("Failed to parse MGH series {:?}", path))
    }
}

fn read_mgh_series_from_reader<B: ComputeBackend, R: Read>(
    reader: &mut R,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let DecodedMgh {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mgh(reader)?;

    volumes
        .into_iter()
        .map(|data| Image::from_flat_on(data, dims, origin, spacing, direction, backend))
        .collect()
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
        if good_ras_flag == GOOD_RAS_VALID {
            RasValidity::Valid
        } else {
            RasValidity::Synthetic
        },
        [nx, ny, nz],
        spacing_xyz,
        direction_columns,
        c_ras,
    );

    let nframes = nframes as usize;
    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| anyhow::anyhow!("Volume dimensions overflow: {}x{}x{}", nx, ny, nz))?;
    let total_voxels = n_voxels.checked_mul(nframes).ok_or_else(|| {
        anyhow::anyhow!("Series voxel count overflow: {n_voxels} voxels × {nframes} frames")
    })?;
    let voxel_type = VoxelType::try_from(mri_type)?;
    let bpv = voxel_type.bytes_per_voxel();
    let data_size = total_voxels.checked_mul(bpv).ok_or_else(|| {
        anyhow::anyhow!("Data size overflow: {total_voxels} voxels × {bpv} bytes")
    })?;

    let volumes = voxel_decode::decode_volumes(reader, voxel_type, n_voxels, nframes)
        .with_context(|| format!("Failed to decode {data_size} bytes of MGH voxel data"))?;

    Ok(DecodedMgh {
        volumes,
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

    /// Read an MGH or MGZ acquisition series as one image per frame.
    pub fn read_series<B: ComputeBackend, P: AsRef<Path>>(
        path: P,
        backend: &B,
    ) -> Result<Vec<Image<f32, B, 3>>> {
        read_mgh_series(path, backend)
    }
}
