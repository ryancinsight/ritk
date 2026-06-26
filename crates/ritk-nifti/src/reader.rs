use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use flate2::read::GzDecoder;
use ritk_core::image::Image;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::header::{NiftiDatatype, NiftiHeader};
use crate::shape::checked_voxel_count;
use crate::spatial::metadata_from_nifti_ras_affine;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

pub fn read_nifti<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        tracing::error!("Failed to read NIfTI file {:?}: {}", path.as_ref(), e);
        anyhow!("Failed to read NIfTI file")
    })?;

    read_nifti_from_bytes(&bytes, device).map_err(|e| {
        tracing::error!("Failed to decode NIfTI file: {e:#}");
        if format!("{e:#}").contains("Invalid NIfTI spatial metadata") {
            anyhow!("Invalid NIfTI spatial metadata")
        } else {
            anyhow!("Failed to read NIfTI file")
        }
    })
}

/// Read a NIfTI payload from in-memory bytes.
///
/// Accepts `.nii` bytes directly and `.nii.gz` bytes by detecting the gzip
/// header. The decoded payload must be a single-file NIfTI-1 stream.
pub fn read_nifti_from_bytes<B: Backend>(bytes: &[u8], device: &B::Device) -> Result<Image<B, 3>> {
    let decoded;
    let payload = if bytes.starts_with(&GZIP_MAGIC) {
        decoded = decode_gzip(bytes).context("Failed to decode gzipped NIfTI bytes")?;
        decoded.as_slice()
    } else {
        bytes
    };

    image_from_single_file_bytes(payload, device)
}

fn image_from_single_file_bytes<B: Backend>(
    bytes: &[u8],
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let header = NiftiHeader::parse(bytes).context("Invalid NIfTI header")?;
    if header.datatype != NiftiDatatype::Float32 {
        anyhow::bail!(
            "read_nifti requires Float32 datatype, got {}",
            header.datatype.code()
        );
    }

    let spatial = metadata_from_nifti_ras_affine(header.affine()?)
        .context("Invalid NIfTI spatial metadata")?;
    let [nx, ny, nz] = dims_xyz(&header)?;
    let voxel_count = checked_voxel_count(nx, ny, nz)?;
    let range = header.volume_byte_range(bytes.len())?;
    let data_bytes = &bytes[range];

    let mut data_vec = vec![0.0_f32; voxel_count];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let file_index = x + nx * (y + ny * z);
                let offset = file_index * 4;
                let value = f32::from_le_bytes(
                    data_bytes[offset..offset + 4]
                        .try_into()
                        .expect("invariant: checked volume byte range gives complete f32 lanes"),
                );
                data_vec[z * ny * nx + y * nx + x] = value;
            }
        }
    }

    let shape_burn = Shape::new([nz, ny, nx]);
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data_vec, shape_burn), device);

    Ok(Image::new(
        tensor,
        spatial.origin,
        spatial.spacing,
        spatial.direction,
    ))
}

/// Read a NIfTI file as an integer label map in ZYX order.
///
/// # Label extraction
///
/// Float32 volumes convert with `max(0.0).round() as u32`; UInt32 volumes are
/// copied exactly. The returned shape is `[nz, ny, nx]`.
pub fn read_nifti_labels<P: AsRef<Path>>(path: P) -> Result<(Vec<u32>, [usize; 3])> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        tracing::error!("Failed to read NIfTI label file: {}", e);
        anyhow!("Failed to read NIfTI label file")
    })?;
    read_nifti_labels_from_bytes(&bytes).map_err(|e| {
        tracing::error!("Failed to decode NIfTI label file: {e:#}");
        anyhow!("Failed to read NIfTI label file")
    })
}

fn read_nifti_labels_from_bytes(bytes: &[u8]) -> Result<(Vec<u32>, [usize; 3])> {
    let decoded;
    let payload = if bytes.starts_with(&GZIP_MAGIC) {
        decoded = decode_gzip(bytes).context("Failed to decode gzipped NIfTI label bytes")?;
        decoded.as_slice()
    } else {
        bytes
    };

    let header = NiftiHeader::parse(payload).context("Invalid NIfTI label header")?;
    let [nx, ny, nz] = dims_xyz(&header)?;
    let voxel_count = checked_voxel_count(nx, ny, nz)?;
    let range = header.volume_byte_range(payload.len())?;
    let data_bytes = &payload[range];
    let mut labels = vec![0_u32; voxel_count];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let file_index = x + nx * (y + ny * z);
                let offset = file_index * 4;
                let raw = data_bytes[offset..offset + 4]
                    .try_into()
                    .expect("invariant: checked volume byte range gives complete 4-byte lanes");
                labels[z * ny * nx + y * nx + x] = match header.datatype {
                    NiftiDatatype::Float32 => f32::from_le_bytes(raw).max(0.0).round() as u32,
                    NiftiDatatype::Uint32 => u32::from_le_bytes(raw),
                };
            }
        }
    }

    Ok((labels, [nz, ny, nx]))
}

fn dims_xyz(header: &NiftiHeader) -> Result<[usize; 3]> {
    Ok([
        usize::from(header.dim[1]),
        usize::from(header.dim[2]),
        usize::from(header.dim[3]),
    ])
}

fn decode_gzip(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(bytes);
    let mut decoded = Vec::new();
    decoder.read_to_end(&mut decoded)?;
    Ok(decoded)
}
