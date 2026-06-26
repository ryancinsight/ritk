use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use flate2::read::GzDecoder;
use ritk_core::image::Image;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::header::NiftiHeader;
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
            e.context("Invalid NIfTI spatial metadata")
        } else {
            e.context("Failed to read NIfTI file")
        }
    })
}

/// Read a NIfTI payload from in-memory bytes.
///
/// Accepts `.nii` bytes directly and `.nii.gz` bytes by detecting the gzip
/// header. The decoded payload must be a single-file NIfTI-1 or NIfTI-2 stream.
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
    let spatial = metadata_from_nifti_ras_affine(header.affine()?)
        .context("Invalid NIfTI spatial metadata")?;
    let [nx, ny, nz] = dims_xyz(&header)?;
    let voxel_count = checked_voxel_count(nx, ny, nz)?;
    let range = header.volume_byte_range(bytes.len())?;
    let data_bytes = &bytes[range];
    let lane_width = header.datatype.byte_width();

    let mut data_vec = vec![0.0_f32; voxel_count];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let file_index = x + nx * (y + ny * z);
                let offset = file_index * lane_width;
                let value = header.read_f32_voxel(&data_bytes[offset..offset + lane_width])?;
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
        e.context("Failed to read NIfTI label file")
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
    let lane_width = header.datatype.byte_width();
    let mut labels = vec![0_u32; voxel_count];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let file_index = x + nx * (y + ny * z);
                let offset = file_index * lane_width;
                labels[z * ny * nx + y * nx + x] =
                    header.read_label_voxel(&data_bytes[offset..offset + lane_width])?;
            }
        }
    }

    Ok((labels, [nz, ny, nx]))
}

fn dims_xyz(header: &NiftiHeader) -> Result<[usize; 3]> {
    Ok([header.dim[1], header.dim[2], header.dim[3]])
}

fn decode_gzip(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(bytes);
    let mut decoded = Vec::new();
    decoder.read_to_end(&mut decoded)?;
    Ok(decoded)
}
