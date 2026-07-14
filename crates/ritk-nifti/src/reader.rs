use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use flate2::read::GzDecoder;
use ritk_image::native::Image;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::header::NiftiHeader;
use crate::shape::checked_voxel_count;
use crate::spatial::{metadata_from_nifti_ras_affine, InternalSpatialMetadata};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const MAX_HEADER_PREFIX_BYTES: u64 = 544;

pub fn read_nifti<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        tracing::error!("Failed to read NIfTI file {:?}: {}", path.as_ref(), e);
        anyhow!("Failed to read NIfTI file")
    })?;

    read_nifti_from_bytes(&bytes, backend).map_err(|e| {
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
pub fn read_nifti_from_bytes<B: ComputeBackend>(
    bytes: &[u8],
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let DecodedNifti {
        data,
        dims,
        spatial,
    } = decode_nifti_bytes(bytes)?;
    Image::from_flat_on(
        data,
        dims,
        spatial.origin,
        spatial.spacing,
        spatial.direction,
        backend,
    )
}

/// Decoded NIfTI volume: voxels in `[nz, ny, nx]` order plus physical metadata.
struct DecodedNifti {
    data: Vec<f32>,
    dims: [usize; 3],
    spatial: InternalSpatialMetadata,
}

/// Decode NIfTI bytes (gzip-detected) into a backend-agnostic [`DecodedNifti`].
fn decode_nifti_bytes(bytes: &[u8]) -> Result<DecodedNifti> {
    let decoded;
    let payload = if bytes.starts_with(&GZIP_MAGIC) {
        decoded = decode_gzip(bytes).context("Failed to decode gzipped NIfTI bytes")?;
        decoded.as_slice()
    } else {
        bytes
    };

    decode_single_file(payload)
}

fn decode_single_file(bytes: &[u8]) -> Result<DecodedNifti> {
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

    Ok(DecodedNifti {
        data: data_vec,
        dims: [nz, ny, nx],
        spatial,
    })
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
    Read::by_ref(&mut decoder)
        .take(MAX_HEADER_PREFIX_BYTES)
        .read_to_end(&mut decoded)?;

    let header = NiftiHeader::parse(&decoded).context("Invalid compressed NIfTI header")?;
    let declared_end = header.volume_byte_range(usize::MAX)?.end;
    let read_limit = declared_end
        .checked_add(1)
        .ok_or_else(|| anyhow!("Compressed NIfTI read limit overflows usize"))?;
    let remaining = read_limit.saturating_sub(decoded.len());
    decoder
        .take(u64::try_from(remaining).context("Compressed NIfTI read limit exceeds u64")?)
        .read_to_end(&mut decoded)?;
    Ok(decoded)
}
