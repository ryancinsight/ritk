use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use flate2::read::GzDecoder;
use ritk_image::Image;
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
    let (data, dims, spatial) = decode_nifti_bytes(bytes)?.into_single_volume()?;
    Image::from_flat_on(
        data,
        dims,
        spatial.origin,
        spatial.spacing,
        spatial.direction,
        backend,
    )
}

/// Read a NIfTI acquisition series as one image per volume.
///
/// A NIfTI file carries its acquisition axis in `dim[4]` — the axis diffusion,
/// functional, and other repeated acquisitions vary along. Every returned image
/// shares the file's single spatial grid, in acquisition order.
///
/// A rank-3 file is a one-volume series, so this reader accepts an ordinary
/// volume and returns it as a single-element series. The inverse is not true:
/// [`read_nifti`] rejects a multi-volume file rather than returning its first
/// volume.
///
/// # Errors
///
/// Returns an error when the header, spatial metadata, or payload length is
/// invalid, or when a voxel lane cannot be decoded.
pub fn read_nifti_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        tracing::error!(
            "Failed to read NIfTI series file {:?}: {}",
            path.as_ref(),
            e
        );
        anyhow!("Failed to read NIfTI series file")
    })?;

    read_nifti_series_from_bytes(&bytes, backend).map_err(|e| {
        tracing::error!("Failed to decode NIfTI series file: {e:#}");
        e.context("Failed to read NIfTI series file")
    })
}

/// Read a NIfTI acquisition series from in-memory bytes.
///
/// The byte-level counterpart of [`read_nifti_series`], accepting `.nii` bytes
/// directly and `.nii.gz` bytes by detecting the gzip header.
///
/// # Errors
///
/// Returns an error when the header, spatial metadata, or payload length is
/// invalid, or when a voxel lane cannot be decoded.
pub fn read_nifti_series_from_bytes<B: ComputeBackend>(
    bytes: &[u8],
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let DecodedNifti {
        volumes,
        dims,
        spatial,
    } = decode_nifti_bytes(bytes)?;

    volumes
        .into_iter()
        .map(|data| {
            Image::from_flat_on(
                data,
                dims,
                spatial.origin,
                spatial.spacing,
                spatial.direction,
                backend,
            )
        })
        .collect()
}

/// Decoded NIfTI payload: one entry per volume, each in `[nz, ny, nx]` order,
/// sharing one spatial grid.
struct DecodedNifti {
    volumes: Vec<Vec<f32>>,
    dims: [usize; 3],
    spatial: InternalSpatialMetadata,
}

impl DecodedNifti {
    /// Take the sole volume, rejecting a series.
    ///
    /// The single-volume readers carry a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through them; returning volume 0 would
    /// discard the rest of the acquisition while reporting success.
    fn into_single_volume(mut self) -> Result<(Vec<f32>, [usize; 3], InternalSpatialMetadata)> {
        if self.volumes.len() != 1 {
            bail!(
                "NIfTI file declares {} volumes; this reader returns one 3-D volume. \
                 Use the series reader to decode an acquisition series (diffusion, \
                 time series) without discarding {} of its volumes.",
                self.volumes.len(),
                self.volumes.len() - 1
            );
        }
        let data = self
            .volumes
            .pop()
            .expect("invariant: length checked to be exactly one above");
        Ok((data, self.dims, self.spatial))
    }
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

    // NIfTI stores x fastest, then y, z, and finally the acquisition axis, so
    // each volume is one contiguous block of `voxel_count` voxels.
    let volumes = (0..header.volume_count())
        .map(|volume| {
            let base = volume * voxel_count * lane_width;
            let mut data_vec = vec![0.0_f32; voxel_count];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let file_index = x + nx * (y + ny * z);
                        let offset = base + file_index * lane_width;
                        let value =
                            header.read_f32_voxel(&data_bytes[offset..offset + lane_width])?;
                        data_vec[z * ny * nx + y * nx + x] = value;
                    }
                }
            }
            Ok(data_vec)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DecodedNifti {
        volumes,
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
    if header.volume_count() != 1 {
        // The label contract is one `[nz, ny, nx]` map. Decoding only the first
        // volume of a series would report success over a discarded acquisition.
        bail!(
            "NIfTI label file declares {} volumes; a label map is a single volume",
            header.volume_count()
        );
    }
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
