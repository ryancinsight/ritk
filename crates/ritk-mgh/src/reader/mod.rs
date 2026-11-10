//! MGH / MGZ reader for 3-D volumetric images and acquisition series.
//!
//! Voxels are stored in MGH Fortran order `x + y*nx + z*nx*ny`, which is
//! identical to RITK row-major `[z, y, x]` order. No axis permutation is
//! required when constructing the tensor. Each frame is a contiguous block
//! of `nx * ny * nz` voxels; the `nframes` header field counts consecutive
//! volumes of identical geometry.

use crate::spatial::{ImageGeometry, MghRasBlock, RasValidity};
use crate::types::VoxelType;
use crate::{is_gzip_path, MghError, GOOD_RAS_VALID, PADDING_LEN, VERSION};
use coeus_core::ComputeBackend;
use consus_core::{read_from, ByteOrder, EndianScalar};
use flate2::read::GzDecoder;
use ritk_codecs::{ImageReadBudget, ImageReadResource};
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
/// The file size, the frame count, and the `f32` voxel storage the header
/// implies are checked against `budget` before any voxel is decoded.
///
/// # Errors
///
/// Returns [`MghError::Budget`] when a declared size exceeds `budget`, and a
/// header variant when the version, dimensions, or data type code are invalid.
/// Returns [`MghError::TruncatedPayload`] when the payload is shorter than the
/// header declares, and [`MghError::MultiFrame`] when the file declares more
/// than one frame — a multi-frame volume has no correct single-volume
/// decoding, so it fails rather than silently yielding frame 0.
pub fn read_mgh<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: ImageReadBudget,
) -> Result<Image<f32, B, 3>, MghError> {
    let mut stream = open_mgh_stream(path.as_ref(), budget)?;
    read_mgh_from_reader(&mut stream, backend, budget)
}

/// Raw or gunzipped byte stream of one MGH / MGZ file.
enum MghStream {
    Plain(BufReader<std::fs::File>),
    Gzip(Box<BufReader<GzDecoder<BufReader<std::fs::File>>>>),
}

impl Read for MghStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(reader) => reader.read(buf),
            Self::Gzip(reader) => reader.read(buf),
        }
    }
}

/// Opens `path`, enforces the encoded-size budget, and selects gzip
/// decompression for `.mgz` and `.mgh.gz` paths.
///
/// Decompression needs no separate cap: the header's voxel count passes the
/// decoded-byte budget before any voxel is read, and the decoder consumes
/// exactly the voxels it declares.
fn open_mgh_stream(path: &Path, budget: ImageReadBudget) -> Result<MghStream, MghError> {
    let open_error = |source| MghError::Open {
        path: path.to_path_buf(),
        source,
    };
    let file = std::fs::File::open(path).map_err(open_error)?;
    let encoded_bytes = file.metadata().map_err(open_error)?.len();
    budget.check(ImageReadResource::EncodedBytes, encoded_bytes)?;

    Ok(if is_gzip_path(path) {
        MghStream::Gzip(Box::new(BufReader::new(GzDecoder::new(BufReader::new(
            file,
        )))))
    } else {
        MghStream::Plain(BufReader::new(file))
    })
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

struct MghHeader {
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
    voxel_type: VoxelType,
    voxels_per_frame: usize,
    nframes: usize,
}

fn read_mgh_from_reader<B: ComputeBackend, R: Read>(
    reader: &mut R,
    backend: &B,
    budget: ImageReadBudget,
) -> Result<Image<f32, B, 3>, MghError> {
    let header = read_mgh_header(reader, budget)?;
    if header.nframes != 1 {
        return Err(MghError::MultiFrame {
            nframes: header.nframes,
        });
    }
    let DecodedMgh {
        mut volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mgh_payload(reader, header)?;
    let data = volumes
        .pop()
        .expect("invariant: single-frame header produces one decoded volume");
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
        .map_err(|source| MghError::Image { source })
}

/// Read an MGH or MGZ acquisition series as one image per frame.
///
/// Each returned image shares the file's single spatial grid, in acquisition
/// order. A single-frame file is a one-image series, so this reader accepts
/// an ordinary volume; [`read_mgh`] does not accept the converse, rejecting a
/// multi-frame file rather than returning its first frame.
///
/// The file size, the header's `nframes` against
/// [`ImageReadBudget::max_series_volumes`], and the `f32` voxel storage of all
/// frames against [`ImageReadBudget::max_decoded_bytes`] are checked before any
/// voxel is decoded.
///
/// # Errors
///
/// Returns [`MghError::Budget`] when a declared size exceeds `budget`, a header
/// variant when the version, dimensions, frame count, or data type code are
/// invalid, and [`MghError::TruncatedPayload`] when the payload is shorter than
/// the header declares.
pub fn read_mgh_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: ImageReadBudget,
) -> Result<Vec<Image<f32, B, 3>>, MghError> {
    let mut stream = open_mgh_stream(path.as_ref(), budget)?;
    let DecodedMgh {
        volumes,
        dims,
        origin,
        spacing,
        direction,
    } = decode_mgh(&mut stream, budget)?;

    volumes
        .into_iter()
        .map(|data| {
            Image::from_flat_on(data, dims, origin, spacing, direction, backend)
                .map_err(|source| MghError::Image { source })
        })
        .collect()
}

fn read_be<T: EndianScalar, R: Read>(reader: &mut R) -> Result<T, MghError> {
    read_from::<T, _>(reader, ByteOrder::BigEndian).map_err(|source| MghError::Header { source })
}

fn read_mgh_header<R: Read>(
    reader: &mut R,
    budget: ImageReadBudget,
) -> Result<MghHeader, MghError> {
    let version = read_be::<i32, _>(reader)?;
    if version != VERSION {
        return Err(MghError::InvalidVersion {
            expected: VERSION,
            found: version,
        });
    }

    let width = read_be::<i32, _>(reader)?;
    let height = read_be::<i32, _>(reader)?;
    let depth = read_be::<i32, _>(reader)?;
    let declared_frames = read_be::<i32, _>(reader)?;
    let mri_type = read_be::<i32, _>(reader)?;
    let _dof = read_be::<i32, _>(reader)?;

    let positive = |value: i32| usize::try_from(value).ok().filter(|&extent| extent > 0);
    let (Some(nx), Some(ny), Some(nz)) = (positive(width), positive(height), positive(depth))
    else {
        return Err(MghError::InvalidDimensions {
            width,
            height,
            depth,
        });
    };
    let Some(nframes) = positive(declared_frames) else {
        return Err(MghError::InvalidFrameCount {
            nframes: declared_frames,
        });
    };

    let good_ras_flag = read_be::<i16, _>(reader)?;
    let spacing_xyz = [
        read_be::<f32, _>(reader)?,
        read_be::<f32, _>(reader)?,
        read_be::<f32, _>(reader)?,
    ];
    let direction_columns = read_direction_columns(reader)?;
    let c_ras = [
        read_be::<f32, _>(reader)?,
        read_be::<f32, _>(reader)?,
        read_be::<f32, _>(reader)?,
    ];

    let mut padding = [0u8; PADDING_LEN];
    reader
        .read_exact(&mut padding)
        .map_err(|source| MghError::Header { source })?;

    let ImageGeometry {
        spacing,
        direction,
        origin,
    } = MghRasBlock::new(spacing_xyz, direction_columns, c_ras)
        .into_image_geometry(
            if good_ras_flag == GOOD_RAS_VALID {
                RasValidity::Valid
            } else {
                RasValidity::Synthetic
            },
            [nx, ny, nz],
        )
        .map_err(|source| MghError::InvalidSpacing { source })?;

    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or(MghError::DimensionOverflow { nx, ny, nz })?;
    let total_voxels = n_voxels
        .checked_mul(nframes)
        .ok_or(MghError::SeriesOverflow {
            voxels: n_voxels,
            frames: nframes,
        })?;
    let voxel_type = VoxelType::try_from(mri_type)?;

    // The header's counts are claims about the file, so they are bounded
    // before any frame is decoded. Decoded storage is the `f32` output, which
    // is never smaller than the stored voxels.
    budget.check_len(ImageReadResource::SeriesVolumes, nframes)?;
    budget.check_len(
        ImageReadResource::DecodedBytes,
        total_voxels.saturating_mul(size_of::<f32>()),
    )?;

    Ok(MghHeader {
        dims: [nz, ny, nx],
        origin,
        spacing,
        direction,
        voxel_type,
        voxels_per_frame: n_voxels,
        nframes,
    })
}

fn decode_mgh<R: Read>(reader: &mut R, budget: ImageReadBudget) -> Result<DecodedMgh, MghError> {
    let header = read_mgh_header(reader, budget)?;
    decode_mgh_payload(reader, header)
}

fn decode_mgh_payload<R: Read>(reader: &mut R, header: MghHeader) -> Result<DecodedMgh, MghError> {
    let volumes = voxel_decode::decode_volumes(
        reader,
        header.voxel_type,
        header.voxels_per_frame,
        header.nframes,
    )?;
    Ok(DecodedMgh {
        volumes,
        dims: header.dims,
        origin: header.origin,
        spacing: header.spacing,
        direction: header.direction,
    })
}

fn read_direction_columns<R: Read>(reader: &mut R) -> Result<[[f32; 3]; 3], MghError> {
    let mut columns = [[0.0f32; 3]; 3];
    for column in &mut columns {
        for value in column {
            *value = read_be(reader)?;
        }
    }
    Ok(columns)
}

/// Stateless reader for MGH / MGZ files.
pub struct MghReader;

impl MghReader {
    /// Read an MGH or MGZ file into a 3-D `Image`.
    ///
    /// # Errors
    ///
    /// See [`read_mgh`].
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        path: P,
        backend: &B,
        budget: ImageReadBudget,
    ) -> Result<Image<f32, B, 3>, MghError> {
        read_mgh(path, backend, budget)
    }

    /// Read an MGH or MGZ acquisition series as one image per frame.
    ///
    /// # Errors
    ///
    /// See [`read_mgh_series`].
    pub fn read_series<B: ComputeBackend, P: AsRef<Path>>(
        path: P,
        backend: &B,
        budget: ImageReadBudget,
    ) -> Result<Vec<Image<f32, B, 3>>, MghError> {
        read_mgh_series(path, backend, budget)
    }
}
