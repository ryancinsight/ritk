use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_codecs::{ByteOrder, SampleType};
use ritk_image::Image;
use ritk_image_io::{ImageReadBudget, SeriesAxis};
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

use crate::axes::AcquisitionAxis;

mod payload;
pub(super) use payload::parse_nrrd_raw;

use super::decode::decode_element_bytes;

#[derive(Clone, Copy)]
pub(super) enum NrrdReadPurpose {
    StoredVolume,
    StoredSeries,
    ComputeF32,
}

impl NrrdReadPurpose {
    const fn sample_width(self, stored_type: SampleType) -> usize {
        match self {
            Self::StoredVolume | Self::StoredSeries => stored_type.byte_width(),
            Self::ComputeF32 => std::mem::size_of::<f32>(),
        }
    }
}

/// Decode of a NRRD file into one flat `[Z, Y, X]` volume per acquisition,
/// sharing one spatial grid.
struct DecodedNrrd {
    volumes: Vec<Vec<f32>>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    /// Acquisition geometry from the header's key/value field; `Cartesian`
    /// when absent, which is what every pre-existing NRRD means.
    coordinate_map: ritk_spatial::CoordinateMap,
}

/// NRRD payload and shared three-dimensional spatial metadata before sample
/// decoding or acquisition-axis separation.
#[derive(Debug)]
pub(super) struct RawNrrd {
    pub(super) series_axis: Option<SeriesAxis>,
    pub(super) raw_bytes: Vec<u8>,
    pub(super) element_type: String,
    pub(super) byte_order: ByteOrder,
    pub(super) acquisition: AcquisitionAxis,
    pub(super) volumes: usize,
    pub(super) voxels_per_volume: usize,
    pub(super) dims: [usize; 3],
    pub(super) origin: Point<3>,
    pub(super) spacing: Spacing<3>,
    pub(super) direction: Direction<3>,
    pub(super) coordinate_map: ritk_spatial::CoordinateMap,
}

impl DecodedNrrd {
    /// Take the sole volume, rejecting a series.
    ///
    /// The single-volume reader carries a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through it; returning volume 0 would
    /// discard the rest of the acquisition while reporting success.
    fn into_single_volume(mut self) -> Result<DecodedNrrd> {
        if self.volumes.len() != 1 {
            return Err(anyhow!(
                "NRRD file declares {} volumes along its acquisition axis; this reader \
                 returns one 3-D volume. Use the series reader to decode an acquisition \
                 series (diffusion, time series) without discarding {} of its volumes.",
                self.volumes.len(),
                self.volumes.len() - 1
            ));
        }
        self.volumes.truncate(1);
        Ok(self)
    }
}

/// Read a NRRD (Nearly Raw Raster Data) file into a 3-D `Image`.
///
/// # Axis convention
/// NRRD files produced by ITK-compatible tools store voxels in `[X, Y, Z]`
/// order with X as the fastest-varying raw axis. That flat raw order is the
/// same byte sequence as a RITK tensor shaped `[Z, Y, X]`, so the returned
/// tensor is constructed directly with shape `[nz, ny, nx]`.
///
/// # Spatial metadata
/// Direction and spacing are derived from `space directions` when that field
/// is present. NRRD file-axis vectors `[x,y,z]` are reordered into RITK
/// metadata columns `[depth,row,col] = [z,y,x]`. If only `spacings` is present,
/// the scalar spacings follow the same axis reorder with axis-aligned
/// directions.
/// Named RAS and LAS patient coordinates are converted to LPS; supported
/// `space units` are converted to millimeters. Anonymous coordinate frames,
/// unsupported spaces, and units without a known millimeter scale are rejected.
///
/// # Encoding
/// `raw`, `ascii` (`text`, `txt`), and `gzip` (`gz`) encodings are supported;
/// any other encoding returns an error with an actionable message. ASCII
/// samples are limited to 128 bytes per scalar to bound parser scratch space.
/// Multi-byte binary element types require an explicit `endian: little` or
/// `endian: big` field; one-byte samples and ASCII values do not.
///
/// # Supported types
/// All ten NRRD scalar types are accepted: signed and unsigned 8-, 16-, 32-,
/// and 64-bit integers, plus `float` and `double`. All are converted to `f32`
/// in this compute-image API; use [`crate::read_nrrd_stored`] to preserve the
/// declared sample type and bits.
///
/// # Inline vs. detached data
/// * Inline: no `data file` field (or `data file: INTERNAL`) — the payload
///   follows the blank header-terminator line in the same file.
/// * Detached: `data file: <filename>` — the payload is in a separate file
///   resolved relative to the NRRD header file's directory. Absolute paths and
///   parent traversal are rejected.
pub fn read_nrrd<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let decoded = decode_nrrd(path)?.into_single_volume()?;
    let DecodedNrrd {
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
        volumes,
    } = decoded;
    Image::from_flat_on(
        volumes
            .into_iter()
            .next()
            .expect("single_volume guaranteed"),
        dims,
        origin,
        spacing,
        direction,
        backend,
    )?
    .with_coordinate_map(coordinate_map)
}

/// Read a NRRD acquisition series as one image per volume.
///
/// # Acquisition axis
///
/// A 4-D NRRD carries one non-spatial axis — the diffusion gradient index of a
/// DWI file, a functional timepoint. Unlike NIfTI, NRRD does not fix its
/// position: the NA-MIC convention Slicer and DTIPrep emit places it first
/// (fastest, volumes interleaved voxel-by-voxel), while other tools place it
/// last (slowest, volumes contiguous). Both are read here, located through
/// `kinds` or the `none` slot in `space directions`.
///
/// Every returned image shares the file's single spatial grid, in acquisition
/// order. A 2-D or 3-D file is a one-volume series, so this reader accepts an
/// ordinary volume; [`read_nrrd`] does not accept the converse, rejecting a
/// series rather than returning its first volume.
///
/// # Errors
///
/// Returns an error when the header is invalid, when the acquisition axis is
/// absent or in an unsupported position on a 4-D file, or when the payload does
/// not match the declared sizes.
pub fn read_nrrd_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Vec<Image<f32, B, 3>>> {
    let DecodedNrrd {
        volumes,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    } = decode_nrrd(path)?;

    volumes
        .into_iter()
        .map(|data| {
            Image::from_flat_on(data, dims, origin, spacing, direction, backend)?
                .with_coordinate_map(coordinate_map.clone())
        })
        .collect()
}

fn decode_nrrd<P: AsRef<Path>>(path: P) -> Result<DecodedNrrd> {
    let RawNrrd {
        series_axis: _,
        raw_bytes,
        element_type,
        byte_order,
        acquisition,
        volumes,
        voxels_per_volume,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    } = parse_nrrd_raw(path, ImageReadBudget::DEFAULT, NrrdReadPurpose::ComputeF32)?;
    let total_voxels = voxels_per_volume
        .checked_mul(volumes)
        .ok_or_else(|| anyhow!("NRRD series element count overflows usize"))?;
    let f32_data = decode_element_bytes(&raw_bytes, &element_type, total_voxels, byte_order)?;
    if f32_data.len() != total_voxels {
        return Err(anyhow!(
            "NRRD voxel count mismatch: sizes implies {total_voxels} voxels but {} were decoded",
            f32_data.len()
        ));
    }

    let mut volume_data = Vec::new();
    volume_data
        .try_reserve_exact(volumes)
        .context("cannot allocate NRRD volume table")?;
    for _ in 0..volumes {
        let mut volume = Vec::new();
        volume
            .try_reserve_exact(voxels_per_volume)
            .context("cannot allocate decoded NRRD volume")?;
        volume_data.push(volume);
    }
    for (flat_index, value) in f32_data.into_iter().enumerate() {
        let volume_index = match acquisition {
            AcquisitionAxis::Fastest => flat_index % volumes,
            AcquisitionAxis::Absent | AcquisitionAxis::Slowest => flat_index / voxels_per_volume,
        };
        volume_data
            .get_mut(volume_index)
            .expect("invariant: declared acquisition layout selects an allocated volume")
            .push(value);
    }

    Ok(DecodedNrrd {
        volumes: volume_data,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    })
}

/// Thin reader struct for NRRD files.
///
/// The backend `B` and device are supplied per-call so a single `NrrdReader`
/// instance can serve multiple backends.
pub struct NrrdReader;

impl NrrdReader {
    /// Read a NRRD file at `path` into an [`Image`] on `device`.
    pub fn read<B: ComputeBackend, P: AsRef<Path>>(
        &self,
        path: P,
        backend: &B,
    ) -> Result<Image<f32, B, 3>> {
        read_nrrd(path, backend)
    }
}
