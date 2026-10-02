use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use flate2::read::GzDecoder;
use ritk_codecs::sample::{Conversion, Rescale, Sample, SampleBuffer};
use ritk_image::Image;
use std::borrow::Cow;
use std::fmt::Display;
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::header::NiftiHeader;
use crate::shape::checked_voxel_count;
use crate::spatial::{metadata_from_nifti_ras_affine, InternalSpatialMetadata};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const MAX_HEADER_PREFIX_BYTES: u64 = 544;

/// Read a NIfTI volume as physical values in `T`.
///
/// The stored samples convert to `T` under `conversion`
/// ([`Exact`](ritk_codecs::sample::Exact) refuses any conversion that could
/// change a value) and the header's `scl_slope`/`scl_inter` rescale is then
/// applied in `T`'s arithmetic.
///
/// # Errors
///
/// Returns an error when the file cannot be read, the header, spatial
/// metadata, or payload length is invalid, the file declares more than one
/// volume, `conversion` refuses the stored type, the header declares a
/// rescale and `T` is an integer type — use [`read_nifti_stored`] for the
/// stored integers and the rescale — or a rescale coefficient lies outside
/// `T`'s range ([`SampleError::RescaleOutOfRange`]).
///
/// [`SampleError::RescaleOutOfRange`]: ritk_codecs::sample::SampleError::RescaleOutOfRange
pub fn read_nifti<T, C, B, P>(path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let bytes = read_file(path.as_ref())?;
    read_nifti_from_bytes(&bytes, backend, conversion).map_err(|e| {
        tracing::error!("Failed to decode NIfTI file: {e:#}");
        if format!("{e:#}").contains("Invalid NIfTI spatial metadata") {
            e.context("Invalid NIfTI spatial metadata")
        } else {
            e.context("Failed to read NIfTI file")
        }
    })
}

/// Read a NIfTI volume from in-memory bytes as physical values in `T`.
///
/// Accepts `.nii` bytes directly and `.nii.gz` bytes by detecting the gzip
/// header. The decoded payload must be a single-file NIfTI-1 or NIfTI-2 stream.
///
/// # Errors
///
/// Returns an error under the conditions of [`read_nifti`] past the file read.
pub fn read_nifti_from_bytes<T, C, B>(
    bytes: &[u8],
    backend: &B,
    conversion: C,
) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
{
    decode_nifti_bytes(bytes, conversion)?
        .into_physical()?
        .into_single_volume()?
        .into_image(backend)
}

/// Read a NIfTI volume as its stored samples in `T`, with the rescale the
/// header declares left unapplied.
///
/// Reading a file in its stored type keeps every sample exact; the returned
/// [`Rescale`] maps those samples to physical values.
///
/// # Errors
///
/// Returns an error when the file cannot be read, the header, spatial
/// metadata, or payload length is invalid, `conversion` refuses the stored
/// type, or the file declares more than one volume.
pub fn read_nifti_stored<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<(Image<T, B, 3>, Rescale)>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let bytes = read_file(path.as_ref())?;
    let decoded =
        decode_nifti_bytes::<T, C>(&bytes, conversion).context("Failed to read NIfTI file")?;
    let rescale = decoded.rescale;
    Ok((decoded.into_single_volume()?.into_image(backend)?, rescale))
}

/// Read a NIfTI acquisition series as one image per volume, in physical values
/// of `T`.
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
/// Returns an error when the file cannot be read, the header, spatial
/// metadata, or payload length is invalid, `conversion` refuses the stored
/// type, the header declares a rescale and `T` is an integer type — use
/// [`read_nifti_series_stored`] for the stored integers and the rescale — or
/// a rescale coefficient lies outside `T`'s range.
pub fn read_nifti_series<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<Vec<Image<T, B, 3>>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let bytes = read_file(path.as_ref())?;
    read_nifti_series_from_bytes(&bytes, backend, conversion).map_err(|e| {
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
/// Returns an error under the conditions of [`read_nifti_series`] past the
/// file read.
pub fn read_nifti_series_from_bytes<T, C, B>(
    bytes: &[u8],
    backend: &B,
    conversion: C,
) -> Result<Vec<Image<T, B, 3>>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
{
    decode_nifti_bytes(bytes, conversion)?
        .into_physical()?
        .into_images(backend)
}

/// Read a NIfTI acquisition series as its stored samples in `T`, one image per
/// volume, with the rescale the header declares left unapplied.
///
/// The series counterpart of [`read_nifti_stored`]: the returned [`Rescale`]
/// maps every volume's samples to physical values.
///
/// # Errors
///
/// Returns an error when the file cannot be read, the header, spatial
/// metadata, or payload length is invalid, or `conversion` refuses the stored
/// type.
pub fn read_nifti_series_stored<T, C, B, P>(
    path: P,
    backend: &B,
    conversion: C,
) -> Result<(Vec<Image<T, B, 3>>, Rescale)>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let bytes = read_file(path.as_ref())?;
    let decoded = decode_nifti_bytes::<T, C>(&bytes, conversion)
        .context("Failed to read NIfTI series file")?;
    let rescale = decoded.rescale;
    Ok((decoded.into_images(backend)?, rescale))
}

/// Read `path`, keeping the I/O failure as the error's source. The context
/// names no path: an `io::Error` from `fs::read` displays only the operating
/// system's message and code, so the chain carries no caller path either.
fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| {
        tracing::error!("Failed to read NIfTI file {path:?}: {e}");
        anyhow::Error::new(e).context("Failed to read NIfTI file")
    })
}

/// Decoded NIfTI payload: one entry per volume, each in `[nz, ny, nx]` order,
/// sharing one spatial grid, with the rescale still to apply.
struct DecodedNifti<T> {
    volumes: Vec<Vec<T>>,
    dims: [usize; 3],
    spatial: InternalSpatialMetadata,
    rescale: Rescale,
}

/// One decoded volume and the grid it lies on.
struct DecodedVolume<T> {
    data: Vec<T>,
    dims: [usize; 3],
    spatial: InternalSpatialMetadata,
}

impl<T: Sample> DecodedNifti<T> {
    /// Apply the header's rescale to every volume.
    fn into_physical(mut self) -> Result<Self> {
        for volume in &mut self.volumes {
            self.rescale.apply(volume).with_context(|| {
                format!(
                    "NIfTI scl_slope {} and scl_inter {} declare a rescale; \
                     read_nifti_stored and read_nifti_series_stored return the \
                     stored samples and the rescale",
                    self.rescale.slope(),
                    self.rescale.intercept()
                )
            })?;
        }
        Ok(self)
    }

    /// One image per volume, every one on the file's spatial grid.
    fn into_images<B: ComputeBackend>(self, backend: &B) -> Result<Vec<Image<T, B, 3>>> {
        let Self {
            volumes,
            dims,
            spatial,
            rescale: _,
        } = self;
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

    /// Take the sole volume, rejecting a series.
    ///
    /// The single-volume readers carry a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through them; returning volume 0 would
    /// discard the rest of the acquisition while reporting success.
    fn into_single_volume(mut self) -> Result<DecodedVolume<T>> {
        let count = self.volumes.len();
        let (Some(data), true) = (self.volumes.pop(), count == 1) else {
            bail!(
                "NIfTI file declares {count} volumes; this reader returns one 3-D volume. \
                 Use the series reader to decode an acquisition series (diffusion, \
                 time series) without discarding {} of its volumes.",
                count.saturating_sub(1)
            );
        };
        Ok(DecodedVolume {
            data,
            dims: self.dims,
            spatial: self.spatial,
        })
    }
}

impl<T: Sample> DecodedVolume<T> {
    fn into_image<B: ComputeBackend>(self, backend: &B) -> Result<Image<T, B, 3>> {
        Image::from_flat_on(
            self.data,
            self.dims,
            self.spatial.origin,
            self.spatial.spacing,
            self.spatial.direction,
            backend,
        )
    }
}

/// Decode NIfTI bytes (gzip-detected) into stored samples converted to `T`
/// under `conversion`.
fn decode_nifti_bytes<T: Sample, C: Conversion>(
    bytes: &[u8],
    conversion: C,
) -> Result<DecodedNifti<T>> {
    let payload = single_file_payload(bytes)?;
    let header = NiftiHeader::parse(&payload).context("Invalid NIfTI header")?;
    let spatial = metadata_from_nifti_ras_affine(header.affine()?)
        .context("Invalid NIfTI spatial metadata")?;
    let [nx, ny, nz] = dims_xyz(&header);
    let range = header.volume_byte_range(payload.len())?;
    let volume_bytes = checked_voxel_count(nx, ny, nz)?
        .checked_mul(header.sample_type.byte_width())
        .ok_or_else(|| anyhow!("NIfTI volume byte count overflows usize"))?;

    // NIfTI stores x fastest, then y, z, and finally the acquisition axis —
    // RITK's `[nz, ny, nx]` flat order — so each volume is one contiguous
    // block decoded in a single pass. The range spans exactly `volume_count`
    // blocks of `volume_bytes` (non-zero: every axis and width is positive).
    let volumes = payload[range]
        .chunks_exact(volume_bytes)
        .map(|block| {
            let samples = SampleBuffer::decode(block, header.sample_type, header.byte_order())?;
            Ok(conversion.convert::<T>(samples)?)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DecodedNifti {
        volumes,
        dims: [nz, ny, nx],
        spatial,
        rescale: header.rescale,
    })
}

/// Read a NIfTI file as an integer label map in ZYX order.
///
/// # Label extraction
///
/// Labels are the stored samples; `scl_slope`/`scl_inter` are not applied,
/// since a label is an identifier rather than a measurement. Integer samples
/// convert exactly; a float sample is accepted only when it is a whole number,
/// and a negative, fractional, or over-`u32` label is an error rather than a
/// rounded or clamped one. The returned shape is `[nz, ny, nx]`.
///
/// # Errors
///
/// Returns an error when the file cannot be read, the header or payload length
/// is invalid, the file declares more than one volume, or a label voxel is not
/// a whole number in the `u32` range.
pub fn read_nifti_labels<P: AsRef<Path>>(path: P) -> Result<(Vec<u32>, [usize; 3])> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        tracing::error!("Failed to read NIfTI label file: {}", e);
        anyhow::Error::new(e).context("Failed to read NIfTI label file")
    })?;
    read_nifti_labels_from_bytes(&bytes).map_err(|e| {
        tracing::error!("Failed to decode NIfTI label file: {e:#}");
        e.context("Failed to read NIfTI label file")
    })
}

fn read_nifti_labels_from_bytes(bytes: &[u8]) -> Result<(Vec<u32>, [usize; 3])> {
    let payload = single_file_payload(bytes)?;
    let header = NiftiHeader::parse(&payload).context("Invalid NIfTI label header")?;
    if header.volume_count() != 1 {
        // The label contract is one `[nz, ny, nx]` map. Decoding only the first
        // volume of a series would report success over a discarded acquisition.
        bail!(
            "NIfTI label file declares {} volumes; a label map is a single volume",
            header.volume_count()
        );
    }
    let [nx, ny, nz] = dims_xyz(&header);
    let range = header.volume_byte_range(payload.len())?;
    let samples = SampleBuffer::decode(&payload[range], header.sample_type, header.byte_order())?;
    Ok((labels_from_samples(samples)?, [nz, ny, nx]))
}

fn labels_from_samples(samples: SampleBuffer) -> Result<Vec<u32>> {
    match samples {
        SampleBuffer::U32(labels) => Ok(labels),
        SampleBuffer::U8(values) => checked_labels(values),
        SampleBuffer::I8(values) => checked_labels(values),
        SampleBuffer::U16(values) => checked_labels(values),
        SampleBuffer::I16(values) => checked_labels(values),
        SampleBuffer::I32(values) => checked_labels(values),
        SampleBuffer::U64(values) => checked_labels(values),
        SampleBuffer::I64(values) => checked_labels(values),
        SampleBuffer::F32(values) => values
            .into_iter()
            .map(|value| float_label(f64::from(value)))
            .collect(),
        SampleBuffer::F64(values) => values.into_iter().map(float_label).collect(),
    }
}

/// A floating-point label voxel holding a whole number in the `u32` range.
fn float_label(value: f64) -> Result<u32> {
    if value.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&value) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole number in 0..=u32::MAX converts to u32 exactly"
        )]
        Ok(value as u32)
    } else {
        Err(label_range_error(value))
    }
}

fn label_range_error(value: impl Display) -> anyhow::Error {
    anyhow!("NIfTI label voxel must be a whole number in 0..=4294967295, got {value}")
}

fn checked_labels<S: Copy + Display>(values: Vec<S>) -> Result<Vec<u32>>
where
    u32: TryFrom<S>,
{
    values
        .into_iter()
        .map(|value| u32::try_from(value).map_err(|_| label_range_error(value)))
        .collect()
}

fn dims_xyz(header: &NiftiHeader) -> [usize; 3] {
    [header.dim[1], header.dim[2], header.dim[3]]
}

/// The single-file NIfTI stream in `bytes`: borrowed for `.nii`, inflated for
/// `.nii.gz`.
fn single_file_payload(bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
    if bytes.starts_with(&GZIP_MAGIC) {
        Ok(Cow::Owned(
            decode_gzip(bytes).context("Failed to decode gzipped NIfTI bytes")?,
        ))
    } else {
        Ok(Cow::Borrowed(bytes))
    }
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
