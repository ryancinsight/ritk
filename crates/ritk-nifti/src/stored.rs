//! Stored-sample NIfTI reads into the shared RITK image contract.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read};
use std::path::Path;

use anyhow::anyhow;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleError, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    CalibrationError, ImageReadBudget, ImageReadBudgetError, ImageReadResource,
    IntensityCalibration, LinearCalibration, StoredVolume, VolumeError,
};
use ritk_spatial::CoordinateMap;
use thiserror::Error;

use crate::header::{HeaderVersion, NiftiDatatype, NiftiHeader, NiftiHeaderError};
use crate::shape::checked_voxel_count;
use crate::spatial::metadata_from_nifti_ras_affine;

const NIFTI1_HEADER_LENGTH: usize = 348;
const NIFTI2_HEADER_LENGTH: usize = 540;
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Reads a NIfTI-1 or NIfTI-2 volume without converting its stored samples.
///
/// The input may be a single-file `.nii` or gzip-compressed `.nii.gz`. The
/// returned volume carries the exact stored sample type and values, LPS
/// millimeter geometry, and NIfTI intensity calibration. Acquisition axes are
/// rejected; use the series reader for rank-four data.
///
/// `budget` bounds encoded file bytes, decompressed bytes through the sample
/// payload, and the decoded sample allocation before it is allocated.
///
/// # Examples
///
/// ```no_run
/// use ritk_nifti::{read_nifti_stored, write_nifti_stored};
/// use ritk_image_io::ImageReadBudget;
///
/// fn main() -> anyhow::Result<()> {
///     let volume = read_nifti_stored("scan.nii.gz", ImageReadBudget::DEFAULT)?;
///     write_nifti_stored("copy.nii.gz", &volume)?;
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns [`NiftiStoredReadError`] for malformed headers, unsupported rank or
/// units, truncated input, invalid calibration or geometry, allocation and
/// codec failures, or a read-budget violation.
pub fn read_nifti_stored<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<StoredVolume, NiftiStoredReadError> {
    let file = File::open(path)?;
    let encoded_bytes = file.metadata()?.len();
    budget.check(ImageReadResource::EncodedBytes, encoded_bytes)?;
    let limited = file.take(budget.max_encoded_bytes());
    let reader = NiftiInput::new(limited)?;
    read_stored(reader, encoded_bytes, budget)
}

/// Reads a NIfTI volume from bytes without converting its stored samples.
///
/// The byte slice may contain a single-file `.nii` or gzip-compressed
/// `.nii.gz` stream. Only one rank-three volume is accepted.
///
/// # Examples
///
/// ```no_run
/// use ritk_image_io::ImageReadBudget;
/// use ritk_nifti::read_nifti_stored_from_bytes;
///
/// fn main() -> anyhow::Result<()> {
///     let bytes = std::fs::read("scan.nii.gz")?;
///     let volume = read_nifti_stored_from_bytes(&bytes, ImageReadBudget::DEFAULT)?;
///     let shape = volume.shape();
///     assert!(shape.iter().all(|length| *length > 0));
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns [`NiftiStoredReadError`] for malformed headers, unsupported rank or
/// units, truncated input, invalid calibration or geometry, allocation and
/// codec failures, or a read-budget violation.
pub fn read_nifti_stored_from_bytes(
    bytes: &[u8],
    budget: ImageReadBudget,
) -> Result<StoredVolume, NiftiStoredReadError> {
    let encoded_bytes =
        u64::try_from(bytes.len()).map_err(|_| NiftiStoredReadError::ImageSizeOverflow)?;
    budget.check(ImageReadResource::EncodedBytes, encoded_bytes)?;
    let limited = Cursor::new(bytes).take(budget.max_encoded_bytes());
    let reader = NiftiInput::new(limited)?;
    read_stored(reader, encoded_bytes, budget)
}

fn read_stored<R: Read>(
    mut reader: NiftiInput<R>,
    encoded_bytes: u64,
    budget: ImageReadBudget,
) -> Result<StoredVolume, NiftiStoredReadError> {
    let header = read_header(&mut reader)?;
    if header.dim[0] != 3 || header.volume_count() != 1 {
        return Err(NiftiStoredReadError::UnsupportedRank {
            rank: header.dim[0],
        });
    }

    let [nx, ny, nz] = [header.dim[1], header.dim[2], header.dim[3]];
    let shape = [nz, ny, nx];
    let sample_count = checked_voxel_count(nx, ny, nz).map_err(header_error)?;
    let payload_range = header.volume_byte_range(usize::MAX).map_err(header_error)?;
    let expanded_bytes =
        u64::try_from(payload_range.end).map_err(|_| NiftiStoredReadError::ImageSizeOverflow)?;
    budget.check(ImageReadResource::DecodedBytes, expanded_bytes)?;
    budget.check(ImageReadResource::EncodedBytes, encoded_bytes)?;

    let header_length = match header.version {
        HeaderVersion::One => NIFTI1_HEADER_LENGTH,
        HeaderVersion::Two => NIFTI2_HEADER_LENGTH,
    };
    let skip_length = payload_range
        .start
        .checked_sub(header_length)
        .ok_or_else(|| {
            header_error(anyhow!("NIfTI voxel offset precedes the parsed header end"))
        })?;
    let skip_length =
        u64::try_from(skip_length).map_err(|_| NiftiStoredReadError::ImageSizeOverflow)?;
    let padding_length = skip_length.checked_sub(4).ok_or_else(|| {
        header_error(anyhow!(
            "NIfTI voxel offset does not leave room for the extension flag"
        ))
    })?;
    let mut extension_flag = [0_u8; 4];
    reader.read_exact(&mut extension_flag)?;
    if extension_flag[0] != 0 {
        return Err(NiftiStoredReadError::UnsupportedExtensions);
    }
    if extension_flag[1..].iter().any(|byte| *byte != 0) {
        return Err(NiftiStoredReadError::InvalidExtensionFlag { extension_flag });
    }
    let skipped = io::copy(&mut reader.by_ref().take(padding_length), &mut io::sink())?;
    if skipped != padding_length {
        return Err(NiftiStoredReadError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "NIfTI voxel offset lies beyond the input stream",
        )));
    }

    let byte_order = match header.byte_order() {
        consus_core::ByteOrder::LittleEndian => ByteOrder::LeastSignificantByteFirst,
        consus_core::ByteOrder::BigEndian => ByteOrder::MostSignificantByteFirst,
    };
    let samples = SampleBuffer::read_from(
        sample_type(header.datatype),
        &mut reader,
        sample_count,
        byte_order,
    )?;
    let calibration = nifti_calibration(header.scl_slope, header.scl_inter)?;
    let spatial = metadata_from_nifti_ras_affine(
        header.affine().map_err(header_error)?,
        header.spatial_unit_scale().map_err(header_error)?,
    )
    .map_err(header_error)?;
    let metadata = ImageMetadata::new(spatial.origin, spatial.spacing, spatial.direction);
    StoredVolume::new(
        shape,
        samples,
        metadata,
        CoordinateMap::Cartesian,
        calibration,
    )
    .map_err(NiftiStoredReadError::Volume)
}

fn read_header<R: Read>(reader: &mut R) -> Result<NiftiHeader, NiftiStoredReadError> {
    let mut bytes = [0_u8; NIFTI2_HEADER_LENGTH];
    reader.read_exact(&mut bytes[..NIFTI1_HEADER_LENGTH])?;
    let size_bytes: [u8; 4] = bytes[..4]
        .try_into()
        .map_err(|_| NiftiStoredReadError::ImageSizeOverflow)?;
    let little_endian_size = i32::from_le_bytes(size_bytes);
    let big_endian_size = i32::from_be_bytes(size_bytes);
    let (version, header_length) = match (little_endian_size, big_endian_size) {
        (348, _) | (_, 348) => (HeaderVersion::One, NIFTI1_HEADER_LENGTH),
        (540, _) | (_, 540) => (HeaderVersion::Two, NIFTI2_HEADER_LENGTH),
        _ => {
            return Err(header_error(anyhow!(
                "invalid NIfTI sizeof_hdr; expected 348 or 540"
            )));
        }
    };
    if matches!(version, HeaderVersion::Two) {
        reader.read_exact(&mut bytes[NIFTI1_HEADER_LENGTH..NIFTI2_HEADER_LENGTH])?;
    }
    NiftiHeader::parse(&bytes[..header_length])
        .map_err(NiftiHeaderError::from)
        .map_err(NiftiStoredReadError::Header)
}

fn header_error(error: anyhow::Error) -> NiftiStoredReadError {
    NiftiStoredReadError::Header(NiftiHeaderError::from(error))
}

fn sample_type(datatype: NiftiDatatype) -> SampleType {
    match datatype {
        NiftiDatatype::Uint8 => SampleType::U8,
        NiftiDatatype::Int8 => SampleType::I8,
        NiftiDatatype::Uint16 => SampleType::U16,
        NiftiDatatype::Int16 => SampleType::I16,
        NiftiDatatype::Uint32 => SampleType::U32,
        NiftiDatatype::Int32 => SampleType::I32,
        NiftiDatatype::Float32 => SampleType::F32,
        NiftiDatatype::Float64 => SampleType::F64,
        NiftiDatatype::Int64 => SampleType::I64,
        NiftiDatatype::Uint64 => SampleType::U64,
    }
}

fn nifti_calibration(
    slope: f64,
    intercept: f64,
) -> Result<IntensityCalibration, NiftiStoredReadError> {
    if slope == 0.0 || (slope == 1.0 && intercept == 0.0) {
        return Ok(IntensityCalibration::Identity);
    }
    if !slope.is_finite() || !intercept.is_finite() {
        return Err(NiftiStoredReadError::NonFiniteCalibration { slope, intercept });
    }
    LinearCalibration::new(slope, intercept)
        .map(IntensityCalibration::Linear)
        .map_err(NiftiStoredReadError::InvalidCalibration)
}

enum NiftiInput<R> {
    Plain(BufReader<R>),
    Gzip(flate2::read::GzDecoder<BufReader<R>>),
}

impl<R: Read> NiftiInput<R> {
    fn new(reader: R) -> io::Result<Self> {
        let mut reader = BufReader::new(reader);
        if reader.fill_buf()?.starts_with(&GZIP_MAGIC) {
            Ok(Self::Gzip(flate2::read::GzDecoder::new(reader)))
        } else {
            Ok(Self::Plain(reader))
        }
    }
}

impl<R: Read> Read for NiftiInput<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(reader) => reader.read(buffer),
            Self::Gzip(reader) => reader.read(buffer),
        }
    }
}

/// An error while reading an exact stored NIfTI volume.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredReadError {
    /// The encoded file or decompression stream failed.
    #[error("NIfTI input stream failed: {0}")]
    Io(#[from] io::Error),
    /// The NIfTI header or declared byte range is invalid.
    #[error(transparent)]
    Header(#[from] NiftiHeaderError),
    /// The input exceeded a configured image read budget.
    #[error(transparent)]
    Budget(#[from] ImageReadBudgetError),
    /// The stored-sample codec failed.
    #[error(transparent)]
    Sample(#[from] SampleError),
    /// The stored volume violated a shared structural or geometry invariant.
    #[error(transparent)]
    Volume(#[from] VolumeError),
    /// The file has no single rank-three volume representation.
    #[error("NIfTI rank {rank} cannot be represented as one stored 3-D volume")]
    UnsupportedRank {
        /// NIfTI rank declared by `dim[0]`.
        rank: usize,
    },
    /// The NIfTI calibration coefficients are not finite.
    #[error("NIfTI scaling coefficients must be finite when the slope is nonzero (slope={slope}, intercept={intercept})")]
    NonFiniteCalibration {
        /// Stored `scl_slope` coefficient.
        slope: f64,
        /// Stored `scl_inter` coefficient.
        intercept: f64,
    },
    /// The calibration coefficients violate the shared calibration contract.
    #[error("invalid NIfTI intensity calibration: {0}")]
    InvalidCalibration(#[source] CalibrationError),
    /// NIfTI extension records are not represented by the shared volume value.
    #[error("NIfTI extension records cannot be preserved by StoredVolume")]
    UnsupportedExtensions,
    /// Reserved NIfTI extension-flag bytes are nonzero.
    #[error("invalid NIfTI extension flag {extension_flag:?}")]
    InvalidExtensionFlag {
        /// Four bytes immediately following the NIfTI header.
        extension_flag: [u8; 4],
    },
    /// A file size cannot be represented by the platform's read budget.
    #[error("NIfTI image byte count does not fit the platform size type")]
    ImageSizeOverflow,
}
