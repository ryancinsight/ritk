//! Exact stored-sample import for supported DICOM series.
//!
//! [`super::loader`] reconstructs a compute-ready `f32` volume and applies the
//! modality transform, which rounds wide integers and discards the source's own
//! sample representation. This module is the stored counterpart: it returns the
//! source pixels as their exact fixed-width values beside the same
//! [`DicomReadMetadata`] inventory, and refuses any series whose encoding,
//! photometry, calibration, or geometry it cannot represent before a
//! [`StoredSeries`] escapes.
//!
//! The projected inventory losses travel with the value. A caller converts the
//! import through [`DicomStoredSeries::prepare_conversion`], which runs the
//! shared preflight — including those losses — before any destination exists.

use std::path::Path;

use anyhow::Context;
use dicom::core::Tag;
use ritk_codecs::{decode_stored_pixel_frame, ByteOrder, Sample, SampleBuffer, SampleType};
use ritk_dicom::{
    parse_bytes_with_budget, parse_file_with_budget, DicomRsBackend, ParseBudget, PixelLayout,
    PixelSignedness, TransferSyntaxKind,
};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    prepare_conversion, ConversionAdapter, ConversionPrepareError, FormatMetadataLoss,
    IntensityCalibration, PreparedConversion, SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

mod error;

pub use error::DicomStoredImportError;

use crate::format::dicom::inventory::dicom_series_metadata_losses;

use super::geometry::{
    analyze_slice_spacing, dot, slice_normal_from_iop, SliceCoverage, SpacingUniformity,
};
use super::pixel::ensure_scalar_samples_per_pixel;
use super::scan::scan_dicom_path_with_budget;
use super::types::{DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata};
use super::DicomReadBudget;

/// Source identifier reported when an import runs a conversion preflight.
pub const DICOM_STORED_SOURCE: &str = "dicom";

/// A DICOM series imported as exact stored samples.
///
/// The single stored volume is the whole series, so the value keeps the
/// series-scope inventory beside the samples it was read from.
#[derive(Debug)]
pub struct DicomStoredSeries {
    series: StoredSeries,
    metadata: DicomReadMetadata,
}

impl DicomStoredSeries {
    /// Returns the imported samples with their validated physical geometry.
    #[must_use]
    pub fn series(&self) -> &StoredSeries {
        &self.series
    }

    /// Returns the source inventory the import preserved.
    #[must_use]
    pub fn metadata(&self) -> &DicomReadMetadata {
        &self.metadata
    }

    /// Projects the source inventory into the shared conversion-loss vocabulary.
    ///
    /// The import produces one volume, so every per-slice retention loss is
    /// reported at volume 0's own frame.
    #[must_use]
    pub fn metadata_losses(&self) -> Box<[FormatMetadataLoss]> {
        dicom_series_metadata_losses(&self.metadata, 0)
    }

    /// Runs the shared conversion preflight for `target` before any output opens.
    ///
    /// The reported losses are the import's own inventory, so a series that
    /// dropped a field the stored model cannot retain is rejected here rather
    /// than written with missing metadata.
    ///
    /// # Errors
    ///
    /// Returns the scoped capability report when the inventory reports a loss,
    /// or the target's typed rejection for a value it cannot represent.
    pub fn prepare_conversion<'a, T: ConversionAdapter>(
        &'a self,
        target: &'a T,
    ) -> Result<PreparedConversion<'a, T>, ConversionPrepareError<T::Rejection>> {
        prepare_conversion(
            target,
            DICOM_STORED_SOURCE,
            &self.series,
            self.metadata_losses(),
        )
    }
}

/// Reads a DICOM series from `path` into exact stored samples.
///
/// # Errors
///
/// Returns [`DicomStoredImportError`] when the scan or an instance parse fails,
/// or when the series declares a photometry, encoding, calibration, sample
/// width, or geometry the stored model cannot carry.
pub fn read_dicom_series_stored<P: AsRef<Path>>(
    path: P,
    budget: &DicomReadBudget,
) -> Result<DicomStoredSeries, DicomStoredImportError> {
    let series =
        scan_dicom_path_with_budget(path, budget).map_err(DicomStoredImportError::Reader)?;
    load_dicom_series_stored(series, budget)
}

/// Imports a pre-scanned DICOM series into exact stored samples.
///
/// This is the zero-disk counterpart of [`read_dicom_series_stored`]: a caller
/// holding a [`DicomSeriesInfo`] from
/// [`scan_dicom_instances`](super::scan::scan_dicom_instances) passes it
/// directly instead of re-scanning a directory.
///
/// # Errors
///
/// Returns [`DicomStoredImportError`] when the series declares a photometry,
/// encoding, calibration, sample width, or geometry the stored model cannot
/// carry, or when a slice's pixels cannot be decoded exactly.
pub fn load_dicom_series_stored(
    series: DicomSeriesInfo,
    budget: &DicomReadBudget,
) -> Result<DicomStoredSeries, DicomStoredImportError> {
    let DicomSeriesInfo { metadata, .. } = series;
    validate_photometry(&metadata)?;
    let [rows, cols, depth] = metadata.dimensions;
    if rows == 0 || cols == 0 || depth == 0 {
        return Err(DicomStoredImportError::EmptyGeometry);
    }
    if metadata.slices.len() != depth {
        return Err(DicomStoredImportError::SliceCountMismatch {
            expected: depth,
            actual: metadata.slices.len(),
        });
    }
    validate_uniform_geometry(&metadata)?;
    let sample_type = series_sample_type(&metadata)?;
    validate_high_bit(&metadata)?;
    let parser_budget = budget.parser();

    let mut frames = Vec::with_capacity(depth);
    for slice in &metadata.slices {
        frames.push(decode_slice_stored(
            slice,
            &parser_budget,
            sample_type,
            rows,
            cols,
        )?);
    }
    let samples = concatenate_frames(frames, sample_type)?;
    let volume = StoredVolume::new(
        [depth, rows, cols],
        samples,
        ImageMetadata::new(
            Point::new(metadata.origin),
            Spacing::new(metadata.spacing),
            Direction::from_column_major(metadata.direction),
        ),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )?;
    let stored = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume)?;
    Ok(DicomStoredSeries {
        series: stored,
        metadata,
    })
}

/// Rejects a photometric interpretation with no scalar stored form.
fn validate_photometry(metadata: &DicomReadMetadata) -> Result<(), DicomStoredImportError> {
    let Some(photometric) = metadata.photometric_interpretation else {
        return Ok(());
    };
    let photometric = photometric.as_str();
    if matches!(photometric, "MONOCHROME1" | "MONOCHROME2") {
        Ok(())
    } else {
        Err(DicomStoredImportError::NonMonochrome {
            photometric: photometric.to_owned(),
        })
    }
}

/// Rejects slice positions that would require resampling stored samples.
///
/// A stored payload is the source's own bytes; interpolating it onto a uniform
/// grid would fabricate samples the source never recorded, so the stored path
/// refuses such a series instead of resampling it.
fn validate_uniform_geometry(metadata: &DicomReadMetadata) -> Result<(), DicomStoredImportError> {
    let slices = &metadata.slices;
    if slices.len() < 2 {
        return Ok(());
    }
    let Some(normal) = slices
        .first()
        .and_then(|slice| slice.image_orientation_patient)
        .and_then(slice_normal_from_iop)
    else {
        return Err(DicomStoredImportError::NonUniformGeometry);
    };
    let mut positions = Vec::with_capacity(slices.len());
    for slice in slices {
        let Some(intercept) = slice.image_position_patient else {
            return Err(DicomStoredImportError::NonUniformGeometry);
        };
        positions.push(dot(intercept, normal));
    }
    let report = analyze_slice_spacing(&positions);
    if report.spacing_uniformity == SpacingUniformity::Nonuniform
        || report.slice_coverage == SliceCoverage::HasMissingSlices
    {
        return Err(DicomStoredImportError::NonUniformGeometry);
    }
    Ok(())
}

/// Returns the one fixed-width sample type the whole series stores.
fn series_sample_type(metadata: &DicomReadMetadata) -> Result<SampleType, DicomStoredImportError> {
    let bits_allocated = metadata.bits_allocated.unwrap_or(16);
    let representation = metadata
        .slices
        .first()
        .map_or(PixelSignedness::Unsigned, |slice| {
            slice.pixel_representation
        });
    stored_sample_type(bits_allocated, representation).ok_or(
        DicomStoredImportError::UnsupportedSampleWidth {
            bits_allocated,
            representation,
        },
    )
}

/// Maps a pixel layout onto the fixed-width sample type it stores.
///
/// BitsAllocated=24 uses the next wider representation because [`SampleType`]
/// has no 24-bit integer variant, matching the codec decoder's own mapping.
const fn stored_sample_type(
    bits_allocated: u16,
    representation: PixelSignedness,
) -> Option<SampleType> {
    match (bits_allocated, representation) {
        (8, PixelSignedness::Unsigned) => Some(SampleType::U8),
        (8, PixelSignedness::Signed) => Some(SampleType::I8),
        (16, PixelSignedness::Unsigned) => Some(SampleType::U16),
        (16, PixelSignedness::Signed) => Some(SampleType::I16),
        (24 | 32, PixelSignedness::Unsigned) => Some(SampleType::U32),
        (24 | 32, PixelSignedness::Signed) => Some(SampleType::I32),
        _ => None,
    }
}

/// Rejects a series whose stored bits do not occupy the low-order arrangement.
///
/// The codec's stored decoder masks the low `BitsStored` bits, which is exactly
/// the DICOM arrangement `HighBit = BitsStored - 1`. Any other HighBit would
/// shift the meaningful bits, so the value would not be the stored one.
fn validate_high_bit(metadata: &DicomReadMetadata) -> Result<(), DicomStoredImportError> {
    let Some(high_bit) = metadata.high_bit else {
        return Ok(());
    };
    let Some(bits_stored) = metadata.bits_stored else {
        return Ok(());
    };
    if high_bit == bits_stored.saturating_sub(1) {
        Ok(())
    } else {
        Err(DicomStoredImportError::UnexpectedHighBit {
            bits_stored,
            high_bit,
        })
    }
}

/// Decodes one slice's pixels into its exact stored samples.
fn decode_slice_stored(
    slice: &DicomSliceMetadata,
    parser_budget: &ParseBudget,
    sample_type: SampleType,
    rows: usize,
    cols: usize,
) -> Result<SampleBuffer, DicomStoredImportError> {
    let transfer_syntax = slice
        .transfer_syntax_uid
        .as_deref()
        .map(TransferSyntaxKind::from_uid)
        .unwrap_or(TransferSyntaxKind::ImplicitVrLittleEndian);
    if transfer_syntax.is_big_endian() {
        return Err(DicomStoredImportError::BigEndianSyntax {
            uid: transfer_syntax.uid().to_owned(),
        });
    }
    if transfer_syntax.is_compressed() {
        return Err(DicomStoredImportError::CompressedSyntax {
            uid: transfer_syntax.uid().to_owned(),
        });
    }
    if slice.rescale_slope != 1.0 || slice.rescale_intercept != 0.0 {
        return Err(DicomStoredImportError::NonIdentityCalibration {
            path: slice.path.clone(),
            slope: slice.rescale_slope,
            intercept: slice.rescale_intercept,
        });
    }

    let object = match &slice.part10_bytes {
        Some(bytes) => parse_bytes_with_budget::<DicomRsBackend>(bytes, parser_budget),
        None => parse_file_with_budget::<DicomRsBackend, _>(&slice.path, parser_budget),
    }
    .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))
    .map_err(DicomStoredImportError::Reader)?;

    let samples_per_pixel = object
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|element| element.to_str().ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1);
    ensure_scalar_samples_per_pixel(samples_per_pixel, slice.path.display())
        .map_err(DicomStoredImportError::Reader)?;

    let layout = PixelLayout {
        rows,
        cols,
        samples_per_pixel,
        bits_allocated: slice.bits_allocated,
        bits_stored: slice.bits_stored,
        pixel_representation: slice.pixel_representation,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    let frame_bytes =
        layout
            .bytes_per_frame()
            .map_err(|_| DicomStoredImportError::UnsupportedSampleWidth {
                bits_allocated: slice.bits_allocated,
                representation: slice.pixel_representation,
            })?;
    let payload = object
        .element(Tag(0x7FE0, 0x0010))
        .with_context(|| format!("missing Pixel Data (7FE0,0010) in {:?}", slice.path))
        .and_then(|element| {
            element
                .value()
                .to_bytes()
                .map_err(|error| anyhow::anyhow!("Pixel Data bytes unreadable: {error:?}"))
        })
        .map_err(DicomStoredImportError::Reader)?;
    let frame = payload
        .get(..frame_bytes)
        .ok_or(DicomStoredImportError::ShortPixelFrame {
            path: slice.path.clone(),
            expected: frame_bytes,
            actual: payload.len(),
        })?;

    let decoded = decode_stored_pixel_frame(frame, layout, ByteOrder::LeastSignificantByteFirst)
        .map_err(|source| DicomStoredImportError::Pixel {
            path: slice.path.clone(),
            source,
        })?;
    if decoded.sample_type() != sample_type {
        return Err(DicomStoredImportError::MixedSampleTypes {
            path: slice.path.clone(),
            expected: sample_type,
            actual: decoded.sample_type(),
        });
    }
    Ok(decoded)
}

/// Concatenates one stored frame per slice into the series' single payload.
fn concatenate_frames(
    frames: Vec<SampleBuffer>,
    sample_type: SampleType,
) -> Result<SampleBuffer, DicomStoredImportError> {
    fn gather<T: Sample>(frames: Vec<SampleBuffer>) -> Result<Vec<T>, DicomStoredImportError> {
        let mut combined = Vec::new();
        for frame in frames {
            let samples = frame
                .try_into_samples::<T>()
                .map_err(DicomStoredImportError::InconsistentSamples)?;
            combined.extend_from_slice(&samples);
        }
        Ok(combined)
    }
    Ok(match sample_type {
        SampleType::U8 => SampleBuffer::from_samples(gather::<u8>(frames)?),
        SampleType::I8 => SampleBuffer::from_samples(gather::<i8>(frames)?),
        SampleType::U16 => SampleBuffer::from_samples(gather::<u16>(frames)?),
        SampleType::I16 => SampleBuffer::from_samples(gather::<i16>(frames)?),
        SampleType::U32 => SampleBuffer::from_samples(gather::<u32>(frames)?),
        SampleType::I32 => SampleBuffer::from_samples(gather::<i32>(frames)?),
        SampleType::U64 => SampleBuffer::from_samples(gather::<u64>(frames)?),
        SampleType::I64 => SampleBuffer::from_samples(gather::<i64>(frames)?),
        SampleType::F32 => SampleBuffer::from_samples(gather::<f32>(frames)?),
        SampleType::F64 => SampleBuffer::from_samples(gather::<f64>(frames)?),
        other => {
            return Err(DicomStoredImportError::UnsupportedSampleType { sample_type: other });
        }
    })
}
