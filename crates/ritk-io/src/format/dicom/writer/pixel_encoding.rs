use crate::format::dicom::writer::elements::PutValue;
use anyhow::Result;
use dicom::core::{Tag, VR};
use dicom::object::InMemDicomObject;
use eunomia::convert::IntegerTarget;
use ritk_codecs::jpeg_2000::encoder::QuantizationStep;
use std::collections::HashSet;
use std::marker::PhantomData;

use super::error::DicomWriteError;

/// Photometric interpretation for scalar/grayscale images in DICOM writers.
pub(crate) const MONOCHROME2: &str = "MONOCHROME2";

pub(crate) const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

/// Unsigned representations supported by scalar DICOM image writers.
pub(crate) trait UnsignedPixelSample: Copy {
    /// Pixel width in bits for DICOM BitsAllocated, BitsStored, and HighBit.
    const BITS: u16;
}

impl UnsignedPixelSample for u8 {
    const BITS: u16 = 8;
}

impl UnsignedPixelSample for u16 {
    const BITS: u16 = 16;
}

impl UnsignedPixelSample for u32 {
    const BITS: u16 = 32;
}

/// Encodings whose complete unsigned code range is exact in f32 arithmetic.
/// RT Dose uses its supplied f64 scale instead of this image-range mapping.
pub(crate) trait RescaledPixelSample: UnsignedPixelSample + IntegerTarget {
    /// Greatest exactly representable unsigned code.
    const MAXIMUM_CODE: f32;
}

impl RescaledPixelSample for u8 {
    const MAXIMUM_CODE: f32 = 255.0;
}

impl RescaledPixelSample for u16 {
    const MAXIMUM_CODE: f32 = 65535.0;
}

/// Quality used for DICOM baseline (lossy) JPEG fragments.
///
/// Baseline JPEG is chosen here because a receiver demanded it, not because it
/// saves space -- DICOM stores a *transport* encoding, and an image archived as
/// lossy JPEG is not archived losslessly no matter what quality is chosen. So
/// the default is high enough that the DCT step is a rounding error against the
/// eight-bit quantisation already inherent in the format, and low enough that
/// the fragment stays recognisably JPEG. Overridable per call.
pub(crate) const JPEG_BASELINE_QUALITY: u8 = 95;

/// Dimensions validated against the DICOM Image Pixel Module and the buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DicomImageShape {
    /// Number of slices or frames.
    pub(crate) depth: usize,
    /// Number of rows.
    pub(crate) rows: usize,
    /// Number of columns.
    pub(crate) columns: usize,
    /// Rows representable in DICOM's US Rows attribute.
    pub(crate) rows_attribute: u16,
    /// Columns representable in DICOM's US Columns attribute.
    pub(crate) columns_attribute: u16,
    /// Samples in one frame.
    pub(crate) frame_samples: usize,
    /// Samples in the complete image.
    pub(crate) total_samples: usize,
}

/// Validate nonzero dimensions, checked sample count, and DICOM row/column limits.
pub(crate) fn validate_image_shape(
    [depth, rows, columns]: [usize; 3],
    actual_samples: usize,
) -> Result<DicomImageShape> {
    if depth == 0 || rows == 0 || columns == 0 {
        return Err(DicomWriteError::InvalidDimensions {
            depth,
            rows,
            columns,
        }
        .into());
    }
    let frame_samples = rows
        .checked_mul(columns)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let total_samples = depth
        .checked_mul(frame_samples)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    if actual_samples != total_samples {
        return Err(DicomWriteError::PixelCountMismatch {
            expected: total_samples,
            actual: actual_samples,
        }
        .into());
    }
    let rows_attribute =
        u16::try_from(rows).map_err(|_| DicomWriteError::RowsOutOfRange { rows })?;
    let columns_attribute =
        u16::try_from(columns).map_err(|_| DicomWriteError::ColumnsOutOfRange { columns })?;
    Ok(DicomImageShape {
        depth,
        rows,
        columns,
        rows_attribute,
        columns_attribute,
        frame_samples,
        total_samples,
    })
}

/// One linear mapping from modality values to the selected unsigned pixel range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PixelEncodingPlan<T> {
    minimum: f32,
    range: f32,
    rescale_slope: f32,
    rescale_intercept: f32,
    sample: PhantomData<T>,
}

impl<T: RescaledPixelSample> PixelEncodingPlan<T> {
    /// Derive a rescale and prove every sample is finite and representable.
    ///
    /// Nonconstant data maps `[minimum, maximum]` to `[0, maximum_code]`.
    /// Constant data uses a unit input range, so every stored value is zero and
    /// the intercept reconstructs the constant exactly without an epsilon.
    pub(crate) fn prepare(data: &[f32], index_start: usize) -> Result<Self> {
        let Some(&first) = data.first() else {
            return Err(DicomWriteError::PixelRangeOutOfRange.into());
        };
        if !first.is_finite() {
            return Err(DicomWriteError::NonFinitePixel { index: index_start }.into());
        }
        let (minimum, maximum) = data.iter().copied().enumerate().try_fold(
            (first, first),
            |(minimum, maximum), (offset, value)| {
                if !value.is_finite() {
                    let index = index_start
                        .checked_add(offset)
                        .ok_or(DicomWriteError::PixelCountOverflow)?;
                    return Err(DicomWriteError::NonFinitePixel { index });
                }
                Ok((minimum.min(value), maximum.max(value)))
            },
        )?;
        let maximum_code = T::MAXIMUM_CODE;
        let range = if minimum == maximum {
            1.0
        } else {
            maximum - minimum
        };
        let rescale_slope = range / maximum_code;
        if !range.is_finite() || range <= 0.0 || !rescale_slope.is_finite() || rescale_slope <= 0.0
        {
            return Err(DicomWriteError::PixelRangeOutOfRange.into());
        }
        let plan = Self {
            minimum,
            range,
            rescale_slope,
            rescale_intercept: minimum,
            sample: PhantomData,
        };
        plan.validate(data, index_start)?;
        Ok(plan)
    }

    /// Prove that conversion through Eunomia's saturating API cannot clamp.
    fn validate(&self, data: &[f32], index_start: usize) -> Result<()> {
        for (offset, value) in data.iter().copied().enumerate() {
            let index = index_start
                .checked_add(offset)
                .ok_or(DicomWriteError::PixelCountOverflow)?;
            self.encoded_sample(value, index)?;
        }
        Ok(())
    }

    /// Encode values with the previously validated mapping.
    pub(crate) fn encode(&self, data: &[f32], index_start: usize) -> Result<Vec<T>> {
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(data.len())
            .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
        for (offset, value) in data.iter().copied().enumerate() {
            let index = index_start
                .checked_add(offset)
                .ok_or(DicomWriteError::PixelCountOverflow)?;
            let sample = self.encoded_sample(value, index)?;
            encoded.push(sample);
        }
        Ok(encoded)
    }

    /// Rescale slope used by DICOM readers to recover modality values.
    #[must_use]
    pub(crate) const fn rescale_slope(self) -> f32 {
        self.rescale_slope
    }

    /// Rescale intercept used by DICOM readers to recover modality values.
    #[must_use]
    pub(crate) const fn rescale_intercept(self) -> f32 {
        self.rescale_intercept
    }

    fn normalized(self, value: f32) -> f32 {
        ((value - self.minimum) / self.range * T::MAXIMUM_CODE).round()
    }

    fn encoded_sample(&self, value: f32, index: usize) -> Result<T> {
        let normalized = self.normalized(value);
        if !normalized.is_finite() || normalized < 0.0 || normalized > T::MAXIMUM_CODE {
            return Err(DicomWriteError::EncodedPixelOutOfRange { index }.into());
        }
        // The range proof above excludes the saturating cases of from_truncated.
        Ok(T::from_truncated(f64::from(normalized)))
    }
}

/// Preflight one sample encoding for every frame without allocating the payload.
pub(crate) fn prepare_frame_encodings<T: RescaledPixelSample>(
    data: &[f32],
    shape: DicomImageShape,
) -> Result<Vec<PixelEncodingPlan<T>>> {
    let mut plans = Vec::new();
    plans
        .try_reserve_exact(shape.depth)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    for frame_index in 0..shape.depth {
        let start = frame_index
            .checked_mul(shape.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let end = start
            .checked_add(shape.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let frame = data
            .get(start..end)
            .ok_or(DicomWriteError::PixelCountMismatch {
                expected: shape.total_samples,
                actual: data.len(),
            })?;
        plans.push(PixelEncodingPlan::<T>::prepare(frame, start)?);
    }
    Ok(plans)
}

/// Validate the geometry values emitted by the scalar image writers.
pub(crate) fn validate_spatial_metadata(
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> Result<()> {
    let spacing_is_valid = spacing
        .iter()
        .all(|value| value.is_finite() && *value > 0.0);
    let origin_is_valid = origin.iter().all(|value| value.is_finite());
    let direction_is_valid = direction.iter().all(|value| value.is_finite());
    if spacing_is_valid && origin_is_valid && direction_is_valid && orthonormal_axes(direction) {
        Ok(())
    } else {
        Err(DicomWriteError::InvalidSpatialMetadata.into())
    }
}

// PS3.3 C.7.6.2.1.1 requires unit, orthogonal direction cosines:
// https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html
// Admit directions rounded once to f32, then widened without further loss.
// For component errors <= u, a three-term squared norm or orthogonal dot
// changes by <= 6u + 3u². Five f64 operations add at most
// 3 * gamma(5) * (1+u)². This is an input accuracy contract, not a fitted epsilon.
const DIRECTION_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0; // 2^-24
const DOT_UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;
const DOT_GAMMA: f64 = 5.0 * DOT_UNIT_ROUNDOFF / (1.0 - 5.0 * DOT_UNIT_ROUNDOFF);
const DIRECTION_DOT_BOUND: f64 = 6.0 * DIRECTION_UNIT_ROUNDOFF
    + 3.0 * DIRECTION_UNIT_ROUNDOFF * DIRECTION_UNIT_ROUNDOFF
    + 3.0 * DOT_GAMMA * (1.0 + DIRECTION_UNIT_ROUNDOFF) * (1.0 + DIRECTION_UNIT_ROUNDOFF);

fn orthonormal_axes(direction: &[f64]) -> bool {
    if direction.is_empty() {
        return true;
    }
    if direction.len() != 6 && direction.len() != 9 {
        return false;
    }
    let axes = direction.chunks_exact(3);
    for (index, axis) in axes.clone().enumerate() {
        let norm_squared = axis.iter().map(|value| value * value).sum::<f64>();
        if (norm_squared - 1.0).abs() > DIRECTION_DOT_BOUND {
            return false;
        }
        for other in axes.clone().skip(index + 1) {
            let dot = axis.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
            if dot.abs() > DIRECTION_DOT_BOUND {
                return false;
            }
        }
    }
    true
}

/// Emit the four DICOM tags that define unsigned pixel format.
///
/// BitsAllocated = BitsStored = `T::BITS`, HighBit = `T::BITS - 1`,
/// PixelRepresentation = 0 (unsigned).
///
/// The serialized sample type determines all four tags, so metadata from a
/// differently encoded source cannot contradict the output pixel payload.
pub(crate) fn emit_pixel_format_tags<T: UnsignedPixelSample>(obj: &mut InMemDicomObject) {
    obj.put_value(Tag(0x0028, 0x0100), VR::US, T::BITS);
    obj.put_value(Tag(0x0028, 0x0101), VR::US, T::BITS);
    obj.put_value(Tag(0x0028, 0x0102), VR::US, T::BITS - 1);
    obj.put_value(Tag(0x0028, 0x0103), VR::US, 0u16);
}

pub(crate) fn generate_series_uid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Format: 2.25.<ns>.<seq> — distinct UIDs guaranteed within a process.
    format!("2.25.{}.{}", t, n)
}

pub(super) fn generate_instance_uid(series_uid: &str, instance: usize) -> String {
    format!("{}.{}", series_uid, instance + 1)
}

/// Map a VR string slice to the dicom VR enum, defaulting to UN for unknown names.
pub(crate) fn str_to_vr(s: &str) -> VR {
    match s {
        "AE" => VR::AE,
        "AS" => VR::AS,
        "AT" => VR::AT,
        "CS" => VR::CS,
        "DA" => VR::DA,
        "DS" => VR::DS,
        "DT" => VR::DT,
        "FL" => VR::FL,
        "FD" => VR::FD,
        "IS" => VR::IS,
        "LO" => VR::LO,
        "LT" => VR::LT,
        "OB" => VR::OB,
        "OD" => VR::OD,
        "OF" => VR::OF,
        "OL" => VR::OL,
        "OW" => VR::OW,
        "PN" => VR::PN,
        "SH" => VR::SH,
        "SL" => VR::SL,
        "SQ" => VR::SQ,
        "SS" => VR::SS,
        "ST" => VR::ST,
        "TM" => VR::TM,
        "UC" => VR::UC,
        "UI" => VR::UI,
        "UL" => VR::UL,
        "UN" => VR::UN,
        "UR" => VR::UR,
        "US" => VR::US,
        "UT" => VR::UT,
        _ => VR::UN,
    }
}

/// Return compact key for a tag (group << 16 | element).
#[inline]
pub(super) fn writer_tag_key(group: u16, element: u16) -> u32 {
    ((group as u32) << 16) | (element as u32)
}

/// Tags explicitly emitted by write_dicom_series_with_metadata.
/// These are excluded from preservation emission to prevent duplication.
pub(super) fn writer_exclusion_tags() -> HashSet<u32> {
    let mut s = HashSet::new();
    s.insert(writer_tag_key(0x0008, 0x0016)); // SOP Class UID
    s.insert(writer_tag_key(0x0008, 0x0018)); // SOP Instance UID
    s.insert(writer_tag_key(0x0008, 0x0020)); // StudyDate
    s.insert(writer_tag_key(0x0008, 0x0021)); // SeriesDate
    s.insert(writer_tag_key(0x0008, 0x0031)); // SeriesTime
    s.insert(writer_tag_key(0x0008, 0x0060)); // Modality
    s.insert(writer_tag_key(0x0008, 0x103E)); // SeriesDescription
    s.insert(writer_tag_key(0x0010, 0x0010)); // PatientName
    s.insert(writer_tag_key(0x0010, 0x0020)); // PatientID
    s.insert(writer_tag_key(0x0018, 0x0050)); // SliceThickness
    s.insert(writer_tag_key(0x0019, 0x10AA)); // Private (hardcoded in writer)
    s.insert(writer_tag_key(0x0020, 0x000D)); // StudyInstanceUID
    s.insert(writer_tag_key(0x0020, 0x000E)); // SeriesInstanceUID
    s.insert(writer_tag_key(0x0020, 0x0013)); // InstanceNumber
    s.insert(writer_tag_key(0x0020, 0x0032)); // ImagePositionPatient
    s.insert(writer_tag_key(0x0020, 0x0037)); // ImageOrientationPatient
    s.insert(writer_tag_key(0x0020, 0x0052)); // FrameOfReferenceUID
    s.insert(writer_tag_key(0x0028, 0x0004)); // PhotometricInterpretation
    s.insert(writer_tag_key(0x0028, 0x0006)); // PlanarConfiguration: scalar output
    s.insert(writer_tag_key(0x0028, 0x0008)); // NumberOfFrames: single-frame slices
    s.insert(writer_tag_key(0x0028, 0x0010)); // Rows
    s.insert(writer_tag_key(0x0028, 0x0011)); // Columns
    s.insert(writer_tag_key(0x0028, 0x0100)); // BitsAllocated
    s.insert(writer_tag_key(0x0028, 0x0101)); // BitsStored
    s.insert(writer_tag_key(0x0028, 0x0102)); // HighBit
    s.insert(writer_tag_key(0x0028, 0x0103)); // PixelRepresentation
    s.insert(writer_tag_key(0x0028, 0x0030)); // PixelSpacing
    s.insert(writer_tag_key(0x0028, 0x1052)); // RescaleIntercept
    s.insert(writer_tag_key(0x0028, 0x1053)); // RescaleSlope
    s.insert(writer_tag_key(0x0029, 0x10BB)); // Private (hardcoded in writer)
    s.insert(writer_tag_key(0x7FE0, 0x0010)); // PixelData
    s.insert(writer_tag_key(0x7FE0, 0x0008)); // FloatPixelData: replaced by integer samples
    s.insert(writer_tag_key(0x7FE0, 0x0009)); // DoubleFloatPixelData
    s.insert(writer_tag_key(0x0008, 0x0064)); // ConversionType
    s.insert(writer_tag_key(0x0008, 0x0090)); // ReferringPhysicianName
    s.insert(writer_tag_key(0x0020, 0x0011)); // SeriesNumber
    s.insert(writer_tag_key(0x0028, 0x0002)); // SamplesPerPixel
    s
}

/// NEAR parameter for the JPEG-LS Lossy (near-lossless) transfer syntax.
///
/// JPEG-LS lossless mode is `NEAR = 0`; `NEAR > 0` permits a per-sample
/// reconstruction error of up to `NEAR` levels. One is the setting that makes
/// the transfer syntax's own contract hold at the tolerance this workspace's
/// round-trip tests already assert, and it is the conservative end of the range:
/// a lossy syntax should be chosen when a receiver demands it, not because a
/// caller wanted a smaller file.
pub(crate) const JPEG_LS_NEAR: u32 = 1;

/// Quantization step for the JPEG 2000 Lossy transfer syntax.
///
/// This is the irreversible 9/7 transform's dead-zone width in stored-sample
/// levels, so a decoded sample can differ from the original by roughly this much
/// per coefficient. `1.0` keeps that at one level -- below the eight-bit
/// quantisation the lossless path is compared against -- which is the same
/// conservative posture as [`JPEG_LS_NEAR`] and [`JPEG_BASELINE_QUALITY`].
pub(crate) const JPEG_2000_QUANTIZATION_STEP: QuantizationStep = QuantizationStep::UNIT;
