use crate::format::dicom::writer::elements::PutValue;
use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};
use dicom::object::InMemDicomObject;
use eunomia::convert::{IntegerConversionError, IntegerTarget};
use ritk_codecs::jpeg_2000::encoder::QuantizationStep;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Maximum u16 pixel value as f32, used for normalization (u16::MAX = 65535).
pub(crate) const U16_MAX_F: f32 = 65535.0;

/// Photometric interpretation for scalar/grayscale images in DICOM writers.
pub(crate) const MONOCHROME2: &str = "MONOCHROME2";

pub(crate) const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

/// Maximum u8 pixel value as f32, used for normalization (u8::MAX = 255).
pub(crate) const U8_MAX_F: f32 = 255.0;

/// Quality used for DICOM baseline (lossy) JPEG fragments.
///
/// Baseline JPEG is chosen here because a receiver demanded it, not because it
/// saves space -- DICOM stores a *transport* encoding, and an image archived as
/// lossy JPEG is not archived losslessly no matter what quality is chosen. So
/// the default is high enough that the DCT step is a rounding error against the
/// eight-bit quantisation already inherent in the format, and low enough that
/// the fragment stays recognisably JPEG. Overridable per call.
pub(crate) const JPEG_BASELINE_QUALITY: u8 = 95;

/// Failure while mapping modality values into a stored DICOM sample range.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PixelEncodingError {
    /// The writer received no samples.
    #[error("DICOM pixel buffer is empty")]
    Empty,
    /// The writer received a non-finite modality value.
    #[error("DICOM pixel value {value} must be finite")]
    NonFinite { value: f32 },
    /// The finite endpoints have a range that cannot be represented in `f32`.
    #[error("DICOM pixel range from {minimum} to {maximum} exceeds f32")]
    RangeOverflow { minimum: f32, maximum: f32 },
    /// A normalized value was outside the target sample's integer range.
    #[error("normalized DICOM pixel is outside the stored sample range")]
    SampleConversion(#[from] IntegerConversionError),
}

/// Unsigned integer widths supported by the DICOM writer's scalar pixel path.
pub(crate) trait DicomStoredSample: IntegerTarget {
    const ALLOCATED_BITS: u16;
    const MAX_PIXEL_VALUE: f32;
}

impl DicomStoredSample for u8 {
    const ALLOCATED_BITS: u16 = 8;
    const MAX_PIXEL_VALUE: f32 = U8_MAX_F;
}

impl DicomStoredSample for u16 {
    const ALLOCATED_BITS: u16 = 16;
    const MAX_PIXEL_VALUE: f32 = U16_MAX_F;
}

/// Normalize modality values to the requested unsigned DICOM sample type.
///
/// For `R = max - min`, each value maps to
/// `round((v - min) / max(R, ε) × T::MAX_PIXEL_VALUE)`, clamped to the
/// destination range. Rounding uses `f32::round` (nearest, ties away from zero)
/// before Eunomia checks representability; this preserves the existing writer's
/// sample values. The resulting reconstruction error is at most half the
/// RescaleSlope, apart from `f32` arithmetic rounding.
///
/// # Errors
///
/// Returns an error for empty input, non-finite values, an overflowing range,
/// or a sample that cannot be represented by `T`.
pub(crate) fn normalize_pixels<T: DicomStoredSample>(
    data: &[f32],
) -> std::result::Result<(Vec<T>, f32, f32), PixelEncodingError> {
    let first = data.first().copied().ok_or(PixelEncodingError::Empty)?;
    if !first.is_finite() {
        return Err(PixelEncodingError::NonFinite { value: first });
    }
    let (minimum, maximum) = data.iter().copied().skip(1).try_fold(
        (first, first),
        |(minimum, maximum), value| {
            if !value.is_finite() {
                return Err(PixelEncodingError::NonFinite { value });
            }
            Ok::<_, PixelEncodingError>((minimum.min(value), maximum.max(value)))
        },
    )?;
    let extent = maximum - minimum;
    if !extent.is_finite() {
        return Err(PixelEncodingError::RangeOverflow { minimum, maximum });
    }
    let range = extent.max(f32::EPSILON);
    let rescale_slope = range / T::MAX_PIXEL_VALUE;
    let rescale_intercept = minimum;
    let pixels = data
        .iter()
        .map(|&value| {
            let normalized = ((value - minimum) / range * T::MAX_PIXEL_VALUE)
                .round()
                .clamp(0.0, T::MAX_PIXEL_VALUE);
            T::try_from_rounded(normalized).map_err(PixelEncodingError::SampleConversion)
        })
        .collect::<std::result::Result<Vec<T>, PixelEncodingError>>()?;
    Ok((pixels, rescale_slope, rescale_intercept))
}

/// Emit the unsigned DICOM pixel-format tags for the stored sample type.
pub(crate) fn emit_pixel_format_tags<T: DicomStoredSample>(obj: &mut InMemDicomObject) {
    let high_bit = T::ALLOCATED_BITS - 1;
    obj.put_value(Tag(0x0028, 0x0100), VR::US, T::ALLOCATED_BITS);
    obj.put_value(Tag(0x0028, 0x0101), VR::US, T::ALLOCATED_BITS);
    obj.put_value(Tag(0x0028, 0x0102), VR::US, high_bit);
    obj.put_value(Tag(0x0028, 0x0103), VR::US, 0u16);
}

pub(super) fn format_triplet(value: [f64; 3]) -> String {
    format!("{:.6}\\{:.6}\\{:.6}", value[0], value[1], value[2])
}

pub(super) fn format_pair(value: [f64; 2]) -> String {
    format!("{:.6}\\{:.6}", value[0], value[1])
}

pub(super) fn format_six(value: [f64; 6]) -> String {
    format!(
        "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
        value[0], value[1], value[2], value[3], value[4], value[5]
    )
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
    s.insert(writer_tag_key(0x0008, 0x0064)); // ConversionType
    s.insert(writer_tag_key(0x0008, 0x0090)); // ReferringPhysicianName
    s.insert(writer_tag_key(0x0020, 0x0011)); // SeriesNumber
    s.insert(writer_tag_key(0x0028, 0x0002)); // SamplesPerPixel
    s
}

pub(super) fn ensure_series_directory(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        if !path.is_dir() {
            bail!("DICOM output path is not a directory");
        }
        return Ok(path.to_path_buf());
    }
    std::fs::create_dir_all(path)
        .with_context(|| "failed to create DICOM series output directory")?;
    Ok(path.to_path_buf())
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

#[cfg(test)]
mod tests {
    use super::{
        emit_pixel_format_tags, normalize_pixels, DicomStoredSample, InMemDicomObject,
        PixelEncodingError, Tag,
    };

    #[test]
    fn normalization_uses_the_destination_sample_width() {
        let (pixels_8, slope_8, intercept_8) = normalize_pixels::<u8>(&[-1.0, 0.0, 1.0])
            .expect("finite range fits eight-bit normalization");
        let (pixels_16, slope_16, intercept_16) = normalize_pixels::<u16>(&[-1.0, 0.0, 1.0])
            .expect("finite range fits sixteen-bit normalization");

        assert_eq!(pixels_8, [0, 128, 255]);
        assert_eq!(pixels_16, [0, 32768, 65535]);
        assert_eq!(slope_8, 2.0 / 255.0);
        assert_eq!(slope_16, 2.0 / 65535.0);
        assert_eq!(intercept_8, -1.0);
        assert_eq!(intercept_16, -1.0);
    }

    #[test]
    fn normalization_handles_constant_and_invalid_ranges() {
        let (pixels, slope, intercept) = normalize_pixels::<u8>(&[5.0, 5.0])
            .expect("constant finite values have a nonzero encoding range");

        assert_eq!(pixels, [0, 0]);
        assert_eq!(slope, f32::EPSILON / 255.0);
        assert_eq!(intercept, 5.0);
        assert!(matches!(
            normalize_pixels::<u8>(&[]),
            Err(PixelEncodingError::Empty)
        ));
        assert!(matches!(
            normalize_pixels::<u8>(&[f32::NAN]),
            Err(PixelEncodingError::NonFinite { value }) if value.is_nan()
        ));
        assert!(matches!(
            normalize_pixels::<u16>(&[-f32::MAX, f32::MAX]),
            Err(PixelEncodingError::RangeOverflow { minimum, maximum })
                if minimum == -f32::MAX && maximum == f32::MAX
        ));
    }

    #[test]
    fn pixel_format_tags_follow_the_stored_sample_type() {
        assert_pixel_format_tags::<u8>(8, 7);
        assert_pixel_format_tags::<u16>(16, 15);
    }

    fn assert_pixel_format_tags<T: DicomStoredSample>(allocated: u16, high_bit: u16) {
        let mut object = InMemDicomObject::new_empty();
        emit_pixel_format_tags::<T>(&mut object);

        assert_eq!(
            object
                .element(Tag(0x0028, 0x0100))
                .expect("BitsAllocated is present")
                .to_int::<u16>()
                .expect("BitsAllocated is unsigned short"),
            allocated
        );
        assert_eq!(
            object
                .element(Tag(0x0028, 0x0101))
                .expect("BitsStored is present")
                .to_int::<u16>()
                .expect("BitsStored is unsigned short"),
            allocated
        );
        assert_eq!(
            object
                .element(Tag(0x0028, 0x0102))
                .expect("HighBit is present")
                .to_int::<u16>()
                .expect("HighBit is unsigned short"),
            high_bit
        );
        assert_eq!(
            object
                .element(Tag(0x0028, 0x0103))
                .expect("PixelRepresentation is present")
                .to_int::<u16>()
                .expect("PixelRepresentation is unsigned short"),
            0
        );
    }
}
