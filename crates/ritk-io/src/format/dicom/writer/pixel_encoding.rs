use crate::format::dicom::writer::elements::PutValue;
use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};
use dicom::object::InMemDicomObject;
use eunomia::CastFrom;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Photometric interpretation for scalar/grayscale images in DICOM writers.
pub(crate) const MONOCHROME2: &str = "MONOCHROME2";

pub(crate) const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

/// Quality used for DICOM baseline (lossy) JPEG fragments.
///
/// Baseline JPEG is chosen here because a receiver demanded it, not because it
/// saves space -- DICOM stores a *transport* encoding, and an image archived as
/// lossy JPEG is not archived losslessly no matter what quality is chosen. So
/// the default is high enough that the DCT step is a rounding error against the
/// eight-bit quantisation already inherent in the format, and low enough that
/// the fragment stays recognisably JPEG. Overridable per call.
pub(crate) const JPEG_BASELINE_QUALITY: u8 = 95;

/// DICOM integer pixel types whose quantization and stored width are supported.
pub(crate) trait DicomPixelSample: CastFrom<f64> {
    /// Largest unsigned stored sample for this type.
    const MAXIMUM: f32;

    /// Bits Allocated and Bits Stored for this type.
    const BITS: u16;
}

impl DicomPixelSample for u8 {
    const MAXIMUM: f32 = 255.0;
    const BITS: u16 = 8;
}

impl DicomPixelSample for u16 {
    const MAXIMUM: f32 = 65_535.0;
    const BITS: u16 = 16;
}

/// Normalize a finite image plane into its DICOM integer sample type.
///
/// A nonconstant plane uses its exact finite range, with slope equal to range
/// divided by the type maximum and intercept equal to the minimum. Quantization
/// rounds to nearest with ties away from zero, so reconstruction error is at
/// most half the rescale slope. A constant plane uses unit range and is
/// reconstructed exactly from the intercept.
///
/// # Errors
/// Returns an error for empty or non-finite input, an unrepresentable range, or
/// a rescale slope that underflows to zero.
pub(crate) fn normalize_samples<T: DicomPixelSample>(data: &[f32]) -> Result<(Vec<T>, f32, f32)> {
    if data.is_empty() {
        bail!("cannot normalize an empty pixel buffer");
    }
    for (index, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            bail!("pixel sample at index {index} is not finite: {value}");
        }
    }

    let minimum = data.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let observed_range = maximum - minimum;
    if !observed_range.is_finite() {
        bail!("pixel range width is not representable as finite f32");
    }
    let range = if observed_range == 0.0 {
        1.0
    } else {
        observed_range
    };
    let rescale_slope = range / T::MAXIMUM;
    if !rescale_slope.is_finite() || rescale_slope <= 0.0 {
        bail!("pixel rescale slope is not representable as positive finite f32");
    }

    let pixels = data
        .iter()
        .map(|&value| {
            let quantized = ((value - minimum) / range * T::MAXIMUM)
                .round()
                .clamp(0.0, T::MAXIMUM);
            T::cast_from(f64::from(quantized))
        })
        .collect();
    Ok((pixels, rescale_slope, minimum))
}

/// Emit DICOM unsigned grayscale pixel tags for a stored sample type.
pub(crate) fn emit_pixel_format_tags<T: DicomPixelSample>(obj: &mut InMemDicomObject) {
    obj.put_value(Tag(0x0028, 0x0100), VR::US, T::BITS);
    obj.put_value(Tag(0x0028, 0x0101), VR::US, T::BITS);
    obj.put_value(Tag(0x0028, 0x0102), VR::US, T::BITS - 1);
    obj.put_value(Tag(0x0028, 0x0103), VR::US, 0u16);
}

/// Validate rows and columns before encoding them in DICOM US tags.
///
/// # Errors
/// Returns an error for zero dimensions or values that exceed u16::MAX.
pub(crate) fn dicom_pixel_dimensions(rows: usize, cols: usize) -> Result<(u16, u16)> {
    if rows == 0 || cols == 0 {
        bail!("DICOM rows and columns must be greater than zero: rows={rows} cols={cols}");
    }
    let rows_tag = u16::try_from(rows)
        .with_context(|| format!("DICOM Rows dimension {rows} exceeds u16::MAX"))?;
    let cols_tag = u16::try_from(cols)
        .with_context(|| format!("DICOM Columns dimension {cols} exceeds u16::MAX"))?;
    Ok((rows_tag, cols_tag))
}

const DICOM_DECIMAL_MAX_BYTES: usize = 16;

/// Format one finite DICOM Decimal String value within its 16-byte limit.
pub(crate) fn format_ds_value(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("DICOM Decimal String values must be finite");
    }
    let shortest = value.to_string();
    if shortest.len() <= DICOM_DECIMAL_MAX_BYTES {
        return Ok(shortest);
    }
    for precision in (0..=14).rev() {
        let candidate = format!("{value:.precision$e}");
        if candidate.len() <= DICOM_DECIMAL_MAX_BYTES {
            return Ok(candidate);
        }
    }
    bail!("finite value cannot fit the DICOM Decimal String limit")
}

/// Format a DICOM multi-value Decimal String with bounded components.
pub(crate) fn format_ds_values<const N: usize>(values: [f64; N]) -> Result<String> {
    let mut result = String::new();
    for (index, value) in values.into_iter().enumerate() {
        if index != 0 {
            result.push('\\');
        }
        result.push_str(&format_ds_value(value)?);
    }
    Ok(result)
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

#[cfg(test)]
mod tests {
    use super::{
        dicom_pixel_dimensions, format_ds_value, format_ds_values, normalize_samples,
        DICOM_DECIMAL_MAX_BYTES,
    };

    #[test]
    fn normalizers_preserve_quantization_and_calibration() {
        assert_eq!(
            normalize_samples::<u8>(&[0.0, 127.5, 255.0]).expect("finite plane"),
            (vec![0, 128, 255], 1.0, 0.0)
        );
        assert_eq!(
            normalize_samples::<u16>(&[0.0, 32_767.5, 65_535.0]).expect("finite plane"),
            (vec![0, 32_768, 65_535], 1.0, 0.0)
        );
        let (pixels, slope, intercept) =
            normalize_samples::<u16>(&[4.0, 4.0]).expect("constant plane");
        assert_eq!(pixels, [0, 0]);
        assert_eq!(slope, 1.0 / 65_535.0);
        assert_eq!(intercept, 4.0);
    }

    #[test]
    fn normalizers_preserve_ranges_below_f32_epsilon() {
        let span = 1.0e-8_f32;
        assert_eq!(
            normalize_samples::<u8>(&[0.0, span / 2.0, span]).expect("finite plane"),
            (vec![0, 128, 255], span / 255.0, 0.0)
        );
        assert_eq!(
            normalize_samples::<u16>(&[0.0, span / 2.0, span]).expect("finite plane"),
            (vec![0, 32_768, 65_535], span / 65_535.0, 0.0)
        );
    }

    #[test]
    fn normalizers_reject_invalid_inputs_and_unrepresentable_calibration() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = normalize_samples::<u16>(&[0.0, invalid])
                .expect_err("non-finite samples are invalid");
            assert_eq!(
                error.to_string(),
                format!("pixel sample at index 1 is not finite: {invalid}")
            );
        }
        assert_eq!(
            normalize_samples::<u8>(&[])
                .expect_err("empty planes have no range")
                .to_string(),
            "cannot normalize an empty pixel buffer"
        );
        assert_eq!(
            normalize_samples::<u16>(&[-f32::MAX, f32::MAX])
                .expect_err("range overflow is not representable")
                .to_string(),
            "pixel range width is not representable as finite f32"
        );
        assert_eq!(
            normalize_samples::<u16>(&[0.0, f32::from_bits(1)])
                .expect_err("zero slope cannot represent distinct samples")
                .to_string(),
            "pixel rescale slope is not representable as positive finite f32"
        );
    }

    #[test]
    fn dimensions_fit_dicom_us_without_truncation() {
        assert_eq!(
            dicom_pixel_dimensions(1, usize::from(u16::MAX)).expect("maximum dimension"),
            (1, u16::MAX)
        );
        assert_eq!(
            dicom_pixel_dimensions(usize::from(u16::MAX) + 1, 1)
                .expect_err("oversized dimensions cannot truncate")
                .to_string(),
            format!(
                "DICOM Rows dimension {} exceeds u16::MAX",
                usize::from(u16::MAX) + 1
            )
        );
    }

    #[test]
    fn decimal_strings_preserve_small_values_within_the_wire_limit() {
        assert_eq!(
            format_ds_value(1.0e-9).expect("finite DS value"),
            "0.000000001"
        );
        let tiny = format_ds_value(1.0e-30).expect("finite DS value");
        assert!(tiny.len() <= DICOM_DECIMAL_MAX_BYTES);
        assert_eq!(tiny.parse::<f64>().expect("valid DS number"), 1.0e-30);
        assert_eq!(
            format_ds_values([1.0e-9, -2.5, 0.0]).expect("finite DS values"),
            "0.000000001\\-2.5\\0"
        );
    }

    #[test]
    fn decimal_strings_reject_non_finite_values() {
        assert_eq!(
            format_ds_value(f64::NAN)
                .expect_err("DICOM DS cannot represent NaN")
                .to_string(),
            "DICOM Decimal String values must be finite"
        );
    }
}
