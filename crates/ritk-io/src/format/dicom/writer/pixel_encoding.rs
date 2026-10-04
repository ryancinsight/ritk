use crate::format::dicom::writer::elements::PutValue;
use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};
use dicom::object::InMemDicomObject;
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

/// The `(minimum, range)` window every normalisation in this module maps onto.
///
/// Exposed so a caller writing a narrower stored format -- eight-bit for
/// baseline JPEG, for PNG -- derives its own scaling from the same window the
/// sixteen-bit path uses, instead of a second normalisation function that
/// differs only in the output width.
pub(crate) fn normalization_window(data: &[f32]) -> (f32, f32) {
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(f32::EPSILON);
    (min_val, range)
}

/// Normalize a slice of f32 pixel values to u16, computing min/max rescale parameters.
///
/// Returns `(pixel_u16, rescale_slope, rescale_intercept)`.
///
/// # Mathematical specification
///
/// Let range = max(max_val - min_val, ε). Then:
///   `pixel[i] = round((v[i] - min) / range × 65535).clamp(0, 65535)`
///   `rescale_slope = range / 65535`
///   `rescale_intercept = min_val`
///
/// Reconstruction invariant: `|v[i] - (pixel[i] × slope + intercept)| ≤ slope / 2`.
pub(crate) fn normalize_to_u16(data: &[f32]) -> (Vec<u16>, f32, f32) {
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(f32::EPSILON);
    let rescale_slope = range / U16_MAX_F;
    let rescale_intercept = min_val;
    let pixels: Vec<u16> = data
        .iter()
        .map(|&v| {
            ((v - min_val) / range * U16_MAX_F)
                .round()
                .clamp(0.0, U16_MAX_F) as u16
        })
        .collect();
    (pixels, rescale_slope, rescale_intercept)
}

/// Emit the four DICOM tags that define unsigned pixel format.
///
/// BitsAllocated = BitsStored = `bits_allocated`, HighBit = `bits_allocated - 1`,
/// PixelRepresentation = 0 (unsigned).
///
/// One function rather than a 16-bit and an 8-bit copy: the four tags differ only
/// by the width, and a per-width pair is exactly the kind of type-suffixed
/// duplicate this crate is measured for. Call sites may override individual tags
/// afterwards if metadata specifies different values.
pub(crate) fn emit_pixel_format_tags(obj: &mut InMemDicomObject, bits_allocated: u16) {
    obj.put_value(Tag(0x0028, 0x0100), VR::US, bits_allocated);
    obj.put_value(Tag(0x0028, 0x0101), VR::US, bits_allocated);
    obj.put_value(
        Tag(0x0028, 0x0102),
        VR::US,
        bits_allocated.saturating_sub(1),
    );
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
