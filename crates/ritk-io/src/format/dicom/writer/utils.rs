use anyhow::{bail, Context, Result};
use dicom::core::VR;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(crate) const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

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
