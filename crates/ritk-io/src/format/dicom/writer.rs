//! DICOM series writer using dicom-rs v0.8.2.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.
//!
//! Stage 1 scope:
//! - preserve metadata-driven tags during series write
//! - keep pixel-module ordering stable
//! - verify private tag propagation for supported scalar tags

use super::object_model::{DicomPreservationSet, DicomSequenceItem, DicomValue};
use super::reader::DicomReadMetadata;
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value as DicomCoreValue};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

fn format_triplet(value: [f64; 3]) -> String {
    format!("{:.6}\\{:.6}\\{:.6}", value[0], value[1], value[2])
}

fn format_pair(value: [f64; 2]) -> String {
    format!("{:.6}\\{:.6}", value[0], value[1])
}

fn format_six(value: [f64; 6]) -> String {
    format!(
        "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
        value[0], value[1], value[2], value[3], value[4], value[5]
    )
}

fn generate_series_uid() -> String {
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

fn generate_instance_uid(series_uid: &str, instance: usize) -> String {
    format!("{}.{}", series_uid, instance + 1)
}

/// Map a VR string slice to the dicom VR enum, defaulting to UN for unknown names.
fn str_to_vr(s: &str) -> VR {
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
fn writer_tag_key(group: u16, element: u16) -> u32 {
    ((group as u32) << 16) | (element as u32)
}

/// Tags explicitly emitted by write_dicom_series_with_metadata.
/// These are excluded from preservation emission to prevent duplication.
fn writer_exclusion_tags() -> HashSet<u32> {
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

/// Recursively convert a DicomSequenceItem into an InMemDicomObject for writing.
fn sequence_item_to_dicom(item: &DicomSequenceItem) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for node in &item.elements {
        let tag = Tag(node.tag.group, node.tag.element);
        let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        match &node.value {
            DicomValue::Text(s) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())));
            }
            DicomValue::Bytes(b) => {
                obj.put(DataElement::new(
                    tag,
                    VR::OB,
                    PrimitiveValue::U8(SmallVec::from_vec(b.clone())),
                ));
            }
            DicomValue::U16(v) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(*v)));
            }
            DicomValue::I32(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{}", v).as_str()),
                ));
            }
            DicomValue::F64(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{:.6}", v).as_str()),
                ));
            }
            DicomValue::Sequence(sub_items) => {
                let dicom_items: Vec<InMemDicomObject> =
                    sub_items.iter().map(sequence_item_to_dicom).collect();
                let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
                let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
                obj.put(DataElement::new(tag, VR::SQ, val));
            }
            DicomValue::Empty => {}
        }
    }
    obj
}

/// Emit preserved nodes from a DicomPreservationSet into obj, skipping tags in exclusion.
///
/// Must be called BEFORE adding PixelData so the Image Pixel Module ordering invariant
/// (BitsAllocated, BitsStored, HighBit before PixelData) is preserved.
fn emit_preservation_nodes(
    obj: &mut InMemDicomObject,
    preservation: &DicomPreservationSet,
    exclusion: &HashSet<u32>,
) {
    for node in &preservation.object.nodes {
        let key = writer_tag_key(node.tag.group, node.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        let tag = Tag(node.tag.group, node.tag.element);
        let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        match &node.value {
            DicomValue::Text(s) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())));
            }
            DicomValue::Bytes(b) => {
                obj.put(DataElement::new(
                    tag,
                    VR::OB,
                    PrimitiveValue::U8(SmallVec::from_vec(b.clone())),
                ));
            }
            DicomValue::U16(v) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(*v)));
            }
            DicomValue::I32(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{}", v).as_str()),
                ));
            }
            DicomValue::F64(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{:.6}", v).as_str()),
                ));
            }
            DicomValue::Sequence(items) => {
                let dicom_items: Vec<InMemDicomObject> =
                    items.iter().map(sequence_item_to_dicom).collect();
                let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
                let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
                obj.put(DataElement::new(tag, VR::SQ, val));
            }
            DicomValue::Empty => {}
        }
    }
    for elem in &preservation.preserved {
        let key = writer_tag_key(elem.tag.group, elem.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        let tag = Tag(elem.tag.group, elem.tag.element);
        let vr = elem.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        obj.put(DataElement::new(
            tag,
            vr,
            PrimitiveValue::U8(SmallVec::from_vec(elem.bytes.clone())),
        ));
    }
}

pub fn write_dicom_series<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;
    let series_uid = generate_series_uid();
    let study_uid = series_uid.clone();
    let series_instance_uid = format!("{}.1", series_uid);
    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
    let slice_len = rows * cols;
    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (min_val, max_val) = slice_f32
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
            (1.0_f32, min_val)
        } else {
            ((max_val - min_val) / 65535.0_f32, min_val)
        };
        let pixel_u16: Vec<u16> = slice_f32
            .iter()
            .map(|&v| {
                ((v - rescale_intercept) / rescale_slope)
                    .round()
                    .clamp(0.0, 65535.0) as u16
            })
            .collect();
        let sop_instance_uid = generate_instance_uid(&series_uid, z);
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(DICOM_SOP_CLASS_SECONDARY_CAPTURE),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("OT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0064),
            VR::CS,
            PrimitiveValue::from("WSD"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from(study_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", z + 1)),
        ));
        // --- Type 2 mandatory: Patient and Study module tags ---
        // PS3.3 C.7.1.1 Patient Module (Type 2 => present with empty value when unknown).
        obj.put(DataElement::new(
            Tag(0x0008, 0x0090),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0010),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0020),
            VR::LO,
            PrimitiveValue::from(""),
        ));
        // PS3.3 C.7.2.1 General Study Module (Type 2).
        obj.put(DataElement::new(
            Tag(0x0008, 0x0020),
            VR::DA,
            PrimitiveValue::from(""),
        ));
        // PS3.3 C.7.3.1 General Series Module: SeriesNumber (Type 2).
        obj.put(DataElement::new(
            Tag(0x0020, 0x0011),
            VR::IS,
            PrimitiveValue::from("0"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_slope)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_intercept)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(DICOM_SOP_CLASS_SECONDARY_CAPTURE)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}

/// Write a DICOM series with optional metadata propagation.
///
/// When `metadata` is `Some`, spatial reference tags (ImagePositionPatient,
/// ImageOrientationPatient, PixelSpacing, SliceThickness) and series-level
/// identifiers are written into each slice. When `None`, the writer falls
/// back to generated UIDs and default tag values (identical to
/// `write_dicom_series`).
///
/// This is the Stage 1 DICOM object-model preservation boundary for the
/// supported series writer: scalar metadata tags are propagated through the
/// write path, and the emitted file layout keeps Image Pixel Module elements
/// before Pixel Data.
pub fn write_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    metadata: Option<&DicomReadMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;

    // Resolve UIDs: prefer metadata, fall back to generated.
    let generated_uid = generate_series_uid();
    let series_uid = metadata
        .and_then(|m| m.series_instance_uid.as_deref())
        .unwrap_or(&generated_uid);
    let study_uid = metadata
        .and_then(|m| m.study_instance_uid.as_deref())
        .unwrap_or(&generated_uid);

    let modality = metadata.and_then(|m| m.modality.as_deref()).unwrap_or("OT");
    let photometric = metadata
        .and_then(|m| m.photometric_interpretation.as_deref())
        .unwrap_or("MONOCHROME2");

    let sop_class = DICOM_SOP_CLASS_SECONDARY_CAPTURE;

    // Spatial parameters from metadata or defaults.
    let spacing = metadata.map(|m| m.spacing).unwrap_or([1.0, 1.0, 1.0]);
    let origin = metadata.map(|m| m.origin).unwrap_or([0.0, 0.0, 0.0]);
    let direction = metadata
        .map(|m| m.direction)
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // Slice normal is column 0 of direction matrix = direction[0..3] = N̂.
    let normal = [direction[0], direction[1], direction[2]];

    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
    let slice_len = rows * cols;

    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (min_val, max_val) = slice_f32
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
            (1.0_f32, min_val)
        } else {
            ((max_val - min_val) / 65535.0_f32, min_val)
        };
        let pixel_u16: Vec<u16> = slice_f32
            .iter()
            .map(|&v| {
                ((v - rescale_intercept) / rescale_slope)
                    .round()
                    .clamp(0.0, 65535.0) as u16
            })
            .collect();

        let sop_instance_uid = generate_instance_uid(series_uid, z);
        let mut obj = InMemDicomObject::new_empty();

        // --- Mandatory DICOM tags ---
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(sop_class),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from(modality),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0064),
            VR::CS,
            PrimitiveValue::from("WSD"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from(study_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", z + 1)),
        ));

        // --- Image pixel module ---
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_slope)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_intercept)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from(photometric),
        ));

        // --- Spatial reference tags ---
        if metadata.is_some() {
            let ipp_x = origin[0] + (z as f64) * spacing[0] * normal[0];
            let ipp_y = origin[1] + (z as f64) * spacing[0] * normal[1];
            let ipp_z = origin[2] + (z as f64) * spacing[0] * normal[2];
            obj.put(DataElement::new(
                Tag(0x0020, 0x0032),
                VR::DS,
                PrimitiveValue::from(format_triplet([ipp_x, ipp_y, ipp_z])),
            ));
            obj.put(DataElement::new(
                Tag(0x0020, 0x0037),
                VR::DS,
                // IOP = [F_r, F_c] = [direction[6..9], direction[3..6]]
                PrimitiveValue::from(format_six([
                    direction[6],
                    direction[7],
                    direction[8],
                    direction[3],
                    direction[4],
                    direction[5],
                ])),
            ));
            obj.put(DataElement::new(
                Tag(0x0028, 0x0030),
                VR::DS,
                // PixelSpacing = [ΔRow, ΔCol] = [spacing[1], spacing[2]]
                PrimitiveValue::from(format_pair([spacing[1], spacing[2]])),
            ));
            obj.put(DataElement::new(
                Tag(0x0018, 0x0050),
                VR::DS,
                // SliceThickness = Δz = spacing[0]
                PrimitiveValue::from(format!("{:.6}", spacing[0])),
            ));
        }

        // --- Type 2 mandatory fallback tags ---
        // DICOM PS3.3 Type 2: tag must be present even when value is unknown; empty string is valid.
        // These defaults are overridden below when metadata provides non-None field values.
        obj.put(DataElement::new(
            Tag(0x0008, 0x0090),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0010),
            VR::PN,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0010, 0x0020),
            VR::LO,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0020),
            VR::DA,
            PrimitiveValue::from(""),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0011),
            VR::IS,
            PrimitiveValue::from("0"),
        ));

        // --- Optional series-level tags from metadata ---
        if let Some(m) = metadata {
            if let Some(ref uid) = m.frame_of_reference_uid {
                obj.put(DataElement::new(
                    Tag(0x0020, 0x0052),
                    VR::UI,
                    PrimitiveValue::from(uid.as_str()),
                ));
            }
            if let Some(ref pid) = m.patient_id {
                obj.put(DataElement::new(
                    Tag(0x0010, 0x0020),
                    VR::LO,
                    PrimitiveValue::from(pid.as_str()),
                ));
            }
            if let Some(ref pn) = m.patient_name {
                obj.put(DataElement::new(
                    Tag(0x0010, 0x0010),
                    VR::PN,
                    PrimitiveValue::from(pn.as_str()),
                ));
            }
            if let Some(ref sd) = m.study_date {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0020),
                    VR::DA,
                    PrimitiveValue::from(sd.as_str()),
                ));
            }
            if let Some(ref desc) = m.series_description {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x103E),
                    VR::LO,
                    PrimitiveValue::from(desc.as_str()),
                ));
            }
            if let Some(ref sd) = m.series_date {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0021),
                    VR::DA,
                    PrimitiveValue::from(sd.as_str()),
                ));
            }
            if let Some(ref st) = m.series_time {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0031),
                    VR::TM,
                    PrimitiveValue::from(st.as_str()),
                ));
            }
            if let Some(bits) = m.bits_allocated {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0100),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(bits) = m.bits_stored {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0101),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(bits) = m.high_bit {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0102),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(ref private_value) = m.private_tags.get("0019,10AA") {
                obj.put(DataElement::new(
                    Tag(0x0019, 0x10AA),
                    VR::LO,
                    PrimitiveValue::from(private_value.as_str()),
                ));
            }
            if let Some(ref private_value) = m.private_tags.get("0029,10BB") {
                obj.put(DataElement::new(
                    Tag(0x0029, 0x10BB),
                    VR::LO,
                    PrimitiveValue::from(private_value.as_str()),
                ));
            }
        }

        // Emit preservation nodes before PixelData to maintain Image Pixel Module ordering.
        if let Some(m) = metadata {
            if !m.preservation.is_empty() {
                let exclusion = writer_exclusion_tags();
                emit_preservation_nodes(&mut obj, &m.preservation, &exclusion);
            }
        }
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        ));

        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(sop_class)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}

pub struct DicomWriter<B> {
    _phantom: PhantomData<B>,
}
impl<B> DicomWriter<B> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    pub fn series_path<P: AsRef<Path>>(path: P) -> PathBuf {
        path.as_ref().to_path_buf()
    }
}
impl<B> Default for DicomWriter<B> {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_series_directory(path: &Path) -> Result<PathBuf> {
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
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    type Backend = burn_ndarray::NdArray<f32>;
    use dicom::core::Tag;
    use dicom::object::open_file;
    use std::collections::HashMap;

    fn make_image(depth: usize, rows: usize, cols: usize, fill: f32) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![fill; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn make_image_with_spatial(
        depth: usize,
        rows: usize,
        cols: usize,
        fill: f32,
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![fill; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn make_test_metadata() -> DicomReadMetadata {
        let mut private_tags = HashMap::new();
        private_tags.insert("0019,10AA".to_string(), "PRIVATE_SERIES_VALUE".to_string());
        private_tags.insert(
            "0029,10BB".to_string(),
            "PRIVATE_SERIES_VALUE_2".to_string(),
        );

        DicomReadMetadata {
            series_instance_uid: Some("1.2.3.4.5.6.789".to_string()),
            study_instance_uid: Some("1.2.3.4.5.6.100".to_string()),
            frame_of_reference_uid: Some("1.2.3.4.5.6.200".to_string()),
            series_description: Some("Test Series".to_string()),
            modality: Some("CT".to_string()),
            patient_id: Some("PAT001".to_string()),
            patient_name: Some("Test^Patient".to_string()),
            study_date: Some("20240101".to_string()),
            series_date: Some("20240102".to_string()),
            series_time: Some("123456".to_string()),
            dimensions: [4, 4, 3],
            // Axial convention: spacing=[Δz,ΔRow,ΔCol], direction cols=[N̂, F_c, F_r]
            // N̂=[0,0,1], F_c=[0,1,0], F_r=[1,0,0]
            spacing: [2.5, 0.5, 0.5],
            origin: [10.0, 20.0, 30.0],
            direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags,
            preservation: crate::format::dicom::DicomPreservationSet::new(),
        }
    }

    #[test]
    fn test_writer_rejects_zero_dimension() {
        let image = make_image(0, 4, 4, 0.5);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        let result = write_dicom_series(&path, &image);
        assert!(result.is_err(), "zero depth must be rejected");
    }

    #[test]
    fn test_writer_creates_correct_number_of_slice_files() {
        let image = make_image(3, 4, 5, 0.5);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        assert!(path.is_dir());
        let count = std::fs::read_dir(&path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("dcm"))
            .count();
        assert_eq!(count, 3, "must produce exactly 3 .dcm files");
    }

    #[test]
    fn test_writer_slice_files_are_nonempty() {
        let image = make_image(2, 8, 8, 100.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        for entry in std::fs::read_dir(&path).unwrap().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("dcm") {
                let size = std::fs::metadata(entry.path()).unwrap().len();
                assert!(size > 200, "DICOM slice must be >200 bytes, got {}", size);
            }
        }
    }

    #[test]
    fn test_writer_dcm_starts_with_dicom_magic() {
        let image = make_image(1, 4, 4, 0.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        let dcm_path = path.join("slice_0000.dcm");
        let bytes = std::fs::read(&dcm_path).expect("slice file must exist");
        assert!(bytes.len() >= 132, "DICOM file must be >=132 bytes");
        assert_eq!(
            &bytes[128..132],
            b"DICM",
            "DICOM magic bytes must be present at offset 128"
        );
    }

    #[test]
    fn test_metadata_writer_spatial_tags_first_slice() {
        let meta = make_test_metadata();
        let image = make_image_with_spatial(3, 4, 4, 50.0, meta.origin, meta.spacing);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("meta_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta))
            .expect("metadata write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");

        let ipp = obj
            .element(Tag(0x0020, 0x0032))
            .expect("IPP tag must exist");
        let ipp_str = ipp.to_str().unwrap();
        let ipp_vals: Vec<f64> = ipp_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(ipp_vals.len(), 3);
        assert!((ipp_vals[0] - 10.0).abs() < 1e-3, "IPP x={}", ipp_vals[0]);
        assert!((ipp_vals[1] - 20.0).abs() < 1e-3, "IPP y={}", ipp_vals[1]);
        assert!((ipp_vals[2] - 30.0).abs() < 1e-3, "IPP z={}", ipp_vals[2]);

        let iop = obj
            .element(Tag(0x0020, 0x0037))
            .expect("IOP tag must exist");
        let iop_str = iop.to_str().unwrap();
        let iop_vals: Vec<f64> = iop_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(iop_vals.len(), 6);
        assert!((iop_vals[0] - 1.0).abs() < 1e-6, "IOP[0]");
        assert!((iop_vals[4] - 1.0).abs() < 1e-6, "IOP[4]");

        let ps = obj
            .element(Tag(0x0028, 0x0030))
            .expect("PixelSpacing must exist");
        let ps_str = ps.to_str().unwrap();
        let ps_vals: Vec<f64> = ps_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(ps_vals.len(), 2);
        assert!((ps_vals[0] - 0.5).abs() < 1e-6, "PS row={}", ps_vals[0]);
        assert!((ps_vals[1] - 0.5).abs() < 1e-6, "PS col={}", ps_vals[1]);

        let st = obj
            .element(Tag(0x0018, 0x0050))
            .expect("SliceThickness must exist");
        let st_val: f64 = st.to_str().unwrap().trim().parse().unwrap();
        assert!((st_val - 2.5).abs() < 1e-6, "SliceThickness={}", st_val);

        let mod_elem = obj
            .element(Tag(0x0008, 0x0060))
            .expect("Modality must exist");
        assert_eq!(mod_elem.to_str().unwrap().trim(), "CT");

        let pid = obj
            .element(Tag(0x0010, 0x0020))
            .expect("PatientID must exist");
        assert_eq!(pid.to_str().unwrap().trim(), "PAT001");

        let sd = obj
            .element(Tag(0x0008, 0x0021))
            .expect("SeriesDate must exist");
        assert_eq!(sd.to_str().unwrap().trim(), "20240102");
        let st = obj
            .element(Tag(0x0008, 0x0031))
            .expect("SeriesTime must exist");
        assert_eq!(st.to_str().unwrap().trim(), "123456");

        let for_uid = obj
            .element(Tag(0x0020, 0x0052))
            .expect("FrameOfReferenceUID must exist");
        assert_eq!(for_uid.to_str().unwrap().trim(), "1.2.3.4.5.6.200");

        let private = obj
            .element(Tag(0x0019, 0x10AA))
            .expect("private tag must exist");
        assert_eq!(private.to_str().unwrap().trim(), "PRIVATE_SERIES_VALUE");
    }

    #[test]
    fn test_metadata_writer_multislice_ipp_increment() {
        let meta = make_test_metadata();
        let image = make_image_with_spatial(3, 4, 4, 75.0, meta.origin, meta.spacing);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("multi_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta))
            .expect("metadata write must succeed");

        let expected_z = [30.0, 32.5, 35.0];
        for (z_idx, &ez) in expected_z.iter().enumerate() {
            let dcm_path = path.join(format!("slice_{z_idx:04}.dcm"));
            let obj = open_file(&dcm_path).unwrap_or_else(|_| panic!("must open slice {z_idx}"));
            let ipp = obj.element(Tag(0x0020, 0x0032)).expect("IPP must exist");
            let ipp_str = ipp.to_str().unwrap();
            let ipp_vals: Vec<f64> = ipp_str
                .split('\\')
                .map(|s| s.trim().parse().unwrap())
                .collect();
            assert!(
                (ipp_vals[2] - ez).abs() < 1e-3,
                "slice {z_idx}: expected z={ez}, got z={}",
                ipp_vals[2]
            );
        }
    }

    #[test]
    fn test_metadata_writer_none_metadata_fallback() {
        let image = make_image(2, 4, 4, 25.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("no_meta_series");
        write_dicom_series_with_metadata(&path, &image, None)
            .expect("write with None metadata must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");

        assert!(
            obj.element(Tag(0x0020, 0x0032)).is_err(),
            "IPP should not exist when metadata is None"
        );
        assert!(
            obj.element(Tag(0x0020, 0x0037)).is_err(),
            "IOP should not exist when metadata is None"
        );

        let mod_elem = obj
            .element(Tag(0x0008, 0x0060))
            .expect("Modality must exist");
        assert_eq!(mod_elem.to_str().unwrap().trim(), "OT");

        assert!(
            obj.element(Tag(0x0019, 0x10AA)).is_err(),
            "private tag should not exist when metadata is None"
        );
    }

    #[test]
    fn test_metadata_writer_rejects_zero_dimension() {
        let meta = make_test_metadata();
        let image = make_image(0, 4, 4, 0.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("zero_series");
        let result = write_dicom_series_with_metadata(&path, &image, Some(&meta));
        assert!(result.is_err(), "zero depth must be rejected");
    }

    #[test]
    fn test_metadata_writer_pixel_tags_precede_pixel_data_and_are_unique() {
        let meta = make_test_metadata();
        let image = make_image_with_spatial(1, 4, 4, 42.0, meta.origin, meta.spacing);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("pixel_tag_order_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta))
            .expect("metadata write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let bytes = std::fs::read(&dcm_path).expect("slice file must exist");

        let bits_allocated = [0x28_u8, 0x00, 0x00, 0x01];
        let bits_stored = [0x28_u8, 0x00, 0x01, 0x01];
        let high_bit = [0x28_u8, 0x00, 0x02, 0x01];
        let pixel_data = [0xE0_u8, 0x7F, 0x10, 0x00];

        fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
            haystack
                .windows(needle.len())
                .enumerate()
                .filter_map(|(idx, window)| (window == needle).then_some(idx))
                .collect()
        }

        let bits_allocated_pos = find_all(&bytes, &bits_allocated[..]);
        let bits_stored_pos = find_all(&bytes, &bits_stored[..]);
        let high_bit_pos = find_all(&bytes, &high_bit[..]);
        let pixel_data_pos = find_all(&bytes, &pixel_data[..]);

        assert_eq!(
            bits_allocated_pos.len(),
            1,
            "BitsAllocated tag must appear exactly once, got {:?}",
            bits_allocated_pos
        );
        assert_eq!(
            bits_stored_pos.len(),
            1,
            "BitsStored tag must appear exactly once, got {:?}",
            bits_stored_pos
        );
        assert_eq!(
            high_bit_pos.len(),
            1,
            "HighBit tag must appear exactly once, got {:?}",
            high_bit_pos
        );
        assert_eq!(
            pixel_data_pos.len(),
            1,
            "PixelData tag must appear exactly once, got {:?}",
            pixel_data_pos
        );

        let pixel_data_offset = pixel_data_pos[0];
        assert!(
            bits_allocated_pos[0] < pixel_data_offset,
            "BitsAllocated must precede PixelData: {:?} vs {}",
            bits_allocated_pos,
            pixel_data_offset
        );
        assert!(
            bits_stored_pos[0] < pixel_data_offset,
            "BitsStored must precede PixelData: {:?} vs {}",
            bits_stored_pos,
            pixel_data_offset
        );
        assert!(
            high_bit_pos[0] < pixel_data_offset,
            "HighBit must precede PixelData: {:?} vs {}",
            high_bit_pos,
            pixel_data_offset
        );
    }
    #[test]
    fn test_preservation_private_text_round_trip() {
        use crate::format::dicom::object_model::{DicomObjectNode, DicomPreservationSet, DicomTag};
        let mut preservation = DicomPreservationSet::new();
        preservation.object.insert(DicomObjectNode::text(
            DicomTag::new(0x0009, 0x0010),
            "LO",
            "PRIVATE_ROUND_TRIP",
        ));
        let mut meta = make_test_metadata();
        meta.preservation = preservation;

        let image = make_image(1, 4, 4, 10.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("priv_rt_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");
        let elem = obj
            .element(Tag(0x0009, 0x0010))
            .expect("private tag (0009,0010) must exist in written DICOM");
        assert_eq!(
            elem.to_str().unwrap().trim(),
            "PRIVATE_ROUND_TRIP",
            "private tag value must survive write"
        );
    }

    #[test]
    fn test_preservation_sequence_round_trip() {
        use crate::format::dicom::object_model::{
            DicomObjectNode, DicomPreservationSet, DicomSequenceItem, DicomTag, DicomValue,
        };
        let mut preservation = DicomPreservationSet::new();
        let mut seq_item = DicomSequenceItem::new();
        seq_item.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0104),
            "LO",
            "TestCodeMeaning",
        ));
        preservation.object.insert(DicomObjectNode {
            tag: DicomTag::new(0x0008, 0x0096),
            vr: Some("SQ".to_string()),
            value: DicomValue::Sequence(vec![seq_item]),
            private: false,
            source: None,
        });

        let mut meta = make_test_metadata();
        meta.preservation = preservation;

        let image = make_image(1, 4, 4, 20.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("seq_rt_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");
        let seq_elem = obj
            .element(Tag(0x0008, 0x0096))
            .expect("sequence tag (0008,0096) must exist in written DICOM");
        assert_eq!(
            seq_elem.vr(),
            dicom::core::VR::SQ,
            "sequence element must have VR=SQ"
        );
        let items = seq_elem.value().items().expect("sequence must have items");
        assert_eq!(items.len(), 1, "sequence must have exactly one item");
        let code_meaning = items[0]
            .element(Tag(0x0008, 0x0104))
            .expect("(0008,0104) must exist inside sequence item");
        assert_eq!(
            code_meaning.to_str().unwrap().trim(),
            "TestCodeMeaning",
            "sequence item value must survive write"
        );
    }

    #[test]
    fn test_preservation_raw_bytes_round_trip() {
        use crate::format::dicom::object_model::{
            DicomPreservationSet, DicomPreservedElement, DicomTag,
        };
        let mut preservation = DicomPreservationSet::new();
        preservation.preserve(DicomPreservedElement::new(
            DicomTag::new(0x0019, 0x1001),
            Some("OB".to_string()),
            vec![0xDE_u8, 0xAD, 0xBE, 0xEF],
        ));

        let mut meta = make_test_metadata();
        meta.preservation = preservation;

        let image = make_image(1, 4, 4, 30.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("raw_rt_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");
        let raw_elem = obj
            .element(Tag(0x0019, 0x1001))
            .expect("raw bytes tag (0019,1001) must exist in written DICOM");
        let bytes = raw_elem.to_bytes().expect("must get raw bytes");
        assert_eq!(
            bytes.as_ref(),
            &[0xDE_u8, 0xAD, 0xBE, 0xEF],
            "raw byte payload must survive write"
        );
    }

    /// Invariant: every DICOM slice file written by write_dicom_series must carry
    /// SamplesPerPixel (0028,0002) = 1 (Type 1 mandatory tag in Image Pixel Module,
    /// DICOM PS3.3 C.7.6.3.1.1).
    #[test]
    fn test_series_writer_has_samples_per_pixel_one() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let series_path = tmp.path().join("spp_series");
        let image = make_image(2, 3, 4, 1.0);
        write_dicom_series(&series_path, &image).expect("write_dicom_series");

        // Verify first slice contains SamplesPerPixel = 1
        let slice_path = series_path.join("slice_0000.dcm");
        let obj = open_file(&slice_path).expect("open_file");
        let spp: u16 = obj
            .element(Tag(0x0028, 0x0002))
            .expect("SamplesPerPixel (0028,0002) must be present in written slice")
            .to_str()
            .expect("SamplesPerPixel must be readable as string")
            .trim()
            .parse()
            .expect("SamplesPerPixel must be numeric");
        assert_eq!(spp, 1, "SamplesPerPixel must equal 1 for grayscale series");
    }

    /// Pixel clamp invariant: no encoded u16 value may exceed 65535 even when
    /// floating-point rounding produces a value slightly above max.
    ///
    /// Analytical construction: fill image with [0.0, 65535.0] range;
    /// slope = 65535/65535 = 1.0, intercept = 0.0. The clamped path must keep
    /// all pixels <= 65535.
    #[test]
    #[allow(unused_comparisons)]
    fn test_series_pixel_clamp_u16_range() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("clamp_series");
        let n_frames = 1_usize;
        let rows = 4_usize;
        let cols = 4_usize;
        // Values: 0.0 to 65535.0 in 16 steps — each u16 encodes exactly 0..=65535.
        let data: Vec<f32> = (0..n_frames * rows * cols)
            .map(|i| (i as f32) * (65535.0_f32 / 15.0_f32))
            .collect();
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([n_frames, rows, cols])),
            &Default::default(),
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        write_dicom_series(&out_path, &image).expect("write_dicom_series");

        // Open each slice and verify no pixel value exceeds 65535.
        for entry in std::fs::read_dir(&out_path).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("dcm") {
                continue;
            }
            let obj = dicom::object::open_file(&path).expect("open_file");
            if let Ok(elem) = obj.element(dicom::core::Tag(0x7FE0, 0x0010)) {
                if let Ok(bytes) = elem.value().to_bytes() {
                    for chunk in bytes.chunks_exact(2) {
                        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                        assert!(v <= 65535, "pixel value {v} exceeds u16 max");
                    }
                }
            }
        }
    }

    /// ConversionType (0008,0064) must equal "WSD" in each slice written by write_dicom_series.
    ///
    /// Invariant: SC Equipment Module (PS3.3 C.8.6.1) mandates ConversionType as Type 1.
    #[test]
    fn test_series_writer_has_conversion_type_wsd() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("conv_type_series");
        let image = make_image(2, 4, 4, 1.0);
        write_dicom_series(&out_path, &image).expect("write_dicom_series");

        let first_slice = out_path.join("slice_0000.dcm");
        let obj = open_file(&first_slice).expect("open_file");
        let conv_type = obj
            .element(Tag(0x0008, 0x0064))
            .expect("ConversionType (0008,0064) must be present")
            .to_str()
            .expect("ConversionType must be a string")
            .trim()
            .to_string();
        assert_eq!(conv_type, "WSD", "ConversionType must be 'WSD'");
    }

    /// write_dicom_series must emit Type 2 mandatory Patient and Study module tags:
    /// (0008,0090) ReferringPhysicianName, (0010,0010) PatientName,
    /// (0010,0020) PatientID, (0008,0020) StudyDate, (0020,0011) SeriesNumber.
    ///
    /// Invariant: DICOM PS3.3 Type 2 = present (may be empty).
    #[test]
    fn test_basic_series_writer_has_type2_patient_tags() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("type2_tags_series");
        let image = make_image(2, 4, 4, 1.0);
        write_dicom_series(&out_path, &image).expect("write_dicom_series");

        let first_slice = out_path.join("slice_0000.dcm");
        let obj = open_file(&first_slice).expect("open_file");
        // PatientName (0010,0010) and PatientID (0010,0020) must be present (value may be empty)
        assert!(
            obj.element(Tag(0x0010, 0x0010)).is_ok(),
            "PatientName (0010,0010) must be present"
        );
        assert!(
            obj.element(Tag(0x0010, 0x0020)).is_ok(),
            "PatientID (0010,0020) must be present"
        );
        // ReferringPhysicianName (0008,0090) must be present
        assert!(
            obj.element(Tag(0x0008, 0x0090)).is_ok(),
            "ReferringPhysicianName (0008,0090) must be present"
        );
        // StudyDate (0008,0020) must be present
        assert!(
            obj.element(Tag(0x0008, 0x0020)).is_ok(),
            "StudyDate (0008,0020) must be present"
        );
        // SeriesNumber (0020,0011) must be present
        assert!(
            obj.element(Tag(0x0020, 0x0011)).is_ok(),
            "SeriesNumber (0020,0011) must be present"
        );
    }

    /// When `write_dicom_series_with_metadata` is called with `None` metadata,
    /// the five Type 2 mandatory DICOM tags must be present in the output slice.
    ///
    /// # Invariants
    /// - PatientName (0010,0010): Type 2, PS3.3 C.7.1.1
    /// - PatientID (0010,0020): Type 2, PS3.3 C.7.1.1
    /// - ReferringPhysicianName (0008,0090): Type 2, PS3.3 C.7.2.1
    /// - StudyDate (0008,0020): Type 2, PS3.3 C.7.2.1
    /// - SeriesNumber (0020,0011): Type 2, PS3.3 C.7.3.1
    #[test]
    fn test_metadata_writer_none_metadata_type2_tags() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("meta_none_type2_series");
        let image = make_image(2, 4, 4, 1.0);
        write_dicom_series_with_metadata(&out_path, &image, None)
            .expect("write_dicom_series_with_metadata(None)");

        let first_slice = out_path.join("slice_0000.dcm");
        let obj = open_file(&first_slice).expect("open_file");

        assert!(
            obj.element(Tag(0x0010, 0x0010)).is_ok(),
            "PatientName (0010,0010) must be present for None metadata"
        );
        assert!(
            obj.element(Tag(0x0010, 0x0020)).is_ok(),
            "PatientID (0010,0020) must be present for None metadata"
        );
        assert!(
            obj.element(Tag(0x0008, 0x0090)).is_ok(),
            "ReferringPhysicianName (0008,0090) must be present for None metadata"
        );
        assert!(
            obj.element(Tag(0x0008, 0x0020)).is_ok(),
            "StudyDate (0008,0020) must be present for None metadata"
        );
        assert!(
            obj.element(Tag(0x0020, 0x0011)).is_ok(),
            "SeriesNumber (0020,0011) must be present for None metadata"
        );
    }

    #[test]
    fn test_series_uid_distinct_on_rapid_successive_calls() {
        // Two rapid calls must not produce identical UIDs regardless of clock resolution.
        // Invariant: generate_series_uid() uses AtomicU64 counter; result is 2.25.<ns>.<seq>.
        let uid_a = generate_series_uid();
        let uid_b = generate_series_uid();
        assert_ne!(uid_a, uid_b, "successive series UIDs must be distinct");
        // Both must be valid 2.25-root UIDs.
        assert!(
            uid_a.starts_with("2.25."),
            "uid_a={uid_a} must start with 2.25."
        );
        assert!(
            uid_b.starts_with("2.25."),
            "uid_b={uid_b} must start with 2.25."
        );
    }
}
