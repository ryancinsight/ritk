//! RT Structure Set writer — serialize an [`RtStructureSet`] to a DICOM Part-10 file.

use anyhow::{Context, Result};
use dicom::core::header::Length;
use dicom::core::value::DataSetSequence;
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::types::{RtContour, RtRoiInfo, RtStructureSet, RT_STRUCT_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;

static RT_STRUCT_UID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write an [`RtStructureSet`] to a DICOM RT Structure Set Storage file at `path`.
///
/// # Write/Read Invariant
///
/// All fields — structure_set_label, structure_set_name, roi_number, roi_name,
/// roi_description, roi_interpreted_type, display_color, geometric_type, and
/// every contour point — are preserved through a write-read cycle without loss.
///
/// # Errors
/// - File cannot be created or written at `path`.
pub fn write_rt_struct<P: AsRef<Path>>(path: P, ss: &RtStructureSet) -> Result<()> {
    let path = path.as_ref();

    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = RT_STRUCT_UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sop_instance_uid = format!("2.25.{}.{}", t, n);

    let roi_seq_items: Vec<InMemDicomObject> = ss.rois.iter().map(build_roi_seq_item).collect();
    let roi_contour_seq_items: Vec<InMemDicomObject> =
        ss.rois.iter().map(build_roi_contour_item).collect();
    let obs_seq_items: Vec<InMemDicomObject> = ss.rois.iter().map(build_obs_item).collect();

    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(RT_STRUCT_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("RTSTRUCT"),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0002),
        VR::LO,
        PrimitiveValue::from(ss.structure_set_label.as_str()),
    ));
    if let Some(name) = &ss.structure_set_name {
        if !name.is_empty() {
            obj.put(DataElement::new(
                Tag(0x3006, 0x0004),
                VR::LO,
                PrimitiveValue::from(name.as_str()),
            ));
        }
    }
    if !roi_seq_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x3006, 0x0020),
            VR::SQ,
            Value::from(DataSetSequence::new(roi_seq_items, Length::UNDEFINED)),
        ));
    }
    if !roi_contour_seq_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x3006, 0x0039),
            VR::SQ,
            Value::from(DataSetSequence::new(
                roi_contour_seq_items,
                Length::UNDEFINED,
            )),
        ));
    }
    if !obs_seq_items.is_empty() {
        obj.put(DataElement::new(
            Tag(0x3006, 0x0080),
            VR::SQ,
            Value::from(DataSetSequence::new(obs_seq_items, Length::UNDEFINED)),
        ));
    }

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_STRUCT_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax(EXPLICIT_VR_LE),
    )
    .with_context(|| "build RT Structure Set file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Structure Set to {}", path.display()))?;

    Ok(())
}

fn build_roi_seq_item(roi: &RtRoiInfo) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x3006, 0x0022),
        VR::IS,
        PrimitiveValue::from(roi.roi_number.to_string().as_str()),
    ));
    item.put(DataElement::new(
        Tag(0x3006, 0x0026),
        VR::LO,
        PrimitiveValue::from(roi.roi_name.as_str()),
    ));
    if let Some(desc) = &roi.roi_description {
        if !desc.is_empty() {
            item.put(DataElement::new(
                Tag(0x3006, 0x0028),
                VR::ST,
                PrimitiveValue::from(desc.as_str()),
            ));
        }
    }
    item
}

fn build_roi_contour_item(roi: &RtRoiInfo) -> InMemDicomObject {
    let contour_items: Vec<InMemDicomObject> =
        roi.contours.iter().map(build_contour_item).collect();

    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x3006, 0x0084),
        VR::IS,
        PrimitiveValue::from(roi.roi_number.to_string().as_str()),
    ));
    if let Some(color) = &roi.display_color {
        let color_str = format!("{}\\{}\\{}", color[0], color[1], color[2]);
        item.put(DataElement::new(
            Tag(0x3006, 0x002A),
            VR::IS,
            PrimitiveValue::from(color_str.as_str()),
        ));
    }
    if !contour_items.is_empty() {
        item.put(DataElement::new(
            Tag(0x3006, 0x0040),
            VR::SQ,
            Value::from(DataSetSequence::new(contour_items, Length::UNDEFINED)),
        ));
    }
    item
}

fn build_contour_item(contour: &RtContour) -> InMemDicomObject {
    let data_str = contour
        .points
        .iter()
        .flat_map(|p| [p[0].to_string(), p[1].to_string(), p[2].to_string()])
        .collect::<Vec<_>>()
        .join("\\");

    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x3006, 0x0042),
        VR::CS,
        PrimitiveValue::from(contour.geometric_type.as_dicom_str()),
    ));
    item.put(DataElement::new(
        Tag(0x3006, 0x0050),
        VR::DS,
        PrimitiveValue::from(data_str.as_str()),
    ));
    item
}

fn build_obs_item(roi: &RtRoiInfo) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x3006, 0x0084),
        VR::IS,
        PrimitiveValue::from(roi.roi_number.to_string().as_str()),
    ));
    if let Some(itype) = &roi.roi_interpreted_type {
        if !itype.is_empty() {
            item.put(DataElement::new(
                Tag(0x3006, 0x00A4),
                VR::CS,
                PrimitiveValue::from(itype.as_str()),
            ));
        }
    }
    item
}
