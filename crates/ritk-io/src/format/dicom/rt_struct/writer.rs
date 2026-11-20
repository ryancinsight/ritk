//! RT Structure Set writer — serialize an [`RtStructureSet`] to a DICOM Part-10 file.

use anyhow::{Context, Result};
use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::DicomTag;
use ritk_dicom::dataset::FileMetaBuilder;
use ritk_dicom::dataset::Length;
use ritk_dicom::dataset::Sequence;
use ritk_dicom::dataset::Value;
use ritk_dicom::dataset::{DataElement, VR};
use std::path::Path;

use super::types::{RtContour, RtRoiInfo, RtStructureSet, RT_STRUCT_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::pixel_encoding::generate_series_uid;

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

    let sop_instance_uid = generate_series_uid();

    let roi_seq_items: Vec<Dataset> = ss.rois.iter().map(build_roi_seq_item).collect();
    let roi_contour_seq_items: Vec<Dataset> = ss.rois.iter().map(build_roi_contour_item).collect();
    let obs_seq_items: Vec<Dataset> = ss.rois.iter().map(build_obs_item).collect();

    let mut obj = Dataset::new();

    obj.put_value(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        RT_STRUCT_SOP_CLASS_UID,
    );
    obj.put_value(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        sop_instance_uid.as_str(),
    );
    obj.put_value(DicomTag::new(0x0008, 0x0060), VR::CS, "RTSTRUCT");
    obj.put_value(
        DicomTag::new(0x3006, 0x0002),
        VR::LO,
        ss.structure_set_label.as_str(),
    );
    if let Some(name) = &ss.structure_set_name
        && !name.is_empty()
    {
        obj.put_value(DicomTag::new(0x3006, 0x0004), VR::LO, name.as_str());
    }
    if !roi_seq_items.is_empty() {
        obj.put(DataElement::new(
            DicomTag::new(0x3006, 0x0020),
            VR::SQ,
            Value::from(Sequence::new(roi_seq_items, Length::Undefined)),
        ));
    }
    if !roi_contour_seq_items.is_empty() {
        obj.put(DataElement::new(
            DicomTag::new(0x3006, 0x0039),
            VR::SQ,
            Value::from(Sequence::new(roi_contour_seq_items, Length::Undefined)),
        ));
    }
    if !obs_seq_items.is_empty() {
        obj.put(DataElement::new(
            DicomTag::new(0x3006, 0x0080),
            VR::SQ,
            Value::from(Sequence::new(obs_seq_items, Length::Undefined)),
        ));
    }

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid(RT_STRUCT_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax(EXPLICIT_VR_LE),
    )
    .with_context(|| "build RT Structure Set file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Structure Set to {}", path.display()))?;

    Ok(())
}

fn build_roi_seq_item(roi: &RtRoiInfo) -> Dataset {
    let mut item = Dataset::new();
    item.put_value(
        DicomTag::new(0x3006, 0x0022),
        VR::IS,
        roi.roi_number.to_string().as_str(),
    );
    item.put_value(DicomTag::new(0x3006, 0x0026), VR::LO, roi.roi_name.as_str());
    if let Some(desc) = &roi.roi_description
        && !desc.is_empty()
    {
        item.put_value(DicomTag::new(0x3006, 0x0028), VR::ST, desc.as_str());
    }
    item
}

fn build_roi_contour_item(roi: &RtRoiInfo) -> Dataset {
    let contour_items: Vec<Dataset> = roi.contours.iter().map(build_contour_item).collect();

    let mut item = Dataset::new();
    item.put_value(
        DicomTag::new(0x3006, 0x0084),
        VR::IS,
        roi.roi_number.to_string().as_str(),
    );
    if let Some(color) = &roi.display_color {
        let color_str = format!("{}\\{}\\{}", color[0], color[1], color[2]);
        item.put_value(DicomTag::new(0x3006, 0x002A), VR::IS, color_str.as_str());
    }
    if !contour_items.is_empty() {
        item.put(DataElement::new(
            DicomTag::new(0x3006, 0x0040),
            VR::SQ,
            Value::from(Sequence::new(contour_items, Length::Undefined)),
        ));
    }
    item
}

fn build_contour_item(contour: &RtContour) -> Dataset {
    let data_str = contour
        .points
        .iter()
        .flat_map(|p| [p[0].to_string(), p[1].to_string(), p[2].to_string()])
        .collect::<Vec<_>>()
        .join("\\");

    let mut item = Dataset::new();
    item.put_value(
        DicomTag::new(0x3006, 0x0042),
        VR::CS,
        contour.geometric_type.as_dicom_str(),
    );
    item.put_value(DicomTag::new(0x3006, 0x0050), VR::DS, data_str.as_str());
    item
}

fn build_obs_item(roi: &RtRoiInfo) -> Dataset {
    let mut item = Dataset::new();
    item.put_value(
        DicomTag::new(0x3006, 0x0084),
        VR::IS,
        roi.roi_number.to_string().as_str(),
    );
    if let Some(itype) = &roi.roi_interpreted_type {
        item.put_value(DicomTag::new(0x3006, 0x00A4), VR::CS, itype.as_dicom_str());
    }
    item
}
