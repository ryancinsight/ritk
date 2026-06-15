use super::utils::{
    emit_u16_pixel_format_tags, ensure_series_directory, generate_instance_uid,
    generate_series_uid, normalize_f32_to_u16, DICOM_SOP_CLASS_SECONDARY_CAPTURE,
};
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use std::path::Path;

use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;

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
    let all_data = image
        .try_data_vec()
        .context("DICOM series writer requires f32 image data")?;
    let slice_len = rows * cols;
    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (pixel_u16, rescale_slope, rescale_intercept) = normalize_f32_to_u16(slice_f32);
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
        emit_u16_pixel_format_tags(&mut obj);
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
                    .transfer_syntax(EXPLICIT_VR_LE),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}
