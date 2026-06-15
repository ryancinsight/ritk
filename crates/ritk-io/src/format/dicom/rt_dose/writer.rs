//! RT Dose writer — serialize an [`RtDoseGrid`] to a DICOM Part-10 file.

use anyhow::{bail, Context, Result};
use dicom::core::smallvec::SmallVec;
use dicom::core::value::DataSetSequence;
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::core::{DataElement, PrimitiveValue, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;

use super::types::{RtDoseGrid, RT_DOSE_SOP_CLASS_UID};
use crate::format::dicom::rt_plan::RT_PLAN_SOP_CLASS_UID;
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::utils::generate_series_uid;

/// Write an [`RtDoseGrid`] to a DICOM RT Dose Storage file at `path`.
///
/// # Write/Read Invariant
///
/// For every voxel index `k`:
///   `dose_gy[k] = raw_u32[k] as f64 * dose_grid_scaling`
///
/// Encoding: `raw_u32[k] = (dose_gy[k] / dose_grid_scaling).round().clamp(0.0, u32::MAX as f64) as u32`.
/// Quantization error: `|dose_gy[k] - reconstructed[k]| ≤ dose_grid_scaling / 2`.
///
/// # Errors
/// - `grid.dose_gy.len() != grid.n_frames * grid.rows * grid.cols`
/// - `grid.frame_offsets.len() != grid.n_frames`
/// - `!grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0`
/// - File cannot be created or written at `path`.
pub fn write_rt_dose<P: AsRef<Path>>(path: P, grid: &RtDoseGrid) -> Result<()> {
    let path = path.as_ref();

    let expected_voxels = grid.n_frames * grid.rows * grid.cols;
    if grid.dose_gy.len() != expected_voxels {
        bail!(
            "dose_gy length {} does not match n_frames={} * rows={} * cols={} = {} voxels",
            grid.dose_gy.len(),
            grid.n_frames,
            grid.rows,
            grid.cols,
            expected_voxels,
        );
    }
    if grid.frame_offsets.len() != grid.n_frames {
        bail!(
            "frame_offsets length {} does not match n_frames={}",
            grid.frame_offsets.len(),
            grid.n_frames,
        );
    }
    if !grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0 {
        bail!(
            "dose_grid_scaling must be finite and positive; got {}",
            grid.dose_grid_scaling,
        );
    }

    let sop_instance_uid = generate_series_uid();

    let inv_scaling = 1.0 / grid.dose_grid_scaling;
    let mut pixel_bytes: Vec<u8> = Vec::with_capacity(expected_voxels * 4);
    for &v in &grid.dose_gy {
        let raw = (v * inv_scaling).round().clamp(0.0, u32::MAX as f64) as u32;
        pixel_bytes.extend_from_slice(&raw.to_le_bytes());
    }

    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(RT_DOSE_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("RTDOSE"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(grid.n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(grid.rows as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(grid.cols as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(32u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(32u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(31u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0002),
        VR::CS,
        PrimitiveValue::from(grid.dose_summation_type.as_dicom_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x0004),
        VR::CS,
        PrimitiveValue::from(grid.dose_type.as_dicom_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x3004, 0x000E),
        VR::DS,
        PrimitiveValue::from(format!("{}", grid.dose_grid_scaling).as_str()),
    ));

    let offset_str = grid
        .frame_offsets
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join("\\");
    obj.put(DataElement::new(
        Tag(0x3004, 0x000C),
        VR::DS,
        PrimitiveValue::from(offset_str.as_str()),
    ));

    if let Some(pos) = grid.image_position {
        let s = format!("{}\\{}\\{}", pos[0], pos[1], pos[2]);
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }
    if let Some(ori) = grid.image_orientation {
        let s = ori
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join("\\");
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }
    if let Some(ps) = grid.pixel_spacing {
        let s = format!("{}\\{}", ps[0], ps[1]);
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from(s.as_str()),
        ));
    }

    if let Some(plan_uid) = grid
        .referenced_rt_plan_sop_instance_uid
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let mut item = InMemDicomObject::new_empty();
        item.put(DataElement::new(
            Tag(0x0008, 0x1150),
            VR::UI,
            PrimitiveValue::from(RT_PLAN_SOP_CLASS_UID),
        ));
        item.put(DataElement::new(
            Tag(0x0008, 0x1155),
            VR::UI,
            PrimitiveValue::from(plan_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x300C, 0x0002),
            VR::SQ,
            Value::from(DataSetSequence::new(
                vec![item],
                dicom::core::header::Length::UNDEFINED,
            )),
        ));
    }

    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));

    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(RT_DOSE_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax(EXPLICIT_VR_LE),
    )
    .with_context(|| "build RT Dose file meta")?
    .write_to_file(path)
    .with_context(|| format!("write RT Dose to {}", path.display()))?;

    Ok(())
}
