//! RT Dose writer — serialize an [`RtDoseGrid`] to a DICOM Part-10 file.

use crate::format::dicom::writer::elements::PutValue;
use anyhow::{Context, Result};
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
use crate::format::dicom::writer::decimal_string::{
    format_dicom_decimal, format_pair, format_six, format_triplet,
};
use crate::format::dicom::writer::pixel_encoding::{
    emit_pixel_format_tags, generate_series_uid, validate_image_shape, validate_spatial_metadata,
    MONOCHROME2,
};
use crate::format::dicom::writer::DicomWriteError;
use eunomia::convert::IntegerTarget;

/// Write an [`RtDoseGrid`] to a DICOM RT Dose Storage file at `path`.
///
/// # Write/Read Invariant
///
/// For every voxel index `k`:
///   `dose_gy[k] = f64::from(raw[k]) * dose_grid_scaling`
///
/// Encoding rounds `dose_gy[k] / dose_grid_scaling` to an unsigned 32-bit sample.
/// Negative, non-finite, and out-of-range values are rejected. The half-step
/// quantization bound applies in exact arithmetic; floating-point division and
/// decimal-string scaling serialization add their respective rounding errors.
/// BitsAllocated/BitsStored/HighBit/PixelRepresentation are 32/32/31/0.
///
/// # Errors
/// - `grid.dose_gy.len() != grid.n_frames * grid.rows * grid.cols`
/// - `grid.frame_offsets.len() != grid.n_frames`
/// - `!grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0`
/// - Invalid dimensions, samples, or spatial metadata return [`DicomWriteError`]
///   before opening the output. The complete file is serialized in memory first.
/// - File cannot be created or written at `path`.
pub fn write_rt_dose<P: AsRef<Path>>(path: P, grid: &RtDoseGrid) -> Result<()> {
    let path = path.as_ref();

    let dimensions =
        validate_image_shape([grid.n_frames, grid.rows, grid.cols], grid.dose_gy.len())?;
    if grid.frame_offsets.len() != grid.n_frames {
        return Err(DicomWriteError::FrameMetadataCountMismatch {
            expected: grid.n_frames,
            actual: grid.frame_offsets.len(),
        }
        .into());
    }
    if !grid.dose_grid_scaling.is_finite() || grid.dose_grid_scaling <= 0.0 {
        return Err(DicomWriteError::PixelRangeOutOfRange.into());
    }

    let sop_instance_uid = generate_series_uid();

    validate_spatial_metadata(
        grid.pixel_spacing.as_ref().map_or(&[], |v| v.as_slice()),
        grid.image_position.as_ref().map_or(&[], |v| v.as_slice()),
        grid.image_orientation
            .as_ref()
            .map_or(&[], |v| v.as_slice()),
    )?;
    if grid.frame_offsets.iter().any(|value| !value.is_finite()) {
        return Err(DicomWriteError::InvalidSpatialMetadata.into());
    }
    let mut pixel_bytes = Vec::new();
    pixel_bytes
        .try_reserve_exact(
            dimensions
                .total_samples
                .checked_mul(4)
                .ok_or(DicomWriteError::PixelCountOverflow)?,
        )
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    for (index, &v) in grid.dose_gy.iter().enumerate() {
        if !v.is_finite() {
            return Err(DicomWriteError::NonFinitePixel { index }.into());
        }
        let encoded = (v / grid.dose_grid_scaling).round();
        if v < 0.0 || !encoded.is_finite() || !(0.0..=f64::from(u32::MAX)).contains(&encoded) {
            return Err(DicomWriteError::EncodedPixelOutOfRange { index }.into());
        }
        let raw = u32::from_truncated(encoded);
        pixel_bytes.extend_from_slice(&raw.to_le_bytes());
    }

    let mut obj = InMemDicomObject::new_empty();

    obj.put_value(Tag(0x0008, 0x0016), VR::UI, RT_DOSE_SOP_CLASS_UID);
    obj.put_value(Tag(0x0008, 0x0018), VR::UI, sop_instance_uid.as_str());
    obj.put_value(Tag(0x0008, 0x0060), VR::CS, "RTDOSE");
    obj.put_value(Tag(0x0028, 0x0002), VR::US, 1u16);
    obj.put_value(Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);
    obj.put_value(
        Tag(0x0028, 0x0008),
        VR::IS,
        grid.n_frames.to_string().as_str(),
    );
    obj.put_value(Tag(0x0028, 0x0010), VR::US, dimensions.rows_attribute);
    obj.put_value(Tag(0x0028, 0x0011), VR::US, dimensions.columns_attribute);
    emit_pixel_format_tags::<u32>(&mut obj);
    obj.put_value(
        Tag(0x3004, 0x0002),
        VR::CS,
        grid.dose_summation_type.as_dicom_str(),
    );
    obj.put_value(Tag(0x3004, 0x0004), VR::CS, grid.dose_type.as_dicom_str());
    obj.put_value(
        Tag(0x3004, 0x000E),
        VR::DS,
        format_dicom_decimal(grid.dose_grid_scaling)?,
    );

    let offset_str = grid
        .frame_offsets
        .iter()
        .map(|v| format_dicom_decimal(*v))
        .collect::<Result<Vec<_>>>()?
        .join("\\");
    obj.put_value(Tag(0x3004, 0x000C), VR::DS, offset_str.as_str());

    if let Some(pos) = grid.image_position {
        let s = format_triplet(pos)?;
        obj.put_value(Tag(0x0020, 0x0032), VR::DS, s.as_str());
    }
    if let Some(ori) = grid.image_orientation {
        let s = format_six(ori)?;
        obj.put_value(Tag(0x0020, 0x0037), VR::DS, s.as_str());
    }
    if let Some(ps) = grid.pixel_spacing {
        let s = format_pair(ps)?;
        obj.put_value(Tag(0x0028, 0x0030), VR::DS, s.as_str());
    }

    if let Some(plan_uid) = grid
        .referenced_rt_plan_sop_instance_uid
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let mut item = InMemDicomObject::new_empty();
        item.put_value(Tag(0x0008, 0x1150), VR::UI, RT_PLAN_SOP_CLASS_UID);
        item.put_value(Tag(0x0008, 0x1155), VR::UI, plan_uid);
        obj.put(DataElement::new(
            Tag(0x300C, 0x0002),
            VR::SQ,
            Value::from(DataSetSequence::new(
                vec![item],
                dicom::core::header::Length::UNDEFINED,
            )),
        ));
    }

    obj.put_value(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    );

    let file = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(RT_DOSE_SOP_CLASS_UID)
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax(EXPLICIT_VR_LE),
        )
        .context("build RT Dose file meta")?;
    crate::format::dicom::writer::output::write_file(path, &file)
}
