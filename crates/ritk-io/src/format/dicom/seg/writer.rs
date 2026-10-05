use crate::format::dicom::writer::elements::PutValue;
use anyhow::{Context, Result};
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;

use super::types::{DicomSegmentation, SegmentationType, SEG_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::decimal_string::{
    format_dicom_decimal, format_pair, format_six, format_triplet,
};
use crate::format::dicom::writer::pixel_encoding::{
    generate_series_uid, validate_image_shape, validate_spatial_metadata, MONOCHROME2,
};
use crate::format::dicom::writer::DicomWriteError;

/// Write a [`DicomSegmentation`] to a DICOM Segmentation Storage file.
///
/// # Invariants
/// - SOP Class UID = 1.2.840.10008.5.1.4.1.1.66.4 (Segmentation Storage).
/// - BitsAllocated = 1 (BINARY) or 8 (FRACTIONAL) from the segmentation type;
///   a contradictory `seg.bits_allocated` is rejected.
/// - BINARY samples occupy consecutive bits, least significant bit first, with
///   padding only after the complete multi-frame payload, per
///   [DICOM PS3.5 D.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_D.html).
/// - `seg.pixel_data.len()` must equal `seg.n_frames`.
/// - Each `seg.pixel_data[f].len()` must equal `seg.rows * seg.cols`.
///
/// # Errors
/// Invalid dimensions, samples, frame counts, or spatial metadata return a
/// [`DicomWriteError`] before opening the output. Serialization completes in
/// memory first; filesystem failures can still leave a partially written file.
pub fn write_dicom_seg<P: AsRef<Path>>(path: P, seg: &DicomSegmentation) -> Result<()> {
    for actual in [
        seg.pixel_data.len(),
        seg.frame_segment_numbers.len(),
        seg.image_position_per_frame.len(),
    ] {
        if actual != seg.n_frames {
            return Err(DicomWriteError::FrameMetadataCountMismatch {
                expected: seg.n_frames,
                actual,
            }
            .into());
        }
    }
    let sample_count = seg.pixel_data.iter().try_fold(0usize, |total, frame| {
        total
            .checked_add(frame.len())
            .ok_or(DicomWriteError::PixelCountOverflow)
    })?;
    let dimensions = validate_image_shape([seg.n_frames, seg.rows, seg.cols], sample_count)?;
    for frame in &seg.pixel_data {
        if frame.len() != dimensions.frame_samples {
            return Err(DicomWriteError::PixelCountMismatch {
                expected: dimensions.frame_samples,
                actual: frame.len(),
            }
            .into());
        }
    }
    let bits = match seg.segmentation_type {
        SegmentationType::Binary => 1,
        SegmentationType::Fractional => 8,
    };
    if seg.bits_allocated != bits {
        return Err(DicomWriteError::PixelDescriptionMismatch {
            declared: seg.bits_allocated,
            encoded: bits,
        }
        .into());
    }
    let spacing = seg.pixel_spacing.unwrap_or([1.0; 2]);
    let spacing = [spacing[0], spacing[1], seg.slice_thickness.unwrap_or(1.0)];
    validate_spatial_metadata(
        &spacing,
        &[],
        seg.image_orientation.as_ref().map_or(&[], |v| v.as_slice()),
    )?;
    for position in seg.image_position_per_frame.iter().flatten() {
        validate_spatial_metadata(&[], position, &[])?;
    }

    let sop_instance_uid = generate_series_uid();
    let study_instance_uid = generate_series_uid();
    let series_instance_uid = generate_series_uid();

    // PS3.5 8.1.1 and Annex D: concatenate frames without per-frame padding;
    // the first BINARY sample occupies the least significant bit.
    let payload_bytes = if bits == 1 {
        sample_count.div_ceil(8)
    } else {
        sample_count
    };
    let padded_bytes = payload_bytes
        .checked_add(payload_bytes % 2)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let mut pixel_bytes = Vec::new();
    pixel_bytes
        .try_reserve_exact(padded_bytes)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;
    pixel_bytes.resize(padded_bytes, 0);
    for (index, &sample) in seg.pixel_data.iter().flatten().enumerate() {
        if bits == 1 {
            if sample > 1 {
                return Err(DicomWriteError::InvalidBinaryPixel { index }.into());
            }
            pixel_bytes[index / 8] |= sample << (index % 8);
        } else {
            pixel_bytes[index] = sample;
        }
    }

    let seg_items: Vec<InMemDicomObject> = seg
        .segments
        .iter()
        .map(|info| {
            let mut item = InMemDicomObject::new_empty();
            item.put_value(Tag(0x0062, 0x0004), VR::US, info.segment_number);
            item.put_value(Tag(0x0062, 0x0005), VR::LO, info.segment_label.as_str());
            item.put_value(
                Tag(0x0062, 0x0006),
                VR::ST,
                info.segment_description.as_deref().unwrap_or(""),
            );
            item.put_value(
                Tag(0x0062, 0x0008),
                VR::CS,
                info.algorithm_type
                    .as_ref()
                    .map(|t| t.as_dicom_str())
                    .unwrap_or("MANUAL"),
            );
            item
        })
        .collect();

    let mut obj = InMemDicomObject::new_empty();

    obj.put_value(Tag(0x0008, 0x0016), VR::UI, SEG_SOP_CLASS_UID);
    obj.put_value(Tag(0x0008, 0x0018), VR::UI, sop_instance_uid.as_str());
    obj.put_value(Tag(0x0008, 0x0060), VR::CS, "SEG");
    obj.put_value(Tag(0x0020, 0x000D), VR::UI, study_instance_uid.as_str());
    obj.put_value(Tag(0x0020, 0x000E), VR::UI, series_instance_uid.as_str());
    obj.put_value(Tag(0x0020, 0x0013), VR::IS, "1");
    obj.put_value(
        Tag(0x0028, 0x0008),
        VR::IS,
        seg.n_frames.to_string().as_str(),
    );
    obj.put_value(Tag(0x0028, 0x0010), VR::US, dimensions.rows_attribute);
    obj.put_value(Tag(0x0028, 0x0011), VR::US, dimensions.columns_attribute);
    obj.put_value(Tag(0x0028, 0x0100), VR::US, bits);
    obj.put_value(Tag(0x0028, 0x0101), VR::US, bits);
    obj.put_value(Tag(0x0028, 0x0102), VR::US, bits - 1);
    obj.put_value(Tag(0x0028, 0x0103), VR::US, 0u16);
    obj.put_value(Tag(0x0028, 0x0002), VR::US, 1u16);
    obj.put_value(Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);
    obj.put_value(
        Tag(0x0062, 0x0001),
        VR::CS,
        seg.segmentation_type.as_dicom_str(),
    );

    if !seg_items.is_empty() {
        let seq = DataSetSequence::new(seg_items, Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x0062, 0x0002),
            VR::SQ,
            Value::from(seq),
        ));
    }

    let mut shared_item = InMemDicomObject::new_empty();
    let mut has_shared_fg = false;

    if let Some(iop) = seg.image_orientation {
        let mut ori_item = InMemDicomObject::new_empty();
        let iop_ds = format_six(iop)?;
        ori_item.put_value(Tag(0x0020, 0x0037), VR::DS, iop_ds.as_str());
        let ori_seq = DataSetSequence::new(vec![ori_item], Length::UNDEFINED);
        shared_item.put(DataElement::new(
            Tag(0x0020, 0x9116),
            VR::SQ,
            Value::from(ori_seq),
        ));
        has_shared_fg = true;
    }

    if seg.pixel_spacing.is_some() || seg.slice_thickness.is_some() {
        let mut px_item = InMemDicomObject::new_empty();
        if let Some(ps) = seg.pixel_spacing {
            let ps_ds = format_pair(ps)?;
            px_item.put_value(Tag(0x0028, 0x0030), VR::DS, ps_ds.as_str());
        }
        if let Some(st) = seg.slice_thickness {
            let st_ds = format_dicom_decimal(st)?;
            px_item.put_value(Tag(0x0018, 0x0050), VR::DS, st_ds.as_str());
        }
        let px_seq = DataSetSequence::new(vec![px_item], Length::UNDEFINED);
        shared_item.put(DataElement::new(
            Tag(0x0028, 0x9110),
            VR::SQ,
            Value::from(px_seq),
        ));
        has_shared_fg = true;
    }

    if has_shared_fg {
        let shared_seq = DataSetSequence::new(vec![shared_item], Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x5200, 0x9229),
            VR::SQ,
            Value::from(shared_seq),
        ));
    }

    let mut per_frame_items: Vec<InMemDicomObject> = Vec::with_capacity(seg.n_frames);
    for frame_idx in 0..seg.n_frames {
        let mut frame_item = InMemDicomObject::new_empty();

        let referenced_segment_number = seg.frame_segment_numbers[frame_idx];
        let mut seg_id_item = InMemDicomObject::new_empty();
        seg_id_item.put_value(Tag(0x0062, 0x000B), VR::US, referenced_segment_number);
        let seg_id_seq = DataSetSequence::new(vec![seg_id_item], Length::UNDEFINED);
        frame_item.put(DataElement::new(
            Tag(0x0062, 0x000A),
            VR::SQ,
            Value::from(seg_id_seq),
        ));

        if let Some(Some(pos)) = seg.image_position_per_frame.get(frame_idx) {
            let mut pos_item = InMemDicomObject::new_empty();
            let pos_ds = format_triplet(*pos)?;
            pos_item.put_value(Tag(0x0020, 0x0032), VR::DS, pos_ds.as_str());
            let pos_seq = DataSetSequence::new(vec![pos_item], Length::UNDEFINED);
            frame_item.put(DataElement::new(
                Tag(0x0020, 0x9113),
                VR::SQ,
                Value::from(pos_seq),
            ));
        }

        per_frame_items.push(frame_item);
    }
    if !per_frame_items.is_empty() {
        let seq = DataSetSequence::new(per_frame_items, Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x5200, 0x9230),
            VR::SQ,
            Value::from(seq),
        ));
    }

    obj.put_value(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    );

    let path = path.as_ref();
    let file = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax(EXPLICIT_VR_LE),
        )
        .context("build DICOM-SEG file meta")?;
    crate::format::dicom::writer::output::write_file(path, &file)
}
