use anyhow::{bail, Context, Result};
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;

use super::types::{DicomSegmentation, SEG_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::utils::{generate_series_uid, MONOCHROME2};

/// Write a [`DicomSegmentation`] to a DICOM Segmentation Storage file.
///
/// # Invariants
/// - SOP Class UID = 1.2.840.10008.5.1.4.1.1.66.4 (Segmentation Storage).
/// - BitsAllocated = 1 (BINARY) or 8 (FRACTIONAL) based on `seg.bits_allocated`.
/// - BINARY pixel data is packed MSB-first within each byte per DICOM PS3.5 §8.2:
///   pixel i → byte = i/8, bit = 7-(i%8).
/// - `seg.pixel_data.len()` must equal `seg.n_frames`.
/// - Each `seg.pixel_data[f].len()` must equal `seg.rows * seg.cols`.
pub fn write_dicom_seg<P: AsRef<Path>>(path: P, seg: &DicomSegmentation) -> Result<()> {
    if seg.pixel_data.len() != seg.n_frames {
        bail!(
            "pixel_data.len()={} != n_frames={}",
            seg.pixel_data.len(),
            seg.n_frames
        );
    }
    if seg.frame_segment_numbers.len() != seg.n_frames {
        bail!(
            "frame_segment_numbers.len()={} != n_frames={}",
            seg.frame_segment_numbers.len(),
            seg.n_frames
        );
    }
    let n_pixels = seg.rows * seg.cols;
    for (f, frame) in seg.pixel_data.iter().enumerate() {
        if frame.len() != n_pixels {
            bail!(
                "pixel_data[{}].len()={} != rows*cols={}",
                f,
                frame.len(),
                n_pixels
            );
        }
    }

    let sop_instance_uid = generate_series_uid();
    let study_instance_uid = generate_series_uid();
    let series_instance_uid = generate_series_uid();

    // BINARY: MSB-first packing — inverse of unpack_pixel_data (BitsAllocated == 1).
    // FRACTIONAL: raw byte-per-pixel concatenation (BitsAllocated == 8).
    let pixel_bytes: Vec<u8> = match seg.bits_allocated {
        1 => {
            let frame_byte_count = n_pixels.div_ceil(8);
            let mut buf = vec![0u8; seg.n_frames * frame_byte_count];
            for (f, frame) in seg.pixel_data.iter().enumerate() {
                let base = f * frame_byte_count;
                for (i, &px) in frame.iter().enumerate() {
                    if px != 0 {
                        buf[base + i / 8] |= 1u8 << (7 - (i % 8));
                    }
                }
            }
            buf
        }
        8 => seg.pixel_data.iter().flatten().copied().collect(),
        _ => bail!("unsupported bits_allocated={}", seg.bits_allocated),
    };

    let seg_items: Vec<InMemDicomObject> = seg
        .segments
        .iter()
        .map(|info| {
            let mut item = InMemDicomObject::new_empty();
            item.put(DataElement::new(
                Tag(0x0062, 0x0004),
                VR::US,
                PrimitiveValue::from(info.segment_number),
            ));
            item.put(DataElement::new(
                Tag(0x0062, 0x0005),
                VR::LO,
                PrimitiveValue::from(info.segment_label.as_str()),
            ));
            item.put(DataElement::new(
                Tag(0x0062, 0x0006),
                VR::ST,
                PrimitiveValue::from(info.segment_description.as_deref().unwrap_or("")),
            ));
            item.put(DataElement::new(
                Tag(0x0062, 0x0008),
                VR::CS,
                PrimitiveValue::from(
                    info.algorithm_type
                        .as_ref()
                        .map(|t| t.as_dicom_str())
                        .unwrap_or("MANUAL"),
                ),
            ));
            item
        })
        .collect();

    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(SEG_SOP_CLASS_UID),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("SEG"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from(study_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from(series_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(seg.n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(seg.rows as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(seg.cols as u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(seg.bits_allocated),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(seg.bits_allocated),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(seg.bits_allocated - 1),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from(MONOCHROME2),
    ));
    obj.put(DataElement::new(
        Tag(0x0062, 0x0001),
        VR::CS,
        PrimitiveValue::from(seg.segmentation_type.as_dicom_str()),
    ));

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
        let iop_ds = format!(
            "{}\\{}\\{}\\{}\\{}\\{}",
            iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]
        );
        ori_item.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(iop_ds.as_str()),
        ));
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
            let ps_ds = format!("{}\\{}", ps[0], ps[1]);
            px_item.put(DataElement::new(
                Tag(0x0028, 0x0030),
                VR::DS,
                PrimitiveValue::from(ps_ds.as_str()),
            ));
        }
        if let Some(st) = seg.slice_thickness {
            let st_ds = st.to_string();
            px_item.put(DataElement::new(
                Tag(0x0018, 0x0050),
                VR::DS,
                PrimitiveValue::from(st_ds.as_str()),
            ));
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
        seg_id_item.put(DataElement::new(
            Tag(0x0062, 0x000B),
            VR::US,
            PrimitiveValue::from(referenced_segment_number),
        ));
        let seg_id_seq = DataSetSequence::new(vec![seg_id_item], Length::UNDEFINED);
        frame_item.put(DataElement::new(
            Tag(0x0062, 0x000A),
            VR::SQ,
            Value::from(seg_id_seq),
        ));

        if let Some(Some(pos)) = seg.image_position_per_frame.get(frame_idx) {
            let mut pos_item = InMemDicomObject::new_empty();
            let pos_ds = format!("{}\\{}\\{}", pos[0], pos[1], pos[2]);
            pos_item.put(DataElement::new(
                Tag(0x0020, 0x0032),
                VR::DS,
                PrimitiveValue::from(pos_ds.as_str()),
            ));
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

    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));

    let path = path.as_ref();
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax(EXPLICIT_VR_LE),
    )
    .with_context(|| "build DICOM-SEG file meta")?
    .write_to_file(path)
    .with_context(|| format!("write DICOM-SEG to {}", path.display()))?;

    Ok(())
}
