use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use std::path::Path;

use super::types::{DicomSegmentation, SEG_SOP_CLASS_UID};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::elements::{put_bytes, put_is, put_sequence, put_text, put_u16};
use crate::format::dicom::writer::pixel_encoding::{generate_series_uid, MONOCHROME2};

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
            put_u16(&mut item, Tag(0x0062, 0x0004), info.segment_number);
            put_text(
                &mut item,
                Tag(0x0062, 0x0005),
                VR::LO,
                info.segment_label.as_str(),
            );
            put_text(
                &mut item,
                Tag(0x0062, 0x0006),
                VR::ST,
                info.segment_description.as_deref().unwrap_or(""),
            );
            put_text(
                &mut item,
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

    put_text(&mut obj, Tag(0x0008, 0x0016), VR::UI, SEG_SOP_CLASS_UID);
    put_text(
        &mut obj,
        Tag(0x0008, 0x0018),
        VR::UI,
        sop_instance_uid.as_str(),
    );
    put_text(&mut obj, Tag(0x0008, 0x0060), VR::CS, "SEG");
    put_text(
        &mut obj,
        Tag(0x0020, 0x000D),
        VR::UI,
        study_instance_uid.as_str(),
    );
    put_text(
        &mut obj,
        Tag(0x0020, 0x000E),
        VR::UI,
        series_instance_uid.as_str(),
    );
    put_is(&mut obj, Tag(0x0020, 0x0013), 1);
    put_is(&mut obj, Tag(0x0028, 0x0008), seg.n_frames);
    put_u16(&mut obj, Tag(0x0028, 0x0010), seg.rows as u16);
    put_u16(&mut obj, Tag(0x0028, 0x0011), seg.cols as u16);
    put_u16(&mut obj, Tag(0x0028, 0x0100), seg.bits_allocated);
    put_u16(&mut obj, Tag(0x0028, 0x0101), seg.bits_allocated);
    put_u16(&mut obj, Tag(0x0028, 0x0102), seg.bits_allocated - 1);
    put_u16(&mut obj, Tag(0x0028, 0x0103), 0);
    put_u16(&mut obj, Tag(0x0028, 0x0002), 1);
    put_text(&mut obj, Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);
    put_text(
        &mut obj,
        Tag(0x0062, 0x0001),
        VR::CS,
        seg.segmentation_type.as_dicom_str(),
    );

    if !seg_items.is_empty() {
        put_sequence(&mut obj, Tag(0x0062, 0x0002), seg_items);
    }

    let mut shared_item = InMemDicomObject::new_empty();
    let mut has_shared_fg = false;

    if let Some(iop) = seg.image_orientation {
        let mut ori_item = InMemDicomObject::new_empty();
        let iop_ds = format!(
            "{}\\{}\\{}\\{}\\{}\\{}",
            iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]
        );
        put_text(&mut ori_item, Tag(0x0020, 0x0037), VR::DS, iop_ds.as_str());
        put_sequence(&mut shared_item, Tag(0x0020, 0x9116), vec![ori_item]);
        has_shared_fg = true;
    }

    if seg.pixel_spacing.is_some() || seg.slice_thickness.is_some() {
        let mut px_item = InMemDicomObject::new_empty();
        if let Some(ps) = seg.pixel_spacing {
            let ps_ds = format!("{}\\{}", ps[0], ps[1]);
            put_text(&mut px_item, Tag(0x0028, 0x0030), VR::DS, ps_ds.as_str());
        }
        if let Some(st) = seg.slice_thickness {
            let st_ds = st.to_string();
            put_text(&mut px_item, Tag(0x0018, 0x0050), VR::DS, st_ds.as_str());
        }
        put_sequence(&mut shared_item, Tag(0x0028, 0x9110), vec![px_item]);
        has_shared_fg = true;
    }

    if has_shared_fg {
        put_sequence(&mut obj, Tag(0x5200, 0x9229), vec![shared_item]);
    }

    let mut per_frame_items: Vec<InMemDicomObject> = Vec::with_capacity(seg.n_frames);
    for frame_idx in 0..seg.n_frames {
        let mut frame_item = InMemDicomObject::new_empty();

        let referenced_segment_number = seg.frame_segment_numbers[frame_idx];
        let mut seg_id_item = InMemDicomObject::new_empty();
        put_u16(
            &mut seg_id_item,
            Tag(0x0062, 0x000B),
            referenced_segment_number,
        );
        put_sequence(&mut frame_item, Tag(0x0062, 0x000A), vec![seg_id_item]);

        if let Some(Some(pos)) = seg.image_position_per_frame.get(frame_idx) {
            let mut pos_item = InMemDicomObject::new_empty();
            let pos_ds = format!("{}\\{}\\{}", pos[0], pos[1], pos[2]);
            put_text(&mut pos_item, Tag(0x0020, 0x0032), VR::DS, pos_ds.as_str());
            put_sequence(&mut frame_item, Tag(0x0020, 0x9113), vec![pos_item]);
        }

        per_frame_items.push(frame_item);
    }
    if !per_frame_items.is_empty() {
        put_sequence(&mut obj, Tag(0x5200, 0x9230), per_frame_items);
    }

    put_bytes(&mut obj, Tag(0x7FE0, 0x0010), VR::OW, pixel_bytes);

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
