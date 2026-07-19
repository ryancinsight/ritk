use super::super::reader::DicomReadMetadata;
use super::preservation::emit_preservation_nodes;
use super::utils::{
    emit_pixel_format_tags, ensure_series_directory, format_pair, format_six, format_triplet,
    generate_instance_uid, generate_series_uid, normalize_to_u16, writer_exclusion_tags,
    DICOM_SOP_CLASS_SECONDARY_CAPTURE,
};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use anyhow::{bail, Context, Result};
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

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
    image: &Image<f32, B, 3>,
    metadata: Option<&DicomReadMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;

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

    let spacing = metadata.map(|m| m.spacing).unwrap_or([1.0, 1.0, 1.0]);
    let origin = metadata.map(|m| m.origin).unwrap_or([0.0, 0.0, 0.0]);
    let direction = metadata
        .map(|m| m.direction)
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // Slice normal is column 0 of direction matrix = direction[0..3] = NÌ‚.
    let normal = [direction[0], direction[1], direction[2]];

    let all_data = image
        .try_data_vec()
        .context("DICOM metadata writer requires f32 image data")?;
    let slice_len = rows * cols;

    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (pixel_u16, rescale_slope, rescale_intercept) = normalize_to_u16(slice_f32);

        let sop_instance_uid = generate_instance_uid(series_uid, z);
        let mut obj = InMemDicomObject::new_empty();

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
        emit_pixel_format_tags(&mut obj);
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
                // PixelSpacing = [Î”Row, Î”Col] = [spacing[1], spacing[2]]
                PrimitiveValue::from(format_pair([spacing[1], spacing[2]])),
            ));
            obj.put(DataElement::new(
                Tag(0x0018, 0x0050),
                VR::DS,
                // SliceThickness = Î”z = spacing[0]
                PrimitiveValue::from(format!("{:.6}", spacing[0])),
            ));
        }

        // DICOM PS3.3 Type 2: tag must be present even when value is unknown; empty string is valid.
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
            if let Some(private_value) = m.private_tags.get("0019,10AA") {
                obj.put(DataElement::new(
                    Tag(0x0019, 0x10AA),
                    VR::LO,
                    PrimitiveValue::from(private_value.as_str()),
                ));
            }
            if let Some(private_value) = m.private_tags.get("0029,10BB") {
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

pub struct DicomWriter<B> {
    _phantom: PhantomData<fn() -> B>,
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
