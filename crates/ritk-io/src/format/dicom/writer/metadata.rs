use super::super::reader::DicomReadMetadata;
use super::decimal_string::{format_dicom_decimal, format_pair, format_six, format_triplet};
use super::error::DicomWriteError;
use super::output::{serialize_file, write_series_files};
use super::pixel_encoding::{
    emit_pixel_format_tags, generate_instance_uid, generate_series_uid, prepare_frame_encodings,
    validate_image_shape, validate_spatial_metadata, writer_exclusion_tags,
    DICOM_SOP_CLASS_SECONDARY_CAPTURE,
};
use super::pixel_preflight::{pixel_bit_description_is_valid, preflight_native_pixel_data};
use super::preservation::emit_preservation_nodes;
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;
use crate::format::dicom::writer::elements::PutValue;
use anyhow::Result;
use dicom::core::smallvec::SmallVec;
use dicom::core::{PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use eunomia::FloatElement;
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
///
/// Encoded samples are unsigned 16-bit. Source bit attributes are validated,
/// but the output bit attributes always describe those encoded samples.
/// All slices are serialized before the output directory or files change;
/// this retains the serialized volume in memory. Persistence errors can leave
/// partial output, but input and serialization errors preserve existing files.
pub fn write_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
    metadata: Option<&DicomReadMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    let all_data = image.data_cow_on(&B::default()).into_owned();
    let dimensions = validate_image_shape(image.shape(), all_data.len())?;
    validate_source_pixel_description(metadata)?;
    let pixel_plans = prepare_frame_encodings::<u16>(&all_data, dimensions)?;

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
    validate_spatial_metadata(&spacing, &origin, &direction)?;
    validate_scalar_photometric_interpretation(photometric)?;
    // Slice normal is column 0 of direction matrix = direction[0..3] = NÌ‚.
    let normal = [direction[0], direction[1], direction[2]];

    if metadata.is_some() {
        for z in 0..dimensions.depth {
            let zf = f64::from_count(z);
            let position = [
                origin[0] + zf * spacing[0] * normal[0],
                origin[1] + zf * spacing[0] * normal[1],
                origin[2] + zf * spacing[0] * normal[2],
            ];
            if position.iter().any(|coordinate| !coordinate.is_finite()) {
                return Err(DicomWriteError::InvalidSpatialMetadata.into());
            }
        }
    }

    let mut slices = Vec::new();
    slices
        .try_reserve_exact(dimensions.depth)
        .map_err(|_| DicomWriteError::PixelAllocationFailed)?;

    for z in 0..dimensions.depth {
        let slice_offset = z
            .checked_mul(dimensions.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let slice_end = slice_offset
            .checked_add(dimensions.frame_samples)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let slice_f32 =
            all_data
                .get(slice_offset..slice_end)
                .ok_or(DicomWriteError::PixelCountMismatch {
                    expected: dimensions.total_samples,
                    actual: all_data.len(),
                })?;
        let plan = pixel_plans
            .get(z)
            .copied()
            .ok_or(DicomWriteError::PixelCountMismatch {
                expected: dimensions.depth,
                actual: pixel_plans.len(),
            })?;
        let pixel_u16 = plan.encode(slice_f32, slice_offset)?;
        let rescale_slope = plan.rescale_slope();
        let rescale_intercept = plan.rescale_intercept();

        let sop_instance_uid = generate_instance_uid(series_uid, z);
        let mut obj = InMemDicomObject::new_empty();

        obj.put_value(Tag(0x0008, 0x0016), VR::UI, sop_class);
        obj.put_value(Tag(0x0008, 0x0018), VR::UI, sop_instance_uid.as_str());
        obj.put_value(Tag(0x0008, 0x0060), VR::CS, modality);
        obj.put_value(Tag(0x0008, 0x0064), VR::CS, "WSD");
        obj.put_value(Tag(0x0020, 0x000D), VR::UI, study_uid);
        obj.put_value(Tag(0x0020, 0x000E), VR::UI, series_uid);
        obj.put_value(Tag(0x0020, 0x0013), VR::IS, format!("{}", z + 1));

        obj.put_value(Tag(0x0028, 0x0002), VR::US, 1_u16);
        obj.put_value(Tag(0x0028, 0x0010), VR::US, dimensions.rows_attribute);
        obj.put_value(Tag(0x0028, 0x0011), VR::US, dimensions.columns_attribute);
        emit_pixel_format_tags::<u16>(&mut obj);
        obj.put_value(
            Tag(0x0028, 0x1053),
            VR::DS,
            format_dicom_decimal(f64::from(rescale_slope))?,
        );
        obj.put_value(
            Tag(0x0028, 0x1052),
            VR::DS,
            format_dicom_decimal(f64::from(rescale_intercept))?,
        );
        obj.put_value(Tag(0x0028, 0x0004), VR::CS, photometric);

        if metadata.is_some() {
            let zf = f64::from_count(z);
            let ipp_x = origin[0] + zf * spacing[0] * normal[0];
            let ipp_y = origin[1] + zf * spacing[0] * normal[1];
            let ipp_z = origin[2] + zf * spacing[0] * normal[2];
            obj.put_value(
                Tag(0x0020, 0x0032),
                VR::DS,
                format_triplet([ipp_x, ipp_y, ipp_z])?,
            );
            // IOP = [F_r, F_c] = [direction[6..9], direction[3..6]]
            obj.put_value(
                Tag(0x0020, 0x0037),
                VR::DS,
                format_six([
                    direction[6],
                    direction[7],
                    direction[8],
                    direction[3],
                    direction[4],
                    direction[5],
                ])?,
            );
            // PixelSpacing = [ΔRow, ΔCol] = [spacing[1], spacing[2]]
            obj.put_value(
                Tag(0x0028, 0x0030),
                VR::DS,
                format_pair([spacing[1], spacing[2]])?,
            );
            obj.put_value(
                Tag(0x0018, 0x0050),
                VR::DS,
                format_dicom_decimal(spacing[0])?,
            );
        }

        // DICOM PS3.3 Type 2: tag must be present even when value is unknown; empty string is valid.
        obj.put_value(Tag(0x0008, 0x0090), VR::PN, "");
        obj.put_value(Tag(0x0010, 0x0010), VR::PN, "");
        obj.put_value(Tag(0x0010, 0x0020), VR::LO, "");
        obj.put_value(Tag(0x0008, 0x0020), VR::DA, "");
        obj.put_value(Tag(0x0020, 0x0011), VR::IS, "0");

        if let Some(m) = metadata {
            if let Some(ref uid) = m.frame_of_reference_uid {
                obj.put_value(Tag(0x0020, 0x0052), VR::UI, uid.as_str());
            }
            if let Some(ref pid) = m.patient_id {
                obj.put_value(Tag(0x0010, 0x0020), VR::LO, pid.as_str());
            }
            if let Some(ref pn) = m.patient_name {
                obj.put_value(Tag(0x0010, 0x0010), VR::PN, pn.as_str());
            }
            if let Some(ref sd) = m.study_date {
                obj.put_value(Tag(0x0008, 0x0020), VR::DA, sd.as_str());
            }
            if let Some(ref desc) = m.series_description {
                obj.put_value(Tag(0x0008, 0x103E), VR::LO, desc.as_str());
            }
            if let Some(ref sd) = m.series_date {
                obj.put_value(Tag(0x0008, 0x0021), VR::DA, sd.as_str());
            }
            if let Some(ref st) = m.series_time {
                obj.put_value(Tag(0x0008, 0x0031), VR::TM, st.as_str());
            }
            if let Some(private_value) = m.private_tags.get("0019,10AA") {
                obj.put_value(Tag(0x0019, 0x10AA), VR::LO, private_value.as_str());
            }
            if let Some(private_value) = m.private_tags.get("0029,10BB") {
                obj.put_value(Tag(0x0029, 0x10BB), VR::LO, private_value.as_str());
            }
        }

        // Emit preservation nodes before PixelData to maintain Image Pixel Module ordering.
        if let Some(m) = metadata
            && !m.preservation.is_empty()
        {
            let exclusion = writer_exclusion_tags();
            emit_preservation_nodes(&mut obj, &m.preservation, &exclusion);
        }
        obj.put_value(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        );
        preflight_native_pixel_data(&obj)?;

        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(sop_class)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax(EXPLICIT_VR_LE),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        slices.push(serialize_file(&file_obj)?);
    }
    write_series_files(path, &slices)
}

fn validate_source_pixel_description(metadata: Option<&DicomReadMetadata>) -> Result<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    match (
        metadata.bits_allocated,
        metadata.bits_stored,
        metadata.high_bit,
    ) {
        (None, None, None) => Ok(()),
        (Some(bits_allocated), Some(bits_stored), Some(high_bit)) => {
            if pixel_bit_description_is_valid(bits_allocated, bits_stored, high_bit) {
                Ok(())
            } else {
                Err(DicomWriteError::InvalidPixelDescription {
                    bits_allocated,
                    bits_stored,
                    high_bit,
                }
                .into())
            }
        }
        _ => Err(DicomWriteError::IncompleteSourcePixelDescription.into()),
    }
}

fn validate_scalar_photometric_interpretation(value: &str) -> Result<()> {
    if matches!(value, "MONOCHROME1" | "MONOCHROME2") {
        Ok(())
    } else {
        Err(DicomWriteError::UnsupportedPhotometricInterpretation.into())
    }
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
