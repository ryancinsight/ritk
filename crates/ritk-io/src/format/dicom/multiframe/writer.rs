//! Multi-frame DICOM writer: serializes a 3-D image as a single DICOM Part 10 file.

use crate::format::dicom::writer::decimal_string::{
    format_dicom_decimal, format_pair, format_six, format_triplet,
};
use crate::format::dicom::writer::elements::PutValue;
use anyhow::{bail, Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use std::path::Path;

use super::types::{MultiFrameSpatialMetadata, MultiFrameWriterConfig};
use crate::format::dicom::writer::pixel_encoding::{
    emit_pixel_format_tags, generate_series_uid, validate_image_shape, validate_spatial_metadata,
    PixelEncodingPlan, MONOCHROME2,
};
mod compression;
use compression::{encode_baseline_jpeg_frames, encode_compressed_frames};

enum EncodedPixelSamples {
    NativeWord(Vec<u16>, PixelEncodingPlan<u16>),
    EncapsulatedWord(Vec<u16>, PixelEncodingPlan<u16>),
    BaselineByte(Vec<u8>, PixelEncodingPlan<u8>),
}

/// Write a 3-D `Image<f32, B, 3>` with shape `[n_frames, rows, cols]` as a single
/// multi-frame DICOM Part 10 file.
///
/// ## Invariants
/// - `n_frames >= 1`, `rows >= 1`, `cols >= 1`; returns `Err` otherwise.
/// - A single linear rescale maps the full finite f32 volume to unsigned
///   16-bit samples, or unsigned 8-bit samples for baseline JPEG.
/// - Constant input stores zero with the constant as intercept.
/// - Pixel bit attributes describe the encoded sample type.
/// - Input validation and serialization finish before the output file changes.
///
/// ## Encoding
/// Defaults to Explicit VR Little Endian (1.2.840.10008.1.2.1). Use
/// [`MultiFrameWriterConfig::transfer_syntax`] for JPEG-LS/JPEG 2000/RLE
/// lossless compressed output.
pub fn write_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
) -> Result<()> {
    write_multiframe_impl(path.as_ref(), image, &MultiFrameWriterConfig::default())
}

/// Write a 3-D `Image<f32, B, 3>` as a multi-frame DICOM file with optional spatial metadata.
///
/// When `spatial` is `None`, behaves identically to [`write_dicom_multiframe`].
/// When `spatial` is `Some`, also emits:
/// - (0020,0032) ImagePositionPatient
/// - (0020,0037) ImageOrientationPatient
/// - (0028,0030) PixelSpacing
/// - (0018,0050) SliceThickness
/// - (0008,0060) Modality (overrides default "OT")
pub fn write_dicom_multiframe_with_options<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
    spatial: Option<&MultiFrameSpatialMetadata>,
) -> Result<()> {
    let config = MultiFrameWriterConfig {
        spatial: spatial.cloned(),
        ..MultiFrameWriterConfig::default()
    };
    write_multiframe_impl(path.as_ref(), image, &config)
}

/// Write a 3-D `Image<f32, B, 3>` as a multi-frame DICOM file with full writer configuration.
///
/// Accepts a [`MultiFrameWriterConfig`] for SOP class override, spatial metadata,
/// and instance number. When `config.spatial` is `None`, no spatial tags are emitted.
///
/// ## Invariants
/// - `n_frames >= 1`, `rows >= 1`, `cols >= 1`; returns `Err` otherwise.
/// - Non-finite pixels, unrepresentable ranges, and invalid spatial metadata
///   are rejected before the output file changes.
pub fn write_dicom_multiframe_with_config<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<f32, B, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    write_multiframe_impl(path.as_ref(), image, config)
}

/// Write a native `Image<f32, MoiraiBackend, 3>` with shape `[n_frames, rows,
/// cols]` as a single multi-frame DICOM Part 10 file.
///
/// Native counterpart of [`write_dicom_multiframe`]: identical byte output for
/// identical voxels (both route through the shared substrate-free encode core),
/// differing only in the source image carrier.
pub fn write_dicom_multiframe_native<P: AsRef<Path>>(
    path: P,
    image: &NativeImage<f32, MoiraiBackend, 3>,
) -> Result<()> {
    write_native_impl(path.as_ref(), image, &MultiFrameWriterConfig::default())
}

/// Write a native `Image<f32, MoiraiBackend, 3>` as a multi-frame DICOM file
/// with optional spatial metadata. Native counterpart of
/// [`write_dicom_multiframe_with_options`].
pub fn write_dicom_multiframe_native_with_options<P: AsRef<Path>>(
    path: P,
    image: &NativeImage<f32, MoiraiBackend, 3>,
    spatial: Option<&MultiFrameSpatialMetadata>,
) -> Result<()> {
    let config = MultiFrameWriterConfig {
        spatial: spatial.cloned(),
        ..MultiFrameWriterConfig::default()
    };
    write_native_impl(path.as_ref(), image, &config)
}

/// Write a native `Image<f32, MoiraiBackend, 3>` as a multi-frame DICOM file
/// with full writer configuration. Native counterpart of
/// [`write_dicom_multiframe_with_config`].
pub fn write_dicom_multiframe_native_with_config<P: AsRef<Path>>(
    path: P,
    image: &NativeImage<f32, MoiraiBackend, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    write_native_impl(path.as_ref(), image, config)
}

fn write_multiframe_impl<B: Backend>(
    path: &Path,
    image: &Image<f32, B, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    let all_data = image.data_cow_on(&B::default()).into_owned();
    write_multiframe_flat(path, &all_data, image.shape(), config)
}

fn write_native_impl(
    path: &Path,
    image: &NativeImage<f32, MoiraiBackend, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    let data = image
        .data_slice()
        .context("DICOM multiframe writer requires contiguous f32 image data")?;
    write_multiframe_flat(path, data, image.shape(), config)
}

/// Serialize a flat `[n_frames, rows, cols]` row-major `f32` buffer as a
/// multi-frame DICOM Part 10 file.
///
/// Substrate-free encode core shared by the Coeus and native writers. The single
/// global linear rescale, tag emission, and file layout are defined here so the
/// two carriers produce byte-identical output for identical voxels.
fn write_multiframe_flat(
    path: &Path,
    all_data: &[f32],
    shape: [usize; 3],
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    let dimensions = validate_image_shape(shape, all_data.len())?;
    let n_frames = dimensions.depth;
    let spatial_tags = if let Some(spatial) = &config.spatial {
        validate_spatial_metadata(
            &[
                spatial.slice_thickness,
                spatial.pixel_spacing[0],
                spatial.pixel_spacing[1],
            ],
            &spatial.origin,
            &spatial.image_orientation,
        )?;
        Some((
            format_triplet(spatial.origin)?,
            format_six(spatial.image_orientation)?,
            format_pair(spatial.pixel_spacing)?,
            format_dicom_decimal(spatial.slice_thickness)?,
        ))
    } else {
        None
    };
    let encoded_pixels = match &config.transfer_syntax {
        TransferSyntaxKind::ExplicitVrLittleEndian => {
            let plan = PixelEncodingPlan::<u16>::prepare(all_data, 0)?;
            EncodedPixelSamples::NativeWord(plan.encode(all_data, 0)?, plan)
        }
        TransferSyntaxKind::JpegLsLossless
        | TransferSyntaxKind::JpegLsLossy
        | TransferSyntaxKind::JpegLosslessFirstOrderPrediction
        | TransferSyntaxKind::JpegLosslessNonHierarchical
        | TransferSyntaxKind::Jpeg2000Lossless
        | TransferSyntaxKind::Jpeg2000Lossy
        | TransferSyntaxKind::RleLossless => {
            let plan = PixelEncodingPlan::<u16>::prepare(all_data, 0)?;
            EncodedPixelSamples::EncapsulatedWord(plan.encode(all_data, 0)?, plan)
        }
        TransferSyntaxKind::JpegBaseline => {
            let plan = PixelEncodingPlan::<u8>::prepare(all_data, 0)?;
            EncodedPixelSamples::BaselineByte(plan.encode(all_data, 0)?, plan)
        }
        syntax => bail!(
            "DICOM multiframe write transfer syntax '{}' is not supported; supported output syntaxes are Explicit VR Little Endian, JPEG Baseline, JPEG-LS Lossless, JPEG-LS Lossy (near-lossless), JPEG 2000 Lossless, JPEG 2000 Lossy, JPEG Lossless (first-order prediction), and RLE Lossless",
            syntax.uid()
        ),
    };

    let sop_instance_uid = generate_series_uid();
    let study_instance_uid = generate_series_uid();
    let series_instance_uid = generate_series_uid();

    let modality_str = config
        .spatial
        .as_ref()
        .map(|s| s.modality.as_str())
        .unwrap_or("OT");

    let mut obj = InMemDicomObject::new_empty();

    obj.put_value(Tag(0x0008, 0x0016), VR::UI, config.sop_class_uid.as_str());
    obj.put_value(Tag(0x0008, 0x0018), VR::UI, sop_instance_uid.as_str());

    // Patient Module — Type 2 mandatory (PS3.3 C.7.1.1)
    obj.put_value(Tag(0x0010, 0x0010), VR::PN, "");
    obj.put_value(Tag(0x0010, 0x0020), VR::LO, "");

    // General Study Module — Type 1/2 mandatory (PS3.3 C.7.2.1)
    obj.put_value(Tag(0x0020, 0x000D), VR::UI, study_instance_uid.as_str());
    obj.put_value(Tag(0x0008, 0x0020), VR::DA, "");
    obj.put_value(Tag(0x0008, 0x0090), VR::PN, "");
    obj.put_value(Tag(0x0020, 0x0010), VR::SH, "");

    // General Series Module — Type 1/2 mandatory (PS3.3 C.7.3.1)
    obj.put_value(Tag(0x0020, 0x000E), VR::UI, series_instance_uid.as_str());
    obj.put_value(Tag(0x0020, 0x0011), VR::IS, "");
    obj.put_value(Tag(0x0008, 0x0060), VR::CS, modality_str);
    obj.put_value(Tag(0x0008, 0x0064), VR::CS, "WSD");
    obj.put_value(
        Tag(0x0020, 0x0013),
        VR::IS,
        format!("{}", config.instance_number),
    );

    obj.put_value(Tag(0x0028, 0x0008), VR::IS, format!("{}", n_frames));
    obj.put_value(Tag(0x0028, 0x0002), VR::US, 1_u16);
    obj.put_value(Tag(0x0028, 0x0010), VR::US, dimensions.rows_attribute);
    obj.put_value(Tag(0x0028, 0x0011), VR::US, dimensions.columns_attribute);
    match &encoded_pixels {
        EncodedPixelSamples::NativeWord(_, plan)
        | EncodedPixelSamples::EncapsulatedWord(_, plan) => {
            emit_pixel_format_tags::<u16>(&mut obj);
            obj.put_value(
                Tag(0x0028, 0x1053),
                VR::DS,
                format_dicom_decimal(f64::from(plan.rescale_slope()))?,
            );
            obj.put_value(
                Tag(0x0028, 0x1052),
                VR::DS,
                format_dicom_decimal(f64::from(plan.rescale_intercept()))?,
            );
        }
        EncodedPixelSamples::BaselineByte(_, plan) => {
            emit_pixel_format_tags::<u8>(&mut obj);
            obj.put_value(
                Tag(0x0028, 0x1053),
                VR::DS,
                format_dicom_decimal(f64::from(plan.rescale_slope()))?,
            );
            obj.put_value(
                Tag(0x0028, 0x1052),
                VR::DS,
                format_dicom_decimal(f64::from(plan.rescale_intercept()))?,
            );
        }
    }
    obj.put_value(Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);

    if let Some((origin, orientation, pixel_spacing, slice_thickness)) = spatial_tags {
        obj.put_value(Tag(0x0020, 0x0032), VR::DS, origin);
        obj.put_value(Tag(0x0020, 0x0037), VR::DS, orientation);
        obj.put_value(Tag(0x0028, 0x0030), VR::DS, pixel_spacing);
        obj.put_value(Tag(0x0018, 0x0050), VR::DS, slice_thickness);
    }

    match (&config.transfer_syntax, encoded_pixels) {
        (
            TransferSyntaxKind::ExplicitVrLittleEndian,
            EncodedPixelSamples::NativeWord(pixel_u16, _),
        ) => {
            obj.put_value(
                Tag(0x7FE0, 0x0010),
                VR::OW,
                PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
            );
        }
        (
            TransferSyntaxKind::JpegLsLossless
            | TransferSyntaxKind::JpegLsLossy
            | TransferSyntaxKind::JpegLosslessFirstOrderPrediction
            | TransferSyntaxKind::JpegLosslessNonHierarchical
            | TransferSyntaxKind::Jpeg2000Lossless
            | TransferSyntaxKind::Jpeg2000Lossy
            | TransferSyntaxKind::RleLossless,
            EncodedPixelSamples::EncapsulatedWord(pixel_u16, _),
        ) => {
            let encoded_fragments =
                encode_compressed_frames(&pixel_u16, dimensions, &config.transfer_syntax)?;
            obj.put(DataElement::new(
                Tag(0x7FE0, 0x0010),
                VR::OB,
                encoded_fragments,
            ));
        }
        (TransferSyntaxKind::JpegBaseline, EncodedPixelSamples::BaselineByte(pixel_u8, _)) => {
            let fragments = encode_baseline_jpeg_frames(&pixel_u8, dimensions)?;
            obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, fragments));
        }
        _ => bail!("DICOM transfer syntax and prepared pixel sample type disagree"),
    }

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(config.sop_class_uid.as_str())
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax(config.transfer_syntax.uid()),
        )
        .map_err(|e| anyhow::anyhow!("DICOM multiframe meta build failed: {e}"))?;

    crate::format::dicom::writer::output::write_file(path, &file_obj)
}
