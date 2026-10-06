//! Multi-frame DICOM writer: serializes a 3-D image as a single DICOM Part 10 file.

use crate::format::dicom::writer::elements::PutValue;
use crate::format::dicom::writer::pixel_encoding::{
    JPEG_2000_QUANTIZATION_STEP, JPEG_BASELINE_QUALITY, JPEG_LS_NEAR,
};
use anyhow::{bail, Context, Result};
use coeus_core::MoiraiBackend;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::PixelFragmentSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_codecs::encode_jpeg_fragment;
use ritk_codecs::encode_rle_lossless_fragment_u16_grayscale;
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, Jpeg2000Encoding};
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;
use ritk_codecs::{PixelLayout, PixelSignedness};
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use std::path::Path;

use super::types::{MultiFrameSpatialMetadata, MultiFrameWriterConfig};
use crate::format::dicom::writer::pixel_encoding::{
    emit_pixel_format_tags, generate_series_uid, normalize_pixels, MONOCHROME2,
};

/// Write a 3-D `Image<f32, B, 3>` with shape `[n_frames, rows, cols]` as a single
/// multi-frame DICOM Part 10 file.
///
/// ## Invariants
/// - `n_frames >= 1`, `rows >= 1`, `cols >= 1`; returns `Err` otherwise.
/// - A single linear rescale (slope/intercept) maps the full f32 volume to
///   the [0, 65535] u16 range. When max == min, slope ≈ ε/65535 and
///   intercept = min_val (flat-image degenerate case; reconstruction is exact).
/// - The emitted file is readable by `load_dicom_multiframe` (round-trip
///   invariant: abs(recovered - original) <= rescale_slope + 1.0).
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
/// - Round-trip invariant: |recovered − original| ≤ rescale_slope + 1.0.
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
    let [n_frames, rows, cols] = shape;
    if n_frames == 0 || rows == 0 || cols == 0 {
        bail!(
            "DICOM multiframe write: n_frames={} rows={} cols={} must all be >0",
            n_frames,
            rows,
            cols
        );
    }

    let (pixel_u16, rescale_slope, rescale_intercept) = normalize_pixels::<u16>(all_data)?;

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
    obj.put_value(Tag(0x0028, 0x0010), VR::US, rows as u16);
    obj.put_value(Tag(0x0028, 0x0011), VR::US, cols as u16);
    emit_pixel_format_tags::<u16>(&mut obj);
    obj.put_value(Tag(0x0028, 0x0004), VR::CS, MONOCHROME2);
    obj.put_value(Tag(0x0028, 0x1053), VR::DS, format!("{:.6}", rescale_slope));
    obj.put_value(
        Tag(0x0028, 0x1052),
        VR::DS,
        format!("{:.6}", rescale_intercept),
    );

    if let Some(s) = &config.spatial {
        let o = &s.origin;
        obj.put_value(
            Tag(0x0020, 0x0032),
            VR::DS,
            format!("{:.6}\\{:.6}\\{:.6}", o[0], o[1], o[2]),
        );

        let iop = &s.image_orientation;
        obj.put_value(
            Tag(0x0020, 0x0037),
            VR::DS,
            format!(
                "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
                iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]
            ),
        );

        let ps = &s.pixel_spacing;
        obj.put_value(
            Tag(0x0028, 0x0030),
            VR::DS,
            format!("{:.6}\\{:.6}", ps[0], ps[1]),
        );
        obj.put_value(
            Tag(0x0018, 0x0050),
            VR::DS,
            format!("{:.6}", s.slice_thickness),
        );
    }

    match &config.transfer_syntax {
        TransferSyntaxKind::ExplicitVrLittleEndian => {
            obj.put_value(
                Tag(0x7FE0, 0x0010),
                VR::OW,
                PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
            );
        }
        TransferSyntaxKind::JpegLsLossless
        | TransferSyntaxKind::JpegLsLossy
        | TransferSyntaxKind::Jpeg2000Lossless
        | TransferSyntaxKind::Jpeg2000Lossy
        | TransferSyntaxKind::RleLossless => {
            let encoded_fragments = encode_compressed_frames(
                &pixel_u16,
                n_frames,
                rows,
                cols,
                &config.transfer_syntax,
            )?;
            obj.put(DataElement::new(
                Tag(0x7FE0, 0x0010),
                VR::OB,
                PixelFragmentSequence::<Vec<u8>>::new_fragments(SmallVec::from_vec(
                    encoded_fragments,
                )),
            ));
        }
        TransferSyntaxKind::JpegBaseline => {
            // Baseline JPEG carries eight-bit samples, so the 16-bit
            // normalisation and the 16-bit pixel tags do not apply to this
            // transfer syntax. Re-derive both from the modality data.
            let (pixel_u8, jpeg_slope, jpeg_intercept) = normalize_pixels::<u8>(all_data)?;
            emit_pixel_format_tags::<u8>(&mut obj);
            let layout = PixelLayout {
                rows,
                cols,
                samples_per_pixel: 1,
                bits_allocated: 8,
                bits_stored: 8,
                pixel_representation: PixelSignedness::Unsigned,
                // `pixel_u8` already holds *stored* samples, so the encoder's
                // layout must be the identity. Carrying the modality rescale
                // here would make `encode_jpeg_fragment` invert it a second
                // time, scaling each sample by 255/range on the way in. The
                // rescale belongs in the DICOM tags below, which is where a
                // reader looks for it.
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            };
            let mut fragments = Vec::with_capacity(n_frames);
            let frame_pixels = rows * cols;
            for frame_index in 0..n_frames {
                let start = frame_index * frame_pixels;
                let frame = pixel_u8[start..start + frame_pixels]
                    .iter()
                    .map(|&v| f32::from(v))
                    .collect::<Vec<_>>();
                let fragment = encode_jpeg_fragment(&frame, layout, JPEG_BASELINE_QUALITY)
                    .with_context(|| {
                        format!("JPEG baseline encode failed for frame {frame_index}")
                    })?;
                fragments.push(fragment);
            }
            obj.put(DataElement::new(
                Tag(0x7FE0, 0x0010),
                VR::OB,
                PixelFragmentSequence::<Vec<u8>>::new_fragments(SmallVec::from_vec(fragments)),
            ));
            if rescale_slope != jpeg_slope || rescale_intercept != jpeg_intercept {
                obj.put_value(Tag(0x0028, 0x1053), VR::DS, format!("{:.10}", jpeg_slope));
                obj.put_value(
                    Tag(0x0028, 0x1052),
                    VR::DS,
                    format!("{:.10}", jpeg_intercept),
                );
            }
        }
        syntax => {
            bail!(
                "DICOM multiframe write transfer syntax '{}' is not supported; supported output syntaxes are Explicit VR Little Endian, JPEG Baseline, \n                JPEG-LS Lossless, JPEG-LS Lossy (near-lossless), JPEG 2000 Lossless, \n                JPEG 2000 Lossy, and RLE Lossless",
                syntax.uid()
            );
        }
    }

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(config.sop_class_uid.as_str())
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax(config.transfer_syntax.uid()),
        )
        .map_err(|e| anyhow::anyhow!("DICOM multiframe meta build failed: {e}"))?;

    file_obj
        .write_to_file(path)
        .map_err(|e| anyhow::anyhow!("DICOM multiframe write to {:?} failed: {e}", path))?;

    Ok(())
}

fn encode_compressed_frames(
    pixels_u16: &[u16],
    n_frames: usize,
    rows: usize,
    cols: usize,
    syntax: &TransferSyntaxKind,
) -> Result<Vec<Vec<u8>>> {
    let frame_pixels = rows
        .checked_mul(cols)
        .context("DICOM multiframe compressed write frame size overflow")?;
    let expected = n_frames
        .checked_mul(frame_pixels)
        .context("DICOM multiframe compressed write total size overflow")?;
    if pixels_u16.len() != expected {
        bail!(
            "DICOM multiframe compressed write expected {} u16 pixels for shape [{}, {}, {}], got {}",
            expected,
            n_frames,
            rows,
            cols,
            pixels_u16.len()
        );
    }

    let mut fragments = Vec::with_capacity(n_frames);
    for frame_index in 0..n_frames {
        let start = frame_index
            .checked_mul(frame_pixels)
            .context("DICOM multiframe compressed write frame offset overflow")?;
        let end = start + frame_pixels;
        let frame = &pixels_u16[start..end];

        let encoded = match syntax {
            TransferSyntaxKind::JpegLsLossless => {
                encode_grayscale_jpeg_ls(frame, rows as u32, cols as u32, 16, 0).with_context(
                    || format!("JPEG-LS lossless encode failed for frame {frame_index}"),
                )?
            }
            TransferSyntaxKind::Jpeg2000Lossless => {
                let frame_i32: Vec<i32> = frame.iter().map(|&v| i32::from(v)).collect();
                encode_grayscale_j2k(
                    &frame_i32,
                    rows as u32,
                    cols as u32,
                    16,
                    ritk_dicom::PixelSignedness::Unsigned,
                    Jpeg2000Encoding::Lossless {
                        decomposition_levels: 1,
                    },
                )
                .with_context(|| {
                    format!("JPEG 2000 lossless encode failed for frame {frame_index}")
                })?
            }
            TransferSyntaxKind::JpegLsLossy => {
                encode_grayscale_jpeg_ls(frame, rows as u32, cols as u32, 16, JPEG_LS_NEAR)
                    .with_context(|| {
                        format!("JPEG-LS near-lossless encode failed for frame {frame_index}")
                    })?
            }
            TransferSyntaxKind::Jpeg2000Lossy => {
                let frame_i32: Vec<i32> = frame.iter().map(|&v| i32::from(v)).collect();
                encode_grayscale_j2k(
                    &frame_i32,
                    rows as u32,
                    cols as u32,
                    16,
                    ritk_dicom::PixelSignedness::Unsigned,
                    Jpeg2000Encoding::Lossy {
                        decomposition_levels: 1,
                        quantization_step: JPEG_2000_QUANTIZATION_STEP,
                    },
                )
                .with_context(|| format!("JPEG 2000 lossy encode failed for frame {frame_index}"))?
            }
            TransferSyntaxKind::RleLossless => encode_rle_lossless_fragment_u16_grayscale(frame),
            _ => bail!(
                "internal error: compressed frame encoder called with non-compressed syntax '{}'",
                syntax.uid()
            ),
        };
        fragments.push(encoded);
    }
    Ok(fragments)
}
