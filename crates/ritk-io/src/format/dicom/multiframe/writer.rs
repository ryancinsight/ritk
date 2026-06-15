//! Multi-frame DICOM writer: serializes a 3-D image as a single DICOM Part 10 file.

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use std::path::Path;

use super::types::{MultiFrameSpatialMetadata, MultiFrameWriterConfig};
use crate::format::dicom::transfer_syntax::EXPLICIT_VR_LE;

/// Generate a DICOM UID using nanoseconds since UNIX epoch under the 2.25 root.
///
/// Invariant: uniqueness holds within a single process under non-repeating system clock.
fn generate_multiframe_uid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: std::sync::atomic::AtomicU64 = AtomicU64::new(0);

    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Format: 2.25.<ns>.<seq> — both components are numeric; total ≤ 64 chars.
    format!("2.25.{}.{}", t, n)
}

/// Write a 3-D `Image<B, 3>` with shape `[n_frames, rows, cols]` as a single
/// multi-frame DICOM Part 10 file.
///
/// ## Invariants
/// - `n_frames >= 1`, `rows >= 1`, `cols >= 1`; returns `Err` otherwise.
/// - A single linear rescale (slope/intercept) maps the full f32 volume to
///   the [0, 65535] u16 range. When max == min, slope = 1.0 and
///   intercept = min_val (flat-image degenerate case).
/// - The emitted file is readable by `load_dicom_multiframe` (round-trip
///   invariant: abs(recovered - original) <= rescale_slope + 1.0).
///
/// ## Encoding
/// Transfer syntax: Explicit VR Little Endian (1.2.840.10008.1.2.1).
/// SOP Class: Multi-Frame Grayscale Word Secondary Capture Image Storage
/// (1.2.840.10008.5.1.4.1.1.7.3).
pub fn write_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
) -> Result<()> {
    write_multiframe_impl(path.as_ref(), image, &MultiFrameWriterConfig::default())
}

/// Write a 3-D `Image<B, 3>` as a multi-frame DICOM file with optional spatial metadata.
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
    image: &Image<B, 3>,
    spatial: Option<&MultiFrameSpatialMetadata>,
) -> Result<()> {
    let config = MultiFrameWriterConfig {
        spatial: spatial.cloned(),
        ..MultiFrameWriterConfig::default()
    };
    write_multiframe_impl(path.as_ref(), image, &config)
}

/// Write a 3-D `Image<B, 3>` as a multi-frame DICOM file with full writer configuration.
///
/// Accepts a [`MultiFrameWriterConfig`] for SOP class override, spatial metadata,
/// and instance number. When `config.spatial` is `None`, no spatial tags are emitted.
///
/// ## Invariants
/// - `n_frames >= 1`, `rows >= 1`, `cols >= 1`; returns `Err` otherwise.
/// - Round-trip invariant: |recovered − original| ≤ rescale_slope + 1.0.
pub fn write_dicom_multiframe_with_config<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    write_multiframe_impl(path.as_ref(), image, config)
}

fn write_multiframe_impl<B: Backend>(
    path: &Path,
    image: &Image<B, 3>,
    config: &MultiFrameWriterConfig,
) -> Result<()> {
    let [n_frames, rows, cols] = image.shape();
    if n_frames == 0 || rows == 0 || cols == 0 {
        bail!(
            "DICOM multiframe write: n_frames={} rows={} cols={} must all be >0",
            n_frames,
            rows,
            cols
        );
    }

    let all_data = image
        .try_data_vec()
        .context("DICOM multiframe writer requires f32 image data")?;

    let (min_val, max_val) = all_data
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
            (mn.min(v), mx.max(v))
        });

    let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
        (1.0_f32, min_val)
    } else {
        ((max_val - min_val) / 65535.0_f32, min_val)
    };

    let pixel_u16: Vec<u16> = all_data
        .iter()
        .map(|&v| {
            ((v - rescale_intercept) / rescale_slope)
                .round()
                .clamp(0.0, 65535.0) as u16
        })
        .collect();

    let sop_instance_uid = generate_multiframe_uid();
    let study_instance_uid = generate_multiframe_uid();
    let series_instance_uid = generate_multiframe_uid();

    let modality_str = config
        .spatial
        .as_ref()
        .map(|s| s.modality.as_str())
        .unwrap_or("OT");

    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(config.sop_class_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid.as_str()),
    ));

    // Patient Module — Type 2 mandatory (PS3.3 C.7.1.1)
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

    // General Study Module — Type 1/2 mandatory (PS3.3 C.7.2.1)
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from(study_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0020),
        VR::DA,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0090),
        VR::PN,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0010),
        VR::SH,
        PrimitiveValue::from(""),
    ));

    // General Series Module — Type 1/2 mandatory (PS3.3 C.7.3.1)
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from(series_instance_uid.as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0011),
        VR::IS,
        PrimitiveValue::from(""),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from(modality_str),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0064),
        VR::CS,
        PrimitiveValue::from("WSD"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from(format!("{}", config.instance_number)),
    ));

    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(format!("{}", n_frames)),
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
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(15_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
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

    if let Some(s) = &config.spatial {
        let o = &s.origin;
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}\\{:.6}\\{:.6}", o[0], o[1], o[2])),
        ));

        let iop = &s.image_orientation;
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(format!(
                "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
                iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]
            )),
        ));

        let ps = &s.pixel_spacing;
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}\\{:.6}", ps[0], ps[1])),
        ));
        obj.put(DataElement::new(
            Tag(0x0018, 0x0050),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", s.slice_thickness)),
        ));
    }

    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
    ));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(config.sop_class_uid.as_str())
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax(EXPLICIT_VR_LE),
        )
        .map_err(|e| anyhow::anyhow!("DICOM multiframe meta build failed: {e}"))?;

    file_obj
        .write_to_file(path)
        .map_err(|e| anyhow::anyhow!("DICOM multiframe write to {:?} failed: {e}", path))?;

    Ok(())
}
