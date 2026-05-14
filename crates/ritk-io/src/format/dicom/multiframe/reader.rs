//! Multi-frame DICOM reader: header extraction, functional group parsing, and volume loading.

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::InMemDicomObject;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::{
    decode_frame_with, parse_file_with, DecodeFrameRequest, DicomRsBackend, PixelLayout,
    TransferSyntaxKind,
};
use std::path::Path;

use super::types::{MultiFrameInfo, PerFrameInfo};

/// Parse a `\`-separated DICOM Decimal String (DS) field into a fixed-size array.
///
/// # Invariant
/// Returns `Some(arr)` iff the input contains at least `N` parseable `f64` values
/// separated by `\`. Non-numeric tokens are skipped. Returns `None` if fewer than
/// `N` valid numeric components exist.
pub(crate) fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
    let parts: Vec<f64> = s
        .trim()
        .split('\\')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .collect();
    if parts.len() >= N {
        let mut arr = [0.0_f64; N];
        for i in 0..N {
            arr[i] = parts[i];
        }
        Some(arr)
    } else {
        None
    }
}

/// Extract all multi-frame header fields from an already-opened DICOM object.
///
/// # Invariants
/// - n_frames defaults to 1 when (0028,0008) is absent.
/// - bits_allocated defaults to 16 when absent.
/// - rescale_slope defaults to 1.0, rescale_intercept to 0.0 when absent.
/// - per_frame is always Vec::new(); call extract_functional_groups separately.
pub(crate) fn extract_multiframe_header(path: &Path, obj: &InMemDicomObject) -> MultiFrameInfo {
    let n_frames: usize = obj
        .element(Tag(0x0028, 0x0008))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);

    let rows: usize = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let cols: usize = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let bits_allocated: u16 = obj
        .element(Tag(0x0028, 0x0100))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(16);

    let samples_per_pixel: usize = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);

    let pixel_representation: u16 = obj
        .element(Tag(0x0028, 0x0103))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let pixel_spacing = obj
        .element(Tag(0x0028, 0x0030))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<2>(&s)));

    let frame_thickness = obj
        .element(Tag(0x0018, 0x0050))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok());

    let modality = obj
        .element(Tag(0x0008, 0x0060))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty());

    let sop_class_uid = obj
        .element(Tag(0x0008, 0x0016))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty());

    let image_position = obj
        .element(Tag(0x0020, 0x0032))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<3>(&s)));

    let image_orientation = obj
        .element(Tag(0x0020, 0x0037))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| parse_ds_backslash::<6>(&s)));

    let rescale_slope: f64 = obj
        .element(Tag(0x0028, 0x1053))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);

    let rescale_intercept: f64 = obj
        .element(Tag(0x0028, 0x1052))
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    MultiFrameInfo {
        path: path.to_path_buf(),
        n_frames,
        rows,
        cols,
        samples_per_pixel,
        bits_allocated,
        pixel_representation,
        pixel_spacing,
        frame_thickness,
        modality,
        sop_class_uid,
        image_position,
        image_orientation,
        rescale_slope,
        rescale_intercept,
        per_frame: Vec::new(),
    }
}

/// Navigate item → seq_tag → items()[0] → inner_tag and parse as `[f64; N]` DS array.
///
/// Returns None if either sequence is absent, has no items, or the DS value fails to
/// parse with at least N components.
fn read_nested_ds<const N: usize>(
    item: &InMemDicomObject,
    seq_tag: Tag,
    inner_tag: Tag,
) -> Option<[f64; N]> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner_item = seq.items().first()?;
        let inner_elem = inner_item.element(inner_tag).ok()?;
        inner_elem
            .to_str()
            .ok()
            .and_then(|s| parse_ds_backslash::<N>(&s))
    } else {
        None
    }
}

/// Navigate item → seq_tag → items()[0] → inner_tag and parse as f64.
///
/// Returns None if either sequence is absent, has no items, or the DS value fails to
/// parse as a finite f64.
fn read_nested_f64(item: &InMemDicomObject, seq_tag: Tag, inner_tag: Tag) -> Option<f64> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner_item = seq.items().first()?;
        let inner_elem = inner_item.element(inner_tag).ok()?;
        inner_elem
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
    } else {
        None
    }
}

/// Extract per-frame metadata from functional group sequences per DICOM PS3.3 C.7.6.16.
///
/// Reads Shared Functional Groups (5200,9229) as fallback defaults, then per-frame
/// overrides from Per-Frame Functional Groups (5200,9230).
///
/// # Algorithm
///
/// 1. If neither (5200,9229) nor (5200,9230) is present, return `Vec::new()`.
/// 2. Parse shared groups (5200,9229)[0] into a `PerFrameInfo` template.
/// 3. For each frame index k in [0, n_frames):
///    a. Start from the shared template.
///    b. If (5200,9230)[k] exists, override non-None fields.
///    c. Push merged result.
///
/// # Fallback chain per attribute A
/// per_frame[k].A = per_frame_groups[k].A   // if Some(...)
///               ?? shared_groups.A          // if Some(...)
///               ?? None
pub(crate) fn extract_functional_groups(
    obj: &InMemDicomObject,
    n_frames: usize,
) -> Vec<PerFrameInfo> {
    let has_shared = obj.element(Tag(0x5200, 0x9229)).is_ok();
    let has_per_frame = obj.element(Tag(0x5200, 0x9230)).is_ok();

    if !has_shared && !has_per_frame {
        tracing::debug!(
            "extract_functional_groups: (5200,9229) and (5200,9230) absent; per_frame=[]"
        );
        return Vec::new();
    }

    // Parse shared functional groups template.
    let shared_template = if let Ok(sg_elem) = obj.element(Tag(0x5200, 0x9229)) {
        if let Value::Sequence(seq) = sg_elem.value() {
            if let Some(item) = seq.items().first() {
                let tpl = PerFrameInfo {
                    image_position: read_nested_ds::<3>(
                        item,
                        Tag(0x0020, 0x9113),
                        Tag(0x0020, 0x0032),
                    ),
                    image_orientation: read_nested_ds::<6>(
                        item,
                        Tag(0x0020, 0x9116),
                        Tag(0x0020, 0x0037),
                    ),
                    pixel_spacing: read_nested_ds::<2>(
                        item,
                        Tag(0x0028, 0x9110),
                        Tag(0x0028, 0x0030),
                    ),
                    slice_thickness: read_nested_f64(
                        item,
                        Tag(0x0028, 0x9110),
                        Tag(0x0018, 0x0050),
                    ),
                    rescale_slope: read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1053)),
                    rescale_intercept: read_nested_f64(
                        item,
                        Tag(0x0028, 0x9145),
                        Tag(0x0028, 0x1052),
                    ),
                };
                tracing::debug!(
                    has_image_position = tpl.image_position.is_some(),
                    has_pixel_spacing = tpl.pixel_spacing.is_some(),
                    has_rescale = tpl.rescale_slope.is_some(),
                    "extract_functional_groups: shared template parsed"
                );
                tpl
            } else {
                PerFrameInfo::default()
            }
        } else {
            PerFrameInfo::default()
        }
    } else {
        PerFrameInfo::default()
    };

    // Collect per-frame items.
    let pf_items: Vec<InMemDicomObject> = if let Ok(pf_elem) = obj.element(Tag(0x5200, 0x9230)) {
        if let Value::Sequence(seq) = pf_elem.value() {
            seq.items().to_vec()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    tracing::debug!(
        n_frames,
        pf_items = pf_items.len(),
        "extract_functional_groups: merging per-frame metadata"
    );

    (0..n_frames)
        .map(|k| {
            let mut info = shared_template.clone();
            if let Some(item) = pf_items.get(k) {
                macro_rules! override_if_some {
                    ($field:ident, $expr:expr) => {
                        if let Some(v) = $expr {
                            info.$field = Some(v);
                        }
                    };
                }
                override_if_some!(
                    image_position,
                    read_nested_ds::<3>(item, Tag(0x0020, 0x9113), Tag(0x0020, 0x0032))
                );
                override_if_some!(
                    image_orientation,
                    read_nested_ds::<6>(item, Tag(0x0020, 0x9116), Tag(0x0020, 0x0037))
                );
                override_if_some!(
                    pixel_spacing,
                    read_nested_ds::<2>(item, Tag(0x0028, 0x9110), Tag(0x0028, 0x0030))
                );
                override_if_some!(
                    slice_thickness,
                    read_nested_f64(item, Tag(0x0028, 0x9110), Tag(0x0018, 0x0050))
                );
                override_if_some!(
                    rescale_slope,
                    read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1053))
                );
                override_if_some!(
                    rescale_intercept,
                    read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1052))
                );
            }
            info
        })
        .collect()
}

/// Read summary information from a multi-frame DICOM file without pixel data.
pub fn read_multiframe_info(path: impl AsRef<Path>) -> Result<MultiFrameInfo> {
    let path = path.as_ref();
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;
    let mut info = extract_multiframe_header(path, &obj);
    info.per_frame = extract_functional_groups(&obj, info.n_frames);
    Ok(info)
}

/// Load a multi-frame DICOM file as a 3-D image with shape [n_frames, rows, cols].
///
/// Applies RescaleSlope and RescaleIntercept to convert stored integers to floats.
/// For Enhanced CT/MR/PET objects that carry per-frame functional groups (5200,9230),
/// per-frame slope/intercept values override the global tags for each frame in the
/// native (uncompressed) decode path.
/// When ImagePositionPatient (0020,0032) is present, the image origin is set
/// accordingly; otherwise the origin defaults to [0, 0, 0].
/// When ImageOrientationPatient (0020,0037) is present, the direction matrix is
/// derived from the row and column cosines with the normal computed as their
/// cross product; otherwise the direction defaults to identity.
pub fn load_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;

    // Guard: compressed transfer syntaxes are not natively decodable by ritk-io.
    // Pixel data from compressed objects cannot be interpreted as raw u16/u8 samples.
    let ts_uid = obj.meta().transfer_syntax();
    let ts = TransferSyntaxKind::from_uid(ts_uid);
    if ts.is_compressed() && !ts.is_codec_supported() {
        bail!(
            "DICOM multiframe: compressed transfer syntax '{}' in {:?} is not supported \
             (not natively decoded and no codec registered); \
             decompress the file or use a supported transfer syntax",
            ts_uid,
            path
        );
    }
    if ts.is_big_endian() {
        bail!(
            "DICOM multiframe: big-endian transfer syntax '{}' in {:?} is not supported; \
             pixel decode requires little-endian byte order",
            ts_uid,
            path
        );
    }

    let info = extract_multiframe_header(path, &obj);
    if info.rows == 0 || info.cols == 0 {
        bail!(
            "DICOM multiframe: rows={} cols={} must be >0 in {:?}",
            info.rows,
            info.cols,
            path
        );
    }
    if info.samples_per_pixel != 1 {
        bail!(
            "DICOM multiframe scalar volume loader supports only SamplesPerPixel=1; \
             {:?} declares SamplesPerPixel={}. \
             Decode RGB/color frames through the codec boundary or a color-volume loader",
            path,
            info.samples_per_pixel
        );
    }

    let per_frame = extract_functional_groups(&obj, info.n_frames);
    tracing::debug!(
        n_frames = info.n_frames,
        per_frame_len = per_frame.len(),
        "load_dicom_multiframe: functional groups extracted"
    );

    let frame_pixels = info.rows * info.cols;
    let mut floats = Vec::with_capacity(info.n_frames * frame_pixels);

    for frame_idx in 0..info.n_frames {
        let frame_info = per_frame.get(frame_idx);
        let slope = frame_info
            .and_then(|pfi| pfi.rescale_slope)
            .unwrap_or(info.rescale_slope) as f32;
        let intercept = frame_info
            .and_then(|pfi| pfi.rescale_intercept)
            .unwrap_or(info.rescale_intercept) as f32;

        let frame = decode_frame_with::<DicomRsBackend>(
            &obj,
            DecodeFrameRequest {
                frame_index: u32::try_from(frame_idx)
                    .context("DICOM multiframe frame index exceeds u32")?,
                transfer_syntax: ts.clone(),
                layout: PixelLayout {
                    rows: info.rows,
                    cols: info.cols,
                    samples_per_pixel: 1,
                    bits_allocated: info.bits_allocated,
                    pixel_representation: info.pixel_representation,
                    rescale_slope: slope,
                    rescale_intercept: intercept,
                },
            },
        )
        .with_context(|| format!("DICOM backend failed for frame {frame_idx} in {:?}", path))?
        .pixels;

        if frame.len() != frame_pixels {
            bail!(
                "DICOM multiframe: frame {} in {:?} decoded {} pixels; expected {}",
                frame_idx,
                path,
                frame.len(),
                frame_pixels
            );
        }

        floats.extend_from_slice(&frame);
    }

    let actual_n = info.n_frames;
    if actual_n == 0 {
        bail!("DICOM multiframe: no pixel data decoded from {:?}", path);
    }

    // Per-frame geometry analysis: derive z-spacing from actual frame positions
    // rather than the global SliceThickness tag, which may be absent or inaccurate
    // for Enhanced CT/MR/PET with varying slice distances.
    //
    // When per_frame is populated (Enhanced MF), each PerFrameInfo may carry
    // image_position. Compute per-adjacent-pair distances projected onto the
    // shared slice normal N̂ = normalize(row × col). Use the median as nominal
    // spacing; warn and resample when nonuniform or missing frames are detected.
    let (final_floats, final_n, spacing_z) = 'geometry: {
        if per_frame.len() < 2 {
            // Fewer than 2 per-frame positions: cannot derive inter-frame spacing.
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        }

        // Resolve IOP: per-frame override first, then global.
        let iop = per_frame
            .iter()
            .find_map(|pf| pf.image_orientation)
            .or(info.image_orientation);

        let Some(normal) = iop.and_then(super::super::reader::slice_normal_from_iop) else {
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        };

        let proj: Vec<Option<f64>> = per_frame
            .iter()
            .take(actual_n)
            .map(|pf| {
                pf.image_position
                    .map(|p| super::super::reader::dot_3d(p, normal))
            })
            .collect();

        if proj.iter().any(|p| p.is_none()) {
            // Not all frames have positions; fall back to global thickness.
            break 'geometry (floats, actual_n, info.frame_thickness.unwrap_or(1.0));
        }

        let src_positions: Vec<f64> = proj.into_iter().map(|p| p.unwrap()).collect();
        let report = super::super::reader::analyze_slice_spacing(&src_positions);

        if report.is_nonuniform {
            tracing::warn!(
                max_relative_deviation = report.max_relative_deviation,
                nominal_spacing_mm = report.nominal_spacing,
                n_frames = actual_n,
                path = ?path,
                "DICOM multiframe: nonuniform inter-frame spacing detected \
                 (max deviation {:.2}%); resampling to uniform grid \
                 with nominal spacing {:.4} mm",
                report.max_relative_deviation * 100.0,
                report.nominal_spacing,
            );
        }
        if report.has_missing_slices {
            tracing::warn!(
                missing_between = ?report.missing_between,
                nominal_spacing_mm = report.nominal_spacing,
                path = ?path,
                "DICOM multiframe: {} gap(s) exceed 1.5× nominal spacing \
                 ({:.4} mm), indicating missing frames; \
                 resampling to fill gaps via linear interpolation",
                report.missing_between.len(),
                report.nominal_spacing,
            );
        }

        if report.is_nonuniform || report.has_missing_slices {
            let frame_pixels = info.rows * info.cols;
            let src_frames: Vec<Vec<f32>> = floats
                .chunks(frame_pixels)
                .take(actual_n)
                .map(|c| c.to_vec())
                .collect();

            let resampled = super::super::reader::resample_frames_linear(
                &src_frames,
                &src_positions,
                report.nominal_spacing,
            );
            let new_n = resampled.len();
            let new_floats: Vec<f32> = resampled.into_iter().flatten().collect();
            (new_floats, new_n, report.nominal_spacing)
        } else {
            (floats, actual_n, report.nominal_spacing)
        }
    };

    let spacing = match info.pixel_spacing {
        Some([rs, cs]) => Spacing::new([spacing_z, rs, cs]),
        None => Spacing::new([spacing_z, 1.0, 1.0]),
    };

    let origin = info.image_position.unwrap_or([0.0, 0.0, 0.0]);

    let direction = if let Some(iop) = info.image_orientation {
        let (rx, ry, rz) = (iop[0], iop[1], iop[2]);
        let (cx, cy, cz) = (iop[3], iop[4], iop[5]);
        let nx = ry * cz - rz * cy;
        let ny = rz * cx - rx * cz;
        let nz = rx * cy - ry * cx;
        let n = super::super::reader::normalize_3d([nx, ny, nz]).unwrap_or([0.0, 0.0, 1.0]);
        let col_data: [f64; 9] = [n[0], n[1], n[2], cx, cy, cz, rx, ry, rz];
        Direction(SMatrix::<f64, 3, 3>::from_column_slice(&col_data))
    } else {
        Direction::identity()
    };

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(final_floats, Shape::new([final_n, info.rows, info.cols])),
        device,
    );

    Ok(Image::new(tensor, Point::new(origin), spacing, direction))
}
