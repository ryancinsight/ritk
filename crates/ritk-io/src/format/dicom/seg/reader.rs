use anyhow::{bail, Context, Result};
use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::InMemDicomObject;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::path::Path;

use super::types::{DicomSegmentInfo, DicomSegmentation, SEG_SOP_CLASS_UID};

/// Read a DICOM Segmentation Storage file at `path` into [`DicomSegmentation`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's MediaStorageSOPClassUID is not `1.2.840.10008.5.1.4.1.1.66.4`.
/// - Required tags (Rows, Columns, BitsAllocated, PixelData) are absent.
/// - PixelData length is inconsistent with declared frame geometry.
pub fn read_dicom_seg<P: AsRef<Path>>(path: P) -> Result<DicomSegmentation> {
    let path = path.as_ref();

    let obj = parse_file_with::<DicomRsBackend, _>(path)
        .with_context(|| format!("open DICOM file: {}", path.display()))?;

    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != SEG_SOP_CLASS_UID {
        bail!("SOP Class {} is not Segmentation Storage", sop);
    }

    let rows: usize = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(|| anyhow::anyhow!("missing or unparseable Rows (0028,0010)"))?;

    let cols: usize = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(|| anyhow::anyhow!("missing or unparseable Columns (0028,0011)"))?;

    let n_frames: usize = obj
        .element(Tag(0x0028, 0x0008))
        .ok()
        .and_then(|e| e.to_int::<i32>().ok())
        .map(|v| v.max(1) as usize)
        .unwrap_or(1);

    let bits_allocated: u16 = obj
        .element(Tag(0x0028, 0x0100))
        .ok()
        .and_then(|e| e.to_str().ok().and_then(|s| s.trim().parse().ok()))
        .unwrap_or(1);

    let segmentation_type = obj
        .element(Tag(0x0062, 0x0001))
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()))
        .unwrap_or_else(|| "BINARY".to_owned());

    tracing::debug!(
        "read_dicom_seg: header rows={} cols={} n_frames={} bits_allocated={} seg_type={}",
        rows,
        cols,
        n_frames,
        bits_allocated,
        segmentation_type
    );

    let segments = parse_segment_sequence(&obj);
    let (frame_segment_numbers, image_position_per_frame) =
        parse_per_frame_functional_groups(&obj, n_frames);
    let (image_orientation, pixel_spacing, slice_thickness) = parse_shared_functional_groups(&obj);

    let px_bytes = obj
        .element(Tag(0x7FE0, 0x0010))
        .context("missing PixelData (7FE0,0010)")?
        .to_bytes()
        .context("PixelData to_bytes")?;

    let pixel_data = unpack_pixel_data(
        &px_bytes,
        n_frames,
        rows,
        cols,
        bits_allocated,
        &segmentation_type,
    )?;

    Ok(DicomSegmentation {
        rows,
        cols,
        n_frames,
        bits_allocated,
        segmentation_type,
        segments,
        frame_segment_numbers,
        pixel_data,
        image_position_per_frame,
        image_orientation,
        pixel_spacing,
        slice_thickness,
    })
}

/// Parse `\`-separated DICOM Decimal String into a fixed-size `f64` array.
fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
    let parts: Vec<f64> = s
        .trim()
        .split('\\')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .collect();
    if parts.len() >= N {
        let mut arr = [0.0_f64; N];
        arr.copy_from_slice(&parts[..N]);
        Some(arr)
    } else {
        None
    }
}

/// Navigate `item → seq_tag → items[0] → inner_tag` and parse as DS array.
fn read_nested_ds<const N: usize>(
    item: &InMemDicomObject,
    seq_tag: Tag,
    inner_tag: Tag,
) -> Option<[f64; N]> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner = seq.items().first()?;
        inner
            .element(inner_tag)
            .ok()?
            .to_str()
            .ok()
            .and_then(|s| parse_ds_backslash::<N>(&s))
    } else {
        None
    }
}

/// Navigate `item → seq_tag → items[0] → inner_tag` and parse as single `f64`.
fn read_nested_f64(item: &InMemDicomObject, seq_tag: Tag, inner_tag: Tag) -> Option<f64> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner = seq.items().first()?;
        inner
            .element(inner_tag)
            .ok()?
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
    } else {
        None
    }
}

/// Extract `Vec<DicomSegmentInfo>` from SegmentSequence (0062,0002).
fn parse_segment_sequence(obj: &InMemDicomObject) -> Vec<DicomSegmentInfo> {
    let elem = match obj.element(Tag(0x0062, 0x0002)) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let seq = match elem.value() {
        Value::Sequence(s) => s,
        _ => return Vec::new(),
    };
    seq.items()
        .iter()
        .map(|item: &InMemDicomObject| {
            let segment_number = item
                .element(Tag(0x0062, 0x0004))
                .ok()
                .and_then(|e| e.to_int::<u16>().ok())
                .unwrap_or(0);
            let segment_label = item
                .element(Tag(0x0062, 0x0005))
                .ok()
                .and_then(|e| {
                    e.to_str()
                        .ok()
                        .map(|s: std::borrow::Cow<str>| s.trim().to_owned())
                })
                .unwrap_or_default();
            let segment_description = item
                .element(Tag(0x0062, 0x0006))
                .ok()
                .and_then(|e| {
                    e.to_str()
                        .ok()
                        .map(|s: std::borrow::Cow<str>| s.trim().to_owned())
                })
                .filter(|s: &String| !s.is_empty());
            let algorithm_type = item
                .element(Tag(0x0062, 0x0008))
                .ok()
                .and_then(|e| {
                    e.to_str()
                        .ok()
                        .map(|s: std::borrow::Cow<str>| s.trim().to_owned())
                })
                .filter(|s: &String| !s.is_empty());
            DicomSegmentInfo {
                segment_number,
                segment_label,
                segment_description,
                algorithm_type,
            }
        })
        .collect()
}

/// Extract per-frame segment numbers and image positions from (5200,9230).
fn parse_per_frame_functional_groups(
    obj: &InMemDicomObject,
    n_frames: usize,
) -> (Vec<u16>, Vec<Option<[f64; 3]>>) {
    let pf_items: Vec<InMemDicomObject> = obj
        .element(Tag(0x5200, 0x9230))
        .ok()
        .and_then(|e| {
            if let Value::Sequence(seq) = e.value() {
                Some(seq.items().to_vec())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let mut seg_numbers = Vec::with_capacity(n_frames);
    let mut positions = Vec::with_capacity(n_frames);

    for k in 0..n_frames {
        if let Some(pf_item) = pf_items.get(k) {
            let seg_num = pf_item
                .element(Tag(0x0062, 0x000A))
                .ok()
                .and_then(|e| {
                    if let Value::Sequence(seq) = e.value() {
                        seq.items().first().and_then(|inner| {
                            inner
                                .element(Tag(0x0062, 0x000B))
                                .ok()
                                .and_then(|ie| ie.to_int::<u16>().ok())
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            let pos = read_nested_ds::<3>(pf_item, Tag(0x0020, 0x9113), Tag(0x0020, 0x0032));

            seg_numbers.push(seg_num);
            positions.push(pos);
        } else {
            seg_numbers.push(0);
            positions.push(None);
        }
    }

    (seg_numbers, positions)
}

/// Extract orientation, pixel spacing, and slice thickness from shared FG (5200,9229).
fn parse_shared_functional_groups(
    obj: &InMemDicomObject,
) -> (Option<[f64; 6]>, Option<[f64; 2]>, Option<f64>) {
    let shared_item: Option<InMemDicomObject> =
        obj.element(Tag(0x5200, 0x9229)).ok().and_then(|e| {
            if let Value::Sequence(seq) = e.value() {
                seq.items().first().cloned()
            } else {
                None
            }
        });

    let item = match shared_item {
        Some(i) => i,
        None => return (None, None, None),
    };

    let orientation = read_nested_ds::<6>(&item, Tag(0x0020, 0x9116), Tag(0x0020, 0x0037));
    let pixel_spacing = read_nested_ds::<2>(&item, Tag(0x0028, 0x9110), Tag(0x0028, 0x0030));
    let slice_thickness = read_nested_f64(&item, Tag(0x0028, 0x9110), Tag(0x0018, 0x0050));

    (orientation, pixel_spacing, slice_thickness)
}

/// Unpack raw PixelData bytes into per-frame decoded pixel vectors.
///
/// # BINARY unpacking (BitsAllocated == 1)
///
/// frame_bytes = ⌈rows × cols / 8⌉
/// Pixel i of frame f:
///   raw_byte = px_bytes[f * frame_bytes + i / 8]
///   bit      = 7 - (i % 8)
///   value    = (raw_byte >> bit) & 1
///
/// # FRACTIONAL (BitsAllocated == 8)
///
/// Pixel i of frame f = px_bytes[f * rows * cols + i]
fn unpack_pixel_data(
    px_bytes: &[u8],
    n_frames: usize,
    rows: usize,
    cols: usize,
    bits_allocated: u16,
    segmentation_type: &str,
) -> Result<Vec<Vec<u8>>> {
    let n_pixels = rows * cols;

    match (bits_allocated, segmentation_type.trim()) {
        (1, _) | (_, "BINARY") => {
            let frame_bytes = n_pixels.div_ceil(8);
            let expected = n_frames * frame_bytes;
            if px_bytes.len() < expected {
                bail!(
                    "PixelData too short for BINARY: got {} bytes, need {} ({}×{}px/8 per frame × {} frames)",
                    px_bytes.len(), expected, rows, cols, n_frames
                );
            }
            let mut frames = Vec::with_capacity(n_frames);
            for f in 0..n_frames {
                let base = f * frame_bytes;
                let mut decoded = Vec::with_capacity(n_pixels);
                for i in 0..n_pixels {
                    let byte_idx = base + i / 8;
                    let bit_pos = 7 - (i % 8);
                    let v = (px_bytes[byte_idx] >> bit_pos) & 1;
                    decoded.push(v);
                }
                frames.push(decoded);
            }
            Ok(frames)
        }
        (8, _) => {
            let frame_bytes = n_pixels;
            let expected = n_frames * frame_bytes;
            if px_bytes.len() < expected {
                bail!(
                    "PixelData too short for FRACTIONAL: got {} bytes, need {}",
                    px_bytes.len(),
                    expected
                );
            }
            let frames = (0..n_frames)
                .map(|f| px_bytes[f * frame_bytes..(f + 1) * frame_bytes].to_vec())
                .collect();
            Ok(frames)
        }
        _ => bail!(
            "unsupported BitsAllocated={} for segmentation_type={}",
            bits_allocated,
            segmentation_type
        ),
    }
}
