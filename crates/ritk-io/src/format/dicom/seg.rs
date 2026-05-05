//! DICOM Segmentation Storage (SOP Class 1.2.840.10008.5.1.4.1.1.66.4) reader.
//!
//! # Specification
//!
//! A DICOM-SEG file encodes N binary or fractional segmentation frames:
//! - (0028,0010) Rows, (0028,0011) Columns, (0028,0008) NumberOfFrames
//! - (0028,0100) BitsAllocated: 1 (BINARY) or 8 (FRACTIONAL)
//! - (0062,0001) SegmentationType: "BINARY" | "FRACTIONAL"
//! - (0062,0002) SegmentSequence: one item per segment label
//! - (5200,9230) Per-Frame Functional Groups: segment identification and plane position
//! - (5200,9229) Shared Functional Groups: orientation and pixel measures
//! - (7FE0,0010) PixelData: packed bits for BINARY, byte-per-pixel for FRACTIONAL
//!
//! ## BINARY pixel unpacking invariant
//!
//! For frame f with rows×cols pixels, frame_bytes = ⌈rows×cols / 8⌉.
//! The flat pixel index i ∈ [0, rows×cols) maps to:
//!   byte   = i / 8
//!   bit    = 7 - (i % 8)   (MSB-first within each byte)
//!   value  = (raw_byte >> bit) & 1
//!
//! FRACTIONAL frames: pixel i of frame f = raw_bytes[f * rows*cols + i].

use anyhow::{bail, Context, Result};
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::{open_file, InMemDicomObject};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// SOP Class UID for Segmentation Storage.
pub const SEG_SOP_CLASS_UID: &str = "1.2.840.10008.5.1.4.1.1.66.4";

// ── Domain types ────────────────────────────────────────────────────────────

/// Metadata for one segment label defined in the Segment Sequence (0062,0002).
#[derive(Debug, Clone)]
pub struct DicomSegmentInfo {
    /// SegmentNumber (0062,0004) US — 1-based segment index.
    pub segment_number: u16,
    /// SegmentLabel (0062,0005) LO.
    pub segment_label: String,
    /// SegmentDescription (0062,0006) ST, optional.
    pub segment_description: Option<String>,
    /// AlgorithmType (0062,0008) CS, optional.
    pub algorithm_type: Option<String>,
}

/// Complete in-memory representation of a DICOM-SEG object.
///
/// # Invariants
/// - `pixel_data.len() == n_frames`
/// - `pixel_data[f].len() == rows * cols` for all f
/// - For BINARY: pixel_data[f][i] ∈ {0, 1}
/// - `frame_segment_numbers.len() == n_frames`
/// - `image_position_per_frame.len() == n_frames`
#[derive(Debug, Clone)]
pub struct DicomSegmentation {
    /// Pixel rows per frame (0028,0010).
    pub rows: usize,
    /// Pixel columns per frame (0028,0011).
    pub cols: usize,
    /// Number of frames (0028,0008); defaults to 1 when absent.
    pub n_frames: usize,
    /// BitsAllocated (0028,0100): 1 for BINARY, 8 for FRACTIONAL.
    pub bits_allocated: u16,
    /// SegmentationType (0062,0001): "BINARY" or "FRACTIONAL".
    pub segmentation_type: String,
    /// One entry per segment defined in SegmentSequence (0062,0002).
    pub segments: Vec<DicomSegmentInfo>,
    /// ReferencedSegmentNumber per frame; length == n_frames.
    pub frame_segment_numbers: Vec<u16>,
    /// Decoded pixel values per frame; each inner vec length == rows*cols.
    /// BINARY: 0 or 1. FRACTIONAL: raw byte values [0, 255].
    pub pixel_data: Vec<Vec<u8>>,
    /// ImagePositionPatient per frame from (5200,9230) → (0020,9113) → (0020,0032).
    pub image_position_per_frame: Vec<Option<[f64; 3]>>,
    /// ImageOrientationPatient from shared FG (5200,9229) → (0020,9116) → (0020,0037).
    pub image_orientation: Option<[f64; 6]>,
    /// PixelSpacing from shared FG (5200,9229) → (0028,9110) → (0028,0030).
    pub pixel_spacing: Option<[f64; 2]>,
    /// SliceThickness from shared FG (5200,9229) → (0028,9110) → (0018,0050).
    pub slice_thickness: Option<f64>,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Read a DICOM Segmentation Storage file at `path` into [`DicomSegmentation`].
///
/// # Errors
/// - `path` does not exist or is not readable.
/// - The file's MediaStorageSOPClassUID is not `1.2.840.10008.5.1.4.1.1.66.4`.
/// - Required tags (Rows, Columns, BitsAllocated, PixelData) are absent.
/// - PixelData length is inconsistent with declared frame geometry.
pub fn read_dicom_seg<P: AsRef<Path>>(path: P) -> Result<DicomSegmentation> {
    let path = path.as_ref();

    let obj = open_file(path).with_context(|| format!("open DICOM file: {}", path.display()))?;

    // Validate SOP Class UID.
    let sop = obj.meta().media_storage_sop_class_uid();
    let sop = sop.trim_end_matches('\0').trim();
    if sop != SEG_SOP_CLASS_UID {
        bail!("SOP Class {} is not Segmentation Storage", sop);
    }

    // ── Scalar header fields ─────────────────────────────────────────────────

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

    // ── Segment Sequence (0062,0002) ─────────────────────────────────────────

    let segments = parse_segment_sequence(&obj);

    // ── Per-Frame Functional Groups (5200,9230) ──────────────────────────────

    let (frame_segment_numbers, image_position_per_frame) =
        parse_per_frame_functional_groups(&obj, n_frames);

    // ── Shared Functional Groups (5200,9229) ─────────────────────────────────

    let (image_orientation, pixel_spacing, slice_thickness) = parse_shared_functional_groups(&obj);

    // ── Pixel data ───────────────────────────────────────────────────────────

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

// ── Private helpers ──────────────────────────────────────────────────────────

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
///
/// For each frame f ∈ [0, n_frames):
/// - `frame_segment_numbers[f]`: (5200,9230)[f] → (0062,000A) → (0062,000B) US
/// - `image_position_per_frame[f]`: (5200,9230)[f] → (0020,9113) → (0020,0032) DS
///
/// Missing items default to 0 and None respectively.
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
            // ReferencedSegmentNumber: (5200,9230)[k] → (0062,000A) → (0062,000B)
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

            // ImagePositionPatient: (5200,9230)[k] → (0020,9113) → (0020,0032)
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
///
/// Paths:
/// - orientation: (5200,9229)[0] → (0020,9116)[0] → (0020,0037)
/// - pixel_spacing: (5200,9229)[0] → (0028,9110)[0] → (0028,0030)
/// - slice_thickness: (5200,9229)[0] → (0028,9110)[0] → (0018,0050)
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
            // MSB-first bit packing: frame_bytes = ⌈n_pixels / 8⌉
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

// ── Writer ────────────────────────────────────────────────────────────────────

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
    // ── Validation ──────────────────────────────────────────────────────────
    if seg.pixel_data.len() != seg.n_frames {
        bail!(
            "pixel_data.len()={} != n_frames={}",
            seg.pixel_data.len(),
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

    // ── SOP Instance UID ────────────────────────────────────────────────────
    static SEG_UID_COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = SEG_UID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sop_instance_uid = format!("2.25.{}.{}", t, n);

    // ── Pixel data packing ──────────────────────────────────────────────────
    // BINARY: MSB-first packing — inverse of unpack_pixel_data (BitsAllocated == 1).
    //   frame_byte_count = ⌈rows×cols / 8⌉
    //   pixel i → bit (7 - i%8) of byte (base + i/8)
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

    // ── Build SegmentSequence (0062,0002) ───────────────────────────────────
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
                PrimitiveValue::from(info.algorithm_type.as_deref().unwrap_or("MANUAL")),
            ));
            item
        })
        .collect();

    // ── Assemble root InMemDicomObject ──────────────────────────────────────
    let mut obj = InMemDicomObject::new_empty();

    // SOP identification
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

    // Study / Series / Instance identifiers (placeholder UIDs)
    obj.put(DataElement::new(
        Tag(0x0020, 0x000D),
        VR::UI,
        PrimitiveValue::from("2.25.999"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x000E),
        VR::UI,
        PrimitiveValue::from("2.25.998"),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from("1"),
    ));

    // Image Pixel Module
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
    // BitsStored = BitsAllocated for SEG IOD
    obj.put(DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(seg.bits_allocated),
    ));
    // HighBit = BitsAllocated - 1
    obj.put(DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::from(seg.bits_allocated - 1),
    ));
    // PixelRepresentation = 0 (unsigned)
    obj.put(DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    // SamplesPerPixel = 1
    obj.put(DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));

    // Segmentation type (0062,0001)
    obj.put(DataElement::new(
        Tag(0x0062, 0x0001),
        VR::CS,
        PrimitiveValue::from(seg.segmentation_type.as_str()),
    ));

    // Segment Sequence (0062,0002)
    if !seg_items.is_empty() {
        let seq = DataSetSequence::new(seg_items, Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x0062, 0x0002),
            VR::SQ,
            Value::from(seq),
        ));
    }

    // Pixel Data (7FE0,0010) OW
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));

    // ── Write Part-10 file ──────────────────────────────────────────────────
    let path = path.as_ref();
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
            .media_storage_sop_instance_uid(sop_instance_uid.as_str())
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .with_context(|| "build DICOM-SEG file meta")?
    .write_to_file(path)
    .with_context(|| format!("write DICOM-SEG to {}", path.display()))?;

    Ok(())
}

/// Convert a `ritk_core` `LabelMap` with spatial metadata to a `DicomSegmentation`.
///
/// # Mathematical Specification
///
/// This implementation creates one 2D frame per Z-slice per segment. Given a 3D
/// label map L: Z³ → N with shape [nz, ny, nx], each unique foreground label ID
/// produces nz frames (one per Z-slice), each with geometry rows=ny, cols=nx.
///
/// Total frames = (number of foreground labels) × nz.
/// Frame f corresponding to segment s and Z-index z contains pixel values 1 where
/// the label equals s, and 0 elsewhere.
///
/// # Invariants
/// - Background label (ID 0) is excluded from segments; only foreground labels are encoded.
/// - Frame count = n_segments × nz.
/// - All label table entries with existing IDs in the map are preserved as segment metadata.
///
/// # Errors
/// - `label_map.shape` has zero dimension (invalid geometry).
/// - No foreground labels exist (empty segmentation).
pub fn label_map_to_dicom_seg(
    label_map: &ritk_core::annotation::LabelMap,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
    use_binary: bool,
) -> Result<DicomSegmentation> {
    // ── Validation ──────────────────────────────────────────────────────────

    if label_map.shape[0] == 0 || label_map.shape[1] == 0 || label_map.shape[2] == 0 {
        bail!("label_map has zero dimension: {:?}", label_map.shape);
    }

    // Get unique foreground labels (excluding background 0)
    let mut foreground_labels = label_map.present_labels();
    foreground_labels.retain(|&id| id != 0);

    if foreground_labels.is_empty() {
        bail!("no foreground labels found in label_map");
    }

    // ── Frame geometry ──────────────────────────────────────────────────────
    // Shape is [nz, ny, nx]; DICOM-SEG uses rows=ny, cols=nx per frame
    let nz = label_map.shape[0];
    let ny = label_map.shape[1];
    let nx = label_map.shape[2];
    let rows = ny as usize;
    let cols = nx as usize;
    let n_pixels_per_frame = rows * cols;
    let n_frames = foreground_labels.len() * nz; // one frame per Z-slice per segment

    // ── Build segment metadata and pixel frames ──────────────────────────────
    let mut segments = Vec::new();
    let mut frame_segment_numbers = Vec::new();
    let mut pixel_data = Vec::new();
    let mut image_position_per_frame = Vec::new();

    for (segment_idx, &label_id) in foreground_labels.iter().enumerate() {
        let segment_number = (segment_idx + 1) as u16; // 1-based segment numbering

        // Look up label entry from table
        let label_name = label_map
            .table
            .get_label(label_id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| format!("Label {}", label_id));

        segments.push(DicomSegmentInfo {
            segment_number,
            segment_label: label_name,
            segment_description: None,
            algorithm_type: None,
        });

        // Create nz frames, one per Z-slice
        for z in 0..nz {
            frame_segment_numbers.push(segment_number);

            // Extract 2D frame at Z-index z: iterate over (y, x)
            let mut frame_pixels = Vec::with_capacity(n_pixels_per_frame);
            for y in 0..ny {
                for x in 0..nx {
                    let at_label = label_map.label_at([z, y, x]);
                    frame_pixels.push(if at_label == label_id { 1u8 } else { 0u8 });
                }
            }
            assert_eq!(
                frame_pixels.len(),
                n_pixels_per_frame,
                "frame pixel count mismatch at z={}: {} != {}",
                z,
                frame_pixels.len(),
                n_pixels_per_frame
            );
            pixel_data.push(frame_pixels);

            // Compute ImagePositionPatient for this frame (Z-slice z)
            let dir_z_col = [direction[2], direction[5], direction[8]];
            let z_offset = z as f64 * spacing[0];
            let pos = [
                origin[0] + z_offset * dir_z_col[0],
                origin[1] + z_offset * dir_z_col[1],
                origin[2] + z_offset * dir_z_col[2],
            ];
            image_position_per_frame.push(Some(pos));
        }
    }

    // ── Construct DicomSegmentation ──────────────────────────────────────────

    let bits_allocated = if use_binary { 1u16 } else { 8u16 };
    let segmentation_type = if use_binary {
        "BINARY".to_owned()
    } else {
        "FRACTIONAL".to_owned()
    };

    // Convert 3×3 direction matrix to 6-element image orientation (row, then column direction)
    let image_orientation_6: [f64; 6] = [
        direction[0], // row direction, x component
        direction[3], // row direction, y component
        direction[6], // row direction, z component
        direction[1], // column direction, x component
        direction[4], // column direction, y component
        direction[7], // column direction, z component
    ];

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
        image_orientation: Some(image_orientation_6),
        pixel_spacing: Some([spacing[1], spacing[2]]),
        slice_thickness: Some(spacing[0]),
    })
}

/// Convert a [`DicomSegmentation`] into a dense `ritk_core` [`LabelMap`].
///
/// # Mathematical specification
///
/// Let `S` be a segmentation with frame geometry `(rows, cols)`, `n_frames`,
/// `segments`, and per-frame segment references `frame_segment_numbers`.
/// This conversion reconstructs a dense tensor `L[z, y, x]` with shape
/// `[nz, rows, cols]`, where `nz = n_frames / n_segments` and `n_segments = segments.len()`.
///
/// For each frame `f`, segment `s = frame_segment_numbers[f]` and in-segment
/// slice index `z_s`, each foreground pixel (`pixel_data[f][i] != 0`) writes
/// `label_id(s)` to `L[z_s, y(i), x(i)]`.
///
/// Overlap policy: later writes in frame order win.
///
/// # Invariants checked
/// - `rows > 0`, `cols > 0`, `n_frames > 0`, `segments` non-empty.
/// - `frame_segment_numbers.len() == n_frames`.
/// - `pixel_data.len() == n_frames` and each frame length is `rows * cols`.
/// - `n_frames % segments.len() == 0`.
/// - Every declared segment receives exactly `nz` frames.
pub fn dicom_seg_to_label_map(seg: &DicomSegmentation) -> Result<ritk_core::annotation::LabelMap> {
    if seg.rows == 0 || seg.cols == 0 {
        bail!("DICOM-SEG has invalid frame geometry: rows={}, cols={}", seg.rows, seg.cols);
    }
    if seg.n_frames == 0 {
        bail!("DICOM-SEG has zero frames");
    }
    if seg.segments.is_empty() {
        bail!("DICOM-SEG has no segments");
    }
    if seg.frame_segment_numbers.len() != seg.n_frames {
        bail!(
            "frame_segment_numbers length {} != n_frames {}",
            seg.frame_segment_numbers.len(),
            seg.n_frames
        );
    }
    if seg.pixel_data.len() != seg.n_frames {
        bail!(
            "pixel_data frame count {} != n_frames {}",
            seg.pixel_data.len(),
            seg.n_frames
        );
    }

    let n_segments = seg.segments.len();
    if seg.n_frames % n_segments != 0 {
        bail!(
            "n_frames {} is not divisible by segment count {}",
            seg.n_frames,
            n_segments
        );
    }

    let nz = seg.n_frames / n_segments;
    if nz == 0 {
        bail!("derived nz is zero");
    }

    let n_pixels_per_frame = seg.rows * seg.cols;
    for (idx, frame) in seg.pixel_data.iter().enumerate() {
        if frame.len() != n_pixels_per_frame {
            bail!(
                "pixel_data[{idx}] length {} != rows*cols {}",
                frame.len(),
                n_pixels_per_frame
            );
        }
    }

    let mut table = ritk_core::annotation::LabelTable::new();
    let mut segment_to_label: HashMap<u16, u32> = HashMap::new();
    for s in &seg.segments {
        if s.segment_number == 0 {
            bail!("segment_number 0 is invalid for DICOM-SEG");
        }
        let label_id = u32::from(s.segment_number);
        let color = segment_color(label_id);
        table
            .add_label(label_id, s.segment_label.clone(), color)
            .map_err(|e| anyhow::anyhow!("invalid or duplicate segment label id {}: {}", label_id, e))?;
        segment_to_label.insert(s.segment_number, label_id);
    }

    let mut seen_per_segment: HashMap<u16, usize> = seg
        .segments
        .iter()
        .map(|s| (s.segment_number, 0usize))
        .collect();

    let mut data = vec![0u32; nz * n_pixels_per_frame];
    for frame_idx in 0..seg.n_frames {
        let segment_number = seg.frame_segment_numbers[frame_idx];
        let Some(&label_id) = segment_to_label.get(&segment_number) else {
            bail!(
                "frame {} references undefined segment_number {}",
                frame_idx,
                segment_number
            );
        };

        let Some(z_counter) = seen_per_segment.get_mut(&segment_number) else {
            bail!(
                "internal segment tracking missing segment_number {}",
                segment_number
            );
        };
        if *z_counter >= nz {
            bail!(
                "segment {} has more than {} frames",
                segment_number,
                nz
            );
        }
        let z = *z_counter;
        *z_counter += 1;

        let frame = &seg.pixel_data[frame_idx];
        for (i, &v) in frame.iter().enumerate() {
            if v != 0 {
                let flat = z * n_pixels_per_frame + i;
                data[flat] = label_id;
            }
        }
    }

    for s in &seg.segments {
        let count = seen_per_segment
            .get(&s.segment_number)
            .copied()
            .unwrap_or(0);
        if count != nz {
            bail!(
                "segment {} has {} frames, expected {}",
                s.segment_number,
                count,
                nz
            );
        }
    }

    ritk_core::annotation::LabelMap::from_data([nz, seg.rows, seg.cols], data, table)
        .map_err(|e| anyhow::anyhow!("failed to build LabelMap from DICOM-SEG: {e}"))
}

fn segment_color(label_id: u32) -> [u8; 4] {
    let seed = label_id.wrapping_mul(0x9E37_79B9);
    let r = 40 + ((seed & 0x7F) as u8);
    let g = 40 + (((seed >> 8) & 0x7F) as u8);
    let b = 40 + (((seed >> 16) & 0x7F) as u8);
    [r, g, b, 180]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::header::Length;
    use dicom::core::value::DataSetSequence;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::meta::FileMetaTableBuilder;

    /// Build a minimal DICOM-SEG InMemDicomObject with the given geometry and raw pixel bytes.
    ///
    /// `seg_items`: list of (segment_number, segment_label) for SegmentSequence.
    /// `pf_items`: per-frame InMemDicomObject items for (5200,9230).
    fn build_seg_obj(
        rows: u16,
        cols: u16,
        n_frames: u32,
        bits_allocated: u16,
        segmentation_type: &str,
        seg_items: Vec<InMemDicomObject>,
        pf_items: Vec<InMemDicomObject>,
        pixel_bytes: Vec<u8>,
    ) -> InMemDicomObject {
        let mut obj = InMemDicomObject::new_empty();

        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0008),
            VR::IS,
            PrimitiveValue::from(n_frames.to_string().as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(bits_allocated),
        ));
        obj.put(DataElement::new(
            Tag(0x0062, 0x0001),
            VR::CS,
            PrimitiveValue::from(segmentation_type),
        ));

        // SegmentSequence (0062,0002)
        if !seg_items.is_empty() {
            let seq = DataSetSequence::new(seg_items, Length::UNDEFINED);
            obj.put(DataElement::new(
                Tag(0x0062, 0x0002),
                VR::SQ,
                Value::from(seq),
            ));
        }

        // Per-Frame Functional Groups (5200,9230)
        if !pf_items.is_empty() {
            let seq = DataSetSequence::new(pf_items, Length::UNDEFINED);
            obj.put(DataElement::new(
                Tag(0x5200, 0x9230),
                VR::SQ,
                Value::from(seq),
            ));
        }

        // PixelData (7FE0,0010) OB
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OB,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
        ));

        obj
    }

    /// Write an InMemDicomObject as a DICOM-SEG Part 10 file.
    fn write_seg_file(obj: InMemDicomObject, path: &std::path::Path) {
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build")
        .write_to_file(path)
        .expect("write DICOM-SEG file");
    }

    /// Build a segment sequence item for a single segment.
    fn make_segment_item(segment_number: u16, label: &str) -> InMemDicomObject {
        let mut item = InMemDicomObject::new_empty();
        item.put(DataElement::new(
            Tag(0x0062, 0x0004),
            VR::US,
            PrimitiveValue::from(segment_number),
        ));
        item.put(DataElement::new(
            Tag(0x0062, 0x0005),
            VR::LO,
            PrimitiveValue::from(label),
        ));
        item
    }

    /// Build a per-frame item containing SegmentIdentification and optional PlanePosition.
    fn make_per_frame_item(
        referenced_segment_number: u16,
        image_position: Option<&str>,
    ) -> InMemDicomObject {
        // SegmentIdentification (0062,000A) → ReferencedSegmentNumber (0062,000B)
        let mut seg_id_item = InMemDicomObject::new_empty();
        seg_id_item.put(DataElement::new(
            Tag(0x0062, 0x000B),
            VR::US,
            PrimitiveValue::from(referenced_segment_number),
        ));
        let seg_id_seq = DataSetSequence::new(vec![seg_id_item], Length::UNDEFINED);

        let mut pf = InMemDicomObject::new_empty();
        pf.put(DataElement::new(
            Tag(0x0062, 0x000A),
            VR::SQ,
            Value::from(seg_id_seq),
        ));

        if let Some(pos_str) = image_position {
            let mut pos_item = InMemDicomObject::new_empty();
            pos_item.put(DataElement::new(
                Tag(0x0020, 0x0032),
                VR::DS,
                PrimitiveValue::from(pos_str),
            ));
            let pos_seq = DataSetSequence::new(vec![pos_item], Length::UNDEFINED);
            pf.put(DataElement::new(
                Tag(0x0020, 0x9113),
                VR::SQ,
                Value::from(pos_seq),
            ));
        }

        pf
    }

    // ── Test 1: missing file ─────────────────────────────────────────────────

    #[test]
    fn test_read_seg_missing_file_returns_error() {
        let result = read_dicom_seg("/nonexistent/path/seg.dcm");
        assert!(result.is_err(), "expected Err for missing file");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("nonexistent") || msg.contains("open DICOM"),
            "error must mention file path or open action; got: {msg}"
        );
    }

    // ── Test 2: wrong SOP class ──────────────────────────────────────────────

    #[test]
    fn test_read_seg_wrong_sop_class_returns_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("ct.dcm");

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99"),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::new()),
        ));
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.99")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta")
        .write_to_file(&path)
        .expect("write CT stub");

        let result = read_dicom_seg(&path);
        assert!(result.is_err(), "expected Err for wrong SOP class");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("SOP"),
            "error message must contain 'SOP'; got: {msg}"
        );
    }

    // ── Test 3: 4×4 BINARY single frame, all-ones ───────────────────────────

    #[test]
    fn test_read_seg_binary_4x4_single_frame() {
        // 4×4 = 16 pixels → 2 bytes, all bits set = [0xFF, 0xFF]
        let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF];

        let seg_items = vec![make_segment_item(1, "TUMOR")];
        let pf_items = vec![make_per_frame_item(1, None)];

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_4x4.dcm");

        let obj = build_seg_obj(4, 4, 1, 1, "BINARY", seg_items, pf_items, pixel_bytes);
        write_seg_file(obj, &path);

        let seg = read_dicom_seg(&path).expect("read_dicom_seg 4x4 binary");

        assert_eq!(seg.rows, 4, "rows");
        assert_eq!(seg.cols, 4, "cols");
        assert_eq!(seg.n_frames, 1, "n_frames");
        assert_eq!(seg.segmentation_type, "BINARY", "segmentation_type");
        assert_eq!(seg.segments.len(), 1, "segment count");
        assert_eq!(seg.segments[0].segment_label, "TUMOR", "segment label");
        assert_eq!(seg.segments[0].segment_number, 1, "segment number");
        assert_eq!(
            seg.frame_segment_numbers.len(),
            1,
            "frame_segment_numbers len"
        );
        assert_eq!(
            seg.frame_segment_numbers[0], 1,
            "frame 0 references segment 1"
        );
        assert_eq!(seg.pixel_data.len(), 1, "pixel_data frames");
        assert_eq!(seg.pixel_data[0].len(), 16, "pixels per frame");
        assert_eq!(
            seg.pixel_data[0],
            vec![1u8; 16],
            "all 16 pixels must be 1 for 0xFF 0xFF"
        );
    }

    // ── Test 4: two frames, two segments ────────────────────────────────────

    #[test]
    fn test_read_seg_two_frames_two_segments() {
        // Frame 0: all-ones (16 pixels, 2 bytes = [0xFF, 0xFF])
        // Frame 1: all-zeros (16 pixels, 2 bytes = [0x00, 0x00])
        let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF, 0x00, 0x00];

        let seg_items = vec![make_segment_item(1, "GTV"), make_segment_item(2, "CTV")];
        let pf_items = vec![make_per_frame_item(1, None), make_per_frame_item(2, None)];

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_two_frames.dcm");

        let obj = build_seg_obj(4, 4, 2, 1, "BINARY", seg_items, pf_items, pixel_bytes);
        write_seg_file(obj, &path);

        let seg = read_dicom_seg(&path).expect("read_dicom_seg two frames");

        assert_eq!(seg.n_frames, 2, "n_frames");
        assert_eq!(seg.segments.len(), 2, "segment count");
        assert_eq!(seg.segments[0].segment_label, "GTV", "segment 0 label");
        assert_eq!(seg.segments[1].segment_label, "CTV", "segment 1 label");
        assert_eq!(
            seg.frame_segment_numbers,
            vec![1u16, 2u16],
            "frame_segment_numbers"
        );
        assert_eq!(seg.pixel_data[0], vec![1u8; 16], "frame 0: all ones");
        assert_eq!(seg.pixel_data[1], vec![0u8; 16], "frame 1: all zeros");
    }

    // ── Test 5: shared FG pixel spacing ─────────────────────────────────────

    #[test]
    fn test_read_seg_preserves_pixel_spacing() {
        let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF];

        let seg_items = vec![make_segment_item(1, "ROI")];
        let pf_items = vec![make_per_frame_item(1, None)];

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_spacing.dcm");

        // Build shared FG containing PixelMeasuresSequence (0028,9110).
        let mut px_meas_item = InMemDicomObject::new_empty();
        px_meas_item.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from("0.5\\0.5"),
        ));
        px_meas_item.put(DataElement::new(
            Tag(0x0018, 0x0050),
            VR::DS,
            PrimitiveValue::from("2.5"),
        ));
        let px_meas_seq = DataSetSequence::new(vec![px_meas_item], Length::UNDEFINED);

        let mut shared_item = InMemDicomObject::new_empty();
        shared_item.put(DataElement::new(
            Tag(0x0028, 0x9110),
            VR::SQ,
            Value::from(px_meas_seq),
        ));
        let shared_seq = DataSetSequence::new(vec![shared_item], Length::UNDEFINED);

        let mut obj = build_seg_obj(4, 4, 1, 1, "BINARY", seg_items, pf_items, pixel_bytes);
        obj.put(DataElement::new(
            Tag(0x5200, 0x9229),
            VR::SQ,
            Value::from(shared_seq),
        ));
        write_seg_file(obj, &path);

        let seg = read_dicom_seg(&path).expect("read_dicom_seg pixel spacing");

        assert_eq!(
            seg.pixel_spacing,
            Some([0.5, 0.5]),
            "pixel_spacing must be [0.5, 0.5]"
        );
        assert_eq!(
            seg.slice_thickness,
            Some(2.5),
            "slice_thickness must be 2.5"
        );
    }

    // ── Test 6: per-frame image positions ────────────────────────────────────

    #[test]
    fn test_read_seg_per_frame_image_position() {
        // 2-frame 4×4 BINARY SEG with explicit per-frame ImagePositionPatient.
        let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF, 0x00, 0x00];

        let seg_items = vec![make_segment_item(1, "S1"), make_segment_item(2, "S2")];
        let pf_items = vec![
            make_per_frame_item(1, Some("0.0\\0.0\\0.0")),
            make_per_frame_item(2, Some("0.0\\0.0\\5.0")),
        ];

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_positions.dcm");

        let obj = build_seg_obj(4, 4, 2, 1, "BINARY", seg_items, pf_items, pixel_bytes);
        write_seg_file(obj, &path);

        let seg = read_dicom_seg(&path).expect("read_dicom_seg per-frame positions");

        assert_eq!(
            seg.image_position_per_frame.len(),
            2,
            "image_position_per_frame length"
        );
        assert_eq!(
            seg.image_position_per_frame[0],
            Some([0.0, 0.0, 0.0]),
            "frame 0 position"
        );
        assert_eq!(
            seg.image_position_per_frame[1],
            Some([0.0, 0.0, 5.0]),
            "frame 1 position"
        );
    }

    // ── Test 7: write_dicom_seg BINARY roundtrip ─────────────────────────────

    /// Invariant: write_dicom_seg packs BINARY frames MSB-first (inverse of unpack_pixel_data).
    /// Frame 0: pixels 0-7 = 1 → byte 0 = 0xFF; pixels 8-15 = 0 → byte 1 = 0x00.
    /// Frame 1: pixels 0-7 = 0 → byte 0 = 0x00; pixels 8-15 = 1 → byte 1 = 0xFF.
    /// read_dicom_seg must recover the original pixel vectors exactly.
    #[test]
    fn test_write_dicom_seg_binary_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_write_binary.dcm");

        let seg = DicomSegmentation {
            rows: 4,
            cols: 4,
            n_frames: 2,
            bits_allocated: 1,
            segmentation_type: "BINARY".to_owned(),
            segments: vec![DicomSegmentInfo {
                segment_number: 1,
                segment_label: "body".to_owned(),
                segment_description: None,
                algorithm_type: None,
            }],
            frame_segment_numbers: vec![1, 1],
            pixel_data: vec![
                vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            image_position_per_frame: vec![None, None],
            image_orientation: None,
            pixel_spacing: None,
            slice_thickness: None,
        };

        write_dicom_seg(&path, &seg).expect("write_dicom_seg binary");
        let result = read_dicom_seg(&path).expect("read_dicom_seg binary roundtrip");

        assert_eq!(result.rows, 4, "rows");
        assert_eq!(result.cols, 4, "cols");
        assert_eq!(result.n_frames, 2, "n_frames");
        assert_eq!(
            result.pixel_data[0],
            vec![1u8, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "frame 0 pixel values"
        );
        assert_eq!(
            result.pixel_data[1],
            vec![0u8, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "frame 1 pixel values"
        );
        assert_eq!(result.segments.len(), 1, "segment count");
        assert_eq!(result.segments[0].segment_label, "body", "segment label");
    }

    // ── Test 8: write_dicom_seg FRACTIONAL roundtrip ─────────────────────────

    /// Invariant: FRACTIONAL frames are stored as raw bytes (byte-per-pixel).
    /// All four pixel values [0, 128, 200, 255] must survive the write-read cycle.
    #[test]
    fn test_write_dicom_seg_fractional_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_write_fractional.dcm");

        let seg = DicomSegmentation {
            rows: 2,
            cols: 2,
            n_frames: 1,
            bits_allocated: 8,
            segmentation_type: "FRACTIONAL".to_owned(),
            segments: vec![DicomSegmentInfo {
                segment_number: 1,
                segment_label: "prob".to_owned(),
                segment_description: None,
                algorithm_type: None,
            }],
            frame_segment_numbers: vec![1],
            pixel_data: vec![vec![0, 128, 200, 255]],
            image_position_per_frame: vec![None],
            image_orientation: None,
            pixel_spacing: None,
            slice_thickness: None,
        };

        write_dicom_seg(&path, &seg).expect("write_dicom_seg fractional");
        let result = read_dicom_seg(&path).expect("read_dicom_seg fractional roundtrip");

        assert_eq!(result.rows, 2, "rows");
        assert_eq!(result.cols, 2, "cols");
        assert_eq!(result.n_frames, 1, "n_frames");
        assert_eq!(result.pixel_data.len(), 1, "frame count");
        assert_eq!(result.pixel_data[0][0], 0u8, "pixel[0]");
        assert_eq!(result.pixel_data[0][1], 128u8, "pixel[1]");
        assert_eq!(result.pixel_data[0][2], 200u8, "pixel[2]");
        assert_eq!(result.pixel_data[0][3], 255u8, "pixel[3]");
    }

    // ── Test 9: write_dicom_seg rejects mismatched frame count ───────────────

    /// Invariant: pixel_data.len() must equal n_frames; otherwise Err is returned
    /// without creating or truncating any file.
    #[test]
    fn test_write_dicom_seg_rejects_mismatched_frame_count() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg_bad.dcm");

        let seg = DicomSegmentation {
            rows: 4,
            cols: 4,
            n_frames: 2, // declares 2 frames
            bits_allocated: 1,
            segmentation_type: "BINARY".to_owned(),
            segments: vec![DicomSegmentInfo {
                segment_number: 1,
                segment_label: "x".to_owned(),
                segment_description: None,
                algorithm_type: None,
            }],
            frame_segment_numbers: vec![1],
            pixel_data: vec![vec![0u8; 16]], // only 1 frame supplied
            image_position_per_frame: vec![None],
            image_orientation: None,
            pixel_spacing: None,
            slice_thickness: None,
        };

        let result = write_dicom_seg(&path, &seg);
        assert!(result.is_err(), "mismatched frame count must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("pixel_data") || msg.contains("n_frames"),
            "error must identify the frame-count mismatch; got: {msg}"
        );
    }

    // ── Tests for label_map_to_dicom_seg converter ──────────────────────────

    #[test]
    fn test_label_map_to_dicom_seg_identity_single_label() {
        // Create a simple 2×2×2 label map with label 1 everywhere (single label)
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "Label 1", [255, 0, 0, 255]).unwrap();

        let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
        let origin = [0.0, 0.0, 0.0];
        let spacing = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
            .expect("label_map_to_dicom_seg");

        assert_eq!(seg.rows, 2, "rows must equal ny");
        assert_eq!(seg.cols, 2, "cols must equal nx");
        assert_eq!(seg.n_frames, 2, "nz=2 frames, one per Z-slice per segment");
        assert_eq!(seg.bits_allocated, 1, "use_binary=true → bits_allocated=1");
        assert_eq!(seg.segmentation_type, "BINARY");
        assert_eq!(seg.segments.len(), 1, "one segment");
        assert_eq!(seg.segments[0].segment_label, "Label 1");
        assert_eq!(seg.segments[0].segment_number, 1);
        // Each frame is 2×2=4 pixels, all 1s
        assert_eq!(seg.pixel_data[0], vec![1u8; 4], "frame 0 all ones");
        assert_eq!(seg.pixel_data[1], vec![1u8; 4], "frame 1 all ones");
    }

    #[test]
    fn test_label_map_to_dicom_seg_multi_label() {
        // Create a 2×2×2 map with two distinct labels (label 1 in z=0, label 2 in z=1)
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "Foreground", [255, 0, 0, 255]).unwrap();
        table.add_label(2, "Other", [0, 255, 0, 255]).unwrap();

        let data = vec![1, 1, 1, 1, 2, 2, 2, 2]; // z=0 all label 1, z=1 all label 2
        let map = LabelMap::from_data([2, 2, 2], data, table).unwrap();
        let origin = [0.0, 0.0, 0.0];
        let spacing = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
            .expect("label_map_to_dicom_seg");

        // nz=2, 2 segments → 4 frames total
        assert_eq!(seg.n_frames, 4, "2 segments × 2 z-slices = 4 frames");
        assert_eq!(seg.segments.len(), 2, "two segments");
        assert_eq!(seg.frame_segment_numbers, vec![1, 1, 2, 2], "segment assignment per frame");
        // Segment 1 frames:
        assert_eq!(seg.pixel_data[0], vec![1u8; 4], "frame 0 (seg 1, z=0) all ones");
        assert_eq!(seg.pixel_data[1], vec![0u8; 4], "frame 1 (seg 1, z=1) all zeros");
        // Segment 2 frames:
        assert_eq!(seg.pixel_data[2], vec![0u8; 4], "frame 2 (seg 2, z=0) all zeros");
        assert_eq!(seg.pixel_data[3], vec![1u8; 4], "frame 3 (seg 2, z=1) all ones");
    }

    #[test]
    fn test_label_map_to_dicom_seg_background_excluded() {
        // Create a map with background (0) and one foreground label
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "Lesion", [255, 0, 0, 255]).unwrap();

        // Half background, half foreground: single Z-slice
        let data = vec![0, 0, 1, 1]; // 1×2×2 map: first two are background, last two are label 1
        let map = LabelMap::from_data([1, 2, 2], data, table).unwrap();
        let origin = [0.0, 0.0, 0.0];
        let spacing = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
            .expect("label_map_to_dicom_seg");

        assert_eq!(seg.n_frames, 1, "1 foreground label × 1 z-slice = 1 frame");
        assert_eq!(seg.pixel_data[0], vec![0, 0, 1, 1], "correct mask for label 1");
    }

    #[test]
    fn test_label_map_to_dicom_seg_spatial_metadata() {
        // Verify spatial metadata is correctly mapped
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "A", [255, 0, 0, 255]).unwrap();

        let map = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();
        let origin = [10.0, 20.0, 30.0];
        let spacing = [2.0, 0.5, 1.5];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // identity

        let seg = label_map_to_dicom_seg(&map, origin, spacing, direction, true)
            .expect("label_map_to_dicom_seg");

        // image_orientation: first 6 elements of flattened direction [row_x, row_y, row_z, col_x, col_y, col_z]
        // For identity: [1, 0, 0, 0, 1, 0]
        assert_eq!(
            seg.image_orientation,
            Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "image_orientation from direction"
        );
        // pixel_spacing: [spacing[1], spacing[2]]
        assert_eq!(
            seg.pixel_spacing,
            Some([0.5, 1.5]),
            "pixel_spacing [ny_spacing, nx_spacing]"
        );
        // slice_thickness: spacing[0]
        assert_eq!(seg.slice_thickness, Some(2.0), "slice_thickness from spacing[0]");
        // Verify image_position_per_frame includes positions for each Z-slice
        assert_eq!(
            seg.image_position_per_frame[0],
            Some([10.0, 20.0, 30.0]),
            "frame 0 position at z=0"
        );
        assert_eq!(
            seg.image_position_per_frame[1],
            Some([10.0, 20.0, 32.0]),
            "frame 1 position at z=1 (z_offset = 1*spacing[0]=2)"
        );
    }

    #[test]
    fn test_label_map_to_dicom_seg_error_empty_geometry() {
        // Label map with zero dimension should be rejected
        use ritk_core::annotation::{LabelMap, LabelTable};

        let table = LabelTable::new();
        // We can't actually create a map with zero size due to validation in LabelMap::from_data
        // Instead test that the validator catches it by creating a valid map and ensuring
        // the conversion is being tested for its own geometry validation
        let map = LabelMap::new([1, 2, 2], table);

        let origin = [0.0, 0.0, 0.0];
        let spacing = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // This map is all background, so should fail with "no foreground labels"
        let result = label_map_to_dicom_seg(&map, origin, spacing, direction, true);
        assert!(result.is_err(), "all-background map should return Err");
    }


    #[test]
    fn test_label_map_to_dicom_seg_error_no_foreground() {
        // Label map containing only background (0)
        use ritk_core::annotation::{LabelMap, LabelTable};

        let table = LabelTable::new();
        let map = LabelMap::from_data([2, 2, 2], vec![0u32; 8], table).unwrap();

        let origin = [0.0, 0.0, 0.0];
        let spacing = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = label_map_to_dicom_seg(&map, origin, spacing, direction, true);
        assert!(result.is_err(), "all-background map should return Err");
    }

    #[test]
    fn test_dicom_seg_to_label_map_roundtrip_single_label() {
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "Label 1", [255, 0, 0, 255]).unwrap();
        let original = LabelMap::from_data([2, 2, 2], vec![1u32; 8], table).unwrap();

        let seg = label_map_to_dicom_seg(
            &original,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            true,
        )
        .expect("label_map_to_dicom_seg");

        let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
        assert_eq!(rebuilt.shape, [2, 2, 2]);
        assert_eq!(rebuilt.as_slice(), original.as_slice());
        assert_eq!(rebuilt.table.get_label(1).map(|e| e.name.as_str()), Some("Label 1"));
    }

    #[test]
    fn test_dicom_seg_to_label_map_roundtrip_multi_label() {
        use ritk_core::annotation::{LabelMap, LabelTable};

        let mut table = LabelTable::new();
        table.add_label(1, "A", [255, 0, 0, 255]).unwrap();
        table.add_label(2, "B", [0, 255, 0, 255]).unwrap();
        let original = LabelMap::from_data([2, 2, 2], vec![1, 1, 0, 2, 2, 0, 1, 2], table).unwrap();

        let seg = label_map_to_dicom_seg(
            &original,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            true,
        )
        .expect("label_map_to_dicom_seg");

        let rebuilt = dicom_seg_to_label_map(&seg).expect("dicom_seg_to_label_map");
        assert_eq!(rebuilt.shape, [2, 2, 2]);
        assert_eq!(rebuilt.as_slice(), original.as_slice());
        assert!(rebuilt.table.get_label(1).is_some());
        assert!(rebuilt.table.get_label(2).is_some());
    }

    #[test]
    fn test_dicom_seg_to_label_map_error_bad_frame_lengths() {
        let seg = DicomSegmentation {
            rows: 2,
            cols: 2,
            n_frames: 1,
            bits_allocated: 1,
            segmentation_type: "BINARY".to_owned(),
            segments: vec![DicomSegmentInfo {
                segment_number: 1,
                segment_label: "A".to_owned(),
                segment_description: None,
                algorithm_type: None,
            }],
            frame_segment_numbers: vec![1],
            pixel_data: vec![vec![1, 0, 1]],
            image_position_per_frame: vec![None],
            image_orientation: None,
            pixel_spacing: None,
            slice_thickness: None,
        };

        let result = dicom_seg_to_label_map(&seg);
        assert!(result.is_err(), "invalid frame length must fail");
    }

    #[test]
    fn test_dicom_seg_to_label_map_error_uneven_frames_per_segment() {
        let seg = DicomSegmentation {
            rows: 2,
            cols: 2,
            n_frames: 3,
            bits_allocated: 1,
            segmentation_type: "BINARY".to_owned(),
            segments: vec![
                DicomSegmentInfo {
                    segment_number: 1,
                    segment_label: "A".to_owned(),
                    segment_description: None,
                    algorithm_type: None,
                },
                DicomSegmentInfo {
                    segment_number: 2,
                    segment_label: "B".to_owned(),
                    segment_description: None,
                    algorithm_type: None,
                },
            ],
            frame_segment_numbers: vec![1, 1, 2],
            pixel_data: vec![vec![1, 0, 0, 0], vec![0, 1, 0, 0], vec![0, 0, 1, 0]],
            image_position_per_frame: vec![None, None, None],
            image_orientation: None,
            pixel_spacing: None,
            slice_thickness: None,
        };

        let result = dicom_seg_to_label_map(&seg);
        assert!(result.is_err(), "non-divisible frame/segment layout must fail");
    }
}
