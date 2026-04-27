//! Multi-frame DICOM image reader and writer.
//!
//! # Reader specification
//!
//! A multi-frame DICOM file stores N frames in one file:
//! - (0028,0008) NumberOfFrames: N (absent ⇒ 1)
//! - (0028,0010) Rows, (0028,0011) Columns
//! - (7FE0,0010) PixelData: `N × Rows × Cols × (BitsAllocated/8)` bytes
//!
//! ## Reader invariants
//! - `n_frames >= 1`
//! - Output tensor shape: `[n_frames, rows, cols]`
//! - RescaleSlope (absent ⇒ 1.0) and RescaleIntercept (absent ⇒ 0.0) applied.
//! - 8-bit and 16-bit BitsAllocated are both supported.
//! - ImagePositionPatient (0020,0032) sets the image origin when present.
//! - ImageOrientationPatient (0020,0037) sets the direction matrix when present;
//!   the normal vector is computed as the cross product of the row and column cosines.
//!
//! # Writer specification (`write_dicom_multiframe`)
//!
//! Writes a 3-D `Image<B, 3>` with shape `[n_frames, rows, cols]` as a single
//! DICOM Part 10 file. The writer enforces the following constraints:
//!
//! ## Encoding constraints
//! - **SOP Class**: Multi-Frame Grayscale Word Secondary Capture Image Storage
//!   (`1.2.840.10008.5.1.4.1.1.7.3`). The output is not an Enhanced Multi-Frame
//!   CT, MR, or PET object. Viewers that enforce strict modality-to-SOP-class
//!   binding may reject the file.
//! - **Transfer Syntax**: Explicit VR Little Endian (`1.2.840.10008.1.2.1`).
//!   Compressed transfer syntaxes (JPEG, JPEG-LS, JPEG 2000) are not supported.
//! - **Pixel depth**: always 16-bit unsigned (BitsAllocated=16, BitsStored=16,
//!   HighBit=15, PixelRepresentation=0).
//!
//! ## Rescale constraints
//! - A **single global linear rescale** maps the entire f32 volume to the u16 range
//!   [0, 65535]: `rescale_slope = (max - min) / 65535; rescale_intercept = min`.
//! - When max == min (flat image), slope = 1.0 and intercept = min_val.
//! - **All frames share one slope/intercept pair.** Per-frame rescaling is not
//!   supported. Images whose frames have widely varying intensity ranges will
//!   lose intra-frame contrast fidelity relative to inter-frame range.
//!
//! ## Spatial metadata
//! - `write_dicom_multiframe` emits no spatial metadata (IPP/IOP/PixelSpacing/
//!   SliceThickness). Use [`write_dicom_multiframe_with_options`] with a
//!   [`MultiFrameSpatialMetadata`] value to include spatial tags.
//!
//! ## Interoperability limits
//! - The file is readable by `load_dicom_multiframe` (round-trip invariant:
//!   |recovered − original| ≤ rescale_slope + 1.0).
//! - DICOM conformance: the file satisfies the Multi-Frame Grayscale Word SC IOD
//!   but does NOT carry a conformance statement or General Series / Frame Of
//!   Reference modules required for Enhanced Multi-Frame objects.

use super::transfer_syntax::TransferSyntaxKind;
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::{open_file, InMemDicomObject};
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

/// SOP Class UID for Multi-Frame Grayscale Word Secondary Capture Image Storage.
const MF_GRAYSCALE_WORD_SC_UID: &str = "1.2.840.10008.5.1.4.1.1.7.3";

/// Parse a `\`-separated DICOM Decimal String (DS) field into a fixed-size array.
///
/// # Invariant
/// Returns `Some(arr)` iff the input contains at least `N` parseable `f64` values
/// separated by `\`. Non-numeric tokens are skipped. Returns `None` if fewer than
/// `N` valid numeric components exist.
fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]> {
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

/// Summary information about a multi-frame DICOM file.
#[derive(Debug, Clone)]
pub struct MultiFrameInfo {
    /// Source file path.
    pub path: PathBuf,
    /// Number of frames.
    pub n_frames: usize,
    /// Pixel rows per frame.
    pub rows: usize,
    /// Pixel columns per frame.
    pub cols: usize,
    /// Bits allocated per sample (8 or 16).
    pub bits_allocated: u16,
    /// PixelRepresentation (0028,0103): 0 = unsigned, 1 = signed two's complement.
    /// Defaults to 0 (unsigned) per DICOM PS3.3 C.7.6.3.1.
    pub pixel_representation: u16,
    /// Pixel spacing [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Frame thickness (SliceThickness) in mm.
    pub frame_thickness: Option<f64>,
    /// Modality string.
    pub modality: Option<String>,
    /// SOP Class UID.
    pub sop_class_uid: Option<String>,
    /// ImagePositionPatient for frame 0: [x, y, z] in mm.
    pub image_position: Option<[f64; 3]>,
    /// ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z].
    pub image_orientation: Option<[f64; 6]>,
    /// RescaleSlope (0028,1053). Defaults to 1.0 when absent.
    pub rescale_slope: f64,
    /// RescaleIntercept (0028,1052). Defaults to 0.0 when absent.
    pub rescale_intercept: f64,
}

/// Extract all multi-frame header fields from an already-opened DICOM object.
///
/// # Invariants
/// - n_frames defaults to 1 when (0028,0008) is absent.
/// - bits_allocated defaults to 16 when absent.
/// - rescale_slope defaults to 1.0, rescale_intercept to 0.0 when absent.
fn extract_multiframe_header(path: &Path, obj: &InMemDicomObject) -> MultiFrameInfo {
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
    }
}

/// Read summary information from a multi-frame DICOM file without pixel data.
pub fn read_multiframe_info(path: impl AsRef<Path>) -> Result<MultiFrameInfo> {
    let path = path.as_ref();
    let obj = open_file(path).with_context(|| format!("failed to open DICOM file {:?}", path))?;
    Ok(extract_multiframe_header(path, &obj))
}

/// Load a multi-frame DICOM file as a 3-D image with shape [n_frames, rows, cols].
///
/// Applies RescaleSlope and RescaleIntercept to convert stored integers to floats.
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
    let obj = open_file(path).with_context(|| format!("failed to open DICOM file {:?}", path))?;

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

    let rescale_slope = info.rescale_slope as f32;
    let rescale_intercept = info.rescale_intercept as f32;

    let (floats, actual_n) = if ts.is_codec_supported() {
        // Compressed TS with registered codec: decode each frame individually.
        let mut all_floats = Vec::with_capacity(info.n_frames * info.rows * info.cols);
        for frame_idx in 0..info.n_frames {
            let frame = super::codec::decode_compressed_frame(
                &obj,
                frame_idx as u32,
                info.bits_allocated,
                info.pixel_representation,
                rescale_slope,
                rescale_intercept,
            )
            .with_context(|| format!("codec failed for frame {frame_idx} in {:?}", path))?;
            if frame.is_empty() {
                bail!(
                    "DICOM multiframe: codec produced empty frame {frame_idx} in {:?}",
                    path
                );
            }
            all_floats.extend_from_slice(&frame);
        }
        let n = info.n_frames;
        (all_floats, n)
    } else {
        // Native (uncompressed) TS: read all pixel bytes at once and apply LUT.
        let pixel_bytes = if let Ok(elem) = obj.element(Tag(0x7FE0, 0x0010)) {
            elem.value()
                .to_bytes()
                .ok()
                .map(|b| b.to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let all_floats = super::reader::decode_pixel_bytes(
            &pixel_bytes,
            info.bits_allocated,
            info.pixel_representation,
            rescale_slope,
            rescale_intercept,
        );
        let n = if info.rows * info.cols > 0 && !all_floats.is_empty() {
            all_floats.len() / (info.rows * info.cols)
        } else {
            info.n_frames
        };
        (all_floats, n)
    };

    if actual_n == 0 {
        bail!("DICOM multiframe: no pixel data decoded from {:?}", path);
    }

    let spacing = match info.pixel_spacing {
        Some([rs, cs]) => Spacing::new([info.frame_thickness.unwrap_or(1.0), rs, cs]),
        None => Spacing::new([info.frame_thickness.unwrap_or(1.0), 1.0, 1.0]),
    };
    let origin = info.image_position.unwrap_or([0.0, 0.0, 0.0]);
    let direction = if let Some(iop) = info.image_orientation {
        let (rx, ry, rz) = (iop[0], iop[1], iop[2]);
        let (cx, cy, cz) = (iop[3], iop[4], iop[5]);
        let nx = ry * cz - rz * cy;
        let ny = rz * cx - rx * cz;
        let nz = rx * cy - ry * cx;
        let col_data: [f64; 9] = [rx, ry, rz, cx, cy, cz, nx, ny, nz];
        Direction(SMatrix::<f64, 3, 3>::from_column_slice(&col_data))
    } else {
        Direction::identity()
    };

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(floats, Shape::new([actual_n, info.rows, info.cols])),
        device,
    );
    Ok(Image::new(tensor, Point::new(origin), spacing, direction))
}

/// Generate a DICOM UID using nanoseconds since UNIX epoch under the 2.25 root.
///
/// Invariant: uniqueness holds within a single process under non-repeating system clock.
fn generate_multiframe_uid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Format: 2.25.<ns>.<seq> — both components are numeric; total ≤ 64 chars.
    format!("2.25.{}.{}", t, n)
}

/// Optional spatial metadata for multi-frame DICOM output.
///
/// When provided to [`write_dicom_multiframe_with_options`], spatial tags are
/// emitted: ImagePositionPatient (0020,0032), ImageOrientationPatient (0020,0037),
/// PixelSpacing (0028,0030), SliceThickness (0018,0050), and Modality (0008,0060).
///
/// When absent, the writer behaves identically to [`write_dicom_multiframe`].
#[derive(Debug, Clone)]
pub struct MultiFrameSpatialMetadata {
    /// ImagePositionPatient for frame 0: [x, y, z] in mm.
    pub origin: [f64; 3],
    /// Pixel spacing [row_spacing, col_spacing] in mm.
    pub pixel_spacing: [f64; 2],
    /// Slice thickness in mm.
    pub slice_thickness: f64,
    /// ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z].
    pub image_orientation: [f64; 6],
    /// Modality string (e.g., "CT", "MR", "OT").
    pub modality: String,
}

/// Builder for multi-frame DICOM write options.
///
/// # Invariants
/// - `sop_class_uid` defaults to [`MF_GRAYSCALE_WORD_SC_UID`].
/// - `instance_number` defaults to 1.
/// - When `spatial` is `None`, no spatial tags are emitted.
///
/// Use [`write_dicom_multiframe_with_config`] to supply an explicit config.
#[derive(Debug, Clone)]
pub struct MultiFrameWriterConfig {
    /// SOP Class UID (0008,0016). Defaults to Multi-Frame Grayscale Word SC UID.
    pub sop_class_uid: String,
    /// Optional spatial metadata emitted as IPP/IOP/PixelSpacing/SliceThickness/Modality.
    pub spatial: Option<MultiFrameSpatialMetadata>,
    /// InstanceNumber (0020,0013). Defaults to 1.
    pub instance_number: u32,
}

impl Default for MultiFrameWriterConfig {
    fn default() -> Self {
        Self {
            sop_class_uid: MF_GRAYSCALE_WORD_SC_UID.to_string(),
            spatial: None,
            instance_number: 1,
        }
    }
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
        bail!("DICOM multiframe write: n_frames={n_frames} rows={rows} cols={cols} must all be >0");
    }
    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
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
                .media_storage_sop_class_uid(&config.sop_class_uid)
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .map_err(|e| anyhow::anyhow!("DICOM multiframe meta build failed: {e}"))?;
    file_obj
        .write_to_file(path)
        .map_err(|e| anyhow::anyhow!("DICOM multiframe write to {:?} failed: {e}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    type B = NdArray<f32>;

    #[test]
    fn test_read_multiframe_info_missing_file_returns_error() {
        let result = read_multiframe_info("/nonexistent/path/file.dcm");
        assert!(result.is_err(), "expected Err for missing file");
    }

    #[test]
    fn test_load_multiframe_missing_file_returns_error() {
        let device = <B as Backend>::Device::default();
        let result = load_dicom_multiframe::<B, _>("/nonexistent/path/file.dcm", &device);
        assert!(result.is_err(), "expected Err for missing file");
    }

    #[test]
    fn test_multiframe_info_and_roundtrip_writer_read_consistency() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("multiframe.dcm");
        let n_frames = 2_usize;
        let rows = 3_usize;
        let cols = 4_usize;

        let mut data: Vec<f32> = Vec::with_capacity(n_frames * rows * cols);
        for frame in 0..n_frames {
            for row in 0..rows {
                for col in 0..cols {
                    data.push((frame * 100 + row * 10 + col) as f32);
                }
            }
        }

        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
        assert!(out_path.exists(), "output file must exist after write");

        let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
        assert_eq!(info.n_frames, n_frames, "n_frames");
        assert_eq!(info.rows, rows, "rows");
        assert_eq!(info.cols, cols, "cols");
        assert_eq!(info.bits_allocated, 16, "bits_allocated");
        assert_eq!(info.modality.as_deref(), Some("OT"), "modality");
        assert_eq!(
            info.sop_class_uid.as_deref(),
            Some("1.2.840.10008.5.1.4.1.1.7.3"),
            "sop_class_uid"
        );

        let loaded =
            load_dicom_multiframe::<B, _>(&out_path, &device).expect("load_dicom_multiframe");
        let [frames, loaded_rows, loaded_cols] = loaded.shape();
        assert_eq!(frames, n_frames, "frames");
        assert_eq!(loaded_rows, rows, "loaded_rows");
        assert_eq!(loaded_cols, cols, "loaded_cols");

        let recovered_td = loaded.data().clone().into_data();
        let recovered: &[f32] = recovered_td
            .as_slice::<f32>()
            .expect("recovered tensor must be f32");
        assert_eq!(recovered.len(), data.len(), "recovered pixel count");

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = if (max_val - min_val).abs() <= f32::EPSILON {
            1.0_f32
        } else {
            (max_val - min_val) / 65535.0_f32
        };
        let tolerance = slope + 1.0_f32;

        for (idx, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {idx}: original={orig:.4} recovered={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    }

    /// Round-trip invariant: write then read must recover pixel values within
    /// quantization error of at most rescale_slope + 1.0 per sample.
    ///
    /// Analytical ground truth: val = (frame * 100 + row * 10 + col) as f32
    /// for shape [3, 4, 5]. min=0.0, max=245.0 => slope = 245.0/65535.0.
    /// Max quantization error per sample <= slope + 1.0 (rounding + slope bound).
    #[test]
    fn test_write_read_multiframe_roundtrip() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("multiframe.dcm");
        let n_frames = 3_usize;
        let rows = 4_usize;
        let cols = 5_usize;
        let mut data: Vec<f32> = Vec::with_capacity(n_frames * rows * cols);
        for frame in 0..n_frames {
            for row in 0..rows {
                for col in 0..cols {
                    data.push((frame * 100 + row * 10 + col) as f32);
                }
            }
        }
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
        assert!(out_path.exists(), "output file must exist after write");
        let loaded = load_dicom_multiframe::<B, _>(&out_path, &device)
            .expect("load_dicom_multiframe roundtrip");
        let [lf, lr, lc] = loaded.shape();
        assert_eq!(lf, n_frames, "recovered n_frames");
        assert_eq!(lr, rows, "recovered rows");
        assert_eq!(lc, cols, "recovered cols");
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = if (max_val - min_val).abs() <= f32::EPSILON {
            1.0_f32
        } else {
            (max_val - min_val) / 65535.0_f32
        };
        let tolerance = slope + 1.0_f32;
        let recovered_td = loaded.data().clone().into_data();
        let recovered: &[f32] = recovered_td
            .as_slice::<f32>()
            .expect("recovered tensor must be f32");
        assert_eq!(recovered.len(), data.len(), "recovered pixel count");
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: original={orig:.4} recovered={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    }

    #[test]
    fn test_read_multiframe_info_reports_scalar_defaults_for_single_frame() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("single_frame.dcm");

        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![7.0_f32; 1 * 2 * 3], Shape::new([1_usize, 2, 3])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
        let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
        assert_eq!(info.n_frames, 1, "single-frame file must report one frame");
        assert_eq!(info.rows, 2, "rows");
        assert_eq!(info.cols, 3, "cols");
        assert_eq!(info.bits_allocated, 16, "bits_allocated");
        assert_eq!(info.modality.as_deref(), Some("OT"), "modality");
        assert_eq!(
            info.sop_class_uid.as_deref(),
            Some("1.2.840.10008.5.1.4.1.1.7.3"),
            "sop_class_uid"
        );
    }

    /// Rejection invariant: any dimension equal to zero must produce Err.
    #[test]
    fn test_write_multiframe_rejects_zero_dimension() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("zero.dcm");
        let data: Vec<f32> = vec![];
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([1_usize, 0_usize, 5_usize])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let result = write_dicom_multiframe(&out_path, &image);
        assert!(
            result.is_err(),
            "write_dicom_multiframe must return Err for zero-row image"
        );
    }

    #[test]
    fn test_multiframe_sop_class_is_mf_grayscale_word() {
        // Verifies that write_dicom_multiframe emits the Multi-Frame Grayscale Word SC SOP class
        // (1.2.840.10008.5.1.4.1.1.7.3) rather than Single-frame SC (1.2.840.10008.5.1.4.1.1.7).
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf.dcm");
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 2 * 3 * 4], Shape::new([2_usize, 3, 4])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
        assert_eq!(
            info.sop_class_uid.as_deref(),
            Some("1.2.840.10008.5.1.4.1.1.7.3"),
            "SOP class must be Multi-Frame Grayscale Word Secondary Capture"
        );
    }

    #[test]
    fn test_write_multiframe_with_spatial_metadata_round_trip() {
        // Analytical invariants:
        // - IPP round-trip: |read_ipp[i] - written_ipp[i]| < 1e-4 (DS string precision)
        // - IOP round-trip: |read_iop[i] - written_iop[i]| < 1e-4
        // - Modality round-trip: exact string match
        // - origin in loaded Image equals IPP to ±1e-4
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf_spatial.dcm");
        let n_frames = 2_usize;
        let rows = 3_usize;
        let cols = 4_usize;
        let data: Vec<f32> = (0..n_frames * rows * cols).map(|i| i as f32).collect();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let spatial = MultiFrameSpatialMetadata {
            origin: [10.0, 20.0, -50.0],
            pixel_spacing: [0.8, 0.8],
            slice_thickness: 2.5,
            image_orientation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            modality: "CT".to_string(),
        };
        write_dicom_multiframe_with_options(&out_path, &image, Some(&spatial))
            .expect("write_dicom_multiframe_with_options");
        assert!(out_path.exists(), "output file must exist");

        // Verify via read_multiframe_info
        let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
        assert_eq!(
            info.modality.as_deref(),
            Some("CT"),
            "modality must round-trip"
        );
        assert_eq!(
            info.sop_class_uid.as_deref(),
            Some("1.2.840.10008.5.1.4.1.1.7.3"),
            "SOP class must be Multi-Frame Grayscale Word SC"
        );
        let ipp = info.image_position.expect("image_position must be Some");
        assert!((ipp[0] - 10.0).abs() < 1e-4, "IPP x round-trip");
        assert!((ipp[1] - 20.0).abs() < 1e-4, "IPP y round-trip");
        assert!((ipp[2] - (-50.0)).abs() < 1e-4, "IPP z round-trip");
        let iop = info
            .image_orientation
            .expect("image_orientation must be Some");
        assert!((iop[0] - 1.0).abs() < 1e-4, "IOP[0] round-trip");
        assert!((iop[4] - 1.0).abs() < 1e-4, "IOP[4] round-trip");

        // Verify via load_dicom_multiframe
        let loaded =
            load_dicom_multiframe::<B, _>(&out_path, &device).expect("load_dicom_multiframe");
        let loaded_origin = loaded.origin();
        assert!((loaded_origin[0] - 10.0).abs() < 1e-4, "loaded origin x");
        assert!((loaded_origin[1] - 20.0).abs() < 1e-4, "loaded origin y");
        assert!((loaded_origin[2] - (-50.0)).abs() < 1e-4, "loaded origin z");

        // Shape invariant
        let [lf, lr, lc] = loaded.shape();
        assert_eq!(lf, n_frames, "frame count");
        assert_eq!(lr, rows, "rows");
        assert_eq!(lc, cols, "cols");

        // Pixel round-trip within quantization tolerance
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = if (max_val - min_val).abs() <= f32::EPSILON {
            1.0_f32
        } else {
            (max_val - min_val) / 65535.0_f32
        };
        let tolerance = slope + 1.0_f32;
        let recovered_td = loaded.data().clone().into_data();
        let recovered: &[f32] = recovered_td.as_slice::<f32>().expect("f32");
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: orig={orig:.4} rec={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    }

    /// Invariant: every file written by write_dicom_multiframe must carry
    /// SamplesPerPixel (0028,0002) = 1 (Type 1 mandatory tag in Image Pixel Module).
    ///
    /// Proof: Multi-Frame Grayscale Word SC IOD mandates SamplesPerPixel = 1 (PS3.3 C.7.6.3.1.1).
    #[test]
    fn test_written_multiframe_has_samples_per_pixel_one() {
        use dicom::object::open_file;
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf_spp.dcm");
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 2 * 4 * 5], Shape::new([2_usize, 4, 5])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");

        let obj = open_file(&out_path).expect("open_file");
        let spp: u16 = obj
            .element(dicom::core::Tag(0x0028, 0x0002))
            .expect("SamplesPerPixel (0028,0002) must be present")
            .to_str()
            .expect("SamplesPerPixel must be readable as string")
            .trim()
            .parse()
            .expect("SamplesPerPixel must be numeric");
        assert_eq!(
            spp, 1,
            "SamplesPerPixel must equal 1 for grayscale multi-frame"
        );
    }

    /// Invariant: MultiFrameWriterConfig.instance_number propagates to InstanceNumber (0020,0013).
    ///
    /// Proof: write_multiframe_impl emits format!("{}", config.instance_number) as IS VR.
    #[test]
    fn test_writer_config_instance_number_propagated() {
        use dicom::object::open_file;
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf_inst.dcm");
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![5.0_f32; 1 * 2 * 3], Shape::new([1_usize, 2, 3])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let config = MultiFrameWriterConfig {
            instance_number: 42,
            ..MultiFrameWriterConfig::default()
        };
        write_dicom_multiframe_with_config(&out_path, &image, &config).expect("write");

        let obj = open_file(&out_path).expect("open_file");
        let inst_num: u32 = obj
            .element(dicom::core::Tag(0x0020, 0x0013))
            .expect("InstanceNumber (0020,0013) must be present")
            .to_str()
            .expect("InstanceNumber must be readable")
            .trim()
            .parse()
            .expect("InstanceNumber must be numeric");
        assert_eq!(
            inst_num, 42,
            "InstanceNumber must match config.instance_number"
        );
    }

    /// Invariant: write+read round-trip for negative-intensity images must satisfy
    /// |recovered − original| ≤ rescale_slope + 1.0 for all samples.
    ///
    /// Analytical: range = [-1024.0, 500.0] = 1524.0, slope = 1524.0/65535.0 ≈ 0.02325.
    #[test]
    fn test_round_trip_negative_intensity_image() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf_neg.dcm");
        let n = 2_usize * 3 * 4;
        let data: Vec<f32> = (0..n)
            .map(|i| -1024.0_f32 + (i as f32) * (1524.0_f32 / (n as f32 - 1.0_f32)))
            .collect();
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([2_usize, 3, 4])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load");

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = (max_val - min_val) / 65535.0_f32;
        let tolerance = slope + 1.0_f32;

        let recovered_td = loaded.data().clone().into_data();
        let recovered: &[f32] = recovered_td.as_slice::<f32>().expect("f32 slice");
        assert_eq!(recovered.len(), data.len(), "pixel count must be preserved");
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: orig={orig:.4} rec={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    }

    /// Invariant: flat image (all-same value) must round-trip with zero quantization error.
    ///
    /// Proof: when min == max, slope = 1.0 and intercept = constant_value.
    /// All stored u16 = 0; recovered = 0 * 1.0 + intercept = intercept = constant_value.
    #[test]
    fn test_round_trip_flat_image_exact() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("mf_flat.dcm");
        let constant = 42.75_f32; // exactly representable; "{:.6}" -> "42.750000" -> 42.75
        let data: Vec<f32> = vec![constant; 2 * 3 * 4];
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.clone(), Shape::new([2_usize, 3, 4])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load");
        let recovered_td = loaded.data().clone().into_data();
        let recovered: &[f32] = recovered_td.as_slice::<f32>().expect("f32 slice");
        assert_eq!(recovered.len(), data.len(), "pixel count must be preserved");
        for (i, &rec) in recovered.iter().enumerate() {
            assert!(
                (rec - constant).abs() <= f32::EPSILON,
                "pixel {i}: expected {constant} got {rec} (diff {})",
                (rec - constant).abs()
            );
        }
    }

    /// Compressed transfer syntax guard: load_dicom_multiframe must return Err
    /// when the file meta declares a compressed transfer syntax.
    ///
    /// Analytical invariant: any TS for which TransferSyntaxKind::is_compressed() is true
    /// must be rejected before pixel decode so no garbage data is produced.
    #[test]
    fn test_load_multiframe_compressed_ts_errors() {
        use dicom::object::meta::FileMetaTableBuilder;
        use dicom::object::InMemDicomObject;
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("compressed.dcm");

        // Construct a minimal DICOM dataset that declares JPEG Baseline TS (1.2.840.10008.1.2.4.50)
        // but contains no real compressed pixels. The TS guard must fire before pixel decode.
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7.3"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.99999"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(vec![0u8; 8])),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                    .media_storage_sop_instance_uid("2.25.99999")
                    .transfer_syntax("1.2.840.10008.1.2.4.80"), // JPEG-LS Lossless (no charls)
            )
            .expect("meta build");
        file_obj
            .write_to_file(&path)
            .expect("write compressed stub");

        let result = load_dicom_multiframe::<B, _>(&path, &device);
        assert!(
            result.is_err(),
            "load_dicom_multiframe must return Err for compressed TS"
        );
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("1.2.840.10008.1.2.4.80") || msg.to_lowercase().contains("compress"),
            "error must name the TS UID or contain 'compress'; got: {msg}"
        );
    }

    /// Verify that a JPEG Baseline multi-frame file (codec-supported compressed TS) loads
    /// successfully and produces pixel values within JPEG quantization tolerance.
    ///
    /// # Specification
    /// Guard condition `is_compressed() && !is_codec_supported()` allows JPEG Baseline.
    /// `load_dicom_multiframe` calls `codec::decode_compressed_frame` for each frame.
    /// Shape must be `[N_frames, H, W]`.
    /// Per-pixel error must satisfy `|decoded - original| ≤ 8` (JPEG Q75 tolerance).
    #[test]
    fn test_load_multiframe_jpeg_baseline_codec_round_trip() {
        use dicom::core::smallvec::SmallVec;
        use dicom::core::value::PixelFragmentSequence;
        use image::{DynamicImage, GrayImage};

        let n_frames = 2usize;
        let rows = 4u32;
        let cols = 4u32;

        // Generate two distinct 4×4 frames.
        let frame0: Vec<u8> = (0u8..16).collect();
        let frame1: Vec<u8> = (100u8..116).collect();
        let original = [frame0.clone(), frame1.clone()];

        // JPEG-encode each frame into a separate fragment.
        let mut fragments: SmallVec<[Vec<u8>; 2]> = SmallVec::new();
        for frame_pixels in &original {
            let gray =
                GrayImage::from_raw(cols, rows, frame_pixels.clone()).expect("GrayImage::from_raw");
            let dyn_img = DynamicImage::ImageLuma8(gray);
            let mut jpeg_bytes: Vec<u8> = Vec::new();
            let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Jpeg)
                .expect("JPEG encode");
            drop(cursor);
            fragments.push(jpeg_bytes);
        }

        let pfs: PixelFragmentSequence<Vec<u8>> = PixelFragmentSequence::new_fragments(fragments);

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.77777701"),
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
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from("2.25.77777702"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.77777703"),
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
            Tag(0x0028, 0x0008),
            VR::IS,
            PrimitiveValue::from(format!("{n_frames}")),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(8u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(7u16),
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
            PrimitiveValue::from("MONOCHROME2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from("1.000000"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from("0.000000"),
        ));
        obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OB, pfs));

        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                    .media_storage_sop_instance_uid("2.25.77777701")
                    .transfer_syntax("1.2.840.10008.1.2.4.50"), // JPEG Baseline
            )
            .expect("meta build");

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mf_jpeg.dcm");
        file_obj.write_to_file(&path).expect("write");

        let device = <B as Backend>::Device::default();
        let img = load_dicom_multiframe::<B, _>(&path, &device)
            .expect("JPEG Baseline multiframe load must succeed via codec path");

        let [lf, lr, lc] = img.shape();
        assert_eq!(lf, n_frames, "shape[0] must equal n_frames");
        assert_eq!(lr, rows as usize, "shape[1] must equal rows");
        assert_eq!(lc, cols as usize, "shape[2] must equal cols");

        let td = img.data().clone().into_data();
        let floats: &[f32] = td.as_slice::<f32>().expect("f32 slice");
        assert_eq!(
            floats.len(),
            n_frames * (rows as usize) * (cols as usize),
            "total pixel count mismatch"
        );

        // Verify each frame independently.
        let frame_size = (rows * cols) as usize;
        for (f_idx, orig_frame) in original.iter().enumerate() {
            let decoded_frame = &floats[f_idx * frame_size..(f_idx + 1) * frame_size];
            let max_error = orig_frame
                .iter()
                .zip(decoded_frame.iter())
                .map(|(&o, &d)| (o as f32 - d).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 8.0,
                "frame {f_idx}: codec round-trip error {max_error} exceeds JPEG tolerance 8.0"
            );
        }
    }

    /// MultiFrameInfo must include rescale_slope and rescale_intercept populated from
    /// the written DICOM file's (0028,1053) and (0028,1052) tags.
    ///
    /// Analytical ground truth:
    /// - image range [0.0, 124.0] => slope = 124.0/65535.0, intercept = 0.0
    /// - DS format "{:.6}" => read-back within f64::EPSILON of written value
    #[test]
    fn test_multiframe_info_rescale_slope_intercept_populated() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("rescale.dcm");

        // Analytically derive expected slope/intercept
        let n_frames = 1_usize;
        let rows = 5_usize;
        let cols = 5_usize;
        let data: Vec<f32> = (0..n_frames * rows * cols).map(|i| i as f32).collect();
        let min_val = 0.0_f32;
        let max_val = 24.0_f32;
        let expected_slope = (max_val - min_val) / 65535.0_f32;
        let expected_intercept = min_val;

        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([n_frames, rows, cols])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let info = read_multiframe_info(&out_path).expect("read_multiframe_info");

        // DS precision is 6 decimal places => tolerance is 5e-7
        let tol = 5e-7_f64;
        assert!(
            (info.rescale_slope - expected_slope as f64).abs() < tol,
            "rescale_slope: expected {:.8} got {:.8}",
            expected_slope,
            info.rescale_slope
        );
        assert!(
            (info.rescale_intercept - expected_intercept as f64).abs() < tol,
            "rescale_intercept: expected {:.8} got {:.8}",
            expected_intercept,
            info.rescale_intercept
        );
    }

    /// ConversionType (0008,0064) must be present and equal to "WSD" in the output file.
    ///
    /// Invariant: SC Equipment Module (PS3.3 C.8.6.1) mandates ConversionType as Type 1.
    #[test]
    fn test_multiframe_has_conversion_type_wsd() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("conv_type.dcm");

        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 1 * 2 * 2], Shape::new([1_usize, 2, 2])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");

        let obj = open_file(&out_path).expect("open");
        let conv_type = obj
            .element(Tag(0x0008, 0x0064))
            .expect("ConversionType (0008,0064) must be present")
            .to_str()
            .expect("ConversionType must be a string")
            .trim()
            .to_string();
        assert_eq!(
            conv_type, "WSD",
            "ConversionType must be 'WSD' (Workstation)"
        );
    }

    /// Type 1 UIDs — StudyInstanceUID (0020,000D) and SeriesInstanceUID (0020,000E)
    /// must be present, non-empty, and distinct in the emitted multiframe DICOM file.
    ///
    /// Invariant: SC Multi-Frame IOD (PS3.3 A.8.5.2) mandates both UIDs as Type 1.
    #[test]
    fn test_multiframe_has_study_and_series_uids() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("uids.dcm");
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 1 * 2 * 2], Shape::new([1_usize, 2, 2])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let obj = open_file(&out_path).expect("open");
        let study_uid = obj
            .element(Tag(0x0020, 0x000D))
            .expect("StudyInstanceUID (0020,000D) must be present")
            .to_str()
            .expect("StudyInstanceUID must be a string")
            .trim()
            .to_string();
        let series_uid = obj
            .element(Tag(0x0020, 0x000E))
            .expect("SeriesInstanceUID (0020,000E) must be present")
            .to_str()
            .expect("SeriesInstanceUID must be a string")
            .trim()
            .to_string();
        assert!(!study_uid.is_empty(), "StudyInstanceUID must be non-empty");
        assert!(
            !series_uid.is_empty(),
            "SeriesInstanceUID must be non-empty"
        );
        assert_ne!(
            study_uid, series_uid,
            "StudyInstanceUID and SeriesInstanceUID must be distinct"
        );
    }

    /// Type 2 mandatory patient/study/series tags must be present in the emitted
    /// multiframe DICOM file even when no user metadata is provided.
    ///
    /// Tags verified: PatientName (0010,0010), PatientID (0010,0020),
    /// StudyDate (0008,0020), ReferringPhysicianName (0008,0090),
    /// StudyID (0020,0010), SeriesNumber (0020,0011).
    ///
    /// Invariant: SC Multi-Frame IOD (PS3.3 A.8.5.2) mandates these as Type 2.
    #[test]
    fn test_multiframe_has_type2_patient_study_series_tags() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("type2.dcm");
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![5.0_f32; 1 * 3 * 3], Shape::new([1_usize, 3, 3])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        write_dicom_multiframe(&out_path, &image).expect("write");
        let obj = open_file(&out_path).expect("open");
        // Assert presence (value may be empty per Type 2 semantics).
        obj.element(Tag(0x0010, 0x0010))
            .expect("PatientName (0010,0010) must be present");
        obj.element(Tag(0x0010, 0x0020))
            .expect("PatientID (0010,0020) must be present");
        obj.element(Tag(0x0008, 0x0020))
            .expect("StudyDate (0008,0020) must be present");
        obj.element(Tag(0x0008, 0x0090))
            .expect("ReferringPhysicianName (0008,0090) must be present");
        obj.element(Tag(0x0020, 0x0010))
            .expect("StudyID (0020,0010) must be present");
        obj.element(Tag(0x0020, 0x0011))
            .expect("SeriesNumber (0020,0011) must be present");
    }

    /// Signed i16 pixel round-trip invariant:
    ///   load_dicom_multiframe with PixelRepresentation=1 must decode two's-complement
    ///   i16 samples correctly: decoded_f32 = i16 × RescaleSlope + RescaleIntercept.
    ///
    /// Analytical ground truth (identity rescale, slope=1.0, intercept=0.0):
    ///   pixels = [-1000, 0, 1000, 2000] → expected f32 = [-1000.0, 0.0, 1000.0, 2000.0]
    #[test]
    fn test_load_multiframe_signed_i16_roundtrip() {
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("signed_i16.dcm");

        // Construct a 1-frame 2×2 DICOM file with PixelRepresentation=1 (signed i16).
        let signed_pixels: [i16; 4] = [-1000, 0, 1000, 2000];
        let pixel_bytes: Vec<u8> = signed_pixels
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();

        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(MF_GRAYSCALE_WORD_SC_UID),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.999888777"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0008),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(1_u16), // signed
        ));
        // Identity rescale: slope=1.0, intercept=0.0 (defaults when absent).
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(MF_GRAYSCALE_WORD_SC_UID)
                    .media_storage_sop_instance_uid("2.25.999888777")
                    .transfer_syntax("1.2.840.10008.1.2.1"), // Explicit VR LE
            )
            .expect("meta build");
        file_obj
            .write_to_file(&out_path)
            .expect("write signed file");

        let loaded = load_dicom_multiframe::<B, _>(&out_path, &device)
            .expect("load_dicom_multiframe signed i16");
        let [frames, rows, cols] = loaded.shape();
        assert_eq!(frames, 1, "frames");
        assert_eq!(rows, 2, "rows");
        assert_eq!(cols, 2, "cols");

        let td = loaded.data().clone().into_data();
        let result: &[f32] = td.as_slice::<f32>().expect("f32 slice");
        assert_eq!(result.len(), 4, "pixel count");

        // Analytical ground truth: i16 × 1.0 + 0.0
        let expected = [-1000.0_f32, 0.0, 1000.0, 2000.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.5,
                "pixel {i}: expected {exp:.1} got {got:.1}"
            );
        }
    }

    #[test]
    fn test_multiframe_rejects_big_endian_ts() {
        // Verify that a DICOM multiframe file with ExplicitVrBigEndian TS
        // is rejected before pixel decode. We construct a file with BigEndian
        // in its file meta and assert load_dicom_multiframe returns an error.
        use dicom::object::meta::FileMetaTableBuilder;
        use dicom::object::InMemDicomObject;
        type B = burn_ndarray::NdArray<f32>;
        let device = Default::default();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("be_test.dcm");

        // Build a minimal multiframe object with BigEndian TS in meta.
        let mut obj = InMemDicomObject::new_empty();
        // PixelData — 4 bytes (1 frame, 1x1 pixel, 16-bit LE; BE interpretation is wrong)
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(dicom::core::smallvec::SmallVec::from_slice(&[0u16])),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(1u16),
        ));
        // Build file meta with BigEndian TS UID
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7.3")
                    .media_storage_sop_instance_uid("2.25.999")
                    .transfer_syntax("1.2.840.10008.1.2.2"), // ExplicitVrBigEndian
            )
            .expect("meta build must succeed");
        file_obj.write_to_file(&path).expect("write must succeed");

        let result = load_dicom_multiframe::<B, _>(&path, &device);
        assert!(
            result.is_err(),
            "load_dicom_multiframe must reject ExplicitVrBigEndian TS"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("big-endian"),
            "error message must mention big-endian; got: {err_msg}"
        );
    }
}
