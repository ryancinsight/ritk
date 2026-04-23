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
//!
//! # Writer specification (`write_dicom_multiframe`)
//!
//! Writes a 3-D `Image<B, 3>` with shape `[n_frames, rows, cols]` as a single
//! DICOM Part 10 file. The writer enforces the following constraints:
//!
//! ## Encoding constraints
//! - **SOP Class**: Secondary Capture Image Storage (`1.2.840.10008.5.1.4.1.1.7`).
//!   The output is not an Enhanced Multi-Frame CT, MR, or PET object. Viewers
//!   that enforce strict modality-to-SOP-class binding may reject the file.
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
//! ## Spatial metadata constraints
//! - No spatial metadata (ImagePositionPatient, ImageOrientationPatient,
//!   PixelSpacing, SliceThickness) is written. The origin is implicitly [0,0,0]
//!   and direction is identity. Use `write_dicom_series` for series with full
//!   spatial metadata.
//!
//! ## Interoperability limits
//! - The file is readable by `load_dicom_multiframe` (round-trip invariant:
//!   |recovered − original| ≤ rescale_slope + 1.0).
//! - DICOM conformance: the file satisfies the Secondary Capture IOD but does
//!   NOT carry a conformance statement or General Series / Frame Of Reference
//!   modules required for Enhanced Multi-Frame objects.
//! - Viewers expecting modality-specific Enhanced objects (e.g., Enhanced CT
//!   Storage) will not recognise the SOP class and may warn or reject.

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::{open_file, InMemDicomObject};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

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
    /// Pixel spacing [row_spacing, col_spacing] in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Frame thickness (SliceThickness) in mm.
    pub frame_thickness: Option<f64>,
    /// Modality string.
    pub modality: Option<String>,
    /// SOP Class UID.
    pub sop_class_uid: Option<String>,
}

/// Read summary information from a multi-frame DICOM file without pixel data.
pub fn read_multiframe_info(path: impl AsRef<Path>) -> Result<MultiFrameInfo> {
    let path = path.as_ref();
    let obj = open_file(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;

    let n_frames: usize = obj.element(Tag(0x0028, 0x0008)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);
    let rows: usize = obj.element(Tag(0x0028, 0x0010)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let cols: usize = obj.element(Tag(0x0028, 0x0011)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let bits_allocated: u16 = obj.element(Tag(0x0028, 0x0100)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(16);
    let pixel_spacing = obj.element(Tag(0x0028, 0x0030)).ok()
        .and_then(|e| e.to_str().ok().map(|s| {
            let parts: Vec<f64> = s.trim().split('\\')
                .filter_map(|p| p.trim().parse::<f64>().ok())
                .collect();
            if parts.len() >= 2 { Some([parts[0], parts[1]]) } else { None }
        }))
        .flatten();
    let frame_thickness = obj.element(Tag(0x0018, 0x0050)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok());
    let modality = obj.element(Tag(0x0008, 0x0060)).ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty());
    let sop_class_uid = obj.element(Tag(0x0008, 0x0016)).ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty());

    Ok(MultiFrameInfo { path: path.to_path_buf(), n_frames, rows, cols,
        bits_allocated, pixel_spacing, frame_thickness, modality, sop_class_uid })
}

/// Load a multi-frame DICOM file as a 3-D image with shape [n_frames, rows, cols].
///
/// Applies RescaleSlope and RescaleIntercept to convert stored integers to floats.
pub fn load_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let obj = open_file(path)
        .with_context(|| format!("failed to open DICOM file {:?}", path))?;

    let n_frames: usize = obj.element(Tag(0x0028, 0x0008)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);
    let rows: usize = obj.element(Tag(0x0028, 0x0010)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let cols: usize = obj.element(Tag(0x0028, 0x0011)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let bits_allocated: u16 = obj.element(Tag(0x0028, 0x0100)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(16);
    if rows == 0 || cols == 0 {
        bail!("DICOM multiframe: rows={rows} cols={cols} must be >0 in {:?}", path);
    }
    let rescale_slope: f32 = obj.element(Tag(0x0028, 0x1053)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0_f32);
    let rescale_intercept: f32 = obj.element(Tag(0x0028, 0x1052)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0_f32);
    let pixel_bytes = if let Ok(elem) = obj.element(Tag(0x7FE0, 0x0010)) {
        elem.value().to_bytes().ok().map(|b| b.to_vec()).unwrap_or_default()
    } else {
        Vec::new()
    };

    let bytes_per_sample = ((bits_allocated as usize) + 7) / 8;
    let floats: Vec<f32> = if bytes_per_sample == 2 {
        pixel_bytes.chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes([c[0], c[1]]) as f32;
                raw * rescale_slope + rescale_intercept
            })
            .collect()
    } else {
        pixel_bytes.iter()
            .map(|&b| b as f32 * rescale_slope + rescale_intercept)
            .collect()
    };
    let actual_n = if rows * cols > 0 && !floats.is_empty() {
        floats.len() / (rows * cols)
    } else {
        n_frames
    };
    if actual_n == 0 {
        bail!("DICOM multiframe: no pixel data decoded from {:?}", path);
    }
    let pixel_spacing_val = obj.element(Tag(0x0028, 0x0030)).ok()
        .and_then(|e| e.to_str().ok().map(|s| {
            let v: Vec<f64> = s.trim().split('\\')
                .filter_map(|p| p.trim().parse::<f64>().ok()).collect();
            if v.len() >= 2 { Some([v[0], v[1]]) } else { None }
        })).flatten();
    let slice_thickness: f64 = obj.element(Tag(0x0018, 0x0050)).ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);
    let spacing = match pixel_spacing_val {
        Some([rs, cs]) => Spacing::new([slice_thickness, rs, cs]),
        None           => Spacing::new([slice_thickness, 1.0, 1.0]),
    };
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(floats, Shape::new([actual_n, rows, cols])),
        device,
    );
    Ok(Image::new(tensor, Point::new([0.0, 0.0, 0.0]), spacing, Direction::identity()))
}


/// Generate a DICOM UID using nanoseconds since UNIX epoch under the 2.25 root.
///
/// Invariant: uniqueness holds within a single process under non-repeating system clock.
fn generate_multiframe_uid() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("2.25.{}", t)
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
/// SOP Class: Secondary Capture (1.2.840.10008.5.1.4.1.1.7).
pub fn write_dicom_multiframe<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
) -> Result<()> {
    let path = path.as_ref();
    let [n_frames, rows, cols] = image.shape();
    if n_frames == 0 || rows == 0 || cols == 0 {
        bail!(
            "DICOM multiframe write: n_frames={n_frames} rows={rows} cols={cols} must all be >0"
        );
    }
    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
    let (min_val, max_val) = all_data
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| (mn.min(v), mx.max(v)));
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
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(Tag(0x0008, 0x0016), VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7")));
    obj.put(DataElement::new(Tag(0x0008, 0x0018), VR::UI, PrimitiveValue::from(sop_instance_uid.as_str())));
    obj.put(DataElement::new(Tag(0x0008, 0x0060), VR::CS, PrimitiveValue::from("OT")));
    obj.put(DataElement::new(Tag(0x0028, 0x0008), VR::IS, PrimitiveValue::from(format!("{}", n_frames))));
    obj.put(DataElement::new(Tag(0x0028, 0x0010), VR::US, PrimitiveValue::from(rows as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0011), VR::US, PrimitiveValue::from(cols as u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0100), VR::US, PrimitiveValue::from(16_u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0101), VR::US, PrimitiveValue::from(16_u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0102), VR::US, PrimitiveValue::from(15_u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0103), VR::US, PrimitiveValue::from(0_u16)));
    obj.put(DataElement::new(Tag(0x0028, 0x0004), VR::CS, PrimitiveValue::from("MONOCHROME2")));
    obj.put(DataElement::new(Tag(0x0028, 0x1053), VR::DS, PrimitiveValue::from(format!("{:.6}", rescale_slope))));
    obj.put(DataElement::new(Tag(0x0028, 0x1052), VR::DS, PrimitiveValue::from(format!("{:.6}", rescale_intercept))));
    obj.put(DataElement::new(Tag(0x7FE0, 0x0010), VR::OW, PrimitiveValue::U16(SmallVec::from_vec(pixel_u16))));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
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
    use burn_ndarray::NdArray;
    use burn::tensor::backend::Backend;
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
    fn test_load_multiframe_single_frame_via_writer() {
        use crate::format::dicom::writer::write_dicom_series;
        use burn::tensor::{Shape, TensorData};
        use ritk_core::image::Image;
        use ritk_core::spatial::{Direction, Point, Spacing};
        let device = <B as Backend>::Device::default();
        let tmp = tempfile::tempdir().expect("tempdir");
        let data: Vec<f32> = vec![0.0_f32; 1 * 4 * 5];
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new([1_usize, 4, 5])), &device);
        let image = Image::new(tensor,
            Point::new([0.0, 0.0, 0.0]), Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity());
        write_dicom_series(tmp.path(), &image).expect("write_dicom_series");
        let slice_path = tmp.path().join("slice_0000.dcm");
        assert!(slice_path.exists(), "slice_0000.dcm must exist");
        let loaded = load_dicom_multiframe::<B, _>(&slice_path, &device)
            .expect("load_dicom_multiframe");
        let [frames, rows, cols] = loaded.shape();
        assert_eq!(frames, 1, "frames");
        assert_eq!(rows, 4, "rows");
        assert_eq!(cols, 5, "cols");
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
        assert!(result.is_err(), "write_dicom_multiframe must return Err for zero-row image");
    }

}
