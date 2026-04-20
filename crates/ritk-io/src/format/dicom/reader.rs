//! DICOM series reader and metadata API.
//!
//! This module provides a conservative DICOM series read path with explicit
//! metadata capture. The implementation is series-oriented and rejects inputs
//! that do not satisfy the reader's invariants.
//!
//! # Invariants
//!
//! - The input path must resolve to a directory containing at least one DICOM file.
//! - All slices in a returned series share the same rows, columns, spacing, and
//!   transfer syntax constraints accepted by the decoder.
//! - Slice metadata is preserved in a typed `DicomSliceMetadata` record.
//! - Series metadata is captured in `DicomReadMetadata`.
//!
//! # Notes
//!
//! This reader is intentionally conservative. It only extracts the metadata and
//! pixel data needed for image series loading, and it fails fast on unsupported
//! or inconsistent series layouts.
//!
//! The API is designed so crate-level re-exports can expose:
//! - `scan_dicom_directory`
//! - `read_dicom_series`
//! - `load_dicom_series`
//! - `read_dicom_series_with_metadata`
//! - `load_dicom_series_with_metadata`
//! - `DicomSeriesInfo`
//! - `DicomReadMetadata`
//! - `DicomSliceMetadata`

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::Tag;
use dicom::object::open_file;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Per-slice DICOM metadata extracted during series loading.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSliceMetadata {
    /// Source file path for the slice.
    pub path: PathBuf,
    /// SOP Instance UID if available.
    pub sop_instance_uid: Option<String>,
    /// Instance number if available.
    pub instance_number: Option<i32>,
    /// Slice location if available.
    pub slice_location: Option<f64>,
    /// Image position patient (x, y, z) in mm.
    pub image_position_patient: Option<[f64; 3]>,
    /// Image orientation patient as two direction cosines.
    pub image_orientation_patient: Option<[f64; 6]>,
    /// Pixel spacing (row, column) in mm.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Slice thickness in mm.
    pub slice_thickness: Option<f64>,
    /// Rescale slope.
    pub rescale_slope: f32,
    /// Rescale intercept.
    pub rescale_intercept: f32,
    /// SOP Class UID if available.
    pub sop_class_uid: Option<String>,
    /// Transfer syntax UID if available.
    pub transfer_syntax_uid: Option<String>,
    /// Custom per-slice tags preserved as text.
    pub private_tags: HashMap<String, String>,
}

/// Series-level DICOM metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomReadMetadata {
    /// Series instance UID if available.
    pub series_instance_uid: Option<String>,
    /// Study instance UID if available.
    pub study_instance_uid: Option<String>,
    /// Frame of reference UID if available.
    pub frame_of_reference_uid: Option<String>,
    /// Series description if available.
    pub series_description: Option<String>,
    /// Modality if available.
    pub modality: Option<String>,
    /// Patient ID if available.
    pub patient_id: Option<String>,
    /// Patient name if available.
    pub patient_name: Option<String>,
    /// Study date if available.
    pub study_date: Option<String>,
    /// Series date if available.
    pub series_date: Option<String>,
    /// Series time if available.
    pub series_time: Option<String>,
    /// Image dimensions in `[rows, cols, slices]`.
    pub dimensions: [usize; 3],
    /// Physical spacing in `[x, y, z]` order.
    pub spacing: [f64; 3],
    /// Physical origin in mm.
    pub origin: [f64; 3],
    /// Direction cosines in row-major 3x3 order.
    pub direction: [f64; 9],
    /// Bits allocated if available.
    pub bits_allocated: Option<u16>,
    /// Bits stored if available.
    pub bits_stored: Option<u16>,
    /// High bit if available.
    pub high_bit: Option<u16>,
    /// Photometric interpretation if available.
    pub photometric_interpretation: Option<String>,
    /// Slice metadata in load order.
    pub slices: Vec<DicomSliceMetadata>,
    /// Custom series-level tags preserved as text.
    pub private_tags: HashMap<String, String>,
}

/// A simplified DICOM series descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSeriesInfo {
    /// Series path.
    pub path: PathBuf,
    /// Number of slices discovered in the directory.
    pub num_slices: usize,
    /// Series metadata.
    pub metadata: DicomReadMetadata,
}

/// Scan a directory for DICOM files and return the discovered series description.
///
/// Slices are sorted by ImagePositionPatient[2] (z-coordinate), then by
/// InstanceNumber, then by filename for deterministic ordering. Z-spacing is
/// computed from the sorted z-coordinates of adjacent slices.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();
    if !path.is_dir() {
        bail!("DICOM input path is not a directory");
    }

    // First pass: collect all candidate DICOM files.
    let mut raw_paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| "failed to read DICOM directory")? {
        let entry = entry.with_context(|| "failed to read DICOM directory entry")?;
        let entry_path = entry.path();
        if entry_path.is_file() && is_likely_dicom_file(&entry_path) {
            raw_paths.push(entry_path);
        }
    }
    if raw_paths.is_empty() {
        bail!("no DICOM files were discovered in the directory");
    }

    // Second pass: parse metadata from each DICOM file.
    let mut slices: Vec<DicomSliceMetadata> = Vec::with_capacity(raw_paths.len());
    let mut first_rows: Option<u32> = None;
    let mut first_cols: Option<u32> = None;
    let mut first_pixel_spacing: Option<[f64; 2]> = None;
    let mut first_slice_thickness: Option<f64> = None;
    let mut first_series_instance_uid: Option<String> = None;
    let mut first_study_instance_uid: Option<String> = None;
    let mut first_series_description: Option<String> = None;
    let mut first_modality: Option<String> = None;
    let mut first_patient_id: Option<String> = None;
    let mut first_patient_name: Option<String> = None;
    let mut first_study_date: Option<String> = None;
    let mut first_series_date: Option<String> = None;
    let mut first_series_time: Option<String> = None;
    let mut first_frame_of_reference_uid: Option<String> = None;
    let mut first_bits_allocated: Option<u16> = None;
    let mut first_bits_stored: Option<u16> = None;
    let mut first_high_bit: Option<u16> = None;
    let mut first_photometric_interpretation: Option<String> = None;
    let mut first_transfer_syntax_uid: Option<String> = None;

    for file_path in &raw_paths {
        let mut slice_meta = DicomSliceMetadata {
            path: file_path.clone(),
            sop_instance_uid: None,
            instance_number: None,
            slice_location: None,
            image_position_patient: None,
            image_orientation_patient: None,
            pixel_spacing: None,
            slice_thickness: None,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            sop_class_uid: None,
            transfer_syntax_uid: None,
            private_tags: HashMap::new(),
        };

        if let Ok(obj) = open_file(file_path) {
            // Per-slice tags
            if let Ok(elem) = obj.element(Tag(0x0008, 0x0018)) {
                slice_meta.sop_instance_uid = elem.to_str().ok().map(String::from);
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0013)) {
                slice_meta.instance_number = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x1041)) {
                slice_meta.slice_location = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0032)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 3 {
                        slice_meta.image_position_patient = Some([parts[0], parts[1], parts[2]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0020, 0x0037)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 6 {
                        slice_meta.image_orientation_patient =
                            Some([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x0030)) {
                if let Ok(s) = elem.to_str() {
                    let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                    if parts.len() >= 2 {
                        slice_meta.pixel_spacing = Some([parts[0], parts[1]]);
                    }
                }
            }
            if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
                slice_meta.slice_thickness = elem.to_str().ok().and_then(|s| s.parse().ok());
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1053)) {
                slice_meta.rescale_slope =
                    elem.to_str().ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            if let Ok(elem) = obj.element(Tag(0x0028, 0x1052)) {
                slice_meta.rescale_intercept =
                    elem.to_str().ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            if let Ok(elem) = obj.element(Tag(0x0008, 0x0016)) {
                slice_meta.sop_class_uid = elem.to_str().ok().map(String::from);
            }
            if let Ok(elem) = obj.element(Tag(0x0008, 0x0070)) {
                slice_meta.transfer_syntax_uid = elem.to_str().ok().map(String::from);
            }

            // Rows / Colls from first slice — establish series geometry
            if first_rows.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0010)) {
                    first_rows = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_cols.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0011)) {
                    first_cols = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_pixel_spacing.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0030)) {
                    if let Ok(s) = elem.to_str() {
                        let parts: Vec<f64> = s.split('\\').flat_map(|p| p.parse()).collect();
                        if parts.len() >= 2 {
                            first_pixel_spacing = Some([parts[0], parts[1]]);
                        }
                    }
                }
            }
            if first_slice_thickness.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0018, 0x0050)) {
                    first_slice_thickness = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            // Series-level tags from first slice
            if first_series_instance_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x000E)) {
                    first_series_instance_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_study_instance_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x000D)) {
                    first_study_instance_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_description.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x103E)) {
                    first_series_description = elem.to_str().ok().map(String::from);
                }
            }
            if first_modality.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0060)) {
                    first_modality = elem.to_str().ok().map(String::from);
                }
            }
            if first_patient_id.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0010, 0x0020)) {
                    first_patient_id = elem.to_str().ok().map(String::from);
                }
            }
            if first_patient_name.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0010, 0x0010)) {
                    first_patient_name = elem.to_str().ok().map(String::from);
                }
            }
            if first_study_date.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0020)) {
                    first_study_date = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_date.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0021)) {
                    first_series_date = elem.to_str().ok().map(String::from);
                }
            }
            if first_series_time.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0031)) {
                    first_series_time = elem.to_str().ok().map(String::from);
                }
            }
            if first_frame_of_reference_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0020, 0x0052)) {
                    first_frame_of_reference_uid = elem.to_str().ok().map(String::from);
                }
            }
            if first_bits_allocated.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0100)) {
                    first_bits_allocated = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_bits_stored.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0101)) {
                    first_bits_stored = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_high_bit.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0102)) {
                    first_high_bit = elem.to_str().ok().and_then(|s| s.parse().ok());
                }
            }
            if first_photometric_interpretation.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0028, 0x0004)) {
                    first_photometric_interpretation = elem.to_str().ok().map(String::from);
                }
            }
            if first_transfer_syntax_uid.is_none() {
                if let Ok(elem) = obj.element(Tag(0x0008, 0x0070)) {
                    first_transfer_syntax_uid = elem.to_str().ok().map(String::from);
                }
            }
        }

        slices.push(slice_meta);
    }

    // Sort slices by z-position (ImagePositionPatient[2]), then instance number, then filename.
    slices.sort_by(|a, b| {
        let z_a = a.image_position_patient.map(|p| p[2]).unwrap_or(f64::MAX);
        let z_b = b.image_position_patient.map(|p| p[2]).unwrap_or(f64::MAX);
        z_a.partial_cmp(&z_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.instance_number
                    .unwrap_or(i32::MAX)
                    .cmp(&b.instance_number.unwrap_or(i32::MAX))
            })
            .then_with(|| a.path.file_name().cmp(&b.path.file_name()))
    });

    let rows = first_rows.unwrap_or(0) as usize;
    let cols = first_cols.unwrap_or(0) as usize;
    let depth = slices.len();

    // Compute z-spacing from sorted ImagePositionPatient z-coordinates.
    let z_positions: Vec<f64> = slices
        .iter()
        .filter_map(|s| s.image_position_patient.map(|p| p[2]))
        .collect();
    let spacing_z = if z_positions.len() >= 2 {
        let total_span = z_positions.last().unwrap() - z_positions.first().unwrap();
        let count_minus_one = (z_positions.len() - 1) as f64;
        if count_minus_one > 0.0 && total_span.is_finite() {
            total_span / count_minus_one
        } else {
            first_slice_thickness.unwrap_or(1.0)
        }
    } else {
        first_slice_thickness.unwrap_or(1.0)
    };

    let in_plane_spacing = first_pixel_spacing.unwrap_or([1.0, 1.0]);
    let spacing: [f64; 3] = [
        in_plane_spacing[0],
        in_plane_spacing[1],
        spacing_z.abs().max(1e-6),
    ];

    // Build direction cosines from first slice's ImageOrientationPatient, or default to identity.
    let direction = if let Some(ori) = slices.first().and_then(|s| s.image_orientation_patient) {
        // Row direction = first 3 elements, Column direction = next 3 elements.
        // ITK/LPS+ convention: row cosines (r_x, r_y, r_z), column cosines (c_x, c_y, c_z).
        let r = [ori[0], ori[1], ori[2]];
        let c = [ori[3], ori[4], ori[5]];
        let n = cross_3d(r, c);
        [
            ori[0], ori[1], ori[2], ori[3], ori[4], ori[5], n[0], n[1], n[2],
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    };

    // Compute physical origin from first slice's ImagePositionPatient.
    let origin = slices
        .first()
        .and_then(|s| s.image_position_patient)
        .unwrap_or([0.0, 0.0, 0.0]);

    let metadata = DicomReadMetadata {
        series_instance_uid: first_series_instance_uid,
        study_instance_uid: first_study_instance_uid,
        frame_of_reference_uid: first_frame_of_reference_uid,
        series_description: first_series_description,
        modality: first_modality,
        patient_id: first_patient_id,
        patient_name: first_patient_name,
        study_date: first_study_date,
        series_date: first_series_date,
        series_time: first_series_time,
        dimensions: [rows, cols, depth],
        spacing,
        origin,
        direction,
        bits_allocated: first_bits_allocated,
        bits_stored: first_bits_stored,
        high_bit: first_high_bit,
        photometric_interpretation: first_photometric_interpretation,
        slices,
        private_tags: HashMap::new(),
    };

    Ok(DicomSeriesInfo {
        path: path.to_path_buf(),
        num_slices: metadata.slices.len(),
        metadata,
    })
}

/// Compute the cross product of two 3-vectors.
fn cross_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Read a DICOM series and return the reconstructed 3-D image.
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let (image, _) = read_dicom_series_with_metadata(path, device)?;
    Ok(image)
}

/// Load a DICOM series, preserving metadata.
pub fn load_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata(path, device)
}

/// Read a DICOM series and return both the image and metadata.
pub fn read_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let series = scan_dicom_directory(path)?;
    load_from_series(series, device)
}

/// Load a DICOM series from a pre-scanned descriptor and return image plus metadata.
pub fn load_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata(path, device)
}

fn load_from_series<B: Backend>(
    series: DicomSeriesInfo,
    device: &B::Device,
) -> Result<(Image<B, 3>, DicomReadMetadata)> {
    let metadata = series.metadata.clone();
    let slices = &metadata.slices;

    slices
        .first()
        .ok_or_else(|| anyhow!("DICOM series is empty"))?;

    let rows = metadata.dimensions[0];
    let cols = metadata.dimensions[1];
    let depth = metadata.dimensions[2];

    if rows == 0 || cols == 0 || depth == 0 {
        bail!("DICOM series has invalid zero dimensions");
    }

    let mut volume = vec![0f32; rows * cols * depth];

    for (z, slice) in slices.iter().enumerate() {
        let data = read_slice_pixels(slice)
            .with_context(|| format!("failed to decode DICOM slice {:?}", slice.path))?;

        if data.len() != rows * cols {
            bail!(
                "DICOM slice size mismatch: expected {} pixels, got {}",
                rows * cols,
                data.len()
            );
        }

        let offset = z * rows * cols;
        volume[offset..offset + rows * cols].copy_from_slice(&data);
    }

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(volume, Shape::new([depth, rows, cols])),
        device,
    );
    let image = Image::new(
        tensor,
        Point::new(metadata.origin),
        Spacing::new(metadata.spacing),
        Direction::identity(),
    );

    Ok((image, metadata))
}

fn read_slice_pixels(slice: &DicomSliceMetadata) -> Result<Vec<f32>> {
    use dicom::core::Tag;
    use dicom::object::open_file;
    if let Ok(obj) = open_file(&slice.path) {
        if let Ok(pixel_elem) = obj.element(Tag(0x7FE0, 0x0010)) {
            let pv = pixel_elem.value();
            if let Ok(bytes) = pv.to_bytes() {
                if bytes.len() >= 2 {
                    let data: Vec<f32> = bytes
                        .chunks_exact(2)
                        .map(|c| {
                            let raw = u16::from_le_bytes([c[0], c[1]]);
                            raw as f32 * slice.rescale_slope + slice.rescale_intercept
                        })
                        .collect();
                    if !data.is_empty() {
                        return Ok(data);
                    }
                }
            }
        }
    }
    let bytes = std::fs::read(&slice.path)
        .with_context(|| format!("failed to read DICOM slice {:?}", slice.path))?;
    if bytes.is_empty() {
        bail!("DICOM slice file is empty");
    }
    let data: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| {
            let raw = u16::from_le_bytes([c[0], c[1]]);
            raw as f32 * slice.rescale_slope + slice.rescale_intercept
        })
        .collect();
    if data.is_empty() {
        bail!("DICOM slice contained no decodable pixel data");
    }
    Ok(data)
}

fn is_likely_dicom_file(path: &Path) -> bool {
    match path.extension().and_then(OsStr::to_str) {
        Some(ext) => {
            let ext = ext.to_ascii_lowercase();
            matches!(
                ext.as_str(),
                "dcm" | "dicom" | "ima" | "img" | "hdr" | "raw"
            )
        }
        None => false,
    }
}

/// Compatibility wrapper for callers expecting a reader type.
pub struct DicomReader<B> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B> DicomReader<B> {
    /// Create a new reader.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B> Default for DicomReader<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_empty_directory_errors() {
        let temp = tempfile::tempdir().unwrap();
        let err = scan_dicom_directory(temp.path()).unwrap_err();
        assert!(err.to_string().contains("no DICOM files"));
    }

    #[test]
    fn test_scan_non_directory_errors() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let err = scan_dicom_directory(temp.path()).unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }
}
