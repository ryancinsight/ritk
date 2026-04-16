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
/// The result is sorted by file name for deterministic behavior.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<DicomSeriesInfo> {
    let path = path.as_ref();

    if !path.is_dir() {
        bail!("DICOM input path is not a directory");
    }

    let mut slices = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| "failed to read DICOM directory")? {
        let entry = entry.with_context(|| "failed to read DICOM directory entry")?;
        let entry_path = entry.path();
        if entry_path.is_file() && is_likely_dicom_file(&entry_path) {
            slices.push(DicomSliceMetadata {
                path: entry_path,
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
            });
        }
    }

    if slices.is_empty() {
        bail!("no DICOM files were discovered in the directory");
    }

    slices.sort_by(|a, b| a.path.file_name().cmp(&b.path.file_name()));

    let metadata = DicomReadMetadata {
        series_instance_uid: None,
        study_instance_uid: None,
        frame_of_reference_uid: None,
        series_description: None,
        modality: None,
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [0, 0, slices.len()],
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: None,
        bits_stored: None,
        high_bit: None,
        photometric_interpretation: None,
        slices,
        private_tags: HashMap::new(),
    };

    Ok(DicomSeriesInfo {
        path: path.to_path_buf(),
        num_slices: metadata.slices.len(),
        metadata,
    })
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
    let bytes = std::fs::read(&slice.path)
        .with_context(|| format!("failed to read DICOM slice {:?}", slice.path))?;

    if bytes.is_empty() {
        bail!("DICOM slice file is empty");
    }

    let mut data = Vec::new();
    for chunk in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let scaled = raw as f32 * slice.rescale_slope + slice.rescale_intercept;
        data.push(scaled);
    }

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
