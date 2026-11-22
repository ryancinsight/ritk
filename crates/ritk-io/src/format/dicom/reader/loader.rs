//! Load a reconstructed scalar DICOM series onto the requested compute backend.

use anyhow::Result;
use coeus_core::ComputeBackend;
use ritk_image::Image;
use std::path::Path;

use super::reconstruction::{decode_dicom_series, DecodedDicomSeries};
use super::scan::scan_dicom_path_with_budget;
use super::types::{DicomReadMetadata, DicomSeriesInfo};
use super::DicomReadBudget;

/// Read a DICOM series and return both the image and metadata.
///
/// Part-10 payloads retained by the scan are used during reconstruction and
/// released before the returned metadata leaves the loader. The returned
/// metadata therefore carries the validated descriptive fields without a
/// second copy of the encoded study.
pub fn read_dicom_series_with_metadata<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata_with_budget(path, backend, &DicomReadBudget::DEFAULT)
}

/// Read a DICOM series with explicit parser, retained-byte, and decoded-buffer ceilings.
pub fn read_dicom_series_with_metadata_with_budget<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    let series = scan_dicom_path_with_budget(path, budget)?;
    load_from_series_with_budget(series, backend, budget)
}

/// Load a DICOM series from a pre-scanned descriptor and return image plus metadata.
pub fn load_dicom_series_with_metadata<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_dicom_series_with_metadata_with_budget(path, backend, &DicomReadBudget::DEFAULT)
}

/// Load a DICOM series with explicit parser, retained-byte, and decoded-buffer ceilings.
pub fn load_dicom_series_with_metadata_with_budget<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    read_dicom_series_with_metadata_with_budget(path, backend, budget)
}

/// Load a DICOM series from a pre-scanned [`DicomSeriesInfo`] and return image plus metadata.
///
/// This is the zero-disk counterpart of [`load_dicom_series_with_metadata`]:
/// callers that have already obtained a `DicomSeriesInfo` (e.g. via
/// [`scan_dicom_instances`](super::scan::scan_dicom_instances)) pass it directly
/// instead of re-scanning a directory. Pixel decode uses `part10_bytes` from the
/// slice metadata when present, falling back to file-path I/O otherwise. Those
/// payloads are released after reconstruction; callers that need the encoded
/// bytes must retain the scan input separately.
pub fn load_dicom_from_series<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series(series, backend)
}

/// Load a pre-scanned DICOM series with explicit decoded-buffer ceilings.
pub fn load_dicom_from_series_with_budget<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series_with_budget(series, backend, budget)
}

pub(crate) fn load_from_series<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    load_from_series_with_budget(series, backend, &DicomReadBudget::DEFAULT)
}

pub(crate) fn load_from_series_with_budget<B: ComputeBackend>(
    series: DicomSeriesInfo,
    backend: &B,
    budget: &DicomReadBudget,
) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
    let DecodedDicomSeries {
        volume,
        shape,
        origin,
        spacing,
        direction,
        metadata,
    } = decode_dicom_series(series, budget)?;
    let image = Image::from_flat_on(volume, shape, origin, spacing, direction, backend)?;

    Ok((image, metadata))
}
