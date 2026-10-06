//! Typed discovery and exact selection for multi-series DICOM studies.

use std::path::Path;

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::Image;

use crate::format::dicom::reader::{
    load_dicom_from_series_with_budget, scan_dicom_files_with_budget, DicomReadBudget,
    DicomReadMetadata,
};

use super::{scan::scan_dicom_directory_with_budget, DicomSeriesInfo};

/// Independently selectable DICOM series discovered beneath one study path.
#[derive(Debug)]
pub struct DicomStudyCatalog {
    series: Vec<DicomSeriesInfo>,
}

impl DicomStudyCatalog {
    /// Scan a directory or DICOMDIR and group every image series by UID.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery cannot read or validate the input file set.
    pub fn scan<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::scan_with_budget(path, &DicomReadBudget::DEFAULT)
    }

    /// Scan and group every image series under explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns an error when discovery cannot read or validate the input file set,
    /// or when a candidate count or encoded-byte ceiling is exceeded.
    pub fn scan_with_budget<P: AsRef<Path>>(path: P, budget: &DicomReadBudget) -> Result<Self> {
        scan_dicom_directory_with_budget(path, budget).map(|series| Self { series })
    }

    /// Return every discovered series in deterministic display order.
    #[must_use]
    pub fn series(&self) -> &[DicomSeriesInfo] {
        &self.series
    }

    /// Consume the catalog and return its discovered series.
    #[must_use]
    pub fn into_series(self) -> Vec<DicomSeriesInfo> {
        self.series
    }

    /// Return the series whose `SeriesInstanceUID` exactly matches `uid`.
    ///
    /// # Errors
    ///
    /// Returns an error when `uid` is empty or absent from the catalog.
    pub fn select(&self, uid: &str) -> Result<&DicomSeriesInfo> {
        let uid = uid.trim();
        if uid.is_empty() {
            bail!("SeriesInstanceUID selection must not be empty");
        }
        self.series
            .iter()
            .find(|series| series.series_instance_uid() == uid)
            .with_context(|| format!("SeriesInstanceUID {uid:?} was not found in DICOM catalog"))
    }

    /// Decode one UID-selected series and return its pixels and geometry metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when selection fails or the selected series cannot be
    /// validated, decoded, or reconstructed.
    pub fn load<B: ComputeBackend>(
        &self,
        uid: &str,
        backend: &B,
    ) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
        self.load_with_budget(uid, backend, &DicomReadBudget::DEFAULT)
    }

    /// Decode one UID-selected series under explicit read resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns an error when selection fails, a resource ceiling is exceeded,
    /// or the selected series cannot be validated, decoded, or reconstructed.
    pub fn load_with_budget<B: ComputeBackend>(
        &self,
        uid: &str,
        backend: &B,
        budget: &DicomReadBudget,
    ) -> Result<(Image<f32, B, 3>, DicomReadMetadata)> {
        let selected = self.select(uid)?;
        let selected_uid = selected.series_instance_uid();
        let scanned = scan_dicom_files_with_budget(&selected.file_paths, budget)
            .with_context(|| format!("failed to validate selected DICOM series {uid:?}"))?;
        let scanned_uid = scanned.metadata.series_instance_uid.as_deref();
        if scanned_uid != Some(selected_uid) {
            bail!(
                "selected DICOM series changed from {selected_uid:?} to {scanned_uid:?} after catalog discovery"
            );
        }
        load_dicom_from_series_with_budget(scanned, backend, budget)
            .with_context(|| format!("failed to load selected DICOM series {uid:?}"))
    }
}
