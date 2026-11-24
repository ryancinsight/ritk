//! Browser series retain their selected Part 10 payloads for later panel assignments.
use super::*;
use crate::dicom::loader::{load_volume_from_preview_bytes, load_volume_from_scanned_series};
use crate::dicom::series_tree::SeriesEntry;
use std::collections::BTreeMap;

impl SeriesBrowser {
    pub(in crate::presentation::native_session) fn from_bytes(
        files: &[(&str, &[u8])],
    ) -> Result<Self> {
        let mut entries = Vec::new();
        let mut sources = BTreeMap::new();
        for scan in ritk_io::scan_dicom_part10_catalog(files)? {
            let meta = &scan.metadata;
            let uid = meta
                .series_instance_uid
                .as_deref()
                .ok_or_else(|| anyhow!("missing series UID"))?;
            let info = DicomSeriesInfo::new(
                uid,
                meta.series_description.clone().unwrap_or_default(),
                meta.modality.as_deref().unwrap_or(""),
                meta.patient_id.clone().unwrap_or_default(),
                meta.slices.iter().map(|slice| slice.path.clone()).collect(),
            );
            entries.push(SeriesEntry::with_metadata(info, meta));
            sources.insert(uid.to_owned(), scan);
        }
        let mut browser = Self::from_tree(&SeriesTree::from_entries(entries), None)?;
        browser.sources = sources;
        Ok(browser)
    }

    pub(in crate::presentation::native_session) fn load(
        &self,
        index: usize,
    ) -> Result<crate::LoadedVolume> {
        let choice = self
            .choice(index)
            .ok_or_else(|| anyhow!("series outside catalog"))?;
        let scan = self
            .sources
            .get(choice.acquisition.series_instance_uid())
            .ok_or_else(|| anyhow!("missing selected series bytes"))?;
        load_volume_from_scanned_series(scan.clone())
    }

    pub(super) fn preview_volume(&self, choice: &SeriesChoice) -> Result<crate::LoadedVolume> {
        let slice = self
            .sources
            .get(choice.acquisition.series_instance_uid())
            .and_then(|scan| scan.metadata.slices.first())
            .ok_or_else(|| anyhow!("series preview requires an instance"))?;
        let bytes = slice
            .part10_bytes
            .as_deref()
            .ok_or_else(|| anyhow!("missing preview bytes"))?;
        load_volume_from_preview_bytes(
            slice
                .path
                .to_str()
                .ok_or_else(|| anyhow!("invalid browser filename"))?,
            bytes,
        )
    }
}
