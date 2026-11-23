//! Stable workstation ordering of discovered acquisitions.
use super::{SeriesEntry, SeriesTree};
use ritk_io::{DicomReadMetadata, DicomSeriesInfo, DicomTag};
use std::cmp::Ordering;

pub(crate) struct DiscoveredSeries {
    number: Option<i32>,
    entry: SeriesEntry<'static>,
}

impl DiscoveredSeries {
    pub(crate) fn new(info: DicomSeriesInfo, metadata: &DicomReadMetadata) -> Self {
        // Both hosts use the first member in path order. The in-memory reader
        // stores slice metadata in spatial order, whereas directory discovery
        // reads the first member's header only.
        let number = metadata
            .slices
            .iter()
            .min_by(|a, b| a.path.cmp(&b.path))
            .and_then(|slice| slice.preservation.object.get(DicomTag::new(0x0020, 0x0011)))
            .and_then(|node| node.value.as_text())
            .and_then(|text| text.trim().parse().ok());
        Self {
            number,
            entry: SeriesEntry::with_metadata(info, metadata),
        }
    }

    /// Basic and Strict retain Series Number, modality and instance membership.
    /// The numbered export is an order-preserving renaming of sorted source
    /// paths, so comparing the first member also preserves the final tie.
    fn compare_acquisition(&self, other: &Self) -> Ordering {
        let a = &self.entry.acquisition;
        let b = &other.entry.acquisition;
        self.number
            .is_none()
            .cmp(&other.number.is_none())
            .then_with(|| self.number.cmp(&other.number))
            .then_with(|| a.modality().cmp(b.modality()))
            .then_with(|| b.file_paths.len().cmp(&a.file_paths.len()))
            .then_with(|| a.file_paths.first().cmp(&b.file_paths.first()))
    }
}

impl SeriesTree<'static> {
    pub(crate) fn from_discovery(mut found: Vec<DiscoveredSeries>) -> Self {
        // The export scope is a single DICOM study. Even inconsistent source
        // patient labels within that study must not outrank an acquisition.
        let single_study = found.first().is_some_and(|first| {
            first.entry.study_uid.is_some()
                && found
                    .iter()
                    .all(|item| item.entry.study_uid == first.entry.study_uid)
        });
        found.sort_by(|a, b| {
            if single_study {
                a.compare_acquisition(b)
            } else {
                a.entry
                    .acquisition
                    .patient_id
                    .cmp(&b.entry.acquisition.patient_id)
                    .then_with(|| a.entry.study_uid.cmp(&b.entry.study_uid))
                    .then_with(|| a.entry.study_date.cmp(&b.entry.study_date))
                    .then_with(|| a.compare_acquisition(b))
            }
        });
        Self::from_entries(found.into_iter().map(|series| series.entry).collect())
    }
}

#[cfg(test)]
mod tests;
