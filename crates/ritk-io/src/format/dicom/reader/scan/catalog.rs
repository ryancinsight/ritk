//! Multi-series discovery over a browser-selected collection of Part 10 instances.

use super::{
    image_series_uid, is_dicomdir, parse_bytes_with_budget, DicomReadBudget, DicomSeriesInfo,
    NativeBackend, SeriesScan,
};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Discover all image series in a bounded in-memory selection, ordered by UID.
/// Each returned scan retains exactly its own admitted Part 10 instances.
pub fn scan_dicom_part10_catalog(files: &[(&str, &[u8])]) -> Result<Vec<DicomSeriesInfo>> {
    let budget = DicomReadBudget::DEFAULT;
    budget.checked_instances(files.len())?;
    let mut series = std::collections::BTreeMap::<_, SeriesScan>::new();
    let mut retained = 0usize;
    for &(name, bytes) in files {
        if is_dicomdir(Path::new(name)) {
            continue; // The selected files themselves establish membership.
        }
        retained = retained
            .checked_add(bytes.len())
            .context("study size overflow")?;
        budget.checked_retained_bytes(retained)?;
        let object = parse_bytes_with_budget::<NativeBackend>(bytes, &budget.parser())
            .context("failed to parse DICOM byte payload")?;
        if let Some(uid) = image_series_uid(&object)? {
            series.entry(uid).or_default().push(
                &object,
                PathBuf::from(name),
                bytes.to_vec(),
                None,
                &budget,
            )?;
        }
    }
    if series.is_empty() {
        bail!("select the DICOM image instances for this study");
    }
    series
        .into_values()
        .map(|scan| scan.finish(PathBuf::from("dropped://series")))
        .collect()
}
